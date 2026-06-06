from __future__ import annotations

try:
    import timm
except ImportError:
    timm = None

import math
import logging
import inspect
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS

from mmseg.registry import MODELS


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear.

    The original linear layer is frozen. Only the low-rank update

        x -> W x + scale * B(A(x))

    is trainable.

    The LoRA branch is initialized such that its initial contribution is zero:
    A is randomly initialized and B is initialized with zeros. Therefore, at
    initialization, the wrapped layer behaves exactly like the original frozen
    linear layer.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear)}")

        if r < 0:
            raise ValueError(f"LoRA rank r must be non-negative, got r={r}")

        self.linear = linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.r if self.r > 0 else 0.0

        # Freeze original pretrained projection.
        for p in self.linear.parameters():
            p.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        if self.r > 0:
            self.lora_A = nn.Linear(in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, out_features, bias=False)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # Standard LoRA initialization:
            # A random, B zero => initially identical to the frozen base model.
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    def forward(self, x):
        out = self.linear(x)

        if self.r > 0:
            out = out + self.scale * self.lora_B(
                self.lora_A(self.dropout(x))
            )

        return out


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module):
    """Replace a submodule given a dotted module path."""
    parts = module_name.split(".")
    parent = root

    for p in parts[:-1]:
        parent = getattr(parent, p)

    setattr(parent, parts[-1], new_module)


def _call_with_supported_kwargs(fn, *args, **kwargs):
    """Call a function while only forwarding kwargs accepted by its signature."""
    if not kwargs:
        return fn(*args)

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args)

    accepted = {
        k: v
        for k, v in kwargs.items()
        if k in sig.parameters
    }

    return fn(*args, **accepted)


@MODELS.register_module()
class LoRAIntermediateTIMMBackbone(BaseModule):
    """Frozen timm DINOv3/EVA ViT backbone with trainable LoRA adapters.

    This backbone returns an intermediate ViT feature map using timm's own
    forward_intermediates(...) implementation.

    Important parameter:

        out_block_index=-1  -> last Transformer block
        out_block_index=-2  -> penultimate / vorletzter Transformer block
        out_block_index=-3  -> third-last Transformer block

    The returned feature map is:

        (B, C, H_patch, W_patch)

    For input (B, 3, 256, 512) and patch_size=16, this gives:

        (B, 384, 16, 32)

    Output:

        tuple(spatial,)
    """

    def __init__(
        self,
        model_name="vit_small_patch16_dinov3",
        pretrained=False,
        checkpoint_path="",
        in_channels=3,
        patch_size=16,
        frozen_base=True,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        target_modules=("attn.qkv", "attn.proj"),
        out_block_index=-2,
        apply_output_norm=True,
        verbose=True,
        init_cfg=None,
        **kwargs,
    ):
        if timm is None:
            raise RuntimeError("timm is not installed")

        super().__init__(init_cfg)

        if "norm_layer" in kwargs:
            kwargs["norm_layer"] = MMENGINE_MODELS.get(kwargs["norm_layer"])

        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)

        self.frozen_base = bool(frozen_base)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.target_modules = tuple(target_modules)

        self.out_block_index = int(out_block_index)
        self.apply_output_norm = bool(apply_output_norm)

        self.verbose = bool(verbose)

        # Important: features_only=False.
        # For local DINOv3 checkpoints, features_only=True may wrap the model
        # and cause state_dict prefix mismatches.
        self.timm_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            in_chans=in_channels,
            **kwargs,
        )

        if hasattr(self.timm_model, "global_pool"):
            self.timm_model.global_pool = None
        if hasattr(self.timm_model, "fc"):
            self.timm_model.fc = None
        if hasattr(self.timm_model, "classifier"):
            self.timm_model.classifier = None

        if self.frozen_base:
            self.freeze_base()

        self.inject_lora()

        self._check_intermediate_support()

        if self.verbose:
            self.log_lora_summary()
            self.log_intermediate_summary()

        if pretrained or checkpoint_path:
            self._is_init = True

    def _log_info(self, msg: str):
        """Small logging helper that works also outside a full MMEngine run."""
        logging.getLogger(__name__).info(msg)

    def freeze_base(self):
        """Freeze all parameters of the timm model before LoRA injection."""
        for param in self.timm_model.parameters():
            param.requires_grad = False

    def inject_lora(self):
        """Inject LoRA into selected Linear layers.

        For timm ViT/EVA, typical module names are e.g.

            blocks.0.attn.qkv
            blocks.0.attn.proj
            blocks.1.attn.qkv
            blocks.1.attn.proj

        Depending on the exact timm model, MLP modules may be named differently.
        """

        replaced = []

        # list(...) is important because we modify modules while iterating.
        target_block_idx = self._resolve_block_index()
        
        for name, module in list(self.timm_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            # If this Linear layer belongs to a Transformer block after the selected
            # output block, skip it. Otherwise DDP will see trainable parameters that
            # never participate in the loss when out_block_index < -1.
            if name.startswith("blocks."):
                parts = name.split(".")
                if len(parts) >= 2:
                    try:
                        block_idx = int(parts[1])
                    except ValueError:
                        block_idx = None

                    if block_idx is not None and block_idx > target_block_idx:
                        continue

            if any(name.endswith(target) for target in self.target_modules):
                lora_module = LoRALinear(
                    module,
                    r=self.lora_r,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                )
                _replace_module(self.timm_model, name, lora_module)
                replaced.append(name)

        if len(replaced) == 0:
            available_linear_modules = [
                name for name, module in self.timm_model.named_modules()
                if isinstance(module, nn.Linear)
            ]

            preview = "\n".join(available_linear_modules[:80])

            raise RuntimeError(
                "No LoRA target modules were found. "
                f"target_modules={self.target_modules}. "
                "Check timm_model.named_modules() for the actual module names.\n"
                "First available Linear modules are:\n"
                f"{preview}"
            )

        self.lora_module_names = replaced

    def _check_intermediate_support(self):
        """Check that the timm model exposes what we need."""
        if not hasattr(self.timm_model, "blocks"):
            raise RuntimeError(
                "LoRAIntermediateTIMMBackbone expects a timm ViT/EVA-like model "
                "with attribute 'blocks'. "
                f"Got model type: {type(self.timm_model)}"
            )

        if len(self.timm_model.blocks) == 0:
            raise RuntimeError("timm_model.blocks is empty.")

        if not hasattr(self.timm_model, "forward_intermediates"):
            raise RuntimeError(
                "This timm model does not expose forward_intermediates(...). "
                "For DINOv3/EVA this method should exist. "
                "Use a timm version/model that supports intermediate features, "
                "or implement a model-specific manual forward."
            )

        _ = self._resolve_block_index()

    def _resolve_block_index(self) -> int:
        """Resolve possibly negative out_block_index to a concrete block index."""
        num_blocks = len(self.timm_model.blocks)

        idx = self.out_block_index
        if idx < 0:
            idx = num_blocks + idx

        if idx < 0 or idx >= num_blocks:
            raise ValueError(
                f"out_block_index={self.out_block_index} is invalid for "
                f"{num_blocks} Transformer blocks. Valid examples: "
                f"-1 last, -2 penultimate, or 0..{num_blocks - 1}."
            )

        return idx

    def train(self, mode=True):
        super().train(mode)

        # Keep frozen DINO/EVA base in eval mode.
        # This disables possible dropout/stochastic-depth behavior in the
        # frozen pretrained backbone.
        if self.frozen_base:
            self.timm_model.eval()

            # LoRA modules should still follow the requested mode.
            # This matters if lora_dropout > 0.
            for module in self.timm_model.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)

        return self

    def _extract_intermediate_from_result(self, result):
        """Normalize different possible forward_intermediates return formats.

        For timm models with intermediates_only=True, the result is usually a
        list/tuple of intermediate tensors.

        We requested exactly one index, so we expect exactly one feature map.
        """

        # Most likely case:
        # result = [spatial]
        if isinstance(result, (list, tuple)):
            if len(result) == 1 and torch.is_tensor(result[0]):
                return result[0]

            # Some implementations may ignore intermediates_only and return:
            # result = (final, intermediates)
            if len(result) == 2:
                first, second = result

                if isinstance(second, (list, tuple)):
                    if len(second) != 1:
                        raise RuntimeError(
                            "Expected exactly one intermediate feature, got "
                            f"{len(second)}."
                        )
                    if not torch.is_tensor(second[0]):
                        raise TypeError(
                            "Expected intermediate feature to be a tensor, got "
                            f"{type(second[0])}."
                        )
                    return second[0]

                if torch.is_tensor(second):
                    return second

            raise RuntimeError(
                "Could not interpret forward_intermediates(...) result. "
                f"Got type={type(result)} with length={len(result)}."
            )

        if torch.is_tensor(result):
            return result

        raise TypeError(
            "Expected forward_intermediates(...) to return a tensor or a list/tuple "
            f"of tensors, got {type(result)}."
        )

    def _forward_intermediate_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Use timm's own forward_intermediates implementation.

        This is preferred over manually calling patch_embed, _pos_embed, blocks,
        etc., because EVA/DINOv3 models may return and use RoPE internally:

            x, rot_pos_embed = self._pos_embed(x)
            x = blk(x, rope=rot_pos_embed)

        forward_intermediates handles this correctly.
        """

        # We pass the original out_block_index, not the resolved positive index,
        # because timm supports negative indices in normal usage. If a timm
        # version does not, use [self._resolve_block_index()] instead.
        kwargs = dict(
            indices=[self.out_block_index],
            norm=self.apply_output_norm,
            output_fmt="NCHW",
            intermediates_only=True,
        )

        result = _call_with_supported_kwargs(
            self.timm_model.forward_intermediates,
            x,
            **kwargs,
        )

        spatial = self._extract_intermediate_from_result(result)

        if not torch.is_tensor(spatial):
            raise TypeError(
                f"Expected intermediate spatial feature to be a tensor, got {type(spatial)}"
            )

        if spatial.ndim != 4:
            raise ValueError(
                "Expected intermediate feature in NCHW format with shape "
                f"(B,C,H,W), got {tuple(spatial.shape)}. "
                "This likely means output_fmt='NCHW' was not supported by this "
                "timm forward_intermediates implementation."
            )

        return spatial

    def forward(self, x):
        # Important:
        # Do NOT wrap this in torch.no_grad().
        # Even though the base model is frozen, gradients must flow through
        # the computation graph into the LoRA adapters.
        spatial = self._forward_intermediate_spatial(x)

        b, c, h, w = spatial.shape

        expected_h = x.shape[-2] // self.patch_size
        expected_w = x.shape[-1] // self.patch_size

        if h != expected_h or w != expected_w:
            raise ValueError(
                "Intermediate feature map has unexpected spatial shape. "
                f"Got {(h, w)}, expected {(expected_h, expected_w)} for "
                f"input shape {tuple(x.shape)} and patch_size={self.patch_size}."
            )

        return (spatial,)

    def get_parameter_summary(self):
        """Return total/trainable parameter counts for the backbone."""
        total = 0
        trainable = 0

        for _, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        frozen = total - trainable
        ratio = 100.0 * trainable / total if total > 0 else 0.0

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_ratio": ratio,
        }

    def get_trainable_parameter_names(self):
        """Return names of trainable backbone parameters."""
        return [
            name for name, p in self.named_parameters()
            if p.requires_grad
        ]

    def log_lora_summary(self):
        """Log which modules were adapted and how many parameters are trainable."""
        summary = self.get_parameter_summary()
        trainable_names = self.get_trainable_parameter_names()

        self._log_info(
            "Injected LoRA adapters into "
            f"{len(self.lora_module_names)} modules: "
            f"{self.lora_module_names}"
        )

        self._log_info(
            "Backbone parameter summary: "
            f"trainable={summary['trainable']:,}, "
            f"frozen={summary['frozen']:,}, "
            f"total={summary['total']:,}, "
            f"trainable_ratio={summary['trainable_ratio']:.4f}%"
        )

        self._log_info(
            "Trainable backbone parameters:\n"
            + "\n".join(f"  {name}" for name in trainable_names)
        )

    def log_intermediate_summary(self):
        """Log which Transformer block is used as output."""
        num_blocks = len(self.timm_model.blocks)
        resolved_idx = self._resolve_block_index()

        self._log_info(
            "Intermediate feature extraction via timm.forward_intermediates: "
            f"out_block_index={self.out_block_index} "
            f"resolved to block {resolved_idx}/{num_blocks - 1}. "
            f"apply_output_norm={self.apply_output_norm}."
        )

    def print_trainable_parameters(self):
        """Print trainable parameter summary.

        Kept as a convenience method for quick manual debugging.
        """
        summary = self.get_parameter_summary()

        print(
            "Trainable backbone parameters: "
            f"{summary['trainable']:,} / {summary['total']:,} "
            f"({summary['trainable_ratio']:.4f}%)"
        )

        print("Trainable parameter names:")
        for name in self.get_trainable_parameter_names():
            print(f"  {name}")