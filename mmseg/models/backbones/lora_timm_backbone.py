from __future__ import annotations

try:
    import timm
except ImportError:
    timm = None

import math
import logging
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
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 0.0

        # Freeze original pretrained projection.
        for p in self.linear.parameters():
            p.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
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


@MODELS.register_module()
class LoRATIMMBackbone(BaseModule):
    """Frozen timm DINOv3 ViT backbone with trainable LoRA adapters.

    Loads a timm DINOv3 ViT with a local checkpoint, freezes the base model,
    injects LoRA adapters into selected Linear layers, calls forward_features(),
    removes prefix tokens, and returns a spatial patch feature map.

    For input (B, 3, 256, 512) and patch_size=16, the output is:
        (B, 384, 16, 32)
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
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.frozen_base = frozen_base
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = tuple(target_modules)
        self.verbose = verbose

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

        if self.verbose:
            self.log_lora_summary()

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

        For timm ViT, typical module names are e.g.
            blocks.0.attn.qkv
            blocks.0.attn.proj
            blocks.1.attn.qkv
            ...
        """

        replaced = []

        # list(...) is important because we modify modules while iterating.
        for name, module in list(self.timm_model.named_modules()):
            if not isinstance(module, nn.Linear):
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

            preview = "\n".join(available_linear_modules[:50])

            raise RuntimeError(
                "No LoRA target modules were found. "
                f"target_modules={self.target_modules}. "
                "Check timm_model.named_modules() for the actual module names.\n"
                "First available Linear modules are:\n"
                f"{preview}"
            )

        self.lora_module_names = replaced

    def train(self, mode=True):
        super().train(mode)

        # Keep frozen DINO base in eval mode.
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

    def forward(self, x):
        # Important:
        # Do NOT wrap this in torch.no_grad().
        # Even though the base model is frozen, gradients must flow through
        # the computation graph into the LoRA adapters.
        features = self.timm_model.forward_features(x)

        if not torch.is_tensor(features):
            raise TypeError(
                f"Expected forward_features to return a tensor, got {type(features)}"
            )

        if features.ndim != 3:
            raise ValueError(
                f"Expected features of shape (B, N, C), got {tuple(features.shape)}"
            )

        b, n, c = features.shape

        h_patch = x.shape[-2] // self.patch_size
        w_patch = x.shape[-1] // self.patch_size
        n_patch = h_patch * w_patch

        if n < n_patch:
            raise ValueError(
                f"Got only {n} tokens, but expected at least {n_patch} patch tokens."
            )

        # DINOv3 returned 517 tokens in your test:
        # 512 patch tokens + 5 prefix tokens.
        # The patch tokens are at the end.
        patch_tokens = features[:, -n_patch:, :]

        spatial = patch_tokens.transpose(1, 2).contiguous().view(
            b, c, h_patch, w_patch
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