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

    The original linear layer is frozen. Only the low-rank update is trainable.
    The LoRA branch is initialized with zero contribution, so the wrapped layer
    initially behaves exactly like the frozen base layer.
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

        for p in self.linear.parameters():
            p.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        if self.r > 0:
            self.lora_A = nn.Linear(in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, out_features, bias=False)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
class LoRAMultiLayerTIMMBackbone(BaseModule):
    """Frozen timm DINOv3/EVA ViT backbone with LoRA and multi-layer fusion.

    It extracts several ViT block outputs, projects each selected feature to a
    smaller channel count, concatenates them, fuses them, and returns one NCHW
    feature map for the existing neck/head.

    Recommended stronger setting for ViT-S/16:

        out_indices=(2, 5, 8, 11)
        each feature:      (B, 384, H_patch, W_patch)
        per-layer project: (B, 128, H_patch, W_patch)
        concat:            (B, 512, H_patch, W_patch)
        final fuse:         (B, 384, H_patch, W_patch)

    Output:

        tuple(fused_spatial,)
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
        out_indices=(2, 5, 8, 11),
        in_feature_channels=384,
        per_layer_out_channels=128,
        out_channels=384,
        apply_output_norm=True,
        use_fuse_conv=True,
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

        self.out_indices = tuple(int(i) for i in out_indices)
        self.in_feature_channels = int(in_feature_channels)
        self.per_layer_out_channels = int(per_layer_out_channels)
        self.out_channels = int(out_channels)
        self.apply_output_norm = bool(apply_output_norm)
        self.use_fuse_conv = bool(use_fuse_conv)
        self.verbose = bool(verbose)

        if len(self.out_indices) == 0:
            raise ValueError("out_indices must contain at least one block index.")

        if self.per_layer_out_channels <= 0:
            raise ValueError(
                f"per_layer_out_channels must be positive, got "
                f"{self.per_layer_out_channels}."
            )

        if self.out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive, got {self.out_channels}."
            )

        self.fusion_in_channels = (
            self.per_layer_out_channels * len(self.out_indices)
        )

        if not self.use_fuse_conv and self.fusion_in_channels != self.out_channels:
            raise ValueError(
                "If use_fuse_conv=False, the concatenated channel count must "
                "already equal out_channels. "
                f"Got fusion_in_channels={self.fusion_in_channels}, "
                f"out_channels={self.out_channels}. "
                "Set use_fuse_conv=True to allow a final projection."
            )

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

        self.layer_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.in_feature_channels,
                    self.per_layer_out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(1, self.per_layer_out_channels),
                nn.GELU(),
            )
            for _ in self.out_indices
        ])

        if self.use_fuse_conv:
            self.fuse = nn.Sequential(
                nn.Conv2d(
                    self.fusion_in_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(1, self.out_channels),
                nn.GELU(),
            )
        else:
            self.fuse = nn.Identity()

        if self.verbose:
            self.log_lora_summary()
            self.log_intermediate_summary()

        if pretrained or checkpoint_path:
            self._is_init = True

    def _log_info(self, msg: str):
        logging.getLogger(__name__).info(msg)

    def freeze_base(self):
        """Freeze all parameters of the timm model before LoRA injection."""
        for param in self.timm_model.parameters():
            param.requires_grad = False

    def _resolve_single_index(self, idx: int) -> int:
        num_blocks = len(self.timm_model.blocks)
        resolved = int(idx)

        if resolved < 0:
            resolved = num_blocks + resolved

        if resolved < 0 or resolved >= num_blocks:
            raise ValueError(
                f"out index {idx} is invalid for {num_blocks} Transformer blocks. "
                f"Valid examples: -1 last, -2 penultimate, or 0..{num_blocks - 1}."
            )

        return resolved

    def _resolve_out_indices(self) -> tuple[int, ...]:
        return tuple(self._resolve_single_index(i) for i in self.out_indices)

    def inject_lora(self):
        """Inject LoRA only into blocks that can influence selected outputs."""
        replaced = []
        max_target_block_idx = max(self._resolve_out_indices())

        for name, module in list(self.timm_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            if name.startswith("blocks."):
                parts = name.split(".")
                if len(parts) >= 2:
                    try:
                        block_idx = int(parts[1])
                    except ValueError:
                        block_idx = None

                    if block_idx is not None and block_idx > max_target_block_idx:
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
        if not hasattr(self.timm_model, "blocks"):
            raise RuntimeError(
                "LoRAMultiLayerTIMMBackbone expects a timm ViT/EVA-like model "
                "with attribute 'blocks'. "
                f"Got model type: {type(self.timm_model)}"
            )

        if len(self.timm_model.blocks) == 0:
            raise RuntimeError("timm_model.blocks is empty.")

        if not hasattr(self.timm_model, "forward_intermediates"):
            raise RuntimeError(
                "This timm model does not expose forward_intermediates(...). "
                "Use a timm version/model that supports intermediate features, "
                "or implement a model-specific manual forward."
            )

        _ = self._resolve_out_indices()

    def train(self, mode=True):
        super().train(mode)

        if self.frozen_base:
            self.timm_model.eval()

            for module in self.timm_model.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)

        return self

    def _extract_intermediates_from_result(self, result) -> list[torch.Tensor]:
        n_expected = len(self.out_indices)

        if isinstance(result, (list, tuple)):
            # Most likely with intermediates_only=True:
            # result = [f2, f5, f8, f11]
            if len(result) == n_expected and all(torch.is_tensor(x) for x in result):
                return list(result)

            # Some timm variants may return:
            # result = (final, intermediates)
            if len(result) == 2:
                _, second = result

                if isinstance(second, (list, tuple)):
                    if len(second) != n_expected:
                        raise RuntimeError(
                            f"Expected {n_expected} intermediate features, "
                            f"got {len(second)}."
                        )

                    if not all(torch.is_tensor(x) for x in second):
                        raise TypeError(
                            "Expected all intermediate features to be tensors."
                        )

                    return list(second)

            raise RuntimeError(
                "Could not interpret forward_intermediates(...) result. "
                f"Got type={type(result)} with length={len(result)}."
            )

        raise TypeError(
            "Expected forward_intermediates(...) to return a list/tuple of tensors, "
            f"got {type(result)}."
        )

    def _forward_intermediate_spatials(self, x: torch.Tensor) -> list[torch.Tensor]:
        kwargs = dict(
            indices=list(self._resolve_out_indices()),
            norm=self.apply_output_norm,
            output_fmt="NCHW",
            intermediates_only=True,
        )

        result = _call_with_supported_kwargs(
            self.timm_model.forward_intermediates,
            x,
            **kwargs,
        )

        spatials = self._extract_intermediates_from_result(result)

        expected_h = x.shape[-2] // self.patch_size
        expected_w = x.shape[-1] // self.patch_size

        for i, spatial in zip(self.out_indices, spatials):
            if spatial.ndim != 4:
                raise ValueError(
                    f"Expected intermediate feature from block {i} in NCHW format, "
                    f"got shape {tuple(spatial.shape)}."
                )

            _, c, h, w = spatial.shape

            if c != self.in_feature_channels:
                raise ValueError(
                    f"Intermediate feature from block {i} has {c} channels, "
                    f"expected {self.in_feature_channels}."
                )

            if h != expected_h or w != expected_w:
                raise ValueError(
                    f"Intermediate feature from block {i} has unexpected spatial shape. "
                    f"Got {(h, w)}, expected {(expected_h, expected_w)} for "
                    f"input shape {tuple(x.shape)} and patch_size={self.patch_size}."
                )

        return spatials

    def forward(self, x):
        # Do not use torch.no_grad().
        # Gradients must flow into LoRA adapters and into the projection/fusion layers.
        spatials = self._forward_intermediate_spatials(x)

        projected = [
            proj(feat)
            for proj, feat in zip(self.layer_projs, spatials)
        ]

        fused = torch.cat(projected, dim=1)
        fused = self.fuse(fused)

        return (fused,)

    def get_parameter_summary(self):
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
        return [
            name for name, p in self.named_parameters()
            if p.requires_grad
        ]

    def log_lora_summary(self):
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
        num_blocks = len(self.timm_model.blocks)
        resolved = self._resolve_out_indices()

        self._log_info(
            "Multi-layer intermediate feature extraction via timm.forward_intermediates: "
            f"out_indices={self.out_indices}, "
            f"resolved={resolved}, "
            f"num_blocks={num_blocks}, "
            f"apply_output_norm={self.apply_output_norm}, "
            f"per_layer_out_channels={self.per_layer_out_channels}, "
            f"fusion_in_channels={self.fusion_in_channels}, "
            f"out_channels={self.out_channels}, "
            f"use_fuse_conv={self.use_fuse_conv}."
        )

    def print_trainable_parameters(self):
        summary = self.get_parameter_summary()

        print(
            "Trainable backbone parameters: "
            f"{summary['trainable']:,} / {summary['total']:,} "
            f"({summary['trainable_ratio']:.4f}%)"
        )

        print("Trainable parameter names:")
        for name in self.get_trainable_parameter_names():
            print(f"  {name}")