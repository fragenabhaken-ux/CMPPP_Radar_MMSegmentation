from __future__ import annotations

try:
    import timm
except ImportError:
    timm = None

import torch

from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS

from mmseg.registry import MODELS


@MODELS.register_module()
class FrozenTIMMBackbone(BaseModule):
    """Frozen timm DINOv3 ViT backbone for MMSegmentation.

    Loads a timm DINOv3 ViT with a local checkpoint, calls forward_features(),
    removes prefix tokens, and returns a spatial patch feature map.

    For input (B, 3, 256, 512) and patch_size=16, the output is:
        (B, 384, 16, 32)
    """

    def __init__(
        self,
        model_name='vit_small_patch16_dinov3',
        pretrained=False,
        checkpoint_path='',
        in_channels=3,
        frozen=True,
        patch_size=16,
        init_cfg=None,
        **kwargs,
    ):
        if timm is None:
            raise RuntimeError('timm is not installed')

        super().__init__(init_cfg)

        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = MMENGINE_MODELS.get(kwargs['norm_layer'])

        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.in_channels = in_channels
        self.frozen = frozen
        self.patch_size = patch_size

        # Important: features_only=False.
        # For local DINOv3 checkpoints, features_only=True wraps the model in
        # FeatureGetterNet and causes state_dict prefix mismatches.
        self.timm_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            in_chans=in_channels,
            **kwargs,
        )

        if hasattr(self.timm_model, 'global_pool'):
            self.timm_model.global_pool = None
        if hasattr(self.timm_model, 'fc'):
            self.timm_model.fc = None
        if hasattr(self.timm_model, 'classifier'):
            self.timm_model.classifier = None

        if self.frozen:
            self.freeze()

        if pretrained or checkpoint_path:
            self._is_init = True

    def freeze(self):
        for param in self.timm_model.parameters():
            param.requires_grad = False
        self.timm_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.timm_model.eval()
        return self

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                features = self.timm_model.forward_features(x)
        else:
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

        # DINOv3 returned 517 tokens for our test:
        # 512 patch tokens + 5 prefix tokens.
        # The patch tokens are at the end.
        patch_tokens = features[:, -n_patch:, :]

        spatial = patch_tokens.transpose(1, 2).contiguous().view(
            b, c, h_patch, w_patch
        )

        return (spatial,)