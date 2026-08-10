from __future__ import annotations

from collections.abc import Sequence

import torch
from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class PolarPPPDataPreProcessor(BaseDataPreprocessor):
    """Batch images without forcing variable-length radar points into masks.

    ``data_samples`` remain a list of ``SegDataSample`` objects. Each sample
    contains ``gt_radar_points`` with variable-length polar/xy/meta tensors.
    """

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        bgr_to_rgb: bool = False,
        pad_val: float = 0.0,
        size: tuple[int, int] | None = None,
        non_blocking: bool = False,
    ) -> None:
        super().__init__(non_blocking=non_blocking)

        if (mean is None) != (std is None):
            raise ValueError('mean and std must either both be set or both be None.')

        self.bgr_to_rgb = bool(bgr_to_rgb)
        self.pad_val = float(pad_val)
        self.size = None if size is None else tuple(int(v) for v in size)

        if mean is None:
            self.register_buffer('mean', torch.empty(0), persistent=False)
            self.register_buffer('std', torch.empty(0), persistent=False)
        else:
            mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
            if torch.any(std_t == 0):
                raise ValueError('std must not contain zero.')
            self.register_buffer('mean', mean_t, persistent=False)
            self.register_buffer('std', std_t, persistent=False)

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = data['inputs']

        if isinstance(inputs, (list, tuple)):
            if not inputs:
                raise ValueError('Received an empty input batch.')
            inputs = torch.stack(list(inputs), dim=0)
        elif not isinstance(inputs, torch.Tensor):
            raise TypeError(f'inputs must be a Tensor or sequence of Tensors, got {type(inputs)}.')

        if inputs.ndim != 4:
            raise ValueError(
                'Expected batched inputs with shape [B, C, H, W], '
                f'got {tuple(inputs.shape)}.'
            )

        inputs = inputs.float()
        if self.bgr_to_rgb:
            if inputs.shape[1] != 3:
                raise ValueError('bgr_to_rgb requires three image channels.')
            inputs = inputs[:, [2, 1, 0], :, :]

        if self.mean.numel() > 0:
            inputs = (inputs - self.mean.to(inputs)) / self.std.to(inputs)

        if self.size is not None and tuple(inputs.shape[-2:]) != self.size:
            raise ValueError(
                'All inputs must already have the configured fixed size. '
                f'Got {tuple(inputs.shape[-2:])}, expected {self.size}.'
            )

        return dict(
            inputs=inputs,
            data_samples=data.get('data_samples', None),
        )
