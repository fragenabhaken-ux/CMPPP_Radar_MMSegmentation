from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class PolarPPPDataPreProcessor(BaseDataPreprocessor):
    """Preprocess images for polar point-list PPP training.

    Unlike SegDataPreProcessor, this class does not access gt_sem_seg.
    It keeps variable-length gt_radar_points inside each data sample.
    """

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        size: tuple[int, int] | None = None,
        size_divisor: int | None = None,
        pad_val: float = 0.0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
    ) -> None:
        super().__init__()

        if bgr_to_rgb and rgb_to_bgr:
            raise ValueError("Only one channel conversion may be enabled.")

        if (mean is None) != (std is None):
            raise ValueError("mean and std must either both be set or both be None.")

        self.size = tuple(size) if size is not None else None
        self.size_divisor = size_divisor
        self.pad_val = float(pad_val)
        self.channel_conversion = bgr_to_rgb or rgb_to_bgr

        if mean is not None:
            self.register_buffer(
                "mean",
                torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor(std, dtype=torch.float32).view(-1, 1, 1),
                persistent=False,
            )
        else:
            self.mean = None
            self.std = None

    def _target_shape(self, inputs: list[torch.Tensor]) -> tuple[int, int]:
        if self.size is not None:
            return self.size

        h = max(x.shape[-2] for x in inputs)
        w = max(x.shape[-1] for x in inputs)

        if self.size_divisor is not None:
            d = self.size_divisor
            h = (h + d - 1) // d * d
            w = (w + d - 1) // d * d

        return h, w

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)

        inputs = data["inputs"]
        data_samples = data.get("data_samples")

        if torch.is_tensor(inputs):
            inputs = list(inputs)

        target_h, target_w = self._target_shape(inputs)
        processed_inputs = []

        for img in inputs:
            img = img.float()

            if self.channel_conversion:
                img = img[[2, 1, 0], ...]

            if self.mean is not None:
                img = (img - self.mean) / self.std

            h, w = img.shape[-2:]
            if h > target_h or w > target_w:
                raise ValueError(
                    f"Input shape {(h, w)} exceeds target shape "
                    f"{(target_h, target_w)}."
                )

            pad_h = target_h - h
            pad_w = target_w - w

            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h), value=self.pad_val)

            processed_inputs.append(img)

        inputs = torch.stack(processed_inputs, dim=0)

        if data_samples is not None:
            for sample, original_img in zip(data_samples, processed_inputs):
                sample.set_metainfo(
                    dict(
                        batch_input_shape=(target_h, target_w),
                        pad_shape=(target_h, target_w),
                    )
                )

        return dict(
            inputs=inputs,
            data_samples=data_samples,
        )