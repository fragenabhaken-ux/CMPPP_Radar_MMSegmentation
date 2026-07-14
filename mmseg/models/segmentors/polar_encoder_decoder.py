# mmseg/models/segmentors/polar_encoder_decoder.py

from __future__ import annotations

import torch
from mmengine.structures import PixelData

from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


@MODELS.register_module()
class PolarEncoderDecoder(EncoderDecoder):
    """EncoderDecoder variant that preserves the polar output resolution."""

    def postprocess_result(
        self,
        seg_logits: torch.Tensor,
        data_samples=None,
    ):
        batch_size = seg_logits.shape[0]

        if data_samples is None:
            data_samples = [
                SegDataSample()
                for _ in range(batch_size)
            ]

        for batch_id, data_sample in enumerate(data_samples):
            # Keep [1, H_polar, W_polar], do not resize to camera resolution.
            data_sample.set_data(
                dict(
                    seg_logits=PixelData(
                        data=seg_logits[batch_id]
                    )
                )
            )

        return data_samples