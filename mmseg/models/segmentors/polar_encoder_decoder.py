from __future__ import annotations

import torch
from mmengine.structures import PixelData
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


@MODELS.register_module()
class PolarEncoderDecoder(EncoderDecoder):
    """EncoderDecoder variant that does not resize polar output to image size."""

    def postprocess_result(
        self,
        seg_logits: torch.Tensor,
        data_samples: list[SegDataSample] | None = None,
    ) -> list[SegDataSample]:
        batch_size = seg_logits.shape[0]

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
        elif len(data_samples) != batch_size:
            raise ValueError(
                'Number of data samples must equal batch size: '
                f'{len(data_samples)} != {batch_size}.'
            )

        for batch_idx, data_sample in enumerate(data_samples):
            data_sample.set_data(
                dict(seg_logits=PixelData(data=seg_logits[batch_idx]))
            )

        return data_samples
