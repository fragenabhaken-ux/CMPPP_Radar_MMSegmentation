from __future__ import annotations

from typing import Any, Dict, List
from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

# This must be importable in your env:
# Either pip-install your aumovio package, or export PYTHONPATH to include it.
from aumovio.sys100dataset import Sys100Dataset


@DATASETS.register_module()
class Sys100PPPHeatmapDataset(BaseDataset):

    def __init__(
        self,
        sys100_root: str,
        out_img_size=(544, 1792),
        undistort=True,
        fovs=(110, 48.4),
        strict=False,
        **kwargs,
    ):
        self.sys100 = Sys100Dataset(
            source=str(sys100_root),
            extract_video=True,
            out_img_size=out_img_size,
            undistort=undistort,
            project_radar=False,
            plot_radar=False,
            display_stats=False,
            fovs=fovs,
            strict=strict,
        )

        super().__init__(**kwargs)

    def load_data_list(self):
        return [{"idx": i} for i in range(len(self.sys100))]
