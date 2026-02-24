from __future__ import annotations

from typing import Any, Dict, List
from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

# This must be importable in your env:
# Either pip-install your aumovio package, or export PYTHONPATH to include it.
from aumovio.sys100dataset import Sys100Dataset


@DATASETS.register_module()
class Sys100PPPHeatmapDataset(BaseDataset):
    """
    This class only does sample indexing!
    Sys100PPPHeatmapDataset turns a SYS100 video+radar sequence into a list of integer indices, 
    and lets the pipeline decide how to materialize each index into data. I.e. Here we simply define 
    what samples exist. It is the transform-pipeline's job is to define how samples are loaded.
    returns a list of samples
    [
    {"idx": 0},
    {"idx": 1},
    {"idx": 2},
    ...
    ]
    Then the data loader can choose an index
    sample_id = 123
    results = data_list[123]  # {"idx": 123}

    Stide argument. I.e. not take every frame a.e. for test run.
    """

    def __init__(
        self,
        sys100_root: str,
        out_img_size=(544, 1792),
        undistort=True,
        fovs=(110, 48.4),
        frame_stride=1,
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
        self.frame_stride = frame_stride

        super().__init__(**kwargs)

    def load_data_list(self):
        return [
            {"idx": i}
            for i in range(0, len(self.sys100), self.frame_stride)
        ]
