from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

# Must be importable in your env:
# from truckscenes import TruckScenes


@DATASETS.register_module()
class TruckScenesPPPHeatmapDataset(BaseDataset):
    def __init__(self,
                 version="v1.1-mini",
                 dataroot=None,
                 scene_tokens=None,
                 frame_stride=1,
                 *args,
                 **kwargs):
        self.truckscenes_version = version
        self.dataroot = dataroot
        self.scene_tokens = scene_tokens  # optional filter
        self.frame_stride = int(frame_stride)

        super().__init__(*args, **kwargs)

    def _iterate_scene_samples(self, trucksc, scene_token):
        scene = trucksc.get("scene", scene_token)
        token = scene["first_sample_token"]
        while token:
            yield token
            sample = trucksc.get("sample", token)
            token = sample["next"]

    def load_data_list(self):
        # IMPORTANT: create TruckScenes only locally here
        trucksc = TruckScenes(
            version=self.truckscenes_version,
            dataroot=self.dataroot,
            verbose=False
        )

        # choose scenes
        if self.scene_tokens is None:
            scenes = [s["token"] for s in trucksc.scene]
        else:
            scenes = list(self.scene_tokens)

        tokens = []
        for st in scenes:
            tokens.extend(list(self._iterate_scene_samples(trucksc, st)))

        # apply stride
        if self.frame_stride > 1:
            tokens = tokens[::self.frame_stride]

        # return mmseg style dicts
        return [{"sample_token": t} for t in tokens]