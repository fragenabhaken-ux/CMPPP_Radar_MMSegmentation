"""
## Why We Use torch.utils.data.Dataset Instead of mmengine.BaseDataset ##

BaseDataset is designed for annotation-driven datasets that provide a fully 
materialized `data_list`. During initialization, BaseDataset automatically 
calls `full_init()` and `_serialize_data()`, which assume that:

  - `self.data_list` is a real Python list containing all sample dictionaries.
  - The entire list can be serialized and shared across workers.
  - The list supports mutation operations such as `.clear()`.

For a dataset with tens of millions of samples, constructing a list like:

  [{"idx": 0}, ..., {"idx": N-1}]

is infeasible in both memory and initialization time. Attempts to replace the 
list with a lazy sequence object failed because BaseDataset still attempts to 
serialize and mutate `data_list`, leading to runtime errors.

In short, BaseDataset enforces full dataset materialization and serialization 
at initialization time, which is incompatible with our lazy, index-only design.

We therefore inherit from `torch.utils.data.Dataset`, which only requires the 
minimal PyTorch dataset contract:

  - `__len__()
  - `__getitem__(i)`

We generate samples lazily by returning `{"idx": i}` in `__getitem__` and 
manually applying the MMSegmentation pipeline via `Compose(pipeline)`.

This results in:

  - O(1) initialization time
  - O(1) memory usage with respect to dataset size
  - No serialization overhead
  - Full compatibility with MMSegmentation training

This design allows the dataset to scale cleanly to tens of millions of samples.


## Why we need `self.pipeline = Compose(pipeline)` ##

When we switched from `mmengine.BaseDataset` to `torch.utils.data.Dataset`,
we also lost the automatic pipeline handling that BaseDataset provides.

With BaseDataset:
  - MMEngine internally calls `load_data_list()`
  - In `__getitem__`, it automatically applies the configured `pipeline`
  - The final output already contains {"inputs": ..., "data_samples": ...}

After switching to torch.utils.data.Dataset:
  - MMEngine no longer wraps our dataset with BaseDataset logic
  - The pipeline is NOT applied automatically
  - `__getitem__` must explicitly apply the transforms

Therefore we store:

  self.pipeline = Compose(pipeline)

`Compose` turns the list of transform configs from the MMSeg config into a
callable transform pipeline.

Then inside `__getitem__`, we manually execute:

  results = self.pipeline(results)

Without this step, the dataset would only return {"idx": i}, the transforms
would never run, and the model would crash with:

  KeyError: 'inputs'

because the required `PackSegInputs` transform would never be executed.
"""
from __future__ import annotations

from typing import Any, Dict
import os
import time

import h5py
import numpy as np

from torch.utils.data import Dataset
from mmseg.registry import DATASETS
from mmengine.dataset import Compose

from aumovio.sys100dataset import Sys100Dataset


def get_rank():
   return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))


@DATASETS.register_module()
class Sys100PPPHeatmapDatasetDemoVid(Dataset):
    def __init__(
        self,
        sys100_root: str,
        pipeline=None,
        out_img_size=(544, 1792),
        undistort=True,
        fovs=(110, 48.4),
        strict=False,
        source_is_summary_csv: bool = False,
        max_files=None,
        target_dataset_size=None,

        # Optional demo-sequence mode.
        # If demo_h5_path is None, the dataset behaves exactly as before.
        demo_h5_path=None,
        demo_frame_start=None,
        demo_frame_end=None,
        demo_frame_stride=1,
        demo_use_video_frame_counter=True,

        # Optional stream filter for h5 files that reference multiple videos,
        # e.g. EVEN / ODD streams.
        # If None, no video stream filtering is applied.
        demo_video_name_contains=None,

        **kwargs,  # ignore mmengine-specific dataset kwargs
    ):
        self.pipeline = Compose(pipeline) if pipeline is not None else None

        self.demo_indices = None

        # ------------------------------------------------------------
        # Optional demo mode.
        # This is only active if demo_h5_path is explicitly provided.
        # Otherwise, the old target_dataset_size / Sys100Dataset logic below
        # is untouched.
        # ------------------------------------------------------------
        if demo_h5_path is not None:
            self.demo_indices = self._build_demo_indices_from_h5(
                h5_path=str(demo_h5_path),
                frame_start=demo_frame_start,
                frame_end=demo_frame_end,
                frame_stride=demo_frame_stride,
                use_video_frame_counter=demo_use_video_frame_counter,
                video_name_contains=demo_video_name_contains,
            )
            self.n = len(self.demo_indices)

            print(
                f"[rank {get_rank()}] Sys100PPPHeatmapDataset demo mode: "
                f"{self.n} samples from {demo_h5_path}, "
                f"frame_start={demo_frame_start}, "
                f"frame_end={demo_frame_end}, "
                f"frame_stride={demo_frame_stride}, "
                f"use_video_frame_counter={demo_use_video_frame_counter}, "
                f"video_name_contains={demo_video_name_contains}",
                flush=True,
            )

        elif target_dataset_size:
          self.n = target_dataset_size

        else:
          sys100 = Sys100Dataset(
              source=str(sys100_root),
              source_is_summary_csv=bool(source_is_summary_csv),
              max_files=max_files,
              target_dataset_size=target_dataset_size,
              extract_video=True,
              out_img_size=out_img_size,
              undistort=undistort,
              project_radar=False,
              plot_radar=False,
              display_stats=False,
              return_metadata=True,
              fovs=fovs,
              strict=strict,
          )
          self.n = len(sys100)

          # self.time_nbr = 0
          # self.time_avg = 0
          #del sys100
        # print(f"[rank {get_rank()}] Sys100PPPHeatmapDataset init", flush=True)

    @staticmethod
    def _decode_h5_attr_string(x):
        if isinstance(x, bytes):
            return x.decode("utf-8")
        return str(x)

    @staticmethod
    def _build_demo_indices_from_h5(
        h5_path: str,
        frame_start,
        frame_end,
        frame_stride=1,
        use_video_frame_counter=True,
        video_name_contains=None,
    ):
        frame_stride = int(frame_stride)

        if frame_stride <= 0:
            raise ValueError("demo_frame_stride must be >= 1")

        if frame_start is None:
            raise ValueError("demo_frame_start must be set when demo_h5_path is used")

        if frame_end is None:
            raise ValueError("demo_frame_end must be set when demo_h5_path is used")

        frame_start = int(frame_start)
        frame_end = int(frame_end)

        if frame_end < frame_start:
            raise ValueError(
                f"demo_frame_end must be >= demo_frame_start, got "
                f"{frame_end} < {frame_start}"
            )

        with h5py.File(h5_path, "r") as f:
            valid = int(f.attrs["valid_entries"])
            recordings = f["recordings"]
            rec = recordings[:valid]

            if use_video_frame_counter:
                frame_numbers = rec["VideoFrameCounter"].astype(np.int64)
            else:
                frame_numbers = np.arange(valid, dtype=np.int64)

            video_ids = rec["VideoId"].astype(np.int64)

            if "VideoId2Path" in f.attrs:
                video_paths = [
                    Sys100PPPHeatmapDatasetDemoVid._decode_h5_attr_string(x)
                    for x in f.attrs["VideoId2Path"]
                ]
            else:
                video_paths = []

        mask = (
            (frame_numbers >= frame_start)
            & (frame_numbers <= frame_end)
            & ((frame_numbers - frame_start) % frame_stride == 0)
        )

        if video_name_contains is not None:
            video_name_contains = str(video_name_contains)

            keep_video_ids = [
                video_id
                for video_id, video_path in enumerate(video_paths)
                if video_name_contains in os.path.basename(video_path)
                or video_name_contains in video_path
            ]

            if len(keep_video_ids) == 0:
                raise RuntimeError(
                    f"No video stream matched demo_video_name_contains={video_name_contains!r}. "
                    f"Available VideoId2Path entries: {video_paths}"
                )

            mask &= np.isin(video_ids, np.asarray(keep_video_ids, dtype=np.int64))

        indices = np.nonzero(mask)[0].astype(np.int64)

        if len(indices) == 0:
            raise RuntimeError(
                f"Demo frame selection produced 0 samples. "
                f"h5_path={h5_path}, "
                f"frame_start={frame_start}, "
                f"frame_end={frame_end}, "
                f"frame_stride={frame_stride}, "
                f"use_video_frame_counter={use_video_frame_counter}, "
                f"video_name_contains={video_name_contains}"
            )

        # Keep chronological order by actual frame number.
        order = np.argsort(frame_numbers[indices])
        indices = indices[order]

        return indices.tolist()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        if i < 0:
            i = self.n + i
        if i < 0 or i >= self.n:
            raise IndexError(i)

        # Old behavior:
        #   results = {"idx": int(i)}
        #
        # Demo behavior:
        #   local demo index i -> real h5 sample index demo_indices[i]
        if self.demo_indices is None:
            results = {"idx": int(i)}
        else:
            results = {
                "idx": int(self.demo_indices[i]),
                "demo_idx": int(i),
            }

        if self.pipeline is not None:

            # t0 = time.perf_counter()
            results = self.pipeline(results)  # must end with PPPPackSegInputs
            # t1 = time.perf_counter()

            # self.time_nbr += 1
            # self.time_avg += ((t1-t0) - self.time_avg)/self.time_nbr
            # if self.time_nbr % 60 == 0:
            #   print(f"[rank {get_rank()}] pipeline_time = {self.time_avg:.4f}s [avg over {self.time_nbr} steps]", flush=True)
        return results