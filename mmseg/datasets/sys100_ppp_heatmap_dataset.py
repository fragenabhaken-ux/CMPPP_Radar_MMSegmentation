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

  - `__len__()`
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
from torch.utils.data import Dataset
from mmseg.registry import DATASETS
from mmengine.dataset import Compose

from aumovio.sys100dataset import Sys100Dataset


@DATASETS.register_module()
class Sys100PPPHeatmapDataset(Dataset):
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
        **kwargs,  # ignore mmengine-specific dataset kwargs
    ):
        self.sys100 = Sys100Dataset(
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
            fovs=fovs,
            strict=strict,
        )
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.n = len(self.sys100)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        if i < 0:
            i = self.n + i
        if i < 0 or i >= self.n:
            raise IndexError(i)

        results = {"idx": int(i)}
        if self.pipeline is not None:
            results = self.pipeline(results)  # must end with PPPPackSegInputs
        return results