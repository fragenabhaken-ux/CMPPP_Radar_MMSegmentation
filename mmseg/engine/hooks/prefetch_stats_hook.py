from collections import deque
from typing import Any, List, Optional
import os 

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

import time

def nested_tensor_dict_print(data, indent=0):
    """
    Recursively pretty-print dictionaries and lists.
    If an element has a .shape attribute, print its type and shape.
    """
    # Define the indentation for nested structures
    indent_str = ' ' * indent

    if isinstance(data, dict):
        print(f"{indent_str}{{")
        for key, value in data.items():
            print(f"{indent_str}  {repr(key)}: ", end='')
            nested_tensor_dict_print(value, indent + 4)
        print(f"{indent_str}}}")
    elif isinstance(data, list):
        print(f"{indent_str}[")
        for item in data:
            nested_tensor_dict_print(item, indent + 4)
        print(f"{indent_str}]")
    else:
        if hasattr(data, 'shape'):
            rep = f"shape={data.shape}"
            if hasattr(data, 'dtype'):
                rep += f", dtype={data.dtype}"
            if hasattr(data, 'device'):
                rep += f", device={data.device}"
            print(f"{indent_str}{type(data).__name__} ({rep})")
        else:
            print(f"{indent_str}{repr(data)}")

 

@HOOKS.register_module()
class PrefetchStatsHook(Hook):
    """
    Logs prefetch-related statistics from `data_batch` to TensorBoard.

    Expected keys in each sample / batch:
      - prefetch_unseen
      - prefetch_unseen_count
      - prefetch_idle_time

    Logged scalars:
      - prefetch/unseen_frac
      - prefetch/unseen_count
      - prefetch/idle_time
    """

    def __init__( self, ):
        self.rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("PROCID", "0")))
        self.start_time = time.time()
        self.last_time = time.time()

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[Any] = None,
        outputs: Optional[Any] = None,
    ) -> None:
        #if not is_main_process():
        #    return

        if data_batch is None:
            print("WARNING: PrefetchStatsHook can't work without a data batch")
            return
        try: 
            unseen_vals = [sample.prefetch_unseen for sample in data_batch['data_samples']]
            unseen_count_vals = [sample.prefetch_unseen_count for sample in data_batch['data_samples']]
            idle_time_vals = [sample.prefetch_idle_time for sample in data_batch['data_samples']]

            runner.visualizer.add_scalar(name=f'prefetch_unseen_frac/{self.rank}', value=sum(unseen_vals)/len(unseen_vals), step=runner.iter)
            runner.visualizer.add_scalar(name=f'prefetch_unseen_count/{self.rank}', value=sum(unseen_count_vals)/len(unseen_count_vals), step=runner.iter)
            runner.visualizer.add_scalar(name=f'prefetch_idle_time/{self.rank}', value=sum(idle_time_vals)/len(idle_time_vals), step=runner.iter)
            
            new_time = time.time()
            runner.visualizer.add_scalar(name=f'total_time', value=new_time-self.start_time, step=runner.iter)
            runner.visualizer.add_scalar(name=f'step_time', value=new_time-self.last_time, step=runner.iter)
            self.last_time = new_time
        except Exception as e:
            print(f"WARNING: PrefetchStatsHook couldn't read batch of type {data_batch.__class__.__name__}: {e.__class__.__name__}: {e}")
            try:
                print(f"         data_samples[0] has attr:", [p for p in dir(data_batch["data_samples"][0]) if not str(p).startswith("_")] )
            except Exception:
                print("          dir(data_batch[\"data_samples\"][0]) din't work either")
                pass
