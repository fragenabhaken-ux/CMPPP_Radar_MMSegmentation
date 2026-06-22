from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from mmseg.datasets.transforms.loading import BaseTransform
from mmseg.registry import TRANSFORMS

from aumovio.prefetcheddataset import AsyncPrefetchedDatasetWrapper
from aumovio.sys100dataset import Sys100Dataset

@TRANSFORMS.register_module()
class LoadSys100PolarSample(BaseTransform):
    """Load one SYS100 sample for polar point-list PPP training.

    Produces:
        img: HWC numpy float32 in [-1, 1]
        radar_points_polar: Tensor[N, 3], columns r, theta, velocity
        radar_points_xy: Tensor[N, 2], columns x_forward, y_lateral
        radar_points_meta: Tensor[N, 6], optional debug/meta values
    """

    def __init__(
        self,
        sys100_root: str,
        out_img_size: Tuple[int, int] = (544, 1792),
        undistort: bool = True,
        fovs: Tuple[float, float] = (110, 48.4),
        strict: bool = False,
        prefetch: bool = False,
        prefetch_dict: Optional[dict] = None,
        source_is_summary_csv: bool = False,
        max_files=None,
        target_dataset_size=None,
        subsampling_step=None,
        random_subsampling=True,
        seed=None,
        debug_polar_origin: bool = False,
        debug_polar_origin_every: int = 0,
    ) -> None:
        if prefetch_dict is None:
            prefetch_dict = {}

        base_ds = Sys100Dataset(
            source=str(sys100_root),
            source_is_summary_csv=bool(source_is_summary_csv),
            max_files=max_files,
            target_dataset_size=target_dataset_size,
            subsampling_step=subsampling_step,
            random_subsampling=random_subsampling,
            seed=seed,
            extract_video=True,
            out_img_size=out_img_size,
            undistort=undistort,
            project_radar=False,
            plot_radar=False,
            display_stats=False,
            return_metadata=True,
            return_polar_points=True,
            debug_polar_origin=debug_polar_origin,
            debug_polar_origin_every=debug_polar_origin_every,
            fovs=fovs,
            strict=strict,
        )

        if prefetch:
            self.ds = AsyncPrefetchedDatasetWrapper(
                base_dataset=base_ds,
                **prefetch_dict,
            )
        else:
            self.ds = base_ds

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(results["idx"])
        sample = self.ds[idx]

        video_path = sample.get("video_path", None)
        md = sample.get("meta_data", None)

        frame_in_video = None
        if md is not None:
            md_names = getattr(getattr(md, "dtype", None), "names", None)
            if md_names is not None and "VideoFrameCounter" in md_names:
                frame_in_video = int(md["VideoFrameCounter"])

        metainfo = results.setdefault("metainfo", {})
        metainfo.update(
            dict(
                sample_id=f"sys100_{idx:06d}",
                frame_idx=idx,
                global_idx=idx,
                video_path=video_path,
                video_name=os.path.basename(video_path) if video_path else None,
                video_frame_idx=frame_in_video,
            )
        )

        img_t = sample["jpg"]  # [3, H, W], torch tensor in [-1, 1]
        img = img_t.detach().cpu().numpy().transpose(1, 2, 0).astype(
            np.float32,
            copy=False,
        )

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        results["radar_points_polar"] = sample["radar_points_polar"]
        results["radar_points_xy"] = sample["radar_points_xy"]

        if "radar_points_meta" in sample:
            results["radar_points_meta"] = sample["radar_points_meta"]

        if "dopplerRange" in sample:
            results["dopplerRange"] = sample["dopplerRange"]

        if "prefetch_unseen" in sample:
            results["prefetch_unseen"] = sample["prefetch_unseen"]
            results["prefetch_unseen_count"] = sample["prefetch_unseen_count"]
            results["prefetch_idle_time"] = sample["prefetch_idle_time"]

        return results


@TRANSFORMS.register_module()
class PPPPolarRandomFlip(BaseTransform):
    """Horizontally flip image and polar point targets."""

    def __init__(self, prob: float = 0.5, direction: str = "horizontal") -> None:
        self.prob = float(prob)
        self.direction = direction

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.prob:
            return results

        if self.direction != "horizontal":
            raise ValueError(f"Unknown direction: {self.direction}")

        results["img"] = np.flip(results["img"], axis=1).copy()

        polar = results["radar_points_polar"]
        polar = polar.clone() if torch.is_tensor(polar) else np.array(polar, copy=True)
        polar[:, 1] *= -1.0  # theta -> -theta
        results["radar_points_polar"] = polar

        xy = results["radar_points_xy"]
        xy = xy.clone() if torch.is_tensor(xy) else np.array(xy, copy=True)
        xy[:, 1] *= -1.0  # y_lateral -> -y_lateral
        results["radar_points_xy"] = xy

        return results

@TRANSFORMS.register_module()
class PPPPolarFilterDomain(BaseTransform):
    """Keep only radar points inside the supervised polar domain."""

    def __init__(
        self,
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
    ) -> None:
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

    def transform(self, results: dict) -> dict:
        polar = results["radar_points_polar"]

        if isinstance(polar, torch.Tensor):
            keep = (
                (polar[:, 0] >= self.r_min)
                & (polar[:, 0] <= self.r_max)
                & (polar[:, 1] >= self.theta_min)
                & (polar[:, 1] <= self.theta_max)
            )
        elif isinstance(polar, np.ndarray):
            keep = (
                (polar[:, 0] >= self.r_min)
                & (polar[:, 0] <= self.r_max)
                & (polar[:, 1] >= self.theta_min)
                & (polar[:, 1] <= self.theta_max)
            )
        else:
            raise TypeError(
                "radar_points_polar must be a torch.Tensor or numpy.ndarray, "
                f"got {type(polar)!r}."
            )

        results["radar_points_polar"] = results["radar_points_polar"][keep]
        results["radar_points_xy"] = results["radar_points_xy"][keep]
        results["radar_points_meta"] = results["radar_points_meta"][keep]

        return results

@TRANSFORMS.register_module()
class PPPPackPolarInputs(BaseTransform):
    """Pack image and variable-length polar radar point targets."""

    @staticmethod
    def _to_float_tensor(x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.contiguous().float()
        return torch.from_numpy(np.ascontiguousarray(x)).float()

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(np.ascontiguousarray(img))
            img = img.permute(2, 0, 1).contiguous().float()
        elif isinstance(img, torch.Tensor):
            img = img.contiguous().float()
        else:
            raise TypeError(f"Unsupported img type: {type(img)}")

        from mmengine.structures import InstanceData
        from mmseg.structures import SegDataSample

        polar = self._to_float_tensor(results["radar_points_polar"])
        xy = self._to_float_tensor(results["radar_points_xy"])

        radar_points = InstanceData()
        radar_points.polar = polar
        radar_points.xy = xy

        if "radar_points_meta" in results:
            radar_points.meta = self._to_float_tensor(results["radar_points_meta"])

        metainfo = {}
        if isinstance(results.get("metainfo", None), dict):
            metainfo.update(results["metainfo"])

        metainfo.update(
            dict(
                img_shape=results.get("img_shape"),
                ori_shape=results.get("ori_shape"),
                num_radar_points=int(polar.shape[0]),
            )
        )

        if "idx" in results:
            metainfo.setdefault("global_idx", int(results["idx"]))
            metainfo.setdefault("frame_idx", int(results["idx"]))

        if "prefetch_unseen" in results:
            metainfo.setdefault("prefetch_unseen", int(results["prefetch_unseen"]))
            metainfo.setdefault(
                "prefetch_unseen_count",
                int(results["prefetch_unseen_count"]),
            )
            metainfo.setdefault(
                "prefetch_idle_time",
                float(results["prefetch_idle_time"]),
            )

        data_sample = SegDataSample()
        data_sample.set_metainfo(metainfo)
        data_sample.gt_radar_points = radar_points

        return dict(
            inputs=img,
            data_samples=data_sample,
        )