"""
A mmsegmentation-compatible adapter around the SYS100 codebase.

Content:
- SYS100 dataset internals (radar → clusters, camera model), directly from SYS100 codebase
- PPP target definition (3-channel heatmap, Poisson view)
- mmseg BaseTransform for PPP-style supervision
- Naming, structure, integration into training loop

Design decisions:
- `img` is kept throughout the pipeline as HWC numpy float32 in [-1, 1]
- `gt_seg_map` is kept as HWC numpy float32
- only in PPPPackSegInputs both are converted to CHW torch tensors
- no expensive conversion from [-1,1] to uint8 [0,255]
- camera objects are cached
- expensive assertions are optional via debug_checks
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple
import time

import cv2
import numpy as np
import torch

import time
import logging
from mmengine.logging import print_log

from mmseg.datasets.transforms.loading import BaseTransform
from mmseg.registry import TRANSFORMS

from aumovio.fisheye import FisheyeCamera, PinholeCamera
from aumovio.prefetcheddataset import AsyncPrefetchedDatasetWrapper
from aumovio.sys100dataset import Sys100Dataset, point_from_gridentry_vectorized


LOGGER_NAME = "ppp.transforms"
LOG_LEVEL = logging.INFO

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOG_LEVEL)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor/array-like to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_float_scalar(x: Any) -> float:
    """Convert scalar-like value (tensor, numpy scalar, python scalar) to float."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

def get_rank():
   return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

@TRANSFORMS.register_module()
class LoadSys100Sample(BaseTransform):
    """
    Materializes one SYS100 sample.

    Expects:
      results["idx"] : global dataset index

    Produces:
      results["img"]                  : HWC numpy float32 in [-1, 1]
      results["radar"]                : torch.Tensor [C, 64, 64]
      results["rad2cam"]              : torch.Tensor [3, 4]
      results["real_hw"]              : torch.Tensor [2]
      results["dopplerRange"]         : scalar-like
      results["radar_normalisation"]  : per-channel radar scaling
      results["fovs"]                 : (hFov, vFov)

    Also sets:
      results["metainfo"] with:
        - sample_id
        - frame_idx
        - global_idx
        - video_path
        - video_name
        - video_frame_idx
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
    ):
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
            fovs=fovs,
            strict=strict,
        )

        # Read base dataset attributes before optional wrapping.
        self.radar_normalisation = base_ds.radar_normalisation
        self.fovs = fovs
        self.default_doppler = 20.0

        if prefetch:
            self.ds = AsyncPrefetchedDatasetWrapper(
                base_dataset=base_ds,
                **prefetch_dict,
            )
        else:
            self.ds = base_ds
        
        # self.time_nbr = 0
        # self.time_avg = 0

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(results["idx"])
        
        # t0 = time.perf_counter()
        sample = self.ds[idx]
        # t1 = time.perf_counter()

        # self.time_nbr += 1
        # self.time_avg += ((t1-t0) - self.time_avg)/self.time_nbr
        # if self.time_nbr % 60 == 0:
        #     print(f"[rank {get_rank()}] SYS100 querry time = {self.time_avg:.4f}s [avg over {self.time_nbr} steps]", flush=True)
        #     self.time_nbr = 0
        #     self.time_avg = 0

        # -------------------------
        # Meta info
        # -------------------------
        video_path = sample.get("video_path", None)
        md = sample.get("meta_data", None)

        frame_in_video = None
        if md is not None:
            md_dtype = getattr(md, "dtype", None)
            md_names = getattr(md_dtype, "names", None)
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

        # -------------------------
        # Image
        # SYS100 provides [3, H, W] torch tensor in [-1, 1].
        # Keep it as HWC numpy float32 in [-1, 1] throughout the pipeline.
        # -------------------------
        img_t = sample["jpg"]  # torch.Tensor [3, H, W]
        img = img_t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32, copy=False)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        # -------------------------
        # Keep these for target building
        # -------------------------
        results["radar"] = sample["radar"]
        results["rad2cam"] = sample["rad2cam"]
        results["real_hw"] = sample["real_hw"]
        results["dopplerRange"] = sample.get("dopplerRange", self.default_doppler)
        results["radar_normalisation"] = self.radar_normalisation
        results["fovs"] = self.fovs
        if "prefetch_unseen" in sample:
            results["prefetch_unseen"] = sample["prefetch_unseen"]
            results["prefetch_unseen_count"] = sample["prefetch_unseen_count"]
            results["prefetch_idle_time"] = sample["prefetch_idle_time"]

        return results


@TRANSFORMS.register_module()
class BuildPPPHeatmap(BaseTransform):
    """
    Builds gt_seg_map with 3 channels:
      0: intensity (1 at projected points)
      1: depth
      2: velocity

    Output:
      results["gt_seg_map"] : HWC float32 with 3 channels
    """

    def __init__(
        self,
        thres: float = 0.05,
        rel_x_centre: float = 0.491,
        rel_y_centre: float = 0.4370,
        clip_vel: Tuple[float, float] = (-10.0, 10.0),
        clip_depth: Tuple[float, float] = (0.0, 150.0),
        debug_checks: bool = False,
    ):
        self.thres = float(thres)
        self.rel_x_centre = float(rel_x_centre)
        self.rel_y_centre = float(rel_y_centre)
        self.clip_vel = clip_vel
        self.clip_depth = clip_depth
        self.debug_checks = debug_checks
        self._cam_cache: Dict[Tuple[Any, ...], PinholeCamera] = {}

    def _get_pinhole_cam(
        self,
        H: int,
        W: int,
        real_hw: np.ndarray,
        fovs: Tuple[float, float],
    ) -> PinholeCamera:
        """Cache camera objects for repeated geometries."""
        real_h = int(real_hw[0])
        real_w = int(real_hw[1])
        hFov, vFov = fovs

        key = (
            int(H),
            int(W),
            real_h,
            real_w,
            float(hFov),
            float(vFov),
            self.rel_x_centre,
            self.rel_y_centre,
        )

        cam = self._cam_cache.get(key)
        if cam is None:
            cam = PinholeCamera(
                height=H,
                width=W,
                hFov=float(hFov),
                vFov=float(vFov),
                rel_x_centre=self.rel_x_centre,
                rel_y_centre=self.rel_y_centre,
                fisheye_cam=FisheyeCamera(height=real_h, width=real_w),
            )
            self._cam_cache[key] = cam
        return cam

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]  # HWC numpy
        H, W = img.shape[:2]

        radar = _to_numpy(results["radar"])  # [C, 64, 64]
        radar_norm = results["radar_normalisation"]

        # Recover sparse radar point cloud from dense 64x64 grid.
        radar_grid = radar.transpose(1, 2, 0)[::-1, ::-1] * radar_norm[None, None, :]
        clusters = point_from_gridentry_vectorized(radar_grid, thres=self.thres)

        if self.debug_checks:
            assert clusters.ndim == 2 and clusters.shape[1] >= 7, (
                f"Expected clusters shape (N,>=7), got {clusters.shape}"
            )

        gt = np.zeros((H, W, 3), dtype=np.float32)

        if len(clusters) == 0:
            results["gt_seg_map"] = gt
            results["seg_fields"] = ["gt_seg_map"]
            return results

        doppler = _to_float_scalar(results["dopplerRange"])

        # Velocity definition like SYS100 (fold into Doppler ambiguity range).
        vel = clusters[:, 5] - clusters[:, 6]
        vel -= doppler * np.floor(vel / doppler + 0.5)

        depth = clusters[:, 0]

        # radar -> camera coordinates
        rad2cam = _to_numpy(results["rad2cam"])  # [3, 4]
        cam_pts = clusters[:, :2] @ rad2cam[:3, :2].T + rad2cam[:3, 3:].T  # [N, 3]

        real_hw = _to_numpy(results["real_hw"])
        pinhole_cam = self._get_pinhole_cam(H, W, real_hw, results["fovs"])

        z = cam_pts[:, 2]
        u = np.divide(cam_pts[:, 0], z, out=np.zeros_like(z), where=z != 0)
        v = np.divide(cam_pts[:, 1], z, out=np.zeros_like(z), where=z != 0)

        x_pix, y_pix = pinhole_cam.real_world_to_pinhole(u, v)
        x_pix = x_pix.astype(np.int32, copy=False)
        y_pix = y_pix.astype(np.int32, copy=False)

        valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
        if not np.any(valid):
            results["gt_seg_map"] = gt
            results["seg_fields"] = ["gt_seg_map"]
            return results

        x_pix = x_pix[valid]
        y_pix = y_pix[valid]
        depth = np.clip(depth[valid], self.clip_depth[0], self.clip_depth[1])
        vel = np.clip(vel[valid], self.clip_vel[0], self.clip_vel[1])

        gt[y_pix, x_pix, 0] = 1.0
        gt[y_pix, x_pix, 1] = depth
        gt[y_pix, x_pix, 2] = vel

        if self.debug_checks:
            occ = gt[..., 0]
            assert np.all((occ == 0) | (occ == 1))
            assert np.all(gt[..., 1][occ == 0] == 0)
            assert np.all(gt[..., 2][occ == 0] == 0)

        results["gt_seg_map"] = gt
        results["seg_fields"] = ["gt_seg_map"]
        return results


@TRANSFORMS.register_module()
class PPPRandomCrop(BaseTransform):
    """Random crop img and gt_seg_map together (with padding if needed)."""

    def __init__(self, crop_size: Tuple[int, int]):
        self.crop_h, self.crop_w = crop_size

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]        # HWC numpy float32
        gt = results["gt_seg_map"]  # HWC numpy float32
        H, W = img.shape[:2]

        ch, cw = self.crop_h, self.crop_w
        if H < ch or W < cw:
            pad_h = max(0, ch - H)
            pad_w = max(0, cw - W)

            img = cv2.copyMakeBorder(
                img,
                0,
                pad_h,
                0,
                pad_w,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
            gt = np.pad(gt, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            H, W = img.shape[:2]

        y1 = np.random.randint(0, H - ch + 1)
        x1 = np.random.randint(0, W - cw + 1)

        results["img"] = img[y1:y1 + ch, x1:x1 + cw]
        results["gt_seg_map"] = gt[y1:y1 + ch, x1:x1 + cw]
        results["img_shape"] = (ch, cw)

        return results


@TRANSFORMS.register_module()
class PPPRandomFlip(BaseTransform):
    """Flip img and gt_seg_map together."""

    def __init__(self, prob: float = 0.5, direction: str = "horizontal"):
        self.prob = float(prob)
        self.direction = direction

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]        # HWC numpy
        gt = results["gt_seg_map"]  # HWC numpy

        if self.direction == "horizontal":
            # Keep HWC layout, return ordinary arrays with positive strides.
            results["img"] = np.flip(img, axis=1).copy()
            results["gt_seg_map"] = np.flip(gt, axis=1).copy()
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        return results


@TRANSFORMS.register_module()
class PPPPackSegInputs(BaseTransform):
    """
    Pack outputs into mmseg's expected format:
    return {
        "inputs": <torch.Tensor>,
        "data_samples": <SegDataSample>
    }

    Supports multi-channel float gt_seg_map.
    This version merges results["metainfo"] into SegDataSample.metainfo.
    """

    def __init__(self, debug_checks: bool = False):
        self.debug_checks = debug_checks

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # -------------------------
        # Image
        # img is HWC numpy float32 in [-1, 1]
        # -------------------------
        img = results["img"]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous().float()
        elif isinstance(img, torch.Tensor):
            img = img.contiguous().float()
        else:
            raise TypeError(f"Unsupported img type: {type(img)}")

        # -------------------------
        # GT heatmap
        # gt is HWC numpy float32
        # -------------------------
        gt = results["gt_seg_map"]
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).contiguous()
        elif isinstance(gt, torch.Tensor):
            gt = gt.contiguous()
        else:
            raise TypeError(f"Unsupported gt_seg_map type: {type(gt)}")

        # -------------------------
        # SegDataSample
        # -------------------------
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample

        metainfo = {}
        if isinstance(results.get("metainfo", None), dict):
            metainfo.update(results["metainfo"])

        metainfo.update(
            dict(
                img_shape=results.get("img_shape"),
                ori_shape=results.get("ori_shape"),
            )
        )

        if "rad2cam" in results:
            metainfo["rad2cam"] = (
                _to_numpy(results["rad2cam"])
                .astype(np.float32, copy=True)
            )

        if "fovs" in results:
            metainfo["fovs"] = tuple(
                float(value) for value in results["fovs"]
            )
            
        if "idx" in results:
            metainfo.setdefault("global_idx", int(results["idx"]))
            metainfo.setdefault("frame_idx", int(results["idx"]))

        if "prefetch_unseen" in results:
            metainfo.setdefault("prefetch_unseen", int(results["prefetch_unseen"]))
            metainfo.setdefault("prefetch_unseen_count", int(results["prefetch_unseen_count"]))
            metainfo.setdefault("prefetch_idle_time", int(results["prefetch_idle_time"]))
        
        data_sample = SegDataSample()
        data_sample.set_metainfo(metainfo)
        data_sample.gt_sem_seg = PixelData(data=gt)

        if self.debug_checks:
            unique_vals = torch.unique(gt[0])
            assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
                f"PPP intensity channel is not binary. Found values: {unique_vals.tolist()}"
            )

        return dict(
            inputs=img,
            data_samples=data_sample,
        )