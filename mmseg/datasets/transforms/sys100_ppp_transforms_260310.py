"""
A mmsegmentation-compatible adapter around the SYS100 codebase.

Content:
- SYS100 dataset internals (radar → clusters, camera model), directly from SYS100 codebase
- PPP target definition (3-channel heatmap, Poisson view)
- mmseg BaseTransform for PPP-style supervision
- Naming, structure, integration into training loop
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple
import time

import cv2
import numpy as np
import torch

from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import BaseTransform

from aumovio.sys100dataset import point_from_gridentry_vectorized
from aumovio.fisheye import PinholeCamera, FisheyeCamera

LOGGER_NAME = "ppp.transforms"
LOG_LEVEL = logging.INFO  # set to logging.DEBUG for verbose diagnostics

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOG_LEVEL)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import BaseTransform


def get_rank():
   return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))


@TRANSFORMS.register_module()
class LoadSys100Sample(BaseTransform):
    """
    All sensor and video logic comes from SYS100Dataset. This transform materializes one sample.

    Expects:
      results["idx"] : global dataset index (provided by the PyTorch Dataset wrapper)

    Produces:
      results["img"]        : (H, W, 3) uint8 in [0, 255]
      results["radar"]      : torch.Tensor [7, 64, 64] (normalized radar grid)
      results["rad2cam"]    : torch.Tensor [3, 4] (affine transform radar→camera)
      results["real_hw"]    : torch.Tensor [2]
      results["dopplerRange"]: torch scalar
      results["radar_normalisation"]: per-channel scaling to unnormalize radar values
      results["fovs"]       : (hFov, vFov)

    Also sets:
      results["metainfo"] with:
        - sample_id
        - frame_idx (legacy: global idx; keep for backwards compatibility)
        - global_idx
        - video_path
        - video_name
        - video_frame_idx (true frame counter in the video)
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
    ):
        from aumovio.sys100dataset import Sys100Dataset
        from aumovio.prefetcheddataset import AsyncPrefetchedDatasetWrapper

        if prefetch_dict is None:
            prefetch_dict = {}

        self.ds = Sys100Dataset(
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

        # for radar un-normalization
        self.radar_normalisation = self.ds.radar_normalisation

        if prefetch:
            print(f"[rank {get_rank()}] init SYS100 with AsyncWrapper")
            self.ds = AsyncPrefetchedDatasetWrapper(
                base_dataset=self.ds,
                **prefetch_dict,
            )

        self.fovs = fovs
        self.time_nbr = 0
        self.time_avg = 0

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(results["idx"])
        
        t0 = time.perf_counter()
        sample = self.ds[idx]
        t1 = time.perf_counter()

        self.time_nbr += 1
        self.time_avg += ((t1-t0) - self.time_avg)/self.time_nbr
        if self.time_nbr % 60 == 0:
            print(f"[rank {get_rank()}] SYS100 querry time = {self.time_avg:.4f}s [avg over {self.time_nbr} steps]", flush=True)
            self.time_nbr = 0
            self.time_avg = 0

        # -------------------------
        # Meta info (video + frame)
        # -------------------------
        video_path = sample.get("video_path", None)

        md = sample.get("meta_data", None)
        frame_in_video = None
        if md is not None and getattr(getattr(md, "dtype", None), "names", None) is not None:
            if "VideoFrameCounter" in md.dtype.names:
                frame_in_video = int(md["VideoFrameCounter"])

        results.setdefault("metainfo", {})
        # Keep legacy key for old code.
        results["metainfo"].update(
            dict(
                sample_id=f"sys100_{idx:06d}",
                frame_idx=idx,          # legacy: global idx
                global_idx=idx,         # explicit
                video_path=video_path,
                video_name=os.path.basename(video_path) if video_path else None,
                video_frame_idx=frame_in_video,  # true frame counter inside the video
            )
        )

        # -------------------------
        # Image (jpg: [3,H,W] in [-1,1] -> HWC uint8 in [0,255])
        # -------------------------
        img_t = sample["jpg"]
        img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(127.5 * img + 127.5, dtype=np.uint8)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        # -------------------------
        # Keep these for target building
        # -------------------------
        results["radar"] = sample["radar"]
        results["rad2cam"] = sample["rad2cam"]
        results["real_hw"] = sample["real_hw"]
        results["dopplerRange"] = sample.get("dopplerRange", torch.tensor(20.0))
        results["radar_normalisation"] = self.radar_normalisation
        results["fovs"] = self.fovs

        return results


@TRANSFORMS.register_module()
class BuildPPPHeatmap(BaseTransform):
    """
    Builds gt_seg_map with 3 channels:
      0: intensity (1 at projected points)
      1: depth (at projected points)
      2: velocity (at projected points)

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
    ):
        self.thres = float(thres)
        self.rel_x_centre = float(rel_x_centre)
        self.rel_y_centre = float(rel_y_centre)
        self.clip_vel = clip_vel
        self.clip_depth = clip_depth

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]
        H, W = img.shape[:2]

        radar = results["radar"].cpu().numpy()  # [C,64,64]
        radar_norm = results["radar_normalisation"]

        # Recover sparse radar point cloud from dense 64×64 grid.
        radar_grid = radar.transpose(1, 2, 0)[::-1, ::-1] * radar_norm[None, None, :]
        clusters = point_from_gridentry_vectorized(radar_grid, thres=self.thres)

        assert clusters.ndim == 2 and clusters.shape[1] >= 7, (
            f"Expected clusters shape (N,>=7), got {clusters.shape}"
        )

        gt = np.zeros((H, W, 3), dtype=np.float32)
        if len(clusters) == 0:
            results["gt_seg_map"] = gt
            results["seg_fields"] = ["gt_seg_map"]
            return results

        doppler = results["dopplerRange"]
        doppler = float(doppler.item() if hasattr(doppler, "item") else doppler)

        # Velocity definition like SYS100 (fold into Doppler ambiguity range).
        vel = clusters[:, 5] - clusters[:, 6]
        vel -= doppler * np.floor(vel / doppler + 0.5)

        depth = clusters[:, 0]

        # radar -> camera coordinates
        rad2cam = results["rad2cam"].cpu().numpy()  # [3,4]
        cam_pts = clusters[:, :2] @ rad2cam[:3, :2].T + rad2cam[:3, 3:].T  # [N,3]

        # pinhole model
        real_hw = results["real_hw"].cpu().numpy()
        hFov, vFov = results["fovs"]

        pinhole_cam = PinholeCamera(
            height=H,
            width=W,
            hFov=float(hFov),
            vFov=float(vFov),
            rel_x_centre=self.rel_x_centre,
            rel_y_centre=self.rel_y_centre,
            fisheye_cam=FisheyeCamera(height=int(real_hw[0]), width=int(real_hw[1])),
        )

        z = cam_pts[:, 2]
        u = np.divide(cam_pts[:, 0], z, out=np.zeros_like(z), where=z != 0)
        v = np.divide(cam_pts[:, 1], z, out=np.zeros_like(z), where=z != 0)
        x_pix, y_pix = pinhole_cam.real_world_to_pinhole(u, v)

        x_pix = x_pix.astype(np.int32)
        y_pix = y_pix.astype(np.int32)

        valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
        x_pix = x_pix[valid]
        y_pix = y_pix[valid]
        depth = depth[valid]
        vel = vel[valid]

        depth = np.clip(depth, self.clip_depth[0], self.clip_depth[1])
        vel = np.clip(vel, self.clip_vel[0], self.clip_vel[1])

        # Write sparse targets (occupied pixels, not point multiplicity).
        gt[y_pix, x_pix, 0] = 1.0
        gt[y_pix, x_pix, 1] = depth
        gt[y_pix, x_pix, 2] = vel

        assert np.all((gt[..., 0] == 0) | (gt[..., 0] == 1))
        assert np.all(gt[..., 1][gt[..., 0] == 0] == 0)
        assert np.all(gt[..., 2][gt[..., 0] == 0] == 0)

        results["gt_seg_map"] = gt
        results["seg_fields"] = ["gt_seg_map"]
        return results


@TRANSFORMS.register_module()
class PPPRandomCrop(BaseTransform):
    """Random crop img and gt_seg_map together (with padding if needed)."""

    def __init__(self, crop_size: Tuple[int, int]):
        self.crop_h, self.crop_w = crop_size

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]
        gt = results["gt_seg_map"]
        H, W = img.shape[:2]

        ch, cw = self.crop_h, self.crop_w
        if H < ch or W < cw:
            pad_h = max(0, ch - H)
            pad_w = max(0, cw - W)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
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
    """Flip img + gt_seg_map together."""

    def __init__(self, prob: float = 0.5, direction: str = "horizontal"):
        self.prob = float(prob)
        self.direction = direction

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        gt = results["gt_seg_map"]

        if self.direction == "horizontal":
            results["img"] = img[:, ::-1, :]
            results["gt_seg_map"] = gt[:, ::-1, :]
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        return results


@TRANSFORMS.register_module()
class PPPPackSegInputs(BaseTransform):
    """
    Pack outputs into mmseg’s expected format:
    return {
        "inputs": <torch.Tensor>,
        "data_samples": <SegDataSample>
    }
    Supports multi-channel float gt_seg_map.

    This version merges results["metainfo"] into SegDataSample.metainfo.
    """

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # -------------------------
        # Image
        # -------------------------
        img = np.ascontiguousarray(results["img"])  # HWC uint8
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        # -------------------------
        # GT heatmap
        # -------------------------
        gt = np.ascontiguousarray(results["gt_seg_map"])  # HWC float32
        gt = torch.from_numpy(gt).permute(2, 0, 1).contiguous()

        # -------------------------
        # SegDataSample
        # -------------------------
        from mmseg.structures import SegDataSample
        from mmengine.structures import PixelData

        metainfo = {}
        if isinstance(results.get("metainfo", None), dict):
            metainfo.update(results["metainfo"])

        metainfo.update(
            dict(
                img_shape=results.get("img_shape"),
                ori_shape=results.get("ori_shape"),
            )
        )

        # Ensure global_idx exists (explicit), but do not overwrite if set upstream.
        if "idx" in results:
            metainfo.setdefault("global_idx", int(results["idx"]))
            metainfo.setdefault("frame_idx", int(results["idx"]))  # legacy key

        data_sample = SegDataSample()
        data_sample.set_metainfo(metainfo)
        data_sample.gt_sem_seg = PixelData(data=gt)

        # Assert: intensity channel must be binary {0,1}
        unique_vals = torch.unique(gt[0])
        assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
            f"PPP intensity channel is not binary. Found values: {unique_vals.tolist()}"
        )

        return dict(
            inputs=img,
            data_samples=data_sample,
        )