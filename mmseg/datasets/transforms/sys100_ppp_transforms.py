from __future__ import annotations

import numpy as np
import torch
import cv2

from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import BaseTransform

from aumovio.sys100dataset import point_from_gridentry_vectorized
from aumovio.fisheye import PinholeCamera, FisheyeCamera

from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
import torch

@TRANSFORMS.register_module()
class LoadSys100Sample(BaseTransform):
    """
    Loads one SYS100 sample given results['idx'] (provided by the dataset).
    Produces:
      results['img']        : HWC uint8
      results['radar']      : torch tensor [C,64,64] (kept for target build)
      results['rad2cam']    : torch tensor [3,4]
      results['real_hw']    : torch tensor [2]
      results['dopplerRange']: torch scalar
    """
    def __init__(self, sys100_root: str, out_img_size=(544, 1792), undistort=True, fovs=(110, 48.4), strict=False):
        from aumovio.sys100dataset import Sys100Dataset

        self.ds = Sys100Dataset(
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

        # for radar un-normalization
        self.radar_normalisation = self.ds.radar_normalisation
        self.fovs = fovs

    def transform(self, results: dict) -> dict:
        idx = int(results["idx"])
        sample = self.ds[idx]

        img_t = sample["jpg"]  # [3,H,W], normalized [-1,1]
        img = img_t.cpu().numpy().transpose(1, 2, 0)
        img = np.ascontiguousarray(127.5 * img + 127.5, dtype=np.uint8)  # HWC uint8

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        # keep these for target building
        results["radar"] = sample["radar"]
        results["rad2cam"] = sample["rad2cam"]
        results["real_hw"] = sample["real_hw"]
        results["dopplerRange"] = sample.get("dopplerRange", torch.tensor(20.0))
        results["radar_normalisation"] = self.radar_normalisation
        results["fovs"] = self.fovs

        results.setdefault("metainfo", {})
        results["metainfo"]["sample_id"] = f"sys100_{idx:06d}"

        return results


@TRANSFORMS.register_module()
class BuildPPPHeatmap(BaseTransform):
    """
    Builds gt_seg_map with 3 channels:
      0: intensity (1 at projected points)
      1: depth (at projected points)
      2: velocity (at projected points)

    Output:
      results['gt_seg_map'] : HWC float32 with 3 channels
    """
    def __init__(
        self,
        thres=0.05,
        rel_x_centre=0.491,
        rel_y_centre=0.4370,
        clip_vel=(-10.0, 10.0),
        clip_depth=(0.0, 150.0),
    ):
        self.thres = float(thres)
        self.rel_x_centre = float(rel_x_centre)
        self.rel_y_centre = float(rel_y_centre)
        self.clip_vel = clip_vel
        self.clip_depth = clip_depth

    def transform(self, results: dict) -> dict:
        img = results["img"]
        H, W = img.shape[:2]

        radar = results["radar"].cpu().numpy()  # [C,64,64]
        radar_norm = results["radar_normalisation"]

        # reconstruct clusters EXACTLY like the SYS100 code
        radar_grid = radar.transpose(1, 2, 0)[::-1, ::-1] * radar_norm[None, None, :]
        clusters = point_from_gridentry_vectorized(radar_grid, thres=self.thres)

        gt = np.zeros((H, W, 3), dtype=np.float32)

        if len(clusters) == 0:
            results["gt_seg_map"] = gt
            return results

        doppler = results["dopplerRange"]
        doppler = float(doppler.item() if hasattr(doppler, "item") else doppler)

        # velocity definition like SYS100
        vel = clusters[:, 5] - clusters[:, 6]
        vel -= doppler * np.floor(vel / doppler + 0.5)

        depth = clusters[:, 0]

        # radar -> camera coordinates
        rad2cam = results["rad2cam"].cpu().numpy()  # [3,4]
        cam_pts = clusters[:, :2] @ rad2cam[:3, :2].T + rad2cam[:3, 3:].T  # -> [N,3]
        # cam_pts columns: (X,Y,Z)

        # build SAME pinhole model (this is your overlay code)
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

        # compute pixel coords WITHOUT drawing:
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

        # optional clipping to keep scale sane
        depth = np.clip(depth, self.clip_depth[0], self.clip_depth[1])
        vel = np.clip(vel, self.clip_vel[0], self.clip_vel[1])

        # write sparse targets
        gt[y_pix, x_pix, 0] = 1.0
        gt[y_pix, x_pix, 1] = depth
        gt[y_pix, x_pix, 2] = vel

        results["gt_seg_map"] = gt
        results["seg_fields"] = ["gt_seg_map"]  # mmseg convention
        return results


@TRANSFORMS.register_module()
class PPPRandomCrop(BaseTransform):
    """Crop img (HWC) and gt_seg_map (HWC float) identically."""
    def __init__(self, crop_size):
        self.crop_h, self.crop_w = crop_size

    def transform(self, results: dict) -> dict:
        img = results["img"]
        gt = results["gt_seg_map"]
        H, W = img.shape[:2]

        ch, cw = self.crop_h, self.crop_w
        if H < ch or W < cw:
            # pad if needed
            pad_h = max(0, ch - H)
            pad_w = max(0, cw - W)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
            gt = np.pad(gt, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            H, W = img.shape[:2]

        y1 = np.random.randint(0, H - ch + 1)
        x1 = np.random.randint(0, W - cw + 1)

        results["img"] = img[y1:y1+ch, x1:x1+cw]
        results["gt_seg_map"] = gt[y1:y1+ch, x1:x1+cw]
        results["img_shape"] = (ch, cw)
        return results


@TRANSFORMS.register_module()
class PPPRandomFlip(BaseTransform):
    """Flip img + gt_seg_map together."""
    def __init__(self, prob=0.5, direction="horizontal"):
        self.prob = prob
        self.direction = direction

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        gt = results["gt_seg_map"]

        if self.direction == "horizontal":
            results["img"] = img[:, ::-1, :]
            results["gt_seg_map"] = gt[:, ::-1, :]
        elif self.direction == "vertical":
            results["img"] = img[::-1, :, :]
            results["gt_seg_map"] = gt[::-1, :, :]
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        return results


@TRANSFORMS.register_module()
class PPPPackSegInputs(BaseTransform):
    """
    Pack outputs into mmseg’s expected format.
    Supports multi-channel float gt_seg_map.
    """

    def transform(self, results: dict) -> dict:
        # -------------------------
        # Image
        # -------------------------
        img = results["img"]  # HWC uint8

        # IMPORTANT: kill negative strides
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        img = img.float()

        # -------------------------
        # GT heatmap
        # -------------------------
        gt = results["gt_seg_map"]  # HWC float32

        # IMPORTANT: same here
        gt = np.ascontiguousarray(gt)

        gt = torch.from_numpy(gt).permute(2, 0, 1).contiguous()

        # -------------------------
        # SegDataSample
        # -------------------------
        from mmseg.structures import SegDataSample
        from mmengine.structures import PixelData

        data_sample = SegDataSample()
        data_sample.set_metainfo(
            dict(
                img_shape=results.get("img_shape"),
                ori_shape=results.get("ori_shape"),
            )
        )
        data_sample.gt_sem_seg = PixelData(data=gt)

        return dict(
            inputs=img,
            data_samples=data_sample,
        )