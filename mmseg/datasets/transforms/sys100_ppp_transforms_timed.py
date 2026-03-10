import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from mmseg.datasets.transforms.loading import BaseTransform
from mmseg.registry import TRANSFORMS

from aumovio.fisheye import FisheyeCamera, PinholeCamera
from aumovio.sys100dataset import point_from_gridentry_vectorized


LOGGER_NAME = "ppp.transforms"
LOG_LEVEL = logging.INFO

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOG_LEVEL)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)


# =============================================================================
# GLOBAL PERIODIC TIMER WITH SOURCE LOCATION
# =============================================================================


class GlobalStepTimer:
    """
    Process-local timing accumulator.

    It prints a summary every `report_every` completed samples.
    A "completed sample" is counted at the end of PPPPackSegInputs.transform(),
    i.e. after the full transform pipeline for one sample has finished.
    """

    def __init__(self, report_every: int = 150, enabled: bool = True):
        self.report_every = int(report_every)
        self.enabled = bool(enabled)

        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.locations = {}

        self.completed_samples = 0
        self.lock = threading.Lock()

    def add(self, key: str, dt: float, location: Optional[str] = None) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.times[key] += float(dt)
            self.counts[key] += 1
            if location is not None and key not in self.locations:
                self.locations[key] = location

    def sample_done(self) -> None:
        if not self.enabled:
            return

        should_report = False
        with self.lock:
            self.completed_samples += 1
            if self.report_every > 0 and self.completed_samples % self.report_every == 0:
                should_report = True

        if should_report:
            self.report()

    def report(self) -> None:
        if not self.enabled:
            return

        with self.lock:
            if not self.times:
                print(
                    f"\n[PPP TIMER] pid={os.getpid()} samples={self.completed_samples}: no timing data.\n",
                    flush=True,
                )
                return

            total = sum(self.times.values())
            rows = sorted(self.times.items(), key=lambda kv: kv[1], reverse=True)
            counts = dict(self.counts)
            locations = dict(self.locations)
            samples = self.completed_samples
            pid = os.getpid()

        print("\n" + "=" * 150, flush=True)
        print(f"[PPP TIMER] pid={pid} | completed_samples={samples}", flush=True)
        print(f"[PPP TIMER] Total measured transform time in this process: {total:.6f} s", flush=True)
        print("-" * 150, flush=True)
        print(
            f"{'name':55s} {'time [s]':>12s} {'calls':>10s} {'avg [ms]':>12s} {'share [%]':>10s} {'location':>45s}",
            flush=True,
        )
        print("-" * 150, flush=True)

        for key, t in rows:
            c = counts.get(key, 0)
            avg_ms = 1000.0 * t / c if c > 0 else 0.0
            share = 100.0 * t / total if total > 0 else 0.0
            loc = locations.get(key, "-")
            print(
                f"{key:55s} {t:12.6f} {c:10d} {avg_ms:12.3f} {share:10.2f} {loc:>45s}",
                flush=True,
            )

        print("-" * 150, flush=True)
        print(
            "[PPP TIMER] This table is cumulative for THIS process since start.\n"
            "[PPP TIMER] With DataLoader workers or DDP, each process prints its own table.\n"
            "[PPP TIMER] For the cleanest profile, set num_workers=0.",
            flush=True,
        )
        print("=" * 150 + "\n", flush=True)


GLOBAL_TIMER = GlobalStepTimer(report_every=150, enabled=True)


class StepTimer:
    """Per-call step timer that accumulates into GLOBAL_TIMER."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.last = time.perf_counter()

    def step(self, name: str) -> None:
        now = time.perf_counter()

        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is not None:
            location = f"{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}"
        else:
            location = "unknown"

        GLOBAL_TIMER.add(f"{self.prefix}.{name}", now - self.last, location=location)
        self.last = now

    def finish(self) -> None:
        now = time.perf_counter()

        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is not None:
            location = f"{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}"
        else:
            location = "unknown"

        GLOBAL_TIMER.add(f"{self.prefix}.finish", now - self.last, location=location)
        self.last = now


# =============================================================================
# HELPERS
# =============================================================================


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_float_scalar(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


# =============================================================================
# TRANSFORMS
# =============================================================================


@TRANSFORMS.register_module()
class LoadSys100Sample(BaseTransform):
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
        from aumovio.prefetcheddataset import AsyncPrefetchedDatasetWrapper
        from aumovio.sys100dataset import Sys100Dataset

        t0 = time.perf_counter()
        if prefetch_dict is None:
            prefetch_dict = {}
        GLOBAL_TIMER.add(
            "LoadSys100Sample.__init__.prefetch_dict",
            time.perf_counter() - t0,
            location=f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno - 4}",
        )

        t0 = time.perf_counter()
        base_ds = Sys100Dataset(
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
        GLOBAL_TIMER.add(
            "LoadSys100Sample.__init__.Sys100Dataset",
            time.perf_counter() - t0,
            location=f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno - 15}",
        )

        t0 = time.perf_counter()
        self.radar_normalisation = base_ds.radar_normalisation
        self.fovs = fovs
        self.default_doppler = 20.0
        GLOBAL_TIMER.add(
            "LoadSys100Sample.__init__.assign_attrs",
            time.perf_counter() - t0,
            location=f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno - 4}",
        )

        if prefetch:
            t0 = time.perf_counter()
            self.ds = AsyncPrefetchedDatasetWrapper(
                base_dataset=base_ds,
                **prefetch_dict,
            )
            GLOBAL_TIMER.add(
                "LoadSys100Sample.__init__.prefetch_wrap",
                time.perf_counter() - t0,
                location=f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno - 5}",
            )
        else:
            t0 = time.perf_counter()
            self.ds = base_ds
            GLOBAL_TIMER.add(
                "LoadSys100Sample.__init__.no_prefetch_assign",
                time.perf_counter() - t0,
                location=f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno - 2}",
            )

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        timer = StepTimer("LoadSys100Sample")

        idx = int(results["idx"])
        timer.step("read_idx")

        sample = self.ds[idx]
        timer.step("ds_getitem")

        video_path = sample.get("video_path", None)
        md = sample.get("meta_data", None)
        timer.step("sample_gets")

        frame_in_video = None
        if md is not None:
            md_dtype = getattr(md, "dtype", None)
            md_names = getattr(md_dtype, "names", None)
            if md_names is not None and "VideoFrameCounter" in md_names:
                frame_in_video = int(md["VideoFrameCounter"])
        timer.step("frame_counter_extract")

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
        timer.step("metainfo_update")

        img_t = sample["jpg"]
        timer.step("img_fetch")

        img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
        timer.step("img_tensor_to_numpy")

        img = np.ascontiguousarray(127.5 * img + 127.5, dtype=np.uint8)
        timer.step("img_rescale_to_uint8")

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        results["radar"] = sample["radar"]
        results["rad2cam"] = sample["rad2cam"]
        results["real_hw"] = sample["real_hw"]
        results["dopplerRange"] = sample.get("dopplerRange", self.default_doppler)
        results["radar_normalisation"] = self.radar_normalisation
        results["fovs"] = self.fovs
        timer.step("assign_outputs")

        timer.finish()
        return results


@TRANSFORMS.register_module()
class BuildPPPHeatmap(BaseTransform):
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
        t0 = time.perf_counter()

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
        GLOBAL_TIMER.add(
            "BuildPPPHeatmap._get_pinhole_cam.lookup",
            time.perf_counter() - t0,
            location=f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno - 10}",
        )

        if cam is None:
            t0 = time.perf_counter()
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
            GLOBAL_TIMER.add(
                "BuildPPPHeatmap._get_pinhole_cam.create",
                time.perf_counter() - t0,
                location=f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno - 9}",
            )

        return cam

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        timer = StepTimer("BuildPPPHeatmap")

        H, W = results["img"].shape[:2]
        timer.step("read_hw")

        radar = _to_numpy(results["radar"])
        timer.step("radar_to_numpy")

        radar_norm = results["radar_normalisation"]
        timer.step("read_radar_norm")

        radar_grid = radar.transpose(1, 2, 0)[::-1, ::-1] * radar_norm[None, None, :]
        timer.step("radar_grid_prepare")

        clusters = point_from_gridentry_vectorized(radar_grid, thres=self.thres)
        timer.step("point_from_gridentry")

        if self.debug_checks:
            assert clusters.ndim == 2 and clusters.shape[1] >= 7, (
                f"Expected clusters shape (N,>=7), got {clusters.shape}"
            )
        timer.step("cluster_assert")

        gt = np.zeros((H, W, 3), dtype=np.float32)
        timer.step("gt_alloc")

        if len(clusters) == 0:
            results["gt_seg_map"] = gt
            results["seg_fields"] = ["gt_seg_map"]
            timer.step("empty_return_assign")
            timer.finish()
            return results

        doppler = _to_float_scalar(results["dopplerRange"])
        timer.step("doppler_scalar")

        vel = clusters[:, 5] - clusters[:, 6]
        timer.step("vel_diff")

        vel -= doppler * np.floor(vel / doppler + 0.5)
        timer.step("vel_wrap")

        depth = clusters[:, 0]
        timer.step("depth_extract")

        rad2cam = _to_numpy(results["rad2cam"])
        timer.step("rad2cam_to_numpy")

        cam_pts = clusters[:, :2] @ rad2cam[:3, :2].T + rad2cam[:3, 3:].T
        timer.step("rad2cam_project")

        real_hw = _to_numpy(results["real_hw"])
        timer.step("real_hw_to_numpy")

        pinhole_cam = self._get_pinhole_cam(H, W, real_hw, results["fovs"])
        timer.step("camera_get")

        z = cam_pts[:, 2]
        timer.step("z_extract")

        u = np.divide(cam_pts[:, 0], z, out=np.zeros_like(z), where=z != 0)
        timer.step("u_compute")

        v = np.divide(cam_pts[:, 1], z, out=np.zeros_like(z), where=z != 0)
        timer.step("v_compute")

        x_pix, y_pix = pinhole_cam.real_world_to_pinhole(u, v)
        timer.step("real_world_to_pinhole")

        x_pix = x_pix.astype(np.int32, copy=False)
        y_pix = y_pix.astype(np.int32, copy=False)
        timer.step("pix_cast_int32")

        valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
        timer.step("valid_mask")

        if not np.any(valid):
            results["gt_seg_map"] = gt
            results["seg_fields"] = ["gt_seg_map"]
            timer.step("no_valid_return_assign")
            timer.finish()
            return results

        x_pix = x_pix[valid]
        y_pix = y_pix[valid]
        timer.step("xy_filter")

        depth = np.clip(depth[valid], self.clip_depth[0], self.clip_depth[1])
        timer.step("depth_filter_clip")

        vel = np.clip(vel[valid], self.clip_vel[0], self.clip_vel[1])
        timer.step("vel_filter_clip")

        gt[y_pix, x_pix, 0] = 1.0
        timer.step("gt_write_intensity")

        gt[y_pix, x_pix, 1] = depth
        timer.step("gt_write_depth")

        gt[y_pix, x_pix, 2] = vel
        timer.step("gt_write_velocity")

        if self.debug_checks:
            occ = gt[..., 0]
            assert np.all((occ == 0) | (occ == 1))
            assert np.all(gt[..., 1][occ == 0] == 0)
            assert np.all(gt[..., 2][occ == 0] == 0)
        timer.step("gt_assert")

        results["gt_seg_map"] = gt
        results["seg_fields"] = ["gt_seg_map"]
        timer.step("final_assign")

        timer.finish()
        return results


@TRANSFORMS.register_module()
class PPPRandomCrop(BaseTransform):
    def __init__(self, crop_size: Tuple[int, int]):
        self.crop_h, self.crop_w = crop_size

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        timer = StepTimer("PPPRandomCrop")

        img = results["img"]
        gt = results["gt_seg_map"]
        H, W = img.shape[:2]
        timer.step("read_inputs")

        ch, cw = self.crop_h, self.crop_w
        timer.step("read_crop_size")

        if H < ch or W < cw:
            pad_h = max(0, ch - H)
            pad_w = max(0, cw - W)
            timer.step("pad_compute")

            img = cv2.copyMakeBorder(
                img,
                0,
                pad_h,
                0,
                pad_w,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
            timer.step("img_padding")

            gt = np.pad(gt, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            timer.step("gt_padding")

            H, W = img.shape[:2]
            timer.step("pad_hw_refresh")
        else:
            timer.step("pad_compute")
            timer.step("img_padding")
            timer.step("gt_padding")
            timer.step("pad_hw_refresh")

        y1 = np.random.randint(0, H - ch + 1)
        x1 = np.random.randint(0, W - cw + 1)
        timer.step("sample_crop_coords")

        results["img"] = img[y1:y1 + ch, x1:x1 + cw]
        results["gt_seg_map"] = gt[y1:y1 + ch, x1:x1 + cw]
        results["img_shape"] = (ch, cw)
        timer.step("crop_assign")

        timer.finish()
        return results


@TRANSFORMS.register_module()
class PPPRandomFlip(BaseTransform):
    def __init__(self, prob: float = 0.5, direction: str = "horizontal"):
        self.prob = float(prob)
        self.direction = direction

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        timer = StepTimer("PPPRandomFlip")

        do_flip = np.random.rand() <= self.prob
        timer.step("sample_flip")

        if not do_flip:
            timer.finish()
            return results

        img = results["img"]
        gt = results["gt_seg_map"]
        timer.step("read_inputs")

        if self.direction == "horizontal":
            results["img"] = img[:, ::-1, :]
            timer.step("img_flip")

            results["gt_seg_map"] = gt[:, ::-1, :]
            timer.step("gt_flip")
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        timer.finish()
        return results


@TRANSFORMS.register_module()
class PPPPackSegInputs(BaseTransform):
    def __init__(self, debug_checks: bool = False):
        self.debug_checks = debug_checks

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        timer = StepTimer("PPPPackSegInputs")

        img = np.ascontiguousarray(results["img"])
        timer.step("img_ascontiguousarray")

        img = torch.from_numpy(img)
        timer.step("img_from_numpy")

        img = img.permute(2, 0, 1).contiguous().float()
        timer.step("img_permute_contiguous_float")

        gt = np.ascontiguousarray(results["gt_seg_map"])
        timer.step("gt_ascontiguousarray")

        gt = torch.from_numpy(gt)
        timer.step("gt_from_numpy")

        gt = gt.permute(2, 0, 1).contiguous()
        timer.step("gt_permute_contiguous")

        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample
        timer.step("import_structures")

        metainfo = {}
        if isinstance(results.get("metainfo", None), dict):
            metainfo.update(results["metainfo"])
        timer.step("metainfo_copy")

        metainfo.update(
            dict(
                img_shape=results.get("img_shape"),
                ori_shape=results.get("ori_shape"),
            )
        )
        timer.step("metainfo_shape_update")

        if "idx" in results:
            metainfo.setdefault("global_idx", int(results["idx"]))
            metainfo.setdefault("frame_idx", int(results["idx"]))
        timer.step("metainfo_idx_defaults")

        data_sample = SegDataSample()
        timer.step("datasample_create")

        data_sample.set_metainfo(metainfo)
        timer.step("datasample_set_metainfo")

        data_sample.gt_sem_seg = PixelData(data=gt)
        timer.step("pixeldata_create_assign")

        if self.debug_checks:
            unique_vals = torch.unique(gt[0])
            assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
                f"PPP intensity channel is not binary. Found values: {unique_vals.tolist()}"
            )
        timer.step("debug_assert")

        out = dict(
            inputs=img,
            data_samples=data_sample,
        )
        timer.step("return_dict_build")

        timer.finish()
        GLOBAL_TIMER.sample_done()

        return out