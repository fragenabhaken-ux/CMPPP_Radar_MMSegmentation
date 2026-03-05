# mmseg/datasets/transforms/truckscenes_ppp_transforms.py
from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from truckscenes import TruckScenes


@TRANSFORMS.register_module()
class LoadTruckScenesSample(BaseTransform):
    """Load TruckScenes camera image for a given sample token."""

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.1-mini",
        cam_channel: str = "CAMERA_LEFT_FRONT",
        radar_channel: str = "RADAR_LEFT_FRONT",
        verbose: bool = False,
    ):
        super().__init__()
        self.dataroot = str(dataroot)
        self.version = str(version)
        self.cam_channel = str(cam_channel)
        self.radar_channel = str(radar_channel)
        self.verbose = bool(verbose)

        # Lazy init so it works with num_workers>0
        self._trucksc: TruckScenes | None = None

    def _get_trucksc(self) -> TruckScenes:
        if self._trucksc is None:
            self._trucksc = TruckScenes(
                version=self.version,
                dataroot=self.dataroot,
                verbose=self.verbose,
            )
        return self._trucksc

    def transform(self, results: dict) -> dict:
        token = results.get("sample_token") or results.get("token")
        if token is None:
            raise KeyError("LoadTruckScenesSample expects results['sample_token'] or results['token'].")

        trucksc = self._get_trucksc()

        results["trucksc"] = trucksc
        results["sample_token"] = token
        results["cam_channel"] = self.cam_channel
        results["radar_channel"] = self.radar_channel

        sample = trucksc.get("sample", token)
        if self.cam_channel not in sample["data"]:
            raise KeyError(
                f"Camera channel {self.cam_channel!r} not in sample['data'] "
                f"keys={list(sample['data'].keys())}"
            )

        cam_token = sample["data"][self.cam_channel]
        img_path = trucksc.get_sample_data_path(cam_token)

        import cv2
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"cv2.imread failed for: {img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)

        h, w = img.shape[:2]
        results["img"] = img
        results["img_shape"] = (h, w)
        results["ori_shape"] = (h, w)

        results["img_path"] = img_path
        results["filename"] = img_path
        results["ori_filename"] = img_path

        return results


@TRANSFORMS.register_module()
class BuildTruckScenesPPPHeatmap(BaseTransform):
    """Build marked PPP targets from TruckScenes radar projected into camera image.

    Produces:
        results["gt_seg_map"] : float32 HxWx3
            channel 0: PPP mask (0/1)
            channel 1: depth (clipped)
            channel 2: velocity (placeholder zeros unless you implement extraction)
        results["seg_fields"] : ["gt_seg_map"]
    """

    def __init__(
        self,
        cam_channel: str = "CAMERA_LEFT_FRONT",
        radar_channel: str = "RADAR_LEFT_FRONT",
        dot_size: int = 2,
        clip_depth=(0.0, 150.0),
        clip_vel=(-20.0, 20.0),
        # optional fallback if results doesn't include trucksc
        dataroot: str | None = None,
        version: str = "v1.1-mini",
        verbose: bool = False,
    ):
        super().__init__()
        self.cam_channel = str(cam_channel)
        self.radar_channel = str(radar_channel)
        self.dot_size = int(dot_size)
        self.clip_depth = clip_depth
        self.clip_vel = clip_vel

        self.dataroot = str(dataroot) if dataroot is not None else None
        self.version = str(version)
        self.verbose = bool(verbose)

        self._trucksc: TruckScenes | None = None

    def _get_trucksc(self) -> TruckScenes:
        if self._trucksc is None:
            if self.dataroot is None:
                raise RuntimeError(
                    "No trucksc in results and no dataroot provided to BuildTruckScenesPPPHeatmap."
                )
            self._trucksc = TruckScenes(
                version=self.version,
                dataroot=self.dataroot,
                verbose=self.verbose,
            )
        return self._trucksc

    @staticmethod
    def _tf(rec) -> np.ndarray:
        """4x4 transform from a record with 'translation' and quaternion 'rotation'."""
        from pyquaternion import Quaternion
        q = Quaternion(rec["rotation"])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = np.array(rec["translation"], dtype=np.float64)
        return T

    @staticmethod
    def _pcd_numpy_dtype(fields, sizes, types, counts):
        """Build a numpy dtype for a PCD binary block."""
        type_map = {
            ("F", 4): np.float32,
            ("F", 8): np.float64,
            ("I", 1): np.int8,
            ("I", 2): np.int16,
            ("I", 4): np.int32,
            ("I", 8): np.int64,
            ("U", 1): np.uint8,
            ("U", 2): np.uint16,
            ("U", 4): np.uint32,
            ("U", 8): np.uint64,
        }

        dtype_fields = []
        for name, sz, tp, cnt in zip(fields, sizes, types, counts):
            key = (tp, sz)
            if key not in type_map:
                raise RuntimeError(f"Unsupported PCD TYPE/SIZE combo: TYPE={tp} SIZE={sz} for field {name!r}")
            base = type_map[key]
            if cnt == 1:
                dtype_fields.append((name, base))
            else:
                dtype_fields.append((name, base, (cnt,)))
        return np.dtype(dtype_fields)

    def _load_radar_pcd_xyz(self, pcd_path: str) -> np.ndarray:
        """Load xyz from a PCD file (supports DATA binary).

        Returns homogeneous 4xN points in the radar sensor frame.
        """
        # Read header (ASCII) first
        header = {}
        header_lines = []
        with open(pcd_path, "rb") as f:
            while True:
                line_b = f.readline()
                if not line_b:
                    raise RuntimeError(f"PCD header ended unexpectedly: {pcd_path}")
                line = line_b.decode("utf-8", errors="ignore").strip()
                header_lines.append(line)
                if line == "" or line.startswith("#"):
                    continue
                parts = line.split()
                key = parts[0].upper()
                vals = parts[1:]
                header[key] = vals
                if key == "DATA":
                    data_type = vals[0].lower()
                    break

            if data_type == "ascii":
                # (unlikely for your files, but support anyway)
                fields = header["FIELDS"]
                data = np.loadtxt(f, dtype=np.float64)
                if data.ndim == 1:
                    data = data[None, :]
                ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
                xyz = data[:, [ix, iy, iz]].T
                ones = np.ones((1, xyz.shape[1]), dtype=xyz.dtype)
                return np.vstack([xyz, ones])

            if data_type != "binary":
                raise RuntimeError(
                    f"Unsupported PCD DATA type {data_type!r} for {pcd_path}. "
                    f"Only 'ascii' and 'binary' supported."
                )

            # Parse required header entries
            fields = header.get("FIELDS")
            sizes = header.get("SIZE")
            types = header.get("TYPE")
            counts = header.get("COUNT")
            width = header.get("WIDTH")
            height = header.get("HEIGHT")
            points = header.get("POINTS")

            if fields is None or sizes is None or types is None:
                raise RuntimeError(f"PCD missing FIELDS/SIZE/TYPE in header: {pcd_path}")

            fields = [str(x) for x in fields]
            sizes = [int(x) for x in sizes]
            types = [str(x) for x in types]
            if counts is None:
                counts = [1] * len(fields)
            else:
                counts = [int(x) for x in counts]

            if points is not None:
                n_pts = int(points[0])
            elif width is not None and height is not None:
                n_pts = int(width[0]) * int(height[0])
            else:
                raise RuntimeError(f"PCD missing POINTS or WIDTH/HEIGHT in header: {pcd_path}")

            dtype = self._pcd_numpy_dtype(fields, sizes, types, counts)

            # Read binary blob for n_pts records
            # np.fromfile can't be used with already-opened file reliably in some contexts,
            # so we read exact bytes.
            row_bytes = dtype.itemsize
            blob = f.read(n_pts * row_bytes)
            if len(blob) != n_pts * row_bytes:
                raise RuntimeError(
                    f"PCD binary payload too short. Expected {n_pts * row_bytes} bytes, got {len(blob)}. "
                    f"File: {pcd_path}"
                )

        arr = np.frombuffer(blob, dtype=dtype, count=n_pts)

        if "x" not in arr.dtype.names or "y" not in arr.dtype.names or "z" not in arr.dtype.names:
            raise RuntimeError(f"PCD fields do not contain x/y/z. fields={arr.dtype.names}, file={pcd_path}")

        x = arr["x"].astype(np.float64, copy=False)
        y = arr["y"].astype(np.float64, copy=False)
        z = arr["z"].astype(np.float64, copy=False)

        xyz = np.vstack([x, y, z])  # 3xN
        ones = np.ones((1, xyz.shape[1]), dtype=xyz.dtype)
        return np.vstack([xyz, ones])  # 4xN

    def _project_radar_to_image(self, trucksc: TruckScenes, sample_token: str):
        """Project radar points into the camera image using devkit-style transforms."""
        sample = trucksc.get("sample", sample_token)

        cam_token = sample["data"][self.cam_channel]
        radar_token = sample["data"][self.radar_channel]

        # Camera intrinsics + poses
        cam_sd = trucksc.get("sample_data", cam_token)
        cam_cs = trucksc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_ep = trucksc.get("ego_pose", cam_sd["ego_pose_token"])
        K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)

        # Radar: TruckScenes get_sample_data returns (path, boxes, ...)
        ret = trucksc.get_sample_data(radar_token)
        radar_path = ret[0] if isinstance(ret, (tuple, list)) else ret
        if not isinstance(radar_path, str):
            radar_path = trucksc.get_sample_data_path(radar_token)

        radar_points = self._load_radar_pcd_xyz(radar_path)  # 4xN in radar sensor frame

        radar_sd = trucksc.get("sample_data", radar_token)
        radar_cs = trucksc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])
        radar_ep = trucksc.get("ego_pose", radar_sd["ego_pose_token"])

        # Chain: radar -> ego(radar) -> global -> ego(cam) -> cam
        T_ego_radar = self._tf(radar_cs)
        T_global_ego_radar = self._tf(radar_ep)
        T_ego_cam_global = np.linalg.inv(self._tf(cam_ep))
        T_cam_ego = np.linalg.inv(self._tf(cam_cs))
        T_cam_radar = T_cam_ego @ T_ego_cam_global @ T_global_ego_radar @ T_ego_radar

        pts_cam = T_cam_radar @ radar_points  # 4xN

        in_front = pts_cam[2, :] > 1e-6
        pts_cam = pts_cam[:, in_front]

        from truckscenes.utils.geometry_utils import view_points
        pts_img = view_points(pts_cam[:3, :], K, normalize=True)  # 3xN

        u = pts_img[0, :]
        v = pts_img[1, :]
        depth = pts_cam[2, :]

        return u, v, depth

    def _stamp(self, gt: np.ndarray, u: np.ndarray, v: np.ndarray, depths: np.ndarray, vel: np.ndarray) -> None:
        """Stamp dot_size x dot_size squares centered at (u,v)."""
        H, W, _ = gt.shape
        r = max(0, self.dot_size // 2)

        for ui, vi, di, vli in zip(u, v, depths, vel):
            x0 = max(0, int(ui) - r)
            x1 = min(W, int(ui) + r + 1)
            y0 = max(0, int(vi) - r)
            y1 = min(H, int(vi) + r + 1)

            gt[y0:y1, x0:x1, 0] = 1.0
            gt[y0:y1, x0:x1, 1] = float(di)
            gt[y0:y1, x0:x1, 2] = float(vli)

    def transform(self, results: dict) -> dict:
        img = results["img"]
        H, W = img.shape[:2]
        token = results["sample_token"]

        trucksc = results.get("trucksc", None)
        if trucksc is None:
            trucksc = self._get_trucksc()
            results["trucksc"] = trucksc

        u, v, depths = self._project_radar_to_image(trucksc, token)

        u = u.astype(np.int32)
        v = v.astype(np.int32)
        depths = np.asarray(depths, dtype=np.float32)

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        depths = depths[valid]

        # TODO: real velocity extraction; placeholder zeros
        vel = np.zeros_like(depths, dtype=np.float32)

        depths = np.clip(depths, *self.clip_depth)
        vel = np.clip(vel, *self.clip_vel)

        gt = np.zeros((H, W, 3), dtype=np.float32)

        if self.dot_size <= 1:
            gt[v, u, 0] = 1.0
            gt[v, u, 1] = depths
            gt[v, u, 2] = vel
        else:
            self._stamp(gt, u, v, depths, vel)

        results["gt_seg_map"] = gt
        results["seg_fields"] = ["gt_seg_map"]
        return results
    
    @TRANSFORMS.register_module()
    class TruckScenesPackSegInputs(BaseTransform):
        """Pack img + gt_seg_map into mmseg SegDataSample.

        Expects:
            results["img"] (H,W,3) uint8
            results["gt_seg_map"] (H,W,3) float32
            results["img_shape"], results["ori_shape"]
            results["sample_token"] (str)

        Produces:
            dict(inputs=..., data_samples=...)
        """
        def __init__(self):
            super().__init__()

        def transform(self, results: dict) -> dict:
            import torch
            import numpy as np
            from mmengine.structures import PixelData
            from mmseg.structures import SegDataSample

            img = results["img"]          # HWC uint8
            gt = results["gt_seg_map"]    # HWC float32

            # ✅ fix negative strides (e.g. after np.flip / slicing)
            img = np.ascontiguousarray(img)
            gt  = np.ascontiguousarray(gt)

            inputs = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            gt_t   = torch.from_numpy(gt).permute(2, 0, 1).contiguous().float()

            data_sample = SegDataSample()
            data_sample.set_metainfo(
                dict(
                    img_shape=results.get("img_shape"),
                    ori_shape=results.get("ori_shape"),
                    sample_token=results.get("sample_token"),
                    idx=int(results["idx"]) if "idx" in results else None,
                )
            )
            data_sample.gt_sem_seg = PixelData(data=gt_t)

            return dict(inputs=inputs, data_samples=data_sample)