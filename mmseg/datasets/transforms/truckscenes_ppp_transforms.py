# mmseg/datasets/transforms/truckscenes_ppp_transforms.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import zlib

from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

# from truckscenes import TruckScenes


@TRANSFORMS.register_module()
class LoadTruckScenesSample(BaseTransform):
    """Loads RGB image for a TruckScenes sample_token and stores TruckScenes handle in results.

    Important:
      - Lazy-init TruckScenes inside each dataloader worker.
      - Stores: img (HWC RGB uint8), img_shape, ori_shape, img_path, sample_token, trucksc, channels.
    """

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

        self._trucksc: Optional[TruckScenes] = None

    def _get_trucksc(self) -> TruckScenes:
        if self._trucksc is None:
            self._trucksc = TruckScenes(
                version=self.version,
                dataroot=self.dataroot,
                verbose=self.verbose,
            )
        return self._trucksc

    def transform(self, results: dict) -> dict:
        token = results.get("sample_token")
        if token is None:
            raise KeyError("LoadTruckScenesSample expects results['sample_token'].")

        trucksc = self._get_trucksc()

        # store for downstream transforms
        results["trucksc"] = trucksc
        results["sample_token"] = token
        results["cam_channel"] = self.cam_channel
        results["radar_channel"] = self.radar_channel

        sample = trucksc.get("sample", token)

        if self.cam_channel not in sample["data"]:
            raise KeyError(
                f"Camera channel {self.cam_channel!r} not in sample['data'] keys={list(sample['data'].keys())}"
            )
        if self.radar_channel not in sample["data"]:
            raise KeyError(
                f"Radar channel {self.radar_channel!r} not in sample['data'] keys={list(sample['data'].keys())}"
            )

        cam_token = sample["data"][self.cam_channel]
        img_path = trucksc.get_sample_data_path(cam_token)

        import cv2

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"cv2.imread failed for: {img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img)

        # Use HWC shapes (safer with visualizers/hooks that expect channel dim)
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape

        # common fields used by various hooks
        results["img_path"] = img_path
        results["filename"] = img_path
        results["ori_filename"] = img_path

        return results


@TRANSFORMS.register_module()
class BuildTruckScenesPPPHeatmap(BaseTransform):
    """Builds gt_seg_map (H, W, 3): [mark(0/1), depth(z_cam), abs_vel] from radar projected into camera image.

    Notes (MAN TruckScenes devkit):
      RadarPointCloud dims are fixed to 7:
        [x, y, z, vrel_x, vrel_y, vrel_z, rcs]
    """

    def __init__(
        self,
        cam_channel: str = "CAMERA_LEFT_FRONT",
        radar_channel: str = "RADAR_LEFT_FRONT",
        dot_size: int = 2,
        clip_depth: Tuple[float, float] = (0.0, 150.0),
        clip_vel: Tuple[float, float] = (0.0, 30.0),
        # optional fallback if trucksc not passed (usually results["trucksc"] exists)
        dataroot: Optional[str] = None,
        version: str = "v1.1-mini",
        verbose: bool = False,
        # whether to use 3D speed sqrt(vx^2+vy^2+vz^2). If False -> planar sqrt(vx^2+vy^2)
        use_3d_speed: bool = True,
    ):
        super().__init__()
        self.cam_channel = str(cam_channel)
        self.radar_channel = str(radar_channel)
        self.dot_size = int(dot_size)
        self.clip_depth = clip_depth
        self.clip_vel = clip_vel
        self.use_3d_speed = bool(use_3d_speed)

        self.dataroot = str(dataroot) if dataroot is not None else None
        self.version = str(version)
        self.verbose = bool(verbose)
        self._trucksc: Optional[TruckScenes] = None

    def _get_trucksc(self) -> TruckScenes:
        if self._trucksc is None:
            if self.dataroot is None:
                raise RuntimeError("No trucksc in results and no dataroot provided to BuildTruckScenesPPPHeatmap.")
            self._trucksc = TruckScenes(
                version=self.version,
                dataroot=self.dataroot,
                verbose=self.verbose,
            )
        return self._trucksc

    @staticmethod
    def _tf_from_rec(rec) -> np.ndarray:
        """Build 4x4 matrix from a record that has 'translation' and 'rotation' (quaternion)."""
        from pyquaternion import Quaternion

        q = Quaternion(rec["rotation"])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = np.array(rec["translation"], dtype=np.float64)
        return T

    def _project_radar_to_image(
        self,
        trucksc: TruckScenes,
        sample_token: str,
        cam_channel: str,
        radar_channel: str,
        min_dist: float = 1.0,   # exakt wie in deren map_pointcloud_to_image()
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        1:1 wie TruckScenesExplorer.map_pointcloud_to_image(), aber liefert zusätzlich abs_vel.

        Returns:
            u, v: float pixel coords
            depth: pc.points[2] in camera frame (z away from camera)
            abs_vel: speed from vrel components (after rotations applied by RadarPointCloud.rotate)
        """
        from pyquaternion import Quaternion
        from truckscenes.utils.geometry_utils import view_points
        from truckscenes.utils.data_classes import RadarPointCloud

        sample = trucksc.get("sample", sample_token)

        cam_token = sample["data"][cam_channel]
        radar_token = sample["data"][radar_channel]

        cam = trucksc.get("sample_data", cam_token)
        pointsensor = trucksc.get("sample_data", radar_token)

        pcl_path = trucksc.get_sample_data_path(radar_token)
        pc = RadarPointCloud.from_file(pcl_path)

        # ---- EXACT SAME TRANSFORM CHAIN AS LIB ----

        # 1) point sensor -> ego (at radar time)
        cs_record = trucksc.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"], dtype=np.float64))

        # 2) ego -> global (at radar time)
        poserecord = trucksc.get("ego_pose", pointsensor["ego_pose_token"])
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc.translate(np.array(poserecord["translation"], dtype=np.float64))

        # 3) global -> ego (at camera time)
        poserecord = trucksc.get("ego_pose", cam["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"], dtype=np.float64))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # 4) ego -> camera
        cs_record = trucksc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        pc.translate(-np.array(cs_record["translation"], dtype=np.float64))
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        # 5) project
        depths = pc.points[2, :]  # z in camera frame (library comment: points away from camera)

        # pixels
        K = np.array(cs_record["camera_intrinsic"], dtype=np.float64)
        pts_img = view_points(pc.points[:3, :], K, normalize=True)  # 3xN

        u = pts_img[0, :]
        v = pts_img[1, :]

        # filter like library
        # (they also filter out very near points to avoid casing / slight async)
        # NOTE: im.size available in their function, here we filter image bounds later in transform()
        mask = depths > float(min_dist)

        u = u[mask]
        v = v[mask]
        depth = depths[mask]

        # velocities:
        # In MAN devkit RadarPointCloud dims are [x,y,z,vrel_x,vrel_y,vrel_z,rcs] (7,N).
        # Rotations above already rotated velocity vectors too (RadarPointCloud.rotate overrides).

        # ACHTUNG TODO !!!: 
        # This is fine only if RadarPointCloud.rotate() rotates the velocity components too (some devkits 
        # do, some don’t; your comment suggests MAN’s does).
        # Quick sanity test:
        # Pick one radar point before/after rotation and verify [vrel_x, vrel_y, vrel_z] 
        # changes consistently with the rotation. If velocities are not rotated, you’ll still get 
        # a valid magnitude (norm invariant under rotation), but the meaning of vx/vy/vz could be 
        # inconsistent if later you ever want signed components. For abs_vel, you’re safe. 
        vx_rel = pc.points[3, mask]
        vy_rel = pc.points[4, mask]
        vz_rel = pc.points[5, mask]

        # pick the time you want (camera time is typical if you already end in camera frame)
        t = cam["timestamp"]

        # trucksc.ego_motion_cabin is the table (list of dicts); pick nearest timestamp
        ego_motion = min(trucksc.ego_motion_cabin, key=lambda r: abs(r["timestamp"] - t))

        # now you can read vx,vy,vz
        v_cabin = np.array([ego_motion["vx"], ego_motion["vy"], ego_motion["vz"]], dtype=np.float64)

        # build a tiny "point cloud" that only carries our 3-vector in the first 3 rows,
        # We need this, because rotate() is a method of RadarPointCloud.
        v_pc = RadarPointCloud(np.zeros((7, 1), dtype=np.float64))
        v_pc.points[0:3, 0] = v_cabin

        # 2) ego -> global (at radar time)  [same rotate as your chain, but NO translate]
        poserecord = trucksc.get("ego_pose", pointsensor["ego_pose_token"])
        v_pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)

        # 3) global -> ego (at camera time) [NO translate]
        poserecord = trucksc.get("ego_pose", cam["ego_pose_token"])
        v_pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # 4) ego -> camera                  [NO translate]
        cs_record = trucksc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        v_pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        v_cabin_cam = v_pc.points[0:3, 0].copy()  # (3,)

        # ---- absolute/object velocity in camera frame: v_obj = v_rel + v_cabin ----
        vx_obj = vx_rel + v_cabin_cam[0]
        vy_obj = vy_rel + v_cabin_cam[1]
        vz_obj = vz_rel + v_cabin_cam[2]

        # project object velocity onto line of sight
        v_rad = np.sqrt(vx_obj**2  + vy_obj**2 + vz_obj**2)

        v_rad = np.nan_to_num(
            v_rad,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return u, v, depth, v_rad

    def _stamp_agg(
        self,
        mark: np.ndarray,
        depth_map: np.ndarray,
        vel_map: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        depth: np.ndarray,
        vel: np.ndarray,
    ) -> None:
        """Aggregate stamping:
        - mark: 1 if any point hits pixel
        - depth_map: min depth
        - vel_map: max abs velocity
        """
        H, W = mark.shape
        r = max(0, self.dot_size // 2)

        for ui, vi, di, vi_abs in zip(u, v, depth, vel):
            x0 = max(0, ui - r)
            x1 = min(W, ui + r + 1)
            y0 = max(0, vi - r)
            y1 = min(H, vi + r + 1)

            mark[y0:y1, x0:x1] = 1
            depth_map[y0:y1, x0:x1] = np.minimum(depth_map[y0:y1, x0:x1], di)
            vel_map[y0:y1, x0:x1] = np.maximum(vel_map[y0:y1, x0:x1], vi_abs)

    def transform(self, results: dict) -> dict:
        img = results["img"]
        H, W = img.shape[:2]
        token = results["sample_token"]

        trucksc = results.get("trucksc")
        if trucksc is None:
            trucksc = self._get_trucksc()
            results["trucksc"] = trucksc

        cam_channel = results.get("cam_channel", self.cam_channel)
        radar_channel = results.get("radar_channel", self.radar_channel)

        u, v, depth, abs_vel = self._project_radar_to_image(
            trucksc, token, cam_channel=cam_channel, radar_channel=radar_channel
        )

        # round to nearest pixel (less bias than floor)
        u = np.rint(u).astype(np.int32)
        v = np.rint(v).astype(np.int32)

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        depth = depth[valid].astype(np.float32)
        abs_vel = abs_vel[valid].astype(np.float32)

        depth = np.clip(depth, *self.clip_depth)
        abs_vel = np.clip(abs_vel, *self.clip_vel)

        # aggregation maps
        mark = np.zeros((H, W), dtype=np.uint8)
        depth_map = np.full((H, W), np.inf, dtype=np.float32)
        vel_map = np.zeros((H, W), dtype=np.float32)

        if u.size > 0:
            self._stamp_agg(mark, depth_map, vel_map, u, v, depth, abs_vel)

        # replace inf (no hit) with 0.0
        depth_map[~np.isfinite(depth_map)] = 0.0

        gt = np.stack([mark.astype(np.float32), depth_map, vel_map], axis=-1).astype(np.float32)

        results["gt_seg_map"] = gt
        results["seg_fields"] = ["gt_seg_map"]
        return results


@TRANSFORMS.register_module()
class TruckScenesPackSegInputs(BaseTransform):
    """Like your PPPPackSegInputs, but does NOT require results['idx'] (uses token fallback),
    and avoids negative-stride numpy -> torch conversion.
    """

    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        img = results["img"]  # HWC RGB uint8
        gt = results["gt_seg_map"]  # HWC float32

        img = np.ascontiguousarray(img)
        gt = np.ascontiguousarray(gt)

        inputs = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).contiguous().float()

        data_sample = SegDataSample()

        if "idx" in results:
            frame_idx = int(results["idx"])
        else:
            tok = results.get("sample_token", "")
            frame_idx = int(zlib.crc32(tok.encode("utf-8")) & 0xFFFFFFFF) if tok else -1

        data_sample.set_metainfo(
            dict(
                img_shape=results.get("img_shape"),
                ori_shape=results.get("ori_shape"),
                frame_idx=frame_idx,
                sample_token=results.get("sample_token", None),
                cam_channel=results.get("cam_channel", None),
                radar_channel=results.get("radar_channel", None),
            )
        )

        data_sample.gt_sem_seg = PixelData(data=gt_t)

        return dict(inputs=inputs, data_samples=data_sample)