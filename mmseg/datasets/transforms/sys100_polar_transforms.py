from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import InstanceData
from mmseg.datasets.transforms.loading import BaseTransform
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

from aumovio.prefetcheddataset import AsyncPrefetchedDatasetWrapper
from aumovio.sys100dataset import Sys100Dataset


@TRANSFORMS.register_module()
class LoadSys100PolarSample(BaseTransform):
    """Load one image and its variable-length polar radar point list."""

    def __init__(
        self,
        sys100_root: str,
        out_img_size: Tuple[int, int] = (544, 1792),
        undistort: bool = True,
        fovs: Tuple[float, float] = (105.0, 48.4),
        strict: bool = False,
        prefetch: bool = False,
        prefetch_dict: Optional[dict] = None,
        source_is_summary_csv: bool = False,
        max_files=None,
        target_dataset_size=None,
        subsampling_step=None,
        random_subsampling: bool = True,
        seed=None,
    ) -> None:
        prefetch_dict = {} if prefetch_dict is None else dict(prefetch_dict)

        base_dataset = Sys100Dataset(
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
            fovs=fovs,
            strict=strict,
        )

        self.default_doppler = 20.0
        self.fovs = tuple(float(v) for v in fovs)
        self.ds = (
            AsyncPrefetchedDatasetWrapper(base_dataset=base_dataset, **prefetch_dict)
            if prefetch else base_dataset
        )

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(results['idx'])
        sample = self.ds[idx]

        image = sample['jpg']
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise TypeError('SYS100 sample jpg must be a [C, H, W] tensor.')

        image_hwc = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32, copy=False)

        for key in (
            'radar_points_polar',
            'radar_points_xy',
            'radar_points_meta',
            'rad2cam',
            'real_hw',
        ):
            if key not in sample:
                raise KeyError(
                    f'SYS100 sample lacks {key!r}. Ensure the modified '
                    'aumovio/sys100dataset.py with return_polar_points=True is installed.'
                )

        video_path = sample.get('video_path', None)
        metadata = sample.get('meta_data', None)
        frame_in_video = None
        dtype = getattr(metadata, 'dtype', None)
        names = getattr(dtype, 'names', None)
        if names is not None and 'VideoFrameCounter' in names:
            frame_in_video = int(metadata['VideoFrameCounter'])

        metainfo = results.setdefault('metainfo', {})
        metainfo.update(
            dict(
                sample_id=f'sys100_{idx:06d}',
                frame_idx=idx,
                global_idx=idx,
                video_path=video_path,
                video_name=os.path.basename(video_path) if video_path else None,
                video_frame_idx=frame_in_video,
            )
        )

        results['img'] = image_hwc
        results['img_shape'] = image_hwc.shape[:2]
        results['ori_shape'] = image_hwc.shape[:2]
        results['radar_points_polar'] = sample['radar_points_polar'].detach().cpu().float()
        results['radar_points_xy'] = sample['radar_points_xy'].detach().cpu().float()
        results['radar_points_meta'] = sample['radar_points_meta'].detach().cpu().float()
        results['rad2cam'] = sample['rad2cam'].detach().cpu().float()
        results['real_hw'] = sample['real_hw'].detach().cpu().float()
        results['dopplerRange'] = sample.get('dopplerRange', self.default_doppler)
        results['fovs'] = self.fovs

        for key in ('prefetch_unseen', 'prefetch_unseen_count', 'prefetch_idle_time'):
            if key in sample:
                results[key] = sample[key]

        return results


@TRANSFORMS.register_module()
class PPPPolarRandomFlip(BaseTransform):
    """Horizontal image reflection with the matching polar/BEV reflection."""

    def __init__(self, prob: float = 0.5, direction: str = 'horizontal') -> None:
        if direction != 'horizontal':
            raise ValueError('Only horizontal flips are defined for the polar target.')
        self.prob = float(prob)
        self.direction = direction

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() >= self.prob:
            return results

        results['img'] = np.ascontiguousarray(results['img'][:, ::-1, :])

        polar = results['radar_points_polar'].clone()
        xy = results['radar_points_xy'].clone()
        polar[:, 1] = -polar[:, 1]
        xy[:, 1] = -xy[:, 1]

        results['radar_points_polar'] = polar
        results['radar_points_xy'] = xy
        results.setdefault('metainfo', {})['flip'] = True
        results['metainfo']['flip_direction'] = 'horizontal'
        return results


@TRANSFORMS.register_module()
class PPPPolarFilterDomain(BaseTransform):
    """Keep only radar returns inside the modeled polar domain."""

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

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        polar = results['radar_points_polar']
        valid = (
            (polar[:, 0] >= self.r_min)
            & (polar[:, 0] <= self.r_max)
            & (polar[:, 1] >= self.theta_min)
            & (polar[:, 1] <= self.theta_max)
        )

        results['radar_points_polar'] = polar[valid].contiguous()
        results['radar_points_xy'] = results['radar_points_xy'][valid].contiguous()
        results['radar_points_meta'] = results['radar_points_meta'][valid].contiguous()
        return results


@TRANSFORMS.register_module()
class PPPPackPolarInputs(BaseTransform):
    """Pack an image and variable-length radar points into a SegDataSample."""

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        image = results['img']
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        elif not isinstance(image, torch.Tensor):
            raise TypeError(f'Unsupported image type: {type(image)}')
        image = image.contiguous().float()

        polar = results['radar_points_polar'].contiguous().float()
        xy = results['radar_points_xy'].contiguous().float()
        meta = results['radar_points_meta'].contiguous().float()

        if polar.ndim != 2 or polar.shape[1] < 3:
            raise ValueError(f'Expected polar radar points [N, >=3], got {tuple(polar.shape)}.')
        if xy.ndim != 2 or xy.shape[0] != polar.shape[0]:
            raise ValueError('radar_points_xy must have the same point count as polar.')
        if meta.ndim != 2 or meta.shape[0] != polar.shape[0]:
            raise ValueError('radar_points_meta must have the same point count as polar.')

        rad2cam = results['rad2cam'].contiguous().float()
        real_hw = results['real_hw'].contiguous().float()
        metainfo = dict(results.get('metainfo', {}))
        metainfo.update(
            dict(
                img_shape=results.get('img_shape'),
                ori_shape=results.get('ori_shape'),
                rad2cam=rad2cam,
                real_hw=real_hw,
                fovs=tuple(results['fovs']),
            )
        )
        if 'idx' in results:
            metainfo.setdefault('global_idx', int(results['idx']))
            metainfo.setdefault('frame_idx', int(results['idx']))

        data_sample = SegDataSample()
        data_sample.set_metainfo(metainfo)

        point_data = InstanceData()
        point_data.polar = polar
        point_data.xy = xy
        point_data.meta = meta
        data_sample.set_data(dict(gt_radar_points=point_data))

        return dict(inputs=image, data_samples=data_sample)
