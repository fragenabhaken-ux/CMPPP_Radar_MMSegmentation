from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Wedge


@HOOKS.register_module()
class PPPPolarDenseSixBEVDemoVisHook(Hook):
    """Write the six dense-BEV demo panels as six separate image series.

    For each selected frame, this hook writes one PNG into each of six panel
    directories:

        <runner.work_dir>/<out_dir>/
            intensity_with_gt/<phase>/<video>/frame_...png
            velocity_with_gt/<phase>/<video>/frame_...png
            rcs_with_gt/<phase>/<video>/frame_...png
            sampled_plain/<phase>/<video>/frame_...png
            sampled_velocity/<phase>/<video>/frame_...png
            sampled_rcs/<phase>/<video>/frame_...png

    The dense prediction is expected to have three channels:
        0: log intensity with respect to dr dtheta
        1: raw velocity mark
        2: raw RCS mark

    Preferred GT source is sample.metainfo['polar_gt_points'] with columns
    (r, theta, velocity, rcs). If unavailable, dense occupied-cell centres are
    used as a fallback.
    """

    PANEL_KEYS = (
        'intensity_with_gt',
        'velocity_with_gt',
        'rcs_with_gt',
        'sampled_plain',
        'sampled_velocity',
        'sampled_rcs',
    )

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = 'demo_frames_polar_dense_six_bev',
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        bev_x_min: float = 0.0,
        bev_x_max: float = 104.0,
        bev_y_min: float = -52.0,
        bev_y_max: float = 52.0,
        intensity_gamma: float = 0.3,
        heatmap_cmap: str = 'hot',
        point_size: float = 20.0,
        sample_point_size: float | None = None,
        point_color: str = 'white',
        sample_point_color: str = 'white',
        point_alpha: float = 0.97,
        sample_point_alpha: float = 0.97,
        point_edge_color: str = 'black',
        point_edge_width: float = 0.45,
        velocity_mode: str = 'signed',
        velocity_link: str = 'cube',
        velocity_vmin: float = -11.0,
        velocity_vmax: float = 11.0,
        velocity_cmap: str = 'coolwarm',
        rcs_link: str = 'cube',
        rcs_vmin: float = -60.0,
        rcs_vmax: float = 10.0,
        rcs_cmap: str = 'viridis',
        sampling_seed_offset: int = 1000,
        show_colorbar: bool = True,
        show_textbox: bool = True,
        show_axis_labels: bool = True,
        show_range_rings: bool = True,
        range_ring_step: float = 10.0,
        range_ring_color: str = 'gray',
        range_ring_alpha: float = 0.12,
        range_ring_linewidth: float = 0.8,
        dpi: int = 150,
        figsize: tuple[float, float] = (8.0, 8.0),
        image_ext: str = 'png',
        write_all_ranks: bool = False,
    ) -> None:
        self.train_interval = int(train_interval)
        self.max_samples = {
            'train': int(max_samples_train),
            'val': int(max_samples_val),
            'test': int(max_samples_test),
        }

        self.out_dir = str(out_dir)
        self.image_ext = str(image_ext).lstrip('.')
        self.write_all_ranks = bool(write_all_ranks)

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.bev_x_min = float(bev_x_min)
        self.bev_x_max = float(bev_x_max)
        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)

        self.intensity_gamma = float(intensity_gamma)
        self.heatmap_cmap = str(heatmap_cmap)

        self.point_size = float(point_size)
        self.sample_point_size = float(sample_point_size) if sample_point_size is not None else float(point_size)
        self.point_color = str(point_color)
        self.sample_point_color = str(sample_point_color)
        self.point_alpha = float(point_alpha)
        self.sample_point_alpha = float(sample_point_alpha)
        self.point_edge_color = str(point_edge_color)
        self.point_edge_width = float(point_edge_width)

        self.velocity_mode = str(velocity_mode).lower()
        self.velocity_link = str(velocity_link).lower()
        self.velocity_vmin = float(velocity_vmin)
        self.velocity_vmax = float(velocity_vmax)
        self.velocity_cmap = str(velocity_cmap)

        self.rcs_link = str(rcs_link).lower()
        self.rcs_vmin = float(rcs_vmin)
        self.rcs_vmax = float(rcs_vmax)
        self.rcs_cmap = str(rcs_cmap)

        self.sampling_seed_offset = int(sampling_seed_offset)
        self.show_colorbar = bool(show_colorbar)
        self.show_textbox = bool(show_textbox)
        self.show_axis_labels = bool(show_axis_labels)
        self.show_range_rings = bool(show_range_rings)
        self.range_ring_step = float(range_ring_step)
        self.range_ring_color = str(range_ring_color)
        self.range_ring_alpha = float(range_ring_alpha)
        self.range_ring_linewidth = float(range_ring_linewidth)
        self.dpi = int(dpi)
        self.figsize = tuple(float(v) for v in figsize)

        if self.r_min >= self.r_max:
            raise ValueError('r_min must be smaller than r_max.')
        if self.theta_min >= self.theta_max:
            raise ValueError('theta_min must be smaller than theta_max.')
        if self.bev_x_min >= self.bev_x_max:
            raise ValueError('bev_x_min must be smaller than bev_x_max.')
        if self.bev_y_min >= self.bev_y_max:
            raise ValueError('bev_y_min must be smaller than bev_y_max.')
        if self.velocity_mode not in {'signed', 'absolute'}:
            raise ValueError("velocity_mode must be 'signed' or 'absolute'.")
        if self.velocity_link not in {'identity', 'cube'}:
            raise ValueError("velocity_link must be 'identity' or 'cube'.")
        if self.velocity_vmin >= self.velocity_vmax:
            raise ValueError('velocity_vmin must be smaller than velocity_vmax.')
        if self.rcs_link not in {'identity', 'cube'}:
            raise ValueError("rcs_link must be 'identity' or 'cube'.")
        if self.rcs_vmin >= self.rcs_vmax:
            raise ValueError('rcs_vmin must be smaller than rcs_vmax.')
        if not 0.0 <= self.point_alpha <= 1.0:
            raise ValueError('point_alpha must be in [0, 1].')
        if not 0.0 <= self.sample_point_alpha <= 1.0:
            raise ValueError('sample_point_alpha must be in [0, 1].')
        if self.range_ring_step <= 0.0:
            raise ValueError('range_ring_step must be positive.')
        if not 0.0 <= self.range_ring_alpha <= 1.0:
            raise ValueError('range_ring_alpha must be in [0, 1].')
        if self.range_ring_linewidth < 0.0:
            raise ValueError('range_ring_linewidth must be non-negative.')

        self._written_in_run = {'val': 0, 'test': 0}

    def _rank_allowed(self) -> bool:
        return self.write_all_ranks or is_main_process()

    def before_val_epoch(self, runner) -> None:
        self._written_in_run['val'] = 0

    def before_test_epoch(self, runner) -> None:
        self._written_in_run['test'] = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples['train'] <= 0 or self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return
        self._run(runner=runner, data_batch=data_batch, phase='train')

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples['val'] <= 0:
            return
        if self._written_in_run['val'] >= self.max_samples['val']:
            return
        self._run(runner=runner, data_batch=data_batch, phase='val')

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        del batch_idx, outputs
        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples['test'] <= 0:
            return
        if self._written_in_run['test'] >= self.max_samples['test']:
            return
        self._run(runner=runner, data_batch=data_batch, phase='test')

    def _run(self, runner, data_batch, phase: str) -> None:
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        was_training = model.training
        model.eval()

        proc = model.data_preprocessor(data_batch)
        inputs = proc['inputs']
        data_samples = proc['data_samples']

        with torch.no_grad():
            preds = model.decode_head(model.extract_feat(inputs))

        if not torch.is_tensor(preds):
            raise TypeError(
                'PPPPolarDenseSixBEVDemoVisHook expects decode_head output to be a tensor, '
                f'got {type(preds)}.'
            )
        if preds.ndim != 4 or preds.shape[1] != 3:
            raise ValueError(
                'Expected dense polar predictions with shape [B, 3, H, W], got '
                f'{tuple(preds.shape)}.'
            )

        if phase == 'train':
            n_vis = min(self.max_samples['train'], preds.shape[0])
            start_idx = 0
        else:
            remaining = self.max_samples[phase] - self._written_in_run[phase]
            if remaining <= 0:
                if was_training:
                    model.train()
                return
            n_vis = min(remaining, preds.shape[0])
            start_idx = self._written_in_run[phase]

        for i in range(n_vis):
            self._visualize_sample(
                runner=runner,
                phase=phase,
                pred_grid=preds[i],
                sample=data_samples[i],
                sample_idx=start_idx + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _extract_dense_gt(sample) -> torch.Tensor:
        if not hasattr(sample, 'gt_sem_seg'):
            raise AttributeError('Data sample has no gt_sem_seg field.')
        gt = sample.gt_sem_seg.data
        if not torch.is_tensor(gt):
            raise TypeError(f'Expected sample.gt_sem_seg.data to be a tensor, got {type(gt)}.')
        if gt.ndim == 4:
            if gt.shape[0] != 1:
                raise ValueError(f'Unexpected batched GT inside one data sample: {tuple(gt.shape)}.')
            gt = gt[0]
        if gt.ndim != 3 or gt.shape[0] != 3:
            raise ValueError(f'Expected dense polar GT with shape [3, H, W], got {tuple(gt.shape)}.')
        return gt.detach().float().cpu()

    def _decode_velocity(self, raw_velocity: torch.Tensor) -> np.ndarray:
        raw_velocity = raw_velocity.detach().float().cpu()
        if self.velocity_link == 'identity':
            velocity = raw_velocity
        elif self.velocity_link == 'cube':
            velocity = raw_velocity.pow(3)
        else:
            raise RuntimeError(f'Unsupported velocity_link={self.velocity_link!r}.')
        if self.velocity_mode == 'absolute':
            velocity = velocity.abs()
        return velocity.numpy()

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        raw_rcs = raw_rcs.detach().float().cpu()
        if self.rcs_link == 'identity':
            rcs = raw_rcs
        elif self.rcs_link == 'cube':
            rcs = raw_rcs.pow(3)
        else:
            raise RuntimeError(f'Unsupported rcs_link={self.rcs_link!r}.')
        return rcs.numpy()

    def _geometry(self, sample, h_r: int, w_theta: int) -> dict[str, Any]:
        meta = sample.metainfo
        r_min = float(meta.get('polar_r_min', self.r_min))
        r_max = float(meta.get('polar_r_max', self.r_max))
        theta_min = float(meta.get('polar_theta_min', self.theta_min))
        theta_max = float(meta.get('polar_theta_max', self.theta_max))

        dr = (r_max - r_min) / h_r
        dtheta = (theta_max - theta_min) / w_theta

        r_edges = np.linspace(r_min, r_max, h_r + 1, dtype=np.float64)
        theta_edges = np.linspace(theta_min, theta_max, w_theta + 1, dtype=np.float64)
        rr_edges, tt_edges = np.meshgrid(r_edges, theta_edges, indexing='ij')

        return {
            'r_min': r_min,
            'r_max': r_max,
            'theta_min': theta_min,
            'theta_max': theta_max,
            'dr': dr,
            'dtheta': dtheta,
            'x_edges': rr_edges * np.cos(tt_edges),
            'y_edges': rr_edges * np.sin(tt_edges),
        }

    def _gt_pack(self, sample, gt: torch.Tensor, geometry: dict[str, Any]) -> dict[str, np.ndarray | str]:
        meta = sample.metainfo
        if 'polar_gt_points' in meta:
            polar_points = self._to_numpy(meta['polar_gt_points']).astype(np.float64, copy=False)
            if polar_points.ndim != 2 or polar_points.shape[1] < 4:
                raise ValueError(
                    "Expected metainfo['polar_gt_points'] with shape [N, >=4], "
                    f'got {polar_points.shape}.'
                )
            finite = np.all(np.isfinite(polar_points[:, :4]), axis=1)
            polar_points = polar_points[finite]
            r = polar_points[:, 0]
            theta = polar_points[:, 1]
            xy = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1)
            velocity = polar_points[:, 2].copy()
            if self.velocity_mode == 'absolute':
                velocity = np.abs(velocity)
            rcs = polar_points[:, 3].copy()
            return {
                'xy': xy,
                'velocity': velocity,
                'rcs': rcs,
                'source': 'continuous',
            }

        occupancy = gt[0].numpy() > 0.0
        h_r, w_theta = occupancy.shape
        r_centres = geometry['r_min'] + (np.arange(h_r, dtype=np.float64) + 0.5) * geometry['dr']
        theta_centres = geometry['theta_min'] + (np.arange(w_theta, dtype=np.float64) + 0.5) * geometry['dtheta']
        rr, tt = np.meshgrid(r_centres, theta_centres, indexing='ij')
        xy = np.stack((rr[occupancy] * np.cos(tt[occupancy]), rr[occupancy] * np.sin(tt[occupancy])), axis=1)
        velocity = gt[1].numpy()[occupancy].astype(np.float64, copy=False)
        if self.velocity_mode == 'absolute':
            velocity = np.abs(velocity)
        rcs = gt[2].numpy()[occupancy].astype(np.float64, copy=False)
        return {
            'xy': xy,
            'velocity': velocity,
            'rcs': rcs,
            'source': 'cell centres',
        }

    def _sample_dense_polar_ppp(self, lam: np.ndarray, velocity: np.ndarray, rcs: np.ndarray, geometry: dict[str, Any], seed: int) -> dict[str, Any]:
        dr = float(geometry['dr'])
        dtheta = float(geometry['dtheta'])
        h_r, w_theta = lam.shape
        r_edges = np.linspace(geometry['r_min'], geometry['r_max'], h_r + 1)
        theta_edges = np.linspace(geometry['theta_min'], geometry['theta_max'], w_theta + 1)

        cell_mass = np.asarray(lam, dtype=np.float64) * dr * dtheta
        cell_mass = np.clip(cell_mass, 0.0, None)

        rng = np.random.default_rng(int(seed) + self.sampling_seed_offset)
        counts = rng.poisson(cell_mass)

        ri, ti = np.nonzero(counts > 0)
        if len(ri) == 0:
            return {
                'xy': np.zeros((0, 2), dtype=np.float64),
                'velocity': np.zeros((0,), dtype=np.float64),
                'rcs': np.zeros((0,), dtype=np.float64),
                'count': 0,
            }

        reps = counts[ri, ti].astype(np.int64)
        total = int(reps.sum())
        ri_rep = np.repeat(ri, reps)
        ti_rep = np.repeat(ti, reps)

        r0 = r_edges[ri_rep]
        r1 = r_edges[ri_rep + 1]
        t0 = theta_edges[ti_rep]
        t1 = theta_edges[ti_rep + 1]

        r = r0 + rng.random(total) * (r1 - r0)
        theta = t0 + rng.random(total) * (t1 - t0)
        x_forward = r * np.cos(theta)
        y_lateral = r * np.sin(theta)

        return {
            'xy': np.stack([x_forward, y_lateral], axis=1).astype(np.float64),
            'velocity': velocity[ri_rep, ti_rep].astype(np.float64),
            'rcs': rcs[ri_rep, ti_rep].astype(np.float64),
            'count': total,
        }

    def _setup_bev_axes(self, ax, r_min: float, r_max: float) -> None:
        del r_min, r_max
        ax.set_xlim(self.bev_y_max, self.bev_y_min)
        ax.set_ylim(self.bev_x_min, self.bev_x_max)
        ax.set_aspect('equal', adjustable='box')
        if self.show_axis_labels:
            ax.set_xlabel('y lateral [m]', labelpad=1)
            ax.set_ylabel('x forward [m]', labelpad=1)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=9, pad=1)

    def _draw_range_rings(self, ax) -> None:
        if not self.show_range_rings:
            return

        max_radius = max(self.bev_x_max, abs(self.bev_y_min), abs(self.bev_y_max))
        n_rings = int(np.ceil(max_radius / self.range_ring_step))

        for ring_idx in range(1, n_rings + 1):
            outer_radius = ring_idx * self.range_ring_step
            inner_radius = max(0.0, outer_radius - self.range_ring_step)

            if ring_idx % 2 == 1 and self.range_ring_alpha > 0.0:
                band = Wedge(
                    center=(0.0, 0.0),
                    r=outer_radius,
                    theta1=0.0,
                    theta2=180.0,
                    width=outer_radius - inner_radius,
                    facecolor=self.range_ring_color,
                    edgecolor='none',
                    alpha=self.range_ring_alpha,
                    zorder=1.6,
                )
                ax.add_patch(band)

            if self.range_ring_linewidth > 0.0:
                arc = Arc(
                    xy=(0.0, 0.0),
                    width=2.0 * outer_radius,
                    height=2.0 * outer_radius,
                    angle=0.0,
                    theta1=0.0,
                    theta2=180.0,
                    color=self.range_ring_color,
                    linewidth=self.range_ring_linewidth,
                    alpha=min(1.0, self.range_ring_alpha * 1.35 if self.range_ring_alpha > 0.0 else 0.18),
                    zorder=1.7,
                )
                ax.add_patch(arc)


    @staticmethod
    def _textbox(ax, text: str) -> None:
        ax.text(
            0.015, 0.975, text,
            transform=ax.transAxes,
            fontsize=9,
            color='white',
            va='top', ha='left',
            bbox=dict(facecolor='black', alpha=0.60, pad=3),
        )

    @staticmethod
    def _add_inset_colorbar(fig: Figure, ax, mappable, label: str) -> None:
        cax = ax.inset_axes([0.69, 0.045, 0.28, 0.035])
        colorbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
        colorbar.ax.tick_params(labelsize=7, pad=1, length=2)
        colorbar.set_label(label, fontsize=7, labelpad=1)
        cax.set_facecolor((1.0, 1.0, 1.0, 0.85))

    @staticmethod
    def _tagify(value: str) -> str:
        return ''.join(c if c.isalnum() or c in '._-/' else '_' for c in value)

    def _output_path(self, runner, panel_key: str, phase: str, video: str, frame_str: str, global_idx_str: str) -> Path:
        root = Path(self.out_dir)
        if not root.is_absolute():
            root = Path(runner.work_dir) / root
        output_dir = root / panel_key / phase / video
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f'frame_{frame_str}_idx_{global_idx_str}.{self.image_ext}'

    def _new_axes_figure(self) -> tuple[Figure, Any]:
        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.085, 0.070, 0.900, 0.875])
        return fig, ax

    def _save_figure(self, fig: Figure, output_path: Path) -> None:
        fig.savefig(output_path, dpi=self.dpi)
        fig.clear()

    def _visualize_sample(self, runner, phase: str, pred_grid: torch.Tensor, sample, sample_idx: int) -> None:
        del sample_idx
        pred_grid = pred_grid.detach().float().cpu()
        gt = self._extract_dense_gt(sample)
        if tuple(gt.shape[-2:]) != tuple(pred_grid.shape[-2:]):
            raise ValueError('Prediction and GT grid shapes differ.')

        log_lambda = pred_grid[0]
        raw_velocity = pred_grid[1]
        raw_rcs = pred_grid[2]

        lam = torch.exp(log_lambda).numpy()
        velocity = self._decode_velocity(raw_velocity)
        rcs = self._decode_rcs(raw_rcs)

        h_r, w_theta = lam.shape
        geometry = self._geometry(sample, h_r, w_theta)
        expected_count = float(lam.sum() * geometry['dr'] * geometry['dtheta'])

        gt_pack = self._gt_pack(sample, gt, geometry)
        gt_xy = gt_pack['xy']
        gt_velocity = gt_pack['velocity']
        gt_rcs = gt_pack['rcs']
        gt_source = gt_pack['source']
        gt_count = int(gt_xy.shape[0])

        meta = sample.metainfo
        video = self._tagify(str(meta.get('video_name', 'unknown_video')))
        frame = meta.get('video_frame_idx')
        global_idx = meta.get('global_idx', meta.get('frame_idx'))
        frame_str = f'{int(frame):06d}' if frame is not None else 'unknown_frame'
        global_idx_str = f'{int(global_idx):08d}' if global_idx is not None else 'unknown_idx'

        seed_base = int(frame) if frame is not None else int(global_idx if global_idx is not None else sample_idx)
        sample_pack = self._sample_dense_polar_ppp(lam=lam, velocity=velocity, rcs=rcs, geometry=geometry, seed=seed_base)
        sample_xy = sample_pack['xy']
        sample_velocity = sample_pack['velocity']
        sample_rcs = sample_pack['rcs']
        sample_count = int(sample_pack['count'])

        intensity_display = np.power(np.clip(lam, 0.0, None), self.intensity_gamma)

        # 1. intensity_with_gt
        fig, ax = self._new_axes_figure()
        mesh = ax.pcolormesh(geometry['y_edges'], geometry['x_edges'], intensity_display, shading='auto', cmap=self.heatmap_cmap)
        self._draw_range_rings(ax)
        if gt_count > 0:
            ax.scatter(gt_xy[:, 1], gt_xy[:, 0], s=self.point_size, c=self.point_color, alpha=self.point_alpha,
                       edgecolors=self.point_edge_color, linewidths=self.point_edge_width, zorder=3)
        self._setup_bev_axes(ax, geometry['r_min'], geometry['r_max'])
        ax.set_title('Predicted PPP intensity with GT radar points', fontsize=11, pad=4)
        if self.show_textbox:
            self._textbox(ax, f'E[N]={expected_count:.1f}\nGT N={gt_count}\nGT={gt_source}')
        if self.show_colorbar:
            self._add_inset_colorbar(fig, ax, mesh, rf'$\lambda^{{{self.intensity_gamma:g}}}$')
        self._save_figure(fig, self._output_path(runner, 'intensity_with_gt', phase, video, frame_str, global_idx_str))

        # 2. velocity_with_gt
        fig, ax = self._new_axes_figure()
        mesh = ax.pcolormesh(geometry['y_edges'], geometry['x_edges'], velocity, shading='auto', cmap=self.velocity_cmap,
                             vmin=self.velocity_vmin, vmax=self.velocity_vmax)
        self._draw_range_rings(ax)
        finite = np.isfinite(gt_velocity)
        if gt_count > 0 and np.any(finite):
            ax.scatter(gt_xy[finite, 1], gt_xy[finite, 0], s=self.point_size, c=gt_velocity[finite],
                       cmap=self.velocity_cmap, vmin=self.velocity_vmin, vmax=self.velocity_vmax,
                       alpha=self.point_alpha, edgecolors=self.point_edge_color, linewidths=self.point_edge_width, zorder=3)
        self._setup_bev_axes(ax, geometry['r_min'], geometry['r_max'])
        ax.set_title('Predicted velocity with GT velocity dots', fontsize=11, pad=4)
        if self.show_textbox:
            self._textbox(ax, f'mode={self.velocity_mode}\nGT velocity dots: N={int(np.isfinite(gt_velocity).sum())}')
        if self.show_colorbar:
            vel_label = 'velocity [m/s]' if self.velocity_mode == 'signed' else '|velocity| [m/s]'
            self._add_inset_colorbar(fig, ax, mesh, vel_label)
        self._save_figure(fig, self._output_path(runner, 'velocity_with_gt', phase, video, frame_str, global_idx_str))

        # 3. rcs_with_gt
        fig, ax = self._new_axes_figure()
        mesh = ax.pcolormesh(geometry['y_edges'], geometry['x_edges'], rcs, shading='auto', cmap=self.rcs_cmap,
                             vmin=self.rcs_vmin, vmax=self.rcs_vmax)
        self._draw_range_rings(ax)
        finite = np.isfinite(gt_rcs)
        if gt_count > 0 and np.any(finite):
            ax.scatter(gt_xy[finite, 1], gt_xy[finite, 0], s=self.point_size, c=gt_rcs[finite],
                       cmap=self.rcs_cmap, vmin=self.rcs_vmin, vmax=self.rcs_vmax,
                       alpha=self.point_alpha, edgecolors=self.point_edge_color, linewidths=self.point_edge_width, zorder=3)
        self._setup_bev_axes(ax, geometry['r_min'], geometry['r_max'])
        ax.set_title('Predicted RCS with GT RCS dots', fontsize=11, pad=4)
        if self.show_textbox:
            self._textbox(ax, f'link={self.rcs_link}\nGT RCS dots: N={int(np.isfinite(gt_rcs).sum())}')
        if self.show_colorbar:
            self._add_inset_colorbar(fig, ax, mesh, 'RCS')
        self._save_figure(fig, self._output_path(runner, 'rcs_with_gt', phase, video, frame_str, global_idx_str))

        # 4. sampled_plain
        fig, ax = self._new_axes_figure()
        self._draw_range_rings(ax)
        if sample_count > 0:
            ax.scatter(sample_xy[:, 1], sample_xy[:, 0], s=self.sample_point_size, c=self.sample_point_color,
                       alpha=self.sample_point_alpha, edgecolors=self.point_edge_color, linewidths=self.point_edge_width, zorder=3)
        self._setup_bev_axes(ax, geometry['r_min'], geometry['r_max'])
        ax.set_title('Sampled predicted radar points', fontsize=11, pad=4)
        if self.show_textbox:
            self._textbox(ax, f'sampled N={sample_count}\nE[N]={expected_count:.1f}')
        self._save_figure(fig, self._output_path(runner, 'sampled_plain', phase, video, frame_str, global_idx_str))

        # 5. sampled_velocity
        fig, ax = self._new_axes_figure()
        self._draw_range_rings(ax)
        sc = None
        finite = np.isfinite(sample_velocity)
        if sample_count > 0 and np.any(finite):
            sc = ax.scatter(sample_xy[finite, 1], sample_xy[finite, 0], s=self.sample_point_size, c=sample_velocity[finite],
                            cmap=self.velocity_cmap, vmin=self.velocity_vmin, vmax=self.velocity_vmax,
                            alpha=self.sample_point_alpha, edgecolors=self.point_edge_color, linewidths=self.point_edge_width, zorder=3)
        self._setup_bev_axes(ax, geometry['r_min'], geometry['r_max'])
        ax.set_title('Sampled predicted points colored by predicted velocity', fontsize=11, pad=4)
        if self.show_textbox:
            self._textbox(ax, f'sampled N={sample_count}\nmode={self.velocity_mode}')
        if self.show_colorbar and sc is not None:
            vel_label = 'velocity [m/s]' if self.velocity_mode == 'signed' else '|velocity| [m/s]'
            self._add_inset_colorbar(fig, ax, sc, vel_label)
        self._save_figure(fig, self._output_path(runner, 'sampled_velocity', phase, video, frame_str, global_idx_str))

        # 6. sampled_rcs
        fig, ax = self._new_axes_figure()
        self._draw_range_rings(ax)
        sc = None
        finite = np.isfinite(sample_rcs)
        if sample_count > 0 and np.any(finite):
            sc = ax.scatter(sample_xy[finite, 1], sample_xy[finite, 0], s=self.sample_point_size, c=sample_rcs[finite],
                            cmap=self.rcs_cmap, vmin=self.rcs_vmin, vmax=self.rcs_vmax,
                            alpha=self.sample_point_alpha, edgecolors=self.point_edge_color, linewidths=self.point_edge_width, zorder=3)
        self._setup_bev_axes(ax, geometry['r_min'], geometry['r_max'])
        ax.set_title('Sampled predicted points colored by predicted RCS', fontsize=11, pad=4)
        if self.show_textbox:
            self._textbox(ax, f'sampled N={sample_count}\nlink={self.rcs_link}')
        if self.show_colorbar and sc is not None:
            self._add_inset_colorbar(fig, ax, sc, 'RCS')
        self._save_figure(fig, self._output_path(runner, 'sampled_rcs', phase, video, frame_str, global_idx_str))