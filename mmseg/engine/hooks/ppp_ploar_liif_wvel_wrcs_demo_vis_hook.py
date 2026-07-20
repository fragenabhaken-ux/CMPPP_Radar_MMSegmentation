"""
Polar LIIF PPP demo-video visualization hook with velocity and RCS.

Writes one PNG per selected frame:

    <runner.work_dir>/<out_dir>/<phase>/<video_name>/
        frame_<frame>_idx_<idx>.png

Layout, 4 x 1:
    1. Camera image, preserving its natural pixel aspect ratio.
    2. Predicted PPP intensity in BEV, with GT radar points overlaid.
    3. Predicted velocity in BEV, with GT points colored by GT velocity.
    4. Predicted RCS in BEV, with GT points colored by GT RCS.

The BEV panels intentionally use aspect="auto" so that they have the same
full width as the camera image. This is preferable for compact video frames,
but it stretches the metric x/y geometry vertically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


@HOOKS.register_module()
class PPPPolarLIIFWVelWRcsDemoVisHook(Hook):
    """Write compact PPP, velocity, and RCS BEV demo frames as PNG files."""

    _VELOCITY_KEYS = (
        "grid_raw_velocity",
        "grid_velocity_raw",
        "grid_velocity",
        "grid_vel",
        "velocity",
    )

    _RCS_RAW_KEYS = (
        "grid_raw_rcs",
        "grid_rcs_raw",
    )

    _RCS_DECODED_KEYS = (
        "grid_rcs",
        "rcs",
    )

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames_polar_wvel_rcs",
        img_mean=None,
        img_std=None,
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        bev_y_min: float = -85.0,
        bev_y_max: float = 85.0,
        intensity_gamma: float = 0.3,
        point_size: float = 20.0,
        point_color: str = "cyan",
        point_edge_color: str = "black",
        point_edge_width: float = 0.45,
        velocity_mode: str = "signed",
        velocity_grid_is_raw: bool = True,
        velocity_vmin: float = -15.0,
        velocity_vmax: float = 15.0,
        velocity_cmap: str = "coolwarm",
        rcs_grid_is_raw: bool = True,
        rcs_link: str = "cube",
        rcs_vmin: float = -50.0,
        rcs_vmax: float = 20.0,
        rcs_cmap: str = "viridis",
        dpi: int = 100,
        figsize=(18, 22),
        image_ext: str = "png",
        write_all_ranks: bool = False,
    ) -> None:
        self.train_interval = int(train_interval)
        self.max_samples = {
            "train": int(max_samples_train),
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.out_dir = str(out_dir)
        self.image_ext = str(image_ext).lstrip(".")
        self.write_all_ranks = bool(write_all_ranks)

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)

        self.intensity_gamma = float(intensity_gamma)

        self.point_size = float(point_size)
        self.point_color = str(point_color)
        self.point_edge_color = str(point_edge_color)
        self.point_edge_width = float(point_edge_width)

        self.velocity_mode = str(velocity_mode)
        self.velocity_grid_is_raw = bool(velocity_grid_is_raw)
        self.velocity_vmin = float(velocity_vmin)
        self.velocity_vmax = float(velocity_vmax)
        self.velocity_cmap = str(velocity_cmap)

        self.rcs_grid_is_raw = bool(rcs_grid_is_raw)
        self.rcs_link = str(rcs_link).lower()
        self.rcs_vmin = float(rcs_vmin)
        self.rcs_vmax = float(rcs_vmax)
        self.rcs_cmap = str(rcs_cmap)

        self.dpi = int(dpi)
        self.figsize = tuple(figsize)

        self.img_mean = (
            None
            if img_mean is None
            else np.asarray(img_mean, dtype=np.float32)[:, None, None]
        )
        self.img_std = (
            None
            if img_std is None
            else np.asarray(img_std, dtype=np.float32)[:, None, None]
        )

        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                "velocity_mode must be 'signed' or 'absolute', "
                f"got {self.velocity_mode!r}."
            )

        if self.rcs_link not in {"identity", "cube", "linear_cube"}:
            raise ValueError(
                "rcs_link must be 'identity', 'cube', or 'linear_cube', "
                f"got {self.rcs_link!r}."
            )

        if self.rcs_vmin >= self.rcs_vmax:
            raise ValueError(
                "rcs_vmin must be smaller than rcs_vmax, got "
                f"{self.rcs_vmin} >= {self.rcs_vmax}."
            )

        self._written_in_run = {
            "val": 0,
            "test": 0,
        }

    def _rank_allowed(self) -> bool:
        return self.write_all_ranks or is_main_process()

    def before_val_epoch(self, runner) -> None:
        self._written_in_run["val"] = 0

    def before_test_epoch(self, runner) -> None:
        self._written_in_run["test"] = 0

    def after_train_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        if not self._rank_allowed():
            return
        if not isinstance(data_batch, dict):
            return
        if self.max_samples["train"] <= 0:
            return
        if self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="train",
            step=runner.iter,
        )

    def after_val_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        if not self._rank_allowed():
            return
        if not isinstance(data_batch, dict):
            return
        if self.max_samples["val"] <= 0:
            return
        if self._written_in_run["val"] >= self.max_samples["val"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="val",
            step=runner.iter,
        )

    def after_test_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        if not self._rank_allowed():
            return
        if not isinstance(data_batch, dict):
            return
        if self.max_samples["test"] <= 0:
            return
        if self._written_in_run["test"] >= self.max_samples["test"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="test",
            step=runner.iter,
        )

    def _run(
        self,
        runner,
        data_batch,
        phase: str,
        step: int,
    ) -> None:
        model = (
            runner.model.module
            if is_model_wrapper(runner.model)
            else runner.model
        )

        was_training = model.training
        model.eval()

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            feats = model.extract_feat(inputs)
            preds = model.decode_head(feats)

        if not isinstance(preds, dict):
            raise TypeError(
                "PPPPolarLIIFWVelDemoVisHook expects decode_head output "
                "to be a dictionary."
            )

        grid_log_lambda = preds["grid_log_lambda"]
        grid_velocity = self._extract_velocity_grid(preds)
        grid_rcs = self._extract_rcs_grid(preds)

        if phase == "train":
            n_vis = min(
                self.max_samples["train"],
                grid_log_lambda.shape[0],
            )
            start_idx = 0
        else:
            remaining = self.max_samples[phase] - self._written_in_run[phase]

            if remaining <= 0:
                if was_training:
                    model.train()
                return

            n_vis = min(remaining, grid_log_lambda.shape[0])
            start_idx = self._written_in_run[phase]

        for i in range(n_vis):
            self._visualize_sample(
                runner=runner,
                step=step,
                phase=phase,
                img_tensor=inputs[i],
                grid_log_lambda=self._select_grid(
                    grid_log_lambda,
                    batch_index=i,
                ),
                grid_velocity=self._select_grid(
                    grid_velocity,
                    batch_index=i,
                ),
                grid_rcs=self._select_grid(
                    grid_rcs,
                    batch_index=i,
                ),
                sample=data_samples[i],
                sample_idx=start_idx + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

    def _extract_velocity_grid(self, preds: dict) -> torch.Tensor:
        for key in self._VELOCITY_KEYS:
            value = preds.get(key)

            if torch.is_tensor(value):
                return value

        available = ", ".join(sorted(preds.keys()))

        raise KeyError(
            "Could not find the predicted velocity grid. "
            f"Tried {self._VELOCITY_KEYS}. Available prediction keys: "
            f"{available}"
        )

    def _extract_rcs_grid(self, preds: dict) -> torch.Tensor:
        keys = (
            self._RCS_RAW_KEYS
            if self.rcs_grid_is_raw
            else self._RCS_DECODED_KEYS
        )

        for key in keys:
            value = preds.get(key)

            if torch.is_tensor(value):
                return value

        available = ", ".join(sorted(preds.keys()))

        expected = "raw" if self.rcs_grid_is_raw else "decoded"
        raise KeyError(
            f"Could not find the predicted {expected} RCS grid. "
            f"Tried {keys}. Available prediction keys: {available}"
        )

    @staticmethod
    def _select_grid(
        tensor: torch.Tensor,
        batch_index: int,
    ) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor[batch_index, 0]

        if tensor.ndim == 3:
            return tensor[batch_index]

        raise ValueError(
            "Expected an intensity/mark grid with shape "
            "(B, 1, H, W) or (B, H, W), got "
            f"{tuple(tensor.shape)}."
        )

    def _decode_velocity(self, raw_velocity: torch.Tensor) -> np.ndarray:
        raw_velocity = raw_velocity.detach().float().cpu()

        if not self.velocity_grid_is_raw:
            velocity = raw_velocity

        elif self.velocity_mode == "signed":
            velocity = raw_velocity.pow(3)

        else:
            velocity = raw_velocity.abs()

        return velocity.numpy()

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        raw_rcs = raw_rcs.detach().float().cpu()

        if not self.rcs_grid_is_raw:
            rcs = raw_rcs
        elif self.rcs_link == "identity":
            rcs = raw_rcs
        elif self.rcs_link == "cube":
            rcs = raw_rcs.pow(3)
        elif self.rcs_link == "linear_cube":
            rcs = raw_rcs + raw_rcs.pow(3)
        else:
            raise RuntimeError(f"Unsupported rcs_link={self.rcs_link!r}")

        return rcs.numpy()

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)

        if self.img_mean is not None and self.img_std is not None:
            x = x * self.img_std + self.img_mean
            x = (x + 1.0) * 127.5

        else:
            if x.min() >= -1.01 and x.max() <= 1.01:
                x = (x + 1.0) * 127.5
            elif x.max() <= 1.01:
                x = x * 255.0

        x = np.clip(x, 0, 255).astype(np.uint8)
        return np.transpose(x, (1, 2, 0))

    @staticmethod
    def _tagify(s: str) -> str:
        return "".join(
            c if c.isalnum() or c in "._-/" else "_"
            for c in s
        )

    @staticmethod
    def _textbox(ax, text: str) -> None:
        ax.text(
            0.015,
            0.975,
            text,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="black",
                alpha=0.60,
                pad=3,
            ),
        )

    @staticmethod
    def _polar_edges(
        r_min: float,
        r_max: float,
        theta_min: float,
        theta_max: float,
        h_r: int,
        w_theta: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r_edges = np.linspace(r_min, r_max, h_r + 1)
        theta_edges = np.linspace(theta_min, theta_max, w_theta + 1)
        return r_edges, theta_edges

    def _polar_to_bev_mesh(
        self,
        h_r: int,
        w_theta: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r_edges, theta_edges = self._polar_edges(
            r_min=self.r_min,
            r_max=self.r_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            h_r=h_r,
            w_theta=w_theta,
        )

        rr, tt = np.meshgrid(
            r_edges,
            theta_edges,
            indexing="ij",
        )

        x_forward = rr * np.cos(tt)
        y_lateral = rr * np.sin(tt)

        return x_forward, y_lateral

    def _setup_bev_axes(self, ax) -> None:
        ax.set_xlim(self.bev_y_max, self.bev_y_min)
        ax.set_ylim(self.r_min, self.r_max)

        # Deliberately fills the same full width as the camera frame.
        ax.set_aspect("auto")

        ax.set_xlabel("y lateral [m]", labelpad=1)
        ax.set_ylabel("x forward [m]", labelpad=1)

        ax.grid(
            True,
            alpha=0.22,
            linewidth=0.6,
        )

        ax.tick_params(
            labelsize=9,
            pad=1,
        )

    @staticmethod
    def _add_inset_colorbar(
        fig: Figure,
        ax,
        mappable,
        label: str,
    ) -> None:
        cax = ax.inset_axes([0.73, 0.045, 0.24, 0.035])

        cb = fig.colorbar(
            mappable,
            cax=cax,
            orientation="horizontal",
        )

        cb.ax.tick_params(
            labelsize=7,
            pad=1,
            length=2,
        )

        cb.set_label(
            label,
            fontsize=7,
            labelpad=1,
        )

        cax.set_facecolor((1.0, 1.0, 1.0, 0.85))

    def _output_path(
        self,
        runner,
        phase: str,
        video: str,
        frame_str: str,
        gidx_str: str,
    ) -> Path:
        root = Path(self.out_dir)

        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        out_dir = root / phase / video
        out_dir.mkdir(parents=True, exist_ok=True)

        return out_dir / (
            f"frame_{frame_str}_idx_{gidx_str}.{self.image_ext}"
        )

    def _visualize_sample(
        self,
        runner,
        step: int,
        phase: str,
        img_tensor: torch.Tensor,
        grid_log_lambda: torch.Tensor,
        grid_velocity: torch.Tensor,
        grid_rcs: torch.Tensor,
        sample,
        sample_idx: int,
    ) -> None:
        del step, sample_idx

        img = self._denorm_img(
            img_tensor.detach().cpu().numpy()
        )

        log_lambda = grid_log_lambda.detach().float().cpu()
        lam = torch.exp(log_lambda).numpy()

        velocity = self._decode_velocity(grid_velocity)
        rcs = self._decode_rcs(grid_rcs)

        h_r, w_theta = lam.shape
        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        cell_mass = lam * dr * dtheta
        expected_count = float(cell_mass.sum())

        polar = sample.gt_radar_points.polar.detach().cpu().numpy()
        meta_points = sample.gt_radar_points.meta.detach().cpu().numpy()
        xy = sample.gt_radar_points.xy.detach().cpu().numpy()

        if polar.ndim != 2 or polar.shape[1] < 3:
            raise ValueError(
                "Expected gt_radar_points.polar with at least three columns: "
                "(range, azimuth, velocity)."
            )

        if (
            meta_points.ndim != 2
            or meta_points.shape[0] != polar.shape[0]
            or meta_points.shape[1] <= 2
        ):
            raise ValueError(
                "Expected GT RCS in gt_radar_points.meta[:, 2] with the "
                "same point count as gt_radar_points.polar, got "
                f"polar={polar.shape}, meta={meta_points.shape}."
            )

        gt_velocity = polar[:, 2].astype(np.float64)
        gt_rcs = meta_points[:, 2].astype(np.float64)

        if self.velocity_mode == "absolute":
            gt_velocity = np.abs(gt_velocity)

        gt_count = int(xy.shape[0])

        meta = sample.metainfo

        video = self._tagify(
            str(meta.get("video_name", "unknown_video"))
        )
        frame = meta.get("video_frame_idx")
        gidx = meta.get("global_idx", meta.get("frame_idx"))

        frame_str = (
            f"{int(frame):06d}"
            if frame is not None
            else "unknown_frame"
        )
        gidx_str = (
            f"{int(gidx):08d}"
            if gidx is not None
            else "unknown_idx"
        )

        x_mesh, y_mesh = self._polar_to_bev_mesh(
            h_r=h_r,
            w_theta=w_theta,
        )

        fig = Figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        FigureCanvasAgg(fig)

        axs = fig.subplots(
            4,
            1,
            gridspec_kw=dict(
                height_ratios=(1.0, 1.0, 1.0, 1.0),
            ),
        )

        fig.subplots_adjust(
            left=0.052,
            right=0.992,
            top=0.958,
            bottom=0.035,
            hspace=0.12,
        )

        fig.suptitle(
            f"{phase} | {video} | "
            f"frame {frame_str} | idx {gidx_str}",
            fontsize=13,
            y=0.988,
        )

        # -------------------------------------------------------------
        # 1. Camera image
        # -------------------------------------------------------------
        ax = axs[0]

        ax.imshow(
            img,
            aspect="equal",
        )
        ax.set_title(
            "Camera image",
            fontsize=11,
            pad=3,
        )
        ax.axis("off")

        # -------------------------------------------------------------
        # 2. PPP intensity in BEV with GT radar points
        # -------------------------------------------------------------
        ax = axs[1]

        intensity_display = np.power(
            lam,
            self.intensity_gamma,
        )

        intensity_pcm = ax.pcolormesh(
            y_mesh,
            x_mesh,
            intensity_display,
            shading="auto",
            cmap="hot",
        )

        if gt_count > 0:
            ax.scatter(
                xy[:, 1],
                xy[:, 0],
                s=self.point_size,
                c=self.point_color,
                alpha=0.97,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax)

        ax.set_title(
            "Predicted PPP intensity in BEV with GT radar points",
            fontsize=11,
            pad=3,
        )

        self._textbox(
            ax,
            (
                f"E[N]={expected_count:.1f}\n"
                f"GT N={gt_count}\n"
                f"GT dots: cyan"
            ),
        )

        self._add_inset_colorbar(
            fig=fig,
            ax=ax,
            mappable=intensity_pcm,
            label=rf"$\lambda^{{{self.intensity_gamma:g}}}$",
        )

        # -------------------------------------------------------------
        # 3. Velocity in BEV with GT velocity-colored radar points
        # -------------------------------------------------------------
        ax = axs[2]

        velocity_pcm = ax.pcolormesh(
            y_mesh,
            x_mesh,
            velocity,
            shading="auto",
            cmap=self.velocity_cmap,
            vmin=self.velocity_vmin,
            vmax=self.velocity_vmax,
        )

        finite_velocity = np.isfinite(gt_velocity)

        if gt_count > 0 and np.any(finite_velocity):
            ax.scatter(
                xy[finite_velocity, 1],
                xy[finite_velocity, 0],
                s=self.point_size,
                c=gt_velocity[finite_velocity],
                cmap=self.velocity_cmap,
                vmin=self.velocity_vmin,
                vmax=self.velocity_vmax,
                alpha=0.98,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax)

        if self.velocity_mode == "signed":
            velocity_title = (
                "Predicted signed velocity in BEV "
                "with GT velocity-colored radar points"
            )
            velocity_label = "velocity [m/s]"
        else:
            velocity_title = (
                "Predicted absolute velocity in BEV "
                "with GT velocity-colored radar points"
            )
            velocity_label = "|velocity| [m/s]"

        ax.set_title(
            velocity_title,
            fontsize=11,
            pad=3,
        )

        self._textbox(
            ax,
            (
                f"mode={self.velocity_mode}\n"
                f"GT velocity dots: N={int(finite_velocity.sum())}"
            ),
        )

        self._add_inset_colorbar(
            fig=fig,
            ax=ax,
            mappable=velocity_pcm,
            label=velocity_label,
        )

        # -------------------------------------------------------------
        # 4. RCS in BEV with GT RCS-colored radar points
        # -------------------------------------------------------------
        ax = axs[3]

        rcs_pcm = ax.pcolormesh(
            y_mesh,
            x_mesh,
            rcs,
            shading="auto",
            cmap=self.rcs_cmap,
            vmin=self.rcs_vmin,
            vmax=self.rcs_vmax,
        )

        finite_rcs = np.isfinite(gt_rcs)

        if gt_count > 0 and np.any(finite_rcs):
            ax.scatter(
                xy[finite_rcs, 1],
                xy[finite_rcs, 0],
                s=self.point_size,
                c=gt_rcs[finite_rcs],
                cmap=self.rcs_cmap,
                vmin=self.rcs_vmin,
                vmax=self.rcs_vmax,
                alpha=0.98,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax)

        ax.set_title(
            "Predicted RCS in BEV with GT RCS-colored radar points",
            fontsize=11,
            pad=3,
        )

        self._textbox(
            ax,
            (
                f"link={self.rcs_link if self.rcs_grid_is_raw else 'decoded'}\n"
                f"GT RCS dots: N={int(finite_rcs.sum())}"
            ),
        )

        self._add_inset_colorbar(
            fig=fig,
            ax=ax,
            mappable=rcs_pcm,
            label="RCS",
        )

        out_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            gidx_str=gidx_str,
        )

        fig.savefig(
            out_path,
            dpi=self.dpi,
        )
        fig.clear()