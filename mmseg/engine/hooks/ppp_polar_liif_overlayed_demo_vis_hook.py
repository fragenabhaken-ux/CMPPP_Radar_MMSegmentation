"""
Polar LIIF PPP demo-video visualization hook with one large BEV overlay.

Writes one PNG per selected frame:

    <runner.work_dir>/<out_dir>/<phase>/<video_name>/
        frame_<frame>_idx_<idx>.png

Layout:
    1. Top:    camera image at its natural aspect ratio.
    2. Bottom: predicted PPP intensity in BEV with GT radar points overlaid.

The BEV panel spans the vertical space previously occupied by the intensity
and velocity panels of PPPPolarLIIFWVelDemoVisHook.
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
class PPPPolarLIIFOverlayedDemoVisHook(Hook):
    """Write camera plus large BEV PPP-overlay demo frames."""

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames_polar_overlayed",
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
        dpi: int = 100,
        figsize=(18, 17),
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
                "PPPPolarLIIFOverlayedDemoVisHook expects decode_head "
                "output to be a dictionary."
            )

        if "grid_log_lambda" not in preds:
            raise KeyError(
                "Could not find preds['grid_log_lambda']."
            )

        grid_log_lambda = preds["grid_log_lambda"]

        if phase == "train":
            n_vis = min(
                self.max_samples["train"],
                grid_log_lambda.shape[0],
            )
            start_idx = 0

        else:
            remaining = (
                self.max_samples[phase]
                - self._written_in_run[phase]
            )

            if remaining <= 0:
                if was_training:
                    model.train()
                return

            n_vis = min(
                remaining,
                grid_log_lambda.shape[0],
            )
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
                sample=data_samples[i],
                sample_idx=start_idx + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

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
            "Expected grid shape (B, 1, H, W) or (B, H, W), "
            f"got {tuple(tensor.shape)}."
        )

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
        # Reversed horizontally:
        # positive y lateral on the left, negative on the right.
        ax.set_xlim(self.bev_y_max, self.bev_y_min)
        ax.set_ylim(self.r_min, self.r_max)

        # The lower panel occupies two original plot rows, so equal metric
        # scaling now fills that space without the former wide-strip look.
        ax.set_aspect("equal", adjustable="box")

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
        sample,
        sample_idx: int,
    ) -> None:
        del step, sample_idx

        img = self._denorm_img(
            img_tensor.detach().cpu().numpy()
        )

        log_lambda = grid_log_lambda.detach().float().cpu()
        lam = torch.exp(log_lambda).numpy()

        h_r, w_theta = lam.shape

        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        cell_mass = lam * dr * dtheta
        expected_count = float(cell_mass.sum())

        xy = sample.gt_radar_points.xy.detach().cpu().numpy()
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

        grid = fig.add_gridspec(
            3,
            1,
            height_ratios=(1.0, 1.0, 1.0),
        )

        ax_camera = fig.add_subplot(grid[0, 0])
        ax_bev = fig.add_subplot(grid[1:, 0])

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
        # Camera image
        # -------------------------------------------------------------
        ax_camera.imshow(
            img,
            aspect="equal",
        )
        ax_camera.set_title(
            "Camera image",
            fontsize=11,
            pad=3,
        )
        ax_camera.axis("off")

        # -------------------------------------------------------------
        # Large PPP intensity overlay in BEV
        # -------------------------------------------------------------
        intensity_display = np.power(
            lam,
            self.intensity_gamma,
        )

        intensity_pcm = ax_bev.pcolormesh(
            y_mesh,
            x_mesh,
            intensity_display,
            shading="auto",
            cmap="hot",
        )

        if gt_count > 0:
            ax_bev.scatter(
                xy[:, 1],
                xy[:, 0],
                s=self.point_size,
                c=self.point_color,
                alpha=0.97,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(ax_bev)

        ax_bev.set_title(
            "Predicted PPP intensity in BEV with GT radar points",
            fontsize=11,
            pad=3,
        )

        self._textbox(
            ax_bev,
            (
                f"E[N]={expected_count:.1f}\n"
                f"GT N={gt_count}\n"
                f"GT dots: cyan"
            ),
        )

        self._add_inset_colorbar(
            fig=fig,
            ax=ax_bev,
            mappable=intensity_pcm,
            label=rf"$\lambda^{{{self.intensity_gamma:g}}}$",
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