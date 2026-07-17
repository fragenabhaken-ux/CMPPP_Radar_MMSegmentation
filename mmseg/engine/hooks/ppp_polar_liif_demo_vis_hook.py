"""
PPP polar LIIF demo visualization hook.

Writes PNG files directly to:

    <runner.work_dir>/<out_dir>/<phase>/<video_name>/
        frame_<frame>_idx_<idx>.png

Plot layout (3x2):
    1. top-left:     camera image
    2. top-right:    predicted intensity warped to BEV
    3. middle-left:  GT radar points in BEV
    4. middle-right: GT radar points in polar coordinates
    5. bottom-left:  predicted polar intensity lambda(r, theta)
    6. bottom-right: predicted polar cell mass lambda * dr * dtheta

For distributed test runs, set write_all_ranks=True.
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
class PPPPolarLIIFDemoVisHook(Hook):
    """Write polar LIIF PPP demo visualizations as PNG files."""

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames_polar",
        img_mean=None,
        img_std=None,
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        gamma: float = 0.3,
        point_size: float = 12.0,
        dpi: int = 100,
        figsize=(18, 18),
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

        self.gamma = float(gamma)
        self.point_size = float(point_size)
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

        grid_log_lambda = preds["grid_log_lambda"]

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
                grid_log_lambda=grid_log_lambda[i, 0],
                sample=data_samples[i],
                sample_idx=start_idx + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

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
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="black",
                alpha=0.55,
                pad=3,
            ),
        )

    def _polar_edges(
        self,
        h_r: int,
        w_theta: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r_edges = np.linspace(
            self.r_min,
            self.r_max,
            h_r + 1,
        )
        theta_edges = np.linspace(
            self.theta_min,
            self.theta_max,
            w_theta + 1,
        )
        return r_edges, theta_edges

    def _polar_to_bev_mesh(
        self,
        h_r: int,
        w_theta: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r_edges, theta_edges = self._polar_edges(h_r, w_theta)

        rr, tt = np.meshgrid(
            r_edges,
            theta_edges,
            indexing="ij",
        )

        x_forward = rr * np.cos(tt)
        y_lateral = rr * np.sin(tt)

        return x_forward, y_lateral

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

        log_lam = grid_log_lambda.detach().float().cpu()
        lam = torch.exp(log_lam).numpy()

        h_r, w_theta = lam.shape

        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        cell_mass = lam * dr * dtheta
        exp_count = float(cell_mass.sum())

        polar = sample.gt_radar_points.polar.detach().cpu().numpy()
        xy = sample.gt_radar_points.xy.detach().cpu().numpy()

        gt_count = int(polar.shape[0])

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

        axs = fig.subplots(3, 2)

        fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.93,
            bottom=0.06,
            hspace=0.28,
            wspace=0.16,
        )

        fig.suptitle(
            f"{phase} | {video} | "
            f"frame {frame_str} | idx {gidx_str}",
            fontsize=14,
        )

        # ---------------------------------------------------------------------
        # 1. Camera image: top-left
        # ---------------------------------------------------------------------
        ax = axs[0, 0]

        ax.imshow(img, aspect="auto")
        ax.axis("off")
        ax.set_title("Camera image")

        # ---------------------------------------------------------------------
        # 2. Predicted intensity in BEV: top-right
        # ---------------------------------------------------------------------
        ax = axs[0, 1]

        ax.set_title("Predicted polar intensity warped to BEV")

        pcm = ax.pcolormesh(
            y_mesh,
            x_mesh,
            np.power(lam, self.gamma),
            shading="auto",
            cmap="hot",
        )

        ax.set_xlim(self.r_max, -self.r_max)
        ax.set_ylim(self.r_min, self.r_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("y lateral [m]")
        ax.set_ylabel("x forward [m]")

        fig.colorbar(
            pcm,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        self._textbox(
            ax,
            f"λ^γ, γ={self.gamma}",
        )

        # ---------------------------------------------------------------------
        # 3. GT BEV points: middle-left
        # ---------------------------------------------------------------------
        ax = axs[1, 0]

        ax.set_title("GT BEV points")

        if xy.shape[0] > 0:
            ax.scatter(
                xy[:, 1],
                xy[:, 0],
                s=self.point_size,
                c="lime",
                alpha=0.85,
                edgecolors="none",
            )

        ax.set_xlim(self.r_max, -self.r_max)
        ax.set_ylim(self.r_min, self.r_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("y lateral [m]")
        ax.set_ylabel("x forward [m]")
        ax.grid(True, alpha=0.25)

        self._textbox(
            ax,
            f"GT N={gt_count}",
        )

        # ---------------------------------------------------------------------
        # 4. GT polar points: middle-right
        # r=0 is at the bottom; r=r_max is at the top.
        # ---------------------------------------------------------------------
        ax = axs[1, 1]

        ax.set_title("GT polar points")

        if gt_count > 0:
            ax.scatter(
                polar[:, 1],
                polar[:, 0],
                s=self.point_size,
                c="lime",
                alpha=0.85,
                edgecolors="none",
            )

        ax.set_xlim(self.theta_max, self.theta_min)
        ax.set_ylim(self.r_min, self.r_max)

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$r$")
        ax.grid(True, alpha=0.25)

        self._textbox(
            ax,
            f"GT N={gt_count}",
        )

        # ---------------------------------------------------------------------
        # 5. Predicted polar intensity: bottom-left
        # r=0 is at the bottom; r=r_max is at the top.
        # ---------------------------------------------------------------------
        ax = axs[2, 0]

        im = ax.imshow(
            lam,
            extent=[
                self.theta_min,
                self.theta_max,
                self.theta_max,
                self.r_min,
                self.r_max,
            ],
            origin="lower",
            cmap="hot",
            aspect="auto",
        )

        ax.set_xlim(self.theta_max, self.theta_min)
        
        ax.set_title(r"Predicted $\lambda(r,\theta)$")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$r$")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        self._textbox(
            ax,
            (
                f"E[N]={exp_count:.2f}\n"
                f"GT={gt_count}\n"
                f"Δ={exp_count - gt_count:+.2f}"
            ),
        )

        # ---------------------------------------------------------------------
        # 6. Predicted polar cell mass: bottom-right
        # r=0 is at the bottom; r=r_max is at the top.
        # ---------------------------------------------------------------------
        ax = axs[2, 1]

        im = ax.imshow(
            np.power(cell_mass, self.gamma),
            extent=[
                self.theta_min,
                self.theta_max,
                self.r_min,
                self.r_max,
            ],
            origin="lower",
            cmap="hot",
            aspect="auto",
        )

        ax.set_title(
            r"Cell mass$^\gamma$: "
            r"$(\lambda\,\Delta r\,\Delta\theta)^\gamma$"
        )
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$r$")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        self._textbox(
            ax,
            f"γ={self.gamma}",
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
            bbox_inches="tight",
        )
        fig.clear()