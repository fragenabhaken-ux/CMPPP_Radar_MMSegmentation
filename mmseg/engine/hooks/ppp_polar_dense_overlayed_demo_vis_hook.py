"""Single-panel demo visualization for the dense polar-BEV PPP model.

Writes one PNG per selected frame to

    <runner.work_dir>/<out_dir>/<phase>/<video_name>/
        frame_<frame>_idx_<idx>.png

Each PNG contains only the predicted PPP intensity warped from the dense
polar grid into Cartesian BEV, with the continuous ground-truth radar points
overlaid.

Prediction convention:
    channel 0: log intensity with respect to dr dtheta
    remaining channels are ignored

Ground-truth convention:
    The preferred GT source is ``sample.metainfo['polar_gt_points']`` with
    columns ``(r, theta, velocity, rcs)``. These are the exact continuous
    in-domain points retained by the updated BuildPPPPolarGrid transform.
    If that metadata is unavailable, occupied dense-target cell centres are
    used as a fallback.
"""

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

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


@HOOKS.register_module()
class PPPPolarDenseOverlayedDemoVisHook(Hook):
    """Write one dense polar-BEV intensity overlay per selected sample."""

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames_polar_dense_overlayed",
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        bev_y_min: float = -52.0,
        bev_y_max: float = 52.0,
        intensity_gamma: float = 0.3,
        heatmap_cmap: str = "hot",
        point_size: float = 20.0,
        point_color: str = "cyan",
        point_alpha: float = 0.97,
        point_edge_color: str = "black",
        point_edge_width: float = 0.45,
        show_colorbar: bool = True,
        show_textbox: bool = True,
        dpi: int = 150,
        figsize: tuple[float, float] = (8.0, 8.0),
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
        self.heatmap_cmap = str(heatmap_cmap)

        self.point_size = float(point_size)
        self.point_color = str(point_color)
        self.point_alpha = float(point_alpha)
        self.point_edge_color = str(point_edge_color)
        self.point_edge_width = float(point_edge_width)

        self.show_colorbar = bool(show_colorbar)
        self.show_textbox = bool(show_textbox)
        self.dpi = int(dpi)
        self.figsize = tuple(float(v) for v in figsize)

        if self.r_min >= self.r_max:
            raise ValueError(
                f"r_min must be smaller than r_max, got {self.r_min} >= {self.r_max}."
            )
        if self.theta_min >= self.theta_max:
            raise ValueError(
                "theta_min must be smaller than theta_max, got "
                f"{self.theta_min} >= {self.theta_max}."
            )
        if self.bev_y_min >= self.bev_y_max:
            raise ValueError(
                "bev_y_min must be smaller than bev_y_max, got "
                f"{self.bev_y_min} >= {self.bev_y_max}."
            )
        if self.intensity_gamma <= 0.0:
            raise ValueError("intensity_gamma must be positive.")
        if self.point_size <= 0.0:
            raise ValueError("point_size must be positive.")
        if not 0.0 <= self.point_alpha <= 1.0:
            raise ValueError("point_alpha must be in [0, 1].")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if len(self.figsize) != 2 or min(self.figsize) <= 0.0:
            raise ValueError("figsize must contain two positive values.")

        self._written_in_run = {"val": 0, "test": 0}

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
        del batch_idx, outputs

        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["train"] <= 0 or self.train_interval <= 0:
            return
        if runner.iter % self.train_interval != 0:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="train",
        )

    def after_val_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        del batch_idx, outputs

        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["val"] <= 0:
            return
        if self._written_in_run["val"] >= self.max_samples["val"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="val",
        )

    def after_test_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        del batch_idx, outputs

        if not self._rank_allowed() or not isinstance(data_batch, dict):
            return
        if self.max_samples["test"] <= 0:
            return
        if self._written_in_run["test"] >= self.max_samples["test"]:
            return

        self._run(
            runner=runner,
            data_batch=data_batch,
            phase="test",
        )

    def _run(self, runner, data_batch, phase: str) -> None:
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model

        was_training = model.training
        model.eval()

        proc = model.data_preprocessor(data_batch)
        inputs = proc["inputs"]
        data_samples = proc["data_samples"]

        with torch.no_grad():
            predictions = model.decode_head(model.extract_feat(inputs))

        if not torch.is_tensor(predictions):
            raise TypeError(
                "PPPPolarDenseOverlayedDemoVisHook expects decode_head output "
                f"to be a tensor, got {type(predictions)}."
            )
        if predictions.ndim != 4 or predictions.shape[1] < 1:
            raise ValueError(
                "Expected dense predictions with shape [B, C, H, W] and C >= 1, "
                f"got {tuple(predictions.shape)}."
            )
        if predictions.shape[0] != len(data_samples):
            raise ValueError(
                "Prediction batch size and data-sample count differ: "
                f"{predictions.shape[0]} != {len(data_samples)}."
            )

        if phase == "train":
            n_vis = min(self.max_samples["train"], predictions.shape[0])
            start_idx = 0
        else:
            remaining = self.max_samples[phase] - self._written_in_run[phase]
            if remaining <= 0:
                if was_training:
                    model.train()
                return
            n_vis = min(remaining, predictions.shape[0])
            start_idx = self._written_in_run[phase]

        for batch_index in range(n_vis):
            self._visualize_sample(
                runner=runner,
                phase=phase,
                pred_grid=predictions[batch_index],
                sample=data_samples[batch_index],
                sample_idx=start_idx + batch_index,
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
        if not hasattr(sample, "gt_sem_seg"):
            raise AttributeError("Data sample has no gt_sem_seg field.")

        gt = sample.gt_sem_seg.data
        if not torch.is_tensor(gt):
            raise TypeError(
                "Expected sample.gt_sem_seg.data to be a tensor, got "
                f"{type(gt)}."
            )

        if gt.ndim == 4:
            if gt.shape[0] != 1:
                raise ValueError(
                    "Unexpected batched GT inside one data sample: "
                    f"{tuple(gt.shape)}."
                )
            gt = gt[0]

        if gt.ndim != 3 or gt.shape[0] < 1:
            raise ValueError(
                "Expected dense polar GT with shape [C, H, W] and C >= 1, got "
                f"{tuple(gt.shape)}."
            )

        return gt.detach().float().cpu()

    def _geometry(
        self,
        sample,
        h_r: int,
        w_theta: int,
    ) -> dict[str, Any]:
        meta = sample.metainfo

        r_min = float(meta.get("polar_r_min", self.r_min))
        r_max = float(meta.get("polar_r_max", self.r_max))
        theta_min = float(meta.get("polar_theta_min", self.theta_min))
        theta_max = float(meta.get("polar_theta_max", self.theta_max))

        if r_min >= r_max:
            raise ValueError(f"Invalid polar range in sample metadata: {r_min}, {r_max}.")
        if theta_min >= theta_max:
            raise ValueError(
                "Invalid polar angle range in sample metadata: "
                f"{theta_min}, {theta_max}."
            )

        meta_shape = meta.get("polar_grid_shape")
        if meta_shape is not None:
            meta_shape = tuple(int(v) for v in meta_shape)
            if meta_shape != (h_r, w_theta):
                raise ValueError(
                    "Prediction shape does not match sample polar_grid_shape: "
                    f"{(h_r, w_theta)} != {meta_shape}."
                )

        dr = (r_max - r_min) / h_r
        dtheta = (theta_max - theta_min) / w_theta

        r_edges = np.linspace(r_min, r_max, h_r + 1, dtype=np.float64)
        theta_edges = np.linspace(
            theta_min,
            theta_max,
            w_theta + 1,
            dtype=np.float64,
        )
        rr_edges, tt_edges = np.meshgrid(
            r_edges,
            theta_edges,
            indexing="ij",
        )

        return {
            "r_min": r_min,
            "r_max": r_max,
            "theta_min": theta_min,
            "theta_max": theta_max,
            "dr": dr,
            "dtheta": dtheta,
            "x_edges": rr_edges * np.cos(tt_edges),
            "y_edges": rr_edges * np.sin(tt_edges),
        }

    def _gt_xy(
        self,
        sample,
        gt: torch.Tensor,
        geometry: dict[str, Any],
    ) -> tuple[np.ndarray, str]:
        meta = sample.metainfo

        if "polar_gt_points" in meta:
            polar_points = self._to_numpy(meta["polar_gt_points"]).astype(
                np.float64,
                copy=False,
            )
            if polar_points.ndim != 2 or polar_points.shape[1] < 2:
                raise ValueError(
                    "Expected metainfo['polar_gt_points'] with shape [N, >=2], "
                    f"got {polar_points.shape}."
                )

            r = polar_points[:, 0]
            theta = polar_points[:, 1]
            finite = np.isfinite(r) & np.isfinite(theta)
            r = r[finite]
            theta = theta[finite]

            xy = np.stack(
                (r * np.cos(theta), r * np.sin(theta)),
                axis=1,
            )
            return xy, "continuous"

        occupancy = gt[0].numpy() > 0.0
        h_r, w_theta = occupancy.shape

        r_centres = geometry["r_min"] + (
            np.arange(h_r, dtype=np.float64) + 0.5
        ) * geometry["dr"]
        theta_centres = geometry["theta_min"] + (
            np.arange(w_theta, dtype=np.float64) + 0.5
        ) * geometry["dtheta"]
        rr, tt = np.meshgrid(r_centres, theta_centres, indexing="ij")

        xy = np.stack(
            (rr[occupancy] * np.cos(tt[occupancy]),
             rr[occupancy] * np.sin(tt[occupancy])),
            axis=1,
        )
        return xy, "cell centres"

    def _setup_bev_axes(
        self,
        ax,
        r_min: float,
        r_max: float,
    ) -> None:
        ax.set_xlim(self.bev_y_max, self.bev_y_min)
        ax.set_ylim(r_min, r_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y lateral [m]", labelpad=1)
        ax.set_ylabel("x forward [m]", labelpad=1)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=9, pad=1)

    @staticmethod
    def _textbox(ax, text: str) -> None:
        ax.text(
            0.015,
            0.975,
            text,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.60, pad=3),
        )

    @staticmethod
    def _add_inset_colorbar(
        fig: Figure,
        ax,
        mappable,
        label: str,
    ) -> None:
        cax = ax.inset_axes([0.69, 0.045, 0.28, 0.035])
        colorbar = fig.colorbar(
            mappable,
            cax=cax,
            orientation="horizontal",
        )
        colorbar.ax.tick_params(labelsize=7, pad=1, length=2)
        colorbar.set_label(label, fontsize=7, labelpad=1)
        cax.set_facecolor((1.0, 1.0, 1.0, 0.85))

    @staticmethod
    def _tagify(value: str) -> str:
        return "".join(
            character
            if character.isalnum() or character in "._-/"
            else "_"
            for character in value
        )

    def _output_path(
        self,
        runner,
        phase: str,
        video: str,
        frame_str: str,
        global_idx_str: str,
    ) -> Path:
        root = Path(self.out_dir)
        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        output_dir = root / phase / video
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / (
            f"frame_{frame_str}_idx_{global_idx_str}.{self.image_ext}"
        )

    def _visualize_sample(
        self,
        runner,
        phase: str,
        pred_grid: torch.Tensor,
        sample,
        sample_idx: int,
    ) -> None:
        del sample_idx

        pred_grid = pred_grid.detach().float().cpu()
        if pred_grid.ndim != 3 or pred_grid.shape[0] < 1:
            raise ValueError(
                "Expected one dense prediction with shape [C, H, W] and C >= 1, "
                f"got {tuple(pred_grid.shape)}."
            )

        gt = self._extract_dense_gt(sample)
        if tuple(gt.shape[-2:]) != tuple(pred_grid.shape[-2:]):
            raise ValueError(
                "Prediction and GT grid shapes differ: "
                f"{tuple(pred_grid.shape[-2:])} != {tuple(gt.shape[-2:])}."
            )

        log_lambda = pred_grid[0]
        lam = torch.exp(log_lambda).numpy()
        if not np.all(np.isfinite(lam)):
            raise ValueError("Predicted intensity contains NaN or Inf after exp().")

        h_r, w_theta = lam.shape
        geometry = self._geometry(sample, h_r=h_r, w_theta=w_theta)
        expected_count = float(lam.sum() * geometry["dr"] * geometry["dtheta"])

        gt_xy, gt_source = self._gt_xy(
            sample=sample,
            gt=gt,
            geometry=geometry,
        )
        gt_count = int(gt_xy.shape[0])

        meta = sample.metainfo
        video = self._tagify(str(meta.get("video_name", "unknown_video")))
        frame = meta.get("video_frame_idx")
        global_idx = meta.get("global_idx", meta.get("frame_idx"))

        frame_str = f"{int(frame):06d}" if frame is not None else "unknown_frame"
        global_idx_str = (
            f"{int(global_idx):08d}"
            if global_idx is not None
            else "unknown_idx"
        )

        intensity_display = np.power(
            np.clip(lam, 0.0, None),
            self.intensity_gamma,
        )

        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)

        # A single axes fills almost the complete frame. The colorbar is inset,
        # so it does not create an additional white column beside the BEV plot.
        ax = fig.add_axes([0.085, 0.070, 0.900, 0.875])

        intensity_mesh = ax.pcolormesh(
            geometry["y_edges"],
            geometry["x_edges"],
            intensity_display,
            shading="auto",
            cmap=self.heatmap_cmap,
        )

        if gt_count > 0:
            ax.scatter(
                gt_xy[:, 1],
                gt_xy[:, 0],
                s=self.point_size,
                c=self.point_color,
                alpha=self.point_alpha,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )

        self._setup_bev_axes(
            ax=ax,
            r_min=geometry["r_min"],
            r_max=geometry["r_max"],
        )
        ax.set_title(
            "Predicted PPP intensity in BEV with GT radar points",
            fontsize=11,
            pad=4,
        )

        if self.show_textbox:
            self._textbox(
                ax,
                (
                    f"E[N]={expected_count:.1f}\n"
                    f"GT N={gt_count}\n"
                    f"GT={gt_source}"
                ),
            )

        if self.show_colorbar:
            self._add_inset_colorbar(
                fig=fig,
                ax=ax,
                mappable=intensity_mesh,
                label=rf"$\lambda^{{{self.intensity_gamma:g}}}$",
            )

        output_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_str=frame_str,
            global_idx_str=global_idx_str,
        )
        fig.savefig(output_path, dpi=self.dpi)
        fig.clear()