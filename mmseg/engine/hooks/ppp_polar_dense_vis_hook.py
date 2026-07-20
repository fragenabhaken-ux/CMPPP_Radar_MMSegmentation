from __future__ import annotations

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


def fig_to_rgb(fig: Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


@HOOKS.register_module()
class PPPPolarDenseVisHook(Hook):
    """Training visualization hook for dense polar-grid PPP prediction.

    Plot layout (4 x 2):
      1. top-left:      input image
      2. top-right:     predicted PPP intensity warped to Cartesian BEV
      3. second-left:   GT occupied cells in Cartesian BEV
      4. second-right:  GT occupied cells in polar coordinates
      5. third-left:    predicted polar intensity lambda(r, theta)
      6. third-right:   predicted polar cell mass
      7. bottom-left:   predicted velocity in BEV with GT velocity points
      8. bottom-right:  predicted RCS in BEV with GT RCS points

    Dense prediction channels:
      0: log intensity
      1: raw velocity mark
      2: raw RCS mark

    Dense target channels:
      0: binary occupancy
      1: GT velocity
      2: GT RCS
    """

    def __init__(
        self,
        train_interval: int = 1000,
        max_samples_train: int = 1,
        max_samples_val: int = 5,
        max_samples_test: int = 5,
        img_mean=None,
        img_std=None,
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        bev_y_min: float = -85.0,
        bev_y_max: float = 85.0,
        gamma: float = 0.3,
        point_size: float = 12.0,
        point_color: str = "cyan",
        point_edge_color: str = "black",
        point_edge_width: float = 0.35,
        velocity_mode: str = "signed",
        velocity_link: str = "cube",
        velocity_vmin: float = -11.0,
        velocity_vmax: float = 11.0,
        velocity_cmap: str = "coolwarm",
        rcs_link: str = "cube",
        rcs_vmin: float = -50.0,
        rcs_vmax: float = 20.0,
        rcs_cmap: str = "viridis",
        dpi: int = 100,
        figsize=(18, 24),
    ) -> None:
        self.train_interval = int(train_interval)
        self.max_samples = {
            "train": int(max_samples_train),
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)
        self.gamma = float(gamma)

        self.point_size = float(point_size)
        self.point_color = str(point_color)
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

        if self.r_min >= self.r_max:
            raise ValueError(
                f"r_min must be smaller than r_max, got "
                f"{self.r_min} >= {self.r_max}."
            )
        if self.theta_min >= self.theta_max:
            raise ValueError(
                f"theta_min must be smaller than theta_max, got "
                f"{self.theta_min} >= {self.theta_max}."
            )
        if self.bev_y_min >= self.bev_y_max:
            raise ValueError(
                f"bev_y_min must be smaller than bev_y_max, got "
                f"{self.bev_y_min} >= {self.bev_y_max}."
            )
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                "velocity_mode must be 'signed' or 'absolute', "
                f"got {self.velocity_mode!r}."
            )
        if self.velocity_link not in {"identity", "cube"}:
            raise ValueError(
                "velocity_link must be 'identity' or 'cube', "
                f"got {self.velocity_link!r}."
            )
        if self.velocity_vmin >= self.velocity_vmax:
            raise ValueError(
                "velocity_vmin must be smaller than velocity_vmax, got "
                f"{self.velocity_vmin} >= {self.velocity_vmax}."
            )
        if self.rcs_link not in {"identity", "cube"}:
            raise ValueError(
                "rcs_link must be 'identity' or 'cube', "
                f"got {self.rcs_link!r}."
            )
        if self.rcs_vmin >= self.rcs_vmax:
            raise ValueError(
                "rcs_vmin must be smaller than rcs_vmax, got "
                f"{self.rcs_vmin} >= {self.rcs_vmax}."
            )

        self._written_in_run = {"val": 0, "test": 0}

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
        if not is_main_process():
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
        if not is_main_process():
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
        if not is_main_process():
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

        if not torch.is_tensor(preds):
            raise TypeError(
                "PPPPolarDenseVisHook expects decode_head output to be a "
                f"tensor, got {type(preds)}."
            )
        if preds.ndim != 4 or preds.shape[1] != 3:
            raise ValueError(
                "Expected dense polar predictions with shape [B, 3, H, W], "
                f"got {tuple(preds.shape)}."
            )

        if phase == "train":
            n_vis = min(self.max_samples["train"], preds.shape[0])
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
                step=step,
                phase=phase,
                img_tensor=inputs[i],
                pred_grid=preds[i],
                sample=data_samples[i],
                sample_idx=start_idx + i,
            )

        if phase in self._written_in_run:
            self._written_in_run[phase] += n_vis

        if was_training:
            model.train()

    def _decode_velocity(self, raw_velocity: torch.Tensor) -> np.ndarray:
        raw_velocity = raw_velocity.detach().float().cpu()

        if self.velocity_link == "identity":
            velocity = raw_velocity
        elif self.velocity_link == "cube":
            velocity = raw_velocity.pow(3)
        else:
            raise RuntimeError(
                f"Unsupported velocity_link={self.velocity_link!r}."
            )

        if self.velocity_mode == "absolute":
            velocity = velocity.abs()

        return velocity.numpy()

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        raw_rcs = raw_rcs.detach().float().cpu()

        if self.rcs_link == "identity":
            rcs = raw_rcs
        elif self.rcs_link == "cube":
            rcs = raw_rcs.pow(3)
        else:
            raise RuntimeError(f"Unsupported rcs_link={self.rcs_link!r}.")

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
    def _tagify(value: str) -> str:
        return "".join(
            c if c.isalnum() or c in "._-/" else "_"
            for c in value
        )

    @staticmethod
    def _label(ax, text: str) -> None:
        ax.text(
            -0.08,
            0.5,
            text,
            transform=ax.transAxes,
            rotation=90,
            fontsize=12,
            va="center",
            ha="center",
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
            bbox=dict(facecolor="black", alpha=0.55, pad=3),
        )

    def _polar_geometry(self, h_r: int, w_theta: int) -> dict:
        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        r_edges = np.linspace(self.r_min, self.r_max, h_r + 1)
        theta_edges = np.linspace(
            self.theta_min,
            self.theta_max,
            w_theta + 1,
        )

        r_centres_1d = (
            self.r_min
            + (np.arange(h_r, dtype=np.float64) + 0.5) * dr
        )
        theta_centres_1d = (
            self.theta_min
            + (np.arange(w_theta, dtype=np.float64) + 0.5) * dtheta
        )

        rr_edges, tt_edges = np.meshgrid(
            r_edges,
            theta_edges,
            indexing="ij",
        )
        rr_centres, tt_centres = np.meshgrid(
            r_centres_1d,
            theta_centres_1d,
            indexing="ij",
        )

        return dict(
            dr=dr,
            dtheta=dtheta,
            x_edges=rr_edges * np.cos(tt_edges),
            y_edges=rr_edges * np.sin(tt_edges),
            r_centres=rr_centres,
            theta_centres=tt_centres,
            x_centres=rr_centres * np.cos(tt_centres),
            y_centres=rr_centres * np.sin(tt_centres),
        )

    def _setup_bev_axes(self, ax) -> None:
        # Positive lateral values on the left, negative values on the right.
        ax.set_xlim(self.bev_y_max, self.bev_y_min)
        ax.set_ylim(self.r_min, self.r_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y lateral [m]")
        ax.set_ylabel("x forward [m]")
        ax.grid(True, alpha=0.25)

    def _setup_polar_axes(self, ax) -> None:
        # Positive azimuth on the left, negative azimuth on the right.
        ax.set_xlim(self.theta_max, self.theta_min)
        ax.set_ylim(self.r_min, self.r_max)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$r$")
        ax.grid(True, alpha=0.25)

    @staticmethod
    def _extract_gt(sample) -> torch.Tensor:
        if not hasattr(sample, "gt_sem_seg"):
            raise AttributeError("Data sample has no gt_sem_seg field.")

        gt = sample.gt_sem_seg.data

        if not torch.is_tensor(gt):
            raise TypeError(
                "Expected sample.gt_sem_seg.data to be a tensor, "
                f"got {type(gt)}."
            )

        if gt.ndim == 4:
            if gt.shape[0] != 1:
                raise ValueError(
                    "Unexpected batched GT inside one data sample: "
                    f"{tuple(gt.shape)}."
                )
            gt = gt[0]

        if gt.ndim != 3 or gt.shape[0] != 3:
            raise ValueError(
                "Expected dense polar GT with shape [3, H, W], "
                f"got {tuple(gt.shape)}."
            )

        return gt.detach().float().cpu()

    def _visualize_sample(
        self,
        runner,
        step: int,
        phase: str,
        img_tensor: torch.Tensor,
        pred_grid: torch.Tensor,
        sample,
        sample_idx: int,
    ) -> None:
        img = self._denorm_img(img_tensor.detach().cpu().numpy())
        pred_grid = pred_grid.detach().float().cpu()
        gt = self._extract_gt(sample)

        if pred_grid.shape != gt.shape:
            raise ValueError(
                "Prediction and GT grids must have identical shapes, "
                f"got prediction={tuple(pred_grid.shape)}, "
                f"GT={tuple(gt.shape)}."
            )

        log_lambda = pred_grid[0]
        raw_velocity = pred_grid[1]
        raw_rcs = pred_grid[2]

        lam = torch.exp(log_lambda).numpy()
        velocity = self._decode_velocity(raw_velocity)
        rcs = self._decode_rcs(raw_rcs)

        occupancy = gt[0].numpy()
        gt_velocity_grid = gt[1].numpy()
        gt_rcs_grid = gt[2].numpy()

        unique_occupancy = np.unique(occupancy)
        if not np.all(np.isin(unique_occupancy, [0.0, 1.0])):
            raise ValueError(
                "Dense polar occupancy must be binary, found "
                f"{unique_occupancy.tolist()}."
            )

        occupied = occupancy > 0.0
        gt_count = int(occupied.sum())

        h_r, w_theta = lam.shape
        geometry = self._polar_geometry(h_r=h_r, w_theta=w_theta)

        dr = float(geometry["dr"])
        dtheta = float(geometry["dtheta"])
        cell_mass = lam * dr * dtheta
        expected_count = float(cell_mass.sum())

        x_edges = geometry["x_edges"]
        y_edges = geometry["y_edges"]
        x_centres = geometry["x_centres"]
        y_centres = geometry["y_centres"]
        r_centres = geometry["r_centres"]
        theta_centres = geometry["theta_centres"]

        gt_x = x_centres[occupied]
        gt_y = y_centres[occupied]
        gt_r = r_centres[occupied]
        gt_theta = theta_centres[occupied]
        gt_velocity = gt_velocity_grid[occupied].astype(
            np.float64,
            copy=False,
        )
        gt_rcs = gt_rcs_grid[occupied].astype(
            np.float64,
            copy=False,
        )

        if self.velocity_mode == "absolute":
            gt_velocity = np.abs(gt_velocity)

        meta = sample.metainfo
        video = self._tagify(
            str(meta.get("video_name", "unknown_video"))
        )
        frame = meta.get("video_frame_idx")
        global_idx = meta.get("global_idx", meta.get("frame_idx"))

        frame_str = (
            f"{int(frame):06d}"
            if frame is not None
            else "unknown_frame"
        )
        global_idx_str = (
            f"{int(global_idx):08d}"
            if global_idx is not None
            else "unknown_idx"
        )

        fig = Figure(figsize=self.figsize, dpi=self.dpi)
        FigureCanvasAgg(fig)
        axs = fig.subplots(4, 2)

        fig.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.95,
            bottom=0.045,
            hspace=0.28,
            wspace=0.16,
        )

        fig.suptitle(
            f"{phase} | {video} | "
            f"frame {frame_str} | idx {global_idx_str}",
            fontsize=14,
        )

        # 1. Input image
        ax = axs[0, 0]
        ax.imshow(img, aspect="auto")
        ax.axis("off")
        self._label(ax, "Image")

        # 2. Predicted intensity warped to Cartesian BEV
        ax = axs[0, 1]
        ax.set_title("Predicted PPP intensity in BEV with GT points")
        intensity_pcm = ax.pcolormesh(
            y_edges,
            x_edges,
            np.power(lam, self.gamma),
            shading="auto",
            cmap="hot",
        )
        if gt_count > 0:
            ax.scatter(
                gt_y,
                gt_x,
                s=self.point_size,
                c=self.point_color,
                alpha=0.9,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )
        self._setup_bev_axes(ax)
        fig.colorbar(intensity_pcm, ax=ax, fraction=0.046, pad=0.04)
        self._textbox(
            ax,
            (
                f"E[N]={expected_count:.2f}\n"
                f"GT={gt_count}\n"
                f"gamma={self.gamma:g}"
            ),
        )

        # 3. GT occupied cells in Cartesian BEV
        ax = axs[1, 0]
        ax.set_title("GT occupied polar cells in BEV")
        if gt_count > 0:
            ax.scatter(
                gt_y,
                gt_x,
                s=self.point_size,
                c="lime",
                alpha=0.85,
                edgecolors="none",
            )
        self._setup_bev_axes(ax)
        self._textbox(ax, f"GT N={gt_count}")

        # 4. GT occupied cells in polar coordinates
        ax = axs[1, 1]
        ax.set_title("GT occupied polar cells")
        if gt_count > 0:
            ax.scatter(
                gt_theta,
                gt_r,
                s=self.point_size,
                c="lime",
                alpha=0.85,
                edgecolors="none",
            )
        self._setup_polar_axes(ax)
        self._textbox(ax, f"GT N={gt_count}")

        # 5. Predicted polar intensity
        ax = axs[2, 0]
        intensity_polar_im = ax.imshow(
            lam,
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
        ax.set_title(r"Predicted $\lambda(r,\theta)$")
        self._setup_polar_axes(ax)
        fig.colorbar(
            intensity_polar_im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        self._textbox(
            ax,
            (
                f"E[N]={expected_count:.2f}\n"
                f"GT={gt_count}\n"
                f"Delta={expected_count - gt_count:+.2f}"
            ),
        )

        # 6. Predicted polar cell mass
        ax = axs[2, 1]
        cell_mass_im = ax.imshow(
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
        self._setup_polar_axes(ax)
        fig.colorbar(cell_mass_im, ax=ax, fraction=0.046, pad=0.04)
        self._textbox(ax, f"gamma={self.gamma:g}")

        # 7. Predicted velocity in BEV with GT velocity points
        ax = axs[3, 0]
        velocity_pcm = ax.pcolormesh(
            y_edges,
            x_edges,
            velocity,
            shading="auto",
            cmap=self.velocity_cmap,
            vmin=self.velocity_vmin,
            vmax=self.velocity_vmax,
        )
        finite_velocity = np.isfinite(gt_velocity)
        if gt_count > 0 and np.any(finite_velocity):
            ax.scatter(
                gt_y[finite_velocity],
                gt_x[finite_velocity],
                s=self.point_size,
                c=gt_velocity[finite_velocity],
                cmap=self.velocity_cmap,
                vmin=self.velocity_vmin,
                vmax=self.velocity_vmax,
                alpha=0.95,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )
        self._setup_bev_axes(ax)

        if self.velocity_mode == "signed":
            velocity_title = (
                "Predicted signed velocity in BEV with GT velocity points"
            )
            velocity_label = "velocity [m/s]"
        else:
            velocity_title = (
                "Predicted absolute velocity in BEV with GT velocity points"
            )
            velocity_label = "|velocity| [m/s]"

        ax.set_title(velocity_title)
        velocity_cbar = fig.colorbar(
            velocity_pcm,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        velocity_cbar.set_label(velocity_label)
        self._textbox(
            ax,
            (
                f"link={self.velocity_link}\n"
                f"mode={self.velocity_mode}\n"
                f"GT mark points={int(finite_velocity.sum())}"
            ),
        )

        # 8. Predicted RCS in BEV with GT RCS points
        ax = axs[3, 1]
        rcs_pcm = ax.pcolormesh(
            y_edges,
            x_edges,
            rcs,
            shading="auto",
            cmap=self.rcs_cmap,
            vmin=self.rcs_vmin,
            vmax=self.rcs_vmax,
        )
        finite_rcs = np.isfinite(gt_rcs)
        if gt_count > 0 and np.any(finite_rcs):
            ax.scatter(
                gt_y[finite_rcs],
                gt_x[finite_rcs],
                s=self.point_size,
                c=gt_rcs[finite_rcs],
                cmap=self.rcs_cmap,
                vmin=self.rcs_vmin,
                vmax=self.rcs_vmax,
                alpha=0.95,
                edgecolors=self.point_edge_color,
                linewidths=self.point_edge_width,
                zorder=3,
            )
        self._setup_bev_axes(ax)
        ax.set_title("Predicted RCS in BEV with GT RCS points")
        rcs_cbar = fig.colorbar(
            rcs_pcm,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        rcs_cbar.set_label("RCS")
        self._textbox(
            ax,
            (
                f"link={self.rcs_link}\n"
                f"GT mark points={int(finite_rcs.sum())}"
            ),
        )

        tag = (
            f"ppp_polar_dense_vis/{phase}/"
            f"{video}_f{frame_str}_idx{global_idx_str}_"
            f"sample{sample_idx}"
        )
        runner.visualizer.add_image(
            tag,
            fig_to_rgb(fig),
            step=step,
        )
        fig.clear()