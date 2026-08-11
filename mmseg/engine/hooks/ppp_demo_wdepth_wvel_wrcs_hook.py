"""
PPP demo visualization hook with depth, velocity, and RCS panels.

Reduced visualization hook for producing demo frames as image files.

The output layout contains:
  1. GT points projected on the camera image, colored by GT depth
  2. PPP sample by thinning projected on the camera image, colored by predicted depth
  3. PPP sample by block approximation projected on the camera image, colored by predicted depth
  4. Predicted PPP intensity mass heatmap
  5. Gamma-adjusted predicted PPP intensity mass heatmap
  6. Predicted depth heatmap with GT depths overlaid as dots
  7. Predicted velocity heatmap with GT velocities overlaid as dots
  8. Predicted RCS heatmap with GT RCS values overlaid as dots

Current four-channel camera-domain convention:

    prediction:
        pred[0] = log intensity parameter
        pred[1] = raw depth, decoded with exp
        pred[2] = raw velocity
        pred[3] = raw RCS

    ground truth:
        gt[0] = support mask
        gt[1] = depth
        gt[2] = velocity
        gt[3] = RCS

Velocity is decoded according to ``decode_head.velocity_mode``:

    signed:   raw_velocity ** 3
    absolute: abs(raw_velocity)

RCS is decoded according to ``rcs_link``:

    identity:    raw_rcs
    cube:        raw_rcs ** 3
    linear_cube: raw_rcs + raw_rcs ** 3

The hook writes PNG files directly to:

    <runner.work_dir>/<out_dir>/<phase>/<video_name>/frame_<frame>_idx_<idx>.png

For distributed test runs, set ``write_all_ranks=True``.

The class name is intentionally kept as ``PPPDemoWDepthWVelVisHook`` so that
existing configs using the previous depth-and-velocity demo hook remain valid.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


def make_synthetic_depth_cmap(name: str = "synthetic_depth_clash"):
    """Create a high-contrast synthetic colormap for depth visualization."""
    return LinearSegmentedColormap.from_list(
        name,
        [
            "#ff00ff",  # neon magenta
            "#7a00ff",  # violet
            "#003cff",  # electric blue
            "#00e5ff",  # cyan
            "#00ff66",  # neon green
            "#ffff00",  # yellow
            "#ff6a00",  # orange
            "#ff0000",  # red
        ],
        N=256,
    )


def extract_gt_points_and_values(
    mask: np.ndarray,
    values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return occupied GT cell centers and values at those cells.

    Coordinates are returned as ``(x, y)`` in image-grid coordinates.
    """
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    points = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)

    if values is None:
        point_values = np.zeros((len(points),), dtype=np.float64)
    else:
        point_values = values[ys, xs].astype(np.float64)

    return points, point_values


def sample_grid_values_at_points(
    grid: np.ndarray | None,
    points: np.ndarray,
) -> np.ndarray:
    """Sample a dense grid at floating-point ``(x, y)`` coordinates.

    The lower integer pixel index is used, matching the cell containing each
    sampled PPP point.
    """
    if grid is None or len(points) == 0:
        return np.zeros((0,), dtype=np.float64)

    height, width = grid.shape[-2:]

    xs = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
    ys = np.clip(points[:, 1].astype(np.int64), 0, height - 1)

    return grid[ys, xs].astype(np.float64)


def sample_ppp_thinning(
    lam: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """Sample a point cloud from a discrete PPP intensity by thinning."""
    rng = np.random.default_rng() if rng is None else rng

    lam = np.asarray(lam, dtype=np.float64)

    if lam.ndim != 2:
        raise ValueError("lambda grid must be 2D")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lambda grid contains NaN/Inf")
    if np.any(lam < 0):
        raise ValueError("lambda grid contains negative values")

    height, width = lam.shape
    lam_max = float(lam.max())

    if lam_max <= 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    num_proposals = int(rng.poisson(lam_max * height * width))

    if num_proposals == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_proposals": 0,
        }

    xs = rng.random(num_proposals) * width
    ys = rng.random(num_proposals) * height

    xi = np.clip(xs.astype(np.int64), 0, width - 1)
    yi = np.clip(ys.astype(np.int64), 0, height - 1)

    keep = rng.random(num_proposals) < (lam[yi, xi] / lam_max)
    points = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)

    return points, {
        "num_points": int(len(points)),
        "num_proposals": int(num_proposals),
    }


def sample_ppp_block(
    lam: np.ndarray,
    block_size: int = 8,
    placement: str = "weighted",
    rng: np.random.Generator | None = None,
):
    """Sample a point cloud using block activation probabilities."""
    rng = np.random.default_rng() if rng is None else rng

    lam = np.asarray(lam, dtype=np.float64)

    if lam.ndim != 2:
        raise ValueError("lambda grid must be 2D")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lambda grid contains NaN/Inf")
    if np.any(lam < 0):
        raise ValueError("lambda grid contains negative values")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if placement not in {"weighted", "uniform"}:
        raise ValueError("placement must be 'weighted' or 'uniform'")

    height, width = lam.shape
    block_size = int(block_size)

    num_blocks_y = (height + block_size - 1) // block_size
    num_blocks_x = (width + block_size - 1) // block_size

    padded_height = num_blocks_y * block_size
    padded_width = num_blocks_x * block_size

    lam_padded = np.zeros((padded_height, padded_width), dtype=np.float64)
    lam_padded[:height, :width] = lam

    lam_blocks = lam_padded.reshape(
        num_blocks_y,
        block_size,
        num_blocks_x,
        block_size,
    ).transpose(0, 2, 1, 3)

    block_mass = lam_blocks.sum(axis=(2, 3))
    activation_probability = 1.0 - np.exp(-block_mass)

    active = rng.random((num_blocks_y, num_blocks_x)) < activation_probability
    block_y, block_x = np.nonzero(active)
    num_points = len(block_y)

    if num_points == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "num_points": 0,
            "num_active_blocks": 0,
        }

    points = np.empty((num_points, 2), dtype=np.float64)

    x_starts = block_x * block_size
    y_starts = block_y * block_size
    block_widths = np.minimum(block_size, width - x_starts)
    block_heights = np.minimum(block_size, height - y_starts)

    if placement == "uniform":
        points[:, 0] = x_starts + rng.random(num_points) * block_widths
        points[:, 1] = y_starts + rng.random(num_points) * block_heights
    else:
        for i, (x0, y0, block_width, block_height) in enumerate(
            zip(x_starts, y_starts, block_widths, block_heights)
        ):
            block = lam[
                y0:y0 + block_height,
                x0:x0 + block_width,
            ]
            block_sum = float(block.sum())

            if block_sum <= 0:
                points[i, 0] = x0 + rng.random() * block_width
                points[i, 1] = y0 + rng.random() * block_height
                continue

            flat_index = int(
                rng.choice(
                    block.size,
                    p=block.ravel() / block_sum,
                )
            )
            pixel_y, pixel_x = divmod(flat_index, int(block_width))

            points[i, 0] = x0 + pixel_x + rng.random()
            points[i, 1] = y0 + pixel_y + rng.random()

    return points, {
        "num_points": int(num_points),
        "num_active_blocks": int(num_points),
    }


@HOOKS.register_module()
class PPPDemoWDepthWVelWRCSVisHook(Hook):
    """Write PPP demo frames with depth, velocity, and RCS predictions."""

    def __init__(
        self,
        train_interval: int = 0,
        max_samples_train: int = 0,
        max_samples_val: int = 0,
        max_samples_test: int = 10,
        out_dir: str = "demo_frames",
        img_mean=None,
        img_std=None,
        dpi: int = 90,
        figsize=(14, 32),
        gamma: float = 0.9,
        sim_block_size: int = 8,
        sim_block_placement: str = "weighted",
        point_size: float = 28,
        image_ext: str = "png",
        write_all_ranks: bool = False,
        depth_vmin=None,
        depth_vmax=None,
        depth_cmap="synthetic",
        depth_point_edge_color: str = "black",
        depth_point_edge_width: float = 0.45,
        vel_vmin=None,
        vel_vmax=None,
        vel_cmap: str = "coolwarm",
        vel_point_edge_color: str = "black",
        vel_point_edge_width: float = 0.45,
        rcs_link: str = "cube",
        rcs_vmin: float | None = -50.0,
        rcs_vmax: float | None = 20.0,
        rcs_cmap: str = "viridis",
        rcs_point_edge_color: str = "black",
        rcs_point_edge_width: float = 0.45,
        require_four_channels: bool = True,
    ) -> None:
        self.train_interval = int(train_interval)
        self.max_samples = {
            "train": int(max_samples_train),
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.out_dir = str(out_dir)
        self.image_ext = str(image_ext).lstrip(".")
        self.dpi = int(dpi)
        self.figsize = tuple(figsize)
        self.gamma = float(gamma)
        self.sim_block_size = int(sim_block_size)
        self.sim_block_placement = str(sim_block_placement)
        self.point_size = float(point_size)
        self.write_all_ranks = bool(write_all_ranks)
        self.require_four_channels = bool(require_four_channels)

        if self.sim_block_size <= 0:
            raise ValueError("sim_block_size must be > 0")

        if self.sim_block_placement not in {"weighted", "uniform"}:
            raise ValueError(
                "sim_block_placement must be 'weighted' or 'uniform', "
                f"got {self.sim_block_placement!r}."
            )

        self.depth_vmin = depth_vmin
        self.depth_vmax = depth_vmax
        self.depth_point_edge_color = str(depth_point_edge_color)
        self.depth_point_edge_width = float(depth_point_edge_width)

        self.vel_vmin = vel_vmin
        self.vel_vmax = vel_vmax
        self.vel_cmap = str(vel_cmap)
        self.vel_point_edge_color = str(vel_point_edge_color)
        self.vel_point_edge_width = float(vel_point_edge_width)

        self.rcs_link = str(rcs_link).lower()
        self.rcs_vmin = rcs_vmin
        self.rcs_vmax = rcs_vmax
        self.rcs_cmap = str(rcs_cmap)
        self.rcs_point_edge_color = str(rcs_point_edge_color)
        self.rcs_point_edge_width = float(rcs_point_edge_width)

        if self.rcs_link not in {"identity", "cube", "linear_cube"}:
            raise ValueError(
                "rcs_link must be 'identity', 'cube', or 'linear_cube', "
                f"got {self.rcs_link!r}."
            )

        if (
            self.rcs_vmin is not None
            and self.rcs_vmax is not None
            and self.rcs_vmin >= self.rcs_vmax
        ):
            raise ValueError(
                "rcs_vmin must be smaller than rcs_vmax, got "
                f"{self.rcs_vmin} >= {self.rcs_vmax}."
            )

        if depth_cmap == "synthetic":
            self.depth_cmap = make_synthetic_depth_cmap()
        else:
            self.depth_cmap = depth_cmap

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
            step=runner.iter,
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
            step=runner.iter,
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
            step=runner.iter,
        )

    def _run(
        self,
        runner,
        data_batch: dict,
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

        try:
            velocity_mode = getattr(
                model.decode_head,
                "velocity_mode",
                "signed",
            )

            if velocity_mode not in {"signed", "absolute"}:
                raise ValueError(
                    "velocity_mode must be 'signed' or 'absolute', "
                    f"got {velocity_mode!r}."
                )

            processed = model.data_preprocessor(data_batch)
            inputs = processed["inputs"]
            data_samples = processed["data_samples"]

            with torch.no_grad():
                predictions = model.decode_head(
                    model.extract_feat(inputs)
                )

            if not torch.is_tensor(predictions) or predictions.ndim != 4:
                raise TypeError(
                    "PPPDemoWDepthWVelVisHook expects decode-head output "
                    "with shape (B, C, H, W), got "
                    f"{type(predictions)!r} with shape "
                    f"{getattr(predictions, 'shape', None)!r}."
                )

            minimum_prediction_channels = 4 if self.require_four_channels else 3
            if predictions.shape[1] < minimum_prediction_channels:
                raise ValueError(
                    "Expected prediction channels "
                    "[log intensity, depth, velocity, RCS], but got shape "
                    f"{tuple(predictions.shape)}."
                )

            if phase == "train":
                num_visualizations = min(
                    self.max_samples["train"],
                    predictions.size(0),
                    len(data_samples),
                )
                start_index = 0
            else:
                remaining = (
                    self.max_samples[phase]
                    - self._written_in_run[phase]
                )

                if remaining <= 0:
                    return

                num_visualizations = min(
                    remaining,
                    predictions.size(0),
                    len(data_samples),
                )
                start_index = self._written_in_run[phase]

            for i in range(num_visualizations):
                self._visualize_sample(
                    runner=runner,
                    step=step,
                    phase=phase,
                    img_tensor=inputs[i],
                    pred=predictions[i],
                    sample=data_samples[i],
                    iter_seed=runner.iter,
                    sample_idx=start_index + i,
                    velocity_mode=velocity_mode,
                )

            if phase in self._written_in_run:
                self._written_in_run[phase] += num_visualizations

        finally:
            if was_training:
                model.train()

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)

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
            character
            if character.isalnum() or character in "._-/"
            else "_"
            for character in value
        )

    @staticmethod
    def _label(ax, text: str) -> None:
        ax.text(
            -0.015,
            0.5,
            text,
            transform=ax.transAxes,
            rotation=90,
            fontsize=13,
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
            fontsize=11,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="black",
                alpha=0.55,
                pad=3,
            ),
        )

    def _imshow(
        self,
        ax,
        array: np.ndarray,
        ylabel: str,
        cmap=None,
        vmin=None,
        vmax=None,
        fig: Figure | None = None,
        colorbar: bool = False,
    ):
        image = ax.imshow(
            array,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        ax.axis("off")
        self._label(ax, ylabel)

        if colorbar and fig is not None:
            colorbar_axis = ax.inset_axes([0.0, -0.06, 1.0, 0.04])
            colorbar_object = fig.colorbar(
                image,
                cax=colorbar_axis,
                orientation="horizontal",
            )
            colorbar_object.ax.tick_params(labelsize=10)

        return image

    def _overlay_depth_points(
        self,
        ax,
        image: np.ndarray,
        points: np.ndarray,
        depth_values: np.ndarray,
        ylabel: str,
        count: int,
        textbox_extra: str | None = None,
    ) -> None:
        ax.imshow(image, aspect="auto")

        self._scatter_values(
            ax=ax,
            points=points,
            values=depth_values,
            cmap=self.depth_cmap,
            vmin=self.depth_vmin,
            vmax=self.depth_vmax,
            edge_color=self.depth_point_edge_color,
            edge_width=self.depth_point_edge_width,
        )

        ax.axis("off")
        self._label(ax, ylabel)

        text = f"N={count}"
        if textbox_extra:
            text += f"\n{textbox_extra}"
        self._textbox(ax, text)

    def _scatter_values(
        self,
        ax,
        points: np.ndarray,
        values: np.ndarray,
        cmap,
        vmin,
        vmax,
        edge_color: str,
        edge_width: float,
    ) -> None:
        if len(points) == 0 or len(values) == 0:
            return

        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)

        if not np.any(finite):
            return

        points_to_plot = points[finite]
        values_to_plot = values[finite]

        ax.scatter(
            points_to_plot[:, 0],
            points_to_plot[:, 1],
            s=self.point_size,
            c=values_to_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.98,
            edgecolors=edge_color,
            linewidths=edge_width,
        )

    def _output_path(
        self,
        runner,
        phase: str,
        video: str,
        frame_string: str,
        global_index_string: str,
    ) -> Path:
        root = Path(self.out_dir)

        if not root.is_absolute():
            root = Path(runner.work_dir) / root

        output_directory = root / phase / video
        output_directory.mkdir(parents=True, exist_ok=True)

        return output_directory / (
            f"frame_{frame_string}_idx_{global_index_string}."
            f"{self.image_ext}"
        )

    @staticmethod
    def _decode_velocity(
        raw_velocity: torch.Tensor,
        velocity_mode: str,
    ) -> tuple[np.ndarray, str]:
        if velocity_mode == "signed":
            velocity = raw_velocity.pow(3)
            label = "Pred vel [m/s]"
        elif velocity_mode == "absolute":
            velocity = raw_velocity.abs()
            label = "Pred |vel| [m/s]"
        else:
            raise RuntimeError(
                f"Unsupported velocity_mode={velocity_mode!r}."
            )

        return velocity.detach().cpu().numpy(), label

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        if self.rcs_link == "identity":
            rcs = raw_rcs
        elif self.rcs_link == "cube":
            rcs = raw_rcs.pow(3)
        elif self.rcs_link == "linear_cube":
            rcs = raw_rcs + raw_rcs.pow(3)
        else:
            raise RuntimeError(
                f"Unsupported rcs_link={self.rcs_link!r}."
            )

        return rcs.detach().cpu().numpy()

    def _visualize_sample(
        self,
        runner,
        step: int,
        phase: str,
        img_tensor: torch.Tensor,
        pred: torch.Tensor,
        sample,
        iter_seed: int,
        sample_idx: int,
        velocity_mode: str,
    ) -> None:
        del step, sample_idx

        gt_all = sample.gt_sem_seg.data.to(pred.device)
        minimum_gt_channels = 4 if self.require_four_channels else 3

        if gt_all.ndim < 3 or gt_all.shape[0] < minimum_gt_channels:
            raise ValueError(
                "Expected GT channels [support, depth, velocity, RCS], "
                f"but got shape {tuple(gt_all.shape)}."
            )

        # Ground truth channel convention.
        gt_mask_tensor = gt_all[0]
        gt_depth_tensor = gt_all[1]
        gt_velocity_tensor = gt_all[2]
        gt_rcs_tensor = gt_all[3] if gt_all.shape[0] > 3 else None

        # Prediction channel convention.
        predicted_lambda = torch.exp(pred[0]).detach().cpu().numpy()
        predicted_depth = torch.exp(pred[1]).detach().cpu().numpy()
        predicted_velocity, predicted_velocity_label = self._decode_velocity(
            raw_velocity=pred[2],
            velocity_mode=velocity_mode,
        )
        predicted_rcs = (
            self._decode_rcs(pred[3])
            if pred.shape[0] > 3
            else None
        )

        gt_mask = gt_mask_tensor.detach().cpu().numpy()
        gt_depth = gt_depth_tensor.detach().cpu().numpy()
        gt_velocity = gt_velocity_tensor.detach().cpu().numpy()
        gt_rcs = (
            gt_rcs_tensor.detach().cpu().numpy()
            if gt_rcs_tensor is not None
            else None
        )

        image = self._denorm_img(
            img_tensor.detach().cpu().numpy()
        )

        metadata = sample.metainfo
        video = self._tagify(
            str(metadata.get("video_name", "unknown_video"))
        )
        frame = metadata.get("video_frame_idx")
        global_index = metadata.get(
            "global_idx",
            metadata.get("frame_idx"),
        )

        frame_string = (
            f"{int(frame):06d}"
            if frame is not None
            else "unknown_frame"
        )
        global_index_string = (
            f"{int(global_index):08d}"
            if global_index is not None
            else "unknown_idx"
        )

        original_shape = metadata.get("ori_shape")

        if (
            isinstance(original_shape, (tuple, list))
            and len(original_shape) >= 2
        ):
            original_height = int(original_shape[0])
            original_width = int(original_shape[1])
        else:
            original_height, original_width = predicted_lambda.shape[-2:]

        if original_height <= 0 or original_width <= 0:
            raise ValueError(
                "Invalid original image shape "
                f"{(original_height, original_width)}."
            )

        # Existing camera-domain convention: exp(pred[0]) is converted to
        # expected-count mass per original image cell before sampling.
        lambda_mass = predicted_lambda / float(
            original_height * original_width
        )

        expected_count = float(lambda_mass.sum())
        ground_truth_count = float(gt_mask.sum())

        gt_points, gt_depth_values = extract_gt_points_and_values(
            gt_mask,
            gt_depth,
        )
        _, gt_velocity_values = extract_gt_points_and_values(
            gt_mask,
            gt_velocity,
        )
        _, gt_rcs_values = extract_gt_points_and_values(
            gt_mask,
            gt_rcs,
        )

        if velocity_mode == "absolute":
            gt_velocity_values = np.abs(gt_velocity_values)

        seed_base = (
            int(frame)
            if frame is not None
            else int(metadata.get("frame_idx", 0))
        )
        seed = seed_base + 1000 * int(iter_seed)

        thinning_points, thinning_stats = sample_ppp_thinning(
            lambda_mass,
            rng=np.random.default_rng(seed + 11),
        )
        block_points, block_stats = sample_ppp_block(
            lambda_mass,
            block_size=self.sim_block_size,
            placement=self.sim_block_placement,
            rng=np.random.default_rng(seed + 29),
        )

        thinning_depths = sample_grid_values_at_points(
            predicted_depth,
            thinning_points,
        )
        block_depths = sample_grid_values_at_points(
            predicted_depth,
            block_points,
        )

        figure = Figure(
            figsize=self.figsize,
            dpi=self.dpi,
        )
        FigureCanvasAgg(figure)

        axes = figure.subplots(8, 1)
        figure.subplots_adjust(
            left=0.025,
            right=0.99,
            top=0.96,
            bottom=0.012,
            hspace=0.10,
        )
        figure.suptitle(
            f"{phase} | {video} | frame {frame_string} | "
            f"idx {global_index_string}",
            fontsize=14,
        )

        self._overlay_depth_points(
            axes[0],
            image=image,
            points=gt_points,
            depth_values=gt_depth_values,
            ylabel="GT points",
            count=len(gt_points),
            textbox_extra="color: GT depth",
        )

        self._overlay_depth_points(
            axes[1],
            image=image,
            points=thinning_points,
            depth_values=thinning_depths,
            ylabel="Sample thinning",
            count=thinning_stats["num_points"],
            textbox_extra="color: pred depth",
        )

        self._overlay_depth_points(
            axes[2],
            image=image,
            points=block_points,
            depth_values=block_depths,
            ylabel="Sample block",
            count=block_stats["num_points"],
            textbox_extra="color: pred depth",
        )

        self._imshow(
            axes[3],
            lambda_mass,
            "Pred λ mass",
            cmap="hot",
            fig=figure,
            colorbar=True,
        )
        self._textbox(
            axes[3],
            f"E[N]={expected_count:.2f}\n"
            f"GT={ground_truth_count:.0f}\n"
            f"Δ={expected_count - ground_truth_count:+.2f}",
        )

        self._imshow(
            axes[4],
            lambda_mass ** self.gamma,
            "Pred λ^γ",
            cmap="hot",
            fig=figure,
            colorbar=True,
        )
        self._textbox(
            axes[4],
            f"γ={self.gamma}",
        )

        self._imshow(
            axes[5],
            predicted_depth,
            "Pred depth [m]",
            cmap=self.depth_cmap,
            vmin=self.depth_vmin,
            vmax=self.depth_vmax,
            fig=figure,
            colorbar=True,
        )
        self._scatter_values(
            ax=axes[5],
            points=gt_points,
            values=gt_depth_values,
            cmap=self.depth_cmap,
            vmin=self.depth_vmin,
            vmax=self.depth_vmax,
            edge_color=self.depth_point_edge_color,
            edge_width=self.depth_point_edge_width,
        )
        self._textbox(
            axes[5],
            f"GT depth dots: N={len(gt_depth_values)}",
        )

        self._imshow(
            axes[6],
            predicted_velocity,
            predicted_velocity_label,
            cmap=self.vel_cmap,
            vmin=self.vel_vmin,
            vmax=self.vel_vmax,
            fig=figure,
            colorbar=True,
        )
        self._scatter_values(
            ax=axes[6],
            points=gt_points,
            values=gt_velocity_values,
            cmap=self.vel_cmap,
            vmin=self.vel_vmin,
            vmax=self.vel_vmax,
            edge_color=self.vel_point_edge_color,
            edge_width=self.vel_point_edge_width,
        )
        self._textbox(
            axes[6],
            f"mode={velocity_mode}\n"
            f"GT vel dots: N={len(gt_velocity_values)}",
        )

        if predicted_rcs is not None:
            self._imshow(
                axes[7],
                predicted_rcs,
                "Pred RCS [dBsm]",
                cmap=self.rcs_cmap,
                vmin=self.rcs_vmin,
                vmax=self.rcs_vmax,
                fig=figure,
                colorbar=True,
            )
            self._scatter_values(
                ax=axes[7],
                points=gt_points,
                values=gt_rcs_values,
                cmap=self.rcs_cmap,
                vmin=self.rcs_vmin,
                vmax=self.rcs_vmax,
                edge_color=self.rcs_point_edge_color,
                edge_width=self.rcs_point_edge_width,
            )
            self._textbox(
                axes[7],
                f"link={self.rcs_link}\n"
                f"GT RCS dots: N={len(gt_rcs_values)}",
            )
        else:
            axes[7].axis("off")
            self._label(axes[7], "Pred RCS")
            self._textbox(axes[7], "missing channel")

        output_path = self._output_path(
            runner=runner,
            phase=phase,
            video=video,
            frame_string=frame_string,
            global_index_string=global_index_string,
        )

        figure.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches="tight",
        )
        figure.clear()