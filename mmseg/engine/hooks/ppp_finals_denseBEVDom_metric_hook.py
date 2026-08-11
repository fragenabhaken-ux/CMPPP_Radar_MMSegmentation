"""
PPPPolarDenseMetricEvalExportHook
================================
Export dense polar-BEV PPP predictions and targets as fixed Cartesian-BEV
rasters for offline metric evaluation.

Each distributed rank writes one HDF5 file:

    {log_dir}/metric_eval_data_dense_bev_cartesian/{phase}/rank00.h5
    {log_dir}/metric_eval_data_dense_bev_cartesian/{phase}/rank01.h5
    ...

Dense polar prediction channels
-------------------------------

    pred[0] = log intensity density lambda(r, theta)
    pred[1] = raw velocity mark
    pred[2] = raw RCS mark

Dense polar target channels
---------------------------

    gt[0] = binary occupied-cell indicator
    gt[1] = GT velocity
    gt[2] = GT RCS

Cartesian output datasets
-------------------------

    /sample_000000/img             (H_img, W_img, 3) uint8
    /sample_000000/pred_lambda     (H_bev, W_bev)    float32
    /sample_000000/pred_velocity   (H_bev, W_bev)    float32
    /sample_000000/pred_rcs        (H_bev, W_bev)    float32
    /sample_000000/gt_support      (H_bev, W_bev)    float32
    /sample_000000/gt_velocity     (H_bev, W_bev)    float32
    /sample_000000/gt_rcs          (H_bev, W_bev)    float32

For exact GT preservation, the hook additionally stores:

    /sample_000000/gt_points_xy        (N, 2) float32
    /sample_000000/gt_point_velocity   (N,)   float32
    /sample_000000/gt_point_rcs        (N,)   float32

The columns of ``gt_points_xy`` are ``x_forward`` and ``y_lateral`` in metres.
These arrays avoid losing individual GT marks when multiple occupied polar
cells map to the same Cartesian raster cell.

Cartesian raster convention
---------------------------

The exported arrays have axis order:

    array[x_forward_index, y_lateral_index]

Rows run from ``bev_x_min`` to ``bev_x_max`` and columns run from
``bev_y_min`` to ``bev_y_max``. Coordinate bounds and resolutions are stored
as HDF5 attributes.

The polar prediction is pushed into Cartesian space as follows:

1. Decode the polar intensity density ``lambda(r, theta)``.
2. Convert each polar cell to expected-count mass
   ``lambda * delta_r * delta_theta``.
3. Map the polar cell centre to ``(x_forward, y_lateral)``.
4. Add the mass to the corresponding Cartesian raster cell.
5. Aggregate velocity and RCS using predicted PPP mass as weights.

This preserves total expected count for all polar cell centres that lie inside
the configured Cartesian bounds. Mass outside the Cartesian rectangle is
reported in sample attributes.

Usage in config
---------------

    custom_hooks = [
        dict(
            type='PPPPolarDenseMetricEvalExportHook',
            max_samples_test=10000,
            max_samples_val=0,
            img_mean=data_rbg_mean,
            img_std=data_rbg_std,
            r_min=0.0,
            r_max=104.0,
            theta_min=-1.5707963267948966,
            theta_max=1.5707963267948966,
            bev_x_min=0.0,
            bev_x_max=104.0,
            bev_y_min=-85.0,
            bev_y_max=85.0,
            bev_height=256,
            bev_width=256,
            velocity_mode='signed',
            velocity_link='cube',
            rcs_link='cube',
        )
    ]
"""

from __future__ import annotations

import os
from typing import Any

import h5py
import numpy as np
import torch
from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmseg.registry import HOOKS


def _weighted_mean_from_bins(
    flat_indices: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted means and total weights for flattened output bins."""
    weight_sum = np.bincount(
        flat_indices,
        weights=weights,
        minlength=num_bins,
    ).astype(np.float64, copy=False)

    value_sum = np.bincount(
        flat_indices,
        weights=weights * values,
        minlength=num_bins,
    ).astype(np.float64, copy=False)

    mean = np.zeros(num_bins, dtype=np.float64)
    nonzero = weight_sum > 0.0
    mean[nonzero] = value_sum[nonzero] / weight_sum[nonzero]
    return mean, weight_sum


@HOOKS.register_module()
class PPPPolarDenseMetricEvalExportHook(Hook):
    """Export dense polar PPP predictions in a fixed Cartesian BEV grid.

    Parameters
    ----------
    max_samples_test : int
        Maximum number of samples written per rank during testing.
    max_samples_val : int
        Maximum number of samples written per rank during validation. Set to
        zero to disable validation export.
    img_mean, img_std : sequence[float] | None
        Image-normalisation parameters used by the model data preprocessor.
    r_min, r_max : float
        Radial limits of the dense polar prediction domain in metres.
    theta_min, theta_max : float
        Angular limits of the dense polar prediction domain in radians.
    bev_x_min, bev_x_max : float | None
        Cartesian forward limits in metres. ``None`` uses ``r_min`` and
        ``r_max``, respectively.
    bev_y_min, bev_y_max : float
        Cartesian lateral limits in metres.
    bev_height, bev_width : int
        Number of Cartesian bins along forward x and lateral y.
    velocity_mode : {"signed", "absolute"}
        Whether exported velocity is signed or absolute.
    velocity_link : {"identity", "cube"}
        Inverse link applied to raw predicted velocity.
    rcs_link : {"identity", "cube"}
        Inverse link applied to raw predicted RCS.
    require_three_channels : bool
        Require prediction and GT tensors to contain all three dense polar
        channels.
    output_subdir : str
        Directory below ``runner.log_dir`` used for the HDF5 files.
    """

    def __init__(
        self,
        max_samples_test: int = 500,
        max_samples_val: int = 0,
        img_mean=None,
        img_std=None,
        r_min: float = 0.0,
        r_max: float = 104.0,
        theta_min: float = -1.5707963267948966,
        theta_max: float = 1.5707963267948966,
        bev_x_min: float | None = None,
        bev_x_max: float | None = None,
        bev_y_min: float = -85.0,
        bev_y_max: float = 85.0,
        bev_height: int = 256,
        bev_width: int = 256,
        velocity_mode: str = "signed",
        velocity_link: str = "cube",
        rcs_link: str = "cube",
        require_three_channels: bool = True,
        output_subdir: str = "metric_eval_data_dense_bev_cartesian",
    ) -> None:
        self.max_samples = {
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        self.bev_x_min = (
            self.r_min if bev_x_min is None else float(bev_x_min)
        )
        self.bev_x_max = (
            self.r_max if bev_x_max is None else float(bev_x_max)
        )
        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)

        self.velocity_mode = str(velocity_mode).lower()
        self.velocity_link = str(velocity_link).lower()
        self.rcs_link = str(rcs_link).lower()
        self.require_three_channels = bool(require_three_channels)
        self.output_subdir = str(output_subdir)

        self.img_mean = (
            np.asarray(img_mean, dtype=np.float32)[:, None, None]
            if img_mean is not None
            else None
        )
        self.img_std = (
            np.asarray(img_std, dtype=np.float32)[:, None, None]
            if img_std is not None
            else None
        )

        self._validate_configuration()

        self._written: dict[str, int] = {"val": 0, "test": 0}
        self._h5: dict[str, h5py.File | None] = {
            "val": None,
            "test": None,
        }

    # ------------------------------------------------------------------
    # Epoch callbacks
    # ------------------------------------------------------------------

    def before_val_epoch(self, runner) -> None:
        self._close_h5("val")
        self._written["val"] = 0

    def before_test_epoch(self, runner) -> None:
        self._close_h5("test")
        self._written["test"] = 0

    def after_val_epoch(self, runner, metrics=None, **kwargs) -> None:
        self._close_h5("val")

    def after_test_epoch(self, runner, metrics=None, **kwargs) -> None:
        self._close_h5("test")

    # ------------------------------------------------------------------
    # Iteration callbacks
    # ------------------------------------------------------------------

    def after_val_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        self._maybe_run(runner, data_batch, phase="val")

    def after_test_iter(
        self,
        runner,
        batch_idx,
        data_batch=None,
        outputs=None,
    ) -> None:
        self._maybe_run(runner, data_batch, phase="test")

    # ------------------------------------------------------------------
    # Core inference/export logic
    # ------------------------------------------------------------------

    def _maybe_run(self, runner, data_batch, phase: str) -> None:
        if not isinstance(data_batch, dict):
            return
        if self.max_samples[phase] <= 0:
            return
        if self._written[phase] >= self.max_samples[phase]:
            return

        model = (
            runner.model.module
            if is_model_wrapper(runner.model)
            else runner.model
        )
        was_training = model.training
        model.eval()

        try:
            proc = model.data_preprocessor(data_batch)
            inputs = proc["inputs"]
            data_samples = proc["data_samples"]

            with torch.no_grad():
                features = model.extract_feat(inputs)
                predictions = model.decode_head(features)

            if not torch.is_tensor(predictions):
                raise TypeError(
                    "PPPPolarDenseMetricEvalExportHook expects the decode "
                    f"head to return a tensor, got {type(predictions)}."
                )
            if predictions.ndim != 4:
                raise ValueError(
                    "Expected dense polar predictions with shape "
                    "[B, C, H_r, W_theta], got "
                    f"{tuple(predictions.shape)}."
                )
            if self.require_three_channels and predictions.shape[1] != 3:
                raise ValueError(
                    "Expected exactly three dense polar prediction channels "
                    "[log_lambda, raw_velocity, raw_rcs], got shape "
                    f"{tuple(predictions.shape)}."
                )
            if predictions.shape[1] < 3:
                raise ValueError(
                    "Dense polar predictions require at least three channels, "
                    f"got shape {tuple(predictions.shape)}."
                )

            remaining = self.max_samples[phase] - self._written[phase]
            n_save = min(
                remaining,
                int(predictions.shape[0]),
                len(data_samples),
            )

            h5f = self._get_h5(runner, phase)
            self._write_file_configuration(h5f)

            for i in range(n_save):
                self._append_sample(
                    h5f=h5f,
                    img_tensor=inputs[i],
                    pred_grid=predictions[i],
                    sample=data_samples[i],
                    local_index=self._written[phase] + i,
                )

            h5f.flush()
            self._written[phase] += n_save
        finally:
            if was_training:
                model.train()

    # ------------------------------------------------------------------
    # HDF5 management
    # ------------------------------------------------------------------

    def _get_out_dir(self, runner, phase: str) -> str:
        out_dir = os.path.join(
            runner.log_dir,
            self.output_subdir,
            phase,
        )
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _get_h5(self, runner, phase: str) -> h5py.File:
        if self._h5[phase] is None:
            out_dir = self._get_out_dir(runner, phase)
            rank = get_rank()
            filepath = os.path.join(out_dir, f"rank{rank:02d}.h5")

            h5f = h5py.File(filepath, "w")
            h5f.attrs["phase"] = phase
            h5f.attrs["rank"] = int(rank)
            h5f.attrs["format"] = (
                "ppp_polar_dense_metric_eval_cartesian_v1"
            )
            h5f.attrs["domain"] = "cartesian_bev"
            h5f.attrs["source_prediction_domain"] = "polar_bev"
            self._h5[phase] = h5f

        return self._h5[phase]

    def _close_h5(self, phase: str) -> None:
        h5f = self._h5[phase]
        if h5f is not None:
            try:
                h5f.close()
            except Exception:
                pass
            self._h5[phase] = None

    def _write_file_configuration(self, h5f: h5py.File) -> None:
        h5f.attrs["prediction_channel_convention"] = (
            "0=log_lambda_polar,1=raw_velocity,2=raw_rcs"
        )
        h5f.attrs["ground_truth_channel_convention"] = (
            "0=occupancy_polar,1=velocity,2=rcs"
        )
        h5f.attrs["output_dataset_convention"] = (
            "pred_lambda=expected_count_mass_per_cartesian_cell"
        )
        h5f.attrs["cartesian_axis_order"] = (
            "array[row=x_forward, column=y_lateral]"
        )
        h5f.attrs["polar_to_cartesian_method"] = (
            "polar_cell_centre_scatter_mass_conserving"
        )
        h5f.attrs["predicted_mark_aggregation"] = (
            "predicted_mass_weighted_mean"
        )
        h5f.attrs["ground_truth_mark_aggregation"] = (
            "point_count_weighted_mean"
        )
        h5f.attrs["gt_support_semantics"] = (
            "number_of_gt_points_per_cartesian_cell"
        )

        h5f.attrs["r_min"] = self.r_min
        h5f.attrs["r_max"] = self.r_max
        h5f.attrs["theta_min"] = self.theta_min
        h5f.attrs["theta_max"] = self.theta_max

        h5f.attrs["bev_x_min"] = self.bev_x_min
        h5f.attrs["bev_x_max"] = self.bev_x_max
        h5f.attrs["bev_y_min"] = self.bev_y_min
        h5f.attrs["bev_y_max"] = self.bev_y_max
        h5f.attrs["bev_height"] = self.bev_height
        h5f.attrs["bev_width"] = self.bev_width
        h5f.attrs["bev_dx"] = (
            (self.bev_x_max - self.bev_x_min) / self.bev_height
        )
        h5f.attrs["bev_dy"] = (
            (self.bev_y_max - self.bev_y_min) / self.bev_width
        )

        h5f.attrs["velocity_mode"] = self.velocity_mode
        h5f.attrs["velocity_link"] = self.velocity_link
        h5f.attrs["rcs_link"] = self.rcs_link

    # ------------------------------------------------------------------
    # Per-sample conversion and writing
    # ------------------------------------------------------------------

    def _append_sample(
        self,
        h5f: h5py.File,
        img_tensor: torch.Tensor,
        pred_grid: torch.Tensor,
        sample,
        local_index: int,
    ) -> None:
        pred_grid = pred_grid.detach().float().cpu()
        gt_grid = self._extract_gt(sample)

        if pred_grid.ndim != 3 or pred_grid.shape[0] < 3:
            raise ValueError(
                "Expected one dense polar prediction with shape [3, H, W], "
                f"got {tuple(pred_grid.shape)}."
            )
        if pred_grid.shape != gt_grid.shape:
            raise ValueError(
                "Prediction and GT polar grids must have identical shapes, "
                f"got prediction={tuple(pred_grid.shape)}, "
                f"GT={tuple(gt_grid.shape)}."
            )

        log_lambda = pred_grid[0].numpy().astype(np.float64, copy=False)
        raw_velocity = pred_grid[1]
        raw_rcs = pred_grid[2]

        if not np.all(np.isfinite(log_lambda)):
            raise ValueError("Predicted log-intensity contains NaN or Inf.")

        polar_lambda = np.exp(log_lambda)
        pred_velocity_polar = self._decode_velocity(raw_velocity)
        pred_rcs_polar = self._decode_rcs(raw_rcs)

        if not np.all(np.isfinite(pred_velocity_polar)):
            raise ValueError("Decoded predicted velocity contains NaN or Inf.")
        if not np.all(np.isfinite(pred_rcs_polar)):
            raise ValueError("Decoded predicted RCS contains NaN or Inf.")

        gt_occupancy = gt_grid[0].numpy().astype(np.float64, copy=False)
        gt_velocity_polar = gt_grid[1].numpy().astype(
            np.float64,
            copy=False,
        )
        gt_rcs_polar = gt_grid[2].numpy().astype(
            np.float64,
            copy=False,
        )

        unique_occupancy = np.unique(gt_occupancy)
        if not np.all(np.isin(unique_occupancy, [0.0, 1.0])):
            raise ValueError(
                "Dense polar GT occupancy must be binary, found "
                f"{unique_occupancy.tolist()}."
            )

        if self.velocity_mode == "absolute":
            gt_velocity_polar = np.abs(gt_velocity_polar)

        converted = self._polar_to_cartesian(
            polar_lambda=polar_lambda,
            pred_velocity=pred_velocity_polar,
            pred_rcs=pred_rcs_polar,
            gt_occupancy=gt_occupancy,
            gt_velocity=gt_velocity_polar,
            gt_rcs=gt_rcs_polar,
        )

        img = self._denorm_img(
            img_tensor.detach().cpu().numpy(),
        )

        meta = sample.metainfo
        grp = h5f.create_group(f"sample_{local_index:06d}")
        self._write_attrs(
            grp,
            meta=meta,
            local_index=local_index,
        )

        grp.attrs["polar_height"] = int(pred_grid.shape[1])
        grp.attrs["polar_width"] = int(pred_grid.shape[2])
        grp.attrs["polar_dr"] = float(converted["polar_dr"])
        grp.attrs["polar_dtheta"] = float(converted["polar_dtheta"])

        grp.attrs["pred_expected_count_polar"] = float(
            converted["pred_expected_count_polar"]
        )
        grp.attrs["pred_expected_count_cartesian"] = float(
            converted["pred_expected_count_cartesian"]
        )
        grp.attrs["pred_expected_count_outside_cartesian"] = float(
            converted["pred_expected_count_outside_cartesian"]
        )
        grp.attrs["gt_count_polar"] = int(converted["gt_count_polar"])
        grp.attrs["gt_count_cartesian"] = int(
            converted["gt_count_cartesian"]
        )
        grp.attrs["gt_count_outside_cartesian"] = int(
            converted["gt_count_outside_cartesian"]
        )
        grp.attrs["cartesian_cells_with_gt_collisions"] = int(
            converted["cartesian_cells_with_gt_collisions"]
        )
        grp.attrs["maximum_gt_points_in_one_cartesian_cell"] = int(
            converted["maximum_gt_points_in_one_cartesian_cell"]
        )

        self._create_dataset(grp, "img", img)
        self._create_dataset(
            grp,
            "pred_lambda",
            converted["pred_lambda"],
        )
        self._create_dataset(
            grp,
            "pred_velocity",
            converted["pred_velocity"],
        )
        self._create_dataset(
            grp,
            "pred_rcs",
            converted["pred_rcs"],
        )
        self._create_dataset(
            grp,
            "gt_support",
            converted["gt_support"],
        )
        self._create_dataset(
            grp,
            "gt_velocity",
            converted["gt_velocity"],
        )
        self._create_dataset(
            grp,
            "gt_rcs",
            converted["gt_rcs"],
        )

        # Exact GT point representation in physical Cartesian coordinates.
        self._create_dataset(
            grp,
            "gt_points_xy",
            converted["gt_points_xy"],
        )
        self._create_dataset(
            grp,
            "gt_point_velocity",
            converted["gt_point_velocity"],
        )
        self._create_dataset(
            grp,
            "gt_point_rcs",
            converted["gt_point_rcs"],
        )

    def _polar_to_cartesian(
        self,
        polar_lambda: np.ndarray,
        pred_velocity: np.ndarray,
        pred_rcs: np.ndarray,
        gt_occupancy: np.ndarray,
        gt_velocity: np.ndarray,
        gt_rcs: np.ndarray,
    ) -> dict[str, Any]:
        """Push dense polar fields into the configured Cartesian raster."""
        arrays = [
            pred_velocity,
            pred_rcs,
            gt_occupancy,
            gt_velocity,
            gt_rcs,
        ]
        if any(array.shape != polar_lambda.shape for array in arrays):
            raise ValueError(
                "All polar arrays must have the same shape, got "
                f"lambda={polar_lambda.shape}, "
                f"velocity={pred_velocity.shape}, "
                f"rcs={pred_rcs.shape}, "
                f"gt_occupancy={gt_occupancy.shape}, "
                f"gt_velocity={gt_velocity.shape}, "
                f"gt_rcs={gt_rcs.shape}."
            )

        h_r, w_theta = polar_lambda.shape
        dr = (self.r_max - self.r_min) / h_r
        dtheta = (self.theta_max - self.theta_min) / w_theta

        r_centres = (
            self.r_min
            + (np.arange(h_r, dtype=np.float64) + 0.5) * dr
        )
        theta_centres = (
            self.theta_min
            + (np.arange(w_theta, dtype=np.float64) + 0.5) * dtheta
        )
        rr, tt = np.meshgrid(
            r_centres,
            theta_centres,
            indexing="ij",
        )

        x_forward = rr * np.cos(tt)
        y_lateral = rr * np.sin(tt)

        dx = (self.bev_x_max - self.bev_x_min) / self.bev_height
        dy = (self.bev_y_max - self.bev_y_min) / self.bev_width

        x_index = np.floor(
            (x_forward - self.bev_x_min) / dx
        ).astype(np.int64)
        y_index = np.floor(
            (y_lateral - self.bev_y_min) / dy
        ).astype(np.int64)

        inside = (
            (x_index >= 0)
            & (x_index < self.bev_height)
            & (y_index >= 0)
            & (y_index < self.bev_width)
        )

        flat_output_size = self.bev_height * self.bev_width
        flat_indices = (
            x_index[inside] * self.bev_width + y_index[inside]
        )

        # Predicted mass in each polar cell. The dense head predicts intensity
        # with respect to dr dtheta, matching the training visualization hook.
        polar_mass = polar_lambda * dr * dtheta
        pred_mass_inside = polar_mass[inside]

        pred_velocity_flat, pred_mass_flat = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=pred_velocity[inside],
            weights=pred_mass_inside,
            num_bins=flat_output_size,
        )
        pred_rcs_flat, pred_mass_flat_rcs = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=pred_rcs[inside],
            weights=pred_mass_inside,
            num_bins=flat_output_size,
        )

        if not np.allclose(
            pred_mass_flat,
            pred_mass_flat_rcs,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Internal error: velocity and RCS mass accumulators differ."
            )

        # Ground-truth count raster. It may exceed one because multiple polar
        # occupied cells can map to the same Cartesian raster cell.
        occupied = gt_occupancy > 0.0
        occupied_inside = occupied & inside
        gt_flat_indices = (
            x_index[occupied_inside] * self.bev_width
            + y_index[occupied_inside]
        )
        gt_weights = gt_occupancy[occupied_inside]

        gt_velocity_flat, gt_count_flat = _weighted_mean_from_bins(
            flat_indices=gt_flat_indices,
            values=gt_velocity[occupied_inside],
            weights=gt_weights,
            num_bins=flat_output_size,
        )
        gt_rcs_flat, gt_count_flat_rcs = _weighted_mean_from_bins(
            flat_indices=gt_flat_indices,
            values=gt_rcs[occupied_inside],
            weights=gt_weights,
            num_bins=flat_output_size,
        )

        if not np.array_equal(gt_count_flat, gt_count_flat_rcs):
            raise RuntimeError(
                "Internal error: GT velocity and RCS count accumulators differ."
            )

        output_shape = (self.bev_height, self.bev_width)
        pred_lambda_cart = pred_mass_flat.reshape(output_shape)
        pred_velocity_cart = pred_velocity_flat.reshape(output_shape)
        pred_rcs_cart = pred_rcs_flat.reshape(output_shape)
        gt_support_cart = gt_count_flat.reshape(output_shape)
        gt_velocity_cart = gt_velocity_flat.reshape(output_shape)
        gt_rcs_cart = gt_rcs_flat.reshape(output_shape)

        gt_points_xy = np.stack(
            [
                x_forward[occupied_inside],
                y_lateral[occupied_inside],
            ],
            axis=1,
        )

        gt_count_polar = int(occupied.sum())
        gt_count_cartesian = int(occupied_inside.sum())
        pred_expected_count_polar = float(polar_mass.sum())
        pred_expected_count_cartesian = float(pred_lambda_cart.sum())

        collision_cells = gt_support_cart > 1.0
        max_points_one_cell = (
            int(gt_support_cart.max())
            if gt_support_cart.size > 0
            else 0
        )

        return {
            "pred_lambda": pred_lambda_cart.astype(
                np.float32,
                copy=False,
            ),
            "pred_velocity": pred_velocity_cart.astype(
                np.float32,
                copy=False,
            ),
            "pred_rcs": pred_rcs_cart.astype(
                np.float32,
                copy=False,
            ),
            "gt_support": gt_support_cart.astype(
                np.float32,
                copy=False,
            ),
            "gt_velocity": gt_velocity_cart.astype(
                np.float32,
                copy=False,
            ),
            "gt_rcs": gt_rcs_cart.astype(
                np.float32,
                copy=False,
            ),
            "gt_points_xy": gt_points_xy.astype(
                np.float32,
                copy=False,
            ),
            "gt_point_velocity": gt_velocity[occupied_inside].astype(
                np.float32,
                copy=False,
            ),
            "gt_point_rcs": gt_rcs[occupied_inside].astype(
                np.float32,
                copy=False,
            ),
            "polar_dr": float(dr),
            "polar_dtheta": float(dtheta),
            "pred_expected_count_polar": pred_expected_count_polar,
            "pred_expected_count_cartesian": (
                pred_expected_count_cartesian
            ),
            "pred_expected_count_outside_cartesian": float(
                pred_expected_count_polar
                - pred_expected_count_cartesian
            ),
            "gt_count_polar": gt_count_polar,
            "gt_count_cartesian": gt_count_cartesian,
            "gt_count_outside_cartesian": (
                gt_count_polar - gt_count_cartesian
            ),
            "cartesian_cells_with_gt_collisions": int(
                collision_cells.sum()
            ),
            "maximum_gt_points_in_one_cartesian_cell": (
                max_points_one_cell
            ),
        }

    # ------------------------------------------------------------------
    # Decoding, validation, and utilities
    # ------------------------------------------------------------------

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

        return velocity.numpy().astype(np.float64, copy=False)

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        raw_rcs = raw_rcs.detach().float().cpu()

        if self.rcs_link == "identity":
            rcs = raw_rcs
        elif self.rcs_link == "cube":
            rcs = raw_rcs.pow(3)
        else:
            raise RuntimeError(
                f"Unsupported rcs_link={self.rcs_link!r}."
            )

        return rcs.numpy().astype(np.float64, copy=False)

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

    def _validate_configuration(self) -> None:
        if self.r_min >= self.r_max:
            raise ValueError(
                f"r_min must be smaller than r_max, got "
                f"{self.r_min} >= {self.r_max}."
            )
        if self.theta_min >= self.theta_max:
            raise ValueError(
                "theta_min must be smaller than theta_max, got "
                f"{self.theta_min} >= {self.theta_max}."
            )
        if self.bev_x_min >= self.bev_x_max:
            raise ValueError(
                "bev_x_min must be smaller than bev_x_max, got "
                f"{self.bev_x_min} >= {self.bev_x_max}."
            )
        if self.bev_y_min >= self.bev_y_max:
            raise ValueError(
                "bev_y_min must be smaller than bev_y_max, got "
                f"{self.bev_y_min} >= {self.bev_y_max}."
            )
        if self.bev_height <= 0 or self.bev_width <= 0:
            raise ValueError(
                "bev_height and bev_width must be positive, got "
                f"{self.bev_height} x {self.bev_width}."
            )
        if self.velocity_mode not in {"signed", "absolute"}:
            raise ValueError(
                "velocity_mode must be 'signed' or 'absolute', got "
                f"{self.velocity_mode!r}."
            )
        if self.velocity_link not in {"identity", "cube"}:
            raise ValueError(
                "velocity_link must be 'identity' or 'cube', got "
                f"{self.velocity_link!r}."
            )
        if self.rcs_link not in {"identity", "cube"}:
            raise ValueError(
                "rcs_link must be 'identity' or 'cube', got "
                f"{self.rcs_link!r}."
            )
        if not self.output_subdir:
            raise ValueError("output_subdir must not be empty.")

    @staticmethod
    def _create_dataset(
        grp: h5py.Group,
        name: str,
        data: np.ndarray,
    ) -> None:
        grp.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=4,
        )

    def _write_attrs(
        self,
        grp: h5py.Group,
        meta: dict[str, Any],
        local_index: int,
    ) -> None:
        grp.attrs["local_index"] = int(local_index)
        grp.attrs["rank"] = int(get_rank())

        for key in [
            "sample_id",
            "frame_idx",
            "global_idx",
            "video_path",
            "video_name",
            "video_frame_idx",
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "doppler_period",
            "ego_longitudinal_velocity",
            "ego_yaw_rate",
        ]:
            if key in meta and meta[key] is not None:
                grp.attrs[key] = self._attr_value(meta[key])

    @staticmethod
    def _attr_value(value: Any):
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                np.integer,
                np.floating,
                np.bool_,
            ),
        ):
            return value

        if isinstance(value, (tuple, list)) and all(
            isinstance(
                element,
                (
                    int,
                    float,
                    bool,
                    np.integer,
                    np.floating,
                    np.bool_,
                ),
            )
            for element in value
        ):
            return np.asarray(value)

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.dtype.kind in {"b", "i", "u", "f", "S", "U"}:
                return value

        return str(value)

    def _denorm_img(self, x: np.ndarray) -> np.ndarray:
        """Convert a (C, H, W) float image to (H, W, C) uint8."""
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