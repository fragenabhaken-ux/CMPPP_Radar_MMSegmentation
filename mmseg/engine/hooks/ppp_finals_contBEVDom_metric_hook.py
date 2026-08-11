"""
PPPPolarLIIFMetricEvalExportHook
===============================
Export polar-LIIF PPP predictions and continuous radar targets as fixed
Cartesian-BEV rasters for offline metric evaluation.

Each distributed rank writes one HDF5 file:

    {log_dir}/metric_eval_data_dense_bev_cartesian/{phase}/rank00.h5
    {log_dir}/metric_eval_data_dense_bev_cartesian/{phase}/rank01.h5
    ...

Prediction representation
-------------------------

The LIIF decode head returns a dictionary containing dense evaluation grids:

    preds["grid_log_lambda"]  = log intensity density lambda(r, theta)
    predicted velocity grid   = raw or already decoded velocity
    predicted RCS grid        = raw or already decoded RCS

The accepted velocity and RCS keys are the same as in the current LIIF
visualization hooks.

Continuous ground truth
-----------------------

The target is not a dense grid. It is read from:

    sample.gt_radar_points.xy      [:, 0:2] = (x_forward, y_lateral)
    sample.gt_radar_points.polar   [:, 2]   = radial velocity
    sample.gt_radar_points.meta    [:, 2]   = RCS

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
Only points inside the configured Cartesian evaluation rectangle are stored in
these three exact-point datasets. Counts of invalid and outside points are
stored as sample attributes.

Cartesian raster convention
---------------------------

The exported arrays have axis order:

    array[x_forward_index, y_lateral_index]

Rows run from ``bev_x_min`` to ``bev_x_max`` and columns run from
``bev_y_min`` to ``bev_y_max``.

The prediction is converted as follows:

1. Decode the dense polar intensity density ``lambda(r, theta)``.
2. Convert every polar evaluation cell to expected-count mass
   ``lambda * delta_r * delta_theta``.
3. Map the polar cell centre to ``(x_forward, y_lateral)``.
4. Add its mass to the corresponding Cartesian raster cell.
5. Aggregate predicted velocity and RCS with predicted PPP mass as weights.

The continuous GT points are rasterized directly from their exact Cartesian
coordinates. ``gt_support`` therefore stores the number of GT points per
Cartesian cell and may exceed one. GT velocity and RCS are point-count-weighted
means in cells containing multiple points.

Usage in config
---------------

    custom_hooks = [
        dict(
            type='PPPPolarLIIFMetricEvalExportHook',
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
            velocity_grid_is_raw=True,
            rcs_grid_is_raw=True,
            rcs_link='cube',
            output_subdir='metric_eval_data_dense_bev_cartesian',
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
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if not (len(flat_indices) == len(values) == len(weights)):
        raise ValueError(
            "flat_indices, values, and weights must have identical lengths, "
            f"got {len(flat_indices)}, {len(values)}, and {len(weights)}."
        )

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
class PPPPolarLIIFMetricEvalExportHook(Hook):
    """Export LIIF polar predictions and continuous GT in Cartesian BEV.

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
        Radial limits of the dense polar prediction grid in metres.
    theta_min, theta_max : float
        Angular limits of the dense polar prediction grid in radians.
    bev_x_min, bev_x_max : float | None
        Cartesian forward limits in metres. ``None`` uses ``r_min`` and
        ``r_max``, respectively.
    bev_y_min, bev_y_max : float
        Cartesian lateral limits in metres.
    bev_height, bev_width : int
        Number of Cartesian bins along forward x and lateral y.
    velocity_mode : {"signed", "absolute"}
        Whether exported velocity and GT velocity are signed or absolute.
    velocity_grid_is_raw : bool
        If true, decode signed velocity with a cube link and absolute velocity
        with ``abs``. If false, the returned grid is treated as already decoded.
    rcs_grid_is_raw : bool
        If true, apply ``rcs_link``. If false, the returned RCS grid is treated
        as already decoded.
    rcs_link : {"identity", "cube", "linear_cube"}
        Inverse link applied to a raw predicted RCS grid.
    output_subdir : str
        Directory below ``runner.log_dir`` used for the HDF5 files. Use the
        same value as the other Cartesian-BEV export hooks.
    """

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
        velocity_grid_is_raw: bool = True,
        rcs_grid_is_raw: bool = True,
        rcs_link: str = "cube",
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
        self.velocity_grid_is_raw = bool(velocity_grid_is_raw)
        self.rcs_grid_is_raw = bool(rcs_grid_is_raw)
        self.rcs_link = str(rcs_link).lower()
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

            if not isinstance(predictions, dict):
                raise TypeError(
                    "PPPPolarLIIFMetricEvalExportHook expects the decode "
                    f"head to return a dictionary, got {type(predictions)}."
                )

            grid_log_lambda = predictions.get("grid_log_lambda")
            if not torch.is_tensor(grid_log_lambda):
                raise KeyError(
                    "Missing tensor predictions['grid_log_lambda']. "
                    f"Available keys: {sorted(predictions.keys())}."
                )

            grid_velocity = self._extract_velocity_grid(predictions)
            grid_rcs = self._extract_rcs_grid(predictions)

            batch_size = self._grid_batch_size(grid_log_lambda)
            if self._grid_batch_size(grid_velocity) != batch_size:
                raise ValueError(
                    "Velocity-grid batch size differs from intensity-grid "
                    f"batch size: {self._grid_batch_size(grid_velocity)} "
                    f"versus {batch_size}."
                )
            if self._grid_batch_size(grid_rcs) != batch_size:
                raise ValueError(
                    "RCS-grid batch size differs from intensity-grid batch "
                    f"size: {self._grid_batch_size(grid_rcs)} versus "
                    f"{batch_size}."
                )

            remaining = self.max_samples[phase] - self._written[phase]
            n_save = min(
                remaining,
                batch_size,
                len(data_samples),
            )

            h5f = self._get_h5(runner, phase)
            self._write_file_configuration(h5f)

            for i in range(n_save):
                self._append_sample(
                    h5f=h5f,
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
                    local_index=self._written[phase] + i,
                )

            h5f.flush()
            self._written[phase] += n_save
        finally:
            if was_training:
                model.train()

    # ------------------------------------------------------------------
    # Prediction-grid extraction
    # ------------------------------------------------------------------

    def _extract_velocity_grid(self, predictions: dict) -> torch.Tensor:
        for key in self._VELOCITY_KEYS:
            value = predictions.get(key)
            if torch.is_tensor(value):
                return value

        raise KeyError(
            "Could not find the predicted velocity grid. "
            f"Tried {self._VELOCITY_KEYS}. "
            f"Available keys: {sorted(predictions.keys())}."
        )

    def _extract_rcs_grid(self, predictions: dict) -> torch.Tensor:
        keys = (
            self._RCS_RAW_KEYS
            if self.rcs_grid_is_raw
            else self._RCS_DECODED_KEYS
        )

        for key in keys:
            value = predictions.get(key)
            if torch.is_tensor(value):
                return value

        expected = "raw" if self.rcs_grid_is_raw else "decoded"
        raise KeyError(
            f"Could not find the predicted {expected} RCS grid. "
            f"Tried {keys}. Available keys: "
            f"{sorted(predictions.keys())}."
        )

    @staticmethod
    def _grid_batch_size(tensor: torch.Tensor) -> int:
        if tensor.ndim not in {3, 4}:
            raise ValueError(
                "Expected a grid with shape (B, H, W) or (B, 1, H, W), "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.ndim == 4 and tensor.shape[1] != 1:
            raise ValueError(
                "Expected singleton grid channel dimension, got "
                f"{tuple(tensor.shape)}."
            )
        return int(tensor.shape[0])

    @staticmethod
    def _select_grid(
        tensor: torch.Tensor,
        batch_index: int,
    ) -> torch.Tensor:
        if tensor.ndim == 4:
            if tensor.shape[1] != 1:
                raise ValueError(
                    "Expected grid shape (B, 1, H, W), got "
                    f"{tuple(tensor.shape)}."
                )
            return tensor[batch_index, 0]

        if tensor.ndim == 3:
            return tensor[batch_index]

        raise ValueError(
            "Expected a grid with shape (B, 1, H, W) or (B, H, W), "
            f"got {tuple(tensor.shape)}."
        )

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
                "ppp_polar_liif_metric_eval_cartesian_v1"
            )
            h5f.attrs["domain"] = "cartesian_bev"
            h5f.attrs["source_prediction_domain"] = "polar_liif_grid"
            h5f.attrs["source_ground_truth_domain"] = (
                "continuous_cartesian_radar_points"
            )
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
        h5f.attrs["prediction_output_convention"] = (
            "dict:grid_log_lambda+velocity_grid+rcs_grid"
        )
        h5f.attrs["ground_truth_convention"] = (
            "gt_radar_points.xy[:,0:2]=x_forward,y_lateral;"
            "polar[:,2]=velocity;meta[:,2]=rcs"
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
        h5f.attrs["velocity_grid_is_raw"] = self.velocity_grid_is_raw
        h5f.attrs["rcs_grid_is_raw"] = self.rcs_grid_is_raw
        h5f.attrs["rcs_link"] = self.rcs_link

    # ------------------------------------------------------------------
    # Per-sample conversion and writing
    # ------------------------------------------------------------------

    def _append_sample(
        self,
        h5f: h5py.File,
        img_tensor: torch.Tensor,
        grid_log_lambda: torch.Tensor,
        grid_velocity: torch.Tensor,
        grid_rcs: torch.Tensor,
        sample,
        local_index: int,
    ) -> None:
        log_lambda = self._as_2d_cpu_grid(
            grid_log_lambda,
            name="grid_log_lambda",
        )
        raw_or_decoded_velocity = self._as_2d_cpu_grid(
            grid_velocity,
            name="grid_velocity",
        )
        raw_or_decoded_rcs = self._as_2d_cpu_grid(
            grid_rcs,
            name="grid_rcs",
        )

        if raw_or_decoded_velocity.shape != log_lambda.shape:
            raise ValueError(
                "Velocity and intensity grids must have identical shapes, "
                f"got velocity={tuple(raw_or_decoded_velocity.shape)} and "
                f"intensity={tuple(log_lambda.shape)}."
            )
        if raw_or_decoded_rcs.shape != log_lambda.shape:
            raise ValueError(
                "RCS and intensity grids must have identical shapes, got "
                f"RCS={tuple(raw_or_decoded_rcs.shape)} and "
                f"intensity={tuple(log_lambda.shape)}."
            )

        log_lambda_np = log_lambda.numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(log_lambda_np)):
            raise ValueError("Predicted grid_log_lambda contains NaN or Inf.")

        polar_lambda = np.exp(log_lambda_np)
        if not np.all(np.isfinite(polar_lambda)):
            raise ValueError("Decoded predicted polar intensity contains NaN or Inf.")

        pred_velocity_polar = self._decode_velocity(
            raw_or_decoded_velocity
        )
        pred_rcs_polar = self._decode_rcs(
            raw_or_decoded_rcs
        )

        if not np.all(np.isfinite(pred_velocity_polar)):
            raise ValueError("Decoded predicted velocity contains NaN or Inf.")
        if not np.all(np.isfinite(pred_rcs_polar)):
            raise ValueError("Decoded predicted RCS contains NaN or Inf.")

        gt = self._extract_continuous_gt(sample)
        if self.velocity_mode == "absolute":
            gt["velocity"] = np.abs(gt["velocity"])

        pred_cart = self._rasterize_polar_prediction(
            polar_lambda=polar_lambda,
            pred_velocity=pred_velocity_polar,
            pred_rcs=pred_rcs_polar,
        )
        gt_cart = self._rasterize_gt_points(
            x_forward=gt["x_forward"],
            y_lateral=gt["y_lateral"],
            velocity=gt["velocity"],
            rcs=gt["rcs"],
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

        grp.attrs["polar_height"] = int(log_lambda.shape[0])
        grp.attrs["polar_width"] = int(log_lambda.shape[1])
        grp.attrs["polar_dr"] = float(pred_cart["polar_dr"])
        grp.attrs["polar_dtheta"] = float(pred_cart["polar_dtheta"])

        grp.attrs["pred_expected_count_polar"] = float(
            pred_cart["total_mass"]
        )
        grp.attrs["pred_expected_count_cartesian"] = float(
            pred_cart["inside_mass"]
        )
        grp.attrs["pred_expected_count_outside_cartesian"] = float(
            pred_cart["outside_mass"]
        )

        # Both names are stored for compatibility and explicit semantics.
        grp.attrs["gt_count_continuous"] = int(gt_cart["total_count"])
        grp.attrs["gt_count_polar"] = int(gt_cart["total_count"])
        grp.attrs["gt_count_cartesian"] = int(gt_cart["inside_count"])
        grp.attrs["gt_count_invalid"] = int(gt_cart["invalid_count"])
        grp.attrs["gt_count_outside_cartesian"] = int(
            gt_cart["outside_count"]
        )
        grp.attrs["cartesian_cells_with_gt_collisions"] = int(
            gt_cart["collision_cells"]
        )
        grp.attrs["maximum_gt_points_in_one_cartesian_cell"] = int(
            gt_cart["max_points_one_cell"]
        )

        self._create_dataset(grp, "img", img)
        self._create_dataset(
            grp,
            "pred_lambda",
            pred_cart["mass"].astype(np.float32, copy=False),
        )
        self._create_dataset(
            grp,
            "pred_velocity",
            pred_cart["velocity"].astype(np.float32, copy=False),
        )
        self._create_dataset(
            grp,
            "pred_rcs",
            pred_cart["rcs"].astype(np.float32, copy=False),
        )
        self._create_dataset(
            grp,
            "gt_support",
            gt_cart["support"].astype(np.float32, copy=False),
        )
        self._create_dataset(
            grp,
            "gt_velocity",
            gt_cart["velocity"].astype(np.float32, copy=False),
        )
        self._create_dataset(
            grp,
            "gt_rcs",
            gt_cart["rcs"].astype(np.float32, copy=False),
        )

        self._create_dataset(
            grp,
            "gt_points_xy",
            gt_cart["points_xy"].astype(np.float32, copy=False),
        )
        self._create_dataset(
            grp,
            "gt_point_velocity",
            gt_cart["point_velocity"].astype(np.float32, copy=False),
        )
        self._create_dataset(
            grp,
            "gt_point_rcs",
            gt_cart["point_rcs"].astype(np.float32, copy=False),
        )

    @staticmethod
    def _as_2d_cpu_grid(
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected {name} to be a tensor, got {type(tensor)}.")

        tensor = tensor.detach().float().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected {name} with shape (H, W), got "
                f"{tuple(tensor.shape)}."
            )
        return tensor

    def _rasterize_polar_prediction(
        self,
        polar_lambda: np.ndarray,
        pred_velocity: np.ndarray,
        pred_rcs: np.ndarray,
    ) -> dict[str, Any]:
        """Push a dense polar prediction into the Cartesian evaluation grid."""
        polar_lambda = np.asarray(polar_lambda, dtype=np.float64)
        pred_velocity = np.asarray(pred_velocity, dtype=np.float64)
        pred_rcs = np.asarray(pred_rcs, dtype=np.float64)

        if polar_lambda.ndim != 2:
            raise ValueError(
                "polar_lambda must be a two-dimensional grid, got "
                f"{polar_lambda.shape}."
            )
        if pred_velocity.shape != polar_lambda.shape:
            raise ValueError(
                "pred_velocity shape does not match polar_lambda: "
                f"{pred_velocity.shape} versus {polar_lambda.shape}."
            )
        if pred_rcs.shape != polar_lambda.shape:
            raise ValueError(
                "pred_rcs shape does not match polar_lambda: "
                f"{pred_rcs.shape} versus {polar_lambda.shape}."
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
        polar_mass = polar_lambda * dr * dtheta

        stable = (
            np.isfinite(x_forward)
            & np.isfinite(y_lateral)
            & np.isfinite(polar_mass)
            & (polar_mass >= 0.0)
            & np.isfinite(pred_velocity)
            & np.isfinite(pred_rcs)
        )
        if not np.all(stable):
            raise ValueError(
                "Polar prediction contains invalid coordinates, mass, or marks."
            )

        inside, flat_indices = self._cartesian_indices(
            x_forward=x_forward,
            y_lateral=y_lateral,
            stable=stable,
        )

        flat_output_size = self.bev_height * self.bev_width
        inside_mass = polar_mass[inside]

        velocity_flat, mass_flat = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=pred_velocity[inside],
            weights=inside_mass,
            num_bins=flat_output_size,
        )
        rcs_flat, mass_flat_rcs = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=pred_rcs[inside],
            weights=inside_mass,
            num_bins=flat_output_size,
        )

        if not np.allclose(
            mass_flat,
            mass_flat_rcs,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Internal error: velocity and RCS mass accumulators differ."
            )

        output_shape = (self.bev_height, self.bev_width)
        total_mass = float(polar_mass.sum())
        inside_mass_total = float(mass_flat.sum())

        return {
            "mass": mass_flat.reshape(output_shape),
            "velocity": velocity_flat.reshape(output_shape),
            "rcs": rcs_flat.reshape(output_shape),
            "polar_dr": float(dr),
            "polar_dtheta": float(dtheta),
            "total_mass": total_mass,
            "inside_mass": inside_mass_total,
            "outside_mass": float(total_mass - inside_mass_total),
        }

    def _rasterize_gt_points(
        self,
        x_forward: np.ndarray,
        y_lateral: np.ndarray,
        velocity: np.ndarray,
        rcs: np.ndarray,
    ) -> dict[str, Any]:
        """Rasterize exact continuous GT points into Cartesian BEV cells."""
        arrays = [
            np.asarray(x_forward, dtype=np.float64).reshape(-1),
            np.asarray(y_lateral, dtype=np.float64).reshape(-1),
            np.asarray(velocity, dtype=np.float64).reshape(-1),
            np.asarray(rcs, dtype=np.float64).reshape(-1),
        ]
        lengths = [len(array) for array in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Continuous GT arrays must have identical lengths, got "
                f"{lengths}."
            )

        x_forward, y_lateral, velocity, rcs = arrays
        total_count = len(x_forward)

        stable = (
            np.isfinite(x_forward)
            & np.isfinite(y_lateral)
            & np.isfinite(velocity)
            & np.isfinite(rcs)
        )
        inside, flat_indices = self._cartesian_indices(
            x_forward=x_forward,
            y_lateral=y_lateral,
            stable=stable,
        )

        flat_output_size = self.bev_height * self.bev_width
        point_weights = np.ones(int(inside.sum()), dtype=np.float64)

        velocity_flat, count_flat = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=velocity[inside],
            weights=point_weights,
            num_bins=flat_output_size,
        )
        rcs_flat, count_flat_rcs = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=rcs[inside],
            weights=point_weights,
            num_bins=flat_output_size,
        )

        if not np.array_equal(count_flat, count_flat_rcs):
            raise RuntimeError(
                "Internal error: GT velocity and RCS count accumulators differ."
            )

        output_shape = (self.bev_height, self.bev_width)
        support = count_flat.reshape(output_shape)
        points_xy = (
            np.stack(
                [x_forward[inside], y_lateral[inside]],
                axis=1,
            )
            if np.any(inside)
            else np.zeros((0, 2), dtype=np.float64)
        )

        stable_count = int(stable.sum())
        inside_count = int(inside.sum())
        collision_cells = support > 1.0
        max_points_one_cell = (
            int(support.max()) if support.size > 0 else 0
        )

        return {
            "support": support,
            "velocity": velocity_flat.reshape(output_shape),
            "rcs": rcs_flat.reshape(output_shape),
            "points_xy": points_xy,
            "point_velocity": velocity[inside],
            "point_rcs": rcs[inside],
            "total_count": int(total_count),
            "inside_count": inside_count,
            "invalid_count": int(total_count - stable_count),
            "outside_count": int(stable_count - inside_count),
            "collision_cells": int(collision_cells.sum()),
            "max_points_one_cell": max_points_one_cell,
        }

    def _cartesian_indices(
        self,
        x_forward: np.ndarray,
        y_lateral: np.ndarray,
        stable: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return inside mask and flattened Cartesian-grid indices."""
        x_forward = np.asarray(x_forward, dtype=np.float64)
        y_lateral = np.asarray(y_lateral, dtype=np.float64)
        stable = np.asarray(stable, dtype=bool)

        if not (
            x_forward.shape == y_lateral.shape == stable.shape
        ):
            raise ValueError(
                "x_forward, y_lateral, and stable must have identical shapes, "
                f"got {x_forward.shape}, {y_lateral.shape}, and "
                f"{stable.shape}."
            )

        dx = (self.bev_x_max - self.bev_x_min) / self.bev_height
        dy = (self.bev_y_max - self.bev_y_min) / self.bev_width

        x_index = np.full(x_forward.shape, -1, dtype=np.int64)
        y_index = np.full(y_lateral.shape, -1, dtype=np.int64)

        x_index[stable] = np.floor(
            (x_forward[stable] - self.bev_x_min) / dx
        ).astype(np.int64)
        y_index[stable] = np.floor(
            (y_lateral[stable] - self.bev_y_min) / dy
        ).astype(np.int64)

        inside = (
            stable
            & (x_index >= 0)
            & (x_index < self.bev_height)
            & (y_index >= 0)
            & (y_index < self.bev_width)
        )
        flat_indices = (
            x_index[inside] * self.bev_width + y_index[inside]
        )
        return inside, flat_indices

    # ------------------------------------------------------------------
    # Decoding, GT extraction, metadata, validation, utilities
    # ------------------------------------------------------------------

    def _decode_velocity(self, grid_velocity: torch.Tensor) -> np.ndarray:
        grid_velocity = grid_velocity.detach().float().cpu()

        if not self.velocity_grid_is_raw:
            velocity = grid_velocity
        elif self.velocity_mode == "signed":
            velocity = grid_velocity.pow(3)
        elif self.velocity_mode == "absolute":
            velocity = grid_velocity.abs()
        else:
            raise RuntimeError(
                f"Unsupported velocity_mode={self.velocity_mode!r}."
            )

        return velocity.numpy().astype(np.float64, copy=False)

    def _decode_rcs(self, grid_rcs: torch.Tensor) -> np.ndarray:
        grid_rcs = grid_rcs.detach().float().cpu()

        if not self.rcs_grid_is_raw:
            rcs = grid_rcs
        elif self.rcs_link == "identity":
            rcs = grid_rcs
        elif self.rcs_link == "cube":
            rcs = grid_rcs.pow(3)
        elif self.rcs_link == "linear_cube":
            rcs = grid_rcs + grid_rcs.pow(3)
        else:
            raise RuntimeError(
                f"Unsupported rcs_link={self.rcs_link!r}."
            )

        return rcs.numpy().astype(np.float64, copy=False)

    @staticmethod
    def _extract_continuous_gt(sample) -> dict[str, np.ndarray]:
        if not hasattr(sample, "gt_radar_points"):
            raise AttributeError(
                "Data sample has no gt_radar_points field."
            )

        radar_points = sample.gt_radar_points
        for field in ["xy", "polar", "meta"]:
            if not hasattr(radar_points, field):
                raise AttributeError(
                    f"sample.gt_radar_points has no {field!r} field."
                )

        xy = radar_points.xy
        polar = radar_points.polar
        meta_points = radar_points.meta

        for name, tensor in [
            ("xy", xy),
            ("polar", polar),
            ("meta", meta_points),
        ]:
            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"Expected gt_radar_points.{name} to be a tensor, "
                    f"got {type(tensor)}."
                )

        xy = xy.detach().float().cpu().numpy()
        polar = polar.detach().float().cpu().numpy()
        meta_points = meta_points.detach().float().cpu().numpy()

        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(
                "Expected gt_radar_points.xy with shape (N, >=2), got "
                f"{xy.shape}."
            )
        if polar.ndim != 2 or polar.shape[1] < 3:
            raise ValueError(
                "Expected gt_radar_points.polar with shape (N, >=3), got "
                f"{polar.shape}."
            )
        if meta_points.ndim != 2 or meta_points.shape[1] <= 2:
            raise ValueError(
                "Expected GT RCS in gt_radar_points.meta[:, 2], got "
                f"shape {meta_points.shape}."
            )

        counts = [xy.shape[0], polar.shape[0], meta_points.shape[0]]
        if len(set(counts)) != 1:
            raise ValueError(
                "GT Cartesian, polar, and metadata point counts differ: "
                f"xy={xy.shape}, polar={polar.shape}, "
                f"meta={meta_points.shape}."
            )

        return {
            "x_forward": xy[:, 0].astype(np.float64, copy=False),
            "y_lateral": xy[:, 1].astype(np.float64, copy=False),
            "velocity": polar[:, 2].astype(np.float64, copy=False),
            "rcs": meta_points[:, 2].astype(np.float64, copy=False),
        }

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
        if self.rcs_link not in {"identity", "cube", "linear_cube"}:
            raise ValueError(
                "rcs_link must be 'identity', 'cube', or 'linear_cube', "
                f"got {self.rcs_link!r}."
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