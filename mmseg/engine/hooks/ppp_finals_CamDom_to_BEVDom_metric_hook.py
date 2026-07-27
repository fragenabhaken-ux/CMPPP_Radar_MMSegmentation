"""
PPPCameraToCartesianBEVMetricEvalExportHook
===========================================
Export four-channel camera-domain marked-PPP predictions as fixed Cartesian
BEV rasters for offline evaluation against models that predict directly in
BEV space.

Each distributed rank writes one HDF5 file:

    {log_dir}/metric_eval_data_dense_bev_cartesian/{phase}/rank00.h5
    {log_dir}/metric_eval_data_dense_bev_cartesian/{phase}/rank01.h5
    ...

Camera-domain prediction channels
---------------------------------

    pred[0] = log intensity parameter
    pred[1] = raw forward-depth mark, decoded with exp
    pred[2] = raw velocity mark
    pred[3] = raw RCS mark

Camera-domain target channels
-----------------------------

    gt[0] = binary support mask
    gt[1] = forward depth x in metres
    gt[2] = velocity
    gt[3] = RCS

Cartesian output datasets
-------------------------

    /sample_000000/img             (H_img, W_img, 3) uint8
    /sample_000000/pred_lambda     (H_bev, W_bev)    float32
    /sample_000000/pred_velocity   (H_bev, W_bev)    float32
    /sample_000000/pred_rcs        (H_bev, W_bev)    float32
    /sample_000000/gt_support      (H_bev, W_bev)    float32
    /sample_000000/gt_velocity     (H_bev, W_bev)    float32
    /sample_000000/gt_rcs          (H_bev, W_bev)    float32

For exact preservation of the projected GT points, the hook additionally
stores:

    /sample_000000/gt_points_xy        (N, 2) float32
    /sample_000000/gt_point_velocity   (N,)   float32
    /sample_000000/gt_point_rcs        (N,)   float32

The columns of ``gt_points_xy`` are ``x_forward`` and ``y_lateral`` in metres.

Cartesian raster convention
---------------------------

The exported arrays have axis order:

    array[x_forward_index, y_lateral_index]

Rows run from ``bev_x_min`` to ``bev_x_max`` and columns run from
``bev_y_min`` to ``bev_y_max``. Use exactly the same bounds and dimensions as
for ``PPPPolarDenseMetricEvalExportHook``.

Camera-to-BEV push-forward
--------------------------

For each camera prediction cell:

1. Decode image-cell expected-count mass as
   ``exp(pred[0]) / (original_image_height * original_image_width)``.
2. Decode forward depth ``x = exp(pred[1])``.
3. Convert the horizontal camera coordinate and forward depth to lateral
   position using the frame-specific ``rad2cam`` matrix.
4. Add the image-cell mass to the corresponding Cartesian BEV cell.
5. Aggregate velocity and RCS by predicted-mass-weighted means.

This is deterministic. It does not sample a point cloud from the image-domain
PPP. Total expected count is preserved for all projected image cells whose
back-projected centres lie inside the configured Cartesian rectangle.

Usage in config
---------------

    custom_hooks = [
        dict(
            type='PPPCameraToCartesianBEVMetricEvalExportHook',
            max_samples_test=10000,
            max_samples_val=0,
            img_mean=data_rbg_mean,
            img_std=data_rbg_std,
            h_fov_deg=105.0,       # fallback; sample metainfo fovs is preferred
            principal_x_rel=0.491,
            lateral_sign=1.0,
            bev_x_min=0.0,
            bev_x_max=104.0,
            bev_y_min=-85.0,
            bev_y_max=85.0,
            bev_height=256,
            bev_width=256,
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


def recover_lateral_from_forward_depth(
    u_px: np.ndarray,
    forward_m: np.ndarray,
    rad2cam: np.ndarray,
    image_width: int,
    h_fov_deg: float,
    principal_x_rel: float,
) -> np.ndarray:
    """Recover radar lateral y from horizontal image coordinate and forward x.

    The target construction projects a ground-plane radar point ``(x, y, 0)``
    into the virtual pinhole camera using the frame-specific ``rad2cam``
    matrix. This function algebraically inverts the horizontal projection when
    x is known from the depth mark.
    """
    u_px = np.asarray(u_px, dtype=np.float64)
    forward_m = np.asarray(forward_m, dtype=np.float64)
    rad2cam = np.asarray(rad2cam, dtype=np.float64)

    if rad2cam.shape != (3, 4):
        raise ValueError(
            f"Expected rad2cam with shape (3, 4), got {rad2cam.shape}."
        )

    width = float(max(int(image_width), 1))
    cx = float(principal_x_rel) * (width - 1.0)
    fx = (width / 2.0) / np.tan(
        np.deg2rad(float(h_fov_deg)) / 2.0
    )

    qx = (u_px - cx) / fx
    qx, forward_m = np.broadcast_arrays(qx, forward_m)

    a00 = rad2cam[0, 0]
    a01 = rad2cam[0, 1]
    a20 = rad2cam[2, 0]
    a21 = rad2cam[2, 1]
    t0 = rad2cam[0, 3]
    t2 = rad2cam[2, 3]

    numerator = (
        qx * (a20 * forward_m + t2)
        - (a00 * forward_m + t0)
    )
    denominator = a01 - qx * a21

    lateral = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=lateral,
        where=np.abs(denominator) > 1e-12,
    )
    return lateral


@HOOKS.register_module()
class PPPCameraToCartesianBEVMetricEvalExportHook(Hook):
    """Push camera-domain marked-PPP fields into a Cartesian BEV raster.

    Parameters
    ----------
    max_samples_test : int
        Maximum number of samples written per rank during testing.
    max_samples_val : int
        Maximum number of samples written per rank during validation. Set to
        zero to disable validation export.
    img_mean, img_std : sequence[float] | None
        Image-normalisation parameters used by the data preprocessor.
    h_fov_deg : float
        Fallback horizontal FOV of the virtual pinhole image. If the packed
        sample contains ``metainfo['fovs']``, its horizontal FOV is used.
    principal_x_rel : float
        Relative horizontal principal point used during target construction.
    lateral_sign : float
        Sign applied to recovered lateral coordinates. Use the same convention
        as the direct BEV representation. The current visualization uses 1.0.
    bev_x_min, bev_x_max : float
        Cartesian forward limits in metres.
    bev_y_min, bev_y_max : float
        Cartesian lateral limits in metres.
    bev_height, bev_width : int
        Number of bins along forward x and lateral y.
    rcs_link : {"identity", "cube", "linear_cube"}
        Inverse link applied to raw predicted RCS.
    require_four_channels : bool
        Require prediction and GT tensors to contain intensity/support, depth,
        velocity, and RCS.
    output_subdir : str
        Directory below ``runner.log_dir`` used for the HDF5 files. Its default
        matches the direct dense-BEV exporter.
    """

    def __init__(
        self,
        max_samples_test: int = 500,
        max_samples_val: int = 0,
        img_mean=None,
        img_std=None,
        h_fov_deg: float = 105.0,
        principal_x_rel: float = 0.491,
        lateral_sign: float = 1.0,
        bev_x_min: float = 0.0,
        bev_x_max: float = 104.0,
        bev_y_min: float = -85.0,
        bev_y_max: float = 85.0,
        bev_height: int = 256,
        bev_width: int = 256,
        rcs_link: str = "cube",
        require_four_channels: bool = True,
        output_subdir: str = "metric_eval_data_dense_bev_cartesian",
    ) -> None:
        self.max_samples = {
            "val": int(max_samples_val),
            "test": int(max_samples_test),
        }

        self.h_fov_deg = float(h_fov_deg)
        self.principal_x_rel = float(principal_x_rel)
        self.lateral_sign = float(lateral_sign)

        self.bev_x_min = float(bev_x_min)
        self.bev_x_max = float(bev_x_max)
        self.bev_y_min = float(bev_y_min)
        self.bev_y_max = float(bev_y_max)
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)

        self.rcs_link = str(rcs_link).lower()
        self.require_four_channels = bool(require_four_channels)
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

            proc = model.data_preprocessor(data_batch)
            inputs = proc["inputs"]
            data_samples = proc["data_samples"]

            with torch.no_grad():
                predictions = model.decode_head(model.extract_feat(inputs))

            if isinstance(predictions, (tuple, list)):
                if len(predictions) == 0:
                    raise ValueError("Decode head returned an empty sequence.")
                predictions = predictions[0]

            if not torch.is_tensor(predictions):
                raise TypeError(
                    "PPPCameraToCartesianBEVMetricEvalExportHook expects "
                    "the decode head to return a tensor, got "
                    f"{type(predictions)}."
                )
            if predictions.ndim != 4:
                raise ValueError(
                    "Expected image-domain predictions with shape "
                    "[B, C, H, W], got "
                    f"{tuple(predictions.shape)}."
                )

            minimum_channels = 4 if self.require_four_channels else 3
            if predictions.shape[1] < minimum_channels:
                raise ValueError(
                    f"Expected at least {minimum_channels} prediction "
                    "channels [log_lambda, depth, velocity, RCS], got "
                    f"shape {tuple(predictions.shape)}."
                )

            remaining = self.max_samples[phase] - self._written[phase]
            n_save = min(
                remaining,
                int(predictions.shape[0]),
                len(data_samples),
            )

            h5f = self._get_h5(runner, phase)
            self._write_file_configuration(
                h5f=h5f,
                velocity_mode=velocity_mode,
            )

            for i in range(n_save):
                self._append_sample(
                    h5f=h5f,
                    img_tensor=inputs[i],
                    pred_grid=predictions[i],
                    sample=data_samples[i],
                    local_index=self._written[phase] + i,
                    velocity_mode=velocity_mode,
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
                "ppp_camera_depth_metric_eval_cartesian_v1"
            )
            h5f.attrs["domain"] = "cartesian_bev"
            h5f.attrs["source_prediction_domain"] = "camera_image"
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

    def _write_file_configuration(
        self,
        h5f: h5py.File,
        velocity_mode: str,
    ) -> None:
        h5f.attrs["prediction_channel_convention"] = (
            "0=log_lambda_image,1=raw_forward_depth,"
            "2=raw_velocity,3=raw_rcs"
        )
        h5f.attrs["ground_truth_channel_convention"] = (
            "0=support_image,1=forward_depth,2=velocity,3=rcs"
        )
        h5f.attrs["output_dataset_convention"] = (
            "pred_lambda=expected_count_mass_per_cartesian_cell"
        )
        h5f.attrs["cartesian_axis_order"] = (
            "array[row=x_forward, column=y_lateral]"
        )
        h5f.attrs["camera_to_cartesian_method"] = (
            "image_cell_centre_forward_depth_backprojection"
        )
        h5f.attrs["predicted_mark_aggregation"] = (
            "predicted_mass_weighted_mean"
        )
        h5f.attrs["ground_truth_mark_aggregation"] = (
            "point_count_weighted_mean"
        )
        h5f.attrs["gt_support_semantics"] = (
            "number_of_projected_gt_points_per_cartesian_cell"
        )
        h5f.attrs["image_intensity_measure"] = (
            "exp(log_lambda)/(original_image_height*original_image_width)"
        )
        h5f.attrs["depth_link"] = "exp"
        h5f.attrs["velocity_mode"] = velocity_mode
        h5f.attrs["velocity_decoding"] = (
            "signed:raw^3; absolute:abs(raw)"
        )
        h5f.attrs["rcs_link"] = self.rcs_link

        h5f.attrs["fallback_h_fov_deg"] = self.h_fov_deg
        h5f.attrs["principal_x_rel"] = self.principal_x_rel
        h5f.attrs["lateral_sign"] = self.lateral_sign

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
        velocity_mode: str,
    ) -> None:
        pred_grid = pred_grid.detach().float().cpu()
        gt_grid = self._extract_gt(sample)

        minimum_channels = 4 if self.require_four_channels else 3
        if pred_grid.ndim != 3 or pred_grid.shape[0] < minimum_channels:
            raise ValueError(
                f"Expected one prediction with at least {minimum_channels} "
                "channels and shape [C, H, W], got "
                f"{tuple(pred_grid.shape)}."
            )
        if gt_grid.shape[0] < minimum_channels:
            raise ValueError(
                f"Expected GT with at least {minimum_channels} channels, "
                f"got {tuple(gt_grid.shape)}."
            )

        meta = sample.metainfo
        rad2cam = self._extract_rad2cam(meta)
        projection_height, projection_width = self._projection_shape(
            meta=meta,
            fallback_hw=gt_grid.shape[-2:],
        )
        h_fov_deg = self._sample_h_fov(meta)

        log_lambda = pred_grid[0].numpy().astype(np.float64, copy=False)
        raw_depth = pred_grid[1]
        raw_velocity = pred_grid[2]

        if not np.all(np.isfinite(log_lambda)):
            raise ValueError("Predicted log-intensity contains NaN or Inf.")

        pred_lambda_image = np.exp(log_lambda) / float(
            projection_height * projection_width
        )
        if not np.all(np.isfinite(pred_lambda_image)):
            raise ValueError(
                "Decoded predicted image-cell mass contains NaN or Inf."
            )

        pred_depth = torch.exp(raw_depth).numpy().astype(
            np.float64,
            copy=False,
        )
        pred_velocity = self._decode_velocity(
            raw_velocity=raw_velocity,
            velocity_mode=velocity_mode,
        )

        pred_rcs = None
        if pred_grid.shape[0] >= 4:
            pred_rcs = self._decode_rcs(pred_grid[3])
        elif self.require_four_channels:
            raise ValueError("Prediction channel 3 for RCS is missing.")
        else:
            pred_rcs = np.zeros_like(pred_lambda_image)

        for name, array in [
            ("predicted depth", pred_depth),
            ("predicted velocity", pred_velocity),
            ("predicted RCS", pred_rcs),
        ]:
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Decoded {name} contains NaN or Inf.")

        gt_support = gt_grid[0].numpy().astype(np.float64, copy=False)
        gt_depth = gt_grid[1].numpy().astype(np.float64, copy=False)
        gt_velocity = gt_grid[2].numpy().astype(np.float64, copy=False)

        if gt_grid.shape[0] >= 4:
            gt_rcs = gt_grid[3].numpy().astype(np.float64, copy=False)
        elif self.require_four_channels:
            raise ValueError("Ground-truth channel 3 for RCS is missing.")
        else:
            gt_rcs = np.zeros_like(gt_support)

        unique_support = np.unique(gt_support)
        if not np.all(np.isin(unique_support, [0.0, 1.0])):
            raise ValueError(
                "Camera-domain GT support must be binary, found "
                f"{unique_support.tolist()}."
            )

        if velocity_mode == "absolute":
            gt_velocity = np.abs(gt_velocity)

        converted = self._camera_to_cartesian(
            pred_lambda_image=pred_lambda_image,
            pred_depth=pred_depth,
            pred_velocity=pred_velocity,
            pred_rcs=pred_rcs,
            gt_support=gt_support,
            gt_depth=gt_depth,
            gt_velocity=gt_velocity,
            gt_rcs=gt_rcs,
            rad2cam=rad2cam,
            projection_height=projection_height,
            projection_width=projection_width,
            h_fov_deg=h_fov_deg,
        )

        img = self._denorm_img(
            img_tensor.detach().cpu().numpy(),
        )

        grp = h5f.create_group(f"sample_{local_index:06d}")
        self._write_attrs(
            grp=grp,
            meta=meta,
            local_index=local_index,
        )

        grp.attrs["projection_height"] = int(projection_height)
        grp.attrs["projection_width"] = int(projection_width)
        grp.attrs["prediction_height"] = int(pred_grid.shape[1])
        grp.attrs["prediction_width"] = int(pred_grid.shape[2])
        grp.attrs["ground_truth_height"] = int(gt_grid.shape[1])
        grp.attrs["ground_truth_width"] = int(gt_grid.shape[2])
        grp.attrs["projection_h_fov_deg"] = float(h_fov_deg)

        grp.attrs["pred_expected_count_image"] = float(
            converted["pred_expected_count_image"]
        )
        grp.attrs["pred_expected_count_cartesian"] = float(
            converted["pred_expected_count_cartesian"]
        )
        grp.attrs["pred_expected_count_invalid_projection"] = float(
            converted["pred_expected_count_invalid_projection"]
        )
        grp.attrs["pred_expected_count_outside_cartesian"] = float(
            converted["pred_expected_count_outside_cartesian"]
        )
        grp.attrs["gt_count_image"] = int(converted["gt_count_image"])
        grp.attrs["gt_count_cartesian"] = int(
            converted["gt_count_cartesian"]
        )
        grp.attrs["gt_count_invalid_projection"] = int(
            converted["gt_count_invalid_projection"]
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

    def _camera_to_cartesian(
        self,
        pred_lambda_image: np.ndarray,
        pred_depth: np.ndarray,
        pred_velocity: np.ndarray,
        pred_rcs: np.ndarray,
        gt_support: np.ndarray,
        gt_depth: np.ndarray,
        gt_velocity: np.ndarray,
        gt_rcs: np.ndarray,
        rad2cam: np.ndarray,
        projection_height: int,
        projection_width: int,
        h_fov_deg: float,
    ) -> dict[str, Any]:
        """Push image-domain prediction and GT fields into Cartesian BEV."""
        prediction_arrays = [pred_depth, pred_velocity, pred_rcs]
        if any(
            array.shape != pred_lambda_image.shape
            for array in prediction_arrays
        ):
            raise ValueError(
                "All predicted image fields must have the same shape, got "
                f"lambda={pred_lambda_image.shape}, "
                f"depth={pred_depth.shape}, "
                f"velocity={pred_velocity.shape}, "
                f"rcs={pred_rcs.shape}."
            )

        gt_arrays = [gt_depth, gt_velocity, gt_rcs]
        if any(array.shape != gt_support.shape for array in gt_arrays):
            raise ValueError(
                "All GT image fields must have the same shape, got "
                f"support={gt_support.shape}, depth={gt_depth.shape}, "
                f"velocity={gt_velocity.shape}, rcs={gt_rcs.shape}."
            )

        pred_h, pred_w = pred_lambda_image.shape
        gt_h, gt_w = gt_support.shape

        # Prediction cell centres expressed in the original projected-image
        # coordinate system. The vertical coordinate is not required because
        # forward depth and the horizontal projection determine ground-plane y.
        pred_u = (
            np.arange(pred_w, dtype=np.float64) + 0.5
        ) * (float(projection_width) / float(pred_w))
        pred_u_grid = np.broadcast_to(
            pred_u[None, :],
            (pred_h, pred_w),
        )

        pred_forward = pred_depth
        pred_lateral = recover_lateral_from_forward_depth(
            u_px=pred_u_grid,
            forward_m=pred_forward,
            rad2cam=rad2cam,
            image_width=projection_width,
            h_fov_deg=h_fov_deg,
            principal_x_rel=self.principal_x_rel,
        )
        pred_lateral = self.lateral_sign * pred_lateral

        pred_cart = self._rasterize_prediction(
            mass=pred_lambda_image,
            x_forward=pred_forward,
            y_lateral=pred_lateral,
            velocity=pred_velocity,
            rcs=pred_rcs,
        )

        # Only occupied image cells are GT points.
        occupied_y, occupied_x = np.where(gt_support > 0.0)
        gt_count_image = int(len(occupied_x))

        if gt_count_image > 0:
            gt_u = (
                occupied_x.astype(np.float64) + 0.5
            ) * (float(projection_width) / float(gt_w))
            gt_forward_points = gt_depth[occupied_y, occupied_x]
            gt_lateral_points = recover_lateral_from_forward_depth(
                u_px=gt_u,
                forward_m=gt_forward_points,
                rad2cam=rad2cam,
                image_width=projection_width,
                h_fov_deg=h_fov_deg,
                principal_x_rel=self.principal_x_rel,
            )
            gt_lateral_points = self.lateral_sign * gt_lateral_points
            gt_velocity_points = gt_velocity[occupied_y, occupied_x]
            gt_rcs_points = gt_rcs[occupied_y, occupied_x]
            gt_point_weights = gt_support[occupied_y, occupied_x]
        else:
            gt_forward_points = np.zeros((0,), dtype=np.float64)
            gt_lateral_points = np.zeros((0,), dtype=np.float64)
            gt_velocity_points = np.zeros((0,), dtype=np.float64)
            gt_rcs_points = np.zeros((0,), dtype=np.float64)
            gt_point_weights = np.zeros((0,), dtype=np.float64)

        gt_cart = self._rasterize_gt_points(
            x_forward=gt_forward_points,
            y_lateral=gt_lateral_points,
            velocity=gt_velocity_points,
            rcs=gt_rcs_points,
            point_weights=gt_point_weights,
        )

        return {
            "pred_lambda": pred_cart["mass"].astype(
                np.float32,
                copy=False,
            ),
            "pred_velocity": pred_cart["velocity"].astype(
                np.float32,
                copy=False,
            ),
            "pred_rcs": pred_cart["rcs"].astype(
                np.float32,
                copy=False,
            ),
            "gt_support": gt_cart["support"].astype(
                np.float32,
                copy=False,
            ),
            "gt_velocity": gt_cart["velocity"].astype(
                np.float32,
                copy=False,
            ),
            "gt_rcs": gt_cart["rcs"].astype(
                np.float32,
                copy=False,
            ),
            "gt_points_xy": gt_cart["points_xy"].astype(
                np.float32,
                copy=False,
            ),
            "gt_point_velocity": gt_cart["point_velocity"].astype(
                np.float32,
                copy=False,
            ),
            "gt_point_rcs": gt_cart["point_rcs"].astype(
                np.float32,
                copy=False,
            ),
            "pred_expected_count_image": float(
                pred_cart["total_mass"]
            ),
            "pred_expected_count_cartesian": float(
                pred_cart["inside_mass"]
            ),
            "pred_expected_count_invalid_projection": float(
                pred_cart["invalid_mass"]
            ),
            "pred_expected_count_outside_cartesian": float(
                pred_cart["outside_mass"]
            ),
            "gt_count_image": gt_count_image,
            "gt_count_cartesian": int(gt_cart["inside_count"]),
            "gt_count_invalid_projection": int(
                gt_cart["invalid_count"]
            ),
            "gt_count_outside_cartesian": int(
                gt_cart["outside_count"]
            ),
            "cartesian_cells_with_gt_collisions": int(
                gt_cart["collision_cells"]
            ),
            "maximum_gt_points_in_one_cartesian_cell": int(
                gt_cart["max_points_one_cell"]
            ),
        }

    def _rasterize_prediction(
        self,
        mass: np.ndarray,
        x_forward: np.ndarray,
        y_lateral: np.ndarray,
        velocity: np.ndarray,
        rcs: np.ndarray,
    ) -> dict[str, Any]:
        """Mass-conserving scatter of dense image cells into Cartesian bins."""
        shape = mass.shape
        for name, array in [
            ("x_forward", x_forward),
            ("y_lateral", y_lateral),
            ("velocity", velocity),
            ("rcs", rcs),
        ]:
            if array.shape != shape:
                raise ValueError(
                    f"Prediction {name} shape {array.shape} does not match "
                    f"mass shape {shape}."
                )

        stable = (
            np.isfinite(x_forward)
            & (x_forward > 0.0)
            & np.isfinite(y_lateral)
        )
        inside, flat_indices = self._cartesian_indices(
            x_forward=x_forward,
            y_lateral=y_lateral,
            stable=stable,
        )

        flat_output_size = self.bev_height * self.bev_width
        inside_mass = mass[inside]

        velocity_flat, mass_flat = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=velocity[inside],
            weights=inside_mass,
            num_bins=flat_output_size,
        )
        rcs_flat, mass_flat_rcs = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=rcs[inside],
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
        total_mass = float(mass.sum())
        valid_mass = float(mass[stable].sum())
        cartesian_mass = float(mass_flat.sum())

        return {
            "mass": mass_flat.reshape(output_shape),
            "velocity": velocity_flat.reshape(output_shape),
            "rcs": rcs_flat.reshape(output_shape),
            "total_mass": total_mass,
            "inside_mass": cartesian_mass,
            "invalid_mass": float(total_mass - valid_mass),
            "outside_mass": float(valid_mass - cartesian_mass),
        }

    def _rasterize_gt_points(
        self,
        x_forward: np.ndarray,
        y_lateral: np.ndarray,
        velocity: np.ndarray,
        rcs: np.ndarray,
        point_weights: np.ndarray,
    ) -> dict[str, Any]:
        """Rasterize projected GT points and preserve their exact coordinates."""
        arrays = [
            np.asarray(x_forward, dtype=np.float64),
            np.asarray(y_lateral, dtype=np.float64),
            np.asarray(velocity, dtype=np.float64),
            np.asarray(rcs, dtype=np.float64),
            np.asarray(point_weights, dtype=np.float64),
        ]
        lengths = [len(array) for array in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Projected GT arrays must have identical lengths, got "
                f"{lengths}."
            )

        x_forward, y_lateral, velocity, rcs, point_weights = arrays

        stable = (
            np.isfinite(x_forward)
            & (x_forward > 0.0)
            & np.isfinite(y_lateral)
            & np.isfinite(velocity)
            & np.isfinite(rcs)
            & np.isfinite(point_weights)
            & (point_weights > 0.0)
        )
        inside, flat_indices = self._cartesian_indices(
            x_forward=x_forward,
            y_lateral=y_lateral,
            stable=stable,
        )

        flat_output_size = self.bev_height * self.bev_width
        velocity_flat, count_flat = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=velocity[inside],
            weights=point_weights[inside],
            num_bins=flat_output_size,
        )
        rcs_flat, count_flat_rcs = _weighted_mean_from_bins(
            flat_indices=flat_indices,
            values=rcs[inside],
            weights=point_weights[inside],
            num_bins=flat_output_size,
        )

        if not np.array_equal(count_flat, count_flat_rcs):
            raise RuntimeError(
                "Internal error: GT velocity and RCS accumulators differ."
            )

        output_shape = (self.bev_height, self.bev_width)
        support = count_flat.reshape(output_shape)
        points_xy = np.stack(
            [x_forward[inside], y_lateral[inside]],
            axis=1,
        ) if np.any(inside) else np.zeros((0, 2), dtype=np.float64)

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
            "inside_count": inside_count,
            "invalid_count": int(len(x_forward) - stable_count),
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
        """Return inside mask and flattened Cartesian indices."""
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
    # Decoding, metadata, validation, and utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_velocity(
        raw_velocity: torch.Tensor,
        velocity_mode: str,
    ) -> np.ndarray:
        raw_velocity = raw_velocity.detach().float().cpu()

        if velocity_mode == "signed":
            velocity = raw_velocity.pow(3)
        elif velocity_mode == "absolute":
            velocity = raw_velocity.abs()
        else:
            raise RuntimeError(
                f"Unsupported velocity_mode={velocity_mode!r}."
            )

        return velocity.numpy().astype(np.float64, copy=False)

    def _decode_rcs(self, raw_rcs: torch.Tensor) -> np.ndarray:
        raw_rcs = raw_rcs.detach().float().cpu()

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

        if gt.ndim != 3:
            raise ValueError(
                "Expected camera-domain GT with shape [C, H, W], "
                f"got {tuple(gt.shape)}."
            )

        return gt.detach().float().cpu()

    @staticmethod
    def _extract_rad2cam(meta: dict[str, Any]) -> np.ndarray:
        if "rad2cam" not in meta:
            raise KeyError(
                "Missing rad2cam in sample metainfo. Ensure "
                "PPPPackSegInputs stores results['rad2cam']."
            )
        rad2cam = np.asarray(meta["rad2cam"], dtype=np.float64)
        if rad2cam.shape != (3, 4):
            raise ValueError(
                f"Expected rad2cam with shape (3, 4), got {rad2cam.shape}."
            )
        if not np.all(np.isfinite(rad2cam)):
            raise ValueError("rad2cam contains NaN or Inf.")
        return rad2cam

    def _sample_h_fov(self, meta: dict[str, Any]) -> float:
        fovs = meta.get("fovs")
        if isinstance(fovs, (tuple, list, np.ndarray)) and len(fovs) >= 1:
            h_fov = float(fovs[0])
        else:
            h_fov = self.h_fov_deg

        if not np.isfinite(h_fov) or not (0.0 < h_fov < 180.0):
            raise ValueError(
                f"Horizontal FOV must lie in (0, 180), got {h_fov}."
            )
        return h_fov

    @staticmethod
    def _projection_shape(
        meta: dict[str, Any],
        fallback_hw: tuple[int, int],
    ) -> tuple[int, int]:
        ori_shape = meta.get("ori_shape")
        if isinstance(ori_shape, (tuple, list, np.ndarray)) and len(ori_shape) >= 2:
            height = int(ori_shape[0])
            width = int(ori_shape[1])
        else:
            height = int(fallback_hw[0])
            width = int(fallback_hw[1])

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid projected-image shape {(height, width)}."
            )
        return height, width

    def _validate_configuration(self) -> None:
        if not np.isfinite(self.h_fov_deg) or not (
            0.0 < self.h_fov_deg < 180.0
        ):
            raise ValueError(
                "h_fov_deg must lie in (0, 180), got "
                f"{self.h_fov_deg}."
            )
        if not np.isfinite(self.principal_x_rel):
            raise ValueError("principal_x_rel must be finite.")
        if not np.isfinite(self.lateral_sign) or self.lateral_sign == 0.0:
            raise ValueError("lateral_sign must be finite and non-zero.")
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
        if self.rcs_link not in {"identity", "cube", "linear_cube"}:
            raise ValueError(
                f"Unsupported rcs_link={self.rcs_link!r}. Use "
                "'identity', 'cube', or 'linear_cube'."
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