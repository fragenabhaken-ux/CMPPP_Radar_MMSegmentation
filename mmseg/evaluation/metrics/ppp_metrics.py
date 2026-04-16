from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

# Stable production wrapper for the sigma-optimized PPP-style metric.
from .ppp_metric_loss_sigma_opt import evaluate_ppp_loss_sigma_opt


# ------------------------------------------------------------------
# Logger used for optional metric debugging.
# Set LOG_LEVEL to logging.DEBUG if you want per-sample debug output.
# ------------------------------------------------------------------
LOGGER_NAME = "ppp_metric_loss"
LOG_LEVEL = logging.INFO  # change to logging.INFO, logging.DEBUG

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOG_LEVEL)


@METRICS.register_module()
class PPPMetric(BaseMetric):
    """
    Evaluation metric for PPP-based models.

    This metric can compute several components:
      - count_abs_error:
            absolute error between expected predicted count and GT count
      - loss_sigma_opt_score:
            best sigma-optimized PPP/MMD-style score
      - loss_sigma_opt_best_sigma:
            sigma at which the best score is achieved

    Optionally, these metrics can be combined into a weighted sum:
      - selection_score

    Interpretation of the prediction
    --------------------------------
    We assume that channel 0 of `seg_logits` contains log-intensity logits.

    Therefore:
      - `torch.exp(pred[0])` is the positive intensity field
      - `torch.exp(pred[0]) / (H * W)` is the discrete prediction grid whose
        total sum equals the expected count

    This discretely scaled `pred_grid` is what we pass to the sigma-optimized
    loss metric helper library.
    """

    def __init__(
        self,
        out_img_size,
        metric_weights: dict[str, float] | None = None,
        # ----------------------------------------------------------
        # Switches controlling which metric components are computed.
        # ----------------------------------------------------------
        compute_count_abs_error: bool = True,
        compute_loss_sigma_opt: bool = False,
        return_selection_score: bool = False,
        # ----------------------------------------------------------
        # Settings for the sigma-optimized loss metric.
        # These are forwarded to evaluate_ppp_loss_sigma_opt(...).
        # ----------------------------------------------------------
        sigma_opt_method: str = "hybrid",
        sigma_opt_score_kind: str = "loglik",
        sigma_opt_verbose: bool = False,
        sigma_opt_normalize_by_gt: bool = True,
        sigma_opt_sigma_max: float = 120.0,
        sigma_opt_n_coarse: int = 21,
        sigma_opt_refine_max_iter: int = 30,
        sigma_opt_refine_tol_sigma: float = 0.5,
        sigma_opt_sigma_min: float = 0.0,
        sigma_opt_max_iter: int = 30,
        sigma_opt_tol_sigma: float = 0.5,
        sigma_opt_truncate: float = 4.0,
        sigma_opt_sigma_grid=None,
        sigma_opt_eps_h: float = 1e-12,
        **kwargs,
    ):
        # Prefix "ppp" means the final logged metric names will look like
        # ppp/count_abs_error, ppp/loss_sigma_opt_score, etc.
        super().__init__(prefix="ppp", **kwargs)

        # Output image size of the prediction grid.
        self.out_img_size = tuple(out_img_size)

        # Which metric components should actually be computed.
        self.compute_count_abs_error = bool(compute_count_abs_error)
        self.compute_loss_sigma_opt = bool(compute_loss_sigma_opt)
        self.return_selection_score = bool(return_selection_score)

        # Optional weighted combination of metrics, e.g.
        # {"loss_sigma_opt_score": 1.0, "count_abs_error": -0.2}
        self.metric_weights = metric_weights or {}

        # Store sigma-optimization settings.
        self.sigma_opt_method = str(sigma_opt_method).lower()
        self.sigma_opt_score_kind = str(sigma_opt_score_kind).lower()
        self.sigma_opt_verbose = bool(sigma_opt_verbose)
        self.sigma_opt_normalize_by_gt = bool(sigma_opt_normalize_by_gt)
        self.sigma_opt_sigma_max = float(sigma_opt_sigma_max)
        self.sigma_opt_n_coarse = int(sigma_opt_n_coarse)
        self.sigma_opt_refine_max_iter = int(sigma_opt_refine_max_iter)
        self.sigma_opt_refine_tol_sigma = float(sigma_opt_refine_tol_sigma)
        self.sigma_opt_sigma_min = float(sigma_opt_sigma_min)
        self.sigma_opt_max_iter = int(sigma_opt_max_iter)
        self.sigma_opt_tol_sigma = float(sigma_opt_tol_sigma)
        self.sigma_opt_truncate = float(sigma_opt_truncate)
        self.sigma_opt_sigma_grid = sigma_opt_sigma_grid
        self.sigma_opt_eps_h = float(sigma_opt_eps_h)

    @staticmethod
    def _get_tensor(sample: Any, key: str):
        """
        Extract a tensor-like object from an MMEngine data sample.

        Handles both dict-style and attribute-style access, and also the common
        wrapper format where the actual tensor lives in `.data` or `["data"]`.
        """
        obj = sample.get(key, None) if isinstance(sample, dict) else getattr(sample, key, None)
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get("data", None)
        return obj.data

    @staticmethod
    def _mean_if_exists(results, key: str):
        """
        Compute the mean of one result field across all samples, if present.
        """
        vals = [r[key] for r in results if key in r]
        return sum(vals) / len(vals) if len(vals) > 0 else None

    def _build_sigma_opt_kwargs(self) -> dict[str, Any]:
        """
        Build the keyword arguments that will be passed to the sigma-optimized
        metric helper, depending on the chosen optimization method.
        """
        # Shared arguments used by all sigma-optimization methods.
        kwargs = {
            "normalize_by_gt": self.sigma_opt_normalize_by_gt,
            "truncate": self.sigma_opt_truncate,
            "eps_h": self.sigma_opt_eps_h,
        }

        # Hybrid method: coarse scan + local refinement.
        if self.sigma_opt_method == "hybrid":
            kwargs.update(
                sigma_max=self.sigma_opt_sigma_max,
                n_coarse=self.sigma_opt_n_coarse,
                refine_max_iter=self.sigma_opt_refine_max_iter,
                refine_tol_sigma=self.sigma_opt_refine_tol_sigma,
            )

        # Golden / ternary: continuous search on [sigma_min, sigma_max].
        elif self.sigma_opt_method in {"golden", "ternary"}:
            kwargs.update(
                sigma_min=self.sigma_opt_sigma_min,
                sigma_max=self.sigma_opt_sigma_max,
                max_iter=self.sigma_opt_max_iter,
                tol_sigma=self.sigma_opt_tol_sigma,
            )

        # Naive grid search: evaluate an explicit sigma grid.
        elif self.sigma_opt_method == "grid_naive":
            if self.sigma_opt_sigma_grid is None:
                kwargs["sigma_grid"] = np.linspace(0.0, self.sigma_opt_sigma_max, 25)
            else:
                kwargs["sigma_grid"] = np.asarray(self.sigma_opt_sigma_grid, dtype=np.float64)

        else:
            raise ValueError(f"Unknown sigma_opt_method: {self.sigma_opt_method}")

        return kwargs

    def process(self, data_batch, data_samples):
        """
        Process one validation/test batch.

        For each sample, we:
          1. read prediction and GT
          2. compute expected-count error if requested
          3. compute sigma-optimized loss metric if requested
          4. append the per-sample result dict to self.results
        """
        H, W = self.out_img_size
        pixel_scale = float(H * W)

        # Build sigma-opt kwargs only once per batch if needed.
        sigma_opt_kwargs = self._build_sigma_opt_kwargs() if self.compute_loss_sigma_opt else None

        for sample_idx, sample in enumerate(data_samples):
            # The model output is expected in seg_logits.
            pred = self._get_tensor(sample, "seg_logits")
            # The GT segmentation-style tensor is expected in gt_sem_seg.
            gt = self._get_tensor(sample, "gt_sem_seg")

            assert pred is not None, "seg_logits missing"
            assert gt is not None, "gt_sem_seg missing"

            # Detach from the computation graph because metrics do not need grads.
            pred = pred.detach()
            gt = gt.detach()

            # --------------------------------------------------------------
            # Channel 0 stores the log-intensity logits.
            # After exp(...), we obtain a positive intensity field.
            # --------------------------------------------------------------
            pred_grid_raw = torch.exp(pred[0]).to(dtype=torch.float64)

            # GT channel 0 is the GT grid. In the sigma-opt helper library,
            # values > 0 are interpreted as GT support locations.
            gt_grid = gt[0].to(dtype=torch.float64)

            # --------------------------------------------------------------
            # Expected count:
            # The raw positive field sums to expected_count * (H * W),
            # so dividing by pixel_scale yields the expected count.
            # --------------------------------------------------------------
            expected_count = pred_grid_raw.sum() / pixel_scale
            gt_count = gt_grid.sum()

            # Per-sample result dict appended to self.results.
            entry: dict[str, Any] = {}

            # --------------------------------------------------------------
            # 1) Simple count calibration metric
            # --------------------------------------------------------------
            if self.compute_count_abs_error:
                entry["count_abs_error"] = torch.abs(expected_count - gt_count).item()

            # --------------------------------------------------------------
            # 2) Sigma-optimized loss metric
            # --------------------------------------------------------------
            if self.compute_loss_sigma_opt:
                # Rescale the prediction grid so that its sum equals the
                # expected count. This is the discrete PPP-style prediction
                # grid used by the sigma-optimized metric helper.
                pred_grid = pred_grid_raw / pixel_scale

                # Evaluate the best sigma-optimized score.
                res = evaluate_ppp_loss_sigma_opt(
                    pred_grid=pred_grid.cpu().numpy(),
                    gt_support=gt_grid.cpu().numpy(),
                    method=self.sigma_opt_method,
                    score_kind=self.sigma_opt_score_kind,
                    verbose=self.sigma_opt_verbose,
                    **sigma_opt_kwargs,
                )

                # Keep only the essential outputs in the per-sample results.
                entry["loss_sigma_opt_score"] = float(res["best_score"])
                entry["loss_sigma_opt_best_sigma"] = float(res["best_sigma"])

                # Optional debug logging for per-sample inspection.
                if self.sigma_opt_verbose:
                    logger.debug(
                        "PPPMetric | sample=%d | method=%s | score_kind=%s | best_sigma=%.6f | best_score=%.6f | status=%s",
                        sample_idx,
                        res["method"],
                        res["score_kind"],
                        res["best_sigma"],
                        res["best_score"],
                        res["status"],
                    )

            # Store the per-sample result. MMEngine will aggregate later via
            # compute_metrics(...).
            self.results.append(entry)

    def compute_metrics(self, results):
        """
        Aggregate the stored per-sample results over the whole dataset.

        Depending on the enabled switches, this returns:
          - count_abs_error
          - loss_sigma_opt_score
          - loss_sigma_opt_best_sigma
          - selection_score (optional weighted sum)
        """
        metrics = {}

        # Aggregate count calibration metric.
        if self.compute_count_abs_error:
            metrics["count_abs_error"] = self._mean_if_exists(results, "count_abs_error")

        # Aggregate sigma-optimized loss metric outputs.
        if self.compute_loss_sigma_opt:
            metrics["loss_sigma_opt_score"] = self._mean_if_exists(results, "loss_sigma_opt_score")
            metrics["loss_sigma_opt_best_sigma"] = self._mean_if_exists(results, "loss_sigma_opt_best_sigma")

        # Optional weighted sum for model selection.
        #
        # Example:
        #   metric_weights = {
        #       "loss_sigma_opt_score": 1.0,
        #       "count_abs_error": -0.2,
        #   }
        #
        # Then:
        #   selection_score =
        #       1.0 * loss_sigma_opt_score
        #     + (-0.2) * count_abs_error
        #
        # This is useful if you want save_best to depend on multiple metrics.
        if self.return_selection_score:
            selection_score = 0.0
            used_any = False

            for metric_name, weight in self.metric_weights.items():
                if metric_name in metrics and metrics[metric_name] is not None:
                    selection_score += float(weight) * float(metrics[metric_name])
                    used_any = True

            if used_any:
                metrics["selection_score"] = float(selection_score)

        return metrics


# from mmengine.evaluator import BaseMetric
# from mmengine.registry import METRICS
# import torch
# from math import log
    

# @METRICS.register_module()
# class PPPMetric(BaseMetric):
#     """
#     Evaluation metric for PPP-based models.

#     Computes:
#       - count_abs_error = |E[N] - N_gt|
#       - distance_mae       (masked by GT intensity)
#       - velocity_mae    (masked by GT intensity)
#     """

#     def __init__(self, out_img_size, **kwargs):
#         super().__init__(prefix='ppp', **kwargs)
#         self.out_img_size = out_img_size

#     def _get_tensor(self, sample, key):
#         obj = sample.get(key, None) if isinstance(sample, dict) else getattr(sample, key, None)
#         if obj is None:
#             return None
#         if isinstance(obj, dict):
#             return obj.get("data", None)
#         return obj.data


#     def process(self, data_batch, data_samples):

#         H, W = self.out_img_size
#         pixel_scale = H * W

#         for sample in data_samples:

#             # --- USE LOGITS, NOT pred_sem_seg ---
#             pred = self._get_tensor(sample, "seg_logits")
#             gt   = self._get_tensor(sample, "gt_sem_seg")

#             assert pred is not None, "seg_logits missing"
#             assert gt is not None, "gt_sem_seg missing"

#             pred = pred.detach()
#             gt   = gt.detach()

#             # channel 0 = intensity logits
#             pred_intensity = torch.exp(pred[0])
#             gt_intensity   = gt[0]

#             expected_count = pred_intensity.sum() / pixel_scale
#             gt_count = gt_intensity.sum()

#             entry = {
#                 "count_abs_error": torch.abs(expected_count - gt_count).item(),
#                 # optional debug:
#                 # "expected_count": expected_count.item(),
#                 # "gt_count": gt_count.item(),
#             }

#             # diff = expected_count - gt_count
#             # print(
#             #     "[PPPMetric]",
#             #     f"E[N]={expected_count.item():.2f}",
#             #     f"N_gt={gt_count.item():.2f}",
#             #     f"|diff|={diff.abs().item():.2f}",
#             # )

#             self.results.append(entry)


#     def compute_metrics(self, results):
#         """
#         Aggregate metrics over the whole dataset.
#         """

#         metrics = {}

#         def mean_if_exists(key):
#             vals = [r[key] for r in results if key in r]
#             return sum(vals) / len(vals) if len(vals) > 0 else None

#         metrics["count_abs_error"] = mean_if_exists("count_abs_error")

#         # for the time being, we only take abs count as a metric. This is, because 
#         # distance and velocity are not yet normalized and can dominate. Once we have
#         # included those, we can uncomment the below code and set in config 
#         # default_hooks/checkpoint/save_best='selection_score'.

#         # distance_mae = mean_if_exists("distance_mae")
#         # if distance_mae is not None:
#         #     metrics["distance_mae"] = distance_mae

#         # vel_mae = mean_if_exists("velocity_mae")
#         # if vel_mae is not None:
#         #     metrics["velocity_mae"] = vel_mae
        
#         # selection_score = metrics["count_abs_error"]

#         # if "distance_mae" in metrics:
#         #     selection_score += 0.5 * metrics["distance_mae"]

#         # if "velocity_mae" in metrics:
#         #     selection_score += 0.5 * metrics["velocity_mae"]

#         # metrics["selection_score"] = selection_score


#         return metrics

