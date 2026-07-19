from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
import numpy as np
import torch

from mmengine.logging import MMLogger

from .ppp_metrics_loss_mmd_tools import compute_optimal_sigma_metric


@METRICS.register_module()
class PPPLossMMD(BaseMetric):
    """
    Evaluation metric based on an optimally Gaussianized prediction grid.

    The metric supports two score variants:
      - "loglik": PPP log-likelihood style score
      - "mmd":    overlap-based score on GT support

    For each sample:
      1. extract the predicted intensity grid
      2. extract the ground-truth support
      3. determine the optimal Gaussian scale sigma via golden-section search
      4. store the resulting best score and sigma

    Dataset-level results are reported as means over all samples.

    Metric parameters:
    - out_img_size: expected spatial size (H, W) of prediction and GT grids; used as a consistency check
    - score_type: score variant to use, either "loglik" (PPP log-likelihood style) or "mmd" (overlap-based score)
    - sigma_min, sigma_max: lower and upper bound of the admissible sigma range for the Gaussian blur
    - max_iter: maximum number of golden-section search iterations
    - tol_sigma: stopping tolerance for the sigma optimization; smaller values give a more precise optimum but require more iterations
    - normalize_by_gt: if True, divide the score by the number of GT support points to improve comparability across samples
    - eps: numerical stabilization constant for the log-likelihood score; prevents taking log(0)
    - check_zero: if True, also evaluate the unsmoothed case sigma = 0 (no Gaussianization)
    - pred_channel: prediction channel used as the evaluation grid (e.g. PPP intensity channel)
    - gt_channel: GT channel used as the support mask; entries > 0 are treated as GT locations
    - use_logits: if True, exponentiate the selected prediction channel before evaluation (for log-intensity / logits outputs)
    - prefix: metric prefix used by MMEngine / MMSeg for logging
    - **kwargs: additional arguments forwarded to BaseMetric
    """

    def __init__(
        self,
        out_img_size,
        score_type="loglik",
        sigma_min=0.0,
        sigma_max=120.0,
        max_iter=30,
        tol_sigma=0.5,
        normalize_by_gt=True,
        eps=1e-12,
        check_zero=True,
        pred_channel=0,
        gt_channel=0,
        use_logits=True,
        cell_measure=None,
        prefix="ppp",
        verbose_logging=True,
        **kwargs,
    ):
        super().__init__(prefix=prefix, **kwargs)

        self.out_img_size = out_img_size
        self.score_type = score_type
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.max_iter = int(max_iter)
        self.tol_sigma = float(tol_sigma)
        self.normalize_by_gt = bool(normalize_by_gt)
        self.eps = float(eps)
        self.check_zero = bool(check_zero)
        self.pred_channel = int(pred_channel)
        self.gt_channel = int(gt_channel)
        self.use_logits = bool(use_logits)
        self.cell_measure = (
            None if cell_measure is None else float(cell_measure)
        )
        self.verbose_logging = bool(verbose_logging)

    def _get_tensor(self, sample, key):
        obj = sample.get(key, None) if isinstance(sample, dict) else getattr(sample, key, None)
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get("data", None)
        return obj.data

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def process(self, data_batch, data_samples):
        expected_h, expected_w = self.out_img_size
        pixel_scale = expected_h * expected_w

        for sample in data_samples:
            pred = self._get_tensor(sample, "seg_logits")
            gt = self._get_tensor(sample, "gt_sem_seg")

            pred = pred.detach()
            gt = gt.detach()

            pred_grid = pred[self.pred_channel]
            if self.use_logits:
                pred_grid = torch.exp(pred_grid)

            if self.cell_measure is None:
                # Existing camera-domain behavior.
                pred_grid = pred_grid / pixel_scale
            else:
                # Physical cell mass for polar intensity defined w.r.t. dr dtheta.
                pred_grid = pred_grid * self.cell_measure

            gt_support = gt[self.gt_channel]

            pred_grid_np = self._to_numpy(pred_grid).astype(np.float64, copy=False)
            gt_support_np = self._to_numpy(gt_support)

            result = compute_optimal_sigma_metric(
                pred_grid=pred_grid_np,
                gt_support=gt_support_np,
                score_type=self.score_type,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                max_iter=self.max_iter,
                tol_sigma=self.tol_sigma,
                eps=self.eps,
                normalize_by_gt=self.normalize_by_gt,
                check_zero=self.check_zero,
            )

            self.results.append(
                {
                    "optimal_score": float(result["best_score"]),
                    "optimal_sigma": float(result["best_sigma"]),
                }
            )

            if self.verbose_logging:
                logger = MMLogger.get_current_instance()
                logger.info(
                    f"optimal_score={result['best_score']:.6f}, "
                    f"optimal_sigma={result['best_sigma']:.3f}"
                )

    def compute_metrics(self, results):
        metrics = {}

        def mean_if_exists(key):
            values = [r[key] for r in results if key in r]
            return sum(values) / len(values) if len(values) > 0 else None

        optimal_score = mean_if_exists("optimal_score")
        optimal_sigma = mean_if_exists("optimal_sigma")

        if optimal_score is not None:
            metrics["optimal_score"] = optimal_score

        if optimal_sigma is not None:
            metrics["optimal_sigma"] = optimal_sigma

        return metrics