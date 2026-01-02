from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
import torch


@METRICS.register_module()
class PPPMetric(BaseMetric):
    """
    Evaluation metric for PPP-based models.

    Computes:
      - count_abs_error = |E[N] - N_gt|
      - distance_mae       (masked by GT intensity)
      - velocity_mae    (masked by GT intensity)
    """

    def __init__(self, **kwargs):
        super().__init__(prefix='ppp', **kwargs)

    def _get_tensor(obj):
        """
        Extract tensor from:
        - PixelData
        - dict {'data': tensor}
        """
        if isinstance(obj, dict):
            return obj["data"]
        else:
            return obj.data

    def process(self, data_batch, data_samples):

        for sample in data_samples:

            def get_data(x):
                return x["data"] if isinstance(x, dict) else x.data
            
            pred = get_data(sample["pred_sem_seg"])
            gt   = get_data(sample["gt_sem_seg"])

            pred = pred.detach()
            gt   = gt.detach()

            # ---------------------------------------------
            # Intensity / count
            # ---------------------------------------------
            pred_intensity = torch.exp(pred[0])
            gt_intensity   = gt[0]

            H, W = pred_intensity.shape
            expected_count = (pred_intensity / (H * W)).sum()
            gt_count = gt_intensity.sum()

            count_abs_error = torch.abs(expected_count - gt_count)

            entry = {
                "count_abs_error": count_abs_error.item()
            }

            # ---------------------------------------------
            # Mask for distance / velocity
            # ---------------------------------------------
            mask = gt_intensity == 1

            if pred.shape[0] > 1 and mask.any():
                entry["distance_mae"] = torch.mean(
                    torch.abs(pred[1][mask] - gt[1][mask])
                ).item()

            if pred.shape[0] > 2 and mask.any():
                entry["velocity_mae"] = torch.mean(
                    torch.abs(pred[2][mask] - gt[2][mask])
                ).item()

            self.results.append(entry)


    def compute_metrics(self, results):
        """
        Aggregate metrics over the whole dataset.
        """

        metrics = {}

        def mean_if_exists(key):
            vals = [r[key] for r in results if key in r]
            return sum(vals) / len(vals) if len(vals) > 0 else None

        metrics["count_abs_error"] = mean_if_exists("count_abs_error")

        # for the time being, we only take abs count as a metric. This is, because 
        # distance and velocity are not yet normalized and can dominate. Once we have
        # included those, we can uncomment the below code and set in config 
        # default_hooks/checkpoint/save_best='selection_score'.

        # distance_mae = mean_if_exists("distance_mae")
        # if distance_mae is not None:
        #     metrics["distance_mae"] = distance_mae

        # vel_mae = mean_if_exists("velocity_mae")
        # if vel_mae is not None:
        #     metrics["velocity_mae"] = vel_mae
        
        # selection_score = metrics["count_abs_error"]

        # if "distance_mae" in metrics:
        #     selection_score += 0.5 * metrics["distance_mae"]

        # if "velocity_mae" in metrics:
        #     selection_score += 0.5 * metrics["velocity_mae"]

        # metrics["selection_score"] = selection_score


        return metrics




# from mmengine.evaluator import BaseMetric
# from mmengine.registry import METRICS

# @METRICS.register_module()
# class PPPMetric(BaseMetric):
#     """
#     Placeholder metric for Poisson Point Process (PPP) based segmentation/detection.

#     This metric currently does nothing and returns an empty dict. It exists
#     only to keep the validation/test pipeline functional until a proper PPP
#     evaluation is implemented.

#     In the future, this class can be extended to compute:
#         - point detection precision/recall/F1
#         - count error (|N_pred - N_gt|)
#         - PPP negative log-likelihood
#         - intensity MSE (λ_pred vs λ_gt)
#         - spatial localization error of predicted points
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def process(self, data_batch, data_samples):
#         """
#         Called once per batch during validation/test.
#         For now, we do nothing and simply ignore all predictions.

#         Args:
#             data_batch (dict): input batch data (unused).
#             data_samples (list): model outputs (unused).
#         """
#         return  # no-op

#     def compute_metrics(self, results):
#         """
#         Called once at the end of validation/test.

#         Returns:
#             dict: Empty dict (valid metric output for MMEngine).
#         """
#         return {}
