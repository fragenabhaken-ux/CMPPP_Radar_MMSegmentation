from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
import torch
from math import log
    

@METRICS.register_module()
class PPPMetric(BaseMetric):
    """
    Evaluation metric for PPP-based models.

    Computes:
      - count_abs_error = |E[N] - N_gt|
      - distance_mae       (masked by GT intensity)
      - velocity_mae    (masked by GT intensity)
    """

    def __init__(self, out_img_size, **kwargs):
        super().__init__(prefix='ppp', **kwargs)
        self.out_img_size = out_img_size

    def _get_tensor(self, sample, key):
        obj = sample.get(key, None) if isinstance(sample, dict) else getattr(sample, key, None)
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get("data", None)
        return obj.data


    def process(self, data_batch, data_samples):

        H, W = self.out_img_size
        pixel_scale = H * W

        for sample in data_samples:

            # --- USE LOGITS, NOT pred_sem_seg ---
            pred = self._get_tensor(sample, "seg_logits")
            gt   = self._get_tensor(sample, "gt_sem_seg")

            assert pred is not None, "seg_logits missing"
            assert gt is not None, "gt_sem_seg missing"

            pred = pred.detach()
            gt   = gt.detach()

            # channel 0 = intensity logits
            pred_intensity = torch.exp(pred[0])
            gt_intensity   = gt[0]

            expected_count = pred_intensity.sum() / pixel_scale
            gt_count = gt_intensity.sum()

            entry = {
                "count_abs_error": torch.abs(expected_count - gt_count).item(),
                # optional debug:
                # "expected_count": expected_count.item(),
                # "gt_count": gt_count.item(),
            }

            diff = expected_count - gt_count
            print(
                "[PPPMetric]",
                f"E[N]={expected_count.item():.2f}",
                f"N_gt={gt_count.item():.2f}",
                f"|diff|={diff.abs().item():.2f}",
            )

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

