from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

@METRICS.register_module()
class PPPMetric(BaseMetric):
    """
    Placeholder metric for Poisson Point Process (PPP) based segmentation/detection.

    This metric currently does nothing and returns an empty dict. It exists
    only to keep the validation/test pipeline functional until a proper PPP
    evaluation is implemented.

    In the future, this class can be extended to compute:
        - point detection precision/recall/F1
        - count error (|N_pred - N_gt|)
        - PPP negative log-likelihood
        - intensity MSE (λ_pred vs λ_gt)
        - spatial localization error of predicted points
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples):
        """
        Called once per batch during validation/test.
        For now, we do nothing and simply ignore all predictions.

        Args:
            data_batch (dict): input batch data (unused).
            data_samples (list): model outputs (unused).
        """
        return  # no-op

    def compute_metrics(self, results):
        """
        Called once at the end of validation/test.

        Returns:
            dict: Empty dict (valid metric output for MMEngine).
        """
        return {}
