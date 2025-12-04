import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

@MODELS.register_module()
class PPPLoss(nn.Module):
    """
    ...
    """

    def __init__(self, test_resolution=(512, 1024), loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.test_resolution = test_resolution

    def forward(self, inputs, targets, avg_factor=1.0):
        """
        inputs:  (N,1,H,W) raw logits
        targets: (N,1,H,W) binary centers
        """

        if inputs.dim() != 4:
            raise ValueError(f"PPPLoss expects inputs (N,1,H,W), got {inputs.shape}")
        if targets.dim() != 4:
            raise ValueError(f"targets must be (N,1,H,W), got {targets.shape}")

        N, C, H, W = inputs.shape
        assert C == 1, "PPPLoss requires 1-channel intensity map."

        device = inputs.device
        # Pixel area normalization
        pixel_scale = torch.tensor(self.test_resolution, device=device).prod()
        # softplus(inputs) / pixel_scale
        lam = F.softplus(inputs) / pixel_scale
        # Integral
        integral_term = lam.sum()
        # Observation term
        log_lam = torch.log(lam + 1e-8)
        observation_term = (targets * log_lam).sum()
        # Poisson loss
        loss = (integral_term - observation_term) / avg_factor

        return loss * self.loss_weight



'''
import torch
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class PPPLoss_V2(nn.Module):
    """
    Pure Poisson Point Process (PPP) negative log-likelihood:

        lambda(x) = exp(f(x))

    Loss:
        L = sum_x lambda(x)*dx  -  sum_i f(x_i)

    Discretized grid:
        dx = 1 / pixel_scale
    """

    def __init__(self, test_resolution=(512, 1024), loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.test_resolution = test_resolution

    def forward(self, inputs, targets, avg_factor=1.0):

        # pixel area scale (dx = 1/s)
        pixel_scale = torch.tensor(self.test_resolution, device=inputs.device).prod()

        # intensity field λ(x) = exp(f(x)) / s
        lam = torch.exp(inputs) / pixel_scale

        # integral term ∑ λ(x)
        integral_term = lam.sum()

        # observation term ∑ f(x_i)
        observation_term = (targets * inputs).sum()

        # PPP negative log-likelihood
        loss = integral_term - observation_term

        return loss * self.loss_weight

'''