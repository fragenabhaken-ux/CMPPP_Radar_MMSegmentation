import torch #
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class PPPLoss(nn.Module):

    def __init__(self, test_resolution=(512,1024), loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.test_resolution = test_resolution

    def forward(self, inputs, targets, avg_factor=1.0):
        """
        inputs:  (N,1,H,W) logits f(x)
        targets: (N,1,H,W) binary mask of observed points
        """

        N, C, H, W = inputs.shape
        assert C == 1

        # pixel area of original resolution
        pixel_scale = torch.tensor(self.test_resolution, device=inputs.device).prod()

        # intensity
        lam = torch.exp(inputs) / pixel_scale      # λ = exp(f)/pixel_area

        # ∫ λ dx (approximated as sum over grid)
        integral_term = lam.sum()

        # sum_i log λ(x_i)
        # but log λ = f(x) - log(pixel_scale)
        log_lambda_at_points = targets * (inputs - torch.log(pixel_scale))

        observation_term = log_lambda_at_points.sum()

        loss = (integral_term - observation_term) / avg_factor
        return loss * self.loss_weight
    



'''
@MODELS.register_module()
class PPPLoss(nn.Module):

    def __init__(self, test_resolution=(1024, 2048), loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.test_resolution = test_resolution

    def forward(self, inputs, targets, avg_factor=1.0):
        N, C, H, W = inputs.shape

        H0, W0 = self.test_resolution
        pixel_scale = (H0 * W0) / (H * W)
        pixel_scale_t = torch.tensor(pixel_scale, dtype=inputs.dtype, device=inputs.device)

        lam = torch.exp(inputs) / pixel_scale_t
        integral_term = lam.sum()

        mask = targets.sum(dim=1)
        observation_term = (mask * (inputs - torch.log(pixel_scale_t))).sum()

        loss = (integral_term - observation_term) / avg_factor
        return loss * self.loss_weight

###################################################################

import torch
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class PPPLoss(nn.Module):

    def __init__(self, test_resolution=(512,1024), loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.test_resolution = test_resolution

    def forward(self, inputs, targets, avg_factor=1.0):
        """
        inputs:  (N,1,H,W) logits f(x)
        targets: (N,1,H,W) binary mask of observed points
        """

        N, C, H, W = inputs.shape
        assert C == 1

        # pixel area of original resolution
        pixel_scale = torch.tensor(self.test_resolution, device=inputs.device).prod()

        # intensity
        lam = torch.exp(inputs) / pixel_scale      # λ = exp(f)/pixel_area

        # ∫ λ dx (approximated as sum over grid)
        integral_term = lam.sum()

        # sum_i log λ(x_i)
        # but log λ = f(x) - log(pixel_scale)
        log_lambda_at_points = targets * (inputs - torch.log(pixel_scale))

        observation_term = log_lambda_at_points.sum()

        loss = (integral_term - observation_term) / avg_factor
        return loss * self.loss_weight



        
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class PPPLoss(nn.Module):

    def __init__(self, orig_resolution=(256, 512), loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.orig_resolution = orig_resolution

    def forward(self, inputs, targets, avg_factor=1.0):
        """
        inputs:  (N,1,H,W) raw logits f(x)
        targets: (N,1,H,W) binary mask of centers
        """
        if inputs.dim() != 4:
            raise ValueError
        if targets.dim() != 4:
            raise ValueError

        N, C, H, W = inputs.shape
        assert C == 1

        # pixel area scaling factor
        H0, W0 = self.orig_resolution
        pixel_area = (H0 * W0) / (H * W)

        # λ(x) = softplus(f)
        lam = F.softplus(inputs)

        # ∫ λ dx  ≈ sum(λ) * pixel_area
        integral = lam.sum() * pixel_area

        # log λ(x) = log(softplus(f(x)))
        log_lam = torch.log(lam + 1e-8)

        # ∑ log λ(x_i)
        obs = (targets * log_lam).sum()

        # PPP negative log likelihood
        loss = (integral - obs) / avg_factor

        return loss * self.loss_weight



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