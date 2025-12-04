import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class PPPLoss_V2(nn.Module):
    """
    Conditional PPP loss (location likelihood):

        p_k = exp(f_k) / sum_j exp(f_j)

        If centers exist:
            Loss_i = - mean_{centers} log p_k
        Else:
            Loss_i = logsumexp(f)        # penalize large intensities

    Numerically stable (logsumexp).
    Includes full analytics:
    - observation term (sum log p)
    - mean log p
    - final loss
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_name = 'loss_ppp'

    def forward(self, inputs, targets, avg_factor=1.0):
        """
        inputs:  (N,1,H,W) raw logits f(x)
        targets: (N,1,H,W) binary mask with 1 at center pixels
        """

        N, C, H, W = inputs.shape

        # flatten spatial dims
        f = inputs.view(N, -1)       # (N, HW)
        t = targets.view(N, -1)      # (N, HW)

        per_image_losses = []
        analytics_data = []

        for i in range(N):
            fi = f[i]      # (HW,)
            ti = t[i]      # (HW,)

            center_mask = ti > 0.5
            num_centers = int(center_mask.sum().item())

            # log ∑_x exp(f)
            lse = torch.logsumexp(fi, dim=0)  # scalar

            # log p_k = f_k - log ∑ exp(f)
            log_p = fi - lse                  # (HW,)

            if num_centers > 0:
                # observation terms at centers
                log_p_centers = log_p[center_mask]   # (K,)

                obs_sum = log_p_centers.sum().item()     # sum log p(x_i)
                obs_mean = log_p_centers.mean().item()   # mean log p(x_i)

                # final loss for this image
                nll_i = - log_p_centers.mean()           # negative mean log-likelihood
            else:
                # no centers: fallback loss = logsumexp
                obs_sum = 0.0
                obs_mean = 0.0
                nll_i = lse

            per_image_losses.append(nll_i)

            analytics_data.append({
                "f_min": float(fi.min().item()),
                "f_max": float(fi.max().item()),
                "f_mean": float(fi.mean().item()),
                "num_centers": num_centers,
                "lse": float(lse.item()),
                "obs_sum": obs_sum,
                "obs_mean": obs_mean,
                "loss_i": float(nll_i.item()),
            })

        loss = torch.stack(per_image_losses).mean() / avg_factor

        self._analytics(analytics_data, loss, every=50)

        return loss * self.loss_weight


    def _analytics(self, data, loss, every=50):
        """Print diagnostics every `every` iterations."""

        if not hasattr(self, "_iter"):
            self._iter = 0
        self._iter += 1

        if self._iter % every != 0:
            return

        d = data[0]  # print only first image in batch

        print("\n=== PPP (Conditional) LOSS ANALYTICS ===")
        print(f"Iteration:         {self._iter}")
        print("----------------------------------")
        print(f"f.min/max/mean:    {d['f_min']:.4f}   "
              f"{d['f_max']:.4f}   {d['f_mean']:.4f}")
        print(f"#Centers:          {d['num_centers']}")
        print(f"logsumexp(f):      {d['lse']:.4f}")

        print(f"Observation (sum): {d['obs_sum']:.4f}")
        print(f"Observation (mean):{d['obs_mean']:.4f}")

        print(f"Loss (img 0):      {d['loss_i']:.4f}")
        print(f"Batch Loss:        {loss.item():.4f}")

        if torch.isnan(loss):
            print("⚠ WARNING: LOSS IS NaN!")
        if torch.isinf(loss):
            print("⚠ WARNING: LOSS IS INF!")

        print("====================================\n")
