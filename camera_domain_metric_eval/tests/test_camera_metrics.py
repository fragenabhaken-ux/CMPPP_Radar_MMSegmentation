from __future__ import annotations

import numpy as np
import pandas as pd

from ppp_expected_count_tools import compute_expected_count_metrics
from ppp_mark_metric_tools import (
    circular_error,
    compute_depth_metrics,
    compute_rcs_metrics,
    compute_velocity_metrics,
)
from ppp_offline_metric_eval import summarize_results
from ppp_spatial_likelihood_tools import (
    gaussianize_mass_grid,
    image_ppp_loglik_score,
    optimize_gaussian_sigma,
)


def _sample() -> dict:
    support = np.zeros((4, 4), dtype=float)
    support[1, 1] = 1.0
    support[2, 2] = 1.0
    mass = np.zeros((4, 4), dtype=float)
    mass[1, 1] = 1.0
    mass[2, 2] = 1.0
    return {
        "pred_lambda": mass,
        "gt_support": support,
        "pred_depth": support * np.array(
            [[0, 0, 0, 0], [0, 10, 0, 0], [0, 0, 20, 0], [0, 0, 0, 0]],
            dtype=float,
        ),
        "gt_depth": support * np.array(
            [[0, 0, 0, 0], [0, 10, 0, 0], [0, 0, 20, 0], [0, 0, 0, 0]],
            dtype=float,
        ),
        "pred_velocity": support * np.array(
            [[0, 0, 0, 0], [0, 9, 0, 0], [0, 0, -9, 0], [0, 0, 0, 0]],
            dtype=float,
        ),
        "gt_velocity": support * np.array(
            [[0, 0, 0, 0], [0, -9, 0, 0], [0, 0, 9, 0], [0, 0, 0, 0]],
            dtype=float,
        ),
        "pred_rcs": support * 3.0,
        "gt_rcs": support * 2.0,
        "attrs": {"velocity_mode": "signed", "doppler_period": 20.0},
    }


def test_expected_count() -> None:
    result = compute_expected_count_metrics(_sample())
    assert result["expected_count"] == 2.0
    assert result["gt_count"] == 2
    assert result["count_abs_error"] == 0.0


def test_mass_preserving_gaussian() -> None:
    mass = _sample()["pred_lambda"]
    blurred = gaussianize_mass_grid(mass, 2.0)
    assert np.isclose(blurred.sum(), mass.sum())


def test_aligned_loglik_better_than_shifted() -> None:
    sample = _sample()
    aligned = image_ppp_loglik_score(sample["pred_lambda"], sample["gt_support"])
    shifted = np.roll(sample["pred_lambda"], 1, axis=1)
    shifted_score = image_ppp_loglik_score(shifted, sample["gt_support"])
    assert aligned > shifted_score


def test_sigma_optimizer_includes_zero() -> None:
    sample = _sample()
    result = optimize_gaussian_sigma(
        sample["pred_lambda"],
        sample["gt_support"],
        sigma_max_px=4.0,
    )
    assert result["best_sigma_px"] == 0.0


def test_circular_velocity() -> None:
    err = circular_error(np.array([9.0, -9.0]), np.array([-9.0, 9.0]), 20.0)
    assert np.allclose(np.abs(err), [2.0, 2.0])
    result = compute_velocity_metrics(_sample())
    assert np.isclose(result["velocity_mae"], 2.0)
    assert result["velocity_status"] == "signed_circular"


def test_depth_and_rcs_metrics() -> None:
    sample = _sample()
    depth = compute_depth_metrics(sample)
    rcs = compute_rcs_metrics(sample)
    assert depth["depth_mae"] == 0.0
    assert rcs["rcs_mae"] == 1.0
    assert rcs["rcs_n_points"] == 2


def test_global_point_weighting() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "A",
                "evaluation_status": "ok",
                "depth_mae": 1.0,
                "depth_rmse": 1.0,
                "depth_bias": 1.0,
                "depth_n_points": 10,
                "depth_abs_error_sum": 10.0,
                "depth_squared_error_sum": 10.0,
                "depth_error_sum": 10.0,
                "depth_abs_rel_error_sum": 1.0,
            },
            {
                "model": "A",
                "evaluation_status": "ok",
                "depth_mae": 3.0,
                "depth_rmse": 3.0,
                "depth_bias": 3.0,
                "depth_n_points": 100,
                "depth_abs_error_sum": 300.0,
                "depth_squared_error_sum": 900.0,
                "depth_error_sum": 300.0,
                "depth_abs_rel_error_sum": 30.0,
            },
        ]
    )
    summary = summarize_results(df)
    assert np.isclose(summary.loc[0, "depth_mae_frame_mean"], 2.0)
    assert np.isclose(summary.loc[0, "depth_mae_global"], 310.0 / 110.0)
