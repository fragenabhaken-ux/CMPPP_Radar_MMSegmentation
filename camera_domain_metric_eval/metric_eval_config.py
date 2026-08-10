"""Configuration for offline camera-domain marked-PPP evaluation."""

H5_DIRS = {
    "1069846_ppp_finals_unet_54k": "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals_cam/1069846_ppp_finals_unet_54k_metric_eval_CamDom/20260728_023617/metric_eval_data/test",
    "1260093_ppp_finals_upernet_run3_50k": "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals_cam/1260093_ppp_finals_upernet_run3_50k_metric_eval_CamDom/20260806_132924/metric_eval_data/test",
    "1069890_ppp_finals_TrafoEnc_CamDom_24k": "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals_cam/1069890_ppp_finals_TrafoEnc_CamDom_24k_CamDom_metric_eval/20260728_024516/metric_eval_data/test",
}

OUTPUT_DIR = "./camera_metric_results"
MAX_SAMPLES_PER_MODEL = None
VERBOSE = True
FAIL_FAST = True

# Model comparison is only meaningful on exactly the same physical frames.
STRICT_SAMPLE_MATCH = False
REQUIRE_STABLE_UID = False

METRICS = {
    "expected_count": {
        "enabled": True,
        "strict_binary_gt": True,
    },
    "likelihood": {
        "enabled": True,
        "sigma_min_px": 0.0,
        "sigma_max_px": 120.0,
        "max_iter": 30,
        "tol_sigma_px": 0.25,
        "normalize_by_gt": True,
        "eps": 1e-12,
        "check_zero": True,
    },
    "depth": {
        "enabled": True,
        "min_gt_depth": 0.0,
        "eps": 1e-8,
    },
    "velocity": {
        "enabled": True,
        "mode": "auto",
        "circular": True,
        "fixed_period": None,
        "period_attr_candidates": (
            "doppler_period",
            "dopplerRange",
            "doppler_range",
        ),
        "require_period": True,
    },
    "rcs": {
        "enabled": True,
    },
}

# Higher optimized log-likelihood is better.
SORT_BY = "likelihood_optimal_score_mean"
ASCENDING = False

BOOTSTRAP = {
    "enabled": True,

    # Number of paired bootstrap resamples.
    "num_resamples": 5000,

    # Percentile confidence interval.
    "confidence_level": 0.95,

    # Fixed seed for reproducibility.
    "seed": 20260731,

    "metrics": [
        # Expected count
        "count_bias",
        "count_error_std",
        "count_rmse",

        # Spatial PPP score
        "likelihood_optimal_score_mean",
        "likelihood_optimal_sigma_px_mean",

        # Mark prediction
        "depth_mae_global",
        "velocity_mae_global",
        "rcs_mae_global",
    ],
}