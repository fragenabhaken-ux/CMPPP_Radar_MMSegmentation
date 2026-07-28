"""Configuration for offline camera-domain marked-PPP evaluation."""

H5_DIRS = {
    "1069846_ppp_finals_unet_54k_metric_eval_CamDom": "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals_cam/1069846_ppp_finals_unet_54k_metric_eval_CamDom/20260728_023617/metric_eval_data/test",
    # "UPerNet": "/path/to/run/timestamp/metric_eval_data/test",
    # "Transformer": "/path/to/run/timestamp/metric_eval_data/test",
}

OUTPUT_DIR = "./camera_metric_results"
MAX_SAMPLES_PER_MODEL = None
VERBOSE = True
FAIL_FAST = True

# Model comparison is only meaningful on exactly the same physical frames.
STRICT_SAMPLE_MATCH = True
REQUIRE_STABLE_UID = True

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
