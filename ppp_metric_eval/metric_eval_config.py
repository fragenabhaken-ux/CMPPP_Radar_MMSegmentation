"""
Config for offline PPP model comparison.

Notebook usage:

    from ppp_offline_metric_eval import run_from_config

    df_samples, df_summary = run_from_config("metric_eval_config.py")
    df_summary
"""

# ---------------------------------------------------------------------------
# Paths: one entry per model run
# ---------------------------------------------------------------------------

H5_DIRS = {
    "resnet34_fcn_160k_8nodes": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
        "/ppp_resnet34_fcn_160k_8nodes_metric_eval_13904846/20260531_230554/metric_eval_data/test"
    ),    
    "ppp_unet_160k_8nodes": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
        "/13905675_ppp_unet_160k_8nodes_metric_eval/20260601_002233/metric_eval_data/test"
    ),
    "ppp_dinov3_LoRA_rope_big_neck": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
        "/13925402_ppp_dinov3_LoRA_rope_big_neck_noCrop_metric_eval/20260604_140742/metric_eval_data/test"
    ),
    # "ppp_dinov3_mid_LoRA_rope_big_neck": (
    #     "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
    #     "/13925403_ppp_dinov3_mid_LoRA_rope_big_neck_noCrop_metric_eval/20260604_140937/metric_eval_data/test"
    # ),
    # "ppp_dinov3_big_LoRA_rope_big_neck": (
    #     "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
    #     "/13925575_ppp_dinov3_big_LoRA_rope_big_neck_noCrop_metric_eval/20260604_142927/metric_eval_data/test"
    # ),
    # "ppp_dinov3_rope_big_neck_depthLoss_multitask_head": (
    #     "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
    #     "/13924592_ppp_dinov3_rope_big_neck_depthLoss_multitask_head_noCrop_metric_eval/20260604_125447/metric_eval_data/test"
    # ),
    # "ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head": (
    #     "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
    #     "/13925243_ppp_dinov3_rope_big_neck_depthLoss_multitask_two_depth_head_noCrop_metric_eval/20260604_135701/metric_eval_data_twodepth/test"
    # ),
    "ppp_dinov3_LoRA_rope_big_neck_V2": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric"
        "/13944563_ppp_dinov3_LoRA_rope_big_neck_noCrop_V2_metric_eval/20260606_232109/metric_eval_data/test"
    ),
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "./metric_comparison_results"

# Use a small number for debugging, e.g. 5 or 20.
# Set to None for the full run.
MAX_SAMPLES_PER_MODEL = None

VERBOSE = True

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS = {
    "mmd": {
        "enabled": True,
        "score_type": "loglik",
        "sigma_min": 0.0,
        "sigma_max": 120.0,
        "max_iter": 30,
        "tol_sigma": 0.5,
        "normalize_by_gt": True,
        "eps": 1e-12,
        "check_zero": True,
    },
    "expected_count": {
        "enabled": True,
    },
    "depth": {
        "enabled": True,
        "min_gt_depth": 0.0,
        "eps": 1e-8,
    },
}

# Higher MMD score is better.
SORT_BY = "mmd_optimal_score_mean"
ASCENDING = False

# # Beispiel:
# SORT_BY = "count_abs_error_mean"
# ASCENDING = True