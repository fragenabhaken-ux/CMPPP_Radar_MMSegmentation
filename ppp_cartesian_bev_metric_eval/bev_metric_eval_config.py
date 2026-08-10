"""Example configuration for Cartesian-BEV marked-PPP evaluation."""

# One exported HDF5 directory per model. All exports must use the same fixed
# Cartesian BEV raster and, in strict alignment mode, the same sample IDs.
H5_DIRS = {
    "1140228_ppp_finals_unet_54k": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
        "/1140228_ppp_finals_unet_54k_metric_eval_BEVDom/20260731_160945/metric_eval_data_dense_bev_cartesian/test"
    ),   
    "1140233_ppp_finals_unet_54k_woVel": (
            "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
            "/1140233_ppp_finals_unet_54k_woVel_metric_eval_BEVDom/20260731_161010/metric_eval_data_dense_bev_cartesian/test"
        ), 
    "1140234_ppp_finals_unet_54k_woVel_woRCS": (
            "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
            "/1140234_ppp_finals_unet_54k_woVel_woRCS_metric_eval_BEVDom/20260731_161028/metric_eval_data_dense_bev_cartesian/test"
        ),
    "1140280_ppp_finals_TrafoEnc_CamDom_24k": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
        "/1140280_ppp_finals_TrafoEnc_CamDom_24k_BEVDom_metric_eval/20260731_161159/metric_eval_data_dense_bev_cartesian/test"
    ),
    "1140226_ppp_finals_TrafoEnc_BEVpolDom_20k": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
        "/1140226_ppp_finals_TrafoEnc_BEVpolDom_20k_metric_eval/20260731_160947/metric_eval_data_dense_bev_cartesian/test"
    ),
    "1140285_ppp_finals_TrafoEncDec_run2_20k": (
        "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
        "/1140285_ppp_finals_TrafoEncDec_run2_20k_metric_eval/20260731_161245/metric_eval_data_dense_bev_cartesian/test"
    ),
    "1140286_ppp_finals_TrafoEncDec_run2_20k_woVel": (
            "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
            "/1140286_ppp_finals_TrafoEncDec_run2_20k_woVel_metric_eval/20260731_161300/metric_eval_data_dense_bev_cartesian/test"
        ), 
    "1222752_ppp_finals_upernet_run2_21k": (
            "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
            "/1222752_ppp_finals_upernet_run2_21k_metric_eval_BEVDom/20260803_140033/metric_eval_data_dense_bev_cartesian/test"
        ), 
    # "1246573_ppp_finals_TrafoEnc_BEVpolDom_19k_8nodes": (
    #         "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
    #         "/1246573_ppp_finals_TrafoEnc_BEVpolDom_19k_8nodes_metric_eval/20260805_113233/metric_eval_data_dense_bev_cartesian/test"
    #     ), 
    # "1246573_ppp_finals_TrafoEnc_BEVpolDom_19k_8nodes_best": (
    #         "/mnt/c/Users/uik04723/Repositories/gen_data_metric_finals"
    #         "/1247574_ppp_finals_TrafoEnc_BEVpolDom_19k_8nodes_metric_eval_best/20260805_131102/metric_eval_data_dense_bev_cartesian/test"
    #     ), 
}

OUTPUT_DIR = "./cartesian_bev_metric_results"
MAX_SAMPLES_PER_MODEL = None
VERBOSE = True
FAIL_FAST = True

# strict: all models must contain exactly the same stable sample IDs.
# intersection: evaluate only sample IDs shared by all models.
# none: evaluate each model independently.
ALIGNMENT_MODE = None # "strict"

# This guards against accidentally comparing incompatible exports.
EXPECTED_BEV_GEOMETRY = {
    "bev_x_min": 0.0,
    "bev_x_max": 170.0,
    "bev_y_min": -85.0,
    "bev_y_max": 85.0,
    "bev_height": 256,
    "bev_width": 256,
}

METRICS = {
    "count": {
        "enabled": True,
    },
    "likelihood": {
        "enabled": True,
        # Sigma is measured in metres, not raster cells.
        "sigma_min_m": 0.0,
        "sigma_max_m": 20.0,
        "positive_sigma_floor_m": 0.01,
        "max_iter": 40,
        "tol_sigma_m": 0.05,
        "eps": 1e-12,
        # physical_density evaluates mass / Cartesian cell area.
        "normalization": "physical_density",
        "check_zero": True,
    },
    "marks": {
        "enabled": True,
        "velocity_mode": "signed",
        "use_circular_velocity": True,
        "require_doppler_period": True,
        "velocity_gt_abs_min": 0.5,
    },
    "radargen": {
        "enabled": True,
        # PPP predictions are distributions, so explicit point clouds are drawn.
        # Use the same two samplers as PPPMarkVisHookLight, adapted to the shared
        # Cartesian BEV mass grid. Metrics are reported separately per method.
        "sampling_methods": ("thinning", "block"),
        "num_draws": 8,
        "seed": 20260726,
        "block_size": 8,
        "block_placement": "weighted",
        "max_sampled_points": None,

        # RadarGen spatial metrics.
        "iou_threshold_m": 1.0,

        # RadarGen distance-attribute matching, with circular SYS100 velocity.
        "distance_threshold_m": 1.0,
        "rcs_threshold": 8.0,
        "velocity_threshold_mps": 2.5,
        "use_circular_velocity_matching": True,

        # Fixed bounds for normalized full-vector Chamfer.
        "rcs_min": -60.0,
        "rcs_max": 10.0,
        "velocity_min": -11.0,
        "velocity_max": 11.0,

        # RadarGen biased squared MMD with five RBF kernels.
        "mmd_n_kernels": 5,
        "mmd_multiplier": 2.0,
        "mmd_max_points": 2048,
        "mmd_negative_tolerance": 1e-10,
    },
}

# Higher likelihood is better. For Chamfer and MMD use ASCENDING=True.
SORT_BY = "likelihood_optimal_score_mean"
ASCENDING = False

BOOTSTRAP = {
    "enabled": True,

    # 5000 is sufficient for stable percentile intervals.
    "num_resamples": 5000,

    "confidence_level": 0.95,
    "seed": 20260731,

    "metrics": [
        # Count
        "count_error_mean",
        "count_error_std",
        "count_rmse",

        # PPP likelihood
        "likelihood_optimal_score_mean",
        "likelihood_optimal_sigma_m_mean",

        # Mark prediction
        "velocity_mae_global",
        "velocity_nonzero_mae_global",
        "rcs_mae_global",

        # RadarGen geometry
        "radargen_thinning_chamfer_location_m_mean",
        "radargen_thinning_chamfer_full_normalized_mean",
        "radargen_thinning_iou_1m_mean",

        # RadarGen distance-attribute matching
        "radargen_thinning_da_recall_mean",
        "radargen_thinning_da_precision_mean",
        "radargen_thinning_da_f1_mean",

        # RadarGen distributional metrics
        "radargen_thinning_mmd_location_mean",
        "radargen_thinning_mmd_rcs_mean",
        "radargen_thinning_mmd_velocity_mean",
    ],
}