"""Optional standalone spatial block for configs/ppp_eda/ppp_eda.py."""

ppp_eda_spatial = dict(
    principal_x_rel=0.491,
    lateral_sign=1.0,
    fallback_h_fov_deg=105.0,
    camera=dict(
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
    ),
    cartesian_bev=dict(
        lateral_min=-50.0,
        lateral_max=50.0,
        forward_min=0.0,
        forward_max=150.0,
    ),
    polar=dict(
        range_min=0.0,
        range_max=150.0,
        azimuth_min_deg=-52.5,
        azimuth_max_deg=52.5,
    ),
)
