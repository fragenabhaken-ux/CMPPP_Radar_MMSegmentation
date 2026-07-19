import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools" / "ppp_eda"
sys.path.insert(0, str(TOOLS_DIR))

from ppp_eda_geometry import (
    camera_points_to_spatial,
    normalize_spatial_config,
)


def test_camera_backprojection_and_none_limits():
    spatial = normalize_spatial_config(
        dict(
            camera=dict(x_min=None, x_max=None, y_min=None, y_max=None),
            cartesian_bev=dict(
                lateral_min=None,
                lateral_max=None,
                forward_min=None,
                forward_max=None,
            ),
            polar=dict(
                range_min=None,
                range_max=None,
                azimuth_min_deg=None,
                azimuth_max_deg=None,
            ),
        )
    )

    width = 1792
    h_fov_deg = 110.0
    cx = spatial["principal_x_rel"] * (width - 1)
    fx = (width / 2) / np.tan(np.deg2rad(h_fov_deg) / 2)

    lateral = np.array([-10.0, 0.0, 20.0])
    forward = np.array([50.0, 30.0, 100.0])

    # camera x = -lateral and camera z = forward
    rad2cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    u_center = (-lateral / forward) * fx + cx
    geometry = camera_points_to_spatial(
        x_pix=u_center - 0.5,
        depth=forward,
        rad2cam=rad2cam,
        image_width=width,
        h_fov_deg=h_fov_deg,
        spatial=spatial,
    )

    np.testing.assert_allclose(geometry["lateral"], lateral)
    np.testing.assert_allclose(geometry["forward"], forward)
    assert geometry["geometry_valid"].all()
