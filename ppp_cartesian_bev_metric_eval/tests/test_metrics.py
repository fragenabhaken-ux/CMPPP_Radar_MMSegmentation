from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from bev_eval.count_metrics import compute_count_metrics
from bev_eval.evaluator import run_evaluation
from bev_eval.likelihood import compute_likelihood_metrics
from bev_eval.mark_metrics import compute_mark_metrics
from bev_eval.pointcloud import (
    MarkedPointCloud,
    sample_ppp_block_grid,
    sample_ppp_thinning_grid,
    sample_predicted_ppp,
)
from bev_eval.radargen_metrics import compute_radargen_pointcloud_metrics
from bev_eval.types import BEVGeometry


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = BEVGeometry(0.0, 2.0, -1.0, 1.0, 2, 2)
        self.sample = {
            "pred_lambda": np.array([[1.0, 0.0], [0.0, 0.0]]),
            "pred_velocity": np.array([[9.0, 0.0], [0.0, 0.0]]),
            "pred_rcs": np.array([[-4.0, 0.0], [0.0, 0.0]]),
            "gt_support": np.array([[1.0, 0.0], [0.0, 0.0]]),
            "gt_velocity": np.array([[-9.0, 0.0], [0.0, 0.0]]),
            "gt_rcs": np.array([[-2.0, 0.0], [0.0, 0.0]]),
            "gt_points_xy": np.array([[0.5, -0.5]]),
            "gt_point_velocity": np.array([-9.0]),
            "gt_point_rcs": np.array([-2.0]),
            "attrs": {"doppler_period": 20.0},
            "geometry": self.geometry,
            "meta": {"sample_uid": "sample-A"},
        }

    def test_count(self) -> None:
        result = compute_count_metrics(self.sample)
        self.assertEqual(result["expected_count"], 1.0)
        self.assertEqual(result["gt_count"], 1.0)
        self.assertEqual(result["count_abs_error"], 0.0)

    def test_likelihood_prefers_zero_for_perfect_single_cell(self) -> None:
        result = compute_likelihood_metrics(
            self.sample,
            {
                "sigma_min_m": 0.0,
                "sigma_max_m": 2.0,
                "positive_sigma_floor_m": 0.01,
                "tol_sigma_m": 0.001,
                "normalization": "physical_density",
            },
        )
        self.assertAlmostEqual(result["likelihood_optimal_sigma_m"], 0.0, places=8)
        self.assertEqual(result["likelihood_status"], "ok")

    def test_circular_velocity_and_rcs_marks(self) -> None:
        result = compute_mark_metrics(self.sample)
        self.assertAlmostEqual(result["velocity_mae"], 2.0)
        self.assertAlmostEqual(result["velocity_bias"], -2.0)
        self.assertAlmostEqual(result["rcs_mae"], 2.0)
        self.assertAlmostEqual(result["rcs_bias"], -2.0)
        self.assertEqual(result["mark_evaluation_source"], "exact_gt_points")

    def test_radargen_identical_clouds(self) -> None:
        cloud = MarkedPointCloud(
            xy=np.array([[0.2, -0.3], [1.4, 0.7]]),
            rcs=np.array([-5.0, 2.0]),
            velocity=np.array([1.0, -3.0]),
        )
        result = compute_radargen_pointcloud_metrics(
            cloud,
            cloud,
            self.geometry,
            doppler_period=20.0,
        )
        self.assertAlmostEqual(result["chamfer_location_m"], 0.0)
        self.assertAlmostEqual(result["chamfer_full_normalized"], 0.0)
        self.assertAlmostEqual(result["iou_1m"], 1.0)
        self.assertAlmostEqual(result["da_f1"], 1.0)
        self.assertAlmostEqual(result["mmd_location"], 0.0, places=10)
        self.assertAlmostEqual(result["mmd_rcs"], 0.0, places=10)
        self.assertAlmostEqual(result["mmd_velocity"], 0.0, places=10)

    def test_thinning_sampling_is_reproducible_and_in_bev(self) -> None:
        a = sample_predicted_ppp(
            self.sample,
            rng=np.random.default_rng(123),
            method="thinning",
        )
        b = sample_predicted_ppp(
            self.sample,
            rng=np.random.default_rng(123),
            method="thinning",
        )
        np.testing.assert_allclose(a.as_nx4(), b.as_nx4())
        if a.n_points:
            self.assertTrue(np.all((a.xy[:, 0] >= 0.0) & (a.xy[:, 0] < 1.0)))
            self.assertTrue(np.all((a.xy[:, 1] >= -1.0) & (a.xy[:, 1] < 0.0)))

    def test_block_sampling_is_reproducible(self) -> None:
        a = sample_predicted_ppp(
            self.sample,
            rng=np.random.default_rng(456),
            method="block",
            block_size=1,
            block_placement="weighted",
        )
        b = sample_predicted_ppp(
            self.sample,
            rng=np.random.default_rng(456),
            method="block",
            block_size=1,
            block_placement="weighted",
        )
        np.testing.assert_allclose(a.as_nx4(), b.as_nx4())

    def test_grid_samplers_match_expected_coordinate_convention(self) -> None:
        thinning_points, _ = sample_ppp_thinning_grid(
            np.array([[10.0, 0.0], [0.0, 0.0]]),
            rng=np.random.default_rng(7),
        )
        if len(thinning_points):
            self.assertTrue(np.all((thinning_points[:, 0] >= 0.0) & (thinning_points[:, 0] < 1.0)))
            self.assertTrue(np.all((thinning_points[:, 1] >= 0.0) & (thinning_points[:, 1] < 1.0)))

        block_points, _ = sample_ppp_block_grid(
            np.array([[100.0, 0.0], [0.0, 0.0]]),
            block_size=2,
            placement="weighted",
            rng=np.random.default_rng(8),
        )
        self.assertEqual(len(block_points), 1)
        self.assertTrue(0.0 <= block_points[0, 0] < 1.0)
        self.assertTrue(0.0 <= block_points[0, 1] < 1.0)

    def test_end_to_end_h5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_dir = root / "model" / "test"
            h5_dir.mkdir(parents=True)
            path = h5_dir / "rank00.h5"
            with h5py.File(path, "w") as h5f:
                h5f.attrs["bev_x_min"] = 0.0
                h5f.attrs["bev_x_max"] = 2.0
                h5f.attrs["bev_y_min"] = -1.0
                h5f.attrs["bev_y_max"] = 1.0
                h5f.attrs["bev_height"] = 2
                h5f.attrs["bev_width"] = 2
                h5f.attrs["source_prediction_domain"] = "test"
                grp = h5f.create_group("sample_000000")
                grp.attrs["video_name"] = "video"
                grp.attrs["video_frame_idx"] = 7
                grp.attrs["doppler_period"] = 20.0
                for name in (
                    "pred_lambda",
                    "pred_velocity",
                    "pred_rcs",
                    "gt_support",
                    "gt_velocity",
                    "gt_rcs",
                ):
                    grp.create_dataset(name, data=self.sample[name])
                grp.create_dataset("gt_points_xy", data=self.sample["gt_points_xy"])
                grp.create_dataset(
                    "gt_point_velocity",
                    data=self.sample["gt_point_velocity"],
                )
                grp.create_dataset("gt_point_rcs", data=self.sample["gt_point_rcs"])

            metrics = {
                "count": {"enabled": True},
                "likelihood": {
                    "enabled": True,
                    "sigma_max_m": 2.0,
                    "positive_sigma_floor_m": 0.01,
                },
                "marks": {"enabled": True},
                "radargen": {
                    "enabled": True,
                    "sampling_methods": ("thinning", "block"),
                    "num_draws": 2,
                    "seed": 1,
                    "block_size": 1,
                    "block_placement": "weighted",
                },
            }
            df_samples, df_summary = run_evaluation(
                {"model": str(h5_dir)},
                metrics_config=metrics,
                output_dir=root / "out",
                expected_geometry={
                    "bev_x_min": 0.0,
                    "bev_x_max": 2.0,
                    "bev_y_min": -1.0,
                    "bev_y_max": 1.0,
                    "bev_height": 2,
                    "bev_width": 2,
                },
                verbose=False,
            )
            self.assertEqual(len(df_samples), 1)
            self.assertEqual(len(df_summary), 1)
            self.assertTrue((root / "out" / "sample_metrics.csv").is_file())
            self.assertTrue((root / "out" / "model_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
