"""Fast standard-library tests for the released method path only."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from magcal import MagCalConfig, MagCalDetector
from magcal.data import load_mat_scene
from magcal.evaluation import compare_scores
from magcal.magnitude import (
    calibrate_score,
    patchwise_magnitude,
    patchwise_same_kernel_rarity,
)
from magcal.preprocessing import minmax, pca_embedding, zscore_bands


def synthetic_scene() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(7)
    cube = rng.normal(size=(16, 16, 12)).astype(np.float64)
    cube[5:8, 9:12, :] += 2.0
    gt = np.zeros((16, 16), dtype=bool)
    gt[5:8, 9:12] = True
    return cube, gt


class MagCalSmokeTests(unittest.TestCase):
    def test_minmax_constant_is_zero(self):
        self.assertTrue(np.all(minmax(np.ones((4, 4))) == 0))

    def test_patchwise_magnitude_is_finite_and_bounded(self):
        cube, _ = synthetic_scene()
        features = pca_embedding(zscore_bands(cube), 3)
        prior, stats = patchwise_magnitude(features, core=4, halo=1)
        self.assertEqual(prior.shape, cube.shape[:2])
        self.assertTrue(np.isfinite(prior).all())
        self.assertGreaterEqual(float(prior.min()), 0)
        self.assertLessEqual(float(prior.max()), 1)
        self.assertEqual(stats["patch_count"], 16)

    def test_constant_prior_does_not_change_rank(self):
        rng = np.random.RandomState(3)
        score = rng.rand(8, 8)
        calibrated = calibrate_score(score, np.full((8, 8), 0.5))
        self.assertTrue(
            np.array_equal(np.argsort(score.ravel()), np.argsort(calibrated.ravel()))
        )

    def test_same_kernel_rarity_is_finite_and_bounded(self):
        cube, _ = synthetic_scene()
        features = pca_embedding(zscore_bands(cube), 3)
        prior = patchwise_same_kernel_rarity(features, core=4, halo=1)
        self.assertEqual(prior.shape, cube.shape[:2])
        self.assertTrue(np.isfinite(prior).all())
        self.assertGreaterEqual(float(prior.min()), 0.0)
        self.assertLessEqual(float(prior.max()), 1.0)

    def test_calibration_normalizes_host_before_multiplication(self):
        score = np.array([[2.0, 3.0], [5.0, 9.0]])
        prior = np.array([[1.0, 0.2], [0.7, 0.0]])
        expected = minmax(minmax(score) * (1.0 + prior))
        actual = calibrate_score(score, prior, alpha=1.0)
        self.assertTrue(np.allclose(actual, expected, atol=0.0, rtol=0.0))

    def test_pipeline_fits_without_ground_truth(self):
        cube, _ = synthetic_scene()
        config = MagCalConfig(
            selected_bands=8,
            pca_components=3,
            core=4,
            halo=1,
            layers=2,
            epochs=1,
            iterations=1,
            output_iteration=1,
            device="cpu",
        )
        maps = MagCalDetector(config).fit(cube).maps()
        self.assertEqual(set(maps), {"original", "magnitude_prior", "calibrated"})
        for value in maps.values():
            self.assertEqual(value.shape, cube.shape[:2])
            self.assertTrue(np.isfinite(value).all())
        self.assertGreaterEqual(float(maps["original"].min()), 0.0)
        self.assertLessEqual(float(maps["original"].max()), 1.0)

    def test_metrics_are_evaluation_only(self):
        cube, gt = synthetic_scene()
        score = cube.mean(axis=2)
        result = compare_scores(gt, score, score)
        self.assertEqual(result["delta"]["AP"], 0)
        self.assertEqual(result["delta"]["ROC_AUC"], 0)

    def test_mat_loader_accepts_optional_gt(self):
        cube, gt = synthetic_scene()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.mat"
            savemat(path, {"data": cube, "map": gt.astype(np.uint8)})
            scene = load_mat_scene(path)
        self.assertEqual(scene.cube.shape, cube.shape)
        self.assertTrue(np.array_equal(scene.gt, gt))

    def test_invalid_calibration_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_score(np.zeros((4, 4)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
