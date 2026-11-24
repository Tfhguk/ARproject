import unittest

import numpy as np

from depth_appearance import DepthAppearance
from depth_completion import complete_surface
from rendering import Renderer
from stereo_live import sample_depth


class SurfaceCompletionTests(unittest.TestCase):
    def test_large_holes_and_border_strip_fill_without_changing_known_samples(self):
        truth = np.full((130, 180), 0.8, np.float32)
        depth = truth.copy()
        depth[20:110, 50:130] = np.nan
        depth[:, :20] = np.nan
        original = depth.copy()
        result, estimated = complete_surface(depth)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, truth, atol=1e-6)
        np.testing.assert_array_equal(estimated, ~np.isfinite(depth))
        np.testing.assert_array_equal(depth, original)
        np.testing.assert_array_equal(result[~estimated], depth[~estimated])

    def test_camera_image_selects_the_correct_side_of_an_occlusion(self):
        depth = np.ones((80, 140), np.float32)
        depth[:, :70] = 0.6
        guide = np.full(depth.shape, 180, np.uint8)
        guide[:, :70] = 40
        depth[:, 40:90] = np.nan
        result, estimated = complete_surface(depth, guide)
        self.assertTrue(estimated[:, 40:90].all())
        self.assertAlmostEqual(float(result[40, 65]), 0.6, places=4)
        self.assertAlmostEqual(float(result[40, 75]), 1.0, places=4)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.6 - 1e-6)
        self.assertLessEqual(float(result.max()), 1.0 + 1e-6)

    def test_empty_and_insufficient_measurements_do_not_produce_a_surface(self):
        for count in (0, 20):
            depth = np.full((80, 120), np.nan, np.float32)
            depth[:count, :1] = 1
            result, estimated = complete_surface(depth)
            np.testing.assert_array_equal(result, depth)
            self.assertFalse(estimated.any())

    def test_surface_is_coloured_everywhere_but_measurement_readout_stays_missing(self):
        depth = np.ones((130, 180), np.float32)
        depth[20:110, 50:130] = np.nan
        result = DepthAppearance(fill="surface").process(depth, 1)
        self.assertTrue(np.isfinite(result.depth).all())
        colors = result.colorize(Renderer(0.3, 4))
        self.assertFalse(np.any(np.all(colors == 0, axis=2)))
        self.assertTrue(result.completed[65, 90])
        self.assertTrue(result.interpolated[65, 90])
        self.assertFalse(result.measured[65, 90])
        self.assertIsNone(sample_depth(depth, (0.5, 0.5)))

    def test_estimated_surface_never_becomes_old_depth_after_scene_changes(self):
        style = DepthAppearance(fill="surface")
        depth = np.ones((130, 180), np.float32)
        depth[20:110, 50:130] = np.nan
        style.process(depth, 1)
        result = style.process(depth * 0.6, 1.4)
        np.testing.assert_allclose(result.depth, 0.6, atol=1e-5)
        result = style.process(np.full_like(depth, np.nan), 1.6)
        self.assertTrue(np.isnan(result.depth).all())

    def test_raw_profile_still_shows_missing_measurements(self):
        depth = np.ones((80, 120), np.float32)
        depth[:, :30] = np.nan
        result = DepthAppearance("raw", fill="surface").process(depth, 1)
        np.testing.assert_array_equal(result.depth, depth)


if __name__ == "__main__":
    unittest.main()
