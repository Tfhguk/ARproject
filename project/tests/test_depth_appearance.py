import unittest

import numpy as np

from depth_appearance import DepthAppearance
from rendering import Renderer


class DepthAppearanceTests(unittest.TestCase):
    def test_spatial_noise_reduces_without_changing_measurements_or_blurring_step(self):
        rng = np.random.default_rng(42)
        truth = np.ones((80, 120), np.float32)
        truth[:, :60] = 0.65
        depth = truth + rng.normal(0, 0.007, truth.shape).astype(np.float32)
        original = depth.copy()
        result = DepthAppearance("smooth").process(depth, 1)
        np.testing.assert_array_equal(depth, original)
        self.assertLess(np.std(result.depth[10:70, 10:50]), np.std(depth[10:70, 10:50]) * 0.6)
        self.assertLess(abs(float(np.median(result.depth[:, 59])) - 0.65), 0.008)
        self.assertLess(abs(float(np.median(result.depth[:, 60])) - 1.0), 0.008)

    def test_small_enclosed_hole_fills_but_border_gaps_and_large_holes_do_not(self):
        depth = np.ones((130, 180), np.float32)
        depth[30:33, 45:48] = np.nan
        depth[45:115, 70:150] = np.nan
        depth[:, :10] = np.nan
        result = DepthAppearance().process(depth, 1)
        self.assertAlmostEqual(float(result.depth[31, 46]), 1.0, places=5)
        self.assertTrue(result.interpolated[31, 46])
        self.assertFalse(result.measured[31, 46])
        self.assertTrue(np.isnan(result.depth[80, 110]))
        self.assertTrue(np.isnan(result.depth[:, :10]).all())
        self.assertTrue(np.isnan(depth[31, 46]))

    def test_hole_at_foreground_edge_is_not_filled_with_fictitious_middle_depth(self):
        depth = np.ones((80, 120), np.float32)
        depth[:, :60] = 0.6
        depth[32:36, 58:62] = np.nan
        result = DepthAppearance().process(depth, 1)
        patch = result.depth[32:36, 58:62]
        finite = patch[np.isfinite(patch)]
        self.assertFalse(np.any((finite > 0.62) & (finite < 0.98)))
        self.assertTrue(np.isnan(result.depth[33, 59]) or abs(result.depth[33, 59] - 0.6) < 0.01)

    def test_narrow_connected_gaps_fill_but_camera_image_edges_can_veto_them(self):
        depth = np.ones((80, 120), np.float32)
        depth[:, 58:63] = np.nan
        guide = np.full(depth.shape, 100, np.uint8)
        result = DepthAppearance().process(depth, 1, guide)
        self.assertAlmostEqual(float(result.depth[40, 60]), 1.0, places=5)
        self.assertTrue(result.interpolated[40, 60])
        # A newly visible dark object in this gap must not inherit the background.
        guide[:, 58:63] = 0
        result = DepthAppearance().process(depth, 1, guide)
        self.assertTrue(np.isnan(result.depth[40, 60]))

    def test_isolated_wrong_depth_is_repaired_and_marked_as_interpolation(self):
        depth = np.ones((80, 120), np.float32)
        depth[40, 60] = 3.0
        result = DepthAppearance().process(depth, 1)
        self.assertAlmostEqual(float(result.depth[40, 60]), 1.0, places=4)
        self.assertTrue(result.interpolated[40, 60])
        self.assertEqual(depth[40, 60], 3.0)

    def test_temporal_filter_settles_jitter_but_resets_for_new_object_or_missing_data(self):
        style = DepthAppearance()
        raw, filtered = [], []
        for index in range(30):
            value = 1 + (0.009 if index % 2 else -0.009)
            result = style.process(np.full((40, 60), value, np.float32), index * 0.05)
            if index >= 10:
                raw.append(value)
                filtered.append(float(result.depth[20, 30]))
        self.assertLess(np.std(filtered), np.std(raw) * 0.4)
        result = style.process(np.full((40, 60), 0.65, np.float32), 1.5)
        self.assertAlmostEqual(float(result.depth[20, 30]), 0.65, places=5)
        result = style.process(np.full((40, 60), np.nan, np.float32), 1.55)
        self.assertTrue(np.isnan(result.depth).all())
        result = style.process(np.full((40, 60), 0.66, np.float32), 1.6)
        self.assertAlmostEqual(float(result.depth[20, 30]), 0.66, places=5)

    def test_camera_image_changes_and_frame_gaps_reset_temporal_blending(self):
        style = DepthAppearance()
        depth = np.ones((40, 60), np.float32)
        guide = np.zeros(depth.shape, np.uint8)
        style.process(depth, 1, guide)
        result = style.process(depth * 1.02, 1.05, guide + 50)
        self.assertAlmostEqual(float(result.depth[20, 30]), 1.02, places=5)
        result = style.process(depth * 1.03, 3.0, guide + 50)
        self.assertAlmostEqual(float(result.depth[20, 30]), 1.03, places=5)

    def test_raw_mode_and_rendering_preserve_unknown_areas(self):
        depth = np.ones((40, 60), np.float32)
        depth[:, :10] = np.nan
        raw = DepthAppearance("raw").process(depth, 1)
        np.testing.assert_array_equal(raw.depth, depth)
        self.assertFalse(raw.interpolated.any())
        smooth = DepthAppearance().process(depth, 1)
        colors = smooth.colorize(Renderer(0.3, 4))
        self.assertFalse(colors[:, :10].any())
        self.assertTrue(colors[20, 30].any())
        self.assertEqual(colors.dtype, np.uint8)

    def test_display_contours_round_corners_without_expanding_into_unknown_space(self):
        depth = np.full((80, 120), np.nan, np.float32)
        depth[20:60, 20:100] = 1
        original = depth.copy()
        result = DepthAppearance().process(depth, 1)
        self.assertTrue(np.isnan(result.depth[20, 20]))
        self.assertAlmostEqual(float(result.depth[20, 60]), 1.0, places=5)
        self.assertLess(float(result.alpha[20, 60]), 1)
        self.assertFalse(np.isfinite(result.depth[~np.isfinite(depth)]).any())
        np.testing.assert_array_equal(depth, original)

    def test_resolution_scaling_preserves_a_flat_surface_and_bounded_output(self):
        for shape, scale in (((90, 140), 1), ((180, 280), 2)):
            with self.subTest(scale=scale):
                depth = np.full(shape, 0.8, np.float32)
                depth[30 * scale:33 * scale, 40 * scale:43 * scale] = np.nan
                result = DepthAppearance(scale=scale).process(depth, 1)
                self.assertTrue(np.isfinite(result.depth).all())
                np.testing.assert_allclose(result.depth, 0.8, atol=1e-5)

    def test_internal_colour_boundaries_round_without_inventing_a_middle_surface(self):
        depth = np.ones((100, 140), np.float32)
        depth[25:75, 35:105] = 0.6
        original = depth.copy()
        result = DepthAppearance().process(depth, 1)
        # The corner is rounded into the background; the straight middle of
        # an edge and the interior retain their own surface's depth.
        self.assertGreater(float(result.depth[25, 35]), 0.98)
        self.assertLess(float(result.depth[25, 70]), 0.62)
        self.assertLess(float(result.depth[50, 70]), 0.62)
        self.assertFalse(np.any((result.depth > 0.62) & (result.depth < 0.98)))
        np.testing.assert_array_equal(depth, original)

    def test_black_hole_corners_round_but_its_unknown_centre_stays_black(self):
        depth = np.ones((130, 180), np.float32)
        depth[30:100, 50:130] = np.nan
        result = DepthAppearance().process(depth, 1)
        self.assertAlmostEqual(float(result.depth[30, 50]), 1, places=5)
        self.assertTrue(result.interpolated[30, 50])
        self.assertTrue(np.isnan(result.depth[65, 90]))
        self.assertFalse(result.measured[30, 50])

    def test_static_hole_uses_short_history_but_held_data_expires(self):
        style = DepthAppearance()
        depth = np.ones((100, 140), np.float32)
        guide = np.full(depth.shape, 100, np.uint8)
        guide[30:65, 50:85] = 30  # A static printed pattern, not a new object.
        style.process(depth, 1, guide)
        missing = depth.copy()
        missing[30:65, 50:85] = np.nan  # Guide contrast vetoes spatial filling.
        result = style.process(missing, 1.6, guide)
        self.assertAlmostEqual(float(result.depth[47, 67]), 1, places=5)
        self.assertTrue(result.held[47, 67])
        self.assertTrue(result.interpolated[47, 67])
        self.assertFalse(result.measured[47, 67])
        result = style.process(missing, 2.2, guide)
        self.assertTrue(result.held[47, 67])
        result = style.process(missing, 2.8, guide)
        self.assertTrue(np.isnan(result.depth[47, 67]))
        self.assertFalse(result.held[47, 67])

    def test_motion_no_guide_and_large_dropouts_cannot_hold_the_old_surface(self):
        depth = np.ones((100, 140), np.float32)
        guide = np.full(depth.shape, 100, np.uint8)
        missing = depth.copy()
        missing[30:65, 50:85] = np.nan
        moving = guide.copy()
        moving[30:65, 50:85] = 30
        for first_guide, next_guide in ((guide, moving), (None, None)):
            style = DepthAppearance()
            style.process(depth, 1, first_guide)
            result = style.process(missing, 1.6, next_guide)
            if next_guide is not None:
                self.assertTrue(np.isnan(result.depth[47, 67]))
            self.assertFalse(result.held.any())
        style = DepthAppearance()
        style.process(depth, 1, guide)
        missing[10:90, 20:120] = np.nan
        result = style.process(missing, 1.6, guide)
        self.assertFalse(result.held.any())
        # Complete loss of depth always clears history, even with a static guide.
        result = style.process(np.full_like(depth, np.nan), 1.8, guide)
        self.assertTrue(np.isnan(result.depth).all())

    def test_new_enclosed_dark_object_vetoes_spatial_background_filling(self):
        depth = np.ones((100, 140), np.float32)
        depth[30:65, 50:85] = np.nan
        guide = np.full(depth.shape, 100, np.uint8)
        guide[30:65, 50:85] = 0
        result = DepthAppearance().process(depth, 1, guide)
        self.assertTrue(np.isnan(result.depth[47, 67]))
        self.assertFalse(result.interpolated[47, 67])


if __name__ == "__main__":
    unittest.main()
