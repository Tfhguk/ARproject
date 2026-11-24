from pathlib import Path
from contextlib import ExitStack
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from depth_appearance import DepthAppearance
from surface_updates import SurfaceUpdates
from work_area import WorkArea, calibrate_area, calibration_pattern, projector_bounds


class WorkAreaTests(unittest.TestCase):
    def area(self):
        return WorkArea([[40, 24], [79, 24], [79, 55], [40, 55]], (120, 80), (200, 100), "stereo-hash")

    def test_small_zone_ignores_outside_motion_and_accepts_inside_changes(self):
        area = self.area()
        mask = area.mask((80, 120))  # Less than 20% of the camera image.
        gate = SurfaceUpdates(block=4, region=mask)
        original = np.ones((80, 120), np.float32)
        for stamp in (0, .5, 1):
            result = gate.observe(original, stamp)
        self.assertIsNotNone(result)
        self.assertTrue(np.isnan(result[~mask]).all())
        for index in range(8):
            outside = original.copy()
            outside[~mask] = .3 + index * .05
            self.assertIsNone(gate.observe(outside, 1.5 + index * .5))
        changed = original.copy()
        changed[32:48, 48:72] = .8
        for stamp in (5.5, 6, 6.5):
            result = gate.observe(changed, stamp)
        self.assertIsNotNone(result)
        self.assertEqual(gate.publications, 2)
        np.testing.assert_array_equal(original, np.ones_like(original))

    def test_outside_measurements_cannot_fill_a_completely_unknown_work_area(self):
        area = self.area()
        mask = area.mask((80, 120))
        depth = np.ones(mask.shape, np.float32)
        depth[mask] = np.nan
        display = DepthAppearance(fill="surface").process(area.restrict(depth), 1)
        area.clip_display(display)
        self.assertTrue(np.isnan(display.depth).all())
        self.assertFalse(display.measured.any())
        self.assertTrue(display.completed is None or not display.completed.any())

    def test_completion_and_all_masks_stay_inside_work_area(self):
        area = self.area()
        depth = np.ones((80, 120), np.float32)
        display = DepthAppearance(fill="surface").process(area.restrict(depth), 1)
        area.clip_display(display)
        outside = ~area.mask(depth.shape)
        self.assertTrue(np.isnan(display.depth[outside]).all())
        self.assertTrue((display.alpha[outside] == 0).all())
        for values in (display.measured, display.interpolated, display.held, display.completed):
            self.assertFalse(values[outside].any())
        self.assertEqual(area.statistics(display)["display_fraction"], 1)

    def test_small_footprint_with_a_hole_is_completed_using_its_own_measurements(self):
        area = WorkArea([[90, 60], [129, 60], [129, 99], [90, 99]], (320, 180), (200, 100), "hash")
        depth = np.ones((180, 320), np.float32)
        depth[70:90, 100:120] = np.nan
        display = area.process_display(DepthAppearance(fill="surface"), depth, 1)
        self.assertTrue(np.isfinite(display.depth[area.mask(depth.shape)]).all())
        self.assertTrue(display.completed[80, 110])
        self.assertTrue(np.isnan(display.depth[~area.mask(depth.shape)]).all())
        self.assertTrue(np.isnan(depth[80, 110]))  # Original missing measurement stays missing.

    def test_perspective_projection_handles_camera_rotation_and_resolution_change(self):
        # Projector top left is at the camera's bottom right after rig rotation.
        area = WorkArea([[95, 65], [15, 60], [20, 12], [100, 18]], (120, 80), (200, 100), "hash")
        colors = ((10, 30, 240), (20, 230, 40), (220, 40, 30), (0, 220, 230))
        for shape in ((80, 120), (160, 240)):
            image = np.zeros((*shape, 3), np.uint8)
            for point, color in zip(area.scaled_points(shape), colors):
                cv2.circle(image, tuple(np.rint(point).astype(int)), 7, color, -1)
            output = area.project(image)
            for (x, y), color in zip(((1, 1), (198, 1), (198, 98), (1, 98)), colors):
                np.testing.assert_allclose(output[y, x], color, atol=1)

    def test_saved_zone_rejects_a_different_stereo_calibration_or_output_size(self):
        area = self.area()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "area.json"
            area.save(path)
            loaded = WorkArea.load(path, (200, 100), "stereo-hash")
            np.testing.assert_array_equal(loaded.mask((80, 120)), area.mask((80, 120)))
            with self.assertRaises(ValueError):
                WorkArea.load(path, (200, 100), "other-calibration")
            with self.assertRaises(ValueError):
                WorkArea.load(path, (1280, 800), "stereo-hash")

    def test_invalid_corner_order_or_incomplete_field_cannot_be_saved(self):
        for points in ([[0, 0], [119, 79], [119, 0], [0, 79]],
                       [[0, 0], [125, 0], [119, 79], [0, 79]],
                       [[1, 1], [2, 1], [2, 2], [1, 2]],
                       [[0, 0], [119, 0], [119, 79], [np.nan, 79]]):
            with self.assertRaises(ValueError):
                WorkArea(points, (120, 80), (200, 100), "hash")

    def test_narrower_pattern_keeps_black_ends_without_changing_its_height(self):
        rectangle = projector_bounds((1280, 800), 83)
        image = calibration_pattern((1280, 800), rectangle)
        self.assertFalse(image[:, :83].any())
        self.assertFalse(image[:, 1197:].any())
        self.assertTrue((image[:, 640] == 255).all())
        self.assertEqual(rectangle, (83, 0, 1197, 800))
        self.assertAlmostEqual((1197 - 83) * 155 / 1280, 135, delta=.2)

    def test_field_is_trimmed_on_all_four_sides_and_remains_centred(self):
        rectangle = projector_bounds((1280, 800), 268, 40)
        self.assertEqual(rectangle, (268, 40, 1012, 760))
        self.assertAlmostEqual((1012 - 268) * 155 / 1280, 90, delta=.2)
        self.assertEqual((760 - 40) * 100 / 800, 90)
        self.assertEqual((268 + 1012, 40 + 760), (1280, 800))
        image = calibration_pattern((1280, 800), rectangle)
        self.assertFalse(image[:, :268].any())
        self.assertFalse(image[:, 1012:].any())
        self.assertFalse(image[:40].any())
        self.assertFalse(image[760:].any())
        self.assertTrue((image[40:760, 640] == 255).all())
        with self.assertRaises(ValueError):
            projector_bounds((1280, 800), 268, 400)

    def test_narrower_zone_stays_inside_its_projector_rectangle_after_save_and_load(self):
        rectangle = projector_bounds((1280, 800), 83)
        area = WorkArea([[0, 0], [119, 0], [119, 79], [0, 79]], (120, 80), (1280, 800), "hash", rectangle)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "area.json"
            area.save(path)
            loaded = WorkArea.load(path, (1280, 800), "hash", rectangle)
            self.assertEqual(loaded.projector_rect, rectangle)
            with self.assertRaises(ValueError):
                WorkArea.load(path, (1280, 800), "hash", projector_bounds((1280, 800)))
        image = np.full((80, 120, 3), 200, np.uint8)
        projected = loaded.project(image)
        self.assertFalse(projected[:, :83].any())
        self.assertFalse(projected[:, 1197:].any())
        self.assertTrue((projected[:, 84:1196] == 200).all())

    def calibration_ui(self, path, select):
        callbacks = []
        steps = [0]

        def key(delay):
            steps[0] += 1
            if select and steps[0] == 1:
                for x, y in ((40, 24), (79, 24), (79, 55), (40, 55)):
                    callbacks[0](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                return -1
            return 13 if select or steps[0] == 1 else 27

        with ExitStack() as stack:
            for name in ("namedWindow", "resizeWindow", "moveWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch.object(cv2, name))
            stack.enter_context(patch.object(cv2, "setMouseCallback", side_effect=lambda name, cb: callbacks.append(cb)))
            stack.enter_context(patch.object(cv2, "waitKey", key))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
            return calibrate_area(lambda: np.zeros((80, 120, 3), np.uint8), "projector", path,
                                  (200, 100), "stereo-hash")

    def test_four_clicks_and_enter_save_the_selected_camera_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "area.json"
            selected = self.calibration_ui(path, True)
            np.testing.assert_array_equal(selected.points, self.area().points)
            loaded = WorkArea.load(path, (200, 100), "stereo-hash")
            np.testing.assert_array_equal(loaded.points, self.area().points)

    def test_enter_without_corners_then_cancel_preserves_the_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "area.json"
            path.write_bytes(b"previous calibration")
            self.assertIsNone(self.calibration_ui(path, False))
            self.assertEqual(path.read_bytes(), b"previous calibration")


if __name__ == "__main__":
    unittest.main()
