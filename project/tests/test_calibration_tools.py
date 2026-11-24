import argparse
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration_tools import BoardStability, assess_calibration, save_npz_atomic, summarize_distance
from stereo_live import Frame


class CalibrationQualityTests(unittest.TestCase):
    def test_real_chessboard_detection_reaches_first_capture_and_saves_observations(self):
        import stereo_calibrate
        image = np.full((720, 1280, 3), 255, np.uint8)
        for row in range(7):
            for column in range(10):
                if (row + column) % 2 == 0:
                    x, y = 400 + column * 40, 180 + row * 40
                    image[y:y + 40, x:x + 40] = 0
        found, corners = stereo_calibrate.find_corners(image, (9, 6))
        self.assertTrue(found)
        self.assertEqual(corners.shape, (54, 1, 2))
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            directory = Path(temporary)
            args = argparse.Namespace(directory=directory, left=0, right=1, focus_left=0, focus_right=0,
                                      columns=9, rows=6, square_mm=25, baseline_mm=120, views=30,
                                      hold_seconds=0.8, still_px=1.5, auto=True, output=directory / "active.npz")
            frames = [Frame(i, stamp, image) for i, stamp in enumerate((100.0, 100.4, 100.9, 101.1), 1)]
            camera = stack.enter_context(patch.object(stereo_calibrate, "Camera"))
            camera.return_value.info = {"focus": 0.0}
            stack.enter_context(patch.object(stereo_calibrate, "dpi_awareness"))
            stack.enter_context(patch.object(stereo_calibrate, "choose_pair", side_effect=[(frame, frame) for frame in frames]))
            stack.enter_context(patch.object(stereo_calibrate.time, "perf_counter", return_value=100.0))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(stereo_calibrate.cv2, name))
            stack.enter_context(patch.object(stereo_calibrate.cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch.object(stereo_calibrate.cv2, "waitKey", side_effect=[-1, -1, -1, 27]))
            with redirect_stdout(StringIO()):
                stereo_calibrate.collect(args)
            session = next(directory.glob("session_*"))
            with np.load(session / "observations.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["left_points"].shape, (1, 54, 1, 2))
                np.testing.assert_allclose(saved["left_points"][0], corners)
                self.assertEqual(float(saved["square_mm"]), 25)
                self.assertEqual(float(saved["measured_baseline_mm"]), 120)
            self.assertTrue((session / "view_01_left.png").is_file())
            self.assertTrue((session / "view_01_right.png").is_file())
            manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["views"], 1)
            self.assertFalse(args.output.exists())

    def test_waiting_for_board_does_not_trigger_camera_timeout(self):
        import stereo_calibrate
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            directory = Path(temporary)
            args = argparse.Namespace(directory=directory, left=0, right=1, focus_left=0, focus_right=0,
                                      columns=9, rows=6, square_mm=25, baseline_mm=120, views=30,
                                      hold_seconds=0.8, still_px=1.5, auto=True, output=directory / "active.npz")
            frame = Frame(1, 35.0, np.zeros((32, 64, 3), np.uint8))
            camera = stack.enter_context(patch.object(stereo_calibrate, "Camera"))
            stack.enter_context(patch.object(stereo_calibrate, "dpi_awareness"))
            stack.enter_context(patch.object(stereo_calibrate, "choose_pair", side_effect=[(frame, frame), None, None]))
            stack.enter_context(patch.object(stereo_calibrate, "find_corners", return_value=(False, None)))
            stack.enter_context(patch.object(stereo_calibrate.time, "perf_counter", side_effect=[0.0, 35.0, 35.05]))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(stereo_calibrate.cv2, name))
            stack.enter_context(patch.object(stereo_calibrate.cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch.object(stereo_calibrate.cv2, "waitKey", side_effect=[-1, -1, 27]))
            with redirect_stdout(StringIO()):
                stereo_calibrate.collect(args)
            self.assertEqual(camera.return_value.close.call_count, 2)
            session = next(directory.glob("session_*"))
            manifest = json.loads((session / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "incomplete")
            self.assertFalse(args.output.exists())

    def test_board_must_be_stationary_in_both_views_and_not_drift_slowly(self):
        corners = np.zeros((54, 1, 2), np.float32)
        gate = BoardStability(hold_seconds=0.8, tolerance_px=1.5)
        for timestamp, offset in ((0, 0), (0.3, 1), (0.6, 2), (0.9, 3), (1.2, 4)):
            self.assertFalse(gate.update(corners, corners + offset, timestamp))
        self.assertFalse(gate.update(corners, corners + 4, 1.5))
        self.assertTrue(gate.update(corners, corners + 4, 2.05))
        self.assertFalse(gate.update(corners + 10, corners + 4, 2.2))
        self.assertFalse(gate.update(None, corners, 2.4))
        self.assertFalse(gate.update(corners, corners, 3.4))
        self.assertFalse(gate.update(corners, corners, 4.3))  # need >= 3 observations
        self.assertTrue(gate.update(corners, corners, 4.35))
        self.assertFalse(gate.update(corners, corners, 6.0))  # stale observations cannot bridge a gap

    def test_low_image_error_does_not_override_bad_physical_baseline(self):
        quality = {"rectified_y_rms_px": [0.3, 0.5], "stereo_rms_px": 0.4,
                   "baseline_m": 0.130, "horizontal_rig": True}
        result = assess_calibration(quality, baseline_mm=120)
        self.assertTrue(result["image_geometry_passed"])
        self.assertFalse(result["accepted"])
        self.assertFalse(result["baseline_check_passed"])
        quality["baseline_m"] = 0.121
        result = assess_calibration(quality, baseline_mm=120)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["metric_validation"], "pending_known_distance_checks")
        quality["rectified_y_rms_px"] = [np.nan]
        self.assertFalse(assess_calibration(quality)["accepted"])

    def test_distance_check_catches_scale_error_jitter_and_missing_frames(self):
        wrong = summarize_distance([0.56] * 20, 0.45, 20)
        self.assertFalse(wrong["passed_this_point"])
        self.assertAlmostEqual(wrong["error_percent"], 24.4444444)
        self.assertTrue(summarize_distance([0.455] * 20, 0.45, 20)["passed_this_point"])
        self.assertFalse(summarize_distance([0.45] * 10, 0.45, 20)["passed_this_point"])
        self.assertFalse(summarize_distance([0.35, 0.55] * 10, 0.45, 20)["passed_this_point"])
        self.assertFalse(summarize_distance([], 0.45, 20)["passed_this_point"])

    def test_replacing_calibration_preserves_previous_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "stereo.npz"
            self.assertIsNone(save_npz_atomic(destination, {"T": np.array([1])}, backup=True))
            backup = save_npz_atomic(destination, {"T": np.array([2])}, backup=True)
            with np.load(backup, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["T"], [1])
            with np.load(destination, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["T"], [2])

    def test_refinement_uses_exact_session_observations_and_preserves_active_if_baseline_bad(self):
        import stereo_refine
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            active = folder / "active.npz"
            save_npz_atomic(active, {"old": np.array([123])})
            objects = np.zeros((12, 54, 3), np.float32)
            points = np.zeros((12, 54, 1, 2), np.float32)
            save_npz_atomic(folder / "observations.npz", {
                "object_points": objects, "left_points": points, "right_points": points,
                "capture_size": [1280, 720], "board_inner_corners": [9, 6], "square_mm": 25.0,
                "camera_indices": [0, 1], "focus_values": [0, 0], "measured_baseline_mm": 120.0})
            quality = {"rectified_y_rms_px": [0.3] * 12, "stereo_rms_px": 0.4,
                       "baseline_m": 0.130, "horizontal_rig": True, "accepted": True}
            argv = ["stereo_refine.py", "--directory", str(folder), "--square-mm", "25", "--output", str(active)]
            with patch.object(sys, "argv", argv), patch.object(stereo_refine, "refine_points", return_value=({}, quality)) as solve, \
                    patch.object(stereo_refine, "find_corners") as detect, redirect_stdout(StringIO()):
                self.assertEqual(stereo_refine.main(), 2)
            detect.assert_not_called()
            np.testing.assert_array_equal(solve.call_args.args[1], points)
            with np.load(active, allow_pickle=False) as saved:
                self.assertEqual(saved.files, ["old"])
            report = json.loads((folder / "refinement_quality.json").read_text(encoding="utf-8"))
            self.assertFalse(report["accepted"])
            self.assertTrue((folder / "active.candidate.npz").is_file())

    def test_measured_square_correction_re_solves_geometry_and_preserves_original_capture(self):
        import stereo_refine
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            grid = np.zeros((54, 3), np.float32)
            grid[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * 0.025
            objects = np.repeat(grid[None], 12, axis=0)
            points = np.zeros((12, 54, 1, 2), np.float32)
            source = folder / "observations.npz"
            save_npz_atomic(source, {
                "object_points": objects, "left_points": points, "right_points": points,
                "capture_size": [1280, 720], "board_inner_corners": [9, 6], "square_mm": 25.0,
                "camera_indices": [0, 1], "focus_values": [0, 0], "measured_baseline_mm": 120.0})
            original = source.read_bytes()
            output = folder / "corrected.npz"
            quality = {"rectified_y_rms_px": [0.3] * 12, "stereo_rms_px": 0.4,
                       "baseline_m": 0.1284, "horizontal_rig": True, "accepted": True}
            argv = ["stereo_refine.py", "--directory", str(folder), "--square-mm", "24.5",
                    "--baseline-mm", "131", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(stereo_refine, "refine_points") as solve:
                with self.assertRaisesRegex(ValueError, "square-mm differs"):
                    stereo_refine.main()
                solve.assert_not_called()
            with patch.object(sys, "argv", argv + ["--correct-square-size"]), \
                    patch.object(stereo_refine, "refine_points", return_value=({}, quality)) as solve, redirect_stdout(StringIO()):
                self.assertEqual(stereo_refine.main(), 0)
            np.testing.assert_allclose(np.asarray(solve.call_args.args[0]), objects * 0.98)
            np.testing.assert_array_equal(solve.call_args.args[1], points)
            self.assertEqual(source.read_bytes(), original)
            with np.load(output, allow_pickle=False) as saved:
                self.assertEqual(float(saved["square_mm"]), 24.5)
                self.assertEqual(float(saved["source_square_mm"]), 25)
                self.assertEqual(float(saved["measured_baseline_mm"]), 131)
                self.assertFalse(bool(saved["metric_validated"]))
            report = json.loads((folder / "refinement_quality.json").read_text(encoding="utf-8"))
            self.assertTrue(report["accepted"])
            self.assertEqual(report["square_size_correction"]["object_point_scale_factor"], 0.98)


if __name__ == "__main__":
    unittest.main()
