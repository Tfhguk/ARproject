from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from stereo_live import Frame, StereoEngine, choose_pair, fit_image, sample_depth
from stereo_calibrate import solve_calibration
from stereo_refine import refine_points


class StereoTests(unittest.TestCase):
    def test_live_defaults_follow_the_calibrated_camera_device_order(self):
        import sys
        from unittest.mock import patch
        import stereo_live
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.npz'
            np.savez(path, camera_indices=[1, 0])
            argv = ['stereo_live.py', '--calibration', str(path)]
            with patch.object(sys, 'argv', argv):
                args = stereo_live.arguments()
                self.assertEqual((args.left, args.right), (1, 0))
            with patch.object(sys, 'argv', argv + ['--left', '0', '--right', '1']):
                args = stereo_live.arguments()
                self.assertEqual((args.left, args.right), (0, 1))

    def test_calibration_requires_measured_square_before_opening_cameras(self):
        from contextlib import redirect_stderr
        from io import StringIO
        import sys
        from unittest.mock import patch
        import stereo_calibrate
        with patch.object(sys, "argv", ["stereo_calibrate.py", "--auto"]), \
                patch.object(stereo_calibrate, "collect") as capture, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                stereo_calibrate.main()
            self.assertEqual(error.exception.code, 2)
            capture.assert_not_called()

    def test_near_object_requires_more_than_legacy_disparity_range(self):
        rng = np.random.default_rng(107)
        left = cv2.GaussianBlur(rng.integers(0, 256, (720, 1280), dtype=np.uint8), (3, 3), 0.7)
        right = np.zeros_like(left)
        right[:, :-192] = left[:, 192:]
        pair = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (left, right)]
        intrinsic = np.float64([[900, 0, 640], [0, 900, 360], [0, 0, 1]])
        expected = 900 * 0.1 / 192
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.npz"
            np.savez(path, K1=intrinsic, K2=intrinsic, D1=np.zeros(5), D2=np.zeros(5),
                     R=np.eye(3), T=np.float64([[-0.1], [0], [0]]), capture_size=[1280, 720])
            for size in ((640, 360), (1280, 720)):
                legacy = StereoEngine(size, calibration=path, disparities=64, lr_check=False)
                improved = StereoEngine(size, calibration=path)
                self.assertGreater(legacy.effective_near, expected)
                self.assertLess(improved.effective_near, expected)
                _, depth, valid = improved.compute(*pair)
                self.assertGreater(valid, 0.15)
                self.assertAlmostEqual(float(np.nanmedian(depth)), expected, delta=0.015)

    def test_reverse_check_rejects_inconsistent_and_missing_matches(self):
        engine = StereoEngine((64, 32))
        forward = np.full((32, 64), 2 * 16, np.int16)
        reverse = -forward.copy()
        reverse[12, 18] = 0  # left x=20 maps here, but does not map back.
        reverse[12, 19] = (engine.reverse_minimum - 1) * 16

        class FixedMatcher:
            def __init__(self, values):
                self.values = values

            def compute(self, left, right):
                padding = (left.shape[1] - self.values.shape[1]) // 2
                return np.pad(self.values, ((0, 0), (padding, padding)), constant_values=-32000)

        engine.matcher = FixedMatcher(forward)
        engine.reverse_matcher = FixedMatcher(reverse)
        image = np.zeros((32, 64, 3), np.uint8)
        colors, _, _ = engine.compute(image, image)
        self.assertFalse(colors[:, :2].any())
        self.assertFalse(colors[12, 20].any())
        self.assertFalse(colors[12, 21].any())
        self.assertTrue(colors[12, 22].any())

    def test_depth_readout_ignores_missing_values_and_is_bounded(self):
        self.assertIsNone(sample_depth(None))
        depth = np.full((10, 10), np.nan, np.float32)
        self.assertIsNone(sample_depth(depth))
        depth[:4, :4] = 1.25
        depth[0, 0] = np.inf
        self.assertAlmostEqual(sample_depth(depth, (0, 0)), 1.25)
        self.assertIsNone(sample_depth(depth, (1, 1)))
        depth[5:8, 5:8] = 2
        depth[6, 6] = 100  # A single bad disparity must not dominate readout.
        self.assertEqual(sample_depth(depth, (0.7, 0.7), radius=1), 2)

    def test_solver_recovers_known_stereo_baseline(self):
        rng = np.random.default_rng(73)
        intrinsic = np.float64([[900, 0, 640], [0, 900, 360], [0, 0, 1]])
        board = np.zeros((54, 3), np.float32)
        board[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * 0.025
        rotation = cv2.Rodrigues(np.float64([0.005, 0.01, 0]))[0]
        translation = np.float64([[-0.12], [0.001], [0.001]])
        objects, left, right = [], [], []
        for _ in range(14):
            rvec = rng.uniform(-0.4, 0.4, (3, 1))
            tvec = np.float64([[rng.uniform(-0.2, 0.05)], [rng.uniform(-0.15, 0.05)], [rng.uniform(0.8, 1.6)]])
            a = cv2.projectPoints(board, rvec, tvec, intrinsic, np.zeros(5))[0]
            right_rvec = cv2.Rodrigues(rotation @ cv2.Rodrigues(rvec)[0])[0]
            b = cv2.projectPoints(board, right_rvec, rotation @ tvec + translation, intrinsic, np.zeros(5))[0]
            objects.append(board.copy())
            left.append(a)
            right.append(b)
        arrays, quality = solve_calibration(objects, left, right)
        self.assertAlmostEqual(quality["baseline_m"], float(np.linalg.norm(translation)), delta=0.002)
        self.assertLess(quality["stereo_rms_px"], 0.1)
        self.assertLess(max(quality["rectified_y_rms_px"]), 0.1)
        self.assertTrue(quality["horizontal_rig"])

        # Simulate two stereo captures taken while the board was moving:
        # the right camera sees a vertically shifted board in those views.
        damaged = [points.copy() for points in right]
        damaged[0][:, :, 1] += 15
        damaged[3][:, :, 1] -= 10
        recovered, report = refine_points(objects, left, damaged)
        self.assertTrue(report["accepted"], report)
        self.assertGreaterEqual(report["views"], 12)
        self.assertEqual({item["view"] for item in report["rejected_views"]}, {0, 3})
        self.assertAlmostEqual(report["baseline_m"], float(np.linalg.norm(translation)), delta=0.002)

    def test_pairing_is_fresh_bounded_and_does_not_reuse_frames(self):
        image = np.zeros((1, 1, 3), np.uint8)
        left = [Frame(1, 9.8, image), Frame(2, 9.95, image)]
        right = [Frame(1, 9.805, image), Frame(2, 9.96, image)]
        pair = choose_pair(left, right, max_skew=0.02, now=10)
        self.assertEqual((pair[0].sequence, pair[1].sequence), (2, 2))
        self.assertIsNone(choose_pair(left, right, previous=(2, 2), now=10))
        self.assertIsNone(choose_pair(left, right, now=11))
        self.assertIsNone(choose_pair(left[-1:], right[-1:], max_skew=0.001, now=10))

    def test_depth_scale_survives_resolution_change_and_camera_order(self):
        rng = np.random.default_rng(29)
        left = cv2.GaussianBlur(rng.integers(0, 256, (720, 1280), dtype=np.uint8), (3, 3), 0.7)
        intrinsic = np.float64([[900, 0, 640], [0, 900, 360], [0, 0, 1]])
        with tempfile.TemporaryDirectory() as folder:
            for sign in (1, -1):
                path = Path(folder) / "stereo.npz"
                np.savez(path, K1=intrinsic, K2=intrinsic, D1=np.zeros(5), D2=np.zeros(5),
                         R=np.eye(3), T=np.float64([-0.1 * sign, 0, 0]), capture_size=[1280, 720])
                right = np.zeros_like(left)
                if sign == 1:
                    right[:, :-32] = left[:, 32:]
                else:
                    right[:, 32:] = left[:, :-32]
                pair = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (left, right)]
                for size in ((640, 360), (1280, 720)):
                    engine = StereoEngine(size, calibration=path)
                    colors, depth, valid = engine.compute(*pair)
                    with self.subTest(sign=sign, size=size):
                        self.assertEqual(colors.shape[:2], size[::-1])
                        self.assertGreater(valid, 0.25)
                        self.assertAlmostEqual(float(np.nanmedian(depth)), 900 * 0.1 / 32, delta=0.05)
                        self.assertTrue(np.all(depth[np.isfinite(depth)] > 0))

    def test_uncalibrated_preview_has_no_metric_depth(self):
        image = np.zeros((360, 640, 3), np.uint8)
        colors, depth, valid = StereoEngine((640, 360)).compute(image, image)
        self.assertIsNone(depth)
        self.assertEqual(valid, 0)
        self.assertFalse(colors.any())

    def test_projector_preserves_aspect_ratio(self):
        image = np.full((360, 640, 3), 255, np.uint8)
        output = fit_image(image, (1280, 800))
        self.assertEqual(output.shape, (800, 1280, 3))
        self.assertFalse(output[:40].any())
        self.assertFalse(output[-40:].any())
        self.assertTrue((output[40:760] == 255).all())


if __name__ == "__main__":
    unittest.main()
