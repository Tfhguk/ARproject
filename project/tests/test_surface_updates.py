from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from depth_appearance import DepthAppearance
import stereo_live
from surface_updates import SurfaceUpdates
from work_area import WorkArea, projector_bounds
from surface_geometry import FloorPlane


class SurfaceUpdateTests(unittest.TestCase):
    def setUp(self):
        self.base = np.ones((80, 120), np.float32)

    def initial(self, gate, depth=None):
        depth = self.base if depth is None else depth
        self.assertIsNone(gate.observe(depth, 0))
        self.assertIsNone(gate.observe(depth, 0.5))
        result = gate.observe(depth, 1)
        self.assertIsNotNone(result)
        return result

    def test_startup_uses_three_measured_frames_and_does_not_modify_inputs(self):
        gate = SurfaceUpdates()
        original = self.base.copy()
        self.initial(gate)
        np.testing.assert_array_equal(self.base, original)
        self.assertEqual(gate.publications, 1)

    def test_stationary_noise_and_missing_pixels_never_repaint_the_saved_map(self):
        gate = SurfaceUpdates()
        saved = self.initial(gate).copy()
        rng = np.random.default_rng(8)
        for index in range(20):
            noisy = self.base + rng.normal(0, 0.004, self.base.shape).astype(np.float32)
            noisy[rng.random(noisy.shape) < 0.15] = np.nan
            noisy[rng.random(noisy.shape) < 0.01] = 1.4
            self.assertIsNone(gate.observe(noisy, 1.5 + index * 0.5))
        np.testing.assert_array_equal(gate.published, saved)
        self.assertEqual(gate.publications, 1)

    def test_persistent_local_change_is_confirmed_and_then_held(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        changed = self.base.copy()
        changed[20:60, 30:80] = 0.94
        self.assertIsNone(gate.observe(changed, 1.5))
        self.assertIsNone(gate.observe(changed, 2))
        result = gate.observe(changed, 2.5)
        np.testing.assert_array_equal(result, changed)
        self.assertEqual(gate.publications, 2)
        self.assertIsNone(gate.observe(changed, 3))

    def test_transient_hand_is_cancelled_when_original_surface_returns(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        hand = self.base.copy()
        hand[15:60, 20:80] = 0.6
        self.assertIsNone(gate.observe(hand, 1.5))
        for stamp in (2, 2.5, 3):
            self.assertIsNone(gate.observe(self.base, stamp))
        self.assertEqual(gate.publications, 1)

    def test_continuous_motion_publishes_latest_depth_once_per_second(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        publications = []
        for index in range(10):
            stamp = 1.5 + index * .5
            depth = self.base.copy()
            depth[20:60, 30:80] = .95 - index * .05
            result = gate.observe(depth, stamp)
            if result is not None:
                publications.append(stamp)
                np.testing.assert_array_equal(result, depth)
        self.assertEqual(publications, [2, 3, 4, 5, 6])
        # Once movement ends, no quiet timer or repeated publication is needed.
        for stamp in np.arange(6.5, 11, .5):
            self.assertIsNone(gate.observe(depth, float(stamp)))

    def test_lateral_motion_does_not_need_one_region_changed_in_all_three_frames(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        publications = []
        for index in range(10):
            stamp = 1.5 + index * .5
            depth = self.base.copy()
            x = (8, 48, 88)[index % 3]
            depth[16:64, x:x + 32] = .6
            result = gate.observe(depth, stamp)
            if result is not None:
                publications.append(stamp)
                np.testing.assert_array_equal(result, depth)  # No old object positions.
        self.assertEqual(publications, [2, 3, 4, 5, 6])

    def test_moving_scene_can_publish_its_first_map_without_stopping(self):
        gate = SurfaceUpdates()
        for stamp, height in ((0, .9), (.5, .8)):
            self.assertIsNone(gate.observe(self.base * height, stamp))
        latest = self.base * .7
        np.testing.assert_array_equal(gate.observe(latest, 1), latest)

    def test_compare_against_published_surface_including_accumulated_small_changes(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        for stamp, shift in ((1.5, .005), (2, .01), (2.5, .015), (3, .025), (3.5, .025)):
            self.assertIsNone(gate.observe(self.base - shift, stamp))
        self.assertIsNotNone(gate.observe(self.base - .025, 4))

    def test_dropouts_duplicate_timestamps_and_long_gaps_cannot_confirm_a_change(self):
        gate = SurfaceUpdates()
        saved = self.initial(gate).copy()
        changed = self.base * .9
        self.assertIsNone(gate.observe(changed, 1.5))
        for _ in range(5):
            self.assertIsNone(gate.observe(changed, 1.5))
        self.assertIsNone(gate.observe(np.full_like(changed, np.nan), 2))
        np.testing.assert_array_equal(gate.published, saved)
        self.assertIsNone(gate.observe(changed, 2.5))
        self.assertIsNone(gate.observe(changed, 10))
        self.assertIsNone(gate.observe(changed, 10.5))
        self.assertIsNotNone(gate.observe(changed, 11))

    def test_latest_snapshot_preserves_unknown_pixels_instead_of_inventing_measurements(self):
        gate = SurfaceUpdates()
        depth = self.base.copy()
        depth[20:50, 30:60] = np.nan
        result = self.initial(gate, depth)
        self.assertTrue(np.isnan(result[20:50, 30:60]).all())
        self.assertTrue(np.isnan(gate.published[20:50, 30:60]).all())

    def test_latest_measurements_are_not_erased_by_older_non_overlapping_samples(self):
        gate = SurfaceUpdates()
        pixel_in_block = (np.indices(self.base.shape)[0] % 8) * 8 + np.indices(self.base.shape)[1] % 8
        for stamp, begin in ((0, 0), (.5, 21), (1, 41)):
            depth = np.where((pixel_in_block >= begin) & (pixel_in_block < begin + 23), self.base, np.nan)
            result = gate.observe(depth, stamp)
            if stamp < 1:
                self.assertIsNone(result)
        np.testing.assert_array_equal(result, depth)
        np.testing.assert_array_equal(gate.published, depth)
        self.assertTrue(np.isfinite(result).any())

    def test_recorded_stationary_scene_only_publishes_once(self):
        recording = Path(__file__).resolve().parents[1] / "data" / "recorded_wall.npz"
        with np.load(recording, allow_pickle=False) as saved:
            gate = SurfaceUpdates()
            for depth, stamp in zip(saved["depth"], saved["timestamps"]):
                gate.observe(depth, float(stamp))
        self.assertEqual(gate.publications, 1)
        self.assertEqual(gate.status, "Map unchanged")

    def test_recorded_ceiling_floor_noise_accepts_one_map_within_two_seconds(self):
        recording = Path(__file__).resolve().parents[1] / "data" / "recorded_floor_grid.npz"
        with np.load(recording, allow_pickle=False) as saved:
            grids, stamps = saved["depth"], saved["timestamps"]
        originals = grids.copy()
        gate = SurfaceUpdates(block=1)
        published = [float(stamp) for grid, stamp in zip(grids, stamps)
                     if gate.observe(grid, float(stamp)) is not None]
        self.assertEqual(len(published), 1)
        self.assertLessEqual(published[0] - stamps[0], 2)
        np.testing.assert_array_equal(grids, originals)

    def test_three_cm_change_is_confirmed_on_top_of_recorded_ceiling_noise(self):
        recording = Path(__file__).resolve().parents[1] / "data" / "recorded_floor_grid.npz"
        with np.load(recording, allow_pickle=False) as saved:
            grids, stamps = saved["depth"].copy(), saved["timestamps"]
        gate = SurfaceUpdates(block=1)
        events = []
        for index, (grid, stamp) in enumerate(zip(grids, stamps)):
            if index >= 20:
                grid[19:27, 34:44] -= .03
            if gate.observe(grid, float(stamp)) is not None:
                events.append(float(stamp))
        self.assertEqual(len(events), 2)
        self.assertGreaterEqual(events[1], stamps[20] + 1)
        self.assertLessEqual(events[1], stamps[20] + 2)

    def test_gradual_motion_updates_during_movement_and_holds_after_it_ends(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        publications = []
        for index in range(7):
            depth = self.base.copy()
            depth[20:60, 30:80] = .9 - .015 * index
            stamp = 1.5 + index * .5
            result = gate.observe(depth, stamp)
            if result is not None:
                publications.append(stamp)
                np.testing.assert_array_equal(result, depth)
        self.assertGreaterEqual(len(publications), 3)
        self.assertLessEqual(publications[0], 2.5)
        self.assertTrue(np.all(np.diff(publications) >= 1))
        # Allow the last position between update slots to be published once.
        for stamp in np.arange(5, 7, .5):
            gate.observe(depth, float(stamp))
        count = gate.publications
        for stamp in np.arange(7, 12, .5):
            self.assertIsNone(gate.observe(depth, float(stamp)))
        self.assertEqual(gate.publications, count)

    def test_different_noisy_regions_cannot_collectively_confirm_a_change(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        areas = ((slice(8, 32), slice(8, 40)), (slice(40, 64), slice(8, 40)),
                 (slice(24, 56), slice(72, 104)))
        for index, stamp in enumerate((1.5, 2, 2.5)):
            depth = self.base.copy()
            for area_index, area in enumerate(areas):
                depth[area] = .985 if area_index == index else .975
            self.assertIsNone(gate.observe(depth, stamp))
        self.assertEqual(gate.publications, 1)
        for area in areas:
            depth[area] = .95
        results = [gate.observe(depth, stamp) for stamp in (3, 3.5, 4)]
        self.assertTrue(any(result is not None for result in results))
        np.testing.assert_array_equal(gate.published, depth)

    def test_too_small_or_too_shallow_changes_are_ignored(self):
        gate = SurfaceUpdates()
        self.initial(gate)
        for index, depth in enumerate((self.base - .01, self.base.copy())):
            depth[30:33, 45:48] = .7
            for stamp in (1.5, 2, 2.5):
                self.assertIsNone(gate.observe(depth, stamp + index * 1.5))
        self.assertEqual(gate.publications, 1)


class ProjectorUpdateTests(unittest.TestCase):
    def simulated_run(self, scene, mode="changes", seconds=6, work_points=None, projector_inset=None,
                      height_mode=False, projector_inset_y=None):
        """Exercise capture pairing, rendering and exports without opening devices."""
        clock = [100.0]
        checks, finishes, projectors, cameras = [], [], [], []
        self.finish_times = []
        finish_times = self.finish_times
        image = np.zeros((720, 1280, 3), np.uint8)

        class FakeCamera:
            def __init__(self, index, focus):
                self.info, self.closed = {"index": index}, False
                cameras.append(self)

            def snapshot(self):
                return [stereo_live.Frame(round(clock[0] * 1000), clock[0], image)]

            def fps(self):
                return 30

            def close(self):
                self.closed = True

        class FakeEngine:
            def __init__(self, size, method, *args):
                self.size = size
                self.method, self.minimum, self.disparities = method, 0, 64
                self.effective_near = .3
                self.last_rectified = (np.zeros(size[::-1] if work_points is not None else (80, 120), np.uint8),) * 2
                self.rectified_intrinsic = np.float64([[120, 0, 60], [0, 120, 40], [0, 0, 1]])

            def compute(self, left, right):
                depth = scene(clock[0] - 100).copy()
                if work_points is not None:
                    depth = cv2.resize(depth, self.size, interpolation=cv2.INTER_NEAREST)
                checks.append(depth)
                clock[0] += .05
                return np.zeros((*depth.shape, 3), np.uint8), depth, float(np.isfinite(depth).mean())

        class CountingAppearance(DepthAppearance):
            def process(self, depth, *args):
                finishes.append(depth.copy())
                finish_times.append(clock[0])
                return super().process(depth, *args)

        def wait_key(delay):
            clock[0] += .1
            return -1

        def show(window, output):
            if window == stereo_live.PROJECTOR:
                projectors.append(output.copy())

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            calibration = root / "calibration.npz"
            np.savez(calibration, camera_indices=[0, 1], focus_values=[0, 0])
            work_area = root / "work_area.json"
            if work_points is not None:
                WorkArea(work_points, (120, 80), (320, 180),
                         hashlib.sha256(calibration.read_bytes()).hexdigest(),
                         projector_bounds((320, 180), projector_inset or 0, projector_inset_y or 0)).save(work_area)
            argv = ["stereo_live.py", "--calibration", str(calibration), "--output", str(root / "output"),
                    "--seconds", str(seconds), "--update-mode", mode, "--width", "320", "--height", "180",
                    "--work-area", str(work_area), "--projector-model", str(root / "projector_model.json")]
            if projector_inset is not None:
                argv += ["--projector-inset-x", str(projector_inset)]
            if projector_inset_y is not None:
                argv += ["--projector-inset-y", str(projector_inset_y)]
            if height_mode:
                floor_file = root / "floor.json"
                FloorPlane([0, 0, -1, 1.5], hashlib.sha256(calibration.read_bytes()).hexdigest()).save(floor_file)
                argv += ["--height-map", "--floor-reference", str(floor_file), "--height-max", "0.5"]
            stack.enter_context(patch.object(sys, "argv", argv))
            args = stereo_live.arguments()
            stack.enter_context(patch.object(stereo_live, "Camera", FakeCamera))
            stack.enter_context(patch.object(stereo_live, "StereoEngine", FakeEngine))
            stack.enter_context(patch.object(stereo_live, "DepthAppearance", CountingAppearance))
            stack.enter_context(patch.object(stereo_live, "dpi_awareness"))
            stack.enter_context(patch.object(stereo_live.time, "perf_counter", lambda: clock[0]))
            for name in ("namedWindow", "resizeWindow", "moveWindow", "setWindowProperty",
                         "setMouseCallback", "destroyAllWindows"):
                stack.enter_context(patch.object(cv2, name))
            stack.enter_context(patch.object(cv2, "waitKey", wait_key))
            stack.enter_context(patch.object(cv2, "imshow", show))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
            stack.enter_context(redirect_stdout(StringIO()))
            stereo_live.run(args)
            report = json.loads((args.output / "live_report.json").read_text(encoding="utf-8"))
            exports = {path.name: np.load(path) for path in args.output.glob("*.npy")}
        self.assertTrue(all(camera.closed for camera in cameras))
        return report["periods"][0], checks, finishes, projectors, exports

    def test_projector_and_expensive_finish_only_run_on_confirmed_changes(self):
        def scene(elapsed):
            depth = np.ones((80, 120), np.float32)
            if elapsed >= 3:
                depth[20:60, 30:80] = .9
            return depth

        report, checks, finishes, projectors, exports = self.simulated_run(scene)
        self.assertEqual(report["publications"], 2)
        self.assertGreater(report["depth_checks"], report["publications"] * 3)
        self.assertEqual(report["depth_checks"], len(checks))
        self.assertEqual(len(finishes), 2)
        self.assertEqual(len(projectors), 3)  # Initial black image, then two publications.
        self.assertFalse(np.array_equal(projectors[1], projectors[2]))
        np.testing.assert_array_equal(exports["640x360_sgbm_published_depth_m.npy"], scene(4))
        for depth in checks:  # The complete pipeline must preserve its raw inputs.
            self.assertTrue(np.all((depth == 1) | (depth == np.float32(.9))))

    def test_live_mode_publishes_every_processed_depth(self):
        report, checks, finishes, projectors, _ = self.simulated_run(
            lambda elapsed: np.ones((80, 120), np.float32), mode="live", seconds=2)
        self.assertEqual(report["publications"], len(checks))
        self.assertEqual(len(finishes), len(checks))
        self.assertEqual(len(projectors), len(checks) + 1)

    def test_projector_updates_while_moving_and_stops_repainting_static_scene(self):
        def scene(elapsed):
            depth = np.ones((80, 120), np.float32)
            depth[20:60, 30:80] = .95 - max(0, min(elapsed - 2, 4)) * .1
            return depth

        report, _, finishes, projectors, exports = self.simulated_run(scene, seconds=10)
        moving_updates = [stamp for stamp in self.finish_times if 103 < stamp < 106]
        self.assertGreaterEqual(len(moving_updates), 2)
        self.assertTrue(np.all(np.diff(self.finish_times) >= 1 - 1e-6))
        self.assertLess(self.finish_times[-1], 108)
        self.assertEqual(report["publications"], len(finishes))
        self.assertEqual(len(projectors), len(finishes) + 1)
        np.testing.assert_array_equal(exports["640x360_sgbm_published_depth_m.npy"], scene(10))

    def test_no_reliable_depth_never_publishes_or_exports_a_fake_surface(self):
        report, _, finishes, _, exports = self.simulated_run(
            lambda elapsed: np.full((80, 120), np.nan, np.float32), seconds=2)
        self.assertEqual(report["publications"], 0)
        self.assertIsNone(report["receipt_to_submit_ms"])
        self.assertEqual(finishes, [])
        self.assertEqual(set(exports), {"640x360_sgbm_depth_m.npy"})
        self.assertTrue(np.isnan(exports["640x360_sgbm_depth_m.npy"]).all())

    def test_projector_ignores_motion_outside_the_saved_work_area(self):
        points = [[40, 24], [79, 24], [79, 55], [40, 55]]
        region = WorkArea(points, (120, 80), (320, 180), "hash").mask((360, 640))

        def scene(elapsed):
            depth = np.ones(region.shape, np.float32)
            depth[~region] = max(.2, 1 - elapsed * .1)
            return depth

        report, checks, finishes, projectors, exports = self.simulated_run(scene, work_points=points)
        self.assertEqual(report["publications"], 1)
        self.assertEqual(len(finishes), 1)
        self.assertEqual(len(projectors), 2)
        self.assertTrue(np.any(checks[-1][~region] < .8))  # Outside movement really reached the engine.
        self.assertTrue(np.all(projectors[-1][2:-2, 2:-2].sum(axis=2) > 0))
        for name in ("640x360_sgbm_depth_m.npy", "640x360_sgbm_published_depth_m.npy",
                     "640x360_sgbm_display_depth_m.npy"):
            self.assertTrue(np.isnan(exports[name][~region]).all())
        self.assertLess(report["work_area_fraction"], .2)

    def test_projector_accepts_continuous_motion_inside_the_saved_work_area(self):
        points = [[40, 24], [79, 24], [79, 55], [40, 55]]
        region = WorkArea(points, (120, 80), (320, 180), "hash").mask((360, 640))

        def scene(elapsed):
            depth = np.ones(region.shape, np.float32)
            depth[region] -= max(0, min(elapsed - 2, 4)) * .1
            return depth

        report, _, finishes, _, _ = self.simulated_run(scene, seconds=10, work_points=points)
        self.assertGreaterEqual(len([t for t in self.finish_times if 103 < t < 106]), 2)
        self.assertTrue(np.all(np.diff(self.finish_times) >= 1 - 1e-6))
        self.assertLess(self.finish_times[-1], 108)
        self.assertEqual(report["publications"], len(finishes))

    def test_projector_insets_apply_with_and_without_a_saved_camera_zone(self):
        points = [[40, 24], [79, 24], [79, 55], [40, 55]]
        for work_points in (None, points):
            with self.subTest(calibrated_zone=work_points is not None):
                report, _, _, images, _ = self.simulated_run(
                    lambda elapsed: np.ones((80, 120), np.float32),
                    work_points=work_points, projector_inset=20, seconds=2)
                self.assertEqual(report["projector_rect"], [20, 0, 300, 180])
                self.assertFalse(images[-1][:, :20].any())
                self.assertFalse(images[-1][:, 300:].any())
                self.assertTrue(images[-1][80:100, 140:180].any())

    def test_height_pipeline_illuminates_low_objects_and_keeps_high_objects_black(self):
        def scene(elapsed):
            depth = np.full((80, 120), 1.5, np.float32)
            depth[20:60, 15:45] = 1.3  # 20cm above the floor.
            depth[20:60, 75:105] = .8  # 70cm: excluded.
            return depth

        report, _, _, images, exports = self.simulated_run(scene, height_mode=True, seconds=2)
        self.assertEqual(report["above_height_range"], "black")
        self.assertEqual(report["height_max_m"], .5)
        self.assertTrue(images[-1][90, 80].any())
        self.assertFalse(images[-1][90, 240].any())
        self.assertAlmostEqual(float(exports["640x360_sgbm_height_m.npy"][40, 90]), .7, places=5)

    def test_saved_zone_preserves_top_bottom_and_side_projector_borders(self):
        points = [[40, 24], [79, 24], [79, 55], [40, 55]]
        report, _, _, images, _ = self.simulated_run(lambda elapsed: np.ones((80, 120), np.float32),
            work_points=points, projector_inset=20, projector_inset_y=10, seconds=2)
        self.assertEqual(report["projector_rect"], [20, 10, 300, 170])
        image = images[-1]
        self.assertFalse(image[:, :20].any())
        self.assertFalse(image[:, 300:].any())
        self.assertFalse(image[:10].any())
        self.assertFalse(image[170:].any())
        self.assertTrue(image[80:100, 140:180].any())


if __name__ == "__main__":
    unittest.main()
