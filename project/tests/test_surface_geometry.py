from pathlib import Path
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from depth_appearance import DisplayDepth

from surface_geometry import FloorPlane, HeightRenderer, ProjectorModel, project_points
from calibrate_projector import tag_pattern, detected_tags, make_detector, floor_alignment
from work_area import WorkArea, projector_bounds
import calibrate_projector
from stereo_live import Frame


class FloorTests(unittest.TestCase):
    def setUp(self):
        self.k = np.float64([[180, 0, 60], [0, 190, 40], [0, 0, 1]])
        self.floor = FloorPlane([-.08, .06, -1, 1.7], 'hash')
        y, x = np.indices((80, 120))
        n = self.floor.plane
        self.factor = n[0] * (x - 60) / 180 + n[1] * (y - 40) / 190 + n[2]
        self.depth = (-n[3] / self.factor).astype(np.float32)

    def test_tilted_floor_is_zero_and_height_is_perpendicular_to_the_plane(self):
        np.testing.assert_allclose(self.floor.heights(self.depth, self.k), 0, atol=2e-7)
        raised = ((.15 - self.floor.plane[3]) / self.factor).astype(np.float32)
        np.testing.assert_allclose(self.floor.heights(raised, self.k), .15, atol=2e-7)
        raised[3:8] = np.nan
        self.assertTrue(np.isnan(self.floor.heights(raised, self.k)[3:8]).all())

    def test_floor_fit_rejects_outliers_without_inventing_unknown_measurements(self):
        rng = np.random.default_rng(8)
        frames = self.depth + rng.normal(0, .002, (9, *self.depth.shape)).astype(np.float32)
        frames[:, :12, :60] = .7
        frames[:, -5:] = np.nan
        original = frames.copy()
        region = np.ones(self.depth.shape, bool)
        fitted = FloorPlane.fit(frames, self.k, region, 'hash')
        heights = fitted.heights(self.depth, self.k)
        self.assertLess(float(np.max(np.abs(heights))), .002)
        np.testing.assert_array_equal(frames, original)
        with self.assertRaises(ValueError):
            FloorPlane.fit(np.full_like(frames, np.nan), self.k, region, 'hash')

    def test_height_colours_emphasize_a_small_object_and_preserve_missing_pixels(self):
        floor = FloorPlane([0, 0, -1, 1.5], 'hash')
        depth = np.float32([[1.5, 1.497, 1.48, 1., .9999, .5, np.nan]])
        k = np.eye(3)
        renderer = HeightRenderer(floor, k)
        colors = renderer.colorize(depth)
        np.testing.assert_array_equal(colors[0, 0], colors[0, 1])  # Noise band stays neutral.
        self.assertGreater(np.linalg.norm(colors[0, 2].astype(float) - colors[0, 0]), 20)
        self.assertTrue(colors[0, 3].any())  # Exactly 50cm is included.
        self.assertFalse(colors[0, 4].any())  # Anything higher stays black.
        self.assertFalse(colors[0, 5].any())
        self.assertFalse(colors[0, -1].any())

    def test_completion_cannot_repaint_a_high_object_or_erase_its_occluding_geometry(self):
        raw = np.full((32, 32), 1.5, np.float32)
        raw[8:24, 8:24] = .8
        raw[15:17, 15:17] = np.nan
        before = raw.copy()
        # An inferred floor behind the excluded object must not become visible.
        result = DisplayDepth(np.full_like(raw, 1.5), np.isfinite(raw), np.ones(raw.shape, bool),
                              np.ones(raw.shape, np.float32))
        renderer = HeightRenderer(FloorPlane([0, 0, -1, 1.5], 'hash'), np.eye(3))
        colors = renderer.render(result, raw)
        self.assertFalse(colors[8:24, 8:24].any())
        self.assertTrue(colors[2, 2].any())
        np.testing.assert_allclose(result.depth[8:24, 8:24], .8)
        self.assertTrue(result.completed[15:17, 15:17].all())
        self.assertFalse(result.measured[15:17, 15:17].any())
        np.testing.assert_array_equal(raw, before)

    def test_saved_floor_rejects_a_different_stereo_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'floor.json'
            self.floor.save(path)
            np.testing.assert_allclose(FloorPlane.load(path, 'hash').plane, self.floor.plane)
            with self.assertRaises(ValueError):
                FloorPlane.load(path, 'other')


class ProjectorTests(unittest.TestCase):
    def cloud(self):
        rng = np.random.default_rng(18)
        groups = np.repeat(np.arange(12), 12)
        points = rng.uniform([-.2, -.15, .6], [.2, .15, 1.8], (144, 3))
        points[:, 2] = np.repeat(np.linspace(.6, 1.8, 12), 12) + points[:, 0] * .1
        k = np.float64([[400, 0, 160], [0, 400, 90], [0, 0, 1]])
        r = cv2.Rodrigues(np.float64([.01, .08, -.01]))[0]
        p = k @ np.c_[r, [-.08, .01, .03]]
        pixels, _ = project_points(p, points)
        return rng, points, pixels, groups, p

    def test_non_coplanar_calibration_recovers_projection_with_outlier_matches(self):
        rng, points, pixels, groups, expected = self.cloud()
        noisy = pixels + rng.normal(0, .12, pixels.shape)
        noisy[::18] += 40
        model = ProjectorModel.fit(points, noisy, groups, (320, 180), 'hash')
        test = rng.uniform([-.15, -.1, .7], [.15, .1, 1.7], (30, 3))
        a, _ = project_points(expected, test)
        b, _ = project_points(model.matrix, test)
        self.assertLess(float(np.sqrt(np.mean((a - b) ** 2))), .2)
        self.assertLess(model.quality['leave_view_out_rms_px'], 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'projector.json'
            model.save(path)
            np.testing.assert_allclose(ProjectorModel.load(path, (320, 180), 'hash').matrix, model.matrix)
            with self.assertRaises(ValueError):
                ProjectorModel.load(path, (320, 180), 'new-camera-calibration')

    def test_a_floor_homography_cannot_be_accepted_as_a_3d_projector_model(self):
        _, points, pixels, groups, _ = self.cloud()
        points[:, 2] = 1.7
        with self.assertRaises(ValueError):
            ProjectorModel.fit(points, pixels, groups, (320, 180), 'hash')

    def test_3d_render_moves_the_raised_object_and_resolves_projector_occlusion(self):
        k = np.float64([[100, 0, 60], [0, 100, 40], [0, 0, 1]])
        p = np.float64([[200, 0, 120, -20], [0, 200, 80, 0], [0, 0, 1, 0]])
        model = ProjectorModel(p, (240, 160), 'hash')
        depth = np.ones((80, 120), np.float32)
        colors = np.zeros((80, 120, 3), np.uint8)
        colors[:] = (255, 0, 0)
        depth[30:50, 80:90] = .5
        colors[30:50, 80:90] = (0, 0, 255)
        output = model.render(colors, depth, k, (10, 0, 230, 160))
        self.assertTrue((output[80, 130] == (0, 0, 255)).all())  # Near surface wins over far floor.
        red_y, red_x = np.nonzero(output[:, :, 2] > 200)
        self.assertAlmostEqual(float(np.median(red_x)), 129, delta=2)
        self.assertFalse(output[:, :10].any())
        self.assertFalse(output[:, 230:].any())
        colors[30:50, 80:90] = 0
        black_occluder = model.render(colors, depth, k, (10, 0, 230, 160))
        self.assertFalse(black_occluder[80, 130].any())
        empty = model.render(colors, np.full_like(depth, np.nan), k, (10, 0, 230, 160))
        self.assertFalse(empty.any())

    def test_projected_marker_ids_and_floor_alignment_have_unambiguous_orientation(self):
        size, rectangle = (1280, 800), projector_bounds((1280, 800), 227)
        image, expected = tag_pattern(size, rectangle, floor=True)
        detected = detected_tags(make_detector(), image)
        self.assertEqual(set(detected), set(expected))
        # A rotated camera sees a perspective transform of the projector image.
        transform = np.float64([[-.38, .02, 560], [.01, -.36, 330], [0, 0, 1]])
        uv = np.concatenate([expected[i] for i in sorted(expected)])
        camera = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), transform).reshape(-1, 2)
        old = WorkArea([[160, 20], [570, 20], [570, 350], [160, 350]], (640, 360), size, 'hash', rectangle)
        aligned, error = floor_alignment(camera, uv, old)
        actual = cv2.perspectiveTransform(aligned.points.reshape(-1, 1, 2), np.linalg.inv(transform)).reshape(-1, 2)
        wanted = np.float32([[227, 0], [1052, 0], [1052, 799], [227, 799]])
        np.testing.assert_allclose(actual, wanted, atol=.001)
        self.assertLess(error, .001)


class ProjectorWorkflowTests(unittest.TestCase):
    def test_unreliable_floor_keeps_the_window_open_and_preserves_the_previous_reference(self):
        import time
        cameras, keys = [], [0]
        image = np.zeros((720, 1280, 3), np.uint8)

        class Camera:
            def __init__(self, *args):
                self.closed = False
                self.sequence = 0
                cameras.append(self)

            def snapshot(self):
                self.sequence += 1
                return [Frame(self.sequence, time.perf_counter(), image)]

            def close(self):
                self.closed = True

        class Engine:
            def __init__(self, *args, **kwargs):
                pass

            def compute(self, *args):
                return None, np.full((360, 640), np.nan, np.float32), 0

        def key(delay):
            keys[0] += 1
            return 27 if keys[0] >= 22 else -1

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            calibration = root / 'stereo.npz'
            k = np.float64([[900, 0, 640], [0, 900, 360], [0, 0, 1]])
            np.savez(calibration, K1=k, K2=k, D1=np.zeros(5), D2=np.zeros(5), R=np.eye(3),
                     T=[-.13, 0, 0], camera_indices=[0, 1], focus_values=[0, 0])
            digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
            area_path = root / 'area.json'
            WorkArea([[100, 50], [500, 50], [500, 300], [100, 300]], (640, 360),
                     (1280, 800), digest).save(area_path)
            floor = root / 'floor.json'
            floor.write_bytes(b'previous reference')
            args = SimpleNamespace(calibration=calibration, work_area=area_path, floor_reference=floor,
                                   model=root / 'projector.json', output=root / 'output', reuse_floor=False,
                                   floor_only=True, poses=12)
            stack.enter_context(patch.object(calibrate_projector, 'Camera', Camera))
            stack.enter_context(patch.object(calibrate_projector, 'StereoEngine', Engine))
            stack.enter_context(patch.object(calibrate_projector, 'detected_tags', return_value={}))
            stack.enter_context(patch.object(calibrate_projector, 'dpi_awareness'))
            rejected = stack.enter_context(patch.object(FloorPlane, 'fit', side_effect=ValueError('unreliable floor')))
            for name in ('namedWindow', 'resizeWindow', 'moveWindow', 'imshow', 'setWindowProperty', 'destroyAllWindows'):
                stack.enter_context(patch.object(cv2, name))
            stack.enter_context(patch.object(cv2, 'waitKey', side_effect=key))
            stack.enter_context(patch.object(cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(redirect_stdout(StringIO()))
            calibrate_projector.run(args)
            self.assertEqual(floor.read_bytes(), b'previous reference')
            self.assertTrue((args.output / 'floor_diagnostic.npz').is_file())
            self.assertFalse(args.model.exists())
            rejected.assert_called_once()
        self.assertEqual(keys[0], 22)
        self.assertTrue(all(camera.closed for camera in cameras))


if __name__ == '__main__':
    unittest.main()
