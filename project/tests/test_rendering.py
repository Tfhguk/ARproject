import unittest

import cv2
import numpy as np

from rendering import Renderer


class RenderingTests(unittest.TestCase):
    def test_fixed_palette_direction_and_unknown_pixels(self):
        values = np.float32([[0, .5, 1, np.nan]])
        image = Renderer(0, 1, ascending=True).colorize(values)
        expected = cv2.applyColorMap(np.uint8([[0, 128, 255]]), cv2.COLORMAP_TURBO)
        np.testing.assert_array_equal(image[:, :3], expected)
        self.assertFalse(image[0, -1].any())
        inverse = Renderer(0, 1).colorize(values)
        np.testing.assert_array_equal(inverse[0, 0], image[0, 2])
        np.testing.assert_array_equal(inverse[0, 2], image[0, 0])

    def test_bright_projector_colours_clip_ranges_without_changing_measurements(self):
        values = np.float32([[-1, 0, .2, 1, 2, np.nan]])
        before = values.copy()
        image = Renderer(0, 1, bright=True, ascending=True).colorize(values)
        self.assertTrue((image[0, :5].max(axis=1) > 100).all())
        np.testing.assert_array_equal(image[0, 0], image[0, 1])
        np.testing.assert_array_equal(image[0, 3], image[0, 4])
        np.testing.assert_array_equal(values, before)

    def test_invalid_colour_limits_are_rejected(self):
        for low, high in ((0, 0), (1, 0), (np.nan, 1), (0, np.inf)):
            with self.assertRaises(ValueError):
                Renderer(low, high)
