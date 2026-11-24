"""A projector footprint in rectified left-camera coordinates, on a flat floor."""

import json
from pathlib import Path
import time

import cv2
import numpy as np

from depth_appearance import DisplayDepth


AREA_WINDOW = "Work area: click light corners 1-2-3-4; Enter=save Esc=cancel"


def projector_bounds(size, inset_x=0, inset_y=0):
    width, height = size
    if not isinstance(inset_x, (int, np.integer)) or inset_x < 0 or inset_x * 2 > width - 2:
        raise ValueError("Projector inset must leave at least two visible columns")
    if not isinstance(inset_y, (int, np.integer)) or inset_y < 0 or inset_y * 2 > height - 2:
        raise ValueError("Projector inset must leave at least two visible rows")
    return (int(inset_x), int(inset_y), width - int(inset_x), height - int(inset_y))


def clip_projector(image, rectangle):
    left, top, right, bottom = rectangle
    image[:, :left] = 0
    image[:, right:] = 0
    image[:top] = 0
    image[bottom:] = 0
    return image


class WorkArea:
    def __init__(self, points, input_size, output_size, calibration_hash, projector_rect=None):
        self.points = np.array(points, dtype=np.float32, copy=True)
        self.input_size = tuple(input_size)
        self.output_size = tuple(output_size)
        if (len(self.input_size) != 2 or len(self.output_size) != 2
                or any(not isinstance(v, (int, np.integer)) or v < 2
                       for v in (*self.input_size, *self.output_size))):
            raise ValueError("Work area needs valid image and projector sizes")
        if self.points.shape != (4, 2) or not np.isfinite(self.points).all():
            raise ValueError("Select four finite light corners in order 1-2-3-4")
        if (np.any(self.points < 0) or np.any(self.points > np.array(self.input_size) - 1)
                or not cv2.isContourConvex(self.points)
                or abs(cv2.contourArea(self.points)) < np.prod(self.input_size) * .01):
            raise ValueError("All four light corners must form a visible convex area inside the camera image")
        self.calibration_hash = calibration_hash
        self.projector_rect = tuple(projector_rect) if projector_rect is not None else projector_bounds(output_size)
        if (len(self.projector_rect) != 4
                or any(not isinstance(v, (int, np.integer)) for v in self.projector_rect)):
            raise ValueError("Projector rectangle must contain four integer bounds")
        left, top, right, bottom = self.projector_rect
        if not (0 <= left < right - 1 < self.output_size[0]
                and 0 <= top < bottom - 1 < self.output_size[1]):
            raise ValueError("Projector rectangle must lie inside the output image")
        self.points.setflags(write=False)
        self._masks = {}
        self._transforms = {}

    def scaled_points(self, shape):
        return self.points * ((np.array(shape[:2][::-1]) - 1) / (np.array(self.input_size) - 1))

    def mask(self, shape):
        shape = tuple(shape[:2])
        if shape not in self._masks:
            mask = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(mask, np.rint(self.scaled_points(shape)).astype(np.int32), 1)
            mask = mask.astype(bool)
            mask.setflags(write=False)
            self._masks[shape] = mask
        return self._masks[shape]

    def restrict(self, depth):
        return np.where(self.mask(depth.shape), depth, np.nan).astype(np.float32)

    def clip_display(self, display):
        outside = ~self.mask(display.depth.shape)
        display.depth[outside] = np.nan
        display.alpha[outside] = 0
        for name in ("measured", "interpolated", "held", "completed"):
            values = getattr(display, name)
            if values is not None:
                values[outside] = False
        return display

    def process_display(self, appearance, depth, timestamp, guide=None):
        # Process the bounding rectangle so a small projector footprint does
        # not look like a nearly empty camera image to the completion filter.
        x, y, width, height = cv2.boundingRect(np.rint(self.scaled_points(depth.shape)).astype(np.int32))
        selection = np.s_[y:y + height, x:x + width]
        cropped = self.restrict(depth)[selection]
        result = appearance.process(cropped, timestamp, guide[selection] if guide is not None else None)

        def expand(values, fill):
            if values is None:
                return None
            full = np.full(depth.shape, fill, dtype=values.dtype)
            full[selection] = values
            return full

        return self.clip_display(DisplayDepth(expand(result.depth, np.nan), expand(result.measured, False),
                                             expand(result.interpolated, False), expand(result.alpha, 0),
                                             expand(result.held, False), expand(result.completed, False)))

    def statistics(self, display):
        mask = self.mask(display.depth.shape)
        result = {"display_fraction": float(np.isfinite(display.depth[mask]).mean())}
        for name in ("measured", "interpolated", "held", "completed"):
            values = getattr(display, name)
            result[name + "_fraction"] = float(values[mask].mean()) if values is not None else 0.
        return result

    def project(self, colors):
        shape = colors.shape[:2]
        if shape not in self._transforms:
            left, top, right, bottom = self.projector_rect
            destination = np.float32([[left, top], [right - 1, top],
                                      [right - 1, bottom - 1], [left, bottom - 1]])
            self._transforms[shape] = cv2.getPerspectiveTransform(
                self.scaled_points(shape).astype(np.float32), destination)
        clipped = colors.copy()
        clipped[~self.mask(shape)] = 0
        output = cv2.warpPerspective(clipped, self._transforms[shape], self.output_size,
                                     flags=cv2.INTER_LINEAR)
        return clip_projector(output, self.projector_rect)

    def save(self, path):
        config = {"version": 1, "coordinate_space": "rectified_left",
                  "input_size": list(self.input_size), "output_size": list(self.output_size),
                  "source_points": self.points.tolist(),
                  "projector_rect": list(self.projector_rect),
                  "stereo_calibration_sha256": self.calibration_hash,
                  "saved_unix_time": time.time(),
                  "model": "flat_floor_homography"}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, output_size, calibration_hash, projector_rect=None):
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if (config.get("version") != 1 or config.get("coordinate_space") != "rectified_left"
                or config.get("stereo_calibration_sha256") != calibration_hash
                or config.get("output_size") != list(output_size)):
            raise ValueError("Work area does not match stereo calibration/projector; run --calibrate-area again")
        area = cls(config["source_points"], config["input_size"], config["output_size"], calibration_hash,
                   config.get("projector_rect"))
        if projector_rect is not None and tuple(projector_rect) != area.projector_rect:
            raise ValueError("Projector field size changed; run --calibrate-area again")
        return area


def calibration_pattern(size, projector_rect=None):
    width, height = size
    image = np.zeros((height, width, 3), np.uint8)
    left, top, right, bottom = projector_rect or projector_bounds(size)
    field = image[top:bottom, left:right]
    field[:] = 255
    height, width = field.shape[:2]
    scale = max(.7, min(width, height) / 300)
    margin = max(12, round(min(width, height) * .04))
    text_height = cv2.getTextSize("4", cv2.FONT_HERSHEY_SIMPLEX, scale, 4)[0][1]
    for index, (x, y) in enumerate(((margin, margin + text_height),
                                   (width - margin - text_height, margin + text_height),
                                   (width - margin - text_height, height - margin),
                                   (margin, height - margin)), 1):
        cv2.putText(field, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    return image


def calibrate_area(read_image, projector_window, path, output_size, calibration_hash, projector_rect=None):
    """Keep a live rectified preview; only Enter after four clicks saves a zone."""
    points = []
    latest_shape = None
    message = "Click OUTER light corners 1 -> 2 -> 3 -> 4, not the printed numbers"

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and latest_shape is not None and len(points) < 4:
            if 0 <= x < latest_shape[1] and 0 <= y < latest_shape[0]:
                points.append((x, y))

    cv2.namedWindow(AREA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(AREA_WINDOW, 1100, 660)
    cv2.moveWindow(AREA_WINDOW, 30, 40)
    cv2.setMouseCallback(AREA_WINDOW, click)
    cv2.imshow(projector_window, calibration_pattern(output_size, projector_rect))
    try:
        while True:
            image = read_image()
            candidate = None
            message = "Click OUTER light corners 1 -> 2 -> 3 -> 4, not the printed numbers"
            if image is not None:
                latest_shape = image.shape[:2]
                image = image.copy()
                if len(points) == 4:
                    try:
                        candidate = WorkArea(points, image.shape[1::-1], output_size, calibration_hash, projector_rect)
                        image[~candidate.mask(image.shape)] //= 3
                        message = "Check outline. Enter=save; Backspace=undo; R=clear; Esc=cancel"
                    except ValueError as exc:
                        message = str(exc)
                if points:
                    cv2.polylines(image, [np.int32(points)], len(points) == 4, (0, 255, 0), 2)
                for number, (x, y) in enumerate(points, 1):
                    cv2.circle(image, (x, y), 5, (0, 128, 255), -1)
                    cv2.putText(image, str(number), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, .65,
                                (0, 128, 255), 2, cv2.LINE_AA)
                footer = np.zeros((56, image.shape[1], 3), np.uint8)
                cv2.putText(footer, message, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(footer, "Keep cameras/projector fixed; use an empty flat floor. R=clear Esc=cancel",
                            (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
                image = np.vstack((image, footer))
            else:
                image = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(image, "Waiting for a fresh camera frame... Esc=cancel", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
                latest_shape = None
            cv2.imshow(AREA_WINDOW, image)
            key = cv2.waitKey(30) & 0xff
            if key == 27 or cv2.getWindowProperty(AREA_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if cv2.getWindowProperty(projector_window, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (8, 127) and points:
                points.pop()
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and candidate is not None:
                candidate.save(path)
                return candidate
    finally:
        cv2.destroyWindow(AREA_WINDOW)
