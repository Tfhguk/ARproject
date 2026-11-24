"""Presentation-only cleanup of metric depth; original measurements stay intact."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from depth_completion import complete_surface, nearest_values


@dataclass(frozen=True)
class Style:
    radius: int
    sigma_log: float
    hole_radius: int
    hole_area: int
    bridge_gap: int
    tau: float
    jump_log: float
    feather: float
    contour_radius: int = 0
    hold_seconds: float = 0
    hold_area: int = 0


STYLES = {
    "raw": Style(0, 0, 0, 0, 0, 0, 0, 0),
    "balanced": Style(2, 0.022, 3, 128, 8, 0.18, 0.035, 1.2, 2, 0.8, 768),
    "smooth": Style(3, 0.038, 16, 2048, 56, 0.38, 0.05, 2.6, 6, 1.5, 3072),
}


@dataclass
class DisplayDepth:
    depth: np.ndarray
    measured: np.ndarray
    interpolated: np.ndarray
    alpha: np.ndarray
    held: np.ndarray | None = None
    completed: np.ndarray | None = None

    def statistics(self):
        return {"measured_fraction": float(np.mean(self.measured)),
                "display_fraction": float(np.mean(np.isfinite(self.depth))),
                "interpolated_fraction": float(np.mean(self.interpolated)),
                "held_fraction": float(np.mean(self.held)) if self.held is not None else 0.0,
                "completed_fraction": float(np.mean(self.completed)) if self.completed is not None else 0.0}

    def colorize(self, renderer):
        colors = renderer.colorize(self.depth)
        return np.rint(colors.astype(np.float32) * self.alpha[:, :, None]).astype(np.uint8)


class DepthAppearance:
    def __init__(self, mode="smooth", scale=1.0, fill="conservative"):
        if mode not in STYLES or not math.isfinite(scale) or scale <= 0 or fill not in ("conservative", "surface"):
            raise ValueError("Unknown appearance mode or invalid resolution scale")
        self.mode, self.scale, self.fill = mode, scale, fill
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_guide = None
        self.timestamp = None
        self.last_measured = None

    def _enclosed_holes(self, valid, area):
        """Bound inference to holes wholly surrounded by the current image."""
        _, labels, stats, _ = cv2.connectedComponentsWithStats((~valid).astype(np.uint8), connectivity=8)
        allowed = stats[:, cv2.CC_STAT_AREA] <= round(area * self.scale ** 2)
        allowed[0] = False
        border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        allowed[border] = False
        return allowed[labels]

    def _round_contours(self, logs, valid, style):
        """Round both sides of depth boundaries with circular morphology.

        Temporary nearest-neighbour extension prevents black holes taking part
        in min/max filtering. It does not make any new pixel valid.
        """
        radius = max(1, round(style.contour_radius * self.scale * (1.6 if self.fill == "surface" else 1)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        extended = logs if valid.all() else nearest_values(logs, valid)
        edges = cv2.dilate(extended, kernel) - cv2.erode(extended, kernel) > 0.10
        # Median prefilter keeps min/max morphology from selecting noise
        # extremes and shifting the colour of an otherwise flat surface.
        baseline = cv2.medianBlur(extended, 5)
        opened = cv2.morphologyEx(baseline, cv2.MORPH_OPEN, kernel)
        rounded = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return np.where(edges, rounded, logs)

    def _small_holes(self, logs, valid, style, guide=None):
        """Interpolate only enclosed, bounded holes surrounded by one surface."""
        empty = np.zeros(valid.shape, bool)
        if valid.all() or not valid.any():
            return logs, valid, empty
        small = self._enclosed_holes(valid, style.hole_area)
        if not small.any():
            return logs, valid, empty
        radius = max(1, round(style.hole_radius * self.scale))
        kernel_size = 2 * radius + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        minimum = cv2.erode(np.where(valid, logs, np.float32(1e10)), kernel)
        maximum = cv2.dilate(np.where(valid, logs, np.float32(-1e10)), kernel)
        distance = cv2.distanceTransform((~valid).astype(np.uint8), cv2.DIST_L2, 3)
        # The range check forbids bridging a foreground/background boundary.
        fill = small & (distance <= radius) & ((maximum - minimum) <= 0.065)
        if not fill.any():
            return logs, valid, empty
        weights = valid.astype(np.float32)
        sigma = max(0.8, radius / 2)
        numerator = cv2.GaussianBlur(np.where(valid, logs, 0), (kernel_size, kernel_size), sigma)
        denominator = cv2.GaussianBlur(weights, (kernel_size, kernel_size), sigma)
        fill &= denominator >= 0.08
        if guide is not None:
            expected = cv2.GaussianBlur(np.where(valid, guide, 0), (kernel_size, kernel_size), sigma)
            np.divide(expected, denominator, out=expected, where=denominator > 0)
            fill &= np.abs(guide - expected) <= 25
        result = logs.copy()
        np.divide(numerator, denominator, out=result, where=fill)
        return result, valid | fill, fill

    def _bridge_gaps(self, logs, valid, guide, style):
        """Interpolate narrow gaps with agreeing observations on opposite sides."""
        if valid.all() or not valid.any():
            return logs, valid, np.zeros(valid.shape, bool)
        limit = max(1, round(style.bridge_gap * self.scale))

        def along_rows(values, present, image):
            rows, columns = values.shape
            x = np.arange(columns, dtype=np.int32)[None, :]
            left = np.maximum.accumulate(np.where(present, x, -columns), axis=1)
            right = np.minimum.accumulate(np.where(present, x, 2 * columns)[:, ::-1], axis=1)[:, ::-1]
            gap = right - left - 1
            bounded = (~present) & (left >= 0) & (right < columns) & (gap <= limit)
            a = np.take_along_axis(values, np.clip(left, 0, columns - 1), axis=1)
            b = np.take_along_axis(values, np.clip(right, 0, columns - 1), axis=1)
            fraction = (x - left) / np.maximum(1, right - left)
            estimate = a + (b - a) * fraction
            bounded &= np.abs(a - b) <= 0.065
            if image is not None:
                ia = np.take_along_axis(image, np.clip(left, 0, columns - 1), axis=1)
                ib = np.take_along_axis(image, np.clip(right, 0, columns - 1), axis=1)
                bounded &= np.abs(image - (ia + (ib - ia) * fraction)) <= 25
            return estimate.astype(np.float32), bounded

        horizontal, supported_h = along_rows(logs, valid, guide)
        vertical, supported_v = along_rows(logs.T, valid.T, guide.T if guide is not None else None)
        vertical, supported_v = vertical.T, supported_v.T
        both = supported_h & supported_v
        fill = (supported_h | supported_v) & ~(both & (np.abs(horizontal - vertical) > 0.05))
        result = logs.copy()
        estimate = np.where(supported_h, horizontal, vertical)
        estimate[both] = (horizontal[both] + vertical[both]) * 0.5
        result[fill] = estimate[fill]
        return result, valid | fill, fill

    def process(self, depth, timestamp, guide=None):
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or not math.isfinite(timestamp):
            raise ValueError("Need a two-dimensional depth map and a finite timestamp")
        measured = np.isfinite(depth) & (depth > 0)
        visible = measured.copy()
        output = np.where(measured, depth, np.nan).astype(np.float32)
        interpolated = np.zeros(depth.shape, bool)
        if self.mode == "raw" or not measured.any():
            self.reset()
            return DisplayDepth(output, measured, interpolated, visible.astype(np.float32))
        style = STYLES[self.mode]
        if guide is not None:
            guide = np.asarray(guide)
            if guide.shape != depth.shape:
                raise ValueError("Guide must be rectified grayscale at depth-map resolution")
            guide = guide.astype(np.float32)
        dt = timestamp - self.timestamp if self.timestamp is not None else 0
        history_ok = (self.previous is not None and self.previous.shape == depth.shape
                      and 0 < dt <= style.hold_seconds)
        moving = None
        if history_ok and guide is not None and self.previous_guide is not None:
            moving = (np.abs(guide - self.previous_guide) > 16).astype(np.uint8)
            motion_radius = max(2, round(3 * self.scale))
            moving = cv2.dilate(moving, np.ones((2 * motion_radius + 1,) * 2, np.uint8)).astype(bool)
        if not history_ok:
            self.last_measured = np.full(depth.shape, -np.inf, np.float64)
        logs = np.zeros(depth.shape, np.float32)
        np.log(depth, out=logs, where=measured)
        sentinel = float(logs[measured].min()) - 1.0

        # Remove unsupported impulses without taking a median across true edges.
        padded = np.pad(logs, 1, mode="edge")
        padded_valid = np.pad(visible, 1, mode="constant")
        similar = np.zeros(depth.shape, np.uint8)
        neighbours = np.zeros(depth.shape, np.uint8)
        height, width = depth.shape
        for dy in range(3):
            for dx in range(3):
                neighbour = padded[dy:dy + height, dx:dx + width]
                good = padded_valid[dy:dy + height, dx:dx + width]
                neighbours += good
                similar += good & (np.abs(neighbour - logs) <= 0.055)
        visible &= ~((similar <= 2) & (neighbours >= 7))

        # Disconnected specks should not remain as bright spots on the projector.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(visible.astype(np.uint8), connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= max(3, round(16 * self.scale ** 2))
        keep[0] = False
        visible &= keep[labels]
        # Only direct, accepted observations refresh the lifetime. Inferred or
        # held values can never perpetuate their own history.
        self.last_measured[visible] = timestamp
        if self.fill == "conservative":
            logs, visible, interpolated = self._small_holes(logs, visible, style, guide)
            logs, visible, bridged = self._bridge_gaps(logs, visible, guide, style)
            interpolated |= bridged

        held = np.zeros(depth.shape, bool)
        if moving is not None and (~visible).any():
            held = (~visible & np.isfinite(self.previous) & ~moving
                    & ((timestamp - self.last_measured) <= style.hold_seconds))
            if held.any():
                held &= self._enclosed_holes(visible, style.hold_area)
                logs[held] = self.previous[held]
                visible |= held
                interpolated |= held

        completed = np.zeros(depth.shape, bool)
        if self.fill == "surface":
            surface = np.full(depth.shape, np.nan, np.float32)
            np.exp(logs, out=surface, where=visible)
            surface, completed = complete_surface(surface, guide)
            np.log(surface, out=logs, where=np.isfinite(surface))
            visible |= completed
            interpolated |= completed
        if visible.any():
            logs = self._round_contours(logs, visible, style)

        radius = max(1, round(style.radius * self.scale))
        filtered = cv2.bilateralFilter(np.where(visible, logs, np.float32(sentinel)),
                                       2 * radius + 1, style.sigma_log, max(1.0, radius * 0.8))
        # Invalid sentinels are far outside sigma_log and receive no range weight.
        filtered[~visible] = np.nan
        if history_ok:
            stable = visible & np.isfinite(self.previous) & (np.abs(filtered - self.previous) <= style.jump_log)
            if moving is not None:
                stable &= ~moving
            alpha = -math.expm1(-dt / style.tau)
            filtered[stable] = alpha * filtered[stable] + (1 - alpha) * self.previous[stable]
        # Round concave as well as convex corners of the coverage mask. Only
        # a narrow, locally supported rim may be inferred; the centre of an
        # unsupported hole is left empty.
        sigma = max(0.1, style.feather * self.scale)
        coverage = cv2.GaussianBlur(visible.astype(np.float32), (0, 0), sigma)
        rim = ~visible & (coverage >= 0.5)
        if rim.any():
            radius = max(1, math.ceil(2 * sigma))
            kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
            low = cv2.erode(np.where(visible, filtered, np.float32(1e10)), kernel)
            high = cv2.dilate(np.where(visible, filtered, np.float32(-1e10)), kernel)
            rim &= (high - low) <= 0.065
            if guide is not None:
                expected = cv2.GaussianBlur(np.where(visible, guide, 0), (0, 0), sigma)
                np.divide(expected, coverage, out=expected, where=coverage > 0)
                rim &= np.abs(guide - expected) <= 25
            numerator = cv2.GaussianBlur(np.where(visible, filtered, 0), (0, 0), sigma)
            np.divide(numerator, coverage, out=filtered, where=rim)
            visible |= rim
            interpolated |= rim
        visible &= coverage >= 0.5
        alpha = np.where(visible, np.minimum(1, coverage / 0.8), 0).astype(np.float32)
        filtered[~visible] = np.nan
        self.previous = filtered.copy()
        self.previous[completed] = np.nan
        self.previous_guide = guide.copy() if guide is not None else None
        self.timestamp = timestamp
        output.fill(np.nan)
        np.exp(filtered, out=output, where=visible)
        interpolated &= visible
        return DisplayDepth(output, measured, interpolated, alpha, held & visible, completed & visible)
