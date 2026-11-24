"""Hold a static map and publish current depth at a limited rate during changes."""

from collections import deque
import math
import warnings

import cv2
import numpy as np


def _median(values, axis):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


class SurfaceUpdates:
    def __init__(self, threshold_m=0.02, interval_seconds=1.0, min_area=0.003,
                 block=8, min_samples=3, max_gap=5.0, region=None):
        if (not all(math.isfinite(v) for v in (threshold_m, interval_seconds, min_area, max_gap))
                or threshold_m <= 0 or interval_seconds <= 0 or not 0 < min_area <= 1
                or block < 1 or min_samples < 2 or max_gap <= 0):
            raise ValueError("Invalid change detection settings")
        self.threshold = threshold_m
        self.interval = interval_seconds
        self.min_area, self.block = min_area, int(block)
        self.min_samples, self.max_gap = int(min_samples), max_gap
        self.region = np.array(region, dtype=bool, copy=True) if region is not None else None
        if self.region is not None and (self.region.ndim != 2 or not self.region.any()):
            raise ValueError("Work area mask must be a nonempty image")
        self.reset()

    def reset(self):
        self.published = None
        self.baseline = None
        self.last_stamp = None
        self.published_stamp = None
        self.shape = None
        self.grid_area = None
        self.publications = 0
        self.status = "Collecting initial depth"
        self._clear_history()

    def _clear_history(self):
        self.grids = deque(maxlen=self.min_samples)
        self.stamps = deque(maxlen=self.min_samples)
        self.history_since = None
        self.motion_since = None
        self.motion_active = False

    def _grid(self, depth):
        height, width = depth.shape
        rows, cols = math.ceil(height / self.block), math.ceil(width / self.block)
        padded = np.full((rows * self.block, cols * self.block), np.nan, np.float32)
        padded[:height, :width] = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
        cells = padded.reshape(rows, self.block, cols, self.block).transpose(0, 2, 1, 3)
        cells = cells.reshape(rows, cols, -1)
        enough = np.isfinite(cells).sum(axis=2) >= max(1, math.ceil(self.block ** 2 * 0.35))
        grid = np.where(enough, _median(cells, axis=2), np.nan)
        # Stereo noise is spatially correlated at floor distance. Compare a
        # robust neighbourhood, while keeping unmeasured cells unmeasured.
        padded = np.pad(grid, 1, mode="edge")
        neighbours = np.stack([padded[y:y + rows, x:x + cols] for y in range(3) for x in range(3)])
        return np.where(np.isfinite(grid), _median(neighbours, axis=0), np.nan)

    def _changed(self, first, second, threshold):
        common = np.isfinite(first) & np.isfinite(second)
        # Missing measurements do not count as a change of height.
        if common.sum() < max(1, (self.grid_area or first.size) * 0.2):
            return None
        changed = common & (np.abs(first - second) > threshold)
        return self._significant(changed)

    def _significant(self, changed):
        if not changed.any():
            return False
        _, _, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
        minimum = max(2, math.ceil((self.grid_area or changed.size) * self.min_area))
        return bool(np.any(stats[1:, cv2.CC_STAT_AREA] >= minimum))

    def _confirmed_change(self):
        grids = np.stack(self.grids)
        common = np.isfinite(self.baseline) & np.isfinite(grids).all(axis=0)
        difference = grids - self.baseline
        changed = common & ((difference > self.threshold).all(axis=0)
                            | (difference < -self.threshold).all(axis=0))
        return self._significant(changed)

    def _moving(self):
        if len(self.grids) < 3:
            return False
        first, middle, last = list(self.grids)[-3:]
        # Motion needs stronger evidence than a persistent height change:
        # correlated floor noise can exceed the ordinary threshold once.
        motion_threshold = self.threshold * 1.5
        if (self._changed(first, middle, motion_threshold) is True
                and self._changed(middle, last, motion_threshold) is True):
            return True
        # Also detect gradual movement below the per-probe threshold. Both
        # increments must have the same direction and exceed tiny stereo jitter.
        common = np.isfinite(first) & np.isfinite(middle) & np.isfinite(last)
        a, b = middle - first, last - middle
        drift = common & (a * b > 0) & (np.abs(a) > self.threshold * .5 + 1e-6)
        drift &= (np.abs(b) > self.threshold * .5 + 1e-6)
        drift &= np.abs(last - first) >= motion_threshold - 1e-6
        return self._significant(drift)

    def observe(self, depth, timestamp):
        depth = np.asarray(depth, np.float32)
        if depth.ndim != 2 or not math.isfinite(timestamp):
            raise ValueError("Need a depth matrix and a finite timestamp")
        if self.region is not None and self.region.shape != depth.shape:
            raise ValueError("Work area mask must match depth resolution")
        if self.shape is not None and self.shape != depth.shape:
            self.reset()
        if self.last_stamp is not None and timestamp <= self.last_stamp:
            return None
        if self.last_stamp is not None and timestamp - self.last_stamp > self.max_gap:
            self._clear_history()
        if self.shape is None and self.region is not None:
            h, w = depth.shape
            rows, cols = math.ceil(h / self.block), math.ceil(w / self.block)
            padded = np.zeros((rows * self.block, cols * self.block), bool)
            padded[:h, :w] = self.region
            cells = padded.reshape(rows, self.block, cols, self.block)
            self.grid_area = int((cells.sum(axis=(1, 3)) >= math.ceil(self.block ** 2 * .35)).sum())
            if not self.grid_area:
                raise ValueError("Work area is too small for the change detector")
        self.shape = depth.shape
        self.last_stamp = timestamp
        if self.region is not None:
            depth = np.where(self.region, depth, np.nan)
        grid = self._grid(depth)
        if np.isfinite(grid).sum() < max(1, (self.grid_area or grid.size) * 0.2):
            self._clear_history()
            self.status = "No reliable depth; holding saved map"
            return None
        if self.history_since is None:
            self.history_since = timestamp
        self.grids.append(grid)
        self.stamps.append(timestamp)
        if len(self.grids) < self.min_samples:
            self.status = "Collecting fresh depth observations"
            return None

        moving = self._moving()
        if self.baseline is None:
            if timestamp - self.history_since < self.interval:
                self.status = "Collecting initial depth"
                return None
        else:
            changed = self._changed(self.baseline, grid, self.threshold)
            if moving:
                if self.motion_since is None:
                    # Include the interval leading up to the first observed move.
                    self.motion_since = self.stamps[0]
            else:
                self.motion_since = None
                self.motion_active = False
            motion_ready = (moving and timestamp - self.motion_since >= self.interval
                            and (changed is True or self.motion_active))
            confirmed = changed is True and self._confirmed_change()
            if not motion_ready and not confirmed:
                self.status = ("Motion detected; checking next observation" if moving
                               else "Map unchanged" if changed is False
                               else "Checking depth change" if changed is True
                               else "Insufficient overlap; holding map")
                return None
            if timestamp - self.published_stamp < self.interval:
                self.status = "Scene changing; waiting for next update slot"
                return None
            if motion_ready:
                self.motion_active = True

        # Publish the latest measured surface. A temporal depth median would
        # merge different object positions and leave trails during motion.
        accepted = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float32)
        self.published = accepted.copy()
        # During motion the reference must describe the published position.
        # Otherwise its lag would trigger a redundant update after stopping.
        reference = grid if moving else _median(np.stack(self.grids), axis=0)
        self.baseline = np.where(np.isfinite(grid), reference, np.nan)
        self.published_stamp = timestamp
        self.publications += 1
        self.status = "Map updated; monitoring changes"
        return accepted
