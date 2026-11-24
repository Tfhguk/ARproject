"""Capture safeguards and quality checks shared by stereo calibration tools."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

import numpy as np


class BoardStability:
    """Require both views to remain close to an anchor for a continuous interval.

    Comparing with the anchor (rather than the preceding frame) also catches
    slow board drift. Receipt timestamps do not provide sensor synchronization.
    """

    def __init__(self, hold_seconds=0.8, tolerance_px=1.5, max_gap=1.0):
        self.hold_seconds = hold_seconds
        self.tolerance_px = tolerance_px
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.anchor = None
        self.since = self.last = None
        self.samples = 0
        self.progress = 0.0

    def update(self, left, right, timestamp):
        if left is None or right is None:
            self.reset()
            return False
        points = np.stack((left, right)).reshape(2, -1, 2)
        if not np.isfinite(points).all():
            self.reset()
            return False
        restart = self.anchor is None or timestamp <= self.last or timestamp - self.last > self.max_gap
        if not restart:
            drift = np.linalg.norm(points - self.anchor, axis=2)
            restart = np.max(np.quantile(drift, 0.95, axis=1)) > self.tolerance_px
        if restart:
            self.anchor = points.copy()
            self.since, self.samples = timestamp, 0
        self.last = timestamp
        self.samples += 1
        self.progress = min(1.0, (timestamp - self.since) / self.hold_seconds)
        return self.progress >= 1.0 and self.samples >= 3


def assess_calibration(quality, baseline_mm=None, baseline_tolerance_percent=5.0,
                       max_stereo_px=1.0, max_vertical_px=1.5):
    """Image residuals and measured rig baseline are necessary, not metric proof."""
    errors = np.asarray(quality["rectified_y_rms_px"], dtype=float)
    finite = (errors.size > 0 and np.isfinite(errors).all()
              and np.isfinite(quality["stereo_rms_px"])
              and np.isfinite(quality["baseline_m"]) and quality["baseline_m"] > 0)
    geometry = bool(finite and quality["horizontal_rig"]
                    and quality["stereo_rms_px"] <= max_stereo_px
                    and errors.max() <= max_vertical_px)
    delta = None if baseline_mm is None else (1000 * quality["baseline_m"] / baseline_mm - 1) * 100
    baseline_ok = None if delta is None else bool(np.isfinite(delta) and abs(delta) <= baseline_tolerance_percent)
    return {"accepted": geometry and baseline_ok is not False,
            "image_geometry_passed": geometry,
            "measured_baseline_mm": baseline_mm,
            "baseline_difference_percent": delta,
            "baseline_check_passed": baseline_ok,
            "limits": {"stereo_rms_px": max_stereo_px, "max_rectified_y_rms_px": max_vertical_px,
                       "baseline_difference_percent": baseline_tolerance_percent},
            "metric_validation": "pending_known_distance_checks"}


def save_npz_atomic(path, arrays, backup=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if backup and path.exists():
        previous = path.with_name(path.name + f".backup_{time.time_ns()}")
        shutil.copy2(path, previous)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)
    return str(previous) if previous is not None else None


def summarize_distance(samples, reference_m, attempts, tolerance_m=0.02, tolerance_percent=5.0):
    """Score one stationary surface against a ruler, including missing frames."""
    if not np.isfinite(reference_m) or reference_m <= 0:
        raise ValueError("Reference distance must be finite and positive")
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if attempts < len(values) or attempts <= 0:
        raise ValueError("Attempts must include all samples")
    median = float(np.median(values)) if len(values) else None
    error = median - reference_m if median is not None else None
    spread = float(np.ptp(np.percentile(values, [10, 90]))) if len(values) else None
    allowed = max(tolerance_m, reference_m * tolerance_percent / 100)
    sufficient = len(values) >= 10 and len(values) / attempts >= 0.8
    return {"reference_m": reference_m, "measured_median_m": median,
            "error_m": error, "error_percent": 100 * error / reference_m if error is not None else None,
            "temporal_p90_minus_p10_m": spread, "valid_frames": len(values), "attempted_frames": attempts,
            "tolerance_m": allowed,
            "passed_this_point": bool(sufficient and abs(error) <= allowed and spread <= 2 * allowed),
            "scope": "Only this selected surface, distance, position and stationary time interval"}
