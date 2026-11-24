"""Re-solve saved stereo board views, rejecting a limited number of outliers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration_tools import assess_calibration, save_npz_atomic
from stereo_calibrate import find_corners, solve_calibration
from stereo_live import CAPTURE_SIZE


def refine_points(objects, left, right, names=None, min_views=12, max_error=1.5):
    if len(objects) != len(left) or len(left) != len(right) or len(objects) < min_views:
        raise ValueError("Not enough matching stereo observations")
    names = list(range(len(objects))) if names is None else list(names)
    kept = list(range(len(objects)))
    rejected, history = [], []
    # Retain at least 75% of the observations and at least min_views.
    maximum_rejections = min(len(objects) - min_views, len(objects) // 4)
    while True:
        arrays, quality = solve_calibration([objects[i] for i in kept],
                                            [left[i] for i in kept], [right[i] for i in kept])
        errors = np.asarray(quality["rectified_y_rms_px"])
        finite = np.isfinite(errors).all() and math.isfinite(quality["stereo_rms_px"])
        checks = assess_calibration(quality, max_stereo_px=min(1.0, max_error), max_vertical_px=max_error)
        accepted = checks["accepted"]
        history.append({"views": len(kept), "stereo_rms_px": quality["stereo_rms_px"],
                        "max_rectified_y_rms_px": float(errors.max()), "baseline_m": quality["baseline_m"]})
        if accepted or not finite or not quality["horizontal_rig"] or len(rejected) >= maximum_rejections:
            quality.update(checks)
            quality.update({"accepted": accepted, "kept_views": [names[i] for i in kept],
                            "rejected_views": rejected, "refinement_history": history})
            return arrays, quality
        worst = int(np.argmax(errors))
        rejected.append({"view": names[kept[worst]], "rectified_y_rms_px": float(errors[worst])})
        kept.pop(worst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("stereo_calibration"))
    parser.add_argument("--metadata", type=Path, help="Legacy JPG session metadata; new sessions carry their own")
    parser.add_argument("--baseline-mm", type=float, help="Measured lens separation (new sessions inherit saved value)")
    parser.add_argument("--square-mm", type=float, required=True, help="Measured PHYSICAL square side")
    parser.add_argument("--correct-square-size", action="store_true",
                        help="Re-solve at a newly measured square size; preserve original captured observations")
    parser.add_argument("--columns", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("stereo_calibration/stereo_calibration.npz"))
    args = parser.parse_args()
    if not math.isfinite(args.square_mm) or args.square_mm <= 0 or min(args.columns, args.rows) < 3:
        parser.error("Need positive finite square-mm and columns/rows >= 3")
    if args.baseline_mm is not None and (not math.isfinite(args.baseline_mm) or args.baseline_mm <= 0):
        parser.error("baseline-mm must be finite and positive")
    objects, left, right, names, failed = [], [], [], [], []
    board = (args.columns, args.rows)
    observations = args.directory / "observations.npz"
    baseline_mm = args.baseline_mm
    square_size_correction = None
    if observations.is_file():
        source = observations
        with np.load(source, allow_pickle=False) as saved:
            if tuple(saved["capture_size"]) != CAPTURE_SIZE or tuple(saved["board_inner_corners"]) != board:
                raise ValueError("Saved capture size or board dimensions do not match")
            source_square_mm = float(saved["square_mm"])
            if not np.isclose(source_square_mm, args.square_mm) and not args.correct_square_size:
                raise ValueError("square-mm differs from this capture session; check the physical board")
            indices, focus = saved["camera_indices"].copy(), saved["focus_values"].copy()
            objects = list(saved["object_points"].copy())
            left, right = list(saved["left_points"].copy()), list(saved["right_points"].copy())
            if baseline_mm is None and np.isfinite(saved["measured_baseline_mm"]):
                baseline_mm = float(saved["measured_baseline_mm"])
        if not np.isclose(source_square_mm, args.square_mm):
            factor = args.square_mm / source_square_mm
            objects = [np.asarray(points * factor, dtype=np.float32) for points in objects]
            square_size_correction = {"captured_square_mm": source_square_mm,
                                      "measured_square_mm": args.square_mm,
                                      "object_point_scale_factor": factor,
                                      "method": "Re-solve original image observations with corrected board geometry"}
            print(f"Correcting board geometry: {source_square_mm:g} -> {args.square_mm:g} mm; originals preserved", flush=True)
        names = [f"view_{i + 1:02d}" for i in range(len(objects))]
        print(f"Loaded {len(objects)} original observations from {source}", flush=True)
    else:
        if any(args.directory.glob("session_*")):
            parser.error("Select the specific session_... directory; do not mix capture sessions")
        source = args.metadata or Path("stereo_calibration.candidate.npz")
        if not source.is_file():
            parser.error("Metadata file missing; use --metadata with this legacy capture session")
        with np.load(source, allow_pickle=False) as saved:
            indices, focus = saved["camera_indices"].copy(), saved["focus_values"].copy()
        grid = np.zeros((args.columns * args.rows, 3), np.float32)
        grid[:, :2] = np.mgrid[:args.columns, :args.rows].T.reshape(-1, 2) * (args.square_mm / 1000)
        for path in sorted(args.directory.glob("view_*_left.jpg")):
            other = path.with_name(path.name.replace("_left.jpg", "_right.jpg"))
            pair = [cv2.imdecode(np.frombuffer(item.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    for item in (path, other)]
            if any(image is None or image.shape[:2] != CAPTURE_SIZE[::-1] for image in pair):
                raise ValueError(f"Invalid capture size in {path.name}")
            found1, corners1 = find_corners(pair[0], board)
            found2, corners2 = find_corners(pair[1], board)
            name = path.name.replace("_left.jpg", "")
            if not found1 or not found2:
                failed.append(name)
                continue
            objects.append(grid.copy())
            left.append(corners1)
            right.append(corners2)
            names.append(name)
            print(f"Detected {name}", flush=True)
    arrays, quality = refine_points(objects, left, right, names)
    quality.update(assess_calibration(quality, baseline_mm))
    destination = args.output if quality["accepted"] else args.output.with_name(args.output.stem + ".candidate.npz")
    quality.update({"square_mm": args.square_mm, "board_inner_corners": list(board),
                    "square_size_correction": square_size_correction,
                    "failed_detection": failed, "camera_indices": indices.tolist(),
                    "focus_values": focus.tolist(), "source_metadata": str(source),
                    "file": str(destination),
                    "notes": "Training-view reprojection checks only. Verify baseline and known distances physically."})
    arrays.update(camera_indices=indices, focus_values=focus, square_mm=np.float64(args.square_mm),
                  board_inner_corners=np.int32(board), metric_validated=np.bool_(False),
                  measured_baseline_mm=np.float64(baseline_mm if baseline_mm else np.nan),
                  source_session=np.str_(args.directory.resolve()))
    if square_size_correction is not None:
        arrays.update(source_square_mm=np.float64(source_square_mm),
                      object_point_scale_factor=np.float64(square_size_correction["object_point_scale_factor"]))
    quality["previous_calibration_backup"] = save_npz_atomic(destination, arrays, backup=True)
    (args.directory / "refinement_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)
    return 0 if quality["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
