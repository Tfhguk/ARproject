"""Create a printable board, then collect stereo chessboard views and calibrate."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from calibration_tools import BoardStability, assess_calibration, save_npz_atomic
from stereo_live import Camera, CAPTURE_SIZE, choose_pair, dpi_awareness, label


def make_board(directory, columns, rows, square_mm):
    """Exact physical-size vector PDF and SVG, A4 landscape."""
    width, height = (columns + 1) * square_mm, (rows + 1) * square_mm
    if width > 277 or height > 185:
        raise ValueError("Board must fit A4 landscape with margins; reduce square-mm")
    directory.mkdir(parents=True, exist_ok=True)
    x0, y0 = (297 - width) / 2, (210 - height) / 2
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210">',
           '<rect width="297" height="210" fill="white"/>']
    mm = 72 / 25.4
    commands = []
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                x, y = x0 + column * square_mm, y0 + row * square_mm
                svg.append(f'<rect x="{x}" y="{y}" width="{square_mm}" height="{square_mm}"/>')
                commands.append(f"{x*mm:.5f} {(210-y-square_mm)*mm:.5f} {square_mm*mm:.5f} {square_mm*mm:.5f} re f")
    text = f"{columns} x {rows} INNER corners | square = {square_mm:g} mm | A4 landscape | PRINT AT 100%"
    svg.append(f'<text x="12" y="204" font-family="sans-serif" font-size="3">{text}</text></svg>')
    (directory / "chessboard.svg").write_text("\n".join(svg), encoding="utf-8")
    commands.append(f"BT /F1 9 Tf 25 12 Td ({text}) Tj ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {297*mm:.5f} {210*mm:.5f}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
               f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    (directory / "chessboard.pdf").write_bytes(pdf)
    print(f"Board: {directory / 'chessboard.pdf'}; print at 100%, measure the actual square", flush=True)


def find_corners(image, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1),
                                  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        # OpenCV builds can return either (N, 2) or (N, 1, 2).
        # Keep one layout for descriptors, saved observations and calibration.
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return found, corners


def solve_calibration(objects, left, right, size=CAPTURE_SIZE):
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
    rms1, k1, d1, _, _ = cv2.calibrateCamera(objects, left, size, None, None)
    rms2, k2, d2, _, _ = cv2.calibrateCamera(objects, right, size, None, None)
    rms, k1, d1, k2, d2, r, t, e, f = cv2.stereoCalibrate(
        objects, left, right, k1, d1, k2, d2, size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, size, r, t,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    errors = []
    for pts1, pts2 in zip(left, right):
        a = cv2.undistortPoints(pts1, k1, d1, R=r1, P=p1)
        b = cv2.undistortPoints(pts2, k2, d2, R=r2, P=p2)
        errors.append(float(np.sqrt(np.mean((a[:, 0, 1] - b[:, 0, 1]) ** 2))))
    arrays = {"K1": k1, "D1": d1, "K2": k2, "D2": d2, "R": r, "T": t,
              "capture_size": np.int32(size)}
    quality = {"left_rms_px": rms1, "right_rms_px": rms2, "stereo_rms_px": rms,
               "rectified_y_rms_px": errors, "baseline_m": float(np.linalg.norm(t)),
               "horizontal_rig": bool(abs(p2[0, 3]) > abs(p2[1, 3])), "views": len(objects)}
    return arrays, quality


def collect(args):
    dpi_awareness()
    args.directory.mkdir(parents=True, exist_ok=True)
    session = args.directory / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    session.mkdir()
    print(f"Capture session: {session}", flush=True)
    cameras = [Camera(args.left, args.focus_left), Camera(args.right, args.focus_right)]
    board = (args.columns, args.rows)
    grid = np.zeros((args.rows * args.columns, 3), np.float32)
    grid[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * (args.square_mm / 1000)
    objects, left_points, right_points, descriptors = [], [], [], []
    timestamps = []
    stability = BoardStability(args.hold_seconds, args.still_px)
    previous, last_saved = (0, 0), -10
    started = time.perf_counter()
    last_pair = started
    window = "Stereo calibration: SPACE=save view  ENTER=solve  Esc=cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 500)
    message = "Show the whole board in BOTH views, hold still, then move to a new pose"
    manifest = {"status": "collecting", "square_mm": args.square_mm,
                "board_inner_corners": list(board), "capture_size": list(CAPTURE_SIZE),
                "camera_indices": [args.left, args.right], "measured_baseline_mm": args.baseline_mm,
                "hold_seconds": args.hold_seconds, "still_tolerance_px": args.still_px}
    (session / "session.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        while True:
            pair = choose_pair(cameras[0].snapshot(), cameras[1].snapshot(), previous, 0.05)
            if pair is None:
                if cv2.waitKey(10) & 0xFF == 27:
                    return
                if time.perf_counter() - last_pair > 30:
                    raise OSError("No paired frames; check cameras and software receipt timing")
                continue
            last_pair = time.perf_counter()
            previous = (pair[0].sequence, pair[1].sequence)
            found1, corners1 = find_corners(pair[0].image, board)
            found2, corners2 = find_corners(pair[1].image, board)
            stable = stability.update(corners1 if found1 else None, corners2 if found2 else None,
                                      max(frame.timestamp for frame in pair))
            different = False
            if found1 and found2:
                descriptor = np.concatenate([
                    np.concatenate((np.mean(points[:, 0], axis=0), np.ptp(points[:, 0], axis=0),
                                    points[-1, 0] - points[0, 0])) / np.tile(CAPTURE_SIZE, 3)
                    for points in (corners1, corners2)])
                different = not descriptors or min(np.linalg.norm(descriptor - old) for old in descriptors) > 0.14
            previews = []
            for frame, found, corners in zip(pair, (found1, found2), (corners1, corners2)):
                image = frame.image.copy()
                if found:
                    cv2.drawChessboardCorners(image, board, corners, True)
                previews.append(cv2.resize(image, (640, 360)))
            preview = np.hstack(previews)
            label(preview, f"Views: {len(objects)}/{args.views}; board visible: {found1}/{found2}")
            status = "READY" if stable else f"HOLD STILL: {stability.progress:.0%}"
            if stable and not different:
                status = "Already saved: move board to a NEW position / distance / tilt"
            label(preview, status, 58, (0, 255, 0) if stable else (0, 220, 255))
            label(preview, message, 335)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return
            if found1 and found2:
                if key == 32 and not stable:
                    message = "Not saved: hold the board STILL until READY, then press SPACE"
                if stable and (key == 32 or (args.auto and different and time.perf_counter() - last_saved > 1.5)):
                    objects.append(grid.copy())
                    left_points.append(corners1.copy())
                    right_points.append(corners2.copy())
                    descriptors.append(descriptor)
                    timestamps.append([frame.timestamp for frame in pair])
                    last_saved = time.perf_counter()
                    for side, frame in zip(("left", "right"), pair):
                        (session / f"view_{len(objects):02d}_{side}.png").write_bytes(cv2.imencode(".png", frame.image)[1].tobytes())
                    save_npz_atomic(session / "observations.npz", {
                        "object_points": np.stack(objects), "left_points": np.stack(left_points),
                        "right_points": np.stack(right_points), "receipt_timestamps": np.asarray(timestamps),
                        "capture_size": np.int32(CAPTURE_SIZE), "square_mm": np.float64(args.square_mm),
                        "board_inner_corners": np.int32(board), "camera_indices": np.int32([args.left, args.right]),
                        "focus_values": np.float64([camera.info.get("focus", 0) for camera in cameras]),
                        "measured_baseline_mm": np.float64(args.baseline_mm if args.baseline_mm else np.nan)})
                    message = f"Saved {len(objects)}. Move to a NEW position / distance / tilt, then hold still"
                    print(f"Captured {len(objects)}/{args.views}", flush=True)
            if len(objects) >= args.views or key in (10, 13) and len(objects) >= 12:
                break
        print("Solving stereo calibration...", flush=True)
        arrays, quality = solve_calibration(objects, left_points, right_points)
        quality.update(assess_calibration(quality, args.baseline_mm))
        accepted = quality["accepted"]
        destination = args.output if accepted else args.output.with_name(args.output.stem + ".candidate.npz")
        arrays.update(camera_indices=np.int32([args.left, args.right]),
                      focus_values=np.float64([camera.info.get("focus", 0) for camera in cameras]),
                      square_mm=np.float64(args.square_mm), board_inner_corners=np.int32(board),
                      measured_baseline_mm=np.float64(args.baseline_mm if args.baseline_mm else np.nan),
                      source_session=np.str_(session.resolve()), metric_validated=np.bool_(False))
        save_npz_atomic(session / "solution.npz", arrays)
        backup = save_npz_atomic(destination, arrays, backup=True)
        quality.update({"accepted": accepted, "file": str(destination),
                        "square_mm": args.square_mm, "board_inner_corners": list(board),
                        "source_session": str(session), "previous_calibration_backup": backup,
                        "camera_settings": [camera.info for camera in cameras]})
        (session / "calibration_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
        (args.directory / "latest_session.txt").write_text(str(session.resolve()), encoding="utf-8")
        manifest["status"] = "solved" if accepted else "candidate_only"
        print(json.dumps(quality, indent=2), flush=True)
        if not accepted:
            print("Candidate only: improve board views/rig geometry and recalibrate; active calibration unchanged.")
    finally:
        if manifest["status"] == "collecting":
            manifest["status"] = "incomplete"
        manifest["views"] = len(objects)
        (session / "session.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for camera in cameras:
            camera.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-board", action="store_true", help="Generate printable PDF/SVG and exit")
    parser.add_argument("--columns", type=int, default=9, help="INNER corners horizontally")
    parser.add_argument("--rows", type=int, default=6, help="INNER corners vertically")
    parser.add_argument("--square-mm", type=float, default=None,
                        help="Measured physical square side; required for calibration (board generation default: 25)")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--focus-left", type=float)
    parser.add_argument("--focus-right", type=float)
    parser.add_argument("--views", type=int, default=30)
    parser.add_argument("--auto", action="store_true", help="Save distinct board views automatically")
    parser.add_argument("--hold-seconds", type=float, default=0.8, help="Continuous stationary interval before saving")
    parser.add_argument("--still-px", type=float, default=1.5, help="Maximum board drift in capture pixels")
    parser.add_argument("--baseline-mm", type=float, help="Measured lens separation; reject discrepancy above 5 percent")
    parser.add_argument("--output", type=Path, default=Path("stereo_calibration/stereo_calibration.npz"))
    parser.add_argument("--directory", type=Path, default=Path("stereo_calibration"))
    args = parser.parse_args()
    if args.square_mm is None:
        if args.make_board:
            args.square_mm = 25
        else:
            parser.error("Specify the measured square size with --square-mm, e.g. --square-mm 25 for a 25 mm square")
    if args.columns < 3 or args.rows < 3 or not np.isfinite(args.square_mm) or args.square_mm <= 0 or args.views < 12:
        parser.error("Need columns/rows >= 3, positive square-mm, views >= 12")
    if args.left == args.right or min(args.left, args.right) < 0:
        parser.error("Select two different nonnegative camera indices")
    if any(not np.isfinite(value) or value <= 0 for value in (args.hold_seconds, args.still_px)):
        parser.error("hold-seconds and still-px must be finite and positive")
    if args.baseline_mm is not None and (not np.isfinite(args.baseline_mm) or args.baseline_mm <= 0):
        parser.error("baseline-mm must be finite and positive")
    if args.make_board:
        make_board(args.directory, args.columns, args.rows, args.square_mm)
    else:
        collect(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
