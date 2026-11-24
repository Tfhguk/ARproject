"""Capture projector tags with both cameras on the floor and a raised white board."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from calibration_tools import BoardStability
from stereo_live import Camera, StereoEngine, choose_pair, dpi_awareness, label, PROJECTOR
from surface_geometry import StereoGeometry, FloorPlane, ProjectorModel
from work_area import WorkArea, clip_projector


WINDOW = 'Projector 3D calibration: 1-5=patch location Enter=solve Esc=cancel'
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def tag_pattern(size, rectangle, floor=False, preset=0):
    width, height = size
    image = np.zeros((height, width, 3), np.uint8)
    left, top, right, bottom = rectangle
    tags = {}
    if floor:
        image[top:bottom, left:right] = 255
        side, gap = min(112, (right - left) // 5, (bottom - top) // 5), 36
        positions = ((left + gap, top + gap), (right - gap - side, top + gap),
                     (right - gap - side, bottom - gap - side), (left + gap, bottom - gap - side))
        ids = (20, 21, 22, 23)
    else:
        side, gap, margin = 80, 20, 14
        patch_width, patch_height = 2 * side + gap + 2 * margin, side + 2 * margin
        centres = (((left + right) // 2, (top + bottom) // 2),
                   (left + patch_width // 2 + 12, top + patch_height // 2 + 12),
                   (right - patch_width // 2 - 12, top + patch_height // 2 + 12),
                   (right - patch_width // 2 - 12, bottom - patch_height // 2 - 12),
                   (left + patch_width // 2 + 12, bottom - patch_height // 2 - 12))
        cx, cy = centres[preset]
        x, y = cx - patch_width // 2, cy - patch_height // 2
        image[y:y + patch_height, x:x + patch_width] = 255
        positions = ((x + margin, y + margin), (x + margin + side + gap, y + margin))
        ids = (10, 11)
    for marker_id, (x, y) in zip(ids, positions):
        tag = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, side)
        image[y:y + side, x:x + side] = tag[:, :, None]
        tags[marker_id] = np.float32([[x, y], [x + side - 1, y],
                                     [x + side - 1, y + side - 1], [x, y + side - 1]])
    return clip_projector(image, rectangle), tags


def make_detector():
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 3
    parameters.minMarkerPerimeterRate = .01
    return cv2.aruco.ArucoDetector(DICTIONARY, parameters)


def detected_tags(detector, image):
    corners, ids, _ = detector.detectMarkers(image)
    return {} if ids is None else {int(marker_id): corner.reshape(4, 2)
                                   for marker_id, corner in zip(ids.ravel(), corners)}


def floor_alignment(rectified, pixels, area):
    matrix, good = cv2.findHomography(np.float64(rectified), np.float64(pixels), cv2.RANSAC, 2)
    if matrix is None or good.sum() < len(pixels) * .85:
        raise ValueError('Floor tags do not agree; clear the field and keep the rig fixed')
    prediction = cv2.perspectiveTransform(np.float64(rectified).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((prediction[good.ravel() != 0] - pixels[good.ravel() != 0]) ** 2, axis=1))))
    if rms > 1.5:
        raise ValueError('Floor alignment is not accurate enough')
    left, top, right, bottom = area.projector_rect
    corners = np.float64([[left, top], [right - 1, top], [right - 1, bottom - 1], [left, bottom - 1]])
    source = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), np.linalg.inv(matrix)).reshape(-1, 2)
    return WorkArea(source, (640, 360), area.output_size, area.calibration_hash, area.projector_rect), rms


def run(args):
    dpi_awareness()
    digest = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    raw_area = json.loads(args.work_area.read_text(encoding='utf-8'))
    area = WorkArea.load(args.work_area, raw_area['output_size'], digest)
    geometry = StereoGeometry(args.calibration)
    with np.load(args.calibration, allow_pickle=False) as saved:
        indices, focus = saved['camera_indices'], saved['focus_values']
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (args.work_area, args.floor_reference, args.model):
        if path.is_file():
            shutil.copy2(path, args.output / (path.name + '.before'))
    floor = FloorPlane.load(args.floor_reference, digest) if args.reuse_floor else None
    engine = None if floor else StereoEngine((640, 360), 'sgbm', args.calibration, low=.4, high=3.)
    detector = make_detector()
    cameras = [Camera(int(index), float(value)) for index, value in zip(indices, focus)]
    records, descriptors, floor_depths = [], [], deque(maxlen=20)
    next_floor_fit = 0.
    previous, phase, preset, last_saved = (0, 0), 'floor', 0, -100.
    stability = BoardStability(.8, 1.5)
    pattern, expected = tag_pattern(area.output_size, area.projector_rect, floor=True)
    cv2.namedWindow(PROJECTOR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PROJECTOR, *area.output_size)
    cv2.moveWindow(PROJECTOR, 1920, 0)
    cv2.imshow(PROJECTOR, pattern)
    cv2.waitKey(1)
    cv2.setWindowProperty(PROJECTOR, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 650)
    message = 'Clear the entire floor square. Four projected tags are detected automatically.'

    def observations():
        xyz = np.concatenate([record[0] for record in records])
        uv = np.concatenate([record[1] for record in records])
        groups = np.concatenate([np.full(len(record[0]), index) for index, record in enumerate(records)])
        np.savez_compressed(args.output / 'observations.npz', points_3d=xyz, projector_uv=uv, groups=groups,
                            stereo_calibration_sha256=digest)
        return xyz, uv, groups

    try:
        while True:
            pair = choose_pair(cameras[0].snapshot(), cameras[1].snapshot(), previous)
            can_save = False
            cloud = pixels = left_points = right_points = normal = None
            current_height = None
            if pair:
                previous = tuple(frame.sequence for frame in pair)
                stamp = min(frame.timestamp for frame in pair)
                detections = [detected_tags(detector, frame.image) for frame in pair]
                complete = all(set(expected) <= set(found) for found in detections)
                if phase == 'floor' and floor is None:
                    _, depth, _ = engine.compute(pair[0].image, pair[1].image)
                    floor_depths.append(depth)
                    if len(floor_depths) >= 15 and time.perf_counter() >= next_floor_fit:
                        try:
                            floor = FloorPlane.fit(np.stack(floor_depths), geometry.intrinsic,
                                                   area.mask((360, 640)), digest)
                            floor.save(args.floor_reference)
                            print(json.dumps({'floor': floor.quality}), flush=True)
                        except ValueError as exc:
                            message = 'Floor not accepted: ' + str(exc)
                            next_floor_fit = time.perf_counter() + 5
                            np.savez_compressed(args.output / 'floor_diagnostic.npz',
                                                depth=np.stack(floor_depths), region=area.mask((360, 640)),
                                                stereo_calibration_sha256=digest)
                            print(json.dumps({'floor_rejected': str(exc), 'retry_in_s': 5}), flush=True)
                if complete and floor is not None:
                    ids = sorted(expected)
                    left_points, right_points = [np.concatenate([found[i] for i in ids]) for found in detections]
                    pixels = np.concatenate([expected[i] for i in ids])
                    cloud, good = geometry.triangulate(left_points, right_points)
                    stable = stability.update(left_points, right_points, stamp)
                    if good.all():
                        current_height = float(np.median(cloud @ floor.plane[:3] + floor.plane[3]))
                        _, _, axes = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
                        normal = axes[-1]
                        if phase == 'floor':
                            can_save = stable and abs(current_height) < .02
                        else:
                            different = all(np.linalg.norm(cloud.mean(axis=0) - centre) > .04
                                            or abs(normal @ old_normal) < np.cos(np.deg2rad(8)) or preset != old_preset
                                            for centre, old_normal, old_preset in descriptors)
                            can_save = (stable and different and .04 < current_height < 1.1
                                        and time.perf_counter() - last_saved > 1.5)
                else:
                    stability.reset()
                if can_save:
                    if phase == 'floor':
                        try:
                            aligned, rms = floor_alignment(geometry.rectify_points(left_points), pixels, area)
                            aligned.save(args.work_area)
                            area = aligned
                            print(json.dumps({'floor_alignment_rms_projector_px': rms}), flush=True)
                            (args.output / 'floor_alignment.json').write_text(json.dumps({'rms_px': rms,
                                'source_points': area.points.tolist()}, indent=2), encoding='utf-8')
                            records.append((cloud, pixels))
                            observations()
                            if args.floor_only:
                                return
                            phase = 'board'
                            pattern, expected = tag_pattern(area.output_size, area.projector_rect, preset=preset)
                            cv2.imshow(PROJECTOR, pattern)
                            message = 'Cover BOTH tags with WHITE A4 board. Raise/tilt it; hold still. Auto-save.'
                            stability.reset()
                        except ValueError as exc:
                            message = str(exc)
                    else:
                        records.append((cloud, pixels))
                        descriptors.append((cloud.mean(axis=0), normal, preset))
                        last_saved = time.perf_counter()
                        observations()
                        print(json.dumps({'board_pose': len(descriptors), 'height_m': current_height,
                                          'patch': preset + 1}), flush=True)
                        message = 'Saved. Move board to another HEIGHT / TILT. Keys 1-5 move the light patch.'
                preview = np.hstack([cv2.resize(frame.image, (640, 360)) for frame in pair])
                footer = np.zeros((150, 1280, 3), np.uint8)
                label(footer, f'Phase: {phase}; board poses {len(descriptors)}/{args.poses}; tags stable {stability.progress:.0%}')
                label(footer, message, 60)
                label(footer, f'Board height: {current_height:.3f}m' if current_height is not None else 'Need both cameras to see all tags', 92)
                label(footer, 'Use several heights (10-50 cm or higher) and 2+ patch locations. Enter=solve; Esc=cancel', 124)
                cv2.imshow(WINDOW, np.vstack((preview, footer)))
            key = cv2.waitKey(10) & 0xff
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            if phase == 'board' and ord('1') <= key <= ord('5'):
                preset = key - ord('1')
                pattern, expected = tag_pattern(area.output_size, area.projector_rect, preset=preset)
                cv2.imshow(PROJECTOR, pattern)
                stability.reset()
            if phase == 'board' and len(descriptors) >= 8 and (len(descriptors) >= args.poses or key in (10, 13)):
                try:
                    model = ProjectorModel.fit(*observations(), area.output_size, digest)
                    model.save(args.model)
                    (args.output / 'quality.json').write_text(json.dumps(model.quality, indent=2), encoding='utf-8')
                    print(json.dumps({'projector_3d_saved': str(args.model), 'quality': model.quality}), flush=True)
                    return
                except ValueError as exc:
                    message = str(exc)
                    args.poses = len(descriptors) + 2
                    print(json.dumps({'need_more_views': message}), flush=True)
    finally:
        for camera in cameras:
            camera.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=Path('stereo_calibration/stereo_calibration.npz'))
    parser.add_argument('--work-area', type=Path, default=Path('stereo_calibration/work_area.json'))
    parser.add_argument('--floor-reference', type=Path, default=Path('stereo_calibration/floor_reference.json'))
    parser.add_argument('--model', type=Path, default=Path('stereo_calibration/projector_model.json'))
    parser.add_argument('--output', type=Path, default=Path('results/projector_calibration') / str(time.time_ns()))
    parser.add_argument('--poses', type=int, default=12)
    parser.add_argument('--floor-only', action='store_true')
    parser.add_argument('--reuse-floor', action='store_true', help='Use the saved floor instead of capturing a fresh empty reference')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
