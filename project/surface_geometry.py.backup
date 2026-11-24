"""Metric floor heights and a 3D camera-to-projector model."""

import json
from pathlib import Path
import time
import warnings

import cv2
import numpy as np

from rendering import Renderer
from depth_completion import nearest_values
from work_area import clip_projector


class StereoGeometry:
    def __init__(self, calibration, size=(640, 360)):
        self.size = tuple(size)
        with np.load(calibration, allow_pickle=False) as saved:
            self.intrinsics = [saved['K1'].copy(), saved['K2'].copy()]
            self.distortions = [saved['D1'].copy(), saved['D2'].copy()]
            scaled = [k.copy() for k in self.intrinsics]
            for k in scaled:
                k[0] *= size[0] / 1280
                k[1] *= size[1] / 720
            r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
                scaled[0], self.distortions[0], scaled[1], self.distortions[1], size,
                saved['R'], saved['T'].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.rotations, self.projections = (r1, r2), (p1, p2)
        self.intrinsic = p1[:, :3]

    def rectify_points(self, points, camera=0):
        return cv2.undistortPoints(np.asarray(points, np.float64).reshape(-1, 1, 2),
                                  self.intrinsics[camera], self.distortions[camera],
                                  R=self.rotations[camera], P=self.projections[camera]).reshape(-1, 2)

    def triangulate(self, left, right):
        a, b = self.rectify_points(left), self.rectify_points(right, 1)
        homogeneous = cv2.triangulatePoints(*self.projections, a.T, b.T)
        with np.errstate(divide='ignore', invalid='ignore'):
            points = (homogeneous[:3] / homogeneous[3]).T
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > .25) & (points[:, 2] < 4)
        valid &= np.abs(a[:, 1] - b[:, 1]) < 1.2
        return points, valid


def xyz_from_depth(depth, intrinsic):
    y, x = np.indices(depth.shape, dtype=np.float32)
    return np.stack(((x - intrinsic[0, 2]) * depth / intrinsic[0, 0],
                     (y - intrinsic[1, 2]) * depth / intrinsic[1, 1], depth), axis=-1)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class FloorPlane:
    def __init__(self, plane, calibration_hash, quality=None):
        plane = np.asarray(plane, np.float64)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
            raise ValueError('Invalid floor plane')
        self.plane = plane / np.linalg.norm(plane[:3])
        if self.plane[2] > 0:
            self.plane *= -1
        if self.plane[3] <= .25 or abs(self.plane[2]) < .4:
            raise ValueError('Floor plane is inconsistent with an overhead camera')
        self.calibration_hash, self.quality = calibration_hash, quality or {}

    @classmethod
    def fit(cls, depths, intrinsic, region, calibration_hash):
        depths = np.asarray(depths, np.float32)
        if depths.ndim != 3 or len(depths) < 5 or region.shape != depths.shape[1:]:
            raise ValueError('Need at least five matching floor depth frames')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median = np.nanmedian(depths, axis=0)
            spread = np.nanmedian(np.abs(depths - median), axis=0)
        reliable = region & np.isfinite(median) & (median > 0) & (spread < .01)
        reliable &= np.isfinite(depths).sum(axis=0) >= len(depths) * .6
        if reliable.sum() < max(200, region.sum() * .4):
            raise ValueError('Not enough stable measured floor; clear the work area and retry')
        cloud = xyz_from_depth(median, intrinsic)[reliable].astype(np.float64)
        cloud = cloud[np.linspace(0, len(cloud) - 1, min(5000, len(cloud))).astype(int)]
        rng, best = np.random.default_rng(197), None
        for _ in range(160):
            a, b, c = cloud[rng.choice(len(cloud), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)
            if length < 1e-6:
                continue
            normal /= length
            good = np.abs(cloud @ normal - a @ normal) < .008
            if best is None or good.sum() > best.sum():
                best = good
        if best is None or best.mean() < .75:
            raise ValueError('The reference is not a clean flat floor')
        inliers = cloud[best]
        centre = inliers.mean(axis=0)
        _, _, axes = np.linalg.svd(inliers - centre, full_matrices=False)
        normal = axes[-1]
        plane = np.r_[normal, -normal @ centre]
        errors = np.abs(inliers @ normal + plane[3])
        quality = {'frames': len(depths), 'measured_points': int(reliable.sum()),
                   'plane_inlier_fraction': float(best.mean()),
                   'plane_rms_m': float(np.sqrt(np.mean(errors ** 2))),
                   'plane_p95_m': float(np.percentile(errors, 95))}
        if quality['plane_rms_m'] > .006:
            raise ValueError('Floor fit is too noisy; keep the empty floor stationary')
        return cls(plane, calibration_hash, quality)

    def heights(self, depth, intrinsic):
        y, x = np.ogrid[:depth.shape[0], :depth.shape[1]]
        n = self.plane
        factor = n[0] * (x - intrinsic[0, 2]) / intrinsic[0, 0]
        factor = factor + n[1] * (y - intrinsic[1, 2]) / intrinsic[1, 1] + n[2]
        return np.where(np.isfinite(depth) & (depth > 0), depth * factor + n[3], np.nan).astype(np.float32)

    def save(self, path):
        _write_json(path, {'version': 1, 'coordinate_space': 'rectified_left', 'plane_m': self.plane.tolist(),
                           'stereo_calibration_sha256': self.calibration_hash, 'quality': self.quality,
                           'saved_unix_time': time.time()})

    @classmethod
    def load(cls, path, calibration_hash):
        config = json.loads(Path(path).read_text(encoding='utf-8'))
        if (config.get('version') != 1 or config.get('coordinate_space') != 'rectified_left'
                or config.get('stereo_calibration_sha256') != calibration_hash):
            raise ValueError('Floor reference does not match stereo calibration; calibrate the empty floor again')
        return cls(config['plane_m'], calibration_hash, config.get('quality'))


class HeightRenderer:
    def __init__(self, floor, intrinsic, maximum=.5, gamma=.5, noise=.005):
        if (not np.isfinite([maximum, gamma, noise]).all() or maximum <= noise
                or noise < 0 or gamma <= 0):
            raise ValueError('Invalid height colour settings')
        self.floor, self.intrinsic = floor, intrinsic
        self.maximum, self.gamma, self.noise = maximum, gamma, noise
        self.palette = Renderer(0, 1, bright=True, ascending=True)

    def colorize(self, depth):
        heights = self.floor.heights(depth, self.intrinsic)
        normalized = np.clip((heights - self.noise) / (self.maximum - self.noise), 0, 1)
        colors = self.palette.colorize(np.power(normalized, self.gamma))
        colors[~self.allowed(depth)] = 0
        return colors

    def allowed(self, depth):
        heights = self.floor.heights(depth, self.intrinsic)
        return np.isfinite(heights) & (heights >= -self.noise) & (heights <= self.maximum + 1e-6)

    def render(self, display, measured_depth):
        heights = self.floor.heights(measured_depth, self.intrinsic)
        above = np.isfinite(heights) & (heights > self.maximum + 1e-6)
        blocked = above.copy()
        if above.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            enclosed = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            blocked |= enclosed & ~np.isfinite(measured_depth)
        # Black high objects remain opaque geometry for the projector z-buffer.
        # Their measured surface must not turn into a lower filled surface.
        display.depth[above] = measured_depth[above]
        display.measured[above] = True
        for name in ('interpolated', 'held', 'completed'):
            values = getattr(display, name)
            if values is not None:
                values[above] = False
        enclosed_unknown = blocked & ~np.isfinite(measured_depth)
        if enclosed_unknown.any():
            estimated = nearest_values(np.where(above, measured_depth, 0), above)
            display.depth[enclosed_unknown] = estimated[enclosed_unknown]
            display.measured[enclosed_unknown] = False
            for name in ('interpolated', 'completed'):
                if getattr(display, name) is None:
                    setattr(display, name, np.zeros(measured_depth.shape, bool))
                getattr(display, name)[enclosed_unknown] = True
        colors = display.colorize(self)
        colors[blocked] = 0
        return colors


def project_points(matrix, points):
    homogeneous = np.asarray(points) @ matrix[:, :3].T + matrix[:, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        return homogeneous[:, :2] / homogeneous[:, 2:3], homogeneous[:, 2]


def _dlt(points, pixels):
    points, pixels = np.asarray(points, np.float64), np.asarray(pixels, np.float64)
    if (len(points) < 6 or not np.isfinite(points).all() or not np.isfinite(pixels).all()
            or np.linalg.matrix_rank(points - points.mean(axis=0)) < 3):
        raise ValueError('Projector samples must include several physical heights')
    centre3, centre2 = points.mean(axis=0), pixels.mean(axis=0)
    scale3 = np.sqrt(3) / np.sqrt(np.mean(np.sum((points - centre3) ** 2, axis=1)))
    scale2 = np.sqrt(2) / np.sqrt(np.mean(np.sum((pixels - centre2) ** 2, axis=1)))
    t3, t2 = np.eye(4), np.eye(3)
    t3[:3, :3] *= scale3
    t3[:3, 3] = -centre3 * scale3
    t2[:2, :2] *= scale2
    t2[:2, 2] = -centre2 * scale2
    xyz = np.c_[(points - centre3) * scale3, np.ones(len(points))]
    uv = (pixels - centre2) * scale2
    a = np.zeros((2 * len(points), 12), np.float64)
    a[0::2, :4], a[1::2, 4:8] = xyz, xyz
    a[0::2, 8:], a[1::2, 8:] = -uv[:, :1] * xyz, -uv[:, 1:] * xyz
    _, _, vectors = np.linalg.svd(a, full_matrices=False)
    matrix = np.linalg.inv(t2) @ vectors[-1].reshape(3, 4) @ t3
    matrix /= np.linalg.norm(matrix[2, :3])
    if np.median(points @ matrix[2, :3] + matrix[2, 3]) < 0:
        matrix *= -1
    return matrix


class ProjectorModel:
    def __init__(self, matrix, output_size, calibration_hash, quality=None):
        self.matrix = np.asarray(matrix, np.float64)
        if self.matrix.shape != (3, 4) or not np.isfinite(self.matrix).all() or np.linalg.matrix_rank(self.matrix[:, :3]) < 3:
            raise ValueError('Invalid 3D projector matrix')
        self.output_size, self.calibration_hash = tuple(output_size), calibration_hash
        self.quality = quality or {}

    @classmethod
    def fit(cls, points, pixels, groups, output_size, calibration_hash):
        points, pixels, groups = np.asarray(points), np.asarray(pixels), np.asarray(groups)
        if points.shape != (len(pixels), 3) or pixels.shape[1:] != (2,) or len(groups) != len(points):
            raise ValueError('Need matching 3D points, projector pixels and capture groups')
        if len(points) < 60 or len(np.unique(groups)) < 8:
            raise ValueError('Collect at least eight poses of the white board')
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[2] / singular[0] < .025 or np.ptp(points[:, 2]) < .10:
            raise ValueError('Need non-coplanar views: raise and tilt the board at least 10 cm')
        rng, best = np.random.default_rng(217), None
        for _ in range(160):
            sample = rng.choice(len(points), 12, replace=False)
            if np.linalg.matrix_rank(np.c_[points[sample], np.ones(12)]) < 4:
                continue
            candidate = _dlt(points[sample], pixels[sample])
            projected, distance = project_points(candidate, points)
            good = (np.linalg.norm(projected - pixels, axis=1) < 4) & (distance > 0)
            if best is None or good.sum() > best.sum():
                best = good
        if best is None or best.mean() < .85:
            raise ValueError('Projector matches are inconsistent; collect steadier board views')
        matrix = _dlt(points[best], pixels[best])
        projected, _ = project_points(matrix, points[best])
        error = np.linalg.norm(projected - pixels[best], axis=1)
        validation = []
        for group in np.unique(groups):
            train, test = best & (groups != group), best & (groups == group)
            if not test.any() or train.sum() < 12:
                continue
            prediction, _ = project_points(_dlt(points[train], pixels[train]), points[test])
            validation.extend(np.linalg.norm(prediction - pixels[test], axis=1))
        quality = {'points': len(points), 'poses': len(np.unique(groups)), 'inlier_fraction': float(best.mean()),
                   'rms_px': float(np.sqrt(np.mean(error ** 2))), 'p95_px': float(np.percentile(error, 95)),
                   'leave_view_out_rms_px': float(np.sqrt(np.mean(np.square(validation)))),
                   'depth_span_m': float(np.ptp(points[:, 2]))}
        if (not all(np.isfinite(value) for value in quality.values())
                or quality['rms_px'] > 2.5 or quality['leave_view_out_rms_px'] > 5):
            raise ValueError('Projector validation error too high; add poses across the work area and heights')
        return cls(matrix, output_size, calibration_hash, quality)

    def save(self, path):
        _write_json(path, {'version': 1, 'coordinate_space': 'rectified_left', 'matrix': self.matrix.tolist(),
                           'output_size': list(self.output_size), 'stereo_calibration_sha256': self.calibration_hash,
                           'quality': self.quality, 'saved_unix_time': time.time()})

    @classmethod
    def load(cls, path, output_size, calibration_hash):
        config = json.loads(Path(path).read_text(encoding='utf-8'))
        if (config.get('version') != 1 or config.get('coordinate_space') != 'rectified_left'
                or config.get('output_size') != list(output_size)
                or config.get('stereo_calibration_sha256') != calibration_hash):
            raise ValueError('3D projector model does not match this camera/projector setup')
        return cls(config['matrix'], output_size, calibration_hash, config.get('quality'))

    def render(self, colors, depth, intrinsic, rectangle):
        # Twice the camera resolution, then a bilinear splat with a z-buffer.
        expanded = cv2.resize(depth, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        color = cv2.resize(colors, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        k = intrinsic.copy()
        k[0, 0] *= 2
        k[1, 1] *= 2
        k[0, 2], k[1, 2] = (k[0, 2] + .5) * 2 - .5, (k[1, 2] + .5) * 2 - .5
        valid = np.isfinite(expanded) & (expanded > 0)
        cloud, color = xyz_from_depth(expanded, k)[valid], color[valid]
        projected, z = project_points(self.matrix, cloud)
        left, top, right, bottom = rectangle
        good = np.isfinite(projected).all(axis=1) & (z > 0)
        good &= (projected[:, 0] >= left) & (projected[:, 0] < right - 1)
        good &= (projected[:, 1] >= top) & (projected[:, 1] < bottom - 1)
        projected, z, color = projected[good], z[good], color[good]
        w, h = self.output_size
        output = np.zeros((h, w, 3), np.uint8)
        if not len(projected):
            return output
        xy = np.floor(projected).astype(np.int32)
        fraction = projected - xy
        indices, distances, weights, samples = [], [], [], []
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            indices.append((xy[:, 1] + dy) * w + xy[:, 0] + dx)
            distances.append(z)
            weights.append((fraction[:, 0] if dx else 1 - fraction[:, 0])
                           * (fraction[:, 1] if dy else 1 - fraction[:, 1]))
            samples.append(color)
        index, distance, weight, sample = map(np.concatenate, (indices, distances, weights, samples))
        buffer = np.full(h * w, np.inf, np.float32)
        np.minimum.at(buffer, index, distance)
        visible = distance <= buffer[index] + .001
        index, weight, sample = index[visible], weight[visible], sample[visible]
        total = np.bincount(index, weights=weight, minlength=h * w)
        present = total > 1e-8
        flat = output.reshape(-1, 3)
        for channel in range(3):
            value = np.bincount(index, weights=weight * sample[:, channel], minlength=h * w)
            flat[present, channel] = np.rint(value[present] / total[present]).clip(0, 255).astype(np.uint8)
        coverage = present.reshape(h, w)
        if coverage.any() and not coverage.all():
            distance, labels = cv2.distanceTransformWithLabels((~coverage).astype(np.uint8), cv2.DIST_L2, 5,
                                                              labelType=cv2.DIST_LABEL_PIXEL)
            lookup = np.zeros((int(labels.max()) + 1, 3), np.uint8)
            lookup[labels[coverage]] = output[coverage]
            holes = ~coverage & (distance <= 2.5)
            output[holes] = lookup[labels[holes]]
        return clip_projector(output, rectangle)
