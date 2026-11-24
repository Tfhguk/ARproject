"""Dense presentation estimates. Never convert these into stereo measurements."""

import cv2
import numpy as np


def nearest_values(values, valid, with_distance=False):
    """Propagate the nearest valid sample, including into image-border gaps."""
    distance, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros((int(labels.max()) + 1,) + values.shape[2:], np.float32)
    lookup[labels[valid]] = values[valid]
    result = lookup[labels]
    return (result, distance) if with_distance else result


def complete_surface(depth, guide=None):
    """Fill presentation gaps using spatial and image similarity of depth layers.

    Reliable samples remain anchors. Large gaps and one-sided border regions
    are estimates, not new measurements. An empty or almost empty map is never
    expanded into a purported full surface. No completed value enters history.
    """
    valid = np.isfinite(depth) & (depth > 0)
    missing = ~valid
    if not missing.any() or valid.sum() < max(32, depth.size * 0.05):
        return depth.copy(), np.zeros(depth.shape, bool)
    logs = np.zeros(depth.shape, np.float32)
    np.log(depth, out=logs, where=valid)
    image = np.zeros(depth.shape, np.float32) if guide is None else np.asarray(guide, np.float32)
    if image.shape != depth.shape:
        raise ValueError("Completion guide must match the depth image")
    values = np.stack((logs, image), axis=-1)
    closest, distance = nearest_values(values, valid, with_distance=True)

    def score(candidate, dist):
        difference = np.minimum(100, np.abs(candidate[:, :, 1] - image))
        return dist * (1 + difference / 60) + difference * 0.35

    best, cost = closest[:, :, 0].copy(), score(closest, distance)
    if guide is not None:
        # Fixed logarithmic bands avoid changing colour levels as objects enter.
        # Limit candidates by population to keep full-resolution costs bounded.
        bands = np.floor(logs / 0.14).astype(np.int32)
        levels, counts = np.unique(bands[valid], return_counts=True)
        for index in np.argsort(counts)[-8:]:
            if counts[index] < 32:
                continue
            seeds = valid & (bands == levels[index])
            candidate, dist = nearest_values(values, seeds, with_distance=True)
            candidate_cost = score(candidate, dist)
            use = missing & (candidate_cost < cost)
            best[use], cost[use] = candidate[:, :, 0][use], candidate_cost[use]
    # Smooth inferred contours without averaging widely separated surfaces.
    best[valid] = logs[valid]
    best = cv2.medianBlur(best, 5)
    best = cv2.bilateralFilter(best, 9, 0.035, 3)
    output = depth.copy()
    np.exp(best, out=output, where=missing)
    return output, missing
