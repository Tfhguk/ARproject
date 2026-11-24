"""Reproduce conservative/dense comparison on a recorded stereo sequence."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_appearance import DepthAppearance
from rendering import Renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=Path("data/recorded_wall.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/surface_check"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.capture, allow_pickle=False) as saved:
        depths, guides, stamps = (saved[key] for key in ("depth", "guide", "timestamps"))
    report = {"source": str(args.capture), "frames": len(depths), "results": {},
              "scope": "Same stereo depths, rectified images and receipt times. Filled pixels are "
                       "presentation estimates; coverage is not metric accuracy."}
    images = {}
    for mode in ("conservative", "surface"):
        appearance, renderer = DepthAppearance(fill=mode), Renderer(.45, 1.2, bright=True)
        sequence, stats, times, blacks = [], [], [], []
        for depth, guide, stamp in zip(depths, guides, stamps):
            original = depth.copy()
            started = time.perf_counter()
            result = appearance.process(depth, float(stamp), guide)
            colors = result.colorize(renderer)
            times.append((time.perf_counter() - started) * 1000)
            np.testing.assert_array_equal(depth, original)
            sequence.append(colors)
            stats.append(result.statistics())
            blacks.append(float(np.all(colors == 0, axis=2).mean()))
        images[mode] = sequence
        report["results"][mode] = {key: float(np.mean([row[key] for row in stats])) for key in stats[0]}
        report["results"][mode].update(mean_finish_ms=float(np.mean(times)), mean_black_fraction=float(np.mean(blacks)))
    writer = cv2.VideoWriter(str(args.output / "comparison.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                             1 / float(np.median(np.diff(stamps))), (1280, 360))
    if not writer.isOpened():
        raise OSError("Cannot write comparison video")
    try:
        for a, b in zip(images["conservative"], images["surface"]):
            comparison = np.hstack([a, b])
            for x, label in ((10, "CONSERVATIVE"), (650, "FULL SURFACE")):
                cv2.putText(comparison, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, .7,
                            (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(comparison)
        (args.output / "comparison.png").write_bytes(cv2.imencode(".png", comparison)[1].tobytes())
    finally:
        writer.release()
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
