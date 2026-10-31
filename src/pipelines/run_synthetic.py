"""Detect holds in a synthetic wall screenshot with one superpixel algorithm (main view: type, x, y, yaw).

Writes outputs/<wall>/<algo>/: main_view.json (detections), detections.png, mask.png, superpixels.jpg.
usage: python3 run_synthetic.py [wall] [algo]      (settings of that algorithm from baseline/algorithms.json)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np

import candidates as cand
import baseline
import evaluate as ev
import orientation as ori
import paths
import superpixels as sp

# The detector was tuned at this resolution (~370 px per metre, hold scale 0.6). Every image is
# resized to it, using the red marker (size from scene.json) to measure the image's resolution.
_W, _C = baseline.P["working_resolution"], baseline.P["candidates"]
TARGET_PX_PER_M = _W["px_per_m"]
SCALE = _W["hold_scale"]


def to_working_resolution(orig, marker):
    """Resize to TARGET_PX_PER_M using the red marker; returns (img, up, to_world) where to_world maps a
    pixel of the resized image to wall metres."""
    H = cand.calibrate_homography(orig, square_m=marker)
    px_per_m = 1.0 / np.linalg.norm(cand.apply_homography(H, [[1, 0]]) - cand.apply_homography(H, [[0, 0]]))
    up = TARGET_PX_PER_M / px_per_m
    img = cv2.resize(orig, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA)
    print(f"red marker {marker} m, about {px_per_m:.1f} px/m near the image origin; image resized x{up:.2f}")
    return img, up, lambda p: cand.apply_homography(H, np.asarray(p, float) / up)[0]


def detect(img, algo, size, thresh, min_sat, max_aspect=_C["max_aspect"]):
    """Superpixels + hold candidates on an image at the working resolution; returns (labels, mask, blobs)."""
    labels = sp.segment(algo, img, size)
    mask, blobs = cand.hold_candidates(img, labels, thresh=thresh, window=max(21, int(_C["window_at_scale1"] * SCALE)),
                                       open_k=_C["open_kernel"], min_area=_C["min_area_at_scale1"] * SCALE ** 2,
                                       max_area=_C["max_area_at_scale1"] * SCALE ** 2, max_aspect=max_aspect,
                                       min_solidity=_C["min_solidity"], min_sat=min_sat)
    return labels, mask, blobs


def main(name="synthetic_wall1", algo=None, size=None, thresh=None, min_sat=None, marker=None):
    """algo defaults to the best algorithm; size / thresh / min_sat default to that algorithm's baseline.
    marker: red marker side in metres; None = from scene.json (1 m if it does not say)."""
    s = baseline.settings(algo)
    algo = algo or baseline.best_algorithm()
    size = size if size is not None else s["region_px"]
    thresh = thresh if thresh is not None else s["contrast_threshold"]
    min_sat = min_sat if min_sat is not None else s["min_saturation"]

    orig = cv2.imread(paths.image_path(name))
    gt = ev.load_gt(paths.gt_path(name))
    marker = marker if marker is not None else ev.marker_size(paths.gt_path(name))
    img, up, to_world = to_working_resolution(orig, marker)

    labels, mask, blobs = detect(img, algo, size, thresh, min_sat)
    r = ev.match(blobs, gt, to_world, gate=_C["match_gate_m"])
    print(f"{algo.upper()} size {size} ({labels.max() + 1} superpixels), contrast threshold {thresh}, "
          f"saturation floor {min_sat:.0f}")
    print(f"TP={r['tp']} FP={r['fp']} FN={r['fn']}   missed: {r['missed']}")

    template = ori.load_template()
    for b in blobs:                                    # yaw from silhouette matching, in the original image
        b["yaw_deg"], b["yaw_iou"] = ori.estimate_yaw(orig, b["centroid"][0] / up, b["centroid"][1] / up,
                                                      np.sqrt(b["area"]) / up, template)

    # one record per detection in the simulator's units; the final output.json is written by run_final.py
    # after merging in the stereo z
    dets = []
    for k, b in enumerate(sorted(blobs, key=lambda b: (b["world"][1], b["world"][0])), 1):
        yaw = np.radians(b["yaw_deg"]) if b["yaw_deg"] is not None else 0.0
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi                       # (-pi, pi], like scene.json
        dets.append({
            "id": k, "type": ev.classify_colour(b["hue"]),
            "x": float(b["world"][0]), "y": float(b["world"][1]), "yaw": float(yaw),
            "pixel": [round(b["centroid"][0] / up), round(b["centroid"][1] / up)],
            "size_px": round(float(np.sqrt(b["area"]) / up), 1),
            "yaw_iou": round(float(b["yaw_iou"]), 2), "matches_gt": b["gt"],
        })
    with open(paths.out_path(name, algo, "main_view.json"), "w") as f:
        json.dump(dets, f, indent=1)

    vis = orig.copy()
    for b in blobs:
        x, y, w, h = (round(v / up) for v in b["bbox"])
        ok = b["gt"] is not None
        cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 200, 0) if ok else (0, 0, 255), 2)
        tag = (b["gt"].replace("hold_", "#") if ok else "FP")
        cv2.putText(vis, tag, (x - 4, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(paths.out_path(name, algo, "detections.png"), vis)
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(paths.out_path(name, algo, "mask.png"), mask * 255)
    cv2.imwrite(paths.out_path(name, algo, "superpixels.jpg"), sp.boundaries(img, labels),   # at the working resolution
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote outputs/{name}/{algo}/ ({len(dets)} detections)")
    return dets


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:1] or ["synthetic_wall1"]), *a[1:2])
