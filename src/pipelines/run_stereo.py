"""Detect holds in both eyes of a stereo PNG with one superpixel algorithm, triangulate them, and score
against scene.json. Writes outputs/<wall>/<algo>/stereo_output.json and stereo_detections.png.

usage: python3 run_stereo.py [wall] [algo]      (settings of that algorithm from baseline/algorithms.json)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np

import baseline
import evaluate as ev
import paths
import stereo
from pipelines import run_synthetic


def fundamental(P0, P1):
    """F such that x1^T F x0 = 0, from two projection matrices."""
    c0 = np.linalg.svd(P0)[2][-1]                      # camera-0 centre (null space of P0)
    e1 = P1 @ c0
    skew = np.array([[0, -e1[2], e1[1]], [e1[2], 0, -e1[0]], [-e1[1], e1[0], 0]])
    return skew @ P1 @ np.linalg.pinv(P0)


_T = baseline.P["stereo"]


def detect(eye_img, up, algo, size, thresh, min_sat):
    """Detect holds in one eye after resizing it by `up`; returned pixels are in the eye's own image."""
    big = cv2.resize(eye_img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA)
    _, _, blobs = run_synthetic.detect(big, algo, size, thresh, min_sat, max_aspect=_T["max_aspect"])
    for b in blobs:
        b["centroid"] = tuple(v / up for v in b["centroid"])
        b["bbox"] = tuple(round(v / up) for v in b["bbox"])
    return blobs


def main(name="synthetic_wall1", algo=None, size=None, thresh=None, min_sat=None, height=None, distance=None):
    """algo defaults to the best algorithm; size / thresh / min_sat default to that algorithm's baseline.
    height / distance: stereo camera position in metres; None = from camera.json."""
    s = baseline.settings(algo)
    algo = algo or baseline.best_algorithm()
    size = size if size is not None else s["region_px"]
    thresh = thresh if thresh is not None else s["contrast_threshold"]
    min_sat = min_sat if min_sat is not None else s["min_saturation"]
    img = cv2.imread(paths.stereo_path(name))
    if height is None or distance is None:                       # fill what is not given from camera.json
        cam = json.load(open(paths.camera_path(name)))["stereo"]
        height = cam["height_m"] if height is None else height
        distance = cam["distance_m"] if distance is None else distance
    cam = {"height_m": height, "distance_m": distance}
    gt = ev.load_gt(paths.gt_path(name))
    rig = stereo.StereoRig(img.shape[1], img.shape[0], cam["height_m"], cam["distance_m"])
    left, right = rig.split(img)
    up = run_synthetic.TARGET_PX_PER_M / (rig.f / cam["distance_m"])   # eye pixels per metre at the wall -> working resolution
    bl, br = (detect(eye, up, algo, size, thresh, min_sat) for eye in (left, right))
    print(f"{name}: eye {left.shape[1]}x{left.shape[0]}, camera height {cam['height_m']} m "
          f"distance {cam['distance_m']} m; blobs left={len(bl)} right={len(br)}")

    # match along epipolar lines, same colour class
    F = fundamental(rig.P[0], rig.P[1])
    pairs, used = [], set()
    for a in bl:
        x0 = np.array([*a["centroid"], 1.0])
        line = F @ x0
        best = None
        for j, b in enumerate(br):
            if j in used or ev.classify_colour(a["hue"]) != ev.classify_colour(b["hue"]):
                continue
            d = abs(line @ np.array([*b["centroid"], 1.0])) / np.hypot(line[0], line[1])
            if d < _T["epipolar_tolerance_px"] and (best is None or d < best[0]):
                best = (d, j)
        if best:
            used.add(best[1])
            pairs.append((a, br[best[1]], best[0]))
    print(f"matched {len(pairs)} left/right pairs")

    ids = list(gt)
    gxyz = np.array([[gt[k]["position"][c] for c in "xyz"] for k in ids])
    out, errs = [], []
    print("\nhold      est (x, y, z) m            true (x, y, z) m           error cm (x, y, z)")
    for a, b, d_epi in pairs:
        X = rig.triangulate([a["centroid"]], [b["centroid"]])[0]
        dist = np.linalg.norm(gxyz - X, axis=1)
        j = int(np.argmin(dist))
        gid = ids[j] if dist[j] < 0.15 else None
        out.append({"type": ev.classify_colour(a["hue"]),
                    "x": float(X[0]), "y": float(X[1]), "z": float(X[2]), "matches_gt": gid})
        if gid:
            e = (X - gxyz[j]) * 100
            errs.append(e)
            print(f"{gid:8s} ({X[0]:6.3f},{X[1]:6.3f},{X[2]:6.3f})   ({gxyz[j][0]:6.3f},{gxyz[j][1]:6.3f},{gxyz[j][2]:6.3f})   "
                  f"({e[0]:5.1f},{e[1]:5.1f},{e[2]:5.1f})")
        else:
            print(f"UNMATCHED ({X[0]:6.3f},{X[1]:6.3f},{X[2]:6.3f})")
    if errs:
        e = np.abs(np.array(errs))
        print(f"\nmean abs error cm: x={e[:, 0].mean():.1f} y={e[:, 1].mean():.1f} z={e[:, 2].mean():.1f}")

    json.dump(out, open(paths.out_path(name, algo, "stereo_output.json"), "w"), indent=1)
    vis = img.copy()
    off = img.shape[1] - left.shape[1]
    for a, b, _ in pairs:
        for blob, dx in ((a, 0), (b, off)):
            x, y, w, h = blob["bbox"]
            cv2.rectangle(vis, (x + dx - 4, y - 4), (x + dx + w + 4, y + h + 4), (0, 200, 0), 2)
    cv2.imwrite(paths.out_path(name, algo, "stereo_detections.png"), vis)
    return {"holds": out, "distance_m": distance, "focal_px": rig.f}


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:1] or ["synthetic_wall1"]), *a[1:2])
