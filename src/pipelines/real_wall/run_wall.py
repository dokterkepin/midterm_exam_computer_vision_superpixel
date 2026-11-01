"""Run a real wall photo with every superpixel algorithm: type, position and orientation of every hold.

Each algorithm runs with its real-wall baseline (baseline/real_wall_algorithm.json, from sweep.py). Writes
outputs/<wall>/<algo>/: output.json (the answer: type, position, orientation), main_view.json (debug),
detections.png, superpixels.jpg and score.json (against the hand-labelled holds, inside the labelled region). Position: the grid calibration (z = 0: one photo has no depth). Orientation: yaw = direction of
the longest straight edge on the wall; roll = pitch = 0 (holds assumed flat on the wall).
usage: python3 run_wall.py wall [algo ...]
"""
import contextlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import cv2
import numpy as np

from pipelines.real_wall.sweep import REAL_FILE
from real_wall import baseline, detection as rw
from shared import paths, superpixels as sp
from shared.matching import f1

TYPE_BGR = {"red": (60, 60, 230), "orange": (40, 140, 255), "yellow": (0, 220, 255), "green": (60, 200, 60),
            "blue": (230, 120, 30), "teal": (180, 180, 60), "brown": (40, 70, 120), "tan": (120, 170, 210),
            "grey": (170, 170, 170)}


def real_settings(algo):
    with open(REAL_FILE) as f:
        return json.load(f)["algorithms"][algo]


def detect(wall, algo, s, prepared, protos):
    orig, img, up, to_world = prepared
    labels = sp.segment(algo, img, s["region_px"])
    mask, blobs = rw.candidates(img, labels, s["contrast_threshold"], s["min_saturation"])
    blobs = rw.describe(rw.on_wall(wall, blobs, to_world), mask, img, orig, up, to_world, protos)
    return labels, blobs


def evaluate(blobs, gt, region, to_world):
    r = rw.score(blobs, gt, region, to_world)
    pairs = [(b["type"] == gt[gid]["color"], d) for b, gid, d in r["pairs"]]
    return {"found": r["tp"], "fp": r["fp"], "n_gt": len(gt), "f1": round(f1(r["tp"], r["fp"], len(gt)), 3),
            "type_ok": int(sum(ok for ok, _ in pairs)),
            "pos_err_cm": round(float(np.mean([d for _, d in pairs]) * 100), 2) if pairs else None,
            "missed": r["missed"]}


def draw(orig, up, blobs, gt, region, H):
    """Boxes in the colour of the hold type; red cross = false positive, yellow circle = labelled hold missed,
    thin grey outline = the labelled region."""
    vis = orig.copy()
    inv = np.linalg.inv(H)
    to_px = lambda xy: cv2.perspectiveTransform(np.float32(xy).reshape(-1, 1, 2), inv).reshape(-1, 2)
    cv2.polylines(vis, [to_px(region).astype(np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)
    for b in blobs:
        x, y, w, h = (round(v / up) for v in b["bbox"])
        cv2.rectangle(vis, (x - 2, y - 2), (x + w + 2, y + h + 2), TYPE_BGR[b["type"]], 2)
        if b.get("gt", "") is None:                        # scored and unmatched
            cx, cy = round(b["centroid"][0] / up), round(b["centroid"][1] / up)
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    matched = {b.get("gt") for b in blobs}
    for gid, g in gt.items():
        if gid not in matched:
            (x, y), = to_px([[g["position"]["x"], g["position"]["y"]]])
            cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 255), 2)
    return vis


def main(wall, algo):
    t = time.time()
    prepared = rw.prepare(wall)
    orig, img, up, to_world = prepared
    H = np.array(rw.calibration(wall)["H_pixel_to_m"])
    protos = rw.colour_prototypes()
    gt, region = rw.labels(wall)
    s = real_settings(algo)

    labels, blobs = detect(wall, algo, s, prepared, protos)
    sc = evaluate(blobs, gt, region, to_world)                        # sets blob["gt"] for the scored blobs
    cv2.imwrite(paths.out_path(wall, algo, "detections.png"), draw(orig, up, blobs, gt, region, H))
    cv2.imwrite(paths.out_path(wall, algo, "superpixels.jpg"), sp.boundaries(img, labels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    final = [{"type": b["type"],
              "position": {"x": round(float(b["world"][0]), 3), "y": round(float(b["world"][1]), 3), "z": 0.0},
              "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": round(b["yaw"], 3)}}
             for b in sorted(blobs, key=lambda b: (b["world"][1], b["world"][0]))]
    with open(paths.out_path(wall, algo, "output.json"), "w") as f:
        json.dump(final, f, indent=1)
    debug = [{"type": b["type"], "x": float(b["world"][0]), "y": float(b["world"][1]), "yaw": b["yaw"],
              "pixel": [round(b["centroid"][0] / up), round(b["centroid"][1] / up)],
              "size_px": round(float(np.sqrt(b["area"]) / up), 1), "matches_label": b.get("gt")} for b in blobs]
    with open(paths.out_path(wall, algo, "main_view.json"), "w") as f:
        json.dump(debug, f, indent=1)

    out = {"wall": wall, "algo": algo, "holds_output": len(final), "settings": s, **sc,
           "labels": f"hand-labelled holds of {baseline.P['detection']['labels_wall']}, scored inside its photo's region"}
    with open(paths.out_path(wall, algo, "score.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{wall} / {algo}: {len(final)} holds output; labelled region: {sc['found']}/{sc['n_gt']} found, "
          f"{sc['fp']} FP, type {sc['type_ok']}/{sc['found']}  ({time.time() - t:.0f}s)")
    return out


def _quiet(job):
    with open(paths.out_path(job[0], job[1], "run.log"), "w") as f, contextlib.redirect_stdout(f):
        return main(*job)


def run(walls, algos=sp.ALGOS):
    """Every (wall, algorithm) in parallel, then the comparison table."""
    walls = [walls] if isinstance(walls, str) else list(walls)
    for w in walls:
        rw.labels(w)                                               # frame offset, computed once
    jobs = [(w, a) for w in walls for a in algos]
    with Pool(len(jobs)) as pool:
        results = list(pool.imap(_quiet, jobs))
    print_results(results)
    return results


def print_results(results):
    best = json.load(open(REAL_FILE))["best_algorithm"]
    print(f"\n{'wall':6s} {'algo':6s} {'output':>6s}   found    FP    F1   type  pos cm")
    for r in results:
        print(f"{r['wall']:6s} {r['algo'] + ('*' if r['algo'] == best else ''):6s} {r['holds_output']:6d}   "
              f"{r['found']:3d}/{r['n_gt']} {r['fp']:4d}  {r['f1']:.2f}  {r['type_ok']:3d}/{r['found']:<3d} {r['pos_err_cm'] or 0:5.1f}")
    print("(* = best algorithm on the tuning photo; scores only inside the labelled region; output = all holds on the wall)")


if __name__ == "__main__":
    run(sys.argv[1], tuple(sys.argv[2:]) or sp.ALGOS)
