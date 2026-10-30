"""Detect holds in a synthetic wall screenshot and score them against its JSON.

usage: python3 run_synthetic.py [name] [region_size] [thresh] [scale] [min_sat]
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

COLOUR_BGR = {"#F2554E": (78, 85, 242), "#59C77B": (123, 199, 89), "#6389ED": (237, 137, 99)}

# The detector was tuned at this resolution (~370 px per metre, hold scale 0.6). Every image is
# resized to it, using the red marker (size from scene.json) to measure the image's resolution.
_W, _S, _C = baseline.P["working_resolution"], baseline.P["superpixels"], baseline.P["candidates"]
TARGET_PX_PER_M = _W["px_per_m"]
SCALE_AT_TARGET = _W["hold_scale"]
TEMPLATE_HOLD = ("synthetic_wall3", "hold_8")   # the one hold the yaw template was cut from: not scored


def main(name="synthetic_wall1", region=_S["region_at_scale1"], thresh=_C["contrast_threshold"], scale=SCALE_AT_TARGET,
         min_sat=_C["min_saturation"], marker=None):
    """scale: hold size relative to a ~22 px hold, multiplies every pixel limit (after resizing).
    marker: red marker side in metres; None = from scene.json (1 m if it does not say)."""
    orig = cv2.imread(paths.image_path(name))
    gt = ev.load_gt(paths.gt_path(name))

    marker = marker if marker is not None else ev.marker_size(paths.gt_path(name))
    H = cand.calibrate_homography(orig, square_m=marker)
    px_per_m = 1.0 / np.linalg.norm(cand.apply_homography(H, [[1, 0]]) - cand.apply_homography(H, [[0, 0]]))
    up = TARGET_PX_PER_M / px_per_m
    img = cv2.resize(orig, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA)
    to_world = lambda p: cand.apply_homography(H, np.asarray(p, float) / up)[0]
    print(f"red marker {marker} m, about {px_per_m:.1f} px/m near the image origin; image resized x{up:.2f}")

    region = max(3, round(region * scale))
    labels, n = sp.slic(img, region_size=region)
    mask, blobs = cand.hold_candidates(img, labels, thresh=thresh, window=max(21, int(_C["window_at_scale1"] * scale)),
                                       open_k=_C["open_kernel"], min_area=_C["min_area_at_scale1"] * scale ** 2,
                                       max_area=_C["max_area_at_scale1"] * scale ** 2, max_aspect=_C["max_aspect"],
                                       min_solidity=_C["min_solidity"], min_sat=min_sat)
    r = ev.match(blobs, gt, to_world, gate=_C["match_gate_m"])
    template = ori.load_template()
    for b in blobs:                                    # yaw from silhouette matching, in the original image
        b["yaw_deg"], b["yaw_iou"] = ori.estimate_yaw(orig, b["centroid"][0] / up, b["centroid"][1] / up,
                                                      np.sqrt(b["area"]) / up, template)
    print(f"SLIC region={region} ({n} superpixels), contrast threshold {thresh}")
    print(f"TP={r['tp']} FP={r['fp']} FN={r['fn']}   missed: {r['missed']}")

    print("\nmatched holds   pos err   type(pred/true)      yaw est/true (deg, mod 180)")
    pos_err, type_ok, yaw_err = [], 0, []
    for b, gid, d in sorted(r["pairs"], key=lambda p: int(p[1].split("_")[1])):
        g = gt[gid]
        pred = ev.classify_colour(b["hue"])
        b["type"] = pred
        type_ok += pred == g["color"]
        est = (b["yaw_deg"] or 0.0) % 180.0
        true_deg = np.degrees(g["orientation"]["yaw"]) % 180.0
        diff = abs((est - true_deg + 90) % 180 - 90)
        skip = (name, gid) == TEMPLATE_HOLD or g["orientation"]["roll"] > 0.1
        pos_err.append(d * 100)
        if not skip:
            yaw_err.append(diff)
        note = "  (yaw template source)" if (name, gid) == TEMPLATE_HOLD else \
               "  (on a volume: yaw not scored)" if g["orientation"]["roll"] > 0.1 else ""
        print(f"  {gid:8s}  {d * 100:5.1f} cm   {pred}/{g['color']} {'ok' if pred == g['color'] else 'WRONG'}"
              f"   {est:6.1f} / {true_deg:6.1f}   (diff {diff:4.1f})  z_true={g['position']['z']:.2f}{note}")
    k = len(pos_err)
    if k:
        print(f"\nposition error: mean {np.mean(pos_err):.1f} cm, max {np.max(pos_err):.1f} cm")
        print(f"type accuracy:  {type_ok}/{k}")
        if yaw_err:
            print(f"yaw error:      mean {np.mean(yaw_err):.1f} deg, median {np.median(yaw_err):.1f} deg  "
                  f"({len(yaw_err)} holds flat on the wall; chance = 45)")

    # one record per detection in the simulator's units; main_view.json is the debug copy,
    # the final output.json is written by run_final.py after merging in the stereo z
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
    out_file = paths.out_path(name, "main_view.json")
    with open(out_file, "w") as f:
        json.dump(dets, f, indent=1)
    print(f"\nwrote {out_file} ({len(dets)} detections)")

    vis = orig.copy()
    for b in r["pairs"] and blobs:
        x, y, w, h = (round(v / up) for v in b["bbox"])
        ok = b["gt"] is not None
        cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 200, 0) if ok else (0, 0, 255), 2)
        tag = (b["gt"].replace("hold_", "#") if ok else "FP")
        cv2.putText(vis, tag, (x - 4, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(paths.out_path(name, "detections.png"), vis)
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(paths.out_path(name, "mask.png"), mask * 255)
    return dets


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:1] or ["synthetic_wall1"]), **({"region": int(a[1])} if len(a) > 1 else {}),
         **({"thresh": float(a[2])} if len(a) > 2 else {}),
         **({"scale": float(a[3])} if len(a) > 3 else {}),
         **({"min_sat": float(a[4])} if len(a) > 4 else {}),
         **({"marker": float(a[5])} if len(a) > 5 else {}))
