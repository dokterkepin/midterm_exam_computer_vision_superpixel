"""Merge the main-view detections with the stereo depth into the final output.json, and score it.

Main view gives type, x, y, yaw. Stereo gives z, but depth noise grows with the square of the camera
distance, so a stereo z is used only where its expected error is small (see MAX_Z_SIGMA); otherwise
z = 0 (hold flat on the wall). Roll and pitch are not estimated and are written as 0.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

import baseline
import evaluate as ev
import paths
import stereo

_T = baseline.P["stereo"]
MAX_Z_SIGMA = _T["max_z_sigma_m"]            # metres: use the stereo z only if its expected error is below this
GATE = _T["merge_gate_m"]                    # metres: main-view and stereo detections closer than this are the same hold
DISPARITY_SIGMA_PX = _T["disparity_sigma_px"]  # sub-pixel matching accuracy assumed for the depth-error estimate


def z_sigma(distance, focal_px):
    """Expected depth error (m) of a stereo pair at this distance: d^2 * sigma_px / (f * baseline)."""
    return distance ** 2 * DISPARITY_SIGMA_PX / (focal_px * stereo.BASELINE)


def main(name, dets, st=None):
    use_z = st is not None and z_sigma(st["distance_m"], st["focal_px"]) <= MAX_Z_SIGMA
    if st is not None:
        print(f"\nstereo depth error at {st['distance_m']} m is about {z_sigma(st['distance_m'], st['focal_px']) * 100:.1f} cm: "
              f"{'z taken from stereo' if use_z else 'too large, z = 0 (flat on the wall)'}")
    final, n_z = [], 0
    for d in dets:
        z = 0.0
        if use_z:
            near = [(np.hypot(h["x"] - d["x"], h["y"] - d["y"]), h) for h in st["holds"]]
            near = [(dist, h) for dist, h in near if dist < GATE]
            if near:
                z = max(0.0, min(near, key=lambda t: t[0])[1]["z"])
                n_z += 1
        final.append({"type": d["type"],
                      "position": {"x": round(d["x"], 3), "y": round(d["y"], 3), "z": round(z, 3)},
                      "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": round(d["yaw"], 3)}})
    out_file = paths.out_path(name, "output.json")
    with open(out_file, "w") as f:
        json.dump(final, f, indent=1)
    print(f"wrote {out_file}: {len(final)} holds, z from stereo for {n_z}")

    # score the final file against scene.json
    gt = ev.load_gt(paths.gt_path(name))
    ids = {d["matches_gt"]: (d, f) for d, f in zip(dets, final) if d["matches_gt"]}
    ez, ep, er = [], [], []
    for gid, (d, f) in ids.items():
        g = gt[gid]
        ez.append(abs(f["position"]["z"] - g["position"]["z"]) * 100)
        er.append(max(abs(f["orientation"]["roll"] - g["orientation"]["roll"]),
                      abs(f["orientation"]["pitch"] - g["orientation"]["pitch"])))
    if ids:
        print(f"final output vs scene.json over {len(ids)} matched holds: z error mean {np.mean(ez):.1f} cm; "
              f"roll/pitch error mean {np.degrees(np.mean(er)):.1f} deg "
              f"({sum(1 for g in ids if gt[g]['orientation']['roll'] > 0.1)} are on a volume)")
