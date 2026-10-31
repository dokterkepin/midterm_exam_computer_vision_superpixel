"""Merge the main-view detections with the stereo depth into the final output.json, and score it.

Main view gives type, x, y, yaw. Stereo gives z, but depth noise grows with the square of the camera
distance, so a stereo z is used only where its expected error is small (see MAX_Z_SIGMA); otherwise
z = 0 (hold flat on the wall). Roll and pitch are not estimated and are written as 0.
Writes outputs/<wall>/<algo>/output.json (the system's answer) and score.json (its score against scene.json).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

import baseline
import evaluate as ev
import orientation as ori
import paths
import stereo

_T = baseline.P["stereo"]
MAX_Z_SIGMA = _T["max_z_sigma_m"]            # metres: use the stereo z only if its expected error is below this
GATE = _T["merge_gate_m"]                    # metres: main-view and stereo detections closer than this are the same hold
DISPARITY_SIGMA_PX = _T["disparity_sigma_px"]  # sub-pixel matching accuracy assumed for the depth-error estimate


def z_sigma(distance, focal_px):
    """Expected depth error (m) of a stereo pair at this distance: d^2 * sigma_px / (f * baseline)."""
    return distance ** 2 * DISPARITY_SIGMA_PX / (focal_px * stereo.BASELINE)


def main(name, algo, dets, st=None):
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
    with open(paths.out_path(name, algo, "output.json"), "w") as f:
        json.dump(final, f, indent=1)
    print(f"wrote outputs/{name}/{algo}/output.json: {len(final)} holds, z from stereo for {n_z}")

    gt = ev.load_gt(paths.gt_path(name))
    sc = ev.score(final, [d["matches_gt"] for d in dets], gt, name, ori.TEMPLATE_SOURCE)
    sc.update(algo=algo, settings=baseline.settings(algo), stereo_pairs=len(st["holds"]) if st else 0)
    with open(paths.out_path(name, algo, "score.json"), "w") as f:
        json.dump(sc, f, indent=1)

    print("\nhold      pos err   type   yaw err   z err   note")
    for h in sc["holds"]:
        yaw = f"{h['yaw_err_deg']:5.1f}" if h["yaw_err_deg"] is not None else "    -"
        note = "on a volume" if h["on_volume"] else "yaw template source" if h["yaw_err_deg"] is None else ""
        print(f"  {h['hold']:8s} {h['pos_err_cm']:4.1f} cm  {'ok' if h['type_ok'] else 'WRONG':5s}  {yaw} deg  "
              f"{h['z_err_cm']:4.1f} cm  {note}")
    return sc


def summary(sc):
    """Pooled numbers of one or more score.json dicts (one per wall)."""
    holds = [h for s in sc for h in s["holds"]]
    yaw = [h["yaw_err_deg"] for h in holds if h["yaw_err_deg"] is not None]
    mean = lambda k: round(float(np.mean([h[k] for h in holds])), 2) if holds else None
    return {"found": sum(s["found"] for s in sc), "n_gt": sum(s["n_gt"] for s in sc), "fp": sum(s["fp"] for s in sc),
            "pos_err_cm": mean("pos_err_cm"), "type_ok": sum(h["type_ok"] for h in holds),
            "yaw_mean_deg": round(float(np.mean(yaw)), 1) if yaw else None,
            "yaw_median_deg": round(float(np.median(yaw)), 1) if yaw else None, "n_yaw": len(yaw),
            "z_err_cm": mean("z_err_cm"), "roll_pitch_err_deg": mean("roll_pitch_err_deg")}
