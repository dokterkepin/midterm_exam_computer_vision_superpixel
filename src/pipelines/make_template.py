"""Rebuild src/yaw_template.png: the silhouette of one training hold (synthetic_wall3, hold_8, yaw 0).

That hold's pixel position comes from the detector (main_view.json), its yaw (0) from scene.json, so run
run_synthetic on synthetic_wall3 first. The hold is excluded from the yaw scores.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2

import evaluate as ev
import orientation as ori
import paths

WALL, HOLD = "synthetic_wall3", "hold_8"

if __name__ == "__main__":
    det = [d for d in json.load(open(paths.out_path(WALL, "main_view.json"))) if d["matches_gt"] == HOLD][0]
    yaw = ev.load_gt(paths.gt_path(WALL))[HOLD]["orientation"]["yaw"]
    img = cv2.imread(paths.image_path(WALL))
    t = ori.build_template(img, det["pixel"][0], det["pixel"][1], 80.0, yaw_deg=yaw * 180 / 3.141592653589793)
    print("template area", int(t.sum()), "->", ori.TEMPLATE_FILE)
