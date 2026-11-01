"""Rebuild baseline/synthetic_wall_yaw_template.png: the silhouette of one training hold (yaw.template_source in
baseline/synthetic_wall_parameters.json, yaw 0).

That hold's pixel position comes from the best algorithm's detections (main_view.json), its yaw from
scene.json, so run that wall first. The hold is excluded from the yaw scores.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import cv2

from shared import paths
from synthetic_wall import baseline, evaluate as ev, orientation as ori

WALL, HOLD = ori.TEMPLATE_SOURCE

if __name__ == "__main__":
    det = [d for d in json.load(open(paths.out_path(WALL, baseline.best_algorithm(), "main_view.json"))) if d["matches_gt"] == HOLD][0]
    yaw = ev.load_gt(paths.gt_path(WALL))[HOLD]["orientation"]["yaw"]
    img = cv2.imread(paths.image_path(WALL))
    t = ori.build_template(img, det["pixel"][0], det["pixel"][1], 80.0, yaw_deg=yaw * 180 / 3.141592653589793)
    print("template area", int(t.sum()), "->", ori.TEMPLATE_FILE)
