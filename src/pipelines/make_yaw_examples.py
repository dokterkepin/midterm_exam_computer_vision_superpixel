"""Draw baseline/yaw_examples.png: for every detected hold on the synthetic walls, its silhouette (white), the
best-matching rotated template (red outline), and the true / estimated yaw. Needs main_view.json of each wall
(run the wall's scripts/run.py first). Also saves the training hold the template was cut from."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np

import baseline
import evaluate as ev
import orientation as ori
import paths

WALLS = paths.synthetic_walls()
TEMPLATE_HOLD = ("synthetic_wall3", "hold_8")


def tile(img, d, gt, template):
    sil = ori.silhouette(img, d["pixel"][0], d["pixel"][1], d["size_px"])
    if sil is None:
        return None
    m = ori.normalise(sil)
    deg, iou = ori.match_yaw(m, template)
    R = cv2.getRotationMatrix2D((ori.CANVAS / 2, ori.CANVAS / 2), deg, 1.0)
    t = cv2.warpAffine(template, R, (ori.CANVAS, ori.CANVAS), flags=cv2.INTER_NEAREST)
    out = np.full((ori.CANVAS, ori.CANVAS, 3), 30, np.uint8)
    out[m > 0] = (235, 235, 235)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (60, 60, 255), 1)
    true = np.degrees(gt["orientation"]["yaw"]) % 180
    err = abs((deg % 180 - true + 90) % 180 - 90)
    flat = gt["orientation"]["roll"] <= 0.1
    return out, f"true {true:3.0f} est {deg % 180:3.0f}", ("on volume" if not flat else f"err {err:.0f}")


def main():
    tpl = ori.load_template()
    tiles = []
    for w in WALLS:
        img = cv2.imread(paths.image_path(w))
        gt = ev.load_gt(paths.gt_path(w))
        for d in sorted(json.load(open(paths.out_path(w, "main_view.json"))), key=lambda d: d["id"]):
            if d["matches_gt"] is None or (w, d["matches_gt"]) == TEMPLATE_HOLD:
                continue
            r = tile(img, d, gt[d["matches_gt"]], tpl)
            if r is None:
                continue
            t, l1, l2 = r
            cv2.putText(t, f"{w[-1]}:{d['matches_gt'][5:]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            cv2.putText(t, l1, (3, 120), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
            cv2.putText(t, l2, (3, 108), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 220, 255), 1)
            tiles.append(t)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(baseline.DIR / "yaw_examples.png"), sheet)
    # the hold the template was cut from, with the template beside it
    d = [d for d in json.load(open(paths.out_path(TEMPLATE_HOLD[0], "main_view.json"))) if d["matches_gt"] == TEMPLATE_HOLD[1]][0]
    img = cv2.imread(paths.image_path(TEMPLATE_HOLD[0]))
    x, y, h = d["pixel"][0], d["pixel"][1], int(d["size_px"] * 1.2)
    crop = cv2.resize(img[y - h:y + h, x - h:x + h], (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(baseline.DIR / "yaw_template_source.png"), np.hstack([crop, cv2.cvtColor(
        cv2.resize(tpl * 255, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)]))
    print(f"{len(tiles)} tiles -> {baseline.DIR}")


if __name__ == "__main__":
    main()
