"""Run the hold detector (SLIC) in pixel space on images without ground truth -> outputs/<name>/slic/detections.png.
Preliminary: the real walls have no calibration or tuned baseline yet.

usage: python3 run_images.py name scale thresh [max_aspect] [min_solidity]

scale = size of a hold in this image relative to synthetic_wall3 (where holds are ~22 px);
all pixel limits (superpixel size, window, areas) are multiplied by it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2

import candidates as cand
import evaluate as ev
import paths
import superpixels as sp


def main(name, scale, thresh, max_aspect=2.5, min_solidity=0.85):
    img = cv2.imread(paths.image_path(name))
    labels, n = sp.slic(img, region_size=max(3, round(8 * scale)))
    _, blobs = cand.hold_candidates(img, labels, thresh=thresh, window=max(21, int(61 * scale)),
                                    min_area=80 * scale ** 2, max_area=1500 * scale ** 2,
                                    max_aspect=max_aspect, min_solidity=min_solidity)
    print(f"{name}: {img.shape[1]}x{img.shape[0]}, {n} superpixels, {len(blobs)} holds detected")
    vis = img.copy()
    colours = {"#F2554E": (60, 60, 255), "#59C77B": (60, 220, 60), "#6389ED": (255, 120, 60)}
    for b in blobs:
        x, y, w, h = b["bbox"]
        col = colours[ev.classify_colour(b["hue"])]
        cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), col, 2)
    cv2.imwrite(paths.out_path(name, "slic", "detections.png"), vis)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), float(a[2]), *(float(v) for v in a[3:]))
