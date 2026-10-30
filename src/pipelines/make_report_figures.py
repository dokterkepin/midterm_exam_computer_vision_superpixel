"""Copy / compose the figures used in report/main.tex into report/figures/."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import paths

FIG = Path(__file__).resolve().parents[2] / "report" / "figures"


def shrink(img, width):
    return cv2.resize(img, None, fx=width / img.shape[1], fy=width / img.shape[1], interpolation=cv2.INTER_AREA)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in ("roc_and_sensitivity.png", "threshold_sensitivity.png"):
        shutil.copy(paths.out_path("all_walls", f), FIG / f)
    # detections on the three synthetic walls, side by side, same height
    tiles = []
    for w in ("synthetic_wall1", "synthetic_wall2", "synthetic_wall3"):
        im = cv2.imread(paths.out_path(w, "detections.png"))
        tiles.append(cv2.resize(im, (int(im.shape[1] * 700 / im.shape[0]), 700), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(FIG / "detections.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # the same patch of synthetic_wall1 (at the working resolution of the experiments) under the three
    # algorithms, size 5, centred on a detected hold
    import json
    import experiments as ex
    import superpixels as sp
    img, _, _ = ex.prepare("synthetic_wall1")
    det = json.load(open(paths.out_path("synthetic_wall1", "main_view.json")))
    cx, cy = [d for d in det if d["matches_gt"] == "hold_4"][0]["pixel"]
    crops = []
    for a in ex.ALGOS:
        labels = ex.make_labels(a, img, 5)
        over = sp.boundaries(img, labels)[cy - 36:cy + 36, cx - 48:cx + 48]
        crops.append(cv2.resize(over, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(FIG / "superpixels.jpg"), np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
