"""Copy / compose the figures used in report/main.tex into report/figures/."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np

from real_wall import baseline as real_baseline
from shared import paths, superpixels as sp
from synthetic_wall import baseline

FIG = Path(__file__).resolve().parents[2] / "report" / "figures"


def shrink(img, width):
    return cv2.resize(img, None, fx=width / img.shape[1], fy=width / img.shape[1], interpolation=cv2.INTER_AREA)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in ("roc_and_sensitivity.png", "threshold_sensitivity.png"):
        shutil.copy(paths.out_path("synthetic_wall_sweep", f), FIG / f"synthetic_wall_{f}")
    # detections of the best algorithm on the synthetic walls, side by side, same height
    algo = baseline.best_algorithm()
    tiles = []
    for w in paths.synthetic_walls():
        im = cv2.imread(paths.out_path(w, algo, "detections.png"))
        tiles.append(cv2.resize(im, (int(im.shape[1] * 700 / im.shape[0]), 700), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(FIG / "synthetic_wall_detections.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # the same patch of synthetic_wall1 (at the working resolution of the experiments) under the three
    # algorithms, size 5, centred on a detected hold
    import json
    from pipelines.synthetic_wall import sweep as ex
    img, _, _ = ex.prepare("synthetic_wall1")
    det = json.load(open(paths.out_path("synthetic_wall1", algo, "main_view.json")))
    cx, cy = [d for d in det if d["matches_gt"] == "hold_4"][0]["pixel"]
    crops = []
    for a in ex.ALGOS:
        labels = sp.segment(a, img, 5)
        over = sp.boundaries(img, labels)[cy - 36:cy + 36, cx - 48:cx + 48]
        crops.append(cv2.resize(over, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(FIG / "synthetic_wall_superpixels.jpg"), np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # real walls: grid calibration, ROC of the real sweep, and wall1 with the best real algorithm (the wall occupies
    # rows 70-640 of the photo)
    shutil.copy(paths.out_path("real_wall_sweep", "roc.png"), FIG / "real_wall_roc.png")
    cv2.imwrite(str(FIG / "real_wall_calibration.jpg"), cv2.imread(paths.out_path("real_wall1", "calibration.png"))[70:640],
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    real = real_baseline.best_algorithm()
    cv2.imwrite(str(FIG / "real_wall_detections.jpg"), cv2.imread(paths.out_path("real_wall1", real, "detections.png"))[70:640],
                [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
