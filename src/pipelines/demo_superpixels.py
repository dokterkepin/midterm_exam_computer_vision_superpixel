"""Run SLIC, LSC and SEEDS on an image as it is (no resizing) and save border overlays to
outputs/<wall>/<algo>/superpixels.png. Used for the real photographs; the synthetic walls get theirs from
run_synthetic.py, at the working resolution and baseline size."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2

import paths
import superpixels as sp

IMAGES = ["wall1", "wall2"]


def main(images, region=25):
    for name in images:
        img = cv2.imread(paths.image_path(name))
        for algo in sp.ALGOS:
            t = time.time()
            labels = sp.segment(algo, img, region)
            dt = time.time() - t
            cv2.imwrite(paths.out_path(name, algo, "superpixels.png"), sp.boundaries(img, labels))
            print(f"{name:20s} {algo:6s} n={labels.max() + 1:6d}  {dt:5.2f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or IMAGES)
