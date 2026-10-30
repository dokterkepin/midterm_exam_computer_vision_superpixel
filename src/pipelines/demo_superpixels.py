"""Run SLIC, LSC and SEEDS on each image and save border overlays to outputs/superpixels/."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2

import paths
import superpixels as sp

IMAGES = ["synthetic_wall1", "wall1", "wall2"]


def main(images):
    for name in images:
        img = cv2.imread(paths.image_path(name))
        # region size relative to image: synthetic holds are tiny, real ones larger
        region = 12 if name.startswith("synthetic") else 25
        n_seeds = (img.shape[0] * img.shape[1]) // (region * region)
        runs = {
            "slic": lambda: sp.slic(img, region_size=region),
            "lsc": lambda: sp.lsc(img, region_size=region),
            "seeds": lambda: sp.seeds(img, num_superpixels=n_seeds),
        }
        for algo, fn in runs.items():
            t = time.time()
            labels, n = fn()
            dt = time.time() - t
            cv2.imwrite(paths.out_path(name, f"superpixels_{algo}.png"), sp.boundaries(img, labels))
            print(f"{name:20s} {algo:6s} n={n:6d}  {dt:5.2f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or IMAGES)
