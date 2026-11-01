"""Calibrate a real wall photo from its 20 cm bolt-hole grid (see src/real_wall/grid_calibration.py).

Writes outputs/<wall>/calibration.npz (homography pixel -> wall metres, scale, residuals) and calibration.png
(fitted grid, holes used, origin and 1 m axes).
usage: python3 calibrate_grid.py [wall ...]      (default: every real wall photo; with no arguments the value at
the bottom is used, so the editor's Run button works too)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import cv2
import numpy as np

from real_wall import grid_calibration as gc
from shared import paths


def main(wall):
    img = cv2.imread(paths.image_path(wall))
    cal = gc.calibrate_grid(img)
    r = cal["residual_px"]
    print(f"{wall}: {cal['n_dots']} dark dots, {cal['n_grid']} on the grid; spacing {cal['spacing_px']:.1f} px "
          f"= {cal['px_per_m']:.0f} px/m, grid angle {cal['angle_deg']:.1f} deg; residual px median {r['median']:.1f}, "
          f"p95 {r['p95']:.1f}, max {r['max']:.1f} ({r['median'] / cal['px_per_m'] * 100:.1f} cm median)")
    np.savez(paths.out_path(wall, "calibration.npz"),
             H_pixel_to_m=cal["H"], px_per_m=cal["px_per_m"], spacing_m=cal["spacing_m"], spacing_px=cal["spacing_px"],
             angle_deg=cal["angle_deg"], origin_px=np.asarray(cal["origin_px"], float),
             extent_m=np.asarray(cal["extent_m"], float), holes_detected=cal["n_dots"], holes_on_grid=cal["n_grid"],
             residual_median_px=r["median"], residual_p95_px=r["p95"], residual_max_px=r["max"],
             frame="origin = bottom-left grid hole found; x right, y up, z out of the wall; metres")
    cv2.imwrite(paths.out_path(wall, "calibration.png"), gc.draw(img, cal))
    return cal


if __name__ == "__main__":
    # ---- Run-button settings: used when the command line gives none ----
    WALLS = None          # e.g. ["real_wall3"]; None = every real wall photo

    ap = argparse.ArgumentParser(description="Calibrate real wall photos from their bolt-hole grid.")
    ap.add_argument("walls", nargs="*", help="wall folder names (default: every real wall photo)")
    a = ap.parse_args()
    for w in a.walls or WALLS or paths.real_walls():
        main(w)
