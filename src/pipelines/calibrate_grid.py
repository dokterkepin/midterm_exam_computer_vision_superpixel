"""Calibrate a real wall photo from its 20 cm bolt-hole grid (see src/grid_calibration.py).

Writes outputs/<wall>/calibration.json (homography pixel -> wall metres, scale, residuals) and calibration.png
(fitted grid, holes used, origin and 1 m axes).
usage: python3 calibrate_grid.py [wall ...]      (default: wall1 wall2)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2

import grid_calibration as gc
import paths

WALLS = ["wall1", "wall2"]


def main(wall):
    img = cv2.imread(paths.image_path(wall))
    cal = gc.calibrate_grid(img)
    r = cal["residual_px"]
    print(f"{wall}: {cal['n_dots']} dark dots, {cal['n_grid']} on the grid; spacing {cal['spacing_px']:.1f} px "
          f"= {cal['px_per_m']:.0f} px/m, grid angle {cal['angle_deg']:.1f} deg; residual px median {r['median']:.1f}, "
          f"p95 {r['p95']:.1f}, max {r['max']:.1f} ({r['median'] / cal['px_per_m'] * 100:.1f} cm median)")
    out = {"H_pixel_to_m": cal["H"].tolist(), "px_per_m": cal["px_per_m"], "spacing_m": cal["spacing_m"],
           "spacing_px": cal["spacing_px"], "angle_deg": cal["angle_deg"], "origin_px": cal["origin_px"],
           "extent_m": cal["extent_m"],
           "holes_detected": cal["n_dots"], "holes_on_grid": cal["n_grid"], "residual_px": cal["residual_px"],
           "frame": "origin = bottom-left grid hole found; x right, y up, z out of the wall; metres"}
    with open(paths.out_path(wall, "calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
    cv2.imwrite(paths.out_path(wall, "calibration.png"), gc.draw(img, cal))
    return cal


if __name__ == "__main__":
    for w in sys.argv[1:] or WALLS:
        main(w)
