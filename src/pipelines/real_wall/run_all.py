"""The whole real-wall pipeline in one go:

1. calibrate the 20 cm bolt-hole grid of each photo -> outputs/<wall>/calibration.npz
2. sweep     SLIC / LSC / SEEDS x size x contrast threshold x saturation floor against the hand-labelled holds
             (tuned on the labelled photo, tested on the other) -> baseline/real_wall_algorithm.json
3. run       every photo with every algorithm at its baseline -> outputs/<wall>/<algo>/

usage: python3 src/pipelines/real_wall/run_all.py [--keep-baseline]
--keep-baseline   skip the best-settings part of step 2: run with the current baseline/real_wall_algorithm.json
With no options the value at the bottom of this file is used, so the editor's Run button works too.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines.real_wall import calibrate_grid, run_wall, sweep
from shared import paths


def main(keep_baseline=False):
    walls = paths.real_walls()
    for w in walls:
        calibrate_grid.main(w)
    sweep.main(save_best=not keep_baseline)
    run_wall.run(walls)


if __name__ == "__main__":
    # ---- Run-button settings: used when the command line gives none ----
    KEEP_BASELINE = False   # True = do not re-pick the baseline; run with the current file

    ap = argparse.ArgumentParser(description="The whole real-wall pipeline: calibrate, sweep, run.")
    ap.add_argument("--keep-baseline", action="store_true", default=KEEP_BASELINE, help="do not re-pick the baseline")
    a = ap.parse_args()
    main(keep_baseline=a.keep_baseline)
