"""The whole real-wall pipeline in one go:

1. calibrate the 20 cm bolt-hole grid of each photo -> outputs/<wall>/calibration.json
2. sweep     SLIC / LSC / SEEDS x size x contrast threshold x saturation floor against the hand-labelled holds
             (tuned on the labelled photo, tested on the other) -> baseline/real_wall_algorithm.json
3. run       every photo with every algorithm at its baseline -> outputs/<wall>/<algo>/

usage: python3 src/pipelines/real_wall/run_all.py [--keep-baseline]
--keep-baseline   skip the best-settings part of step 2: run with the current baseline/real_wall_algorithm.json
"""
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
    main(keep_baseline="--keep-baseline" in sys.argv)
