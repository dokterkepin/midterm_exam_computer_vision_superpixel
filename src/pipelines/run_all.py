"""The whole project in one go.

Synthetic walls
1. sweep     SLIC / LSC / SEEDS x size x contrast threshold x saturation floor on every synthetic wall
2. best      each algorithm's best setting -> baseline/algorithms.json (and the best algorithm)
3. run       every wall with every algorithm at its baseline -> outputs/<wall>/<algo>/
4. compare   a printed table (end-to-end: found, FP, position, type, yaw, z); details in outputs/<wall>/<algo>/score.json
Real walls
5. calibrate the 20 cm bolt-hole grid of each photo -> outputs/<wall>/calibration.json
6. sweep     the same grid plus shape limits, against the hand-labelled holds (tuned on wall2, tested on wall1)
             -> each algorithm's best real-wall setting in baseline/algorithms_real.json
7. run       both photos with every algorithm (real baseline, and the synthetic baseline for comparison)

usage: python3 src/pipelines/run_all.py [--force] [--keep-baseline]
--force           recompute the synthetic sweep of every wall (otherwise only walls whose inputs changed)
--keep-baseline   skip steps 2 and 6: run with the current baseline files
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paths
from pipelines import calibrate_grid, experiments, make_yaw_examples, run_real, run_wall, sweep_real

REAL_WALLS = ["wall2", "wall1"]

if __name__ == "__main__":
    save_best = "--keep-baseline" not in sys.argv
    experiments.main(force="--force" in sys.argv, save_best=save_best)
    run_wall.run(paths.synthetic_walls())                # every wall x algorithm, in parallel
    make_yaw_examples.main()
    print("\n===== real walls " + "=" * 50)
    for w in REAL_WALLS:
        calibrate_grid.main(w)
    sweep_real.main(save_best=save_best)
    run_real.run(REAL_WALLS)
