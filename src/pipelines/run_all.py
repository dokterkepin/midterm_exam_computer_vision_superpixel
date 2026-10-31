"""The whole synthetic-wall pipeline in one go:

1. sweep     SLIC / LSC / SEEDS x size x contrast threshold x saturation floor on every synthetic wall
2. best      each algorithm's best setting -> baseline/algorithms.json (and the best algorithm)
3. run       every wall with every algorithm at its baseline -> outputs/<wall>/<algo>/
4. compare   a printed table (end-to-end: found, FP, position, type, yaw, z); details in outputs/<wall>/<algo>/score.json

usage: python3 src/pipelines/run_all.py [--force] [--keep-baseline]
--force           recompute the sweep of every wall (otherwise only walls whose inputs changed)
--keep-baseline   skip step 2: run with the current baseline/algorithms.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paths
from pipelines import experiments, make_yaw_examples, run_wall

if __name__ == "__main__":
    experiments.main(force="--force" in sys.argv, save_best="--keep-baseline" not in sys.argv)
    run_wall.run(paths.synthetic_walls())                # every wall x algorithm, in parallel
    make_yaw_examples.main()
