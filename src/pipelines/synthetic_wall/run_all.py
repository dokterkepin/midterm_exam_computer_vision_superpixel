"""The whole synthetic-wall pipeline in one go:

1. sweep     SLIC / LSC / SEEDS x size x contrast threshold x saturation floor on every synthetic wall
2. best      each algorithm's best setting -> baseline/synthetic_wall_algorithm.json (and the best algorithm)
3. run       every wall with every algorithm at its baseline -> outputs/<wall>/<algo>/
4. compare   a printed table (end-to-end: found, FP, position, type, yaw, z); details in outputs/<wall>/<algo>/score.json
5. yaw       baseline/synthetic_wall_yaw_examples.png, the template matched to every scored hold

usage: python3 src/pipelines/synthetic_wall/run_all.py [--force] [--keep-baseline]
--force           recompute the sweep of every wall (otherwise only walls whose inputs changed)
--keep-baseline   skip step 2: run with the current baseline/synthetic_wall_algorithm.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines.synthetic_wall import make_yaw_examples, run_wall, sweep
from shared import paths


def main(force=False, keep_baseline=False):
    sweep.main(force=force, save_best=not keep_baseline)
    run_wall.run(paths.synthetic_walls())                # every wall x algorithm, in parallel
    make_yaw_examples.main()


if __name__ == "__main__":
    main(force="--force" in sys.argv, keep_baseline="--keep-baseline" in sys.argv)
