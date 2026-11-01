"""Everything for real_wall2 (real photo): grid calibration, then SLIC, LSC and SEEDS at their frozen real-wall baselines
(baseline/real_wall_algorithm.json) into outputs/real_wall2/<algo>/, scored against the hand-labelled holds where labelled.

usage: python3 run.py      (to re-tune the real-wall baselines: python3 src/pipelines/real_wall/sweep.py --save-best)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines.real_wall import calibrate_grid, run_wall

WALL = "real_wall2"

if __name__ == "__main__":
    calibrate_grid.main(WALL)                  # 20 cm bolt-hole grid -> pixel-to-metre homography
    run_wall.run(WALL)                         # type, position, orientation with each algorithm
