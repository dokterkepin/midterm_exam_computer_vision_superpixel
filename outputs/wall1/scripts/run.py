"""Everything for wall1 (real photo): grid calibration, then SLIC, LSC and SEEDS at their frozen real-wall baselines
(baseline/algorithms_real.json) into outputs/wall1/<algo>/, scored against the hand-labelled holds where labelled.

usage: python3 run.py      (to re-tune the real-wall baselines: python3 src/pipelines/sweep_real.py --save-best)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines import calibrate_grid, run_real

WALL = "wall1"

if __name__ == "__main__":
    calibrate_grid.main(WALL)                  # 20 cm bolt-hole grid -> pixel-to-metre homography
    run_real.run(WALL)                         # type, position, orientation with each algorithm
