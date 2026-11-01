"""Everything for synthetic_wall1: runs SLIC, LSC and SEEDS at their frozen baselines into outputs/synthetic_wall1/<algo>/.

usage: python3 run.py [--algo A] [--marker M] [--stereo-height H] [--stereo-distance D] [--thresh T] [--min-sat S]
                      [--no-pooled] [--force]          (python3 run.py -h for details)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines.synthetic_wall import run_wall

if __name__ == "__main__":
    run_wall.cli("synthetic_wall1")
