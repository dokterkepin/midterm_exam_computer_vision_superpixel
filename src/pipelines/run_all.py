"""The whole project in one go: the synthetic walls, then the real walls (each has its own baseline).

usage: python3 src/pipelines/run_all.py [--force] [--keep-baseline]
--force           recompute the synthetic sweep of every wall (otherwise only walls whose inputs changed)
--keep-baseline   do not re-pick the best settings of either kind of wall: run with the current baseline files
Each part also runs alone: pipelines/synthetic_wall/run_all.py, pipelines/real_wall/run_all.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipelines.real_wall import run_all as real_wall
from pipelines.synthetic_wall import run_all as synthetic_wall

if __name__ == "__main__":
    keep = "--keep-baseline" in sys.argv
    synthetic_wall.main(force="--force" in sys.argv, keep_baseline=keep)
    print("\n===== real walls " + "=" * 50)
    real_wall.main(keep_baseline=keep)
