"""The whole project in one go: the synthetic walls, then the real walls (each has its own baseline).

usage: python3 src/pipelines/run_all.py [--force] [--keep-baseline]
--force           recompute the synthetic sweep of every wall (otherwise only walls whose inputs changed)
--keep-baseline   do not re-pick the best settings of either kind of wall: run with the current baseline files
Each part also runs alone: pipelines/synthetic_wall/run_all.py, pipelines/real_wall/run_all.py.
With no options the values at the bottom of this file are used, so the editor's Run button works too.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipelines.real_wall import run_all as real_wall
from pipelines.synthetic_wall import run_all as synthetic_wall

if __name__ == "__main__":
    # ---- Run-button settings: used when the command line gives none ----
    FORCE = False           # True = recompute the synthetic sweep of every wall
    KEEP_BASELINE = False   # True = do not re-pick either baseline; run with the current files

    ap = argparse.ArgumentParser(description="The whole project: synthetic walls, then real walls.")
    ap.add_argument("--force", action="store_true", default=FORCE, help="recompute the synthetic sweep of every wall")
    ap.add_argument("--keep-baseline", action="store_true", default=KEEP_BASELINE, help="do not re-pick the baselines")
    a = ap.parse_args()
    synthetic_wall.main(force=a.force, keep_baseline=a.keep_baseline)
    print("\n===== real walls " + "=" * 50)
    real_wall.main(keep_baseline=a.keep_baseline)
