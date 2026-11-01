"""Create outputs/<wall>/scripts/run.py for a wall added under dataset/<wall>/.

usage: python3 src/pipelines/synthetic_wall/new_wall.py synthetic_wall4 [--overwrite]   (or set WALL at the bottom and press Run)
Then: python3 outputs/synthetic_wall4/scripts/run.py   (uses the frozen baseline: a held-out score)
Only rerun pipelines/synthetic_wall/run_all.py (which re-picks the best settings) if the new wall should join the parameter choice.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from shared import paths

TEMPLATE = '''"""Everything for {wall}: runs SLIC, LSC and SEEDS at their frozen baselines into outputs/{wall}/<algo>/.

Run it with the editor's Run button (set the values below) or from a terminal:
usage: python3 run.py [--algo A] [--marker M] [--stereo-height H] [--stereo-distance D] [--thresh T] [--min-sat S]
                      [--no-pooled] [--force]          (python3 run.py -h for details; an option overrides the value below)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines.synthetic_wall import run_wall

# ---- Run-button settings: used when you give no command-line options. None = the default ----
ALGO = None               # "slic", "lsc" or "seeds" (or a list of them); None = all three
MARKER = None             # red square side in metres; None = read from scene.json
STEREO_HEIGHT = None      # stereo camera height in metres; None = read from camera.json
STEREO_DISTANCE = None    # stereo camera distance to the wall in metres; None = read from camera.json
THRESH = None             # contrast threshold; None = each algorithm's baseline
MIN_SAT = None            # saturation floor; None = each algorithm's baseline
NO_POOLED = False         # True = skip refreshing the sweep over all walls (quicker)
FORCE = False             # True = recompute the whole sweep

if __name__ == "__main__":
    run_wall.cli("{wall}", algo=ALGO, marker=MARKER, stereo_height=STEREO_HEIGHT, stereo_distance=STEREO_DISTANCE,
                 thresh=THRESH, min_sat=MIN_SAT, no_pooled=NO_POOLED, force=FORCE)
'''


def write(wall, overwrite=False):
    dst = Path(paths.out_path(wall, "scripts", "run.py"))
    if dst.exists() and not overwrite:
        sys.exit(f"{dst} already exists")
    dst.write_text(TEMPLATE.format(wall=wall))
    print("created", dst)


if __name__ == "__main__":
    # ---- Run-button settings: used when no wall is given on the command line ----
    WALL = "synthetic_wall4"
    ap = argparse.ArgumentParser(description="Create outputs/<wall>/scripts/run.py for a wall added under dataset/<wall>/.")
    ap.add_argument("wall", nargs="?", default=WALL, help="wall folder name in dataset/")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run.py")
    a = ap.parse_args()
    if a.wall not in paths.synthetic_walls():
        sys.exit(f"dataset/{a.wall}/ needs scene.json and image.png (or wall.png)")
    write(a.wall, overwrite=a.overwrite)
