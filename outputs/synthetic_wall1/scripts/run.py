"""Everything for synthetic_wall1: runs SLIC, LSC and SEEDS at their frozen baselines into outputs/synthetic_wall1/<algo>/.

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
    run_wall.cli("synthetic_wall1", algo=ALGO, marker=MARKER, stereo_height=STEREO_HEIGHT, stereo_distance=STEREO_DISTANCE,
                 thresh=THRESH, min_sat=MIN_SAT, no_pooled=NO_POOLED, force=FORCE)
