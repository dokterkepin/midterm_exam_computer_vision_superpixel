"""Create outputs/<wall>/scripts/run.py for a wall added under dataset/<wall>/.

usage: python3 src/pipelines/synthetic_wall/new_wall.py synthetic_wall4
Then: python3 outputs/synthetic_wall4/scripts/run.py   (uses the frozen baseline: a held-out score)
Only rerun pipelines/synthetic_wall/run_all.py (which re-picks the best settings) if the new wall should join the parameter choice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from shared import paths

TEMPLATE = '''"""Everything for {wall}: runs SLIC, LSC and SEEDS at their frozen baselines into outputs/{wall}/<algo>/.

usage: python3 run.py [--algo A] [--marker M] [--stereo-height H] [--stereo-distance D] [--thresh T] [--min-sat S]
                      [--no-pooled] [--force]          (python3 run.py -h for details)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines.synthetic_wall import run_wall

if __name__ == "__main__":
    run_wall.cli("{wall}")
'''


def write(wall, overwrite=False):
    dst = Path(paths.out_path(wall, "scripts", "run.py"))
    if dst.exists() and not overwrite:
        sys.exit(f"{dst} already exists")
    dst.write_text(TEMPLATE.format(wall=wall))
    print("created", dst)


if __name__ == "__main__":
    wall = sys.argv[1]
    if wall not in paths.synthetic_walls():
        sys.exit(f"dataset/{wall}/ needs scene.json and image.png (or wall.png)")
    write(wall)
