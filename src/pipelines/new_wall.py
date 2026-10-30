"""Create outputs/<wall>/scripts/run.py for a wall added under dataset/<wall>/ (copy of an existing one).

usage: python3 src/pipelines/new_wall.py synthetic_wall4
Then: python3 outputs/synthetic_wall4/scripts/run.py   (uses the frozen baseline: a held-out score)
Only rerun experiments.py afterwards if the new wall should join the parameter sweep.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import paths

if __name__ == "__main__":
    wall = sys.argv[1]
    if wall not in paths.synthetic_walls():
        sys.exit(f"dataset/{wall}/ needs scene.json and image.png (or wall.png)")
    src = Path(paths.out_path("synthetic_wall1", "scripts/run.py"))
    dst = Path(paths.out_path(wall, "scripts")) / "run.py"
    if dst.exists():
        sys.exit(f"{dst} already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text().replace("synthetic_wall1", wall))
    print("created", dst)
