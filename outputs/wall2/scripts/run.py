"""Everything for wall2 (real photo, no ground truth): settings here, logic in src/pipelines/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines import demo_superpixels, run_images

WALL = "wall2"

if __name__ == "__main__":
    demo_superpixels.main([WALL])
    run_images.main(WALL, scale=1.0, thresh=45.0, max_aspect=3.5, min_solidity=0.75)
