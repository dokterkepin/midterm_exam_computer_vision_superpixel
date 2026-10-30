"""Everything for synthetic_wall1: its settings live here, the logic lives in src/pipelines/.

usage: python3 run.py [--marker M] [--stereo-height H] [--stereo-distance D] [--thresh T] [--min-sat S] [--sweep]
Marker size (m) and stereo camera height / distance (m) default to scene.json / camera.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import baseline
from pipelines import demo_superpixels, experiments, run_final, run_stereo, run_synthetic

WALL = "synthetic_wall1"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", type=float, help="red marker side in metres (default: scene.json, else 1)")
    ap.add_argument("--stereo-height", type=float, help="stereo camera height in metres (default: camera.json)")
    ap.add_argument("--stereo-distance", type=float, help="stereo camera distance to the wall in metres (default: camera.json)")
    ap.add_argument("--thresh", type=float, default=baseline.P["candidates"]["contrast_threshold"], help="local-contrast threshold")
    ap.add_argument("--min-sat", type=float, default=baseline.P["candidates"]["min_saturation"], help="saturation floor")
    ap.add_argument("--sweep", action="store_true", help="also run the slow algorithm x size x threshold sweep")
    a = ap.parse_args()

    demo_superpixels.main([WALL])
    dets = run_synthetic.main(WALL, thresh=a.thresh, min_sat=a.min_sat, marker=a.marker)    # main view; scale from the red marker
    st = run_stereo.main(WALL, thresh=a.thresh, min_sat=a.min_sat,
                    height=a.stereo_height, distance=a.stereo_distance)              # stereo: x, y, z
    run_final.main(WALL, dets, st)                                                   # merge -> output.json
    if a.sweep:
        experiments.main(WALL)
