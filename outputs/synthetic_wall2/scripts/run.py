"""Everything for synthetic_wall2: its settings live here, the logic lives in src/pipelines/.

usage: python3 run.py [--marker M] [--stereo-height H] [--stereo-distance D] [--thresh T] [--min-sat S] [--no-pooled] [--force]
Marker size (m) and stereo camera height / distance (m) default to scene.json / camera.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import baseline
from pipelines import demo_superpixels, experiments, run_final, run_stereo, run_synthetic

WALL = "synthetic_wall2"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", type=float, help="red marker side in metres (default: scene.json, else 1)")
    ap.add_argument("--stereo-height", type=float, help="stereo camera height in metres (default: camera.json)")
    ap.add_argument("--stereo-distance", type=float, help="stereo camera distance to the wall in metres (default: camera.json)")
    ap.add_argument("--thresh", type=float, default=baseline.P["candidates"]["contrast_threshold"], help="local-contrast threshold")
    ap.add_argument("--min-sat", type=float, default=baseline.P["candidates"]["min_saturation"], help="saturation floor")
    ap.add_argument("--no-pooled", action="store_true", help="skip updating outputs/all_walls/ (the sweep over all walls)")
    ap.add_argument("--force", action="store_true", help="recompute the sweep of every wall, not only the changed ones")
    a = ap.parse_args()

    demo_superpixels.main([WALL])
    dets = run_synthetic.main(WALL, thresh=a.thresh, min_sat=a.min_sat, marker=a.marker)    # main view; scale from the red marker
    st = run_stereo.main(WALL, thresh=a.thresh, min_sat=a.min_sat,
                    height=a.stereo_height, distance=a.stereo_distance)              # stereo: x, y, z
    run_final.main(WALL, dets, st)                                                   # merge -> output.json
    if not a.no_pooled:                                  # refresh outputs/all_walls/; only changed walls are recomputed
        experiments.main(force=a.force)
