"""The synthetic walls' baseline.

baseline/synthetic_wall_parameters.json   fixed design values, set by hand (resolution, size limits, yaw, stereo)
baseline/synthetic_wall_algorithm.json    per superpixel algorithm: size, contrast threshold, saturation floor, picked by
                                          the sweep (pipelines/synthetic_wall/sweep.py --save-best) and frozen until the
                                          sweep is asked again
"""
import json
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "baseline"
P = json.load(open(DIR / "synthetic_wall_parameters.json"))
ALGO_FILE = DIR / "synthetic_wall_algorithm.json"


def algorithms():
    """Contents of synthetic_wall_algorithm.json: {"best_algorithm": algo, "algorithms": {algo: settings}, ...}."""
    if not ALGO_FILE.exists():
        raise FileNotFoundError(f"{ALGO_FILE} missing: run python3 src/pipelines/synthetic_wall/run_all.py (sweep + best settings) first")
    return json.load(open(ALGO_FILE))


def best_algorithm():
    """The algorithm the final system uses (best of the three by the rule in synthetic_wall_algorithm.json)."""
    return algorithms()["best_algorithm"]


def settings(algo=None):
    """{"region_px", "contrast_threshold", "min_saturation", ...} of one algorithm (default: the best algorithm)."""
    a = algorithms()
    return a["algorithms"][algo or a["best_algorithm"]]
