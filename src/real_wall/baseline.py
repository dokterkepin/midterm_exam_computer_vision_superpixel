"""The real walls' baseline.

baseline/real_wall_parameters.json   fixed design values, set by hand (grid calibration, scale, size limits)
baseline/real_wall_algorithm.json    per superpixel algorithm: size, contrast threshold, saturation floor, picked by
                                     the sweep (pipelines/real_wall/sweep.py --save-best) on the labelled photo and
                                     frozen until the sweep is asked again
"""
import json
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "baseline"
P = json.load(open(DIR / "real_wall_parameters.json"))
ALGO_FILE = DIR / "real_wall_algorithm.json"


def algorithms():
    """Contents of real_wall_algorithm.json: {"best_algorithm": algo, "algorithms": {algo: settings}, ...}."""
    if not ALGO_FILE.exists():
        raise FileNotFoundError(f"{ALGO_FILE} missing: run python3 src/pipelines/real_wall/run_all.py (sweep + best settings) first")
    return json.load(open(ALGO_FILE))


def best_algorithm():
    return algorithms()["best_algorithm"]


def settings(algo=None):
    """{"region_px", "contrast_threshold", "min_saturation", ...} of one algorithm (default: the best algorithm)."""
    a = algorithms()
    return a["algorithms"][algo or a["best_algorithm"]]
