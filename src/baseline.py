"""The baseline's values.

baseline/parameters.json   fixed design values, set by hand (resolution, size limits, yaw, stereo)
baseline/algorithms.json   per superpixel algorithm: size, contrast threshold, saturation floor, picked by
                           the sweep (experiments.py --save-best) and frozen until the sweep is asked again
"""
import json
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "baseline"
P = json.load(open(DIR / "parameters.json"))
ALGO_FILE = DIR / "algorithms.json"


def algorithms():
    """Contents of algorithms.json: {"best_algorithm": algo, "algorithms": {algo: settings}, ...}."""
    if not ALGO_FILE.exists():
        raise FileNotFoundError(f"{ALGO_FILE} missing: run python3 src/pipelines/run_all.py (sweep + best settings) first")
    return json.load(open(ALGO_FILE))


def best_algorithm():
    """The algorithm the final system uses (best of the three by the rule in algorithms.json)."""
    return algorithms()["best_algorithm"]


def settings(algo=None):
    """{"region_px", "contrast_threshold", "min_saturation", ...} of one algorithm (default: the best algorithm)."""
    a = algorithms()
    return a["algorithms"][algo or a["best_algorithm"]]
