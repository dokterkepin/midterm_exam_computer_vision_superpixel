"""The baseline's tuning values, read from baseline/parameters.json (the single place they are set)."""
import json
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "baseline"
P = json.load(open(DIR / "parameters.json"))
