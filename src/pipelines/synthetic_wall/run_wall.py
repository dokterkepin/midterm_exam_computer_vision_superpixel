"""Run the full pipeline on one synthetic wall for every superpixel algorithm, each at its own baseline.

For each algorithm: main view (detect) -> stereo (stereo_depth, if the wall has a stereo.png) ->
merge and score (final_output), all into outputs/<wall>/<algo>/. Several runs go in parallel processes, each
printing to its own outputs/<wall>/<algo>/run.log. At the end a table comparing the three algorithms
(from every wall's score.json) is printed.

The baseline (baseline/synthetic_wall_algorithm.json) is read, never changed, so a new wall run this way is a held-out score.
Each wall's outputs/<wall>/scripts/run.py calls cli() below.
"""
import argparse
import contextlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipelines.synthetic_wall import detect, final_output, stereo_depth, sweep
from shared import paths, superpixels as sp
from synthetic_wall import baseline

def run_one(wall, algo, marker=None, stereo_height=None, stereo_distance=None, thresh=None, min_sat=None):
    """Main view -> stereo -> final output and score of one wall with one algorithm."""
    dets = detect.main(wall, algo, thresh=thresh, min_sat=min_sat, marker=marker)
    st = None
    if (paths.DATASET / wall / "stereo.png").exists():
        st = stereo_depth.main(wall, algo, thresh=thresh, min_sat=min_sat, height=stereo_height, distance=stereo_distance)
    return final_output.main(wall, algo, dets, st)


def _logged(job):
    wall, algo, kw = job
    t = time.time()
    with open(paths.out_path(wall, algo, "run.log"), "w") as f, contextlib.redirect_stdout(f):
        sc = run_one(wall, algo, **kw)
    return wall, algo, sc, time.time() - t


def run(walls, algos=sp.ALGOS, **kw):
    """Run every (wall, algorithm) in parallel; kw: marker, stereo_height, stereo_distance, and thresh / min_sat
    to override the baseline (for trying values by hand)."""
    jobs = [(w, a, kw) for w in ([walls] if isinstance(walls, str) else walls) for a in algos]
    with Pool(len(jobs)) as pool:
        for wall, algo, sc, dt in pool.imap_unordered(_logged, jobs):
            print(f"  {wall} / {algo}: {sc['found']}/{sc['n_gt']} found, {sc['fp']} FP  ({dt:.0f}s, "
                  f"log in outputs/{wall}/{algo}/run.log)")
    print_results()


def print_results():
    """Print one line per (wall, algorithm) and a pooled line per algorithm, from every wall's score.json."""
    best = baseline.best_algorithm()
    rows = []
    for algo in sp.ALGOS:
        scores = []
        for wall in paths.synthetic_walls():
            f = paths.OUT / wall / algo / "score.json"
            if f.exists():
                scores.append(json.load(open(f)))
                rows.append(dict(wall=wall, algo=algo, **final_output.summary([scores[-1]])))
        if scores:
            rows.append(dict(wall="all", algo=algo, **final_output.summary(scores)))
    rows.sort(key=lambda r: (r["wall"] == "all", r["wall"], sp.ALGOS.index(r["algo"])))

    print(f"\n{'wall':16s} {'algo':6s} found   FP  pos cm  type   yaw mean/med (n)   z cm")
    for r in rows:
        yaw = f"{r['yaw_mean_deg']:5.1f} / {r['yaw_median_deg']:4.1f} ({r['n_yaw']:2d})" if r["n_yaw"] else " " * 18
        pos = f"{r['pos_err_cm']:5.1f}" if r["pos_err_cm"] is not None else "    -"
        z = f"{r['z_err_cm']:4.1f}" if r["z_err_cm"] is not None else "   -"
        star = "*" if r["wall"] == "all" and r["algo"] == best else ""
        print(f"{r['wall']:16s} {r['algo'] + star:6s} {r['found']:2d}/{r['n_gt']:2d}  {r['fp']:3d}  "
              f"{pos}  {r['type_ok']:2d}/{r['found']:2d}  {yaw}  {z}")
    print("(* = the winning algorithm; each wall's numbers are in outputs/<wall>/<algo>/score.json)")


def cli(wall):
    ap = argparse.ArgumentParser(description=f"Run {wall} for every superpixel algorithm at its frozen baseline.")
    ap.add_argument("--algo", choices=sp.ALGOS, action="append", help="only this algorithm (repeatable; default: all)")
    ap.add_argument("--marker", type=float, help="red marker side in metres (default: scene.json, else 1)")
    ap.add_argument("--stereo-height", type=float, help="stereo camera height in metres (default: camera.json)")
    ap.add_argument("--stereo-distance", type=float, help="stereo camera distance to the wall in metres (default: camera.json)")
    ap.add_argument("--thresh", type=float, help="contrast threshold (default: each algorithm's baseline)")
    ap.add_argument("--min-sat", type=float, help="saturation floor (default: each algorithm's baseline)")
    ap.add_argument("--no-pooled", action="store_true", help="skip updating the sweep over all walls (outputs/synthetic_wall_sweep/)")
    ap.add_argument("--force", action="store_true", help="recompute the sweep of every wall, not only the changed ones")
    a = ap.parse_args()
    run(wall, tuple(a.algo or sp.ALGOS), marker=a.marker, stereo_height=a.stereo_height,
        stereo_distance=a.stereo_distance, thresh=a.thresh, min_sat=a.min_sat)
    if not a.no_pooled:                                  # refresh outputs/synthetic_wall_sweep/; only changed walls are recomputed
        sweep.main(force=a.force)                  # never re-picks: the baseline stays frozen


if __name__ == "__main__":
    cli(sys.argv[1])
