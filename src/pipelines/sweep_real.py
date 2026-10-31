"""Sweep on the real photos: algorithm x superpixel size x contrast threshold x saturation floor x shape limits,
scored against the hand-labelled holds (tuned on the labels wall, tested on the overlapping photo).

Writes outputs/real_walls/sweep_real.csv and roc_real.png; with --save-best each algorithm's best setting on the
labels wall goes to baseline/algorithms_real.json (the real-wall baseline, frozen until the next --save-best).
usage: python3 sweep_real.py [--save-best]
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baseline
import candidates as cand
import paths
import real_wall as rw
import superpixels as sp
from pipelines.experiments import SIZES, THRESH, f1

WALLS = ["wall2", "wall1"]                  # tuning photo (labelled) first, then the test photo
SATS = (0.0, 60.0, 118.0)
SHAPES = tuple(rw.SHAPES)                   # "synthetic" limits (the frozen baseline's) and "real" limits
REAL_FILE = baseline.DIR / "algorithms_real.json"
RULE = ("per algorithm, the (size, threshold, saturation floor, shape limits) with the highest F1 on the labels wall; "
        "ties go to fewer false positives, then to the higher mean F1 of the two neighbouring thresholds.")


def run_one(task):
    wall, algo, size = task
    cv2.setNumThreads(1)
    _, img, _, to_world = rw.prepare(wall)
    gt, region = rw.labels(wall)
    labels = sp.segment(algo, img, size)
    contrast = cand.local_contrast(img, labels, window=rw.WINDOW)
    rows = []
    for shape in SHAPES:
        for sat in SATS:
            for T in THRESH:
                _, blobs = rw.candidates(img, labels, T, sat, shape, contrast=contrast)
                r = rw.score(rw.on_wall(wall, blobs, to_world), gt, region, to_world)
                rows.append(dict(wall=wall, algo=algo, size=size, thresh=T, min_sat=sat, shape=shape,
                                 tp=r["tp"], fp=r["fp"], fn=r["fn"], n_gt=len(gt)))
    return rows


def acc_of(rows, wall):
    return {(r["algo"], r["size"], r["thresh"], r["min_sat"], r["shape"]): r for r in rows if r["wall"] == wall}


def rank(acc, key):
    a = acc[key]
    algo, size, T, sat, shape = key
    nb = [f1(acc[k]["tp"], acc[k]["fp"], acc[k]["n_gt"]) for k in
          ((algo, size, T - 5, sat, shape), (algo, size, T + 5, sat, shape)) if k in acc]
    return f1(a["tp"], a["fp"], a["n_gt"]), -a["fp"], float(np.mean(nb)) if nb else 0.0


def result(r):
    return {"tp": r["tp"], "fp": r["fp"], "n_gt": r["n_gt"], "f1": round(f1(r["tp"], r["fp"], r["n_gt"]), 3)}


def choose(rows):
    tune, test = acc_of(rows, WALLS[0]), acc_of(rows, WALLS[1])
    best = {a: max((k for k in tune if k[0] == a), key=lambda k: rank(tune, k)) for a in sp.ALGOS}
    out = {"best_algorithm": max(sp.ALGOS, key=lambda a: rank(tune, best[a])), "rule": RULE,
           "chosen_on": WALLS[0], "tested_on": WALLS[1], "algorithms": {}}
    for a, k in best.items():
        out["algorithms"][a] = {"region_px": k[1], "contrast_threshold": k[2], "min_saturation": k[3], "shape": k[4],
                                "tuning": result(tune[k]), "test": result(test[k])}
    return out


def synthetic_setting(algo):
    """The frozen synthetic baseline of `algo` as a sweep key (it is on the grid: same sizes, thresholds, floors)."""
    s = baseline.settings(algo)
    return (algo, s["region_px"], s["contrast_threshold"], s["min_saturation"], "synthetic")


def plot(rows, best):
    col = {"slic": "tab:blue", "lsc": "tab:orange", "seeds": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a, wall in zip(ax, WALLS):
        acc = acc_of(rows, wall)
        for algo in sp.ALGOS:
            s = best["algorithms"][algo]
            pts = sorted((acc[(algo, s["region_px"], T, s["min_saturation"], s["shape"])]["fp"],
                          acc[(algo, s["region_px"], T, s["min_saturation"], s["shape"])]["tp"] / acc[
                              (algo, s["region_px"], T, s["min_saturation"], s["shape"])]["n_gt"]) for T in THRESH)
            xs = range(0, max(p[0] for p in pts) + 1)
            a.step(list(xs), [max([y for x, y in pts if x <= k] or [0]) for k in xs], where="post", lw=2,
                   color=col[algo], label=f"{algo.upper()} (size {s['region_px']}, sat {s['min_saturation']:.0f}, {s['shape']})")
            r = acc[(algo, s["region_px"], s["contrast_threshold"], s["min_saturation"], s["shape"])]
            a.plot(r["fp"], r["tp"] / r["n_gt"], "*", ms=13, color=col[algo], mec="k")
            k = synthetic_setting(algo)
            a.plot(acc[k]["fp"], acc[k]["tp"] / acc[k]["n_gt"], "x", ms=9, mew=2, color=col[algo])
        a.set(xlabel=f"false positives ({wall})", title=f"{wall} ({'tuning' if wall == WALLS[0] else 'test'})",
              xlim=(0, 30), ylim=(0, 1))
        a.grid(alpha=.3)
        a.legend(fontsize=7, loc="lower right")
    ax[0].set_ylabel("recall (labelled holds found)")
    fig.suptitle("Real walls, contrast threshold 10-100;  * = real-wall baseline,  x = synthetic baseline", fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.out_path("real_walls", "roc_real.png"), dpi=150)


def main(save_best=False):
    tasks = [(w, a, s) for w in WALLS for a in sp.ALGOS for s in SIZES]
    for w in WALLS:
        rw.labels(w)                                          # computes and caches the frame offset once
    t0 = time.time()
    with Pool(min(len(tasks), max(1, (os.cpu_count() or 2) // 2))) as pool:
        rows = [r for rs in pool.imap_unordered(run_one, tasks) for r in rs]
    rows.sort(key=lambda r: (WALLS.index(r["wall"]), sp.ALGOS.index(r["algo"]), r["size"], r["shape"], r["min_sat"], r["thresh"]))
    with open(paths.out_path("real_walls", "sweep_real.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"real sweep: {len(rows)} evaluations in {time.time() - t0:.0f}s")
    best = choose(rows)
    if save_best:
        with open(REAL_FILE, "w") as f:
            json.dump(best, f, indent=2)
        print(f"wrote {REAL_FILE}")
    frozen = json.load(open(REAL_FILE)) if REAL_FILE.exists() else best
    tune, test = acc_of(rows, WALLS[0]), acc_of(rows, WALLS[1])
    print(f"\n{'algo':6s} {'setting':38s} {'tuning ' + WALLS[0]:>22s}   {'test ' + WALLS[1]:>22s}")
    fmt = lambda r: f"{r['tp']:3d}/{r['n_gt']} FP {r['fp']:3d} F1 {f1(r['tp'], r['fp'], r['n_gt']):.2f}"
    for algo in sp.ALGOS:
        k = synthetic_setting(algo)
        print(f"{algo:6s} {'synthetic baseline ' + str(k[1:4]):38s} {fmt(tune[k]):>22s}   {fmt(test[k]):>22s}")
        s = frozen["algorithms"][algo]
        k = (algo, s["region_px"], s["contrast_threshold"], s["min_saturation"], s["shape"])
        star = "*" if algo == frozen["best_algorithm"] else ""
        print(f"{algo + star:6s} {'real baseline ' + str(k[1:]):38s} {fmt(tune[k]):>22s}   {fmt(test[k]):>22s}")
    plot(rows, frozen)
    return rows


if __name__ == "__main__":
    main(save_best="--save-best" in sys.argv)
