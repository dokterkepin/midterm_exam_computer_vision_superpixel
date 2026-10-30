"""Compare SLIC / LSC / SEEDS on the synthetic walls: superpixel size x contrast threshold x saturation floor.

Every wall is resized to the working resolution first (see run_synthetic.py), so a superpixel size in
pixels means the same thing on every wall. Per-wall results go to outputs/<wall>/sweep.csv; the pooled
table and the ROC / sensitivity plots go to outputs/all_walls/.

usage: python3 experiments.py [wall ...]      (default: all synthetic walls)
"""
import csv
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
import evaluate as ev
import paths
import superpixels as sp

WALLS = ["synthetic_wall1", "synthetic_wall2", "synthetic_wall3"]
ALGOS = ("slic", "lsc", "seeds")
SIZES = [3, 4, 5, 6, 8, 10, 14]            # superpixel side in pixels at the working resolution
THRESH = list(range(10, 105, 5))           # contrast threshold: the ROC parameter
_W, _C = baseline.P["working_resolution"], baseline.P["candidates"]
SATS = (0.0, float(_C["min_saturation"]))   # saturation floor: off / baseline
TARGET_PX_PER_M = _W["px_per_m"]
SCALE = _W["hold_scale"]                   # hold size relative to a ~22 px hold at the working resolution
WINDOW = max(21, int(_C["window_at_scale1"] * SCALE))
BASELINE = dict(algo=baseline.P["superpixels"]["algorithm"], size=round(baseline.P["superpixels"]["region_at_scale1"] * SCALE),
                thresh=_C["contrast_threshold"], min_sat=float(_C["min_saturation"]))


def prepare(wall):
    orig = cv2.imread(paths.image_path(wall))
    gt = ev.load_gt(paths.gt_path(wall))
    H = cand.calibrate_homography(orig, square_m=ev.marker_size(paths.gt_path(wall)))
    ppm = 1.0 / np.linalg.norm(cand.apply_homography(H, [[1, 0]]) - cand.apply_homography(H, [[0, 0]]))
    up = TARGET_PX_PER_M / ppm
    img = cv2.resize(orig, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA)
    to_world = lambda p: cand.apply_homography(H, np.asarray(p, float) / up)[0]
    return img, gt, to_world


def make_labels(algo, img, size):
    if algo == "slic":
        return sp.slic(img, region_size=size)[0]
    if algo == "lsc":
        return sp.lsc(img, region_size=size)[0]
    return sp.seeds(img, num_superpixels=img.shape[0] * img.shape[1] // (size * size))[0]


def run_one(task):
    """All thresholds and saturation floors for one (wall, algorithm, size)."""
    wall, algo, size = task
    cv2.setNumThreads(1)
    img, gt, to_world = prepare(wall)
    t = time.time()
    labels = make_labels(algo, img, size)
    t_sp = time.time() - t
    contrast = cand.local_contrast(img, labels, window=WINDOW)
    rows = []
    for sat in SATS:
        for T in THRESH:
            _, blobs = cand.hold_candidates(img, labels, thresh=T, window=WINDOW,
                                            open_k=_C["open_kernel"], min_area=_C["min_area_at_scale1"] * SCALE ** 2,
                                            max_area=_C["max_area_at_scale1"] * SCALE ** 2, max_aspect=_C["max_aspect"],
                                            min_solidity=_C["min_solidity"], min_sat=sat, contrast=contrast)
            r = ev.match(blobs, gt, to_world)
            err = np.mean([d for _, _, d in r["pairs"]]) * 100 if r["pairs"] else np.nan
            rows.append(dict(wall=wall, algo=algo, size=size, n_sp=int(labels.max() + 1), thresh=T,
                             min_sat=sat, tp=r["tp"], fp=r["fp"], fn=r["fn"], n_gt=len(gt),
                             err_cm=round(err, 3), t_sp=round(t_sp, 3)))
    return rows


def f1(tp, fp, n):
    p = tp / max(tp + fp, 1)
    rec = tp / n
    return 2 * p * rec / (p + rec) if p + rec else 0.0


def pooled(rows, **fixed):
    """Sum tp / fp over walls for each remaining free key (algo, size, thresh, min_sat) not in fixed."""
    acc = {}
    for r in rows:
        if all(r[k] == v for k, v in fixed.items()):
            key = (r["algo"], r["size"], r["thresh"], r["min_sat"])
            a = acc.setdefault(key, dict(tp=0, fp=0, n=0, err=[], t_sp=0.0))
            a["tp"] += r["tp"]; a["fp"] += r["fp"]; a["n"] += r["n_gt"]
            a["t_sp"] += r["t_sp"]
            if not np.isnan(r["err_cm"]):
                a["err"].append(r["err_cm"])
    return acc


def load_rows():
    ints = ("size", "n_sp", "thresh", "tp", "fp", "fn", "n_gt")
    rows = []
    with open(paths.out_path("all_walls", "sweep_all.csv")) as f:
        for r in csv.DictReader(f):
            r = {k: (int(v) if k in ints else float(v) if k in ("min_sat", "err_cm", "t_sp") else v)
                 for k, v in r.items()}
            rows.append(r)
    return rows


def main(walls=WALLS, replot=False):
    if replot:
        rows = load_rows()
        walls = sorted({r["wall"] for r in rows})
    else:
        tasks = [(w, a, s) for w in walls for a in ALGOS for s in SIZES]
        t0 = time.time()
        with Pool(8) as pool:
            rows = [r for rs in pool.imap_unordered(run_one, tasks) for r in rs]
        print(f"{len(tasks)} runs, {len(rows)} detector evaluations, {time.time() - t0:.0f}s")

    allw = "all_walls"
    if not replot:
        for w in walls:
            with open(paths.out_path(w, "sweep.csv"), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows([r for r in rows if r["wall"] == w])
        with open(paths.out_path(allw, "sweep_all.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    # ---- pooled over the walls -------------------------------------------------------------
    acc = pooled(rows)
    n_gt = next(iter(acc.values()))["n"]

    def best(algo, size, sat):
        cands = [(f1(a["tp"], a["fp"], a["n"]), k) for k, a in acc.items()
                 if k[0] == algo and k[1] == size and k[3] == sat]
        return max(cands, key=lambda c: (c[0], -c[1][2]))

    print(f"\nbest pooled F1 over thresholds ({n_gt} holds in {len(walls)} walls), saturation floor 118")
    best_size = {}
    for algo in ALGOS:
        for size in SIZES:
            sc, k = best(algo, size, 118.0)
            a = acc[k]
            print(f"  {algo:5s} size={size:2d}  F1={sc:.2f}  TP={a['tp']:2d}/{a['n']} FP={a['fp']:2d} "
                  f"T={k[2]:3d}  err={np.mean(a['err']):.2f} cm")
        best_size[algo] = max(SIZES, key=lambda s: (best(algo, s, 118.0)[0], -s))
    print("best size per algorithm:", best_size)

    # timings at the baseline size, summed over walls
    print("\nsuperpixel time (s, summed over the walls) and count at size", BASELINE["size"])
    timing = []
    for algo in ALGOS:
        rs = [r for r in rows if r["algo"] == algo and r["size"] == BASELINE["size"]
              and r["thresh"] == THRESH[0] and r["min_sat"] == SATS[0]]
        timing.append(dict(algo=algo, seconds=round(sum(r["t_sp"] for r in rs), 2),
                           n_superpixels=sum(r["n_sp"] for r in rs)))
        print(f"  {algo:5s} {timing[-1]['seconds']:6.2f}s  {timing[-1]['n_superpixels']:8d} superpixels")
    with open(paths.out_path(allw, "timing.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(timing[0]))
        wr.writeheader()
        wr.writerows(timing)

    # ---- plots -----------------------------------------------------------------------------
    col = {"slic": "tab:blue", "lsc": "tab:orange", "seeds": "tab:green"}

    def envelope(algo, size, sat):
        pts = [(a["fp"], a["tp"] / a["n"]) for k, a in acc.items()
               if k[0] == algo and k[1] == size and k[3] == sat]
        xs = range(0, max(p[0] for p in pts) + 1)
        return list(xs), [max([y for x, y in pts if x <= k] or [0]) for k in xs], pts

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (sat, title, xmax) in enumerate(((0.0, "ROC, no saturation floor", 45),
                                            (118.0, "ROC, saturation floor 118", 8))):
        for algo in ALGOS:
            xs, ys, pts = envelope(algo, best_size[algo], sat)
            ax[k].scatter(*zip(*pts), s=9, alpha=.25, color=col[algo])
            ax[k].step(xs, ys, where="post", lw=2, color=col[algo],
                       label=f"{algo.upper()} (size {best_size[algo]})")
        ax[k].set(xlabel="false positives (all 3 walls)", ylabel="recall (true positive rate)",
                  title=title + " (threshold 10-100)", xlim=(0, xmax), ylim=(0, 0.9))
    base = acc[(BASELINE["algo"], BASELINE["size"], BASELINE["thresh"], BASELINE["min_sat"])]
    ax[1].plot(base["fp"], base["tp"] / base["n"], "k*", ms=12, label="baseline (SLIC 5, T=40)")
    for algo in ALGOS:
        ax[2].plot(SIZES, [best(algo, s, 118.0)[0] for s in SIZES], "o-", color=col[algo], label=algo.upper())
    ax[2].set(xlabel="superpixel size (px at 370 px/m)", ylabel="best F1 over thresholds",
              title="Sensitivity to superpixel size", ylim=(0, 1))
    for a in ax:
        a.grid(alpha=.3)
        a.legend(loc="lower right" if a is not ax[2] else "lower left", fontsize=8)
    fig.tight_layout()
    fig_file = paths.out_path(allw, "roc_and_sensitivity.png")
    fig.savefig(fig_file, dpi=150)
    print("saved", fig_file)

    # sensitivity to the contrast threshold at the baseline size, SLIC
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ts = THRESH
    tp = [acc[("slic", BASELINE["size"], T, 118.0)]["tp"] for T in ts]
    fp = [acc[("slic", BASELINE["size"], T, 118.0)]["fp"] for T in ts]
    ax.plot(ts, tp, "o-", label=f"true positives (of {n_gt})")
    ax.plot(ts, fp, "s-", label="false positives")
    ax.axvline(BASELINE["thresh"], color="k", ls=":", label="baseline T=40")
    ax.set(xlabel="contrast threshold T", ylabel="holds (all walls)", title="SLIC, size 5, saturation floor 118")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.out_path(allw, "threshold_sensitivity.png"), dpi=150)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or WALLS, replot="--replot" in sys.argv)
