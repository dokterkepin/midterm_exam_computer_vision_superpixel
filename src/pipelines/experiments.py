"""Compare SLIC / LSC / SEEDS on the synthetic walls: superpixel size x contrast threshold x saturation floor.

Every wall is resized to the working resolution first (see run_synthetic.py), so a superpixel size in
pixels means the same thing on every wall. Per-wall results go to outputs/<wall>/sweep.csv; the pooled
table and the ROC / sensitivity plots go to outputs/all_walls/.

With --save-best, the best setting of every algorithm (pooled over the walls, see choose()) is written to
baseline/algorithms.json, which is the baseline every run uses. Without it the baseline stays as it is
(frozen), so a wall added later is scored held-out.

usage: python3 experiments.py [wall ...] [--force] [--replot] [--save-best]   (default: every synthetic wall in dataset/)
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
import evaluate as ev
import paths
import superpixels as sp
from pipelines.run_synthetic import SCALE, TARGET_PX_PER_M

ALGOS = sp.ALGOS
SIZES = [3, 4, 5, 6, 8, 10, 14]            # superpixel side in pixels at the working resolution
THRESH = list(range(10, 105, 5))           # contrast threshold: the ROC parameter
SATS = (0.0, 118.0)                        # saturation floor: off / on
_C = baseline.P["candidates"]
WINDOW = max(21, int(_C["window_at_scale1"] * SCALE))
WORKERS = max(1, (os.cpu_count() or 2) // 2)  # more gave no speed-up: the slowest task (LSC, size 3) sets the floor
BEST_RULE = ("per algorithm, the (size, threshold, saturation floor) with the highest pooled F1; ties go to "
                  "fewer false positives, then to the higher mean F1 of the two neighbouring thresholds (a stable "
                  "plateau). The best algorithm is the best of the three by the same rule.")


def prepare(wall):
    orig = cv2.imread(paths.image_path(wall))
    gt = ev.load_gt(paths.gt_path(wall))
    H = cand.calibrate_homography(orig, square_m=ev.marker_size(paths.gt_path(wall)))
    ppm = 1.0 / np.linalg.norm(cand.apply_homography(H, [[1, 0]]) - cand.apply_homography(H, [[0, 0]]))
    up = TARGET_PX_PER_M / ppm
    img = cv2.resize(orig, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA)
    to_world = lambda p: cand.apply_homography(H, np.asarray(p, float) / up)[0]
    return img, gt, to_world


def run_one(task):
    """All thresholds and saturation floors for one (wall, algorithm, size)."""
    wall, algo, size = task
    cv2.setNumThreads(1)
    img, gt, to_world = prepare(wall)
    t = time.time()
    labels = sp.segment(algo, img, size)
    t_sp = time.time() - t
    contrast = cand.local_contrast(img, labels, window=WINDOW)
    rows = []
    for sat in SATS:
        for T in THRESH:
            _, blobs = cand.hold_candidates(img, labels, thresh=T, window=WINDOW,
                                            open_k=_C["open_kernel"], min_area=_C["min_area_at_scale1"] * SCALE ** 2,
                                            max_area=_C["max_area_at_scale1"] * SCALE ** 2, max_aspect=_C["max_aspect"],
                                            min_solidity=_C["min_solidity"], min_sat=sat, contrast=contrast)
            r = ev.match(blobs, gt, to_world, gate=_C["match_gate_m"])
            err = np.mean([d for _, _, d in r["pairs"]]) * 100 if r["pairs"] else np.nan
            rows.append(dict(wall=wall, algo=algo, size=size, n_sp=int(labels.max() + 1), thresh=T,
                             min_sat=sat, tp=r["tp"], fp=r["fp"], fn=r["fn"], n_gt=len(gt),
                             err_cm=round(err, 3), t_sp=round(t_sp, 3)))
    return rows


def f1(tp, fp, n):
    """2 TP / (2 TP + FP + FN), exact for equal fractions (so ties are real ties)."""
    return 2 * tp / (2 * tp + fp + (n - tp)) if tp else 0.0


def pooled(rows):
    """Sum tp / fp over walls for each (algo, size, thresh, min_sat)."""
    acc = {}
    for r in rows:
        key = (r["algo"], r["size"], r["thresh"], r["min_sat"])
        a = acc.setdefault(key, dict(tp=0, fp=0, n=0, err=[], t_sp=0.0))
        a["tp"] += r["tp"]; a["fp"] += r["fp"]; a["n"] += r["n_gt"]
        a["t_sp"] += r["t_sp"]
        if not np.isnan(r["err_cm"]):
            a["err"].append(r["err_cm"])
    return acc


def rank(acc, key):
    """Sort key of a setting for BEST_RULE: (F1, fewer FP, F1 of the neighbouring thresholds)."""
    a = acc[key]
    algo, size, T, sat = key
    nb = [f1(acc[k]["tp"], acc[k]["fp"], acc[k]["n"]) for k in ((algo, size, T - 5, sat), (algo, size, T + 5, sat))
          if k in acc]
    return f1(a["tp"], a["fp"], a["n"]), -a["fp"], float(np.mean(nb)) if nb else 0.0


def choose(acc, walls):
    """Best setting per algorithm and the best algorithm, in the format of baseline/algorithms.json."""
    best = {algo: max((k for k in acc if k[0] == algo), key=lambda k: rank(acc, k)) for algo in ALGOS}
    out = {"best_algorithm": max(ALGOS, key=lambda a: rank(acc, best[a])), "rule": BEST_RULE,
           "chosen_on": list(walls), "algorithms": {}}
    for algo, (_, size, T, sat) in best.items():
        a = acc[best[algo]]
        out["algorithms"][algo] = {"region_px": size, "contrast_threshold": T, "min_saturation": sat,
                                   "pooled": {"tp": a["tp"], "fp": a["fp"], "n_gt": a["n"],
                                              "f1": round(f1(a["tp"], a["fp"], a["n"]), 3)}}
    return out


def read_csv(path):
    ints = ("size", "n_sp", "thresh", "tp", "fp", "fn", "n_gt")
    with open(path) as f:
        return [{k: (int(v) if k in ints else float(v) if k in ("min_sat", "err_cm", "t_sp") else v)
                 for k, v in r.items()} for r in csv.DictReader(f)]


def load_rows():
    return read_csv(paths.out_path("all_walls", "sweep_all.csv"))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def fingerprint():
    """Everything in parameters.json that changes the sweep's numbers. The per-algorithm baseline
    (algorithms.json) is left out: it is a point on the swept grid, so moving it does not change the sweep."""
    fixed = {k: v for k, v in _C.items() if k != "note"}
    return json.dumps(dict(working=[TARGET_PX_PER_M, SCALE], candidates=fixed,
                           grid=dict(algos=ALGOS, sizes=SIZES, thresh=THRESH, sats=SATS)), sort_keys=True)


def is_fresh(wall):
    """The wall's cached sweep (outputs/<wall>/sweep.csv) is newer than its inputs and the code, and was made
    with the same sweep settings."""
    cache = Path(paths.out_path(wall, "sweep.csv"))
    meta = Path(paths.out_path(wall, "sweep_inputs.json"))
    if not cache.exists() or not meta.exists() or meta.read_text() != fingerprint():
        return False
    here = Path(__file__).resolve().parents[1]
    deps = [paths.gt_path(wall), paths.image_path(wall),
            here / "candidates.py", here / "superpixels.py", here / "evaluate.py", Path(__file__)]
    return cache.stat().st_mtime > max(Path(d).stat().st_mtime for d in deps)


def sweep(walls, force=False):
    """Rows of the sweep over `walls`; a wall whose cached sweep is up to date is not recomputed."""
    stale = [w for w in walls if force or not is_fresh(w)]
    if stale:
        tasks = [(w, a, s) for w in stale for a in ALGOS for s in SIZES]
        t0 = time.time()
        with Pool(min(WORKERS, len(tasks))) as pool:
            new = [r for rs in pool.imap_unordered(run_one, tasks) for r in rs]
        print(f"computed {', '.join(stale)}: {len(tasks)} runs, {len(new)} evaluations, {time.time() - t0:.0f}s")
        for w in stale:
            write_csv(paths.out_path(w, "sweep.csv"), [r for r in new if r["wall"] == w])
            Path(paths.out_path(w, "sweep_inputs.json")).write_text(fingerprint())
    else:
        print("all walls up to date, nothing recomputed")
    rows = [r for w in walls for r in read_csv(paths.out_path(w, "sweep.csv"))]
    write_csv(paths.out_path("all_walls", "sweep_all.csv"), rows)
    return rows


def main(walls=None, replot=False, force=False, save_best=False):
    """Pool the sweep over every wall in dataset/, plot it, and (save_best=True) write the per-algorithm
    baseline to baseline/algorithms.json. force=True recomputes every wall."""
    if replot:
        rows = load_rows()
        walls = sorted({r["wall"] for r in rows})
    else:
        walls = walls or paths.synthetic_walls()
        rows = sweep(walls, force)
    acc = pooled(rows)
    n_gt = next(iter(acc.values()))["n"]

    best = choose(acc, walls)
    if save_best:
        with open(baseline.ALGO_FILE, "w") as f:
            json.dump(best, f, indent=2)
        print(f"\nwrote {baseline.ALGO_FILE}")
    frozen = baseline.algorithms() if baseline.ALGO_FILE.exists() else best

    print(f"\nbest pooled setting per algorithm ({n_gt} holds in {len(walls)} walls)")
    for algo in ALGOS:
        s, b = frozen["algorithms"][algo], best["algorithms"][algo]
        p = s["pooled"]
        print(f"  {algo:5s} size={s['region_px']:2d} T={s['contrast_threshold']:3d} sat={s['min_saturation']:3.0f}  "
              f"TP={p['tp']:2d}/{p['n_gt']} FP={p['fp']:2d} F1={p['f1']:.2f}"
              + ("" if b == s else f"   (sweep now prefers size={b['region_px']} T={b['contrast_threshold']} "
                                   f"sat={b['min_saturation']:.0f}, F1={b['pooled']['f1']:.2f}; rerun with --save-best)"))
    print(f"  best algorithm: {frozen['best_algorithm']}")

    # timings at the best algorithm's size, summed over walls
    size = frozen["algorithms"][frozen["best_algorithm"]]["region_px"]
    print("\nsuperpixel time (s, summed over the walls) and count at size", size)
    timing = []
    for algo in ALGOS:
        rs = [r for r in rows if r["algo"] == algo and r["size"] == size
              and r["thresh"] == THRESH[0] and r["min_sat"] == SATS[0]]
        timing.append(dict(algo=algo, seconds=round(sum(r["t_sp"] for r in rs), 2),
                           n_superpixels=sum(r["n_sp"] for r in rs)))
        print(f"  {algo:5s} {timing[-1]['seconds']:6.2f}s  {timing[-1]['n_superpixels']:8d} superpixels")
    write_csv(paths.out_path("all_walls", "timing.csv"), timing)

    plot(acc, frozen, n_gt)


def plot(acc, base, n_gt):
    """ROC curves (threshold sweep) and sensitivity to superpixel size and threshold, every algorithm drawn at
    its own baseline size; the baseline of each algorithm is a star."""
    col = {"slic": "tab:blue", "lsc": "tab:orange", "seeds": "tab:green"}
    algos = base["algorithms"]

    def envelope(algo, size, sat):
        pts = [(a["fp"], a["tp"] / a["n"]) for k, a in acc.items()
               if k[0] == algo and k[1] == size and k[3] == sat]
        xs = range(0, max(p[0] for p in pts) + 1)
        return list(xs), [max([y for x, y in pts if x <= k] or [0]) for k in xs], pts

    def best_f1(algo, size):
        return max(f1(a["tp"], a["fp"], a["n"]) for k, a in acc.items() if k[0] == algo and k[1] == size)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (sat, title, xmax) in enumerate(((0.0, "ROC, no saturation floor", 45),
                                            (118.0, "ROC, saturation floor 118", 8))):
        for algo in ALGOS:
            s = algos[algo]
            xs, ys, pts = envelope(algo, s["region_px"], sat)
            ax[k].scatter(*zip(*pts), s=9, alpha=.25, color=col[algo])
            ax[k].step(xs, ys, where="post", lw=2, color=col[algo], label=f"{algo.upper()} (size {s['region_px']})")
            if s["min_saturation"] == sat:
                p = s["pooled"]
                ax[k].plot(p["fp"], p["tp"] / p["n_gt"], "*", ms=14, color=col[algo], mec="k",
                           label=f"{algo.upper()} baseline (T={s['contrast_threshold']})")
        ax[k].set(xlabel="false positives (all walls)", ylabel="recall (true positive rate)",
                  title=title + " (threshold 10-100)", xlim=(0, xmax), ylim=(0, 0.9))
    for algo in ALGOS:
        ax[2].plot(SIZES, [best_f1(algo, s) for s in SIZES], "o-", color=col[algo], label=algo.upper())
    ax[2].set(xlabel=f"superpixel size (px at {TARGET_PX_PER_M} px/m)", ylabel="best F1 over thresholds and floors",
              title="Sensitivity to superpixel size", ylim=(0, 1))
    for a in ax:
        a.grid(alpha=.3)
        a.legend(loc="lower right" if a is not ax[2] else "lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.out_path("all_walls", "roc_and_sensitivity.png"), dpi=150)

    # sensitivity to the contrast threshold, every algorithm at its baseline size and saturation floor
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for a, algo in zip(ax, ALGOS):
        s = algos[algo]
        key = lambda T: (algo, s["region_px"], T, s["min_saturation"])
        a.plot(THRESH, [acc[key(T)]["tp"] for T in THRESH], "o-", label=f"true positives (of {n_gt})")
        a.plot(THRESH, [acc[key(T)]["fp"] for T in THRESH], "s-", label="false positives")
        a.axvline(s["contrast_threshold"], color="k", ls=":", label=f"baseline T={s['contrast_threshold']}")
        a.set(xlabel="contrast threshold T", title=f"{algo.upper()}, size {s['region_px']}, "
                                                   f"saturation floor {s['min_saturation']:.0f}")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("holds (all walls)")
    fig.tight_layout()
    fig.savefig(paths.out_path("all_walls", "threshold_sensitivity.png"), dpi=150)
    print("saved the plots in outputs/all_walls/")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or None, replot="--replot" in sys.argv, force="--force" in sys.argv, save_best="--save-best" in sys.argv)
