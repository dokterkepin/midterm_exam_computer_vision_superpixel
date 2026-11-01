"""Match detections to ground-truth holds (the simulator's JSON, or hand labels on the real walls)."""
import numpy as np


def match(blobs, gt, to_world, gate=0.10):
    """Greedy nearest-neighbour matching in world coordinates (metres).

    to_world maps a pixel (u, v) to wall (x, y) metres, e.g. a calibration homography.

    Adds blob["world"] (x, y) and blob["gt"] (hold id or None) in place.
    Returns dict with tp, fp, fn and the list of matched (blob, gt_id, dist).
    """
    ids = list(gt)
    gxy = np.array([[gt[k]["position"]["x"], gt[k]["position"]["y"]] for k in ids])
    used, pairs = set(), []
    order = sorted(range(len(blobs)), key=lambda i: -blobs[i]["area"])
    for i in order:
        b = blobs[i]
        b["world"] = np.asarray(to_world(b["centroid"]), float).reshape(2)
        d = np.linalg.norm(gxy - b["world"], axis=1)
        j = int(np.argmin(d))
        b["gt"] = None
        if d[j] < gate and j not in used:
            used.add(j)
            b["gt"] = ids[j]
            pairs.append((b, ids[j], float(d[j])))
    tp = len(pairs)
    return {"tp": tp, "fp": len(blobs) - tp, "fn": len(ids) - tp, "pairs": pairs,
            "missed": [k for j, k in enumerate(ids) if j not in used]}


def f1(tp, fp, n):
    """2 TP / (2 TP + FP + FN), exact for equal fractions (so ties are real ties)."""
    return 2 * tp / (2 * tp + fp + (n - tp)) if tp else 0.0
