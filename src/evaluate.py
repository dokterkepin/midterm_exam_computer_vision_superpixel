"""Match detections to the simulator's JSON ground truth and score them."""
import json

import numpy as np

# hold colours from the simulator JSON, as OpenCV hue (0..179)
HUE_ANCHORS = {"#F2554E": 1.0, "#59C77B": 70.0, "#6389ED": 112.0}


def load_gt(path):
    return json.load(open(path))["holds"]


def classify_colour(hue):
    """Nearest of the three simulator hold colours by (circular) hue distance."""
    dist = {c: min(abs(hue - h), 180 - abs(hue - h)) for c, h in HUE_ANCHORS.items()}
    return min(dist, key=dist.get)


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


def marker_size(path):
    """Side of the red origin marker in metres. The simulator (main.js, originMarkerDimensions) makes it
    1 m on walls at least 2 m wide, as the handout says, and 0.1 m on narrower walls (synthetic_wall3 is
    0.8 m wide). scene.json states the size; 1 m is the fallback."""
    try:
        with open(path) as f:
            return float(json.load(f)["scene"]["wall"]["originMarker"]["width"])
    except (OSError, KeyError, ValueError):
        return 1.0
