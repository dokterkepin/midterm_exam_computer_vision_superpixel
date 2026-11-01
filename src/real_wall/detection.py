"""Hold detection on the real climbing-wall photos: the shared superpixel and candidate code with the real-wall values
(baseline/real_wall_parameters.json "detection", baseline/real_wall_algorithm.json).

prepare     photo + grid calibration -> working image (px_per_m) and pixel -> metres
candidates  local-contrast hold candidates with the real shape limits; on_wall keeps those on the calibrated wall
describe    type (nearest labelled hold colour), position (x, y; z = 0), yaw (longest straight edge, in wall coordinates)
labels      hand-labelled holds of the labels wall, and the same holds in the frame of an overlapping photo
"""
import json

import cv2
import numpy as np

from real_wall import baseline
from shared import candidates as cand, matching, paths, superpixels as sp

_R = baseline.P["detection"]
WORK_PX_PER_M = _R["px_per_m"]
WINDOW = _R["window_px"]
_S = _R["real_shape"]
SHAPE = dict(min_area=(_S["min_hold_m"] * WORK_PX_PER_M) ** 2, max_area=(_S["max_hold_m"] * WORK_PX_PER_M) ** 2,
             max_aspect=_S["max_aspect"], min_solidity=_S["min_solidity"])


def calibration(wall):
    """The saved grid calibration of a wall photo (outputs/<wall>/calibration.npz, written by calibrate_grid.py)."""
    with np.load(paths.out_path(wall, "calibration.npz")) as z:
        return {k: z[k] for k in z.files}


def prepare(wall):
    """(original photo, working image, up, to_world) where to_world maps a working-image pixel to wall metres."""
    cal = calibration(wall)
    H = np.array(cal["H_pixel_to_m"])
    orig = cv2.imread(paths.image_path(wall))
    up = WORK_PX_PER_M / float(cal["px_per_m"])
    img = cv2.resize(orig, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA)
    to_world = lambda p: cv2.perspectiveTransform(np.float32(p).reshape(-1, 1, 2) / up, H).reshape(-1, 2)[0].astype(float)
    return orig, img, up, to_world


def candidates(img, labels, thresh, min_sat, contrast=None):
    """(mask, blobs) of hold candidates on the working image."""
    return cand.hold_candidates(img, labels, thresh=thresh, window=WINDOW, open_k=_R["open_kernel"],
                                min_sat=min_sat, contrast=contrast, **SHAPE)


def on_wall(wall, blobs, to_world):
    """Blobs whose centre lies on the calibrated wall area (not the floor, ceiling or side walls)."""
    x0, x1, y0, y1 = calibration(wall)["extent_m"]
    m = 0.1
    keep = []
    for b in blobs:
        b["world"] = to_world(b["centroid"])
        if x0 - m <= b["world"][0] <= x1 + m and y0 - m <= b["world"][1] <= y1 + m:
            keep.append(b)
    return keep


# ---- labels -----------------------------------------------------------------------------------------------

def _labelled(wall):
    with open(paths.DATASET / wall / "labels.json") as f:
        return json.load(f)["holds"]


def frame_offset(wall, ref):
    """Shift (metres) from photo `ref`'s wall frame to photo `wall`'s. Both frames sit on the same 20 cm grid, so the
    shift is a whole number of cells: every pair of candidates (one per photo) votes for its rounded cell shift."""
    pts = {}
    for w in (wall, ref):
        _, img, _, to_world = prepare(w)
        labels = sp.segment("slic", img, 5)
        _, blobs = candidates(img, labels, 20, 0)
        pts[w] = np.array([to_world(b["centroid"]) for b in on_wall(w, blobs, to_world)])
    s = baseline.P["grid"]["spacing_m"]
    d = (pts[wall][:, None] - pts[ref][None]).reshape(-1, 2)
    cells = np.round(d / s)
    close = np.linalg.norm(d - cells * s, axis=1) < 0.05
    vals, counts = np.unique(cells[close], axis=0, return_counts=True)
    return (vals[np.argmax(counts)] * s).tolist()


def labels(wall):
    """({hold id: {"position": {"x", "y"}, "color"}}, labelled region) in `wall`'s frame. The region is the part of the
    wall the labels wall's photo shows (a polygon in metres): only detections inside it can be scored.

    The labels wall uses its own labels; another photo of the same wall gets them shifted by frame_offset()."""
    ref = _R["labels_wall"]
    H = np.array(calibration(ref)["H_pixel_to_m"])
    holds = _labelled(ref)
    xy = cv2.perspectiveTransform(np.float32([h["pixel"] for h in holds]).reshape(-1, 1, 2), H).reshape(-1, 2)
    if wall != ref:
        cache = paths.out_path(wall, "frame_offset.json")
        try:
            with open(cache) as f:
                off = json.load(f)[ref]
        except (OSError, KeyError):
            off = frame_offset(wall, ref)
            with open(cache, "w") as f:
                json.dump({ref: off, "note": f"shift (m) from the {ref} frame to the {wall} frame"}, f)
        shift = np.array(off)
    else:
        shift = np.zeros(2)
    gt = {h["id"]: {"position": {"x": float(x), "y": float(y)}, "color": h["color"]} for h, (x, y) in zip(holds, xy + shift)}
    # the labels wall's photo, 4 px inside its border, on the wall, in metres
    hh, ww = cv2.imread(paths.image_path(ref)).shape[:2]
    corners = np.float32([[4, 4], [ww - 5, 4], [ww - 5, hh - 5], [4, hh - 5]]).reshape(-1, 1, 2)
    region = cv2.perspectiveTransform(corners, H).reshape(-1, 2) + shift
    return gt, region.astype(np.float32)


def in_region(blobs, region):
    return [b for b in blobs if cv2.pointPolygonTest(region.reshape(-1, 1, 2), tuple(float(v) for v in b["world"]), False) >= 0]


def score(blobs, gt, region, to_world):
    """Match the blobs inside the labelled region to the labels (greedy, within match_gate_m)."""
    return matching.match(in_region(blobs, region), gt, to_world, gate=_R["match_gate_m"])


# ---- type, position, orientation ------------------------------------------------------------------------

def _lab(img):
    return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)


def hold_colour(lab_pixels):
    """Lab colour of a hold from its pixels: median of the more saturated half (drops shadow, board, bolt holes)."""
    chroma = np.hypot(lab_pixels[:, 1] - 128, lab_pixels[:, 2] - 128)
    return np.median(lab_pixels[chroma >= np.median(chroma)], axis=0)


def colour_prototypes(min_examples=3):
    """Mean Lab colour of each labelled hold colour (with at least min_examples holds), measured around the
    labelled holds of the labels wall."""
    ref = _R["labels_wall"]
    lab = _lab(cv2.imread(paths.image_path(ref)))
    r = _R["colour_radius_px"]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = xx ** 2 + yy ** 2 <= r * r
    per = {}
    for h in _labelled(ref):
        x, y = h["pixel"]
        patch = lab[y - r:y + r + 1, x - r:x + r + 1]
        if patch.shape[:2] == disc.shape:
            per.setdefault(h["color"], []).append(hold_colour(patch[disc]))
    return {c: np.mean(v, axis=0) for c, v in per.items() if len(v) >= min_examples}


def classify_colour(lab_colour, protos):
    """Nearest prototype; lightness counts half, since shading changes it more than it changes the hue."""
    w = np.array([0.5, 1.0, 1.0])
    return min(protos, key=lambda c: float(np.linalg.norm((lab_colour - protos[c]) * w)))


def describe(blobs, mask, img, orig, up, to_world, protos):
    """Type, wall position and yaw of every blob, in place (blob["type"], ["world"], ["yaw"])."""
    lab = _lab(img)
    for b in blobs:
        x, y, w, h = b["bbox"]
        region = mask[y:y + h, x:x + w] > 0
        b["type"] = classify_colour(hold_colour(lab[y:y + h, x:x + w][region]), protos)
        b["world"] = to_world(b["centroid"])
        # holding edge: direction of the longest straight edge (image degrees) -> angle on the wall (x right, y up)
        t = np.radians(b["long_deg"])
        p, q = np.array(b["centroid"]), np.array(b["centroid"]) + 10 * np.array([np.cos(t), np.sin(t)])
        dx, dy = to_world(q) - to_world(p)
        b["yaw"] = float((np.arctan2(dy, dx) + np.pi / 2) % np.pi - np.pi / 2)            # (-90, 90] deg, no direction
    return blobs
