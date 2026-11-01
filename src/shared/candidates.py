"""Superpixels -> hold candidates (local contrast, blob filtering); used by the synthetic and the real walls."""
import cv2
import numpy as np


def patch_mean_lab(img, labels):
    """Mean Lab colour of every superpixel, as an (n, 3) array."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    n = labels.max() + 1
    flat = labels.ravel()
    cnt = np.maximum(np.bincount(flat, minlength=n), 1)
    return np.stack([np.bincount(flat, lab[..., c].ravel(), minlength=n) / cnt
                     for c in range(3)], axis=1)


def local_contrast(img, labels, window=61, ds=4):
    """Per-pixel Lab distance between a superpixel's colour and the median colour
    of the superpixels in a window around it.

    A hold is much smaller than the window, so it always differs from the median.
    On a straight edge a pixel's own side holds most of the window, so edges are
    mostly not flagged (only corners and thin tips are).
    """
    mean_img = patch_mean_lab(img, labels)[labels].astype(np.uint8)
    small = cv2.resize(mean_img, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(small, int(window / ds) | 1)
    med = cv2.resize(med, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.linalg.norm(mean_img.astype(np.float32) - med.astype(np.float32), axis=2)


def refine_mask(lab_img, bbox, pad=10, frac=0.5):
    """Pixel-level hold mask inside a blob's bbox.

    Superpixels are ~8 px wide and holds ~20 px, so the superpixel mask is too coarse
    for an edge direction. Here every pixel is compared with the median colour of the
    surrounding ring; pixels further than frac * (95th percentile) count as hold.
    """
    x, y, w, h = bbox
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    patch = lab_img[y0:y + h + pad, x0:x + w + pad]
    inner = np.zeros(patch.shape[:2], np.uint8)
    inner[y - y0:y - y0 + h, x - x0:x - x0 + w] = 1
    grown = cv2.dilate(inner, np.ones((5, 5), np.uint8)).astype(bool)
    bg = np.median(patch[~grown], axis=0)
    dist = np.linalg.norm(patch - bg, axis=2)
    pm = ((dist > frac * np.percentile(dist[inner > 0], 95)) & grown).astype(np.uint8)
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(pm)
    if n < 2:
        return None, (x0, y0)
    return (cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8), (x0, y0)


def longest_edge_deg(pm):
    """Direction (image degrees, [0, 180)) of the longest side of the hold's polygon."""
    cnts, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)[:, 0, :].astype(float)
    best_len, best_deg = 0.0, None
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        length = np.hypot(*(q - p))
        if length > best_len:
            best_len, best_deg = length, np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])) % 180.0
    return best_deg


def hold_candidates(img, labels, thresh=50.0, window=61, open_k=3, min_area=80,
                    max_area=1500, max_aspect=2.5, min_solidity=0.85, min_sat=0.0, contrast=None):
    """Return (mask, blobs). A blob is a compact, small region of high local contrast.

    min_sat drops blobs whose colour is not saturated (HSV S, 0-255): holds are brightly
    coloured, while shadows and the faces of the volumes are not.
    """
    if contrast is None:
        contrast = local_contrast(img, labels, window)
    mask = (contrast > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, cc, stats, cent = cv2.connectedComponentsWithStats(mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        # work inside the blob's bounding box only (a full-image mask per blob dominated the run time)
        bx, by, bw, bh = (int(v) for v in stats[i, :4])
        region = cc[by:by + bh, bx:bx + bw] == i
        cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(bx, by))
        c = max(cnts, key=cv2.contourArea)
        solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
        (_, _), (w, h), ang = cv2.minAreaRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect or solidity < min_solidity:
            continue
        # direction of the longest side of the min-area rectangle, image degrees in [0, 180)
        long_deg = (ang if w >= h else ang + 90.0) % 180.0
        # a finer estimate from the pixel-level mask; keep the coarse one as fallback
        pm, _ = refine_mask(lab_img, tuple(int(v) for v in stats[i, :4]))
        edge_deg = longest_edge_deg(pm) if pm is not None else None
        if edge_deg is not None:
            long_deg = edge_deg
        # colour: hue/saturation of the most saturated half of the blob's pixels
        box = hsv[by:by + bh, bx:bx + bw]
        s = box[..., 1][region]
        top = region.copy()
        top[region] = s >= np.median(s)
        sat = float(np.median(box[..., 1][top]))
        if sat < min_sat:
            continue
        blobs.append({
            "centroid": tuple(float(v) for v in cent[i]), "area": area,
            "bbox": tuple(int(v) for v in stats[i, :4]),
            "solidity": float(solidity), "aspect": float(aspect), "long_deg": float(long_deg),
            "hue": float(np.median(box[..., 0][top])), "sat": sat,
        })
    return mask, blobs
