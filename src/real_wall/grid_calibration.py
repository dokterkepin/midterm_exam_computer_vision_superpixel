"""Calibration of a real climbing-wall photo from the wall's grid of bolt holes (T-nuts), 20 cm apart.

The real photos have no red marker; the handout gives the grid spacing instead. The holes show as small dark
dots on the plywood. Steps:
  1. detect_dots       dark dots on plain board (black-hat filter, small blob, uniform surroundings)
  2. spacing_and_angle grid spacing (px) and direction from the distances / directions to neighbouring dots
  3. seed_lattice      integer grid cell (i, j) of the dots around one well-connected dot, walking neighbour to neighbour
  4. grow_lattice      fit a homography cell -> pixel to the assigned dots, assign dots that land on nearby
                       predicted cells, drop dots the fit disagrees with, refit; repeat (bridges holds and dark panels)
  5. calibrate_grid    homography pixel -> wall metres: origin at the bottom-left grid hole found, x right, y up

Holes on the dark panels and under holds are not seen, but the plane's homography covers them; holds off the
plane (volumes) get the position of the point of the wall behind them.
"""
from collections import deque

import cv2
import numpy as np

from real_wall import baseline

_G = baseline.P["grid"]


def detect_dots(img):
    """Centres (n, 2) of small dark dots whose surroundings are plain board."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = _G["blackhat_kernel_px"]
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, st, cen = cv2.connectedComponentsWithStats((bh > _G["dot_contrast"]).astype(np.uint8))
    # local spread of the image with the dots filled in: low on plain board, high on holds, edges and scratches
    filled = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    mu = cv2.blur(filled, (9, 9))
    sd = np.sqrt(np.maximum(cv2.blur(filled ** 2, (9, 9)) - mu ** 2, 0))
    keep = [i for i in range(1, n)
            if 2 <= st[i, cv2.CC_STAT_AREA] <= _G["max_dot_area_px"]
            and max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) <= _G["max_dot_width_px"]
            and sd[int(cen[i, 1]), int(cen[i, 0])] < _G["max_board_std"]]
    return cen[keep].astype(float)


def _neighbours(pts, k):
    """Indices (n, k) and distances (n, k) of the k nearest other dots."""
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    return idx, np.take_along_axis(d, idx, axis=1)


def spacing_and_angle(pts):
    """Grid spacing in px (most common neighbour distance) and grid direction in radians (mod 90 deg)."""
    idx, dist = _neighbours(pts, 4)
    hist, edges = np.histogram(dist.ravel(), bins=np.arange(8, 120, 1))
    s = edges[np.argmax(hist)] + 0.5
    v = (pts[idx] - pts[:, None]).reshape(-1, 2)[np.abs(dist.ravel() - s) < 0.25 * s]
    a = np.arctan2(v[:, 1], v[:, 0])
    return float(s), float(np.angle(np.mean(np.exp(4j * a))) / 4)      # 4x: the four grid directions fold together


def seed_lattice(pts, s, th):
    """{dot index: (i, j)} for the dots reachable from the best-connected dot by steps of one grid cell."""
    A = s * np.array([np.cos(th), np.sin(th)])
    B = s * np.array([-np.sin(th), np.cos(th)])
    nearest = lambda q: (np.linalg.norm(pts - q, axis=1).min(), int(np.linalg.norm(pts - q, axis=1).argmin()))
    seed = max(range(len(pts)), key=lambda k: sum(nearest(pts[k] + D)[0] < 0.3 * s for D in (A, -A, B, -B)))
    ij, todo = {seed: (0, 0)}, deque([(seed, A, B)])
    while todo:
        k, a, b = todo.popleft()
        for D, (di, dj) in ((a, (1, 0)), (-a, (-1, 0)), (b, (0, 1)), (-b, (0, -1))):
            d, m = nearest(pts[k] + D)
            if d < 0.3 * s and m not in ij:
                ij[m] = (ij[k][0] + di, ij[k][1] + dj)
                step = pts[m] - pts[k]                                   # follow the local spacing (perspective)
                todo.append((m, step * di if di else a, step * dj if dj else b))
    return ij


def _fit(pts, ij, tol):
    ks = list(ij)
    cells = np.float32([ij[k] for k in ks]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(cells, np.float32(pts[ks]).reshape(-1, 1, 2), cv2.RANSAC, tol)
    res = np.linalg.norm(cv2.perspectiveTransform(cells, H).reshape(-1, 2) - pts[ks], axis=1)
    return H, dict(zip(ks, res))


def _one_per_cell(ij, res):
    best = {}
    for k, c in ij.items():
        if c not in best or res[k] < res[best[c]]:
            best[c] = k
    return {k: c for c, k in best.items()}


def grow_lattice(pts, ij, s):
    """Region growing of the lattice; returns ({dot: (i, j)}, H cell -> pixel, residual px per dot)."""
    tol = _G["tolerance_cells"] * s
    reach = _G["reach_cells"]
    for _ in range(500):
        H, res = _fit(pts, ij, tol)
        ij = _one_per_cell({k: c for k, c in ij.items() if res[k] < tol}, res)     # drop what the fit disagrees with
        occupied = set(ij.values())
        free = np.array([k for k in range(len(pts)) if k not in ij])
        if not len(free):
            break
        cell = np.round(cv2.perspectiveTransform(np.float32(pts[free]).reshape(-1, 1, 2),
                                                 np.linalg.inv(H)).reshape(-1, 2)).astype(int)
        err = np.linalg.norm(cv2.perspectiveTransform(np.float32(cell).reshape(-1, 1, 2), H).reshape(-1, 2)
                             - pts[free], axis=1)
        before = len(ij)
        for k, c, e in sorted(zip(free.tolist(), map(tuple, cell.tolist()), err), key=lambda r: r[2]):
            near = any((c[0] + di, c[1] + dj) in occupied
                       for di in range(-reach, reach + 1) for dj in range(-reach, reach + 1))
            if e < tol and c not in occupied and near:
                ij[k] = c
                occupied.add(c)
        if len(ij) == before:
            break
    H, res = _fit(pts, ij, tol)
    return ij, H, np.array([res[k] for k in ij])


def calibrate_grid(img, spacing_m=_G["spacing_m"]):
    """Calibration of a wall photo from its bolt-hole grid; a dict with H (pixel -> wall metres, 3x3), px_per_m,
    the residuals, and the lattice (for drawing)."""
    pts = detect_dots(img)
    if len(pts) < 20:
        raise ValueError(f"only {len(pts)} grid holes found; cannot calibrate")
    s, th = spacing_and_angle(pts)
    ij, H_cell, res = grow_lattice(pts, seed_lattice(pts, s, th), s)
    cells = np.array(list(ij.values()))
    i0, j0 = int(cells[:, 0].min()), int(cells[:, 1].max())           # bottom-left hole found = origin
    # cell (i, j) -> metres: x = spacing (i - i0), y = spacing (j0 - j)  (image rows go down, wall y goes up)
    to_m = np.array([[spacing_m, 0, -spacing_m * i0], [0, -spacing_m, spacing_m * j0], [0, 0, 1.0]])
    H = to_m @ np.linalg.inv(H_cell)
    H /= H[2, 2]
    # local scale at every hole: pixels per metre along the grid, from the cell -> pixel map
    pix = lambda c: cv2.perspectiveTransform(np.float32(c).reshape(-1, 1, 2), H_cell).reshape(-1, 2)
    step = np.linalg.norm(pix(cells + [1, 0]) - pix(cells), axis=1)
    px_per_m = float(np.median(step) / spacing_m)
    extent = [0.0, spacing_m * float(cells[:, 0].max() - i0), 0.0, spacing_m * float(j0 - cells[:, 1].min())]
    return {"H": H, "H_cell": H_cell, "px_per_m": px_per_m, "spacing_px": s, "angle_deg": float(np.degrees(th)),
            "spacing_m": spacing_m, "n_dots": len(pts), "n_grid": len(ij), "cells": cells, "extent_m": extent,
            "grid_px": pts[list(ij)], "origin_cell": (i0, j0), "origin_px": pix([(i0, j0)])[0].tolist(),
            "residual_px": {"median": float(np.median(res)), "p95": float(np.percentile(res, 95)),
                            "max": float(res.max())}}


def draw(img, cal):
    """The fitted grid (every cell the holes cover), the holes used, the origin and the axes."""
    vis = img.copy()
    pix = lambda c: tuple(int(round(v)) for v in
                          cv2.perspectiveTransform(np.float32(c).reshape(1, 1, 2), cal["H_cell"])[0, 0])
    c = cal["cells"]
    (ia, ja), (ib, jb) = c.min(axis=0), c.max(axis=0)
    for i in range(ia, ib + 1):
        cv2.line(vis, pix((i, ja)), pix((i, jb)), (255, 0, 255), 1, cv2.LINE_AA)
    for j in range(ja, jb + 1):
        cv2.line(vis, pix((ia, j)), pix((ib, j)), (255, 0, 255), 1, cv2.LINE_AA)
    for x, y in cal["grid_px"]:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 220, 0), -1)
    i0, j0 = cal["origin_cell"]
    o = pix((i0, j0))
    cv2.arrowedLine(vis, o, pix((i0 + 5, j0)), (0, 0, 255), 3, tipLength=0.15)       # 1 m along x
    cv2.arrowedLine(vis, o, pix((i0, j0 - 5)), (255, 120, 0), 3, tipLength=0.15)     # 1 m along y
    cv2.circle(vis, o, 7, (0, 0, 255), 2)
    cv2.putText(vis, "x (1 m)", pix((i0 + 5, j0 + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, "y (1 m)", pix((i0 + 1, j0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
    return vis
