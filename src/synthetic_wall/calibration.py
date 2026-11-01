"""Calibration of a synthetic wall screenshot from the red origin square (the real walls use the bolt-hole grid instead)."""
import cv2
import numpy as np


def calibrate_red_square(img, square_m=1.0):
    """Find the red origin marker; return (px_per_m, origin_px, (w, h)).

    origin_px is the bottom-left corner of the square in pixels. The marker is a
    pale salmon rectangle, less saturated than the red holds, so we pick the
    largest salmon-coloured connected component.
    """
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    salmon = (((h < 10) | (h > 170)) & (s > 60) & (s < 150) & (v > 150)).astype(np.uint8)
    salmon = cv2.morphologyEx(salmon, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(salmon)
    if n < 2:
        raise ValueError("red square not found")
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, hh, _ = stats[i]
    return (w + hh) / 2.0 / square_m, (x, y + hh), (w, hh)


def pixel_to_world(pts_px, px_per_m, origin_px):
    """Pixel (u, v) -> wall coordinates (x, y) in metres; y points up."""
    pts = np.asarray(pts_px, float).reshape(-1, 2)
    return np.stack([(pts[:, 0] - origin_px[0]) / px_per_m,
                     (origin_px[1] - pts[:, 1]) / px_per_m], axis=1)


def subpixel_quad(img, quad, bbox, pad=6):
    """Enlarge the contour corners of the square to its true size.

    Contour points lie on the centres of the edge pixels, so the square comes out 1-2 px too small: 2.7% on a
    65 px marker (0.1 m marker in a small screenshot), which grows to 4 cm over a 1.4 m wall. The true area is
    the soft coverage of the square (anti-aliased edge pixels count by how far their colour is from the wall's
    towards the marker's, in chroma); the corners are scaled about their centre to that area.
    """
    x, y, w, h = bbox
    y0, y1, x0, x1 = max(y - pad, 0), min(y + h + pad, img.shape[0]), max(x - pad, 0), min(x + w + pad, img.shape[1])
    # chroma (Lab a, b) only: the marker differs from the wall by its color, and the dark floor under the wall,
    # which can touch the square, must not count by its lightness
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)[..., 1:]
    inner = np.zeros(lab.shape[:2], bool)
    inner[y - y0 + 3:y - y0 + h - 3, x - x0 + 3:x - x0 + w - 3] = True
    ring = np.ones(lab.shape[:2], bool)
    ring[y - y0 - 2:y - y0 + h + 2, x - x0 - 2:x - x0 + w + 2] = False
    if inner.sum() < 9 or ring.sum() < 9:
        return quad
    fg, bg = np.median(lab[inner], axis=0), np.median(lab[ring], axis=0)
    d = fg - bg
    cover = np.clip((lab - bg) @ d / (d @ d), 0, 1)
    cover[ring] = 0                                     # the ring is the wall by definition
    area = float(cover.sum())
    c = quad.mean(axis=0)
    k = np.sqrt(area / max(cv2.contourArea(quad), 1.0))
    return (c + (quad - c) * k).astype(np.float32)


def calibrate_homography(img, square_m=1.0):
    """Homography from image pixels to wall metres, from the red origin square's 4 corners.

    Unlike a single scale + offset this also corrects a tilted camera. World frame:
    bottom-left corner of the square = (0, 0), x right, y up.  Needs the whole square
    in view. Returns the 3x3 matrix H.
    """
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    salmon = (((h < 10) | (h > 170)) & (s > 60) & (s < 150) & (v > 150)).astype(np.uint8)
    salmon = cv2.morphologyEx(salmon, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(salmon)
    if n < 2:
        raise ValueError("red square not found")
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, hh = stats[i, :4]
    if x == 0 or y == 0 or x + w >= img.shape[1] or y + hh >= img.shape[0]:
        raise ValueError("red square is cut off by the image border; cannot get its 4 corners")
    cnts, _ = cv2.findContours((cc == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)[:, 0, :].astype(np.float32)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    quad = subpixel_quad(img, quad, (x, y, w, hh))
    quad = quad[np.argsort(quad[:, 1])]                 # two top corners, then two bottom
    tl, tr = sorted(quad[:2], key=lambda p: p[0])
    bl, br = sorted(quad[2:], key=lambda p: p[0])
    world = np.float32([[0, 1], [1, 1], [0, 0], [1, 0]]) * square_m
    H, _ = cv2.findHomography(np.float32([tl, tr, bl, br]), world)
    return H


def save(path, img, square_m=1.0):
    """Calibrate the screenshot, save the result as a .npz and return the homography.

    Fields: H_pixel_to_m (3x3), px_per_m (near the square), marker_m, origin_px (bottom-left corner of the square),
    corners_px (the square's 4 corners: bottom-left, bottom-right, top-right, top-left), image_wh, frame.
    """
    H = calibrate_homography(img, square_m=square_m)
    Hi = np.linalg.inv(H)
    corners = np.array([apply_homography(Hi, [[x * square_m, y * square_m]])[0] for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))])
    side = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[3] - corners[0])) / 2
    np.savez(path, H_pixel_to_m=H, px_per_m=side / square_m, marker_m=square_m, origin_px=corners[0],
             corners_px=corners, image_wh=np.array([img.shape[1], img.shape[0]]),
             frame="origin = bottom-left corner of the red square; x right, y up, z out of the wall; metres")
    return H


def _paths():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shared import paths
    return paths


def calibrate_wall(wall, marker=None):
    """The calibration step of a synthetic wall: find the red square in dataset/<wall>/ and save
    outputs/<wall>/calibration.npz. marker: the square's side in metres (default: scene.json, else 1).
    Run once per wall before the detection (run_wall.run and the sweep do it); returns the homography."""
    from synthetic_wall import evaluate as ev
    paths = _paths()
    img = cv2.imread(paths.image_path(wall))
    if marker is None:
        marker = ev.marker_size(paths.gt_path(wall))
    return save(paths.out_path(wall, "calibration.npz"), img, square_m=marker)


def load_homography(wall):
    """Homography pixel -> metres from outputs/<wall>/calibration.npz (calibrates the wall first if the file is missing)."""
    from pathlib import Path
    f = Path(_paths().out_path(wall, "calibration.npz"))
    if not f.exists():
        return calibrate_wall(wall)
    with np.load(f) as z:
        return z["H_pixel_to_m"]


def apply_homography(H, pts_px):
    """Pixel (u, v) -> wall coordinates (x, y) in metres with a homography."""
    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(float)


def check(image_path, marker=None, out_dir=None):
    """Run the calibration on one screenshot and show what it found (for testing by hand).

    Prints the square's side in pixels as the plain contour gave it and after the sub-pixel step, and the scale.
    marker: side of the red square in metres; default = scene.json next to the image, else 1.
    out_dir: where to write calibration_check.png (the corners on the screenshot) and calibration_check_zoom.png
    (the square enlarged 6x); default = the image's folder in outputs/ if it is in dataset/, else not written.
    """
    import json
    from pathlib import Path

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    if marker is None:
        scene = image_path.parent / "scene.json"
        try:
            marker = float(json.load(open(scene))["scene"]["wall"]["originMarker"]["width"])
        except (OSError, KeyError, ValueError):
            marker = 1.0

    def corners_and_side(H):
        Hi = np.linalg.inv(H)
        c = [apply_homography(Hi, [[x * marker, y * marker]])[0] for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
        return c, float(np.mean([np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[3] - c[0])]))

    corners, side = corners_and_side(calibrate_homography(img, square_m=marker))
    global subpixel_quad
    keep, subpixel_quad = subpixel_quad, (lambda img, quad, bbox, **k: quad)    # same calibration, step switched off
    try:
        _, contour_side = corners_and_side(calibrate_homography(img, square_m=marker))
    finally:
        subpixel_quad = keep
    print(f"{image_path.parent.name}: marker {marker} m | contour side {contour_side:6.1f} px -> sub-pixel side "
          f"{side:6.1f} px | scale {side / marker:7.1f} px/m ({100 * (side / contour_side - 1):+.1f}%)")

    if out_dir is None and image_path.parent.parent.name == "dataset":
        out_dir = image_path.parents[2] / "outputs" / image_path.parent.name
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pts = np.array(corners, np.float32).round().astype(np.int32)
        out = img.copy()
        cv2.polylines(out, [pts], True, (0, 200, 0), 1)
        for p in pts:
            cv2.circle(out, tuple(int(c) for c in p), 3, (0, 0, 255), -1)
        x, y, w, h = cv2.boundingRect(pts)
        pad = max(20, w // 2)
        crop = out[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        cv2.imwrite(str(out_dir / "calibration_check.png"), out)
        cv2.imwrite(str(out_dir / "calibration_check_zoom.png"),
                    cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
        save(out_dir / "calibration.npz", img, square_m=marker)
        print(f"  wrote {out_dir}/calibration.npz, calibration_check.png and calibration_check_zoom.png")
    return side


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    # ---- Run-button settings: used when the command line gives none ----
    IMAGES = []           # screenshot paths, e.g. ["dataset/synthetic_wall4/wall.png"]; empty = every synthetic wall
    MARKER = None         # red square side in metres; None = scene.json next to the image, else 1
    OUT = None            # folder for calibration_check*.png; None = outputs/<wall>/ for images in dataset/

    ap = argparse.ArgumentParser(description="Test the red-square calibration on screenshots.")
    ap.add_argument("images", nargs="*", help="screenshot files (default: every synthetic wall in dataset/)")
    ap.add_argument("--marker", type=float, default=MARKER, help="red square side in metres (default: scene.json next to the image, else 1)")
    ap.add_argument("--out", default=OUT, help="folder for calibration_check*.png (default: outputs/<wall>/ for images in dataset/)")
    a = ap.parse_args()
    files = a.images or IMAGES
    if not files:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from shared import paths
        files = [paths.image_path(w) for w in paths.synthetic_walls()]
    for f in files:
        check(f, marker=a.marker, out_dir=a.out)
