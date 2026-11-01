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
    quad = quad[np.argsort(quad[:, 1])]                 # two top corners, then two bottom
    tl, tr = sorted(quad[:2], key=lambda p: p[0])
    bl, br = sorted(quad[2:], key=lambda p: p[0])
    world = np.float32([[0, 1], [1, 1], [0, 0], [1, 0]]) * square_m
    H, _ = cv2.findHomography(np.float32([tl, tr, bl, br]), world)
    return H


def apply_homography(H, pts_px):
    """Pixel (u, v) -> wall coordinates (x, y) in metres with a homography."""
    pts = np.asarray(pts_px, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(float)
