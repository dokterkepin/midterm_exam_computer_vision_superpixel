"""Yaw of a hold by matching its silhouette against a rotated template.

All holds are the same object (a flat plate with two screw holes on a rounded bowl), so a hold
lying flat on the wall looks like the template rotated in the image plane by its yaw. The template
is the silhouette of one training hold at yaw 0 (see build_template); a detected hold's silhouette is
cut out of the original screenshot, normalised to the template's area and centroid, and the rotation
with the best overlap (IoU) is the yaw. Holds on the faces of a volume are foreshortened and not
handled (roll and pitch are not estimated).
"""
import cv2
import numpy as np

from synthetic_wall import baseline

_Y = baseline.P["yaw"]
CANVAS = _Y["canvas"]                  # square canvas for template and silhouettes
AREA = float(_Y["area"])               # silhouette area (px^2) both are normalised to
WORK_WIDTH = _Y["work_width"]          # hold width (px) the crop is resized to before segmentation
TEMPLATE_FILE = baseline.DIR / _Y["template"]
TEMPLATE_SOURCE = tuple(_Y["template_source"])   # (wall, hold id) the template was cut from: not scored


def silhouette(img, cx, cy, size):
    """Binary mask of the hold at (cx, cy) in the original image; size = rough hold width in px."""
    f = WORK_WIDTH / max(size, 3.0)
    half = int(round(max(size, 6.0) * 1.8))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REPLICATE)
    crop = pad[y0 + half:y0 + 3 * half, x0 + half:x0 + 3 * half]
    crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border[:, 1:], axis=0)                       # chroma of the wall; shadows keep it
    dist = np.linalg.norm(lab[..., 1:] - bg, axis=2)
    d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    # component nearest the crop centre, with enough area
    best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA] - 4 * np.hypot(*(np.array(
        [st[i, 0] + st[i, 2] / 2 - w / 2, st[i, 1] + st[i, 3] / 2 - h / 2]))))
    return (cc == best).astype(np.uint8)


def normalise(mask):
    """Centre the silhouette's centroid on a CANVAS x CANVAS image, scaled to area AREA."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    s = np.sqrt(AREA / len(xs))
    M = np.array([[s, 0, CANVAS / 2 - s * cx], [0, s, CANVAS / 2 - s * cy]])
    out = cv2.warpAffine(mask.astype(np.float32), M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
    return (out > 0.5).astype(np.uint8)


def build_template(img, cx, cy, size, yaw_deg=0.0):
    """Template from a training hold with known yaw: its silhouette, rotated back to yaw 0."""
    m = normalise(silhouette(img, cx, cy, size))
    R = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), -yaw_deg, 1.0)
    t = cv2.warpAffine(m, R, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST)
    cv2.imwrite(str(TEMPLATE_FILE), t * 255)
    return t


def load_template():
    return (cv2.imread(str(TEMPLATE_FILE), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)


def match_yaw(mask, template, step=_Y["angle_step_deg"], shifts=tuple(_Y["shifts_px"])):
    """Rotation (degrees, counter-clockwise on screen, [0, 360)) of the template that best overlaps
    the silhouette, and its IoU. mask must already be normalised."""
    best_iou, best_deg = -1.0, 0.0
    m = mask.astype(bool)
    for deg in range(0, 360, step):
        R = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), deg, 1.0)
        t = cv2.warpAffine(template, R, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST).astype(bool)
        for dx in shifts:
            for dy in shifts:
                ts = np.roll(np.roll(t, dy, axis=0), dx, axis=1)
                iou = (ts & m).sum() / max((ts | m).sum(), 1)
                if iou > best_iou:
                    best_iou, best_deg = iou, deg
    return best_deg, best_iou


def estimate_yaw(img, cx, cy, size, template):
    """Yaw in degrees [0, 360) of the hold at (cx, cy) in the original image, and the match IoU."""
    sil = silhouette(img, cx, cy, size)
    if sil is None or sil.sum() < 20:
        return None, 0.0
    return match_yaw(normalise(sil), template)
