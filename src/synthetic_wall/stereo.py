"""Camera model of the simulator's "Front stereo camera" (read from its main.js).

Two pinhole cameras 0.064 m apart at (+-0.032, height, distance), both looking at the
wall point (0, height, 0); vertical field of view 74 deg. The PNG is the left-eye view
(left half) next to the right-eye view (right half). World frame = the scene JSON's
frame: x right, y up, z out of the wall (wall front at z = 0), metres.
"""
import cv2
import numpy as np

BASELINE = 0.064
FOV_Y_DEG = 74.0


class StereoRig:
    def __init__(self, png_width, png_height, height, distance):
        self.eye_w = png_width / 2.0
        self.eye_h = float(png_height)
        self.f = (self.eye_h / 2.0) / np.tan(np.radians(FOV_Y_DEG / 2.0))
        self.K = np.array([[self.f, 0, self.eye_w / 2.0],
                           [0, self.f, self.eye_h / 2.0],
                           [0, 0, 1.0]])
        target = np.array([0.0, height, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        self.centres, self.R, self.P = [], [], []
        for sign in (-1.0, +1.0):                       # left eye, right eye
            c = np.array([sign * BASELINE / 2.0, height, distance])
            fwd = (target - c) / np.linalg.norm(target - c)
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            true_up = np.cross(right, fwd)
            R = np.stack([right, -true_up, fwd])        # camera axes: x right, y down, z forward
            self.centres.append(c)
            self.R.append(R)
            self.P.append(self.K @ np.hstack([R, (-R @ c).reshape(3, 1)]))

    def project(self, pts_world, eye):
        """3D points (n, 3) -> pixel (u, v) inside that eye's own image (n, 2)."""
        p = np.asarray(pts_world, float).reshape(-1, 3)
        cam = (self.R[eye] @ (p - self.centres[eye]).T).T
        pix = (self.K @ cam.T).T
        return pix[:, :2] / pix[:, 2:3]

    def triangulate(self, uv_left, uv_right):
        """Matched pixels (n, 2) in each eye's own image -> 3D world points (n, 3)."""
        a = np.asarray(uv_left, float).reshape(-1, 2).T
        b = np.asarray(uv_right, float).reshape(-1, 2).T
        X = cv2.triangulatePoints(self.P[0], self.P[1], a, b)
        return (X[:3] / X[3]).T

    def split(self, img):
        """Cut the side-by-side PNG into (left, right) eye images."""
        w = int(round(self.eye_w))
        return img[:, :w], img[:, img.shape[1] - w:]
