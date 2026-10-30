"""Thin wrappers around the three OpenCV ximgproc superpixel algorithms.

Each function takes a BGR image and returns (labels, n_superpixels).
"""
import cv2
import numpy as np


def slic(img, region_size=20, ruler=10.0, iterations=10):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB)
    sp = cv2.ximgproc.createSuperpixelSLIC(lab, cv2.ximgproc.SLICO, region_size, ruler)
    sp.iterate(iterations)
    return sp.getLabels(), sp.getNumberOfSuperpixels()


def lsc(img, region_size=20, ratio=0.075, iterations=10):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    sp = cv2.ximgproc.createSuperpixelLSC(blur, region_size, ratio)
    sp.iterate(iterations)
    return sp.getLabels(), sp.getNumberOfSuperpixels()


def seeds(img, num_superpixels=400, num_levels=4, prior=2, histogram_bins=5,
          double_step=False, iterations=10):
    h, w, c = img.shape
    sp = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, c, num_superpixels, num_levels, prior, histogram_bins, double_step)
    sp.iterate(img, iterations)
    return sp.getLabels(), sp.getNumberOfSuperpixels()


def boundaries(img, labels, color=(0, 0, 0)):
    """Draw superpixel borders on a copy of img."""
    out = img.copy()
    mask = np.zeros(labels.shape, np.uint8)
    mask[:, :-1] |= (labels[:, :-1] != labels[:, 1:]).astype(np.uint8)
    mask[:-1, :] |= (labels[:-1, :] != labels[1:, :]).astype(np.uint8)
    out[mask > 0] = color
    return out
