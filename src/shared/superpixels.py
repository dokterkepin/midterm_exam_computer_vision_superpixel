"""Thin wrappers around the three OpenCV ximgproc superpixel algorithms.

Each function takes a BGR image and returns (labels, n_superpixels). They run OpenCV single-threaded:
multithreaded LSC gives slightly different labels from run to run, and multithreaded SLIC is slower (4x at
32 threads). Speed comes from running walls / settings in parallel processes instead.
"""
from contextlib import contextmanager

import cv2
import numpy as np

ALGOS = ("slic", "lsc", "seeds")


def segment(algo, img, size):
    """Labels of one of ALGOS with superpixels about size x size pixels."""
    if algo == "slic":
        return slic(img, region_size=size)[0]
    if algo == "lsc":
        return lsc(img, region_size=size)[0]
    if algo == "seeds":
        return seeds(img, num_superpixels=img.shape[0] * img.shape[1] // (size * size))[0]
    raise ValueError(f"unknown superpixel algorithm {algo!r}")


@contextmanager
def _single_thread():
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(threads)


def slic(img, region_size=20, ruler=10.0, iterations=10):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB)
    sp = cv2.ximgproc.createSuperpixelSLIC(lab, cv2.ximgproc.SLICO, region_size, ruler)
    with _single_thread():
        sp.iterate(iterations)
    return sp.getLabels(), sp.getNumberOfSuperpixels()


def lsc(img, region_size=20, ratio=0.075, iterations=10):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    sp = cv2.ximgproc.createSuperpixelLSC(blur, region_size, ratio)
    with _single_thread():
        sp.iterate(iterations)
    return sp.getLabels(), sp.getNumberOfSuperpixels()


def seeds(img, num_superpixels=400, num_levels=4, prior=2, histogram_bins=5,
          double_step=False, iterations=10):
    h, w, c = img.shape
    sp = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, c, num_superpixels, num_levels, prior, histogram_bins, double_step)
    with _single_thread():
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
