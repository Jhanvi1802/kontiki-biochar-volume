#!/usr/bin/env python
"""
Stage 1 segmentation methods + IoU utility.

  classical_fill_mask : no-training control baseline (Method 3 in the plan).
                        Finds the dark granular region (charcoal) by Otsu on the
                        object interior. Works from RGB only.
  YOLO masks          : produced by predict_yolo using a trained YOLO11-seg model
                        (Method 1). Training/export handled in train_yolo.py.
"""
import cv2, numpy as np


def iou(a, b):
    a = a > 0; b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def _largest_cc(bw):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8))
    if n <= 1:
        return bw.astype(np.uint8) * 0
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8) * 255


def classical_fill_mask(rgb):
    """Segment the dark charcoal fill region. The scene is: bright background,
    mid-tone container, dark charcoal. We (1) split foreground object from the
    bright background, (2) within the object keep the DARK pixels (charcoal),
    (3) clean up and take the largest blob."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # foreground = anything notably darker than the bright background
    bg = np.median(blur[[0, -1], :].ravel().tolist() + blur[:, [0, -1]].ravel().tolist())
    fg = (blur < bg - 25).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = _largest_cc(fg)                                   # the container+content blob
    if fg.max() == 0:
        return np.zeros_like(gray)
    # within the object, charcoal is the darker sub-population -> Otsu on fg pixels
    vals = blur[fg > 0]
    if len(vals) < 50:
        return np.zeros_like(gray)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((blur <= thr) & (fg > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _largest_cc(dark)


def classical_container_mask(rgb):
    """Segment the whole container blob (foreground vs bright background).
    Used for the real-track container-IoU baseline."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bg = np.median(blur[[0, -1], :].ravel().tolist() + blur[:, [0, -1]].ravel().tolist())
    fg = (np.abs(blur.astype(int) - bg) > 22).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _largest_cc(fg)
