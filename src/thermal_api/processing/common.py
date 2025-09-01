import cv2 as cv
import numpy as np


def to_gray8(temp_c: np.ndarray, tmin: float = 0.0, tmax: float = 140.0) -> np.ndarray:
    g = np.clip((temp_c - tmin) * (255.0/(tmax - tmin)), 0, 255)
    return g.astype(np.uint8)




def band_mask(temp_c: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return ((temp_c >= tmin) & (temp_c <= tmax)).astype(np.uint8) * 255




def morph_open(mask: np.ndarray, k: int = 3) -> np.ndarray:
    k = max(1, int(k) | 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    return cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)




def cc_boxes(mask: np.ndarray, min_area: int = 8):
    num, lab, stats, cent = cv.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            boxes.append((x, y, x+w, y+h, a))
    return boxes, lab