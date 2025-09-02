
import numpy as np
import cv2 as cv

def to_gray8(temp_c, tmin=None, tmax=None):
    
    import numpy as np
    # NaN を無視して統計を取る
    valid = temp_c[np.isfinite(temp_c)]
    if valid.size == 0:
        return np.zeros_like(temp_c, dtype=np.uint8)

    # 自動スケーリング：デフォルトはパーセンタイル
    if tmin is None:
        tmin = float(np.percentile(valid, 2))   # 下位2%を黒に
    if tmax is None:
        tmax = float(np.percentile(valid, 98))  # 上位98%を白に

    # 異常に狭い範囲を防ぐ
    if tmax - tmin < 1e-3:
        return np.zeros_like(temp_c, dtype=np.uint8)

    g = np.clip((temp_c - tmin) * (255.0 / (tmax - tmin)), 0, 255)
    return g.astype(np.uint8)



def band_mask(temp_c: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return ((temp_c >= tmin) & (temp_c <= tmax)).astype(np.uint8) * 255

def morph_open(mask: np.ndarray, k: int = 3) -> np.ndarray:
    k = max(1, int(k) | 1)  # force odd
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    return cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

def cc_boxes(mask: np.ndarray, min_area: int = 8):
    num, lab, stats, cent = cv.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num):  # 0 is background
        x, y, w, h, a = stats[i]
        if a >= min_area:
            boxes.append((x, y, x+w, y+h, a))
    return boxes, lab
