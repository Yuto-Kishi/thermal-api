# src/thermal_api/processing/human.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2

@dataclass
class HumanFilterParams:
    body_min: float = 25.0
    body_max: float = 39.0
    morph_kernel: int = 3
    min_region_area: int = 200
    line_h_max: int = 6
    aspect_ratio_max: float = 12.0
    nan_frac_max: float = 0.25

def _get_params_from_cfg(cfg) -> HumanFilterParams:
    p = cfg.processing
    t = cfg.thresholds
    return HumanFilterParams(
        body_min = float(getattr(t, "human_body_c_min", 25.0)),
        body_max = float(getattr(t, "human_body_c_max", 39.0)),
        morph_kernel = int(getattr(p, "morph_kernel", 3)),
        min_region_area = int(getattr(p, "min_region_area", 200)),
        line_h_max = int(getattr(p, "line_h_max", 6)),
        aspect_ratio_max = float(getattr(p, "aspect_ratio_max", 12.0)),
        nan_frac_max = float(getattr(p, "nan_frac_max", 0.25)),
    )

def detect_humans(temp_c: np.ndarray, cfg) -> List[Tuple[int,int,int,int]]:
    """
    Returns list of person boxes [x, y, w, h] in 160x120 image coordinates.
    """
    params = _get_params_from_cfg(cfg)
    H, W = temp_c.shape[:2]

    # 1) 有効画素だけ（NaN除外）
    valid = np.isfinite(temp_c)

    # 2) 人体レンジの温度だけを残す（背景は除去）
    mask = valid & (temp_c >= params.body_min) & (temp_c <= params.body_max)
    mask[:2,:] = False; mask[-2:,:] = False  # 端のノイズをカット
    mask[:, :2] = False; mask[:, -2:] = False

    if not np.any(mask):
        return []

    # 3) 形態素処理（小ノイズ除去と穴埋め）
    k = max(1, int(params.morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) 連結成分で水平線/小領域/NaN多めを除外
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    keep = np.zeros_like(mask_u8, dtype=np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < params.min_region_area:
            continue
        if h <= params.line_h_max or (w / max(1.0, float(h))) > params.aspect_ratio_max:
            # 横線や横長すぎるものを除外
            continue
        # NaN が多い領域は除外
        roi = temp_c[y:y+h, x:x+w]
        frac_nan = 1.0 - np.isfinite(roi).mean()
        if frac_nan > params.nan_frac_max:
            continue
        keep[labels == i] = 255

    # 5) バウンディングボックス取得（重複はNMSでまとめる）
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    if not boxes:
        return []

    # 簡易NMS（IoU > 0.5 はまとめる）
    boxes = _nms(boxes, iou_thresh=0.5)
    return boxes

def _nms(boxes: List[Tuple[int,int,int,int]], iou_thresh=0.5):
    if not boxes:
        return []
    # 面積大きい順
    areas = [w*h for (_,_,w,h) in boxes]
    order = np.argsort(-np.array(areas))
    kept = []
    while len(order):
        i = order[0]
        kept.append(boxes[i])
        if len(order) == 1:
            break
        rest = order[1:]
        iou_mask = []
        xi, yi, wi, hi = boxes[i]
        Ai = wi * hi
        for j in rest:
            xj, yj, wj, hj = boxes[j]
            Aj = wj * hj
            xx1 = max(xi, xj); yy1 = max(yi, yj)
            xx2 = min(xi+wi, xj+wj); yy2 = min(yi+hi, yj+hj)
            iw = max(0, xx2-xx1); ih = max(0, yy2-yy1)
            inter = iw * ih
            iou = inter / (Ai + Aj - inter + 1e-6)
            iou_mask.append(iou <= iou_thresh)
        order = rest[np.array(iou_mask)]
    return kept

def draw_overlays(gray8_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    out = gray8_bgr.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(out, "person", (x, max(10, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
    return out
