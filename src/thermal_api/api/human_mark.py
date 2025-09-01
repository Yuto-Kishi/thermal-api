import cv2 as cv
import numpy as np

from ..processing.common import band_mask, cc_boxes, morph_open
from ..utils.geometry import top_third


class HumanMark:
    def __init__(self, body_min=24.0, body_max=39.0, head_min=29.0, head_max=39.0, k=3, min_area=8):
        self.body_min = body_min
        self.body_max = body_max
        self.head_min = head_min
        self.head_max = head_max
        self.k = k
        self.min_area = min_area


    def detect(self, temp_c: np.ndarray, motion_mask: np.ndarray):
# 人体候補: 動き ∧ 温度帯(24–39℃)
        body_mask = cv.bitwise_and(motion_mask, band_mask(temp_c, self.body_min, self.body_max))
        body_mask = morph_open(body_mask, self.k)
        boxes, lab = cc_boxes(body_mask, self.min_area)


        bodies = []
        heads = []
        for (x1, y1, x2, y2, a) in boxes:
            roi = temp_c[y1:y2, x1:x2]
# 胴体上 1/3 を頭探索領域
            ty1, ty2 = top_third(y1, y2)
            head_band = band_mask(temp_c, self.head_min, self.head_max)
            head_roi = head_band[ty1:ty2, x1:x2]


# 円検出（HoughCircles）。小さな頭部を想定して半径制限
            circ = None
            try:
                blurred = cv.GaussianBlur(head_roi, (0, 0), 1.0)
                c = cv.HoughCircles(
                    blurred, cv.HOUGH_GRADIENT, dp=1.2, minDist=6,
                    param1=80, param2=10, minRadius=2, maxRadius=max(3, (y2-y1)//6)
                )
                if c is not None:
                    c = np.uint16(np.around(c))
# 最大応答を採用
                    cx, cy, r = c[0][0]
                    heads.append({
                        "bbox": (x1, ty1, x2, ty2),
                        "circle": (x1+int(cx), ty1+int(cy), int(r))
                    })
            except Exception:
                pass


# 体温（最大温度）
            max_temp = float(np.max(roi)) if roi.size else float("nan")
            bodies.append({
                "bbox": (x1, y1, x2, y2),
                "area": int(a),
                "max_temp_c": max_temp,
                })


        return {"bodies": bodies, "heads": heads, "mask": body_mask}