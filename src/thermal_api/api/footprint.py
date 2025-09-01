import cv2 as cv

from ..processing.common import band_mask, cc_boxes, morph_open


class FootPrint:
    def __init__(self, tmin=17.0, tmax=22.0, k=3, min_area=8):
        self.tmin = tmin
        self.tmax = tmax
        self.k = k
        self.min_area = min_area


    def detect(self, temp_c):
        mask = band_mask(temp_c, self.tmin, self.tmax)
        mask = morph_open(mask, self.k)
        boxes, _ = cc_boxes(mask, self.min_area)
        footprints = [
        {"bbox": (x1, y1, x2, y2), "area": int(a)} for (x1, y1, x2, y2, a) in boxes
        ]
        return {"footprints": footprints, "mask": mask}