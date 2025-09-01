
from ..processing.common import band_mask, morph_open, cc_boxes

class AbnormalAlarm:
    def __init__(self, low_max=10.0, high_min=60.0, high_max=140.0, fire_min=140.0, k=3, min_area=8):
        self.low_max = low_max
        self.high_min = high_min
        self.high_max = high_max
        self.fire_min = fire_min
        self.k = k
        self.min_area = min_area

    def detect(self, temp_c):
        low = band_mask(temp_c, -273.0, self.low_max)
        high = band_mask(temp_c, self.high_min, self.high_max)
        fire = band_mask(temp_c, self.fire_min, 1000.0)

        out = {}
        for name, m in ("low", low), ("high", high), ("fire", fire):
            mm = morph_open(m, self.k)
            boxes, _ = cc_boxes(mm, self.min_area)
            out[name] = [{"bbox": (x1, y1, x2, y2), "area": int(a)} for (x1, y1, x2, y2, a) in boxes]
            out[name + "_mask"] = mm
        return out
