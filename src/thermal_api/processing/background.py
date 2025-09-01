
import numpy as np

class ExpMovingBG:
    def __init__(self, shape, alpha: float = 0.02):
        self.alpha = float(alpha)
        self.bg = np.zeros(shape, np.float32)
        self.initialized = False

    def apply(self, temp_c: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.bg[...] = temp_c
            self.initialized = True
        else:
            self.bg = (1.0 - self.alpha) * self.bg + self.alpha * temp_c
        diff = np.abs(temp_c - self.bg)
        mask = (diff > 0.5).astype(np.uint8) * 255
        return mask
