
from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass
class ThermalFrame:
    temp_c: np.ndarray   # float32 (H, W)
    gray8: np.ndarray    # uint8   (H, W) for visualization
    timestamp: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.temp_c.shape
