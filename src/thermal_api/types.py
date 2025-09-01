from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass

class ThermalFrame:
    """1フレーム分のサーマルデータ。"""
    temp_c: np.ndarray # float32, shape: (H, W)
    gray8: np.ndarray # uint8, shape: (H, W) 可視化用
    timestamp: float
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.temp_c.shape