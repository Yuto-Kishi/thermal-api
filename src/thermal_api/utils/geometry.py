
from typing import Tuple

def top_third(ymin: int, ymax: int) -> Tuple[int, int]:
    h = max(1, ymax - ymin)
    return ymin, ymin + h // 3
