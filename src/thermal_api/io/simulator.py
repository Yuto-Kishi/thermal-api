import time
import numpy as np
import cv2 as cv
from typing import Iterator
from .base import ThermalSource
from ..types import ThermalFrame


class SimulatorSource(ThermalSource):
"""160x120 の温度場を合成し、人物/足跡/異常を模擬。"""
    def __init__(self, width=160, height=120, fps=8.7, seed=0):
        self.W, self.H = width, height
        self.period = 1.0 / fps
        self.rng = np.random.default_rng(seed)
        self.t = time.perf_counter()


# ベース温度: 22℃
        self.base = 22.0 * np.ones((self.H, self.W), np.float32)


    def _insert_person(self, temp: np.ndarray, cx: int, cy: int, r: int = 10):
        yy, xx = np.ogrid[:self.H, :self.W]
        mask = (yy - cy)**2 + (xx - cx)**2 <= r*r
        temp[mask] = np.maximum(temp[mask], 34.0 + 2.0*np.sin(0.2*time.time()))


    def _insert_head(self, temp: np.ndarray, cx: int, cy: int, r: int = 4):
        yy, xx = np.ogrid[:self.H, :self.W]
        mask = (yy - cy)**2 + (xx - cx)**2 <= r*r
        temp[mask] = np.maximum(temp[mask], 36.0)


    def _insert_footprints(self, temp: np.ndarray):
# 床に 20℃ 近辺の足跡（17–22℃帯）
        for i in range(3):
            x = self.rng.integers(5, self.W-5)
            y = self.H - 1 - self.rng.integers(5, 15)
            temp[y-2:y+2, x-3:x+3] = self.rng.uniform(18.0, 21.5)


    def _insert_abnormal(self, temp: np.ndarray):
        # 高温点（~90℃）
        temp[10:14, 10:14] = 90.0
        # 低温点（~5℃）
        temp[30:34, 50:54] = 5.0
        # 発火（>140℃）
        temp[70:75, 90:95] = 160.0


    def frames(self) -> Iterator[ThermalFrame]:
        while True:
            start = time.perf_counter()
            temp = self.base.copy()


# 動く人物（左→右）
            cx = int(20 + 40*np.sin(time.time()*0.4)) + 60
            cy = 60
            self._insert_person(temp, cx, cy, r=12)
            self._insert_head(temp, cx, cy-10, r=5)


            self._insert_footprints(temp)
            self._insert_abnormal(temp)


            # 可視化用 8bit
            tmin, tmax = 0.0, 140.0
            gray8 = np.clip((temp - tmin) * (255.0/(tmax - tmin)), 0, 255).astype(np.uint8)


            yield ThermalFrame(temp_c=temp.astype(np.float32), gray8=gray8, timestamp=time.time())


#FPS 制御
            dt = time.perf_counter() - start
            remain = self.period - dt
            if remain > 0:
                time.sleep(remain)