
"""Raspberry Pi 3B + Breakout (SPI) using pylepton to capture radiometric frames."""
from typing import Iterator
import time
import numpy as np
from ..types import ThermalFrame
from .base import ThermalSource
from ..processing.common import to_gray8

try:
    from pylepton import Lepton
except Exception as e:
    Lepton = None  # handled in frames()

class LeptonSPISource(ThermalSource):
    def __init__(self, bus=0, device=0, width=160, height=120, fps=8.7, devpath=None):
        self.bus = bus
        self.device = device
        self.W = width
        self.H = height
        self.period = 1.0 / float(fps)
        self.dev = devpath or f"/dev/spidev{bus}.{device}"
        self._lepton = None

    def frames(self) -> Iterator[ThermalFrame]:
        if Lepton is None:
            raise RuntimeError("pylepton not installed. `pip install pylepton` on the Pi.")
        if self._lepton is None:
            self._lepton = Lepton(self.dev)
        try:
            while True:
                t0 = time.perf_counter()
                raw, _ = self._lepton.capture()  # uint16 centi-K
                if raw.shape != (self.H, self.W):
                    import cv2 as cv
                    raw = cv.resize(raw, (self.W, self.H), interpolation=cv.INTER_NEAREST)
                temp_k = raw.astype(np.float32) / 100.0
                temp_c = temp_k - 273.15
                gray8 = to_gray8(temp_c)
                yield ThermalFrame(temp_c=temp_c, gray8=gray8, timestamp=time.time())
                dt = time.perf_counter() - t0
                rem = self.period - dt
                if rem > 0:
                    time.sleep(rem)
        finally:
            try:
                if self._lepton is not None:
                    self._lepton.close()
            except Exception:
                pass
