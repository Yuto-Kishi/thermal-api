from typing import Iterator

from ..types import ThermalFrame
from .base import ThermalSource


class PureThermalUSBSource(ThermalSource):
    def __init__(self, device="/dev/video0", width=160, height=120, fps=8.7):
        self.device = device
        self.W = width
        self.H = height
        self.fps = fps


    def frames(self) -> Iterator[ThermalFrame]:
        raise NotImplementedError("Implement PureThermal(UVC) capture for your hardware.")