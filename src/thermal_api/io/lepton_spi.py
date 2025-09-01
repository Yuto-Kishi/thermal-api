from typing import Iterator

from ..types import ThermalFrame
from .base import ThermalSource


class LeptonSPISource(ThermalSource):
    def __init__(self, bus=0, device=0, width=160, height=120, fps=8.7):
        self.bus = bus
        self.device = device
        self.W = width
        self.H = height
        self.fps = fps



    def frames(self) -> Iterator[ThermalFrame]:
        raise NotImplementedError("Implement Lepton SPI capture for your hardware.")