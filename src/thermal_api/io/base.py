from abc import ABC, abstractmethod
from typing import Iterator

from ..types import ThermalFrame


class ThermalSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[ThermalFrame]:
        ...