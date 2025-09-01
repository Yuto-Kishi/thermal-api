
from pydantic import BaseModel
from typing import Literal
import yaml

class IOConfig(BaseModel):
    driver: Literal["simulator", "lepton_spi", "purethermal"] = "simulator"
    fps: float = 8.7
    width: int = 160
    height: int = 120
    spi_bus: int = 0
    spi_device: int = 0
    usb_device: str = "/dev/video0"

class Thresholds(BaseModel):
    human_body_c_min: float = 24.0
    human_body_c_max: float = 39.0
    human_head_c_min: float = 29.0
    human_head_c_max: float = 39.0
    footprint_c_min: float = 17.0
    footprint_c_max: float = 22.0
    abnormal_low_c_max: float = 10.0
    abnormal_high_c_min: float = 60.0
    abnormal_high_c_max: float = 140.0
    abnormal_fire_c_min: float = 140.0

class ProcConfig(BaseModel):
    bg_alpha: float = 0.02
    morph_kernel: int = 3
    min_region_area: int = 8

class VizConfig(BaseModel):
    show: bool = True
    save_dir: str = "outputs"
    font_scale: float = 0.5
    thickness: int = 1

class AppConfig(BaseModel):
    io: IOConfig = IOConfig()
    thresholds: Thresholds = Thresholds()
    processing: ProcConfig = ProcConfig()
    viz: VizConfig = VizConfig()

    @staticmethod
    def load(path: str) -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)
