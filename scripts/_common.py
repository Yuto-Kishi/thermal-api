
import argparse
import os

import cv2 as cv

from thermal_api.config import AppConfig
from thermal_api.io.lepton_spi import LeptonSPISource
from thermal_api.io.purethermal_usb import PureThermalUSBSource
from thermal_api.io.simulator import SimulatorSource


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = AppConfig.load(args.config)
    os.makedirs(cfg.viz.save_dir, exist_ok=True)
    return cfg

def make_source(cfg: AppConfig):
    if cfg.io.driver == "simulator":
        return SimulatorSource(cfg.io.width, cfg.io.height, cfg.io.fps)
    elif cfg.io.driver == "lepton_spi":
        return LeptonSPISource(cfg.io.spi_bus, cfg.io.spi_device, cfg.io.width, cfg.io.height, cfg.io.fps)
    elif cfg.io.driver == "purethermal":
        return PureThermalUSBSource(cfg.io.usb_device, cfg.io.width, cfg.io.height, cfg.io.fps)
    else:
        raise ValueError(f"unknown io.driver: {cfg.io.driver}")

def imwrite_seq(save_dir: str, name: str, idx: int, img):
    cv.imwrite(os.path.join(save_dir, f"{name}_{idx:05d}.png"), img)
