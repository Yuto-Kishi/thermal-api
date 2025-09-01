
# Thermal API Reproduction Kit (Fixed)

Raspberry Pi 3 Model B + FLIR Lepton 3.5 (Breakout) implementation with working simulator and SPI code.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

export PYTHONPATH="$PWD/src:$PWD/scripts:$PYTHONPATH"
python scripts/run_human_front.py -c configs/default.yaml
```
Switch to SPI device by editing `configs/default.yaml`:
```yaml
io:
  driver: lepton_spi
  spi_bus: 0
  spi_device: 0
```
