"""
Lepton SPI capture (auto-detect 80x60 vs 160x120).

- 80x60: Lepton 1.x/2.x 系（1フレーム=60ライン, パケット=164B, セグメントなし）
- 160x120: Lepton 3.x 系（1フレーム=4セグメント×60ライン, パケット=164B）

既定は:
  - SPI mode = 0   （あなたの動作実績に合わせる）
  - SPI speed = 20MHz
必要に応じて configs から fps/width/height を渡す。

依存: sudo apt install -y python3-spidev
"""

from typing import Iterator, Tuple, Optional
import time
import numpy as np
import spidev

from .base import ThermalSource
from ..types import ThermalFrame
from ..processing.common import to_gray8

# 共通（1パケット = 4Bヘッダ + 160B payload = 164B）
_HDR_BYTES = 4
_LINE_PIXELS = 80               # 16bit words → 80 words = 160 bytes
_PAYLOAD_BYTES = _LINE_PIXELS * 2
_PACKET_BYTES = _HDR_BYTES + _PAYLOAD_BYTES

# 80x60（Lepton 1/2）
L2_WIDTH  = 80
L2_HEIGHT = 60
L2_LINES_PER_FRAME = 60

# 160x120（Lepton 3）
L3_WIDTH  = 160
L3_HEIGHT = 120
L3_SEGMENTS = 4
L3_LINES_PER_SEG = 60

_MAX_SYNC_TRIES = 4000

def _parse_header(h0: int, h1: int) -> Tuple[int, int]:
    """
    VoSPI header: 2 bytes -> (segment_id, line_id)
    - 80x60: segment_id は通常 0、line_id は 0..59
    - 160x120: segment_id は上位ニブル (1..4) が入る個体あり、line_id は 0..59
    - 0x0Fxx: リセット/ダミー
    """
    seg_id = (h0 >> 4) & 0x0F
    line_id = ((h0 & 0x0F) << 8) | h1
    return seg_id, line_id

class LeptonSPISource(ThermalSource):
    def __init__(
        self,
        bus: int = 0,
        device: int = 0,
        width: int = L3_WIDTH,
        height: int = L3_HEIGHT,
        fps: float = 8.7,
        speed_hz: int = 20_000_000,
        mode: int = 0,
    ):
        # configのwidth/heightは目安。実際は自動判別で上書きする
        self.W_hint, self.H_hint = width, height
        self.period = 1.0 / float(fps)

        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed_hz
        self.spi.mode = mode

        # 自動判別
        self.is_l2, self.W, self.H = self._detect_mode()

    def _raw_packet(self) -> bytearray:
        # 安定実績のある readbytes を既定に（xfer2 も可）
        return bytearray(self.spi.readbytes(_PACKET_BYTES))

    def _detect_mode(self) -> Tuple[bool, int, int]:
        """
        何十パケットか見て、80x60 か 160x120 かを判定する。
        0x0Fxx を捨てつつ、line_id の最大値や seg_id の出現で判断。
        """
        seen_seg = set()
        seen_lines = set()
        tries = 0
        while tries < _MAX_SYNC_TRIES:
            raw = self._raw_packet()
            h0, h1 = raw[0], raw[1]
            seg_id, line_id = _parse_header(h0, h1)

            # 0x0Fxx（下位ニブル=0x0F）はダミー
            if (h0 & 0x0F) == 0x0F:
                tries += 1
                continue

            if line_id < 1024:
                seen_lines.add(line_id)
            if seg_id in (1, 2, 3, 4):
                seen_seg.add(seg_id)

            # 判定条件：
            # - 160x120: seg 1..4 が観測される or line_id が 60未満で安定しつつ seg 出る
            # - 80x60  : seg が一切出ず、line_id が 0..59 で回っている
            if len(seen_seg) >= 2:
                return False, L3_WIDTH, L3_HEIGHT
            if tries > 200 and len(seen_lines) >= 40 and len(seen_seg) == 0:
                # seg見えず、lineが0..59帯 → 80x60 の可能性が高い
                return True, L2_WIDTH, L2_HEIGHT
            tries += 1

        # フォールバック：ユーザ指定を優先、なければ 80x60 を優先（あなたの実機に合わせる）
        if self.W_hint == L3_WIDTH and self.H_hint == L3_HEIGHT:
            # 指定が160x120ならそれに従う
            return False, L3_WIDTH, L3_HEIGHT
        return True, L2_WIDTH, L2_HEIGHT

    # --- 80x60 読み ---
    def _read_frame_l2(self) -> Optional[np.ndarray]:
        frame = np.zeros((L2_HEIGHT, L2_WIDTH), np.uint16)
        filled = 0
        tries = 0
        while filled < L2_LINES_PER_FRAME and tries < _MAX_SYNC_TRIES:
            raw = self._raw_packet()
            h0, h1 = raw[0], raw[1]
            seg_id, line_id = _parse_header(h0, h1)
            tries += 1

            # リセット/ダミー
            if (h0 & 0x0F) == 0x0F:
                continue
            if line_id >= L2_LINES_PER_FRAME:
                continue

            payload = raw[_HDR_BYTES:]
            if len(payload) != _PAYLOAD_BYTES:
                continue
            words = np.frombuffer(payload, dtype=">u2")
            if words.size != _LINE_PIXELS:
                continue

            frame[line_id, :] = words
            filled += 1

        return frame if filled == L2_LINES_PER_FRAME else None

    # --- 160x120 読み ---
    def _read_frame_l3(self) -> Optional[np.ndarray]:
        frame = np.zeros((L3_HEIGHT, L3_WIDTH), np.uint16)
        seg_filled = [0, 0, 0, 0]  # 各セグメントの埋まり数
        # seg偶奇で上下 60 行にマップする
        while sum(seg_filled) < L3_SEGMENTS * L3_LINES_PER_SEG:
            raw = self._raw_packet()
            h0, h1 = raw[0], raw[1]
            seg_id, line_id = _parse_header(h0, h1)

            # 0x0Fxx は捨てる
            if (h0 & 0x0F) == 0x0F:
                continue
            if line_id >= L3_LINES_PER_SEG:
                continue

            payload = raw[_HDR_BYTES:]
            if len(payload) != _PAYLOAD_BYTES:
                continue
            words = np.frombuffer(payload, dtype=">u2")
            if words.size != _LINE_PIXELS:
                continue

            # 80 words → 160 pixels（左右並べ）
            # Lepton3 は各パケット80ワード×2列で160pxを構成する
            # 多くの実装では1ラインに80 words×2segで160pxを得るが、
            # ここでは「偶奇パリティで上下 60 行に配置」し、後段で最近傍拡大する簡易法にする。
            seg_parity = ((seg_id - 1) & 1) if seg_id in (1, 2, 3, 4) else 0
            y = line_id + seg_parity * L3_LINES_PER_SEG
            if 0 <= y < L3_HEIGHT:
                # とりあえず words を 2 倍に並べて 160px に埋める（簡易）
                line = np.repeat(words, 2)[:L3_WIDTH]
                frame[y, :] = line
                seg_filled[seg_parity * 2] += 1

        return frame

    def frames(self) -> Iterator[ThermalFrame]:
        try:
            while True:
                t0 = time.perf_counter()

                if self.is_l2:
                    raw_u16 = self._read_frame_l2()
                    if raw_u16 is None:
                        continue
                    temp_k = raw_u16.astype(np.float32) / 100.0
                    temp_c = temp_k - 273.15
                    # 80x60 → 160x120 に最近傍拡大（処理系は160x120想定のため）
                    temp_c = np.repeat(np.repeat(temp_c, 2, axis=0), 2, axis=1)
                else:
                    raw_u16 = self._read_frame_l3()
                    if raw_u16 is None:
                        continue
                    temp_k = raw_u16.astype(np.float32) / 100.0
                    temp_c = temp_k - 273.15

                gray8 = to_gray8(temp_c)
                yield ThermalFrame(temp_c=temp_c, gray8=gray8, timestamp=time.time())

                dt = time.perf_counter() - t0
                rem = self.period - dt
                if rem > 0:
                    time.sleep(rem)

        finally:
            try:
                self.spi.close()
            except Exception:
                pass
