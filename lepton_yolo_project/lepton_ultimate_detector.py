# -*- coding: utf-8 -*-
import spidev
import time
import numpy as np
import cv2
import os

# --------------------------------定数設定 (Constants)--------------------------------
# Leptonカメラ設定
WIDTH, HEIGHT = 80, 60
PKT_SIZE = 164

# 表示設定
DISPLAY_SCALE_FACTOR = 10

# MobileNet-SSD AIモデル設定
MODEL_GRAPH_PATH = 'frozen_inference_graph.pb'
MODEL_CONFIG_PATH = 'ssd_mobilenet_v2_coco.pbtxt'
CONFIDENCE_THRESHOLD = 0.55
PERSON_CLASS_ID = 1

# ハイブリッド検証設定
THERMAL_VERIFICATION_THRESHOLD_C = 29.0
REQUIRED_HOTSPOT_PIXEL_COUNT = 3

# --- AIモデル設定データ (直接埋め込み) ---
MODEL_CONFIG_CONTENT = b'node {\n  name: "FeatureExtractor/MobilenetV2/Conv/weights"\n  op: "Const"\n  attr {\n    key: "dtype"\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: "value"\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 32\n          }\n        }\n        tensor_content: "\\3... (very long byte string)'

# --- AIモデル本体データ (直接埋め込み) ---
MODEL_GRAPH_CONTENT = b'\n\x96\x03\n\x08frozen_inference_graph\x12\x8b\x03\n\x04name\x1a\x06...... (very very long byte string)'

# --------------------------------関数定義 (Functions)--------------------------------
def create_model_files_from_embedded_data():
    """埋め込みバイトデータからモデルと設定ファイルを生成する"""
    files_ok = True
    
    if not os.path.exists(MODEL_CONFIG_PATH):
        print(f"設定ファイル '{MODEL_CONFIG_PATH}' を生成します...")
        try:
            with open(MODEL_CONFIG_PATH, 'wb') as f:
                f.write(MODEL_CONFIG_CONTENT)
            print("✅ 設定ファイルの生成に成功しました。")
        except Exception as e:
            print(f"❌ 設定ファイルの生成に失敗しました: {e}")
            files_ok = False

    if not os.path.exists(MODEL_GRAPH_PATH):
        print(f"モデルファイル '{MODEL_GRAPH_PATH}' を生成します...")
        try:
            with open(MODEL_GRAPH_PATH, 'wb') as f:
                f.write(MODEL_GRAPH_CONTENT)
            print("✅ モデルファイルの生成に成功しました。")
        except Exception as e:
            print(f"❌ モデルファイルの生成に失敗しました: {e}")
            files_ok = False
            
    return files_ok

def read_frame(spi):
    """Leptonカメラから1フレームを読み取る"""
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    lines_read = 0
    while lines_read < HEIGHT:
        try:
            pkt = spi.readbytes(PKT_SIZE)
            if len(pkt) != PKT_SIZE or (pkt[0] & 0x0F) == 0x0F: continue
            line_num = pkt[1]
            if line_num < HEIGHT:
                data = np.frombuffer(bytearray(pkt[4:]), dtype=">u2")
                if data.size == WIDTH:
                    frame[line_num] = data
                    lines_read += 1
        except Exception:
            time.sleep(0.01)
            return None, False
    return frame, True

def normalize_and_colorize(raw_frame):
    """生データをカラー画像に変換"""
    i_min, i_max = np.min(raw_frame), np.max(raw_frame)
    if i_max == i_min: return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    normalized = ((raw_frame - i_min) * 255.0 / (i_max - i_min)).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

# --------------------------------メイン処理 (Main Process)--------------------------------
def main():
    if not create_model_files_from_embedded_data():
        print("必要なファイルの準備に失敗したため、プログラムを終了します。")
        return

    # --- SPI初期化 (Typoの修正) ---
    spi = spidev.SpiDev() # 'Sdev' から 'SpiDev' に修正しました
    spi.open(0, 0)
    spi.max_speed_hz = 18000000

    try:
        print(f"AIモデル '{MODEL_GRAPH_PATH}' を読み込んでいます...")
        net = cv2.dnn.readNetFromTensorflow(MODEL_GRAPH_PATH, MODEL_CONFIG_PATH)
        print("✅ モデルの読み込みに成功しました。")
    except Exception as e:
        print(f"❌ モデルの読み込みに失敗しました: {e}")
        spi.close()
        return

    print("検知を開始します... ('q'キーで終了)")
    while True:
        raw_frame, success = read_frame(spi)
        if not success or raw_frame is None: continue

        temp_c = (raw_frame / 100.0) - 273.15
        color_image = normalize_and_colorize(raw_frame)
        
        net.setInput(cv2.dnn.blobFromImage(color_image, size=(300, 300), swapRB=True, crop=False))
        detections = net.forward()
        
        display_image = cv2.resize(color_image, (WIDTH * DISPLAY_SCALE_FACTOR, HEIGHT * DISPLAY_SCALE_FACTOR), interpolation=cv2.INTER_NEAREST)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                box_x = int(detections[0, 0, i, 3] * WIDTH)
                box_y = int(detections[0, 0, i, 4] * HEIGHT)
                box_w = int(detections[0, 0, i, 5] * WIDTH) - box_x
                box_h = int(detections[0, 0, i, 6] * HEIGHT) - box_y
                
                box_x, box_y = max(0, box_x), max(0, box_y)

                roi_temp = temp_c[box_y:box_y+box_h, box_x:box_x+box_w]

                if roi_temp.size > 0:
                    hotspot_pixel_count = np.sum(roi_temp > THERMAL_VERIFICATION_THRESHOLD_C)
                    if hotspot_pixel_count >= REQUIRED_HOTSPOT_PIXEL_COUNT:
                        disp_x, disp_y = box_x * DISPLAY_SCALE_FACTOR, box_y * DISPLAY_SCALE_FACTOR
                        disp_w, disp_h = box_w * DISPLAY_SCALE_FACTOR, box_h * DISPLAY_SCALE_FACTOR
                        
                        cv2.rectangle(display_image, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
                        cv2.putText(display_image, "Person", (disp_x, disp_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Lepton Detector - Final Fix", display_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    spi.close()
    cv2.destroyAllWindows()
    print("\nプログラムを終了しました。")

if __name__ == '__main__':
    main()

