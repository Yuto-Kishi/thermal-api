# -*- coding: utf-8 -*-
# This program runs on a Raspberry Pi to perform real-time person detection
# using a Lepton thermal camera and a YOLOv5n ONNX model.

import spidev
import time
import numpy as np
import cv2

# --- Constants ---
# Lepton Camera Settings
WIDTH, HEIGHT = 80, 60
PKT_SIZE = 164

# Display Settings
DISPLAY_SCALE_FACTOR = 10

# YOLOv5n AI Model Settings
MODEL_PATH = 'yolov5n.onnx' # This file must be in the same directory
YOLO_INPUT_WIDTH = 320
YOLO_INPUT_HEIGHT = 240
CONFIDENCE_THRESHOLD = 0.40 # AIの信頼度の閾値 (AI confidence threshold)
NMS_THRESHOLD = 0.45      # 重複ボックスを除去する際の閾値 (Threshold for removing overlapping boxes)
PERSON_CLASS_ID = 0         # YOLOが学習した「人物」クラスのID (ID for the 'person' class in YOLO)

# Hybrid Thermal Verification (誤検知を減らすための追加検証)
# (An additional verification to reduce false positives)
THERMAL_VERIFICATION_THRESHOLD_C = 29.0 # 人の肌として認識する最低温度 (Minimum temperature to be considered human skin)
REQUIRED_HOTSPOT_PIXEL_COUNT = 3       # 最低限必要な高温ピクセル数 (Minimum number of required hot pixels)

# --- Functions ---
def read_frame(spi):
    """Reads a complete frame from the Lepton camera via SPI."""
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    lines_read = 0
    while lines_read < HEIGHT:
        try:
            pkt = spi.readbytes(PKT_SIZE)
            if len(pkt) != PKT_SIZE or (pkt[0] & 0x0F) == 0x0F:
                continue
            line_num = pkt[1]
            if line_num < HEIGHT:
                data = np.frombuffer(bytearray(pkt[4:]), dtype=">u2")
                if data.size == WIDTH:
                    frame[line_num] = data
                    lines_read += 1
        except Exception as e:
            print(f"SPI read error: {e}")
            time.sleep(0.01)
            return None, False
    return frame, True

def normalize_and_colorize(raw_frame):
    """Normalizes the raw thermal data and applies a color map."""
    i_min, i_max = np.min(raw_frame), np.max(raw_frame)
    if i_max == i_min:
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    normalized = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

# --- Main Execution ---
def main():
    """Main function to run the person detection loop."""
    # --- Initialize SPI for Lepton Camera ---
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 18000000

    # --- Load YOLOv5n AI Model ---
    try:
        print(f"AIモデル '{MODEL_PATH}' を読み込んでいます...")
        # Loading AI model...
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        print("✅ モデルの読み込みに成功しました。")
        # Model loaded successfully.
    except Exception as e:
        print(f"❌ モデルの読み込みに失敗しました: {e}")
        # Failed to load model.
        print(f"'{MODEL_PATH}' がこのスクリプトと同じディレクトリにあることを確認してください。")
        # Ensure '{MODEL_PATH}' is in the same directory as this script.
        spi.close()
        return

    print("検知を開始します... ('q'キーで終了)")
    # Starting detection... (Press 'q' to quit)
    while True:
        raw_frame, success = read_frame(spi)
        if not success or raw_frame is None:
            continue

        # Convert raw data to Celsius for thermal verification
        temp_c = (raw_frame / 100.0) - 273.15
        
        # Prepare image for display and AI model
        color_image = normalize_and_colorize(raw_frame)
        
        # --- AI Inference ---
        blob = cv2.dnn.blobFromImage(color_image, 1/255.0, (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        
        # --- Process Detections ---
        boxes, confidences, class_ids = [], [], []
        rows = outputs[0].shape[0]

        for i in range(rows):
            row = outputs[0][i]
            confidence = row[4]
            if confidence > CONFIDENCE_THRESHOLD:
                scores = row[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > 0.25 and class_id == PERSON_CLASS_ID:
                    center_x, center_y, w, h = row[0:4]
                    x = int((center_x - w / 2) * WIDTH)
                    y = int((center_y - h / 2) * HEIGHT)
                    width = int(w * WIDTH)
                    height = int(h * HEIGHT)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        # Prepare display image
        display_image = cv2.resize(color_image, (WIDTH * DISPLAY_SCALE_FACTOR, HEIGHT * DISPLAY_SCALE_FACTOR), interpolation=cv2.INTER_NEAREST)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # --- Hybrid Thermal Verification ---
                roi_temp = temp_c[max(0, y):y+h, max(0, x):x+w]
                if roi_temp.size > 0:
                    hotspot_pixel_count = np.sum(roi_temp > THERMAL_VERIFICATION_THRESHOLD_C)
                    if hotspot_pixel_count >= REQUIRED_HOTSPOT_PIXEL_COUNT:
                        # If verified, draw the box on the display image
                        disp_x = x * DISPLAY_SCALE_FACTOR
                        disp_y = y * DISPLAY_SCALE_FACTOR
                        disp_w = w * DISPLAY_SCALE_FACTOR
                        disp_h = h * DISPLAY_SCALE_FACTOR
                        cv2.rectangle(display_image, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (0, 255, 0), 2)
                        label = f"Person: {confidences[i]:.2f}"
                        cv2.putText(display_image, label, (disp_x, disp_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Display Result ---
        cv2.imshow("Lepton YOLOv5 Person Detector (Press 'q' to quit)", display_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    spi.close()
    cv2.destroyAllWindows()
    print("\nプログラムを終了しました。")
    # Program terminated.

if __name__ == '__main__':
    main()