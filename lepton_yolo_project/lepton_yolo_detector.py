# -*- coding: utf-8 -*-
import spidev
import time
import numpy as np
import cv2

# --------------------------------定数設定 (Constants for Configuration)--------------------------------

# --- Leptonカメラ設定 (Lepton Camera Settings) ---
WIDTH, HEIGHT = 80, 60  # Lepton 1.6/2.5の解像度 (Resolution for Lepton 1.6/2.5)
PKT_SIZE = 164          # 1パケットのバイトサイズ (Byte size of one packet)

# --- 表示設定 (Display Settings) ---
DISPLAY_SCALE_FACTOR = 10 # 表示ウィンドウの拡大率 (Magnification factor for the display window)

# --- YOLOv8 AIモデル設定 (YOLOv8 AI Model Settings) ---
MODEL_PATH = 'yolov8n.onnx'          # Macから転送したONNXモデルのパス (Path to the ONNX model transferred from Mac)
YOLO_INPUT_WIDTH = 320               # YOLOモデルの入力画像の幅 (Input image width for the YOLO model)
YOLO_INPUT_HEIGHT = 240              # YOLOモデルの入力画像の高さ (Input image height for the YOLO model)
CONFIDENCE_THRESHOLD = 0.40          # 物体検出の信頼度の閾値 (Confidence threshold for object detection)
NMS_THRESHOLD = 0.45                 # Non-Maximum Suppressionの閾値 (Threshold for Non-Maximum Suppression)
PERSON_CLASS_ID = 0                  # YOLOのクラスIDにおける'person'のID (Class ID for 'person' in YOLO)

# --- ハイブリッド検証設定 (Hybrid Verification Settings) ---
# YOLOが人を検出した後、その領域内にこの温度以上のピクセルが一定数あるかを確認する
# After YOLO detects a person, verify if there are a certain number of pixels above this temperature in that area.
THERMAL_VERIFICATION_THRESHOLD_C = 29.0 # 人の肌温度と見なす摂氏温度の閾値 (Temperature threshold in Celsius to be considered as human skin)
REQUIRED_HOTSPOT_PIXEL_COUNT = 3        # 上記の温度を持つべき最低ピクセル数 (Minimum number of pixels required to have the above temperature)

# --------------------------------Leptonカメラ通信 (Lepton Camera Communication)--------------------------------

def read_frame(spi):
    """
    SPI通信を介してLeptonカメラから1フレーム分の生データを読み取る関数
    Function to read one frame of raw data from the Lepton camera via SPI communication.
    """
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    lines_read = 0
    while lines_read < HEIGHT:
        try:
            pkt = spi.readbytes(PKT_SIZE)
            if len(pkt) != PKT_SIZE:
                continue

            # 同期パケットは無視する (Ignore synchronization packets)
            if (pkt[0] & 0x0F) == 0x0F:
                continue

            line_num = pkt[1]
            if line_num < HEIGHT:
                # ビッグエンディアンの16ビットデータを読み込み、フレームに格納
                # Read 16-bit big-endian data and store it in the frame
                data = np.frombuffer(bytearray(pkt[4:]), dtype=">u2")
                if data.size == WIDTH:
                    frame[line_num] = data
                    lines_read += 1
        except Exception as e:
            # SPI通信エラーが発生した場合 (In case of an SPI communication error)
            print(f"SPI read error: {e}")
            time.sleep(0.1) # 少し待ってから再試行 (Wait a bit before retrying)
            return None, False # エラーを示すタプルを返す
            
    return frame, True

# --------------------------------画像処理 (Image Processing)--------------------------------

def normalize_and_colorize(raw_frame):
    """
    生の16ビット温度データを、表示とAI推論に適した8ビットのカラー画像に変換する関数
    Function to convert raw 16-bit temperature data into an 8-bit color image suitable for display and AI inference.
    """
    i_min, i_max = np.min(raw_frame), np.max(raw_frame)
    if i_max == i_min:
        # 画像が真っ黒などの場合 (If the image is completely black, etc.)
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # データを0-255の範囲に正規化 (Normalize data to the 0-255 range)
    normalized = ((raw_frame - i_min) * 255.0 / (i_max - i_min)).astype(np.uint8)
    
    # カラーマップを適用して視覚的に分かりやすくする (Apply a colormap for better visualization)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return colored

# --------------------------------メイン処理 (Main Process)--------------------------------

def main():
    """
    メインの実行関数。カメラの初期化、AIモデルの読み込み、リアルタイムでの人物検知を行う。
    Main execution function. Initializes the camera, loads the AI model, and performs real-time person detection.
    """
    # --- SPI初期化 (Initialize SPI) ---
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 18000000 # 18MHz
    
    # --- AIモデルの読み込み (Load AI Model) ---
    try:
        print(f"AIモデル '{MODEL_PATH}' を読み込んでいます...")
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        print("✅ モデルの読み込みに成功しました。検知を開始します。")
    except Exception as e:
        print(f"❌ モデルの読み込みに失敗しました: {e}")
        print("'{MODEL_PATH}'がこのスクリプトと同じディレクトリにあることを確認してください。")
        spi.close()
        return

    # --- メインループ (Main Loop) ---
    while True:
        # フレームを読み込む (Read a frame)
        raw_frame, success = read_frame(spi)
        if not success or raw_frame is None:
            continue

        # 温度データ（ケルビン）を摂氏に変換 (Convert temperature data (Kelvin) to Celsius)
        temp_c = (raw_frame / 100.0) - 273.15

        # 表示用のカラー画像を作成 (Create a color image for display)
        color_image = normalize_and_colorize(raw_frame)
        
        # 表示用に画像を拡大 (Scale up the image for display)
        display_image = cv2.resize(color_image, 
                                   (WIDTH * DISPLAY_SCALE_FACTOR, HEIGHT * DISPLAY_SCALE_FACTOR), 
                                   interpolation=cv2.INTER_NEAREST)

        # --- AI推論の準備 (Prepare for AI Inference) ---
        blob = cv2.dnn.blobFromImage(color_image, 1/255.0, (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        
        # --- AI推論の実行 (Execute AI Inference) ---
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        
        # --- 検出結果の処理 (Process Detection Results) ---
        boxes, confidences, class_ids = [], [], []
        
        for output in outputs:
            for detection in output[0]:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                    # バウンディングボックスの座標を計算
                    # Calculate bounding box coordinates
                    center_x, center_y, w, h = detection[0:4] * np.array([YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maximum Suppressionを適用して重複するボックスを削除
        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                # --- ハイブリッド検証 (Hybrid Verification) ---
                # 元のLepton解像度でのバウンディングボックス座標に変換
                # Convert bounding box coordinates to the original Lepton resolution
                orig_x = int(x / (YOLO_INPUT_WIDTH / WIDTH))
                orig_y = int(y / (YOLO_INPUT_HEIGHT / HEIGHT))
                orig_w = int(w / (YOLO_INPUT_WIDTH / WIDTH))
                orig_h = int(h / (YOLO_INPUT_HEIGHT / HEIGHT))

                # 範囲外アクセスを防ぐ (Prevent out-of-bounds access)
                orig_x = max(0, orig_x)
                orig_y = max(0, orig_y)
                
                # 検出領域内の温度データを抽出 (Extract temperature data within the detected region)
                roi_temp = temp_c[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]

                if roi_temp.size > 0:
                    # ホットスポット（人の肌）があるか検証 (Verify if a hotspot (human skin) exists)
                    hotspot_pixel_count = np.sum(roi_temp > THERMAL_VERIFICATION_THRESHOLD_C)
                    
                    if hotspot_pixel_count >= REQUIRED_HOTSPOT_PIXEL_COUNT:
                        # 検証成功！表示用画像に描画 (Verification successful! Draw on the display image)
                        disp_x = x * (DISPLAY_SCALE_FACTOR * WIDTH / YOLO_INPUT_WIDTH)
                        disp_y = y * (DISPLAY_SCALE_FACTOR * HEIGHT / YOLO_INPUT_HEIGHT)
                        disp_w = w * (DISPLAY_SCALE_FACTOR * WIDTH / YOLO_INPUT_WIDTH)
                        disp_h = h * (DISPLAY_SCALE_FACTOR * HEIGHT / YOLO_INPUT_HEIGHT)

                        cv2.rectangle(display_image, (int(disp_x), int(disp_y)), (int(disp_x + disp_w), int(disp_y + disp_h)), (0, 255, 0), 2)
                        cv2.putText(display_image, "Person", (int(disp_x), int(disp_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # --- 結果の表示 (Display the result) ---
        cv2.imshow("FLIR Lepton YOLOv8 Person Detector", display_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # --- クリーンアップ (Cleanup) ---
    print("\n終了します。")
    spi.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
