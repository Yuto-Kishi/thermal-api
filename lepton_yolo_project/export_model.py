# -*- coding: utf-8 -*-

# ultralyticsライブラリからYOLOクラスをインポートします
from ultralytics import YOLO

# --------------------------------定数設定 (Constants)--------------------------------

# Raspberry Piで使用するONNXモデルのファイルパス
# This is the file path for the ONNX model that will be used on the Raspberry Pi.
MODEL_PATH = 'yolov8n.onnx'

# モデルのエクスポート時に指定する入力画像のサイズ (高さ, 幅)
# This is the input image size (height, width) specified when exporting the model.
# Leptonの解像度(60x80)を縦横4倍にしたサイズです。
# This size is 4 times the Lepton's resolution (60x80) in both height and width.
YOLO_INPUT_HEIGHT = 240
YOLO_INPUT_WIDTH = 320

# --------------------------------メイン処理 (Main Process)--------------------------------

def main():
    """
    YOLOv8nモデルをダウンロードし、Raspberry Piでの高速推論用に
    最適化されたONNX形式に変換するメイン関数です。
    This is the main function that downloads the YOLOv8n model and converts it
    to the optimized ONNX format for fast inference on the Raspberry Pi.
    """
    print("--- YOLOv8nモデルのエクスポートを開始します ---")
    print("Pre-trained model 'yolov8n.pt' をダウンロードまたはロードしています...")

    # 1. 事前学習済みのYOLOv8nモデルをロードします。
    #    .ptファイルが存在しない場合は、自動的にダウンロードされます。
    # 1. Load the pre-trained YOLOv8n model.
    #    If the .pt file does not exist, it will be downloaded automatically.
    model = YOLO('yolov8n.pt')
    print("モデルのロードが完了しました。")

    print("\nONNX形式への変換を開始します...")
    print(f"入力サイズ: {YOLO_INPUT_HEIGHT}x{YOLO_INPUT_WIDTH}")
    print(f"出力ファイル: {MODEL_PATH}")

    try:
        # 2. モデルをONNX形式にエクスポートします。
        #    imgsz: 推論時の入力画像サイズを指定します。
        #    opset: ONNXのバージョンセット。12はOpenCVのDNNモジュールとの互換性が高いです。
        #    simplify: ONNXモデルのグラフを最適化し、推論を高速化します。
        # 2. Export the model to ONNX format.
        #    imgsz: Specifies the input image size for inference.
        #    opset: ONNX version set. 12 is highly compatible with OpenCV's DNN module.
        #    simplify: Optimizes the ONNX model graph for faster inference.
        model.export(
            format='onnx',
            imgsz=[YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH],
            opset=12,
            simplify=True
        )

        print("\n-------------------------------------------")
        print("✅ モデルのエクスポートが正常に完了しました！")
        print(f"ファイル '{MODEL_PATH}' が生成されました。")
        print("このファイルをRaspberry Piに転送してください。")
        print("-------------------------------------------")

    except Exception as e:
        print("\n-------------------------------------------")
        print(f"❌ エラーが発生しました: {e}")
        print("インターネット接続やライブラリのインストールを確認してください。")
        print("-------------------------------------------")

if __name__ == '__main__':
    main()

