# -*- coding: utf-8 -*-
# This script downloads and converts the YOLOv5n model to the ONNX format.
# It should be run on a PC or Mac.

from ultralytics import YOLO

def main():
    """
    Downloads the YOLOv5n model and converts it to the ONNX format,
    which is optimized for high performance on devices like the Raspberry Pi.
    """
    print("--- Starting YOLOv5n model export ---")
    
    # Load the pre-trained YOLOv5n model.
    # The '.pt' file will be downloaded automatically on the first run.
    print("Loading pre-trained YOLOv5n model...")
    model = YOLO('yolov5n.pt')
    print("Model loaded successfully.")

    print("\nConverting model to ONNX format...")
    
    try:
        # Export the model to ONNX format.
        # opset=11 is highly compatible with various OpenCV versions.
        # imgsz specifies the input image size [height, width].
        model.export(
            format='onnx',
            imgsz=[240, 320],
            opset=11,
            simplify=True
        )
        print("\n-------------------------------------------")
        print("✅ Model export successful!")
        print("File 'yolov5n.onnx' has been created.")
        print("Please transfer this file to your Raspberry Pi.")
        print("-------------------------------------------")

    except Exception as e:
        print(f"\n❌ An error occurred during export: {e}")

if __name__ == '__main__':
    main()

