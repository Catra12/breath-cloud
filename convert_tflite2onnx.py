import os
import sys

# Set protobuf implementation to python to avoid protobuf version errors
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tflite2onnx
    print("tflite2onnx imported successfully")
except Exception as e:
    print("Failed to import tflite2onnx:", e)
    sys.exit(1)

tflite_path = "modelAI/breath_v3.tflite"
onnx_path = "modelAI/breath_v3.onnx"

print(f"Converting {tflite_path} to {onnx_path}...")
try:
    tflite2onnx.convert(tflite_path, onnx_path)
    print("Conversion completed successfully!")
except Exception as e:
    print("Error during conversion:", e)
    sys.exit(1)
