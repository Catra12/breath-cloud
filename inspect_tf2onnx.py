import os
import sys

# Set protobuf implementation to python to avoid protobuf version errors
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tf2onnx.convert
    print("tf2onnx.convert imported successfully!")
    print("Attributes:")
    for attr in dir(tf2onnx.convert):
        if not attr.startswith("_"):
            print("  ", attr)
except Exception as e:
    import traceback
    print("Failed to import tf2onnx.convert:")
    traceback.print_exc()
