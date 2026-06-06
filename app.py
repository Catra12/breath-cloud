from flask import Flask, request, jsonify
from datetime import datetime

import numpy as np
import joblib

app = Flask(__name__)

# ==================================================
# AI MODEL — TFLite
# ==================================================

MODEL_OK = False

try:
    # Dùng TFLite thay vì tf.keras.models.load_model
    try:
        # Render dùng tflite-runtime (nhẹ hơn)
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(
            model_path="modelAI/breath_v3.tflite"
        )
    except ImportError:
        # Fallback: dùng tensorflow nếu có
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(
            model_path="modelAI/breath_v3.tflite"
        )

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    scaler = joblib.load(
        "modelAI/breath_scaler_v3.joblib"
    )

    MODEL_OK = True

    print("=" * 50)
    print("AI MODEL LOADED SUCCESS (TFLite)")
    print(f"Input  : {input_details[0]['shape']}")
    print(f"Output : {output_details[0]['shape']}")
    print("=" * 50)

except Exception as e:
    print("=" * 50)
    print("AI LOAD FAILED")
    print(e)
    print("=" * 50)

# ==================================================
# CONFIG
# ==================================================

WINDOW_SIZE = 500   # 20s × 25Hz
FS = 25
N_FEATURES = 9      # ax,ay,az,gx,gy,gz,acc_mag,gyro_mag,spectral_power

buffer = []         # lưu raw features (8 cột, chưa có spectral_power)

# ==================================================
# DATA
# ==================================================

latest_data = {
    "posture": "WAITING",
    "bpm": "WAITING",
    "ax": 0,
    "ay": 0,
    "az": 0,
    "gx": 0,
    "gy": 0,
    "gz": 0,
    "time": "-"
}

# ==================================================
# POSTURE
# ==================================================


def detect_posture(ax, ay, az):
    # MPU6050 gửi đơn vị m/s² → chuẩn hóa về g
    g_val = np.sqrt(ax**2 + ay**2 + az**2)
    ax_n = ax / g_val if g_val > 0 else ax
    ay_n = ay / g_val if g_val > 0 else ay
    az_n = az / g_val if g_val > 0 else az

    threshold = 0.3

    if abs(az_n - 1.0) < threshold:
        return "NAM"
    elif abs(az_n + 1.0) < threshold:
        return "NAM_NGUA"
    elif abs(ay_n - 1.0) < threshold:
        return "NAM_NGHIENG"
    elif abs(ax_n - 1.0) < threshold or abs(ax_n + 1.0) < threshold:
        return "DUNG"
    return "NGOI"

