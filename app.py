from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

app = Flask(__name__)

# ==================================================
# AI MODEL - TFLite only
# ==================================================

MODEL_OK = False
MODEL_ERROR = None
MODEL_BACKEND = "tflite"
interpreter = None
input_details = None
output_details = None
scaler = None

try:
    scaler = joblib.load("modelAI/breath_scaler_v3.joblib")

    # File breath_v3.tflite co the dung Select TF Ops, nen cloud van can
    # package tensorflow de chay tf.lite.Interpreter. Khong load file .keras.
    interpreter = tf.lite.Interpreter(
        model_path="modelAI/breath_v3.tflite"
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    MODEL_OK = True

    print("=" * 50)
    print("AI MODEL LOADED SUCCESS (TFLite)")
    print(f"Input  : {input_details[0]['shape']}")
    print(f"Output : {output_details[0]['shape']}")
    print("=" * 50)

except Exception as e:
    MODEL_ERROR = f"{type(e).__name__}: {e}"

    print("=" * 50)
    print("AI LOAD FAILED")
    print(MODEL_ERROR)
    print("=" * 50)

# ==================================================
# CONFIG
# ==================================================

WINDOW_SIZE = 500   # 20s x 25Hz
FS = 25
N_FEATURES = 8      # ax,ay,az,gx,gy,gz,acc_mag,gyro_mag

# Luu 8 features: ax, ay, az, gx, gy, gz, acc_mag, gyro_mag
buffer = []

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
    # MPU6050 gui don vi g tu Arduino code. Chuan hoa vector de doc tu the.
    g_val = np.sqrt(ax**2 + ay**2 + az**2)

    if g_val <= 0:
        return "UNKNOWN"

    ax_n = ax / g_val
    ay_n = ay / g_val
    az_n = az / g_val

    threshold = 0.3

    if abs(az_n - 1.0) < threshold:
        return "NAM"
    if abs(az_n + 1.0) < threshold:
        return "NAM_NGUA"
    if abs(ay_n - 1.0) < threshold:
        return "NAM_NGHIENG"
    if abs(ax_n - 1.0) < threshold or abs(ax_n + 1.0) < threshold:
        return "DUNG"

    return "NGOI"

# ==================================================
# AI BPM
# ==================================================


def predict_bpm():
    if not MODEL_OK:
        return "AI_FAILED"

    if len(buffer) < WINDOW_SIZE:
        return f"BUFFER {len(buffer)}/{WINDOW_SIZE}"

    try:
        window = np.array(
            buffer[-WINDOW_SIZE:],
            dtype=np.float32
        )  # shape (500, 8)

        scaled = scaler.transform(window)
        scaled = scaled.reshape(1, WINDOW_SIZE, N_FEATURES).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], scaled)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]
        pred = float(np.clip(pred, 12, 20))

        return round(pred, 1)

    except Exception as e:
        print("AI ERROR:", type(e).__name__, e)
        return "AI_ERROR"

# ==================================================
# DATA INGESTION
# ==================================================


def process_sample(sample):
    ax = float(sample.get("ax", 0))
    ay = float(sample.get("ay", 0))
    az = float(sample.get("az", 0))
    gx = float(sample.get("gx", 0))
    gy = float(sample.get("gy", 0))
    gz = float(sample.get("gz", 0))

    acc_mag = float(np.sqrt(ax**2 + ay**2 + az**2))
    gyro_mag = float(np.sqrt(gx**2 + gy**2 + gz**2))

    feature_row = [ax, ay, az, gx, gy, gz, acc_mag, gyro_mag]
    buffer.append(feature_row)

    if len(buffer) > WINDOW_SIZE:
        del buffer[:-WINDOW_SIZE]

    return ax, ay, az, gx, gy, gz


def extract_samples(payload):
    if "batch" in payload:
        samples = payload.get("batch")

        if not isinstance(samples, list) or len(samples) == 0:
            raise ValueError("batch must be a non-empty list")

        return samples

    return [payload]

# ==================================================
# DASHBOARD
# ==================================================


@app.route("/")
def home():
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="1">
<title>Breath AI Cloud</title>
<style>
body {{
    font-family: Arial;
    background:#f2f2f2;
    text-align:center;
}}
.card {{
    width:800px;
    margin:auto;
    margin-top:30px;
    background:white;
    border-radius:10px;
    overflow:hidden;
}}
.header {{
    background:#188038;
    color:white;
    padding:20px;
}}
.section {{
    margin:20px;
    padding:20px;
    background:#fafafa;
    border-left:5px solid green;
    text-align:left;
}}
.big {{
    font-size:50px;
    color:green;
    font-weight:bold;
}}
.grid {{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:10px;
    margin:20px;
}}
.box {{
    background:#f7f7f7;
    padding:20px;
    border-radius:10px;
}}
.value {{
    font-size:28px;
    font-weight:bold;
}}
.footer {{
    padding:20px;
    color:gray;
}}
</style>
</head>
<body>
<div class="card">
<div class="header">
<h1>Breath AI Cloud</h1>
<p>Real-time Posture & Breath Monitor</p>
</div>
<div class="section">
<h3>TU THE HIEN TAI</h3>
<div class="big">{latest_data["posture"]}</div>
</div>
<div class="section">
<h3>NHIP THO AI</h3>
<div class="big">{latest_data["bpm"]}</div>
</div>
<div class="grid">
<div class="box">AX<div class="value">{latest_data["ax"]}</div></div>
<div class="box">AY<div class="value">{latest_data["ay"]}</div></div>
<div class="box">AZ<div class="value">{latest_data["az"]}</div></div>
<div class="box">GX<div class="value">{latest_data["gx"]}</div></div>
<div class="box">GY<div class="value">{latest_data["gy"]}</div></div>
<div class="box">GZ<div class="value">{latest_data["gz"]}</div></div>
</div>
<div class="footer">Last Update: {latest_data["time"]}</div>
</div>
</body>
</html>
"""

# ==================================================
# API ROUTES
# ==================================================


@app.route("/status")
def status():
    return jsonify({
        "server": "online",
        "model": MODEL_OK,
        "backend": MODEL_BACKEND,
        "model_error": MODEL_ERROR,
        "input_shape": (
            input_details[0]["shape"].tolist() if input_details else None
        ),
        "posture": latest_data["posture"],
        "bpm": latest_data["bpm"],
        "buffer": len(buffer)
    })


@app.route("/current")
def current():
    return jsonify(latest_data)


@app.route("/test")
def test():
    return jsonify({
        "success": True,
        "message": "Cloud Running",
        "model_loaded": MODEL_OK,
        "backend": MODEL_BACKEND,
        "model_error": MODEL_ERROR,
        "buffer_size": len(buffer),
        "window_size": WINDOW_SIZE,
        "features": N_FEATURES
    })


@app.route("/posture", methods=["POST"])
def posture():
    global latest_data

    try:
        payload = request.get_json(force=True)
        samples = extract_samples(payload)

        last_values = None
        for sample in samples:
            last_values = process_sample(sample)

        ax, ay, az, gx, gy, gz = last_values
        posture_result = detect_posture(ax, ay, az)
        bpm = predict_bpm()

        latest_data = {
            "posture": posture_result,
            "bpm": bpm,
            "ax": round(ax, 3),
            "ay": round(ay, 3),
            "az": round(az, 3),
            "gx": round(gx, 3),
            "gy": round(gy, 3),
            "gz": round(gz, 3),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print({
            **latest_data,
            "received": len(samples),
            "buffer": len(buffer)
        })

        return jsonify({
            "success": True,
            "posture": posture_result,
            "bpm": bpm,
            "buffer": len(buffer),
            "received": len(samples)
        })

    except Exception as e:
        print("POSTURE ERROR:", type(e).__name__, e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ==================================================
# MAIN
# ==================================================


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
