from datetime import datetime
import time

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
DATA_TIMEOUT_SECONDS = 5
MAX_GYRO_ABS = 20.0
GYRO_DPS_AUTO_CONVERT_ABS = 20.0
GYRO_DPS_MAX_REASONABLE = 500.0
MIN_ACC_MAG_STD = 0.003
MIN_GYRO_MAG_STD = 0.0005
MIN_STATIC_AXIS_RANGE = 0.012
MIN_CLEAR_AXIS_RANGE = 0.025
MIN_BREATH_BAND_RATIO = 0.25
BREATH_FREQ_MIN = 0.10
BREATH_FREQ_MAX = 0.55

# Luu 8 features: ax, ay, az, gx, gy, gz, acc_mag, gyro_mag
buffer = []

# ==================================================
# DATA
# ==================================================

DEFAULT_DATA = {
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

latest_data = DEFAULT_DATA.copy()
last_data_at = None

# ==================================================
# POSTURE
# ==================================================


def detect_posture(ax, ay, az):
    g = 1.0
    threshold = 0.2

    if (
        abs(az - g) < threshold
        and abs(ax) < 0.4
        and abs(ay) < 0.4
    ):
        return "NAM"

    if (
        abs(ay - g) < threshold
        and abs(ax) < 0.4
        and abs(az) < 0.4
    ):
        return "NAM_NGHIENG"

    if (
        abs(ax - g) < threshold
        and abs(ay) < 0.4
        and abs(az) < 0.4
    ):
        return "DUNG"

    return "NGOI"

# ==================================================
# AI BPM
# ==================================================


def analyze_breath_signal(window):
    gyro_abs_max = float(np.max(np.abs(window[:, 3:6])))
    if gyro_abs_max > MAX_GYRO_ABS:
        return False, "SENSOR_ERROR", {
            "gyro_abs_max": round(gyro_abs_max, 4)
        }

    gyro_mag_std = float(np.std(window[:, 7]))
    candidate_channels = {
        "ax": window[:, 0],
        "ay": window[:, 1],
        "az": window[:, 2],
        "acc_mag": window[:, 6],
    }

    best = {
        "channel": None,
        "range": 0.0,
        "band_ratio": 0.0,
        "dominant_bpm": None,
    }

    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / FS)
    breath_mask = (freqs >= BREATH_FREQ_MIN) & (freqs <= BREATH_FREQ_MAX)
    useful_mask = (freqs >= 0.03) & (freqs <= 2.0)

    for name, values in candidate_channels.items():
        values = np.asarray(values, dtype=np.float32)
        axis_range = float(np.percentile(values, 95) - np.percentile(values, 5))

        signal = values - np.mean(values)
        signal = signal * np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(signal)) ** 2

        useful_power = float(np.sum(spectrum[useful_mask]))
        breath_power = float(np.sum(spectrum[breath_mask]))
        band_ratio = breath_power / (useful_power + 1e-12)

        dominant_bpm = None
        if np.any(breath_mask) and breath_power > 0:
            breath_freqs = freqs[breath_mask]
            breath_spectrum = spectrum[breath_mask]
            dominant_bpm = float(breath_freqs[np.argmax(breath_spectrum)] * 60.0)

        if band_ratio > best["band_ratio"] or axis_range > best["range"]:
            best = {
                "channel": name,
                "range": axis_range,
                "band_ratio": band_ratio,
                "dominant_bpm": dominant_bpm,
            }

    if best["range"] < MIN_STATIC_AXIS_RANGE and gyro_mag_std < MIN_GYRO_MAG_STD:
        return False, "NO_BREATH", {
            "reason": "static_signal",
            "axis_range": round(best["range"], 5),
            "gyro_mag_std": round(gyro_mag_std, 5),
            "band_ratio": round(best["band_ratio"], 4),
            "dominant_bpm": best["dominant_bpm"],
        }

    if (
        best["range"] < MIN_CLEAR_AXIS_RANGE
        and best["band_ratio"] < MIN_BREATH_BAND_RATIO
    ):
        return False, "NO_BREATH", {
            "reason": "unclear_breath_band",
            "axis_range": round(best["range"], 5),
            "gyro_mag_std": round(gyro_mag_std, 5),
            "band_ratio": round(best["band_ratio"], 4),
            "dominant_bpm": best["dominant_bpm"],
        }

    return True, "OK", {
        "channel": best["channel"],
        "axis_range": round(best["range"], 5),
        "gyro_mag_std": round(gyro_mag_std, 5),
        "band_ratio": round(best["band_ratio"], 4),
        "dominant_bpm": (
            round(best["dominant_bpm"], 2)
            if best["dominant_bpm"] is not None
            else None
        ),
    }


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

        signal_ok, signal_status, signal_info = analyze_breath_signal(window)
        if not signal_ok:
            print("SIGNAL CHECK:", signal_status, signal_info)
            return signal_status

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


def reset_runtime_state():
    global latest_data, last_data_at

    buffer.clear()
    latest_data = DEFAULT_DATA.copy()
    last_data_at = None


def get_live_data():
    if last_data_at is None:
        return latest_data

    if time.monotonic() - last_data_at > DATA_TIMEOUT_SECONDS:
        reset_runtime_state()

    return latest_data


def process_sample(sample):
    ax = float(sample.get("ax", 0))
    ay = float(sample.get("ay", 0))
    az = float(sample.get("az", 0))
    gx = float(sample.get("gx", 0))
    gy = float(sample.get("gy", 0))
    gz = float(sample.get("gz", 0))

    gyro_abs_max = max(abs(gx), abs(gy), abs(gz))
    if GYRO_DPS_AUTO_CONVERT_ABS < gyro_abs_max <= GYRO_DPS_MAX_REASONABLE:
        gx, gy, gz = np.deg2rad([gx, gy, gz])

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
    current_data = get_live_data()

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
<div class="big">{current_data["posture"]}</div>
</div>
<div class="section">
<h3>NHIP THO AI</h3>
<div class="big">{current_data["bpm"]}</div>
</div>
<div class="grid">
<div class="box">AX<div class="value">{current_data["ax"]}</div></div>
<div class="box">AY<div class="value">{current_data["ay"]}</div></div>
<div class="box">AZ<div class="value">{current_data["az"]}</div></div>
<div class="box">GX<div class="value">{current_data["gx"]}</div></div>
<div class="box">GY<div class="value">{current_data["gy"]}</div></div>
<div class="box">GZ<div class="value">{current_data["gz"]}</div></div>
</div>
<div class="footer">Last Update: {current_data["time"]}</div>
</div>
</body>
</html>
"""

# ==================================================
# API ROUTES
# ==================================================


@app.route("/status")
def status():
    current_data = get_live_data()

    return jsonify({
        "server": "online",
        "model": MODEL_OK,
        "backend": MODEL_BACKEND,
        "model_error": MODEL_ERROR,
        "input_shape": (
            input_details[0]["shape"].tolist() if input_details else None
        ),
        "posture": current_data["posture"],
        "bpm": current_data["bpm"],
        "buffer": len(buffer)
    })


@app.route("/current")
def current():
    return jsonify(get_live_data())


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


@app.route("/reset", methods=["GET", "POST"])
def reset():
    reset_runtime_state()

    return jsonify({
        "success": True,
        "message": "Runtime state reset",
        "buffer": len(buffer),
        "data": latest_data
    })


@app.route("/posture", methods=["POST"])
def posture():
    global latest_data, last_data_at

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
        last_data_at = time.monotonic()

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
