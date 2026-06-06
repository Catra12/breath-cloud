from datetime import datetime
import time

import joblib
import numpy as np
try:
    import tensorflow as tf
except Exception as e:
    tf = None
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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Breath AI Cloud</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
    --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    --card-bg: rgba(30, 41, 59, 0.7);
    --border-color: rgba(255, 255, 255, 0.08);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --primary: #10b981;
    --primary-glow: rgba(16, 185, 129, 0.15);
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.15);
}}

body {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: var(--bg-gradient);
    min-height: 100vh;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-primary);
    padding: 20px;
    box-sizing: border-box;
}}

.card {{
    width: 100%;
    max-width: 800px;
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    overflow: hidden;
    transition: transform 0.3s ease;
}}

.header {{
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0) 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 32px 24px;
    text-align: center;
}}

.header h1 {{
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    background: linear-gradient(to right, #34d399, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.header p {{
    margin: 8px 0 0 0;
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 500;
}}

.main-sections {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 24px;
}}

@media (max-width: 600px) {{
    .main-sections {{
        grid-template-columns: 1fr;
    }}
}}

.section {{
    padding: 24px;
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}}

.section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}}

.section.posture-sec::before {{
    background: var(--accent);
}}

.section.bpm-sec::before {{
    background: var(--primary);
}}

.section h3 {{
    margin: 0 0 12px 0;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.live-indicator {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 0 0 var(--primary-glow);
    animation: pulse 1.5s infinite;
}}

@keyframes pulse {{
    0% {{
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }}
    70% {{
        transform: scale(1);
        box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
    }}
    100% {{
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }}
}}

.big {{
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
    min-height: 56px;
    display: flex;
    align-items: center;
}}

.posture-sec .big {{
    color: #3b82f6;
    text-shadow: 0 0 20px var(--accent-glow);
}}

.bpm-sec .big {{
    color: #10b981;
    text-shadow: 0 0 20px var(--primary-glow);
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    padding: 0 24px 24px 24px;
}}

.box {{
    background: rgba(15, 23, 42, 0.3);
    border: 1px solid var(--border-color);
    padding: 16px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
    transition: background 0.2s ease;
}}

.box:hover {{
    background: rgba(15, 23, 42, 0.5);
}}

.value {{
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 4px;
    font-family: monospace;
}}

.footer {{
    padding: 16px 24px;
    color: var(--text-secondary);
    font-size: 0.8rem;
    text-align: center;
    border-top: 1px solid var(--border-color);
    background: rgba(15, 23, 42, 0.2);
}}
</style>
</head>
<body>
<div class="card">
    <div class="header">
        <h1>Breath AI Cloud</h1>
        <p>Real-time Posture & Breath Monitor</p>
    </div>
    
    <div class="main-sections">
        <div class="section posture-sec">
            <h3>Tư thế hiện tại</h3>
            <div class="big" id="posture">{current_data["posture"]}</div>
        </div>
        <div class="section bpm-sec">
            <h3>Nhịp thở AI <span class="live-indicator"></span></h3>
            <div class="big" id="bpm">{current_data["bpm"]}</div>
        </div>
    </div>

    <div class="grid">
        <div class="box">AX <div class="value" id="val-ax">{current_data["ax"]}</div></div>
        <div class="box">AY <div class="value" id="val-ay">{current_data["ay"]}</div></div>
        <div class="box">AZ <div class="value" id="val-az">{current_data["az"]}</div></div>
        <div class="box">GX <div class="value" id="val-gx">{current_data["gx"]}</div></div>
        <div class="box">GY <div class="value" id="val-gy">{current_data["gy"]}</div></div>
        <div class="box">GZ <div class="value" id="val-gz">{current_data["gz"]}</div></div>
    </div>
    <div class="footer" id="last-update">Last Update: {current_data["time"]}</div>
</div>

<script>
function updateData() {{
    fetch('/current')
        .then(response => response.json())
        .then(data => {{
            document.getElementById('posture').innerText = data.posture;
            document.getElementById('bpm').innerText = data.bpm;
            document.getElementById('val-ax').innerText = data.ax;
            document.getElementById('val-ay').innerText = data.ay;
            document.getElementById('val-az').innerText = data.az;
            document.getElementById('val-gx').innerText = data.gx;
            document.getElementById('val-gy').innerText = data.gy;
            document.getElementById('val-gz').innerText = data.gz;
            document.getElementById('last-update').innerText = "Last Update: " + data.time;
        }})
        .catch(error => console.error('Error fetching data:', error));
}}
setInterval(updateData, 1000);
</script>
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
