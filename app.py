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
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(
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

# ==================================================
# SPECTRAL POWER FEATURE
# ==================================================


def compute_spectral_power(acc_mag_window, fs=25):
    """
    Tính spectral power trong dải thở 0.1–0.55 Hz
    Normalize về 0–1 theo window hiện tại
    Đây là feature thứ 9 — giống Cell 3B trong notebook
    """
    sig = acc_mag_window.copy()
    sig -= sig.mean()
    sig *= np.hanning(len(sig))

    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(sig)) ** 2
    mask = (freqs >= 0.10) & (freqs <= 0.55)

    power = float(np.sum(spectrum[mask]))

    # Normalize đơn giản — dùng log scale để tránh outlier
    power_norm = np.log1p(power) / 30.0
    power_norm = float(np.clip(power_norm, 0.0, 1.0))

    return power_norm

# ==================================================
# AI BPM
# ==================================================


def predict_bpm():
    if not MODEL_OK:
        return "AI_FAILED"

    if len(buffer) < WINDOW_SIZE:
        return f"BUFFER {len(buffer)}/{WINDOW_SIZE}"

    try:
        # Lấy window 500 samples gần nhất (8 features)
        window_8 = np.array(
            buffer[-WINDOW_SIZE:],
            dtype=np.float32
        )  # shape (500, 8)

        # Tính spectral power cho cột acc_mag (index 6)
        acc_mag_col = window_8[:, 6]
        spectral_pow = compute_spectral_power(acc_mag_col, fs=FS)

        # Ghép thành 9 features
        sp_col = np.full((WINDOW_SIZE, 1), spectral_pow, dtype=np.float32)
        window_9 = np.concatenate([window_8, sp_col], axis=1)  # (500, 9)

        # Scale
        scaled = scaler.transform(window_9)  # (500, 9)
        scaled = scaled.reshape(1, WINDOW_SIZE, N_FEATURES).astype(np.float32)

        # TFLite inference
        interpreter.set_tensor(input_details[0]["index"], scaled)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0][0]

        pred = float(np.clip(pred, 12, 20))
        return round(pred, 1)

    except Exception as e:
        print("AI ERROR:", e)
        return "AI_ERROR"

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
<h1>🚀 Breath AI Cloud</h1>
<p>Real-time Posture & Breath Monitor</p>
</div>
<div class="section">
<h3>TƯ THẾ HIỆN TẠI</h3>
<div class="big">{latest_data["posture"]}</div>
</div>
<div class="section">
<h3>NHỊP THỞ (AI)</h3>
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
# STATUS
# ==================================================


@app.route("/status")
def status():
    return jsonify({
        "server": "online",
        "model": MODEL_OK,
        "posture": latest_data["posture"],
        "bpm": latest_data["bpm"],
        "buffer": len(buffer)
    })

# ==================================================
# CURRENT
# ==================================================


@app.route("/current")
def current():
    return jsonify(latest_data)

# ==================================================
# TEST
# ==================================================


@app.route("/test")
def test():
    return jsonify({
        "success": True,
        "message": "Cloud Running",
        "model_loaded": MODEL_OK,
        "buffer_size": len(buffer),
        "window_size": WINDOW_SIZE
    })

# ==================================================
# POSTURE API
# ==================================================


@app.route("/posture", methods=["POST"])
def posture():
    global latest_data

    try:
        data = request.get_json(force=True)

        ax = float(data.get("ax", 0))
        ay = float(data.get("ay", 0))
        az = float(data.get("az", 0))

        gx = float(data.get("gx", 0))
        gy = float(data.get("gy", 0))
        gz = float(data.get("gz", 0))

        posture_result = detect_posture(ax, ay, az)

        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # Buffer 8 features (spectral_power tính sau từ window đầy đủ)
        feature_row = [ax, ay, az, gx, gy, gz, acc_mag, gyro_mag]
        buffer.append(feature_row)

        if len(buffer) > WINDOW_SIZE:
            buffer.pop(0)

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

        print(latest_data)

        return jsonify({
            "success": True,
            "posture": posture_result,
            "bpm": bpm,
            "buffer": len(buffer)
        })

    except Exception as e:
        print(e)

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
