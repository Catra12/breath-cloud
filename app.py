from flask import Flask, request, jsonify
from datetime import datetime

import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# ==================================================
# AI MODEL
# ==================================================

MODEL_OK = False

try:
    model = tf.keras.models.load_model(
        "modelAI/best_breath_v3.keras",
        compile=False
    )

    scaler = joblib.load(
        "modelAI/breath_scaler_v3.joblib"
    )

    MODEL_OK = True

    print("=" * 50)
    print("AI MODEL LOADED SUCCESS")
    print("=" * 50)

except Exception as e:

    print("=" * 50)
    print("AI LOAD FAILED")
    print(e)
    print("=" * 50)

# ==================================================
# CONFIG
# ==================================================

WINDOW_SIZE = 500

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

    g = 1.0
    threshold = 0.2

    if (
        abs(az - g) < threshold
        and abs(ax) < 0.4
        and abs(ay) < 0.4
    ):
        return "NAM"

    elif (
        abs(ay - g) < threshold
        and abs(ax) < 0.4
        and abs(az) < 0.4
    ):
        return "NAM_NGHIENG"

    elif (
        abs(ax - g) < threshold
        and abs(ay) < 0.4
        and abs(az) < 0.4
    ):
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

        data = np.array(buffer[-WINDOW_SIZE:])

        scaled = scaler.transform(data)

        scaled = scaled.reshape(
            1,
            WINDOW_SIZE,
            8
        )

        pred = model.predict(
            scaled,
            verbose=0
        )[0][0]

        pred = float(
            np.clip(pred, 12, 20)
        )

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

<div class="big">
{latest_data["posture"]}
</div>

</div>

<div class="section">

<h3>NHỊP THỞ (AI)</h3>

<div class="big">
{latest_data["bpm"]}
</div>

</div>

<div class="grid">

<div class="box">
AX
<div class="value">{latest_data["ax"]}</div>
</div>

<div class="box">
AY
<div class="value">{latest_data["ay"]}</div>
</div>

<div class="box">
AZ
<div class="value">{latest_data["az"]}</div>
</div>

<div class="box">
GX
<div class="value">{latest_data["gx"]}</div>
</div>

<div class="box">
GY
<div class="value">{latest_data["gy"]}</div>
</div>

<div class="box">
GZ
<div class="value">{latest_data["gz"]}</div>
</div>

</div>

<div class="footer">

Last Update:
{latest_data["time"]}

</div>

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
        "bpm": latest_data["bpm"]
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
        "model_loaded": MODEL_OK
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

        posture_result = detect_posture(
            ax,
            ay,
            az
        )

        acc_mag = np.sqrt(
            ax**2 +
            ay**2 +
            az**2
        )

        gyro_mag = np.sqrt(
            gx**2 +
            gy**2 +
            gz**2
        )

        feature_row = [
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
            acc_mag,
            gyro_mag
        ]

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

            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
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