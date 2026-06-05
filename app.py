from flask import Flask, request, jsonify
from datetime import datetime

import numpy as np
import tensorflow as tf
import joblib
import os
import math

app = Flask(__name__)

# ==================================================
# LOAD AI MODEL
# ==================================================

MODEL_PATH = "modelAI/best_breath_v3.keras"
SCALER_PATH = "modelAI/breath_scaler_v3.joblib"

model = None
scaler = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("===================================")
    print("AI MODEL LOADED")
    print(MODEL_PATH)
    print("===================================")

except Exception as e:
    print("===================================")
    print("AI LOAD FAILED")
    print(str(e))
    print("===================================")

# ==================================================
# BUFFER AI
# ==================================================

WINDOW_SIZE = 500

window_buffer = []

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

    "acc_mag": 0,
    "gyro_mag": 0,

    "time": "-"
}

# ==================================================
# POSTURE
# ==================================================

def detect_posture(ax, ay, az):

    G = 1.0
    TH = 0.2

    if (
        abs(az - G) < TH and
        abs(ax) < 0.4 and
        abs(ay) < 0.4
    ):
        return "NAM"

    elif (
        abs(ay - G) < TH and
        abs(ax) < 0.4 and
        abs(az) < 0.4
    ):
        return "NAM_NGHIENG"

    elif (
        abs(ax - G) < TH and
        abs(ay) < 0.4 and
        abs(az) < 0.4
    ):
        return "DUNG"

    return "NGOI"

# ==================================================
# AI BPM
# ==================================================

def predict_bpm():

    global window_buffer

    if model is None:
        return "AI_ERROR"

    if scaler is None:
        return "SCALER_ERROR"

    try:

        data = np.array(window_buffer)

        scaled = scaler.transform(data)

        scaled = scaled.reshape(
            1,
            WINDOW_SIZE,
            8
        )

        pred = model.predict(
            scaled,
            verbose=0
        )

        bpm = float(pred[0][0])

        bpm = max(
            8,
            min(40, bpm)
        )

        return round(bpm, 2)

    except Exception as e:

        print("PREDICT ERROR:", str(e))

        return "ERROR"

# ==================================================
# DASHBOARD
# ==================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

<meta charset="utf-8">

<title>Breath AI Cloud</title>

<meta http-equiv="refresh" content="1">

<style>

body{
font-family:Arial;
background:#f2f2f2;
padding:30px;
}

.card{
max-width:900px;
margin:auto;
background:white;
border-radius:15px;
overflow:hidden;
box-shadow:0 0 15px rgba(0,0,0,0.1);
}

.header{
background:#1f8b3f;
color:white;
padding:25px;
text-align:center;
}

.content{
padding:25px;
}

.box{
background:#f7f7f7;
padding:20px;
margin-bottom:15px;
border-left:5px solid green;
border-radius:10px;
}

.value{
font-size:40px;
font-weight:bold;
color:green;
}

.grid{
display:grid;
grid-template-columns:repeat(3,1fr);
gap:10px;
}

.metric{
background:#f7f7f7;
padding:15px;
border-radius:10px;
text-align:center;
}

.footer{
padding:20px;
text-align:center;
color:gray;
}

</style>

</head>

<body>

<div class="card">

<div class="header">

<h1>🚀 Breath AI Cloud</h1>

<p>Real-time Posture & Breath Monitor</p>

</div>

<div class="content">

<div class="box">

<h3>TƯ THẾ HIỆN TẠI</h3>

<div class="value">{posture}</div>

</div>

<div class="box">

<h3>NHỊP THỞ AI</h3>

<div class="value">{bpm}</div>

</div>

<h3>Accelerometer</h3>

<div class="grid">

<div class="metric">
AX<br><b>{ax}</b>
</div>

<div class="metric">
AY<br><b>{ay}</b>
</div>

<div class="metric">
AZ<br><b>{az}</b>
</div>

</div>

<br>

<h3>Gyroscope</h3>

<div class="grid">

<div class="metric">
GX<br><b>{gx}</b>
</div>

<div class="metric">
GY<br><b>{gy}</b>
</div>

<div class="metric">
GZ<br><b>{gz}</b>
</div>

</div>

</div>

<div class="footer">

Last Update:
{time}

</div>

</div>

</body>

</html>
"""

@app.route("/")
def home():
    return HTML.format(**latest_data)

# ==================================================
# STATUS
# ==================================================

@app.route("/status")
def status():

    return jsonify({
        "server": "online",
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
# POSTURE API
# ==================================================

@app.route("/posture", methods=["POST"])
def posture():

    global latest_data
    global window_buffer

    try:

        data = request.get_json(force=True)

        ax = float(data.get("ax", 0))
        ay = float(data.get("ay", 0))
        az = float(data.get("az", 0))

        gx = float(data.get("gx", 0))
        gy = float(data.get("gy", 0))
        gz = float(data.get("gz", 0))

        acc_mag = math.sqrt(
            ax*ax +
            ay*ay +
            az*az
        )

        gyro_mag = math.sqrt(
            gx*gx +
            gy*gy +
            gz*gz
        )

        posture_result = detect_posture(
            ax,
            ay,
            az
        )

        feature = [
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
            acc_mag,
            gyro_mag
        ]

        window_buffer.append(feature)

        if len(window_buffer) > WINDOW_SIZE:
            window_buffer.pop(0)

        bpm_value = "WAITING"

        if len(window_buffer) == WINDOW_SIZE:
            bpm_value = predict_bpm()

        latest_data = {

            "posture": posture_result,
            "bpm": bpm_value,

            "ax": round(ax,3),
            "ay": round(ay,3),
            "az": round(az,3),

            "gx": round(gx,3),
            "gy": round(gy,3),
            "gz": round(gz,3),

            "acc_mag": round(acc_mag,3),
            "gyro_mag": round(gyro_mag,3),

            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

        print(
            "POSTURE:",
            posture_result,
            "BPM:",
            bpm_value
        )

        return jsonify({
            "success": True,
            "posture": posture_result,
            "bpm": bpm_value
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ==================================================
# TEST
# ==================================================

@app.route("/test")
def test():

    return jsonify({
        "success": True,
        "message": "Cloud Running"
    })

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )