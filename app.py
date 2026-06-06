from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np

app = Flask(__name__)

# ==================================================
# CONFIG
# ==================================================

WINDOW_SIZE = 500

buffer = []

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

    g_val = np.sqrt(ax**2 + ay**2 + az**2)

    if g_val == 0:
        return "NGOI"

    ax_n = ax / g_val
    ay_n = ay / g_val
    az_n = az / g_val

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
Last Update: {latest_data["time"]}
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
        "buffer": len(buffer)
    })

# ==================================================
# POSTURE API
# ==================================================

@app.route("/posture", methods=["POST"])
def posture():

    global latest_data

    try:

        data = request.get_json(force=True)

        # =====================================
        # ESP32 gửi batch
        # =====================================

        if "batch" in data:

            samples = data["batch"]

            for sample in samples:

                ax = float(sample.get("ax", 0))
                ay = float(sample.get("ay", 0))
                az = float(sample.get("az", 0))

                gx = float(sample.get("gx", 0))
                gy = float(sample.get("gy", 0))
                gz = float(sample.get("gz", 0))

                posture_result = detect_posture(ax, ay, az)

                buffer.append([
                    ax, ay, az,
                    gx, gy, gz
                ])

                if len(buffer) > WINDOW_SIZE:
                    buffer.pop(0)

                latest_data = {
                    "posture": posture_result,
                    "bpm": "TEST_MODE",
                    "ax": round(ax, 3),
                    "ay": round(ay, 3),
                    "az": round(az, 3),
                    "gx": round(gx, 3),
                    "gy": round(gy, 3),
                    "gz": round(gz, 3),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        else:

            ax = float(data.get("ax", 0))
            ay = float(data.get("ay", 0))
            az = float(data.get("az", 0))

            gx = float(data.get("gx", 0))
            gy = float(data.get("gy", 0))
            gz = float(data.get("gz", 0))

            posture_result = detect_posture(ax, ay, az)

            latest_data = {
                "posture": posture_result,
                "bpm": "TEST_MODE",
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
            "posture": latest_data["posture"],
            "buffer": len(buffer)
        })

    except Exception as e:

        print("ERROR:", e)

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