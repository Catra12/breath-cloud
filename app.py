from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# =====================================
# DỮ LIỆU MỚI NHẤT
# =====================================

latest_data = {
    "posture": "WAITING",

    "ax": 0,
    "ay": 0,
    "az": 0,

    "gx": 0,
    "gy": 0,
    "gz": 0,

    "time": "-"
}

# =====================================
# THUẬT TOÁN PHÂN LOẠI TƯ THẾ
# =====================================

def detect_posture(ax, ay, az):

    eps = 0.25

    # Nằm
    if abs(az - 1.0) < eps:
        return "NAM"

    # Đứng
    elif abs(ax - 1.0) < eps:
        return "DUNG"

    # Ngồi
    else:
        return "NGOI"

# =====================================
# DASHBOARD
# =====================================

@app.route("/")
def home():

    return f"""
    <html>

    <head>

        <title>Breath AI Cloud</title>

        <meta http-equiv="refresh" content="1">

        <style>

        body {{
            font-family: Arial;
            text-align: center;
            margin-top: 40px;
        }}

        .card {{
            width: 500px;
            margin: auto;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
        }}

        h1 {{
            color: green;
        }}

        </style>

    </head>

    <body>

        <h1>🚀 Breath AI Cloud</h1>

        <div class="card">

            <h2>Tư thế hiện tại</h2>

            <h1>{latest_data["posture"]}</h1>

            <hr>

            <p>AX = {latest_data["ax"]}</p>
            <p>AY = {latest_data["ay"]}</p>
            <p>AZ = {latest_data["az"]}</p>

            <p>GX = {latest_data["gx"]}</p>
            <p>GY = {latest_data["gy"]}</p>
            <p>GZ = {latest_data["gz"]}</p>

            <hr>

            <p>Cập nhật lần cuối:</p>

            <b>{latest_data["time"]}</b>

        </div>

    </body>

    </html>
    """

# =====================================
# STATUS
# =====================================

@app.route("/status")
def status():

    return jsonify({
        "server": "online",
        "posture": latest_data["posture"]
    })

# =====================================
# CURRENT DATA
# =====================================

@app.route("/current")
def current():

    return jsonify(latest_data)

# =====================================
# NHẬN DỮ LIỆU ESP32
# =====================================

@app.route("/posture", methods=["POST"])
def posture():

    global latest_data

    try:

        data = request.get_json()

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

        latest_data = {
            "posture": posture_result,

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
            "posture": posture_result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )