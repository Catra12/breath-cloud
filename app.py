from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# =====================================
# DỮ LIỆU MỚI NHẤT
# =====================================

latest_data = {
    "posture": "WAITING",
    "bpm": "WAITING",
    "ax": 0.0,
    "ay": 0.0,
    "az": 0.0,
    "gx": 0.0,
    "gy": 0.0,
    "gz": 0.0,
    "time": "-"
}

# =====================================
# THUẬT TOÁN PHÂN LOẠI TƯ THẾ
# =====================================

def detect_posture(ax: float, ay: float, az: float) -> str:
    """
    Phân loại tư thế dựa trên dữ liệu gia tốc.
    Trả về: NAM | NAM_NGHIENG | DUNG | NGOI
    """
    G = 1.0
    THRESHOLD = 0.2
    SIDE_LIMIT = 0.4

    # Nằm ngửa (az ≈ 1g)
    if abs(az - G) < THRESHOLD and abs(ax) < SIDE_LIMIT and abs(ay) < SIDE_LIMIT:
        return "NAM"

    # Nằm nghiêng (ay ≈ 1g)
    if abs(ay - G) < THRESHOLD and abs(ax) < SIDE_LIMIT and abs(az) < SIDE_LIMIT:
        return "NAM_NGHIENG"

    # Đứng (ax ≈ 1g)
    if abs(ax - G) < THRESHOLD and abs(ay) < SIDE_LIMIT and abs(az) < SIDE_LIMIT:
        return "DUNG"

    # Mặc định: ngồi
    return "NGOI"

# =====================================
# DASHBOARD
# =====================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Breath AI Cloud</title>
    <meta http-equiv="refresh" content="1">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: Arial, sans-serif;
            background: #f0f2f5;
            display: flex;
            justify-content: center;
            padding: 30px 16px;
        }}

        .card {{
            width: 100%;
            max-width: 720px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .card-header {{
            background: #1a7f3c;
            color: white;
            padding: 20px;
            text-align: center;
        }}

        .card-header h1 {{
            font-size: 26px;
            letter-spacing: 1px;
        }}

        .card-header p {{
            font-size: 13px;
            opacity: 0.8;
            margin-top: 4px;
        }}

        .card-body {{
            padding: 24px;
        }}

        .section {{
            margin-bottom: 20px;
            padding: 16px;
            background: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #1a7f3c;
        }}

        .section-title {{
            font-size: 13px;
            text-transform: uppercase;
            color: #888;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}

        .section-value {{
            font-size: 32px;
            font-weight: bold;
            color: #1a7f3c;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}

        .metric {{
            background: #f9f9f9;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}

        .metric-label {{
            font-size: 12px;
            color: #aaa;
            margin-bottom: 4px;
        }}

        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }}

        .footer {{
            text-align: center;
            font-size: 13px;
            color: #bbb;
            padding: 16px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="card-header">
            <h1>🚀 Breath AI Cloud</h1>
            <p>Real-time Posture &amp; Breath Monitor</p>
        </div>

        <div class="card-body">

            <div class="section">
                <div class="section-title">Tư thế hiện tại</div>
                <div class="section-value">{posture}</div>
            </div>

            <div class="section">
                <div class="section-title">Nhịp thở (AI)</div>
                <div class="section-value">{bpm}</div>
            </div>

            <p style="font-size:13px;color:#888;margin-bottom:10px;font-weight:bold;">
                Accelerometer (g)
            </p>
            <div class="grid" style="margin-bottom:20px;">
                <div class="metric">
                    <div class="metric-label">AX</div>
                    <div class="metric-value">{ax}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">AY</div>
                    <div class="metric-value">{ay}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">AZ</div>
                    <div class="metric-value">{az}</div>
                </div>
            </div>

            <p style="font-size:13px;color:#888;margin-bottom:10px;font-weight:bold;">
                Gyroscope (°/s)
            </p>
            <div class="grid">
                <div class="metric">
                    <div class="metric-label">GX</div>
                    <div class="metric-value">{gx}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">GY</div>
                    <div class="metric-value">{gy}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">GZ</div>
                    <div class="metric-value">{gz}</div>
                </div>
            </div>

        </div>

        <div class="footer">
            Cập nhật lần cuối: <strong>{time}</strong>
        </div>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return DASHBOARD_HTML.format(**latest_data)

# =====================================
# STATUS
# =====================================

@app.route("/status")
def status():
    return jsonify({
        "server": "online",
        "posture": latest_data["posture"],
        "bpm": latest_data["bpm"]
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
        data = request.get_json(force=True)

        ax = float(data.get("ax", 0))
        ay = float(data.get("ay", 0))
        az = float(data.get("az", 0))
        gx = float(data.get("gx", 0))
        gy = float(data.get("gy", 0))
        gz = float(data.get("gz", 0))

        posture_result = detect_posture(ax, ay, az)

        latest_data = {
            "posture": posture_result,
            "bpm": "WAITING",          # AI chưa bật
            "ax": round(ax, 3),
            "ay": round(ay, 3),
            "az": round(az, 3),
            "gx": round(gx, 3),
            "gy": round(gy, 3),
            "gz": round(gz, 3),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("[DATA]", latest_data)

        return jsonify({
            "success": True,
            "posture": posture_result,
            "bpm": latest_data["bpm"]
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"success": False, "error": str(e)}), 400

# =====================================
# TEST
# =====================================

@app.route("/test")
def test():
    return jsonify({"success": True, "message": "Cloud Running"})

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)