from flask import Flask, request, jsonify
import math

app = Flask(__name__)

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Breath AI Cloud</title>
    </head>

    <body>
        <h1>🚀 Breath AI Cloud Server</h1>

        <p>Server Status: Online</p>

        <h3>Available APIs</h3>

        <ul>
            <li>GET /status</li>
            <li>POST /posture</li>
        </ul>

    </body>
    </html>
    """


# =========================
# STATUS
# =========================
@app.route("/status")
def status():
    return jsonify({
        "server": "online",
        "service": "Breath AI Cloud"
    })


# =========================
# POSTURE API
# =========================
@app.route("/posture", methods=["POST"])
def posture():

    try:

        data = request.get_json()

        ax = float(data.get("ax", 0))
        ay = float(data.get("ay", 0))
        az = float(data.get("az", 0))

        gx = float(data.get("gx", 0))
        gy = float(data.get("gy", 0))
        gz = float(data.get("gz", 0))

        # Độ lớn gia tốc
        acc_mag = math.sqrt(
            ax**2 +
            ay**2 +
            az**2
        )

        # Tính góc nghiêng
        pitch = math.degrees(
            math.atan2(
                ax,
                math.sqrt(ay*ay + az*az)
            )
        )

        roll = math.degrees(
            math.atan2(
                ay,
                math.sqrt(ax*ax + az*az)
            )
        )

        # =====================
        # PHÂN LOẠI TƯ THẾ
        # =====================

        posture = "UNKNOWN"

        if abs(az) > 8:

            if abs(pitch) < 20:
                posture = "GOOD_POSTURE"
            else:
                posture = "BAD_POSTURE"

        elif abs(ay) > 8:
            posture = "LYING"

        elif abs(ax) > 8:
            posture = "SIDE"

        else:
            posture = "MOVING"

        print(
            f"AX={ax:.2f} "
            f"AY={ay:.2f} "
            f"AZ={az:.2f} "
            f"PITCH={pitch:.2f} "
            f"ROLL={roll:.2f} "
            f"==> {posture}"
        )

        return jsonify({

            "success": True,

            "posture": posture,

            "pitch": round(pitch, 2),

            "roll": round(roll, 2),

            "acc_mag": round(acc_mag, 2)

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )