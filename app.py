from flask import Flask, request, jsonify
import math

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Breath Cloud Server</h1>
    <p>Server Status: Online</p>

    <ul>
        <li>GET /status</li>
        <li>POST /posture</li>
    </ul>
    """

@app.route("/status")
def status():
    return jsonify({
        "server": "online",
        "url": "https://breath-cloud.onrender.com"
    })

@app.route("/posture", methods=["POST"])
def posture():

    data = request.get_json()

    ax = float(data.get("ax", 0))
    ay = float(data.get("ay", 0))
    az = float(data.get("az", 0))

    if abs(az) > 8:
        posture = "UPRIGHT"
    elif abs(ay) > 8:
        posture = "LYING"
    elif abs(ax) > 8:
        posture = "SIDE"
    else:
        posture = "MOVING"

    return jsonify({
        "success": True,
        "posture": posture,
        "ax": ax,
        "ay": ay,
        "az": az
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)