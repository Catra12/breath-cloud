from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "ESP32 Cloud Running"

@app.route("/posture", methods=["POST"])
def posture():

    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    ax = data.get("ax", 0)
    ay = data.get("ay", 0)
    az = data.get("az", 0)

    result = "UNKNOWN"

    if az > 8:
        result = "STANDING"
    elif ay > 8:
        result = "LYING"
    elif ax > 8:
        result = "SIDE"

    return jsonify({
        "posture": result
    })

if __name__ == "__main__":
    app.run()
