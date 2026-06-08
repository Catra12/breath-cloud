from datetime import datetime
import time

import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# ==================================================
# AI MODEL - TFLite only (with fallback)
# ==================================================

MODEL_OK = False
MODEL_ERROR = None
MODEL_BACKEND = "tflite"
interpreter = None
input_details = None
output_details = None
scaler = None

Interpreter = None
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        Interpreter = None

try:
    scaler = joblib.load("modelAI/breath_scaler_v3.joblib")

    if Interpreter is None:
        raise ImportError("Neither tensorflow nor tflite_runtime is installed.")

    interpreter = Interpreter(model_path="modelAI/breath_v3.tflite")
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
N_FEATURES = 5      # acc_mag_filtered, gyro_mag, spectral_power, fft_bpm_norm, fft_confidence
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

# Precomputed FFT constants and BPM classes
freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / FS)
breath_mask = (freqs >= 0.10) & (freqs <= 0.55)   # 6-33 BPM
useful_mask = (freqs >= 0.03) & (freqs <= 2.0)
hann = np.hanning(WINDOW_SIZE)
classes_bpm = [12, 13, 14, 15, 16, 17, 18, 19, 20]

# Stores 8-element raw data: ax, ay, az, gx, gy, gz, acc_mag, gyro_mag
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
    "time": "-",
    "method": "-",
    "expected_bpm": None,
    "argmax_bpm": None,
    "confidence": None,
    "probabilities": [0.0] * len(classes_bpm),
    "fft_bpm": None,
    "fft_confidence": None,
    "spectral_power": None,
    "signal_status": "WAITING",
    "buffer_size": 0
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
        return "nằm"

    if (
        abs(ay - g) < threshold
        and abs(ax) < 0.4
        and abs(az) < 0.4
    ):
        return "nằm nghiêng"

    if (
        abs(ax - g) < threshold
        and abs(ay) < 0.4
        and abs(az) < 0.4
    ):
        return "đứng"

    return "ngồi"

# ==================================================
# PREPROCESSING & AI BPM
# ==================================================


def rolling_mean_numpy(x, window_len=5):
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window_len // 2)
        end = min(n, i + window_len // 2 + 1)
        out[i] = np.mean(x[start:end])
    return out


def compute_fft_features(signal):
    seg = signal.astype(np.float64).copy()
    seg -= seg.mean()
    seg *= hann

    spectrum = np.abs(np.fft.rfft(seg)) ** 2
    breath_power = spectrum[breath_mask]
    useful_power = spectrum[useful_mask].sum() + 1e-9

    spectral_power = np.log1p(breath_power.sum())
    fft_bpm = freqs[breath_mask][np.argmax(breath_power)] * 60.0
    fft_confidence = breath_power.max() / useful_power

    return float(spectral_power), float(fft_bpm / 60.0), float(fft_confidence)


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
    if len(buffer) < WINDOW_SIZE:
        return {
            "status": f"BUFFER {len(buffer)}/{WINDOW_SIZE}",
            "bpm": f"BUFFER {len(buffer)}/{WINDOW_SIZE}"
        }

    try:
        window = np.array(
            buffer[-WINDOW_SIZE:],
            dtype=np.float32
        )  # shape (500, 8)

        signal_ok, signal_status, signal_info = analyze_breath_signal(window)
        if not signal_ok:
            print("SIGNAL CHECK:", signal_status, signal_info)
            return {
                "status": signal_status,
                "bpm": signal_status,
                "info": signal_info
            }

        acc_mag = window[:, 6]
        gyro_mag = window[:, 7]
        acc_mag_filtered = rolling_mean_numpy(acc_mag, window_len=5)

        spectral_power, fft_bpm_norm, fft_confidence = compute_fft_features(acc_mag_filtered)
        fft_bpm = fft_bpm_norm * 60.0

        if not MODEL_OK:
            # Fallback to FFT estimation
            return {
                "status": "OK",
                "bpm": f"{round(fft_bpm, 1)} (FFT)",
                "method": "FFT",
                "fft_bpm": round(fft_bpm, 1),
                "fft_confidence": round(fft_confidence, 3),
                "spectral_power": round(spectral_power, 3)
            }

        # Construct 5-feature matrix [acc_mag_filtered, gyro_mag, spectral_power, fft_bpm_norm, fft_confidence]
        window_base = np.column_stack((acc_mag_filtered, gyro_mag)).astype(np.float32)
        constant_features = np.tile(
            np.array([spectral_power, fft_bpm_norm, fft_confidence], dtype=np.float32),
            (WINDOW_SIZE, 1)
        )
        model_input = np.concatenate([window_base, constant_features], axis=1)

        # Scale and predict
        scaled = scaler.transform(model_input)
        scaled = scaled.reshape(1, WINDOW_SIZE, 5).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], scaled)
        interpreter.invoke()
        output_probs = interpreter.get_tensor(output_details[0]["index"])[0]  # shape (9,)

        pred_idx = int(np.argmax(output_probs))
        argmax_bpm = classes_bpm[pred_idx]
        expected_bpm = float(np.sum(output_probs * classes_bpm))
        confidence = float(output_probs[pred_idx])

        return {
            "status": "OK",
            "bpm": f"{round(expected_bpm, 1)}",
            "method": "AI",
            "expected_bpm": round(expected_bpm, 2),
            "argmax_bpm": argmax_bpm,
            "confidence": round(confidence, 3),
            "probabilities": [round(float(p), 4) for p in output_probs],
            "fft_bpm": round(fft_bpm, 1),
            "fft_confidence": round(fft_confidence, 3),
            "spectral_power": round(spectral_power, 3)
        }

    except Exception as e:
        print("AI ERROR:", type(e).__name__, e)
        return {
            "status": "AI_ERROR",
            "bpm": "AI_ERROR",
            "error": str(e)
        }

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
    return render_template("index.html")

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
        "classes_bpm": classes_bpm,
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
        "features": N_FEATURES,
        "classes_bpm": classes_bpm
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
        bpm_result = predict_bpm()

        latest_data = {
            "posture": posture_result,
            "bpm": bpm_result.get("bpm", "—"),
            "ax": round(ax, 3),
            "ay": round(ay, 3),
            "az": round(az, 3),
            "gx": round(gx, 3),
            "gy": round(gy, 3),
            "gz": round(gz, 3),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": bpm_result.get("method", "-"),
            "expected_bpm": bpm_result.get("expected_bpm"),
            "argmax_bpm": bpm_result.get("argmax_bpm"),
            "confidence": bpm_result.get("confidence"),
            "probabilities": bpm_result.get("probabilities", [0.0] * len(classes_bpm)),
            "fft_bpm": bpm_result.get("fft_bpm"),
            "fft_confidence": bpm_result.get("fft_confidence"),
            "spectral_power": bpm_result.get("spectral_power"),
            "signal_status": bpm_result.get("status", "-"),
            "buffer_size": len(buffer)
        }
        last_data_at = time.monotonic()

        print({
            "posture": posture_result,
            "bpm": bpm_result.get("bpm"),
            "method": bpm_result.get("method"),
            "confidence": bpm_result.get("confidence"),
            "received": len(samples),
            "buffer": len(buffer)
        })

        return jsonify({
            "success": True,
            "posture": posture_result,
            "bpm": bpm_result.get("bpm"),
            "method": bpm_result.get("method", "-"),
            "expected_bpm": bpm_result.get("expected_bpm"),
            "argmax_bpm": bpm_result.get("argmax_bpm"),
            "confidence": bpm_result.get("confidence"),
            "probabilities": bpm_result.get("probabilities", []),
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
