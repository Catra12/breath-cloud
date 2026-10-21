import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from datetime import datetime
import time

import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template

from email_service import (
    send_fall_alert_email,
    send_test_email,
    test_email_connection_sync,
    load_env_vars,
    save_env_vars
)

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
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except (ImportError, AttributeError):
            Interpreter = None

try:
    scaler = joblib.load("modelAI/breath_scaler_v3.joblib")

    if Interpreter is None:
        raise ImportError("Neither ai_edge_litert, tflite_runtime, nor tensorflow is installed.")

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

# Cấu hình ngưỡng phát hiện té ngã (Fall Detection) nhạy và thông minh
FALL_IMPACT_THRESHOLD_G = 1.35    # Ngưỡng va đập cực đại SVM (1.35g để bắt nhạy mọi cú rơi thử nghiệm thực tế)
FALL_FREE_FALL_THRESHOLD_G = 0.70 # Ngưỡng rơi tự do (< 0.70g)
FALL_GYRO_MAG_THRESHOLD = 1.0     # Ngưỡng vận tốc góc biến thiên (rad/s, ~57 deg/s)
FALL_EMAIL_COOLDOWN_SECONDS = 15  # Giảm xuống 15s để người dùng test thoải mái mà không bị nghẽn
FALL_MEMORY_WINDOW_SECONDS = 6.0  # Cửa sổ ghi nhớ va đập (6 giây để kết hợp cú va chạm và tư thế nằm sau đó)

# Precomputed FFT constants and BPM classes
freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / FS)
breath_mask = (freqs >= 0.10) & (freqs <= 0.55)   # 6-33 BPM
useful_mask = (freqs >= 0.03) & (freqs <= 2.0)
hann = np.hanning(WINDOW_SIZE)
classes_bpm = [12, 13, 14, 15, 16, 17, 18, 19, 20]

# Stores 8-element raw data: ax, ay, az, gx, gy, gz, acc_mag, gyro_mag
buffer = []

# Buffer ngắn hạn theo dõi va đập và rơi tự do trong 25 mẫu gần nhất (~1 giây)
recent_samples_buffer = []

# Bộ nhớ ghi nhận va đập gần nhất trong cửa sổ thời gian
recent_impact = {
    "time": 0.0,
    "impact_g": 0.0
}

# Trạng thái cảnh báo té ngã
fall_state = {
    "alert": False,
    "time": None,
    "impact_g": 0.0,
    "posture_after": None,
    "email_status": "Chưa có sự cố",
    "last_email_time": 0.0,
    "dismissed": False
}

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
    "buffer_size": 0,
    "fall_alert": False,
    "impact_g": 0.0,
    "fall_time": None,
    "posture_detail": "WAITING",
    "email_status": "Sẵn sàng"
}

latest_data = DEFAULT_DATA.copy()
last_data_at = None

# ==================================================
# POSTURE DETECTION (NÂNG CAO CHO CẢM BIẾN Ở NGỰC)
# ==================================================


def detect_posture(ax, ay, az):
    """
    Nhận diện tư thế chi tiết khi cảm biến MPU6050 gắn trước ngực:
    - Nằm ngửa (ngực hướng lên trời: az ~ +1g)
    - Nằm úp (ngực ép xuống giường: az ~ -1g)
    - Nằm úp mặt xuống bàn (ngồi ghế gập người, ngực/mặt tì úp xuống bàn)
    - Nằm nghiêng trái / Nằm nghiêng phải (trọng lực dồn sang trục Y: ay ~ ±1g)
    - Đứng thẳng (thân thẳng đứng: ax ~ +1g)
    - Ngồi thẳng / Ngồi tựa lưng
    """
    # Tự động nhận diện và chuẩn hóa nếu chip gửi m/s² (~9.8) hoặc raw ADC (~16384)
    mag = float(np.sqrt(ax**2 + ay**2 + az**2))
    if 5.0 < mag < 50.0:
        ax, ay, az = ax / 9.80665, ay / 9.80665, az / 9.80665
    elif mag > 1000.0:
        ax, ay, az = ax / 16384.0, ay / 16384.0, az / 16384.0

    # 1. NẰM ÚP MẶT XUỐNG BÀN (Gục đầu / ngực úp xuống mặt bàn làm việc)
    if az <= -0.32 and 0.15 <= ax <= 0.75 and abs(ay) < 0.50:
        return "úp mặt xuống bàn"

    # 2. NẰM NGỬA: Ngực hướng thẳng lên trời (trục Z dương lớn, thân phẳng ngang)
    if az >= 0.65 and abs(ax) < 0.45 and abs(ay) < 0.50:
        return "nằm ngửa"

    # 3. NẰM ÚP: Nằm phẳng trên giường, ngực ép xuống mặt phẳng (trục Z âm lớn)
    if az <= -0.65 and abs(ax) < 0.35 and abs(ay) < 0.50:
        return "nằm úp"

    # 4. NẰM NGHIÊNG TRÁI / PHẢI: Trọng lực dồn sang trục Y
    if ay >= 0.50 and abs(ax) < 0.45:
        return "nằm nghiêng trái"
    if ay <= -0.50 and abs(ax) < 0.45:
        return "nằm nghiêng phải"

    # 5. NGỒI TỰA LƯNG: Lưng ngả ra sau tựa vào ghế
    # Góc ngả từ ~10 độ trở lên (az >= 0.18g, ví dụ trong ảnh thực tế ax=0.92, az=0.50)
    if az >= 0.18 and ax >= 0.35 and abs(ay) < 0.50:
        return "ngồi tựa lưng"

    # 6. ĐỨNG THẲNG: Chip bắt buộc phải ở tư thế thẳng đứng
    # Thân thẳng đứng: trọng lực chủ đạo trên trục X (ax >= 0.70g), lưng thẳng không ngửa (|az| < 0.18g, |ay| < 0.45g)
    if ax >= 0.70 and -0.32 < az < 0.18 and abs(ay) < 0.45:
        return "đứng thẳng"

    # Mặc định suy luận dựa trên trục có giá trị tuyệt đối lớn nhất
    abs_x, abs_y, abs_z = abs(ax), abs(ay), abs(az)
    if abs_z > abs_x and abs_z > abs_y:
        if az < 0:
            return "nằm úp" if abs(ax) < 0.25 else "úp mặt xuống bàn"
        return "nằm ngửa"
    elif abs_y > abs_x and abs_y > abs_z:
        return "nằm nghiêng trái" if ay > 0 else "nằm nghiêng phải"
    elif abs_x > abs_y and abs_x > abs_z:
        if az >= 0.18:
            return "ngồi tựa lưng"
        return "đứng thẳng"

    return "đứng thẳng"

# ==================================================
# FALL DETECTION (PHÁT HIỆN TÉ NGÃ)
# ==================================================


def check_fall_event(samples, current_posture, current_bpm="—", is_chip_impact=False, chip_impact_g=0.0):
    """
    Phát hiện sự cố té ngã thông minh:
    1. Ghi nhận cú va đập SVM >= 1.25g, hoặc rơi tự do, hoặc cờ va đập từ chip ESP32.
    2. Lưu giữ vào bộ nhớ va đập trong cửa sổ 6.0 giây.
    3. Khi tư thế chuyển sang NẰM (hoặc ĐỔ GỤC) trong vòng 6.0 giây sau va chạm
       -> KÍCH HOẠT CẢNH BÁO TÉ NGÃ NGAY LẬP TỨC.
    """
    global fall_state, recent_impact

    max_impact = 0.0
    min_impact = 999.0
    has_high_gyro = False
    last_ax = 0.0

    for s in samples:
        ax = float(s.get("ax", 0))
        ay = float(s.get("ay", 0))
        az = float(s.get("az", 0))
        gx = float(s.get("gx", 0))
        gy = float(s.get("gy", 0))
        gz = float(s.get("gz", 0))

        # Chuẩn hóa về g nếu cần
        mag = float(np.sqrt(ax**2 + ay**2 + az**2))
        if 5.0 < mag < 50.0:
            ax, ay, az = ax / 9.80665, ay / 9.80665, az / 9.80665
            mag = mag / 9.80665
        elif mag > 1000.0:
            ax, ay, az = ax / 16384.0, ay / 16384.0, az / 16384.0
            mag = mag / 16384.0

        last_ax = ax

        if mag > max_impact:
            max_impact = mag
        if mag < min_impact:
            min_impact = mag

        gyro_mag = float(np.sqrt(gx**2 + gy**2 + gz**2))
        if gyro_mag >= 1.0:
            has_high_gyro = True

    now = time.time()

    # 1. Phát hiện va đập / rơi tự do (Siêu nhạy - bắt trọn ngay lần rơi đầu tiên)
    if is_chip_impact:
        recent_impact["time"] = now
        recent_impact["impact_g"] = round(max(recent_impact.get("impact_g", 0.0), chip_impact_g, 1.8), 2)
        fall_state["dismissed"] = False
        print(f"⚡ [CHIP BÁO VA ĐẬP TRỰC TIẾP] Lực va đập: {recent_impact['impact_g']}g (Ghi nhớ trong {FALL_MEMORY_WINDOW_SECONDS}s)")

    is_impact = (max_impact >= 1.25) or \
                (max_impact >= 1.15 and has_high_gyro) or \
                (min_impact <= 0.75 and max_impact >= 1.10) or \
                is_chip_impact

    if is_impact:
        recent_impact["time"] = now
        recent_impact["impact_g"] = round(max(recent_impact.get("impact_g", 0.0), max_impact, chip_impact_g), 2)
        fall_state["dismissed"] = False
        print(f"⚡ [VA ĐẬP PHÁT HIỆN] Lực va đập: {recent_impact['impact_g']}g (Ghi nhớ trong {FALL_MEMORY_WINDOW_SECONDS}s)")

    # 2. Kiểm tra tư thế nằm và thời gian va chạm
    time_since_impact = now - recent_impact.get("time", 0.0)
    lying_postures = ["nằm ngửa", "nằm úp", "nằm nghiêng trái", "nằm nghiêng phải", "úp mặt xuống bàn"]

    # 3. TỰ ĐỘNG PHỤC HỒI (AUTO-RECOVERY):
    # Khi người dùng đứng dậy (thân đứng thẳng tuyệt đối, không còn nằm):
    if current_posture == "đứng thẳng":
        if fall_state["alert"] or fall_state.get("dismissed", False):
            print(f"✅ [TỰ ĐỘNG PHỤC HỒI] Người dùng đã đứng dậy ({current_posture}). Reset trạng thái, sẵn sàng cho lần té ngã tiếp theo!")
            fall_state["alert"] = False
            fall_state["dismissed"] = False

    # 4. Tư thế té ngã: hoặc thuộc lying_postures, hoặc khi thân đổ gục (không phải đứng thẳng)
    is_fallen = (current_posture in lying_postures) or (abs(last_ax) < 0.60 and current_posture != "đứng thẳng")

    # Nếu vừa có va chạm VÀ tư thế hiện tại là Nằm / Đổ gục
    if time_since_impact <= FALL_MEMORY_WINDOW_SECONDS and is_fallen:
        if not fall_state["alert"] and not fall_state.get("dismissed", False):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            impact_val = recent_impact.get("impact_g", max_impact)
            fall_state["alert"] = True
            fall_state["time"] = now_str
            fall_state["impact_g"] = impact_val if impact_val > 0 else round(max_impact, 2)
            fall_state["posture_after"] = current_posture

            print(f"🚨 [CẢNH BÁO TÉ NGÃ ĐÃ KÍCH HOẠT!] Thời gian: {now_str} | Lực va đập: {fall_state['impact_g']}g | Tư thế: {current_posture}")

            # Reset bộ nhớ va đập để không kích hoạt lặp
            recent_impact["time"] = 0.0
            recent_impact["impact_g"] = 0.0

            # Gửi email cảnh báo khẩn cấp
            if (now - fall_state["last_email_time"]) > FALL_EMAIL_COOLDOWN_SECONDS:
                fall_state["last_email_time"] = now
                fall_state["email_status"] = "Đang gửi email..."

                def on_email_done(success, message):
                    fall_state["email_status"] = "Đã gửi email thành công" if success else f"Lỗi gửi email: {message}"

                send_fall_alert_email({
                    "time": now_str,
                    "impact_g": fall_state["impact_g"],
                    "posture": current_posture,
                    "bpm": current_bpm
                }, callback=on_email_done)
            else:
                print(f"[Email Alert] Đang trong thời gian cooldown ({FALL_EMAIL_COOLDOWN_SECONDS}s), tạm hoãn gửi email.")

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
    global latest_data, last_data_at, fall_state

    buffer.clear()
    recent_samples_buffer.clear()
    latest_data = DEFAULT_DATA.copy()
    last_data_at = None
    fall_state = {
        "alert": False,
        "time": None,
        "impact_g": 0.0,
        "posture_after": None,
        "email_status": "Chưa có sự cố",
        "last_email_time": 0.0,
        "dismissed": False
    }


def get_live_data():
    if last_data_at is None:
        data = latest_data.copy()
        data["fall_alert"] = fall_state["alert"]
        data["impact_g"] = fall_state["impact_g"]
        data["fall_time"] = fall_state["time"]
        data["email_status"] = fall_state["email_status"]
        return data

    if time.monotonic() - last_data_at > DATA_TIMEOUT_SECONDS:
        # Giữ lại trạng thái fall alert nếu đang có cảnh báo chưa tắt
        saved_fall = fall_state.copy()
        reset_runtime_state()
        fall_state.update(saved_fall)

    data = latest_data.copy()
    data["fall_alert"] = fall_state["alert"]
    data["impact_g"] = fall_state["impact_g"]
    data["fall_time"] = fall_state["time"]
    data["email_status"] = fall_state["email_status"]
    return data


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
        "buffer": len(buffer),
        "fall_alert": fall_state["alert"]
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
        "classes_bpm": classes_bpm,
        "fall_alert": fall_state["alert"]
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


@app.route("/dismiss_alert", methods=["POST"])
def dismiss_alert():
    """Tắt còi báo động và xoá trạng thái cảnh báo ngã"""
    global fall_state
    fall_state["alert"] = False
    fall_state["dismissed"] = True
    print("[Alert] Người dùng đã tắt còi báo động té ngã.")
    return jsonify({
        "success": True,
        "message": "Đã tắt cảnh báo té ngã"
    })


@app.route("/test_fall", methods=["POST"])
def test_fall():
    """Kích hoạt sự kiện té ngã nhân tạo để test còi và email"""
    global fall_state
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_bpm = latest_data.get("bpm", "16.0")

    fall_state["alert"] = True
    fall_state["time"] = now_str
    fall_state["impact_g"] = 3.25
    fall_state["posture_after"] = "nằm úp"
    fall_state["dismissed"] = False
    fall_state["email_status"] = "Đang gửi email..."

    def on_email_done(success, message):
        fall_state["email_status"] = "Đã gửi email thành công" if success else f"Lỗi gửi email: {message}"

    send_fall_alert_email({
        "time": now_str,
        "impact_g": 3.25,
        "posture": "nằm úp",
        "bpm": current_bpm
    }, callback=on_email_done)

    return jsonify({
        "success": True,
        "message": "Đã kích hoạt sự kiện té ngã giả lập thành công!"
    })


@app.route("/api/email_config", methods=["GET", "POST"])
def email_config():
    if request.method == "POST":
        data = request.get_json(force=True)
        ok, msg = save_env_vars(data)
        return jsonify({"success": ok, "message": msg})
    else:
        cfg = load_env_vars()
        # Ẩn bớt mật khẩu ứng dụng để bảo mật
        pwd = cfg.get("EMAIL_PASSWORD", "")
        masked_pwd = (pwd[:4] + "*" * (len(pwd) - 4)) if len(pwd) > 4 else ("*" * len(pwd))
        return jsonify({
            "EMAIL_SENDER": cfg.get("EMAIL_SENDER", ""),
            "EMAIL_PASSWORD": masked_pwd,
            "ALERT_RECEIVER_EMAIL": cfg.get("ALERT_RECEIVER_EMAIL", ""),
            "ENABLE_EMAIL_ALERT": cfg.get("ENABLE_EMAIL_ALERT", "true"),
            "GOOGLE_SCRIPT_URL": cfg.get("GOOGLE_SCRIPT_URL", "")
        })


@app.route("/api/test_email", methods=["POST"])
def test_email_api():
    ok, msg = test_email_connection_sync()
    return jsonify({
        "success": ok,
        "message": msg
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

        # Kiểm tra sự kiện té ngã trên mảng mẫu nhận được (kết hợp cờ từ chip ESP32)
        is_chip_impact = bool(payload.get("impact", False))
        chip_impact_g = float(payload.get("impact_g", 0.0))
        check_fall_event(samples, posture_result, bpm_result.get("bpm", "—"), is_chip_impact, chip_impact_g)

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
            "buffer_size": len(buffer),
            "fall_alert": fall_state["alert"],
            "impact_g": fall_state["impact_g"],
            "fall_time": fall_state["time"],
            "posture_detail": posture_result,
            "email_status": fall_state["email_status"]
        }
        last_data_at = time.monotonic()

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
            "received": len(samples),
            "fall_alert": fall_state["alert"],
            "impact_g": fall_state["impact_g"]
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
