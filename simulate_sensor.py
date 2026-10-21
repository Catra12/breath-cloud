import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import math
import random
import time
import requests

SERVER_URL = "http://127.0.0.1:5000/posture"
FS = 25          # 25 Hz
BATCH_SIZE = 25  # Gửi 1 giây dữ liệu mỗi lần (25 mẫu)
TARGET_BPM = 16  # Nhịp thở giả lập (~16 nhịp/phút)


def generate_posture_sample(posture, t, freq):
    breath_wave = 0.045 * math.sin(2 * math.pi * freq * t)

    if posture == "nam_up":
        # Nằm úp trên giường: ngực ép xuống sàn (az ~ -1g)
        ax = random.gauss(0, 0.01)
        ay = random.gauss(0, 0.01)
        az = -0.98 - breath_wave + random.gauss(0, 0.003)
    elif posture == "up_mat_ban":
        # Nằm úp mặt xuống bàn: ngồi ghế gập người úp ngực lên bàn (ax ~ 0.35g, az ~ -0.75g)
        ax = 0.35 + random.gauss(0, 0.01)
        ay = random.gauss(0, 0.01)
        az = -0.75 - breath_wave + random.gauss(0, 0.003)
    elif posture == "nam_nghieng":
        # Nằm nghiêng: trọng lực rơi vào trục Y
        ax = random.gauss(0, 0.01)
        ay = 0.95 + random.gauss(0, 0.01)
        az = breath_wave + random.gauss(0, 0.003)
    elif posture == "dung":
        # Đứng thẳng: trọng lực rơi vào trục X (ax ~ 0.98g)
        ax = 0.98 + random.gauss(0, 0.01)
        ay = random.gauss(0, 0.01)
        az = breath_wave + random.gauss(0, 0.003)
    elif posture == "ngoi":
        # Ngồi thẳng: ax ~ 0.70g, az ~ 0.20g
        ax = 0.70 + random.gauss(0, 0.01)
        ay = random.gauss(0, 0.01)
        az = 0.20 + breath_wave + random.gauss(0, 0.003)
    else:  # nam_ngua (mặc định)
        # Nằm ngửa trên giường: ngực hướng lên trời (az ~ +0.98g)
        ax = random.gauss(0, 0.01)
        ay = random.gauss(0, 0.01)
        az = 0.98 + breath_wave + random.gauss(0, 0.003)

    return {
        "ax": round(ax, 4),
        "ay": round(ay, 4),
        "az": round(az, 4),
        "gx": round(random.gauss(0, 0.002), 4),
        "gy": round(0.01 * math.cos(2 * math.pi * freq * t) + random.gauss(0, 0.002), 4),
        "gz": round(random.gauss(0, 0.002), 4),
    }


def generate_fall_batch():
    """Tạo 25 mẫu mô phỏng cú té ngã: Rơi tự do (0.3g) -> Va đập mạnh (3.2g) -> Nằm sấp bất động"""
    batch = []
    # 5 mẫu đầu: rơi tự do
    for _ in range(5):
        batch.append({
            "ax": round(random.gauss(0.1, 0.05), 4),
            "ay": round(random.gauss(0.1, 0.05), 4),
            "az": round(random.gauss(0.3, 0.05), 4),  # Rơi tự do trọng lượng giảm
            "gx": round(random.gauss(1.5, 0.2), 4),
            "gy": round(random.gauss(2.0, 0.3), 4),
            "gz": round(random.gauss(1.8, 0.2), 4),
        })

    # 5 mẫu tiếp theo: va đập cực mạnh xuống nền nhà
    for _ in range(5):
        batch.append({
            "ax": round(random.gauss(1.2, 0.3), 4),
            "ay": round(random.gauss(0.8, 0.2), 4),
            "az": round(random.gauss(-3.1, 0.4), 4),  # Đỉnh va đập SVM ~ 3.4g
            "gx": round(random.gauss(3.5, 0.5), 4),
            "gy": round(random.gauss(4.2, 0.5), 4),
            "gz": round(random.gauss(3.0, 0.5), 4),
        })

    # 15 mẫu sau: nằm úp bất động trên sàn nhà
    for _ in range(15):
        batch.append({
            "ax": round(random.gauss(0.05, 0.01), 4),
            "ay": round(random.gauss(0.05, 0.01), 4),
            "az": round(random.gauss(-0.98, 0.01), 4),  # Nằm úp
            "gx": round(random.gauss(0, 0.001), 4),
            "gy": round(random.gauss(0, 0.001), 4),
            "gz": round(random.gauss(0, 0.001), 4),
        })

    return batch


def main():
    parser = argparse.ArgumentParser(description="Script Giả Lập Cảm Biến Hô Hấp, Đa Tư Thế & Té Ngã")
    parser.add_argument("--url", default=SERVER_URL, help="URL nhận dữ liệu")
    parser.add_argument(
        "--posture",
        default="nam_ngua",
        choices=["nam_ngua", "nam_up", "up_mat_ban", "nam_nghieng", "dung", "ngoi"],
        help="Tư thế mô phỏng ban đầu",
    )
    parser.add_argument("--fall-at", type=int, default=0, help="Tự động kích hoạt té ngã sau X giây (0 = không tự kích hoạt)")
    args = parser.parse_args()

    print("=" * 65)
    print("      BREATH AI CLOUD — SENSOR SIMULATOR (ĐA TƯ THẾ & TÉ NGÃ)     ")
    print("=" * 65)
    print(f"URL Server: {args.url}")
    print(f"Tư thế    : {args.posture}")
    if args.fall_at > 0:
        print(f"Té ngã    : Sẽ kích hoạt sau {args.fall_at} giây!")
    print("=" * 65)

    freq = TARGET_BPM / 60.0
    sample_index = 0
    start_time = time.time()
    fall_triggered = False
    current_posture = args.posture

    try:
        while True:
            elapsed = time.time() - start_time

            # Kích hoạt té ngã nếu đến thời gian
            if args.fall_at > 0 and elapsed >= args.fall_at and not fall_triggered:
                print("\n🚨 [MÔ PHỎNG] KÍCH HOẠT SỰ CỐ TÉ NGÃ ĐỘT NGỘT!")
                batch = generate_fall_batch()
                fall_triggered = True
                current_posture = "nam_up"
            else:
                batch = []
                for _ in range(BATCH_SIZE):
                    t = sample_index / float(FS)
                    sample = generate_posture_sample(current_posture, t, freq)
                    batch.append(sample)
                    sample_index += 1

            try:
                res = requests.post(args.url, json={"batch": batch}, timeout=2)
                if res.status_code == 200:
                    data = res.json()
                    alert_tag = " 🚨[FALL ALERT!]" if data.get("fall_alert") else ""
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Buf: {data.get('buffer')}/500 | "
                        f"Tư thế: {data.get('posture'):<18} | "
                        f"BPM: {data.get('bpm'):<12}{alert_tag}"
                    )
                else:
                    print(f"Lỗi HTTP {res.status_code}: {res.text}")
            except requests.exceptions.RequestException as e:
                print(f"Không thể kết nối đến server {args.url}: {e}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n⏹ Đã dừng giả lập cảm biến.")


if __name__ == "__main__":
    main()
