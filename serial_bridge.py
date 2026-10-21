import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import json
import math
import re
import time
import requests
import serial
import serial.tools.list_ports


def find_default_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "COM3" in p.device:
            return "COM3"
    if ports:
        return ports[0].device
    return "COM3"


def parse_sensor_line(line):
    """
    Hỗ trợ tự động nhận diện nhiều định dạng dữ liệu từ chip:
    1. JSON: {"ax": 0.1, "ay": 0.2, "az": 0.98, "gx": 0, "gy": 0, "gz": 0}
    2. CSV: 0.1, 0.2, 0.98, 0.01, 0.02, 0.01
    3. Key-Value: ax: 0.1 ay: 0.2 az: 0.98 gx: 0 gy: 0 gz: 0
    """
    line = line.strip()
    if not line:
        return None

    # 1. Thử parse JSON
    if line.startswith("{") and line.endswith("}"):
        try:
            data = json.loads(line)
            if all(k in data for k in ["ax", "ay", "az"]):
                return normalize_sample(data)
        except Exception:
            pass

    # 2. Thử parse dạng Key-Value (ax:..., ay:...)
    kv_pattern = r"(ax|ay|az|gx|gy|gz)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    matches = re.findall(kv_pattern, line, re.IGNORECASE)
    if matches and len(matches) >= 3:
        d = {k.lower(): float(v) for k, v in matches}
        return normalize_sample({
            "ax": d.get("ax", 0.0),
            "ay": d.get("ay", 0.0),
            "az": d.get("az", 0.0),
            "gx": d.get("gx", 0.0),
            "gy": d.get("gy", 0.0),
            "gz": d.get("gz", 0.0),
        })

    # 3. Thử parse dạng CSV phân tách bằng dấu phẩy, khoảng trắng hoặc tab
    parts = re.split(r"[,;\s\t]+", line)
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass

    if len(nums) >= 3:
        sample = {
            "ax": nums[0],
            "ay": nums[1],
            "az": nums[2],
            "gx": nums[3] if len(nums) > 3 else 0.0,
            "gy": nums[4] if len(nums) > 4 else 0.0,
            "gz": nums[5] if len(nums) > 5 else 0.0,
        }
        return normalize_sample(sample)

    return None


def normalize_sample(sample):
    """
    Tự động chuẩn hóa đơn vị gia tốc về g (1.0g):
    - Nếu đơn vị là m/s² (mag ~ 9.8) -> chia cho 9.80665
    - Nếu đơn vị là ADC 16-bit raw (mag > 1000) -> chia cho 16384.0
    """
    ax = float(sample.get("ax", 0.0))
    ay = float(sample.get("ay", 0.0))
    az = float(sample.get("az", 0.0))
    gx = float(sample.get("gx", 0.0))
    gy = float(sample.get("gy", 0.0))
    gz = float(sample.get("gz", 0.0))

    mag = math.sqrt(ax**2 + ay**2 + az**2)

    # Nếu đang ở đơn vị m/s² (~ 9.8)
    if 5.0 < mag < 50.0:
        ax /= 9.80665
        ay /= 9.80665
        az /= 9.80665
    # Nếu đang ở đơn vị raw ADC 16-bit (~ 16384 cho 1g)
    elif mag > 1000.0:
        ax /= 16384.0
        ay /= 16384.0
        az /= 16384.0

    return {
        "ax": round(ax, 4),
        "ay": round(ay, 4),
        "az": round(az, 4),
        "gx": round(gx, 4),
        "gy": round(gy, 4),
        "gz": round(gz, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Serial Bridge: Đọc dữ liệu từ Chip USB (COM3) và gửi lên Web Server.")
    parser.add_argument("--port", default=find_default_port(), help="Cổng COM (Mặc định: COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Tốc độ Baud (Mặc định: 115200)")
    parser.add_argument("--url", default="http://127.0.0.1:5000/posture", help="URL Server nhận dữ liệu")
    parser.add_argument("--batch-size", type=int, default=5, help="Số mẫu gửi mỗi đợt (Mặc định: 5)")
    args = parser.parse_args()

    print("=" * 65)
    print("      BREATH AI CLOUD - SERIAL BRIDGE (USB -> WEB/CLOUD)      ")
    print("=" * 65)
    print(f"Cổng COM : {args.port} | Baud: {args.baud}")
    print(f"Server   : {args.url}")
    print(f"Gom batch: {args.batch_size} mẫu / request")
    print("-" * 65)
    print("⚠️  LƯU Ý QUAN TRỌNG:")
    print("Nếu gặp lỗi 'Access is denied', hãy ĐÓNG Serial Monitor trong Arduino IDE!")
    print("=" * 65)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"\n❌ LỖI MỞ CỔNG {args.port}: {e}")
        if "Access is denied" in str(e) or "PermissionError" in str(e):
            print("\n👉 NGUYÊN NHÂN: Cổng này đang bị Arduino IDE Serial Monitor chiếm giữ!")
            print("👉 KHẮC PHỤC : Vui lòng bấm tắt cửa sổ Serial Monitor trong Arduino IDE, sau đó chạy lại lệnh này.")
        sys.exit(1)

    print(f"✅ Đã kết nối thành công tới {args.port}. Đang đọc dữ liệu từ chip...\n")

    batch = []
    last_print = time.time()

    try:
        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            sample = parse_sensor_line(raw_line)
            if not sample:
                # In ra dòng chưa parse được để người dùng xem định dạng chip đang gửi
                if time.time() - last_print > 2:
                    print(f"[Chip raw]: {raw_line[:80]}")
                    last_print = time.time()
                continue

            batch.append(sample)

            if len(batch) >= args.batch_size:
                try:
                    res = requests.post(args.url, json={"batch": batch}, timeout=1.5)
                    if res.status_code == 200:
                        data = res.json()
                        posture = data.get("posture", "—")
                        bpm = data.get("bpm", "—")
                        buf = data.get("buffer", 0)
                        fall = data.get("fall_alert", False)
                        last_s = batch[-1]
                        alert_tag = " | 🚨 CẢNH BÁO TÉ NGÃ!" if fall else ""
                        print(f"[{time.strftime('%H:%M:%S')}] ax:{last_s['ax']:+0.2f} ay:{last_s['ay']:+0.2f} az:{last_s['az']:+0.2f} | Tư thế: {posture.upper():<18} | Nhịp thở: {bpm:<12} | Buf: {buf}/500{alert_tag}")
                        if fall:
                            try:
                                import winsound
                                winsound.Beep(1200, 300)
                            except Exception:
                                print("\a", end="")
                    else:
                        print(f"⚠️ Server trả về mã lỗi {res.status_code}: {res.text}")
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Không thể kết nối tới server {args.url}: {e}")

                batch = []

    except KeyboardInterrupt:
        print("\n⏹ Đã dừng Serial Bridge.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
