import json
import os
import smtplib
import threading
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


ENV_FILE = Path(__file__).resolve().parent / ".env"


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        try:
            text = " ".join(str(a) for a in args)
            print(text.encode("ascii", errors="replace").decode("ascii"), **kwargs)
        except Exception:
            pass


def load_env_vars():
    config = {
        "SMTP_SERVER": "smtp.gmail.com",
        "SMTP_PORT": "587",
        "EMAIL_SENDER": "",
        "EMAIL_PASSWORD": "",
        "ALERT_RECEIVER_EMAIL": "",
        "ENABLE_EMAIL_ALERT": "true",
        "GOOGLE_SCRIPT_URL": "",
    }
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        config[k.strip()] = v.strip().strip("\"'")
        except Exception as e:
            safe_print(f"[Email Config] Lỗi đọc .env: {e}")

    # Ưu tiên biến môi trường từ hệ thống (Render Cloud, Docker, OS)
    for k in config.keys():
        env_val = os.environ.get(k)
        if env_val and env_val.strip():
            config[k] = env_val.strip().strip("\"'")

    return config


def save_env_vars(new_config):
    current = load_env_vars()
    current.update({k: str(v) for k, v in new_config.items() if v is not None})
    for k, v in current.items():
        os.environ[k] = str(v)
    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write("# ==================================================\n")
            f.write("# BREATH AI CLOUD - CẤU HÌNH GỬI EMAIL CẢNH BÁO TÉ NGÃ\n")
            f.write("# Hướng dẫn Gmail: Tạo Mật khẩu ứng dụng (App Password 16 chữ cái)\n")
            f.write("# tại: https://myaccount.google.com/apppasswords\n")
            f.write("# ==================================================\n\n")
            for k, v in current.items():
                f.write(f"{k}={v}\n")
        return True, "Lưu cấu hình thành công"
    except Exception as e:
        return False, f"Lỗi lưu file .env: {e}"


def _send_via_google_script(script_url, receiver, subject, html_body):
    """
    Gửi email qua Google Apps Script Webhook (Port 443 HTTPS - Không bao giờ bị chặn trên Render Cloud).
    """
    try:
        payload = json.dumps({
            "to": receiver,
            "subject": subject,
            "html": html_body
        }).encode("utf-8")
        req = urllib.request.Request(
            script_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "BreathAI/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True, "Đã gửi thành công qua Google Apps Script (HTTPS Port 443)"
    except Exception as e:
        return False, f"Lỗi Google Apps Script: {e}"


def _send_email_thread(subject, html_body, callback=None):
    config = load_env_vars()
    sender = config.get("EMAIL_SENDER", "").strip()
    password = config.get("EMAIL_PASSWORD", "").strip()
    receiver = config.get("ALERT_RECEIVER_EMAIL", "").strip()
    smtp_server = config.get("SMTP_SERVER", "smtp.gmail.com").strip()
    smtp_port = int(config.get("SMTP_PORT", 587))
    google_script_url = config.get("GOOGLE_SCRIPT_URL", "").strip()
    enabled = config.get("ENABLE_EMAIL_ALERT", "true").lower() in ("true", "1", "yes")

    if not enabled:
        safe_print("[Email Alert] Tính năng email đang tắt (ENABLE_EMAIL_ALERT=false)")
        if callback:
            callback(False, "Tính năng gửi email đang bị tắt trong cấu hình.")
        return

    if not receiver:
        err_msg = "Chưa cấu hình Email người nhận (ALERT_RECEIVER_EMAIL)."
        safe_print(f"[Email Alert] ⚠️ {err_msg}")
        if callback:
            callback(False, err_msg)
        return

    # 1. Ưu tiên gửi qua Google Apps Script Webhook (HTTPS Port 443 - không bị Render chặn)
    if google_script_url.startswith("http"):
        safe_print(f"[Email Alert] Đang gửi qua Google Apps Script Webhook tới {receiver}...")
        ok, msg = _send_via_google_script(google_script_url, receiver, subject, html_body)
        if ok:
            safe_print(f"[Email Alert] ✅ Đã gửi email thành công qua Google Apps Script tới {receiver}!")
            if callback:
                callback(True, f"Đã gửi thành công tới {receiver} (qua Google Apps Script)")
            return
        else:
            safe_print(f"[Email Alert] ⚠️ Gửi qua Google Apps Script thất bại: {msg}. Thử lại qua SMTP...")

    # 2. Gửi qua SMTP truyền thống (Gmail SMTP port 587)
    if not sender or not password:
        err_msg = "Chưa cấu hình đầy đủ EMAIL_SENDER hoặc EMAIL_PASSWORD."
        safe_print(f"[Email Alert] ⚠️ {err_msg}")
        if callback:
            callback(False, err_msg)
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Breath AI Alert System <{sender}>"
        msg["To"] = receiver

        part = MIMEText(html_body, "html", "utf-8")
        msg.attach(part)

        safe_print(f"[Email Alert] Đang gửi email tới {receiver} qua {smtp_server}:{smtp_port}...")
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=12)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()

        safe_print(f"[Email Alert] ✅ Đã gửi email cảnh báo thành công tới {receiver}!")
        if callback:
            callback(True, f"Đã gửi thành công tới {receiver}")

    except OSError as e:
        err_str = str(e).lower()
        if "101" in err_str or "unreachable" in err_str or getattr(e, 'errno', None) == 101:
            err_text = (
                "[Errno 101] Network unreachable: Render Cloud Free chặn cổng SMTP 587. "
                "Hãy chuyển ESP32 sang Local (http://192.168.1.94:5000) hoặc dùng Google Apps Script Webhook!"
            )
        else:
            err_text = f"Lỗi mạng SMTP: {type(e).__name__}: {e}"
        safe_print(f"[Email Alert] ❌ {err_text}")
        if callback:
            callback(False, err_text)

    except Exception as e:
        err_text = f"Lỗi kết nối SMTP/Gửi thư: {type(e).__name__}: {e}"
        safe_print(f"[Email Alert] ❌ {err_text}")
        if callback:
            callback(False, err_text)


def send_fall_alert_email(fall_info, callback=None):
    """
    Gửi email cảnh báo té ngã không làm nghẽn luồng xử lý chính.
    """
    time_str = fall_info.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    impact_g = fall_info.get("impact_g", 0.0)
    posture = fall_info.get("posture", "Không xác định")
    bpm = fall_info.get("bpm", "Đang đo")
    subject = f"🚨 [CẢNH BÁO TÉ NGÃ KHẨN CẤP] Phát hiện té ngã lúc {time_str}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #0b1120; color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; background: #1e293b; border-radius: 12px; border: 2px solid #ef4444; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
            .header {{ background: linear-gradient(135deg, #b91c1c, #ef4444); padding: 24px; text-align: center; color: white; }}
            .header h1 {{ margin: 0; font-size: 24px; letter-spacing: 0.5px; text-transform: uppercase; }}
            .header p {{ margin: 6px 0 0; font-size: 14px; opacity: 0.9; }}
            .body {{ padding: 24px; color: #e2e8f0; line-height: 1.6; }}
            .alert-box {{ background: rgba(239, 68, 68, 0.15); border-left: 4px solid #ef4444; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; font-size: 15px; font-weight: bold; color: #fca5a5; }}
            .grid {{ display: table; width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .row {{ display: table-row; border-bottom: 1px solid #334155; }}
            .cell-label {{ display: table-cell; padding: 12px 8px; color: #94a3b8; font-weight: 600; width: 40%; }}
            .cell-value {{ display: table-cell; padding: 12px 8px; color: #ffffff; font-weight: bold; font-size: 16px; }}
            .badge-danger {{ color: #ef4444; font-size: 18px; }}
            .footer {{ background: #0f172a; padding: 16px; text-align: center; font-size: 12px; color: #64748b; }}
            .btn {{ display: inline-block; background: #ef4444; color: white; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: bold; margin-top: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚨 CẢNH BÁO TÉ NGÃ PHÁT HIỆN</h1>
                <p>Hệ thống giám sát hô hấp & tư thế thông minh Breath AI Cloud</p>
            </div>
            <div class="body">
                <div class="alert-box">
                    ⚠️ PHÁT HIỆN SỰ CỐ TÉ NGÃ ĐỘT NGỘT! Vui lòng kiểm tra ngay người bệnh hoặc người cao tuổi!
                </div>
                
                <div class="grid">
                    <div class="row">
                        <div class="cell-label">Thời gian phát hiện:</div>
                        <div class="cell-value">{time_str}</div>
                    </div>
                    <div class="row">
                        <div class="cell-label">Lực va đập (Impact G):</div>
                        <div class="cell-value badge-danger">{impact_g:.2f} G</div>
                    </div>
                    <div class="row">
                        <div class="cell-label">Tư thế sau va chạm:</div>
                        <div class="cell-value" style="color:#fbbf24;">{posture.upper()}</div>
                    </div>
                    <div class="row">
                        <div class="cell-label">Trạng thái hô hấp (BPM):</div>
                        <div class="cell-value" style="color:#34d399;">{bpm}</div>
                    </div>
                </div>

                <p style="margin-top:20px; font-size: 14px; color:#cbd5e1;">
                    Hệ thống đo được gia tốc cơ thể thay đổi vượt ngưỡng an toàn kèm biến thiên vận tốc góc cao, sau đó cơ thể chuyển sang trạng thái bất động hoặc nằm sấp/nằm ngửa.
                </p>
            </div>
            <div class="footer">
                Tin nhắn tự động từ Hệ thống Breath AI Cloud IoT • Đừng trả lời trực tiếp email này
            </div>
        </div>
    </body>
    </html>
    """

    t = threading.Thread(
        target=_send_email_thread,
        args=(subject, html_content, callback),
        daemon=True
    )
    t.start()
    return t


def send_test_email(callback=None):
    """
    Gửi email thử nghiệm cấu hình qua luồng ngầm.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"✅ [KIỂM TRA HỆ THỐNG] Kiểm tra kết nối Email Breath AI ({now})"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #0b1120; color: #f8fafc; padding: 20px;">
        <div style="max-width: 500px; margin: 0 auto; background: #1e293b; border-radius: 10px; border: 2px solid #10b981; padding: 24px; text-align: center;">
            <h2 style="color: #10b981; margin-top: 0;">🎉 Kết Nối Email Thành Công!</h2>
            <p style="color: #cbd5e1; font-size: 15px;">
                Hệ thống <b>Breath AI Cloud</b> đã xác thực tài khoản Gmail của bạn thành công.<br>
                Khi có sự cố té ngã xảy ra, thông báo khẩn cấp sẽ lập tức được gửi đến hòm thư này!
            </p>
            <div style="background: rgba(16, 185, 129, 0.1); padding: 10px; border-radius: 6px; font-size: 13px; color: #6ee7b7; margin-top: 15px;">
                Thời gian kiểm tra: {now}
            </div>
        </div>
    </body>
    </html>
    """
    t = threading.Thread(
        target=_send_email_thread,
        args=(subject, html_content, callback),
        daemon=True
    )
    t.start()
    return t


def test_email_connection_sync():
    """
    Kiểm tra kết nối gửi email đồng bộ để phản hồi ngay lập tức cho giao diện người dùng.
    """
    config = load_env_vars()
    sender = config.get("EMAIL_SENDER", "").strip()
    password = config.get("EMAIL_PASSWORD", "").strip()
    receiver = config.get("ALERT_RECEIVER_EMAIL", "").strip()
    smtp_server = config.get("SMTP_SERVER", "smtp.gmail.com").strip()
    smtp_port = int(config.get("SMTP_PORT", 587))

    google_script_url = config.get("GOOGLE_SCRIPT_URL", "").strip()

    if not receiver:
        return False, "Vui lòng nhập Email người nhận cảnh báo (ALERT_RECEIVER_EMAIL)!"

    # 1. Kiểm tra gửi qua Google Apps Script Webhook nếu có cấu hình
    if google_script_url.startswith("http"):
        safe_print(f"[Email Test] Đang kiểm tra kết nối qua Google Apps Script tới {receiver}...")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_subject = f"✅ [THỬ NGHIỆM THÀNH CÔNG] Xác thực Google Apps Script ({now})"
        test_html = f"""
        <div style="font-family: Arial, sans-serif; background:#0f172a; color:#f8fafc; padding:25px; border-radius:10px;">
            <h2 style="color:#10b981;">🎉 KẾT NỐI GOOGLE APPS SCRIPT THÀNH CÔNG!</h2>
            <p>Xin chào! Hệ thống <b>Breath AI Cloud</b> đã gửi thư thử nghiệm thành công qua Google Apps Script Webhook (Port 443 HTTPS) tới hòm thư <b>{receiver}</b>.</p>
            <p>Cơ chế này hoạt động hoàn hảo trên Render Cloud mà không bao giờ bị chặn cổng!</p>
            <p style="color:#94a3b8; font-size:12px;">Thời gian xác thực: {now}</p>
        </div>
        """
        ok, msg = _send_via_google_script(google_script_url, receiver, test_subject, test_html)
        if ok:
            return True, f"✅ ĐÃ GỬI THÀNH CÔNG QUA GOOGLE APPS SCRIPT! Vui lòng kiểm tra hòm thư {receiver}."
        else:
            return False, f"❌ Lỗi khi gửi qua Google Apps Script Webhook: {msg}"

    # 2. Kiểm tra gửi qua SMTP thông thường
    if not sender:
        return False, "Chưa điền Gmail người gửi (EMAIL_SENDER)! Để gửi được email đến người nhận, bạn cần nhập tài khoản Gmail gửi."

    if not password:
        return False, "Chưa điền Mật khẩu ứng dụng Google (16 chữ cái)! Google không cho phép dùng mật khẩu đăng nhập thông thường, bạn cần tạo Mật khẩu ứng dụng tại myaccount.google.com/apppasswords."

    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"✅ [THỬ NGHIỆM THÀNH CÔNG] Kết nối hệ thống Breath AI ({now})"
        msg["From"] = f"Breath AI Alert System <{sender}>"
        msg["To"] = receiver

        html_body = f"""
        <div style="font-family: Arial, sans-serif; background:#0f172a; color:#f8fafc; padding:25px; border-radius:10px;">
            <h2 style="color:#10b981;">🎉 KẾT NỐI EMAIL THÀNH CÔNG!</h2>
            <p>Xin chào! Hệ thống <b>Breath AI Cloud</b> đã gửi thư thử nghiệm thành công tới hòm thư <b>{receiver}</b>.</p>
            <p>Từ bây giờ, khi cảm biến phát hiện sự cố té ngã bất thường, cảnh báo khẩn cấp sẽ lập tức được gửi đến hòm thư này.</p>
            <p style="color:#94a3b8; font-size:12px;">Thời gian xác thực: {now}</p>
        </div>
        """
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        safe_print(f"[Email Test] Đang kết nối tới {smtp_server}:{smtp_port} bằng {sender}...")
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=12)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()

        safe_print(f"[Email Test] ✅ Đã gửi thư thử nghiệm thành công tới {receiver}!")
        return True, f"✅ ĐÃ GỬI THÀNH CÔNG! Vui lòng mở hộp thư đến (hoặc hòm thư Spam) của {receiver} để xem email xác nhận!"

    except smtplib.SMTPAuthenticationError as e:
        err_msg = (
            "❌ Lỗi xác thực tài khoản Google (535 Authentication Failed)!\n"
            "Nguyên nhân: Mật khẩu ứng dụng 16 ký tự của bạn chưa đúng hoặc tài khoản chưa bật Xác thực 2 bước.\n"
            "Cách xử lý: Vào myaccount.google.com/apppasswords để tạo lại chuỗi 16 chữ cái mới và dán vào ô Mật khẩu ứng dụng."
        )
        safe_print(f"[Email Test] ❌ {err_msg}")
        return False, err_msg

    except OSError as e:
        err_str = str(e).lower()
        if "101" in err_str or "unreachable" in err_str or getattr(e, 'errno', None) == 101:
            err_msg = (
                "⚠️ LỖI [Errno 101] Network is unreachable (RENDER CLOUD CHẶN CỔNG SMTP)!\n\n"
                "📌 NGUYÊN NHÂN: Bạn đang chạy trên Render.com gói Free. Tường lửa của Render chặn các cổng gửi thư (587, 465, 25) để chống spam, nên không thể kết nối tới smtp.gmail.com.\n\n"
                "✅ 2 CÁCH KHẮC PHỤC TRIỆT ĐỂ:\n"
                "1. Chạy trên Local (Khuyên dùng): Mở http://localhost:5000 và chuyển ESP32 sang http://192.168.1.94:5000/posture. Mạng máy tính của bạn KHÔNG bị chặn nên gửi email cực mượt!\n"
                "2. Dùng Google Apps Script Webhook: Tạo 1 Web App trên script.google.com (gửi qua cổng 443 HTTPS không bao giờ bị chặn)."
            )
        else:
            err_msg = f"❌ Lỗi mạng SMTP: {type(e).__name__}: {e}"
        safe_print(f"[Email Test] ❌ {err_msg}")
        return False, err_msg

    except Exception as e:
        err_msg = f"❌ Không thể gửi email qua Gmail: {type(e).__name__}: {e}"
        safe_print(f"[Email Test] ❌ {err_msg}")
        return False, err_msg

