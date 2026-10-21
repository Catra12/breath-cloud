/*
 * =============================================
 * BREATH AI CLOUD — ESP32 Client v2
 * =============================================
 * Fix quan trọng so với code cũ:
 *   1. WiFiClientSecure + setInsecure()
 *      → Cho phép HTTPS tới Render.com
 *      → Code cũ dùng HTTPClient thường = fail -1
 *   2. Gyro gửi đúng đơn vị rad/s
 *   3. WiFi tự reconnect khi mất kết nối
 *   4. Retry 3 lần khi gửi batch thất bại
 *      → Xử lý Render cold start (~30s)
 *   5. Tự động phát hiện va đập té ngã in ra Serial
 * =============================================
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>   // BẮT BUỘC để dùng HTTPS
#include <HTTPClient.h>
#include <Wire.h>
#include "MPU6050.h"

MPU6050 mpu;

// =============================================
// CẤU HÌNH — Sửa theo môi trường của bạn
// =============================================

const char* WIFI_SSID     = "Whisper Coffee";
const char* WIFI_PASSWORD = "caphemuoi";

/* 
 * CHỌN ĐƯỜNG DẪN SERVER:
 * - Render Cloud : https://breath-cloud.onrender.com/posture
 * - Local Test   : http://<IP_MAY_TINH>:5000/posture
 */
const char* SERVER_URL    = "https://breath-cloud.onrender.com/posture";

// =============================================
// TIMING & NGƯỠNG TÉ NGÃ
// =============================================

const unsigned long SAMPLE_INTERVAL_MS = 40;     // 25 Hz = mỗi 40ms 1 mẫu
const int           BATCH_SIZE         = 5;      // Gửi mỗi 5 mẫu (200ms/lần)
const int           HTTP_TIMEOUT_MS    = 15000;  // Render free tier có thể chậm
const int           RETRY_COUNT        = 3;      // Số lần retry khi lỗi
const int           RETRY_DELAY_MS     = 3000;   // Chờ 3s giữa các lần retry
const float         FALL_THRESHOLD_G   = 2.4f;   // Ngưỡng phát hiện va đập té ngã (g)

// =============================================
// BUFFER CỤC BỘ
// =============================================

float ax_buf[BATCH_SIZE];
float ay_buf[BATCH_SIZE];
float az_buf[BATCH_SIZE];
float gx_buf[BATCH_SIZE];
float gy_buf[BATCH_SIZE];
float gz_buf[BATCH_SIZE];

int           buf_idx      = 0;
unsigned long lastSampleAt = 0;
int           failCount    = 0;

// =============================================
// SETUP
// =============================================

void setup() {
    Serial.begin(115200);
    delay(300);

    Serial.println("\n==============================");
    Serial.println("  BREATH AI — ESP32 Client v2 ");
    Serial.println("==============================");

    // I2C: SDA=5, SCL=4 (hoặc mặc định SDA=21, SCL=22)
    Wire.begin(5, 4);

    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("[MPU] Thử lại chân mặc định SDA=21, SCL=22...");
        Wire.begin(21, 22);
        delay(100);
        if (!mpu.testConnection()) {
            Serial.println("[MPU] FAIL — Kiểm tra dây SDA/SCL!");
            while (true) { delay(1000); }
        }
    }
    Serial.println("[MPU] OK — Cảm biến MPU6050 đã sẵn sàng!");

    // In ra để kiểm tra giá trị raw khi không chuyển đổi
    debugRawValues();

    connectWiFi();
}

// =============================================
// DEBUG RAW — Chạy 1 lần khi khởi động
// =============================================

void debugRawValues() {
    Serial.println("[DEBUG] Raw values khi khởi động (FS ±2g / ±250dps):");
    int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
    mpu.getMotion6(&ax_r, &ay_r, &az_r, &gx_r, &gy_r, &gz_r);

    float ax = ax_r / 16384.0f;
    float ay = ay_r / 16384.0f;
    float az = az_r / 16384.0f;
    float gx_dps  = gx_r / 131.0f;               // đơn vị DPS
    float gx_rads = gx_dps * (PI / 180.0f);       // đổi sang rad/s

    Serial.printf("  AX=%.4f AY=%.4f AZ=%.4f (g)\n", ax, ay, az);
    Serial.printf("  GX_raw=%d  GX_dps=%.4f  GX_rads=%.5f\n",
        gx_r, gx_dps, gx_rads);
    Serial.println("  → Giá trị gửi lên server sẽ là rad/s (cột GX_rads)");
}

// =============================================
// KẾT NỐI WIFI
// =============================================

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;

    Serial.printf("[WiFi] Đang kết nối: %s\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 40) {
        delay(500);
        Serial.print(".");
        tries++;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[WiFi] OK — IP: %s  RSSI: %d dBm\n",
            WiFi.localIP().toString().c_str(),
            WiFi.RSSI());
    } else {
        Serial.println("[WiFi] TIMEOUT — Không kết nối được WiFi (vẫn xuất qua cáp USB)");
    }
}

// =============================================
// LOOP CHÍNH
// =============================================

void loop() {
    unsigned long now = millis();
    if (now - lastSampleAt < SAMPLE_INTERVAL_MS) return;
    lastSampleAt = now;

    // ── Đọc MPU6050 ──────────────────────────────
    int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
    mpu.getMotion6(&ax_r, &ay_r, &az_r, &gx_r, &gy_r, &gz_r);

    // Accel: ÷ 16384 → đơn vị g (±2g full scale)
    float ax = ax_r / 16384.0f;
    float ay = ay_r / 16384.0f;
    float az = az_r / 16384.0f;

    // Gyro: ÷ 131 → DPS, rồi × π/180 → rad/s
    float gx = (gx_r / 131.0f) * (PI / 180.0f);
    float gy = (gy_r / 131.0f) * (PI / 180.0f);
    float gz = (gz_r / 131.0f) * (PI / 180.0f);

    // Kiểm tra phát hiện va đập té ngã tại chip
    float svm = sqrt(ax * ax + ay * ay + az * az);
    if (svm >= FALL_THRESHOLD_G) {
        Serial.printf("🚨 [CHIP CẢNH BÁO TÉ NGÃ] Va đập cực mạnh: SVM=%.2fg!\n", svm);
    }

    // ── Lưu vào buffer ───────────────────────────
    ax_buf[buf_idx] = ax;
    ay_buf[buf_idx] = ay;
    az_buf[buf_idx] = az;
    gx_buf[buf_idx] = gx;
    gy_buf[buf_idx] = gy;
    gz_buf[buf_idx] = gz;

    buf_idx++;

    // In log ra Serial USB để serial_bridge.py có thể đọc song song
    Serial.printf("BUF=%d  AX=%.3f AY=%.3f AZ=%.3f  GX=%.5f GY=%.5f GZ=%.5f\n",
        buf_idx, ax, ay, az, gx, gy, gz);

    // ── Gửi batch ────────────────────────────────
    if (buf_idx >= BATCH_SIZE) {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[WiFi] Mất kết nối — đang reconnect...");
            connectWiFi();
        }

        if (WiFi.status() == WL_CONNECTED) {
            sendBatch();
        } else {
            Serial.println("[HTTP] Bỏ qua batch Wi-Fi — xuất qua cáp USB");
        }

        buf_idx = 0;
    }
}

// =============================================
// BUILD JSON
// =============================================

String buildJson() {
    String json = "{\"batch\":[";
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (i > 0) json += ",";
        json += "{";
        json += "\"ax\":"  + String(ax_buf[i], 4);
        json += ",\"ay\":" + String(ay_buf[i], 4);
        json += ",\"az\":" + String(az_buf[i], 4);
        json += ",\"gx\":" + String(gx_buf[i], 6);
        json += ",\"gy\":" + String(gy_buf[i], 6);
        json += ",\"gz\":" + String(gz_buf[i], 6);
        json += "}";
    }
    json += "]}";
    return json;
}

// =============================================
// GỬI BATCH — HTTPS POST với retry
// =============================================

void sendBatch() {
    String json = buildJson();

    for (int attempt = 1; attempt <= RETRY_COUNT; attempt++) {
        bool isHttps = String(SERVER_URL).startsWith("https://");

        HTTPClient http;
        bool beginOk = false;

        WiFiClient clientHttp;
        WiFiClientSecure clientHttps;

        if (isHttps) {
            clientHttps.setInsecure();
            clientHttps.setTimeout(HTTP_TIMEOUT_MS / 1000);
            beginOk = http.begin(clientHttps, SERVER_URL);
        } else {
            clientHttp.setTimeout(HTTP_TIMEOUT_MS / 1000);
            beginOk = http.begin(clientHttp, SERVER_URL);
        }

        if (!beginOk) {
            http.end();
            if (attempt < RETRY_COUNT) delay(RETRY_DELAY_MS);
            continue;
        }

        http.addHeader("Content-Type", "application/json");
        http.setTimeout(HTTP_TIMEOUT_MS);

        int code = http.POST(json);

        if (code == 200) {
            // Thành công: Server trả về kết quả tư thế và té ngã ngay lập tức
            String resp = http.getString();
            failCount = 0;
            Serial.printf("[HTTP] 200 OK: %s\n", resp.c_str());
            http.end();
            return;
        } else if (code > 0) {
            Serial.printf("[HTTP] Code %d: %s\n", code, http.getString().c_str());
            failCount++;
            http.end();
            return;
        } else {
            Serial.printf("[HTTP] Lỗi kết nối code=%d (%s) [lần %d/%d]\n",
                code,
                http.errorToString(code).c_str(),
                attempt,
                RETRY_COUNT);
            http.end();

            if (attempt < RETRY_COUNT) {
                delay(RETRY_DELAY_MS);
            }
        }
    }

    failCount++;
    if (failCount >= 5) {
        Serial.println("[WiFi] Quá nhiều lỗi — reconnect WiFi...");
        WiFi.disconnect(true);
        delay(1000);
        connectWiFi();
        failCount = 0;
    }
}
