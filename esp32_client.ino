/*
 * ====================================================================
 * BREATH AI CLOUD — ESP32 Client v3 (Dual-Core FreeRTOS Non-Blocking)
 * ====================================================================
 * Cải tiến đột phá v3:
 *   1. Chạy đa nhân FreeRTOS (Core 0 + Core 1):
 *      - Core 0: Đọc cảm biến liên tục mỗi 40ms (25 Hz) không bao giờ bị dừng
 *      - Core 1: Gửi HTTP/HTTPS qua Wi-Fi
 *      → Dù HTTPS mất 500ms-1500ms phản hồi, Core 0 VẪN ĐỌC LIÊN TỤC,
 *        không bị mù (blind spot), BẮT CHUẨN CÚ NGÃ NGAY LẦN ĐẦU TIÊN!
 *   2. Chốt va đập tức thì (Latched Impact):
 *      Bất kỳ cú rơi / va đập nào (>= 1.35g hoặc rơi tự do <= 0.65g) đều
 *      được ghi nhớ và gửi ngay cờ {"impact": true, "impact_g": ...} lên Server.
 *   3. WiFiClientSecure + setInsecure() cho Render Cloud.
 * ====================================================================
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>   // BẮT BUỘC để dùng HTTPS tới Render
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
 * - Local Test   : http://192.168.1.94:5000/posture
 */
const char* SERVER_URL    = "https://breath-cloud.onrender.com/posture";

// =============================================
// TIMING & NGƯỠNG TÉ NGÃ
// =============================================

const unsigned long SAMPLE_INTERVAL_MS = 40;     // 25 Hz = mỗi 40ms 1 mẫu
const int           BATCH_SIZE         = 5;      // Gửi mỗi 5 mẫu (200ms/lần)
const int           HTTP_TIMEOUT_MS    = 6000;   // Timeout 6 giây
const int           RETRY_COUNT        = 2;      // Số lần retry
const int           RETRY_DELAY_MS     = 1000;   // Chờ 1s giữa các lần retry
const float         FALL_THRESHOLD_G   = 1.80f;  // Ngưỡng va đập thực tế (g) - chống báo bậy khi chạm nhẹ

// =============================================
// BUFFER & BIẾN ĐỒNG BỘ DUAL-CORE
// =============================================

float ax_buf[BATCH_SIZE];
float ay_buf[BATCH_SIZE];
float az_buf[BATCH_SIZE];
float gx_buf[BATCH_SIZE];
float gy_buf[BATCH_SIZE];
float gz_buf[BATCH_SIZE];

float send_ax[BATCH_SIZE];
float send_ay[BATCH_SIZE];
float send_az[BATCH_SIZE];
float send_gx[BATCH_SIZE];
float send_gy[BATCH_SIZE];
float send_gz[BATCH_SIZE];

volatile int   buf_idx         = 0;
volatile bool  batch_ready     = false;
volatile bool  latched_impact  = false;
volatile float max_impact_seen = 0.0f;
int            failCount       = 0;

TaskHandle_t TaskSampling = NULL;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

// =============================================
// KHAI BÁO HÀM
// =============================================
void connectWiFi();
void debugRawValues();
void sendBatch(bool has_impact, float peak_g);
String buildJson(bool has_impact, float peak_g);

// =============================================
// TASK ĐỌC CẢM BIẾN TRÊN CORE 0 (ĐỘC LẬP 100%)
// =============================================

void samplingTask(void * pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(SAMPLE_INTERVAL_MS);

    for (;;) {
        int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
        mpu.getMotion6(&ax_r, &ay_r, &az_r, &gx_r, &gy_r, &gz_r);

        float ax = ax_r / 16384.0f;
        float ay = ay_r / 16384.0f;
        float az = az_r / 16384.0f;
        float gx = (gx_r / 131.0f) * (PI / 180.0f);
        float gy = (gy_r / 131.0f) * (PI / 180.0f);
        float gz = (gz_r / 131.0f) * (PI / 180.0f);

        // Tính lực gia tốc toàn phần SVM & vận tốc góc
        float svm = sqrt(ax * ax + ay * ay + az * az);
        float gyro_mag = sqrt(gx * gx + gy * gy + gz * gz);

        // Bắt đỉnh va đập tức thì
        if (svm > max_impact_seen) {
            max_impact_seen = svm;
        }

        // Bắt sự kiện va chạm thật sự (tránh nhầm khi chỉ chạm tay nhẹ vào vỏ chip):
        // 1. Va đập mạnh (SVM >= 1.80g khi rơi)
        // 2. Va đập vừa kèm xoay người dữ dội (SVM >= 1.60g và Gyro >= 2.0 rad/s)
        // 3. Rơi tự do không trọng lượng thực sự (SVM <= 0.40g)
        if (svm >= FALL_THRESHOLD_G || (svm >= 1.60f && gyro_mag >= 2.0f) || svm <= 0.40f) {
            latched_impact = true;
            Serial.printf("⚡ [CORE 0 BẮT VA ĐẬP] SVM=%.2fg (Gyro=%.2f rad/s)\n", svm, gyro_mag);
        }

        portENTER_CRITICAL(&mux);
        if (buf_idx < BATCH_SIZE) {
            ax_buf[buf_idx] = ax;
            ay_buf[buf_idx] = ay;
            az_buf[buf_idx] = az;
            gx_buf[buf_idx] = gx;
            gy_buf[buf_idx] = gy;
            gz_buf[buf_idx] = gz;
            buf_idx++;

            if (buf_idx >= BATCH_SIZE) {
                batch_ready = true;
            }
        }
        portEXIT_CRITICAL(&mux);

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

// =============================================
// SETUP
// =============================================

void setup() {
    Serial.begin(115200);
    delay(300);

    Serial.println("\n============================================");
    Serial.println("  BREATH AI — ESP32 Dual-Core Non-Blocking  ");
    Serial.println("============================================");

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

    debugRawValues();
    connectWiFi();

    // Tạo Task chạy riêng biệt trên Core 0 để đọc mẫu liên tục 25 Hz
    xTaskCreatePinnedToCore(
        samplingTask,
        "SamplingTask",
        4096,
        NULL,
        2,
        &TaskSampling,
        0   // Ghim vào Core 0 (Wi-Fi chạy trên Core 1)
    );
    Serial.println("[SYSTEM] Task đọc cảm biến Core 0 đã khởi chạy thành công!");
}

// =============================================
// DEBUG RAW
// =============================================

void debugRawValues() {
    int16_t ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
    mpu.getMotion6(&ax_r, &ay_r, &az_r, &gx_r, &gy_r, &gz_r);

    float ax = ax_r / 16384.0f;
    float ay = ay_r / 16384.0f;
    float az = az_r / 16384.0f;
    float gx_rads = (gx_r / 131.0f) * (PI / 180.0f);

    Serial.printf("[DEBUG] Khởi động: AX=%.3f AY=%.3f AZ=%.3f (g) | GX=%.4f (rad/s)\n",
        ax, ay, az, gx_rads);
}

// =============================================
// KẾT NỐI WIFI
// =============================================

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;

    Serial.printf("[WiFi] Đang kết nối: %s\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 30) {
        delay(300);
        Serial.print(".");
        tries++;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[WiFi] OK — IP: %s  RSSI: %d dBm\n",
            WiFi.localIP().toString().c_str(),
            WiFi.RSSI());
    } else {
        Serial.println("[WiFi] TIMEOUT — Tiếp tục thử lại ở background");
    }
}

// =============================================
// LOOP CHÍNH (CHẠY TRÊN CORE 1)
// =============================================

void loop() {
    bool ready_to_send = false;
    bool send_impact = false;
    float send_impact_g = 0.0f;

    portENTER_CRITICAL(&mux);
    if (batch_ready) {
        memcpy(send_ax, ax_buf, sizeof(ax_buf));
        memcpy(send_ay, ay_buf, sizeof(ay_buf));
        memcpy(send_az, az_buf, sizeof(az_buf));
        memcpy(send_gx, gx_buf, sizeof(gx_buf));
        memcpy(send_gy, gy_buf, sizeof(gy_buf));
        memcpy(send_gz, gz_buf, sizeof(gz_buf));

        send_impact = latched_impact;
        send_impact_g = max_impact_seen;

        // Reset buffer cho Core 0 tiếp tục lưu mẫu
        latched_impact = false;
        max_impact_seen = 0.0f;
        buf_idx = 0;
        batch_ready = false;
        ready_to_send = true;
    }
    portEXIT_CRITICAL(&mux);

    if (ready_to_send) {
        if (WiFi.status() != WL_CONNECTED) {
            connectWiFi();
        }

        if (WiFi.status() == WL_CONNECTED) {
            sendBatch(send_impact, send_impact_g);
        } else {
            // Khi mất Wi-Fi, in ra Serial để serial_bridge.py bắt
            Serial.printf("USB_DATA: AX=%.3f AY=%.3f AZ=%.3f IMPACT=%d\n",
                send_ax[BATCH_SIZE-1], send_ay[BATCH_SIZE-1], send_az[BATCH_SIZE-1], send_impact);
        }
    }

    delay(10);
}

// =============================================
// BUILD JSON
// =============================================

String buildJson(bool has_impact, float peak_g) {
    String json = "{\"impact\":";
    json += has_impact ? "true" : "false";
    json += ",\"impact_g\":" + String(peak_g, 2);
    json += ",\"batch\":[";
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (i > 0) json += ",";
        json += "{";
        json += "\"ax\":"  + String(send_ax[i], 4);
        json += ",\"ay\":" + String(send_ay[i], 4);
        json += ",\"az\":" + String(send_az[i], 4);
        json += ",\"gx\":" + String(send_gx[i], 6);
        json += ",\"gy\":" + String(send_gy[i], 6);
        json += ",\"gz\":" + String(send_gz[i], 6);
        json += "}";
    }
    json += "]}";
    return json;
}

// =============================================
// GỬI BATCH (CHẠY TRÊN CORE 1 — KHÔNG ẢNH HƯỞNG CORE 0)
// =============================================

void sendBatch(bool has_impact, float peak_g) {
    String json = buildJson(has_impact, peak_g);

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
            String resp = http.getString();
            failCount = 0;
            Serial.printf("[HTTP] 200 OK: %s\n", resp.c_str());
            http.end();
            return;
        } else if (code > 0) {
            Serial.printf("[HTTP] Code %d\n", code);
            failCount++;
            http.end();
            return;
        } else {
            http.end();
            if (attempt < RETRY_COUNT) {
                delay(RETRY_DELAY_MS);
            }
        }
    }

    failCount++;
    if (failCount >= 6) {
        Serial.println("[WiFi] Reconnect WiFi...");
        WiFi.disconnect(true);
        delay(500);
        connectWiFi();
        failCount = 0;
    }
}
