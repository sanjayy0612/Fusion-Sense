/*
 * FusionSense ESP32-CAM — timestamped 10 FPS JPEG stream
 *
 * Target: AI Thinker ESP32-CAM with OV2640 and PSRAM.
 * Endpoint: http://<device-ip>:81/stream
 *
 * Every multipart JPEG frame includes:
 *   X-Device-Id / X-Session-Id: capture identity
 *   X-Frame-Sequence: monotonically increasing frame number
 *   X-Capture-Timestamp-Us: ESP32 monotonic timestamp from camera_fb_t
 *
 * Control endpoints remain responsive while streaming:
 *   http://<device-ip>/session?id=<session_id>
 *   http://<device-ip>/sync?id=<request_id>
 *   http://<device-ip>/health
 *
 * Copy secrets.example.h to secrets.h and set the local Wi-Fi credentials.
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include <WiFi.h>
#include <cstring>

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef FUSIONSENSE_WIFI_SSID
#define FUSIONSENSE_WIFI_SSID "Fibrsol-2CC88B"
#endif

#ifndef FUSIONSENSE_WIFI_PASSWORD
#define FUSIONSENSE_WIFI_PASSWORD "12345678"
#endif

namespace {

// AI Thinker ESP32-CAM pin map.
constexpr int PWDN_GPIO_NUM = 32;
constexpr int RESET_GPIO_NUM = -1;
constexpr int XCLK_GPIO_NUM = 0;
constexpr int SIOD_GPIO_NUM = 26;
constexpr int SIOC_GPIO_NUM = 27;
constexpr int Y9_GPIO_NUM = 35;
constexpr int Y8_GPIO_NUM = 34;
constexpr int Y7_GPIO_NUM = 39;
constexpr int Y6_GPIO_NUM = 36;
constexpr int Y5_GPIO_NUM = 21;
constexpr int Y4_GPIO_NUM = 19;
constexpr int Y3_GPIO_NUM = 18;
constexpr int Y2_GPIO_NUM = 5;
constexpr int VSYNC_GPIO_NUM = 25;
constexpr int HREF_GPIO_NUM = 23;
constexpr int PCLK_GPIO_NUM = 22;

constexpr uint16_t STREAM_PORT = 81;
constexpr uint16_t CONTROL_PORT = 80;
constexpr uint32_t TARGET_FPS = 10;
constexpr uint64_t FRAME_INTERVAL_US = 1000000ULL / TARGET_FPS;
constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS = 30000;
constexpr size_t IDENTIFIER_CAPACITY = 33;
constexpr char DEVICE_ID[] = "cam01";

constexpr char STREAM_CONTENT_TYPE[] =
    "multipart/x-mixed-replace;boundary=frame";
constexpr char STREAM_BOUNDARY[] = "\r\n--frame\r\n";
constexpr char STREAM_PART_HEADER[] =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "X-Device-Id: %s\r\n"
    "X-Session-Id: %s\r\n"
    "X-Frame-Sequence: %llu\r\n"
    "X-Capture-Timestamp-Us: %llu\r\n\r\n";

httpd_handle_t streamServer = nullptr;
httpd_handle_t controlServer = nullptr;
volatile uint64_t framesCaptured = 0;
volatile uint32_t captureErrors = 0;
volatile uint32_t streamDisconnects = 0;
uint64_t healthWindowStartUs = 0;
uint64_t healthWindowStartFrames = 0;
char sessionId[IDENTIFIER_CAPACITY] = "unassigned";

bool validIdentifier(const char *value) {
  const size_t length = strlen(value);
  if (length == 0 || length >= IDENTIFIER_CAPACITY) {
    return false;
  }
  for (size_t index = 0; index < length; ++index) {
    const char character = value[index];
    const bool allowed =
        (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9') || character == '-' ||
        character == '_' || character == '.';
    if (!allowed) {
      return false;
    }
  }
  return true;
}

bool queryIdentifier(httpd_req_t *request, const char *key, char *destination,
                     size_t capacity) {
  char query[96];
  if (httpd_req_get_url_query_str(request, query, sizeof(query)) != ESP_OK) {
    return false;
  }
  if (httpd_query_key_value(query, key, destination, capacity) != ESP_OK) {
    return false;
  }
  return validIdentifier(destination);
}

uint64_t frameCaptureTimestampUs(const camera_fb_t *frame) {
  if (frame->timestamp.tv_sec != 0 || frame->timestamp.tv_usec != 0) {
    return static_cast<uint64_t>(frame->timestamp.tv_sec) * 1000000ULL +
           static_cast<uint64_t>(frame->timestamp.tv_usec);
  }
  // Defensive fallback for camera-driver builds that do not populate timeval.
  return static_cast<uint64_t>(esp_timer_get_time());
}

esp_err_t healthHandler(httpd_req_t *request) {
  char payload[320];
  const int length = snprintf(
      payload, sizeof(payload),
      "{\"status\":\"ok\",\"schema_version\":1,"
      "\"device_id\":\"%s\",\"session_id\":\"%s\","
      "\"target_fps\":%u,\"frames\":%llu,"
      "\"capture_errors\":%u,\"stream_disconnects\":%u,\"uptime_us\":%llu}",
      DEVICE_ID, sessionId, TARGET_FPS,
      static_cast<unsigned long long>(framesCaptured),
      captureErrors, streamDisconnects,
      static_cast<unsigned long long>(esp_timer_get_time()));
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(request, payload, length);
}

esp_err_t sessionHandler(httpd_req_t *request) {
  char requestedSession[IDENTIFIER_CAPACITY];
  if (!queryIdentifier(request, "id", requestedSession,
                       sizeof(requestedSession))) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "invalid session id");
  }
  strncpy(sessionId, requestedSession, IDENTIFIER_CAPACITY - 1);
  sessionId[IDENTIFIER_CAPACITY - 1] = '\0';
  char payload[160];
  const int length = snprintf(
      payload, sizeof(payload),
      "{\"status\":\"ok\",\"schema_version\":1,\"device_id\":\"%s\","
      "\"session_id\":\"%s\"}",
      DEVICE_ID, sessionId);
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, payload, length);
}

esp_err_t syncHandler(httpd_req_t *request) {
  char requestId[IDENTIFIER_CAPACITY];
  if (!queryIdentifier(request, "id", requestId, sizeof(requestId))) {
    return httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST,
                               "invalid sync request id");
  }
  const uint64_t deviceTimeUs =
      static_cast<uint64_t>(esp_timer_get_time());
  char payload[224];
  const int length = snprintf(
      payload, sizeof(payload),
      "{\"status\":\"ok\",\"schema_version\":1,\"device_id\":\"%s\","
      "\"session_id\":\"%s\",\"request_id\":\"%s\","
      "\"device_time_us\":%llu}",
      DEVICE_ID, sessionId, requestId,
      static_cast<unsigned long long>(deviceTimeUs));
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, payload, length);
}

esp_err_t streamHandler(httpd_req_t *request) {
  httpd_resp_set_type(request, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  Serial.println("# stream_client_connected");

  uint64_t nextFrameDueUs = static_cast<uint64_t>(esp_timer_get_time());
  while (true) {
    const uint64_t nowUs = static_cast<uint64_t>(esp_timer_get_time());
    if (nowUs < nextFrameDueUs) {
      const uint64_t remainingUs = nextFrameDueUs - nowUs;
      if (remainingUs >= 1000) {
        delay(static_cast<uint32_t>(remainingUs / 1000));
      } else {
        delayMicroseconds(static_cast<uint32_t>(remainingUs));
      }
    }

    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
      ++captureErrors;
      Serial.println("# camera_capture_error");
      delay(10);
      nextFrameDueUs = static_cast<uint64_t>(esp_timer_get_time()) +
                       FRAME_INTERVAL_US;
      continue;
    }

    const uint64_t captureTimestampUs = frameCaptureTimestampUs(frame);
    const uint64_t sequence = ++framesCaptured;
    char partHeader[256];
    const int partHeaderLength = snprintf(
        partHeader, sizeof(partHeader), STREAM_PART_HEADER,
        static_cast<unsigned int>(frame->len),
        DEVICE_ID, sessionId,
        static_cast<unsigned long long>(sequence),
        static_cast<unsigned long long>(captureTimestampUs));

    esp_err_t result = httpd_resp_send_chunk(
        request, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, partHeader, partHeaderLength);
    }
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
          request, reinterpret_cast<const char *>(frame->buf), frame->len);
    }
    esp_camera_fb_return(frame);

    if (result != ESP_OK) {
      ++streamDisconnects;
      Serial.println("# stream_client_disconnected");
      break;
    }

    nextFrameDueUs += FRAME_INTERVAL_US;
    const uint64_t afterSendUs = static_cast<uint64_t>(esp_timer_get_time());
    if (afterSendUs > nextFrameDueUs + FRAME_INTERVAL_US) {
      // Do not emit a burst after a slow network send; resume from now.
      nextFrameDueUs = afterSendUs + FRAME_INTERVAL_US;
    }
  }

  return ESP_OK;
}

bool initializeCamera() {
  camera_config_t config{};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 2;
  } else {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
    Serial.println("# warning,psram_not_found,using_single_frame_buffer");
  }

  const esp_err_t result = esp_camera_init(&config);
  if (result != ESP_OK) {
    Serial.printf("# error,camera_init_failed,code=0x%x\n", result);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_framesize(sensor, FRAMESIZE_QVGA);
  }
  return true;
}

bool connectWifi() {
  if (strcmp(FUSIONSENSE_WIFI_SSID, "YOUR_WIFI_SSID") == 0) {
    Serial.println("# error,wifi_credentials_not_configured");
    return false;
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(FUSIONSENSE_WIFI_SSID, FUSIONSENSE_WIFI_PASSWORD);
  Serial.print("# wifi_connecting");
  const uint32_t startedMs = millis();
  while (WiFi.status() != WL_CONNECTED &&
         millis() - startedMs < WIFI_CONNECT_TIMEOUT_MS) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("# error,wifi_connect_timeout");
    return false;
  }
  return true;
}

bool startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = STREAM_PORT;
  config.ctrl_port = 32769;
  config.max_uri_handlers = 4;
  config.max_open_sockets = 2;
  config.lru_purge_enable = true;

  if (httpd_start(&streamServer, &config) != ESP_OK) {
    Serial.println("# error,http_server_start_failed");
    return false;
  }

  httpd_uri_t streamUri{};
  streamUri.uri = "/stream";
  streamUri.method = HTTP_GET;
  streamUri.handler = streamHandler;
  streamUri.user_ctx = nullptr;
  if (httpd_register_uri_handler(streamServer, &streamUri) != ESP_OK) {
    Serial.println("# error,stream_route_failed");
    return false;
  }

  httpd_uri_t healthUri{};
  healthUri.uri = "/health";
  healthUri.method = HTTP_GET;
  healthUri.handler = healthHandler;
  healthUri.user_ctx = nullptr;
  if (httpd_register_uri_handler(streamServer, &healthUri) != ESP_OK) {
    Serial.println("# error,health_route_failed");
    return false;
  }
  return true;
}

bool startControlServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = CONTROL_PORT;
  config.ctrl_port = 32768;
  config.max_uri_handlers = 4;
  config.max_open_sockets = 4;
  config.lru_purge_enable = true;

  if (httpd_start(&controlServer, &config) != ESP_OK) {
    Serial.println("# error,control_server_start_failed");
    return false;
  }

  httpd_uri_t sessionUri{};
  sessionUri.uri = "/session";
  sessionUri.method = HTTP_GET;
  sessionUri.handler = sessionHandler;
  if (httpd_register_uri_handler(controlServer, &sessionUri) != ESP_OK) {
    return false;
  }

  httpd_uri_t syncUri{};
  syncUri.uri = "/sync";
  syncUri.method = HTTP_GET;
  syncUri.handler = syncHandler;
  if (httpd_register_uri_handler(controlServer, &syncUri) != ESP_OK) {
    return false;
  }

  httpd_uri_t healthUri{};
  healthUri.uri = "/health";
  healthUri.method = HTTP_GET;
  healthUri.handler = healthHandler;
  if (httpd_register_uri_handler(controlServer, &healthUri) != ESP_OK) {
    return false;
  }
  return true;
}

void emitHealth() {
  const uint64_t nowUs = static_cast<uint64_t>(esp_timer_get_time());
  const uint64_t elapsedUs = nowUs - healthWindowStartUs;
  if (elapsedUs < 5000000ULL) {
    return;
  }

  const uint64_t windowFrames = framesCaptured - healthWindowStartFrames;
  const float effectiveFps =
      windowFrames * 1000000.0f / static_cast<float>(elapsedUs);
  Serial.printf(
      "# health,t_ms=%llu,frames=%llu,capture_errors=%u,stream_disconnects=%u,"
      "effective_fps=%.2f,rssi_dbm=%d\n",
      static_cast<unsigned long long>(nowUs / 1000ULL),
      static_cast<unsigned long long>(framesCaptured), captureErrors,
      streamDisconnects,
      effectiveFps, WiFi.RSSI());
  healthWindowStartUs = nowUs;
  healthWindowStartFrames = framesCaptured;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("# FusionSense ESP32-CAM timestamped JPEG Step 2");
  Serial.println("# target_fps=10,frame_size=QVGA,pixel_format=JPEG");

  if (!initializeCamera()) {
    Serial.println("# fatal,camera_initialization_failed");
    return;
  }
  Serial.println("# camera_ready");

  if (!connectWifi()) {
    Serial.println("# fatal,wifi_connection_failed");
    return;
  }

  Serial.print("# wifi_connected,ip=");
  Serial.print(WiFi.localIP());
  Serial.print(",rssi_dbm=");
  Serial.println(WiFi.RSSI());

  if (!startControlServer()) {
    Serial.println("# fatal,control_server_failed");
    return;
  }

  if (!startStreamServer()) {
    Serial.println("# fatal,stream_server_failed");
    return;
  }

  Serial.print("# stream_url=http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
  Serial.print("# health_url=http://");
  Serial.print(WiFi.localIP());
  Serial.println("/health");
  Serial.print("# sync_url=http://");
  Serial.print(WiFi.localIP());
  Serial.println("/sync?id=<request_id>");
  healthWindowStartUs = static_cast<uint64_t>(esp_timer_get_time());
}

void loop() {
  emitHealth();
  delay(100);
}
