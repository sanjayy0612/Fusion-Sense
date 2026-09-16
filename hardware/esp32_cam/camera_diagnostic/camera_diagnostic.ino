/*
 * FusionSense ESP32-CAM diagnostic firmware
 *
 * Target: AI Thinker ESP32-CAM with OV2640 and PSRAM.
 *
 * Test order:
 *   1. Capture 50 JPEGs locally before Wi-Fi starts.
 *   2. Serve JSON health data at http://<ip>/health.
 *   3. Serve one JPEG at http://<ip>/capture.
 *   4. Serve timestamped 10 FPS MJPEG at http://<ip>:8080/stream.
 *
 * Copy secrets.example.h to secrets.h and enter the Wi-Fi credentials.
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include <WiFi.h>
#include <cstring>

#if __has_include("secrets.h")
#include "secrets.h"
#elif __has_include("../fusionsense_camera/secrets.h")
// Reuse the ignored credentials from the normal camera sketch when available.
#include "../fusionsense_camera/secrets.h"
#else
#error "Copy secrets.example.h to secrets.h and enter the Wi-Fi credentials"
#endif

namespace {

constexpr char FIRMWARE_BUILD[] = "camera-diagnostic-v1";
constexpr char DEVICE_ID[] = "cam01";

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

constexpr uint16_t CONTROL_PORT = 80;
constexpr uint16_t STREAM_PORT = 8080;
constexpr uint32_t TARGET_FPS = 10;
constexpr uint64_t FRAME_INTERVAL_US = 1000000ULL / TARGET_FPS;
constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS = 30000;
constexpr uint32_t SELF_TEST_FRAMES = 50;
constexpr size_t JPEG_SEND_CHUNK_BYTES = 1024;

constexpr char STREAM_CONTENT_TYPE[] =
    "multipart/x-mixed-replace;boundary=frame";
constexpr char STREAM_BOUNDARY[] = "\r\n--frame\r\n";
constexpr char STREAM_PART_HEADER[] =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "X-Device-Id: %s\r\n"
    "X-Frame-Sequence: %llu\r\n"
    "X-Capture-Timestamp-Us: %llu\r\n\r\n";

httpd_handle_t controlServer = nullptr;
httpd_handle_t streamServer = nullptr;

bool localSelfTestPassed = false;
uint32_t localCaptureSuccesses = 0;
uint32_t localCaptureErrors = 0;
uint64_t localCaptureBytes = 0;
float localCaptureFps = 0.0F;

volatile uint64_t snapshotAttempts = 0;
volatile uint64_t snapshotSuccesses = 0;
volatile uint64_t streamCaptureAttempts = 0;
volatile uint64_t streamFramesSent = 0;
volatile uint32_t captureErrors = 0;
volatile uint32_t streamSendErrors = 0;
volatile uint32_t streamDisconnects = 0;
uint64_t streamStartedUs = 0;
uint64_t lastHealthPrintMs = 0;

uint64_t frameCaptureTimestampUs(const camera_fb_t *frame) {
  if (frame->timestamp.tv_sec != 0 || frame->timestamp.tv_usec != 0) {
    return static_cast<uint64_t>(frame->timestamp.tv_sec) * 1000000ULL +
           static_cast<uint64_t>(frame->timestamp.tv_usec);
  }
  return static_cast<uint64_t>(esp_timer_get_time());
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
  config.fb_location = CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_count = 1;

  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_count = 2;
  }

  const esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    Serial.printf("# camera_init,result=FAIL,error=0x%x\n", error);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    sensor->set_framesize(sensor, FRAMESIZE_QVGA);
  }
  Serial.printf("# camera_init,result=PASS,psram=%s\n",
                psramFound() ? "true" : "false");
  return true;
}

void runLocalSelfTest() {
  Serial.printf("# local_self_test,start,frames=%u\n", SELF_TEST_FRAMES);

  // Discard startup frames while auto-exposure settles.
  for (uint32_t index = 0; index < 3; ++index) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame != nullptr) {
      esp_camera_fb_return(frame);
    }
  }

  const uint64_t startedUs = static_cast<uint64_t>(esp_timer_get_time());
  for (uint32_t index = 0; index < SELF_TEST_FRAMES; ++index) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
      ++localCaptureErrors;
      continue;
    }
    ++localCaptureSuccesses;
    localCaptureBytes += frame->len;
    esp_camera_fb_return(frame);
  }
  const uint64_t elapsedUs =
      static_cast<uint64_t>(esp_timer_get_time()) - startedUs;
  localCaptureFps = elapsedUs > 0
                        ? static_cast<float>(localCaptureSuccesses) * 1000000.0F /
                              static_cast<float>(elapsedUs)
                        : 0.0F;
  const uint64_t meanBytes = localCaptureSuccesses > 0
                                 ? localCaptureBytes / localCaptureSuccesses
                                 : 0;
  localSelfTestPassed = localCaptureSuccesses == SELF_TEST_FRAMES &&
                        localCaptureErrors == 0 && localCaptureFps >= 8.5F;

  Serial.printf(
      "# local_self_test,result=%s,successes=%u,errors=%u,fps=%.2f,"
      "mean_jpeg_bytes=%llu\n",
      localSelfTestPassed ? "PASS" : "FAIL", localCaptureSuccesses,
      localCaptureErrors, localCaptureFps,
      static_cast<unsigned long long>(meanBytes));
}

float effectiveStreamFps() {
  if (streamStartedUs == 0 || streamFramesSent == 0) {
    return 0.0F;
  }
  const uint64_t elapsedUs =
      static_cast<uint64_t>(esp_timer_get_time()) - streamStartedUs;
  return elapsedUs > 0
             ? static_cast<float>(streamFramesSent) * 1000000.0F /
                   static_cast<float>(elapsedUs)
             : 0.0F;
}

esp_err_t healthHandler(httpd_req_t *request) {
  char payload[1024];
  const uint64_t meanBytes = localCaptureSuccesses > 0
                                 ? localCaptureBytes / localCaptureSuccesses
                                 : 0;
  const int length = snprintf(
      payload, sizeof(payload),
      "{\"status\":\"ok\",\"firmware\":\"%s\",\"device_id\":\"%s\","
      "\"target_fps\":%u,\"uptime_us\":%llu,\"rssi_dbm\":%d,"
      "\"psram\":%s,\"local_self_test\":{\"result\":\"%s\","
      "\"successes\":%u,\"errors\":%u,\"fps\":%.3f,"
      "\"mean_jpeg_bytes\":%llu},\"snapshot_attempts\":%llu,"
      "\"snapshot_successes\":%llu,\"stream_capture_attempts\":%llu,"
      "\"stream_frames_sent\":%llu,\"capture_errors\":%u,"
      "\"stream_send_errors\":%u,\"stream_disconnects\":%u,"
      "\"effective_stream_fps\":%.3f}",
      FIRMWARE_BUILD, DEVICE_ID, TARGET_FPS,
      static_cast<unsigned long long>(esp_timer_get_time()), WiFi.RSSI(),
      psramFound() ? "true" : "false",
      localSelfTestPassed ? "PASS" : "FAIL", localCaptureSuccesses,
      localCaptureErrors, localCaptureFps,
      static_cast<unsigned long long>(meanBytes),
      static_cast<unsigned long long>(snapshotAttempts),
      static_cast<unsigned long long>(snapshotSuccesses),
      static_cast<unsigned long long>(streamCaptureAttempts),
      static_cast<unsigned long long>(streamFramesSent), captureErrors,
      streamSendErrors, streamDisconnects, effectiveStreamFps());
  if (length < 0 || static_cast<size_t>(length) >= sizeof(payload)) {
    return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                               "health payload overflow");
  }
  httpd_resp_set_type(request, "application/json");
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, payload, length);
}

esp_err_t captureHandler(httpd_req_t *request) {
  ++snapshotAttempts;
  const uint64_t startedUs = static_cast<uint64_t>(esp_timer_get_time());
  camera_fb_t *frame = esp_camera_fb_get();
  if (frame == nullptr) {
    ++captureErrors;
    Serial.println("# snapshot,result=FAIL,reason=capture_error");
    return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                               "camera capture failed");
  }

  char timestamp[24];
  snprintf(timestamp, sizeof(timestamp), "%llu",
           static_cast<unsigned long long>(frameCaptureTimestampUs(frame)));
  httpd_resp_set_type(request, "image/jpeg");
  httpd_resp_set_hdr(request, "Content-Disposition",
                     "inline; filename=capture.jpg");
  httpd_resp_set_hdr(request, "X-Capture-Timestamp-Us", timestamp);
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  const esp_err_t result = httpd_resp_send(
      request, reinterpret_cast<const char *>(frame->buf), frame->len);
  const size_t frameLength = frame->len;
  esp_camera_fb_return(frame);

  const uint64_t elapsedUs =
      static_cast<uint64_t>(esp_timer_get_time()) - startedUs;
  if (result == ESP_OK) {
    ++snapshotSuccesses;
  } else {
    ++streamSendErrors;
  }
  Serial.printf("# snapshot,result=%s,bytes=%u,total_ms=%.2f\n",
                result == ESP_OK ? "PASS" : "FAIL",
                static_cast<unsigned>(frameLength),
                static_cast<double>(elapsedUs) / 1000.0);
  return result;
}

esp_err_t streamHandler(httpd_req_t *request) {
  httpd_resp_set_type(request, STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  Serial.println("# stream_client_connected");

  uint64_t sequence = 0;
  uint64_t nextFrameDueUs = static_cast<uint64_t>(esp_timer_get_time());
  if (streamStartedUs == 0) {
    streamStartedUs = nextFrameDueUs;
  }

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

    ++streamCaptureAttempts;
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
      ++captureErrors;
      Serial.println("# stream_capture,result=FAIL");
      nextFrameDueUs = static_cast<uint64_t>(esp_timer_get_time()) +
                       FRAME_INTERVAL_US;
      continue;
    }

    ++sequence;
    const uint64_t captureTimestampUs = frameCaptureTimestampUs(frame);
    char partHeader[256];
    const int headerLength = snprintf(
        partHeader, sizeof(partHeader), STREAM_PART_HEADER,
        static_cast<unsigned>(frame->len), DEVICE_ID,
        static_cast<unsigned long long>(sequence),
        static_cast<unsigned long long>(captureTimestampUs));

    esp_err_t result = httpd_resp_send_chunk(
        request, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(request, partHeader, headerLength);
    }

    size_t bytesSent = 0;
    const uint64_t sendStartedUs = static_cast<uint64_t>(esp_timer_get_time());
    while (result == ESP_OK && bytesSent < frame->len) {
      const size_t remaining = frame->len - bytesSent;
      const size_t chunkLength =
          remaining < JPEG_SEND_CHUNK_BYTES ? remaining : JPEG_SEND_CHUNK_BYTES;
      result = httpd_resp_send_chunk(
          request,
          reinterpret_cast<const char *>(frame->buf + bytesSent), chunkLength);
      if (result == ESP_OK) {
        bytesSent += chunkLength;
        delay(1);
      }
    }
    const uint64_t sendElapsedUs =
        static_cast<uint64_t>(esp_timer_get_time()) - sendStartedUs;
    const size_t jpegLength = frame->len;
    esp_camera_fb_return(frame);

    if (result != ESP_OK) {
      ++streamSendErrors;
      ++streamDisconnects;
      Serial.printf(
          "# stream_send,result=FAIL,error=0x%x,sent=%u,jpeg=%u,send_ms=%.2f\n",
          result, static_cast<unsigned>(bytesSent),
          static_cast<unsigned>(jpegLength),
          static_cast<double>(sendElapsedUs) / 1000.0);
      break;
    }

    ++streamFramesSent;
    if (sequence <= 3 || sequence % 10 == 0) {
      Serial.printf("# stream_frame,seq=%llu,bytes=%u,send_ms=%.2f\n",
                    static_cast<unsigned long long>(sequence),
                    static_cast<unsigned>(jpegLength),
                    static_cast<double>(sendElapsedUs) / 1000.0);
    }

    nextFrameDueUs += FRAME_INTERVAL_US;
    const uint64_t afterSendUs = static_cast<uint64_t>(esp_timer_get_time());
    if (nextFrameDueUs < afterSendUs) {
      nextFrameDueUs = afterSendUs;
    }
  }

  Serial.println("# stream_client_disconnected");
  return ESP_OK;
}

bool startServers() {
  httpd_config_t controlConfig = HTTPD_DEFAULT_CONFIG();
  controlConfig.server_port = CONTROL_PORT;
  controlConfig.ctrl_port = 32768;
  controlConfig.max_uri_handlers = 4;
  if (httpd_start(&controlServer, &controlConfig) != ESP_OK) {
    Serial.println("# control_server,result=FAIL");
    return false;
  }

  httpd_uri_t healthUri{};
  healthUri.uri = "/health";
  healthUri.method = HTTP_GET;
  healthUri.handler = healthHandler;
  healthUri.user_ctx = nullptr;
  httpd_uri_t captureUri{};
  captureUri.uri = "/capture";
  captureUri.method = HTTP_GET;
  captureUri.handler = captureHandler;
  captureUri.user_ctx = nullptr;
  httpd_register_uri_handler(controlServer, &healthUri);
  httpd_register_uri_handler(controlServer, &captureUri);

  httpd_config_t streamConfig = HTTPD_DEFAULT_CONFIG();
  streamConfig.server_port = STREAM_PORT;
  streamConfig.ctrl_port = 32769;
  streamConfig.max_uri_handlers = 2;
  streamConfig.stack_size = 8192;
  if (httpd_start(&streamServer, &streamConfig) != ESP_OK) {
    Serial.println("# stream_server,result=FAIL");
    return false;
  }

  httpd_uri_t streamUri{};
  streamUri.uri = "/stream";
  streamUri.method = HTTP_GET;
  streamUri.handler = streamHandler;
  streamUri.user_ctx = nullptr;
  httpd_register_uri_handler(streamServer, &streamUri);
  return true;
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(FUSIONSENSE_WIFI_SSID, FUSIONSENSE_WIFI_PASSWORD);
  Serial.printf("# wifi_connecting,ssid=%s\n", FUSIONSENSE_WIFI_SSID);

  const uint32_t startedMs = millis();
  while (WiFi.status() != WL_CONNECTED &&
         millis() - startedMs < WIFI_CONNECT_TIMEOUT_MS) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("# wifi,result=FAIL,restart_after_checking_secrets");
    return;
  }

  const String ip = WiFi.localIP().toString();
  Serial.printf("# wifi,result=PASS,ip=%s,rssi_dbm=%d\n", ip.c_str(), WiFi.RSSI());
  if (!startServers()) {
    Serial.println("# diagnostic_servers,result=FAIL");
    return;
  }
  Serial.printf("# health_url=http://%s/health\n", ip.c_str());
  Serial.printf("# capture_url=http://%s/capture\n", ip.c_str());
  Serial.printf("# stream_url=http://%s:%u/stream\n", ip.c_str(), STREAM_PORT);
  Serial.println("# diagnostic_ready");
}

void printHealth() {
  Serial.printf(
      "# health,local=%s,snapshots=%llu/%llu,stream_frames=%llu,"
      "capture_errors=%u,send_errors=%u,disconnects=%u,effective_fps=%.2f,"
      "rssi_dbm=%d\n",
      localSelfTestPassed ? "PASS" : "FAIL",
      static_cast<unsigned long long>(snapshotSuccesses),
      static_cast<unsigned long long>(snapshotAttempts),
      static_cast<unsigned long long>(streamFramesSent), captureErrors,
      streamSendErrors, streamDisconnects, effectiveStreamFps(), WiFi.RSSI());
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.printf("# FusionSense %s\n", FIRMWARE_BUILD);
  if (!initializeCamera()) {
    Serial.println("# fatal,camera_not_available");
    return;
  }
  runLocalSelfTest();
  connectWifi();
}

void loop() {
  const uint64_t nowMs = millis();
  if (nowMs - lastHealthPrintMs >= 5000) {
    lastHealthPrintMs = nowMs;
    printHealth();
  }
  delay(20);
}
