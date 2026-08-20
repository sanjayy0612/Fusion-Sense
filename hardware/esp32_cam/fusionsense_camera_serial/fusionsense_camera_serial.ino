/*
 * FusionSense ESP32-CAM — timestamped JPEGs over USB serial
 *
 * Target: AI Thinker ESP32-CAM with OV2640 and PSRAM.
 * Runtime serial: 921600 baud, 8N1.
 * Capture: QVGA JPEG at a 10 FPS target.
 *
 * The camera waits for an ASCII "START\n" command before emitting binary
 * packets. Send "STOP\n" to stop. This keeps startup diagnostics readable and
 * prevents binary output from flooding Serial Monitor before the laptop is
 * ready.
 *
 * Binary packet (little-endian, 36-byte packed header followed by JPEG):
 *   magic[4]              "FSC1"
 *   schema_version        uint8  = 1
 *   packet_type           uint8  = 1 (JPEG)
 *   header_bytes          uint16 = 36
 *   sequence              uint32
 *   capture_timestamp_us  uint64
 *   jpeg_bytes            uint32
 *   width                 uint16
 *   height                uint16
 *   capture_errors        uint32 (cumulative)
 *   jpeg_crc32            uint32
 */

#include <Arduino.h>
#include "esp_camera.h"
#include "esp_timer.h"
#include <cstring>

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

constexpr uint32_t SERIAL_BAUD = 921600;
constexpr uint32_t TARGET_FPS = 10;
constexpr uint64_t FRAME_INTERVAL_US = 1000000ULL / TARGET_FPS;
constexpr uint32_t LOCAL_TEST_FRAMES = 20;
constexpr uint8_t SCHEMA_VERSION = 1;
constexpr uint8_t PACKET_TYPE_JPEG = 1;
constexpr uint8_t PACKET_MAGIC[4] = {'F', 'S', 'C', '1'};

#pragma pack(push, 1)
struct CameraPacketHeader {
  uint8_t magic[4];
  uint8_t schemaVersion;
  uint8_t packetType;
  uint16_t headerBytes;
  uint32_t sequence;
  uint64_t captureTimestampUs;
  uint32_t jpegBytes;
  uint16_t width;
  uint16_t height;
  uint32_t captureErrors;
  uint32_t jpegCrc32;
};
#pragma pack(pop)

static_assert(sizeof(CameraPacketHeader) == 36,
              "camera serial header must remain 36 bytes");

bool cameraReady = false;
bool streaming = false;
uint32_t sequence = 0;
uint32_t framesSent = 0;
uint32_t captureErrors = 0;
uint32_t transmitErrors = 0;
uint64_t nextFrameDueUs = 0;
char commandBuffer[48];
size_t commandLength = 0;

uint64_t frameCaptureTimestampUs(const camera_fb_t *frame) {
  if (frame->timestamp.tv_sec != 0 || frame->timestamp.tv_usec != 0) {
    return static_cast<uint64_t>(frame->timestamp.tv_sec) * 1000000ULL +
           static_cast<uint64_t>(frame->timestamp.tv_usec);
  }
  return static_cast<uint64_t>(esp_timer_get_time());
}

uint32_t jpegCrc32(const uint8_t *data, size_t length) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t index = 0; index < length; ++index) {
    crc ^= data[index];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      const uint32_t mask = -(crc & 1UL);
      crc = (crc >> 1) ^ (0xEDB88320UL & mask);
    }
  }
  return ~crc;
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
  return true;
}

bool runLocalCameraTest() {
  for (uint32_t index = 0; index < 3; ++index) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame != nullptr) {
      esp_camera_fb_return(frame);
    }
  }

  uint32_t successes = 0;
  uint32_t errors = 0;
  uint64_t totalBytes = 0;
  const uint64_t startedUs = static_cast<uint64_t>(esp_timer_get_time());
  for (uint32_t index = 0; index < LOCAL_TEST_FRAMES; ++index) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame == nullptr) {
      ++errors;
      continue;
    }
    ++successes;
    totalBytes += frame->len;
    esp_camera_fb_return(frame);
  }
  const uint64_t elapsedUs =
      static_cast<uint64_t>(esp_timer_get_time()) - startedUs;
  const float fps = elapsedUs > 0
                        ? successes * 1000000.0F / static_cast<float>(elapsedUs)
                        : 0.0F;
  const uint64_t meanBytes = successes > 0 ? totalBytes / successes : 0;
  const bool passed = successes == LOCAL_TEST_FRAMES && errors == 0;
  Serial.printf(
      "# local_test,result=%s,frames=%u,errors=%u,fps=%.2f,"
      "mean_jpeg_bytes=%llu\n",
      passed ? "PASS" : "FAIL", successes, errors, fps,
      static_cast<unsigned long long>(meanBytes));
  return passed;
}

void printInfo() {
  Serial.printf(
      "# info,transport=usb_serial,baud=%u,target_fps=%u,frame=320x240,"
      "psram=%s,frames_sent=%u,capture_errors=%u,transmit_errors=%u\n",
      SERIAL_BAUD, TARGET_FPS, psramFound() ? "true" : "false", framesSent,
      captureErrors, transmitErrors);
}

void handleCommand() {
  commandBuffer[commandLength] = '\0';
  if (strcmp(commandBuffer, "START") == 0) {
    if (!cameraReady) {
      Serial.println("# error,camera_not_ready");
    } else if (!streaming) {
      sequence = 0;
      framesSent = 0;
      captureErrors = 0;
      transmitErrors = 0;
      nextFrameDueUs = static_cast<uint64_t>(esp_timer_get_time());
      Serial.println("# stream_starting,binary_protocol=FSC1");
      Serial.flush();
      streaming = true;
    }
  } else if (strcmp(commandBuffer, "STOP") == 0) {
    if (streaming) {
      streaming = false;
      Serial.printf(
          "\n# stream_stopped,frames_sent=%u,capture_errors=%u,"
          "transmit_errors=%u\n",
          framesSent, captureErrors, transmitErrors);
    }
  } else if (strcmp(commandBuffer, "INFO") == 0 && !streaming) {
    printInfo();
  } else if (commandLength > 0 && !streaming) {
    Serial.println("# error,unknown_command,use_START_STOP_INFO");
  }
  commandLength = 0;
}

void processCommands() {
  while (Serial.available() > 0) {
    const int value = Serial.read();
    if (value < 0) {
      return;
    }
    const char character = static_cast<char>(value);
    if (character == '\r') {
      continue;
    }
    if (character == '\n') {
      handleCommand();
      continue;
    }
    if (commandLength + 1 < sizeof(commandBuffer)) {
      commandBuffer[commandLength++] = character;
    } else {
      commandLength = 0;
    }
  }
}

void waitForFrameSlot() {
  while (streaming) {
    const uint64_t nowUs = static_cast<uint64_t>(esp_timer_get_time());
    if (nowUs >= nextFrameDueUs) {
      return;
    }
    processCommands();
    if (!streaming) {
      return;
    }
    const uint64_t remainingUs = nextFrameDueUs - nowUs;
    if (remainingUs >= 2000) {
      delay(1);
    } else {
      delayMicroseconds(static_cast<uint32_t>(remainingUs));
    }
  }
}

void sendOneFrame() {
  waitForFrameSlot();
  if (!streaming) {
    return;
  }

  camera_fb_t *frame = esp_camera_fb_get();
  if (frame == nullptr) {
    ++captureErrors;
    nextFrameDueUs = static_cast<uint64_t>(esp_timer_get_time()) +
                     FRAME_INTERVAL_US;
    return;
  }

  ++sequence;
  CameraPacketHeader header{};
  memcpy(header.magic, PACKET_MAGIC, sizeof(PACKET_MAGIC));
  header.schemaVersion = SCHEMA_VERSION;
  header.packetType = PACKET_TYPE_JPEG;
  header.headerBytes = sizeof(CameraPacketHeader);
  header.sequence = sequence;
  header.captureTimestampUs = frameCaptureTimestampUs(frame);
  header.jpegBytes = static_cast<uint32_t>(frame->len);
  header.width = static_cast<uint16_t>(frame->width);
  header.height = static_cast<uint16_t>(frame->height);
  header.captureErrors = captureErrors;
  header.jpegCrc32 = jpegCrc32(frame->buf, frame->len);

  const size_t headerWritten = Serial.write(
      reinterpret_cast<const uint8_t *>(&header), sizeof(header));
  const size_t payloadWritten = Serial.write(frame->buf, frame->len);
  esp_camera_fb_return(frame);

  if (headerWritten != sizeof(header) || payloadWritten != header.jpegBytes) {
    ++transmitErrors;
    streaming = false;
    Serial.printf(
        "\n# transmit_error,header_written=%u,payload_written=%u,expected=%u\n",
        static_cast<unsigned>(headerWritten),
        static_cast<unsigned>(payloadWritten),
        static_cast<unsigned>(header.jpegBytes));
    return;
  }
  ++framesSent;

  nextFrameDueUs += FRAME_INTERVAL_US;
  const uint64_t nowUs = static_cast<uint64_t>(esp_timer_get_time());
  if (nextFrameDueUs < nowUs) {
    nextFrameDueUs = nowUs;
  }
}

}  // namespace

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);
  Serial.println();
  Serial.println("# FusionSense ESP32-CAM USB Serial v1");
  Serial.printf("# format=FSC1,header_bytes=%u,baud=%u,target_fps=%u\n",
                static_cast<unsigned>(sizeof(CameraPacketHeader)), SERIAL_BAUD,
                TARGET_FPS);
  Serial.printf("# psram=%s\n", psramFound() ? "true" : "false");

  if (!initializeCamera()) {
    Serial.println("# fatal,camera_initialization_failed");
    return;
  }
  cameraReady = runLocalCameraTest();
  if (!cameraReady) {
    Serial.println("# fatal,local_camera_test_failed");
    return;
  }
  printInfo();
  Serial.println("# ready,close_serial_monitor_then_run_laptop_receiver");
}

void loop() {
  processCommands();
  if (streaming) {
    sendOneFrame();
  } else {
    delay(10);
  }
}
