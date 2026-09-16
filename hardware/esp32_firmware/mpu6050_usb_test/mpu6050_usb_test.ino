/*
 * FusionSense IMU — stable, calibrated, timestamped USB serial stream
 *
 * Default wiring for a classic ESP32 DevKit:
 *   MPU-6050 VCC -> 3V3
 *   MPU-6050 GND -> GND
 *   MPU-6050 SDA -> GPIO21
 *   MPU-6050 SCL -> GPIO22
 *
 * Versioned serial packet:
 *   IMU,1,imu01,<session>,<seq>,<t_device_us>,ax,ay,az,gx,gy,gz
 *
 * Lines beginning with '#' are status/health messages and are ignored by the
 * laptop validator. Keep the board still during startup calibration.
 */

#include <Wire.h>
#include <esp_timer.h>
#include <math.h>
#include <stdio.h>
#include <cstring>

namespace {

constexpr int I2C_SDA_PIN = 21;
constexpr int I2C_SCL_PIN = 22;
constexpr uint32_t SERIAL_BAUD = 115200;
constexpr char DEVICE_ID[] = "imu01";
constexpr size_t SESSION_ID_CAPACITY = 33;
constexpr size_t COMMAND_CAPACITY = 96;
// 100 kHz is ample for 14-byte reads at 50 Hz and is more tolerant of the
// temporary jumper/header connections used during bring-up.
constexpr uint32_t I2C_FREQUENCY_HZ = 100000;

constexpr uint32_t SAMPLE_RATE_HZ = 50;
constexpr uint64_t SAMPLE_PERIOD_US = 1000000ULL / SAMPLE_RATE_HZ;
constexpr uint16_t CALIBRATION_SAMPLES = 250;
constexpr float ACCEL_LSB_PER_G = 16384.0f;       // +/- 2 g
constexpr float GYRO_LSB_PER_DPS = 131.0f;        // +/- 250 deg/s
constexpr float MAX_CAL_ACCEL_MAG_STD_G = 0.030f;
constexpr float MAX_CAL_GYRO_MAG_STD_DPS = 1.000f;
constexpr float MIN_CAL_ACCEL_MAG_G = 0.75f;
constexpr float MAX_CAL_ACCEL_MAG_G = 1.25f;

constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_WHO_AM_I = 0x75;

float accelBiasG[3] = {0.0f, 0.0f, 0.0f};
float gyroBiasDps[3] = {0.0f, 0.0f, 0.0f};
uint8_t mpuAddress = 0;

uint64_t nextSampleUs = 0;
uint64_t healthWindowStartUs = 0;
uint32_t successfulSamples = 0;
uint32_t healthWindowSamples = 0;
uint32_t readErrors = 0;
uint32_t missedSlots = 0;
uint32_t healthWindowReadErrors = 0;
uint32_t healthWindowMissedSlots = 0;
uint64_t sampleSequence = 0;
char sessionId[SESSION_ID_CAPACITY] = "unassigned";
char commandBuffer[COMMAND_CAPACITY] = {};
size_t commandLength = 0;

struct RawSample {
  int16_t accel[3];
  int16_t gyro[3];
};

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

bool readRegister(uint8_t reg, uint8_t &value) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const size_t received = Wire.requestFrom(
      mpuAddress, static_cast<size_t>(1), true);
  if (received != 1 || Wire.available() < 1) {
    return false;
  }

  value = static_cast<uint8_t>(Wire.read());
  return true;
}

int16_t readSigned16() {
  const uint16_t high = static_cast<uint16_t>(Wire.read());
  const uint16_t low = static_cast<uint16_t>(Wire.read());
  return static_cast<int16_t>((high << 8) | low);
}

bool readRawSample(RawSample &sample) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const size_t received = Wire.requestFrom(
      mpuAddress, static_cast<size_t>(14), true);
  if (received != 14 || Wire.available() < 14) {
    while (Wire.available()) {
      Wire.read();
    }
    return false;
  }

  sample.accel[0] = readSigned16();
  sample.accel[1] = readSigned16();
  sample.accel[2] = readSigned16();
  readSigned16();  // Temperature is not used in the Step 1 contract.
  sample.gyro[0] = readSigned16();
  sample.gyro[1] = readSigned16();
  sample.gyro[2] = readSigned16();
  return true;
}

uint8_t scanForMpu() {
  uint8_t candidate = 0;
  uint8_t foundCount = 0;
  Serial.print("# i2c_scan,pins=");
  Serial.print(I2C_SDA_PIN);
  Serial.print('|');
  Serial.print(I2C_SCL_PIN);
  Serial.print(",addresses=");

  for (uint8_t address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission(true) == 0) {
      if (foundCount > 0) {
        Serial.print('|');
      }
      Serial.print("0x");
      if (address < 0x10) {
        Serial.print('0');
      }
      Serial.print(address, HEX);
      ++foundCount;
      if (address == 0x68 || address == 0x69) {
        candidate = address;
      }
    }
  }

  if (foundCount == 0) {
    Serial.print("none");
  }
  Serial.println();
  return candidate;
}

const char *modelForIdentity(uint8_t identity) {
  switch (identity) {
    case 0x68:
      return "MPU-6050";
    case 0x70:
      return "MPU-6500";
    case 0x71:
      return "MPU-9250";
    case 0x73:
      return "MPU-9255";
    default:
      return nullptr;
  }
}

bool configureMpu() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(I2C_FREQUENCY_HZ);
  Wire.setTimeOut(20);

  mpuAddress = scanForMpu();
  if (mpuAddress == 0) {
    return false;
  }

  if (!writeRegister(REG_PWR_MGMT_1, 0x80)) {  // Device reset.
    return false;
  }
  delay(100);

  // PLL with X-axis gyro reference, DLPF enabled, 50 Hz sensor sample rate,
  // +/-2 g accelerometer, and +/-250 deg/s gyroscope.
  if (!writeRegister(REG_PWR_MGMT_1, 0x01) ||
      !writeRegister(REG_CONFIG, 0x03) ||
      !writeRegister(REG_SMPLRT_DIV, 19) ||
      !writeRegister(REG_GYRO_CONFIG, 0x00) ||
      !writeRegister(REG_ACCEL_CONFIG, 0x00)) {
    return false;
  }

  delay(100);
  uint8_t identity = 0;
  if (!readRegister(REG_WHO_AM_I, identity)) {
    return false;
  }

  // These devices share the accelerometer/gyroscope register layout and the
  // +/-2 g, +/-250 deg/s scale factors configured by this sketch.
  const char *model = modelForIdentity(identity);
  if (model == nullptr) {
    Serial.print("# error,unexpected_who_am_i=0x");
    Serial.println(identity, HEX);
    return false;
  }

  Serial.print("# mpu_who_am_i=0x");
  Serial.println(identity, HEX);
  Serial.print("# mpu_i2c_address=0x");
  Serial.println(mpuAddress, HEX);
  Serial.print("# mpu_model=");
  Serial.println(model);
  return true;
}

float standardDeviation(double sum, double sumSquares, uint16_t count) {
  if (count < 2) {
    return 0.0f;
  }
  const double mean = sum / count;
  const double variance = fmax(0.0, (sumSquares / count) - (mean * mean));
  return static_cast<float>(sqrt(variance));
}

bool calibrateStationary() {
  Serial.println("# calibration,keep_sensor_stationary,seconds=5");

  double accelSumG[3] = {0.0, 0.0, 0.0};
  double gyroSumDps[3] = {0.0, 0.0, 0.0};
  double accelMagSum = 0.0;
  double accelMagSquares = 0.0;
  double gyroMagSum = 0.0;
  double gyroMagSquares = 0.0;

  for (uint16_t index = 0; index < CALIBRATION_SAMPLES; ++index) {
    const uint64_t slotStartUs = static_cast<uint64_t>(esp_timer_get_time());
    RawSample raw{};
    if (!readRawSample(raw)) {
      Serial.println("# calibration_failed,reason=i2c_read");
      return false;
    }

    float accelG[3];
    float gyroDps[3];
    for (uint8_t axis = 0; axis < 3; ++axis) {
      accelG[axis] = raw.accel[axis] / ACCEL_LSB_PER_G;
      gyroDps[axis] = raw.gyro[axis] / GYRO_LSB_PER_DPS;
      accelSumG[axis] += accelG[axis];
      gyroSumDps[axis] += gyroDps[axis];
    }

    const float accelMagnitude = sqrtf(
        accelG[0] * accelG[0] + accelG[1] * accelG[1] +
        accelG[2] * accelG[2]);
    const float gyroMagnitude = sqrtf(
        gyroDps[0] * gyroDps[0] + gyroDps[1] * gyroDps[1] +
        gyroDps[2] * gyroDps[2]);
    accelMagSum += accelMagnitude;
    accelMagSquares += accelMagnitude * accelMagnitude;
    gyroMagSum += gyroMagnitude;
    gyroMagSquares += gyroMagnitude * gyroMagnitude;

    const int64_t remainingUs = static_cast<int64_t>(
        slotStartUs + SAMPLE_PERIOD_US - esp_timer_get_time());
    if (remainingUs > 1000) {
      delay(static_cast<uint32_t>(remainingUs / 1000));
    }
    while (esp_timer_get_time() <
           static_cast<int64_t>(slotStartUs + SAMPLE_PERIOD_US)) {
      delayMicroseconds(50);
    }
  }

  float meanAccelG[3];
  float meanGyroDps[3];
  for (uint8_t axis = 0; axis < 3; ++axis) {
    meanAccelG[axis] = static_cast<float>(accelSumG[axis] /
                                         CALIBRATION_SAMPLES);
    meanGyroDps[axis] = static_cast<float>(gyroSumDps[axis] /
                                           CALIBRATION_SAMPLES);
  }

  const float meanAccelMagnitude = sqrtf(
      meanAccelG[0] * meanAccelG[0] + meanAccelG[1] * meanAccelG[1] +
      meanAccelG[2] * meanAccelG[2]);
  const float accelMagnitudeStd = standardDeviation(
      accelMagSum, accelMagSquares, CALIBRATION_SAMPLES);
  const float gyroMagnitudeStd = standardDeviation(
      gyroMagSum, gyroMagSquares, CALIBRATION_SAMPLES);

  if (meanAccelMagnitude < MIN_CAL_ACCEL_MAG_G ||
      meanAccelMagnitude > MAX_CAL_ACCEL_MAG_G ||
      accelMagnitudeStd > MAX_CAL_ACCEL_MAG_STD_G ||
      gyroMagnitudeStd > MAX_CAL_GYRO_MAG_STD_DPS) {
    Serial.print("# calibration_failed,reason=movement_or_bad_scale");
    Serial.print(",accel_mean_g=");
    Serial.print(meanAccelMagnitude, 4);
    Serial.print(",accel_std_g=");
    Serial.print(accelMagnitudeStd, 4);
    Serial.print(",gyro_mag_std_dps=");
    Serial.println(gyroMagnitudeStd, 4);
    return false;
  }

  // Preserve the stationary 1 g gravity vector in whatever orientation the
  // sensor was calibrated. Remove only the magnitude/axis bias beyond it.
  for (uint8_t axis = 0; axis < 3; ++axis) {
    const float expectedGravityAxis = meanAccelG[axis] / meanAccelMagnitude;
    accelBiasG[axis] = meanAccelG[axis] - expectedGravityAxis;
    gyroBiasDps[axis] = meanGyroDps[axis];
  }

  Serial.print("# calibration_ok,accel_bias_g=");
  Serial.print(accelBiasG[0], 5);
  Serial.print('|');
  Serial.print(accelBiasG[1], 5);
  Serial.print('|');
  Serial.print(accelBiasG[2], 5);
  Serial.print(",gyro_bias_dps=");
  Serial.print(gyroBiasDps[0], 4);
  Serial.print('|');
  Serial.print(gyroBiasDps[1], 4);
  Serial.print('|');
  Serial.println(gyroBiasDps[2], 4);
  return true;
}

bool validIdentifier(const char *value) {
  const size_t length = strlen(value);
  if (length == 0 || length >= SESSION_ID_CAPACITY) {
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

void processCommand(char *command) {
  if (strncmp(command, "SESSION,", 8) == 0) {
    const char *requestedSession = command + 8;
    if (!validIdentifier(requestedSession)) {
      Serial.println("# error,invalid_session_id");
      return;
    }
    strncpy(sessionId, requestedSession, SESSION_ID_CAPACITY - 1);
    sessionId[SESSION_ID_CAPACITY - 1] = '\0';
    Serial.print("# session_set,id=");
    Serial.println(sessionId);
    return;
  }

  if (strncmp(command, "SYNC,", 5) == 0) {
    const char *requestId = command + 5;
    if (!validIdentifier(requestId)) {
      Serial.println("# error,invalid_sync_request_id");
      return;
    }
    const uint64_t deviceTimeUs =
        static_cast<uint64_t>(esp_timer_get_time());
    Serial.printf("# SYNC_RESP,%s,%llu\n", requestId,
                  static_cast<unsigned long long>(deviceTimeUs));
    return;
  }

  if (strcmp(command, "INFO") == 0) {
    Serial.printf(
        "# info,device_id=%s,session_id=%s,schema=1,sample_hz=%u,"
        "sequence=%llu\n",
        DEVICE_ID, sessionId, SAMPLE_RATE_HZ,
        static_cast<unsigned long long>(sampleSequence));
    return;
  }

  Serial.println("# error,unknown_command");
}

void pollSerialCommands() {
  while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());
    if (character == '\r') {
      continue;
    }
    if (character == '\n') {
      if (commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        processCommand(commandBuffer);
        commandLength = 0;
      }
      continue;
    }
    if (commandLength + 1 < COMMAND_CAPACITY) {
      commandBuffer[commandLength++] = character;
    } else {
      commandLength = 0;
      Serial.println("# error,command_too_long");
    }
  }
}

void emitSample(uint64_t sequence, uint64_t captureTimeUs,
                const RawSample &raw) {
  float accelG[3];
  float gyroDps[3];
  for (uint8_t axis = 0; axis < 3; ++axis) {
    accelG[axis] = raw.accel[axis] / ACCEL_LSB_PER_G - accelBiasG[axis];
    gyroDps[axis] = raw.gyro[axis] / GYRO_LSB_PER_DPS - gyroBiasDps[axis];
  }

  char line[192];
  const int length = snprintf(
      line, sizeof(line),
      "IMU,1,%s,%s,%llu,%llu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
      DEVICE_ID, sessionId, static_cast<unsigned long long>(sequence),
      static_cast<unsigned long long>(captureTimeUs), accelG[0], accelG[1],
      accelG[2], gyroDps[0], gyroDps[1], gyroDps[2]);
  if (length > 0 && length < static_cast<int>(sizeof(line))) {
    Serial.write(reinterpret_cast<const uint8_t *>(line),
                 static_cast<size_t>(length));
  }
}

void emitHealth(uint64_t nowUs) {
  const uint64_t elapsedUs = nowUs - healthWindowStartUs;
  if (elapsedUs < 5000000ULL) {
    return;
  }

  const float effectiveRate =
      healthWindowSamples * 1000000.0f / static_cast<float>(elapsedUs);
  Serial.print("# health,t_ms=");
  Serial.print(static_cast<unsigned long>(nowUs / 1000ULL));
  Serial.print(",samples=");
  Serial.print(successfulSamples);
  Serial.print(",read_errors_total=");
  Serial.print(readErrors);
  Serial.print(",read_errors_window=");
  Serial.print(healthWindowReadErrors);
  Serial.print(",missed_slots_total=");
  Serial.print(missedSlots);
  Serial.print(",missed_slots_window=");
  Serial.print(healthWindowMissedSlots);
  Serial.print(",effective_hz=");
  Serial.println(effectiveRate, 2);

  healthWindowStartUs = nowUs;
  healthWindowSamples = 0;
  healthWindowReadErrors = 0;
  healthWindowMissedSlots = 0;
}

}  // namespace

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);
  Serial.println("# FusionSense timestamped IMU USB stream");
  Serial.println(
      "# format=IMU,1,device_id,session_id,seq,t_device_us,"
      "ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
  Serial.println("# configured_hz=50,accel_range_g=2,gyro_range_dps=250");

  while (!configureMpu()) {
    Serial.println("# error,mpu_not_found_check_power_sda_scl,retry_ms=2000");
    delay(2000);
  }

  while (!calibrateStationary()) {
    Serial.println("# calibration_retry,keep_sensor_stationary");
    delay(1000);
  }

  const uint64_t nowUs = static_cast<uint64_t>(esp_timer_get_time());
  nextSampleUs = nowUs + SAMPLE_PERIOD_US;
  healthWindowStartUs = nowUs;
  Serial.println("# acquisition_started");
}

void loop() {
  pollSerialCommands();
  const uint64_t nowUs = static_cast<uint64_t>(esp_timer_get_time());
  if (nowUs < nextSampleUs) {
    const uint64_t waitUs = nextSampleUs - nowUs;
    if (waitUs > 2000) {
      delay(1);
    } else if (waitUs > 100) {
      delayMicroseconds(static_cast<uint32_t>(waitUs - 50));
    }
    return;
  }

  const uint64_t latenessUs = nowUs - nextSampleUs;
  if (latenessUs >= SAMPLE_PERIOD_US) {
    const uint32_t skipped = static_cast<uint32_t>(latenessUs /
                                                   SAMPLE_PERIOD_US);
    missedSlots += skipped;
    healthWindowMissedSlots += skipped;
    sampleSequence += skipped;
    nextSampleUs += static_cast<uint64_t>(skipped) * SAMPLE_PERIOD_US;
  }

  const uint64_t captureTimeUs = static_cast<uint64_t>(esp_timer_get_time());
  const uint64_t sequence = sampleSequence++;
  nextSampleUs += SAMPLE_PERIOD_US;

  RawSample raw{};
  if (readRawSample(raw)) {
    emitSample(sequence, captureTimeUs, raw);
    ++successfulSamples;
    ++healthWindowSamples;
  } else {
    ++readErrors;
    ++healthWindowReadErrors;
  }

  emitHealth(captureTimeUs);
}
