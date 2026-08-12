/*
 * ESP32 + MPU-6050 USB test
 *
 * Wiring:
 *   MPU-6050 VCC -> ESP32 3V3
 *   MPU-6050 GND -> ESP32 GND
 *   MPU-6050 SDA -> ESP32 D21 (GPIO 21)
 *   MPU-6050 SCL -> ESP32 D22 (GPIO 22)
 *
 * Open Arduino IDE Serial Monitor at 115200 baud.
 */

#include <Wire.h>

constexpr uint8_t MPU_ADDRESS = 0x68;
constexpr uint8_t MPU_PWR_MGMT_1 = 0x6B;
constexpr uint8_t MPU_ACCEL_XOUT_H = 0x3B;
bool mpuReady = false;

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDRESS);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool mpuConnected() {
  Wire.beginTransmission(MPU_ADDRESS);
  return Wire.endTransmission() == 0;
}

bool initializeMpu() {
  if (!mpuConnected()) {
    return false;
  }
  if (!writeRegister(MPU_PWR_MGMT_1, 0x00)) {
    return false;
  }
  delay(100);
  return true;
}

bool readMotion(int16_t &ax, int16_t &ay, int16_t &az,
                int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDRESS);
  Wire.write(MPU_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(MPU_ADDRESS, static_cast<uint8_t>(14), true) != 14) {
    return false;
  }

  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read();
  Wire.read(); // Skip temperature.
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(21, 22); // SDA = D21, SCL = D22
  Wire.setClock(100000); // Slower I2C is friendlier to temporary contacts.

  Serial.println();
  Serial.println("ESP32 MPU-6050 USB test starting...");

  mpuReady = initializeMpu();
  if (mpuReady) {
    Serial.println("MPU-6050 found at 0x68.");
    Serial.println("t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
  } else {
    Serial.println("MPU-6050 not found; retrying every second.");
    Serial.println("Check 3V3, GND, SDA=D21 and SCL=D22.");
  }
}

void loop() {
  if (!mpuReady) {
    mpuReady = initializeMpu();
    if (mpuReady) {
      Serial.println("MPU-6050 found at 0x68.");
      Serial.println("t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
    } else {
      Serial.println("Waiting for MPU-6050 connection...");
      delay(1000);
      return;
    }
  }

  int16_t ax, ay, az, gx, gy, gz;

  if (!readMotion(ax, ay, az, gx, gy, gz)) {
    Serial.println("READ ERROR - hold the MPU board firmly against its header.");
    mpuReady = false;
    delay(500);
    return;
  }

  // Default ranges: accelerometer +/-2 g and gyroscope +/-250 degrees/s.
  Serial.printf("%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
                static_cast<unsigned long>(millis()),
                ax / 16384.0f, ay / 16384.0f, az / 16384.0f,
                gx / 131.0f, gy / 131.0f, gz / 131.0f);

  delay(100); // 10 readings per second for an easy first test.
}
