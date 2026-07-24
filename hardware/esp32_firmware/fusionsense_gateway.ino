/*
 * FusionSense — ESP32 Sensor Gateway firmware
 * -------------------------------------------------
 * Role: bare-metal sensor acquisition + timestamping + framing.
 * Reads:
 *   - MPU-6050 6-DoF IMU over I2C  (accel x/y/z + gyro x/y/z)
 *   - HLK-LD2410 mmWave radar over UART (target distance + energy)
 * Emits: one CSV line per sample over USB serial @ 115200, consumed by the
 *        edge node (Raspberry Pi, or your laptop during development).
 *
 * CSV format (one row per IMU tick, ~50 Hz):
 *   t_ms, ax, ay, az, gx, gy, gz, radar_dist_cm, radar_energy
 *
 * Wiring (see hardware/wokwi/diagram.json):
 *   MPU-6050:  SDA->GPIO21, SCL->GPIO22, VCC->3V3, GND->GND
 *   LD2410:    TX->GPIO16 (ESP32 RX2), RX->GPIO17 (ESP32 TX2), VCC->5V, GND->GND
 *
 * NOTE: In the Wokwi simulation only the MPU-6050 is wired (Wokwi has no
 * LD2410 part). RADAR_PRESENT guards the radar code so the same sketch runs
 * both in simulation and on real hardware.
 */

#include <Wire.h>

// ---- toggle radar for Wokwi vs real hardware ----
#define RADAR_PRESENT 0        // set to 1 when the LD2410 is physically wired

// ---- MPU-6050 registers ----
#define MPU_ADDR      0x68
#define REG_PWR_MGMT  0x6B
#define REG_ACCEL_X   0x3B

const uint32_t SAMPLE_HZ = 50;
const uint32_t SAMPLE_US = 1000000UL / SAMPLE_HZ;
uint32_t last_us = 0;

// radar state (updated by LD2410 frames)
volatile uint16_t radar_dist_cm = 0;
volatile uint8_t  radar_energy  = 0;

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg); Wire.write(val);
  Wire.endTransmission();
}

void mpuInit() {
  Wire.begin(21, 22);          // SDA, SCL
  mpuWrite(REG_PWR_MGMT, 0x00); // wake up
  delay(100);
}

// read 6 signed 16-bit values (accel[3] + temp + gyro[3]); we skip temp
void mpuRead(int16_t* a, int16_t* g) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_X);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  a[0] = (Wire.read() << 8) | Wire.read();
  a[1] = (Wire.read() << 8) | Wire.read();
  a[2] = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();              // temperature (discard)
  g[0] = (Wire.read() << 8) | Wire.read();
  g[1] = (Wire.read() << 8) | Wire.read();
  g[2] = (Wire.read() << 8) | Wire.read();
}

#if RADAR_PRESENT
// Minimal LD2410 UART frame parser (engineering/basic mode).
// The LD2410 streams frames on Serial2; we extract detection distance + energy.
void radarPoll() {
  static uint8_t buf[64]; static int idx = 0;
  while (Serial2.available()) {
    uint8_t b = Serial2.read();
    // frame header F4 F3 F2 F1 ... tail F8 F7 F6 F5 (basic target frame)
    buf[idx++] = b;
    if (idx >= (int)sizeof(buf)) idx = 0;
    if (idx >= 13 && buf[idx-4]==0xF8 && buf[idx-3]==0xF7) {
      // bytes 8-9: moving target distance (cm), byte 10: moving energy
      radar_dist_cm = buf[8] | (buf[9] << 8);
      radar_energy  = buf[10];
      idx = 0;
    }
  }
}
#endif

void setup() {
  Serial.begin(115200);
  mpuInit();
#if RADAR_PRESENT
  Serial2.begin(256000, SERIAL_8N1, 16, 17);  // LD2410 default baud
#endif
  Serial.println("# FusionSense gateway online");
  Serial.println("# t_ms,ax,ay,az,gx,gy,gz,radar_dist_cm,radar_energy");
}

void loop() {
#if RADAR_PRESENT
  radarPoll();
#endif
  uint32_t now = micros();
  if (now - last_us >= SAMPLE_US) {
    last_us = now;
    int16_t a[3], g[3];
    mpuRead(a, g);
    // stream CSV row
    Serial.print(millis());        Serial.print(',');
    Serial.print(a[0]); Serial.print(','); Serial.print(a[1]); Serial.print(','); Serial.print(a[2]); Serial.print(',');
    Serial.print(g[0]); Serial.print(','); Serial.print(g[1]); Serial.print(','); Serial.print(g[2]); Serial.print(',');
    Serial.print(radar_dist_cm);   Serial.print(',');
    Serial.println(radar_energy);
  }
}
