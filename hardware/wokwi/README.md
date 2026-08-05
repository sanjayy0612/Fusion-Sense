# FusionSense — Wokwi hardware simulation

Runs the ESP32/Arduino-style sensor-gateway firmware **with no physical parts**, so you can
demonstrate a working hardware implementation for the review. In the updated
prototype, this gateway streams samples to the laptop inference API.

## How to run (no downloads, no parts)

1. Go to https://wokwi.com → **New Project** → **ESP32**.
2. Click the `diagram.json` tab and paste the contents of `diagram.json` (this file's folder).
3. Click `sketch.ino` and paste the contents of
   `../esp32_firmware/fusionsense_gateway.ino` (keep `RADAR_PRESENT 0`).
4. Press **Play**. Open the Serial Monitor.
5. You'll see the CSV stream:
   `t_ms,ax,ay,az,gx,gy,gz,radar_dist_cm,radar_energy`
6. Drag the MPU-6050 in the canvas / change its acceleration values to see the
   IMU numbers respond — that's your live "hardware" reading motion.
7. Screenshot the canvas + serial monitor for the deck (Physical Design /
   Hardware Implementation slides).

## What this proves for the review

- The ESP32 firmware compiles and runs.
- I2C communication with the MPU-6050 works and produces the exact CSV format
  the FusionSense pipeline expects (matches the `FusionWindow` IMU channels).
- The acquisition loop runs at the target 50 Hz.

## What's simulated vs. real

| Component | Wokwi | Real hardware |
|-----------|-------|---------------|
| ESP32 | ✅ full | ✅ |
| MPU-6050 IMU (I2C) | ✅ full | ✅ |
| LD2410 radar (UART) | ❌ not a Wokwi part — code is guarded by `RADAR_PRESENT` | ✅ set `RADAR_PRESENT 1` |
| Camera | n/a (handled on the laptop API, not the ESP32/Arduino gateway) | ✅ webcam/laptop camera |

When you get the real LD2410, set `RADAR_PRESENT 1`, wire TX→GPIO16 / RX→GPIO17,
and the same sketch streams radar distance + energy alongside the IMU.
