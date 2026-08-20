# FusionSense ESP32-CAM USB-serial firmware

This firmware replaces Wi-Fi with one persistent USB serial connection through
the ESP32-CAM-MB programmer. It captures QVGA JPEG frames at a 10 FPS target and
puts the sequence, device capture timestamp, dimensions, payload length,
cumulative capture-error count, and CRC32 in every binary packet.

## Upload

1. Plug the AI Thinker ESP32-CAM into the ESP32-CAM-MB socket with `5V` and
   `3.3V` aligned.
2. In Arduino IDE select **AI Thinker ESP32-CAM**, the programmer COM port, and
   an upload speed of **115200**.
3. Upload `fusionsense_camera_serial.ino`. The upload speed and runtime serial
   speed are separate settings.
4. Reset the camera and open Serial Monitor at **921600 baud**. Confirm:

   ```text
   # psram=true
   # local_test,result=PASS,...
   # ready,close_serial_monitor_then_run_laptop_receiver
   ```

5. Close Serial Monitor before starting the laptop receiver. Only one program
   can own the COM port.

## Run the camera-only gate

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_esp32_camera_serial.py --port COM10 --duration 30
```

Replace `COM10` with the ESP32-CAM-MB port. The receiver opens the port at
921600 baud, sends `START`, verifies every JPEG CRC, and sends `STOP` when the
gate finishes. Serial Monitor cannot display the binary frame stream.
