# ESP32-CAM diagnostic firmware

This sketch tests the camera in layers so a camera/power fault is not confused
with a Wi-Fi streaming fault.

## Prepare and upload

1. Copy `secrets.example.h` to `secrets.h` in this directory and enter the
   current 2.4 GHz Wi-Fi credentials.
2. Open `camera_diagnostic.ino` in Arduino IDE.
3. Select **AI Thinker ESP32-CAM** and enable PSRAM.
4. Upload with GPIO 0 connected to GND. After upload, disconnect GPIO 0 from
   GND and press RESET.
5. Open Serial Monitor at **115200 baud**.

The serial output must first contain:

```text
# camera_init,result=PASS,psram=true
# local_self_test,result=PASS,...
```

That self-test captures 50 JPEG frames before Wi-Fi starts. A failure there
points to the board selection, power supply, ribbon cable, camera, or PSRAM.

Once `# diagnostic_ready` appears, run the laptop test from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_esp32_camera.py --host CAMERA_IP
```

The laptop test checks `/health`, three individual `/capture` JPEGs, and ten
seconds of the persistent timestamped stream on port 8080. Its final `result`
must be `PASS`.
