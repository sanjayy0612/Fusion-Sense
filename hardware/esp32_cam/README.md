# ESP32-CAM camera node

The ESP32-CAM is FusionSense V1's primary camera. It captures OV2640 JPEG frames
and exposes a Wi-Fi MJPEG stream. All pose extraction and ML remain on the
laptop.

## Firmware

Use Espressif's Arduino example:

```text
File > Examples > ESP32 > Camera > CameraWebServer
```

Select `CAMERA_MODEL_AI_THINKER`, set Wi-Fi credentials, select the **AI Thinker
ESP32-CAM** board, enable PSRAM, and upload through the ESP32-CAM-MB. Start with
QVGA (320x240), JPEG, and approximately 10-15 FPS.

After boot, verify both endpoints in a browser:

```text
http://<esp32-ip>/
http://<esp32-ip>:81/stream
```

Then test the complete laptop path:

```bash
python scripts/download_pose_model.py
python scripts/test_esp32_camera.py --host <esp32-ip>
```

The regular wearable ESP32 and MPU-6050 are a separate node. Do not connect the
OV2640 ribbon camera to that board.

