from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OUTPUT_DIRECTORY = Path(
    r"C:\Users\SHIRDITHAN\Pictures\FusionSense"
)

MAX_IMAGE_SIZE = 5_000_000


class ImageReceiver(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/upload":
            self.send_error(404, "Use POST /upload")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return

        if length <= 0 or length > MAX_IMAGE_SIZE:
            self.send_error(413, "Invalid image size")
            return

        if "image/jpeg" not in self.headers.get("Content-Type", "").lower():
            self.send_error(415, "Expected image/jpeg")
            return

        image = self.rfile.read(length)

        if len(image) != length:
            self.send_error(400, "Incomplete upload")
            return

        if not image.startswith(b"\xff\xd8"):
            self.send_error(400, "Invalid JPEG header")
            return

        if not image.endswith(b"\xff\xd9"):
            self.send_error(400, "Incomplete JPEG")
            return

        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        destination = OUTPUT_DIRECTORY / f"esp32cam_{timestamp}.jpg"
        temporary = destination.with_suffix(".jpg.part")

        temporary.write_bytes(image)
        temporary.replace(destination)

        message = f"Saved {destination}"
        print(f"{message} ({len(image) / 1024:.1f} KB)")

        response = message.encode("utf-8")

        self.send_response(201)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        print(f"{self.client_address[0]}: {format % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8080), ImageReceiver)

    print("ESP32-CAM receiver running")
    print(f"Saving images to: {OUTPUT_DIRECTORY}")
    print("Listening on port 8080")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReceiver stopped")
    finally:
        server.server_close()