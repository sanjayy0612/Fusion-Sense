"""Serve the local FusionSense dashboard without third-party dependencies."""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import webbrowser


class DashboardHandler(SimpleHTTPRequestHandler):
    """Prevent stale inference JSON and UI assets during live polling."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    parser.add_argument(
        "--data",
        default="mentor_demo.json",
        help="dashboard JSON file, for example session_output.json",
    )
    args = parser.parse_args()

    dashboard_dir = Path(__file__).resolve().parents[1] / "dashboard"
    os.chdir(dashboard_dir)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}/?data={args.data}"
    print(f"FusionSense dashboard: {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
