"""Control-plane client for the FusionSense ESP32-CAM."""

from __future__ import annotations

from http.client import HTTPConnection
import json
import threading
import time
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request

from fusionsense.data.clock_sync import SyncObservation


class CameraControlError(RuntimeError):
    """The ESP32-CAM control endpoint failed or returned invalid data."""


def normalize_esp32_host(host: str) -> str:
    value = host.strip().rstrip("/")
    if value.startswith(("http://", "https://")):
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0].split(":", 1)[0]
    if not value:
        raise ValueError("ESP32-CAM host must not be empty")
    return value


class CameraControlClient:
    """Configure sessions, probe the camera clock, and retrieve health."""

    def __init__(
        self,
        host: str,
        *,
        port: int = 80,
        timeout: float = 3.0,
        opener: Callable[..., object] | None = None,
        connection_factory: Callable[..., object] = HTTPConnection,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.host = normalize_esp32_host(host)
        self.port = port
        self.timeout = timeout
        self._opener = opener
        self._connection_factory = connection_factory
        self._clock_ns = clock_ns
        self._connection: object | None = None
        self._lock = threading.Lock()

    def _url(self, path: str, parameters: dict[str, str] | None = None) -> str:
        query = f"?{urlencode(parameters)}" if parameters else ""
        return f"http://{self.host}:{self.port}{path}{query}"

    def _request_json(
        self, path: str, parameters: dict[str, str] | None = None
    ) -> dict[str, object]:
        if self._opener is not None:
            return self._request_json_with_opener(path, parameters)

        target = path
        if parameters:
            target += f"?{urlencode(parameters)}"
        last_error: Exception | None = None
        with self._lock:
            for _attempt in range(2):
                try:
                    if self._connection is None:
                        self._connection = self._connection_factory(
                            self.host, self.port, timeout=self.timeout
                        )
                    self._connection.request(
                        "GET",
                        target,
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                    response = self._connection.getresponse()
                    payload = response.read()
                    if not 200 <= int(response.status) < 300:
                        raise CameraControlError(
                            f"ESP32-CAM returned HTTP {response.status} for {target}"
                        )
                    return self._decode_json(payload)
                except Exception as error:
                    last_error = error
                    self._close_unlocked()
        raise CameraControlError(
            f"ESP32-CAM persistent control request failed: {target}"
        ) from last_error

    def _request_json_with_opener(
        self, path: str, parameters: dict[str, str] | None
    ) -> dict[str, object]:
        assert self._opener is not None
        request = Request(
            self._url(path, parameters),
            headers={"Cache-Control": "no-cache"},
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                payload = response.read()
        except Exception as error:
            raise CameraControlError(
                f"ESP32-CAM control request failed: {request.full_url}"
            ) from error
        return self._decode_json(payload)

    @staticmethod
    def _decode_json(payload: bytes) -> dict[str, object]:
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CameraControlError("ESP32-CAM returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise CameraControlError("ESP32-CAM control response is not an object")
        return decoded

    def _close_unlocked(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def set_session(self, session_id: str) -> dict[str, object]:
        response = self._request_json("/session", {"id": session_id})
        if response.get("status") != "ok" or response.get("session_id") != session_id:
            raise CameraControlError("ESP32-CAM did not acknowledge the session")
        return response

    def sync(self, request_id: str) -> SyncObservation:
        host_send_ns = self._clock_ns()
        response = self._request_json("/sync", {"id": request_id})
        host_receive_ns = self._clock_ns()
        try:
            device_id = str(response["device_id"])
            returned_request_id = str(response["request_id"])
            device_time_us = int(response["device_time_us"])
        except (KeyError, TypeError, ValueError) as error:
            raise CameraControlError("invalid ESP32-CAM sync response") from error
        if returned_request_id != request_id:
            raise CameraControlError("ESP32-CAM sync request ID mismatch")
        return SyncObservation(
            device_id=device_id,
            request_id=request_id,
            device_time_us=device_time_us,
            host_send_ns=host_send_ns,
            host_receive_ns=host_receive_ns,
        )

    def health(self) -> dict[str, object]:
        return self._request_json("/health")
