import json
import unittest
from io import BytesIO

from fusionsense.data.camera_control import CameraControlClient


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class CameraControlTests(unittest.TestCase):
    def test_default_control_connection_is_reused(self) -> None:
        created_connections = []

        class Response:
            status = 200

            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

        class Connection:
            def __init__(self, host, port, timeout):
                self.requests = []
                self.responses = []
                created_connections.append(self)

            def request(self, method, target, headers):
                self.requests.append((method, target, headers))
                if target.startswith("/session"):
                    self.responses.append(
                        Response(
                            {
                                "status": "ok",
                                "device_id": "cam01",
                                "session_id": "fusion_test",
                            }
                        )
                    )
                else:
                    request_id = target.split("id=", 1)[1]
                    self.responses.append(
                        Response(
                            {
                                "status": "ok",
                                "device_id": "cam01",
                                "request_id": request_id,
                                "device_time_us": 123456,
                            }
                        )
                    )

            def getresponse(self):
                return self.responses.pop(0)

            def close(self):
                pass

        clock_values = iter([10_000_000, 11_000_000, 20_000_000, 21_000_000])
        client = CameraControlClient(
            "192.168.1.5",
            connection_factory=Connection,
            clock_ns=lambda: next(clock_values),
        )
        client.set_session("fusion_test")
        first = client.sync("c1")
        second = client.sync("c2")
        client.close()

        self.assertEqual(len(created_connections), 1)
        self.assertEqual(first.rtt_ns, 1_000_000)
        self.assertEqual(second.rtt_ns, 1_000_000)
        self.assertEqual(len(created_connections[0].requests), 3)
        self.assertTrue(
            all(
                request[2]["Connection"] == "keep-alive"
                for request in created_connections[0].requests
            )
        )

    def test_session_and_sync_contract(self) -> None:
        requested_urls: list[str] = []

        def opener(request, timeout):
            requested_urls.append(request.full_url)
            if "/session" in request.full_url:
                payload = {
                    "status": "ok",
                    "device_id": "cam01",
                    "session_id": "fusion_test",
                }
            else:
                payload = {
                    "status": "ok",
                    "device_id": "cam01",
                    "request_id": "c000001",
                    "device_time_us": 123456,
                }
            return FakeResponse(json.dumps(payload).encode("utf-8"))

        clock_values = iter([10_000_000, 12_000_000])
        client = CameraControlClient(
            "http://192.168.1.5:81/stream",
            opener=opener,
            clock_ns=lambda: next(clock_values),
        )
        client.set_session("fusion_test")
        observation = client.sync("c000001")

        self.assertEqual(observation.device_id, "cam01")
        self.assertEqual(observation.rtt_ns, 2_000_000)
        self.assertIn("http://192.168.1.5:80/session?id=fusion_test", requested_urls)
        self.assertIn("http://192.168.1.5:80/sync?id=c000001", requested_urls)


if __name__ == "__main__":
    unittest.main()
