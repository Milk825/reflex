import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from analysis.autonomy.core.bridge import BridgeServer, _safe_unlink_socket


class BridgeServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.socket_path = str(Path(self.tmpdir.name) / "autonomy-test.sock")
        self.server = BridgeServer(self.socket_path, handler_timeout_ms=90)
        await self.server.start()

    async def asyncTearDown(self) -> None:
        await self.server.stop()
        self.tmpdir.cleanup()

    async def _request(self, method: str, path: str, body: dict | None = None):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        encoded = b""
        if body is not None:
            encoded = json.dumps(body).encode("utf-8")

        req = (
            f"{method} {path} HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(encoded)}\r\n"
            "\r\n"
        ).encode("utf-8") + encoded

        writer.write(req)
        await writer.drain()

        status = await reader.readline()
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, v = line.decode("utf-8").split(":", 1)
            headers[k.strip().lower()] = v.strip()

        body_len = int(headers.get("content-length", "0"))
        body_bytes = await reader.readexactly(body_len) if body_len else b"{}"
        writer.close()
        await writer.wait_closed()

        code = int(status.decode("utf-8").split(" ")[1])
        return code, json.loads(body_bytes.decode("utf-8"))

    async def test_health_endpoint(self):
        code, payload = await self._request("GET", "/health")
        self.assertEqual(code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("coordinator", payload)

    async def test_before_action_envelope(self):
        code, payload = await self._request(
            "POST",
            "/before_action",
            {
                "action_id": "a1",
                "hook_type": "before_action",
                "context": {"toolName": "read", "params": {"path": "README.md"}},
                "timestamp": time.time(),
            },
        )
        self.assertEqual(code, 200)
        self.assertIn("directive", payload)
        self.assertIn("confidence", payload)
        self.assertIn("metadata", payload)
        self.assertIn("meta_state", payload)

    async def test_blocking_high_risk_low_confidence(self):
        huge_cmd = "x" * 5000
        code, payload = await self._request(
            "POST",
            "/before_action",
            {
                "action_id": "a2",
                "hook_type": "before_action",
                "context": {"toolName": "exec", "params": {"command": f"rm -rf / {huge_cmd}"}},
                "timestamp": time.time(),
            },
        )
        self.assertEqual(code, 200)
        self.assertEqual(payload["directive"], "BLOCK")

    async def test_after_action_queue_non_blocking(self):
        code, payload = await self._request(
            "POST",
            "/after_action",
            {
                "action_id": "a3",
                "hook_type": "after_action",
                "context": {"session": {"sessionId": "s-a3"}, "toolName": "read", "ok": True, "durationMs": 4},
                "timestamp": time.time(),
            },
        )
        self.assertEqual(code, 200)
        self.assertEqual(payload["directive"], "PROCEED")
        self.assertTrue(payload["metadata"]["queued"])
        self.assertIn("micro_eval", payload)
        self.assertIn("eval_score", payload["micro_eval"])

    async def test_end_to_end_low_confidence_clarification(self):
        code, payload = await self._request(
            "POST",
            "/v1/preflight",
            {
                "action_id": "a4",
                "hook_type": "before_action",
                "context": {
                    "toolName": "browser",
                    "params": {"targetUrl": "https://example.com"},
                    "predictive": {"p_hist": 0.55, "sample_size": 0},
                    "health": {"overall_score": 70},
                    "ambiguity": 0.7,
                    "complexity": 0.6,
                    "meta_penalty": 0.6,
                },
                "timestamp": time.time(),
            },
        )

        self.assertEqual(code, 200)
        self.assertEqual(payload["directive"], "ASK_CLARIFICATION")

    async def test_assistant_turn_can_emit_reflection_and_lessons(self):
        session = {"sessionId": "bridge-reflect"}

        for idx in range(4):
            await self._request(
                "POST",
                "/after_action",
                {
                    "action_id": f"af-{idx}",
                    "hook_type": "after_action",
                    "context": {
                        "session": session,
                        "toolName": "exec" if idx % 2 else "read",
                        "ok": idx != 2,
                        "durationMs": 180 + idx * 20,
                        "retries": 2 if idx == 2 else 0,
                        "confidence": 0.9 if idx == 2 else 0.6,
                    },
                    "timestamp": time.time(),
                },
            )

        code, payload = await self._request(
            "POST",
            "/on_assistant_turn",
            {
                "action_id": "turn-1",
                "hook_type": "on_assistant_turn",
                "context": {
                    "session": session,
                    "taskBoundary": True,
                    "taskId": "task-bridge-reflect",
                    "sessionSummary": {
                        "taskId": "task-bridge-reflect",
                        "actionCount": 4,
                        "successCount": 3,
                        "failureCount": 1,
                        "mixedOutcomes": True,
                        "timeSpentMs": 180000,
                    },
                },
                "timestamp": time.time(),
            },
        )

        self.assertEqual(code, 200)
        self.assertIn("reflection", payload)
        self.assertIn("lesson_candidates", payload)
        self.assertGreaterEqual(len(payload["lesson_candidates"]), 1)

    async def test_start_refuses_to_remove_regular_file_at_socket_path(self):
        unsafe_path = Path(self.tmpdir.name) / "not-a-socket.sock"
        unsafe_path.write_text("keep-me")

        server = BridgeServer(str(unsafe_path), handler_timeout_ms=90)
        with self.assertRaises(RuntimeError):
            await server.start()

        self.assertTrue(unsafe_path.exists())
        self.assertEqual(unsafe_path.read_text(), "keep-me")

    async def test_safe_unlink_socket_does_not_delete_regular_file(self):
        unsafe_path = Path(self.tmpdir.name) / "still-not-a-socket.sock"
        unsafe_path.write_text("keep-me-too")

        removed = _safe_unlink_socket(str(unsafe_path))
        self.assertFalse(removed)
        self.assertTrue(unsafe_path.exists())
        self.assertEqual(unsafe_path.read_text(), "keep-me-too")


if __name__ == "__main__":
    unittest.main()
