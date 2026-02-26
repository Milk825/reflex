import asyncio
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from analysis.autonomy.bridge import BridgeServer


class AutonomyCliTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.socket_path = str(Path(self.tmpdir.name) / "autonomy-cli.sock")
        self.server = BridgeServer(self.socket_path, handler_timeout_ms=90)
        await self.server.start()

    async def asyncTearDown(self) -> None:
        await self.server.stop()
        self.tmpdir.cleanup()

    async def _run_cli(self, *args: str):
        cmd = [
            "python3",
            "-m",
            "analysis.cli",
            "--socket-path",
            self.socket_path,
            "telemetry",
            "autonomy",
            *args,
        ]

        def _run():
            return subprocess.run(
                cmd,
                cwd="/home/willclawd/.openclaw/workspace",
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        proc = await asyncio.to_thread(_run)
        payload = json.loads(proc.stdout or "{}")
        return proc.returncode, payload

    async def test_status_and_recent_commands(self):
        await self.server._dispatch(
            "POST",
            "/before_action",
            json.dumps(
                {
                    "action_id": "cli-a1",
                    "hook_type": "before_action",
                    "context": {
                        "toolName": "read",
                        "params": {"path": "README.md"},
                        "predictive": {"p_hist": 0.9, "sample_size": 10},
                        "health": {"overall_score": 95},
                        "ambiguity": 0.1,
                        "complexity": 0.1,
                    },
                    "timestamp": time.time(),
                }
            ).encode("utf-8"),
        )

        code, status_payload = await self._run_cli("status")
        self.assertEqual(code, 0)
        self.assertEqual(status_payload["http_status"], 200)
        self.assertIn("autonomy", status_payload)

        code, recent_payload = await self._run_cli("recent")
        self.assertEqual(code, 0)
        self.assertEqual(recent_payload["http_status"], 200)
        self.assertGreaterEqual(len(recent_payload.get("decisions", [])), 1)

    async def test_calibration_and_mode_switch(self):
        # Seed calibration so the endpoint has non-empty output.
        await self.server._coordinator._process_after_action(
            {
                "session": {"sessionId": "cli-calib"},
                "toolName": "read",
                "ok": True,
                "confidence": 0.8,
                "mode": "shadow",
            }
        )

        code, calibration_payload = await self._run_cli("calibration")
        self.assertEqual(code, 0)
        self.assertEqual(calibration_payload["http_status"], 200)
        self.assertIn("calibration", calibration_payload)

        code, mode_payload = await self._run_cli("mode", "suggest")
        self.assertEqual(code, 0)
        self.assertEqual(mode_payload["http_status"], 200)
        self.assertTrue(mode_payload.get("ok"))
        self.assertEqual(mode_payload.get("to"), "suggest")


if __name__ == "__main__":
    unittest.main()
