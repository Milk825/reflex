#!/usr/bin/env python3
"""Autonomy bridge server for OpenClaw TypeScript plugin (Chunks A-G).

Protocol:
  - Transport: HTTP/1.1 over Unix Domain Socket
  - Request envelope: {action_id, hook_type, context, timestamp}
  - Response envelope: {directive, confidence, metadata, timestamp}

Design goals:
  - async server + concurrent handlers
  - strict timeout (<100ms) per request
  - fail-open friendly responses (defaults to PROCEED)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import stat
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from .integration import AutonomyCoordinator

JSONDict = Dict[str, Any]
Handler = Callable[[str, JSONDict], Awaitable[Tuple[int, JSONDict]]]


@dataclass
class RequestEnvelope:
    action_id: str
    hook_type: str
    context: JSONDict
    timestamp: float


@dataclass
class ResponseEnvelope:
    directive: str
    confidence: float
    metadata: JSONDict
    timestamp: float

    def as_dict(self) -> JSONDict:
        return {
            "directive": self.directive,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class BridgeServer:
    """HTTP/JSON bridge on Unix Domain Socket."""

    def __init__(
        self,
        socket_path: str,
        *,
        handler_timeout_ms: int = 90,
        max_event_history: int = 1000,
        logger: Optional[logging.Logger] = None,
        coordinator: Optional[AutonomyCoordinator] = None,
    ) -> None:
        self.socket_path = str(Path(socket_path).expanduser())
        self.handler_timeout_ms = max(1, min(99, int(handler_timeout_ms)))
        self.max_event_history = max(1, int(max_event_history))
        self.logger = logger or logging.getLogger("autonomy.bridge")

        self._server: Optional[asyncio.AbstractServer] = None
        self._shutdown = asyncio.Event()
        self._events: Deque[JSONDict] = deque(maxlen=self.max_event_history)
        self._started_at = 0.0
        self._coordinator = coordinator or AutonomyCoordinator()

        self._post_routes: Dict[str, Handler] = {
            "/before_action": self._handle_before_action,
            "/after_action": self._handle_after_action,
            "/on_user_message": self._handle_on_user_message,
            "/on_assistant_turn": self._handle_on_assistant_turn,
            "/on_health_update": self._handle_on_health_update,
            # v1 compatibility aliases from broader design doc
            "/v1/preflight": self._handle_before_action,
            "/v1/postflight": self._handle_after_action,
            "/v1/message-event": self._handle_on_user_message,
            "/v1/run-end": self._handle_on_assistant_turn,
            "/v1/health": self._handle_on_health_update,
            "/v1/mode": self._handle_set_mode,
        }

    async def start(self) -> None:
        if self._server is not None:
            return

        socket_dir = os.path.dirname(self.socket_path)
        if socket_dir:
            os.makedirs(socket_dir, exist_ok=True)

        if os.path.lexists(self.socket_path):
            removed = _safe_unlink_socket(self.socket_path, logger=self.logger)
            if not removed:
                raise RuntimeError(
                    f"Refusing to remove non-socket path at '{self.socket_path}'. "
                    "Fix --socket-path to point to a UNIX socket location."
                )

        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)
        self._started_at = time.time()
        await self._coordinator.start()
        self.logger.info("Bridge server listening on UDS %s", self.socket_path)

    async def stop(self) -> None:
        self._shutdown.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        await self._coordinator.stop()

        try:
            _safe_unlink_socket(self.socket_path, logger=self.logger)
        except OSError:
            self.logger.warning("Failed removing socket: %s", self.socket_path, exc_info=True)

    async def serve_forever(self) -> None:
        await self.start()
        await self._shutdown.wait()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return

            line = request_line.decode("utf-8", errors="replace").strip()
            parts = line.split(" ")
            if len(parts) < 2:
                await self._write_json(writer, 400, {"error": "invalid request line"})
                return

            method = parts[0].upper()
            path = parts[1]
            headers = await self._read_headers(reader)
            body_bytes = b""

            content_length = int(headers.get("content-length", "0") or "0")
            if content_length > 0:
                body_bytes = await reader.readexactly(content_length)

            status, payload = await asyncio.wait_for(
                self._dispatch(method, path, body_bytes),
                timeout=self.handler_timeout_ms / 1000.0,
            )
            await self._write_json(writer, status, payload)
        except asyncio.TimeoutError:
            self._coordinator.record_fail_open(reason="handler_timeout")
            fail_open = self._fail_open("handler_timeout")
            await self._write_json(writer, 504, fail_open)
        except asyncio.IncompleteReadError:
            await self._write_json(writer, 400, {"error": "incomplete request body"})
        except Exception:
            self.logger.exception("Unhandled bridge request error")
            self._coordinator.record_fail_open(reason="internal_error")
            fail_open = self._fail_open("internal_error")
            await self._write_json(writer, 500, fail_open)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch(self, method: str, path: str, body_bytes: bytes) -> Tuple[int, JSONDict]:
        parsed = urlsplit(path)
        route = parsed.path
        query = parse_qs(parsed.query, keep_blank_values=False)

        if method == "GET":
            if route in {"/health", "/v1/status"}:
                return 200, {
                    "status": "ok",
                    "socket_path": self.socket_path,
                    "uptime_s": max(0.0, time.time() - self._started_at),
                    "timeout_ms": self.handler_timeout_ms,
                    "events_recorded": len(self._events),
                    "coordinator": self._coordinator.stats(),
                    "autonomy": self._coordinator.mode_status(),
                    "timestamp": time.time(),
                }

            if route == "/v1/recent":
                limit = _query_int(query, "limit", default=10)
                return 200, {
                    "mode": self._coordinator.current_mode(),
                    "decisions": self._coordinator.recent_decisions(limit=limit),
                    "timestamp": time.time(),
                }

            if route == "/v1/calibration":
                return 200, {
                    "mode": self._coordinator.current_mode(),
                    "calibration": self._coordinator.calibration_summary(),
                    "timestamp": time.time(),
                }

            return 404, {"error": "not_found", "path": route}

        if method != "POST":
            return 405, {"error": "method_not_allowed", "allowed": ["GET", "POST"]}

        handler = self._post_routes.get(route)
        if handler is None:
            return 404, {"error": "not_found", "path": route}

        try:
            payload = json.loads(body_bytes.decode("utf-8") if body_bytes else "{}")
        except json.JSONDecodeError:
            return 400, {"error": "invalid_json"}

        envelope = self._normalize_request(payload)
        return await handler(route, envelope)

    async def _handle_before_action(self, path: str, envelope: JSONDict) -> Tuple[int, JSONDict]:
        context = envelope["context"]
        decision = await self._coordinator.before_action(context)
        payload = decision.as_response()
        payload["metadata"] = {
            **(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
            "endpoint": path,
            "action_id": envelope["action_id"],
        }

        response = ResponseEnvelope(
            directive=str(payload.get("directive", "PROCEED")),
            confidence=float(payload.get("confidence", 0.0)),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            timestamp=float(payload.get("timestamp", time.time())),
        )

        self._record_event(path, envelope, response)

        out = response.as_dict()
        meta_block = response.metadata.get("meta") if isinstance(response.metadata.get("meta"), dict) else {}
        if meta_block:
            out["meta_state"] = meta_block.get("state")
            out["meta_confidence"] = meta_block.get("confidence")

        health_block = response.metadata.get("health") if isinstance(response.metadata.get("health"), dict) else {}
        adaptive_block = response.metadata.get("adaptive") if isinstance(response.metadata.get("adaptive"), dict) else {}
        intervention_block = (
            response.metadata.get("intervention") if isinstance(response.metadata.get("intervention"), dict) else {}
        )
        if health_block:
            out["health"] = health_block
        if adaptive_block:
            out["adaptive"] = adaptive_block
        if intervention_block:
            out["intervention"] = intervention_block

        communication = response.metadata.get("communication") if isinstance(response.metadata.get("communication"), dict) else {}
        if communication:
            out["communication"] = communication

        return 200, out

    async def _handle_after_action(self, path: str, envelope: JSONDict) -> Tuple[int, JSONDict]:
        context = envelope["context"]
        result = await self._coordinator.after_action(context)

        meta = self._coordinator.current_meta_state(context).as_dict()  # lightweight snapshot only
        health = self._coordinator.current_health_state(context)
        adaptive = self._coordinator.current_adaptive_state(context)
        interventions = self._coordinator.current_intervention_state(context)

        response = ResponseEnvelope(
            directive="PROCEED",
            confidence=0.65,
            metadata={
                "endpoint": path,
                "action_id": envelope.get("action_id"),
                "queued": bool(result.get("queued")),
                "micro_eval": result.get("micro_eval") if isinstance(result.get("micro_eval"), dict) else {},
                "deep_eval_queued": bool(result.get("deep_eval_queued", False)),
                "lesson_candidates": result.get("lesson_candidates") if isinstance(result.get("lesson_candidates"), list) else [],
                "meta": meta,
                "health": health,
                "adaptive": adaptive,
                "interventions": interventions,
            },
            timestamp=time.time(),
        )
        self._record_event(path, envelope, response)

        out = response.as_dict()
        out["meta_state"] = meta.get("state")
        out["meta_confidence"] = meta.get("confidence")
        out["health"] = health
        out["adaptive"] = adaptive
        out["interventions"] = interventions
        out["micro_eval"] = response.metadata.get("micro_eval")
        out["deep_eval_queued"] = response.metadata.get("deep_eval_queued")
        out["lesson_candidates"] = response.metadata.get("lesson_candidates")
        return 200, out

    async def _handle_on_user_message(self, path: str, envelope: JSONDict) -> Tuple[int, JSONDict]:
        meta = await self._coordinator.on_user_message(envelope["context"])
        response = self._default_proceed(path, envelope, confidence=0.7)
        response.metadata["meta"] = meta.as_dict()
        self._record_event(path, envelope, response)

        out = response.as_dict()
        out["meta_state"] = meta.state
        out["meta_confidence"] = meta.confidence
        return 200, out

    async def _handle_on_assistant_turn(self, path: str, envelope: JSONDict) -> Tuple[int, JSONDict]:
        context = envelope["context"]
        meta = await self._coordinator.on_assistant_turn(context)
        notification = self._coordinator.consume_reflection_notification(context)

        response = self._default_proceed(path, envelope, confidence=0.68)
        response.metadata["meta"] = meta.as_dict()

        if notification is not None:
            reflection = notification.get("reflection") if isinstance(notification.get("reflection"), dict) else {}
            lessons = notification.get("lesson_candidates") if isinstance(notification.get("lesson_candidates"), list) else []
            response.metadata["reflection"] = reflection
            response.metadata["lesson_candidates"] = lessons
            completion_message = notification.get("completion_message")
            if isinstance(completion_message, str) and completion_message:
                response.metadata["completion_message"] = completion_message

        self._record_event(path, envelope, response)

        out = response.as_dict()
        out["meta_state"] = meta.state
        out["meta_confidence"] = meta.confidence
        if isinstance(response.metadata.get("reflection"), dict):
            out["reflection"] = response.metadata.get("reflection")
        if isinstance(response.metadata.get("lesson_candidates"), list):
            out["lesson_candidates"] = response.metadata.get("lesson_candidates")
        if isinstance(response.metadata.get("completion_message"), str):
            out["completion_message"] = response.metadata.get("completion_message")
        return 200, out

    async def _handle_on_health_update(self, path: str, envelope: JSONDict) -> Tuple[int, JSONDict]:
        context = envelope["context"]
        meta = await self._coordinator.on_health_update(context)
        health = self._coordinator.current_health_state(context)
        adaptive = self._coordinator.current_adaptive_state(context)
        interventions = self._coordinator.current_intervention_state(context)

        response = self._default_proceed(path, envelope, confidence=0.6)
        response.metadata["meta"] = meta.as_dict()
        response.metadata["health"] = health
        response.metadata["adaptive"] = adaptive
        response.metadata["interventions"] = interventions
        self._record_event(path, envelope, response)

        out = response.as_dict()
        out["meta_state"] = meta.state
        out["meta_confidence"] = meta.confidence
        out["health"] = health
        out["adaptive"] = adaptive
        out["interventions"] = interventions
        return 200, out

    async def _handle_set_mode(self, path: str, envelope: JSONDict) -> Tuple[int, JSONDict]:
        context = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
        mode = str(context.get("mode") or "")
        force = bool(context.get("force", False))
        result = self._coordinator.switch_mode(mode, force=force)

        status = 200 if bool(result.get("ok")) else 409
        payload = {
            **result,
            "currentMode": self._coordinator.current_mode(),
            "timestamp": time.time(),
        }
        return status, payload

    def _default_proceed(self, path: str, envelope: JSONDict, *, confidence: float) -> ResponseEnvelope:
        return ResponseEnvelope(
            directive="PROCEED",
            confidence=confidence,
            metadata={
                "endpoint": path,
                "action_id": envelope.get("action_id"),
            },
            timestamp=time.time(),
        )

    def _normalize_request(self, payload: Any) -> JSONDict:
        if not isinstance(payload, dict):
            payload = {}

        action_id = str(payload.get("action_id") or uuid.uuid4())
        hook_type = str(payload.get("hook_type") or "unknown")
        timestamp = payload.get("timestamp", time.time())

        context = payload.get("context")
        if not isinstance(context, dict):
            context = {}

        # Allow light-weight payloads for control endpoints (e.g., /v1/mode).
        if not context:
            if "mode" in payload:
                context["mode"] = payload.get("mode")
            if "force" in payload:
                context["force"] = payload.get("force")

        return {
            "action_id": action_id,
            "hook_type": hook_type,
            "context": context,
            "timestamp": float(timestamp),
        }

    def _record_event(self, endpoint: str, envelope: JSONDict, response: ResponseEnvelope) -> None:
        meta_state: Optional[str] = None
        if isinstance(response.metadata.get("meta"), dict):
            value = response.metadata["meta"].get("state")
            if isinstance(value, str):
                meta_state = value

        self._events.append(
            {
                "endpoint": endpoint,
                "action_id": envelope.get("action_id"),
                "hook_type": envelope.get("hook_type"),
                "directive": response.directive,
                "confidence": response.confidence,
                "meta_state": meta_state,
                "timestamp": response.timestamp,
            }
        )

    def _fail_open(self, reason: str) -> JSONDict:
        return {
            "directive": "PROCEED",
            "confidence": 0.0,
            "metadata": {
                "failOpen": True,
                "reason": reason,
            },
            "timestamp": time.time(),
        }

    async def _read_headers(self, reader: asyncio.StreamReader) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break

            text = line.decode("utf-8", errors="replace").strip()
            if ":" not in text:
                continue
            k, v = text.split(":", 1)
            headers[k.strip().lower()] = v.strip()

        return headers

    async def _write_json(self, writer: asyncio.StreamWriter, status: int, payload: JSONDict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        reason = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            409: "Conflict",
            500: "Internal Server Error",
            504: "Gateway Timeout",
        }.get(status, "OK")

        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8")

        writer.write(header)
        writer.write(body)
        await writer.drain()


def _safe_unlink_socket(socket_path: str, *, logger: Optional[logging.Logger] = None) -> bool:
    """Remove path only when it is a UNIX socket.

    Returns True when a socket was removed, False otherwise.
    """
    try:
        st = os.lstat(socket_path)
    except FileNotFoundError:
        return False

    if stat.S_ISSOCK(st.st_mode):
        os.remove(socket_path)
        return True

    if logger is not None:
        logger.warning("Refusing to unlink non-socket path: %s", socket_path)
    return False


def _query_int(query: Dict[str, list[str]], key: str, *, default: int) -> int:
    values = query.get(key)
    if not values:
        return default
    try:
        parsed = int(values[0])
    except (TypeError, ValueError):
        return default
    return max(1, min(100, parsed))


async def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = BridgeServer(args.socket_path, handler_timeout_ms=args.timeout_ms)

    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        loop.create_task(server.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            # add_signal_handler may not exist on some platforms/tests.
            pass

    await server.start()
    await server.serve_forever()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomy UDS bridge server")
    parser.add_argument(
        "--socket-path",
        default="~/.openclaw/state/autonomy/autonomy.sock",
        help="Unix socket path for HTTP bridge",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=90,
        help="Per-request timeout in milliseconds (clamped to <100ms)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.socket_path = str(Path(args.socket_path).expanduser())
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
