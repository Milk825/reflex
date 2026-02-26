from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

JSONDict = Dict[str, Any]


@dataclass
class ReflectionRecord:
    id: str
    session_key: str
    task_id: str
    timestamp: float
    summary: str
    action_count: int
    success_count: int
    failure_count: int
    mixed_outcomes: bool
    time_spent_ms: float
    lesson_candidates: List[JSONDict]

    def as_dict(self) -> JSONDict:
        return {
            "id": self.id,
            "session_key": self.session_key,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "action_count": self.action_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "mixed_outcomes": self.mixed_outcomes,
            "time_spent_ms": self.time_spent_ms,
            "lesson_candidates": list(self.lesson_candidates),
        }


@dataclass
class _SessionReflectionState:
    last_reflection_at: float = 0.0
    reflected_task_ids: Set[str] = field(default_factory=set)


class AutoReflectionEngine:
    """Chunk F auto-reflection engine with significance + guardrails."""

    def __init__(
        self,
        *,
        min_spacing_seconds: int = 300,
        min_actions: int = 4,
        min_time_spent_ms: int = 120_000,
    ) -> None:
        self._min_spacing_seconds = max(30, int(min_spacing_seconds))
        self._min_actions = max(2, int(min_actions))
        self._min_time_spent_ms = max(30_000, int(min_time_spent_ms))
        self._sessions: Dict[str, _SessionReflectionState] = {}

        self.generated = 0
        self.skipped_guardrail = 0
        self.skipped_trivial = 0

    def evaluate(self, context: JSONDict, *, now: Optional[float] = None, deep_evaluations: Optional[List[JSONDict]] = None) -> Optional[ReflectionRecord]:
        context = context if isinstance(context, dict) else {}
        now_ts = float(now if now is not None else time.time())
        session_key = _session_key(context)
        state = self._sessions.setdefault(session_key, _SessionReflectionState())

        task_id = self._task_id(context)
        boundary = _as_bool(context.get("taskBoundary"), fallback=False) or _as_bool(context.get("sessionEnd"), fallback=False)

        summary = context.get("sessionSummary") if isinstance(context.get("sessionSummary"), dict) else {}
        action_count = int(max(0.0, _as_float(summary.get("actionCount", context.get("actionCount", 0.0)), fallback=0.0)))
        success_count = int(max(0.0, _as_float(summary.get("successCount", context.get("successCount", 0.0)), fallback=0.0)))
        failure_count = int(max(0.0, _as_float(summary.get("failureCount", context.get("failureCount", 0.0)), fallback=0.0)))

        if action_count <= 0 and isinstance(summary.get("recentActions"), list):
            action_count = len(summary.get("recentActions", []))
        if success_count <= 0 and isinstance(summary.get("recentActions"), list):
            success_count = sum(1 for item in summary.get("recentActions", []) if isinstance(item, dict) and item.get("ok") is True)
        if failure_count <= 0 and isinstance(summary.get("recentActions"), list):
            failure_count = sum(1 for item in summary.get("recentActions", []) if isinstance(item, dict) and item.get("ok") is False)

        if success_count + failure_count < action_count:
            success_count = max(success_count, action_count - failure_count)

        mixed_outcomes = success_count > 0 and failure_count > 0

        time_spent_ms = _as_float(summary.get("timeSpentMs", context.get("durationMs")), fallback=0.0) or 0.0
        if time_spent_ms <= 0 and isinstance(summary.get("recentActions"), list) and summary.get("recentActions"):
            first = summary["recentActions"][0] if isinstance(summary["recentActions"][0], dict) else {}
            last = summary["recentActions"][-1] if isinstance(summary["recentActions"][-1], dict) else {}
            first_ts = _as_float(first.get("timestamp"), fallback=0.0) or 0.0
            last_ts = _as_float(last.get("timestamp"), fallback=0.0) or 0.0
            if last_ts > first_ts:
                time_spent_ms = max(time_spent_ms, last_ts - first_ts)

        significant = self._is_significant_work(
            boundary=boundary,
            action_count=action_count,
            mixed_outcomes=mixed_outcomes,
            time_spent_ms=time_spent_ms,
            success_count=success_count,
            failure_count=failure_count,
        )
        if not significant:
            self.skipped_trivial += 1
            return None

        if task_id in state.reflected_task_ids:
            self.skipped_guardrail += 1
            return None

        if state.last_reflection_at > 0 and (now_ts - state.last_reflection_at) < self._min_spacing_seconds:
            self.skipped_guardrail += 1
            return None

        reflection_summary = self.generate_reflection(
            task_id=task_id,
            action_count=action_count,
            success_count=success_count,
            failure_count=failure_count,
            mixed_outcomes=mixed_outcomes,
            time_spent_ms=time_spent_ms,
            deep_evaluations=deep_evaluations or [],
        )

        lesson_candidates = self._build_lesson_candidates(
            summary_text=reflection_summary,
            task_id=task_id,
            action_count=action_count,
            success_count=success_count,
            failure_count=failure_count,
            deep_evaluations=deep_evaluations or [],
        )

        record = ReflectionRecord(
            id=str(uuid.uuid4()),
            session_key=session_key,
            task_id=task_id,
            timestamp=now_ts,
            summary=reflection_summary,
            action_count=action_count,
            success_count=success_count,
            failure_count=failure_count,
            mixed_outcomes=mixed_outcomes,
            time_spent_ms=time_spent_ms,
            lesson_candidates=lesson_candidates,
        )

        state.reflected_task_ids.add(task_id)
        state.last_reflection_at = now_ts
        self.generated += 1
        return record

    def stats(self) -> JSONDict:
        return {
            "generated": self.generated,
            "skipped_guardrail": self.skipped_guardrail,
            "skipped_trivial": self.skipped_trivial,
            "sessions": len(self._sessions),
            "min_spacing_seconds": self._min_spacing_seconds,
        }

    def _task_id(self, context: JSONDict) -> str:
        explicit = context.get("taskId") or context.get("task_id")
        if isinstance(explicit, str) and explicit:
            return explicit

        summary = context.get("sessionSummary") if isinstance(context.get("sessionSummary"), dict) else {}
        for key in ("taskId", "task_id", "boundaryId", "runId", "run_id"):
            value = summary.get(key)
            if isinstance(value, str) and value:
                return value

        return f"{_session_key(context)}:{int(time.time()) // 300}"

    def _is_significant_work(
        self,
        *,
        boundary: bool,
        action_count: int,
        mixed_outcomes: bool,
        time_spent_ms: float,
        success_count: int,
        failure_count: int,
    ) -> bool:
        if not boundary:
            return False

        multiple_actions = action_count >= self._min_actions
        enough_time = time_spent_ms >= self._min_time_spent_ms

        if not (multiple_actions or mixed_outcomes or enough_time):
            return False

        # Skip trivial one-shot outcomes.
        if action_count <= 1 and failure_count == 0 and not enough_time:
            return False

        if action_count <= 2 and failure_count == 0 and not mixed_outcomes and not enough_time:
            return False

        return True

    def generate_reflection(
        self,
        *,
        task_id: str,
        action_count: int,
        success_count: int,
        failure_count: int,
        mixed_outcomes: bool,
        time_spent_ms: float,
        deep_evaluations: List[JSONDict],
    ) -> str:
        minutes = max(0.0, time_spent_ms / 60000.0)
        tone = "stable"
        if mixed_outcomes:
            tone = "mixed"
        elif failure_count > 0:
            tone = "recovery"

        deep_flags: List[str] = []
        for item in deep_evaluations[-3:]:
            if not isinstance(item, dict):
                continue
            severity = item.get("severity")
            recommendation = item.get("recommendation")
            if isinstance(severity, str) and severity:
                deep_flags.append(severity)
            if isinstance(recommendation, str) and recommendation:
                deep_flags.append(recommendation)

        deep_hint = ""
        if deep_flags:
            deep_hint = f" Deep-eval signals: {', '.join(sorted(set(deep_flags))[:3])}."

        return (
            f"Task {task_id} completed with {tone} outcomes: "
            f"{success_count}/{action_count} actions succeeded, {failure_count} failed over {minutes:.1f} min."
            f" Primary lesson: preserve successful pattern, and tighten validation before risky/slow retries."
            f"{deep_hint}"
        )

    def _build_lesson_candidates(
        self,
        *,
        summary_text: str,
        task_id: str,
        action_count: int,
        success_count: int,
        failure_count: int,
        deep_evaluations: List[JSONDict],
    ) -> List[JSONDict]:
        base_conf = 0.72
        if failure_count == 0 and success_count >= action_count:
            base_conf = 0.66
        elif failure_count > 0:
            base_conf = 0.78

        candidates: List[JSONDict] = [
            {
                "id": str(uuid.uuid4()),
                "source": "auto_reflection",
                "task_id": task_id,
                "text": summary_text,
                "confidence": _clamp(base_conf, lo=0.5, hi=0.95),
                "evidence": {
                    "action_count": action_count,
                    "success_count": success_count,
                    "failure_count": failure_count,
                },
            }
        ]

        for item in deep_evaluations[-2:]:
            if not isinstance(item, dict):
                continue
            recommendation = item.get("recommendation")
            if not isinstance(recommendation, str) or not recommendation:
                continue

            candidates.append(
                {
                    "id": str(uuid.uuid4()),
                    "source": "deep_eval",
                    "task_id": task_id,
                    "text": f"When {recommendation}, prefer safer fallback before repeating failing action.",
                    "confidence": 0.74,
                    "evidence": {
                        "deep_eval_id": item.get("id"),
                        "severity": item.get("severity"),
                        "reasons": item.get("reasons") if isinstance(item.get("reasons"), list) else [],
                    },
                }
            )

        return candidates


def _session_key(context: JSONDict) -> str:
    session = context.get("session") if isinstance(context.get("session"), dict) else {}
    for key in ("session_key", "sessionKey", "session_id", "sessionId", "run_id", "runId"):
        value = session.get(key)
        if isinstance(value, str) and value:
            return value

    for key in ("session_key", "sessionKey", "session_id", "sessionId", "run_id", "runId"):
        value = context.get(key)
        if isinstance(value, str) and value:
            return value

    return "default"


def _as_float(value: Any, fallback: Optional[float]) -> Optional[float]:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return fallback
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            return fallback
        if math.isnan(parsed) or math.isinf(parsed):
            return fallback
        return parsed
    return fallback


def _as_bool(value: Any, *, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return fallback


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
