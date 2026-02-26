from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

JSONDict = Dict[str, Any]


@dataclass
class MicroEvalResult:
    success: float
    efficiency: float
    expectation_alignment: float
    eval_score: float
    duration_ms: float
    expected_duration_ms: float
    predicted_confidence: Optional[float]
    deep_eval_queued: bool
    deep_eval_reasons: List[str]
    timestamp: float

    def as_dict(self) -> JSONDict:
        return {
            "success": self.success,
            "efficiency": self.efficiency,
            "expectation_alignment": self.expectation_alignment,
            "eval_score": self.eval_score,
            "duration_ms": self.duration_ms,
            "expected_duration_ms": self.expected_duration_ms,
            "predicted_confidence": self.predicted_confidence,
            "deep_eval_queued": self.deep_eval_queued,
            "deep_eval_reasons": list(self.deep_eval_reasons),
            "timestamp": self.timestamp,
        }


class SelfEvaluator:
    """Chunk F self-evaluation engine.

    - micro_eval(): synchronous hot-path scoring (<2ms target)
    - deep_eval(): queued background analysis for significant events
    """

    def __init__(
        self,
        *,
        duration_window: int = 120,
        deep_eval_queue_size: int = 512,
        deep_eval_history: int = 200,
    ) -> None:
        self._duration_window = max(20, int(duration_window))
        self._durations_by_tool: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self._duration_window))

        self._deep_queue: "queue.Queue[JSONDict]" = queue.Queue(maxsize=max(32, int(deep_eval_queue_size)))
        self._deep_results: Deque[JSONDict] = deque(maxlen=max(20, int(deep_eval_history)))
        self._deep_result_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._deep_worker, name="autonomy-deep-eval", daemon=True)
        self._worker.start()

        self.micro_evals = 0
        self.deep_eval_triggered = 0
        self.deep_eval_completed = 0
        self.deep_eval_dropped = 0
        self.deep_eval_sync_wait_over_1ms = 0
        self.micro_eval_over_2ms = 0

        self._calibration_pairs: Deque[tuple[float, float]] = deque(maxlen=800)
        self._calibration_error = 0.0

    def stop(self) -> None:
        self._shutdown.set()
        if self._worker.is_alive():
            self._worker.join(timeout=0.5)

    def micro_eval(self, context: JSONDict) -> MicroEvalResult:
        started = time.perf_counter()
        context = context if isinstance(context, dict) else {}

        tool_name = self._tool_name(context)
        success_bool = bool(context.get("ok")) and not bool(context.get("error"))
        success = 1.0 if success_bool else 0.0

        duration_ms = max(0.0, _as_float(context.get("durationMs", context.get("duration_ms")), fallback=0.0))
        expected_duration_ms = self._expected_duration_ms(tool_name=tool_name, context=context)

        efficiency = self._efficiency_score(duration_ms=duration_ms, expected_duration_ms=expected_duration_ms)

        predicted_confidence = _as_float(context.get("confidence"), fallback=None)
        if predicted_confidence is not None:
            predicted_confidence = _clamp(predicted_confidence, lo=0.0, hi=1.0)

        expectation_alignment = self._expectation_alignment(predicted_confidence=predicted_confidence, success=success)
        eval_score = _clamp(0.6 * success + 0.25 * efficiency + 0.15 * expectation_alignment, lo=0.0, hi=1.0)

        if duration_ms > 0:
            self._durations_by_tool[tool_name].append(duration_ms)

        if predicted_confidence is not None:
            self._calibration_pairs.append((predicted_confidence, success))
            self._calibration_error = self._brier_error()

        reasons = self.should_deep_eval(context, predicted_confidence=predicted_confidence, success=success, duration_ms=duration_ms)
        deep_eval_queued = False
        if reasons:
            deep_eval_queued = self._enqueue_deep_eval(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time(),
                    "toolName": tool_name,
                    "context": dict(context),
                    "reasons": reasons,
                    "micro_eval": {
                        "success": success,
                        "efficiency": efficiency,
                        "expectation_alignment": expectation_alignment,
                        "eval_score": eval_score,
                        "duration_ms": duration_ms,
                        "expected_duration_ms": expected_duration_ms,
                        "predicted_confidence": predicted_confidence,
                    },
                }
            )
            if deep_eval_queued:
                self.deep_eval_triggered += 1

        result = MicroEvalResult(
            success=success,
            efficiency=efficiency,
            expectation_alignment=expectation_alignment,
            eval_score=eval_score,
            duration_ms=duration_ms,
            expected_duration_ms=expected_duration_ms,
            predicted_confidence=predicted_confidence,
            deep_eval_queued=deep_eval_queued,
            deep_eval_reasons=reasons,
            timestamp=time.time(),
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.micro_evals += 1
        if elapsed_ms > 2.0:
            self.micro_eval_over_2ms += 1

        return result

    def should_deep_eval(
        self,
        context: JSONDict,
        *,
        predicted_confidence: Optional[float],
        success: float,
        duration_ms: float,
    ) -> List[str]:
        reasons: List[str] = []

        if success <= 0.0:
            reasons.append("failure")

        retries = int(max(0.0, _as_float(context.get("retries", context.get("retryCount", 0.0)), fallback=0.0)))
        if retries >= 2:
            reasons.append("retries>=2")

        if predicted_confidence is not None:
            mismatch = abs(predicted_confidence - success)
            if mismatch >= 0.65:
                reasons.append("confidence_outcome_mismatch")

        if duration_ms > 0.0:
            tool_name = self._tool_name(context)
            p90 = self._p90_duration_ms(tool_name)
            if p90 is not None and duration_ms > p90:
                reasons.append("slow_gt_p90")

        deduped: List[str] = []
        seen = set()
        for reason in reasons:
            if reason in seen:
                continue
            seen.add(reason)
            deduped.append(reason)
        return deduped

    def consume_deep_evaluations(self, *, session_key: Optional[str] = None, limit: int = 50) -> List[JSONDict]:
        if limit <= 0:
            return []

        with self._deep_result_lock:
            items = list(self._deep_results)
            if session_key is not None:
                items = [item for item in items if _session_key(item.get("context") if isinstance(item, dict) else {}) == session_key]

            if not items:
                return []

            selected = items[-limit:]
            selected_ids = {item.get("id") for item in selected}
            self._deep_results = deque(
                [item for item in self._deep_results if item.get("id") not in selected_ids],
                maxlen=self._deep_results.maxlen,
            )
            return selected

    def calibration_snapshot(self) -> JSONDict:
        return {
            "pairs": len(self._calibration_pairs),
            "brier": self._calibration_error,
        }

    def stats(self) -> JSONDict:
        success_rate = 0.0
        if self.micro_evals > 0:
            # Rough success estimate via stored calibration outcomes when available.
            outcomes = [obs for _, obs in self._calibration_pairs]
            if outcomes:
                success_rate = sum(outcomes) / len(outcomes)

        return {
            "micro_evals": self.micro_evals,
            "deep_eval_triggered": self.deep_eval_triggered,
            "deep_eval_completed": self.deep_eval_completed,
            "deep_eval_dropped": self.deep_eval_dropped,
            "deep_eval_sync_wait_over_1ms": self.deep_eval_sync_wait_over_1ms,
            "micro_eval_over_2ms": self.micro_eval_over_2ms,
            "deep_queue_depth": self._deep_queue.qsize(),
            "calibration": self.calibration_snapshot(),
            "estimated_success_rate": success_rate,
        }

    def _enqueue_deep_eval(self, payload: JSONDict) -> bool:
        started = time.perf_counter()
        try:
            self._deep_queue.put_nowait(payload)
            waited_ms = (time.perf_counter() - started) * 1000.0
            if waited_ms > 1.0:
                self.deep_eval_sync_wait_over_1ms += 1
            return True
        except queue.Full:
            self.deep_eval_dropped += 1
            return False

    def _deep_worker(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._deep_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self.deep_eval(item)
                with self._deep_result_lock:
                    self._deep_results.append(result)
                self.deep_eval_completed += 1
            except Exception:
                self.deep_eval_dropped += 1
            finally:
                self._deep_queue.task_done()

    def deep_eval(self, event: JSONDict) -> JSONDict:
        """Heavier background analysis for significant events."""
        context = event.get("context") if isinstance(event.get("context"), dict) else {}
        micro_eval = event.get("micro_eval") if isinstance(event.get("micro_eval"), dict) else {}
        reasons = event.get("reasons") if isinstance(event.get("reasons"), list) else []

        success = _as_float(micro_eval.get("success"), fallback=0.0) or 0.0
        efficiency = _as_float(micro_eval.get("efficiency"), fallback=0.0) or 0.0
        expectation_alignment = _as_float(micro_eval.get("expectation_alignment"), fallback=0.0) or 0.0

        severity = "medium"
        if "failure" in reasons or success <= 0.0:
            severity = "high"
        if "confidence_outcome_mismatch" in reasons and success <= 0.0:
            severity = "critical"

        recommendation = "maintain_strategy"
        if "slow_gt_p90" in reasons:
            recommendation = "optimize_tool_path"
        if "retries>=2" in reasons:
            recommendation = "switch_or_clarify"
        if "confidence_outcome_mismatch" in reasons:
            recommendation = "recalibrate_confidence"

        return {
            "id": event.get("id") or str(uuid.uuid4()),
            "timestamp": time.time(),
            "session_key": _session_key(context),
            "toolName": self._tool_name(context),
            "reasons": list(reasons),
            "severity": severity,
            "summary": {
                "success": success,
                "efficiency": efficiency,
                "expectation_alignment": expectation_alignment,
                "eval_score": _as_float(micro_eval.get("eval_score"), fallback=0.0) or 0.0,
            },
            "recommendation": recommendation,
        }

    def _p90_duration_ms(self, tool_name: str) -> Optional[float]:
        samples = self._durations_by_tool.get(tool_name)
        if not samples or len(samples) < 10:
            return None
        ordered = sorted(samples)
        index = int(round(0.9 * (len(ordered) - 1)))
        return float(ordered[index])

    def _expected_duration_ms(self, *, tool_name: str, context: JSONDict) -> float:
        explicit = _as_float(context.get("expectedDurationMs", context.get("expected_duration_ms")), fallback=None)
        if explicit is not None and explicit > 0:
            return explicit

        p90 = self._p90_duration_ms(tool_name)
        if p90 is not None and p90 > 0:
            return p90

        defaults = {
            "read": 8.0,
            "web_fetch": 22.0,
            "web_search": 26.0,
            "edit": 28.0,
            "write": 30.0,
            "browser": 45.0,
            "exec": 55.0,
            "bash": 55.0,
        }
        return defaults.get(tool_name, 30.0)

    def _efficiency_score(self, *, duration_ms: float, expected_duration_ms: float) -> float:
        if expected_duration_ms <= 0:
            return 1.0

        ratio = duration_ms / expected_duration_ms if duration_ms > 0 else 0.0
        if ratio <= 1.0:
            return 1.0
        if ratio >= 3.0:
            return 0.0
        return _clamp(1.0 - ((ratio - 1.0) / 2.0), lo=0.0, hi=1.0)

    def _expectation_alignment(self, *, predicted_confidence: Optional[float], success: float) -> float:
        if predicted_confidence is None:
            return 0.6 if success > 0.0 else 0.4
        return _clamp(1.0 - abs(predicted_confidence - success), lo=0.0, hi=1.0)

    def _tool_name(self, context: JSONDict) -> str:
        tool = str(context.get("toolName") or context.get("tool") or "unknown").strip().lower()
        return tool if tool else "unknown"

    def _brier_error(self) -> float:
        if not self._calibration_pairs:
            return 0.0
        return sum((pred - obs) ** 2 for pred, obs in self._calibration_pairs) / len(self._calibration_pairs)


def _session_key(context: JSONDict) -> str:
    context = context if isinstance(context, dict) else {}
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


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
