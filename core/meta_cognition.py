from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

JSONDict = Dict[str, Any]

_META_STATES = {"stable", "uncertain", "confused", "looping", "over_eager"}
_EDIT_TOOLS = {"edit", "write", "apply_patch"}
_REVERT_PATTERN = re.compile(r"\b(git\s+(?:restore|checkout\s+--)|revert|undo)\b", re.IGNORECASE)


@dataclass
class ActionEvent:
    timestamp: float
    tool_name: str
    args_signature: str
    success: bool
    confidence: Optional[float]
    ambiguity: float
    progress: float
    explicit_approval: bool
    long_workflow_requested: bool
    batch_operation: bool
    estimated_actions: Optional[int]
    alternative_available: bool
    is_edit: bool
    target_path: Optional[str]
    is_revert: bool
    error_type: Optional[str]


@dataclass
class MetaAssessment:
    state: str = "stable"
    confidence: float = 0.0
    penalty: float = 0.0
    triggers: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> JSONDict:
        return {
            "state": self.state,
            "confidence": self.confidence,
            "penalty": self.penalty,
            "triggers": self.triggers,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class _SessionState:
    session_key: str
    actions: Deque[ActionEvent]
    confidence_history: Deque[Tuple[float, float]]
    last_assessment: MetaAssessment = field(default_factory=MetaAssessment)
    last_clarification_request_at: Optional[float] = None
    last_clarification_response_at: Optional[float] = None
    explicit_approval_for_extension: bool = False
    long_workflow_requested: bool = False
    batch_mode: bool = False
    estimated_actions: Optional[int] = None
    health_score: float = 0.75
    last_edit_by_path: Dict[str, Tuple[str, str]] = field(default_factory=dict)


class MetaCognitionMonitor:
    """Chunk D meta-cognition monitor with fast rolling-window heuristics."""

    def __init__(
        self,
        *,
        max_actions: int = 180,
        confidence_window: int = 32,
    ) -> None:
        self._max_actions = max(32, int(max_actions))
        self._confidence_window = max(8, int(confidence_window))
        self._sessions: Dict[str, _SessionState] = {}

    def assess_before_action(self, context: JSONDict, *, now: Optional[float] = None) -> MetaAssessment:
        now_ts = float(now if now is not None else time.time())
        session = self._session_for(context)
        self._ingest_intent_hints(session, context)

        if _as_bool(context.get("clarificationProvided"), fallback=False):
            session.last_clarification_response_at = now_ts
        if _as_bool(context.get("clarificationNeeded"), fallback=False):
            session.last_clarification_request_at = now_ts

        assessment = self._compute_assessment(session, now_ts, context)
        session.last_assessment = assessment
        return assessment

    def update_after_action(self, context: JSONDict, *, now: Optional[float] = None) -> MetaAssessment:
        now_ts = float(now if now is not None else time.time())
        session = self._session_for(context)
        self._ingest_intent_hints(session, context)

        event = self._build_event(context, session, now_ts)
        session.actions.append(event)

        if event.confidence is not None:
            session.confidence_history.append((event.timestamp, event.confidence))

        if event.ambiguity >= 0.70 and not _as_bool(context.get("clarificationSought"), fallback=False):
            if session.last_clarification_request_at is None:
                session.last_clarification_request_at = event.timestamp

        if _as_bool(context.get("clarificationSought"), fallback=False):
            session.last_clarification_request_at = event.timestamp

        if _as_bool(context.get("clarificationProvided"), fallback=False):
            session.last_clarification_response_at = event.timestamp

        assessment = self._compute_assessment(session, now_ts, context)
        session.last_assessment = assessment
        return assessment

    def on_user_message(self, context: JSONDict, *, now: Optional[float] = None) -> MetaAssessment:
        now_ts = float(now if now is not None else time.time())
        session = self._session_for(context)
        self._ingest_intent_hints(session, context)

        text = str(context.get("prompt") or context.get("message") or "")
        lowered = text.lower()

        if "clarif" in lowered or "to be clear" in lowered or "i mean" in lowered:
            session.last_clarification_response_at = now_ts

        if any(token in lowered for token in ("yes continue", "go ahead", "proceed", "approved", "you can continue")):
            session.explicit_approval_for_extension = True

        if any(token in lowered for token in ("full workflow", "end to end", "all steps", "batch", "in bulk")):
            session.long_workflow_requested = True

        assessment = self._compute_assessment(session, now_ts, context)
        session.last_assessment = assessment
        return assessment

    def on_health_update(self, context: JSONDict, *, now: Optional[float] = None) -> MetaAssessment:
        now_ts = float(now if now is not None else time.time())
        session = self._session_for(context)
        self._ingest_intent_hints(session, context)

        raw_health = _extract_health_score(context)
        if raw_health is not None:
            session.health_score = raw_health

        assessment = self._compute_assessment(session, now_ts, context)
        session.last_assessment = assessment
        return assessment

    def get_meta_penalty(self, meta: str | MetaAssessment | JSONDict | None) -> float:
        if isinstance(meta, MetaAssessment):
            return _clamp(meta.penalty, lo=0.0, hi=0.95)

        if isinstance(meta, dict):
            state = str(meta.get("state") or "stable")
            confidence = _clamp(_as_float(meta.get("confidence"), fallback=0.0), lo=0.0, hi=1.0)
            return self._penalty_from_state(state, confidence)

        if isinstance(meta, str):
            return self._penalty_from_state(meta, 0.6)

        return 0.0

    def get_state(self, context: JSONDict) -> MetaAssessment:
        return self._session_for(context).last_assessment

    def stats(self) -> JSONDict:
        states = {"stable": 0, "uncertain": 0, "confused": 0, "looping": 0, "over_eager": 0}
        for session in self._sessions.values():
            state = session.last_assessment.state
            if state in states:
                states[state] += 1
        return {
            "sessions": len(self._sessions),
            "states": states,
        }

    def _session_for(self, context: JSONDict) -> _SessionState:
        session_key = _session_key(context)
        state = self._sessions.get(session_key)
        if state is not None:
            return state

        state = _SessionState(
            session_key=session_key,
            actions=deque(maxlen=self._max_actions),
            confidence_history=deque(maxlen=self._confidence_window),
        )
        self._sessions[session_key] = state
        return state

    def _ingest_intent_hints(self, session: _SessionState, context: JSONDict) -> None:
        estimated = _extract_estimated_actions(context)
        if estimated is not None and estimated > 0:
            session.estimated_actions = estimated

        if _as_bool(context.get("explicitUserApproval"), fallback=False) or _as_bool(
            context.get("approvedExtendedWorkflow"), fallback=False
        ):
            session.explicit_approval_for_extension = True

        if _as_bool(context.get("longWorkflowRequested"), fallback=False):
            session.long_workflow_requested = True

        if _as_bool(context.get("batchOperation"), fallback=False):
            session.batch_mode = True

    def _build_event(self, context: JSONDict, session: _SessionState, now_ts: float) -> ActionEvent:
        tool_name = str(context.get("toolName") or context.get("tool") or "").strip().lower() or "unknown"
        params = context.get("params") if isinstance(context.get("params"), dict) else {}
        args_signature = _signature(tool_name, params)

        success = _as_bool(context.get("ok"), fallback=not bool(context.get("error")))
        confidence = _normalize_optional_probability(context.get("confidence"))
        ambiguity = _normalize_probability(context.get("ambiguity"), default=0.25)

        progress = _normalize_optional_probability(
            context.get("progress")
            if context.get("progress") is not None
            else context.get("progressScore")
            if context.get("progressScore") is not None
            else context.get("incrementalProgress")
        )
        if progress is None:
            progress = 0.35 if success else 0.08

        explicit_approval = _as_bool(context.get("explicitUserApproval"), fallback=False) or session.explicit_approval_for_extension
        long_workflow_requested = _as_bool(context.get("longWorkflowRequested"), fallback=False) or session.long_workflow_requested
        batch_operation = _as_bool(context.get("batchOperation"), fallback=False) or session.batch_mode
        estimated_actions = _extract_estimated_actions(context) or session.estimated_actions

        alternative_available = _as_bool(context.get("alternativeAvailable"), fallback=False)
        if not alternative_available:
            alternative_available = _as_string(context.get("alternativeTool")) is not None
        if not alternative_available:
            alternative_available = tool_name in {"exec", "bash", "write"}

        is_edit = tool_name in _EDIT_TOOLS
        target_path = _extract_target_path(params)
        is_revert = _detect_revert(context=context, tool_name=tool_name, params=params)

        if is_edit and target_path:
            old_text = _as_string(params.get("oldText") or params.get("old_string"))
            new_text = _as_string(params.get("newText") or params.get("new_string"))
            if old_text is not None and new_text is not None:
                previous = session.last_edit_by_path.get(target_path)
                if previous and previous[0] == new_text and previous[1] == old_text:
                    is_revert = True
                session.last_edit_by_path[target_path] = (old_text, new_text)

        error_type = _as_string(context.get("errorType"))

        return ActionEvent(
            timestamp=now_ts,
            tool_name=tool_name,
            args_signature=args_signature,
            success=success,
            confidence=confidence,
            ambiguity=ambiguity,
            progress=_clamp(progress, lo=0.0, hi=1.0),
            explicit_approval=explicit_approval,
            long_workflow_requested=long_workflow_requested,
            batch_operation=batch_operation,
            estimated_actions=estimated_actions,
            alternative_available=alternative_available,
            is_edit=is_edit,
            target_path=target_path,
            is_revert=is_revert,
            error_type=error_type,
        )

    def _compute_assessment(self, session: _SessionState, now_ts: float, context: JSONDict) -> MetaAssessment:
        health_degradation = _clamp(max(0.0, 0.45 - session.health_score) * 1.4, lo=0.0, hi=0.9)

        scores: JSONDict = {
            "repetition_loop": self._score_repetition_loop(session, now_ts),
            "backtracking_churn": self._score_backtracking_churn(session, now_ts),
            "error_spike": self._score_error_spike(session),
            "ambiguity_unresolved": self._score_ambiguity_unresolved(session, now_ts, context),
            "over_eager_drift": self._score_over_eager_drift(session, context),
            "strategy_lock_in": self._score_strategy_lock_in(session),
            "confidence_collapse": self._score_confidence_collapse(session, now_ts),
            "health_degradation": health_degradation,
        }

        state, state_confidence = self._state_from_scores(scores)
        penalty = self._penalty_from_state(state, state_confidence)

        recent_tools = [a.tool_name for a in list(session.actions)[-6:]]
        metrics = {
            "recent_actions": len(session.actions),
            "recent_tool_sequence": recent_tools,
            "estimated_actions": session.estimated_actions,
            "health_score": session.health_score,
        }

        return MetaAssessment(
            state=state,
            confidence=state_confidence,
            penalty=penalty,
            triggers={k: v for k, v in scores.items() if v >= 0.45},
            metrics=metrics,
            timestamp=now_ts,
        )

    def _state_from_scores(self, scores: JSONDict) -> Tuple[str, float]:
        repetition = _as_float(scores.get("repetition_loop"), fallback=0.0)
        over_eager = _as_float(scores.get("over_eager_drift"), fallback=0.0)
        backtracking = _as_float(scores.get("backtracking_churn"), fallback=0.0)
        error_spike = _as_float(scores.get("error_spike"), fallback=0.0)
        ambiguity = _as_float(scores.get("ambiguity_unresolved"), fallback=0.0)
        strategy = _as_float(scores.get("strategy_lock_in"), fallback=0.0)
        collapse = _as_float(scores.get("confidence_collapse"), fallback=0.0)
        health = _as_float(scores.get("health_degradation"), fallback=0.0)

        if repetition >= 0.75:
            return "looping", _clamp(repetition, lo=0.0, hi=0.99)

        if over_eager >= 0.72:
            return "over_eager", _clamp(over_eager, lo=0.0, hi=0.99)

        confused_score = max(backtracking, error_spike * 0.95, strategy * 0.90, collapse * 0.90, health * 0.85)
        if confused_score >= 0.66 or (backtracking + error_spike + strategy + collapse + health) >= 1.45:
            return "confused", _clamp(confused_score, lo=0.0, hi=0.99)

        uncertain_score = max(ambiguity, error_spike * 0.65, collapse * 0.55, health * 0.8)
        if uncertain_score >= 0.45:
            return "uncertain", _clamp(uncertain_score, lo=0.0, hi=0.99)

        stable_conf = _clamp(1.0 - max(repetition, over_eager, confused_score, uncertain_score), lo=0.50, hi=0.99)
        return "stable", stable_conf

    def _penalty_from_state(self, state: str, confidence: float) -> float:
        normalized_state = state if state in _META_STATES else "stable"
        conf = _clamp(confidence, lo=0.0, hi=1.0)

        if normalized_state == "stable":
            return 0.0
        if normalized_state == "uncertain":
            return _clamp(0.10 + 0.20 * conf, lo=0.08, hi=0.35)
        if normalized_state == "confused":
            return _clamp(0.26 + 0.32 * conf, lo=0.22, hi=0.62)
        if normalized_state == "looping":
            return _clamp(0.52 + 0.34 * conf, lo=0.50, hi=0.92)
        if normalized_state == "over_eager":
            return _clamp(0.42 + 0.32 * conf, lo=0.40, hi=0.80)
        return 0.0

    def _score_repetition_loop(self, session: _SessionState, now_ts: float) -> float:
        cutoff = now_ts - 120.0
        counts: Dict[Tuple[str, str], int] = {}

        for action in reversed(session.actions):
            if action.timestamp < cutoff:
                break
            if action.success:
                continue
            key = (action.tool_name, action.args_signature)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= 3:
                return _clamp(0.80 + 0.05 * (counts[key] - 3), lo=0.0, hi=0.98)

        return 0.0

    def _score_backtracking_churn(self, session: _SessionState, now_ts: float) -> float:
        cutoff = now_ts - 300.0
        edits = [a for a in session.actions if a.is_edit and a.timestamp >= cutoff]
        if len(edits) < 4:
            return 0.0

        by_path: Dict[str, list[ActionEvent]] = {}
        for action in edits:
            if not action.target_path:
                continue
            by_path.setdefault(action.target_path, []).append(action)

        best = 0.0
        for path_actions in by_path.values():
            if len(path_actions) < 4:
                continue

            reverts = sum(1 for action in path_actions if action.is_revert)
            failures = sum(1 for action in path_actions if not action.success)
            churn = reverts * 0.30 + failures * 0.12 + max(0, len(path_actions) - 3) * 0.08
            best = max(best, churn)

        return _clamp(best, lo=0.0, hi=0.95)

    def _score_error_spike(self, session: _SessionState) -> float:
        actions = list(session.actions)
        if len(actions) < 8:
            return 0.0

        recent_n = min(8, max(4, len(actions) // 3))
        recent = actions[-recent_n:]
        baseline = actions[:-recent_n]
        if len(baseline) < 4:
            return 0.0

        recent_failure = sum(1 for a in recent if not a.success) / len(recent)
        baseline_failure = sum(1 for a in baseline if not a.success) / len(baseline)
        jump = recent_failure - baseline_failure

        if recent_failure >= 0.60 and jump >= 0.28:
            return _clamp(0.55 + jump, lo=0.0, hi=0.95)
        return 0.0

    def _score_ambiguity_unresolved(self, session: _SessionState, now_ts: float, context: JSONDict) -> float:
        explicit = _normalize_optional_probability(
            context.get("ambiguity")
            if context.get("ambiguity") is not None
            else context.get("ambiguityScore")
            if context.get("ambiguityScore") is not None
            else context.get("requestAmbiguity")
        )

        if explicit is None and session.actions:
            explicit = session.actions[-1].ambiguity

        ambiguity = explicit if explicit is not None else 0.0
        if ambiguity < 0.65:
            return 0.0

        last_request = session.last_clarification_request_at
        last_response = session.last_clarification_response_at

        if last_request is None:
            return _clamp(0.48 + 0.40 * ambiguity, lo=0.0, hi=0.92)

        if last_response is None or (last_response < last_request and (now_ts - last_request) > 20.0):
            return _clamp(0.45 + 0.35 * ambiguity, lo=0.0, hi=0.90)

        return 0.0

    def _score_over_eager_drift(self, session: _SessionState, context: JSONDict) -> float:
        estimated = _extract_estimated_actions(context) or session.estimated_actions
        if estimated is None or estimated <= 0:
            return 0.0

        actions = list(session.actions)
        actual_actions = len(actions)
        threshold = max(estimated * 1.8, estimated + 3)
        if actual_actions <= threshold:
            return 0.0

        recent = actions[-4:] if len(actions) >= 4 else actions
        if not recent:
            return 0.0

        progress = sum(a.progress for a in recent) / len(recent)
        success_rate = sum(1 for a in recent if a.success) / len(recent)
        explicit_approval = _as_bool(context.get("explicitUserApproval"), fallback=False) or session.explicit_approval_for_extension

        suppressors = (
            session.long_workflow_requested
            or session.batch_mode
            or _as_bool(context.get("longWorkflowRequested"), fallback=False)
            or _as_bool(context.get("batchOperation"), fallback=False)
            or success_rate >= 0.75
        )
        if suppressors:
            return 0.0

        if explicit_approval:
            return 0.0

        if progress >= 0.25:
            return 0.0

        exceed_ratio = (actual_actions - threshold) / max(1.0, threshold)
        raw = 0.62 + min(0.28, exceed_ratio) + (0.25 - progress)
        return _clamp(raw, lo=0.0, hi=0.97)

    def _score_strategy_lock_in(self, session: _SessionState) -> float:
        actions = list(session.actions)[-12:]
        if len(actions) < 5:
            return 0.0

        tool_stats: Dict[str, Dict[str, float]] = {}
        for action in actions:
            stats = tool_stats.setdefault(action.tool_name, {"n": 0.0, "success": 0.0, "alt": 0.0})
            stats["n"] += 1
            if action.success:
                stats["success"] += 1
            if action.alternative_available:
                stats["alt"] += 1

        best = 0.0
        for stats in tool_stats.values():
            if stats["n"] < 4:
                continue
            success_rate = stats["success"] / stats["n"]
            if success_rate >= 0.40:
                continue
            alt_ratio = stats["alt"] / stats["n"]
            if alt_ratio < 0.30:
                continue
            score = 0.52 + (0.40 - success_rate) + min(0.25, alt_ratio)
            best = max(best, score)

        return _clamp(best, lo=0.0, hi=0.93)

    def _score_confidence_collapse(self, session: _SessionState, now_ts: float) -> float:
        cutoff = now_ts - 180.0
        points = [(ts, conf) for ts, conf in session.confidence_history if ts >= cutoff]
        if len(points) < 3:
            return 0.0

        oldest_conf = points[0][1]
        latest_conf = points[-1][1]
        drop = oldest_conf - latest_conf
        if drop <= 0.25:
            return 0.0

        return _clamp(0.52 + (drop - 0.25) * 1.5, lo=0.0, hi=0.95)


def _extract_health_score(context: JSONDict) -> Optional[float]:
    health = context.get("health") if isinstance(context.get("health"), dict) else {}
    value = _as_float(
        health.get("normalized")
        if "normalized" in health
        else health.get("score")
        if "score" in health
        else health.get("overall")
        if "overall" in health
        else health.get("overall_score")
        if "overall_score" in health
        else context.get("health"),
        fallback=None,
    )
    if value is None:
        return None
    if value > 1.0:
        value = value / 100.0
    return _clamp(value, lo=0.0, hi=1.0)


def _extract_estimated_actions(context: JSONDict) -> Optional[int]:
    candidates: Iterable[Any] = (
        context.get("estimated_actions"),
        context.get("estimatedActions"),
        context.get("estimated_total_actions"),
        context.get("planSteps"),
    )
    for candidate in candidates:
        value = _as_float(candidate, fallback=None)
        if value is None:
            continue
        value_i = int(max(0, round(value)))
        if value_i > 0:
            return value_i

    plan = context.get("plan") if isinstance(context.get("plan"), dict) else {}
    for key in ("estimated_actions", "estimatedActions", "steps"):
        value = _as_float(plan.get(key), fallback=None)
        if value is None:
            continue
        value_i = int(max(0, round(value)))
        if value_i > 0:
            return value_i

    return None


def _session_key(context: JSONDict) -> str:
    session = context.get("session") if isinstance(context.get("session"), dict) else {}
    for key in (
        "session_key",
        "sessionKey",
        "session_id",
        "sessionId",
        "run_id",
        "runId",
    ):
        value = session.get(key)
        if isinstance(value, str) and value:
            return value

    for key in (
        "session_key",
        "sessionKey",
        "session_id",
        "sessionId",
        "run_id",
        "runId",
    ):
        value = context.get(key)
        if isinstance(value, str) and value:
            return value

    return "default"


def _signature(tool_name: str, params: JSONDict) -> str:
    serialized: str
    try:
        serialized = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        serialized = str(params)

    coarse = re.sub(r"\d+", "#", serialized.lower())
    if len(coarse) > 220:
        coarse = coarse[:220]
    return f"{tool_name}:{coarse}"


def _extract_target_path(params: JSONDict) -> Optional[str]:
    for key in ("path", "file_path", "filePath", "targetPath"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _detect_revert(*, context: JSONDict, tool_name: str, params: JSONDict) -> bool:
    if _as_bool(context.get("isRevert"), fallback=False):
        return True

    if tool_name in {"exec", "bash", "shell"}:
        command = params.get("command") or params.get("cmd")
        if isinstance(command, str) and _REVERT_PATTERN.search(command):
            return True

    old_text = _as_string(params.get("oldText") or params.get("old_string"))
    new_text = _as_string(params.get("newText") or params.get("new_string"))
    if old_text and new_text and old_text.strip() == new_text.strip():
        return True

    return False


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


def _as_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


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


def _normalize_probability(value: Any, *, default: float) -> float:
    parsed = _as_float(value, fallback=default)
    if parsed is None:
        parsed = default
    return _clamp(parsed, lo=0.0, hi=1.0)


def _normalize_optional_probability(value: Any) -> Optional[float]:
    parsed = _as_float(value, fallback=None)
    if parsed is None:
        return None
    return _clamp(parsed, lo=0.0, hi=1.0)


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
