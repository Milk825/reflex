from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from analysis.adaptive.engine import AdaptiveBehaviorEngine
from analysis.health.monitor import HealthMonitor
from analysis.interventions.engine import InterventionEngine, InterventionPriority, InterventionRecord

from .auto_reflection import AutoReflectionEngine
from .communication import CommunicationPolicy
from .meta_cognition import MetaAssessment, MetaCognitionMonitor
from .preflight import PreflightAssessment, PreflightChecker
from .rollout import RolloutTracker
from .self_evaluation import SelfEvaluator

JSONDict = Dict[str, Any]


@dataclass
class AutonomyDecision:
    directive: str
    confidence: float
    reason: str
    metadata: JSONDict

    def as_response(self) -> JSONDict:
        return {
            "directive": self.directive,
            "confidence": self.confidence,
            "metadata": {
                "reason": self.reason,
                **self.metadata,
            },
            "timestamp": time.time(),
        }


@dataclass
class ConfidenceResult:
    confidence: float
    band: str
    directive_hint: str
    signals: JSONDict
    sample_penalty: float
    raw_score: float
    calibrated_temperature: float


class LessonPipeline:
    """Chunk F lesson candidate ingestion pipeline.

    Flow: candidate -> namespace resolution -> tier assignment -> confirmation queue.
    """

    def __init__(self, *, confirmation_queue_maxlen: int = 2000) -> None:
        self._confirmation_queue: Deque[JSONDict] = deque(maxlen=max(100, int(confirmation_queue_maxlen)))
        self._dedupe_keys: Deque[str] = deque(maxlen=5000)
        self._dedupe_index = set()

        self.ingest_attempts = 0
        self.ingest_success = 0

    def ingest_candidates(self, candidates: List[JSONDict], *, session_key: str, source: str) -> List[JSONDict]:
        accepted: List[JSONDict] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            text = str(candidate.get("text") or "").strip()
            if not text:
                continue

            namespace = self._resolve_namespace(candidate)
            confidence = _as_float(candidate.get("confidence"), fallback=0.65)
            confidence = max(0.0, min(1.0, confidence if confidence is not None else 0.65))
            tier = self._assign_tier(candidate, confidence=confidence)

            dedupe_key = f"{namespace}::{text.lower()[:240]}"
            if dedupe_key in self._dedupe_index:
                continue

            self.ingest_attempts += 1

            item = {
                "id": candidate.get("id") if isinstance(candidate.get("id"), str) and candidate.get("id") else f"lesson-{int(time.time() * 1000)}",
                "session_key": session_key,
                "source": str(candidate.get("source") or source),
                "namespace": namespace,
                "tier": tier,
                "confidence": confidence,
                "text": text,
                "task_id": candidate.get("task_id") if isinstance(candidate.get("task_id"), str) else None,
                "evidence": candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {},
                "timestamp": time.time(),
                "status": "pending_confirmation",
            }

            self._confirmation_queue.append(item)
            self._remember_dedupe(dedupe_key)
            self.ingest_success += 1
            accepted.append(item)

        return accepted

    def drain_confirmation_queue(self, limit: int = 50) -> List[JSONDict]:
        take = max(0, int(limit))
        out: List[JSONDict] = []
        for _ in range(min(take, len(self._confirmation_queue))):
            out.append(self._confirmation_queue.popleft())
        return out

    def ingest_success_rate(self) -> float:
        if self.ingest_attempts <= 0:
            return 1.0
        return self.ingest_success / self.ingest_attempts

    def stats(self) -> JSONDict:
        return {
            "attempts": self.ingest_attempts,
            "success": self.ingest_success,
            "success_rate": self.ingest_success_rate(),
            "confirmation_queue_depth": len(self._confirmation_queue),
        }

    def _resolve_namespace(self, candidate: JSONDict) -> str:
        explicit = candidate.get("namespace")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower()

        text = str(candidate.get("text") or "").lower()
        source = str(candidate.get("source") or "").lower()

        if any(token in text for token in ("retry", "slow", "timeout", "fallback", "tool")):
            return "execution.tools"
        if any(token in text for token in ("clarif", "confirm", "question", "ambiguous")):
            return "interaction.dialogue"
        if source in {"auto_reflection", "deep_eval"}:
            return "autonomy.reflection"
        return "autonomy.general"

    def _assign_tier(self, candidate: JSONDict, *, confidence: float) -> str:
        explicit = candidate.get("tier")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower()

        if confidence >= 0.82:
            return "high"
        if confidence >= 0.62:
            return "medium"
        return "low"

    def _remember_dedupe(self, key: str) -> None:
        self._dedupe_keys.append(key)
        self._dedupe_index.add(key)
        while len(self._dedupe_index) > self._dedupe_keys.maxlen:
            oldest = self._dedupe_keys.popleft()
            self._dedupe_index.discard(oldest)


class ConfidenceScorer:
    """Hot-path confidence scorer for before_action decisions."""

    def __init__(
        self,
        *,
        default_temperature: float = 1.0,
        calibration_window: int = 500,
    ) -> None:
        self._temperature = self._clamp(default_temperature, lo=0.5, hi=2.5)
        self._calibration: Deque[Tuple[float, float]] = deque(maxlen=max(10, calibration_window))

    @property
    def temperature(self) -> float:
        return self._temperature

    def score_before_action(
        self,
        *,
        p_hist: float,
        health: float,
        risk: float,
        ambiguity: float,
        complexity: float,
        meta_penalty: float,
        sample_size: int,
        temperature: Optional[float] = None,
    ) -> ConfidenceResult:
        p_hist = self._unit(p_hist)
        health = self._unit(health)
        risk = self._unit(risk)
        ambiguity = self._unit(ambiguity)
        complexity = self._unit(complexity)
        meta_penalty = self._unit(meta_penalty)

        raw = (
            0.35 * p_hist
            + 0.20 * health
            + 0.15 * (1.0 - risk)
            + 0.10 * (1.0 - ambiguity)
            + 0.10 * (1.0 - complexity)
            + 0.10 * (1.0 - meta_penalty)
        )

        n_similar = max(0, int(sample_size))
        sample_penalty = min(0.15, 0.15 * math.exp(-n_similar / 5.0))
        raw = self._clamp(raw - sample_penalty, lo=0.01, hi=0.99)

        calibrated_temperature = self._clamp(temperature if temperature is not None else self._temperature, lo=0.5, hi=2.5)
        confidence = self._sigmoid(self._logit(raw) / calibrated_temperature)
        confidence = self._clamp(confidence, lo=0.01, hi=0.99)

        band = self._band(confidence)
        directive_hint = self._directive_hint(confidence=confidence, risk=risk, ambiguity=ambiguity)

        return ConfidenceResult(
            confidence=confidence,
            band=band,
            directive_hint=directive_hint,
            signals={
                "p_hist": p_hist,
                "health": health,
                "risk": risk,
                "ambiguity": ambiguity,
                "complexity": complexity,
                "meta_penalty": meta_penalty,
                "sample_size": n_similar,
            },
            sample_penalty=sample_penalty,
            raw_score=raw,
            calibrated_temperature=calibrated_temperature,
        )

    def update_calibration(self, *, predicted_confidence: float, success: bool) -> None:
        predicted = self._unit(predicted_confidence)
        observed = 1.0 if success else 0.0
        self._calibration.append((predicted, observed))

        if len(self._calibration) < 20:
            return

        brier = sum((pred - obs) ** 2 for pred, obs in self._calibration) / len(self._calibration)

        # Lightweight online temperature adaptation.
        if brier > 0.22:
            self._temperature = self._clamp(self._temperature + 0.05, lo=0.5, hi=2.5)
        elif brier < 0.12:
            self._temperature = self._clamp(self._temperature - 0.03, lo=0.5, hi=2.5)

    def _directive_hint(self, *, confidence: float, risk: float, ambiguity: float) -> str:
        if confidence >= 0.78:
            return "PROCEED"
        if confidence >= 0.55:
            if risk >= 0.55:
                return "PROCEED_WITH_CAUTION"
            return "PROCEED"
        if confidence >= 0.35:
            if ambiguity >= 0.55:
                return "ASK_CLARIFICATION"
            return "SWITCH_ALTERNATIVE"
        return "ASK_CONFIRMATION"

    def _band(self, confidence: float) -> str:
        if confidence >= 0.78:
            return "high"
        if confidence >= 0.55:
            return "medium"
        if confidence >= 0.35:
            return "low"
        return "critical"

    def _logit(self, p: float) -> float:
        p = self._clamp(p, lo=1e-6, hi=1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _unit(self, value: Any) -> float:
        return self._clamp(_as_float(value, fallback=0.0), lo=0.0, hi=1.0)

    def _clamp(self, value: float, *, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))


class AutonomyCoordinator:
    """Chunk C+D+F+G integration coordinator used by the bridge server."""

    def __init__(
        self,
        *,
        queue_maxsize: int = 1000,
        mode: str = "shadow",
        preflight_checker: Optional[PreflightChecker] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        meta_monitor: Optional[MetaCognitionMonitor] = None,
        intervention_engine: Optional[InterventionEngine] = None,
        health_monitor: Optional[HealthMonitor] = None,
        adaptive_engine: Optional[AdaptiveBehaviorEngine] = None,
        self_evaluator: Optional[SelfEvaluator] = None,
        auto_reflection: Optional[AutoReflectionEngine] = None,
        lesson_pipeline: Optional[LessonPipeline] = None,
        communication_policy: Optional[CommunicationPolicy] = None,
        rollout_tracker: Optional[RolloutTracker] = None,
    ) -> None:
        self._after_action_queue: asyncio.Queue[JSONDict] = asyncio.Queue(maxsize=max(1, queue_maxsize))
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._running = False

        self._preflight_checker = preflight_checker or PreflightChecker()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._meta_monitor = meta_monitor or MetaCognitionMonitor()
        self._intervention_engine = intervention_engine or InterventionEngine()
        self._health_monitor = health_monitor or HealthMonitor()
        self._adaptive_engine = adaptive_engine or AdaptiveBehaviorEngine()
        self._self_evaluator = self_evaluator or SelfEvaluator()
        self._auto_reflection = auto_reflection or AutoReflectionEngine()
        self._lesson_pipeline = lesson_pipeline or LessonPipeline()
        self._communication_policy = communication_policy or CommunicationPolicy()
        self._rollout = rollout_tracker or RolloutTracker()

        self._mode = self._normalize_mode(mode)
        self._recent_decisions: Deque[JSONDict] = deque(maxlen=500)
        self._pending_reflection_notifications: Dict[str, JSONDict] = {}

        self.processed_after_actions = 0
        self.dropped_after_actions = 0
        self.calibration_updates = 0
        self.last_error: Optional[str] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_after_actions())

    async def stop(self) -> None:
        self._running = False
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        self._self_evaluator.stop()

    def current_mode(self) -> str:
        return self._mode

    def mode_status(self) -> JSONDict:
        health_summary = self._rollout.recent_health()
        directives = [item.get("directive") for item in list(self._recent_decisions)[-10:]]
        return {
            "mode": self._mode,
            "recentDirectives": directives,
            "health": health_summary,
            "rollout": self._rollout.metrics_snapshot(),
        }

    def recent_decisions(self, *, limit: int = 10) -> List[JSONDict]:
        take = max(1, min(100, int(limit)))
        return list(self._recent_decisions)[-take:]

    def calibration_summary(self) -> JSONDict:
        return self._rollout.calibration_stats()

    def switch_mode(self, target_mode: str, *, force: bool = False) -> JSONDict:
        target = self._normalize_mode(target_mode)
        evaluation = self._rollout.evaluate_transition(current_mode=self._mode, target_mode=target)

        if force or evaluation.allowed:
            previous = self._mode
            self._mode = target
            return {
                "ok": True,
                "forced": bool(force),
                "from": previous,
                "to": target,
                "gate": evaluation.as_dict(),
            }

        return {
            "ok": False,
            "forced": False,
            "from": self._mode,
            "to": target,
            "gate": evaluation.as_dict(),
        }

    def record_fail_open(self, *, reason: str) -> None:
        self._rollout.record_fail_open(proceeded=True)
        self.last_error = reason

    async def before_action(self, context: JSONDict) -> AutonomyDecision:
        started = time.perf_counter()
        context = context if isinstance(context, dict) else {}
        tool_name = str(context.get("toolName") or context.get("tool") or "")
        params = context.get("params") if isinstance(context.get("params"), dict) else {}
        mode = self._resolve_mode(context)

        self._health_monitor.observe_before_action(context)
        health_snapshot = self._health_monitor.current_snapshot(context)

        preflight = self._preflight_checker.check_before_action(tool_name, params, context=context)
        meta_assessment = self._meta_monitor.assess_before_action(context)

        p_hist, sample_size = self._extract_predictive(context)
        health = self._extract_health(context, default=health_snapshot.overall)
        ambiguity = self._extract_ambiguity(context, params)
        complexity = self._extract_complexity(context, tool_name, params)
        explicit_meta_penalty = self._extract_meta_penalty(context)
        meta_penalty = max(explicit_meta_penalty, meta_assessment.penalty)
        temperature = self._extract_temperature(context)

        score = self._confidence_scorer.score_before_action(
            p_hist=p_hist,
            health=health,
            risk=preflight.risk,
            ambiguity=ambiguity,
            complexity=complexity,
            meta_penalty=meta_penalty,
            sample_size=sample_size,
            temperature=temperature,
        )

        base_directive, base_reason, extra = self._route_directive(
            tool_name=tool_name,
            score=score,
            preflight=preflight,
            ambiguity=ambiguity,
            complexity=complexity,
            meta_assessment=meta_assessment,
        )

        session_key = _session_key(context)
        adaptive_state = self._adaptive_engine.update_from_health(session_key, health_snapshot.as_dict())

        intervention = self._intervention_engine.evaluate_before_action(
            context=context,
            mode=mode,
            base_directive=base_directive,
            confidence=score.confidence,
            risk=preflight.risk,
            meta_state=meta_assessment.state,
            health=health_snapshot.as_dict(),
        )

        if intervention is not None:
            adaptive_state = self._adaptive_engine.apply_intervention(session_key, intervention)

        directive, reason, intervention_meta = self._apply_intervention_override(
            base_directive=base_directive,
            base_reason=base_reason,
            intervention=intervention,
            mode=mode,
        )

        alternatives: List[str] = []
        alt = extra.get("alternativeTool") if isinstance(extra.get("alternativeTool"), str) else None
        if alt:
            alternatives.append(alt)
        intervention_alt = (
            intervention_meta.get("alternativeTool") if isinstance(intervention_meta.get("alternativeTool"), str) else None
        )
        if intervention_alt and intervention_alt not in alternatives:
            alternatives.append(intervention_alt)

        communication = self._communication_policy.evaluate(
            session_key=session_key,
            directive=directive,
            confidence=score.confidence,
            risk=preflight.risk,
            risk_band=preflight.risk_band,
            reason=reason,
            alternatives=alternatives,
            tool_name=tool_name or None,
            switched_to=alternatives[0] if directive == "SWITCH_ALTERNATIVE" and alternatives else None,
            context=context,
        )

        overhead_ms = (time.perf_counter() - started) * 1000.0
        self._rollout.record_preflight_overhead(overhead_ms)

        over_eager_fp = meta_assessment.state == "over_eager" and (
            bool(context.get("explicitUserApproval"))
            or bool(context.get("longWorkflowRequested"))
            or bool(context.get("batchOperation"))
            or bool(context.get("overEagerFalsePositive"))
        )
        self._rollout.record_decision(
            mode=mode,
            override=bool(intervention_meta.get("interventionOverride", False)),
            proactive_prompt=communication.proactive and communication.should_message,
            noncritical_prompt=communication.noncritical_prompt and communication.should_message,
            over_eager_alert=meta_assessment.state == "over_eager",
            over_eager_false_positive=over_eager_fp,
        )

        metadata: JSONDict = {
            "mode": mode,
            "modeSource": "coordinator",
            "requestedMode": context.get("mode") if isinstance(context.get("mode"), str) else None,
            "risk": preflight.risk,
            "riskBand": preflight.risk_band,
            "preflight": preflight.as_dict(),
            "signals": score.signals,
            "confidenceBand": score.band,
            "directiveHint": score.directive_hint,
            "samplePenalty": score.sample_penalty,
            "rawScore": score.raw_score,
            "temperature": score.calibrated_temperature,
            "meta": meta_assessment.as_dict(),
            "metaState": meta_assessment.state,
            "metaPenalty": meta_penalty,
            "health": health_snapshot.as_dict(),
            "adaptive": adaptive_state.as_dict(),
            "communication": communication.as_dict(),
            "overheadMs": overhead_ms,
            **extra,
            **intervention_meta,
        }

        decision = AutonomyDecision(
            directive=directive,
            confidence=score.confidence,
            reason=reason,
            metadata=metadata,
        )

        self._recent_decisions.append(
            {
                "timestamp": time.time(),
                "mode": mode,
                "directive": directive,
                "confidence": score.confidence,
                "risk": preflight.risk,
                "riskBand": preflight.risk_band,
                "reason": reason,
                "overheadMs": overhead_ms,
                "communication": communication.as_dict(),
                "interventionOverride": bool(intervention_meta.get("interventionOverride", False)),
                "metaState": meta_assessment.state,
                "healthState": health_snapshot.state,
            }
        )

        return decision

    async def after_action(self, context: JSONDict) -> JSONDict:
        merged = self._merge_after_action_context(context if isinstance(context, dict) else {})
        micro_eval = self._self_evaluator.micro_eval(merged)
        merged["self_eval"] = micro_eval.as_dict()

        queued = await self.enqueue_after_action(merged)

        # Outcome feed directly into lesson extraction pipeline.
        outcome_candidates = self._build_outcome_lesson_candidates(merged, micro_eval=micro_eval.as_dict())
        ingested = self._lesson_pipeline.ingest_candidates(
            outcome_candidates,
            session_key=_session_key(merged),
            source="after_action",
        )

        return {
            "queued": queued,
            "micro_eval": micro_eval.as_dict(),
            "deep_eval_queued": micro_eval.deep_eval_queued,
            "lesson_candidates": ingested,
        }

    async def enqueue_after_action(self, context: JSONDict) -> bool:
        try:
            self._after_action_queue.put_nowait(context)
            return True
        except asyncio.QueueFull:
            self.dropped_after_actions += 1
            return False

    async def on_user_message(self, context: JSONDict) -> MetaAssessment:
        context = context if isinstance(context, dict) else {}
        return self._meta_monitor.on_user_message(context)

    async def on_assistant_turn(self, context: JSONDict) -> MetaAssessment:
        context = context if isinstance(context, dict) else {}
        meta = self._meta_monitor.assess_before_action(context)

        session_key = _session_key(context)
        deep_evals = self._self_evaluator.consume_deep_evaluations(session_key=session_key, limit=20)
        reflection = self._auto_reflection.evaluate(context, deep_evaluations=deep_evals)

        if reflection is not None:
            ingested = self._lesson_pipeline.ingest_candidates(
                reflection.lesson_candidates,
                session_key=session_key,
                source="auto_reflection",
            )
            improvement_text = ""
            if ingested:
                candidate = ingested[0]
                if isinstance(candidate, dict):
                    improvement_text = str(candidate.get("text") or "")
            completion = self._communication_policy.completion_message(improvement_text)

            payload: JSONDict = {
                "reflection": reflection.as_dict(),
                "lesson_candidates": ingested,
            }
            if completion:
                payload["completion_message"] = completion

            self._pending_reflection_notifications[session_key] = payload

        return meta

    async def on_health_update(self, context: JSONDict) -> MetaAssessment:
        context = context if isinstance(context, dict) else {}
        meta = self._meta_monitor.on_health_update(context)
        health_snapshot = self._health_monitor.on_health_update(context)
        session_key = _session_key(context)
        self._adaptive_engine.update_from_health(session_key, health_snapshot.as_dict())
        return meta

    def current_meta_state(self, context: JSONDict) -> MetaAssessment:
        context = context if isinstance(context, dict) else {}
        return self._meta_monitor.get_state(context)

    def current_health_state(self, context: JSONDict) -> JSONDict:
        context = context if isinstance(context, dict) else {}
        return self._health_monitor.current_snapshot(context).as_dict()

    def current_adaptive_state(self, context: JSONDict) -> JSONDict:
        context = context if isinstance(context, dict) else {}
        return self._adaptive_engine.current(_session_key(context)).as_dict()

    def current_intervention_state(self, context: JSONDict) -> JSONDict:
        context = context if isinstance(context, dict) else {}
        interventions = [item.as_dict() for item in self._intervention_engine.active_for_session(context)]
        interventions.sort(key=lambda item: int(item.get("priorityValue", 0)), reverse=True)
        return {
            "active": interventions,
            "count": len(interventions),
        }

    def consume_reflection_notification(self, context: JSONDict) -> Optional[JSONDict]:
        context = context if isinstance(context, dict) else {}
        session_key = _session_key(context)
        return self._pending_reflection_notifications.pop(session_key, None)

    def stats(self) -> JSONDict:
        return {
            "mode": self._mode,
            "processed_after_actions": self.processed_after_actions,
            "dropped_after_actions": self.dropped_after_actions,
            "calibration_updates": self.calibration_updates,
            "queue_depth": self._after_action_queue.qsize(),
            "last_error": self.last_error,
            "confidence_temperature": self._confidence_scorer.temperature,
            "recent_decisions": self.recent_decisions(limit=10),
            "mode_status": self.mode_status(),
            "calibration": self.calibration_summary(),
            "rollout": self._rollout.metrics_snapshot(),
            "meta": self._meta_monitor.stats(),
            "health": self._health_monitor.stats(),
            "interventions": self._intervention_engine.stats(),
            "adaptive": self._adaptive_engine.stats(),
            "self_evaluation": self._self_evaluator.stats(),
            "auto_reflection": self._auto_reflection.stats(),
            "lessons": self._lesson_pipeline.stats(),
        }

    async def _consume_after_actions(self) -> None:
        while self._running:
            try:
                event = await self._after_action_queue.get()
                await self._process_after_action(event)
                self._after_action_queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)

    async def _process_after_action(self, context: JSONDict) -> None:
        context = context if isinstance(context, dict) else {}

        merged_for_meta = self._merge_after_action_context(context)
        self_eval = merged_for_meta.get("self_eval") if isinstance(merged_for_meta.get("self_eval"), dict) else {}

        predicted = _as_float(
            merged_for_meta.get("confidence", self_eval.get("predicted_confidence")),
            fallback=None,
        )
        success = bool(merged_for_meta.get("ok"))

        if predicted is not None:
            self._confidence_scorer.update_calibration(predicted_confidence=predicted, success=success)
            mode = self._normalize_mode(str(merged_for_meta.get("mode") or self._mode))
            self._rollout.record_calibration(mode=mode, predicted_confidence=predicted, success=success)
            self.calibration_updates += 1

        self._meta_monitor.update_after_action(merged_for_meta)

        health_snapshot = self._health_monitor.observe_after_action(merged_for_meta)
        self._intervention_engine.observe_after_action(merged_for_meta)

        session_key = _session_key(merged_for_meta)
        self._adaptive_engine.update_from_health(session_key, health_snapshot.as_dict())

        await asyncio.sleep(0)
        self.processed_after_actions += 1

    def _merge_after_action_context(self, context: JSONDict) -> JSONDict:
        nested = context.get("context") if isinstance(context.get("context"), dict) else {}
        merged: JSONDict = {**nested, **context}

        if "session" not in merged and isinstance(nested.get("session"), dict):
            merged["session"] = nested.get("session")

        if "ambiguity" not in merged and nested.get("ambiguity") is not None:
            merged["ambiguity"] = nested.get("ambiguity")
        if "complexity" not in merged and nested.get("complexity") is not None:
            merged["complexity"] = nested.get("complexity")

        return merged

    def _build_outcome_lesson_candidates(self, context: JSONDict, *, micro_eval: JSONDict) -> List[JSONDict]:
        tool_name = str(context.get("toolName") or context.get("tool") or "unknown").lower()
        eval_score = _as_float(micro_eval.get("eval_score"), fallback=0.0) or 0.0
        success = _as_float(micro_eval.get("success"), fallback=1.0) or 0.0

        if success >= 1.0 and eval_score >= 0.75:
            return []

        confidence = _as_float(context.get("confidence", micro_eval.get("predicted_confidence")), fallback=None)
        if confidence is None:
            confidence = 0.62

        text = (
            f"After {tool_name}, outcome score={eval_score:.2f}. "
            f"Add a pre-check and fallback path before repeating the same action."
        )

        return [
            {
                "source": "outcome_feed",
                "text": text,
                "confidence": max(0.55, min(0.9, 0.9 - eval_score * 0.35)),
                "evidence": {
                    "tool_name": tool_name,
                    "eval_score": eval_score,
                    "success": success,
                    "predicted_confidence": confidence,
                },
            }
        ]

    def _route_directive(
        self,
        *,
        tool_name: str,
        score: ConfidenceResult,
        preflight: PreflightAssessment,
        ambiguity: float,
        complexity: float,
        meta_assessment: MetaAssessment,
    ) -> Tuple[str, str, JSONDict]:
        confidence = score.confidence
        risk = preflight.risk

        if preflight.critical:
            return (
                "BLOCK",
                "Critical preflight risk: protected file + destructive pattern",
                {"criticalRisk": True},
            )

        if preflight.has_destructive_pattern and risk >= 0.95:
            return (
                "BLOCK",
                "Critical destructive command pattern detected",
                {"criticalRisk": True},
            )

        if meta_assessment.state == "looping":
            return (
                "ASK_CLARIFICATION",
                "Detected repeated failed loop; clarification required before retrying",
                {"metaIntervention": "looping"},
            )

        if meta_assessment.state == "over_eager" and confidence < 0.90:
            return (
                "ASK_CONFIRMATION",
                "Action plan appears over-eager relative to expected workflow; confirm continuation",
                {"metaIntervention": "over_eager"},
            )

        if preflight.has_protected_file and (risk >= 0.65 or confidence < 0.78):
            return (
                "ASK_CONFIRMATION",
                "Protected file operation requires explicit confirmation",
                {"protectedFileGuard": True},
            )

        if risk >= 0.70 and confidence < 0.55:
            return (
                "ASK_CONFIRMATION",
                "High-risk + low-confidence action requires explicit confirmation",
                {"highRiskLowConfidenceGuard": True},
            )

        if confidence >= 0.78:
            directive: Tuple[str, str, JSONDict] = ("PROCEED", "High confidence", {})
        elif 0.55 <= confidence < 0.78:
            if risk >= 0.55:
                directive = (
                    "PROCEED_WITH_CAUTION",
                    "Medium confidence with elevated risk",
                    {"caution": True},
                )
            else:
                directive = ("PROCEED", "Medium confidence and manageable risk", {})
        elif 0.35 <= confidence < 0.55:
            alternative = self._suggest_alternative(tool_name)
            if alternative and (risk >= 0.35 or complexity >= 0.45 or ambiguity < 0.55):
                directive = (
                    "SWITCH_ALTERNATIVE",
                    f"Low confidence: safer alternative '{alternative}' available",
                    {"alternativeTool": alternative},
                )
            else:
                directive = (
                    "ASK_CLARIFICATION",
                    "Low confidence: need clarification before continuing",
                    {},
                )
        else:
            directive = (
                "ASK_CONFIRMATION",
                "Critical confidence band: explicit confirmation required",
                {},
            )

        return self._apply_meta_overrides(directive, meta_assessment=meta_assessment)

    def _apply_meta_overrides(
        self,
        base: Tuple[str, str, JSONDict],
        *,
        meta_assessment: MetaAssessment,
    ) -> Tuple[str, str, JSONDict]:
        directive, reason, metadata = base

        if meta_assessment.state == "confused" and directive in {"PROCEED", "PROCEED_WITH_CAUTION"}:
            return (
                "ASK_CLARIFICATION",
                "Execution pattern indicates confusion; ask for clarification",
                {**metadata, "metaIntervention": "confused"},
            )

        if meta_assessment.state == "uncertain" and directive == "PROCEED":
            return (
                "PROCEED_WITH_CAUTION",
                "Meta-cognition uncertainty detected; proceed cautiously",
                {**metadata, "metaIntervention": "uncertain"},
            )

        return base

    def _apply_intervention_override(
        self,
        *,
        base_directive: str,
        base_reason: str,
        intervention: Optional[InterventionRecord],
        mode: str,
    ) -> Tuple[str, str, JSONDict]:
        if intervention is None:
            return base_directive, base_reason, {}

        intervention_dict = intervention.as_dict()
        auto_apply = mode == "active"
        suggested = mode in {"suggest", "shadow"}

        # CRITICAL interventions always force a safe directive.
        shared_meta: JSONDict = {
            "intervention": intervention_dict,
            "interventionMode": mode,
            "alternativeTool": intervention.metadata.get("alternativeTool")
            if isinstance(intervention.metadata.get("alternativeTool"), str)
            else intervention.metadata.get("fallbackTool")
            if isinstance(intervention.metadata.get("fallbackTool"), str)
            else None,
        }

        if intervention.priority == InterventionPriority.CRITICAL:
            forced = intervention.directive
            if forced not in {"BLOCK", "ASK_CONFIRMATION"}:
                forced = "ASK_CONFIRMATION"
            return (
                forced,
                intervention.reason,
                {
                    **shared_meta,
                    "interventionOverride": True,
                    "interventionAutoApplied": auto_apply,
                    "interventionSuggested": suggested,
                },
            )

        if auto_apply:
            return (
                intervention.directive,
                intervention.reason,
                {
                    **shared_meta,
                    "interventionOverride": intervention.directive != base_directive,
                    "interventionAutoApplied": True,
                    "interventionSuggested": False,
                },
            )

        return (
            base_directive,
            base_reason,
            {
                **shared_meta,
                "interventionOverride": False,
                "interventionAutoApplied": False,
                "interventionSuggested": True,
            },
        )

    def _resolve_mode(self, context: JSONDict) -> str:
        context = context if isinstance(context, dict) else {}
        requested = context.get("mode")
        allow_override = bool(context.get("allowModeOverride") or context.get("allow_mode_override"))

        if allow_override and isinstance(requested, str) and requested.strip():
            return self._normalize_mode(requested)

        # Coordinator mode is authoritative for runtime behavior.
        return self._mode

    def _normalize_mode(self, mode: str) -> str:
        normalized = (mode or "").strip().lower()
        if normalized in {"shadow", "suggest", "active"}:
            return normalized
        return "shadow"

    def _extract_predictive(self, context: JSONDict) -> Tuple[float, int]:
        predictive = context.get("predictive") if isinstance(context.get("predictive"), dict) else {}

        p_hist = _as_float(
            predictive.get("p_hist", predictive.get("p_success", predictive.get("predicted_success", context.get("p_hist", 0.5)))),
            fallback=0.5,
        )
        p_hist = max(0.0, min(1.0, p_hist))

        sample_size = int(
            max(
                0.0,
                _as_float(
                    predictive.get("sample_size", predictive.get("n_similar", predictive.get("count", context.get("n_similar", 0)))),
                    fallback=0.0,
                ),
            )
        )
        return p_hist, sample_size

    def _extract_health(self, context: JSONDict, *, default: float = 0.75) -> float:
        health_block = context.get("health") if isinstance(context.get("health"), dict) else {}
        raw = _as_float(
            health_block.get(
                "normalized",
                health_block.get("score", health_block.get("overall", health_block.get("overall_score", context.get("health", default)))),
            ),
            fallback=default,
        )

        # Accept either 0..1 or 0..100 inputs.
        if raw > 1.0:
            raw = raw / 100.0
        return max(0.0, min(1.0, raw))

    def _extract_ambiguity(self, context: JSONDict, params: JSONDict) -> float:
        explicit = _as_float(
            context.get("ambiguity", context.get("ambiguityScore", context.get("requestAmbiguity", None))),
            fallback=None,
        )
        if explicit is not None:
            return max(0.0, min(1.0, explicit))

        # Heuristic fallback when explicit ambiguity state is unavailable.
        if not params:
            return 0.55

        if isinstance(params.get("query"), str) and "?" in params.get("query", ""):
            return 0.45

        serialized = str(params)
        if len(serialized) < 20:
            return 0.50
        return 0.25

    def _extract_complexity(self, context: JSONDict, tool_name: str, params: JSONDict) -> float:
        explicit = _as_float(context.get("complexity", context.get("complexityScore", None)), fallback=None)
        if explicit is not None:
            return max(0.0, min(1.0, explicit))

        normalized_tool = tool_name.lower()
        base = {
            "read": 0.15,
            "web_fetch": 0.22,
            "web_search": 0.24,
            "write": 0.45,
            "edit": 0.48,
            "exec": 0.65,
            "bash": 0.65,
            "browser": 0.55,
        }.get(normalized_tool, 0.35)

        payload_size = len(str(params))
        size_factor = min(0.25, payload_size / 8000.0)
        return max(0.0, min(1.0, base + size_factor))

    def _extract_meta_penalty(self, context: JSONDict) -> float:
        meta_block = context.get("meta") if isinstance(context.get("meta"), dict) else {}
        penalty = _as_float(
            context.get("meta_penalty", meta_block.get("penalty", meta_block.get("meta_penalty", 0.0))),
            fallback=0.0,
        )
        return max(0.0, min(1.0, penalty))

    def _extract_temperature(self, context: JSONDict) -> Optional[float]:
        calibration = context.get("calibration") if isinstance(context.get("calibration"), dict) else {}
        temp = _as_float(
            context.get("temperature", calibration.get("temperature", calibration.get("temp", None))),
            fallback=None,
        )
        if temp is None:
            return None
        return max(0.5, min(2.5, temp))

    def _suggest_alternative(self, tool_name: str) -> Optional[str]:
        mapping = {
            "exec": "read",
            "bash": "read",
            "write": "edit",
        }
        return mapping.get(tool_name.lower())


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
