from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

JSONDict = Dict[str, Any]
_ALLOWED_MODES = ("shadow", "suggest", "active")


@dataclass
class ModeMetrics:
    decisions: int = 0
    turns: int = 0
    override_events: int = 0
    proactive_prompts: int = 0
    noncritical_prompts: int = 0
    calibration_samples: int = 0
    brier_sum: float = 0.0
    brier_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))

    def record_decision(self, *, override: bool, proactive_prompt: bool, noncritical_prompt: bool) -> None:
        self.decisions += 1
        self.turns += 1
        if override:
            self.override_events += 1
        if proactive_prompt:
            self.proactive_prompts += 1
        if noncritical_prompt:
            self.noncritical_prompts += 1

    def record_brier(self, value: float) -> None:
        self.calibration_samples += 1
        self.brier_sum += value
        self.brier_history.append(value)

    @property
    def prompt_rate(self) -> float:
        if self.turns <= 0:
            return 0.0
        return self.noncritical_prompts / self.turns

    @property
    def override_rate(self) -> float:
        if self.decisions <= 0:
            return 0.0
        return self.override_events / self.decisions

    @property
    def brier(self) -> Optional[float]:
        if self.calibration_samples <= 0:
            return None
        return self.brier_sum / self.calibration_samples


@dataclass
class GateEvaluation:
    from_mode: str
    to_mode: str
    allowed: bool
    reasons: List[str]
    metrics: JSONDict

    def as_dict(self) -> JSONDict:
        return {
            "from": self.from_mode,
            "to": self.to_mode,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }


class RolloutTracker:
    """Tracks rollout readiness metrics and evaluates mode gate transitions."""

    def __init__(self, *, overhead_window: int = 20_000) -> None:
        self._preflight_overhead_ms: Deque[float] = deque(maxlen=max(1000, int(overhead_window)))
        self._fail_open_events = 0
        self._fail_open_proceed = 0

        self._over_eager_events = 0
        self._over_eager_false_positive = 0

        self._mode_metrics: Dict[str, ModeMetrics] = {mode: ModeMetrics() for mode in _ALLOWED_MODES}

    def record_preflight_overhead(self, overhead_ms: float) -> None:
        if math.isnan(overhead_ms) or math.isinf(overhead_ms):
            return
        self._preflight_overhead_ms.append(max(0.0, float(overhead_ms)))

    def record_fail_open(self, *, proceeded: bool = True) -> None:
        self._fail_open_events += 1
        if proceeded:
            self._fail_open_proceed += 1

    def record_decision(
        self,
        *,
        mode: str,
        override: bool,
        proactive_prompt: bool,
        noncritical_prompt: bool,
        over_eager_alert: bool,
        over_eager_false_positive: bool,
    ) -> None:
        mode_metrics = self._mode_metrics.get(_normalize_mode(mode), self._mode_metrics["shadow"])
        mode_metrics.record_decision(
            override=override,
            proactive_prompt=proactive_prompt,
            noncritical_prompt=noncritical_prompt,
        )

        if over_eager_alert:
            self._over_eager_events += 1
            if over_eager_false_positive:
                self._over_eager_false_positive += 1

    def record_calibration(self, *, mode: str, predicted_confidence: float, success: bool) -> None:
        pred = max(0.0, min(1.0, float(predicted_confidence)))
        obs = 1.0 if success else 0.0
        brier = (pred - obs) ** 2
        mode_metrics = self._mode_metrics.get(_normalize_mode(mode), self._mode_metrics["shadow"])
        mode_metrics.record_brier(brier)

    def evaluate_transition(self, *, current_mode: str, target_mode: str) -> GateEvaluation:
        current = _normalize_mode(current_mode)
        target = _normalize_mode(target_mode)

        if current == target:
            return GateEvaluation(current, target, True, ["already_in_target_mode"], self.metrics_snapshot())

        if _mode_rank(target) < _mode_rank(current):
            return GateEvaluation(current, target, True, ["downgrade_allowed"], self.metrics_snapshot())

        if current == "shadow" and target == "suggest":
            return self._evaluate_shadow_to_suggest()

        if current == "suggest" and target == "active":
            return self._evaluate_suggest_to_active()

        if current == "shadow" and target == "active":
            first = self._evaluate_shadow_to_suggest()
            second = self._evaluate_suggest_to_active()
            allowed = first.allowed and second.allowed
            reasons = first.reasons + second.reasons
            return GateEvaluation("shadow", "active", allowed, reasons, self.metrics_snapshot())

        return GateEvaluation(current, target, False, ["unsupported_transition"], self.metrics_snapshot())

    def metrics_snapshot(self) -> JSONDict:
        mode_block: JSONDict = {}
        for mode, metrics in self._mode_metrics.items():
            mode_block[mode] = {
                "decisions": metrics.decisions,
                "turns": metrics.turns,
                "override_events": metrics.override_events,
                "proactive_prompts": metrics.proactive_prompts,
                "noncritical_prompts": metrics.noncritical_prompts,
                "prompt_rate": metrics.prompt_rate,
                "override_rate": metrics.override_rate,
                "calibration_samples": metrics.calibration_samples,
                "brier": metrics.brier,
            }

        return {
            "preflight": {
                "samples": len(self._preflight_overhead_ms),
                "p95_overhead_ms": self.preflight_p95_ms,
                "mean_overhead_ms": self.preflight_mean_ms,
            },
            "fail_open": {
                "events": self._fail_open_events,
                "proceed_events": self._fail_open_proceed,
                "proceed_rate": self.fail_open_proceed_rate,
            },
            "over_eager": {
                "alerts": self._over_eager_events,
                "false_positive": self._over_eager_false_positive,
                "false_positive_rate": self.over_eager_false_positive_rate,
            },
            "modes": mode_block,
        }

    @property
    def preflight_p95_ms(self) -> float:
        return _percentile(self._preflight_overhead_ms, 95.0)

    @property
    def preflight_mean_ms(self) -> float:
        if not self._preflight_overhead_ms:
            return 0.0
        return statistics.fmean(self._preflight_overhead_ms)

    @property
    def fail_open_proceed_rate(self) -> float:
        if self._fail_open_events <= 0:
            return 1.0
        return self._fail_open_proceed / self._fail_open_events

    @property
    def over_eager_false_positive_rate(self) -> float:
        if self._over_eager_events <= 0:
            return 0.0
        return self._over_eager_false_positive / self._over_eager_events

    def calibration_stats(self) -> JSONDict:
        out: JSONDict = {}
        for mode, metrics in self._mode_metrics.items():
            brier = metrics.brier
            out[mode] = {
                "samples": metrics.calibration_samples,
                "brier": brier,
            }
        shadow_brier = out["shadow"]["brier"]
        suggest_brier = out["suggest"]["brier"]
        out["degradation_vs_shadow"] = (
            None
            if shadow_brier is None or suggest_brier is None
            else suggest_brier - shadow_brier
        )
        return out

    def recent_health(self) -> JSONDict:
        # Convenience status summary for CLI/status endpoints.
        suggest = self._mode_metrics["suggest"]
        return {
            "preflightP95Ms": self.preflight_p95_ms,
            "failOpenProceedRate": self.fail_open_proceed_rate,
            "overEagerFalsePositiveRate": self.over_eager_false_positive_rate,
            "suggestPromptRate": suggest.prompt_rate,
            "suggestOverrideRate": suggest.override_rate,
        }

    def _evaluate_shadow_to_suggest(self) -> GateEvaluation:
        reasons: List[str] = []
        metrics = self.metrics_snapshot()

        if self.preflight_p95_ms >= 10.0:
            reasons.append(f"p95_overhead_ms={self.preflight_p95_ms:.3f} must be <10")

        if self.fail_open_proceed_rate < 1.0:
            reasons.append(f"fail_open_proceed_rate={self.fail_open_proceed_rate:.3f} must be 1.0")

        if self.over_eager_false_positive_rate >= 0.05:
            reasons.append(
                f"over_eager_false_positive_rate={self.over_eager_false_positive_rate:.3f} must be <0.05"
            )

        return GateEvaluation("shadow", "suggest", len(reasons) == 0, reasons or ["pass"], metrics)

    def _evaluate_suggest_to_active(self) -> GateEvaluation:
        reasons: List[str] = []
        metrics = self.metrics_snapshot()

        suggest = self._mode_metrics["suggest"]
        if suggest.prompt_rate > (1.0 / 3.0):
            reasons.append(f"prompt_rate={suggest.prompt_rate:.3f} must be <=0.333")

        if suggest.override_rate > 0.20:
            reasons.append(f"override_rate={suggest.override_rate:.3f} must be <=0.20")

        shadow_brier = self._mode_metrics["shadow"].brier
        suggest_brier = suggest.brier
        if shadow_brier is None or suggest_brier is None:
            reasons.append("insufficient_calibration_data_for_brier_gate")
        elif suggest_brier > shadow_brier:
            reasons.append(
                f"brier_degradation={suggest_brier - shadow_brier:.4f} must be <=0.0000"
            )

        return GateEvaluation("suggest", "active", len(reasons) == 0, reasons or ["pass"], metrics)


def _percentile(values: Deque[float], p: float) -> float:
    if not values:
        return 0.0
    sample = sorted(float(v) for v in values)
    rank = max(0, min(len(sample) - 1, int(math.ceil((p / 100.0) * len(sample))) - 1))
    return sample[rank]


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized not in _ALLOWED_MODES:
        return "shadow"
    return normalized


def _mode_rank(mode: str) -> int:
    normalized = _normalize_mode(mode)
    if normalized == "shadow":
        return 0
    if normalized == "suggest":
        return 1
    return 2
