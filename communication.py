from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

JSONDict = Dict[str, Any]


@dataclass
class CommunicationDecision:
    kind: str
    should_message: bool
    message: Optional[str]
    requires_confirmation: bool = False
    critical: bool = False
    proactive: bool = False
    noncritical_prompt: bool = False
    collapsed_repeat: bool = False
    suppressed_by_cooldown: bool = False
    turn_index: int = 0

    def as_dict(self) -> JSONDict:
        return {
            "kind": self.kind,
            "shouldMessage": self.should_message,
            "message": self.message,
            "requiresConfirmation": self.requires_confirmation,
            "critical": self.critical,
            "proactive": self.proactive,
            "noncriticalPrompt": self.noncritical_prompt,
            "collapsedRepeat": self.collapsed_repeat,
            "suppressedByCooldown": self.suppressed_by_cooldown,
            "turnIndex": self.turn_index,
        }


@dataclass
class _SessionState:
    turns: int = 0
    last_noncritical_prompt_turn: int = -10_000
    last_warning_signature: Optional[str] = None
    last_warning_turn: int = -10_000
    updated_at: float = 0.0


class CommunicationPolicy:
    """Concise, throttled user-facing autonomy messaging policy.

    Rules encoded (Chunk G):
      - high confidence: silent
      - medium confidence + low risk: silent
      - medium confidence + medium/high risk: one-line caution
      - low confidence: ask or propose alternatives
      - critical risk: require confirmation

    Anti-annoyance:
      - max 1 non-critical proactive prompt every N turns (default: 3)
      - collapse repeated warnings
      - terse default phrasing
    """

    def __init__(self, *, cooldown_turns: int = 3, repeat_collapse_window_turns: int = 3) -> None:
        self._cooldown_turns = max(1, int(cooldown_turns))
        self._repeat_collapse_window_turns = max(1, int(repeat_collapse_window_turns))
        self._sessions: Dict[str, _SessionState] = {}

    def evaluate(
        self,
        *,
        session_key: str,
        directive: str,
        confidence: float,
        risk: float,
        risk_band: str,
        reason: str,
        alternatives: Optional[List[str]] = None,
        tool_name: Optional[str] = None,
        switched_to: Optional[str] = None,
        context: Optional[JSONDict] = None,
    ) -> CommunicationDecision:
        state = self._sessions.setdefault(session_key or "default", _SessionState())
        context = context if isinstance(context, dict) else {}

        explicit_turn = _extract_turn_index(context)
        if explicit_turn is None:
            state.turns += 1
        else:
            state.turns = max(state.turns + 1, explicit_turn)
        turn = state.turns
        state.updated_at = time.time()

        conf = _clamp(confidence, lo=0.0, hi=1.0)
        risk = _clamp(risk, lo=0.0, hi=1.0)
        normalized_risk_band = (risk_band or "").strip().lower()
        critical = normalized_risk_band == "critical" or risk >= 0.90 or directive in {"BLOCK", "ASK_CONFIRMATION"} and risk >= 0.70

        if critical:
            message = self._critical_confirmation_message(confidence=conf, risk=risk)
            return CommunicationDecision(
                kind="critical_confirmation",
                should_message=True,
                message=message,
                requires_confirmation=True,
                critical=True,
                proactive=False,
                noncritical_prompt=False,
                turn_index=turn,
            )

        if conf >= 0.78:
            return CommunicationDecision(kind="silent_high_confidence", should_message=False, message=None, turn_index=turn)

        if 0.55 <= conf < 0.78:
            if risk < 0.45:
                return CommunicationDecision(kind="silent_medium_confidence_low_risk", should_message=False, message=None, turn_index=turn)

            caution = f"Caution: medium confidence ({int(round(conf * 100))}%) with elevated risk ({int(round(risk * 100))}%)."
            return self._throttled_warning(
                state=state,
                turn=turn,
                kind="medium_confidence_caution",
                message=caution,
                signature=f"caution::{int(risk * 100)}::{_normalize_reason(reason)}",
            )

        # Low confidence branch.
        alts = _clean_alternatives(alternatives)
        if directive == "SWITCH_ALTERNATIVE" and switched_to:
            low_message = (
                f"I switched from {tool_name or 'the current approach'} to {switched_to} because similar runs often fail."
            )
        else:
            low_message = self._low_confidence_message(confidence=conf, reason=reason, alternatives=alts)

        return self._throttled_warning(
            state=state,
            turn=turn,
            kind="low_confidence",
            message=low_message,
            signature=f"low::{directive}::{_normalize_reason(reason)}::{','.join(alts[:2])}",
        )

    def completion_message(self, improvement: str) -> Optional[str]:
        text = (improvement or "").strip()
        if not text:
            return None
        concise = re.sub(r"\s+", " ", text)
        if len(concise) > 140:
            concise = concise[:137].rstrip() + "..."
        return f"Completed. One improvement logged for next time: {concise}"

    def _critical_confirmation_message(self, *, confidence: float, risk: float) -> str:
        return (
            f"High-risk action ({int(round(risk * 100))}% risk) with "
            f"{int(round(confidence * 100))}% confidence. Confirm before I proceed."
        )

    def _low_confidence_message(self, *, confidence: float, reason: str, alternatives: List[str]) -> str:
        because = _normalize_reason(reason)
        pct = int(round(confidence * 100))
        if alternatives:
            rendered = _render_alternatives(alternatives[:2])
            return f"Confidence is low ({pct}%) because {because}. I can do {rendered}—pick one?"
        return f"Confidence is low ({pct}%) because {because}. Want me to continue or change approach?"

    def _throttled_warning(
        self,
        *,
        state: _SessionState,
        turn: int,
        kind: str,
        message: str,
        signature: str,
    ) -> CommunicationDecision:
        same_as_last = state.last_warning_signature == signature and (turn - state.last_warning_turn) <= self._repeat_collapse_window_turns
        if same_as_last:
            return CommunicationDecision(
                kind=kind,
                should_message=False,
                message=None,
                proactive=True,
                noncritical_prompt=True,
                collapsed_repeat=True,
                turn_index=turn,
            )

        cooldown_remaining = self._cooldown_turns - (turn - state.last_noncritical_prompt_turn)
        if cooldown_remaining > 0:
            return CommunicationDecision(
                kind=kind,
                should_message=False,
                message=None,
                proactive=True,
                noncritical_prompt=True,
                suppressed_by_cooldown=True,
                turn_index=turn,
            )

        state.last_noncritical_prompt_turn = turn
        state.last_warning_signature = signature
        state.last_warning_turn = turn
        return CommunicationDecision(
            kind=kind,
            should_message=True,
            message=message,
            proactive=True,
            noncritical_prompt=True,
            turn_index=turn,
        )


def _extract_turn_index(context: JSONDict) -> Optional[int]:
    for key in ("turn", "turnIndex", "assistantTurnIndex", "messageIndex"):
        value = context.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed >= 0:
                return parsed
        if isinstance(value, str):
            try:
                parsed = int(value.strip())
            except Exception:
                continue
            if parsed >= 0:
                return parsed
    return None


def _clean_alternatives(alternatives: Optional[List[str]]) -> List[str]:
    if not alternatives:
        return []

    out: List[str] = []
    seen = set()
    for alt in alternatives:
        if not isinstance(alt, str):
            continue
        item = alt.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= 3:
            break
    return out


def _render_alternatives(alternatives: List[str]) -> str:
    if not alternatives:
        return ""
    if len(alternatives) == 1:
        return alternatives[0]
    return f"{alternatives[0]} or {alternatives[1]}"


def _normalize_reason(reason: str) -> str:
    text = re.sub(r"\s+", " ", (reason or "").strip())
    if not text:
        return "signals are mixed"
    if len(text) > 72:
        return text[:69].rstrip() + "..."
    return text


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
