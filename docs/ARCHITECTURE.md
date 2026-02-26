# Autonomy Architecture (Chunk G)

## Runtime Flow

1. **Bridge (`bridge.py`)** receives UDS HTTP hooks.
2. **Coordinator (`integration.py`)** performs:
   - preflight risk scoring
   - confidence scoring + calibration
   - meta-cognition assessment
   - intervention override
   - communication policy evaluation
   - rollout metric tracking
3. **Response** returns directive + confidence + metadata.
4. **After-action pipeline** updates calibration, health, interventions, self-eval, reflection lessons.

## New Chunk G Components

### Communication Policy (`communication.py`)
- Converts risk/confidence state into concise user-facing messaging.
- Tracks per-session turn state to enforce anti-annoyance cooldown.
- Collapses repeated non-critical warnings.
- Emits completion message pattern for reflection outcomes.

### Rollout Tracker (`rollout.py`)
- Tracks performance + safety metrics:
  - preflight overhead distribution
  - fail-open proceed rate
  - over-eager FP rate
  - prompt/override rates per mode
  - per-mode Brier calibration
- Evaluates gated transitions:
  - shadow→suggest
  - suggest→active
- Downgrades are always allowed.

### CLI (`analysis/cli.py`)
- Talks directly to bridge UDS endpoints:
  - `/v1/status`
  - `/v1/recent`
  - `/v1/calibration`
  - `/v1/mode`

## Safety Invariants

- Critical risk requires confirmation messaging.
- High-risk + low-confidence paths force `ASK_CONFIRMATION`.
- Fail-open always returns `PROCEED` (tracked for gate health).
- Mode escalation is metric-gated, not schedule-based.
