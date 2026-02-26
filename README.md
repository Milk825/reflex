# Autonomy Bridge Server (Chunks A–G)

Run the Python bridge:

```bash
python3 analysis/autonomy/bridge.py \
  --socket-path ~/.openclaw/state/autonomy/autonomy.sock
```

## Endpoints

Core:
- `GET /health`
- `POST /before_action`
- `POST /after_action`
- `POST /on_user_message`
- `POST /on_assistant_turn`
- `POST /on_health_update`

Compatibility aliases:
- `/v1/status`, `/v1/preflight`, `/v1/postflight`, `/v1/message-event`, `/v1/run-end`, `/v1/health`

Chunk G control/telemetry:
- `GET /v1/recent?limit=10` — recent autonomy decisions
- `GET /v1/calibration` — confidence calibration stats
- `POST /v1/mode` with `{ "mode": "shadow|suggest|active", "force": false }`

## Communication Policy (Chunk G)

`analysis/autonomy/communication.py` applies concise anti-annoyance rules:
- High confidence: silent
- Medium confidence + low risk: silent
- Medium confidence + medium/high risk: one-line caution
- Low confidence: ask with 1–2 alternatives
- Critical risk: confirmation required

Guardrails:
- Non-critical proactive prompts are throttled to **<= 1 per 3 turns**
- Repeated warnings are collapsed
- Terse default phrasing

## Rollout Gates (Metrics-gated)

Implemented in `analysis/autonomy/rollout.py`:

### Shadow → Suggest
- p95 overhead < 10ms
- fail-open proceed rate == 100%
- over-eager false-positive rate < 5%

### Suggest → Active
- proactive prompt rate <= 1/3 turns
- intervention override rate <= 20%
- no Brier score degradation vs shadow

## CLI (analysis/cli.py)

```bash
python3 -m analysis.cli telemetry autonomy status
python3 -m analysis.cli telemetry autonomy recent
python3 -m analysis.cli telemetry autonomy calibration
python3 -m analysis.cli telemetry autonomy mode suggest
```

Optional:
- `--socket-path ~/.openclaw/state/autonomy/autonomy.sock`
- `--timeout-ms 2000`
- `--force` for `mode` command

## Test + Benchmark

```bash
python3 -m unittest discover -s analysis/autonomy/tests -p 'test_*.py' -v
```

Load/perf coverage includes:
- 10k-action p95 regression (`test_performance_regression.py`)
- Communication throttling + confirmation safety
- Rollout gate evaluation
- Bridge + CLI E2E checks
