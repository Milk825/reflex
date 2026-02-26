# Autonomy Troubleshooting

## `telemetry autonomy status` fails with socket error

- Ensure bridge is running:
  ```bash
  python3 analysis/autonomy/bridge.py
  ```
- Confirm socket path matches CLI `--socket-path`.

## Mode switch returns conflict (`status: 409`)

Rollout gates blocked escalation. Check:
- `telemetry autonomy status` → `autonomy.rollout`
- `telemetry autonomy calibration`

Typical blockers:
- p95 overhead >= 10ms
- prompt rate > 1/3 in suggest
- override rate > 20% in suggest
- insufficient calibration data or Brier degradation

If intentionally overriding for controlled testing:
```bash
python3 -m analysis.cli telemetry autonomy mode active --force
```

## Too many proactive prompts

The communication policy throttles non-critical prompts to <=1 every 3 turns.
If you still see more prompts:
- verify calls include stable session keys
- inspect `communication` block in `before_action` responses
- ensure repeated warning signatures are not being changed unnecessarily

## Reflection completion message missing

`completion_message` is only emitted when reflection produced at least one lesson candidate with text.

## Performance regressions

Run:
```bash
python3 -m unittest analysis.autonomy.tests.test_performance_regression -v
```

If p95 exceeds target:
- look for heavy sync operations in `before_action`
- verify no blocking file/network calls were added to hot path
