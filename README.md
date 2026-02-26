# Reflex

**Self-correcting intelligence for AI agents.**

Reflex is a real-time autonomy layer for OpenClaw that intercepts tool calls, detects failure patterns, and automatically switches approaches—giving any language model the resilience and adaptability of premium systems, locally and at zero API cost.

---

## Why Reflex Exists

Most AI agents execute tools blindly. When they fail, they loop. When they loop, they hallucinate. When they hallucinate, you pay for wasted tokens and broken workflows.

**Reflex changes the equation.**

Instead of asking "how do we get better models?" we asked "how do we make the models we have *smarter about when they're failing?*"

The answer: intercept, evaluate, adapt—*before* execution, not after.

---

## What It Does

Reflex sits between your agent's reasoning and the outside world. Every tool call gets evaluated through six signals of confidence. When patterns indicate trouble, Reflex intervenes—blocking, suggesting alternatives, or switching approaches entirely.

| Without Reflex | With Reflex |
|----------------|-------------|
| Try → Fail → Try same → Fail → Hallucinate → Loop forever | Try → Fail → Detect pattern → Switch approach → Try different → Succeed |
| Reacts to failure after it happens | Prevents failure before it occurs |
| Manual intervention required | Self-correcting autonomy |
| $200/month Opus-level reliability | $0.10 model with Opus-level resilience |

---

## Core Capabilities

**🔍 Pattern Detection**
- Identifies looping, over-eagerness, and confidence collapse in real-time
- Learns from telemetry without manual teaching
- Fingerprint-based matching, not keyword filtering

**🛡️ Intervention Engine**
- Circuit breaker for failing tools
- Pre-flight risk assessment
- Automatic fallback to alternative approaches

**📊 Confidence Calibration**
- Six-signal scoring: history, health, risk, ambiguity, complexity, meta-penalty
- Calibrated thresholds that adapt to context
- Transparent decision logging

**⚡ Zero Overhead**
- Local Python execution—no API calls
- Sub-10ms latency per interception
- Bounded memory, graceful degradation

---

## Three Modes of Operation

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Shadow** | Observe, log, never intervene | Learning patterns, building trust |
| **Suggest** | Recommend, let user decide | Collaborative workflows |
| **Active** | Intervene when confidence is low | Autonomous operation |

Graduate from Shadow → Suggest → Active as confidence builds.

---

## Democratizing Intelligence

The best AI capabilities shouldn't require the most expensive models.

**Reflex enables:**
- Students running local models
- Indie devs on tight budgets  
- Teams scaling cost-effectively
- Anyone who wants careful AI without the premium price

**Same models. Smarter behavior. Fraction of the cost.**

---

## Quick Start

```bash
# Install the OpenClaw plugin
npm install -g openclaw-plugin-reflex

# Enable in your OpenClaw config
openclaw config set agents.defaults.memorySearch.enabled true
openclaw config set agents.defaults.reflex.mode shadow

# Start in shadow mode—watch, learn, zero risk
openclaw chat
```

Check what Reflex observed:
```bash
openclaw telemetry reflex recent
```

---

## Architecture

Reflex implements the MAPE-K loop (Monitor, Analyze, Plan, Execute, Knowledge) at the tool interception layer:

```
User Request → LLM Reasoning → Tool Decision → [REFLEX INTERCEPTION] → Execution
                                               ↓
                                         Confidence Scoring
                                         Pattern Matching
                                         Intervention Decision
                                               ↓
                                         PROCEED / BLOCK / SWITCH / ASK
```

- **TypeScript Plugin:** Hooks into OpenClaw's `before_tool_call` and `after_tool_call`
- **Python Core:** Confidence scoring, pattern detection, intervention logic
- **Bridge:** Unix Domain Socket for sub-10ms communication
- **Telemetry:** SQLite + JSONL for pattern learning

---

## Safety First

Reflex is designed to fail safely:

- **Fail-open:** If Reflex crashes, tools proceed normally—no deadlock
- **Circuit breaker:** Auto-disables after repeated faults
- **Shadow mode:** Learn patterns without risk before going active
- **Transparent:** Every decision logged, every intervention auditable

---

## Performance

- **Latency:** <10ms per tool call (p95: ~2.6ms)
- **Memory:** ~50-100MB bounded cache
- **CPU:** Local Python, negligible overhead
- **Tokens:** Zero additional API cost

---

## Documentation

- [Installation Guide](./docs/installation.md)
- [Configuration Reference](./docs/configuration.md)
- [Architecture Deep-Dive](./docs/architecture.md)
- [Contributing](./CONTRIBUTING.md)

---

## License

MIT License—use it, fork it, improve it.

---

**Built for OpenClaw. Designed for builders.**
