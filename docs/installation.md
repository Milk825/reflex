# Installation Guide

## Prerequisites

- OpenClaw 2026.2.23 or later
- Node.js 18+ and npm
- Python 3.10+
- Unix-like environment (Linux/macOS) with Unix Domain Socket support

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Milk825/reflex.git
cd reflex
```

### 2. Install the Python Components

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### 3. Install the OpenClaw Plugin

```bash
cd plugin-typescript
npm install
npm run build

# Install into OpenClaw (dev profile recommended for testing)
openclaw --dev plugins install .
```

### 4. Configure OpenClaw

Add to your OpenClaw config (`~/.openclaw/openclaw.json`):

```json
{
  "agents": {
    "defaults": {
      "reflex": {
        "enabled": true,
        "mode": "shadow",
        "bridge": {
          "socketPath": "/tmp/openclaw-reflex.sock",
          "requestTimeoutMs": 80,
          "preflightTimeoutMs": 6,
          "asyncTimeoutMs": 20
        },
        "circuitBreaker": {
          "failureThreshold": 3,
          "cooldownMs": 5000
        },
        "failOpen": true
      }
    }
  }
}
```

### 5. Start the Bridge

```bash
# Start the Python bridge service
python -m analysis.autonomy.bridge

# Or run in background
python -m analysis.autonomy.bridge --daemon
```

### 6. Test in Shadow Mode

```bash
# Start OpenClaw with Reflex in shadow mode
openclaw --dev chat

# Use normally - Reflex watches and logs but doesn't intervene

# Check what Reflex observed
openclaw telemetry reflex recent
```

## Gradual Rollout

### Phase 1: Shadow Mode (Recommended: 1-2 weeks)

```bash
openclaw config set agents.defaults.reflex.mode shadow
```

- Reflex observes and logs all decisions
- No behavioral changes
- Review logs to understand patterns

### Phase 2: Suggest Mode

```bash
openclaw config set agents.defaults.reflex.mode suggest
```

- Reflex recommends interventions
- You decide whether to accept
- Tune thresholds based on feedback

### Phase 3: Active Mode

```bash
openclaw config set agents.defaults.reflex.mode active
```

- Reflex intervenes automatically
- Blocks dangerous actions
- Switches approaches when stuck

## Troubleshooting

### Bridge Won't Start

Check if socket path is valid:
```bash
ls -la /tmp/openclaw-reflex.sock
```

If it exists and is stale:
```bash
rm /tmp/openclaw-reflex.sock
python -m analysis.autonomy.bridge
```

### Plugin Not Loading

```bash
# Check plugin status
openclaw --dev plugins list | grep reflex

# Reinstall if needed
openclaw --dev plugins remove reflex
openclaw --dev plugins install ./plugin-typescript
```

### No Telemetry Showing

Ensure reflex is enabled:
```bash
openclaw config get agents.defaults.reflex.enabled
```

Check mode:
```bash
openclaw config get agents.defaults.reflex.mode
```

## Uninstallation

```bash
# Remove plugin
openclaw plugins remove reflex

# Remove Python package
pip uninstall reflex

# Clean up config
openclaw config unset agents.defaults.reflex
```

## Next Steps

- Read the [Architecture Guide](./architecture.md)
- Review [Configuration Options](./configuration.md)
- Check [Examples](./examples/)

## Getting Help

- GitHub Issues: https://github.com/Milk825/reflex/issues
- Discussions: https://github.com/Milk825/reflex/discussions