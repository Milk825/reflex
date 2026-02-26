# OpenClaw Autonomy Plugin (Chunk A)

TypeScript plugin scaffold that bridges OpenClaw hooks to a local Python service over UDS.

## Build

```bash
npm install
npm run build
```

## Dev (watch / hot compile)

```bash
npm run dev
```

## Tests

```bash
npm test
```

## Install into OpenClaw (link mode)

```bash
openclaw plugins install -l /absolute/path/to/openclaw-plugin-autonomy
```

## Config snippet

```json
{
  "plugins": {
    "entries": {
      "autonomy": {
        "enabled": true,
        "config": {
          "mode": "shadow",
          "bridge": {
            "socketPath": "~/.openclaw/state/autonomy/autonomy.sock",
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
}
```
