import os from "node:os";
import path from "node:path";

export type PluginMode = "shadow" | "suggest" | "active";

export interface AutonomyPluginConfig {
  mode: PluginMode;
  failOpen: boolean;
  bridge: {
    socketPath: string;
    requestTimeoutMs: number;
    preflightTimeoutMs: number;
    asyncTimeoutMs: number;
  };
  circuitBreaker: {
    failureThreshold: number;
    cooldownMs: number;
  };
}

const DEFAULT_SOCKET_PATH = "~/.openclaw/state/autonomy/autonomy.sock";

export const DEFAULT_CONFIG: AutonomyPluginConfig = {
  mode: "shadow",
  failOpen: true,
  bridge: {
    socketPath: expandHome(DEFAULT_SOCKET_PATH),
    requestTimeoutMs: 80,
    preflightTimeoutMs: 6,
    asyncTimeoutMs: 20,
  },
  circuitBreaker: {
    failureThreshold: 3,
    cooldownMs: 5_000,
  },
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asNumber(value: unknown, fallback: number): number {
  if (typeof value !== "number" || Number.isNaN(value) || !Number.isFinite(value)) {
    return fallback;
  }
  return value;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function asMode(value: unknown, fallback: PluginMode): PluginMode {
  if (value === "shadow" || value === "suggest" || value === "active") {
    return value;
  }
  return fallback;
}

export function expandHome(inputPath: string): string {
  if (inputPath.startsWith("~/")) {
    return path.join(os.homedir(), inputPath.slice(2));
  }
  if (inputPath === "~") {
    return os.homedir();
  }
  return inputPath;
}

export function resolvePluginConfig(rawConfig: unknown): AutonomyPluginConfig {
  const cfg = isRecord(rawConfig) ? rawConfig : {};
  const bridge = isRecord(cfg.bridge) ? cfg.bridge : {};
  const breaker = isRecord(cfg.circuitBreaker) ? cfg.circuitBreaker : {};

  const requestTimeoutMs = Math.min(100, Math.max(1, asNumber(bridge.requestTimeoutMs, DEFAULT_CONFIG.bridge.requestTimeoutMs)));
  const preflightTimeoutMs = Math.min(100, Math.max(1, asNumber(bridge.preflightTimeoutMs, DEFAULT_CONFIG.bridge.preflightTimeoutMs)));
  const asyncTimeoutMs = Math.min(100, Math.max(1, asNumber(bridge.asyncTimeoutMs, DEFAULT_CONFIG.bridge.asyncTimeoutMs)));

  return {
    mode: asMode(cfg.mode, DEFAULT_CONFIG.mode),
    failOpen: asBoolean(cfg.failOpen, DEFAULT_CONFIG.failOpen),
    bridge: {
      socketPath: expandHome(
        typeof bridge.socketPath === "string" && bridge.socketPath.length > 0
          ? bridge.socketPath
          : DEFAULT_SOCKET_PATH,
      ),
      requestTimeoutMs,
      preflightTimeoutMs,
      asyncTimeoutMs,
    },
    circuitBreaker: {
      failureThreshold: Math.max(1, Math.floor(asNumber(breaker.failureThreshold, DEFAULT_CONFIG.circuitBreaker.failureThreshold))),
      cooldownMs: Math.max(100, Math.floor(asNumber(breaker.cooldownMs, DEFAULT_CONFIG.circuitBreaker.cooldownMs))),
    },
  };
}
