import { describe, expect, it } from "vitest";
import { BridgeClient } from "../src/api.js";
import type { AutonomyPluginConfig } from "../src/config.js";

const baseConfig: AutonomyPluginConfig = {
  mode: "shadow",
  failOpen: true,
  bridge: {
    socketPath: "/tmp/definitely-missing-autonomy.sock",
    requestTimeoutMs: 20,
    preflightTimeoutMs: 6,
    asyncTimeoutMs: 20,
  },
  circuitBreaker: {
    failureThreshold: 2,
    cooldownMs: 1000,
  },
};

describe("BridgeClient degraded behavior", () => {
  it("returns PROCEED when bridge is unavailable in fail-open mode", async () => {
    const client = new BridgeClient(baseConfig);
    const decision = await client.beforeAction({ toolName: "read" });

    expect(decision.directive).toBe("PROCEED");
    expect(decision.metadata.failOpen).toBe(true);
    expect(decision.metadata.failClosed).toBe(false);
  });

  it("returns BLOCK when bridge is unavailable in fail-closed mode", async () => {
    const client = new BridgeClient({ ...baseConfig, failOpen: false });
    const decision = await client.beforeAction({ toolName: "read" });

    expect(decision.directive).toBe("BLOCK");
    expect(decision.metadata.failOpen).toBe(false);
    expect(decision.metadata.failClosed).toBe(true);
  });

  it("opens circuit breaker after repeated failures", async () => {
    const client = new BridgeClient(baseConfig);

    await client.beforeAction({ toolName: "read" });
    await client.beforeAction({ toolName: "read" });

    const snap = client.getCircuitBreakerSnapshot();
    expect(snap.state).toBe("OPEN");
  });

  it("returns undefined for currentMode when bridge status is unavailable", async () => {
    const client = new BridgeClient(baseConfig);
    const mode = await client.currentMode();
    expect(mode).toBeUndefined();
  });
});
