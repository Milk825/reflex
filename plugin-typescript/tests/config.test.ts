import { describe, expect, it } from "vitest";
import { resolvePluginConfig } from "../src/config.js";

describe("resolvePluginConfig", () => {
  it("applies defaults", () => {
    const cfg = resolvePluginConfig(undefined);
    expect(cfg.mode).toBe("shadow");
    expect(cfg.failOpen).toBe(true);
    expect(cfg.bridge.preflightTimeoutMs).toBe(6);
  });

  it("clamps timeout values to <=100ms", () => {
    const cfg = resolvePluginConfig({
      bridge: {
        requestTimeoutMs: 999,
        preflightTimeoutMs: 250,
        asyncTimeoutMs: 250,
      },
    });

    expect(cfg.bridge.requestTimeoutMs).toBe(100);
    expect(cfg.bridge.preflightTimeoutMs).toBe(100);
    expect(cfg.bridge.asyncTimeoutMs).toBe(100);
  });
});
