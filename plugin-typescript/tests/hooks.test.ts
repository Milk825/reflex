import { describe, expect, it, vi } from "vitest";
import { registerAutonomyHooks } from "../src/hooks.js";
import type { AutonomyPluginConfig } from "../src/config.js";
import type { BridgeClient, BridgeDecision } from "../src/api.js";

function makeConfig(mode: AutonomyPluginConfig["mode"]): AutonomyPluginConfig {
  return {
    mode,
    failOpen: true,
    bridge: {
      socketPath: "/tmp/missing-autonomy.sock",
      requestTimeoutMs: 20,
      preflightTimeoutMs: 6,
      asyncTimeoutMs: 20,
    },
    circuitBreaker: {
      failureThreshold: 2,
      cooldownMs: 1000,
    },
  };
}

function decision(partial: Partial<BridgeDecision>): BridgeDecision {
  return {
    directive: "PROCEED",
    confidence: 0.8,
    metadata: {},
    timestamp: Date.now(),
    ...partial,
  };
}

function setup(
  mode: AutonomyPluginConfig["mode"],
  beforeDecision: BridgeDecision,
  runtimeMode: AutonomyPluginConfig["mode"] = mode,
) {
  const handlers = new Map<string, (event: any, ctx: any) => Promise<any> | any>();
  const client: Partial<BridgeClient> = {
    currentMode: vi.fn(async () => runtimeMode),
    beforeAction: vi.fn(async () => beforeDecision),
    afterAction: vi.fn(async () => decision({ directive: "PROCEED" })),
    onUserMessage: vi.fn(async () => decision({ directive: "PROCEED" })),
    onAssistantTurn: vi.fn(async () => decision({ directive: "PROCEED" })),
    onHealthUpdate: vi.fn(async () => decision({ directive: "PROCEED" })),
  };

  const logger = {
    info: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    error: vi.fn(),
  };

  const api = {
    on: (name: string, handler: (event: any, ctx: any) => Promise<any> | any) => {
      handlers.set(name, handler);
    },
    logger,
  };

  registerAutonomyHooks({ api, client: client as BridgeClient, config: makeConfig(mode) });
  return { handlers, client, logger };
}

describe("before_tool_call directive enforcement", () => {
  it("passes rich scoring context to beforeAction", async () => {
    const { handlers, client } = setup("active", decision({ directive: "PROCEED" }));
    const beforeTool = handlers.get("before_tool_call");

    await beforeTool?.(
      {
        toolName: "exec",
        params: { command: "ls -la" },
        messages: [
          { role: "user", content: "please inspect workspace" },
          { role: "assistant", content: "running checks" },
        ],
      },
      {
        sessionId: "s1",
        runId: "r1",
        health: { overall_score: 82 },
        predictive: { p_success: 0.74, n_similar: 9 },
        meta: { penalty: 0.2 },
        ambiguityScore: 0.3,
        complexityScore: 0.45,
        recentTools: ["read", "exec"],
      },
    );

    expect(client.beforeAction).toHaveBeenCalledTimes(1);
    const payload = (client.beforeAction as any).mock.calls[0][0];
    expect(payload.toolName).toBe("exec");
    expect(payload.params.command).toBe("ls -la");
    expect(payload.health.normalized).toBeCloseTo(0.82, 2);
    expect(payload.predictive.p_hist).toBeCloseTo(0.74, 2);
    expect(payload.predictive.sample_size).toBe(9);
    expect(payload.meta.penalty).toBeCloseTo(0.2, 2);
    expect(payload.history.messagesCount).toBe(2);
    expect(payload.history.lastUserMessage).toContain("inspect");
    expect(payload.history.recentTools).toEqual(["read", "exec"]);
    expect(payload.session.sessionId).toBe("s1");
  });

  it("fail-open path still proceeds", async () => {
    const { handlers } = setup("active", decision({ directive: "PROCEED", degraded: true }));
    const beforeTool = handlers.get("before_tool_call");
    const result = await beforeTool?.({ toolName: "read", params: { path: "README.md" } }, {});
    expect(result).toBeUndefined();
  });

  it("blocks on BLOCK directive in active mode", async () => {
    const { handlers } = setup("active", decision({ directive: "BLOCK", metadata: { reason: "Denied" } }));
    const beforeTool = handlers.get("before_tool_call");
    const result = await beforeTool?.({ toolName: "exec", params: { command: "rm -rf /" } }, {});
    expect(result?.block).toBe(true);
    expect(result?.blockReason).toContain("Denied");
  });

  it("requires confirmation on ASK_CONFIRMATION", async () => {
    const { handlers } = setup("active", decision({ directive: "ASK_CONFIRMATION", metadata: { reason: "Need approval" } }));
    const beforeTool = handlers.get("before_tool_call");

    const first = await beforeTool?.({ toolName: "exec", params: { command: "danger" } }, { sessionId: "s-confirm" });
    expect(first?.block).toBe(true);
    expect(first?.requiresConfirmation).toBe(true);

    const second = await beforeTool?.(
      { toolName: "exec", params: { command: "danger" }, autonomyConfirmed: true },
      { sessionId: "s-confirm" },
    );
    expect(second).toBeUndefined();
  });

  it("does not bypass confirmation on second identical call without explicit flag", async () => {
    const { handlers } = setup("active", decision({ directive: "ASK_CONFIRMATION", metadata: { reason: "Need approval" } }));
    const beforeTool = handlers.get("before_tool_call");

    const first = await beforeTool?.({ toolName: "exec", params: { command: "danger" } }, { sessionId: "s-bypass" });
    const second = await beforeTool?.({ toolName: "exec", params: { command: "danger" } }, { sessionId: "s-bypass" });

    expect(first?.block).toBe(true);
    expect(second?.block).toBe(true);
    expect(second?.requiresConfirmation).toBe(true);
  });

  it("keeps confirmations session-scoped", async () => {
    const { handlers } = setup("active", decision({ directive: "ASK_CONFIRMATION", metadata: { reason: "Need approval" } }));
    const beforeTool = handlers.get("before_tool_call");

    await beforeTool?.({ toolName: "exec", params: { command: "danger" } }, { sessionId: "session-a" });

    const crossSession = await beforeTool?.(
      { toolName: "exec", params: { command: "danger" }, autonomyConfirmed: true },
      { sessionId: "session-b" },
    );

    expect(crossSession?.block).toBe(true);
    expect(crossSession?.requiresConfirmation).toBe(true);
  });

  it("clears pending confirmations on session end", async () => {
    const { handlers } = setup("active", decision({ directive: "ASK_CONFIRMATION", metadata: { reason: "Need approval" } }));
    const beforeTool = handlers.get("before_tool_call");
    const sessionEnd = handlers.get("session_end");

    await beforeTool?.({ toolName: "exec", params: { command: "danger" } }, { sessionId: "session-z" });
    await sessionEnd?.({}, { sessionId: "session-z" });

    const staleConfirm = await beforeTool?.(
      { toolName: "exec", params: { command: "danger" }, autonomyConfirmed: true },
      { sessionId: "session-z" },
    );

    expect(staleConfirm?.block).toBe(true);
    expect(staleConfirm?.requiresConfirmation).toBe(true);
  });

  it("switches tool when SWITCH_ALTERNATIVE provided", async () => {
    const { handlers } = setup(
      "active",
      decision({ directive: "SWITCH_ALTERNATIVE", metadata: { alternativeTool: "read", reason: "safer" } }),
    );
    const beforeTool = handlers.get("before_tool_call");
    const result = await beforeTool?.({ toolName: "exec", params: { command: "cat file" } }, {});

    expect(result?.toolName).toBe("read");
    expect(result?.replaceToolCall?.toolName).toBe("read");
  });

  it("logs intervention notifications", async () => {
    const { handlers, logger } = setup(
      "active",
      decision({
        directive: "ASK_CONFIRMATION",
        metadata: {
          reason: "intervention",
          intervention: {
            id: "i-1",
            type: "config",
            priority: "HIGH",
            directive: "ASK_CONFIRMATION",
          },
          interventionAutoApplied: true,
        },
      }),
    );

    const beforeTool = handlers.get("before_tool_call");
    await beforeTool?.({ toolName: "write", params: { path: "config.json", content: "{}" } }, {});

    expect(logger.info).toHaveBeenCalledWith(
      "autonomy.intervention",
      expect.objectContaining({
        intervention: expect.objectContaining({ type: "config" }),
      }),
    );
  });

  it("asks clarification in active mode", async () => {
    const { handlers } = setup("active", decision({ directive: "ASK_CLARIFICATION", metadata: { reason: "Need context" } }));
    const beforeTool = handlers.get("before_tool_call");
    const result = await beforeTool?.({ toolName: "exec", params: { command: "do-it" } }, {});

    expect(result?.block).toBe(true);
    expect(result?.requiresClarification).toBe(true);
  });

  it("uses meta-state to block looping even when directive is PROCEED", async () => {
    const { handlers } = setup(
      "active",
      decision({ directive: "PROCEED", metadata: { meta: { state: "looping", confidence: 0.9 } } }),
    );
    const beforeTool = handlers.get("before_tool_call");
    const result = await beforeTool?.({ toolName: "exec", params: { command: "retry" } }, {});

    expect(result?.block).toBe(true);
    expect(result?.requiresClarification).toBe(true);
  });

  it("does not block in suggest mode", async () => {
    const { handlers } = setup("suggest", decision({ directive: "BLOCK", metadata: { reason: "too risky" } }));
    const beforeTool = handlers.get("before_tool_call");
    const result = await beforeTool?.({ toolName: "exec", params: { command: "rm -rf /" } }, {});

    expect(result).toBeUndefined();
  });

  it("uses runtime mode from bridge instead of static config mode", async () => {
    const { handlers, client } = setup(
      "shadow",
      decision({ directive: "BLOCK", metadata: { reason: "runtime active block" } }),
      "active",
    );
    const beforeTool = handlers.get("before_tool_call");
    const result = await beforeTool?.({ toolName: "exec", params: { command: "rm -rf /" } }, {});

    expect((client.currentMode as any)).toHaveBeenCalledTimes(1);
    expect(result?.block).toBe(true);
    expect(result?.blockReason).toContain("runtime active block");
  });
});

describe("after_tool_call telemetry", () => {
  it("sends outcome asynchronously", async () => {
    const { handlers, client } = setup("active", decision({ directive: "PROCEED" }));
    const beforeTool = handlers.get("before_tool_call");
    const afterTool = handlers.get("after_tool_call");

    await beforeTool?.({ toolName: "read", params: { path: "a" } }, { sessionId: "s-1", estimatedActions: 4 });
    await afterTool?.({ toolName: "read", params: { path: "a" }, result: "ok", durationMs: 3 }, { sessionId: "s-1" });

    expect(client.afterAction).toHaveBeenCalledTimes(1);
    const payload = (client.afterAction as any).mock.calls[0][0];
    expect(payload.ok).toBe(true);
    expect(payload.history.actionCount).toBe(1);
    expect(payload.history.recentActions[0].toolName).toBe("read");
    expect(payload.estimated_actions).toBe(4);
  });

  it("tracks intervention effectiveness in after_action payload", async () => {
    const { handlers, client } = setup(
      "active",
      decision({
        directive: "ASK_CONFIRMATION",
        metadata: {
          reason: "verify",
          intervention: {
            id: "i-2",
            type: "interaction",
            priority: "CRITICAL",
            directive: "ASK_CONFIRMATION",
          },
          interventionAutoApplied: true,
        },
      }),
    );
    const beforeTool = handlers.get("before_tool_call");
    const afterTool = handlers.get("after_tool_call");

    await beforeTool?.(
      { toolName: "exec", params: { command: "risky" }, autonomyConfirmed: true },
      { sessionId: "s-2" },
    );
    await afterTool?.({ toolName: "exec", params: { command: "risky" }, result: "ok", durationMs: 8 }, { sessionId: "s-2" });

    const payload = (client.afterAction as any).mock.calls[0][0];
    expect(payload.interventionOutcome.intervention.type).toBe("interaction");
    expect(payload.interventionOutcome.effective).toBe(true);
    expect(payload.history.interventionStats.effective).toBe(1);
  });
});

describe("agent_end reflection handshake", () => {
  it("sends task-boundary session summary and logs reflection notifications", async () => {
    const { handlers, client, logger } = setup("active", decision({ directive: "PROCEED" }));
    const afterTool = handlers.get("after_tool_call");
    const agentEnd = handlers.get("agent_end");

    (client.onAssistantTurn as any).mockResolvedValue(
      decision({
        directive: "PROCEED",
        metadata: {
          reflection: {
            task_id: "task-1",
            summary: "Task task-1 completed with mixed outcomes",
          },
          lesson_candidates: [{ id: "l-1" }],
        },
      }),
    );

    await afterTool?.({ toolName: "read", params: { path: "a" }, result: "ok", durationMs: 3 }, { sessionId: "s-9" });
    await afterTool?.(
      { toolName: "exec", params: { command: "cat missing" }, error: { name: "ENOENT" }, durationMs: 14 },
      { sessionId: "s-9" },
    );

    await agentEnd?.({ success: true, durationMs: 180000, messages: [] }, { sessionId: "s-9" });

    expect(client.onAssistantTurn).toHaveBeenCalledTimes(1);
    const payload = (client.onAssistantTurn as any).mock.calls[0][0];
    expect(payload.taskBoundary).toBe(true);
    expect(payload.sessionSummary.actionCount).toBe(2);
    expect(payload.sessionSummary.mixedOutcomes).toBe(true);

    await Promise.resolve();
    expect(logger.info).toHaveBeenCalledWith(
      "autonomy.reflection.generated",
      expect.objectContaining({
        taskId: "task-1",
        lessonCandidates: 1,
      }),
    );
  });
});
