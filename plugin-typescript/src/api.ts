import http from "node:http";
import { randomUUID } from "node:crypto";
import type { AutonomyPluginConfig, PluginMode } from "./config.js";

export type BridgeDirective =
  | "PROCEED"
  | "PROCEED_WITH_CAUTION"
  | "SWITCH_ALTERNATIVE"
  | "ASK_CLARIFICATION"
  | "ASK_CONFIRMATION"
  | "BLOCK";

export interface BridgeRequestEnvelope {
  action_id: string;
  hook_type: string;
  context: Record<string, unknown>;
  timestamp: number;
}

export interface BridgeResponseEnvelope {
  directive: BridgeDirective;
  confidence: number;
  metadata: Record<string, unknown>;
  timestamp: number;
  meta_state?: string;
  meta_confidence?: number;
  health?: Record<string, unknown>;
  adaptive?: Record<string, unknown>;
  intervention?: Record<string, unknown>;
  interventions?: Record<string, unknown>;
}

export interface BridgeDecision extends BridgeResponseEnvelope {
  degraded?: boolean;
  roundTripMs?: number;
}

interface BridgeHttpResponse {
  statusCode: number;
  body: string;
  elapsedMs: number;
}

type BreakerState = "CLOSED" | "OPEN" | "HALF_OPEN";

class CircuitBreaker {
  private state: BreakerState = "CLOSED";
  private failures = 0;
  private openUntilMs = 0;

  constructor(
    private readonly failureThreshold: number,
    private readonly cooldownMs: number,
  ) {}

  canRequest(now = Date.now()): boolean {
    if (this.state === "OPEN") {
      if (now >= this.openUntilMs) {
        this.state = "HALF_OPEN";
        return true;
      }
      return false;
    }
    return true;
  }

  onSuccess(): void {
    this.failures = 0;
    this.state = "CLOSED";
    this.openUntilMs = 0;
  }

  onFailure(now = Date.now()): void {
    this.failures += 1;
    if (this.failures >= this.failureThreshold) {
      this.state = "OPEN";
      this.openUntilMs = now + this.cooldownMs;
    }
  }

  snapshot() {
    return {
      state: this.state,
      failures: this.failures,
      openUntilMs: this.openUntilMs,
    };
  }
}

export class BridgeClient {
  private readonly breaker: CircuitBreaker;

  constructor(private readonly config: AutonomyPluginConfig) {
    this.breaker = new CircuitBreaker(config.circuitBreaker.failureThreshold, config.circuitBreaker.cooldownMs);
  }

  async health(timeoutMs = this.config.bridge.requestTimeoutMs): Promise<{ ok: boolean; elapsedMs?: number; details?: unknown }> {
    try {
      const res = await this.request("GET", "/health", undefined, timeoutMs);
      const parsed = JSON.parse(res.body) as Record<string, unknown>;
      return { ok: res.statusCode >= 200 && res.statusCode < 300, elapsedMs: res.elapsedMs, details: parsed };
    } catch (error) {
      return { ok: false, details: String(error) };
    }
  }

  async currentMode(timeoutMs = this.config.bridge.requestTimeoutMs): Promise<PluginMode | undefined> {
    try {
      const res = await this.request("GET", "/v1/status", undefined, timeoutMs);
      if (res.statusCode < 200 || res.statusCode >= 300) {
        return undefined;
      }

      const parsed = JSON.parse(res.body) as Record<string, unknown>;
      const autonomy = this.isRecord(parsed.autonomy) ? parsed.autonomy : undefined;
      const coordinator = this.isRecord(parsed.coordinator) ? parsed.coordinator : undefined;

      const modeCandidate =
        (typeof autonomy?.mode === "string" ? autonomy.mode : undefined) ??
        (typeof coordinator?.mode === "string" ? coordinator.mode : undefined) ??
        (typeof parsed.mode === "string" ? parsed.mode : undefined);

      return this.normalizeMode(modeCandidate);
    } catch {
      return undefined;
    }
  }

  async beforeAction(context: Record<string, unknown>): Promise<BridgeDecision> {
    return this.call("/v1/preflight", "before_action", context, this.config.bridge.preflightTimeoutMs);
  }

  async afterAction(context: Record<string, unknown>): Promise<BridgeDecision> {
    return this.call("/v1/postflight", "after_action", context, this.config.bridge.asyncTimeoutMs);
  }

  async onUserMessage(context: Record<string, unknown>): Promise<BridgeDecision> {
    return this.call("/v1/message-event", "on_user_message", context, this.config.bridge.asyncTimeoutMs);
  }

  async onAssistantTurn(context: Record<string, unknown>): Promise<BridgeDecision> {
    return this.call("/v1/run-end", "on_assistant_turn", context, this.config.bridge.asyncTimeoutMs);
  }

  async onHealthUpdate(context: Record<string, unknown>): Promise<BridgeDecision> {
    return this.call("/v1/health", "on_health_update", context, this.config.bridge.asyncTimeoutMs);
  }

  getCircuitBreakerSnapshot() {
    return this.breaker.snapshot();
  }

  private async call(
    endpoint: string,
    hookType: string,
    context: Record<string, unknown>,
    timeoutMs: number,
  ): Promise<BridgeDecision> {
    if (!this.breaker.canRequest()) {
      return this.degradedDecision(hookType, "circuit_open", { circuitBreaker: this.breaker.snapshot() });
    }

    const envelope: BridgeRequestEnvelope = {
      action_id: randomUUID(),
      hook_type: hookType,
      context,
      timestamp: Date.now(),
    };

    try {
      const response = await this.request("POST", endpoint, envelope, timeoutMs);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw new Error(`Bridge HTTP ${response.statusCode}: ${response.body}`);
      }

      const parsed = JSON.parse(response.body) as Partial<BridgeResponseEnvelope>;
      const directive = parsed.directive ?? "PROCEED";
      const metadata: Record<string, unknown> = {
        ...(this.isRecord(parsed.metadata) ? parsed.metadata : {}),
        endpoint,
      };
      if (typeof parsed.meta_state === "string" && parsed.meta_state.length > 0) {
        metadata.metaState = parsed.meta_state;
      }
      if (typeof parsed.meta_confidence === "number") {
        metadata.metaConfidence = parsed.meta_confidence;
      }
      if (this.isRecord(parsed.health)) {
        metadata.health = parsed.health;
      }
      if (this.isRecord(parsed.adaptive)) {
        metadata.adaptive = parsed.adaptive;
      }
      if (this.isRecord(parsed.intervention)) {
        metadata.intervention = parsed.intervention;
      }
      if (this.isRecord(parsed.interventions)) {
        metadata.interventions = parsed.interventions;
      }

      const decision: BridgeDecision = {
        directive: this.normalizeDirective(directive),
        confidence: typeof parsed.confidence === "number" ? parsed.confidence : 0,
        metadata,
        timestamp: typeof parsed.timestamp === "number" ? parsed.timestamp : Date.now(),
        roundTripMs: response.elapsedMs,
      };

      this.breaker.onSuccess();
      return decision;
    } catch (error) {
      this.breaker.onFailure();
      return this.degradedDecision(hookType, "bridge_error", {
        error: String(error),
        circuitBreaker: this.breaker.snapshot(),
      });
    }
  }

  private normalizeDirective(input: string): BridgeDirective {
    const normalized = String(input || "").toUpperCase();
    if (
      normalized === "BLOCK" ||
      normalized === "PROCEED" ||
      normalized === "PROCEED_WITH_CAUTION" ||
      normalized === "SWITCH_ALTERNATIVE" ||
      normalized === "ASK_CLARIFICATION" ||
      normalized === "ASK_CONFIRMATION"
    ) {
      return normalized;
    }

    // Back-compat alias from Chunk A plan wording.
    if (normalized === "CAUTION") {
      return "PROCEED_WITH_CAUTION";
    }

    return "PROCEED";
  }

  private degradedDecision(hookType: string, reason: string, metadata: Record<string, unknown>): BridgeDecision {
    const failOpen = this.config.failOpen;
    return {
      directive: failOpen ? "PROCEED" : "BLOCK",
      confidence: 0,
      metadata: {
        failOpen,
        failClosed: !failOpen,
        hookType,
        reason,
        ...metadata,
      },
      timestamp: Date.now(),
      degraded: true,
    };
  }

  private normalizeMode(mode: unknown): PluginMode | undefined {
    if (mode === "shadow" || mode === "suggest" || mode === "active") {
      return mode;
    }
    return undefined;
  }

  private request(
    method: "GET" | "POST",
    endpoint: string,
    payload: BridgeRequestEnvelope | undefined,
    timeoutMs: number,
  ): Promise<BridgeHttpResponse> {
    const started = process.hrtime.bigint();

    return new Promise((resolve, reject) => {
      const body = payload ? JSON.stringify(payload) : "";

      const req = http.request(
        {
          socketPath: this.config.bridge.socketPath,
          method,
          path: endpoint,
          headers: {
            "content-type": "application/json",
            "content-length": Buffer.byteLength(body),
          },
        },
        (res) => {
          const chunks: Buffer[] = [];
          res.on("data", (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk))));
          res.on("end", () => {
            const elapsedMs = Number(process.hrtime.bigint() - started) / 1_000_000;
            resolve({
              statusCode: res.statusCode ?? 500,
              body: Buffer.concat(chunks).toString("utf8"),
              elapsedMs,
            });
          });
        },
      );

      req.setTimeout(Math.min(100, timeoutMs), () => {
        req.destroy(new Error(`Bridge timeout after ${timeoutMs}ms`));
      });

      req.on("error", reject);

      if (body.length > 0) {
        req.write(body);
      }
      req.end();
    });
  }

  private isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
  }
}
