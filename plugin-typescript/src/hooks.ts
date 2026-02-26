import type { BridgeClient, BridgeDecision } from "./api.js";
import type { AutonomyPluginConfig } from "./config.js";

type LoggerLike = {
  info?: (message: string, meta?: unknown) => void;
  warn?: (message: string, meta?: unknown) => void;
  error?: (message: string, meta?: unknown) => void;
  debug?: (message: string, meta?: unknown) => void;
};

type PluginApiLike = {
  on: (hookName: string, handler: (event: any, ctx: any) => Promise<any> | any, opts?: { priority?: number }) => void;
  logger?: LoggerLike;
};

interface RegisterHooksParams {
  api: PluginApiLike;
  client: BridgeClient;
  config: AutonomyPluginConfig;
}

interface SessionActionEntry {
  index: number;
  toolName: string;
  ok: boolean;
  durationMs?: number;
  errorType?: string;
  confidence?: number;
  progress?: number;
  timestamp: number;
}

interface InterventionSummary {
  id?: string;
  type?: string;
  priority?: string;
  directive?: string;
  reason?: string;
  autoApplied?: boolean;
  suggested?: boolean;
}

interface InterventionStats {
  applied: number;
  suggested: number;
  effective: number;
  ineffective: number;
}

interface SessionAutonomyState {
  key: string;
  actions: SessionActionEntry[];
  actionCounter: number;
  pendingConfirmations: Set<string>;
  lastDirective?: string;
  lastMetaState?: string;
  lastMetaConfidence?: number;
  estimatedActions?: number;
  explicitUserApproval?: boolean;
  longWorkflowRequested?: boolean;
  batchOperation?: boolean;
  lastIntervention?: InterventionSummary;
  interventionStats: InterventionStats;
  lastAdaptiveProfile?: string;
}

const BEFORE_TOOL_BUDGET_MS = 6;
const RECENT_TOOL_LIMIT = 5;
const SESSION_ACTION_HISTORY_LIMIT = 24;

function toRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

function toArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function toErrorType(error: unknown): string | undefined {
  if (!error) return undefined;
  if (typeof error === "string") return "string_error";
  const rec = toRecord(error);
  if (typeof rec.name === "string") return rec.name;
  return "unknown_error";
}

function getDirectiveReason(decision: BridgeDecision, fallback: string): string {
  const reason = decision.metadata.reason;
  return typeof reason === "string" ? reason : fallback;
}

function signature(toolName: string, params: Record<string, unknown>): string {
  return `${toolName}::${JSON.stringify(params)}`;
}

function firstRecord(...candidates: unknown[]): Record<string, unknown> {
  for (const candidate of candidates) {
    if (typeof candidate === "object" && candidate !== null && !Array.isArray(candidate)) {
      return candidate as Record<string, unknown>;
    }
  }
  return {};
}

function firstArray(...candidates: unknown[]): unknown[] {
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      return candidate;
    }
  }
  return [];
}

function normalizeHealth(value: unknown): Record<string, unknown> {
  const health = toRecord(value);
  const normalizedCandidate =
    asNumber(health.normalized) ??
    asNumber(health.score) ??
    asNumber(health.overall) ??
    asNumber(health.overall_score) ??
    asNumber(health.healthScore);

  let normalized = normalizedCandidate;
  if (typeof normalized === "number" && normalized > 1) {
    normalized = normalized / 100;
  }

  return {
    ...health,
    ...(typeof normalized === "number" ? { normalized: Math.min(1, Math.max(0, normalized)) } : {}),
  };
}

function normalizePredictive(value: unknown): Record<string, unknown> {
  const predictive = toRecord(value);
  const pHist =
    asNumber(predictive.p_hist) ?? asNumber(predictive.p_success) ?? asNumber(predictive.predicted_success) ?? undefined;
  const sampleSize =
    asNumber(predictive.sample_size) ?? asNumber(predictive.n_similar) ?? asNumber(predictive.count) ?? undefined;

  return {
    ...predictive,
    ...(typeof pHist === "number" ? { p_hist: pHist } : {}),
    ...(typeof sampleSize === "number" ? { sample_size: Math.max(0, Math.floor(sampleSize)) } : {}),
  };
}

function normalizeMeta(value: unknown): Record<string, unknown> {
  const meta = toRecord(value);
  const penalty = asNumber(meta.penalty) ?? asNumber(meta.meta_penalty) ?? undefined;
  return {
    ...meta,
    ...(typeof penalty === "number" ? { penalty: Math.min(1, Math.max(0, penalty)) } : {}),
  };
}

function extractLastMessage(messages: unknown[], role: "user" | "assistant"): string | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = toRecord(messages[i]);
    if (msg.role !== role) {
      continue;
    }

    const content = msg.content;
    if (typeof content === "string" && content.length > 0) {
      return content;
    }

    if (Array.isArray(content)) {
      const firstText = content.find((entry) => typeof toRecord(entry).text === "string");
      if (firstText) {
        const text = toRecord(firstText).text;
        if (typeof text === "string" && text.length > 0) {
          return text;
        }
      }
    }
  }

  return undefined;
}

function extractRecentTools(eventRec: Record<string, unknown>, ctxRec: Record<string, unknown>): string[] {
  const history = toRecord(ctxRec.history);
  const candidate = firstArray(ctxRec.recentTools, eventRec.recentTools, history.recentTools);

  return candidate
    .map((item) =>
      typeof item === "string" ? item : typeof toRecord(item).toolName === "string" ? String(toRecord(item).toolName) : undefined,
    )
    .filter((item): item is string => typeof item === "string" && item.length > 0)
    .slice(-RECENT_TOOL_LIMIT);
}

function resolveSession(eventRec: Record<string, unknown>, ctxRec: Record<string, unknown>) {
  const historyRec = toRecord(ctxRec.history);

  const session = {
    sessionId:
      asString(ctxRec.sessionId) ??
      asString(ctxRec.session_id) ??
      asString(historyRec.sessionId) ??
      asString(eventRec.sessionId) ??
      asString(eventRec.session_id),
    sessionKey:
      asString(ctxRec.sessionKey) ??
      asString(ctxRec.session_key) ??
      asString(historyRec.sessionKey) ??
      asString(eventRec.sessionKey),
    runId: asString(ctxRec.runId) ?? asString(ctxRec.run_id) ?? asString(eventRec.runId),
    agentId: asString(ctxRec.agentId) ?? asString(ctxRec.agent_id) ?? asString(eventRec.agentId),
  };

  const key = session.sessionKey ?? session.sessionId ?? session.runId ?? "default";
  return { session, key, historyRec };
}

function getSessionState(store: Map<string, SessionAutonomyState>, key: string): SessionAutonomyState {
  const existing = store.get(key);
  if (existing) {
    return existing;
  }

  const created: SessionAutonomyState = {
    key,
    actions: [],
    actionCounter: 0,
    pendingConfirmations: new Set<string>(),
    interventionStats: {
      applied: 0,
      suggested: 0,
      effective: 0,
      ineffective: 0,
    },
  };
  store.set(key, created);
  return created;
}

function extractMetaState(decision: BridgeDecision): { state?: string; confidence?: number } {
  const metaRecord = toRecord(decision.metadata.meta);
  const state =
    asString(metaRecord.state) ??
    asString(decision.metadata.metaState) ??
    asString((decision.metadata as Record<string, unknown>).meta_state);
  const confidence =
    asNumber(metaRecord.confidence) ??
    asNumber(decision.metadata.metaConfidence) ??
    asNumber((decision.metadata as Record<string, unknown>).meta_confidence);

  const out: { state?: string; confidence?: number } = {};
  if (typeof state === "string") {
    out.state = state;
  }
  if (typeof confidence === "number") {
    out.confidence = confidence;
  }
  return out;
}

function extractIntervention(decision: BridgeDecision): InterventionSummary | undefined {
  const intervention = toRecord(decision.metadata.intervention);
  if (Object.keys(intervention).length === 0) {
    return undefined;
  }

  const reason = asString(intervention.reason) ?? asString(decision.metadata.reason);
  const id = asString(intervention.id);
  const type = asString(intervention.type);
  const priority = asString(intervention.priority);
  const directive = asString(intervention.directive) ?? decision.directive;

  const out: InterventionSummary = {
    directive,
    autoApplied: decision.metadata.interventionAutoApplied === true,
    suggested: decision.metadata.interventionSuggested === true,
  };

  if (id) {
    out.id = id;
  }
  if (type) {
    out.type = type;
  }
  if (priority) {
    out.priority = priority;
  }
  if (reason) {
    out.reason = reason;
  }

  return out;
}

function compactActions(actions: SessionActionEntry[]): SessionActionEntry[] {
  if (actions.length <= SESSION_ACTION_HISTORY_LIMIT) {
    return actions;
  }
  return actions.slice(actions.length - SESSION_ACTION_HISTORY_LIMIT);
}

function buildSessionSummary(
  key: string,
  state: SessionAutonomyState,
  eventRec: Record<string, unknown>,
  session: Record<string, unknown>,
): Record<string, unknown> {
  const recentActions = state.actions.slice(-SESSION_ACTION_HISTORY_LIMIT);
  const actionCount = state.actionCounter;
  const successCount = recentActions.filter((item) => item.ok).length;
  const failureCount = recentActions.filter((item) => !item.ok).length;
  const mixedOutcomes = successCount > 0 && failureCount > 0;

  const firstAction = recentActions[0];
  const lastAction = recentActions.length > 0 ? recentActions[recentActions.length - 1] : undefined;
  const inferredDurationMs =
    firstAction && lastAction ? Math.max(0, lastAction.timestamp - firstAction.timestamp) : 0;

  const timeSpentMs = asNumber(eventRec.durationMs) ?? inferredDurationMs;

  const boundaryId =
    asString(eventRec.taskId) ??
    asString(eventRec.task_id) ??
    asString(eventRec.boundaryId) ??
    asString(session.runId) ??
    `${key}:task:${Date.now()}`;

  return {
    taskId: boundaryId,
    boundaryId,
    actionCount,
    successCount,
    failureCount,
    mixedOutcomes,
    timeSpentMs,
    recentActions,
    lastDirective: state.lastDirective,
    lastMetaState: state.lastMetaState,
    interventionStats: state.interventionStats,
    taskBoundary: true,
    significantWorkCandidate: actionCount > 3 || mixedOutcomes || timeSpentMs > 120000,
  };
}

function buildBeforeActionContext(
  event: unknown,
  ctx: unknown,
  mode: AutonomyPluginConfig["mode"],
  state: SessionAutonomyState,
): Record<string, unknown> {
  const eventRec = toRecord(event);
  const ctxRec = toRecord(ctx);
  const { session, historyRec } = resolveSession(eventRec, ctxRec);

  const toolName = String(eventRec.toolName ?? "");
  const params = toRecord(eventRec.params);

  const messages = firstArray(eventRec.messages, ctxRec.messages, historyRec.messages);
  const recentTools = extractRecentTools(eventRec, ctxRec);

  const health = normalizeHealth(firstRecord(ctxRec.health, eventRec.health, ctxRec.autonomyHealth, historyRec.health));
  const predictive = normalizePredictive(
    firstRecord(ctxRec.predictive, eventRec.predictive, ctxRec.autonomyPredictive, historyRec.predictive),
  );
  const meta = normalizeMeta(firstRecord(ctxRec.meta, eventRec.meta, ctxRec.autonomyMeta, historyRec.meta));

  const ambiguity =
    asNumber(ctxRec.ambiguity) ??
    asNumber(ctxRec.ambiguityScore) ??
    asNumber(eventRec.ambiguity) ??
    asNumber(historyRec.ambiguity);
  const complexity =
    asNumber(ctxRec.complexity) ??
    asNumber(ctxRec.complexityScore) ??
    asNumber(eventRec.complexity) ??
    asNumber(historyRec.complexity);
  const metaPenalty = asNumber(ctxRec.meta_penalty) ?? asNumber(meta.penalty) ?? asNumber(eventRec.meta_penalty);

  const calibration = firstRecord(ctxRec.calibration, eventRec.calibration, historyRec.calibration);

  const estimatedActions =
    asNumber(ctxRec.estimated_actions) ??
    asNumber(ctxRec.estimatedActions) ??
    asNumber(eventRec.estimated_actions) ??
    asNumber(eventRec.estimatedActions) ??
    state.estimatedActions;
  if (typeof estimatedActions === "number" && estimatedActions > 0) {
    state.estimatedActions = Math.max(1, Math.floor(estimatedActions));
  }

  const explicitUserApproval =
    asBoolean(ctxRec.explicitUserApproval) ??
    asBoolean(ctxRec.approvedExtendedWorkflow) ??
    asBoolean(eventRec.explicitUserApproval) ??
    state.explicitUserApproval;
  if (explicitUserApproval) {
    state.explicitUserApproval = true;
  }

  const longWorkflowRequested =
    asBoolean(ctxRec.longWorkflowRequested) ?? asBoolean(eventRec.longWorkflowRequested) ?? state.longWorkflowRequested;
  if (longWorkflowRequested) {
    state.longWorkflowRequested = true;
  }

  const batchOperation = asBoolean(ctxRec.batchOperation) ?? asBoolean(eventRec.batchOperation) ?? state.batchOperation;
  if (batchOperation) {
    state.batchOperation = true;
  }

  return {
    toolName,
    params,
    mode,
    session,
    history: {
      messagesCount: messages.length,
      lastUserMessage: extractLastMessage(messages, "user"),
      lastAssistantMessage: extractLastMessage(messages, "assistant"),
      recentTools,
      actionCount: state.actionCounter,
      recentActions: state.actions.slice(-8),
      lastDirective: state.lastDirective,
      lastMetaState: state.lastMetaState,
    },
    health,
    predictive,
    meta,
    ...(typeof ambiguity === "number" ? { ambiguity } : {}),
    ...(typeof complexity === "number" ? { complexity } : {}),
    ...(typeof metaPenalty === "number" ? { meta_penalty: metaPenalty } : {}),
    ...(typeof state.estimatedActions === "number" ? { estimated_actions: state.estimatedActions } : {}),
    ...(explicitUserApproval ? { explicitUserApproval: true } : {}),
    ...(longWorkflowRequested ? { longWorkflowRequested: true } : {}),
    ...(batchOperation ? { batchOperation: true } : {}),
    ...(Object.keys(calibration).length > 0 ? { calibration } : {}),
    runtime: {
      channel: asString(ctxRec.channel) ?? asString(eventRec.channel),
      workspaceRoot: asString(ctxRec.workspaceRoot) ?? asString(ctxRec.workspace) ?? asString(process.cwd()),
    },
  };
}

async function resolveCurrentMode(
  client: BridgeClient,
  fallback: AutonomyPluginConfig["mode"],
  logger?: LoggerLike,
): Promise<AutonomyPluginConfig["mode"]> {
  try {
    const runtimeMode = await client.currentMode();
    if (runtimeMode === "shadow" || runtimeMode === "suggest" || runtimeMode === "active") {
      return runtimeMode;
    }
  } catch (error) {
    logger?.debug?.("autonomy.mode.fetch.error", { error: String(error) });
  }

  return fallback;
}

export function registerAutonomyHooks({ api, client, config }: RegisterHooksParams): void {
  const sessionStates = new Map<string, SessionAutonomyState>();

  api.on("before_agent_start", async (event, ctx) => {
    const decision = await client.onUserMessage({
      prompt: event?.prompt,
      messagesCount: Array.isArray(event?.messages) ? event.messages.length : undefined,
      context: toRecord(ctx),
    });

    api.logger?.debug?.("autonomy.before_agent_start", {
      directive: decision.directive,
      confidence: decision.confidence,
      degraded: decision.degraded,
    });
  });

  api.on("before_tool_call", async (event, ctx) => {
    const started = process.hrtime.bigint();
    const eventRec = toRecord(event);
    const ctxRec = toRecord(ctx);
    const originalToolName = String(eventRec.toolName ?? "");
    const originalParams = toRecord(eventRec.params);

    const { key } = resolveSession(eventRec, ctxRec);
    const state = getSessionState(sessionStates, key);

    const runtimeMode = await resolveCurrentMode(client, config.mode, api.logger);
    const decision = await client.beforeAction(buildBeforeActionContext(event, ctx, runtimeMode, state));
    const metaDecision = extractMetaState(decision);
    if (metaDecision.state) {
      state.lastMetaState = metaDecision.state;
    }
    if (typeof metaDecision.confidence === "number") {
      state.lastMetaConfidence = metaDecision.confidence;
    }
    state.lastDirective = decision.directive;

    const intervention = extractIntervention(decision);
    if (intervention) {
      state.lastIntervention = intervention;
      if (intervention.autoApplied) {
        state.interventionStats.applied += 1;
      } else {
        state.interventionStats.suggested += 1;
      }

      api.logger?.info?.("autonomy.intervention", {
        toolName: originalToolName,
        mode: runtimeMode,
        directive: decision.directive,
        intervention,
      });
    }

    const adaptive = toRecord(decision.metadata.adaptive);
    const adaptiveProfile = asString(adaptive.profile);
    if (adaptiveProfile && adaptiveProfile !== state.lastAdaptiveProfile) {
      state.lastAdaptiveProfile = adaptiveProfile;
      api.logger?.info?.("autonomy.adaptive.profile", {
        sessionKey: key,
        profile: adaptiveProfile,
        decisionDirective: decision.directive,
      });
    }

    const elapsedMs = Number(process.hrtime.bigint() - started) / 1_000_000;
    if (elapsedMs > BEFORE_TOOL_BUDGET_MS) {
      api.logger?.warn?.("autonomy.before_tool_call.budget_exceeded", {
        toolName: originalToolName,
        elapsedMs,
        budgetMs: BEFORE_TOOL_BUDGET_MS,
        degraded: decision.degraded,
      });
    }

    const effectiveDirective =
      intervention?.autoApplied && typeof intervention.directive === "string" ? intervention.directive : decision.directive;

    if (runtimeMode === "shadow") {
      return undefined;
    }

    if (runtimeMode === "suggest") {
      if (decision.directive !== "PROCEED" || intervention?.suggested) {
        api.logger?.info?.("autonomy.before_tool_call.suggest", {
          toolName: originalToolName,
          directive: effectiveDirective,
          reason: intervention?.reason ?? decision.metadata.reason,
          confidence: decision.confidence,
          metaState: state.lastMetaState,
          intervention,
        });
      }
      return undefined;
    }

    if (effectiveDirective === "PROCEED" && state.lastMetaState === "looping") {
      return {
        block: true,
        blockReason: "Meta-cognition detected a repetition loop; ask for clarification before retrying.",
        requiresClarification: true,
      };
    }

    if (effectiveDirective === "PROCEED" && state.lastMetaState === "over_eager") {
      return {
        block: true,
        blockReason: "Meta-cognition detected over-eager drift; explicit confirmation required to continue.",
        requiresConfirmation: true,
      };
    }

    if (effectiveDirective === "BLOCK") {
      const reason = getDirectiveReason(decision, "Blocked by autonomy policy");
      api.logger?.warn?.("autonomy.before_tool_call.block", { toolName: originalToolName, reason, decision });
      return { block: true, blockReason: reason };
    }

    if (effectiveDirective === "ASK_CONFIRMATION") {
      const sig = signature(originalToolName, originalParams);
      const explicitConfirmation =
        toRecord(ctx).autonomyConfirmed === true ||
        toRecord(event).autonomyConfirmed === true ||
        toRecord(event).userConfirmed === true;

      const hasPending = state.pendingConfirmations.has(sig);
      if (explicitConfirmation && hasPending) {
        state.pendingConfirmations.delete(sig);
        return undefined;
      }

      state.pendingConfirmations.add(sig);
      const reason = getDirectiveReason(decision, "Action requires explicit user confirmation");
      return {
        block: true,
        blockReason: `${reason}. Re-run the tool call with autonomyConfirmed=true to proceed.`,
        requiresConfirmation: true,
      };
    }

    if (effectiveDirective === "SWITCH_ALTERNATIVE") {
      const altTool = typeof decision.metadata.alternativeTool === "string" ? decision.metadata.alternativeTool : undefined;
      if (altTool) {
        api.logger?.info?.("autonomy.before_tool_call.switch_alternative", {
          from: originalToolName,
          to: altTool,
          reason: decision.metadata.reason,
        });

        return {
          // keep both shapes for runtime compatibility
          toolName: altTool,
          params: originalParams,
          replaceToolCall: {
            toolName: altTool,
            params: originalParams,
            reason: decision.metadata.reason,
          },
        };
      }
    }

    if (effectiveDirective === "PROCEED_WITH_CAUTION") {
      api.logger?.info?.("autonomy.before_tool_call.caution", {
        toolName: originalToolName,
        confidence: decision.confidence,
        metadata: decision.metadata,
      });
      return undefined;
    }

    if (effectiveDirective === "ASK_CLARIFICATION") {
      const reason = getDirectiveReason(decision, "Need clarification before proceeding");
      return { block: true, blockReason: reason, requiresClarification: true };
    }

    return undefined;
  });

  api.on("after_tool_call", async (event, ctx) => {
    const eventRec = toRecord(event);
    const ctxRec = toRecord(ctx);
    const toolName = String(eventRec.toolName ?? "");

    const { key, session } = resolveSession(eventRec, ctxRec);
    const state = getSessionState(sessionStates, key);

    const progress =
      asNumber(eventRec.progress) ??
      asNumber(eventRec.progressScore) ??
      asNumber(eventRec.incrementalProgress) ??
      (eventRec.error ? 0.05 : 0.35);

    state.actionCounter += 1;
    const durationMs = asNumber(eventRec.durationMs);
    const errorType = toErrorType(eventRec.error);
    const confidence = asNumber(eventRec.confidence);
    const entry: SessionActionEntry = {
      index: state.actionCounter,
      toolName,
      ok: !eventRec.error,
      progress,
      timestamp: Date.now(),
      ...(typeof durationMs === "number" ? { durationMs } : {}),
      ...(typeof errorType === "string" ? { errorType } : {}),
      ...(typeof confidence === "number" ? { confidence } : {}),
    };
    state.actions = compactActions([...state.actions, entry]);

    const appliedIntervention = state.lastIntervention?.autoApplied ? state.lastIntervention : undefined;
    if (appliedIntervention) {
      if (!eventRec.error) {
        state.interventionStats.effective += 1;
      } else {
        state.interventionStats.ineffective += 1;
      }
    }

    const estimatedActions =
      asNumber(ctxRec.estimated_actions) ??
      asNumber(ctxRec.estimatedActions) ??
      asNumber(eventRec.estimated_actions) ??
      asNumber(eventRec.estimatedActions) ??
      state.estimatedActions;
    if (typeof estimatedActions === "number" && estimatedActions > 0) {
      state.estimatedActions = Math.max(1, Math.floor(estimatedActions));
    }

    const explicitUserApproval =
      asBoolean(ctxRec.explicitUserApproval) ??
      asBoolean(ctxRec.approvedExtendedWorkflow) ??
      asBoolean(eventRec.explicitUserApproval) ??
      state.explicitUserApproval;
    if (explicitUserApproval) {
      state.explicitUserApproval = true;
    }

    const longWorkflowRequested =
      asBoolean(ctxRec.longWorkflowRequested) ?? asBoolean(eventRec.longWorkflowRequested) ?? state.longWorkflowRequested;
    if (longWorkflowRequested) {
      state.longWorkflowRequested = true;
    }

    const batchOperation = asBoolean(ctxRec.batchOperation) ?? asBoolean(eventRec.batchOperation) ?? state.batchOperation;
    if (batchOperation) {
      state.batchOperation = true;
    }

    const payload = {
      toolName,
      params: toRecord(eventRec.params),
      result: eventRec.result,
      durationMs: eventRec.durationMs,
      ok: !eventRec.error,
      confidence: typeof eventRec.confidence === "number" ? eventRec.confidence : undefined,
      errorType: toErrorType(eventRec.error),
      error: eventRec.error,
      progress,
      session,
      history: {
        actionCount: state.actionCounter,
        recentActions: state.actions.slice(-8),
        lastDirective: state.lastDirective,
        lastMetaState: state.lastMetaState,
        interventionStats: state.interventionStats,
      },
      ...(appliedIntervention
        ? {
            interventionOutcome: {
              intervention: appliedIntervention,
              effective: !eventRec.error,
            },
          }
        : {}),
      ...(typeof state.estimatedActions === "number" ? { estimated_actions: state.estimatedActions } : {}),
      ...(explicitUserApproval ? { explicitUserApproval: true } : {}),
      ...(longWorkflowRequested ? { longWorkflowRequested: true } : {}),
      ...(batchOperation ? { batchOperation: true } : {}),
      ...(state.lastMetaState
        ? {
            meta: {
              state: state.lastMetaState,
              ...(typeof state.lastMetaConfidence === "number" ? { confidence: state.lastMetaConfidence } : {}),
            },
          }
        : {}),
      context: ctxRec,
    };

    delete state.lastIntervention;

    void client.afterAction(payload).catch((error) => api.logger?.warn?.("autonomy.after_tool_call.error", { error: String(error) }));
  });

  api.on("agent_end", async (event, ctx) => {
    const eventRec = toRecord(event);
    const ctxRec = toRecord(ctx);
    const { key, session } = resolveSession(eventRec, ctxRec);
    const state = getSessionState(sessionStates, key);

    const sessionSummary = buildSessionSummary(key, state, eventRec, session as Record<string, unknown>);

    void client
      .onAssistantTurn({
        success: eventRec.success,
        error: eventRec.error,
        durationMs: eventRec.durationMs,
        messageCount: Array.isArray(eventRec.messages) ? eventRec.messages.length : undefined,
        session,
        taskBoundary: true,
        taskId: sessionSummary.taskId,
        actionCount: state.actionCounter,
        sessionSummary,
        context: ctxRec,
      })
      .then((decision) => {
        const reflection = toRecord(decision.metadata.reflection);
        const lessonCandidates = toArray((decision.metadata as Record<string, unknown>).lesson_candidates ?? (decision.metadata as Record<string, unknown>).lessonCandidates);
        if (Object.keys(reflection).length > 0) {
          api.logger?.info?.("autonomy.reflection.generated", {
            sessionKey: key,
            taskId: reflection.task_id ?? reflection.taskId,
            summary: reflection.summary,
            lessonCandidates: lessonCandidates.length,
          });
        }
      })
      .catch((error) => api.logger?.warn?.("autonomy.agent_end.error", { error: String(error) }));
  });

  api.on("session_start", async (event, ctx) => {
    const eventRec = toRecord(event);
    const ctxRec = toRecord(ctx);
    const { key } = resolveSession(eventRec, ctxRec);
    const existing = sessionStates.get(key);
    if (existing) {
      existing.pendingConfirmations.clear();
      sessionStates.delete(key);
    }

    const runtimeMode = await resolveCurrentMode(client, config.mode, api.logger);

    void client
      .onHealthUpdate({
        status: "session_start",
        mode: runtimeMode,
        context: ctxRec,
      })
      .catch((error) => api.logger?.debug?.("autonomy.session_start.health.error", { error: String(error) }));
  });

  api.on("health_update", async (event, ctx) => {
    const eventRec = toRecord(event);
    const ctxRec = toRecord(ctx);
    const { key } = resolveSession(eventRec, ctxRec);
    const state = getSessionState(sessionStates, key);

    const runtimeMode = await resolveCurrentMode(client, config.mode, api.logger);

    await client
      .onHealthUpdate({
        source: "hook",
        mode: runtimeMode,
        health: normalizeHealth(firstRecord(eventRec.health, ctxRec.health)),
        context: ctxRec,
      })
      .then((healthDecision) => {
        const adaptive = toRecord(healthDecision.metadata.adaptive);
        const profile = asString(adaptive.profile);
        if (profile) {
          state.lastAdaptiveProfile = profile;
          api.logger?.info?.("autonomy.health_update.profile", {
            sessionKey: key,
            profile,
          });
        }
      })
      .catch((error) => api.logger?.warn?.("autonomy.health_update.error", { error: String(error) }));
  });

  api.on("session_end", async (event, ctx) => {
    const eventRec = toRecord(event);
    const ctxRec = toRecord(ctx);
    const { key } = resolveSession(eventRec, ctxRec);
    const existing = sessionStates.get(key);
    if (existing) {
      existing.pendingConfirmations.clear();
      sessionStates.delete(key);
    }
  });
}
