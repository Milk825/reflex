import { BridgeClient } from "./api.js";
import { registerAutonomyHooks } from "./hooks.js";
import { resolvePluginConfig } from "./config.js";

type LoggerLike = {
  info?: (message: string, meta?: unknown) => void;
  warn?: (message: string, meta?: unknown) => void;
  error?: (message: string, meta?: unknown) => void;
  debug?: (message: string, meta?: unknown) => void;
};

type PluginApiLike = {
  id: string;
  pluginConfig?: unknown;
  logger?: LoggerLike;
  on: (hookName: string, handler: (event: any, ctx: any) => Promise<any> | any, opts?: { priority?: number }) => void;
};

const autonomyPlugin = {
  id: "autonomy",
  name: "Autonomy Bridge",
  description: "Bridge OpenClaw hooks to Python autonomy engine over UDS",
  version: "0.1.0",
  async register(api: PluginApiLike) {
    const config = resolvePluginConfig(api.pluginConfig);
    const client = new BridgeClient(config);

    registerAutonomyHooks({ api, client, config });

    const health = await client.health(config.bridge.requestTimeoutMs);
    if (!health.ok) {
      api.logger?.warn?.("autonomy.bridge.health.degraded", {
        socketPath: config.bridge.socketPath,
        failOpen: config.failOpen,
        details: health.details,
      });
      return;
    }

    api.logger?.info?.("autonomy.bridge.ready", {
      socketPath: config.bridge.socketPath,
      elapsedMs: health.elapsedMs,
      mode: config.mode,
    });
  },
};

export default autonomyPlugin;
