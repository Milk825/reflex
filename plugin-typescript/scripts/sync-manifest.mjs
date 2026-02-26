import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(process.cwd());
const srcManifest = resolve(root, "src/manifest.json");
const rootManifest = resolve(root, "openclaw.plugin.json");
const distDir = resolve(root, "dist");
const distManifest = resolve(distDir, "openclaw.plugin.json");

const raw = readFileSync(srcManifest, "utf8");
JSON.parse(raw); // validate JSON early

writeFileSync(rootManifest, raw + "\n", "utf8");

if (existsSync(distDir)) {
  mkdirSync(distDir, { recursive: true });
  writeFileSync(distManifest, raw + "\n", "utf8");
}

console.log("Synced plugin manifest -> openclaw.plugin.json");
