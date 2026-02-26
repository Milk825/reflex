from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

JSONDict = Dict[str, Any]


@dataclass(frozen=True)
class PreflightWarning:
    code: str
    severity: str
    message: str
    details: JSONDict = field(default_factory=dict)

    def as_dict(self) -> JSONDict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class PreflightAssessment:
    warnings: List[PreflightWarning]
    risk: float
    risk_band: str
    has_protected_file: bool
    has_destructive_pattern: bool
    critical: bool

    def as_dict(self) -> JSONDict:
        return {
            "risk": self.risk,
            "riskBand": self.risk_band,
            "critical": self.critical,
            "hasProtectedFile": self.has_protected_file,
            "hasDestructivePattern": self.has_destructive_pattern,
            "warnings": [warning.as_dict() for warning in self.warnings],
        }


class PreflightChecker:
    """Fast preflight guardrails used by autonomy before-action scoring."""

    _PROTECTED_BASENAMES = {
        "MEMORY.md",
        "SOUL.md",
        "IDENTITY.md",
        "USER.md",
        "AGENTS.md",
        "TOOLS.md",
        "config.yaml",
        "config.json",
        ".env",
        "secrets.yaml",
    }

    _DESTRUCTIVE_PATTERNS = {
        "rm_rf": re.compile(r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\b", re.IGNORECASE),
        "mkfs": re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b", re.IGNORECASE),
        "dd_dev": re.compile(r"\bdd\b[^\n;]*\bof\s*=\s*/dev/", re.IGNORECASE),
        "shutdown": re.compile(r"\b(?:shutdown|reboot|poweroff|halt)\b", re.IGNORECASE),
        "fork_bomb": re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\};:", re.IGNORECASE),
        "chmod_root": re.compile(r"\bchmod\s+-R\s+777\s+/(?:\s|$)", re.IGNORECASE),
    }

    _MODIFYING_TOOLS = {"write", "edit", "exec", "bash", "shell", "python"}

    _SEVERITY_TO_RISK = {
        "none": 0.0,
        "low": 0.20,
        "medium": 0.45,
        "high": 0.70,
        "critical": 0.95,
    }

    def check_before_action(self, tool_name: str, params: JSONDict, *, context: JSONDict | None = None) -> PreflightAssessment:
        normalized_tool = str(tool_name or "").strip().lower()
        params = params if isinstance(params, dict) else {}
        context = context if isinstance(context, dict) else {}

        warnings: List[PreflightWarning] = []

        file_paths = list(self._extract_paths(params))
        protected_matches = self._find_protected_files(file_paths)
        if protected_matches and normalized_tool in self._MODIFYING_TOOLS:
            for file_path in protected_matches:
                warnings.append(
                    PreflightWarning(
                        code="protected_file",
                        severity="high",
                        message=f"Operation touches protected file: {file_path}",
                        details={"path": file_path, "tool": normalized_tool},
                    )
                )

        command_blob = self._extract_command_blob(normalized_tool, params)
        matched_destructive = self._find_destructive_patterns(command_blob)
        for pattern_name in matched_destructive:
            severity = "critical" if pattern_name in {"rm_rf", "mkfs", "dd_dev", "fork_bomb"} else "high"
            warnings.append(
                PreflightWarning(
                    code=f"destructive_{pattern_name}",
                    severity=severity,
                    message=f"Potentially destructive command pattern detected ({pattern_name})",
                    details={"tool": normalized_tool},
                )
            )

        command_protected_refs = self._find_protected_basename_refs(command_blob)
        for basename in command_protected_refs:
            warnings.append(
                PreflightWarning(
                    code="protected_file",
                    severity="high",
                    message=f"Command references protected file: {basename}",
                    details={"path": basename, "tool": normalized_tool, "source": "command"},
                )
            )

        # Context-aware warning: destructive command references workspace root directly.
        runtime = context.get("runtime") if isinstance(context.get("runtime"), dict) else {}
        workspace_root = str(
            context.get("workspaceRoot")
            or context.get("workspace")
            or runtime.get("workspaceRoot")
            or ""
        ).strip()
        if command_blob and workspace_root and workspace_root in command_blob and matched_destructive:
            warnings.append(
                PreflightWarning(
                    code="workspace_destructive_reference",
                    severity="critical",
                    message="Destructive command references workspace root",
                    details={"workspaceRoot": workspace_root},
                )
            )

        has_protected_file = any(w.code == "protected_file" for w in warnings)
        has_destructive_pattern = any(w.code.startswith("destructive_") or w.code == "workspace_destructive_reference" for w in warnings)
        critical = has_protected_file and has_destructive_pattern

        risk = self._compute_risk(warnings, critical=critical)
        risk_band = self._risk_band(risk)

        return PreflightAssessment(
            warnings=warnings,
            risk=risk,
            risk_band=risk_band,
            has_protected_file=has_protected_file,
            has_destructive_pattern=has_destructive_pattern,
            critical=critical,
        )

    def _extract_paths(self, params: JSONDict) -> Iterable[str]:
        path_like_keys = {
            "path",
            "file_path",
            "filePath",
            "oldPath",
            "newPath",
            "targetPath",
            "cwd",
        }

        def _walk(value: Any, *, key: str | None = None) -> Iterable[str]:
            if isinstance(value, dict):
                for inner_key, inner_val in value.items():
                    yield from _walk(inner_val, key=str(inner_key))
                return

            if isinstance(value, list):
                for item in value:
                    yield from _walk(item, key=key)
                return

            if isinstance(value, str):
                if key in path_like_keys:
                    yield value
                    return

                if "/" in value or value.endswith(".md") or value.startswith("~"):
                    # Opportunistic path detection for arbitrary payload keys.
                    yield value

        yield from _walk(params)

    def _find_protected_files(self, paths: Iterable[str]) -> List[str]:
        matches: List[str] = []
        seen: set[str] = set()

        for raw_path in paths:
            if not raw_path:
                continue

            expanded = str(Path(raw_path).expanduser())
            basename = Path(expanded).name

            is_protected = basename in self._PROTECTED_BASENAMES
            is_memory_journal = "/memory/" in expanded.replace("\\", "/") and basename.endswith(".md")

            if is_protected or is_memory_journal:
                canonical = expanded
                if canonical not in seen:
                    seen.add(canonical)
                    matches.append(canonical)

        return matches

    def _extract_command_blob(self, tool_name: str, params: JSONDict) -> str:
        if tool_name not in {"exec", "bash", "shell", "python"}:
            return ""

        command_candidates: List[str] = []
        for key in ("command", "cmd", "script", "invokeCommand"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                command_candidates.append(value)
            elif isinstance(value, list):
                command_candidates.append(" ".join(str(v) for v in value))

        if not command_candidates:
            try:
                return json.dumps(params, ensure_ascii=False).lower()
            except Exception:
                return ""

        return "\n".join(command_candidates).lower()

    def _find_destructive_patterns(self, command_blob: str) -> List[str]:
        if not command_blob:
            return []

        matches: List[str] = []
        for name, pattern in self._DESTRUCTIVE_PATTERNS.items():
            if pattern.search(command_blob):
                matches.append(name)
        return matches

    def _find_protected_basename_refs(self, command_blob: str) -> List[str]:
        if not command_blob:
            return []

        lower_blob = command_blob.lower()
        matches: List[str] = []
        for basename in sorted(self._PROTECTED_BASENAMES):
            if basename.lower() in lower_blob:
                matches.append(basename)

        if "memory/" in lower_blob and any(ext in lower_blob for ext in (".md", ".markdown")):
            matches.append("memory/*.md")

        deduped: List[str] = []
        seen: set[str] = set()
        for item in matches:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _compute_risk(self, warnings: List[PreflightWarning], *, critical: bool) -> float:
        if not warnings:
            return 0.0

        risk = 0.0
        for warning in warnings:
            risk = max(risk, self._SEVERITY_TO_RISK.get(warning.severity, 0.0))

        # Multiple independent warnings increase uncertainty/risk.
        risk = min(0.99, risk + max(0, len(warnings) - 1) * 0.05)

        if critical:
            risk = max(risk, 0.99)

        return risk

    def _risk_band(self, risk: float) -> str:
        if risk >= 0.90:
            return "critical"
        if risk >= 0.70:
            return "high"
        if risk >= 0.45:
            return "medium"
        if risk > 0:
            return "low"
        return "none"
