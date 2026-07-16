#!/usr/bin/env python3
"""Fail closed when capability-reducing runtime boundaries are unreviewed.

The registry is intentionally source-oriented. It records where Hermes makes,
propagates, enforces, clears, or delivers a reduced-capability decision. The
checker discovers matching functions from the runtime tree and rejects new
sites until they receive an explicit registry disposition.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REGISTRY_PATH = Path("docs/security/execution-boundary-registry.json")

REQUIRED_CONTRACTS: dict[str, list[str]] = {
    "artifact_request": ["decision", "propagation", "enforcement", "recovery"],
    "artifact_delivery": ["allocation", "validation", "delivery", "recovery"],
    "safe_mode": ["decision", "enforcement", "lifecycle"],
    "incident_fallback": ["decision", "enforcement", "recovery"],
    "cron_restrictions": ["decision", "enforcement", "recovery"],
}

ENTRYPOINTS = (
    "cli.py",
    "run_agent.py",
    "batch_runner.py",
    "mcp_serve.py",
    "gateway/run.py",
    "hermes_cli/main.py",
    "tui_gateway/server.py",
    "acp_adapter/server.py",
)

RUNTIME_ROOTS = (
    "agent",
    "gateway",
    "hermes_cli",
    "cron",
    "tools",
    "plugins",
    "providers",
    "tui_gateway",
    "acp_adapter",
    "acp_registry",
)


@dataclass(frozen=True)
class DiscoveryRule:
    contract: str
    paths: tuple[str, ...]
    tokens: tuple[str, ...]


DISCOVERY_RULES = (
    DiscoveryRule(
        "artifact_request",
        (
            "agent/task_execution_contract.py",
            "agent/turn_context.py",
            "agent/turn_finalizer.py",
            "agent/conversation_loop.py",
            "agent/tool_guardrails.py",
            "agent/tool_executor.py",
        ),
        (
            "artifact_only",
            "execution_contract",
            "_task_execution_contract",
            "set_execution_contract",
            "clear_task_execution_contract",
        ),
    ),
    DiscoveryRule(
        "artifact_delivery",
        (
            "agent/task_execution_contract.py",
            "agent/turn_finalizer.py",
            "gateway/platforms/base.py",
            "gateway/run.py",
            "plugins/platforms/telegram/adapter.py",
        ),
        (
            "artifact_output_path",
            "validate_artifact_output_path",
            "validate_media_delivery_path",
            "media_delivery_safe_roots",
            "media_tag_cleanup_re",
            "send_document",
        ),
    ),
    DiscoveryRule(
        "safe_mode",
        (
            "hermes_cli/main.py",
            "hermes_cli/plugins.py",
            "agent/shell_hooks.py",
            "tools/mcp_tool.py",
        ),
        ("hermes_safe_mode", "safe_mode", "_apply_safe_mode"),
    ),
    DiscoveryRule(
        "incident_fallback",
        (
            "agent/openrouter_fallback_guard.py",
            "agent/chat_completion_helpers.py",
            "agent/agent_runtime_helpers.py",
            "agent/conversation_loop.py",
            "gateway/run.py",
        ),
        ("openrouter_fallback", "try_activate_fallback"),
    ),
    DiscoveryRule(
        "cron_restrictions",
        (
            "cron/jobs.py",
            "cron/scheduler.py",
            "tools/cronjob_tools.py",
            "hermes_cli/web_server.py",
            "tui_gateway/server.py",
            "acp_adapter/session.py",
        ),
        ("provider_snapshot", "enabled_toolsets", "disabled_toolsets"),
    ),
)


@dataclass(frozen=True)
class Site:
    contract: str
    path: str
    symbol: str
    line: int

    @property
    def key(self) -> str:
        return f"{self.path}:{self.symbol}"


@dataclass
class AuditReport:
    discovered: list[Site]
    unclassified: list[str]
    invalid: list[str]

    @property
    def ok(self) -> bool:
        return not self.unclassified and not self.invalid


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.stack: list[str] = []
        self.symbols: list[tuple[str, ast.AST, str]] = []

    def _visit_symbol(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        symbol = ".".join(self.stack)
        segment = ast.get_source_segment(self.source, node) or ""
        self.symbols.append((symbol, node, segment))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node, node.name)


def _tracked_python_paths(repo_root: Path, runtime_roots: Iterable[str]) -> list[Path]:
    roots = tuple(runtime_roots)
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *roots],
        cwd=repo_root,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if result.returncode == 0:
        return [
            repo_root / item.decode("utf-8")
            for item in result.stdout.split(b"\0")
            if item.endswith(b".py")
        ]
    paths: list[Path] = []
    for root_name in roots:
        root = repo_root / root_name
        if root.exists():
            paths.extend(root.rglob("*.py"))
    return paths


def discover_sites(repo_root: Path, runtime_roots: Iterable[str]) -> list[Site]:
    sites: dict[tuple[str, str], Site] = {}
    all_tokens = tuple(token for rule in DISCOVERY_RULES for token in rule.tokens)
    for path in _tracked_python_paths(repo_root, runtime_roots):
        relative = path.relative_to(repo_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        lowered_source = source.lower()
        if not any(token in lowered_source for token in all_tokens):
            continue
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError:
            continue
        visitor = _SymbolVisitor(source)
        visitor.visit(tree)
        for symbol, node, segment in visitor.symbols:
            lowered = segment.lower()
            for rule in DISCOVERY_RULES:
                if any(token in lowered for token in rule.tokens):
                    site = Site(rule.contract, relative, symbol, node.lineno)
                    sites[(rule.contract, site.key)] = site
    return sorted(sites.values(), key=lambda site: (site.contract, site.path, site.line, site.symbol))


def _all_symbols(repo_root: Path, relative_path: str) -> set[str]:
    path = repo_root / relative_path
    if path.suffix != ".py":
        return set()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
    except (OSError, SyntaxError, UnicodeError):
        return set()
    visitor = _SymbolVisitor(source)
    visitor.visit(tree)
    return {symbol for symbol, _node, _segment in visitor.symbols}


def validate_registry(repo_root: Path, registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    symbol_cache: dict[str, set[str]] = {}
    if registry.get("schema_version") != 1:
        errors.append("registry schema_version must be 1")
    entrypoints = registry.get("entrypoints", [])
    runtime_roots = registry.get("runtime_roots", [])
    if entrypoints != list(ENTRYPOINTS):
        errors.append("entrypoints must match the shipped entrypoint denominator")
    if runtime_roots != list(RUNTIME_ROOTS):
        errors.append("runtime_roots must match the dynamic runtime denominator")
    for entrypoint in entrypoints:
        if not (repo_root / entrypoint).is_file():
            errors.append(f"missing entrypoint: {entrypoint}")
    for root in runtime_roots:
        if not (repo_root / root).is_dir():
            errors.append(f"missing runtime root: {root}")

    seen: set[tuple[str, str]] = set()
    roles: dict[str, set[str]] = {}
    for index, site in enumerate(registry.get("sites", [])):
        prefix = f"sites[{index}]"
        contract = site.get("contract")
        role = site.get("role")
        path = site.get("path")
        symbol = site.get("symbol")
        site_id = site.get("id")
        disposition = site.get("disposition")
        if contract not in REQUIRED_CONTRACTS:
            errors.append(f"{prefix}: unknown contract {contract!r}")
        if role not in {"decision", "propagation", "enforcement", "recovery", "allocation", "validation", "delivery", "lifecycle"}:
            errors.append(f"{prefix}: invalid role {role!r}")
        if disposition not in {"covered", "reviewed_exclusion"}:
            errors.append(f"{prefix}: invalid disposition {disposition!r}")
        if not isinstance(site.get("rationale"), str) or not site["rationale"].strip():
            errors.append(f"{prefix}: rationale is required")
        if not isinstance(path, str) or not (repo_root / path).is_file():
            errors.append(f"{prefix}: missing path {path!r}")
            continue
        if path not in symbol_cache:
            symbol_cache[path] = _all_symbols(repo_root, path)
        available_symbols = symbol_cache[path]
        if not isinstance(symbol, str) or symbol not in available_symbols:
            errors.append(f"{prefix}: missing symbol {path}:{symbol}")
        expected_id = f"{path}:{symbol}"
        if site_id != expected_id:
            errors.append(f"{prefix}: id must equal {expected_id!r}")
        key = (str(contract), f"{path}:{symbol}")
        if key in seen:
            errors.append(f"{prefix}: duplicate site {key[1]} for {contract}")
        seen.add(key)
        roles.setdefault(str(contract), set()).add(str(role))

    required = registry.get("contracts")
    if required != REQUIRED_CONTRACTS:
        errors.append("contracts must match REQUIRED_CONTRACTS")
    for contract, required_roles in REQUIRED_CONTRACTS.items():
        missing = set(required_roles) - roles.get(contract, set())
        if missing:
            errors.append(f"{contract}: missing required roles {sorted(missing)}")
    return errors


def audit_repository(repo_root: Path, registry: dict[str, Any]) -> AuditReport:
    invalid = validate_registry(repo_root, registry)
    discovered = discover_sites(repo_root, registry.get("runtime_roots", RUNTIME_ROOTS))
    classified = {
        (site.get("contract"), f"{site.get('path')}:{site.get('symbol')}")
        for site in registry.get("sites", [])
    }
    unclassified = [
        site.key
        for site in discovered
        if (site.contract, site.key) not in classified
    ]
    return AuditReport(discovered, sorted(set(unclassified)), invalid)


def _git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _secret_like(name: str) -> bool:
    upper = name.upper()
    return any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE"))


def build_runtime_manifest(*, repo_root: Path, environment_names: Iterable[str]) -> dict[str, Any]:
    environment: dict[str, dict[str, Any]] = {}
    for name in sorted(set(environment_names)):
        value = os.environ.get(name)
        secret_like = _secret_like(name)
        record: dict[str, Any] = {"present": value is not None, "secret_like": secret_like}
        if value is not None and not secret_like:
            record["value"] = value
        environment[name] = record
    return {
        "schema_version": 1,
        "git_head": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_tree": _git_value(repo_root, "rev-parse", "HEAD^{tree}"),
        "entrypoint_hashes": {
            path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
            for path in ENTRYPOINTS
            if (repo_root / path).is_file()
        },
        "environment": environment,
    }


def _load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("registry root must be an object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.root.resolve()
    registry_path = (args.registry or (repo_root / REGISTRY_PATH)).resolve()
    try:
        registry = _load_registry(registry_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"execution-boundary audit error: {exc}", file=sys.stderr)
        return 2
    report = audit_repository(repo_root, registry)
    if args.manifest_out:
        manifest = build_runtime_manifest(
            repo_root=repo_root,
            environment_names=registry.get("manifest_environment_names", []),
        )
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for item in report.invalid:
        print(f"INVALID {item}")
    for item in report.unclassified:
        print(f"UNCLASSIFIED {item}")
    print(
        "execution-boundary audit: "
        f"discovered={len(report.discovered)} "
        f"unclassified={len(report.unclassified)} invalid={len(report.invalid)}"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
