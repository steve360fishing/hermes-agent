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

REQUIRED_ARTIFACT_LIFECYCLE_NODES = {
    "writer": ("tools/file_tools.py", "write_file_tool"),
    "verifier": ("agent/task_execution_contract.py", "record_artifact_written"),
    "finalizer": ("agent/turn_finalizer.py", "_finalize_turn_impl"),
    "gateway_dispatch": (
        "gateway/platforms/base.py",
        "BasePlatformAdapter._process_message_background",
    ),
    "stream_gateway_dispatch": (
        "gateway/run.py",
        "GatewayRunner._deliver_media_from_response",
    ),
    "telegram_descriptor": (
        "plugins/platforms/telegram/adapter.py",
        "_open_verified_document_descriptor",
    ),
    "receipt_transition": (
        "agent/task_execution_contract.py",
        "record_artifact_dispatch",
    ),
}
REQUIRED_ARTIFACT_LIFECYCLE_EDGES = {
    ("writer", "verifier"),
    ("verifier", "finalizer"),
    ("finalizer", "gateway_dispatch"),
    ("finalizer", "stream_gateway_dispatch"),
    ("gateway_dispatch", "telegram_descriptor"),
    ("stream_gateway_dispatch", "telegram_descriptor"),
    ("gateway_dispatch", "receipt_transition"),
    ("stream_gateway_dispatch", "receipt_transition"),
    ("telegram_descriptor", "receipt_transition"),
}

LIFECYCLE_ROLE_TRANSITIONS: dict[str, set[tuple[str, str]]] = {
    "artifact_request": {("decision", "propagation"), ("propagation", "enforcement"), ("enforcement", "recovery")},
    "artifact_delivery": {("allocation", "validation"), ("validation", "delivery"), ("delivery", "recovery")},
    "safe_mode": {("decision", "enforcement"), ("enforcement", "lifecycle")},
    "incident_fallback": {("decision", "enforcement"), ("enforcement", "recovery")},
    "cron_restrictions": {("decision", "enforcement"), ("enforcement", "recovery")},
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
            "cli.py",
            "agent/task_execution_contract.py",
            "agent/turn_context.py",
            "agent/turn_finalizer.py",
            "agent/conversation_loop.py",
            "agent/tool_guardrails.py",
            "agent/tool_executor.py",
        ),
        (
            "artifact_only",
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
            "tools/file_tools.py",
        ),
        (
            "artifact_output_path",
            "validate_artifact_output_path",
            "validate_media_delivery_path",
            "media_delivery_safe_roots",
            "media_tag_cleanup_re",
            "send_document",
            "write_registered_artifact",
            "record_artifact_dispatch",
            "_open_verified_document_descriptor",
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
    def __init__(self, source: str, *, include_segments: bool = True) -> None:
        self.lines = source.splitlines(keepends=True) if include_segments else []
        self.include_segments = include_segments
        self.stack: list[str] = []
        self.symbols: list[tuple[str, ast.AST, str]] = []

    def _visit_symbol(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        symbol = ".".join(self.stack)
        segment = ""
        if self.include_segments:
            end_line = getattr(node, "end_lineno", node.lineno)
            segment = "".join(self.lines[node.lineno - 1 : end_line])
        self.symbols.append((symbol, node, segment))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node, node.name)


def _tracked_python_paths(repo_root: Path, scan_paths: Iterable[str]) -> list[Path]:
    roots = tuple(dict.fromkeys(scan_paths))
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
        if root.is_file() and root.suffix == ".py":
            paths.append(root)
        elif root.exists():
            paths.extend(root.rglob("*.py"))
    return paths


def _discover_sites_with_errors(repo_root: Path, scan_paths: Iterable[str]) -> tuple[list[Site], list[str]]:
    sites: dict[tuple[str, str], Site] = {}
    errors: list[str] = []
    all_tokens = tuple(token for rule in DISCOVERY_RULES for token in rule.tokens)
    for path in _tracked_python_paths(repo_root, scan_paths):
        relative = path.relative_to(repo_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"unreadable tracked source: {relative}: {exc}")
            continue
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            errors.append(f"unparseable tracked source: {relative}: {exc.msg}")
            continue
        lowered_source = source.lower()
        if not any(token in lowered_source for token in all_tokens):
            continue
        visitor = _SymbolVisitor(source)
        visitor.visit(tree)
        module_segment = "\n".join(
            ast.get_source_segment(source, node) or ""
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        )
        if module_segment:
            visitor.symbols.append(("__module__", tree, module_segment))
        for symbol, node, segment in visitor.symbols:
            lowered = segment.lower()
            for rule in DISCOVERY_RULES:
                if relative not in rule.paths:
                    continue
                if any(token in lowered for token in rule.tokens):
                    site = Site(rule.contract, relative, symbol, getattr(node, "lineno", 1))
                    sites[(rule.contract, site.key)] = site
    return sorted(sites.values(), key=lambda site: (site.contract, site.path, site.line, site.symbol)), errors


def _discovery_denominator(scan_paths: Iterable[str]) -> tuple[str, ...]:
    """Return the complete static denominator, including semantic rule paths."""
    return tuple(dict.fromkeys((*scan_paths, *(path for rule in DISCOVERY_RULES for path in rule.paths))))


def discover_sites(repo_root: Path, scan_paths: Iterable[str]) -> list[Site]:
    return _discover_sites_with_errors(repo_root, _discovery_denominator(scan_paths))[0]


def _all_symbols(repo_root: Path, relative_path: str) -> set[str]:
    path = repo_root / relative_path
    if path.suffix != ".py":
        return set()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
    except (OSError, SyntaxError, UnicodeError):
        return set()
    visitor = _SymbolVisitor(source, include_segments=False)
    visitor.visit(tree)
    return {"__module__", *(symbol for symbol, _node, _segment in visitor.symbols)}


def validate_registry(
    repo_root: Path,
    registry: dict[str, Any],
    *,
    discovered_sites: Iterable[Site] | None = None,
) -> list[str]:
    errors: list[str] = []
    symbol_cache: dict[str, set[str]] = {}
    if registry.get("schema_version") != 1:
        errors.append("registry schema_version must be 1")
    lifecycle = registry.get("lifecycle_graphs", {}).get("artifact_delivery", {})
    lifecycle_nodes = {
        str(node.get("id")): (node.get("path"), node.get("symbol"))
        for node in lifecycle.get("nodes", [])
        if isinstance(node, dict)
    }
    for node_id, expected in REQUIRED_ARTIFACT_LIFECYCLE_NODES.items():
        if lifecycle_nodes.get(node_id) != expected:
            errors.append(
                f"artifact lifecycle node {node_id!r} must equal {expected!r}"
            )
    lifecycle_edges = {
        (str(edge.get("from")), str(edge.get("to")))
        for edge in lifecycle.get("edges", [])
        if isinstance(edge, dict)
    }
    missing_edges = REQUIRED_ARTIFACT_LIFECYCLE_EDGES - lifecycle_edges
    if missing_edges:
        errors.append(
            f"artifact lifecycle missing edges {sorted(missing_edges)!r}"
        )
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
        available_symbols = symbol_cache.get(path, set())
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

    sites_by_contract_id = {
        (site.get("contract"), site.get("id")): site
        for site in registry.get("sites", [])
    }
    relationship_roles: dict[str, set[str]] = {}
    lifecycle_edges: dict[str, set[tuple[str, str]]] = {}
    relationships = registry.get("lifecycle_relationships")
    if not isinstance(relationships, list):
        errors.append("lifecycle relationships must be a list")
        relationships = []
    for index, relationship in enumerate(relationships):
        prefix = f"lifecycle_relationships[{index}]"
        if not isinstance(relationship, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        contract = relationship.get("contract")
        source = sites_by_contract_id.get((contract, relationship.get("from")))
        target = sites_by_contract_id.get((contract, relationship.get("to")))
        invariant = relationship.get("invariant")
        if contract not in REQUIRED_CONTRACTS:
            errors.append(f"{prefix}: unknown contract {contract!r}")
            continue
        if source is None or target is None:
            errors.append(f"{prefix}: endpoints must reference registered sites")
            continue
        if source.get("contract") != contract or target.get("contract") != contract:
            errors.append(f"{prefix}: endpoints must belong to {contract}")
            continue
        if not isinstance(invariant, str) or not invariant.strip():
            errors.append(f"{prefix}: invariant is required")
        source_id = str(relationship.get("from"))
        target_id = str(relationship.get("to"))
        edge = (source_id, target_id)
        contract_edges = lifecycle_edges.setdefault(contract, set())
        if edge in contract_edges:
            errors.append(f"{prefix}: duplicate lifecycle edge {source_id} -> {target_id}")
        if (target_id, source_id) in contract_edges:
            errors.append(f"{prefix}: reverse lifecycle edge {source_id} -> {target_id}")
        contract_edges.add(edge)
        role_transition = (str(source.get("role")), str(target.get("role")))
        if role_transition not in LIFECYCLE_ROLE_TRANSITIONS[contract]:
            errors.append(
                f"{prefix}: invalid lifecycle role transition {role_transition[0]} -> {role_transition[1]} for {contract}"
            )
        relationship_roles.setdefault(contract, set()).update(
            {str(source.get("role")), str(target.get("role"))}
        )
    for contract, required_roles in REQUIRED_CONTRACTS.items():
        missing = set(required_roles) - relationship_roles.get(contract, set())
        if missing:
            errors.append(f"{contract}: lifecycle relationships missing roles {sorted(missing)}")
    return errors


def audit_repository(repo_root: Path, registry: dict[str, Any]) -> AuditReport:
    scan_paths = _discovery_denominator(
        (*tuple(registry.get("entrypoints", ENTRYPOINTS)), *tuple(registry.get("runtime_roots", RUNTIME_ROOTS)))
    )
    discovered, discovery_errors = _discover_sites_with_errors(repo_root, scan_paths)
    invalid = discovery_errors + validate_registry(repo_root, registry, discovered_sites=discovered)
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _sanitize_inventory(data: Any, kind: str) -> dict[str, list[str]]:
    key = "plugins" if kind == "plugins" else "transports"
    if not isinstance(data, dict) or set(data) != {key}:
        raise ValueError(f"{kind} inventory schema must contain only {key!r}")
    names = data[key]
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{kind} inventory schema requires a list of non-empty strings")
    return {key: list(names)}


def build_runtime_manifest(
    *,
    repo_root: Path,
    environment_names: Iterable[str],
    boundary_core_modules: Iterable[str] = (),
    plugin_inventory: Any = None,
    transport_inventory: Any = None,
    registry_path: Path | None = None,
    plugin_inventory_path: Path | None = None,
    transport_inventory_path: Path | None = None,
) -> dict[str, Any]:
    registry_path = registry_path or (repo_root / REGISTRY_PATH)
    if not registry_path.is_file():
        raise ValueError(f"missing registry for manifest: {registry_path}")
    missing_core = [path for path in sorted(set(boundary_core_modules)) if not (repo_root / path).is_file()]
    if missing_core:
        raise ValueError(f"missing boundary core module(s): {', '.join(missing_core)}")
    missing_entrypoints = [path for path in ENTRYPOINTS if not (repo_root / path).is_file()]
    if missing_entrypoints:
        raise ValueError(f"missing entrypoint(s) for manifest: {', '.join(missing_entrypoints)}")
    if plugin_inventory_path is not None:
        plugin_inventory = _load_inventory(plugin_inventory_path, "plugins")
    elif plugin_inventory is not None:
        plugin_inventory = _sanitize_inventory(plugin_inventory, "plugins")
    if transport_inventory_path is not None:
        transport_inventory = _load_inventory(transport_inventory_path, "transports")
    elif transport_inventory is not None:
        transport_inventory = _sanitize_inventory(transport_inventory, "transports")
    environment: dict[str, dict[str, bool]] = {}
    for name in sorted(set(environment_names)):
        environment[name] = {"present": os.environ.get(name) is not None}
    return {
        "schema_version": 1,
        "source_identity": {
            "git_head": _git_value(repo_root, "rev-parse", "HEAD"),
            "git_tree": _git_value(repo_root, "rev-parse", "HEAD^{tree}"),
        },
        "entrypoint_hashes": {
            path: _sha256(repo_root / path)
            for path in ENTRYPOINTS
        },
        "boundary_core_hashes": {
            path: _sha256(repo_root / path)
            for path in sorted(set(boundary_core_modules))
        },
        "checker_hash": _sha256(Path(__file__)),
        "registry_hash": _sha256(registry_path),
        "plugin_inventory_hash": _sha256(plugin_inventory_path) if plugin_inventory_path is not None else None,
        "transport_inventory_hash": _sha256(transport_inventory_path) if transport_inventory_path is not None else None,
        "plugin_inventory": plugin_inventory,
        "transport_inventory": transport_inventory,
        "environment": environment,
    }


def _load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, dict):
        raise ValueError("registry root must be an object")
    return data


def _load_inventory(path: Path, kind: str) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return _sanitize_inventory(json.load(handle, object_pairs_hook=_reject_duplicate_keys), kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--plugin-inventory", type=Path)
    parser.add_argument("--transport-inventory", type=Path)
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
        if args.plugin_inventory is None or args.transport_inventory is None:
            print("execution-boundary audit error: --manifest-out requires explicit plugin and transport inventories", file=sys.stderr)
            return 2
        try:
            plugin_inventory = _load_inventory(args.plugin_inventory, "plugins")
            transport_inventory = _load_inventory(args.transport_inventory, "transports")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"execution-boundary audit error: {exc}", file=sys.stderr)
            return 2
        manifest = build_runtime_manifest(
            repo_root=repo_root,
            environment_names=registry.get("manifest_environment_names", []),
            boundary_core_modules=registry.get("boundary_core_modules", []),
            plugin_inventory=plugin_inventory,
            transport_inventory=transport_inventory,
            registry_path=registry_path,
            plugin_inventory_path=args.plugin_inventory,
            transport_inventory_path=args.transport_inventory,
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
