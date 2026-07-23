"""Read-only helpers for optional tournament fact validation."""
from __future__ import annotations
import hashlib, json, os, stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
AUDIT_SCHEMA_VERSION = "tournament_route_preflight.v2"
AUDIT_SOURCE_REPOSITORY = "sportfishhub-audit"
AUDIT_CONTRACT_COMMIT = "8904a313dfae6cd364c34c0b247c1a71d9b5cc01"
@dataclass(frozen=True)
class RuntimeRoots:
    receipt_root: Path
    journal_root: Path
    source_snapshot_root: Path
def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
def configured_runtime_roots() -> RuntimeRoots | None:
    try:
        from hermes_cli.config import load_config_readonly
        config=(load_config_readonly() or {}).get("tournament_truth_gate", {})
        values=[config.get(k) for k in ("receipt_root","journal_root","source_snapshot_root")]
        if not all(isinstance(v,str) and v.strip() for v in values): return None
        roots=RuntimeRoots(*(Path(v).resolve(strict=True) for v in values))
        return roots if all(p.is_dir() for p in roots.__dict__.values()) else None
    except Exception: return None
def contained_path(root: Path, candidate: str | Path) -> Path | None:
    try:
        resolved=Path(candidate).resolve(strict=True); resolved.relative_to(root.resolve(strict=True)); return resolved
    except (OSError, ValueError): return None
def secure_read_contained_text(root: Path, candidate: str | Path) -> str | None:
    path=contained_path(root,candidate)
    if path is None: return None
    try:
        info=os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode): return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError): return None
def build_artifact_payload(candidate: str, destination: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload={k:v for k,v in metadata.items() if k not in {"content","destination"}}; payload["content"]=candidate; payload["destination"]=destination; return payload
