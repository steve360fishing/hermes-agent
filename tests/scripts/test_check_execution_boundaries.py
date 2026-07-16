from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_execution_boundaries.py"
REGISTRY = REPO_ROOT / "docs" / "security" / "execution-boundary-registry.json"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_execution_boundary_checker", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_execution_boundary_registry_is_complete():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "unclassified=0" in result.stdout
    assert "invalid=0" in result.stdout


def test_unclassified_restriction_site_fails_closed(tmp_path):
    checker = _load_checker()
    (tmp_path / "agent").mkdir()
    synthetic = tmp_path / "agent" / "new_boundary.py"
    synthetic.write_text(
        "def inject_execution_contract():\n"
        "    artifact_only = True\n"
        "    return artifact_only\n",
        encoding="utf-8",
    )
    registry = {
        "schema_version": 1,
        "entrypoints": ["agent/new_boundary.py"],
        "runtime_roots": ["agent"],
        "contracts": checker.REQUIRED_CONTRACTS,
        "sites": [],
    }

    report = checker.audit_repository(tmp_path, registry)

    assert report.unclassified
    assert report.ok is False
    assert "agent/new_boundary.py:inject_execution_contract" in report.unclassified


def test_registry_rejects_missing_contract_roles():
    checker = _load_checker()
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    broken = dict(data)
    broken["sites"] = [
        site
        for site in data["sites"]
        if not (
            site["contract"] == "artifact_delivery"
            and site["role"] == "recovery"
        )
    ]

    errors = checker.validate_registry(REPO_ROOT, broken)

    assert any("artifact_delivery" in error and "recovery" in error for error in errors)


def test_registry_rejects_denominator_shrinkage_and_stale_ids():
    checker = _load_checker()
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    broken = dict(data)
    broken["entrypoints"] = data["entrypoints"][:-1]
    broken["runtime_roots"] = data["runtime_roots"][:-1]
    broken["sites"] = [dict(site) for site in data["sites"]]
    broken["sites"][0]["id"] = "stale:identifier"

    errors = checker.validate_registry(REPO_ROOT, broken)

    assert "entrypoints must match the shipped entrypoint denominator" in errors
    assert "runtime_roots must match the dynamic runtime denominator" in errors
    assert any("id must equal" in error for error in errors)


def test_runtime_manifest_redacts_secret_values(tmp_path, monkeypatch):
    checker = _load_checker()
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-leak")
    monkeypatch.setenv("HERMES_PROFILE", "default")

    manifest = checker.build_runtime_manifest(
        repo_root=REPO_ROOT,
        environment_names=["OPENROUTER_API_KEY", "HERMES_PROFILE"],
    )
    serialized = json.dumps(manifest, sort_keys=True)

    assert "must-not-leak" not in serialized
    assert manifest["environment"]["OPENROUTER_API_KEY"] == {
        "present": True,
        "secret_like": True,
    }
    assert manifest["environment"]["HERMES_PROFILE"]["value"] == "default"
