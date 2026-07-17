from __future__ import annotations

import importlib.util
import copy
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
    synthetic = tmp_path / "agent" / "task_execution_contract.py"
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
    assert "agent/task_execution_contract.py:inject_execution_contract" in report.unclassified


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


def test_registry_requires_lifecycle_relationships_between_registered_sites():
    checker = _load_checker()
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    broken = dict(data)
    broken["lifecycle_relationships"] = []

    errors = checker.validate_registry(REPO_ROOT, broken)

    assert any("lifecycle relationships" in error for error in errors)


def test_registry_requires_real_artifact_lifecycle_graph():
    checker = _load_checker()
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    graph = data.get("lifecycle_graphs", {}).get("artifact_delivery", {})
    nodes = {item.get("id") for item in graph.get("nodes", [])}
    required_nodes = {
        "writer",
        "verifier",
        "finalizer",
        "gateway_dispatch",
        "telegram_descriptor",
        "receipt_transition",
    }

    assert required_nodes <= nodes
    broken = copy.deepcopy(data)
    broken["lifecycle_graphs"]["artifact_delivery"]["edges"] = []
    errors = checker.validate_registry(REPO_ROOT, broken)
    assert any("artifact lifecycle" in error for error in errors)


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
    assert manifest["environment"]["OPENROUTER_API_KEY"] == {"present": True}
    assert manifest["environment"]["HERMES_PROFILE"] == {"present": True}


def test_discovery_covers_entrypoints_runtime_roots_and_module_scope(tmp_path):
    checker = _load_checker()
    (tmp_path / "agent").mkdir()
    (tmp_path / "cli.py").write_text("artifact_only = True\n", encoding="utf-8")
    (tmp_path / "agent" / "task_execution_contract.py").write_text(
        "def boundary():\n    artifact_only = True\n",
        encoding="utf-8",
    )

    sites = checker.discover_sites(tmp_path, ["cli.py", "agent"])

    assert {(site.path, site.symbol) for site in sites} == {
        ("cli.py", "__module__"),
        ("agent/task_execution_contract.py", "boundary"),
    }


def test_discovery_rejects_unparseable_and_unreadable_tracked_source(tmp_path, monkeypatch):
    checker = _load_checker()
    broken = tmp_path / "agent.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr(checker, "_tracked_python_paths", lambda *_: [broken])

    report = checker.audit_repository(
        tmp_path,
        {"schema_version": 1, "entrypoints": list(checker.ENTRYPOINTS), "runtime_roots": list(checker.RUNTIME_ROOTS), "contracts": checker.REQUIRED_CONTRACTS, "sites": []},
    )

    assert any(error.startswith("unparseable tracked source: agent.py") for error in report.invalid)


def test_manifest_hashes_boundary_core_and_explicit_inventories():
    checker = _load_checker()
    manifest = checker.build_runtime_manifest(
        repo_root=REPO_ROOT,
        environment_names=["HERMES_PROFILE"],
        boundary_core_modules=["cron/scheduler.py"],
        plugin_inventory={"plugins": ["example"]},
        transport_inventory={"transports": ["telegram"]},
    )

    assert manifest["boundary_core_hashes"]["cron/scheduler.py"]
    assert manifest["plugin_inventory"] == {"plugins": ["example"]}
    assert manifest["transport_inventory"] == {"transports": ["telegram"]}
    assert manifest["source_identity"]["git_tree"]
