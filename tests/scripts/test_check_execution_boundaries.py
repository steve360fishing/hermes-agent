from __future__ import annotations

import importlib.util
import copy
import json
import subprocess
import sys
from copy import deepcopy
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
    discovered = int(result.stdout.split("discovered=", 1)[1].split()[0])
    assert discovered >= 225


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


def test_registry_rejects_duplicate_reverse_and_invalid_lifecycle_edges():
    checker = _load_checker()
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    first = data["lifecycle_relationships"][0]

    duplicate = deepcopy(data)
    duplicate["lifecycle_relationships"].append(dict(first))
    assert any("duplicate lifecycle edge" in error for error in checker.validate_registry(REPO_ROOT, duplicate))

    reverse = deepcopy(data)
    reverse["lifecycle_relationships"].append(
        {**first, "from": first["to"], "to": first["from"]}
    )
    assert any("reverse lifecycle edge" in error for error in checker.validate_registry(REPO_ROOT, reverse))

    invalid = deepcopy(data)
    invalid["lifecycle_relationships"][0] = {
        **first,
        "to": "agent/turn_finalizer.py:_finalize_turn_impl",
    }
    assert any("invalid lifecycle role transition" in error for error in checker.validate_registry(REPO_ROOT, invalid))


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


def test_discovery_denominator_is_the_union_of_entrypoints_roots_and_rule_paths(tmp_path, monkeypatch):
    checker = _load_checker()
    (tmp_path / "outside").mkdir()
    target = tmp_path / "outside" / "boundary.py"
    target.write_text("def boundary():\n    artifact_only = True\n", encoding="utf-8")
    rule = checker.DiscoveryRule("artifact_request", ("outside/boundary.py",), ("artifact_only",))
    monkeypatch.setattr(checker, "DISCOVERY_RULES", (rule,))
    monkeypatch.setattr(checker, "ENTRYPOINTS", ("cli.py",))
    monkeypatch.setattr(checker, "RUNTIME_ROOTS", ("agent",))

    sites = checker.discover_sites(tmp_path, ["cli.py", "agent"])

    assert [(site.path, site.symbol) for site in sites] == [("outside/boundary.py", "boundary")]


def test_discovery_finds_token_bearing_symbol_in_arbitrary_runtime_root_path(tmp_path):
    checker = _load_checker()
    (tmp_path / "agent").mkdir()
    target = tmp_path / "agent" / "new_boundary.py"
    target.write_text("def boundary():\n    artifact_only = True\n", encoding="utf-8")

    sites = checker.discover_sites(tmp_path, ["agent"])

    assert ("artifact_request", "agent/new_boundary.py", "boundary") in {
        (site.contract, site.path, site.symbol) for site in sites
    }


def test_lifecycle_relationships_must_match_registered_transition_graph():
    checker = _load_checker()
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert data.get("lifecycle_transition_graph"), "registry must declare its canonical transition graph"
    first = data["lifecycle_relationships"][0]

    unrelated = deepcopy(data)
    unrelated["lifecycle_relationships"][0] = {
        **first,
        "from": "agent/task_execution_contract.py:build_task_execution_contract",
    }
    assert any(
        "registered transition graph" in error
        for error in checker.validate_registry(REPO_ROOT, unrelated)
    )

    contradictory = deepcopy(data)
    contradictory["lifecycle_relationships"].append(
        {**first, "to": "agent/turn_finalizer.py:_finalize_turn_impl"}
    )
    errors = checker.validate_registry(REPO_ROOT, contradictory)
    assert any("contradictory lifecycle edge" in error for error in errors)

    wrong_type = deepcopy(data)
    wrong_type["lifecycle_relationships"][0] = {
        **first,
        "type": "enforcement_to_recovery",
    }
    assert any(
        "registered transition graph" in error
        for error in checker.validate_registry(REPO_ROOT, wrong_type)
    )


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


def test_discovery_rejects_unreadable_tracked_source(tmp_path, monkeypatch):
    checker = _load_checker()
    source = tmp_path / "agent.py"
    source.write_text("artifact_only = True\n", encoding="utf-8")
    monkeypatch.setattr(checker, "_tracked_python_paths", lambda *_: [source])
    original_read_text = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == source:
            raise OSError("access denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    report = checker.audit_repository(
        tmp_path,
        {"schema_version": 1, "entrypoints": list(checker.ENTRYPOINTS), "runtime_roots": list(checker.RUNTIME_ROOTS), "contracts": checker.REQUIRED_CONTRACTS, "sites": []},
    )

    assert any(error.startswith("unreadable tracked source: agent.py") for error in report.invalid)


def test_registry_loader_rejects_duplicate_json_keys(tmp_path):
    checker = _load_checker()
    registry = tmp_path / "registry.json"
    registry.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")

    try:
        checker._load_registry(registry)
    except ValueError as exc:
        assert "duplicate JSON key" in str(exc)
    else:
        raise AssertionError("duplicate registry keys must be rejected")


def test_manifest_rejects_missing_core_files_and_hashes_supplied_inputs(tmp_path):
    checker = _load_checker()
    registry = tmp_path / "registry.json"
    plugin_inventory = tmp_path / "plugins.json"
    transport_inventory = tmp_path / "transports.json"
    registry.write_text('{"registry": true}', encoding="utf-8")
    plugin_inventory.write_text('{"plugins": ["example"]}', encoding="utf-8")
    transport_inventory.write_text('{"transports": ["telegram"]}', encoding="utf-8")

    try:
        checker.build_runtime_manifest(
            repo_root=REPO_ROOT,
            environment_names=[],
            boundary_core_modules=["missing/core.py"],
            registry_path=registry,
            plugin_inventory_path=plugin_inventory,
            transport_inventory_path=transport_inventory,
        )
    except ValueError as exc:
        assert "missing boundary core module" in str(exc)
    else:
        raise AssertionError("missing declared core modules must fail the manifest")

    manifest = checker.build_runtime_manifest(
        repo_root=REPO_ROOT,
        environment_names=["HERMES_PROFILE"],
        boundary_core_modules=["cron/scheduler.py"],
        registry_path=registry,
        plugin_inventory_path=plugin_inventory,
        transport_inventory_path=transport_inventory,
    )

    assert manifest["boundary_core_hashes"]["cron/scheduler.py"]
    assert manifest["plugin_inventory"] == {"plugins": ["example"]}
    assert manifest["transport_inventory"] == {"transports": ["telegram"]}
    assert manifest["registry_hash"] == checker._sha256(registry)
    assert manifest["plugin_inventory_hash"] == checker._sha256(plugin_inventory)
    assert manifest["transport_inventory_hash"] == checker._sha256(transport_inventory)
    assert manifest["source_identity"]["git_tree"]


def test_manifest_inventory_schema_drops_secret_bearing_fields(tmp_path):
    checker = _load_checker()
    inventory = tmp_path / "plugins.json"
    inventory.write_text('{"plugins": ["example"], "token": "must-not-leak"}', encoding="utf-8")

    try:
        checker._load_inventory(inventory, "plugins")
    except ValueError as exc:
        assert "inventory schema" in str(exc)
    else:
        raise AssertionError("secret-bearing inventory fields must be rejected")
