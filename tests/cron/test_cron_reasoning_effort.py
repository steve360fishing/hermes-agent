"""Per-job cron reasoning-effort persistence, tool, and scheduler contracts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    """Redirect the JSON cron store without reloading shared modules."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
    return cron_dir


def test_create_persists_canonical_reasoning_effort(cron_store):
    from cron.jobs import create_job, load_jobs

    job = create_job(
        prompt="Analyze the report",
        schedule="every 1h",
        reasoning_effort=" MAX ",
    )

    assert job["reasoning_effort"] == "max"
    assert load_jobs()[0]["reasoning_effort"] == "max"


def test_create_omits_unset_reasoning_effort_for_legacy_compatibility(cron_store):
    from cron.jobs import create_job, load_jobs

    job = create_job(prompt="Legacy-compatible", schedule="every 1h")

    assert "reasoning_effort" not in job
    assert "reasoning_effort" not in load_jobs()[0]


def test_create_rejects_unknown_reasoning_effort(cron_store):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="Invalid reasoning_effort"):
        create_job(
            prompt="Analyze",
            schedule="every 1h",
            reasoning_effort="turbo",
        )


def test_load_normalizes_valid_hand_edited_reasoning_effort(cron_store):
    from cron.jobs import get_job, load_jobs, save_jobs

    save_jobs([
        {
            "id": "legacy-effort",
            "name": "legacy effort",
            "prompt": "Analyze",
            "reasoning_effort": " HIGH ",
            "schedule": {
                "kind": "interval",
                "minutes": 60,
                "display": "every 60m",
            },
            "schedule_display": "every 60m",
            "enabled": True,
        }
    ])

    assert load_jobs()[0]["reasoning_effort"] == "high"
    assert get_job("legacy-effort")["reasoning_effort"] == "high"


def test_load_legacy_job_without_reasoning_effort_is_unchanged(cron_store):
    from cron.jobs import get_job, save_jobs

    save_jobs([
        {
            "id": "legacy-no-effort",
            "name": "legacy",
            "prompt": "Analyze",
            "schedule": {
                "kind": "interval",
                "minutes": 60,
                "display": "every 60m",
            },
            "schedule_display": "every 60m",
            "enabled": True,
        }
    ])

    assert "reasoning_effort" not in get_job("legacy-no-effort")


def test_update_sets_and_clears_reasoning_effort(cron_store):
    from cron.jobs import create_job, get_job, load_jobs, update_job

    job = create_job(prompt="Analyze", schedule="every 1h")

    updated = update_job(job["id"], {"reasoning_effort": " ULTRA "})
    assert updated["reasoning_effort"] == "ultra"
    assert get_job(job["id"])["reasoning_effort"] == "ultra"

    cleared = update_job(job["id"], {"reasoning_effort": ""})
    assert "reasoning_effort" not in cleared
    assert "reasoning_effort" not in load_jobs()[0]


def test_update_rejects_invalid_reasoning_effort_without_mutating(cron_store):
    from cron.jobs import create_job, get_job, update_job

    job = create_job(
        prompt="Analyze",
        schedule="every 1h",
        reasoning_effort="medium",
    )

    with pytest.raises(ValueError, match="Invalid reasoning_effort"):
        update_job(job["id"], {"reasoning_effort": "turbo"})

    assert get_job(job["id"])["reasoning_effort"] == "medium"


def test_cronjob_tool_create_update_and_format_reasoning_effort(cron_store):
    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(
            action="create",
            prompt="Analyze",
            schedule="every 1h",
            reasoning_effort="max",
        )
    )
    assert created["success"] is True
    assert created["job"]["reasoning_effort"] == "max"

    updated = json.loads(
        cronjob(
            action="update",
            job_id=created["job_id"],
            reasoning_effort="low",
        )
    )
    assert updated["success"] is True
    assert updated["job"]["reasoning_effort"] == "low"

    cleared = json.loads(
        cronjob(
            action="update",
            job_id=created["job_id"],
            reasoning_effort="",
        )
    )
    assert cleared["success"] is True
    assert "reasoning_effort" not in cleared["job"]


def test_cronjob_schema_exposes_canonical_reasoning_efforts_including_max():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    prop = CRONJOB_SCHEMA["parameters"]["properties"]["reasoning_effort"]
    assert prop["type"] == "string"
    assert "max" in prop["enum"]
    assert set(prop["enum"]) == {
        "",
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    }
    assert "no_agent" in prop["description"]


def test_registered_cronjob_handler_forwards_reasoning_effort():
    from tools import cronjob_tools
    from tools.registry import registry

    entry = registry.get_entry("cronjob")
    assert entry is not None

    with patch.object(
        cronjob_tools, "cronjob", return_value='{"success": true}'
    ) as call:
        result = entry.handler(
            {"action": "list", "reasoning_effort": "max"},
            task_id="test-task",
        )

    assert json.loads(result)["success"] is True
    assert call.call_args.kwargs["reasoning_effort"] == "max"


def test_scheduler_job_effort_overrides_global_and_accepts_max(tmp_path, monkeypatch):
    from cron import scheduler

    (tmp_path / "config.yaml").write_text(
        "agent:\n  reasoning_effort: low\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    job = {
        "id": "reasoning-override",
        "name": "reasoning override",
        "prompt": "Analyze",
        "reasoning_effort": "max",
    }
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "ok"}

    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=MagicMock()),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        ),
        patch("run_agent.AIAgent", return_value=agent) as agent_cls,
    ):
        success, _doc, _final, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    assert agent_cls.call_args.kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "max",
    }


def test_scheduler_without_job_effort_uses_global(tmp_path, monkeypatch):
    from cron import scheduler

    (tmp_path / "config.yaml").write_text(
        "agent:\n  reasoning_effort: high\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    job = {"id": "global-effort", "name": "global effort", "prompt": "Analyze"}
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "ok"}

    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=MagicMock()),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            },
        ),
        patch("run_agent.AIAgent", return_value=agent) as agent_cls,
    ):
        success, _doc, _final, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    assert agent_cls.call_args.kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "high",
    }


def test_scheduler_invalid_persisted_effort_fails_before_inference(
    tmp_path, monkeypatch
):
    from cron import scheduler

    (tmp_path / "config.yaml").write_text(
        "agent:\n  reasoning_effort: low\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    job = {
        "id": "invalid-effort",
        "name": "invalid effort",
        "prompt": "Analyze",
        "reasoning_effort": "turbo",
    }

    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=MagicMock()),
        patch("run_agent.AIAgent") as agent_cls,
    ):
        success, _doc, _final, error = scheduler.run_job(job)

    assert success is False
    assert "Invalid reasoning_effort" in error
    agent_cls.assert_not_called()


def test_no_agent_job_does_not_parse_reasoning_or_construct_agent(tmp_path):
    from cron import scheduler

    job = {
        "id": "model-free",
        "name": "model free",
        "no_agent": True,
        "script": "watch.py",
        "reasoning_effort": "max",
    }

    with (
        patch("cron.scheduler._run_job_script", return_value=(True, "healthy")),
        patch("hermes_constants.parse_reasoning_effort") as parse_effort,
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as resolve_provider,
        patch("run_agent.AIAgent") as agent_cls,
    ):
        success, _doc, final, error = scheduler.run_job(job)

    assert success is True
    assert final == "healthy"
    assert error is None
    parse_effort.assert_not_called()
    resolve_provider.assert_not_called()
    agent_cls.assert_not_called()
