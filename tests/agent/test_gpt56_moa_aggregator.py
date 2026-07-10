from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import moa_loop


def _enabled_config() -> dict:
    return {
        "delegation": {
            "gpt56_routing": {
                "enabled": True,
                "contract": "gpt56-routing-v3",
                "provider": "openai-codex",
                "max_children": 3,
                "max_depth": 1,
            }
        },
        "moa": {
            "default_preset": "review",
            "presets": {
                "review": {
                    "reference_models": [],
                    "aggregator": {
                        "provider": "openrouter",
                        "model": "stale-aggregator-model",
                    },
                }
            },
        },
    }


def _disabled_config() -> dict:
    config = _enabled_config()
    config["delegation"]["gpt56_routing"] = {"enabled": False}
    return config


def _response(text: str = "canonical synthesis") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=None,
    )


def _stale_runtime(_slot: dict) -> dict:
    return {
        "provider": "openrouter",
        "model": "stale-aggregator-model",
        "base_url": "https://stale.example/v1",
        "api_key": "stale-secret-must-not-leak",
        "api_mode": "chat_completions",
    }


def test_one_shot_aggregator_binds_canonical_protected_route(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("hermes_cli.config.load_config", _enabled_config)
    monkeypatch.setattr(moa_loop, "_slot_runtime", _stale_runtime)
    monkeypatch.setattr(
        moa_loop,
        "_run_references_parallel",
        lambda *_args, **_kwargs: [("advisor", "useful advice", None)],
    )

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response()

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    output = moa_loop.aggregate_moa_context(
        user_prompt="finish",
        api_messages=[{"role": "user", "content": "finish"}],
        reference_models=[{"provider": "openrouter", "model": "advisor"}],
        aggregator={"provider": "openrouter", "model": "stale-aggregator-model"},
    )

    decision = captured["_route_decision"]
    assert decision.provider == "openai-codex"
    assert decision.model == "gpt-5.6-sol"
    assert decision.api_mode == "codex_responses"
    assert captured.get("provider") is None
    assert captured.get("base_url") is None
    assert "canonical synthesis" in output


def test_one_shot_protected_aggregator_failure_does_not_return_raw_references(
    monkeypatch,
) -> None:
    monkeypatch.setattr("hermes_cli.config.load_config", _enabled_config)
    monkeypatch.setattr(moa_loop, "_slot_runtime", _stale_runtime)
    monkeypatch.setattr(
        moa_loop,
        "_run_references_parallel",
        lambda *_args, **_kwargs: [("advisor", "raw reference text", None)],
    )
    monkeypatch.setattr(
        moa_loop,
        "call_llm",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("aggregator down")),
    )

    with pytest.raises(RuntimeError, match="protected.*aggregator"):
        moa_loop.aggregate_moa_context(
            user_prompt="finish",
            api_messages=[{"role": "user", "content": "finish"}],
            reference_models=[{"provider": "openrouter", "model": "advisor"}],
            aggregator={"provider": "openrouter", "model": "stale-aggregator-model"},
        )


def test_persistent_aggregator_binds_canonical_protected_route(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("hermes_cli.config.load_config", _enabled_config)
    monkeypatch.setattr(moa_loop, "_slot_runtime", _stale_runtime)
    monkeypatch.setattr(moa_loop, "_run_references_parallel", lambda *_a, **_k: [])

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _response("acted")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    response = moa_loop.MoAChatCompletions("review").create(
        messages=[{"role": "user", "content": "finish"}],
        tools=[],
    )

    assert response.choices[0].message.content == "acted"
    decision = captured["_route_decision"]
    assert decision.provider == "openai-codex"
    assert decision.model == "gpt-5.6-sol"
    assert decision.policy_spec.protected is True
    assert captured.get("api_key") is None


def test_disabled_policy_preserves_one_shot_joined_reference_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr("hermes_cli.config.load_config", _disabled_config)
    monkeypatch.setattr(moa_loop, "_slot_runtime", _stale_runtime)
    monkeypatch.setattr(
        moa_loop,
        "_run_references_parallel",
        lambda *_args, **_kwargs: [("advisor", "legacy joined fallback", None)],
    )
    monkeypatch.setattr(
        moa_loop,
        "call_llm",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("legacy down")),
    )

    output = moa_loop.aggregate_moa_context(
        user_prompt="finish",
        api_messages=[{"role": "user", "content": "finish"}],
        reference_models=[{"provider": "openrouter", "model": "advisor"}],
        aggregator={"provider": "openrouter", "model": "stale-aggregator-model"},
    )

    assert "legacy joined fallback" in output


def test_protected_aggregator_cache_shaping_uses_bound_codex_runtime(
    monkeypatch,
) -> None:
    shaped_runtimes = []
    monkeypatch.setattr("hermes_cli.config.load_config", _enabled_config)
    monkeypatch.setattr(moa_loop, "_slot_runtime", _stale_runtime)
    monkeypatch.setattr(
        moa_loop,
        "_run_references_parallel",
        lambda *_args, **_kwargs: [("advisor", "useful advice", None)],
    )

    def capture_cache_runtime(messages, runtime):
        shaped_runtimes.append(dict(runtime))
        return messages

    monkeypatch.setattr(
        moa_loop,
        "_maybe_apply_moa_cache_control",
        capture_cache_runtime,
    )
    monkeypatch.setattr(moa_loop, "call_llm", lambda **_kwargs: _response())

    moa_loop.aggregate_moa_context(
        user_prompt="finish",
        api_messages=[{"role": "user", "content": "finish"}],
        reference_models=[{"provider": "openrouter", "model": "advisor"}],
        aggregator={"provider": "openrouter", "model": "stale-aggregator-model"},
    )

    assert shaped_runtimes == [
        {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "base_url": None,
            "api_key": None,
            "api_mode": "codex_responses",
        }
    ]
