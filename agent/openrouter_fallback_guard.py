from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRIMARY_ROUTE_PROVIDER = "openai-codex"
PRIMARY_ROUTE_MODEL = "gpt-5.6-sol"
OPENROUTER_FALLBACK_PROVIDER = "openrouter"
OPENROUTER_FALLBACK_MODEL = "minimax/minimax-m3"
SECONDARY_FALLBACK_PROVIDER = "openai-api"
SECONDARY_FALLBACK_MODEL = "gpt-5.6-luna"
OPENROUTER_FALLBACK_NOTICE = (
    "🚨⚠️ FALLBACK ACTIVE: MiniMax M3 through OpenRouter.\n"
    "GPT-5.6 Sol through the subscription is currently unavailable. ⚠️🚨"
)
SECONDARY_FALLBACK_NOTICE = (
    "🚨⚠️ SECONDARY FALLBACK ACTIVE: GPT-5.6 Luna through the OpenAI API.\n"
    "GPT-5.6 Sol and MiniMax M3 are currently unavailable. ⚠️🚨"
)
PRIMARY_ROUTE_RESTORED_NOTICE = (
    "PRIMARY ROUTE RESTORED: gpt-5.6-sol · openai-codex"
)
CONTINUITY_FALLBACK_ROUTES = (
    (OPENROUTER_FALLBACK_PROVIDER, OPENROUTER_FALLBACK_MODEL),
    (SECONDARY_FALLBACK_PROVIDER, SECONDARY_FALLBACK_MODEL),
)
_OPENROUTER_GPT55_MODELS = {"openai/gpt-5.5", "openai/gpt-5.5-pro"}


def openrouter_fallback_activation_allowed(
    agent: Any,
    provider: str,
    model: str,
    *,
    reason: Any = None,
) -> tuple[bool, str]:
    provider_norm = _norm(provider)
    model_norm = _norm(model)
    if provider_norm == "openrouter" and model_norm in _OPENROUTER_GPT55_MODELS:
        return (
            False,
            "OpenRouter GPT-5.5 is explicit-only and cannot be used as "
            "automatic fallback.",
        )
    if not _is_gpt56_subscription_primary(agent):
        return True, ""

    tier = continuity_fallback_tier(provider_norm, model_norm)
    if tier is None:
        return (
            False,
            "Automatic continuity fallback rejected unexpected route "
            f"{provider or '<empty>'}/{model or '<empty>'}; only "
            "openrouter/minimax/minimax-m3 followed by "
            "openai-api/gpt-5.6-luna is allowed.",
        )
    chain_error = _protected_chain_error(agent)
    if chain_error:
        return False, chain_error
    fallback_index = getattr(agent, "_fallback_index", None)
    if isinstance(fallback_index, int) and fallback_index != tier:
        return (
            False,
            f"Automatic continuity fallback route order violation: tier {tier} "
            f"was evaluated at chain position {fallback_index}.",
        )
    reason_norm = _norm(getattr(reason, "value", reason))
    if reason_norm == "timeout":
        return (
            False,
            "Automatic continuity fallback is blocked for transport timeout.",
        )
    cap_message = fallback_cap_message_if_exhausted(
        agent, provider=provider_norm, model=model
    )
    if cap_message:
        return False, cap_message
    return True, ""


def record_openrouter_fallback_activation(
    agent: Any, *, reason: str | None = None
) -> dict[str, Any]:
    if not _is_gpt56_subscription_primary(agent):
        return {}
    provider = _norm(getattr(agent, "provider", ""))
    model = str(getattr(agent, "model", "") or "").strip()
    tier = continuity_fallback_tier(provider, model)
    if tier is None:
        return {}
    setattr(agent, "_openrouter_fallback_notice_required", True)
    setattr(agent, "_openrouter_fallback_model", model)
    setattr(agent, "_continuity_fallback_tier", tier)
    if not hasattr(agent, "_openrouter_fallback_started_at_monotonic"):
        setattr(agent, "_openrouter_fallback_started_at_monotonic", time.monotonic())
    if not hasattr(agent, "_openrouter_fallback_turns"):
        setattr(agent, "_openrouter_fallback_turns", 0)
    _cap_fallback_output(agent)

    path = gateway_health_path()
    payload = _load_health(path)
    owner = _fallback_owner(agent)
    if payload.get("fallback_owner") != owner:
        payload = {}
    now_epoch = time.time()
    if not payload.get("fallback_started_at_epoch"):
        payload["fallback_started_at_epoch"] = now_epoch
    payload.update(
        {
            "status": "degraded",
            "checked_at": _now_iso(),
            "active_provider": provider,
            "active_model": model,
            "fallback_tier": tier,
            "fallback_active": True,
            "fallback_notice_required": True,
            "fallback_max_turns": _max_turns(),
            "fallback_max_seconds": _max_seconds(),
            "fallback_turns": int(getattr(agent, "_openrouter_fallback_turns", 0)),
            "fallback_owner": owner,
            "last_failure": {
                "category": str(reason or "primary_failed"),
                "safe_summary": _fallback_safe_summary(tier),
            },
        }
    )
    _write_health(path, payload)
    return payload


def record_gateway_primary_route(agent: Any) -> None:
    if is_continuity_fallback_active(agent):
        return
    _clear_cap_state(agent)
    path = gateway_health_path()
    existing = _load_health(path)
    if (
        existing.get("fallback_active") is True
        and existing.get("fallback_owner")
        and existing.get("fallback_owner") != _fallback_owner(agent)
    ):
        return
    payload = {
        "status": "ok",
        "checked_at": _now_iso(),
        "active_provider": str(getattr(agent, "provider", "") or ""),
        "active_model": str(getattr(agent, "model", "") or ""),
        "fallback_active": False,
        "fallback_notice_required": False,
    }
    _write_health(path, payload)


def apply_openrouter_fallback_notice(
    agent: Any, final_response: str
) -> tuple[str, bool]:
    fallback_notice = fallback_notice_for_agent(agent)
    if not fallback_notice:
        existing = _load_health(gateway_health_path())
        restored = (
            _is_gpt56_subscription_primary(agent)
            and _norm(getattr(agent, "provider", "")) == "openai-codex"
            and _norm(getattr(agent, "model", "")) == "gpt-5.6-sol"
            and existing.get("fallback_active") is True
            and existing.get("fallback_owner") == _fallback_owner(agent)
        )
        record_gateway_primary_route(agent)
        if restored:
            body = str(final_response or "").strip()
            if body.startswith(PRIMARY_ROUTE_RESTORED_NOTICE):
                return body, False
            return (
                f"{PRIMARY_ROUTE_RESTORED_NOTICE}\n\n{body}".rstrip(),
                True,
            )
        return final_response, False

    cap_message = fallback_cap_message_if_exhausted(agent)
    if cap_message:
        return cap_message, True

    _record_fallback_turn(agent)
    original_body = str(final_response or "").strip()
    body = _strip_known_fallback_notice(original_body)
    body = body or "Fallback produced no response."
    if original_body.startswith(fallback_notice):
        return original_body, False
    return f"{fallback_notice}\n\n{body}", True


def fallback_cap_message_if_exhausted(
    agent: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str | None:
    provider_norm = _norm(
        provider if provider is not None else getattr(agent, "provider", "")
    )
    model_value = str(model if model is not None else getattr(agent, "model", "") or "")
    if provider_norm == "openrouter" and _norm(model_value) in _OPENROUTER_GPT55_MODELS:
        return (
            f"{OPENROUTER_FALLBACK_NOTICE}\n\nStopped: OpenRouter GPT-5.5 is "
            "explicit-only."
        )
    if continuity_fallback_tier(provider_norm, model_value) is None:
        return None
    notice = _fallback_notice_for_route(provider_norm, model_value)

    turns = getattr(agent, "_openrouter_fallback_turns", None)
    started = getattr(agent, "_openrouter_fallback_started_at_monotonic", None)
    if bool(getattr(agent, "_fallback_activated", False)) and (
        not isinstance(turns, int) or not isinstance(started, (int, float))
    ):
        return _cap_message("fallback cap state became unavailable", notice)
    turns = int(turns or 0)
    if turns >= _max_turns():
        return _cap_message(f"{_max_turns()} fallback turns", notice)
    if started is not None and time.monotonic() - started >= _max_seconds():
        minutes = int(_max_seconds() / 60)
        return _cap_message(f"{minutes} minutes", notice)
    return None


def is_emergency_openrouter_fallback_active(agent: Any) -> bool:
    return (
        is_continuity_fallback_active(agent)
        and continuity_fallback_tier(
            getattr(agent, "provider", ""), getattr(agent, "model", "")
        )
        == 1
    )


def is_continuity_fallback_active(agent: Any) -> bool:
    return (
        bool(getattr(agent, "_openrouter_fallback_notice_required", False))
        and bool(getattr(agent, "_fallback_activated", False))
        and continuity_fallback_tier(
            getattr(agent, "provider", ""), getattr(agent, "model", "")
        )
        is not None
        and _is_gpt56_subscription_primary(agent)
    )


def continuity_fallback_tier(provider: Any, model: Any) -> int | None:
    route = (_norm(provider), _norm(model))
    try:
        return CONTINUITY_FALLBACK_ROUTES.index(route) + 1
    except ValueError:
        return None


def fallback_notice_for_agent(agent: Any) -> str:
    if not is_continuity_fallback_active(agent):
        return ""
    tier = continuity_fallback_tier(
        getattr(agent, "provider", ""), getattr(agent, "model", "")
    )
    if tier == 1:
        return OPENROUTER_FALLBACK_NOTICE
    if tier == 2:
        return SECONDARY_FALLBACK_NOTICE
    return ""


def fallback_notice_from_text(text: Any) -> str:
    body = str(text or "").strip()
    for notice in (OPENROUTER_FALLBACK_NOTICE, SECONDARY_FALLBACK_NOTICE):
        if body.startswith(notice):
            return notice
    return ""


def gateway_health_path() -> Path:
    raw = os.getenv("HERMES_GATEWAY_HEALTH_PATH", "").strip()
    if raw:
        return Path(raw)
    if Path("/opt/data").exists():
        return Path("/opt/data/gateway-health.json")
    return Path.home() / ".hermes" / "gateway-health.json"


def _record_fallback_turn(agent: Any) -> None:
    path = gateway_health_path()
    payload = _load_health(path)
    if not hasattr(agent, "_openrouter_fallback_started_at_monotonic"):
        setattr(agent, "_openrouter_fallback_started_at_monotonic", time.monotonic())
    turns = int(getattr(agent, "_openrouter_fallback_turns", 0)) + 1
    setattr(agent, "_openrouter_fallback_turns", turns)
    if not payload.get("fallback_started_at_epoch"):
        payload["fallback_started_at_epoch"] = time.time()
    payload.update(
        {
            "status": "degraded",
            "checked_at": _now_iso(),
            "active_provider": str(getattr(agent, "provider", "") or ""),
            "active_model": str(getattr(agent, "model", "") or ""),
            "fallback_tier": continuity_fallback_tier(
                getattr(agent, "provider", ""), getattr(agent, "model", "")
            ),
            "fallback_active": True,
            "fallback_notice_required": True,
            "fallback_max_turns": _max_turns(),
            "fallback_max_seconds": _max_seconds(),
            "fallback_turns": turns,
            "fallback_owner": _fallback_owner(agent),
        }
    )
    _write_health(path, payload)


def _cap_fallback_output(agent: Any) -> None:
    if not hasattr(agent, "_openrouter_fallback_original_output"):
        setattr(
            agent,
            "_openrouter_fallback_original_output",
            {
                "max_tokens": getattr(agent, "max_tokens", None),
                "ephemeral_present": hasattr(agent, "_ephemeral_max_output_tokens"),
                "ephemeral_max_output_tokens": getattr(
                    agent, "_ephemeral_max_output_tokens", None
                ),
            },
        )
    cap = _max_output_tokens()
    current = getattr(agent, "max_tokens", None)
    if current is None or current > cap:
        try:
            setattr(agent, "max_tokens", cap)
        except Exception:
            pass
    current_ephemeral = getattr(agent, "_ephemeral_max_output_tokens", None)
    if current_ephemeral is None or current_ephemeral > cap:
        try:
            setattr(agent, "_ephemeral_max_output_tokens", cap)
        except Exception:
            pass


def restore_openrouter_fallback_state(agent: Any) -> None:
    original = getattr(agent, "_openrouter_fallback_original_output", None)
    if isinstance(original, dict):
        try:
            setattr(agent, "max_tokens", original.get("max_tokens"))
        except Exception:
            pass
        if original.get("ephemeral_present"):
            try:
                setattr(
                    agent,
                    "_ephemeral_max_output_tokens",
                    original.get("ephemeral_max_output_tokens"),
                )
            except Exception:
                pass
        else:
            try:
                delattr(agent, "_ephemeral_max_output_tokens")
            except (AttributeError, TypeError):
                pass
    for name in (
        "_openrouter_fallback_original_output",
        "_openrouter_fallback_model",
        "_continuity_fallback_tier",
    ):
        try:
            delattr(agent, name)
        except (AttributeError, TypeError):
            pass
    setattr(agent, "_openrouter_fallback_notice_required", False)


def _clear_cap_state(agent: Any) -> None:
    for name in (
        "_openrouter_fallback_started_at_monotonic",
        "_openrouter_fallback_turns",
    ):
        try:
            delattr(agent, name)
        except (AttributeError, TypeError):
            pass


def _cap_message(reason: str, notice: str) -> str:
    return (
        f"{notice}\n\n"
        f"Stopped: automatic fallback spend cap reached after {reason}. "
        "Hermes will not continue paid fallback use automatically. Restore "
        "GPT-5.6 subscription access or explicitly approve more fallback use."
    )


def _load_health(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_health(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return True
    except OSError:
        return False


def _max_turns() -> int:
    return max(1, _int_env("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_TURNS", 10))


def _max_seconds() -> int:
    return max(60, _int_env("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_SECONDS", 1800))


def _max_output_tokens() -> int:
    return max(
        256, _int_env("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_OUTPUT_TOKENS", 1200)
    )


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _fallback_notice_for_route(provider: Any, model: Any) -> str:
    tier = continuity_fallback_tier(provider, model)
    if tier == 1:
        return OPENROUTER_FALLBACK_NOTICE
    if tier == 2:
        return SECONDARY_FALLBACK_NOTICE
    return ""


def _strip_known_fallback_notice(text: Any) -> str:
    body = str(text or "").strip()
    for notice in (OPENROUTER_FALLBACK_NOTICE, SECONDARY_FALLBACK_NOTICE):
        if body.startswith(notice):
            return body[len(notice) :].lstrip()
    return body


def _fallback_safe_summary(tier: int) -> str:
    if tier == 1:
        return (
            "GPT-5.6 subscription route failed; Hermes entered visible capped "
            "OpenRouter MiniMax M3 fallback."
        )
    return (
        "GPT-5.6 subscription and MiniMax M3 routes were unavailable; Hermes "
        "entered visible capped direct OpenAI API GPT-5.6 Luna fallback."
    )


def _protected_chain_error(agent: Any) -> str:
    chain = getattr(agent, "_fallback_chain", None)
    if not isinstance(chain, list):
        return ""
    normalized = tuple(
        (
            (_norm(entry.get("provider")), _norm(entry.get("model")))
            if isinstance(entry, dict)
            else ("", "")
        )
        for entry in chain
    )
    if normalized == CONTINUITY_FALLBACK_ROUTES:
        return ""
    return (
        "Automatic continuity fallback chain must contain exactly "
        "openrouter/minimax/minimax-m3 followed by "
        "openai-api/gpt-5.6-luna, with no additional routes."
    )


def _is_gpt56_subscription_primary(agent: Any) -> bool:
    runtime = getattr(agent, "_primary_runtime", None)
    if not isinstance(runtime, dict):
        return False
    return (
        _norm(runtime.get("provider")) == PRIMARY_ROUTE_PROVIDER
        and _norm(runtime.get("model")) == PRIMARY_ROUTE_MODEL
    )


def _fallback_owner(agent: Any) -> str:
    return str(
        getattr(agent, "session_id", "")
        or getattr(agent, "task_id", "")
        or f"agent-{id(agent)}"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
