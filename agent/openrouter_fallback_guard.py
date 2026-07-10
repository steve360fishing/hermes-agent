from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OPENROUTER_FALLBACK_MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_FALLBACK_NOTICE = (
    "OPENROUTER FALLBACK ACTIVE: using anthropic/claude-haiku-4.5 because "
    "GPT-5.6 subscription access failed."
)
_OPENROUTER_GPT55_MODELS = {"openai/gpt-5.5", "openai/gpt-5.5-pro"}


def openrouter_fallback_activation_allowed(
    agent: Any, provider: str, model: str
) -> tuple[bool, str]:
    provider_norm = _norm(provider)
    model_norm = _norm(model)
    if provider_norm != "openrouter":
        return True, ""
    if model_norm in _OPENROUTER_GPT55_MODELS:
        return (
            False,
            "OpenRouter GPT-5.5 is explicit-only and cannot be used as "
            "automatic fallback.",
        )
    if not _is_gpt56_subscription_primary(agent):
        return True, ""
    if model_norm != OPENROUTER_FALLBACK_MODEL:
        return True, ""
    cap_message = fallback_cap_message_if_exhausted(
        agent, provider=provider_norm, model=model
    )
    if cap_message:
        return False, cap_message
    return True, ""


def record_openrouter_fallback_activation(
    agent: Any, *, reason: str | None = None
) -> dict[str, Any]:
    if _norm(getattr(agent, "provider", "")) != "openrouter":
        return {}
    if not _is_gpt56_subscription_primary(agent):
        return {}
    model = str(getattr(agent, "model", "") or "").strip()
    if _norm(model) != OPENROUTER_FALLBACK_MODEL:
        return {}
    setattr(agent, "_openrouter_fallback_notice_required", True)
    setattr(agent, "_openrouter_fallback_model", model)
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
            "active_provider": "openrouter",
            "active_model": model,
            "fallback_active": True,
            "fallback_notice_required": True,
            "fallback_max_turns": _max_turns(),
            "fallback_max_seconds": _max_seconds(),
            "fallback_turns": int(getattr(agent, "_openrouter_fallback_turns", 0)),
            "fallback_owner": owner,
            "last_failure": {
                "category": str(reason or "primary_failed"),
                "safe_summary": (
                    "GPT-5.6 subscription route failed; Hermes entered visible "
                    "capped OpenRouter Haiku fallback."
                ),
            },
        }
    )
    _write_health(path, payload)
    return payload


def record_gateway_primary_route(agent: Any) -> None:
    if is_emergency_openrouter_fallback_active(agent):
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
    if not is_emergency_openrouter_fallback_active(agent):
        record_gateway_primary_route(agent)
        return final_response, False

    cap_message = fallback_cap_message_if_exhausted(agent)
    if cap_message:
        return cap_message, True

    _record_fallback_turn(agent)
    body = str(final_response or "").strip() or "Fallback produced no response."
    if body.startswith(OPENROUTER_FALLBACK_NOTICE):
        return body, False
    return f"{OPENROUTER_FALLBACK_NOTICE}\n\n{body}", True


def fallback_cap_message_if_exhausted(
    agent: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str | None:
    provider_norm = _norm(
        provider if provider is not None else getattr(agent, "provider", "")
    )
    if provider_norm != "openrouter":
        return None
    model_value = str(model if model is not None else getattr(agent, "model", "") or "")
    if _norm(model_value) in _OPENROUTER_GPT55_MODELS:
        return (
            f"{OPENROUTER_FALLBACK_NOTICE}\n\nStopped: OpenRouter GPT-5.5 is "
            "explicit-only."
        )
    if _norm(model_value) != OPENROUTER_FALLBACK_MODEL:
        return None

    turns = getattr(agent, "_openrouter_fallback_turns", None)
    started = getattr(agent, "_openrouter_fallback_started_at_monotonic", None)
    if bool(getattr(agent, "_fallback_activated", False)) and (
        not isinstance(turns, int) or not isinstance(started, (int, float))
    ):
        return _cap_message("fallback cap state became unavailable")
    turns = int(turns or 0)
    if turns >= _max_turns():
        return _cap_message(f"{_max_turns()} fallback turns")
    if started is not None and time.monotonic() - started >= _max_seconds():
        minutes = int(_max_seconds() / 60)
        return _cap_message(f"{minutes} minutes")
    return None


def is_emergency_openrouter_fallback_active(agent: Any) -> bool:
    return (
        bool(getattr(agent, "_openrouter_fallback_notice_required", False))
        and bool(getattr(agent, "_fallback_activated", False))
        and _norm(getattr(agent, "provider", "")) == "openrouter"
        and _norm(getattr(agent, "model", "")) == OPENROUTER_FALLBACK_MODEL
        and _is_gpt56_subscription_primary(agent)
    )


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
            "active_provider": "openrouter",
            "active_model": str(getattr(agent, "model", "") or ""),
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


def _cap_message(reason: str) -> str:
    return (
        f"{OPENROUTER_FALLBACK_NOTICE}\n\n"
        f"Stopped: automatic fallback spend cap reached after {reason}. "
        "Hermes will not continue spending OpenRouter credits automatically. "
        "Restore GPT-5.6 subscription access or explicitly approve more fallback use."
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


def _is_gpt56_subscription_primary(agent: Any) -> bool:
    runtime = getattr(agent, "_primary_runtime", None)
    if not isinstance(runtime, dict):
        return False
    return (
        _norm(runtime.get("provider")) == "openai-codex"
        and _norm(runtime.get("model")) == "gpt-5.6-sol"
    )


def _fallback_owner(agent: Any) -> str:
    return str(
        getattr(agent, "session_id", "")
        or getattr(agent, "task_id", "")
        or f"agent-{id(agent)}"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
