"""Fail-closed GPT-5.6 routing policy for Steve's Hermes deployment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


ROUTING_CONTRACT = "gpt56-routing-v3"
PRIMARY_PROVIDER = "openai-codex"
CANONICAL_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
MAX_CHILDREN = 3
MAX_DEPTH = 1


class RoutingPolicyError(ValueError):
    """Raised when a requested route violates the GPT-5.6 contract."""


def validate_codex_route_runtime(
    spec: "RouteSpec",
    *,
    provider: Any,
    model: Any,
    api_mode: Any,
    base_url: Any,
    api_key: Any,
) -> None:
    """Fail closed unless a routed runtime is canonical Codex OAuth.

    Error text deliberately names only violated fields.  Runtime values can
    originate in persisted credential pools, so serializing them here could
    disclose endpoint-embedded credentials or access tokens.
    """
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip().lower()
    normalized_api_mode = str(api_mode or "").strip().lower()
    raw_base_url = str(base_url or "").strip()
    parsed = urlsplit(raw_base_url)
    canonical_endpoint = (
        parsed.scheme.lower() == "https"
        and (parsed.hostname or "").lower() == "chatgpt.com"
        and parsed.port in (None, 443)
        and parsed.path.rstrip("/") == "/backend-api/codex"
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )

    errors: list[str] = []
    if normalized_provider != spec.provider:
        errors.append("provider is not canonical")
    if normalized_model != spec.model:
        errors.append("model is not canonical")
    if normalized_api_mode != "codex_responses":
        errors.append("api_mode is not codex_responses")
    if not canonical_endpoint:
        errors.append("base_url is not the canonical Codex endpoint")
    if not isinstance(api_key, str) or not api_key.strip():
        errors.append("Codex OAuth token is missing")
    if errors:
        raise RoutingPolicyError(
            f"GPT-5.6 route {spec.route_id!r} runtime validation failed closed: "
            + "; ".join(errors)
        )


def validate_operator_config(config: Mapping[str, Any]) -> bool:
    """Validate the complete opt-in contract and return whether it is enabled."""
    if not isinstance(config, Mapping):
        raise RoutingPolicyError("delegation.gpt56_routing must be a mapping")
    raw_enabled = config.get("enabled", False)
    enabled = raw_enabled is True or str(raw_enabled).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return False

    errors = []
    if config.get("contract") != ROUTING_CONTRACT:
        errors.append(f"contract must be {ROUTING_CONTRACT!r}")
    if config.get("provider") != PRIMARY_PROVIDER:
        errors.append(f"provider must be {PRIMARY_PROVIDER!r}")
    if config.get("max_children") != MAX_CHILDREN:
        errors.append(f"max_children must be exactly {MAX_CHILDREN}")
    if config.get("max_depth") != MAX_DEPTH:
        errors.append(f"max_depth must be exactly {MAX_DEPTH}")
    if errors:
        raise RoutingPolicyError(
            "Invalid delegation.gpt56_routing configuration: " + "; ".join(errors)
        )
    return True


class RouteId(str, Enum):
    MAIN = "main"
    EXPLORER = "explorer"
    WORKER = "worker"
    REVIEWER = "reviewer"
    EXPERT = "expert"


class ModelAlias(str, Enum):
    SOL = "gpt56_sol"
    TERRA = "gpt56_terra"
    LUNA = "gpt56_luna"


MODEL_IDS = {
    ModelAlias.SOL.value: "gpt-5.6-sol",
    ModelAlias.TERRA.value: "gpt-5.6-terra",
    ModelAlias.LUNA.value: "gpt-5.6-luna",
}

MODEL_EFFORTS = {
    "gpt-5.6-sol": frozenset({"low", "medium", "high", "xhigh", "max", "ultra"}),
    "gpt-5.6-terra": frozenset({"low", "medium", "high", "xhigh", "max", "ultra"}),
    "gpt-5.6-luna": frozenset({"low", "medium", "high", "xhigh", "max"}),
}


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    model_alias: str
    model: str
    effort: str
    reason: str
    fanout_depth: int
    fallback_used: bool = False
    provider: str = PRIMARY_PROVIDER
    protected: bool = False

    @property
    def allow_fallback(self) -> bool:
        return not self.protected

    def as_contract(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "model_alias": self.model_alias,
            "effort": self.effort,
            "reason": self.reason,
            "fanout_depth": self.fanout_depth,
            "fallback_used": self.fallback_used,
            "provider": self.provider,
        }


_CANONICAL_ROUTES = {
    RouteId.MAIN.value: RouteSpec(
        route_id=RouteId.MAIN.value,
        model_alias=ModelAlias.SOL.value,
        model=MODEL_IDS[ModelAlias.SOL.value],
        effort="max",
        reason="primary controller",
        fanout_depth=0,
        protected=True,
    ),
    RouteId.EXPLORER.value: RouteSpec(
        route_id=RouteId.EXPLORER.value,
        model_alias=ModelAlias.LUNA.value,
        model=MODEL_IDS[ModelAlias.LUNA.value],
        effort="low",
        reason="search, inventory, extraction, formatting, or simple checks",
        fanout_depth=1,
    ),
    RouteId.WORKER.value: RouteSpec(
        route_id=RouteId.WORKER.value,
        model_alias=ModelAlias.TERRA.value,
        model=MODEL_IDS[ModelAlias.TERRA.value],
        effort="medium",
        reason="bounded implementation, deterministic analysis, or routine tests",
        fanout_depth=1,
    ),
    RouteId.REVIEWER.value: RouteSpec(
        route_id=RouteId.REVIEWER.value,
        model_alias=ModelAlias.TERRA.value,
        model=MODEL_IDS[ModelAlias.TERRA.value],
        effort="high",
        reason="code review, edge cases, QA, or moderately ambiguous diagnosis",
        fanout_depth=1,
    ),
    RouteId.EXPERT.value: RouteSpec(
        route_id=RouteId.EXPERT.value,
        model_alias=ModelAlias.SOL.value,
        model=MODEL_IDS[ModelAlias.SOL.value],
        effort="max",
        reason="architecture, security, credentials, production, migration, or unresolved ambiguity",
        fanout_depth=1,
        protected=True,
    ),
}


def validate_model_effort(model: str, effort: str) -> None:
    normalized_model = str(model or "").strip().lower()
    normalized_effort = str(effort or "").strip().lower()
    supported = MODEL_EFFORTS.get(normalized_model)
    if supported is None:
        raise RoutingPolicyError(
            f"unsupported model for {ROUTING_CONTRACT}: {normalized_model or '<empty>'}"
        )
    if normalized_effort not in supported:
        raise RoutingPolicyError(
            f"{normalized_model} does not support effort {normalized_effort or '<empty>'}"
        )


def route_spec(
    route_id: str | RouteId,
    *,
    reason: str | None = None,
    fanout_depth: int | None = None,
    fallback_used: bool = False,
) -> RouteSpec:
    normalized = str(route_id.value if isinstance(route_id, RouteId) else route_id or "").strip().lower()
    template = _CANONICAL_ROUTES.get(normalized)
    if template is None:
        raise RoutingPolicyError(f"unknown route: {normalized or '<empty>'}")
    final_reason = str(reason if reason is not None else template.reason).strip()
    if not final_reason:
        raise RoutingPolicyError(f"route {normalized} requires a non-secret reason")
    depth = template.fanout_depth if fanout_depth is None else fanout_depth
    if not isinstance(depth, int) or depth < 0:
        raise RoutingPolicyError(f"route {normalized} has invalid fanout_depth: {depth!r}")
    spec = replace(
        template,
        reason=final_reason,
        fanout_depth=depth,
        fallback_used=bool(fallback_used),
    )
    validate_model_effort(spec.model, spec.effort)
    return spec


def _aux(
    route_id: str,
    model_alias: str,
    effort: str,
    reason: str,
    *,
    protected: bool = False,
) -> RouteSpec:
    model = MODEL_IDS[model_alias]
    validate_model_effort(model, effort)
    return RouteSpec(
        route_id=route_id,
        model_alias=model_alias,
        model=model,
        effort=effort,
        reason=reason,
        fanout_depth=0,
        protected=protected,
    )


AUXILIARY_ROUTES: dict[str, RouteSpec] = {
    "skills_hub": _aux("explorer", "gpt56_luna", "low", "mechanical skill discovery"),
    "mcp": _aux("explorer", "gpt56_luna", "low", "mechanical MCP routing"),
    "title_generation": _aux("explorer", "gpt56_luna", "low", "short title generation"),
    "tts_audio_tags": _aux("explorer", "gpt56_luna", "low", "short TTS tag rewrite"),
    "monitor": _aux("explorer", "gpt56_luna", "low", "high-volume item scoring"),
    "profile_describer": _aux("explorer", "gpt56_luna", "low", "short profile description"),
    "web_extract": _aux("explorer", "gpt56_luna", "medium", "bounded page extraction"),
    "compression": _aux("explorer", "gpt56_luna", "medium", "context-preserving compression"),
    "triage_specifier": _aux("worker", "gpt56_terra", "medium", "bounded task specification"),
    "coding-qa-worker": _aux("worker", "gpt56_terra", "medium", "bounded coding QA"),
    "moa_reference": _aux("worker", "gpt56_terra", "medium", "independent reference analysis"),
    "vision": _aux("reviewer", "gpt56_terra", "high", "vision interpretation and verification"),
    "kanban_decomposer": _aux("reviewer", "gpt56_terra", "high", "task-graph decomposition"),
    "approval": _aux("expert", "gpt56_sol", "max", "protected approval decision", protected=True),
    "goal_judge": _aux("expert", "gpt56_sol", "max", "protected completion verdict", protected=True),
    "curator": _aux("expert", "gpt56_sol", "max", "protected skill-lifecycle review", protected=True),
    "background_review": _aux("expert", "gpt56_sol", "max", "protected memory and skill review", protected=True),
    "moa_aggregator": _aux("expert", "gpt56_sol", "max", "protected final synthesis", protected=True),
}


# Provider-owned plugin tasks keep their native provider-selection contract.
# They are explicitly classified here so the GPT overlay does not reject them
# as unknown or invent a GPT model alias for a workflow it does not own.
PROVIDER_LOCKED_AUXILIARY_TASKS: dict[str, dict[str, str]] = {
    "call": {
        "owner": "teams_pipeline",
        "complexity": "medium",
        "reason": "Teams meeting-summary provider remains plugin-owned",
    },
}


def provider_locked_auxiliary_task(task: str) -> dict[str, str] | None:
    """Return native-provider metadata for an explicitly locked task."""
    normalized = str(task or "").strip()
    metadata = PROVIDER_LOCKED_AUXILIARY_TASKS.get(normalized)
    return dict(metadata) if metadata is not None else None


def auxiliary_spec(task: str) -> RouteSpec:
    normalized = str(task or "").strip()
    spec = AUXILIARY_ROUTES.get(normalized)
    if spec is None:
        raise RoutingPolicyError(f"unclassified auxiliary task: {normalized or '<empty>'}")
    return spec


def validate_ultra_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    parent_depth: int,
) -> list[RouteSpec]:
    if parent_depth != 0:
        raise RoutingPolicyError("Ultra fan-out is available only to the top-level parent")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not 2 <= len(tasks) <= 3:
        raise RoutingPolicyError("Ultra fan-out requires two or three independent tasks")

    resolved: list[RouteSpec] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise RoutingPolicyError(f"Ultra task {index + 1} must be an object")
        if task.get("independent") is not True:
            raise RoutingPolicyError(f"Ultra task {index + 1} must declare independent=true")
        route_id = str(task.get("route") or "").strip()
        if not route_id:
            raise RoutingPolicyError(f"Ultra task {index + 1} requires an explicit route")
        if str(task.get("role") or "leaf").strip().lower() != "leaf":
            raise RoutingPolicyError("Ultra tasks must be leaf workstreams; recursive fan-out is forbidden")
        reason = str(task.get("reason") or "").strip()
        if not reason:
            raise RoutingPolicyError(f"Ultra task {index + 1} requires a routing reason")
        resolved.append(route_spec(route_id, reason=reason, fanout_depth=1))
    return resolved


__all__ = [
    "AUXILIARY_ROUTES",
    "CANONICAL_CODEX_BASE_URL",
    "MODEL_EFFORTS",
    "MODEL_IDS",
    "MAX_CHILDREN",
    "MAX_DEPTH",
    "ModelAlias",
    "PRIMARY_PROVIDER",
    "PROVIDER_LOCKED_AUXILIARY_TASKS",
    "ROUTING_CONTRACT",
    "RouteId",
    "RouteSpec",
    "RoutingPolicyError",
    "auxiliary_spec",
    "provider_locked_auxiliary_task",
    "route_spec",
    "validate_model_effort",
    "validate_codex_route_runtime",
    "validate_operator_config",
    "validate_ultra_tasks",
]
