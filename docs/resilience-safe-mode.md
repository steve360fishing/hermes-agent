# Hermes safe-mode resilience

`HERMES_SAFE_MODE=1` is equivalent to `hermes chat --safe-mode` before
configuration, rules, plugins, MCP discovery, and shell hooks load. Safe mode
uses built-in defaults only and disables user plugins, skills, tools, models,
configuration, and rules.

The assigned startup surfaces cannot safely preserve the bundled Telegram
transport alone: its construction is owned by `gateway/run.py` and
`gateway/platforms/base.py`. Until their owner wires the recovery transport,
safe mode reports degraded readiness rather than claiming Telegram availability.
The owning gateway path must publish
`gateway.status.safe_mode_transport_readiness()` after its first runtime-status
write whenever `HERMES_SAFE_MODE=1`.

Cached gateway agents must call
`agent.agent_runtime_helpers.fallback_cap_message_after_primary_eligibility(agent)`
at `gateway/run.py`'s pre-`run_conversation` fallback-cap check (currently near
line 18835). This restores and re-evaluates primary eligibility before a cached
fallback spend cap can block recovery.

Profile reconciliation permission or registration failure leaves container
liveness available for recovery where possible, and records a safe degraded
readiness diagnostic in `gateway_state.json`.
