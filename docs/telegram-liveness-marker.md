# Telegram liveness marker contract

The gateway best-effort atomically replaces `${HERMES_HOME:-/opt/data}/telegram-liveness.json` only after defensible receive-path evidence: an inbound update delivered to an active polling handler, or a periodic poll-health observation that finds a running updater, no reconnect in flight, and Telegram reporting `pending_update_count == 0`. A bare `start_polling()` return and `get_me()` only prove setup/send-path health and never refresh the marker.

The marker is mode `0600`, regular (never symlink-followed), and contains only:

```json
{"generated_at_monotonic":0.0,"generated_at_unix":0.0,"producer_source_sha":"<40-or-64-lowercase-hex>","schema_version":1,"telegram_last_success_monotonic":0.0,"telegram_last_success_unix":0.0}
```

All four timestamps are finite numbers. `generated_*` and `telegram_last_success_*` are captured from the same successful evidence event. The host watchdog accepts the marker only when every wall-clock and monotonic age is between zero and 900 seconds and their paired ages differ by at most 30 seconds.

The producer is bound to the running artifact: it requires a valid baked `/opt/hermes/.hermes_build_sha`. If `HERMES_REVISION` is set, it must exactly equal that baked SHA; a missing baked SHA, malformed value, or mismatch fails closed. No caller-supplied SHA can override this binding.

The heartbeat checks every 90 seconds, so a silent wedge with no inbound traffic can remain undetected for up to one heartbeat interval plus the 15-second probe timeout (about 105 seconds) before this producer stops refreshing and recovery begins. A nonzero pending count, absent/stopped updater, webhook mode, or reconnect in progress never refreshes the marker.

Writes remain atomic and symlink-safe: the container process creates a regular `0600` file, so its natural owner is the container runtime UID `10000`. The root-run host watchdog reads it without following symlinks and enforces regular-file status, UID `10000`, exact `0600`, schema/SHA/timestamp validity, TTL, and clock skew. The paired consumer source is `C:\\Users\\steve\\Documents\\Codex\\2026-06-21\\goal-steve-s-desktop-hermes-remote\\work\\audit-resilience-host-20260716\\hermes\\scripts\\hermes_host_watchdog.py`; producer and consumer must ship in the same rollout because the watchdog requires the exact marker contract.

Consumer escalation: that host code currently validates only the SHA's syntax, not equality to an independently configured expected/deployed SHA. Its README makes reviewed same-rollout matching an operational readiness requirement, but runtime expected-SHA enforcement is a host-consumer gap and requires a separately authorized host-repo repair.
