# Telegram liveness marker contract

After a successful Telegram polling start or successful `get_me` connectivity probe, the gateway best-effort atomically replaces `${HERMES_HOME:-/opt/data}/telegram-liveness.json`.

The marker is mode `0600`, regular (never symlink-followed), and contains only:

```json
{"generated_at_monotonic":0.0,"generated_at_unix":0.0,"producer_source_sha":"<40-or-64-lowercase-hex>","schema_version":1,"telegram_last_success_monotonic":0.0,"telegram_last_success_unix":0.0}
```

All four timestamps are finite numbers. `generated_*` and `telegram_last_success_*` are captured from the same successful evidence event. The host watchdog accepts the marker only when every wall-clock and monotonic age is between zero and 900 seconds and their paired ages differ by at most 30 seconds.

The producer requires a valid `HERMES_REVISION` or baked `/opt/hermes/.hermes_build_sha`; it refuses to invent a SHA. Any unsafe path, invalid SHA, or write failure is swallowed by the gateway and leaves no new healthy signal. A restart/reload cannot refresh a prior marker: only new successful polling/provider evidence emits fresh timestamps.
