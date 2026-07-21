# Tournament research contract

Hermes protects tournament-result turns against the SportFish audit repair head
`271141abc8fd63cee3c9773fc800bdd786d871ef` and
`tournament_route_preflight.v2`. This reference remains under audit re-review;
it is not described as frozen.

`tournament_truth_gate` is self-registering and invokes the installed audit
command with a fixed argument list and a 45-second timeout. It never invokes a
shell, retries, or a provider fetch. It reads only snapshots already contained
in the process-configured trusted source root; absent snapshots are a hold.

Configure the process-owned `tournament_truth_gate` section with absolute,
existing `receipt_root`, `journal_root`, and `source_snapshot_root` values.
The model cannot supply or override these paths. The gate always uses
`LATEST-JOURNAL.json` inside that journal root.

Before and after a current receipt is bound, tournament turns can only read
trusted local sources or call the preflight tool. The receipt authorizes only
the final text release. Writes, terminal/code execution, image and provider
actions, delegation, delivery, and broad web discovery remain denied.
The receipt binds the exact candidate bytes, request task/session/turn,
runtime-derived destination/entrypoint, nonce, and the audit's 15-minute
expiry. It is single-use. Private receipts authorize only `private_answer`;
they never authorize a public turn.

Finalization re-runs the audit preflight against current trusted journal and
snapshot bytes before assistant persistence and delivery. Any failed,
expired, altered, missing, or untrusted receipt replaces the candidate with
`ROUTE_HOLD` (private) or `PUBLIC_ARTIFACT_BLOCKED` (public); buffered deltas
are discarded. Cleanup runs on success, error, timeout, cancellation, reload,
and the next normal turn.
