# Tournament research contract

Hermes protects tournament-result turns with the SportFish audit contract pinned
to `a23b30167011d888900630eb2c0412a94634035d` and
`tournament_route_preflight.v2`.

`tournament_truth_gate` is self-registering and invokes the installed audit
command with a fixed argument list and a 45-second timeout. It never invokes a
shell, retries, or a provider fetch. It reads only snapshots already contained
in the process-configured trusted source root; absent snapshots are a hold.

Configure the process-owned `tournament_truth_gate` section with absolute,
existing `receipt_root`, `journal_root`, and `source_snapshot_root` values.
The model cannot supply or override these paths. The gate always uses
`LATEST-JOURNAL.json` inside that journal root.

Before a current receipt is bound, tournament turns can only read trusted local
sources or call the preflight tool. Writes, terminal/code execution, image and
provider actions, delegation, delivery, and broad web discovery are denied.
The receipt binds the exact candidate bytes, request task/session/turn,
runtime-derived destination/entrypoint, nonce, and the audit's 15-minute
expiry. It is single-use. Private receipts authorize only `private_answer`;
they never authorize a public turn.

Finalization happens before assistant persistence and delivery. Any failed,
expired, altered, missing, or untrusted receipt replaces the candidate with
`ROUTE_HOLD` (private) or `PUBLIC_ARTIFACT_BLOCKED` (public); buffered deltas
are discarded. Cleanup runs on success, error, timeout, cancellation, reload,
and the next normal turn.
