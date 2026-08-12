# Completion Contract: Hermes tournament copy gate repair

**Created:** 2026-08-12
**Source request:** Steve's 2026-08-12 GOAL and P1-P20 acceptance contract
**Current phase:** EXECUTING
**Terminal state:** Source changes, deterministic tests, bounded review, provenance, rollback, written canaries, and one immutable live deployment packet are complete; execution stops before live mutation for exact packet-bound approval.
**Finalization authority:** Source-only local edits and tests. On 2026-08-12 Steve approved the exact revised source-verification pass: add `run_agent.py` and `agent/tool_executor.py` to the frozen Hermes scope, add real registered-provider-shape capture/gate fixtures and normalization, use a task-local non-global test environment for missing ACP/POSIX-compatible full-suite dependencies, and reconcile the existing live file drift read-only. This approval does not authorize image build/pull, live apply/recreate/restart, cron changes, external send/publication, or live canaries. Commit/push/merge/PR authority remains governed by the original explicit exclusion and is not inferred from blanket wording.

## Scope

### In scope
- Repair tournament intent/authority classification for P1-P12 and P17, including quote, negation, clause, and trusted continuation context.
- Implement exact pending-publication packets and authenticated, expiring, one-use approval intake without executing during intake.
- Implement the PREPARED -> APPROVED -> IN_FLIGHT -> CONSUMED | AMBIGUOUS | FAILED_PRE_DISPATCH state machine.
- Expose the audit repo's existing R2 trusted snapshot/journal authority through one narrow, runtime-owned Hermes capture adapter; preserve its route/source-map boundary.
- Make public-draft research -> trusted capture -> final transform -> byte freeze -> truth gate -> exact private delivery executable.
- Scope tool guardrails by action semantics/sink; preserve harmless memory, diagnostics, handoffs, research, capture, and gate reachability.
- Preserve unrelated private content; finalize only claim-bearing candidate bytes; distinguish draft HOLD from publication PREPARED_NOT_RELEASED.
- Add deterministic P1-P20 and receipt/approval matrix coverage at the requested layers, plus source/build/runtime provenance, rollback, and written canaries.
- Create one immutable live deployment packet and stop for exact packet-bound approval.

### Approved revision 2 source-verification pass

- Add `run_agent.py` and `agent/tool_executor.py` to the Hermes frozen scope because they are required schema/dispatcher consumers.
- Add provider-realistic, registered-source capture -> trusted manifest -> truth gate -> private-delivery fixtures and only the normalization required for those registered provider shapes.
- Permit a task-local, non-global test environment containing the missing ACP/POSIX-compatible full-suite dependencies.
- Permit read-only reconciliation of the existing live `task_execution_contract.py` drift for provenance and rollback planning.
- Preserve the original deployment packet gate. No free-form approval, including “approve everything,” authorizes build/pull, live apply, recreate, canaries, rollback, send, or publication.

### Out of scope
- Any live Hermes mutation, restart, recreate, build/pull, cron mutation, external send/post/publish, credential action, production test action, commit, push, merge, PR, or global dependency installation. Task-local non-global test dependencies are permitted for the approved source-verification pass.
- Any broad remediation of legacy tournament fetchers except blocking their use as trusted R2 capture authority.
- Any sink expansion beyond the currently supported `send_message` publication action.
- Any cleanup or repair unrelated to R1 intent/authority, R2 trusted capture, or R3 finalizer/guardrail/delivery.

### Assumptions
- Hermes source authority is `steve360fishing/hermes-agent`, base `9945e25592f717f36507ecad65e35ca1ecc311c9`, in the task-owned worktree.
- Audit source authority is `steve360fishing/sportfishhub-tournament-audit`, base `d7ca49b9782c2437d10dacfe055dcccc3eeef319`, in the task-owned worktree.
- Live Hermes serves image `sha256:d7badc9b0d3ed057a56776da5a680c97a15e9ca87234a6b3cadcf6c3483283bc`; live mutation is not authorized.
- The audit R2 journal/source-map and `newsletter_truth_refresh` pipeline is authoritative; legacy `tournament_truth.fetchers` is not an approved R2 adapter.
- A live post-build `task_execution_contract.py` patch is a provenance HOLD for the later deployment packet and will not be copied from the container into source.

## Acceptance criteria

| ID | Criterion | Required evidence | Status |
|---|---|---|---|
| AC-01 | P1-P12 and P17 classify exactly; only real drafts/publications install a contract | Table tests plus contract-install tests | PENDING |
| AC-02 | P1 leaves memory/internal tools available and preserves a useful answer | Guardrail and finalizer regression | PENDING |
| AC-03 | P3 completes trusted capture -> receipt -> byte-identical private delivery without approval language | Real local audit gate E2E fixture | PENDING |
| AC-04 | P10/P11 publication continuations require resolved pending action plus exact truth and approval | Context/approval matrix tests | PENDING |
| AC-05 | P17 intake requires authenticated metadata and exact action/checksum and never executes in intake turn | Gateway/intake state tests | PENDING |
| AC-06 | Approval state transitions and P16/P19 ambiguity/no-replay behavior are deterministic | State-machine/crash-restart tests | PENDING |
| AC-07 | Trusted capture uses registered R2 source authority and rejects P14 threats without leaked secrets or leftover files | Fake DNS/HTTP/security tests | PENDING |
| AC-08 | P13 makes capture and truth gate reachable in one active public-draft contract | Schema/registry/dispatcher/task/session E2E test | PENDING |
| AC-09 | P18 rejects post-receipt byte changes; no-op transport preserves identity | Transform/finalizer tests | PENDING |
| AC-10 | Receipt/approval matrix and P15 fail closed without irrelevant authority consumption | Matrix tests | PENDING |
| AC-11 | Streaming/non-streaming/persistence/callback cleanup are equivalent, including exceptions/interruption | Finalizer/cleanup suite | PENDING |
| AC-12 | Cron bypass is direct, unmasked, and passing | Dedicated cron tests | PENDING |
| AC-13 | Focused and affected full suites plus compile/static checks pass with no relevant skip/xfail | Command receipts | PENDING |
| AC-14 | R1-R3 bounded reviews find no blocking issue after one targeted repair rerun | Review receipts | PENDING |
| AC-15 | Source/build/container provenance, backup/rollback, canaries, and immutable packet are complete | Hash-bound packet | PENDING |
| AC-16 | No permanent blanket authority exists and no external action/live mutation occurred | Source search, packet exclusion, live read-back | PENDING |

## Execution phases and traceability

1. D1: source/runtime authority, ownership, failure reproduction, approval reachability, and capture-deadlock proof -> AC-01, AC-03, AC-05, AC-15.
2. T1: add behavior-valid RED tests in Hermes and audit worktrees -> AC-01 through AC-12.
3. I1: implement Hermes typed intent/pending approval/state machine/guardrail/finalizer/adapter changes -> AC-01, AC-02, AC-04 through AC-11.
4. I2: implement audit R2 capture adapter/security hardening needed by Hermes -> AC-03, AC-07, AC-08.
5. V1: focused/affected suites, compile/static checks, bounded R1-R3 reviews, Fable review, one repair rerun if needed -> AC-13, AC-14.
6. P1: provenance/rollback/canary/immutable packet and exact live approval request -> AC-15, AC-16.

## NO DRIFT. DO NOT DRIFT.

**Follow this plan exactly. Nothing outside it gets acted on.** Not a smarter approach spotted
mid-task, not a better name, not an extra test, not sharper copy, not a bug noticed in passing, not a
"while I'm already in this file" cleanup. **"It's a real problem" and "this is obviously better" are
not justifications** — they are the rationalisation pattern this section exists to stop.

**Everything else discovered gets RECORDED, not done.** At closure, run one bounded Drift disposition
pass. At confidence 90 or above, duplicate, resolved, superseded, untraceable, out-of-scope, or
non-beneficial items receive a durable terminal receipt; valuable separate-scope work receives an
owner and trigger. Only consequential unresolved choices remain active for Steve.

**The only exception, and it is not a judgment call:** act mid-plan *only* if the problem is inside
plan scope, **or** it blocks verification of a planned step, **or** it is actively breaking production,
security, or data integrity right now. Unsure means hold it and say so.

**Declared checks and declared skills are closed sets.** A check or skill wanted mid-execution is
itself a recorded discovery. Every completion report carries `Declared checks run:` and the
closure-ledger summary.

**If the plan turns out to be wrong:** stop, report what breaks it, and wait. Do not improvise a
replacement — changing the plan is Steve's call, not a mid-execution correction.

### The four drift modes, named from real incidents

- **A discovery is not automatically a work item.** Re-reading it mid-execution and deciding it is
  worth doing is drift. At closure it is adopted because it is required, terminally closed, routed
  separately, or kept active as a genuine blocker.
  *(2026-08-03: "the Vercel registry is stale" — logged for a ruling — became a live-API portfolio
  inventory with deletion recommendations.)*
- **Every task traces to a sentence the requester wrote.** Before each phase, confirm each task traces.
  Anything that does not is deleted, not executed. *(2026-08-03: a pricing edit entered a plan at
  write time with no source and survived review.)*
- **No predicted number is reported as an outcome.** Predictions are labelled predictions; only
  measured values appear in a result. If a prediction misses, say so in the same breath as the number.
  *(2026-08-03: "~65KB" predicted, flat delivered.)*
- **A heuristic is a hypothesis, never a conclusion.** When a known pattern seems to explain something,
  run the one command that would disprove it before recording it as fact.
  *(2026-08-03: a real uncommitted hook registration was filed as a CRLF artifact.)*

### Required alongside this block

- **`Hard out-of-bounds`** — a list naming the specific things this plan must not touch (other projects
  per Iron Law #8, protected files, credential rotation, destructive data actions, outbound customer
  contact, printing secrets, live production systems).
- **`DECLARED SKILLS`** — the closed set of skills this plan may invoke, run-mode and point-fire
  separately, each tied to its triggering phase.
- **`Declared checks`** — every audit, council, test run, verification, pixel pass and outside review,
  named before execution starts. If it is not declared, it is not run.

**Phase-boundary ritual.** Before starting each phase, emit one line:
`Phase N — tasks: <list>. Traced: yes. Recorded discoveries: <n>.`
If that line cannot be written truthfully, the phase does not start.

## Hard out-of-bounds

- Live VPS/container files, service state, cron, images, compose, providers, external sinks, and publication.
- Credentials, secret values, credential rotation, auth copying, billing, paid research providers, and global dependencies. Task-local source-verification dependencies are the sole dependency exception.
- Dirty user checkouts, unrelated repositories, unrelated branches/worktrees, unrelated test failures, and cleanup.
- Commit, push, merge, PR creation/update, public remote, or new remote.
- Broad legacy fetcher remediation, sink expansion, permanent approval, arbitrary trusted-root writes, or weakened truth/receipt gates.

## DECLARED SKILLS

- `grill-me` — discovery/clarity gate before execution.
- `debugging-investigator` — D1/T1 root-cause reproduction before repair.
- `agent-routing-intelligence` — bounded read-only discovery and disjoint implementation/review lanes.
- `delegated-engineering-decisions` — safe architecture/implementation choices within the frozen scope.
- `complete-to-closure` — lifecycle owner through the deployment-packet HOLD.
- `drift` — freeze/guard at phase boundaries and one closure disposition.
- `fable` plus `openrouter-consult` — one explicit review after completed artifact/evidence packet; reconciliation only for material disagreement.
- `verification-before-completion` — one native final source/predeploy verdict.

## Declared checks

- C01 exact deployed P1 classifier/contract/tool/finalizer reproduction and cleanup trace.
- C02 trusted-snapshot circular-dependency reproduction and approval-intake caller trace.
- C03 classifier table P1-P12/P17 plus quote/code/negation/clause/metadata/context variants.
- C04 contract-install, provider-schema, registry, dispatcher, task/session binding tests.
- C05 guardrail semantics tests for research/capture/memory/internal artifact/arbitrary write/external action.
- C06 capture fake HTTP/DNS tests for P13/P14, containment, SSRF, redirects, bounds, atomicity, redaction, cleanup.
- C07 truth-gate manifest/bytes/freshness/provenance/sink tests for P15/P18.
- C08 approval/state-machine/matrix/crash-restart tests for P10/P11/P16/P17/P19/P20.
- C09 finalizer streaming/non-streaming/transform/persistence/callback/exception/interruption tests.
- C10 local E2E gateway tests for P1/P3/P4/P10/P13/P16-P19.
- C11 cron bypass dedicated regression with no expectedFailure/skip/mock PASS.
- C12 Hermes focused suite, affected suite, compile, and repository-native static checks.
- C13 Audit focused suite, affected suite, compile, and repository-native static checks.
- C14 exactly one focused review for R1, one for R2, one for R3, plus one targeted rerun only after an in-scope repair.
- C15 one explicit Fable review against the completed redacted evidence packet.
- C16 source/diff/patch/config/image/preimage/rollback/canary/packet hash verification.
- C17 one final `verification-before-completion` HIGH_RISK source/predeploy packet.

## User-only gates

- LIVE-APPLY: exact authenticated approval of the final immutable packet ID/checksum, including enumerated build/pull, recreate, ordered canaries, and rollback. Status: PENDING, reached only after source/predeploy PASS.
- CREDENTIAL-INCIDENT: separate secret-isolated credential rotation/revocation authority for values exposed during a read-only compose/config inspection. Status: HOLD; independent source work continues, no values repeated or persisted.

## Recorded discoveries

- LIVE-PROVENANCE-01: active `/opt/hermes/agent/task_execution_contract.py` differs from the sealed image source. Classification: KEEP_REQUIRED for the deployment packet state guard; do not copy or repair live.
- LEGACY-FETCHER-01: legacy generic fetcher does not reassert authoritative host after redirects. Classification: CLOSED_NOT_REQUIRED for broad repair; R2 adapter must not use it.
- CREDENTIAL-OUTPUT-01: one read-only inspection emitted credential-bearing connection values into model-visible output. Classification: ROUTED_SEPARATE_SCOPE to secret-isolated security remediation after Steve authority; never repeat values.

## Verification contract

- Profile: HIGH_RISK.
- Mutation anchor: final Hermes/audit HEADs plus SHA-256 of both working-tree diffs and immutable packet bytes.
- Evidence packet: `C:/Users/steve/Documents/Codex/2026-08-12/hermes-tournament-copy-gate/evidence/verification-packet.json`.
- Required specialists: bounded R1-R3 reviewers, explicit Fable review, native `verification-before-completion`.
- Final predeploy verdict: `VERIFICATION_PASS`, `VERIFICATION_HOLD`, or `VERIFICATION_BLOCKED`; live proof remains pending until exact deployment approval.
