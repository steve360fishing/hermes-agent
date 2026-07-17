# Execution Boundary Audit

This audit prevents capability-reducing policy decisions from drifting away
from their enforcement, delivery, and cleanup paths. It covers the incident
class where a request contract can authorize an output route that a later
layer rejects, or where a turn-scoped restriction survives into another turn.

## Coverage denominator

`scripts/check_execution_boundaries.py` scans the exact union of:

- eight shipped Python entrypoints;
- ten runtime roots that can be loaded dynamically by profiles, plugins,
  providers, cron, gateways, or adapters; and
- every contract-specific rule path, even when it is outside an entrypoint or
  runtime root; and
- contract-specific semantic tokens in those files.

The checked-in registry currently has 231 source-specific records for the 111
discoverable source sites across five contract families. The counts are
recomputed by the checker; registry records may legitimately cover one site
under more than one contract.

| Contract | Required roles |
| --- | --- |
| Artifact request | decision, propagation, enforcement, recovery |
| Artifact delivery | allocation, validation, delivery, recovery |
| Safe mode | decision, enforcement, lifecycle |
| Incident fallback | decision, enforcement, recovery |
| Cron restrictions | decision, enforcement, recovery |

The scan is intentionally a conservative static boundary audit, not a claim
that Python import analysis can enumerate every runtime plugin. CI fails when:

- a discovered site has no registry disposition;
- an entrypoint or dynamic root is removed from the denominator;
- a registered symbol or identifier is stale;
- a contract is missing a required role; or
- the registry references a missing source path.
- a tracked source cannot be read or parsed;
- a lifecycle edge is duplicate, reversed, or outside its contract's explicit
  role-transition order; or
- duplicate JSON keys or a non-redacted inventory schema are supplied.

Reviewed exclusions require a written rationale. New sites are never accepted
implicitly.

## Documented boundary families

The registry records source-specific evidence rather than assigning generic
labels. In the owned runtime paths, the contract review also maps discoverable
recovery, session, credential/provider, profile/plugin, channel, deployment/
watchdog, resource, approval/lease, clock/network, and backup/restore
boundaries to their concrete source owners. A family is added to the enforced
registry only when a path restriction and lifecycle relationship can be named;
otherwise this document records it as a review family rather than pretending
that a broad token scan proves coverage.

## Cross-layer lifecycle checks

| Failure boundary | Required result | Regression proof |
| --- | --- | --- |
| Artifact allocation/preflight | Model is not invoked when no compatible writable root exists | `tests/run_agent/test_run_agent.py` artifact preflight cases |
| Writer rejection | Contract is cleared and the next turn is unrestricted | `tests/agent/test_turn_finalizer_cleanup_guard.py` |
| Delivery rejection | Contract is cleared; no silent inline substitution | finalizer cleanup and Telegram document tests |
| Session reload | Request-local contract is rebuilt or cleared, never resurrected from history | `tests/agent/test_turn_context.py` |
| TXT delivery | Filename, bytes, and MIME remain aligned through the real writer and mocked Telegram network boundary | task contract and Telegram document tests |
| Protected path | Credentials, traversal, and symlink escapes remain denied | task contract and platform-base tests |
| Incident fallback | Timeout policy cannot silently widen provider authority | OpenRouter fallback guard tests |
| Cron restrictions | Job-local allowlists cannot bypass globally disabled toolsets | cron scheduler tests |

## Findings and repairs

### HOME identity mismatch

On Windows, `os.path.expanduser("~")` prefers `USERPROFILE` even when Hermes is
deliberately running under a different `HOME`. The delivery denylist could
therefore evaluate a different home directory from the writer/runtime policy.
The shared delivery validator now uses `HOME` when explicitly configured and
falls back to platform expansion only when it is absent.

### Control characters in MEDIA paths

MEDIA tag normalization could retain NUL and other control characters long
enough for a malformed path to enter the extraction list. Normalization now
rejects every ASCII control character before path validation or delivery.

### Cross-platform test gaps

The affected tests assumed POSIX separators and unrestricted symlink creation.
They now compare `Path.parts` and skip only when the operating system denies
symlink creation. This keeps the security assertions meaningful on Linux and
Windows.

## Effective runtime manifest

The checker can emit a sanitized manifest with `--manifest-out`, plus explicit
`--plugin-inventory` and `--transport-inventory` JSON inputs. It records:

- source commit and tree;
- SHA-256 hashes for all shipped entrypoints and registry-declared boundary
  core modules, plus the checker, the actual supplied registry, and the actual
  supplied inventories;
- explicit effective plugin and transport inventories using only
  `{ "plugins": ["name"] }` and `{ "transports": ["name"] }` schemas; and
- configured environment names and presence only.

No environment values are emitted, including values for non-secret names. A
manifest fails rather than silently omitting a declared core module. The
manifest is evidence, not configuration, and it does not mutate the runtime.

## Live read-only snapshot (2026-07-16)

- Container: `hermes-sportfish-newsletter`, running, restart count 0, OOM false.
- Image source label: `a064a4fa896df69c9c069fa0dbe6cf77645ef624`.
- `/opt/data` is a writable Docker volume.
- `/opt/data/hermes-artifacts` exists as a `0700` directory owned by the Hermes
  runtime user.
- `/tmp/hermes-artifacts` is absent, so the reviewed persistent fallback is the
  effective artifact root.
- The live hashes for task-contract, turn-context, turn-finalizer,
  platform-base, and Telegram adapter exactly match the canonical Git bytes at
  the source label.
- Only environment-variable names/presence were inspected; values were not
  printed or stored.

No live file, environment variable, container, provider, Telegram message, or
session state was changed during this audit.

## Deployment boundary

Source merge, image build, VPS promotion/recreation, and a Telegram canary are
separate gates. The post-deploy canary should create one harmless `.txt`
document, verify its bytes and MIME metadata, deliver it once, and then prove a
normal subsequent turn has ordinary tools. Any mismatch requires rollback to
the previously recorded image without retrying the canary automatically.
