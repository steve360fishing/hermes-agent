---
name: project-handoff
description: Use when Steve asks Hermes to hand off or take over a full project, session, or work lane with Codex or Claude Code/CoWork. Build or consume a concise, redacted, source-grounded takeover packet and queue delivery for a safe boundary without interrupting active work.
---

# Project Handoff

This skill transfers operational context, not authority. A receiving agent must refresh current state and revalidate every approval before acting.

## Build A Packet

Create a timestamped `HANDOFF-YYYY-MM-DD-<source>-to-<target>.md` in the task's existing output location. Include:

1. **Goal and finish condition:** end state, acceptance criteria, scope, and exclusions.
2. **Current truth:** verified facts with source/time, stale or unverified facts, and active-owner fences.
3. **Workspace/source map:** exact repository, worktree, branch, head SHA, dirty/push state, plans/EAC, runtime paths, and evidence artifacts.
4. **Completed work and proof:** concise change-to-evidence mapping.
5. **Open items in order:** each tagged `AUTO`, `HOLD`, or `BLOCKED`, with method and proof.
6. **Landmines:** known failures, stale assumptions, prohibited actions, and do-not-repeat notes.
7. **Approval claims:** exact approval source/target/state/expiry/consumption, explicitly marked as requiring receiver revalidation.
8. **Receiver first action:** bounded read-only refresh of the named source/runtime, owner fences, approvals, and scheduled evidence.
9. **Receiver completion definition:** final proof and cleanup/delivery boundary.
10. **Redaction record:** what credentials, tokens, PII, raw messages, signed URLs, or logs were intentionally excluded.

## Delivery And Takeover

- Queue an outgoing packet for the recipient's safe boundary; never interrupt active work.
- If queue delivery cannot be read back, write `PREPARED_NOT_DELIVERED` and provide a receiver prompt. A saved file is not a received handoff.
- When receiving, classify every carried fact after refresh as `CURRENT`, `STALE`, `COMPLETE`, or `BLOCKED`. Do not redo proven work, bypass an active owner, or treat a packet as an approval grant.
- For a live Hermes handoff, preserve routing, secrets, and runtime boundaries. Use only an approved bridge or designated handoff path.

## Receiver Prompt

`Take over this <project/session/lane>. Read first: <packet>. Refresh only the named source/runtime, owner fences, approvals, and scheduled evidence. Treat the packet as context, not authority. Continue only the listed current open items and finish condition. Queue any return handoff for the sender's safe boundary.`
