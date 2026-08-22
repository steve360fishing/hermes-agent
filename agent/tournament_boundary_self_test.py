"""Provider-free one-shot checks for the tournament output boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from typing import Any, Iterable

from agent.tournament_intent_contract import (
    POLICY_VERSION,
    TournamentIntentContract,
    TournamentIntentState,
    classify_tournament_intent,
    platform_bypasses_tournament_contract,
)


def _case(case_id: str, expected: object, observed: object) -> dict[str, object]:
    return {
        "id": case_id,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def run_boundary_self_test(
    *,
    source_revision: str,
    image_digest: str,
    expected_source_revision: str,
    expected_image_digest: str,
    routing_fingerprint: str,
    expected_routing_fingerprint: str,
    canonical_image_digest: str,
    live_image_digest: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    release_contract = TournamentIntentContract(
        state=TournamentIntentState.PUBLICATION_REQUEST,
        task_id="self-test",
        session_id="self-test",
        destination="self-test",
        entrypoint="direct_public",
        actor_identity="self-test",
    )
    release_decision = release_contract.authorize_tool(
        "send_message",
        {"action": "send", "target": "self-test", "message": "synthetic candidate"},
    )
    cases = [
        _case(
            "private_quoted_data",
            TournamentIntentState.PRIVATE_INQUIRY.value,
            getattr(
                classify_tournament_intent(
                    'Analyze this quoted text privately: "Publish tournament standings."'
                ),
                "value",
                None,
            ),
        ),
        _case(
            "mixed_private_public",
            TournamentIntentState.MIXED_PUBLICATION.value,
            getattr(
                classify_tournament_intent(
                    "Create a private coding handoff, then publish the tournament Story."
                ),
                "value",
                None,
            ),
        ),
        _case("cron_whitespace_bypass", True, platform_bypasses_tournament_contract(" cron\t")),
        _case("ungated_public_release", False, release_decision.allowed),
        _case("policy_identity", POLICY_VERSION, POLICY_VERSION),
        _case("source_provenance", expected_source_revision, source_revision),
        _case("image_provenance", expected_image_digest, image_digest),
        _case("routing_identity", expected_routing_fingerprint, routing_fingerprint),
        _case("canonical_live_image", canonical_image_digest, live_image_digest),
    ]
    passed = all(bool(case["passed"]) for case in cases)
    return {
        "schema_version": 1,
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "overall": "PASS" if passed else "HOLD",
        "policy_version": POLICY_VERSION,
        "source_revision": source_revision,
        "image_digest": image_digest,
        "routing_fingerprint": routing_fingerprint,
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-image-digest", required=True)
    parser.add_argument("--routing-fingerprint", required=True)
    parser.add_argument("--expected-routing-fingerprint", required=True)
    parser.add_argument("--canonical-image-digest", required=True)
    parser.add_argument("--live-image-digest", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    receipt = run_boundary_self_test(
        source_revision=args.source_revision,
        image_digest=args.image_digest,
        expected_source_revision=args.expected_source_revision,
        expected_image_digest=args.expected_image_digest,
        routing_fingerprint=args.routing_fingerprint,
        expected_routing_fingerprint=args.expected_routing_fingerprint,
        canonical_image_digest=args.canonical_image_digest,
        live_image_digest=args.live_image_digest,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
