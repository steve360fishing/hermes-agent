"""Independent s6 reporter for signed Hermes quiescence snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from agent.rescue_plane_core import RescueExecutionTelemetry, _canonical_snapshot


class QuiescenceReporter:
    def __init__(self, *, telemetry_path: Path, output_path: Path, key_path: Path, key_id: str, source_sha: str, image_id: str, gateway_state: Callable[[], tuple[int | None, str]]) -> None:
        self.telemetry_path = telemetry_path
        self.output_path = output_path
        self.key_path = key_path
        self.key_id = key_id
        self.source_sha = source_sha
        self.image_id = image_id
        self.gateway_state = gateway_state
        self._telemetry = RescueExecutionTelemetry()

    def emit_once(self, *, now: float | None = None) -> dict:
        state = json.loads(self.telemetry_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != "hermes-rescue-turn-telemetry-v1":
            raise ValueError("invalid rescue telemetry")
        for turn in state.get("turns", []):
            self._telemetry.start_turn(turn["turn_id"], lane=turn["lane"], artifact_requested=turn["artifact_requested"], now=turn["started_at"])
            if "completed_at" in turn:
                self._telemetry.finish_turn(turn["turn_id"], now=turn["completed_at"])
        # Counters above are authoritative only when backed by the common writer.
        self._telemetry._active_tools = int(state["active_tool_count"])
        self._telemetry._active_provider_actions = int(state["active_provider_action_count"])
        key = self.key_path.read_bytes()
        gateway_pid, gateway_state = self.gateway_state()
        # The s6 run script launches a fresh Python process every interval;
        # seed from the last signed file so verifier replay protection sees a
        # monotonic sequence across those independent reporter invocations.
        try:
            prior = json.loads(self.output_path.read_text(encoding="utf-8"))
            self._telemetry._sequence = max(0, int(prior.get("sequence", 0)))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        snapshot = self._telemetry.quiescence_snapshot(key=key, key_id=self.key_id, gateway_pid=gateway_pid, gateway_state=gateway_state, source_sha=self.source_sha, image_id=self.image_id, now=now)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(f".{self.output_path.name}.tmp")
        temporary.write_bytes(_canonical_snapshot(snapshot))
        temporary.replace(self.output_path)
        return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=Path("/run/hermes-rescue/turn-telemetry-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("/run/hermes-rescue/quiescence-v1.json"))
    parser.add_argument("--key", type=Path, default=Path("/run/secrets/hermes-rescue-quiescence-hmac-v1"))
    parser.add_argument("--key-id-file", type=Path, default=Path("/run/hermes-rescue/key-id"))
    parser.add_argument("--source-sha-file", type=Path, default=Path("/run/hermes-rescue/source-sha"))
    parser.add_argument("--image-id-file", type=Path, default=Path("/run/hermes-rescue/image-id"))
    args = parser.parse_args()
    QuiescenceReporter(
        telemetry_path=args.telemetry,
        output_path=args.output,
        key_path=args.key,
        key_id=args.key_id_file.read_text(encoding="utf-8").strip(),
        source_sha=args.source_sha_file.read_text(encoding="utf-8").strip(),
        image_id=args.image_id_file.read_text(encoding="utf-8").strip(),
        gateway_state=lambda: (None, "unknown"),
    ).emit_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
