from __future__ import annotations

import os
import tempfile
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("LOCALAPPDATA", tempfile.gettempdir())
os.environ.setdefault("USERPROFILE", tempfile.gettempdir())

import pytest

from agent.conversation_loop import _clear_request_contract_after_turn
from agent.tournament_intent_contract import TournamentIntentContract
from agent.tournament_intent_contract import clear_tournament_intent_contract
from agent.turn_origin import TurnProvenance


def _contract(token: str) -> TournamentIntentContract:
    return TournamentIntentContract(
        state=__import__(
            "agent.tournament_intent_contract", fromlist=["TournamentIntentState"]
        ).TournamentIntentState.PUBLIC_FACING_DRAFT,
        task_id="task",
        session_id="session",
        destination="telegram:private",
        entrypoint="direct_public",
        actor_identity="steve",
        turn_provenance=TurnProvenance.authenticated_direct_user("steve"),
        turn_token=token,
    )


def test_turn_token_is_immutable_and_malformed_token_cannot_claim_active_turn():
    stale = _contract("a" * 32)
    active = _contract("b" * 32)
    agent = SimpleNamespace(_tournament_intent_contract=active)

    with pytest.raises(AttributeError, match="immutable"):
        stale.turn_token = active.turn_token

    object.__setattr__(stale, "turn_token", None)
    stale.cleanup(agent)
    assert agent._tournament_intent_contract is active
    assert active.closed is False


def test_shared_agent_turn_lifecycle_is_serialized_before_contract_mutation():
    agent = SimpleNamespace(
        _task_execution_contract=None,
        _tournament_intent_contract=None,
    )
    entered_first = threading.Event()
    release_first = threading.Event()
    entered_second = threading.Event()
    active_count = 0
    max_active = 0
    state_lock = threading.Lock()

    @_clear_request_contract_after_turn
    def run(_agent, label: str):
        nonlocal active_count, max_active
        with state_lock:
            active_count += 1
            max_active = max(max_active, active_count)
        if label == "first":
            entered_first.set()
            assert release_first.wait(timeout=2)
        else:
            entered_second.set()
        with state_lock:
            active_count -= 1

    first = threading.Thread(target=run, args=(agent, "first"))
    second = threading.Thread(target=run, args=(agent, "second"))
    first.start()
    assert entered_first.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    assert not entered_second.is_set()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert entered_second.is_set()
    assert max_active == 1


def test_cleanup_fault_restores_every_other_owned_surface_and_remains_retryable():
    contract = _contract("c" * 32)

    class RaisingGuardrail:
        def __init__(self):
            self._tournament_contract = contract
            self.calls = 0

        def set_tournament_contract(self, value):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("injected cleanup failure")
            self._tournament_contract = value

    guardrail = RaisingGuardrail()
    contract.buffer_callback = contract.buffer
    contract.original_stream_delta_callback = lambda _value: None
    contract.original_stream_callback = lambda _value: None
    contract.original_persist_session = lambda *_args: None
    contract.added_tool_schema = True
    contract.added_tool_schemas = {"tournament_truth_gate"}
    contract.added_valid_tool_name = True
    contract.added_valid_tool_names = {"tournament_truth_gate"}
    agent = SimpleNamespace(
        _tournament_intent_contract=contract,
        _tool_guardrails=guardrail,
        stream_delta_callback=contract.buffer_callback,
        _stream_callback=contract.buffer_callback,
        _persist_session=contract._defer_persistence,
        tools=[
            {"function": {"name": "memory"}},
            {"function": {"name": "tournament_truth_gate"}},
        ],
        valid_tool_names={"memory", "tournament_truth_gate"},
    )

    assert contract.cleanup(agent) is True
    assert contract.closed is True
    assert agent.stream_delta_callback is contract.original_stream_delta_callback
    assert agent._stream_callback is contract.original_stream_callback
    assert agent._persist_session is contract.original_persist_session
    assert guardrail._tournament_contract is None
    assert [tool["function"]["name"] for tool in agent.tools] == ["memory"]
    assert agent.valid_tool_names == {"memory"}


def test_persistent_cleanup_failure_blocks_next_contract_admission():
    contract = _contract("d" * 32)

    class PersistentFailureAgent(SimpleNamespace):
        def __setattr__(self, name, value):
            if name == "stream_delta_callback" and hasattr(self, name):
                raise RuntimeError("persistent restore failure")
            super().__setattr__(name, value)

    contract.buffer_callback = contract.buffer
    contract.original_stream_delta_callback = lambda _value: None
    agent = PersistentFailureAgent(
        _tournament_intent_contract=contract,
        _tool_guardrails=None,
        stream_delta_callback=contract.buffer_callback,
        _stream_callback=None,
        _persist_session=None,
        tools=[],
        valid_tool_names=set(),
    )

    assert clear_tournament_intent_contract(agent) is False
    assert agent._tournament_intent_contract is contract
    assert contract.closed is False
