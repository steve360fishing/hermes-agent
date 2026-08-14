from __future__ import annotations

import os
import tempfile
import threading
from types import SimpleNamespace

os.environ.setdefault("LOCALAPPDATA", tempfile.gettempdir())
os.environ.setdefault("USERPROFILE", tempfile.gettempdir())

import pytest

from agent.conversation_loop import _clear_request_contract_after_turn
from agent.tournament_intent_contract import (
    TournamentIntentContract,
    _install_stable_runtime_muxes,
    bind_tournament_contract,
    clear_tournament_intent_contract,
    current_tournament_contract,
)
from gateway.run import _mint_gateway_turn_provenance


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
        turn_provenance=_mint_gateway_turn_provenance(
            SimpleNamespace(text="test direct request", message_id="message-1"),
            SimpleNamespace(user_id="steve", platform="telegram", profile="test", chat_id="chat-1", scope_id="telegram:test:chat-1:"),
            is_internal=False,
        ),
        turn_token=token,
    )


def test_turn_token_is_immutable_and_malformed_token_cannot_claim_active_turn():
    stale = _contract("a" * 32)
    active = _contract("b" * 32)
    agent = SimpleNamespace()
    bind_tournament_contract(active)

    with pytest.raises(AttributeError, match="immutable"):
        stale.turn_token = active.turn_token

    object.__setattr__(stale, "turn_token", None)
    stale.cleanup(agent)
    assert current_tournament_contract() is active
    assert active.closed is False
    active.cleanup(agent)


def test_concurrent_turns_keep_request_local_contracts_without_serializing_agent():
    agent = SimpleNamespace(
        _task_execution_contract=None,
        _tool_guardrails=SimpleNamespace(set_execution_contract=lambda _value: None),
    )
    entered_first = threading.Event()
    release_first = threading.Event()
    entered_second = threading.Event()
    release_second = threading.Event()
    observed = {}
    active_count = 0
    max_active = 0
    state_lock = threading.Lock()

    @_clear_request_contract_after_turn
    def run(_agent, label: str):
        nonlocal active_count, max_active
        contract = _contract(("a" if label == "first" else "b") * 32)
        bind_tournament_contract(contract)
        with state_lock:
            active_count += 1
            max_active = max(max_active, active_count)
        observed[label] = current_tournament_contract()
        if label == "first":
            entered_first.set()
            assert release_first.wait(timeout=2)
        else:
            entered_second.set()
            assert release_second.wait(timeout=2)
        with state_lock:
            active_count -= 1

    first = threading.Thread(target=run, args=(agent, "first"))
    second = threading.Thread(target=run, args=(agent, "second"))
    first.start()
    assert entered_first.wait(timeout=2)
    second.start()
    assert entered_second.wait(timeout=2)
    release_second.set()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert max_active == 2
    assert observed["first"].turn_token != observed["second"].turn_token
    assert current_tournament_contract() is None


def test_stable_callback_and_persistence_muxes_never_need_per_turn_restore():
    contract = _contract("c" * 32)
    streamed = []
    persisted = []
    agent = SimpleNamespace(
        stream_delta_callback=streamed.append,
        _persist_session=lambda messages, history: persisted.append((messages, history)),
    )
    base_stream, base_persist = _install_stable_runtime_muxes(agent)
    stream_mux = agent.stream_delta_callback
    persist_mux = agent._persist_session
    bind_tournament_contract(contract)
    stream_mux("unverified public bytes")
    persist_mux([{"role": "assistant", "content": "candidate"}], None)
    assert streamed == []
    assert persisted == []
    assert contract.pending_persistence is not None
    assert contract.cleanup(agent) is True
    assert contract.closed is True
    assert agent.stream_delta_callback is stream_mux
    assert agent._persist_session is persist_mux
    assert callable(base_stream)
    assert callable(base_persist)
    stream_mux("ordinary private response")
    persist_mux([{"role": "assistant", "content": "ordinary"}], None)
    assert streamed == ["ordinary private response"]
    assert persisted[-1][0][-1]["content"] == "ordinary"
    _install_stable_runtime_muxes(agent)
    assert agent.stream_delta_callback is stream_mux
    assert agent._persist_session is persist_mux


def test_tournament_context_propagates_to_concurrent_tool_worker():
    from tools.thread_context import propagate_context_to_thread

    contract = _contract("f" * 32)
    observed = []
    bind_tournament_contract(contract)
    worker = threading.Thread(
        target=propagate_context_to_thread(
            lambda: observed.append(current_tournament_contract())
        )
    )
    worker.start()
    worker.join(timeout=2)
    assert observed == [contract]
    contract.cleanup(SimpleNamespace())


def test_persistent_cleanup_failure_quarantines_only_the_old_candidate_and_admits_next_turn():
    contract = _contract("d" * 32)

    class PersistentFailureAgent(SimpleNamespace):
        def __setattr__(self, name, value):
            if name == "stream_delta_callback" and hasattr(self, name):
                raise RuntimeError("persistent restore failure")
            super().__setattr__(name, value)

    contract.buffer_callback = contract.buffer
    contract.original_stream_delta_callback = lambda _value: None
    agent = PersistentFailureAgent(
        stream_delta_callback=contract.buffer_callback,
    )
    bind_tournament_contract(contract)
    assert clear_tournament_intent_contract(agent) is True
    assert current_tournament_contract() is None
    assert contract.closed is True


def test_exception_closes_only_current_contract_and_next_private_turn_runs():
    agent = SimpleNamespace(
        _task_execution_contract=None,
        _tool_guardrails=SimpleNamespace(set_execution_contract=lambda _value: None),
    )
    contract = _contract("e" * 32)
    next_turn_observed = []

    @_clear_request_contract_after_turn
    def interrupted(_agent):
        bind_tournament_contract(contract)
        raise RuntimeError("injected interruption")

    @_clear_request_contract_after_turn
    def private_turn(_agent):
        next_turn_observed.append(current_tournament_contract())
        return "private answer"

    with pytest.raises(RuntimeError, match="injected interruption"):
        interrupted(agent)
    assert contract.closed is True
    assert current_tournament_contract() is None
    assert private_turn(agent) == "private answer"
    assert next_turn_observed == [None]
