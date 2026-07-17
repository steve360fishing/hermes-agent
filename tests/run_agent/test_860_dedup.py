"""Tests for issue #860 — SQLite session transcript deduplication.

Verifies that:
1. _flush_messages_to_session_db uses _last_flushed_db_idx to avoid re-writing
2. Multiple _persist_session calls don't duplicate messages
3. append_to_transcript(skip_db=True) skips SQLite but writes JSONL
4. The gateway doesn't double-write messages the agent already persisted
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch



# ---------------------------------------------------------------------------
# Test: _flush_messages_to_session_db only writes new messages
# ---------------------------------------------------------------------------

class TestFlushDeduplication:
    """Verify _flush_messages_to_session_db tracks what it already wrote."""

    def _make_agent(self, session_db):
        """Create a minimal AIAgent with a real session DB."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="test-session-860",
                skip_context_files=True,
                skip_memory=True,
            )
        # Simulate lazy session creation (normally done by run_conversation)
        agent._ensure_db_session()
        return agent

    def test_flush_writes_only_new_messages(self):
        """First flush writes all new messages, second flush writes none."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)
            try:
                agent = self._make_agent(db)

                conversation_history = [
                    {"role": "user", "content": "old message"},
                ]
                messages = list(conversation_history) + [
                    {"role": "user", "content": "new question"},
                    {"role": "assistant", "content": "new answer"},
                ]

                # First flush — should write 2 new messages
                agent._flush_messages_to_session_db(messages, conversation_history)

                rows = db.get_messages(agent.session_id)
                assert len(rows) == 2, f"Expected 2 messages, got {len(rows)}"

                # Second flush with SAME messages — should write 0 new messages
                agent._flush_messages_to_session_db(messages, conversation_history)

                rows = db.get_messages(agent.session_id)
                assert len(rows) == 2, f"Expected still 2 messages after second flush, got {len(rows)}"
            finally:
                db.close()

    def test_flush_writes_incrementally(self):
        """Messages added between flushes are written exactly once."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)
            try:
                agent = self._make_agent(db)

                conversation_history = []
                messages = [
                    {"role": "user", "content": "hello"},
                ]

                # First flush — 1 message
                agent._flush_messages_to_session_db(messages, conversation_history)
                rows = db.get_messages(agent.session_id)
                assert len(rows) == 1

                # Add more messages
                messages.append({"role": "assistant", "content": "hi there"})
                messages.append({"role": "user", "content": "follow up"})

                # Second flush — should write only 2 new messages
                agent._flush_messages_to_session_db(messages, conversation_history)
                rows = db.get_messages(agent.session_id)
                assert len(rows) == 3, f"Expected 3 total messages, got {len(rows)}"
            finally:
                db.close()

    def test_persist_session_multiple_calls_no_duplication(self):
        """Multiple _persist_session calls don't duplicate DB entries."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)
            try:
                agent = self._make_agent(db)

                conversation_history = [{"role": "user", "content": "old"}]
                messages = list(conversation_history) + [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]

                # Simulate multiple persist calls (like the agent's many exit paths)
                for _ in range(5):
                    agent._persist_session(messages, conversation_history)

                rows = db.get_messages(agent.session_id)
                assert len(rows) == 4, f"Expected 4 messages, got {len(rows)} (duplication bug!)"
            finally:
                db.close()

    def test_sanitized_artifact_history_is_not_reappended(self):
        """Gateway reuse plus filtering keeps durable message identities."""
        from agent.task_execution_contract import build_task_execution_contract
        from gateway.run import GatewayRunner
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)
            try:
                agent = self._make_agent(db)
                artifact_path = (
                    "/opt/data/hermes-artifacts/"
                    "68fc2177cc474858a2c9b998f3b8be6f/recovery.txt"
                )
                history = [
                    {"role": "user", "content": "Create recovery.txt and attach it."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "artifact-call",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": (
                                        '{"path":"' + artifact_path + '","content":"x"}'
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "artifact-call",
                        "tool_name": "write_file",
                        "content": '{"resolved_path":"' + artifact_path + '"}',
                    },
                    {"role": "assistant", "content": "MEDIA:" + artifact_path},
                    {"role": "user", "content": "Safe prior question."},
                    {"role": "assistant", "content": "Safe prior answer."},
                ]
                for message in history:
                    db.append_message(
                        agent.session_id,
                        role=message["role"],
                        content=message.get("content"),
                        tool_name=message.get("tool_name"),
                        tool_calls=message.get("tool_calls"),
                        tool_call_id=message.get("tool_call_id"),
                    )

                stale_contract = build_task_execution_contract(
                    "Return only a paste-ready prompt.",
                    task_id="prior-artifact-mode",
                    platform="telegram",
                )
                agent._task_execution_contract = stale_contract
                GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)
                assert stale_contract.active is False
                assert agent._task_execution_contract is None

                with patch.dict(
                    os.environ,
                    {"HERMES_ARTIFACT_ROOT": "/opt/data/hermes-artifacts"},
                ):
                    contract = build_task_execution_contract(
                        "Continue normal work.",
                        task_id="dedup-after-artifact",
                        platform="telegram",
                    )
                    messages = contract.bound_conversation_history(history)

                assert messages == history[4:]
                assert messages[0] is history[4]
                assert messages[1] is history[5]
                messages.append({"role": "user", "content": "Continue normal work."})
                agent._flush_messages_to_session_db(messages, history)

                rows = db.get_messages(agent.session_id)
                assert len(rows) == 7
                assert rows[-1]["content"] == "Continue normal work."
            finally:
                db.close()

    def test_sanitized_api_history_never_overwrites_enabled_json_snapshot(self):
        """API-only elision must not erase durable optional JSON history."""
        from agent.task_execution_contract import build_task_execution_contract
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db = SessionDB(db_path=root / "test.db")
            try:
                agent = self._make_agent(db)
                agent._session_json_enabled = True
                agent.logs_dir = root / "sessions"
                agent.logs_dir.mkdir()
                artifact_path = (
                    "/opt/data/hermes-artifacts/"
                    "68fc2177cc474858a2c9b998f3b8be6f/recovery.txt"
                )
                history = [
                    {"role": "user", "content": "Create recovery.txt and attach it."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "artifact-call",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": (
                                        '{"path":"' + artifact_path + '","content":"x"}'
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "artifact-call",
                        "tool_name": "write_file",
                        "content": '{"resolved_path":"' + artifact_path + '"}',
                    },
                    {"role": "assistant", "content": "MEDIA:" + artifact_path},
                    {"role": "user", "content": "Safe prior question."},
                    {"role": "assistant", "content": "Safe prior answer."},
                ]
                for message in history:
                    db.append_message(
                        agent.session_id,
                        role=message["role"],
                        content=message.get("content"),
                        tool_name=message.get("tool_name"),
                        tool_calls=message.get("tool_calls"),
                        tool_call_id=message.get("tool_call_id"),
                    )

                with patch.dict(
                    os.environ,
                    {"HERMES_ARTIFACT_ROOT": "/opt/data/hermes-artifacts"},
                ):
                    contract = build_task_execution_contract(
                        "Continue normal work.",
                        task_id="json-snapshot-after-artifact",
                        platform="telegram",
                    )
                    messages = contract.bound_conversation_history(history)

                assert messages == history[4:]
                current_user = {"role": "user", "content": "Continue normal work."}
                messages.append(current_user)
                agent._session_log_durable_history = list(history)
                agent._session_log_current_turn_user = current_user
                agent._persist_user_message_idx = len(messages) - 1

                for index in range(8):
                    messages.append(
                        {"role": "assistant", "content": f"current response {index}"}
                    )
                    agent._persist_session(messages, history)

                snapshot_path = agent.logs_dir / f"session_{agent.session_id}.json"
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                contents = [message.get("content") for message in snapshot["messages"]]
                assert snapshot["message_count"] == len(history) + 9
                assert contents.count("Create recovery.txt and attach it.") == 1
                assert contents.count("MEDIA:" + artifact_path) == 1
                assert contents.count("Continue normal work.") == 1
                assert contents[-1] == "current response 7"

                rows = db.get_messages(agent.session_id)
                assert len(rows) == len(history) + 9
                assert sum(
                    row["content"] == "Continue normal work." for row in rows
                ) == 1
            finally:
                db.close()

    def test_flush_reset_after_compression(self):
        """After compression creates a new session, flush index resets."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)
            try:
                agent = self._make_agent(db)

                # Write some messages
                messages = [
                    {"role": "user", "content": "msg1"},
                    {"role": "assistant", "content": "reply1"},
                ]
                agent._flush_messages_to_session_db(messages, [])

                old_session = agent.session_id
                assert agent._last_flushed_db_idx == 2

                # Simulate what _compress_context does: new session, reset idx
                agent.session_id = "compressed-session-new"
                db.create_session(session_id=agent.session_id, source="test")
                agent._last_flushed_db_idx = 0

                # Now flush compressed messages to new session
                compressed_messages = [
                    {"role": "user", "content": "summary of conversation"},
                ]
                agent._flush_messages_to_session_db(compressed_messages, [])

                new_rows = db.get_messages(agent.session_id)
                assert len(new_rows) == 1

                # Old session should still have its 2 messages
                old_rows = db.get_messages(old_session)
                assert len(old_rows) == 2
            finally:
                db.close()


# ---------------------------------------------------------------------------
# Test: append_to_transcript skip_db parameter
# ---------------------------------------------------------------------------

class TestAppendToTranscriptSkipDb:
    """Verify skip_db=True skips the SQLite write."""

    def test_skip_db_prevents_sqlite_write(self, tmp_path):
        """With skip_db=True and a real DB, message does NOT appear in SQLite."""
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from hermes_state import SessionDB

        db_path = tmp_path / "test_skip.db"
        db = SessionDB(db_path=db_path)

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        session_id = "test-skip-db-real"
        db.create_session(session_id=session_id, source="test")

        msg = {"role": "assistant", "content": "hello world"}
        store.append_to_transcript(session_id, msg, skip_db=True)

        # SQLite should NOT have the message
        rows = db.get_messages(session_id)
        assert len(rows) == 0, f"Expected 0 DB rows with skip_db=True, got {len(rows)}"

    def test_default_writes_to_sqlite(self, tmp_path):
        """Without skip_db, message appears in SQLite."""
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from hermes_state import SessionDB

        db_path = tmp_path / "test_both.db"
        db = SessionDB(db_path=db_path)

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        session_id = "test-default-write"
        db.create_session(session_id=session_id, source="test")

        msg = {"role": "user", "content": "test message"}
        store.append_to_transcript(session_id, msg)

        # SQLite should have the message
        rows = db.get_messages(session_id)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Test: _last_flushed_db_idx initialization
# ---------------------------------------------------------------------------

class TestFlushIdxInit:
    """Verify _last_flushed_db_idx is properly initialized."""

    def test_init_zero(self):
        """Agent starts with _last_flushed_db_idx = 0."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert agent._last_flushed_db_idx == 0

    def test_no_session_db_noop(self):
        """Without session_db, flush is a no-op and doesn't crash."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        messages = [{"role": "user", "content": "test"}]
        agent._flush_messages_to_session_db(messages, [])
        # Should not crash, idx should remain 0
        assert agent._last_flushed_db_idx == 0
