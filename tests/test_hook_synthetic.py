"""Synthetic integration tests - exercises the hook end-to-end with realistic event data.

Uses the mock TCP server (no real CXDB needed). Event data is based on
actual Amplifier session events extracted from a real session.
"""

from __future__ import annotations

import pytest

from amplifier_module_hooks_cxdb_events.hook import CXDBEventHook
from amplifier_module_hooks_cxdb_events.protocol import CXDBTcpClient


SESSION_ID = "9839b7c5-007b-4b54-acf4-a7e8f114c358"
CHILD_SESSION_ID = (
    "9839b7c5-007b-4b54-acf4-a7e8f114c358-55a6dde8d5a74951_foundation:session-analyst"
)

REALISTIC_EVENTS: list[tuple[str, dict]] = [
    (
        "session:start",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
        },
    ),
    (
        "prompt:submit",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "prompt": "use session analyzer to see where left off with session 8725de75 before we proceed",
        },
    ),
    (
        "execution:start",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "prompt": "use session analyzer to see where left off with session 8725de75 before we proceed",
        },
    ),
    (
        "provider:request",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "provider": "anthropic",
            "iteration": 1,
        },
    ),
    (
        "provider:response",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "provider": "anthropic",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 238,
                "total_tokens": 241,
                "reasoning_tokens": None,
                "cache_read_input_tokens": 17307,
                "cache_creation_input_tokens": 29490,
            },
        },
    ),
    (
        "content_block:end",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "block_index": 0,
            "total_blocks": 2,
            "block_type": "text",
            "block": "I'll delegate to the session analyst to investigate where you left off.",
        },
    ),
    (
        "tool:pre",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "tool_name": "delegate",
            "tool_call_id": "toolu_01BjimyKD6u3gQBMgiDYwGNL",
            "tool_input": {
                "agent": "foundation:session-analyst",
                "instruction": "Find and analyze the session with ID starting with 8725de75",
                "context_depth": "none",
            },
            "parallel_group_id": "pg-001",
        },
    ),
    (
        "task:agent_spawned",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "agent": "foundation:session-analyst",
            "sub_session_id": CHILD_SESSION_ID,
            "parent_session_id": SESSION_ID,
            "context_depth": "none",
            "context_scope": "conversation",
        },
    ),
    (
        "task:agent_completed",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "agent": "foundation:session-analyst",
            "sub_session_id": CHILD_SESSION_ID,
            "parent_session_id": SESSION_ID,
            "success": True,
        },
    ),
    (
        "tool:post",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "tool_name": "delegate",
            "tool_call_id": "toolu_01BjimyKD6u3gQBMgiDYwGNL",
            "tool_input": {
                "agent": "foundation:session-analyst",
                "instruction": "Find and analyze the session",
            },
            "result": {
                "success": True,
                "output": {
                    "response": "## Session Analysis: 8725de75\n\nSession was about CXDB Hook Module...",
                    "session_id": CHILD_SESSION_ID,
                    "agent": "foundation:session-analyst",
                    "turn_count": 1,
                },
            },
            "parallel_group_id": "pg-001",
        },
    ),
    (
        "provider:request",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "provider": "anthropic",
            "iteration": 2,
        },
    ),
    (
        "provider:response",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "provider": "anthropic",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 850,
                "cache_read_input_tokens": 46797,
                "cache_creation_input_tokens": 5200,
            },
        },
    ),
    (
        "content_block:end",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "block_index": 0,
            "total_blocks": 1,
            "block_type": "text",
            "block": "Here's where you left off:\n\n**Session 8725de75** - CXDB Hook Module Implementation",
        },
    ),
    (
        "execution:end",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
        },
    ),
    (
        "prompt:complete",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "prompt": "use session analyzer to see where left off",
            "response": "Here's where you left off...",
        },
    ),
    (
        "orchestrator:complete",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "orchestrator": "loop-streaming",
            "turn_count": 2,
        },
    ),
]

VARIANT_EVENTS: list[tuple[str, dict]] = [
    (
        "session:start:raw",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "mount_plan": {
                "providers": ["anthropic"],
                "tools": ["delegate", "read_file"],
            },
        },
    ),
    (
        "session:start:debug",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
            "mount_plan": {"providers": ["anthropic"]},
        },
    ),
    (
        "session:start",
        {
            "session_id": SESSION_ID,
            "parent_id": None,
        },
    ),
]

CHILD_SESSION_EVENTS: list[tuple[str, dict]] = [
    (
        "session:start",
        {
            "session_id": CHILD_SESSION_ID,
            "parent_id": SESSION_ID,
        },
    ),
    (
        "prompt:submit",
        {
            "session_id": CHILD_SESSION_ID,
            "parent_id": SESSION_ID,
            "prompt": "Find and analyze the session with ID starting with 8725de75",
        },
    ),
    (
        "provider:request",
        {
            "session_id": CHILD_SESSION_ID,
            "parent_id": SESSION_ID,
            "provider": "anthropic",
            "iteration": 1,
        },
    ),
    (
        "tool:pre",
        {
            "session_id": CHILD_SESSION_ID,
            "parent_id": SESSION_ID,
            "tool_name": "bash",
            "tool_input": {"command": "find ~/.amplifier -name 'metadata.json'"},
        },
    ),
    (
        "tool:post",
        {
            "session_id": CHILD_SESSION_ID,
            "parent_id": SESSION_ID,
            "tool_name": "bash",
            "result": "/home/user/.amplifier/projects/.../metadata.json",
        },
    ),
    (
        "content_block:end",
        {
            "session_id": CHILD_SESSION_ID,
            "parent_id": SESSION_ID,
            "block_type": "text",
            "block": "## Session Analysis\n\nFound the session at the expected location.",
        },
    ),
    (
        "orchestrator:complete",
        {
            "session_id": CHILD_SESSION_ID,
            "parent_id": SESSION_ID,
            "orchestrator": "loop-streaming",
            "turn_count": 1,
        },
    ),
]


def _make_hook(mock_tcp_server, session_id=SESSION_ID, parent_id=None):
    """Create a hook for testing."""
    return CXDBEventHook(
        client=CXDBTcpClient("127.0.0.1", mock_tcp_server.port),
        config={"cxdb_host": "127.0.0.1", "cxdb_http_port": 19999},
        session_id=session_id,
        parent_id=parent_id,
        root_session_id=SESSION_ID,
    )


class TestFullSessionReplay:
    """Replay a full realistic session through the hook."""

    @pytest.mark.asyncio
    async def test_replay_all_events(self, mock_tcp_server):
        """All events from a realistic session flow through without errors."""
        hook = _make_hook(mock_tcp_server)
        for event_name, data in REALISTIC_EVENTS:
            result = await hook.handle_event(event_name, data)
            assert result.action == "continue", f"Event {event_name} blocked"
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_hook_initializes_on_first_event(self, mock_tcp_server):
        """Hook lazily initializes on the first event."""
        hook = _make_hook(mock_tcp_server)
        assert not hook.initialized
        await hook.handle_event(*REALISTIC_EVENTS[0])
        assert hook.initialized

    @pytest.mark.asyncio
    async def test_contexts_created(self, mock_tcp_server):
        """Both CXDB contexts are created during session replay."""
        hook = _make_hook(mock_tcp_server)
        await hook.handle_event(*REALISTIC_EVENTS[0])
        assert hook.turns_context_id is not None
        assert hook.everything_context_id is not None
        assert hook.turns_context_id != hook.everything_context_id
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_turn_accumulation_and_flush(self, mock_tcp_server):
        """Turns accumulate and flush on orchestrator:complete."""
        hook = _make_hook(mock_tcp_server)
        for event_name, data in REALISTIC_EVENTS[:-1]:
            await hook.handle_event(event_name, data)
        await hook.handle_event(*REALISTIC_EVENTS[-1])
        assert hook._turn_accumulator.flush() is None
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_tool_calls_accumulated(self, mock_tcp_server):
        """Tool calls (delegate) are accumulated during the session."""
        hook = _make_hook(mock_tcp_server)
        for event_name, data in REALISTIC_EVENTS[:10]:
            await hook.handle_event(event_name, data)
        tool_calls = hook._turn_accumulator._current.tool_calls
        assert len(tool_calls) >= 1
        assert tool_calls[0].tool_name == "delegate"
        assert tool_calls[0].has_result
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_provider_metrics_captured(self, mock_tcp_server):
        """Provider metrics with Anthropic token math are captured."""
        hook = _make_hook(mock_tcp_server)
        for event_name, data in REALISTIC_EVENTS[:5]:
            await hook.handle_event(event_name, data)
        metrics = hook._turn_accumulator._current.metrics
        assert metrics is not None
        assert metrics["provider"] == "anthropic"
        assert metrics["total_input_tokens"] == 3 + 17307 + 29490
        assert metrics["output_tokens"] == 238
        await hook.cleanup()


class TestVariantDeduplication:
    """Test variant dedup with realistic variant events."""

    @pytest.mark.asyncio
    async def test_raw_processed_others_suppressed(self, mock_tcp_server):
        """When :raw arrives first, :debug and base are suppressed."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        for event_name, data in VARIANT_EVENTS:
            result = await hook.handle_event(event_name, data)
            assert result.action == "continue"
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_dedup_resets_on_orchestrator_complete(self, mock_tcp_server):
        """Variant tracking resets after orchestrator:complete."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event(
            "session:start:raw",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
            },
        )
        await hook.handle_event(
            "orchestrator:complete",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "orchestrator": "loop-streaming",
                "turn_count": 1,
            },
        )
        assert hook._variant_dedup.should_process("session:start:raw")
        await hook.cleanup()


class TestChildSessionEvents:
    """Test child session events flowing through the hook."""

    @pytest.mark.asyncio
    async def test_child_session_replay(self, mock_tcp_server):
        """Child session events flow through without errors."""
        hook = _make_hook(
            mock_tcp_server, session_id=CHILD_SESSION_ID, parent_id=SESSION_ID
        )
        for event_name, data in CHILD_SESSION_EVENTS:
            result = await hook.handle_event(event_name, data)
            assert result.action == "continue"
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_child_detected_as_non_root(self, mock_tcp_server):
        """Child session hook is detected as non-root."""
        hook = _make_hook(
            mock_tcp_server, session_id=CHILD_SESSION_ID, parent_id=SESSION_ID
        )
        assert not hook.is_root

    @pytest.mark.asyncio
    async def test_child_agent_name_extracted(self, mock_tcp_server):
        """Agent name extracted from child session ID."""
        hook = _make_hook(
            mock_tcp_server, session_id=CHILD_SESSION_ID, parent_id=SESSION_ID
        )
        assert hook._agent_name == "foundation:session-analyst"

    @pytest.mark.asyncio
    async def test_child_accumulates_and_flushes(self, mock_tcp_server):
        """Child session accumulates turns and flushes on orchestrator:complete."""
        hook = _make_hook(
            mock_tcp_server, session_id=CHILD_SESSION_ID, parent_id=SESSION_ID
        )
        for event_name, data in CHILD_SESSION_EVENTS:
            await hook.handle_event(event_name, data)
        assert hook._turn_accumulator.flush() is None
        await hook.cleanup()


class TestStragglerSuppression:
    """Test straggler suppression with realistic event sequence."""

    @pytest.mark.asyncio
    async def test_llm_after_execution_end_suppressed(self, mock_tcp_server):
        """Straggler llm:* events after execution:end are suppressed."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "test",
            },
        )
        await hook.handle_event(
            "execution:end",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
            },
        )
        result = await hook.handle_event(
            "llm:request:raw",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
            },
        )
        assert result.action == "continue"
        await hook.cleanup()


class TestDelegationEventFlow:
    """Test delegation event sequence from the parent perspective."""

    @pytest.mark.asyncio
    async def test_delegation_cycle(self, mock_tcp_server):
        """Full delegation flow: tool:pre -> spawn -> complete -> tool:post."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event(
            "tool:pre",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "delegate",
                "tool_call_id": "toolu_test",
                "tool_input": {"agent": "foundation:explorer", "instruction": "survey"},
            },
        )
        await hook.handle_event(
            "task:agent_spawned",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "agent": "foundation:explorer",
                "sub_session_id": "0000-aaaa_foundation:explorer",
                "parent_session_id": SESSION_ID,
            },
        )
        await hook.handle_event(
            "task:agent_completed",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "agent": "foundation:explorer",
                "sub_session_id": "0000-aaaa_foundation:explorer",
                "parent_session_id": SESSION_ID,
                "success": True,
            },
        )
        await hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "delegate",
                "tool_call_id": "toolu_test",
                "result": {"success": True, "output": {"response": "Survey complete."}},
            },
        )
        tool_calls = hook._turn_accumulator._current.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "delegate"
        assert tool_calls[0].has_result
        await hook.cleanup()


class TestMultipleTurnCycles:
    """Test multiple orchestrator cycles (multi-turn conversation)."""

    @pytest.mark.asyncio
    async def test_two_turn_conversation(self, mock_tcp_server):
        """Two complete turn cycles work correctly."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        # Turn 1
        await hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "What is CXDB?",
            },
        )
        await hook.handle_event(
            "content_block:end",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "block_type": "text",
                "block": "CXDB is a context database.",
            },
        )
        await hook.handle_event(
            "orchestrator:complete",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "orchestrator": "loop-streaming",
                "turn_count": 1,
            },
        )
        assert hook._turn_accumulator.flush() is None
        # Turn 2
        await hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "How does it store data?",
            },
        )
        await hook.handle_event(
            "tool:pre",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "read_file",
                "tool_input": {"file_path": "docs/arch.md"},
            },
        )
        await hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "read_file",
                "result": "Turn DAG with blob CAS",
            },
        )
        await hook.handle_event(
            "content_block:end",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "block_type": "text",
                "block": "CXDB uses a Turn DAG.",
            },
        )
        await hook.handle_event(
            "orchestrator:complete",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "orchestrator": "loop-streaming",
                "turn_count": 2,
            },
        )
        assert hook._turn_accumulator.flush() is None
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, mock_tcp_server):
        """Multiple parallel tool calls in a single turn."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "Read both files",
            },
        )
        await hook.handle_event(
            "tool:pre",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "read_file",
                "tool_input": {"file_path": "a.yaml"},
                "parallel_group_id": "pg-001",
            },
        )
        await hook.handle_event(
            "tool:pre",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "read_file",
                "tool_input": {"file_path": "b.json"},
                "parallel_group_id": "pg-001",
            },
        )
        await hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "read_file",
                "result": "port: 9009",
                "parallel_group_id": "pg-001",
            },
        )
        await hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "read_file",
                "result": '{"host": "localhost"}',
                "parallel_group_id": "pg-001",
            },
        )
        tool_calls = hook._turn_accumulator._current.tool_calls
        assert len(tool_calls) == 2
        assert all(tc.tool_name == "read_file" for tc in tool_calls)
        assert all(tc.has_result for tc in tool_calls)
        await hook.cleanup()
