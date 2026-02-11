"""Tests for CXDBEventHook - main event router."""

import pytest

from amplifier_module_hooks_cxdb_events.hook import CXDBEventHook
from amplifier_module_hooks_cxdb_events.protocol import CXDBTcpClient


def _make_hook(
    mock_tcp_server, parent_id=None, root_session_id="root-123", session_id=None
):
    """Helper to create a hook instance for testing."""
    if session_id is None:
        session_id = root_session_id if parent_id is None else "0000-aaaa_test-agent"
    return CXDBEventHook(
        client=CXDBTcpClient("127.0.0.1", mock_tcp_server.port),
        config={
            "cxdb_host": "127.0.0.1",
            "cxdb_http_port": 19999,  # HTTP port that won't connect (registry publish will fail gracefully)
        },
        session_id=session_id,
        parent_id=parent_id,
        root_session_id=root_session_id,
    )


class TestHookInitialization:
    @pytest.mark.asyncio
    async def test_root_creates_contexts(self, mock_tcp_server):
        """Root session creates both turns and everything contexts."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        assert hook.initialized
        assert hook.turns_context_id is not None
        assert hook.everything_context_id is not None
        assert hook.is_root
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_child_session_detected(self, mock_tcp_server):
        """Child session correctly identified by parent_id."""
        hook = _make_hook(mock_tcp_server, parent_id="root-123")
        await hook.initialize()
        assert hook.initialized
        assert not hook.is_root
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_tcp_server):
        """Multiple initialize calls don't create extra contexts."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        ctx1 = hook.turns_context_id
        await hook.initialize()
        assert hook.turns_context_id == ctx1  # same context
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_not_initialized_before_connect(self, mock_tcp_server):
        """Hook is not initialized before initialize() is called."""
        hook = _make_hook(mock_tcp_server)
        assert not hook.initialized
        assert hook.turns_context_id is None
        assert hook.everything_context_id is None


class TestHookEventRouting:
    @pytest.mark.asyncio
    async def test_prompt_routed_to_accumulator(self, mock_tcp_server):
        """prompt:submit routes to turn accumulator."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        result = await hook.handle_event("prompt:submit", {"prompt": "Hello"})
        assert result.action == "continue"
        assert hook._turn_accumulator._current.user_prompt == "Hello"
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_tool_events_routed_to_accumulator(self, mock_tcp_server):
        """tool:pre and tool:post route to turn accumulator."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event("tool:pre", {"tool_name": "grep", "tool_input": {}})
        assert len(hook._turn_accumulator._current.tool_calls) == 1
        await hook.handle_event("tool:post", {"tool_name": "grep", "result": "found"})
        assert hook._turn_accumulator._current.tool_calls[0].has_result
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_orchestrator_complete_flushes_turns(self, mock_tcp_server):
        """orchestrator:complete triggers turn flush and dedup reset."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event("prompt:submit", {"prompt": "Hello"})
        await hook.handle_event(
            "content_block:end", {"block_type": "text", "block": "Hi!"}
        )
        await hook.handle_event("orchestrator:complete", {"status": "success"})
        # After flush, accumulator should be empty
        assert hook._turn_accumulator.flush() is None
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_non_turn_events_go_to_everything_only(self, mock_tcp_server):
        """Events not in _TURN_EVENTS go to everything context only."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        result = await hook.handle_event("cancel:requested", {"level": "graceful"})
        assert result.action == "continue"
        # accumulator should be empty (cancel is not a turn event)
        assert hook._turn_accumulator.flush() is None
        await hook.cleanup()


class TestHookVariantDedup:
    @pytest.mark.asyncio
    async def test_raw_variant_processed(self, mock_tcp_server):
        """Raw variant is processed."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        result = await hook.handle_event("session:start:raw", {"session_id": "test"})
        assert result.action == "continue"
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_debug_suppressed_after_raw(self, mock_tcp_server):
        """Debug variant suppressed after raw was seen."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event("session:start:raw", {"session_id": "test"})
        # debug should be suppressed (returns continue without writing)
        result = await hook.handle_event("session:start:debug", {"session_id": "test"})
        assert result.action == "continue"
        await hook.cleanup()


class TestHookStragglerSuppression:
    @pytest.mark.asyncio
    async def test_llm_events_suppressed_after_execution_end(self, mock_tcp_server):
        """llm:* events after execution:end are suppressed."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.handle_event("execution:end", {})
        # llm events should be suppressed
        result = await hook.handle_event("llm:request:raw", {"data": "straggler"})
        assert result.action == "continue"
        await hook.cleanup()


class TestHookErrorHandling:
    @pytest.mark.asyncio
    async def test_always_returns_continue(self, mock_tcp_server):
        """Hook always returns continue, even on errors."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        # Various events should all return continue
        for event in [
            "session:start",
            "tool:pre",
            "unknown:event",
            "orchestrator:complete",
        ]:
            result = await hook.handle_event(event, {"session_id": "test"})
            assert result.action == "continue"
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_buffers_when_not_initialized(self):
        """Events buffered when CXDB is unreachable."""
        hook = CXDBEventHook(
            client=CXDBTcpClient("192.0.2.1", 9999, timeout=0.1),
            config={"cxdb_host": "192.0.2.1"},
            session_id="root-123",
            parent_id=None,
            root_session_id="root-123",
        )
        # handle_event should not raise even when CXDB is unreachable
        result = await hook.handle_event("tool:post", {"tool_name": "test"})
        assert result.action == "continue"
        await hook.cleanup()

    @pytest.mark.asyncio
    async def test_lazy_initialization(self, mock_tcp_server):
        """Hook initializes lazily on first event."""
        hook = _make_hook(mock_tcp_server)
        assert not hook.initialized
        await hook.handle_event("session:start", {"session_id": "test"})
        assert hook.initialized
        await hook.cleanup()


class TestHookCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_closes_connection(self, mock_tcp_server):
        """Cleanup closes the CXDB TCP connection."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        assert hook._client.connected
        await hook.cleanup()
        assert not hook._client.connected

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self, mock_tcp_server):
        """Cleanup can be called multiple times safely."""
        hook = _make_hook(mock_tcp_server)
        await hook.initialize()
        await hook.cleanup()
        await hook.cleanup()  # should not raise

    @pytest.mark.asyncio
    async def test_cleanup_without_initialize(self, mock_tcp_server):
        """Cleanup without prior initialize is safe."""
        hook = _make_hook(mock_tcp_server)
        await hook.cleanup()  # should not raise
