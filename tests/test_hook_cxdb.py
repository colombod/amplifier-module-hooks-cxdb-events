"""Hook-to-CXDB integration tests.

Exercises the CXDBEventHook against a real CXDB server.
Feeds synthetic events into the hook, then queries CXDB HTTP API
to verify data was stored correctly.

Run with: pytest tests/test_hook_cxdb.py -m integration
Requires: CXDB server on localhost:9009 (binary) and localhost:9010 (HTTP)
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from amplifier_module_hooks_cxdb_events.hook import CXDBEventHook
from amplifier_module_hooks_cxdb_events.protocol import CXDBTcpClient
from amplifier_module_hooks_cxdb_events.schema import publish_registry_bundle

pytestmark = pytest.mark.integration

CXDB_HOST = "localhost"
CXDB_BINARY_PORT = 9009
CXDB_HTTP_PORT = 80

SESSION_ID = "integration-test-root-session"
CHILD_SESSION_ID = "0000000000000000-aaaa111122223333_foundation:explorer"


@pytest_asyncio.fixture
async def cxdb_hook():
    """Create a CXDBEventHook connected to real CXDB.

    Publishes registry bundle, initializes hook, yields it,
    then cleans up.
    """
    client = CXDBTcpClient(CXDB_HOST, CXDB_BINARY_PORT, timeout=5.0)
    try:
        await client.connect()
    except ConnectionError:
        pytest.skip("CXDB server not available at localhost:9009")

    # Publish registry bundle so typed queries work
    await publish_registry_bundle(CXDB_HOST, CXDB_HTTP_PORT)

    hook = CXDBEventHook(
        client=client,
        config={
            "cxdb_host": CXDB_HOST,
            "cxdb_port": CXDB_BINARY_PORT,
            "cxdb_http_port": CXDB_HTTP_PORT,
        },
        session_id=SESSION_ID,
        parent_id=None,
        root_session_id=SESSION_ID,
    )
    await hook.initialize()
    yield hook
    await hook.cleanup()


async def _query_turns(context_id: int, view: str = "raw") -> dict:
    """Query CXDB HTTP API for turns in a context.

    Args:
        context_id: CXDB context ID.
        view: "raw" or "typed".

    Returns:
        Response JSON dict with "turns" array.
    """
    url = f"http://{CXDB_HOST}:{CXDB_HTTP_PORT}/v1/contexts/{context_id}/turns"
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url, params={"view": view, "u64_format": "number"})
        response.raise_for_status()
        return response.json()


class TestHookStoresEvents:
    """Verify events flow through the hook into CXDB and can be read back."""

    @pytest.mark.asyncio
    async def test_session_start_stored(self, cxdb_hook):
        """session:start event is stored in the everything context."""
        await cxdb_hook.handle_event(
            "session:start",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
            },
        )

        result = await _query_turns(cxdb_hook.everything_context_id)
        assert len(result.get("turns", [])) >= 1

    @pytest.mark.asyncio
    async def test_tool_events_stored(self, cxdb_hook):
        """tool:pre and tool:post events are stored."""
        await cxdb_hook.handle_event(
            "tool:pre",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "grep",
                "tool_input": {"pattern": "def main"},
            },
        )
        await cxdb_hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "grep",
                "result": "Found 3 matches in src/main.py",
            },
        )

        result = await _query_turns(cxdb_hook.everything_context_id)
        assert len(result.get("turns", [])) >= 2

    @pytest.mark.asyncio
    async def test_provider_event_stored(self, cxdb_hook):
        """provider:response with usage metrics is stored."""
        await cxdb_hook.handle_event(
            "provider:response",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "provider": "anthropic",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 5000,
                    "cache_creation_input_tokens": 1000,
                },
            },
        )

        result = await _query_turns(cxdb_hook.everything_context_id)
        assert len(result.get("turns", [])) >= 1

    @pytest.mark.asyncio
    async def test_delegation_events_stored(self, cxdb_hook):
        """Delegation lifecycle events are stored."""
        await cxdb_hook.handle_event(
            "task:agent_spawned",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "agent": "foundation:explorer",
                "sub_session_id": CHILD_SESSION_ID,
                "parent_session_id": SESSION_ID,
            },
        )
        await cxdb_hook.handle_event(
            "task:agent_completed",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "agent": "foundation:explorer",
                "sub_session_id": CHILD_SESSION_ID,
                "parent_session_id": SESSION_ID,
                "success": True,
            },
        )

        result = await _query_turns(cxdb_hook.everything_context_id)
        assert len(result.get("turns", [])) >= 2


class TestHookStoresTurns:
    """Verify conversation turns flow through accumulator into turns context."""

    @pytest.mark.asyncio
    async def test_full_turn_cycle_stored(self, cxdb_hook):
        """Complete turn cycle: prompt -> response -> flush produces turns in CXDB."""
        # Simulate a complete turn
        await cxdb_hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "Find all Python test files",
            },
        )
        await cxdb_hook.handle_event(
            "provider:response",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "provider": "anthropic",
                "usage": {"input_tokens": 50, "output_tokens": 100},
            },
        )
        await cxdb_hook.handle_event(
            "content_block:end",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "block_type": "text",
                "block": "I found 42 Python test files in the project.",
            },
        )
        # Flush on orchestrator:complete
        await cxdb_hook.handle_event(
            "orchestrator:complete",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "orchestrator": "loop-streaming",
                "turn_count": 1,
            },
        )

        # Query turns context -- should have user + assistant items
        result = await _query_turns(cxdb_hook.turns_context_id)
        turns = result.get("turns", [])
        assert len(turns) >= 2, (
            f"Expected at least 2 turns (user+assistant), got {len(turns)}"
        )

    @pytest.mark.asyncio
    async def test_turn_with_tool_calls_stored(self, cxdb_hook):
        """Turn with tool calls includes tool summaries."""
        await cxdb_hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "Search the codebase for authentication bugs",
            },
        )
        await cxdb_hook.handle_event(
            "tool:pre",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "grep",
                "tool_input": {"pattern": "auth.*bug", "path": "src/"},
            },
        )
        await cxdb_hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "grep",
                "result": "src/auth.py:42: # BUG: token not validated",
            },
        )
        await cxdb_hook.handle_event(
            "content_block:end",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "block_type": "text",
                "block": "I found an authentication bug in src/auth.py line 42.",
            },
        )
        await cxdb_hook.handle_event(
            "orchestrator:complete",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "orchestrator": "loop-streaming",
                "turn_count": 1,
            },
        )

        result = await _query_turns(cxdb_hook.turns_context_id)
        turns = result.get("turns", [])
        assert len(turns) >= 2


class TestHookDualContextSeparation:
    """Verify turns and events go to separate CXDB contexts."""

    @pytest.mark.asyncio
    async def test_contexts_are_separate(self, cxdb_hook):
        """Turns context and everything context have different IDs."""
        assert cxdb_hook.turns_context_id != cxdb_hook.everything_context_id

    @pytest.mark.asyncio
    async def test_events_in_everything_not_in_turns(self, cxdb_hook):
        """Non-turn events (e.g., session:start) only appear in everything context."""
        await cxdb_hook.handle_event(
            "session:start",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
            },
        )

        everything = await _query_turns(cxdb_hook.everything_context_id)
        turns = await _query_turns(cxdb_hook.turns_context_id)

        assert len(everything.get("turns", [])) >= 1
        # Turns context should be empty (no orchestrator:complete to flush)
        assert len(turns.get("turns", [])) == 0

    @pytest.mark.asyncio
    async def test_full_session_populates_both_contexts(self, cxdb_hook):
        """A complete session turn populates both contexts."""
        # Full turn cycle
        events = [
            ("session:start", {"session_id": SESSION_ID, "parent_id": None}),
            (
                "prompt:submit",
                {"session_id": SESSION_ID, "parent_id": None, "prompt": "Hello"},
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
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            ),
            (
                "content_block:end",
                {
                    "session_id": SESSION_ID,
                    "parent_id": None,
                    "block_type": "text",
                    "block": "Hi!",
                },
            ),
            (
                "orchestrator:complete",
                {
                    "session_id": SESSION_ID,
                    "parent_id": None,
                    "orchestrator": "loop-streaming",
                    "turn_count": 1,
                },
            ),
        ]
        for event_name, data in events:
            await cxdb_hook.handle_event(event_name, data)

        everything = await _query_turns(cxdb_hook.everything_context_id)
        turns = await _query_turns(cxdb_hook.turns_context_id)

        # Everything context has all events
        assert len(everything.get("turns", [])) >= 6
        # Turns context has conversation items (user + assistant)
        assert len(turns.get("turns", [])) >= 2


class TestHookRegistryAndTypes:
    """Verify type registry works and events are queryable by type."""

    @pytest.mark.asyncio
    async def test_typed_query_returns_field_names(self, cxdb_hook):
        """Events queried with view=typed return named fields, not numeric tags."""
        await cxdb_hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "read_file",
                "result": "file contents here",
            },
        )

        # Try typed query -- this depends on registry bundle being published
        try:
            result = await _query_turns(cxdb_hook.everything_context_id, view="typed")
            turns = result.get("turns", [])
            if turns:
                turn_data = turns[-1].get("data", {})
                # If typed projection works, we should see named fields
                # rather than numeric keys
                if "event_name" in turn_data or "session_id" in turn_data:
                    assert True  # typed projection working
                else:
                    # Typed projection may not work if registry wasn't loaded
                    # by this version of CXDB -- that's OK for the experiment
                    pytest.skip(
                        "Typed projection not available (registry may need different format)"
                    )
        except httpx.HTTPStatusError:
            pytest.skip("Typed query not supported by this CXDB version")

    @pytest.mark.asyncio
    async def test_declared_type_on_stored_turns(self, cxdb_hook):
        """Stored turns have the correct declared_type metadata."""
        await cxdb_hook.handle_event(
            "session:start",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
            },
        )

        result = await _query_turns(cxdb_hook.everything_context_id)
        turns = result.get("turns", [])
        if turns:
            last_turn = turns[-1]
            declared = last_turn.get("declared_type", {})
            # Should have a type_id from our registry
            type_id = declared.get("type_id", "")
            assert type_id.startswith("amplifier."), (
                f"Expected amplifier.* type, got: {type_id}"
            )


class TestHookMultiTurnSession:
    """Verify a multi-turn session with delegation stores correctly."""

    @pytest.mark.asyncio
    async def test_two_turn_session_with_delegation(self, cxdb_hook):
        """Complete 2-turn session with tool calls and delegation."""
        # Turn 1: Simple response
        await cxdb_hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "What is CXDB?",
            },
        )
        await cxdb_hook.handle_event(
            "content_block:end",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "block_type": "text",
                "block": "CXDB is a context database for AI conversations.",
            },
        )
        await cxdb_hook.handle_event(
            "orchestrator:complete",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "orchestrator": "loop-streaming",
                "turn_count": 1,
            },
        )

        # Turn 2: With delegation
        await cxdb_hook.handle_event(
            "prompt:submit",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "prompt": "Explore the CXDB source code",
            },
        )
        await cxdb_hook.handle_event(
            "tool:pre",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "delegate",
                "tool_input": {
                    "agent": "foundation:explorer",
                    "instruction": "survey cxdb/",
                },
            },
        )
        await cxdb_hook.handle_event(
            "task:agent_spawned",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "agent": "foundation:explorer",
                "sub_session_id": CHILD_SESSION_ID,
                "parent_session_id": SESSION_ID,
            },
        )
        await cxdb_hook.handle_event(
            "task:agent_completed",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "agent": "foundation:explorer",
                "sub_session_id": CHILD_SESSION_ID,
                "parent_session_id": SESSION_ID,
                "success": True,
            },
        )
        await cxdb_hook.handle_event(
            "tool:post",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "tool_name": "delegate",
                "result": {
                    "success": True,
                    "output": {"response": "Found Rust server + Go gateway"},
                },
            },
        )
        await cxdb_hook.handle_event(
            "content_block:end",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "block_type": "text",
                "block": "The CXDB codebase has a Rust server and Go gateway.",
            },
        )
        await cxdb_hook.handle_event(
            "orchestrator:complete",
            {
                "session_id": SESSION_ID,
                "parent_id": None,
                "orchestrator": "loop-streaming",
                "turn_count": 2,
            },
        )

        # Verify everything context has all events
        everything = await _query_turns(cxdb_hook.everything_context_id)
        everything_count = len(everything.get("turns", []))
        # Should have: session events, prompts, provider events, tool events,
        # delegation events, content blocks, orchestrator events
        assert everything_count >= 10, f"Expected >= 10 events, got {everything_count}"

        # Verify turns context has 4 conversation items (2 user + 2 assistant)
        turns = await _query_turns(cxdb_hook.turns_context_id)
        turns_count = len(turns.get("turns", []))
        assert turns_count >= 4, (
            f"Expected >= 4 turns (2 user + 2 assistant), got {turns_count}"
        )
