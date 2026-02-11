"""Integration tests requiring a running CXDB server.

Run with: pytest tests/test_integration.py -m integration
Skip without CXDB: pytest -m "not integration"

Requires:
- CXDB server running on localhost:9009 (binary) and localhost:9010 (HTTP)
- Start with: ./run-cxdb.sh from the cxdb-investigation directory
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from amplifier_module_hooks_cxdb_events.protocol import CXDBTcpClient

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def cxdb_client():
    """Connect to local CXDB server for integration tests.

    Requires CXDB running on localhost:9009.
    """
    client = CXDBTcpClient("localhost", 9009, timeout=5.0)
    try:
        await client.connect()
    except ConnectionError:
        pytest.skip("CXDB server not available at localhost:9009")
    yield client
    await client.close()


class TestCLIChange:
    def test_session_runner_sets_root_session_id(self):
        """Verify session_runner.py contains the cxdb_root_session_id line."""
        session_runner = Path(
            "/home/dicolomb/cxdb-investigation/app-cli-fork/"
            "amplifier_app_cli/session_runner.py"
        )
        assert session_runner.exists(), "session_runner.py not found"
        source = session_runner.read_text()
        assert "cxdb_root_session_id" in source, (
            "session_runner.py should set cxdb_root_session_id in config"
        )


class TestCXDBConnection:
    @pytest.mark.asyncio
    async def test_connect_to_cxdb(self, cxdb_client):
        """Verify we can connect to the local CXDB server."""
        assert cxdb_client.connected

    @pytest.mark.asyncio
    async def test_create_context(self, cxdb_client):
        """Verify context creation works on real CXDB."""
        context_id, head_turn_id = await cxdb_client.create_context()
        assert context_id > 0


class TestCXDBEventStorage:
    @pytest.mark.asyncio
    async def test_append_event_turn(self, cxdb_client):
        """Verify event turns can be appended to a real CXDB context."""
        ctx_id, _ = await cxdb_client.create_context()
        turn_id, depth = await cxdb_client.append_turn(
            context_id=ctx_id,
            payload={1: "session:start", 2: "test-session", 4: 1707600000000},
            declared_type_id="amplifier.SessionEvent",
        )
        assert turn_id > 0

    @pytest.mark.asyncio
    async def test_append_multiple_event_types(self, cxdb_client):
        """Verify different event types can be stored."""
        ctx_id, _ = await cxdb_client.create_context()

        # Session event
        await cxdb_client.append_turn(
            context_id=ctx_id,
            payload={1: "session:start", 2: "test"},
            declared_type_id="amplifier.SessionEvent",
        )

        # Tool event
        await cxdb_client.append_turn(
            context_id=ctx_id,
            payload={1: "tool:post", 2: "test", 10: "grep"},
            declared_type_id="amplifier.ToolEvent",
        )

        # Provider event
        await cxdb_client.append_turn(
            context_id=ctx_id,
            payload={1: "provider:response", 2: "test", 10: "anthropic"},
            declared_type_id="amplifier.ProviderEvent",
        )

    @pytest.mark.asyncio
    async def test_conversation_item_turn(self, cxdb_client):
        """Verify ConversationItem v3 turns can be stored."""
        ctx_id, _ = await cxdb_client.create_context()

        # User input
        await cxdb_client.append_turn(
            context_id=ctx_id,
            payload={1: "user_input", 2: "Hello, world!", 4: 1707600000000},
            declared_type_id="cxdb.ConversationItem",
            declared_type_version=3,
        )

        # Assistant turn
        await cxdb_client.append_turn(
            context_id=ctx_id,
            payload={
                1: "assistant_turn",
                2: "Hi there!",
                3: [{"tool_name": "grep", "has_result": True}],
                4: 1707600001000,
                5: {"total_tokens": 150, "model": "test"},
            },
            declared_type_id="cxdb.ConversationItem",
            declared_type_version=3,
        )


class TestDualContextModel:
    @pytest.mark.asyncio
    async def test_separate_contexts_for_turns_and_events(self, cxdb_client):
        """Verify dual-context model: turns in one, events in another."""
        turns_ctx, _ = await cxdb_client.create_context()
        events_ctx, _ = await cxdb_client.create_context()
        assert turns_ctx != events_ctx

        # Write conversation to turns context
        await cxdb_client.append_turn(
            context_id=turns_ctx,
            payload={1: "user_input", 2: "Search for bugs", 4: 1707600000000},
            declared_type_id="cxdb.ConversationItem",
            declared_type_version=3,
        )

        # Write event to events context
        await cxdb_client.append_turn(
            context_id=events_ctx,
            payload={1: "tool:pre", 2: "test-session", 10: "grep"},
            declared_type_id="amplifier.ToolEvent",
        )

    @pytest.mark.asyncio
    async def test_child_session_events_in_root_context(self, cxdb_client):
        """Verify child session events can be written to root's context."""
        root_ctx, _ = await cxdb_client.create_context()

        # Root session event
        await cxdb_client.append_turn(
            context_id=root_ctx,
            payload={1: "session:start", 2: "root-123", 4: 1707600000000},
            declared_type_id="amplifier.SessionEvent",
        )

        # Child session event (same context, different session_id)
        await cxdb_client.append_turn(
            context_id=root_ctx,
            payload={
                1: "tool:post",
                2: "0000-aaaa_explorer",
                3: "root-123",
                4: 1707600001000,
                5: "foundation:explorer",
                10: "grep",
            },
            declared_type_id="amplifier.ToolEvent",
        )
