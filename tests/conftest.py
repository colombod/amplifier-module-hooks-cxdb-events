"""Shared test fixtures for hooks-cxdb-events module."""

from __future__ import annotations

import asyncio
import struct

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from amplifier_module_hooks_cxdb_events.protocol import (
    FRAME_HEADER_FORMAT,
    FRAME_HEADER_SIZE,
    MSG_APPEND_TURN,
    MSG_CTX_CREATE,
    MSG_CTX_FORK,
    MSG_HELLO,
    encode_frame,
)


@pytest.fixture
def sample_event_data():
    """Sample Amplifier event data for testing."""
    return {
        "session_id": "test-session-123",
        "parent_id": None,
        "ts": 1707600000000,
    }


@pytest.fixture
def mock_coordinator():
    """Minimal mock of ModuleCoordinator."""
    coordinator = MagicMock()
    coordinator.session_id = "test-session-123"
    coordinator.parent_id = None
    coordinator.config = {"root_session_id": "test-session-123"}
    coordinator.hooks = MagicMock()
    coordinator.hooks.register = MagicMock()
    coordinator.collect_contributions = AsyncMock(return_value=[])
    return coordinator


class MockCXDBServer:
    """Minimal mock CXDB server for testing."""

    def __init__(self) -> None:
        self.port: int = 0
        self._server: asyncio.AbstractServer | None = None
        self._next_context_id: int = 100
        self._next_turn_id: int = 1000

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        addr = self._server.sockets[0].getsockname()
        self.port = addr[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                # Read frame header
                header_data = await reader.readexactly(FRAME_HEADER_SIZE)
                payload_len, msg_type, _flags, request_id = struct.unpack(
                    FRAME_HEADER_FORMAT, header_data
                )

                # Read payload
                payload = b""
                if payload_len > 0:
                    payload = await reader.readexactly(payload_len)

                # Handle message
                response_payload = self._handle_message(msg_type, payload)

                # Send response (echo back same msg_type)
                response_frame = encode_frame(msg_type, request_id, response_payload)
                writer.write(response_frame)
                await writer.drain()

        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    def _handle_message(self, msg_type: int, payload: bytes) -> bytes:
        if msg_type == MSG_HELLO:
            # Return session_id(u64 LE) + protocol_version(u16 LE)
            return struct.pack("<QH", 1, 1)

        if msg_type == MSG_CTX_CREATE:
            # Return context_id(u64) + head_turn_id(u64)
            ctx_id = self._next_context_id
            self._next_context_id += 1
            turn_id = self._next_turn_id
            self._next_turn_id += 1
            return struct.pack("<QQ", ctx_id, turn_id)

        if msg_type == MSG_CTX_FORK:
            # Read base_turn_id, return new_context_id + head_turn_id + head_depth
            base_turn_id = (
                struct.unpack("<Q", payload[:8])[0] if len(payload) >= 8 else 0
            )
            ctx_id = self._next_context_id
            self._next_context_id += 1
            return struct.pack("<QQI", ctx_id, base_turn_id, 1)

        if msg_type == MSG_APPEND_TURN:
            # Parse enough to extract context_id, return ACK
            if len(payload) >= 8:
                context_id = struct.unpack("<Q", payload[:8])[0]
            else:
                context_id = 0
            turn_id = self._next_turn_id
            self._next_turn_id += 1
            depth = 1
            # ACK: context_id(u64) + new_turn_id(u64) + new_depth(u32) + content_hash(32)
            dummy_hash = b"\x00" * 32
            return struct.pack("<QQI", context_id, turn_id, depth) + dummy_hash

        # Unknown message type - return error
        return f"Unknown message type: {msg_type}".encode()


@pytest_asyncio.fixture
async def mock_tcp_server():
    """Start a mock CXDB TCP server for testing."""
    server = MockCXDBServer()
    await server.start()
    yield server
    await server.stop()
