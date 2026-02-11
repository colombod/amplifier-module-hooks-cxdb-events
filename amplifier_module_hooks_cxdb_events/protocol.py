"""Binary TCP client for CXDB protocol (MSG_HELLO, MSG_CTX_CREATE, MSG_CTX_FORK, MSG_APPEND_TURN)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct

import blake3
import msgpack

logger = logging.getLogger(__name__)

# Message type constants
MSG_HELLO = 1
MSG_CTX_CREATE = 2
MSG_CTX_FORK = 3
MSG_GET_HEAD = 4
MSG_APPEND_TURN = 5
MSG_ERROR = 255

ENCODING_MSGPACK = 1
COMPRESSION_NONE = 0

# Frame header: len(4B) + msg_type(2B) + flags(2B) + req_id(8B) = 16 bytes
# ALL LITTLE-ENDIAN per CXDB server/src/protocol/mod.rs
FRAME_HEADER_SIZE = 16
FRAME_HEADER_FORMAT = (
    "<IHHQ"  # little-endian: u32 len, u16 msg_type, u16 flags, u64 req_id
)


def encode_frame(
    msg_type: int, request_id: int, payload: bytes, flags: int = 0
) -> bytes:
    """Encode a binary protocol frame.

    Args:
        msg_type: Message type constant (MSG_HELLO, MSG_CTX_CREATE, etc.)
        request_id: Request identifier for matching responses
        payload: Raw payload bytes
        flags: Frame flags (default 0)

    Returns:
        Complete frame bytes including header and payload.
    """
    header = struct.pack(FRAME_HEADER_FORMAT, len(payload), msg_type, flags, request_id)
    return header + payload


def decode_frame(data: bytes) -> tuple[int, int, int, bytes]:
    """Decode a binary protocol frame.

    Args:
        data: Raw frame bytes (header + payload)

    Returns:
        Tuple of (msg_type, flags, request_id, payload)

    Raises:
        ValueError: If data is too short for a valid frame header.
    """
    if len(data) < FRAME_HEADER_SIZE:
        raise ValueError(
            f"Frame too short: {len(data)} bytes, need at least {FRAME_HEADER_SIZE}"
        )

    payload_len, msg_type, flags, request_id = struct.unpack(
        FRAME_HEADER_FORMAT, data[:FRAME_HEADER_SIZE]
    )
    payload = data[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + payload_len]

    if len(payload) < payload_len:
        raise ValueError(
            f"Incomplete payload: got {len(payload)} bytes, expected {payload_len}"
        )

    return msg_type, flags, request_id, payload


def encode_append_turn_payload(
    context_id: int,
    msgpack_bytes: bytes,
    content_hash: bytes,
    declared_type_id: str,
    declared_type_version: int = 1,
    parent_turn_id: int = 0,
) -> bytes:
    """Build the APPEND_TURN binary payload per CLIENT_SPEC.md 5.1.2.

    Args:
        context_id: Target context.
        msgpack_bytes: Pre-encoded msgpack payload.
        content_hash: BLAKE3-256 hash of msgpack_bytes (32 bytes).
        declared_type_id: Type identifier string (e.g., "amplifier.ToolEvent").
        declared_type_version: Type version number.
        parent_turn_id: Parent turn (0 = append to current head).

    Returns:
        Complete APPEND_TURN payload bytes ready for framing.
    """
    type_id_bytes = declared_type_id.encode("utf-8")
    idempotency_key = generate_idempotency_key(context_id, content_hash)

    parts = [
        struct.pack("<Q", context_id),
        struct.pack("<Q", parent_turn_id),
        struct.pack("<I", len(type_id_bytes)),
        type_id_bytes,
        struct.pack("<I", declared_type_version),
        struct.pack("<I", ENCODING_MSGPACK),
        struct.pack("<I", COMPRESSION_NONE),
        struct.pack("<I", len(msgpack_bytes)),
        content_hash,
        struct.pack("<I", len(msgpack_bytes)),
        msgpack_bytes,
        struct.pack("<I", len(idempotency_key)),
        idempotency_key,
    ]
    return b"".join(parts)


def generate_idempotency_key(context_id: int, content_hash: bytes) -> bytes:
    """Generate idempotency key as SHA-256(context_id:content_hash).

    Args:
        context_id: Context ID.
        content_hash: BLAKE3-256 content hash (32 bytes).

    Returns:
        SHA-256 digest bytes (32 bytes).
    """
    key_input = struct.pack("<Q", context_id) + b":" + content_hash
    return hashlib.sha256(key_input).digest()


def serialize_payload(payload: dict) -> tuple[bytes, bytes]:
    """Serialize a payload dict to msgpack and compute BLAKE3 hash.

    Payload keys MUST be integers (tag numbers per CXDB spec).
    Keys are sorted ascending for deterministic encoding.

    Args:
        payload: Dict with integer keys mapping to values.

    Returns:
        Tuple of (msgpack_bytes, blake3_hash_bytes).
    """
    # Sort by key for deterministic encoding
    sorted_payload = dict(sorted(payload.items()))
    msgpack_bytes = msgpack.packb(sorted_payload, use_bin_type=True)
    content_hash = blake3.blake3(msgpack_bytes).digest()
    return msgpack_bytes, content_hash


class CXDBProtocolError(Exception):
    """Raised when CXDB server returns an error frame."""


class CXDBTcpClient:
    """Async TCP client for CXDB binary protocol."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 5.0,
        client_tag: str = "amplifier-hooks-cxdb",
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client_tag = client_tag
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id: int = 0
        self._connected: bool = False
        self._session_id: int | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def session_id(self) -> int | None:
        """Server-assigned session ID from HELLO handshake."""
        return self._session_id

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def connect(self) -> None:
        """Connect to CXDB and perform MSG_HELLO handshake.

        Raises:
            ConnectionError: If connection fails or times out.
            CXDBProtocolError: If handshake fails.
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout,
            )
        except (OSError, asyncio.TimeoutError) as e:
            raise ConnectionError(
                f"Failed to connect to CXDB at {self.host}:{self.port}: {e}"
            ) from e

        # Build HELLO payload (new format, matching Go client)
        tag_bytes = self.client_tag.encode("utf-8")
        hello_payload = struct.pack(
            "<HH", 1, len(tag_bytes)
        )  # protocol_version=1, tag_len
        hello_payload += tag_bytes
        hello_payload += struct.pack("<I", 0)  # client_meta_json_len=0

        response = await self._send_and_recv(MSG_HELLO, hello_payload)
        # Response: session_id(u64 LE) [+ optional protocol_version(u16 LE)]
        if len(response) >= 8:
            self._session_id = struct.unpack("<Q", response[:8])[0]
            logger.info(
                "Connected to CXDB (session=%d, tag=%s)",
                self._session_id,
                self.client_tag,
            )

        self._connected = True

    async def create_context(self, base_turn_id: int = 0) -> tuple[int, int, int]:
        """Create a new CXDB context.

        Args:
            base_turn_id: Turn ID to base the context on. 0 = empty context.

        Returns:
            Tuple of (context_id, head_turn_id, head_depth).

        Raises:
            ConnectionError: If not connected.
            CXDBProtocolError: If server returns error.
        """
        self._ensure_connected()
        response = await self._send_and_recv(
            MSG_CTX_CREATE, struct.pack("<Q", base_turn_id)
        )
        # Response: context_id(u64) + head_turn_id(u64) + head_depth(u32) = 20 bytes
        if len(response) < 20:
            raise CXDBProtocolError(
                f"CTX_CREATE response too short: {len(response)} bytes"
            )
        context_id, head_turn_id, head_depth = struct.unpack("<QQI", response[:20])
        logger.debug(
            "Created context %d (head=%d, depth=%d)",
            context_id,
            head_turn_id,
            head_depth,
        )
        return context_id, head_turn_id, head_depth

    async def get_head(self, context_id: int) -> tuple[int, int]:
        """Get the current head of a context.

        Args:
            context_id: Context to query.

        Returns:
            Tuple of (head_turn_id, head_depth).

        Raises:
            ConnectionError: If not connected.
            CXDBProtocolError: If server returns error.
        """
        self._ensure_connected()
        response = await self._send_and_recv(
            MSG_GET_HEAD, struct.pack("<Q", context_id)
        )
        if len(response) < 12:
            raise CXDBProtocolError(
                f"GET_HEAD response too short: {len(response)} bytes"
            )
        head_turn_id, head_depth = struct.unpack("<QI", response[:12])
        return head_turn_id, head_depth

    async def fork_context(self, base_turn_id: int) -> tuple[int, int, int]:
        """Fork a context from a specific turn.

        Args:
            base_turn_id: Turn ID to fork from.

        Returns:
            Tuple of (new_context_id, head_turn_id, head_depth).

        Raises:
            ConnectionError: If not connected.
            CXDBProtocolError: If server returns error.
        """
        self._ensure_connected()
        payload = struct.pack("<Q", base_turn_id)
        response = await self._send_and_recv(MSG_CTX_FORK, payload)
        # Response: new_context_id(u64) + head_turn_id(u64) + head_depth(u32) = 20 bytes
        if len(response) < 20:
            raise CXDBProtocolError(
                f"CTX_FORK response too short: {len(response)} bytes"
            )
        new_context_id, head_turn_id, head_depth = struct.unpack("<QQI", response[:20])
        logger.debug("Forked context %d from turn %d", new_context_id, base_turn_id)
        return new_context_id, head_turn_id, head_depth

    async def append_turn(
        self,
        context_id: int,
        payload: dict,
        declared_type_id: str,
        declared_type_version: int = 1,
        parent_turn_id: int = 0,
    ) -> tuple[int, int]:
        """Append a turn to a CXDB context.

        Serializes payload as msgpack, computes BLAKE3 hash, generates
        idempotency key, and sends MSG_APPEND_TURN.

        Args:
            context_id: Target context ID.
            payload: Dict with integer keys (tag numbers) to store.
            declared_type_id: CXDB type identifier (e.g., "amplifier.ToolEvent").
            declared_type_version: Type version number.
            parent_turn_id: Parent turn to append after (0 = current head).

        Returns:
            Tuple of (new_turn_id, new_depth).

        Raises:
            ConnectionError: If not connected.
            CXDBProtocolError: If server returns error.
        """
        self._ensure_connected()

        # Serialize and hash
        msgpack_bytes, content_hash = serialize_payload(payload)

        # Build APPEND_TURN payload
        turn_payload = encode_append_turn_payload(
            context_id=context_id,
            msgpack_bytes=msgpack_bytes,
            content_hash=content_hash,
            declared_type_id=declared_type_id,
            declared_type_version=declared_type_version,
            parent_turn_id=parent_turn_id,
        )

        # Send and receive
        response = await self._send_and_recv(MSG_APPEND_TURN, turn_payload)

        # Parse APPEND_TURN_ACK: context_id(u64) + new_turn_id(u64) + new_depth(u32) + hash(32)
        if len(response) < 20:
            raise CXDBProtocolError(f"APPEND_TURN_ACK too short: {len(response)} bytes")
        _, new_turn_id, new_depth = struct.unpack("<QQI", response[:20])

        logger.debug(
            f"Appended turn {new_turn_id} (depth {new_depth}) to context {context_id}"
        )
        return new_turn_id, new_depth

    async def close(self) -> None:
        """Close the TCP connection."""
        self._connected = False
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass  # Best effort cleanup
            self._writer = None
            self._reader = None

    async def _send_and_recv(self, msg_type: int, payload: bytes) -> bytes:
        """Send a frame and read the response frame.

        Args:
            msg_type: Message type to send.
            payload: Payload bytes.

        Returns:
            Response payload bytes.

        Raises:
            ConnectionError: If send/recv fails.
            CXDBProtocolError: If server returns MSG_ERROR.
        """
        if not self._writer or not self._reader:
            raise ConnectionError("Not connected to CXDB")

        request_id = self._next_request_id()
        frame = encode_frame(msg_type, request_id, payload)

        try:
            self._writer.write(frame)
            await self._writer.drain()

            # Read response header
            header_data = await asyncio.wait_for(
                self._reader.readexactly(FRAME_HEADER_SIZE),
                timeout=self.timeout,
            )
            resp_payload_len, resp_type, _resp_flags, _resp_req_id = struct.unpack(
                FRAME_HEADER_FORMAT, header_data
            )

            # Read response payload
            if resp_payload_len > 0:
                resp_payload = await asyncio.wait_for(
                    self._reader.readexactly(resp_payload_len),
                    timeout=self.timeout,
                )
            else:
                resp_payload = b""

            # Check for error response
            if resp_type == MSG_ERROR:
                if len(resp_payload) >= 8:
                    error_code, detail_len = struct.unpack("<II", resp_payload[:8])
                    detail = resp_payload[8 : 8 + detail_len].decode(
                        "utf-8", errors="replace"
                    )
                    raise CXDBProtocolError(f"CXDB error (code={error_code}): {detail}")
                else:
                    # Fallback for unexpected error format
                    error_msg = resp_payload.decode("utf-8", errors="replace")
                    raise CXDBProtocolError(f"CXDB error: {error_msg}")

            return resp_payload

        except asyncio.TimeoutError as e:
            raise ConnectionError(f"CXDB response timeout after {self.timeout}s") from e
        except (asyncio.IncompleteReadError, OSError) as e:
            self._connected = False
            raise ConnectionError(f"CXDB connection lost: {e}") from e

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected:
            raise ConnectionError("Not connected to CXDB. Call connect() first.")
