"""Tests for binary protocol frame encoding/decoding."""

import struct

import pytest

import blake3 as blake3_mod
import msgpack

from amplifier_module_hooks_cxdb_events.protocol import (
    CXDBTcpClient,
    FRAME_HEADER_SIZE,
    MSG_APPEND_TURN,
    MSG_CTX_CREATE,
    MSG_CTX_FORK,
    MSG_ERROR,
    MSG_GET_HEAD,
    MSG_HELLO,
    decode_frame,
    encode_append_turn_payload,
    encode_frame,
    generate_idempotency_key,
    serialize_payload,
)


class TestEncodeFrame:
    def test_hello_frame(self):
        """MSG_HELLO with empty payload (legacy format)."""
        frame = encode_frame(MSG_HELLO, request_id=1, payload=b"")
        assert len(frame) == FRAME_HEADER_SIZE

    def test_empty_payload(self):
        """Frame with empty payload (e.g., CTX_CREATE)."""
        frame = encode_frame(MSG_CTX_CREATE, request_id=1, payload=b"")
        assert len(frame) == FRAME_HEADER_SIZE

    def test_large_payload(self):
        """Frame with a larger payload."""
        payload = b"x" * 10000
        frame = encode_frame(MSG_APPEND_TURN, request_id=99, payload=payload)
        assert len(frame) == FRAME_HEADER_SIZE + 10000

    def test_flags_default_zero(self):
        """Flags default to 0."""
        frame = encode_frame(MSG_HELLO, request_id=1, payload=b"")
        _, _, flags, _ = struct.unpack("<IHHQ", frame[:FRAME_HEADER_SIZE])
        assert flags == 0

    def test_flags_custom(self):
        """Custom flags are encoded."""
        frame = encode_frame(MSG_HELLO, request_id=1, payload=b"", flags=42)
        _, _, flags, _ = struct.unpack("<IHHQ", frame[:FRAME_HEADER_SIZE])
        assert flags == 42

    def test_request_id_encoded(self):
        """Request ID is correctly encoded in the frame."""
        frame = encode_frame(MSG_CTX_CREATE, request_id=12345, payload=b"")
        _, _, _, req_id = struct.unpack("<IHHQ", frame[:FRAME_HEADER_SIZE])
        assert req_id == 12345

    def test_payload_len_encoded(self):
        """Payload length is correctly encoded in the header."""
        payload = b"hello"
        frame = encode_frame(MSG_HELLO, request_id=1, payload=payload)
        payload_len, _, _, _ = struct.unpack("<IHHQ", frame[:FRAME_HEADER_SIZE])
        assert payload_len == 5


class TestDecodeFrame:
    def test_roundtrip(self):
        """Encode then decode should return original values."""
        payload = b"test_payload"
        frame = encode_frame(MSG_CTX_CREATE, request_id=42, payload=payload)
        msg_type, flags, req_id, decoded_payload = decode_frame(frame)
        assert msg_type == MSG_CTX_CREATE
        assert flags == 0
        assert req_id == 42
        assert decoded_payload == payload

    def test_roundtrip_all_msg_types(self):
        """Roundtrip works for all message types."""
        for msg_type in [
            MSG_HELLO,
            MSG_CTX_CREATE,
            MSG_CTX_FORK,
            MSG_GET_HEAD,
            MSG_APPEND_TURN,
            MSG_ERROR,
        ]:
            frame = encode_frame(msg_type, request_id=1, payload=b"data")
            decoded_type, _, _, _ = decode_frame(frame)
            assert decoded_type == msg_type

    def test_error_frame(self):
        """Error frames decode correctly."""
        error_msg = b"context not found"
        frame = encode_frame(MSG_ERROR, request_id=7, payload=error_msg)
        msg_type, _, req_id, payload = decode_frame(frame)
        assert msg_type == MSG_ERROR
        assert req_id == 7
        assert payload == error_msg

    def test_empty_payload_roundtrip(self):
        """Frames with empty payloads roundtrip correctly."""
        frame = encode_frame(MSG_CTX_CREATE, request_id=1, payload=b"")
        _, _, _, payload = decode_frame(frame)
        assert payload == b""

    def test_too_short_raises(self):
        """Data shorter than header size raises ValueError."""
        with pytest.raises(ValueError, match="Frame too short"):
            decode_frame(b"short")

    def test_incomplete_payload_raises(self):
        """Truncated payload raises ValueError."""
        # Create a valid header claiming 100 bytes payload, but only provide 5
        header = struct.pack("<IHHQ", 100, MSG_HELLO, 0, 1)
        with pytest.raises(ValueError, match="Incomplete payload"):
            decode_frame(header + b"short")

    def test_flags_preserved(self):
        """Custom flags survive encode/decode roundtrip."""
        frame = encode_frame(MSG_HELLO, request_id=1, payload=b"", flags=255)
        _, flags, _, _ = decode_frame(frame)
        assert flags == 255


class TestCXDBTcpClientConnect:
    @pytest.mark.asyncio
    async def test_connect_handshake(self, mock_tcp_server):
        """Client connects and completes MSG_HELLO handshake."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        assert client.connected
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """Connection to unreachable host raises ConnectionError."""
        client = CXDBTcpClient("192.0.2.1", 9999, timeout=0.1)
        with pytest.raises(ConnectionError):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_refused(self):
        """Connection to closed port raises ConnectionError."""
        client = CXDBTcpClient("127.0.0.1", 1, timeout=0.5)
        with pytest.raises(ConnectionError):
            await client.connect()

    @pytest.mark.asyncio
    async def test_close_disconnects(self, mock_tcp_server):
        """Close sets connected to False."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        assert client.connected
        await client.close()
        assert not client.connected

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_tcp_server):
        """Close can be called multiple times safely."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        await client.close()
        await client.close()  # Should not raise


class TestHelloNewFormat:
    def test_client_tag_stored(self):
        """Client tag is stored on the client instance."""
        client = CXDBTcpClient("127.0.0.1", 9999, client_tag="test-client")
        assert client.client_tag == "test-client"

    def test_default_client_tag(self):
        """Default client tag is amplifier-hooks-cxdb."""
        client = CXDBTcpClient("127.0.0.1", 9999)
        assert client.client_tag == "amplifier-hooks-cxdb"

    def test_session_id_none_before_connect(self):
        """Session ID is None before connect."""
        client = CXDBTcpClient("127.0.0.1", 9999)
        assert client.session_id is None

    @pytest.mark.asyncio
    async def test_session_id_stored(self, mock_tcp_server):
        """Session ID from HELLO response is stored."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        assert client.session_id is not None
        assert isinstance(client.session_id, int)
        assert client.session_id == 42  # MockCXDBServer returns 42
        await client.close()

    @pytest.mark.asyncio
    async def test_hello_with_custom_tag(self, mock_tcp_server):
        """HELLO handshake works with custom client tag."""
        client = CXDBTcpClient(
            "127.0.0.1", mock_tcp_server.port, client_tag="my-custom-tag"
        )
        await client.connect()
        assert client.connected
        assert client.session_id is not None
        await client.close()


class TestGetHead:
    @pytest.mark.asyncio
    async def test_get_head(self, mock_tcp_server):
        """GetHead returns head_turn_id and head_depth."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        ctx_id, _, _ = await client.create_context()
        head_turn_id, head_depth = await client.get_head(ctx_id)
        assert isinstance(head_turn_id, int)
        assert isinstance(head_depth, int)
        await client.close()

    @pytest.mark.asyncio
    async def test_get_head_not_connected(self):
        """GetHead without connect raises ConnectionError."""
        client = CXDBTcpClient("127.0.0.1", 9999)
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.get_head(1)


class TestCXDBTcpClientContextOps:
    @pytest.mark.asyncio
    async def test_create_context(self, mock_tcp_server):
        """MSG_CTX_CREATE returns valid context_id and head_turn_id."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        context_id, head_turn_id, head_depth = await client.create_context()
        assert isinstance(context_id, int)
        assert context_id > 0
        assert isinstance(head_turn_id, int)
        assert isinstance(head_depth, int)
        await client.close()

    @pytest.mark.asyncio
    async def test_create_multiple_contexts(self, mock_tcp_server):
        """Multiple context creates return unique IDs."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        ctx1, _, _ = await client.create_context()
        ctx2, _, _ = await client.create_context()
        assert ctx1 != ctx2
        await client.close()

    @pytest.mark.asyncio
    async def test_fork_context(self, mock_tcp_server):
        """MSG_CTX_FORK returns new context with correct types."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        ctx_id, head, _ = await client.create_context()
        new_ctx, new_head, depth = await client.fork_context(head)
        assert isinstance(new_ctx, int)
        assert new_ctx != ctx_id
        assert isinstance(depth, int)
        await client.close()

    @pytest.mark.asyncio
    async def test_create_context_not_connected(self):
        """Operations without connect raise ConnectionError."""
        client = CXDBTcpClient("127.0.0.1", 9999)
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.create_context()

    @pytest.mark.asyncio
    async def test_fork_context_not_connected(self):
        """Fork without connect raises ConnectionError."""
        client = CXDBTcpClient("127.0.0.1", 9999)
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.fork_context(1)


class TestAppendTurnPayload:
    def test_payload_starts_with_context_id(self):
        """First 8 bytes are context_id as little-endian u64."""
        payload_data = {1: "test_event", 2: "session-123"}
        msgpack_bytes, content_hash = serialize_payload(payload_data)
        encoded = encode_append_turn_payload(
            context_id=42,
            msgpack_bytes=msgpack_bytes,
            content_hash=content_hash,
            declared_type_id="amplifier.GenericEvent",
        )
        context_id = struct.unpack("<Q", encoded[0:8])[0]
        assert context_id == 42

    def test_parent_turn_id_follows_context_id(self):
        """Bytes 8-16 are parent_turn_id as little-endian u64."""
        payload_data = {1: "test"}
        msgpack_bytes, content_hash = serialize_payload(payload_data)
        encoded = encode_append_turn_payload(
            context_id=1,
            msgpack_bytes=msgpack_bytes,
            content_hash=content_hash,
            declared_type_id="amplifier.GenericEvent",
            parent_turn_id=99,
        )
        parent_turn_id = struct.unpack("<Q", encoded[8:16])[0]
        assert parent_turn_id == 99

    def test_default_parent_turn_id_zero(self):
        """Default parent_turn_id is 0 (append to head)."""
        payload_data = {1: "test"}
        msgpack_bytes, content_hash = serialize_payload(payload_data)
        encoded = encode_append_turn_payload(
            context_id=1,
            msgpack_bytes=msgpack_bytes,
            content_hash=content_hash,
            declared_type_id="amplifier.GenericEvent",
        )
        parent_turn_id = struct.unpack("<Q", encoded[8:16])[0]
        assert parent_turn_id == 0

    def test_type_id_encoded_as_length_prefixed_utf8(self):
        """Type ID is length-prefixed UTF-8 string."""
        payload_data = {1: "test"}
        msgpack_bytes, content_hash = serialize_payload(payload_data)
        type_id = "amplifier.ToolEvent"
        encoded = encode_append_turn_payload(
            context_id=1,
            msgpack_bytes=msgpack_bytes,
            content_hash=content_hash,
            declared_type_id=type_id,
        )
        # After context_id(8) + parent_turn_id(8) = offset 16
        type_id_len = struct.unpack("<I", encoded[16:20])[0]
        assert type_id_len == len(type_id.encode("utf-8"))
        extracted_type_id = encoded[20 : 20 + type_id_len].decode("utf-8")
        assert extracted_type_id == type_id


class TestIdempotencyKey:
    def test_deterministic(self):
        """Same inputs produce same key."""
        content_hash = blake3_mod.blake3(b"test_data").digest()
        key1 = generate_idempotency_key(42, content_hash)
        key2 = generate_idempotency_key(42, content_hash)
        assert key1 == key2

    def test_unique_per_context(self):
        """Different context IDs produce different keys."""
        content_hash = blake3_mod.blake3(b"test_data").digest()
        key1 = generate_idempotency_key(1, content_hash)
        key2 = generate_idempotency_key(2, content_hash)
        assert key1 != key2

    def test_unique_per_content(self):
        """Different content produces different keys."""
        hash1 = blake3_mod.blake3(b"data_a").digest()
        hash2 = blake3_mod.blake3(b"data_b").digest()
        key1 = generate_idempotency_key(1, hash1)
        key2 = generate_idempotency_key(1, hash2)
        assert key1 != key2

    def test_key_is_32_bytes(self):
        """Idempotency key is SHA-256 digest (32 bytes)."""
        content_hash = blake3_mod.blake3(b"test").digest()
        key = generate_idempotency_key(1, content_hash)
        assert len(key) == 32


class TestSerializePayload:
    def test_integer_keys(self):
        """Serialized payload uses integer keys."""
        payload = {1: "session:start", 2: "abc-123", 4: 1707600000000}
        msgpack_bytes, _ = serialize_payload(payload)
        decoded = msgpack.unpackb(msgpack_bytes, raw=False, strict_map_key=False)
        assert all(isinstance(k, int) for k in decoded.keys())

    def test_values_preserved(self):
        """Payload values survive serialization roundtrip."""
        payload = {1: "event_name", 2: "session-id", 4: 999}
        msgpack_bytes, _ = serialize_payload(payload)
        decoded = msgpack.unpackb(msgpack_bytes, raw=False, strict_map_key=False)
        assert decoded[1] == "event_name"
        assert decoded[2] == "session-id"
        assert decoded[4] == 999

    def test_blake3_hash_is_32_bytes(self):
        """Content hash is 32 bytes (BLAKE3-256)."""
        payload = {1: "test"}
        _, content_hash = serialize_payload(payload)
        assert len(content_hash) == 32

    def test_deterministic_encoding(self):
        """Same payload produces same bytes (sorted keys)."""
        payload1 = {3: "c", 1: "a", 2: "b"}
        payload2 = {1: "a", 2: "b", 3: "c"}
        bytes1, hash1 = serialize_payload(payload1)
        bytes2, hash2 = serialize_payload(payload2)
        assert bytes1 == bytes2
        assert hash1 == hash2

    def test_nested_values(self):
        """Nested dicts and lists serialize correctly."""
        payload = {
            1: "tool:post",
            10: {"tool_name": "grep", "result": "5 matches"},
            11: [1, 2, 3],
        }
        msgpack_bytes, _ = serialize_payload(payload)
        decoded = msgpack.unpackb(msgpack_bytes, raw=False, strict_map_key=False)
        assert decoded[10]["tool_name"] == "grep"
        assert decoded[11] == [1, 2, 3]


class TestClientAppendTurn:
    @pytest.mark.asyncio
    async def test_append_turn(self, mock_tcp_server):
        """append_turn sends MSG_APPEND_TURN and returns turn_id, depth."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        ctx_id, _, _ = await client.create_context()
        turn_id, depth = await client.append_turn(
            context_id=ctx_id,
            payload={1: "session:start", 2: "test-session"},
            declared_type_id="amplifier.SessionEvent",
        )
        assert isinstance(turn_id, int)
        assert turn_id > 0
        assert isinstance(depth, int)
        await client.close()

    @pytest.mark.asyncio
    async def test_append_multiple_turns(self, mock_tcp_server):
        """Multiple appends return incrementing turn IDs."""
        client = CXDBTcpClient("127.0.0.1", mock_tcp_server.port)
        await client.connect()
        ctx_id, _, _ = await client.create_context()
        tid1, _ = await client.append_turn(
            context_id=ctx_id,
            payload={1: "event1"},
            declared_type_id="amplifier.GenericEvent",
        )
        tid2, _ = await client.append_turn(
            context_id=ctx_id,
            payload={1: "event2"},
            declared_type_id="amplifier.GenericEvent",
        )
        assert tid2 > tid1
        await client.close()

    @pytest.mark.asyncio
    async def test_append_turn_not_connected(self):
        """append_turn without connect raises ConnectionError."""
        client = CXDBTcpClient("127.0.0.1", 9999)
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.append_turn(
                context_id=1,
                payload={1: "test"},
                declared_type_id="amplifier.GenericEvent",
            )


class TestErrorResponseParsing:
    def test_error_response_parsed_correctly(self):
        """Error frames with binary prefix are parsed correctly."""
        # Build error payload: code(u32) + detail_len(u32) + detail_text
        error_code = 404
        detail = b"context not found"
        error_payload = struct.pack("<II", error_code, len(detail)) + detail

        # Build error frame
        error_frame = encode_frame(MSG_ERROR, request_id=1, payload=error_payload)

        # Decode and verify the payload is structured
        _, _, _, payload = decode_frame(error_frame)
        code, detail_len = struct.unpack("<II", payload[:8])
        assert code == 404
        detail_text = payload[8 : 8 + detail_len].decode("utf-8")
        assert detail_text == "context not found"
