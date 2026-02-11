"""Tests for EventBuffer - in-memory deque with retry logic."""

import pytest

from amplifier_module_hooks_cxdb_events.buffer import EventBuffer


class TestEventBufferBasic:
    def test_empty_on_init(self):
        """Buffer starts empty."""
        buf = EventBuffer(max_size=10)
        assert buf.size == 0
        assert buf.overflow_count == 0
        assert buf.total_enqueued == 0
        assert buf.total_sent == 0

    def test_enqueue_increases_size(self):
        """Enqueue adds items to the buffer."""
        buf = EventBuffer(max_size=10)
        buf.enqueue(1, b"event1", "amplifier.GenericEvent")
        assert buf.size == 1
        buf.enqueue(1, b"event2", "amplifier.GenericEvent")
        assert buf.size == 2

    def test_total_enqueued_tracks_all(self):
        """total_enqueued counts every enqueue, including overflows."""
        buf = EventBuffer(max_size=2)
        buf.enqueue(1, b"e1", "t")
        buf.enqueue(1, b"e2", "t")
        buf.enqueue(1, b"e3", "t")  # overflow
        assert buf.total_enqueued == 3

    def test_max_size_property(self):
        """max_size returns configured capacity."""
        buf = EventBuffer(max_size=42)
        assert buf.max_size == 42

    def test_repr(self):
        """repr shows useful state."""
        buf = EventBuffer(max_size=10)
        buf.enqueue(1, b"e1", "t")
        r = repr(buf)
        assert "size=1" in r
        assert "max_size=10" in r


class TestEventBufferOverflow:
    def test_drops_oldest_on_overflow(self):
        """When full, oldest event is dropped."""
        buf = EventBuffer(max_size=3)
        buf.enqueue(1, b"event1", "amplifier.GenericEvent")
        buf.enqueue(1, b"event2", "amplifier.GenericEvent")
        buf.enqueue(1, b"event3", "amplifier.GenericEvent")
        assert buf.size == 3
        assert buf.overflow_count == 0

        buf.enqueue(1, b"event4", "amplifier.GenericEvent")
        assert buf.size == 3  # still 3
        assert buf.overflow_count == 1

    def test_overflow_count_increments(self):
        """Each overflow increments the counter."""
        buf = EventBuffer(max_size=2)
        buf.enqueue(1, b"e1", "t")
        buf.enqueue(1, b"e2", "t")
        buf.enqueue(1, b"e3", "t")  # overflow 1
        buf.enqueue(1, b"e4", "t")  # overflow 2
        assert buf.overflow_count == 2

    def test_overflow_preserves_newest(self):
        """After overflow, buffer contains the newest events."""
        buf = EventBuffer(max_size=2)
        buf.enqueue(1, b"old", "t")
        buf.enqueue(1, b"mid", "t")
        buf.enqueue(1, b"new", "t")  # drops "old"
        # Verify by flushing
        sent = []

        async def capture(ctx, payload, type_id, type_ver):
            sent.append(payload)
            return (1, 0)

        import asyncio
        asyncio.get_event_loop().run_until_complete(buf.flush(capture))
        assert sent == [b"mid", b"new"]


class TestEventBufferFlush:
    @pytest.mark.asyncio
    async def test_flush_sends_all(self):
        """Flush sends all buffered events."""
        buf = EventBuffer(max_size=10)
        sent = []

        async def mock_send(ctx, payload, type_id, type_ver):
            sent.append(payload)
            return (1, 0)

        buf.enqueue(1, b"event1", "amplifier.GenericEvent")
        buf.enqueue(1, b"event2", "amplifier.GenericEvent")
        count = await buf.flush(mock_send)
        assert count == 2
        assert buf.size == 0
        assert sent == [b"event1", b"event2"]

    @pytest.mark.asyncio
    async def test_flush_fifo_order(self):
        """Events are sent in FIFO order."""
        buf = EventBuffer(max_size=10)
        order = []

        async def track_order(ctx, payload, type_id, type_ver):
            order.append(payload.decode())
            return (1, 0)

        buf.enqueue(1, b"first", "t")
        buf.enqueue(1, b"second", "t")
        buf.enqueue(1, b"third", "t")
        await buf.flush(track_order)
        assert order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_flush_stops_on_error(self):
        """Flush stops on first error, keeping remaining events."""
        buf = EventBuffer(max_size=10)
        call_count = 0

        async def failing_send(ctx, payload, type_id, type_ver):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("CXDB unreachable")
            return (1, 0)

        buf.enqueue(1, b"event1", "t")
        buf.enqueue(1, b"event2", "t")
        buf.enqueue(1, b"event3", "t")
        count = await buf.flush(failing_send)
        assert count == 1  # only first succeeded
        assert buf.size == 2  # remaining stay in buffer

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self):
        """Flushing empty buffer returns 0."""
        buf = EventBuffer(max_size=10)

        async def mock_send(ctx, payload, type_id, type_ver):
            return (1, 0)

        count = await buf.flush(mock_send)
        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_all_fail(self):
        """When first event fails, nothing is sent."""
        buf = EventBuffer(max_size=10)

        async def always_fail(ctx, payload, type_id, type_ver):
            raise ConnectionError("down")

        buf.enqueue(1, b"event1", "t")
        buf.enqueue(1, b"event2", "t")
        count = await buf.flush(always_fail)
        assert count == 0
        assert buf.size == 2  # all retained

    @pytest.mark.asyncio
    async def test_flush_tracks_total_sent(self):
        """total_sent increments across multiple flushes."""
        buf = EventBuffer(max_size=10)

        async def mock_send(ctx, payload, type_id, type_ver):
            return (1, 0)

        buf.enqueue(1, b"e1", "t")
        buf.enqueue(1, b"e2", "t")
        await buf.flush(mock_send)
        assert buf.total_sent == 2

        buf.enqueue(1, b"e3", "t")
        await buf.flush(mock_send)
        assert buf.total_sent == 3

    @pytest.mark.asyncio
    async def test_flush_preserves_type_info(self):
        """Flush passes correct type_id and type_version to send function."""
        buf = EventBuffer(max_size=10)
        received = []

        async def capture_all(ctx, payload, type_id, type_ver):
            received.append((ctx, type_id, type_ver))
            return (1, 0)

        buf.enqueue(42, b"data", "amplifier.ToolEvent", 2)
        await buf.flush(capture_all)
        assert received == [(42, "amplifier.ToolEvent", 2)]


class TestEventBufferClear:
    def test_clear_empties_buffer(self):
        """Clear removes all events."""
        buf = EventBuffer(max_size=10)
        buf.enqueue(1, b"e1", "t")
        buf.enqueue(1, b"e2", "t")
        buf.clear()
        assert buf.size == 0

    def test_clear_preserves_counters(self):
        """Clear doesn't reset overflow/enqueued counters."""
        buf = EventBuffer(max_size=2)
        buf.enqueue(1, b"e1", "t")
        buf.enqueue(1, b"e2", "t")
        buf.enqueue(1, b"e3", "t")  # overflow
        buf.clear()
        assert buf.overflow_count == 1
        assert buf.total_enqueued == 3

    def test_clear_allows_reuse(self):
        """Buffer works normally after clear."""
        buf = EventBuffer(max_size=10)
        buf.enqueue(1, b"before", "t")
        buf.clear()
        buf.enqueue(1, b"after", "t")
        assert buf.size == 1
