"""EventBuffer - in-memory deque with configurable max size and piggyback retry."""

from __future__ import annotations

import collections
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Type alias for the send function signature
SendFn = Callable[[int, bytes, str, int], Awaitable[tuple[int, int]]]


class EventBuffer:
    """In-memory buffer for CXDB events with configurable max size.

    When CXDB is unreachable, events queue in the buffer. On each new event,
    the buffer attempts to flush all queued events first (piggyback retry).
    When the buffer exceeds max size, oldest events are dropped automatically.

    Each buffered item is a tuple of:
        (context_id, msgpack_bytes, declared_type_id, declared_type_version)
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the event buffer.

        Args:
            max_size: Maximum number of events to buffer. When exceeded,
                      oldest events are dropped automatically.
        """
        self._buffer: collections.deque[tuple[int, bytes, str, int]] = (
            collections.deque(maxlen=max_size)
        )
        self._max_size = max_size
        self._overflow_count: int = 0
        self._total_enqueued: int = 0
        self._total_sent: int = 0

    @property
    def size(self) -> int:
        """Current number of buffered events."""
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        """Maximum buffer capacity."""
        return self._max_size

    @property
    def overflow_count(self) -> int:
        """Number of events dropped due to buffer overflow."""
        return self._overflow_count

    @property
    def total_enqueued(self) -> int:
        """Total number of events ever enqueued."""
        return self._total_enqueued

    @property
    def total_sent(self) -> int:
        """Total number of events successfully sent."""
        return self._total_sent

    def enqueue(
        self,
        context_id: int,
        payload: bytes,
        declared_type_id: str,
        declared_type_version: int = 1,
    ) -> None:
        """Add an event to the buffer.

        If the buffer is full, the oldest event is dropped automatically
        (deque maxlen behavior).

        Args:
            context_id: Target CXDB context ID.
            payload: Pre-serialized msgpack bytes.
            declared_type_id: CXDB type identifier.
            declared_type_version: CXDB type version.
        """
        was_full = len(self._buffer) == self._max_size
        self._buffer.append(
            (context_id, payload, declared_type_id, declared_type_version)
        )
        self._total_enqueued += 1

        if was_full:
            self._overflow_count += 1
            if self._overflow_count == 1 or self._overflow_count % 100 == 0:
                logger.warning(
                    f"Event buffer overflow: {self._overflow_count} events dropped "
                    f"(buffer size: {self._max_size})"
                )

    async def flush(self, send_fn: SendFn) -> int:
        """Attempt to send all buffered events via the send function.

        Sends events in FIFO order. Stops on the first error (keeping
        remaining events in the buffer for retry).

        Args:
            send_fn: Async callable with signature
                     (context_id, payload, type_id, type_version) -> (turn_id, depth)

        Returns:
            Number of events successfully sent.
        """
        sent = 0
        while self._buffer:
            context_id, payload, type_id, type_version = self._buffer[0]
            try:
                await send_fn(context_id, payload, type_id, type_version)
                self._buffer.popleft()
                sent += 1
                self._total_sent += 1
            except Exception:
                # Stop flushing on first error -- keep remaining in buffer
                if sent == 0:
                    logger.debug("Buffer flush failed on first event, will retry later")
                else:
                    logger.debug(
                        f"Buffer flush stopped after {sent} events, "
                        f"{len(self._buffer)} remaining"
                    )
                break
        return sent

    def clear(self) -> None:
        """Clear all buffered events."""
        self._buffer.clear()

    def __repr__(self) -> str:
        return (
            f"EventBuffer(size={self.size}, max_size={self._max_size}, "
            f"overflow={self._overflow_count})"
        )
