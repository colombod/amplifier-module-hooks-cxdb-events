"""CXDBEventHook - main event router coordinating protocol client, buffer, and turn accumulator."""

from __future__ import annotations

import logging
from typing import Any

from amplifier_module_hooks_cxdb_events.buffer import EventBuffer
from amplifier_module_hooks_cxdb_events.protocol import CXDBTcpClient
from amplifier_module_hooks_cxdb_events.schema import (
    build_event_envelope,
    extract_agent_name,
    publish_registry_bundle,
    serialize_envelope,
)
from amplifier_module_hooks_cxdb_events.turns import TurnAccumulator
from amplifier_module_hooks_cxdb_events.types import VariantDeduplicator, get_cxdb_type

logger = logging.getLogger(__name__)

# Events that feed the turn accumulator (for the turns context)
_TURN_EVENTS = frozenset(
    {
        "prompt:submit",
        "content_block:end",
        "tool:pre",
        "tool:post",
        "provider:response",
        "orchestrator:complete",
        "execution:end",
    }
)


class CXDBEventHook:
    """Main event hook that captures Amplifier lifecycle events and stores them in CXDB.

    Manages:
    - Dual CXDB contexts (turns + everything) per root session
    - Event routing to turn accumulator and/or everything context
    - Variant deduplication (:raw > :debug > base)
    - Buffered retry when CXDB is unreachable
    - Graceful degradation (never crashes the session)
    """

    def __init__(
        self,
        client: CXDBTcpClient,
        config: dict[str, Any],
        session_id: str,
        parent_id: str | None,
        root_session_id: str,
    ) -> None:
        self._client = client
        self._config = config
        self._session_id = session_id
        self._parent_id = parent_id
        self._root_session_id = root_session_id
        self._is_root = parent_id is None
        self._agent_name = extract_agent_name(session_id)

        # CXDB context IDs (set during initialize)
        self._turns_context_id: int | None = None
        self._everything_context_id: int | None = None

        # Components
        self._buffer = EventBuffer(max_size=config.get("buffer_size", 1000))
        self._turn_accumulator = TurnAccumulator()
        self._variant_dedup = VariantDeduplicator()

        # State
        self._initialized = False
        self._registry_published = False

    @property
    def is_root(self) -> bool:
        """Whether this hook instance is in the root session."""
        return self._is_root

    @property
    def initialized(self) -> bool:
        """Whether the hook has been initialized (connected + contexts created)."""
        return self._initialized

    @property
    def turns_context_id(self) -> int | None:
        return self._turns_context_id

    @property
    def everything_context_id(self) -> int | None:
        return self._everything_context_id

    async def initialize(self) -> None:
        """Connect to CXDB, publish registry bundle, and create/resolve contexts.

        For root sessions: creates both CXDB contexts.
        For child sessions: resolves context IDs from root session naming convention.

        Safe to call multiple times (idempotent after first success).
        """
        if self._initialized:
            return

        try:
            # Connect to CXDB
            if not self._client.connected:
                await self._client.connect()

            # Publish registry bundle (HTTP, idempotent)
            if not self._registry_published:
                http_port = self._config.get("cxdb_http_port", 9010)
                published = await publish_registry_bundle(self._client.host, http_port)
                if published:
                    self._registry_published = True
                    logger.info("CXDB registry bundle published")
                else:
                    logger.warning(
                        "Failed to publish registry bundle, continuing anyway"
                    )

            # Create or resolve contexts
            if self._is_root:
                # Root session creates both contexts
                self._turns_context_id, _ = await self._client.create_context()
                self._everything_context_id, _ = await self._client.create_context()
                logger.info(
                    f"Created CXDB contexts: turns={self._turns_context_id}, "
                    f"everything={self._everything_context_id}"
                )
            else:
                # Child sessions need to discover root's context IDs.
                # For now, create own contexts (will be linked via session_id in events).
                # TODO: Implement shared context lookup by root_session_id naming convention
                self._turns_context_id, _ = await self._client.create_context()
                self._everything_context_id, _ = await self._client.create_context()
                logger.info(
                    f"Child session created CXDB contexts: turns={self._turns_context_id}, "
                    f"everything={self._everything_context_id}"
                )

            self._initialized = True

        except Exception as e:
            logger.warning(f"CXDB initialization failed: {e}")
            # Don't set _initialized -- will retry on next event

    async def handle_event(self, event: str, data: dict[str, Any]) -> Any:
        """Main event handler. Routes events to accumulator, buffer, and contexts.

        This method is registered as the hook handler for all subscribed events.
        It NEVER raises exceptions -- all errors are caught and logged.

        Args:
            event: Amplifier event name.
            data: Event data dict (includes session_id, parent_id via hook defaults).

        Returns:
            HookResult(action="continue") -- never blocks.
        """
        # Import here to avoid circular dependency issues during testing
        try:
            from amplifier_core.models import HookResult
        except ImportError:
            # Fallback for testing without amplifier-core
            class HookResult:  # type: ignore[no-redef]
                def __init__(self, action: str = "continue") -> None:
                    self.action = action

        try:
            # Lazy initialization on first event
            if not self._initialized:
                await self.initialize()

            # Variant deduplication
            if not self._variant_dedup.should_process(event):
                return HookResult(action="continue")

            # Straggler suppression
            if self._turn_accumulator.is_straggler(event):
                return HookResult(action="continue")

            # Route to turn accumulator for turn-related events
            if event in _TURN_EVENTS:
                self._handle_turn_event(event, data)

            # Write to everything context (all events)
            await self._write_to_everything_context(event, data)

            # Flush turns on orchestrator:complete
            if event == "orchestrator:complete":
                await self._flush_turns()
                self._variant_dedup.reset()

        except Exception as e:
            logger.debug(f"Error handling event {event}: {e}")

        return HookResult(action="continue")

    async def cleanup(self) -> None:
        """Flush remaining buffer and close CXDB connection.

        Called during session teardown. Best-effort -- errors are logged.
        """
        try:
            # Flush any remaining turns
            turn = self._turn_accumulator.flush()
            if turn is not None and self._turns_context_id is not None:
                items = self._turn_accumulator.to_conversation_items(turn)
                for item in items:
                    await self._write_turn(
                        self._turns_context_id, item, "cxdb.ConversationItem"
                    )
        except Exception as e:
            logger.debug(f"Error flushing turns during cleanup: {e}")

        try:
            # Flush event buffer
            if self._buffer.size > 0 and self._client.connected:
                await self._buffer.flush(self._send_turn)
        except Exception as e:
            logger.debug(f"Error flushing buffer during cleanup: {e}")

        try:
            await self._client.close()
        except Exception as e:
            logger.debug(f"Error closing CXDB connection: {e}")

    def _handle_turn_event(self, event: str, data: dict[str, Any]) -> None:
        """Route turn-related events to the turn accumulator.

        Args:
            event: Amplifier event name.
            data: Event data dict.
        """
        if event == "prompt:submit":
            self._turn_accumulator.on_prompt_submit(data)
        elif event == "content_block:end":
            self._turn_accumulator.on_content_block_end(data)
        elif event == "tool:pre":
            self._turn_accumulator.on_tool_pre(data)
        elif event == "tool:post":
            self._turn_accumulator.on_tool_post(data)
        elif event == "provider:response":
            self._turn_accumulator.on_provider_response(data)
        elif event == "execution:end":
            self._turn_accumulator.on_execution_end()

    async def _write_to_everything_context(
        self, event: str, data: dict[str, Any]
    ) -> None:
        """Write an event to the everything context.

        Builds the envelope, serializes, and sends (or buffers on failure).
        """
        if self._everything_context_id is None:
            return

        cxdb_type = get_cxdb_type(event)
        envelope = build_event_envelope(
            event_name=event,
            data=data,
            session_id=self._session_id,
            parent_session_id=self._parent_id,
            agent_name=self._agent_name,
        )
        payload_bytes = serialize_envelope(envelope)

        try:
            if self._client.connected:
                # Flush buffer first (piggyback retry)
                if self._buffer.size > 0:
                    await self._buffer.flush(self._send_turn)

                await self._client.append_turn(
                    context_id=self._everything_context_id,
                    payload=dict(envelope),  # serialize_payload expects dict
                    declared_type_id=cxdb_type.type_id,
                    declared_type_version=cxdb_type.type_version,
                )
            else:
                self._buffer.enqueue(
                    self._everything_context_id,
                    payload_bytes,
                    cxdb_type.type_id,
                    cxdb_type.type_version,
                )
        except Exception:
            # Buffer the event for retry
            self._buffer.enqueue(
                self._everything_context_id,
                payload_bytes,
                cxdb_type.type_id,
                cxdb_type.type_version,
            )

    async def _flush_turns(self) -> None:
        """Flush accumulated turns to the turns context."""
        turn = self._turn_accumulator.flush()
        if turn is None or self._turns_context_id is None:
            return

        items = self._turn_accumulator.to_conversation_items(turn)
        for item in items:
            await self._write_turn(
                self._turns_context_id, item, "cxdb.ConversationItem"
            )

    async def _write_turn(
        self, context_id: int, payload: dict[int, object], type_id: str
    ) -> None:
        """Write a single turn to a CXDB context with error handling."""
        try:
            if self._client.connected:
                await self._client.append_turn(
                    context_id=context_id,
                    payload=payload,
                    declared_type_id=type_id,
                )
            else:
                serialized = serialize_envelope(payload)
                self._buffer.enqueue(context_id, serialized, type_id)
        except Exception:
            serialized = serialize_envelope(payload)
            self._buffer.enqueue(context_id, serialized, type_id)

    async def _send_turn(
        self,
        context_id: int,
        payload_bytes: bytes,
        type_id: str,
        type_version: int,
    ) -> tuple[int, int]:
        """Send a pre-serialized turn via the client. Used as flush callback."""
        import msgpack

        decoded = msgpack.unpackb(payload_bytes, raw=False, strict_map_key=False)
        return await self._client.append_turn(
            context_id=context_id,
            payload=decoded,
            declared_type_id=type_id,
            declared_type_version=type_version,
        )
