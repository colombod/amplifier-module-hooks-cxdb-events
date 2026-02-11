"""TurnAccumulator - buffers conversation data during orchestrator loop, flushes on complete."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation during an orchestrator cycle."""

    tool_name: str
    input_summary: str
    call_id: str | None = None
    result: str | None = None
    error: str | None = None
    has_result: bool = False


@dataclass
class AccumulatedTurn:
    """Data accumulated during one orchestrator cycle (prompt -> response)."""

    user_prompt: str | None = None
    assistant_text_blocks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    metrics: dict | None = None
    agent_name: str | None = None
    finish_reason: str | None = None


class TurnAccumulator:
    """Buffers conversation data during an orchestrator cycle.

    Accumulates user prompts, tool call records, assistant text blocks,
    and provider metrics. Flushes a complete turn on orchestrator:complete.

    Handles:
    - Tool result matching (tool:post matched to tool:pre by tool_name)
    - Post-execution straggler suppression (llm:* after execution:end)
    - Anthropic token math (total = input + cache_read + cache_creation)
    - ConversationItem v3 output format
    """

    def __init__(self) -> None:
        self._current = AccumulatedTurn()
        self._execution_ended = False
        self._request_start_mono: float | None = None

    def on_prompt_submit(self, data: dict) -> None:
        """Buffer user message from prompt:submit event.

        Args:
            data: Event data containing 'prompt' field.
        """
        prompt = data.get("prompt")
        if prompt is not None:
            self._current.user_prompt = str(prompt)
            # Reset execution_ended for new cycle
            self._execution_ended = False

    def on_tool_pre(self, data: dict) -> None:
        """Start a tool call record from tool:pre event.

        Args:
            data: Event data with tool_name, tool_input, etc.
        """
        tool_name = data.get("tool_name", "unknown")
        tool_input = data.get("tool_input", {})
        # Amplifier uses "tool_call_id", some events use "call_id"
        call_id = data.get("tool_call_id") or data.get("call_id")

        # Create abbreviated input summary
        if isinstance(tool_input, dict):
            summary = str(tool_input)[:200]
        else:
            summary = str(tool_input)[:200]

        record = ToolCallRecord(
            tool_name=tool_name,
            input_summary=summary,
            call_id=call_id,
        )
        self._current.tool_calls.append(record)

    def on_tool_post(self, data: dict) -> None:
        """Complete a tool call record from tool:post event.

        Matches by call_id first (preferred, handles parallel same-tool calls),
        then falls back to tool_name matching for events without call_id.

        Args:
            data: Event data with tool_name, result, etc.
        """
        tool_name = data.get("tool_name", "unknown")
        call_id = data.get("tool_call_id") or data.get("call_id")
        result = data.get("result")
        error = data.get("error")

        # Try matching by call_id first (handles parallel same-tool calls)
        if call_id:
            for record in reversed(self._current.tool_calls):
                if record.call_id == call_id and not record.has_result:
                    record.has_result = True
                    if result is not None:
                        record.result = str(result)[:500]
                    if error is not None:
                        record.error = str(error)[:500]
                    return

        # Fall back to tool_name matching (for events without call_id)
        for record in reversed(self._current.tool_calls):
            if record.tool_name == tool_name and not record.has_result:
                record.has_result = True
                if result is not None:
                    record.result = str(result)[:500]
                if error is not None:
                    record.error = str(error)[:500]
                return

    def on_content_block_end(self, data: dict) -> None:
        """Accumulate assistant text block from content_block:end event.

        Only processes blocks with block_type='text'. Other block types
        (tool_use, etc.) are ignored for the turns context.

        Handles two data layouts:
        - Top-level: data["block_type"] = "text", data["block"] = "text content"
        - Nested:    data["block"] = {"type": "text", "text": "text content"}

        Args:
            data: Event data with block_type and/or block fields.
        """
        block = data.get("block")

        # Determine block type from either top-level field or nested block.type
        block_type = data.get("block_type")
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")

        if block_type != "text":
            return

        # Extract text from either a dict block or a plain string
        if isinstance(block, dict):
            text = block.get("text", "")
        elif block is not None:
            text = str(block)
        else:
            return

        if text:
            self._current.assistant_text_blocks.append(text)

    def on_provider_request(self, data: dict) -> None:
        """Capture start time for latency measurement."""
        import time

        self._request_start_mono = time.monotonic()

    def on_provider_response(self, data: dict) -> None:
        """Capture metrics from provider:response event.

        Handles Anthropic token math:
        total_input = input_tokens + cache_read_input_tokens + cache_creation_input_tokens

        Args:
            data: Event data with usage, provider, model fields.
        """
        usage = data.get("usage", {})
        if not usage:
            return

        # Calculate total input tokens (Anthropic-aware)
        input_tokens = usage.get("input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
        total_input = input_tokens + cache_read + cache_creation

        output_tokens = usage.get("output_tokens", 0) or 0
        reasoning_tokens = (
            usage.get("reasoning_tokens", 0) or usage.get("thinking_tokens", 0) or 0
        )
        # Check nested OpenAI-style reasoning tokens
        completion_details = usage.get("completion_tokens_details", {})
        if completion_details and not reasoning_tokens:
            reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0

        self._current.metrics = {
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
            "total_input_tokens": total_input,
            "output_tokens": output_tokens,
            "total_tokens": total_input + output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cache_read,
            "model": data.get("model") or usage.get("model", ""),
            "provider": data.get("provider", ""),
        }

        # Compute wall-clock latency if provider:request was seen
        if self._request_start_mono is not None:
            import time

            duration_ms = int((time.monotonic() - self._request_start_mono) * 1000)
            self._current.metrics["duration_ms"] = duration_ms
            self._request_start_mono = None

    def on_execution_end(self) -> None:
        """Mark execution as ended for straggler suppression."""
        self._execution_ended = True

    def is_straggler(self, event_name: str) -> bool:
        """Check if an event should be suppressed as a post-execution straggler.

        After execution:end, llm:* events from summarization hooks
        should not be included in the conversation transcript.

        Args:
            event_name: Amplifier event name.

        Returns:
            True if the event should be suppressed.
        """
        if not self._execution_ended:
            return False
        return event_name.startswith("llm:")

    def flush(self) -> AccumulatedTurn | None:
        """Return the accumulated turn and reset state.

        Returns:
            AccumulatedTurn if there is data to flush, None if empty.
        """
        turn = self._current

        # Check if there's anything meaningful to flush
        has_data = (
            turn.user_prompt is not None
            or turn.assistant_text_blocks
            or turn.tool_calls
        )

        # Reset state
        self._current = AccumulatedTurn()
        self._execution_ended = False
        self._request_start_mono = None

        return turn if has_data else None

    def to_conversation_items(self, turn: AccumulatedTurn) -> list[dict[int, object]]:
        """Convert an AccumulatedTurn to ConversationItem v3 msgpack dicts.

        Produces up to 2 items: one user_input and one assistant_turn.
        Uses Brian's discriminated union layout matching the CXDB UI:

        ConversationItem envelope (tags 1-4):
          tag 1: item_type  ("user_input" | "assistant_turn")
          tag 2: status     ("complete")
          tag 3: timestamp  (i64, unix_ms)
          tag 4: id         (deterministic hash)

        Variant subtrees (exactly one populated):
          tag 10: UserInput   {1: text}
          tag 11: AssistantTurn {1: text, 2: tool_calls[], 3: reasoning,
                                 4: TurnMetrics, 5: agent, 8: finish_reason}

        Args:
            turn: The accumulated turn data.

        Returns:
            List of msgpack-ready dicts (0-2 items).
        """
        items: list[dict[int, object]] = []
        import hashlib
        import time

        timestamp_ms = int(time.time() * 1000)

        def _make_id(label: str) -> str:
            raw = f"{label}:{timestamp_ms}"
            return hashlib.sha256(raw.encode()).hexdigest()[:24]

        # User input item
        if turn.user_prompt is not None:
            user_item: dict[int, object] = {
                1: "user_input",
                2: "complete",
                3: timestamp_ms,
                4: _make_id("user_input"),
                10: {1: turn.user_prompt},  # UserInput.text
            }
            items.append(user_item)

        # Assistant turn item
        has_assistant_data = turn.assistant_text_blocks or turn.tool_calls
        if has_assistant_data:
            text = "".join(turn.assistant_text_blocks)

            # Build the AssistantTurn subtree (tag 11)
            assistant_turn: dict[int, object] = {}

            if text:
                assistant_turn[1] = text  # AssistantTurn.text

            if turn.tool_calls:
                assistant_turn[2] = [  # AssistantTurn.tool_calls
                    self._build_tool_call_item(tc)
                    for tc in turn.tool_calls
                ]

            if turn.metrics:
                assistant_turn[4] = self._build_metrics(turn.metrics)

            if turn.agent_name:
                assistant_turn[5] = turn.agent_name  # AssistantTurn.agent

            if turn.finish_reason:
                assistant_turn[8] = turn.finish_reason  # AssistantTurn.finish_reason

            assistant_item: dict[int, object] = {
                1: "assistant_turn",
                2: "complete",
                3: timestamp_ms,
                4: _make_id("assistant_turn"),
                11: assistant_turn,  # ConversationItem.turn
            }
            items.append(assistant_item)

        return items

    @staticmethod
    def _build_tool_call_item(tc: ToolCallRecord) -> dict[int, object]:
        """Build a ToolCallItem dict matching cxdb.ToolCallItem v3 tags."""
        item: dict[int, object] = {
            1: tc.call_id or "",  # ToolCallItem.id
            2: tc.tool_name,  # ToolCallItem.name
            3: tc.input_summary,  # ToolCallItem.args
            4: "complete" if tc.has_result else "pending",  # ToolCallItem.status
        }
        if tc.result is not None:
            item[8] = {1: tc.result, 3: True}  # ToolCallResult {content, success}
        if tc.error is not None:
            item[9] = {2: tc.error}  # ToolCallError {message}
        return item

    @staticmethod
    def _build_metrics(metrics: dict) -> dict[int, object]:
        """Build a TurnMetrics dict matching cxdb.TurnMetrics v3 tags."""
        m: dict[int, object] = {}
        if "input_tokens" in metrics:
            m[1] = metrics["input_tokens"]
        if "output_tokens" in metrics:
            m[2] = metrics["output_tokens"]
        if "total_tokens" in metrics:
            m[3] = metrics["total_tokens"]
        if "cached_tokens" in metrics:
            m[4] = metrics["cached_tokens"]
        if "reasoning_tokens" in metrics:
            m[5] = metrics["reasoning_tokens"]
        if "duration_ms" in metrics:
            m[6] = metrics["duration_ms"]
        if "model" in metrics:
            m[7] = metrics["model"]
        if "provider" in metrics:
            m[8] = metrics["provider"]
        return m
