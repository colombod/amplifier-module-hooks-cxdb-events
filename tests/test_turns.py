"""Tests for TurnAccumulator - conversation turn buffering and flush."""

from amplifier_module_hooks_cxdb_events.turns import (
    AccumulatedTurn,
    ToolCallRecord,
    TurnAccumulator,
)


class TestTurnAccumulatorPrompt:
    def test_accumulate_user_prompt(self):
        """prompt:submit buffers the user message."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": "Find all Python files"})
        turn = acc.flush()
        assert turn is not None
        assert turn.user_prompt == "Find all Python files"

    def test_prompt_overwrites_previous(self):
        """Second prompt:submit overwrites the first (new cycle)."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": "first"})
        acc.on_prompt_submit({"prompt": "second"})
        turn = acc.flush()
        assert turn.user_prompt == "second"

    def test_prompt_missing_field(self):
        """Missing 'prompt' field is handled gracefully."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({})
        turn = acc.flush()
        assert turn is None  # nothing meaningful

    def test_prompt_none_value(self):
        """None prompt value is handled."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": None})
        turn = acc.flush()
        assert turn is None


class TestTurnAccumulatorContentBlocks:
    def test_accumulate_text_block(self):
        """content_block:end with text type accumulates."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": "Hello"})
        acc.on_content_block_end({"block_type": "text", "block": "Hi there!"})
        turn = acc.flush()
        assert turn.assistant_text_blocks == ["Hi there!"]

    def test_multiple_text_blocks(self):
        """Multiple text blocks accumulate in order."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": "Hello"})
        acc.on_content_block_end({"block_type": "text", "block": "Part 1. "})
        acc.on_content_block_end({"block_type": "text", "block": "Part 2."})
        turn = acc.flush()
        assert len(turn.assistant_text_blocks) == 2
        assert turn.assistant_text_blocks[0] == "Part 1. "
        assert turn.assistant_text_blocks[1] == "Part 2."

    def test_ignore_tool_use_blocks(self):
        """Non-text block types are ignored."""
        acc = TurnAccumulator()
        acc.on_content_block_end({"block_type": "tool_use", "block": {"name": "grep"}})
        turn = acc.flush()
        assert turn is None

    def test_ignore_thinking_blocks(self):
        """Thinking block type is ignored."""
        acc = TurnAccumulator()
        acc.on_content_block_end({"block_type": "thinking", "block": "reasoning..."})
        turn = acc.flush()
        assert turn is None

    def test_text_block_without_prompt(self):
        """Text blocks without a prompt still accumulate (edge case)."""
        acc = TurnAccumulator()
        acc.on_content_block_end({"block_type": "text", "block": "orphan response"})
        turn = acc.flush()
        assert turn is not None
        assert turn.assistant_text_blocks == ["orphan response"]


class TestTurnAccumulatorToolCalls:
    def test_tool_pre_creates_record(self):
        """tool:pre starts a new tool call record."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": "search"})
        acc.on_tool_pre({"tool_name": "grep", "tool_input": {"pattern": "def"}})
        turn = acc.flush()
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].tool_name == "grep"
        assert turn.tool_calls[0].has_result is False

    def test_tool_post_completes_record(self):
        """tool:post matches and completes the tool call record."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": "search"})
        acc.on_tool_pre({"tool_name": "grep", "tool_input": {"pattern": "def"}})
        acc.on_tool_post({"tool_name": "grep", "result": "5 matches"})
        turn = acc.flush()
        assert turn.tool_calls[0].has_result is True
        assert "5 matches" in turn.tool_calls[0].result

    def test_multiple_tools_matched_correctly(self):
        """Multiple tools are matched by name in reverse order."""
        acc = TurnAccumulator()
        acc.on_tool_pre({"tool_name": "grep", "tool_input": {"pattern": "test"}})
        acc.on_tool_pre({"tool_name": "glob", "tool_input": {"pattern": "*.py"}})
        acc.on_tool_post({"tool_name": "glob", "result": "found files"})
        acc.on_tool_post({"tool_name": "grep", "result": "found matches"})
        turn = acc.flush()
        assert len(turn.tool_calls) == 2
        assert all(tc.has_result for tc in turn.tool_calls)

    def test_tool_error(self):
        """Tool errors are captured."""
        acc = TurnAccumulator()
        acc.on_tool_pre({"tool_name": "bash", "tool_input": {"command": "bad"}})
        acc.on_tool_post({"tool_name": "bash", "error": "command failed"})
        turn = acc.flush()
        assert turn.tool_calls[0].error is not None
        assert "command failed" in turn.tool_calls[0].error

    def test_tool_input_summary_truncated(self):
        """Long tool inputs are truncated in summary."""
        acc = TurnAccumulator()
        long_input = {"data": "x" * 1000}
        acc.on_tool_pre({"tool_name": "test", "tool_input": long_input})
        turn = acc.flush()
        assert len(turn.tool_calls[0].input_summary) <= 200


class TestTurnAccumulatorMetrics:
    def test_basic_metrics(self):
        """Provider response captures basic metrics."""
        acc = TurnAccumulator()
        acc.on_provider_response(
            {
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
            }
        )
        acc.on_prompt_submit({"prompt": "test"})
        turn = acc.flush()
        assert turn.metrics is not None
        assert turn.metrics["input_tokens"] == 100
        assert turn.metrics["output_tokens"] == 50
        assert turn.metrics["provider"] == "anthropic"

    def test_anthropic_token_math(self):
        """Anthropic total_input = input + cache_read + cache_creation."""
        acc = TurnAccumulator()
        acc.on_provider_response(
            {
                "usage": {
                    "input_tokens": 5,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 200,
                    "output_tokens": 500,
                },
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
            }
        )
        acc.on_prompt_submit({"prompt": "test"})
        turn = acc.flush()
        assert turn.metrics["total_input_tokens"] == 1205  # 5 + 1000 + 200
        assert turn.metrics["total_tokens"] == 1705  # 1205 + 500

    def test_reasoning_tokens_anthropic(self):
        """Reasoning tokens captured from usage.reasoning_tokens."""
        acc = TurnAccumulator()
        acc.on_provider_response(
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 50,
                    "reasoning_tokens": 30,
                },
            }
        )
        acc.on_prompt_submit({"prompt": "test"})
        turn = acc.flush()
        assert turn.metrics["reasoning_tokens"] == 30

    def test_reasoning_tokens_openai_nested(self):
        """Reasoning tokens from OpenAI nested completion_tokens_details."""
        acc = TurnAccumulator()
        acc.on_provider_response(
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 50,
                    "completion_tokens_details": {"reasoning_tokens": 25},
                },
            }
        )
        acc.on_prompt_submit({"prompt": "test"})
        turn = acc.flush()
        assert turn.metrics["reasoning_tokens"] == 25

    def test_no_usage_skips_metrics(self):
        """Missing usage dict means no metrics captured."""
        acc = TurnAccumulator()
        acc.on_provider_response({"provider": "test"})
        acc.on_prompt_submit({"prompt": "test"})
        turn = acc.flush()
        assert turn.metrics is None


class TestTurnAccumulatorStragglers:
    def test_straggler_after_execution_end(self):
        """llm:* events after execution:end are stragglers."""
        acc = TurnAccumulator()
        acc.on_execution_end()
        assert acc.is_straggler("llm:request:raw") is True
        assert acc.is_straggler("llm:response") is True
        assert acc.is_straggler("llm:response:debug") is True

    def test_non_llm_not_straggler(self):
        """Non-llm:* events are never stragglers."""
        acc = TurnAccumulator()
        acc.on_execution_end()
        assert acc.is_straggler("tool:post") is False
        assert acc.is_straggler("session:end") is False
        assert acc.is_straggler("provider:response") is False

    def test_not_straggler_before_execution_end(self):
        """Before execution:end, nothing is a straggler."""
        acc = TurnAccumulator()
        assert acc.is_straggler("llm:request:raw") is False

    def test_prompt_submit_resets_execution_ended(self):
        """New prompt:submit resets the execution_ended flag."""
        acc = TurnAccumulator()
        acc.on_execution_end()
        assert acc.is_straggler("llm:request") is True
        acc.on_prompt_submit({"prompt": "new cycle"})
        assert acc.is_straggler("llm:request") is False


class TestTurnAccumulatorFlush:
    def test_flush_returns_none_when_empty(self):
        """Empty accumulator flushes to None."""
        acc = TurnAccumulator()
        assert acc.flush() is None

    def test_flush_resets_state(self):
        """After flush, accumulator is empty."""
        acc = TurnAccumulator()
        acc.on_prompt_submit({"prompt": "Hello"})
        acc.flush()
        assert acc.flush() is None

    def test_flush_resets_execution_ended(self):
        """Flush resets the execution_ended flag."""
        acc = TurnAccumulator()
        acc.on_execution_end()
        acc.on_prompt_submit({"prompt": "test"})
        acc.flush()
        assert acc.is_straggler("llm:request") is False


class TestToConversationItems:
    def test_user_and_assistant(self):
        """Full turn produces user_input + assistant_turn items."""
        acc = TurnAccumulator()
        turn = AccumulatedTurn(
            user_prompt="Hello",
            assistant_text_blocks=["Hi there!"],
        )
        items = acc.to_conversation_items(turn)
        assert len(items) == 2
        # User input: tag 1=item_type, tag 2=status, tag 10=UserInput{1:text}
        assert items[0][1] == "user_input"
        assert items[0][2] == "complete"
        assert items[0][10] == {1: "Hello"}
        # Assistant turn: tag 1=item_type, tag 11=AssistantTurn{1:text}
        assert items[1][1] == "assistant_turn"
        assert items[1][2] == "complete"
        assert items[1][11][1] == "Hi there!"

    def test_user_only(self):
        """Turn with only user prompt produces 1 item."""
        acc = TurnAccumulator()
        turn = AccumulatedTurn(user_prompt="Hello")
        items = acc.to_conversation_items(turn)
        assert len(items) == 1
        assert items[0][1] == "user_input"

    def test_assistant_with_tool_calls(self):
        """Assistant turn includes tool_calls in AssistantTurn subtree (tag 11.2)."""
        acc = TurnAccumulator()
        turn = AccumulatedTurn(
            user_prompt="Search",
            assistant_text_blocks=["Found results."],
            tool_calls=[
                ToolCallRecord(
                    tool_name="grep",
                    input_summary="pattern=test",
                    has_result=True,
                    result="5 matches",
                ),
            ],
        )
        items = acc.to_conversation_items(turn)
        assert len(items) == 2
        assistant_turn = items[1][11]  # AssistantTurn subtree
        assert 2 in assistant_turn  # tool_calls at AssistantTurn.tag 2
        assert len(assistant_turn[2]) == 1
        assert assistant_turn[2][0][2] == "grep"  # ToolCallItem.name
        assert assistant_turn[2][0][4] == "complete"  # ToolCallItem.status

    def test_assistant_with_metrics(self):
        """Assistant turn includes metrics in AssistantTurn subtree (tag 11.4)."""
        acc = TurnAccumulator()
        turn = AccumulatedTurn(
            user_prompt="Test",
            assistant_text_blocks=["Response."],
            metrics={"total_tokens": 100, "model": "test-model"},
        )
        items = acc.to_conversation_items(turn)
        assistant_turn = items[1][11]  # AssistantTurn subtree
        assert 4 in assistant_turn  # TurnMetrics at AssistantTurn.tag 4
        assert assistant_turn[4][3] == 100  # TurnMetrics.total_tokens at tag 3
        assert assistant_turn[4][7] == "test-model"  # TurnMetrics.model at tag 7

    def test_multiple_text_blocks_concatenated(self):
        """Multiple text blocks are concatenated in AssistantTurn.text."""
        acc = TurnAccumulator()
        turn = AccumulatedTurn(
            assistant_text_blocks=["Part 1. ", "Part 2."],
        )
        items = acc.to_conversation_items(turn)
        assert len(items) == 1  # no user, just assistant
        assert items[0][11][1] == "Part 1. Part 2."  # AssistantTurn.text

    def test_empty_turn_produces_no_items(self):
        """Turn with no data produces empty list."""
        acc = TurnAccumulator()
        turn = AccumulatedTurn()
        items = acc.to_conversation_items(turn)
        assert items == []

    def test_items_have_envelope_fields(self):
        """All items have status(tag 2), timestamp(tag 3), and id(tag 4)."""
        acc = TurnAccumulator()
        turn = AccumulatedTurn(
            user_prompt="Hello",
            assistant_text_blocks=["Hi"],
        )
        items = acc.to_conversation_items(turn)
        for item in items:
            assert item[2] == "complete"  # status
            assert isinstance(item[3], int) and item[3] > 0  # timestamp
            assert isinstance(item[4], str) and len(item[4]) > 0  # id


class TestToolCallIdMatching:
    def test_tool_post_matches_by_call_id(self):
        """Parallel same-tool calls matched by call_id, not tool_name."""
        acc = TurnAccumulator()
        acc.on_tool_pre(
            {
                "tool_name": "read_file",
                "tool_call_id": "call_A",
                "tool_input": {"file_path": "a.py"},
            }
        )
        acc.on_tool_pre(
            {
                "tool_name": "read_file",
                "tool_call_id": "call_B",
                "tool_input": {"file_path": "b.py"},
            }
        )
        # Results arrive in reverse order
        acc.on_tool_post(
            {
                "tool_name": "read_file",
                "tool_call_id": "call_B",
                "result": "content of b",
            }
        )
        acc.on_tool_post(
            {
                "tool_name": "read_file",
                "tool_call_id": "call_A",
                "result": "content of a",
            }
        )
        turn = acc.flush()
        assert len(turn.tool_calls) == 2
        assert turn.tool_calls[0].call_id == "call_A"
        assert "content of a" in turn.tool_calls[0].result
        assert turn.tool_calls[1].call_id == "call_B"
        assert "content of b" in turn.tool_calls[1].result

    def test_tool_post_falls_back_to_name_without_call_id(self):
        """Without call_id, falls back to tool_name matching."""
        acc = TurnAccumulator()
        acc.on_tool_pre({"tool_name": "grep", "tool_input": {"pattern": "test"}})
        acc.on_tool_post({"tool_name": "grep", "result": "found"})
        turn = acc.flush()
        assert turn.tool_calls[0].has_result


class TestDurationMs:
    def test_duration_ms_captured(self):
        """Duration computed from provider:request to provider:response."""
        import time

        acc = TurnAccumulator()
        acc.on_provider_request({"provider": "anthropic", "iteration": 1})
        time.sleep(0.01)  # small delay
        acc.on_provider_response(
            {
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "provider": "anthropic",
            }
        )
        acc.on_prompt_submit({"prompt": "test"})
        turn = acc.flush()
        assert turn.metrics is not None
        assert "duration_ms" in turn.metrics
        assert turn.metrics["duration_ms"] >= 10  # at least 10ms

    def test_duration_ms_absent_without_request(self):
        """No duration if provider:request wasn't seen."""
        acc = TurnAccumulator()
        acc.on_provider_response(
            {
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "provider": "anthropic",
            }
        )
        acc.on_prompt_submit({"prompt": "test"})
        turn = acc.flush()
        assert "duration_ms" not in turn.metrics
