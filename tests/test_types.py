"""Tests for event-to-type mapping and variant deduplication."""

import pytest

from amplifier_module_hooks_cxdb_events.types import (
    APPROVAL_EVENT,
    ARTIFACT_EVENT,
    CANCEL_EVENT,
    CONTENT_BLOCK_EVENT,
    CONTEXT_EVENT,
    DELEGATION_EVENT,
    EVENT_TYPE_MAP,
    GENERIC_EVENT,
    NOTIFICATION_EVENT,
    ORCHESTRATOR_EVENT,
    PLAN_EVENT,
    PROMPT_EVENT,
    PROVIDER_EVENT,
    SESSION_EVENT,
    THINKING_EVENT,
    TOOL_EVENT,
    VariantDeduplicator,
    get_base_event,
    get_cxdb_type,
    has_variants,
)


class TestGetCxdbType:
    def test_all_canonical_events_mapped(self):
        """Every event in ALL_EVENTS should map to a non-generic type."""
        from amplifier_core.events import ALL_EVENTS

        for event in ALL_EVENTS:
            cxdb_type = get_cxdb_type(event)
            assert cxdb_type is not None, f"Event {event} not mapped"
            # All canonical events should have specific types, not generic
            assert cxdb_type != GENERIC_EVENT, f"Event {event} mapped to GenericEvent"

    def test_tool_task_events_mapped(self):
        """Module-specific delegation events map to DelegationEvent."""
        assert get_cxdb_type("task:agent_spawned") == DELEGATION_EVENT
        assert get_cxdb_type("task:agent_completed") == DELEGATION_EVENT
        assert get_cxdb_type("task:agent_resumed") == DELEGATION_EVENT

    def test_unknown_event_falls_back_to_generic(self):
        """Unknown events fall back to GenericEvent."""
        assert get_cxdb_type("custom:my_event") == GENERIC_EVENT
        assert get_cxdb_type("unknown:anything") == GENERIC_EVENT
        assert get_cxdb_type("") == GENERIC_EVENT

    def test_session_events(self):
        """Session lifecycle events map to SessionEvent."""
        for event in ["session:start", "session:start:debug", "session:start:raw",
                       "session:end", "session:fork", "session:fork:raw",
                       "session:resume", "session:resume:debug"]:
            assert get_cxdb_type(event) == SESSION_EVENT, f"{event} should be SessionEvent"

    def test_prompt_events(self):
        assert get_cxdb_type("prompt:submit") == PROMPT_EVENT
        assert get_cxdb_type("prompt:complete") == PROMPT_EVENT

    def test_plan_events(self):
        assert get_cxdb_type("plan:start") == PLAN_EVENT
        assert get_cxdb_type("plan:end") == PLAN_EVENT

    def test_provider_events(self):
        for event in ["provider:request", "provider:response", "provider:error",
                       "llm:request", "llm:request:raw", "llm:response:debug"]:
            assert get_cxdb_type(event) == PROVIDER_EVENT, f"{event} should be ProviderEvent"

    def test_content_block_events(self):
        for event in ["content_block:start", "content_block:delta", "content_block:end"]:
            assert get_cxdb_type(event) == CONTENT_BLOCK_EVENT

    def test_thinking_events(self):
        assert get_cxdb_type("thinking:delta") == THINKING_EVENT
        assert get_cxdb_type("thinking:final") == THINKING_EVENT

    def test_tool_events(self):
        for event in ["tool:pre", "tool:post", "tool:error"]:
            assert get_cxdb_type(event) == TOOL_EVENT

    def test_context_events(self):
        for event in ["context:pre_compact", "context:post_compact",
                       "context:compaction", "context:include"]:
            assert get_cxdb_type(event) == CONTEXT_EVENT

    def test_orchestrator_events(self):
        for event in ["orchestrator:complete", "execution:start", "execution:end"]:
            assert get_cxdb_type(event) == ORCHESTRATOR_EVENT

    def test_approval_events(self):
        for event in ["approval:required", "approval:granted",
                       "approval:denied", "policy:violation"]:
            assert get_cxdb_type(event) == APPROVAL_EVENT

    def test_cancel_events(self):
        assert get_cxdb_type("cancel:requested") == CANCEL_EVENT
        assert get_cxdb_type("cancel:completed") == CANCEL_EVENT

    def test_notification_event(self):
        assert get_cxdb_type("user:notification") == NOTIFICATION_EVENT

    def test_artifact_events(self):
        assert get_cxdb_type("artifact:write") == ARTIFACT_EVENT
        assert get_cxdb_type("artifact:read") == ARTIFACT_EVENT


class TestGetBaseEvent:
    def test_strip_raw_suffix(self):
        assert get_base_event("session:start:raw") == "session:start"
        assert get_base_event("llm:request:raw") == "llm:request"

    def test_strip_debug_suffix(self):
        assert get_base_event("session:start:debug") == "session:start"
        assert get_base_event("llm:response:debug") == "llm:response"

    def test_no_suffix(self):
        assert get_base_event("tool:post") == "tool:post"
        assert get_base_event("session:end") == "session:end"
        assert get_base_event("prompt:submit") == "prompt:submit"

    def test_double_colon_events(self):
        """Events with colons in base name work correctly."""
        assert get_base_event("content_block:start") == "content_block:start"
        assert get_base_event("context:pre_compact") == "context:pre_compact"


class TestHasVariants:
    def test_events_with_variants(self):
        assert has_variants("session:start") is True
        assert has_variants("session:start:raw") is True
        assert has_variants("session:start:debug") is True
        assert has_variants("llm:request") is True
        assert has_variants("llm:response:raw") is True

    def test_events_without_variants(self):
        assert has_variants("tool:post") is False
        assert has_variants("session:end") is False
        assert has_variants("prompt:submit") is False


class TestVariantDeduplicator:
    def test_raw_preferred_over_debug(self):
        """When :raw seen first, :debug is suppressed."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start:raw") is True
        assert dedup.should_process("session:start:debug") is False
        assert dedup.should_process("session:start") is False

    def test_debug_accepted_if_no_raw(self):
        """When :debug seen first (no :raw), base is suppressed."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start:debug") is True
        assert dedup.should_process("session:start") is False

    def test_base_accepted_if_first(self):
        """Base event accepted if seen first."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start") is True

    def test_raw_overrides_previously_seen_debug(self):
        """:raw overrides previously seen :debug."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start:debug") is True
        assert dedup.should_process("session:start:raw") is True  # richer, accepted

    def test_raw_overrides_previously_seen_base(self):
        """:raw overrides previously seen base."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start") is True
        assert dedup.should_process("session:start:raw") is True  # richer

    def test_independent_base_events(self):
        """Different base events are tracked independently."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start:raw") is True
        assert dedup.should_process("llm:request:raw") is True  # different base
        assert dedup.should_process("session:start:debug") is False  # same base as first

    def test_non_variant_events_always_processed(self):
        """Events without variants are always processed."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("tool:post") is True
        assert dedup.should_process("tool:post") is True  # not deduplicated
        assert dedup.should_process("session:end") is True

    def test_reset_clears_tracking(self):
        """Reset clears all state."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start:raw") is True
        dedup.reset()
        assert dedup.should_process("session:start:raw") is True

    def test_reset_between_cycles(self):
        """Simulates orchestrator cycles with reset between them."""
        dedup = VariantDeduplicator()
        # Cycle 1
        assert dedup.should_process("session:start:raw") is True
        assert dedup.should_process("session:start:debug") is False
        dedup.reset()
        # Cycle 2 - fresh
        assert dedup.should_process("session:start:debug") is True  # accepted in new cycle


class TestEventTypeMapCompleteness:
    def test_all_15_types_represented(self):
        """All 15 CXDB types appear in the mapping."""
        all_types = {t for t in EVENT_TYPE_MAP.values()}
        expected = {
            SESSION_EVENT, PROMPT_EVENT, PLAN_EVENT, PROVIDER_EVENT,
            CONTENT_BLOCK_EVENT, THINKING_EVENT, TOOL_EVENT, CONTEXT_EVENT,
            ORCHESTRATOR_EVENT, DELEGATION_EVENT, APPROVAL_EVENT, CANCEL_EVENT,
            NOTIFICATION_EVENT, ARTIFACT_EVENT,
        }
        # GenericEvent is not in the map (it's the fallback)
        assert expected.issubset(all_types)
        assert GENERIC_EVENT not in all_types  # only used as fallback

    def test_map_has_50_entries(self):
        """Mapping covers exactly 50 events (47 canonical + 3 module-specific)."""
        assert len(EVENT_TYPE_MAP) == 50

    def test_cxdb_type_is_frozen(self):
        """CXDBType instances are immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            SESSION_EVENT.type_id = "changed"
