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
    """Tests for pre-computed variant map deduplication.

    The deduplicator scans known_events at construction to determine the
    richest variant per family, then only that variant passes should_process().
    """

    # Standard set: all three variants registered for session:start and llm:request
    ALL_VARIANTS = [
        "session:start", "session:start:debug", "session:start:raw",
        "llm:request", "llm:request:debug", "llm:request:raw",
        "llm:response", "llm:response:debug", "llm:response:raw",
        "tool:post", "session:end", "prompt:submit",
    ]

    def test_only_raw_passes_when_all_registered(self):
        """When all three variants are registered, only :raw passes."""
        dedup = VariantDeduplicator(known_events=self.ALL_VARIANTS)
        # :raw passes
        assert dedup.should_process("session:start:raw") is True
        # :debug and base are suppressed
        assert dedup.should_process("session:start:debug") is False
        assert dedup.should_process("session:start") is False

    def test_arrival_order_irrelevant(self):
        """All three variants suppressed except :raw, regardless of arrival order."""
        dedup = VariantDeduplicator(known_events=self.ALL_VARIANTS)
        # Amplifier emits base first, then debug, then raw
        assert dedup.should_process("session:start") is False
        assert dedup.should_process("session:start:debug") is False
        assert dedup.should_process("session:start:raw") is True

    def test_debug_accepted_if_no_raw_registered(self):
        """When only base and :debug are registered, :debug wins."""
        events = ["session:start", "session:start:debug", "tool:post"]
        dedup = VariantDeduplicator(known_events=events)
        assert dedup.should_process("session:start:debug") is True
        assert dedup.should_process("session:start") is False

    def test_base_accepted_if_only_variant(self):
        """When only the base is registered, it passes."""
        events = ["session:start", "tool:post"]
        dedup = VariantDeduplicator(known_events=events)
        assert dedup.should_process("session:start") is True

    def test_independent_base_events(self):
        """Different base event families are tracked independently."""
        dedup = VariantDeduplicator(known_events=self.ALL_VARIANTS)
        assert dedup.should_process("session:start:raw") is True
        assert dedup.should_process("llm:request:raw") is True
        assert dedup.should_process("session:start:debug") is False

    def test_non_variant_events_always_processed(self):
        """Events without variants always pass through."""
        dedup = VariantDeduplicator(known_events=self.ALL_VARIANTS)
        assert dedup.should_process("tool:post") is True
        assert dedup.should_process("tool:post") is True  # repeatable
        assert dedup.should_process("session:end") is True

    def test_unknown_variant_family_passes(self):
        """Events from unknown families (not in known_events) pass through."""
        dedup = VariantDeduplicator(known_events=["tool:post"])
        # session:start has variants but wasn't in known_events
        assert dedup.should_process("session:start:raw") is True

    def test_empty_known_events(self):
        """With no known events, everything passes (backward compat)."""
        dedup = VariantDeduplicator()
        assert dedup.should_process("session:start:raw") is True
        assert dedup.should_process("session:start:debug") is True
        assert dedup.should_process("session:start") is True

    def test_deterministic_across_calls(self):
        """Same event always gets the same answer (no state mutation)."""
        dedup = VariantDeduplicator(known_events=self.ALL_VARIANTS)
        for _ in range(3):
            assert dedup.should_process("session:start:raw") is True
            assert dedup.should_process("session:start:debug") is False
            assert dedup.should_process("session:start") is False


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
