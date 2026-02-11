"""Event-to-CXDB-type mapping and variant deduplication."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CXDBType:
    """Represents a CXDB type for event storage."""

    type_id: str
    type_version: int = 1


# --- The 15 CXDB Types ---

SESSION_EVENT = CXDBType("amplifier.SessionEvent")
PROMPT_EVENT = CXDBType("amplifier.PromptEvent")
PLAN_EVENT = CXDBType("amplifier.PlanEvent")
PROVIDER_EVENT = CXDBType("amplifier.ProviderEvent")
CONTENT_BLOCK_EVENT = CXDBType("amplifier.ContentBlockEvent")
THINKING_EVENT = CXDBType("amplifier.ThinkingEvent")
TOOL_EVENT = CXDBType("amplifier.ToolEvent")
CONTEXT_EVENT = CXDBType("amplifier.ContextEvent")
ORCHESTRATOR_EVENT = CXDBType("amplifier.OrchestratorEvent")
DELEGATION_EVENT = CXDBType("amplifier.DelegationEvent")
APPROVAL_EVENT = CXDBType("amplifier.ApprovalEvent")
CANCEL_EVENT = CXDBType("amplifier.CancelEvent")
NOTIFICATION_EVENT = CXDBType("amplifier.NotificationEvent")
ARTIFACT_EVENT = CXDBType("amplifier.ArtifactEvent")
GENERIC_EVENT = CXDBType("amplifier.GenericEvent")

# --- Complete Event-to-Type Mapping (47 events -> 15 types) ---

EVENT_TYPE_MAP: dict[str, CXDBType] = {
    # Session lifecycle (10 events)
    "session:start": SESSION_EVENT,
    "session:start:debug": SESSION_EVENT,
    "session:start:raw": SESSION_EVENT,
    "session:end": SESSION_EVENT,
    "session:fork": SESSION_EVENT,
    "session:fork:debug": SESSION_EVENT,
    "session:fork:raw": SESSION_EVENT,
    "session:resume": SESSION_EVENT,
    "session:resume:debug": SESSION_EVENT,
    "session:resume:raw": SESSION_EVENT,
    # Prompt lifecycle (2 events)
    "prompt:submit": PROMPT_EVENT,
    "prompt:complete": PROMPT_EVENT,
    # Planning (2 events)
    "plan:start": PLAN_EVENT,
    "plan:end": PLAN_EVENT,
    # Provider / LLM (9 events)
    "provider:request": PROVIDER_EVENT,
    "provider:response": PROVIDER_EVENT,
    "provider:error": PROVIDER_EVENT,
    "llm:request": PROVIDER_EVENT,
    "llm:request:debug": PROVIDER_EVENT,
    "llm:request:raw": PROVIDER_EVENT,
    "llm:response": PROVIDER_EVENT,
    "llm:response:debug": PROVIDER_EVENT,
    "llm:response:raw": PROVIDER_EVENT,
    # Content blocks (3 events)
    "content_block:start": CONTENT_BLOCK_EVENT,
    "content_block:delta": CONTENT_BLOCK_EVENT,
    "content_block:end": CONTENT_BLOCK_EVENT,
    # Thinking (2 events)
    "thinking:delta": THINKING_EVENT,
    "thinking:final": THINKING_EVENT,
    # Tool (3 events)
    "tool:pre": TOOL_EVENT,
    "tool:post": TOOL_EVENT,
    "tool:error": TOOL_EVENT,
    # Context management (4 events)
    "context:pre_compact": CONTEXT_EVENT,
    "context:post_compact": CONTEXT_EVENT,
    "context:compaction": CONTEXT_EVENT,
    "context:include": CONTEXT_EVENT,
    # Orchestrator (3 events)
    "orchestrator:complete": ORCHESTRATOR_EVENT,
    "execution:start": ORCHESTRATOR_EVENT,
    "execution:end": ORCHESTRATOR_EVENT,
    # Delegation (3 module-specific events from tool-task)
    "task:agent_spawned": DELEGATION_EVENT,
    "task:agent_completed": DELEGATION_EVENT,
    "task:agent_resumed": DELEGATION_EVENT,
    # Approvals (4 events)
    "approval:required": APPROVAL_EVENT,
    "approval:granted": APPROVAL_EVENT,
    "approval:denied": APPROVAL_EVENT,
    "policy:violation": APPROVAL_EVENT,
    # Cancellation (2 events)
    "cancel:requested": CANCEL_EVENT,
    "cancel:completed": CANCEL_EVENT,
    # Notifications (1 event)
    "user:notification": NOTIFICATION_EVENT,
    # Artifacts (2 events)
    "artifact:write": ARTIFACT_EVENT,
    "artifact:read": ARTIFACT_EVENT,
}

# --- Variant Deduplication ---

# Events that have :debug and :raw variants.
# Preference order: :raw > :debug > base (higher index = richer)
_VARIANT_SUFFIXES = [":raw", ":debug"]

# Base events that have variants
_EVENTS_WITH_VARIANTS = {
    "session:start",
    "session:fork",
    "session:resume",
    "llm:request",
    "llm:response",
}


def get_cxdb_type(event_name: str) -> CXDBType:
    """Return the CXDB type for an Amplifier event.

    Falls back to GenericEvent for unknown events.

    Args:
        event_name: Amplifier event name (e.g., "tool:post").

    Returns:
        CXDBType instance.
    """
    return EVENT_TYPE_MAP.get(event_name, GENERIC_EVENT)


def get_base_event(event_name: str) -> str:
    """Strip :debug or :raw suffix to get the base event name.

    Args:
        event_name: Amplifier event name, possibly with variant suffix.

    Returns:
        Base event name without variant suffix.

    Examples:
        >>> get_base_event("session:start:raw")
        'session:start'
        >>> get_base_event("tool:post")
        'tool:post'
    """
    for suffix in _VARIANT_SUFFIXES:
        if event_name.endswith(suffix):
            return event_name[: -len(suffix)]
    return event_name


def has_variants(event_name: str) -> bool:
    """Check if an event name (base or variant) belongs to a variant group.

    Args:
        event_name: Any event name.

    Returns:
        True if this event has :debug/:raw variants.
    """
    base = get_base_event(event_name)
    return base in _EVENTS_WITH_VARIANTS


def _variant_rank(event_name: str) -> int:
    """Return the richness rank of a variant (higher = richer).

    :raw = 2, :debug = 1, base = 0
    """
    if event_name.endswith(":raw"):
        return 2
    if event_name.endswith(":debug"):
        return 1
    return 0


class VariantDeduplicator:
    """Tracks seen event variants per orchestrator cycle.

    Prefers :raw > :debug > base. Once a richer variant is seen,
    less-rich variants of the same base event are suppressed.

    Call reset() on orchestrator:complete to start a new cycle.
    """

    def __init__(self) -> None:
        # Maps base_event -> highest rank seen
        self._seen: dict[str, int] = {}

    def should_process(self, event_name: str) -> bool:
        """Check if this event should be processed.

        Returns True if:
        - Event has no variants (always process), OR
        - No richer variant of the same base event was already seen

        Args:
            event_name: Amplifier event name.

        Returns:
            True if the event should be processed.
        """
        if not has_variants(event_name):
            return True

        base = get_base_event(event_name)
        rank = _variant_rank(event_name)
        prev_rank = self._seen.get(base, -1)

        if rank > prev_rank:
            # This is richer than anything we've seen -- process it
            self._seen[base] = rank
            return True

        # We already have an equal or richer variant
        return False

    def reset(self) -> None:
        """Clear variant tracking. Call on orchestrator:complete."""
        self._seen.clear()
