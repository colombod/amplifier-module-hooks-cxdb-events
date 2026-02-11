"""Amplifier hook module for CXDB event capture.

Captures all meaningful lifecycle events from Amplifier sessions and stores
them in CXDB using the binary TCP protocol. Requires zero CXDB server changes.

Usage in bundle config:
    hooks:
      - module: hooks-cxdb-events
        config:
          cxdb_host: "localhost"
          cxdb_port: 9009
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from typing import Any

from amplifier_core.events import ALL_EVENTS

from amplifier_module_hooks_cxdb_events.hook import CXDBEventHook
from amplifier_module_hooks_cxdb_events.protocol import CXDBTcpClient

logger = logging.getLogger(__name__)

# Default events to exclude (high-frequency streaming noise)
DEFAULT_EXCLUDES = frozenset({"content_block:delta", "thinking:delta"})

# Module-specific events not in ALL_EVENTS but discoverable at runtime
_KNOWN_MODULE_EVENTS = [
    "task:agent_spawned",
    "task:agent_completed",
    "task:agent_resumed",
]


def _should_exclude(event: str, exclude_patterns: set[str]) -> bool:
    """Check if an event should be excluded by exact match or glob pattern.

    Args:
        event: Event name to check.
        exclude_patterns: Set of exact names or glob patterns.

    Returns:
        True if the event should be excluded.
    """
    if event in exclude_patterns:
        return True
    return any(fnmatch(event, pattern) for pattern in exclude_patterns)


async def mount(coordinator: Any, config: dict[str, Any]) -> Any:
    """Module entry point. Registers hook handlers for CXDB event capture.

    No-op if cxdb_host is not configured (absent or empty).

    Args:
        coordinator: Amplifier ModuleCoordinator instance.
        config: Module configuration dict.

    Returns:
        Async cleanup function, or None if no-op.
    """
    # No-op check: if cxdb_host is not configured, do nothing
    cxdb_host = config.get("cxdb_host", "")
    if not cxdb_host:
        logger.debug("hooks-cxdb-events: no cxdb_host configured, skipping")
        return None

    cxdb_port = config.get("cxdb_port", 9009)
    timeout = config.get("flush_timeout_seconds", 5)
    priority = config.get("priority", 100)

    # Build the event list
    # Layer 1: All canonical kernel events
    events: list[str] = list(ALL_EVENTS)

    # Layer 2: Known module-specific events
    events.extend(_KNOWN_MODULE_EVENTS)

    # Layer 3: Discovered module events via contribution channel
    try:
        contributions = await coordinator.collect_contributions("observability.events")
        for contribution in contributions:
            if isinstance(contribution, list):
                events.extend(contribution)
    except Exception as e:
        logger.debug(f"Failed to collect observability.events contributions: {e}")

    # Layer 4: Additional events from config
    additional = config.get("additional_events", [])
    if additional:
        events.extend(additional)

    # Deduplicate
    events = list(dict.fromkeys(events))

    # Apply exclusion filter
    user_excludes = set(config.get("exclude_events", []))
    exclude_patterns: set[str] = set(DEFAULT_EXCLUDES) | user_excludes
    events = [e for e in events if not _should_exclude(e, exclude_patterns)]

    # Get session info from coordinator
    session_id = coordinator.session_id
    parent_id = coordinator.parent_id
    root_session_id = coordinator.config.get("root_session_id", session_id)

    # Build client_tag: "amplifier - <project> - <root_session_id_short>"
    # The client_tag identifies this connection in CXDB's UI and CQL queries.
    #
    # Note: the CLI does NOT put project_name or bundle_name into
    # coordinator.config (only root_session_id is set there). We derive
    # project name from cwd, matching how the CLI scopes sessions.
    import os

    project_name = os.path.basename(os.getcwd())
    tag_parts = ["amplifier", project_name, root_session_id[:12]]
    client_tag = " - ".join(tag_parts)

    # Create the TCP client and hook
    client = CXDBTcpClient(
        host=cxdb_host, port=cxdb_port, timeout=timeout, client_tag=client_tag
    )

    # Stash project info into config for context_metadata in hook.initialize()
    config["_project_name"] = project_name

    hook = CXDBEventHook(
        client=client,
        config=config,
        session_id=session_id,
        parent_id=parent_id,
        root_session_id=root_session_id,
        known_events=events,
    )

    # Register handler for each event
    registrations = []
    for event in events:
        reg = coordinator.hooks.register(
            event,
            hook.handle_event,
            priority=priority,
            name="cxdb-events",
        )
        registrations.append(reg)

    logger.info(
        f"hooks-cxdb-events: registered {len(registrations)} events "
        f"(priority={priority}, host={cxdb_host}:{cxdb_port})"
    )

    # Return async cleanup function
    async def cleanup() -> None:
        """Cleanup: flush buffer, close connection, unregister hooks."""
        await hook.cleanup()
        for unreg in registrations:
            if callable(unreg):
                try:
                    unreg()
                except Exception:
                    pass

    return cleanup
