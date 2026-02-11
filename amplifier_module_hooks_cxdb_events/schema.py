"""Event envelope construction, payload serialization, and registry bundle management."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx
import msgpack

logger = logging.getLogger(__name__)

_BUNDLE_PATH = Path(__file__).parent / "conversation_bundle.json"
_BUNDLE_ID = "amplifier.events-v1"


def load_bundle_json() -> dict:
    """Load the CXDB registry bundle from package data.

    Returns:
        Parsed JSON dict containing types array.

    Raises:
        FileNotFoundError: If conversation_bundle.json is missing.
    """
    return json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))


async def publish_registry_bundle(http_host: str, http_port: int = 80) -> bool:
    """Publish the registry bundle to CXDB via HTTP PUT.

    Args:
        http_host: CXDB HTTP gateway hostname.
        http_port: CXDB HTTP gateway port (default 9010).

    Returns:
        True if published successfully (201 or 204), False otherwise.
    """
    bundle = load_bundle_json()
    url = f"http://{http_host}:{http_port}/v1/registry/bundles/{_BUNDLE_ID}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.put(
                url,
                json=bundle,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code in (201, 204):
                logger.info(f"Registry bundle published: {_BUNDLE_ID}")
                return True
            else:
                logger.warning(
                    f"Registry publish unexpected status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return False
    except httpx.HTTPError as e:
        logger.warning(f"Registry publish failed: {e}")
        return False


# UUID v4 pattern for detecting root session IDs
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def extract_agent_name(session_id: str) -> str | None:
    """Parse agent name from a child session ID.

    Child session IDs encode the agent name after the last underscore:
        {parent_span}-{child_span}_{agent_name}

    Root session IDs are plain UUIDs with no underscore-delimited agent name.
    Uses rsplit("_", 1) to extract the agent name.

    Args:
        session_id: Amplifier session ID.

    Returns:
        Agent name string, or None for root sessions.
    """
    if not session_id or "_" not in session_id:
        return None

    # Root UUIDs don't have underscores
    if _UUID_RE.match(session_id):
        return None

    # Extract agent name from after the last underscore
    parts = session_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1]

    return None


def build_event_envelope(
    event_name: str,
    data: dict,
    session_id: str,
    parent_session_id: str | None,
    agent_name: str | None,
    root_session_id: str | None = None,
) -> dict[int, object]:
    """Build a msgpack-ready event envelope with integer tag keys.

    The envelope follows the common schema across all amplifier.* types:
      tag 1: event_name
      tag 2: session_id
      tag 3: parent_session_id (if set)
      tag 4: timestamp_ms
      tag 5: agent_name (if set)
      tag 6: payload_bytes (set after serialization)
      tag 7: raw event data
      tag 8: root_session_id (if set)

    Args:
        event_name: Amplifier event name (e.g., "tool:post").
        data: Raw event data dict from the hook handler.
        session_id: Session that emitted this event.
        parent_session_id: Parent session ID, or None for root.
        agent_name: Agent name, or None.
        root_session_id: Root session ID for cross-context lineage, or None.

    Returns:
        Dict with integer keys ready for msgpack serialization.
        The payload_bytes field (tag 6) reflects the serialized size.
    """
    envelope: dict[int, object] = {
        1: event_name,
        2: session_id,
        4: int(time.time() * 1000),
    }

    if parent_session_id is not None:
        envelope[3] = parent_session_id

    if agent_name is not None:
        envelope[5] = agent_name

    if root_session_id is not None:
        envelope[8] = root_session_id

    # Include raw event data at tag 7
    if data:
        # Filter out default fields already in envelope to avoid duplication
        filtered_data = {
            k: v for k, v in data.items() if k not in ("session_id", "parent_id", "ts")
        }
        if filtered_data:
            envelope[7] = filtered_data

    # Calculate payload_bytes: serialize without the size field, measure, then set it
    envelope[6] = 0  # placeholder
    serialized = serialize_envelope(envelope)
    envelope[6] = len(serialized)

    return envelope


def serialize_envelope(envelope: dict[int, object]) -> bytes:
    """Serialize an event envelope to msgpack bytes.

    Keys are sorted ascending for deterministic encoding per CXDB spec.

    Args:
        envelope: Dict with integer tag keys.

    Returns:
        Msgpack-encoded bytes.
    """
    sorted_envelope = dict(sorted(envelope.items()))
    return msgpack.packb(sorted_envelope, use_bin_type=True)


def calculate_payload_bytes(envelope: dict[int, object]) -> int:
    """Calculate the serialized size of an envelope in bytes.

    Args:
        envelope: Dict with integer tag keys.

    Returns:
        Size in bytes when serialized to msgpack.
    """
    return len(serialize_envelope(envelope))
