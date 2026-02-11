"""Event envelope construction, payload serialization, and registry bundle management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import msgpack

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ConversationItem tag numbers (cxdb.ConversationItem v3)
# ---------------------------------------------------------------------------
TAG_ITEM_TYPE = 1
TAG_STATUS = 2
TAG_TIMESTAMP = 3
TAG_ID = 4
TAG_SYSTEM = 12  # SystemMessage subtree
TAG_CTX_META = 30  # ContextMetadata subtree

# SystemMessage subtags (inside tag 12)
SM_TAG_KIND = 1
SM_TAG_TITLE = 2
SM_TAG_CONTENT = 3

# ContextMetadata subtags (inside tag 30)
CM_TAG_CLIENT = 1  # client_tag string
CM_TAG_TITLE = 2  # display title
CM_TAG_LABELS = 3  # list of string labels
CM_TAG_PROVENANCE = 10  # provenance subtree

# Provenance subtags (inside CM_TAG_PROVENANCE)
PROV_TAG_PARENT_CTX = 1
PROV_TAG_SPAWN_REASON = 2
PROV_TAG_ROOT_CTX = 3
PROV_TAG_TRACE_ID = 10
PROV_TAG_SPAN_ID = 11
PROV_TAG_ON_BEHALF_OF = 20
PROV_TAG_ON_BEHALF_OF_SOURCE = 21
PROV_TAG_SERVICE_NAME = 40
PROV_TAG_SERVICE_VERSION = 41
PROV_TAG_SERVICE_INSTANCE_ID = 42
PROV_TAG_PROCESS_PID = 43
PROV_TAG_PROCESS_OWNER = 44
PROV_TAG_HOST_NAME = 45
PROV_TAG_HOST_ARCH = 46
PROV_TAG_ENV = 60
PROV_TAG_SDK_NAME = 70
PROV_TAG_CAPTURED_AT = 80

# Item type constants
ITEM_SYSTEM = "system"

# Environment variables safe to capture in provenance
_ENV_ALLOWLIST = [
    "USER", "HOSTNAME", "HOME", "ENVIRONMENT", "ENV", "STAGE",
    "AWS_REGION", "AWS_DEFAULT_REGION", "GOOGLE_CLOUD_PROJECT",
    "K8S_NAMESPACE", "K8S_POD_NAME",
]


# ---------------------------------------------------------------------------
# Process-level provenance (captured once at import, reused for all contexts)
# ---------------------------------------------------------------------------

def _get_amplifier_version() -> str:
    """Get the amplifier-core version string, or empty on failure."""
    try:
        import importlib.metadata
        return importlib.metadata.version("amplifier-core")
    except Exception:
        return ""


def _get_current_user() -> str:
    """Get the OS username, or empty on failure."""
    try:
        return os.getlogin()
    except Exception:
        try:
            import pwd
            return pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            return os.environ.get("USER", "")


def _capture_env_vars(allowlist: list[str]) -> dict[str, str]:
    """Capture environment variables from allowlist (non-empty values only)."""
    return {k: v for k in allowlist if (v := os.environ.get(k, ""))}


def _capture_process_provenance() -> dict[int, Any]:
    """Capture process-level provenance once at startup.

    Returns a dict with integer keys matching CXDB Provenance tag numbers.
    Stable fields that don't change across contexts within the same process.
    """
    import uuid

    prov: dict[int, Any] = {
        PROV_TAG_SERVICE_NAME: "amplifier",
        PROV_TAG_SERVICE_INSTANCE_ID: str(uuid.uuid4()),
        PROV_TAG_PROCESS_PID: os.getpid(),
        PROV_TAG_HOST_ARCH: platform.machine(),
        PROV_TAG_SDK_NAME: "amplifier-hooks-cxdb-events",
    }
    version = _get_amplifier_version()
    if version:
        prov[PROV_TAG_SERVICE_VERSION] = version
    user = _get_current_user()
    if user:
        prov[PROV_TAG_PROCESS_OWNER] = user
        prov[PROV_TAG_ON_BEHALF_OF] = user
        prov[PROV_TAG_ON_BEHALF_OF_SOURCE] = "cli"
    hostname = socket.gethostname()
    if hostname:
        prov[PROV_TAG_HOST_NAME] = hostname
    env = _capture_env_vars(_ENV_ALLOWLIST)
    if env:
        prov[PROV_TAG_ENV] = env
    return prov


_PROCESS_PROVENANCE = _capture_process_provenance()


def _ts_ms() -> int:
    """Current time as milliseconds since epoch."""
    return int(time.time() * 1000)


def _make_item_id(session_id: str, label: str, ts: int) -> str:
    """Deterministic item ID from session + label + timestamp."""
    raw = f"{session_id}:{label}:{ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def build_provenance(
    session_id: str,
    *,
    parent_context_id: int | None = None,
    root_context_id: int | None = None,
    spawn_reason: str = "",
    agent_name: str = "",
    bundle_name: str = "",
) -> dict[int, Any]:
    """Build a full provenance map merging process-level and per-context fields.

    Starts with cached _PROCESS_PROVENANCE and layers on per-context identity.
    """
    prov: dict[int, Any] = dict(_PROCESS_PROVENANCE)

    if parent_context_id is not None:
        prov[PROV_TAG_PARENT_CTX] = parent_context_id
    if root_context_id is not None:
        prov[PROV_TAG_ROOT_CTX] = root_context_id
    elif parent_context_id is not None:
        prov[PROV_TAG_ROOT_CTX] = parent_context_id
    if spawn_reason:
        prov[PROV_TAG_SPAWN_REASON] = spawn_reason

    prov[PROV_TAG_TRACE_ID] = session_id

    env: dict[str, str] = dict(prov.get(PROV_TAG_ENV, {}))
    if agent_name:
        env["AMPLIFIER_AGENT"] = agent_name
    if bundle_name:
        env["AMPLIFIER_BUNDLE"] = bundle_name
    env["AMPLIFIER_SESSION"] = session_id
    if env:
        prov[PROV_TAG_ENV] = env

    prov[PROV_TAG_CAPTURED_AT] = _ts_ms()
    return prov


def build_context_metadata(
    session_id: str,
    context_label: str,
    *,
    client_tag: str = "amplifier",
    project_name: str = "",
    agent_name: str = "",
    bundle_name: str = "",
    parent_context_id: int | None = None,
    root_context_id: int | None = None,
    spawn_reason: str = "",
) -> dict[int, object]:
    """Build a context_metadata ConversationItem as the first turn of a context.

    CXDB's server extracts tag 30 from the first turn to populate the context's
    title, labels, and provenance in the UI. This makes contexts identifiable
    instead of anonymous.

    Args:
        session_id: Amplifier session ID.
        context_label: Human label like "Turns" or "Events".
        client_tag: Client application tag for CQL filtering.
        project_name: Project name (from workspace directory).
        agent_name: Agent name if child session.
        bundle_name: Active bundle name.
        parent_context_id: Parent CXDB context ID (for child sessions).
        root_context_id: Root CXDB context ID (for child sessions).
        spawn_reason: Why this context was created.

    Returns:
        Dict with integer keys ready for msgpack as cxdb.ConversationItem v3.
    """
    ts = _ts_ms()
    sid_short = session_id[:12]

    # Build title: "Amplifier Turns: ac344fc2..." or "Amplifier Events: ac344fc2..."
    title_parts = ["Amplifier", context_label]
    if project_name:
        title_parts.insert(1, project_name)
    title = " ".join(title_parts) + f": {sid_short}"

    # Labels for CQL filtering
    labels = ["amplifier", context_label.lower()]
    if project_name:
        labels.append(project_name)

    # Provenance
    provenance = build_provenance(
        session_id,
        parent_context_id=parent_context_id,
        root_context_id=root_context_id,
        spawn_reason=spawn_reason,
        agent_name=agent_name,
        bundle_name=bundle_name,
    )

    return {
        TAG_ITEM_TYPE: ITEM_SYSTEM,
        TAG_STATUS: "complete",
        TAG_TIMESTAMP: ts,
        TAG_ID: _make_item_id(session_id, f"context_metadata_{context_label.lower()}", ts),
        TAG_SYSTEM: {
            SM_TAG_KIND: "info",
            SM_TAG_TITLE: f"Context: {context_label}",
            SM_TAG_CONTENT: f"Amplifier session {session_id}",
        },
        TAG_CTX_META: {
            CM_TAG_CLIENT: client_tag,
            CM_TAG_TITLE: title,
            CM_TAG_LABELS: labels,
            CM_TAG_PROVENANCE: provenance,
        },
    }

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
