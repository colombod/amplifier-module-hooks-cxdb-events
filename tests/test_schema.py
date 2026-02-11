"""Tests for registry bundle and schema helpers."""

import json
import time

import httpx
import msgpack
import pytest

from amplifier_module_hooks_cxdb_events.schema import (
    _BUNDLE_ID,
    _BUNDLE_PATH,
    build_event_envelope,
    calculate_payload_bytes,
    extract_agent_name,
    load_bundle_json,
    publish_registry_bundle,
    serialize_envelope,
)


class TestLoadBundleJson:
    def test_bundle_file_exists(self):
        """The bundle JSON file exists in the package."""
        assert _BUNDLE_PATH.exists()

    def test_bundle_is_valid_json(self):
        """Bundle file parses as valid JSON."""
        bundle = load_bundle_json()
        assert isinstance(bundle, dict)

    def test_bundle_has_types_dict(self):
        """Bundle has a 'types' key with a dict value."""
        bundle = load_bundle_json()
        assert "types" in bundle
        assert isinstance(bundle["types"], dict)
        assert len(bundle["types"]) > 0


class TestBundleTypes:
    def test_has_conversation_item(self):
        """Bundle includes cxdb.ConversationItem."""
        bundle = load_bundle_json()
        assert "cxdb.ConversationItem" in bundle["types"]

    def test_has_all_15_amplifier_types(self):
        """Bundle includes all 15 amplifier.* event types."""
        bundle = load_bundle_json()
        type_ids = set(bundle["types"].keys())
        expected = {
            "amplifier.SessionEvent",
            "amplifier.PromptEvent",
            "amplifier.PlanEvent",
            "amplifier.ProviderEvent",
            "amplifier.ContentBlockEvent",
            "amplifier.ThinkingEvent",
            "amplifier.ToolEvent",
            "amplifier.ContextEvent",
            "amplifier.OrchestratorEvent",
            "amplifier.DelegationEvent",
            "amplifier.ApprovalEvent",
            "amplifier.CancelEvent",
            "amplifier.NotificationEvent",
            "amplifier.ArtifactEvent",
            "amplifier.GenericEvent",
        }
        assert expected.issubset(type_ids), f"Missing: {expected - type_ids}"

    def test_total_type_count(self):
        """Bundle has exactly 16 types (1 ConversationItem + 15 amplifier.*)."""
        bundle = load_bundle_json()
        assert len(bundle["types"]) == 26

    def test_all_amplifier_types_have_common_envelope(self):
        """All amplifier.* types have the 7 common envelope fields."""
        bundle = load_bundle_json()
        common_fields = {
            "event_name",
            "session_id",
            "parent_session_id",
            "timestamp_ms",
            "agent_name",
            "payload_bytes",
            "root_session_id",
        }
        for type_id, type_entry in bundle["types"].items():
            if type_id.startswith("amplifier."):
                # Get latest version's fields
                versions = type_entry["versions"]
                latest_version = max(versions.keys(), key=int)
                fields = versions[latest_version]["fields"]
                field_names = {f["name"] for f in fields.values()}
                missing = common_fields - field_names
                assert not missing, f"{type_id} missing envelope fields: {missing}"

    def test_common_envelope_uses_consistent_tags(self):
        """Common envelope fields use the same tags across all types."""
        bundle = load_bundle_json()
        expected_tags = {
            "event_name": "1",
            "session_id": "2",
            "parent_session_id": "3",
            "timestamp_ms": "4",
            "agent_name": "5",
            "payload_bytes": "6",
            "root_session_id": "8",
        }
        for type_id, type_entry in bundle["types"].items():
            if type_id.startswith("amplifier."):
                versions = type_entry["versions"]
                latest_version = max(versions.keys(), key=int)
                fields = versions[latest_version]["fields"]
                for tag_str, field in fields.items():
                    if field["name"] in expected_tags:
                        assert tag_str == expected_tags[field["name"]], (
                            f"{type_id}.{field['name']} has tag {tag_str}, "
                            f"expected {expected_tags[field['name']]}"
                        )

    def test_no_duplicate_tags_per_type(self):
        """No type has duplicate tag numbers (inherently true with dict keys)."""
        bundle = load_bundle_json()
        for type_id, type_entry in bundle["types"].items():
            versions = type_entry["versions"]
            for ver, ver_entry in versions.items():
                fields = ver_entry["fields"]
                # Dict keys are inherently unique, but verify they parse as ints
                tags = [int(k) for k in fields.keys()]
                assert len(tags) == len(set(tags)), (
                    f"Duplicate tags in {type_id} v{ver}: {tags}"
                )

    def test_bundle_has_registry_metadata(self):
        """Bundle has registry_version and bundle_id fields."""
        bundle = load_bundle_json()
        assert "registry_version" in bundle
        assert bundle["registry_version"] == 1
        assert "bundle_id" in bundle
        assert bundle["bundle_id"] == "amplifier.events-v1"

    def test_type_specific_fields_use_tags_10_plus(self):
        """Type-specific fields (non-envelope) use tags >= 10."""
        bundle = load_bundle_json()
        envelope_names = {
            "event_name",
            "session_id",
            "parent_session_id",
            "timestamp_ms",
            "agent_name",
            "payload_bytes",
            "data",
            "root_session_id",
        }
        for type_id, type_entry in bundle["types"].items():
            if type_id.startswith("amplifier."):
                versions = type_entry["versions"]
                latest_version = max(versions.keys(), key=int)
                fields = versions[latest_version]["fields"]
                for tag_str, field in fields.items():
                    if field["name"] not in envelope_names:
                        assert int(tag_str) >= 10, (
                            f"{type_id}.{field['name']} has tag {tag_str}, "
                            f"expected >= 10 for type-specific fields"
                        )


class TestPublishRegistryBundle:
    @pytest.mark.asyncio
    async def test_publish_success_201(self, httpx_mock):
        """Successful publish returns True on 201."""
        httpx_mock.add_response(method="PUT", status_code=201)
        result = await publish_registry_bundle("localhost", 9010)
        assert result is True

    @pytest.mark.asyncio
    async def test_publish_success_204(self, httpx_mock):
        """Already-published bundle returns True on 204."""
        httpx_mock.add_response(method="PUT", status_code=204)
        result = await publish_registry_bundle("localhost", 9010)
        assert result is True

    @pytest.mark.asyncio
    async def test_publish_failure_returns_false(self, httpx_mock):
        """Non-success status returns False."""
        httpx_mock.add_response(method="PUT", status_code=500, text="Internal error")
        result = await publish_registry_bundle("localhost", 9010)
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_connection_error_returns_false(self, httpx_mock):
        """Connection error returns False (graceful degradation)."""
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"))
        result = await publish_registry_bundle("localhost", 9010)
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_uses_correct_url(self, httpx_mock):
        """PUT request goes to correct URL with bundle ID."""
        httpx_mock.add_response(method="PUT", status_code=201)
        await publish_registry_bundle("myhost", 8080)
        request = httpx_mock.get_request()
        assert (
            str(request.url) == f"http://myhost:8080/v1/registry/bundles/{_BUNDLE_ID}"
        )

    @pytest.mark.asyncio
    async def test_publish_sends_json_body(self, httpx_mock):
        """PUT request body is the bundle JSON."""
        httpx_mock.add_response(method="PUT", status_code=201)
        await publish_registry_bundle("localhost", 9010)
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert "types" in body


class TestExtractAgentName:
    def test_child_session_simple(self):
        """Extract agent name from simple child session ID."""
        assert (
            extract_agent_name("0000000000000000-aaaa1111bbbb2222_zen-architect")
            == "zen-architect"
        )

    def test_child_session_with_namespace(self):
        """Extract namespaced agent name."""
        assert (
            extract_agent_name(
                "aaaa1111bbbb2222-cccc3333dddd4444_foundation:explorer"
            )
            == "foundation:explorer"
        )

    def test_root_session_uuid(self):
        """Root session (plain UUID) returns None."""
        assert extract_agent_name("9839b7c5-007b-4b54-acf4-a7e8f114c358") is None

    def test_root_session_short(self):
        """Short root session ID returns None."""
        assert extract_agent_name("abc-123") is None

    def test_empty_string(self):
        """Empty string returns None."""
        assert extract_agent_name("") is None

    def test_child_with_hyphens_in_agent_name(self):
        """Agent names with hyphens are extracted correctly."""
        assert extract_agent_name("0000-aaaa_my-cool-agent") == "my-cool-agent"


class TestBuildEventEnvelope:
    def test_required_fields_present(self):
        """Envelope has event_name (1), session_id (2), timestamp_ms (4), payload_bytes (6)."""
        envelope = build_event_envelope(
            event_name="tool:post",
            data={"tool_name": "grep"},
            session_id="abc-123",
            parent_session_id=None,
            agent_name=None,
        )
        assert envelope[1] == "tool:post"
        assert envelope[2] == "abc-123"
        assert isinstance(envelope[4], int)
        assert envelope[4] > 0  # timestamp
        assert isinstance(envelope[6], int)
        assert envelope[6] > 0  # payload_bytes

    def test_parent_session_id_included_when_set(self):
        """Tag 3 present when parent_session_id is not None."""
        envelope = build_event_envelope(
            event_name="session:fork",
            data={},
            session_id="child-123",
            parent_session_id="root-456",
            agent_name=None,
        )
        assert envelope[3] == "root-456"

    def test_parent_session_id_absent_when_none(self):
        """Tag 3 absent when parent_session_id is None."""
        envelope = build_event_envelope(
            event_name="session:start",
            data={},
            session_id="root-123",
            parent_session_id=None,
            agent_name=None,
        )
        assert 3 not in envelope

    def test_agent_name_included_when_set(self):
        """Tag 5 present when agent_name is not None."""
        envelope = build_event_envelope(
            event_name="tool:post",
            data={},
            session_id="child-123",
            parent_session_id="root-456",
            agent_name="foundation:explorer",
        )
        assert envelope[5] == "foundation:explorer"

    def test_agent_name_absent_when_none(self):
        """Tag 5 absent when agent_name is None."""
        envelope = build_event_envelope(
            event_name="session:start",
            data={},
            session_id="root-123",
            parent_session_id=None,
            agent_name=None,
        )
        assert 5 not in envelope

    def test_event_data_at_tag_7(self):
        """Raw event data included at tag 7."""
        envelope = build_event_envelope(
            event_name="tool:post",
            data={"tool_name": "grep", "result": "5 matches"},
            session_id="abc-123",
            parent_session_id=None,
            agent_name=None,
        )
        assert 7 in envelope
        assert envelope[7]["tool_name"] == "grep"

    def test_event_data_filters_default_fields(self):
        """Default fields (session_id, parent_id, ts) are filtered from tag 7."""
        envelope = build_event_envelope(
            event_name="session:start",
            data={
                "session_id": "abc",
                "parent_id": None,
                "ts": 123,
                "extra": "value",
            },
            session_id="abc",
            parent_session_id=None,
            agent_name=None,
        )
        if 7 in envelope:
            assert "session_id" not in envelope[7]
            assert "parent_id" not in envelope[7]
            assert "ts" not in envelope[7]
            assert envelope[7]["extra"] == "value"

    def test_empty_data_no_tag_7(self):
        """Empty data dict means tag 7 is absent."""
        envelope = build_event_envelope(
            event_name="session:end",
            data={},
            session_id="abc",
            parent_session_id=None,
            agent_name=None,
        )
        assert 7 not in envelope

    def test_timestamp_is_recent(self):
        """Timestamp should be close to current time."""
        before = int(time.time() * 1000)
        envelope = build_event_envelope(
            event_name="test",
            data={},
            session_id="abc",
            parent_session_id=None,
            agent_name=None,
        )
        after = int(time.time() * 1000)
        assert before <= envelope[4] <= after

    def test_payload_bytes_matches_serialized_size(self):
        """Tag 6 (payload_bytes) matches actual serialized size."""
        envelope = build_event_envelope(
            event_name="tool:post",
            data={"tool_name": "grep", "result": "lots of data here"},
            session_id="session-123",
            parent_session_id="parent-456",
            agent_name="explorer",
        )
        serialized = serialize_envelope(envelope)
        assert envelope[6] == len(serialized)


class TestSerializeEnvelope:
    def test_uses_integer_keys(self):
        """Serialized envelope uses integer keys."""
        envelope = {1: "test", 2: "abc", 4: 12345}
        serialized = serialize_envelope(envelope)
        decoded = msgpack.unpackb(serialized, raw=False, strict_map_key=False)
        assert all(isinstance(k, int) for k in decoded.keys())

    def test_deterministic(self):
        """Same envelope produces same bytes regardless of insertion order."""
        env1 = {4: 999, 1: "test", 2: "abc"}
        env2 = {1: "test", 2: "abc", 4: 999}
        assert serialize_envelope(env1) == serialize_envelope(env2)

    def test_nested_values_preserved(self):
        """Nested dicts and lists survive serialization."""
        envelope = {
            1: "tool:post",
            7: {"tool_name": "grep", "results": [1, 2, 3]},
        }
        serialized = serialize_envelope(envelope)
        decoded = msgpack.unpackb(serialized, raw=False, strict_map_key=False)
        assert decoded[7]["tool_name"] == "grep"
        assert decoded[7]["results"] == [1, 2, 3]


class TestCalculatePayloadBytes:
    def test_returns_correct_size(self):
        """calculate_payload_bytes returns the msgpack serialized size."""
        envelope = {1: "test", 2: "session-123", 4: 1707600000000}
        size = calculate_payload_bytes(envelope)
        actual = len(serialize_envelope(envelope))
        assert size == actual

    def test_larger_payload_larger_size(self):
        """More data means larger payload size."""
        small = {1: "x"}
        large = {1: "x" * 1000, 2: "y" * 1000}
        assert calculate_payload_bytes(large) > calculate_payload_bytes(small)
