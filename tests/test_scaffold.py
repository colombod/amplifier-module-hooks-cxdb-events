"""Scaffolding tests - verify package structure and imports."""


def test_package_importable():
    """Package should be importable after pip install."""
    import amplifier_module_hooks_cxdb_events

    assert amplifier_module_hooks_cxdb_events is not None


def test_submodules_importable():
    """All stub modules should be importable."""
    from amplifier_module_hooks_cxdb_events import hook
    from amplifier_module_hooks_cxdb_events import buffer
    from amplifier_module_hooks_cxdb_events import turns
    from amplifier_module_hooks_cxdb_events import types
    from amplifier_module_hooks_cxdb_events import protocol
    from amplifier_module_hooks_cxdb_events import schema

    assert all(m is not None for m in [hook, buffer, turns, types, protocol, schema])


def test_conftest_fixtures_available(sample_event_data, mock_coordinator):
    """Conftest fixtures should be discoverable by pytest."""
    assert sample_event_data["session_id"] == "test-session-123"
    assert mock_coordinator.session_id == "test-session-123"
