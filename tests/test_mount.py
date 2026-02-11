"""Tests for mount() entry point - event registration and lifecycle."""

import asyncio

import pytest

from amplifier_module_hooks_cxdb_events import mount


class TestMountNoop:
    @pytest.mark.asyncio
    async def test_noop_when_no_config(self, mock_coordinator):
        """mount() returns None when config is empty."""
        result = await mount(mock_coordinator, {})
        assert result is None
        mock_coordinator.hooks.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_host_empty(self, mock_coordinator):
        """mount() returns None when cxdb_host is empty string."""
        result = await mount(mock_coordinator, {"cxdb_host": ""})
        assert result is None
        mock_coordinator.hooks.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_host_missing(self, mock_coordinator):
        """mount() returns None when cxdb_host key is absent."""
        result = await mount(mock_coordinator, {"cxdb_port": 9009})
        assert result is None
        mock_coordinator.hooks.register.assert_not_called()


class TestMountRegistration:
    @pytest.mark.asyncio
    async def test_registers_hooks_when_configured(self, mock_coordinator):
        """mount() registers hooks when cxdb_host is set."""
        result = await mount(mock_coordinator, {"cxdb_host": "localhost"})
        assert result is not None
        # Should register many events (ALL_EVENTS + module events - excludes)
        assert mock_coordinator.hooks.register.call_count > 40

    @pytest.mark.asyncio
    async def test_registers_canonical_events(self, mock_coordinator):
        """Canonical events from ALL_EVENTS are registered."""
        await mount(mock_coordinator, {"cxdb_host": "localhost"})
        registered = [call.args[0] for call in mock_coordinator.hooks.register.call_args_list]
        # Spot check some canonical events
        assert "session:start" in registered
        assert "tool:pre" in registered
        assert "tool:post" in registered
        assert "provider:response" in registered
        assert "orchestrator:complete" in registered

    @pytest.mark.asyncio
    async def test_registers_module_events(self, mock_coordinator):
        """Module-specific events (task:*) are registered."""
        await mount(mock_coordinator, {"cxdb_host": "localhost"})
        registered = [call.args[0] for call in mock_coordinator.hooks.register.call_args_list]
        assert "task:agent_spawned" in registered
        assert "task:agent_completed" in registered
        assert "task:agent_resumed" in registered


class TestMountExclusions:
    @pytest.mark.asyncio
    async def test_default_excludes(self, mock_coordinator):
        """Default excluded events are not registered."""
        await mount(mock_coordinator, {"cxdb_host": "localhost"})
        registered = [call.args[0] for call in mock_coordinator.hooks.register.call_args_list]
        assert "content_block:delta" not in registered
        assert "thinking:delta" not in registered

    @pytest.mark.asyncio
    async def test_custom_exact_exclusion(self, mock_coordinator):
        """Custom exact-match exclusions work."""
        await mount(mock_coordinator, {
            "cxdb_host": "localhost",
            "exclude_events": ["session:end"],
        })
        registered = [call.args[0] for call in mock_coordinator.hooks.register.call_args_list]
        assert "session:end" not in registered
        assert "session:start" in registered  # other session events still registered

    @pytest.mark.asyncio
    async def test_custom_glob_exclusion(self, mock_coordinator):
        """Glob pattern exclusions work."""
        await mount(mock_coordinator, {
            "cxdb_host": "localhost",
            "exclude_events": ["artifact:*"],
        })
        registered = [call.args[0] for call in mock_coordinator.hooks.register.call_args_list]
        assert "artifact:write" not in registered
        assert "artifact:read" not in registered
        assert "tool:post" in registered  # non-matching events still registered

    @pytest.mark.asyncio
    async def test_custom_exclusions_add_to_defaults(self, mock_coordinator):
        """Custom exclusions are added to defaults, not replacing them."""
        await mount(mock_coordinator, {
            "cxdb_host": "localhost",
            "exclude_events": ["session:end"],
        })
        registered = [call.args[0] for call in mock_coordinator.hooks.register.call_args_list]
        # Default excludes still active
        assert "content_block:delta" not in registered
        # Custom exclude also active
        assert "session:end" not in registered


class TestMountAdditionalEvents:
    @pytest.mark.asyncio
    async def test_additional_events_registered(self, mock_coordinator):
        """Additional events from config are registered."""
        await mount(mock_coordinator, {
            "cxdb_host": "localhost",
            "additional_events": ["custom:my_event", "custom:other"],
        })
        registered = [call.args[0] for call in mock_coordinator.hooks.register.call_args_list]
        assert "custom:my_event" in registered
        assert "custom:other" in registered


class TestMountPriority:
    @pytest.mark.asyncio
    async def test_default_priority_100(self, mock_coordinator):
        """Default priority is 100 (runs last, observation-only)."""
        await mount(mock_coordinator, {"cxdb_host": "localhost"})
        for call in mock_coordinator.hooks.register.call_args_list:
            assert call.kwargs.get("priority") == 100

    @pytest.mark.asyncio
    async def test_custom_priority(self, mock_coordinator):
        """Priority is configurable."""
        await mount(mock_coordinator, {"cxdb_host": "localhost", "priority": 50})
        for call in mock_coordinator.hooks.register.call_args_list:
            assert call.kwargs.get("priority") == 50


class TestMountCleanup:
    @pytest.mark.asyncio
    async def test_returns_async_cleanup(self, mock_coordinator):
        """mount() returns an async cleanup function."""
        cleanup = await mount(mock_coordinator, {"cxdb_host": "localhost"})
        assert cleanup is not None
        assert asyncio.iscoroutinefunction(cleanup)

    @pytest.mark.asyncio
    async def test_cleanup_callable(self, mock_coordinator):
        """Cleanup function can be called without errors."""
        cleanup = await mount(mock_coordinator, {"cxdb_host": "localhost"})
        await cleanup()  # should not raise


class TestMountHookName:
    @pytest.mark.asyncio
    async def test_all_hooks_named_cxdb_events(self, mock_coordinator):
        """All hooks are registered with name='cxdb-events'."""
        await mount(mock_coordinator, {"cxdb_host": "localhost"})
        for call in mock_coordinator.hooks.register.call_args_list:
            assert call.kwargs.get("name") == "cxdb-events"
