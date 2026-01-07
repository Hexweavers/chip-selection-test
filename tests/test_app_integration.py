"""Tests for App integration with MonitorScreen and ResultsScreen."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from textual.widgets import TabbedContent

from tui.app import ChipBenchmarkApp
from tui.screens.monitor import MonitorScreen
from tui.screens.configure import ConfigureScreen
from tui.screens.results import ResultsScreen


@pytest.mark.asyncio
async def test_app_has_monitor_screen():
    """App should have MonitorScreen in the monitor tab."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        assert monitor is not None


@pytest.mark.asyncio
async def test_app_initializes_runner():
    """App should initialize AsyncRunner."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        assert hasattr(app, "runner")
        assert app.runner is not None


@pytest.mark.asyncio
async def test_app_initializes_database():
    """App should initialize database connection and repository."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        assert hasattr(app, "db_conn")
        assert hasattr(app, "repo")
        assert app.db_conn is not None
        assert app.repo is not None


@pytest.mark.asyncio
async def test_app_handles_run_requested():
    """App should handle RunRequested message by switching to monitor tab."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        # Post a mock RunRequested message
        configure = app.query_one(ConfigureScreen)
        configure.post_message(
            ConfigureScreen.RunRequested(
                models=["gpt-4o"],
                persona_id="test-persona",
                prompt_style="terse",
                flow="basic",
                constraint_type="none",
                chip_count=15,
                dry_run=False,
            )
        )
        await pilot.pause()

        # Should have switched to monitor tab
        tabs = app.query_one(TabbedContent)
        assert tabs.active == "monitor"


@pytest.mark.asyncio
async def test_app_handles_cancel_requested():
    """App should handle CancelRequested by calling runner.cancel()."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        # Setup monitor with a run
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o"],
            run_name="Test",
            config_str="test",
        )
        await pilot.pause()

        # Mock the runner's cancel method
        app.runner.cancel = MagicMock()

        # Post CancelRequested
        monitor.post_message(MonitorScreen.CancelRequested())
        await pilot.pause()

        # Should have called cancel
        app.runner.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_app_handles_view_results_requested():
    """App should handle ViewResultsRequested by switching to results tab."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        # Setup monitor with a run
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o"],
            run_name="Test",
            config_str="test",
        )
        monitor.mark_complete()
        await pilot.pause()

        # Post ViewResultsRequested
        monitor.post_message(MonitorScreen.ViewResultsRequested())
        await pilot.pause()

        # Should have switched to results tab
        tabs = app.query_one(TabbedContent)
        assert tabs.active == "results"


# === ResultsScreen Integration Tests ===


@pytest.mark.asyncio
async def test_app_has_results_screen():
    """App should have ResultsScreen in the results tab."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        results_screen = app.query_one(ResultsScreen)
        assert results_screen is not None


@pytest.mark.asyncio
async def test_app_results_screen_has_repo():
    """App should pass repo to ResultsScreen."""
    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        results_screen = app.query_one(ResultsScreen)
        assert results_screen.repo is app.repo


@pytest.mark.asyncio
async def test_app_handles_run_selected():
    """App should handle RunSelected message by pushing ComparisonScreen."""
    from tui.screens.comparison import ComparisonScreen

    app = ChipBenchmarkApp()
    async with app.run_test() as pilot:
        # Create a test run so ResultsScreen has something
        run_id = app.repo.create_run(
            name="Test Run",
            persona="architect",
            prompt_style="terse",
            flow="basic",
            constraint_type="none",
            chip_count=15,
        )
        # Create a result so the ComparisonScreen has data
        app.repo.create_result(
            run_id=run_id,
            model="test-model",
            chips=[],
            tokens_in=100,
            tokens_out=200,
            cost_usd=0.01,
            latency_ms=500,
            situation_count=1,
            jargon_count=1,
            role_task_count=0,
            environment_count=0,
        )

        # Switch to results tab
        tabs = app.query_one(TabbedContent)
        tabs.active = "results"
        await pilot.pause()

        # Post RunSelected message
        results_screen = app.query_one(ResultsScreen)
        results_screen.post_message(ResultsScreen.RunSelected(run_id))
        await pilot.pause()

        # ComparisonScreen should be pushed
        assert isinstance(app.screen, ComparisonScreen)
        assert app.screen.run_id == run_id
