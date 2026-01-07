"""Tests for the MonitorScreen widget."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, ProgressBar, Button

from tui.screens.monitor import MonitorScreen, ModelProgress


class MonitorScreenTestApp(App):
    """Test app for MonitorScreen."""

    def compose(self) -> ComposeResult:
        yield MonitorScreen()


class ModelProgressTestApp(App):
    """Test app for ModelProgress widget."""

    def compose(self) -> ComposeResult:
        yield ModelProgress(model_id="gpt-4o")


# === ModelProgress Tests ===


@pytest.mark.asyncio
async def test_model_progress_displays_model_name():
    """ModelProgress should display the model name."""
    app = ModelProgressTestApp()
    async with app.run_test() as pilot:
        progress = app.query_one(ModelProgress)
        assert "gpt-4o" in progress.render_str()


@pytest.mark.asyncio
async def test_model_progress_has_progress_bar():
    """ModelProgress should contain a progress bar."""
    app = ModelProgressTestApp()
    async with app.run_test() as pilot:
        progress = app.query_one(ModelProgress)
        bar = progress.query_one(ProgressBar)
        assert bar is not None


@pytest.mark.asyncio
async def test_model_progress_update_progress():
    """ModelProgress.update_progress should update the progress bar."""
    app = ModelProgressTestApp()
    async with app.run_test() as pilot:
        progress = app.query_one(ModelProgress)
        progress.update_progress(50, "Step 1")
        bar = progress.query_one(ProgressBar)
        assert bar.progress == 50


@pytest.mark.asyncio
async def test_model_progress_update_stats():
    """ModelProgress.update_stats should update the stats display."""
    app = ModelProgressTestApp()
    async with app.run_test() as pilot:
        progress = app.query_one(ModelProgress)
        progress.update_stats(tokens=1500, cost=0.05)
        # Check that the stats text contains the values
        stats_text = progress.get_stats_text()
        assert "1500" in stats_text or "1,500" in stats_text
        assert "0.05" in stats_text


# === MonitorScreen Tests ===


@pytest.mark.asyncio
async def test_monitor_screen_idle_state():
    """MonitorScreen should show idle message when no run in progress."""
    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        assert monitor.is_idle is True
        # Check idle message is displayed (verify widget exists with expected ID)
        idle_text = monitor.query_one("#idle-message", Static)
        assert idle_text is not None


@pytest.mark.asyncio
async def test_monitor_screen_setup_run_creates_progress():
    """MonitorScreen.setup_run should create ModelProgress widgets."""
    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o", "claude-3-5-sonnet-20241022"],
            run_name="Test Run",
            config_str="persona=test, style=terse",
        )
        await pilot.pause()
        # Should no longer be idle
        assert monitor.is_idle is False
        # Should have two ModelProgress widgets
        progress_widgets = monitor.query(ModelProgress)
        assert len(progress_widgets) == 2


@pytest.mark.asyncio
async def test_monitor_screen_setup_run_creates_log_panes():
    """MonitorScreen.setup_run should create LogPane widgets for each model."""
    # Import here to avoid import errors if module doesn't exist yet
    from tui.widgets.log_pane import LogPane

    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o", "claude-3-5-sonnet-20241022"],
            run_name="Test Run",
            config_str="persona=test, style=terse",
        )
        await pilot.pause()
        # Should have two LogPane widgets
        log_panes = monitor.query(LogPane)
        assert len(log_panes) == 2


@pytest.mark.asyncio
async def test_monitor_screen_add_log_to_correct_pane():
    """MonitorScreen.add_log should add log to the correct model's pane."""
    from tui.widgets.log_pane import LogPane

    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o", "claude-3-5-sonnet-20241022"],
            run_name="Test Run",
            config_str="persona=test, style=terse",
        )
        await pilot.pause()

        # Add log to first model
        monitor.add_log("gpt-4o", "Test message for GPT")

        # Find the log pane for gpt-4o (using sanitized ID)
        sanitized_id = "gpt-4o".replace("/", "-").replace(".", "-")
        log_pane = monitor.query_one(f"#log-{sanitized_id}", LogPane)
        assert "Test message for GPT" in log_pane.get_log_text()


@pytest.mark.asyncio
async def test_monitor_screen_update_progress():
    """MonitorScreen.update_progress should update the correct model's progress."""
    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o"],
            run_name="Test Run",
            config_str="persona=test",
        )
        await pilot.pause()

        monitor.update_progress("gpt-4o", 75, "Step 2")

        sanitized_id = "gpt-4o".replace("/", "-").replace(".", "-")
        progress = monitor.query_one(f"#progress-{sanitized_id}", ModelProgress)
        bar = progress.query_one(ProgressBar)
        assert bar.progress == 75


@pytest.mark.asyncio
async def test_monitor_screen_update_stats():
    """MonitorScreen.update_stats should update the correct model's stats."""
    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o"],
            run_name="Test Run",
            config_str="persona=test",
        )
        await pilot.pause()

        monitor.update_stats("gpt-4o", tokens=2000, cost=0.10)

        sanitized_id = "gpt-4o".replace("/", "-").replace(".", "-")
        progress = monitor.query_one(f"#progress-{sanitized_id}", ModelProgress)
        stats_text = progress.get_stats_text()
        assert "2000" in stats_text or "2,000" in stats_text


@pytest.mark.asyncio
async def test_monitor_screen_mark_complete():
    """MonitorScreen.mark_complete should enable View Results and disable Cancel."""
    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o"],
            run_name="Test Run",
            config_str="persona=test",
        )
        await pilot.pause()

        # Initially, View Results should be disabled
        view_btn = monitor.query_one("#view-results-btn", Button)
        assert view_btn.disabled is True

        monitor.mark_complete()

        # Now View Results should be enabled
        assert view_btn.disabled is False

        # Cancel button should be disabled
        cancel_btn = monitor.query_one("#cancel-btn", Button)
        assert cancel_btn.disabled is True


@pytest.mark.asyncio
async def test_monitor_screen_posts_cancel_requested():
    """MonitorScreen should post CancelRequested when Cancel is clicked."""
    messages = []

    class CaptureApp(App):
        def compose(self) -> ComposeResult:
            yield MonitorScreen()

        def on_monitor_screen_cancel_requested(
            self, event: MonitorScreen.CancelRequested
        ) -> None:
            messages.append(event)

    app = CaptureApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o"],
            run_name="Test Run",
            config_str="persona=test",
        )
        await pilot.pause()

        cancel_btn = monitor.query_one("#cancel-btn", Button)
        await pilot.click(cancel_btn)

        assert len(messages) == 1


@pytest.mark.asyncio
async def test_monitor_screen_posts_view_results_requested():
    """MonitorScreen should post ViewResultsRequested when View Results is clicked."""
    messages = []

    class CaptureApp(App):
        def compose(self) -> ComposeResult:
            yield MonitorScreen()

        def on_monitor_screen_view_results_requested(
            self, event: MonitorScreen.ViewResultsRequested
        ) -> None:
            messages.append(event)

    app = CaptureApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        monitor.setup_run(
            models=["gpt-4o"],
            run_name="Test Run",
            config_str="persona=test",
        )
        await pilot.pause()

        # Mark complete to enable the button
        monitor.mark_complete()

        view_btn = monitor.query_one("#view-results-btn", Button)
        await pilot.click(view_btn)

        assert len(messages) == 1


@pytest.mark.asyncio
async def test_monitor_screen_sanitizes_model_ids():
    """MonitorScreen should sanitize model IDs with / and . for element IDs."""
    from tui.widgets.log_pane import LogPane

    app = MonitorScreenTestApp()
    async with app.run_test() as pilot:
        monitor = app.query_one(MonitorScreen)
        # Use a model ID with / and . characters
        monitor.setup_run(
            models=["openai/gpt-4.5-preview"],
            run_name="Test Run",
            config_str="persona=test",
        )
        await pilot.pause()

        # Should be able to query with sanitized ID
        sanitized_id = "openai-gpt-4-5-preview"
        log_pane = monitor.query_one(f"#log-{sanitized_id}", LogPane)
        assert log_pane is not None

        progress = monitor.query_one(f"#progress-{sanitized_id}", ModelProgress)
        assert progress is not None
