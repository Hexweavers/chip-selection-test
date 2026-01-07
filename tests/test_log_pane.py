import pytest
from textual.app import App, ComposeResult

from tui.widgets.log_pane import LogPane


class LogPaneTestApp(App):
    def compose(self) -> ComposeResult:
        yield LogPane(title="Test Model")


@pytest.mark.asyncio
async def test_log_pane_add_line():
    """LogPane should allow adding log lines."""
    app = LogPaneTestApp()
    async with app.run_test() as pilot:
        pane = app.query_one(LogPane)
        pane.add_line("Test message")
        assert "Test message" in pane.get_log_text()


@pytest.mark.asyncio
async def test_log_pane_auto_scroll():
    """LogPane should auto-scroll by default."""
    app = LogPaneTestApp()
    async with app.run_test() as pilot:
        pane = app.query_one(LogPane)
        assert pane.auto_scroll is True
