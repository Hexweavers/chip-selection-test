"""Log pane widget with auto-scroll support."""

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, RichLog
from textual.widget import Widget
from textual.reactive import reactive


class LogPane(Widget):
    """A log pane with auto-scroll and pause-on-interaction."""

    DEFAULT_CSS = """
    LogPane {
        height: 100%;
        border: solid $primary;
    }
    LogPane .header {
        height: 3;
        padding: 0 1;
        background: $primary;
        color: $text;
    }
    LogPane .header-title {
        text-style: bold;
    }
    LogPane .header-status {
        dock: right;
    }
    LogPane RichLog {
        height: 1fr;
        padding: 0 1;
    }
    """

    auto_scroll: reactive[bool] = reactive(True)

    def __init__(self, title: str = "Log", **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self._log_lines: list[str] = []

    def compose(self) -> ComposeResult:
        with Static(classes="header"):
            yield Static(self.title, classes="header-title")
            yield Static("[auto]", classes="header-status", id="scroll-status")
        yield RichLog(highlight=True, markup=True, id="log")

    def add_line(self, message: str, timestamp: bool = True) -> None:
        """Add a line to the log."""
        log = self.query_one("#log", RichLog)
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[dim]{ts}[/dim] {message}"
        else:
            line = message
        self._log_lines.append(line)
        log.write(line)
        if self.auto_scroll:
            log.scroll_end(animate=False)

    def get_log_text(self) -> str:
        """Get all log text for testing."""
        return "\n".join(self._log_lines)

    def clear(self) -> None:
        """Clear the log."""
        self._log_lines = []
        log = self.query_one("#log", RichLog)
        log.clear()

    def watch_auto_scroll(self, value: bool) -> None:
        """Update status when auto_scroll changes."""
        status = self.query_one("#scroll-status", Static)
        status.update("[auto]" if value else "[paused]")

    def toggle_auto_scroll(self) -> None:
        """Toggle auto-scroll state."""
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll:
            log = self.query_one("#log", RichLog)
            log.scroll_end(animate=False)
