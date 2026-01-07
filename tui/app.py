"""Main Textual application."""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static
from textual.binding import Binding

from tui.screens.configure import ConfigureScreen


class ChipBenchmarkApp(App):
    """Chip benchmark TUI application."""

    TITLE = "Chip Benchmark"
    CSS = """
    Screen {
        background: $surface;
    }
    TabbedContent {
        height: 100%;
    }
    TabPane {
        padding: 1;
    }
    .placeholder {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
        Binding("1", "switch_tab('configure')", "Configure", show=False),
        Binding("2", "switch_tab('monitor')", "Monitor", show=False),
        Binding("3", "switch_tab('results')", "Results", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="configure"):
            with TabPane("Configure", id="configure"):
                yield ConfigureScreen()
            with TabPane("Monitor", id="monitor"):
                yield Static("Monitor screen placeholder", classes="placeholder")
            with TabPane("Results", id="results"):
                yield Static("Results screen placeholder", classes="placeholder")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        self.query_one(TabbedContent).active = tab_id

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press 1/2/3 to switch tabs, q to quit")

    def on_configure_screen_run_requested(
        self, event: ConfigureScreen.RunRequested
    ) -> None:
        """Handle run request from configure screen."""
        mode = "Dry run" if event.dry_run else "Run"
        self.notify(
            f"{mode}: {len(event.models)} model(s), persona={event.persona_id}, "
            f"style={event.prompt_style}, flow={event.flow}, "
            f"constraint={event.constraint_type}, chips={event.chip_count}"
        )
        self.query_one(TabbedContent).active = "monitor"


def run():
    """Run the application."""
    app = ChipBenchmarkApp()
    app.run()


if __name__ == "__main__":
    run()
