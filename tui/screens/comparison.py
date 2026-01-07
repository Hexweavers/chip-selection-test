"""ComparisonScreen for viewing detailed run results."""

import json

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Button, TabbedContent, TabPane, DataTable
from textual.binding import Binding

from db.repository import Repository
from tui.widgets.chip_panel import ChipPanel
from tui.widgets.rating_bar import RatingBar


class ComparisonScreen(Screen):
    """Screen for comparing model results from a run."""

    DEFAULT_CSS = """
    ComparisonScreen {
        background: $surface;
    }
    ComparisonScreen #screen-header {
        height: auto;
        padding: 1;
        border-bottom: solid $primary;
        background: $surface;
    }
    ComparisonScreen #header-row {
        height: auto;
        layout: horizontal;
    }
    ComparisonScreen #back-btn {
        width: auto;
        margin-right: 2;
    }
    ComparisonScreen #comparison-title {
        width: 1fr;
        text-style: bold;
    }
    ComparisonScreen #config-summary {
        color: $text-muted;
        margin-top: 1;
    }
    ComparisonScreen #main-content {
        height: 1fr;
    }
    ComparisonScreen TabbedContent {
        height: 100%;
    }
    ComparisonScreen TabPane {
        padding: 1;
    }
    ComparisonScreen #chips-container {
        height: 100%;
        layout: horizontal;
    }
    ComparisonScreen #stats-container {
        height: 100%;
    }
    ComparisonScreen DataTable {
        height: 100%;
    }
    ComparisonScreen #raw-container {
        height: 100%;
    }
    ComparisonScreen #raw-json {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("backspace", "go_back", "Back", show=False),
    ]

    def __init__(self, run_id: str, repo: Repository, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        self.repo = repo
        self._run: dict | None = None
        self._results: list[dict] = []

    def on_mount(self) -> None:
        """Load data when screen is mounted."""
        self._load_data()
        self._populate_ui()

    def _load_data(self) -> None:
        """Load run and results from repository."""
        self._run = self.repo.get_run(self.run_id)
        self._results = self.repo.get_results_for_run(self.run_id)

    def _get_model_names(self) -> str:
        """Get formatted model names for display."""
        if not self._results:
            return "No results"

        names = [r.get("model", "Unknown").split("/")[-1] for r in self._results]
        if len(names) == 1:
            return names[0]
        return " vs ".join(names)

    def _get_config_summary(self) -> str:
        """Get config summary string."""
        if not self._run:
            return ""

        return (
            f"Persona: {self._run.get('persona', '?')} | "
            f"Style: {self._run.get('prompt_style', '?')} | "
            f"Flow: {self._run.get('flow', '?')} | "
            f"Chips: {self._run.get('chip_count', '?')}"
        )

    def compose(self) -> ComposeResult:
        # Header
        with Container(id="screen-header"):
            with Horizontal(id="header-row"):
                yield Button("\u2190 Back", id="back-btn", variant="default")
                yield Static("Loading...", id="comparison-title")
            yield Static("", id="config-summary")

        # Main content with tabs
        with Container(id="main-content"):
            with TabbedContent(initial="chips-tab"):
                with TabPane("Chips", id="chips-tab"):
                    with Horizontal(id="chips-container"):
                        pass  # Will be populated dynamically

                with TabPane("Stats", id="stats-tab"):
                    with VerticalScroll(id="stats-container"):
                        yield DataTable(id="stats-table")

                with TabPane("Raw", id="raw-tab"):
                    with VerticalScroll(id="raw-container"):
                        yield Static("", id="raw-json")

        # Rating bar at bottom
        yield RatingBar(results=[])  # Will be updated with actual results

    def _populate_ui(self) -> None:
        """Populate UI with loaded data."""
        # Update title
        title = self.query_one("#comparison-title", Static)
        title.update(self._get_model_names())

        # Update config summary
        config = self.query_one("#config-summary", Static)
        config.update(self._get_config_summary())

        # Populate chip panels
        chips_container = self.query_one("#chips-container", Horizontal)
        chips_container.remove_children()

        for i, result in enumerate(self._results):
            model = result.get("model", "Unknown")
            chips = result.get("chips", [])
            rating = result.get("rating")

            panel = ChipPanel(
                model=model,
                chips=chips,
                rating=rating,
                id=f"chip-panel-{i}",
            )
            chips_container.mount(panel)

        # Populate stats table
        self._populate_stats_table()

        # Populate raw JSON
        raw_json = self.query_one("#raw-json", Static)
        raw_data = {
            "run": self._run,
            "results": self._results,
        }
        raw_json.update(json.dumps(raw_data, indent=2, default=str))

        # Update rating bar
        old_bar = self.query_one(RatingBar)
        old_bar.remove()

        new_bar = RatingBar(results=self._results)
        self.mount(new_bar)

    def _populate_stats_table(self) -> None:
        """Populate the stats comparison table."""
        table = self.query_one("#stats-table", DataTable)
        table.clear(columns=True)

        # Add columns
        table.add_column("Metric", key="metric")
        for i, result in enumerate(self._results):
            model_name = result.get("model", "Unknown").split("/")[-1]
            table.add_column(model_name, key=f"model_{i}")

        # Add rows
        metrics = [
            ("Tokens In", "tokens_in"),
            ("Tokens Out", "tokens_out"),
            ("Total Tokens", None),  # Calculated
            ("Cost (USD)", "cost_usd"),
            ("Latency (ms)", "latency_ms"),
            ("Situation Count", "situation_count"),
            ("Jargon Count", "jargon_count"),
            ("Role/Task Count", "role_task_count"),
            ("Environment Count", "environment_count"),
            ("Rating", "rating"),
        ]

        for label, key in metrics:
            row_data = [label]
            for result in self._results:
                if key is None:
                    # Total tokens
                    value = result.get("tokens_in", 0) + result.get("tokens_out", 0)
                elif key == "cost_usd":
                    value = f"${result.get(key, 0):.4f}"
                elif key == "rating":
                    rating = result.get(key)
                    value = self._format_rating(rating)
                else:
                    value = result.get(key, "N/A")

                row_data.append(str(value))

            table.add_row(*row_data)

    def _format_rating(self, rating: int | None) -> str:
        """Format rating as stars."""
        if rating is None:
            return "--"
        filled = "\u2605" * rating
        empty = "\u2606" * (5 - rating)
        return filled + empty

    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-btn":
            self.app.pop_screen()

    def on_rating_bar_rating_changed(self, event: RatingBar.RatingChanged) -> None:
        """Handle rating changes from the RatingBar."""
        # Save rating to database
        self.repo.update_rating(event.result_id, event.rating)

        # Reload data and refresh stats
        self._load_data()
        self._populate_stats_table()

        # Update chip panel ratings
        for i, result in enumerate(self._results):
            try:
                panel = self.query_one(f"#chip-panel-{i}", ChipPanel)
                # Update the panel's rating
                panel.rating = result.get("rating")
                # Refresh the rating display
                rating_static = panel.query_one(".rating-stars", Static)
                rating_static.update(panel.format_rating_stars())
            except Exception:
                pass  # Panel not found, skip

        self.notify(f"Rating saved: {event.rating} stars")
