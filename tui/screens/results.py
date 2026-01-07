"""Results screen for browsing test run history."""

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, ListView, ListItem, Select, Button
from textual.widget import Widget
from textual.message import Message

from db.repository import Repository


def format_rating_stars(rating: int | None) -> str:
    """Format a rating as star characters."""
    if rating is None:
        return "--"
    filled = "★" * rating
    empty = "☆" * (5 - rating)
    return filled + empty


def parse_time(iso_str: str) -> str:
    """Extract HH:MM from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return "--:--"


def parse_date(iso_str: str) -> str:
    """Extract date from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return "Unknown"


class RunListItem(ListItem):
    """A list item displaying a single run with its results summary."""

    DEFAULT_CSS = """
    RunListItem {
        padding: 1;
        height: auto;
    }
    RunListItem:hover {
        background: $surface-lighten-1;
    }
    RunListItem .run-models {
        text-style: bold;
    }
    RunListItem .run-ratings {
        color: $warning;
        margin-left: 2;
    }
    RunListItem .run-config {
        color: $text-muted;
    }
    RunListItem .run-time {
        color: $text-muted;
        margin-left: 2;
    }
    """

    def __init__(self, run: dict, results: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self._run = run
        self._results = results

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._run.get("id", "")

    def compose(self) -> ComposeResult:
        # Build display strings
        models_str = self._format_models()
        ratings_str = self._format_ratings()
        config_str = self._format_config()
        time_str = parse_time(self._run.get("created_at", ""))

        with Vertical():
            with Horizontal():
                yield Static(models_str, classes="run-models")
                yield Static(ratings_str, classes="run-ratings")
                yield Static(time_str, classes="run-time")
            yield Static(config_str, classes="run-config")

    def _format_models(self) -> str:
        """Format model names as 'ModelA vs ModelB' or just 'ModelA'."""
        models = [r.get("model", "Unknown") for r in self._results]
        if len(models) == 0:
            return "No models"
        elif len(models) == 1:
            return models[0]
        else:
            return " vs ".join(models)

    def _format_ratings(self) -> str:
        """Format ratings as stars for each model."""
        if not self._results:
            return "--"
        ratings = [format_rating_stars(r.get("rating")) for r in self._results]
        return "  ".join(ratings)

    def _format_config(self) -> str:
        """Format config as 'persona | style | flow | chip_count chips'."""
        persona = self._run.get("persona", "?")
        style = self._run.get("prompt_style", "?")
        flow = self._run.get("flow", "?")
        chips = self._run.get("chip_count", 0)
        return f"{persona} | {style} | {flow} | {chips} chips"

    def get_display_text(self) -> str:
        """Get display text for testing purposes."""
        models = self._format_models()
        ratings = self._format_ratings()
        config = self._format_config()
        time = parse_time(self._run.get("created_at", ""))
        return f"{models} {ratings} {config} {time}"


class DateHeader(ListItem):
    """A non-selectable date header in the list."""

    DEFAULT_CSS = """
    DateHeader {
        padding: 0 1;
        height: auto;
        background: $surface;
        border-bottom: solid $primary;
    }
    DateHeader Static {
        text-style: bold;
        color: $text;
    }
    """

    def __init__(self, date_str: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._date_str = date_str
        self.disabled = True

    def compose(self) -> ComposeResult:
        yield Static(self._date_str)


class ResultsScreen(Widget):
    """Results browser screen for viewing and managing test runs."""

    DEFAULT_CSS = """
    ResultsScreen {
        height: 100%;
        layout: vertical;
    }
    ResultsScreen #filter-row {
        height: auto;
        padding: 1;
        layout: horizontal;
    }
    ResultsScreen #filter-row Select {
        width: 1fr;
        margin-right: 1;
    }
    ResultsScreen #filter-row Button {
        width: auto;
    }
    ResultsScreen #runs-list {
        height: 1fr;
        border: solid $primary;
        margin: 0 1;
    }
    ResultsScreen #button-row {
        height: auto;
        padding: 1;
        layout: horizontal;
        align: center middle;
    }
    ResultsScreen #button-row Button {
        margin: 0 1;
    }
    ResultsScreen #open-btn {
        background: $success;
    }
    ResultsScreen #delete-btn {
        background: $error;
    }
    """

    class RunSelected(Message):
        """Posted when a run is selected for viewing."""

        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            super().__init__()

    def __init__(self, repo: Repository, **kwargs) -> None:
        super().__init__(**kwargs)
        self.repo = repo
        self._item_counter = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="filter-row"):
            yield Select(
                [("All Personas", "all")],
                id="persona-filter",
                prompt="Filter by persona",
                value="all",
            )
            yield Select(
                [("All Models", "all")],
                id="model-filter",
                prompt="Filter by model",
                value="all",
            )
            yield Select(
                [
                    ("All Ratings", "all"),
                    ("Rated only", "rated"),
                    ("Unrated only", "unrated"),
                ],
                id="rating-filter",
                prompt="Filter by rating",
                value="all",
            )
            yield Button("Refresh", id="refresh-btn", variant="default")

        yield ListView(id="runs-list")

        with Horizontal(id="button-row"):
            yield Button("Open", id="open-btn", variant="success")
            yield Button("Delete", id="delete-btn", variant="error")
            yield Button("Export CSV", id="export-btn", variant="default")

    def on_mount(self) -> None:
        """Refresh runs when mounted."""
        self.refresh_runs()

    def refresh_runs(self) -> None:
        """Refresh the runs list from the repository."""
        runs = self.repo.list_runs()

        list_view = self.query_one("#runs-list", ListView)

        # Remove existing items properly
        for child in list(list_view.children):
            child.remove()

        # Increment counter to ensure unique IDs for new batch
        self._item_counter += 1

        # Group runs by date
        runs_by_date: dict[str, list[tuple[dict, list[dict]]]] = {}
        for run in runs:
            date_str = parse_date(run.get("created_at", ""))
            results = self.repo.get_results_for_run(run["id"])
            if date_str not in runs_by_date:
                runs_by_date[date_str] = []
            runs_by_date[date_str].append((run, results))

        # Add items to list view grouped by date
        header_idx = 0
        for date_str in sorted(runs_by_date.keys(), reverse=True):
            # Add date header with unique ID
            list_view.append(DateHeader(date_str, id=f"header-{self._item_counter}-{header_idx}"))
            header_idx += 1

            # Add run items for this date with unique IDs
            for run, results in runs_by_date[date_str]:
                item = RunListItem(
                    run=run,
                    results=results,
                    id=f"run-{self._item_counter}-{run['id']}"
                )
                list_view.append(item)

    def _get_selected_run_id(self) -> str | None:
        """Get the run_id of the currently selected item."""
        list_view = self.query_one("#runs-list", ListView)
        if list_view.highlighted_child is None:
            return None

        item = list_view.highlighted_child
        if isinstance(item, RunListItem):
            return item.run_id
        return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh-btn":
            self.refresh_runs()
        elif event.button.id == "open-btn":
            run_id = self._get_selected_run_id()
            if run_id:
                self.post_message(self.RunSelected(run_id))
            else:
                self.notify("Please select a run first", severity="warning")
        elif event.button.id == "delete-btn":
            run_id = self._get_selected_run_id()
            if run_id:
                self.repo.delete_run(run_id)
                self.refresh_runs()
                self.notify("Run deleted", severity="information")
            else:
                self.notify("Please select a run first", severity="warning")
        elif event.button.id == "export-btn":
            self.notify("Export CSV not implemented yet", severity="information")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection (Enter key or double-click)."""
        if isinstance(event.item, RunListItem):
            self.post_message(self.RunSelected(event.item.run_id))
