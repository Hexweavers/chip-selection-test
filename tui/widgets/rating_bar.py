"""RatingBar widget for rating model results."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive
from textual.binding import Binding


class RatingBar(Widget, can_focus=True):
    """Widget for rating model results with 1-5 stars."""

    DEFAULT_CSS = """
    RatingBar {
        height: auto;
        padding: 1;
        border-top: solid $primary;
        background: $surface;
        dock: bottom;
    }
    RatingBar #rating-container {
        height: auto;
        align: center middle;
    }
    RatingBar .arrow-hint {
        color: $text-muted;
        margin: 0 1;
    }
    RatingBar #model-label {
        text-style: bold;
        min-width: 30;
        text-align: center;
    }
    RatingBar #saved-indicator {
        color: $success;
        margin-left: 2;
    }
    RatingBar #rating-hint {
        color: $text-muted;
        margin-top: 1;
        text-align: center;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("left", "prev_model", "Previous model", show=False),
        Binding("right", "next_model", "Next model", show=False),
        Binding("1", "rate(1)", "Rate 1", show=False),
        Binding("2", "rate(2)", "Rate 2", show=False),
        Binding("3", "rate(3)", "Rate 3", show=False),
        Binding("4", "rate(4)", "Rate 4", show=False),
        Binding("5", "rate(5)", "Rate 5", show=False),
    ]

    current_model_index: reactive[int] = reactive(0)
    show_saved: reactive[bool] = reactive(False)

    class RatingChanged(Message):
        """Posted when a rating is changed."""

        def __init__(self, result_id: str, rating: int) -> None:
            self.result_id = result_id
            self.rating = rating
            super().__init__()

    def __init__(self, results: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self.results = results
        self._saved_timer = None

    def get_current_model_short(self) -> str:
        """Get the short name of the current model."""
        if not self.results:
            return "No models"
        result = self.results[self.current_model_index]
        model = result.get("model", "Unknown")
        return model.split("/")[-1]

    def compose(self) -> ComposeResult:
        with Horizontal(id="rating-container"):
            yield Static("\u2190", classes="arrow-hint")
            yield Static(self.get_current_model_short(), id="model-label")
            yield Static("\u2192", classes="arrow-hint")
            yield Static("", id="saved-indicator")

        yield Static("Press 1-5 to rate, \u2190/\u2192 to switch models", id="rating-hint")

    def watch_current_model_index(self, value: int) -> None:
        """Update display when model index changes."""
        try:
            label = self.query_one("#model-label", Static)
            label.update(self.get_current_model_short())
        except Exception:
            pass  # Widget not yet mounted

    def watch_show_saved(self, value: bool) -> None:
        """Update saved indicator visibility."""
        try:
            indicator = self.query_one("#saved-indicator", Static)
            if value:
                indicator.update("Saved \u2713")
            else:
                indicator.update("")
        except Exception:
            pass  # Widget not yet mounted

    def action_prev_model(self) -> None:
        """Switch to previous model (with wrap-around)."""
        if not self.results:
            return
        self.current_model_index = (self.current_model_index - 1) % len(self.results)

    def action_next_model(self) -> None:
        """Switch to next model (with wrap-around)."""
        if not self.results:
            return
        self.current_model_index = (self.current_model_index + 1) % len(self.results)

    def action_rate(self, rating: int) -> None:
        """Rate the current model."""
        if not self.results:
            return

        result = self.results[self.current_model_index]
        result_id = result.get("id", "")

        if not result_id:
            return

        # Post the rating changed message
        self.post_message(self.RatingChanged(result_id=result_id, rating=rating))

        # Show saved indicator
        self.show_saved = True

        # Clear saved indicator after delay
        if self._saved_timer is not None:
            self._saved_timer.stop()
        self._saved_timer = self.set_timer(1.5, self._clear_saved)

    def _clear_saved(self) -> None:
        """Clear the saved indicator."""
        self.show_saved = False
        self._saved_timer = None
