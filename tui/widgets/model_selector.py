"""Model selector widget for choosing 1-2 models."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, Checkbox
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from config import MODELS


class ModelSelector(Widget):
    """Widget for selecting 1-2 models for head-to-head comparison."""

    DEFAULT_CSS = """
    ModelSelector {
        height: auto;
        border: solid $primary;
        padding: 1;
    }
    ModelSelector .title {
        text-style: bold;
        margin-bottom: 1;
    }
    ModelSelector .model-grid {
        height: auto;
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
    }
    ModelSelector .status {
        margin-top: 1;
        text-align: right;
        color: $text-muted;
    }
    ModelSelector .status.head-to-head {
        color: $success;
    }
    """

    selected: reactive[set[str]] = reactive(set, init=False)
    max_selections: int = 2

    class SelectionChanged(Message):
        """Posted when model selection changes."""

        def __init__(self, selected: set[str]) -> None:
            self.selected = selected
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self.models = [(m.id, m.name) for m in MODELS]
        self.selected = set()

    def compose(self) -> ComposeResult:
        yield Static("Models (pick 1-2 for head-to-head)", classes="title")
        with Container(classes="model-grid"):
            for model_id, model_name in self.models:
                safe_id = model_id.replace("/", "-").replace(".", "-")
                yield Checkbox(model_name, id=f"model-{safe_id}", name=model_id)
        yield Static("Selected: 0", classes="status", id="status")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes."""
        model_id = event.checkbox.name

        if event.value:
            if len(self.selected) >= self.max_selections:
                # Deselect the checkbox - can't select more than max
                event.checkbox.value = False
                self.notify(f"Maximum {self.max_selections} models allowed", severity="warning")
                return
            self.selected = self.selected | {model_id}
        else:
            self.selected = self.selected - {model_id}

        self._update_status()
        self.post_message(self.SelectionChanged(self.selected.copy()))

    def _update_status(self) -> None:
        """Update the status display."""
        status = self.query_one("#status", Static)
        count = len(self.selected)
        if count == 2:
            status.update("Selected: 2 (head-to-head)")
            status.add_class("head-to-head")
        else:
            status.update(f"Selected: {count}")
            status.remove_class("head-to-head")
