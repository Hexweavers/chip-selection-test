"""Monitor screen for tracking test run progress."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, ProgressBar
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from tui.widgets.log_pane import LogPane


def sanitize_id(model_id: str) -> str:
    """Sanitize model ID for use as element ID (replace / and . with -)."""
    return model_id.replace("/", "-").replace(".", "-")


class ModelProgress(Widget):
    """Progress display widget for a single model."""

    DEFAULT_CSS = """
    ModelProgress {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }
    ModelProgress .model-header {
        height: auto;
        layout: horizontal;
    }
    ModelProgress .model-name {
        width: 1fr;
        text-style: bold;
    }
    ModelProgress .model-stats {
        width: auto;
        color: $text-muted;
    }
    ModelProgress .progress-row {
        height: auto;
        margin-top: 1;
        layout: horizontal;
    }
    ModelProgress ProgressBar {
        width: 1fr;
    }
    ModelProgress .step-label {
        width: 20;
        text-align: right;
        margin-left: 1;
    }
    """

    def __init__(self, model_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self._tokens = 0
        self._cost = 0.0
        self._step = ""

    def compose(self) -> ComposeResult:
        with Container(classes="model-header"):
            yield Static(self.model_id, classes="model-name")
            yield Static("Tokens: 0 | Cost: $0.00", classes="model-stats", id="stats")
        with Container(classes="progress-row"):
            yield ProgressBar(total=100, show_eta=False, id="progress-bar")
            yield Static("", classes="step-label", id="step-label")

    def update_progress(self, percent: float, step: str) -> None:
        """Update the progress bar and step label."""
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(progress=percent)
        self._step = step
        step_label = self.query_one("#step-label", Static)
        step_label.update(step)

    def update_stats(self, tokens: int, cost: float) -> None:
        """Update the tokens and cost display."""
        self._tokens = tokens
        self._cost = cost
        stats = self.query_one("#stats", Static)
        stats.update(f"Tokens: {tokens:,} | Cost: ${cost:.2f}")

    def get_stats_text(self) -> str:
        """Get the current stats text for testing."""
        return f"Tokens: {self._tokens:,} | Cost: ${self._cost:.2f}"

    def render_str(self) -> str:
        """Get rendered text (for testing)."""
        return self.model_id


class MonitorScreen(Widget):
    """Monitor screen for tracking test run progress."""

    DEFAULT_CSS = """
    MonitorScreen {
        height: 100%;
    }
    MonitorScreen #idle-container {
        height: 100%;
        align: center middle;
    }
    MonitorScreen #idle-message {
        text-style: italic;
        color: $text-muted;
    }
    MonitorScreen #run-container {
        height: 100%;
    }
    MonitorScreen #run-header {
        height: auto;
        padding: 1;
        background: $surface;
        border-bottom: solid $primary;
    }
    MonitorScreen #run-name {
        text-style: bold;
        margin-bottom: 1;
    }
    MonitorScreen #config-summary {
        color: $text-muted;
    }
    MonitorScreen #progress-area {
        height: auto;
        max-height: 40%;
        padding: 1;
    }
    MonitorScreen #logs-area {
        height: 1fr;
        padding: 1;
    }
    MonitorScreen .log-column {
        width: 1fr;
        height: 100%;
        margin: 0 1;
    }
    MonitorScreen #buttons {
        height: auto;
        padding: 1;
        align: center middle;
        border-top: solid $primary;
    }
    MonitorScreen Button {
        margin: 0 1;
    }
    MonitorScreen #cancel-btn {
        background: $error;
    }
    """

    is_idle: reactive[bool] = reactive(True)

    class CancelRequested(Message):
        """Posted when user clicks Cancel."""

        pass

    class ViewResultsRequested(Message):
        """Posted when user clicks View Results."""

        pass

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._models: list[str] = []
        self._run_name = ""
        self._config_str = ""

    def compose(self) -> ComposeResult:
        with Container(id="idle-container"):
            yield Static(
                "No run in progress. Configure and start a test.",
                id="idle-message",
            )
        with Container(id="run-container"):
            with Container(id="run-header"):
                yield Static("", id="run-name")
                yield Static("", id="config-summary")
            with VerticalScroll(id="progress-area"):
                pass  # Will be populated dynamically
            with Horizontal(id="logs-area"):
                pass  # Will be populated dynamically
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel-btn", variant="error")
                yield Button("Pause", id="pause-btn", variant="warning", disabled=True)
                yield Button(
                    "View Results", id="view-results-btn", variant="success", disabled=True
                )

    def on_mount(self) -> None:
        """Set initial visibility."""
        self._update_visibility()

    def watch_is_idle(self, value: bool) -> None:
        """Update visibility when idle state changes."""
        self._update_visibility()

    def _update_visibility(self) -> None:
        """Update container visibility based on idle state."""
        idle_container = self.query_one("#idle-container")
        run_container = self.query_one("#run-container")
        idle_container.display = self.is_idle
        run_container.display = not self.is_idle

    def setup_run(self, models: list[str], run_name: str, config_str: str) -> None:
        """Prepare the monitor for a new run."""
        self._models = models
        self._run_name = run_name
        self._config_str = config_str

        # Update header
        self.query_one("#run-name", Static).update(run_name)
        self.query_one("#config-summary", Static).update(config_str)

        # Clear and populate progress area
        progress_area = self.query_one("#progress-area")
        progress_area.remove_children()
        for model in models:
            safe_id = sanitize_id(model)
            progress_widget = ModelProgress(model_id=model, id=f"progress-{safe_id}")
            progress_area.mount(progress_widget)

        # Clear and populate logs area
        logs_area = self.query_one("#logs-area")
        logs_area.remove_children()
        for model in models:
            safe_id = sanitize_id(model)
            log_pane = LogPane(title=model, id=f"log-{safe_id}", classes="log-column")
            logs_area.mount(log_pane)

        # Reset button states
        self.query_one("#cancel-btn", Button).disabled = False
        self.query_one("#pause-btn", Button).disabled = True
        self.query_one("#view-results-btn", Button).disabled = True

        # Switch to running state
        self.is_idle = False

    def add_log(self, model: str, message: str) -> None:
        """Add a log line to the specified model's log pane."""
        safe_id = sanitize_id(model)
        try:
            log_pane = self.query_one(f"#log-{safe_id}", LogPane)
            log_pane.add_line(message)
        except Exception:
            pass  # Model not found, ignore

    def update_progress(self, model: str, percent: float, step: str) -> None:
        """Update the progress for a specific model."""
        safe_id = sanitize_id(model)
        try:
            progress = self.query_one(f"#progress-{safe_id}", ModelProgress)
            progress.update_progress(percent, step)
        except Exception:
            pass  # Model not found, ignore

    def update_stats(self, model: str, tokens: int, cost: float) -> None:
        """Update the stats for a specific model."""
        safe_id = sanitize_id(model)
        try:
            progress = self.query_one(f"#progress-{safe_id}", ModelProgress)
            progress.update_stats(tokens, cost)
        except Exception:
            pass  # Model not found, ignore

    def mark_complete(self) -> None:
        """Mark the run as complete, enabling View Results and disabling Cancel."""
        self.query_one("#cancel-btn", Button).disabled = True
        self.query_one("#view-results-btn", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.post_message(self.CancelRequested())
        elif event.button.id == "view-results-btn":
            self.post_message(self.ViewResultsRequested())
