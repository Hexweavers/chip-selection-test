"""Configure screen for setting up test runs."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Select, Button, RadioButton, RadioSet
from textual.widget import Widget
from textual.message import Message

from config import MODELS
from tui.widgets.model_selector import ModelSelector


# Load personas for the dropdown
import json
from config import PERSONAS_FILE

def load_persona_options() -> list[tuple[str, str]]:
    with open(PERSONAS_FILE) as f:
        data = json.load(f)
    return [(p["id"], f"{p['desired_role']} ({p['sector']})") for p in data["personas"]]


class ConfigureScreen(Widget):
    """Configuration screen for test setup."""

    DEFAULT_CSS = """
    ConfigureScreen {
        height: 100%;
        padding: 1;
    }
    ConfigureScreen .section {
        margin-bottom: 1;
    }
    ConfigureScreen .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ConfigureScreen .options-row {
        height: auto;
        layout: horizontal;
    }
    ConfigureScreen .option-group {
        width: 1fr;
        margin-right: 2;
    }
    ConfigureScreen .option-group-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ConfigureScreen .buttons {
        margin-top: 2;
        height: auto;
        align: center middle;
    }
    ConfigureScreen Button {
        margin: 0 1;
    }
    ConfigureScreen #run-btn {
        background: $success;
    }
    """

    class RunRequested(Message):
        """Posted when user wants to run a test."""

        def __init__(
            self,
            models: list[str],
            persona_id: str,
            prompt_style: str,
            flow: str,
            constraint_type: str,
            chip_count: int,
            dry_run: bool = False,
        ) -> None:
            self.models = models
            self.persona_id = persona_id
            self.prompt_style = prompt_style
            self.flow = flow
            self.constraint_type = constraint_type
            self.chip_count = chip_count
            self.dry_run = dry_run
            super().__init__()

    def compose(self) -> ComposeResult:
        yield ModelSelector()

        yield Static("", classes="section")

        with Horizontal(classes="options-row"):
            with Vertical(classes="option-group"):
                yield Static("Persona", classes="option-group-title")
                yield Select(
                    load_persona_options(),
                    id="persona-select",
                    prompt="Select persona",
                )

            with Vertical(classes="option-group"):
                yield Static("Prompt Style", classes="option-group-title")
                with RadioSet(id="prompt-style"):
                    yield RadioButton("Terse", value=True, id="terse")
                    yield RadioButton("Guided", id="guided")

            with Vertical(classes="option-group"):
                yield Static("Flow", classes="option-group-title")
                with RadioSet(id="flow"):
                    yield RadioButton("Basic", value=True, id="basic")
                    yield RadioButton("Enriched", id="enriched")

        with Horizontal(classes="options-row"):
            with Vertical(classes="option-group"):
                yield Static("Constraints", classes="option-group-title")
                with RadioSet(id="constraint"):
                    yield RadioButton("None", value=True, id="constraint-none")
                    yield RadioButton("2-per-type", id="constraint-2-per-type")

            with Vertical(classes="option-group"):
                yield Static("Chip Count", classes="option-group-title")
                with RadioSet(id="chip-count"):
                    yield RadioButton("15", value=True, id="count-15")
                    yield RadioButton("35", id="count-35")

        with Horizontal(classes="buttons"):
            yield Button("Run Test", id="run-btn", variant="success")
            yield Button("Dry Run", id="dry-run-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id in ("run-btn", "dry-run-btn"):
            self._request_run(dry_run=event.button.id == "dry-run-btn")

    def _request_run(self, dry_run: bool = False) -> None:
        """Gather config and post run request."""
        selector = self.query_one(ModelSelector)
        models = list(selector.selected)

        if not models:
            self.notify("Please select at least one model", severity="error")
            return

        persona_select = self.query_one("#persona-select", Select)
        if persona_select.value == Select.BLANK:
            self.notify("Please select a persona", severity="error")
            return

        # Get radio selections
        prompt_style = self._get_radio_value("prompt-style")
        flow = self._get_radio_value("flow")
        constraint = self._get_radio_value("constraint")
        chip_count = int(self._get_radio_value("chip-count"))

        self.post_message(
            self.RunRequested(
                models=models,
                persona_id=str(persona_select.value),
                prompt_style=prompt_style,
                flow=flow,
                constraint_type=constraint,
                chip_count=chip_count,
                dry_run=dry_run,
            )
        )

    def _get_radio_value(self, radio_set_id: str) -> str:
        """Get the selected value from a RadioSet."""
        radio_set = self.query_one(f"#{radio_set_id}", RadioSet)
        pressed = radio_set.pressed_button
        if not pressed:
            return ""
        # Strip prefixes added to make IDs valid (constraint-, count-)
        btn_id = pressed.id
        if btn_id.startswith("constraint-"):
            return btn_id[len("constraint-") :]
        if btn_id.startswith("count-"):
            return btn_id[len("count-") :]
        return btn_id
