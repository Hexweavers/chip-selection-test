"""ChipPanel widget for displaying model chips grouped by type."""

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static
from textual.widget import Widget

from config import CHIP_TYPES


class ChipPanel(Widget):
    """Widget for displaying chips for a single model result."""

    DEFAULT_CSS = """
    ChipPanel {
        width: 1fr;
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    ChipPanel .panel-header {
        height: auto;
        padding: 1 0;
        border-bottom: solid $primary;
    }
    ChipPanel .model-name {
        text-style: bold;
    }
    ChipPanel .rating-stars {
        color: $warning;
        margin-left: 1;
    }
    ChipPanel .chip-content {
        height: 1fr;
    }
    ChipPanel .type-section {
        height: auto;
        margin-top: 1;
    }
    ChipPanel .type-header {
        text-style: bold;
        color: $primary;
    }
    ChipPanel .chip-item {
        margin-left: 2;
        color: $text;
    }
    ChipPanel .empty-chips {
        color: $text-muted;
        text-style: italic;
        margin-left: 2;
    }
    """

    def __init__(
        self, model: str, chips: list[dict], rating: int | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.chips = chips
        self.rating = rating

    @property
    def short_model_name(self) -> str:
        """Get the short model name (without provider prefix)."""
        return self.model.split("/")[-1]

    def format_rating_stars(self) -> str:
        """Format rating as star characters."""
        if self.rating is None:
            return "--"
        filled = "\u2605" * self.rating  # Black star
        empty = "\u2606" * (5 - self.rating)  # White star
        return filled + empty

    def group_chips_by_type(self) -> dict[str, list[dict]]:
        """Group chips by their type."""
        grouped: dict[str, list[dict]] = {chip_type: [] for chip_type in CHIP_TYPES}

        for chip in self.chips:
            chip_type = chip.get("type", "")
            if chip_type in grouped:
                grouped[chip_type].append(chip)

        return grouped

    def compose(self) -> ComposeResult:
        # Header with model name and rating
        with Vertical(classes="panel-header"):
            yield Static(self.short_model_name, classes="model-name")
            yield Static(self.format_rating_stars(), classes="rating-stars")

        # Scrollable content area
        with VerticalScroll(classes="chip-content"):
            grouped = self.group_chips_by_type()

            for chip_type in CHIP_TYPES:
                chips_in_type = grouped.get(chip_type, [])
                count = len(chips_in_type)

                with Vertical(classes="type-section"):
                    # Type header with count
                    type_label = chip_type.replace("_", " ").title()
                    yield Static(f"{type_label} ({count})", classes="type-header")

                    # Chip items
                    if chips_in_type:
                        for chip in chips_in_type:
                            display = chip.get("display", chip.get("key", "Unknown"))
                            yield Static(f"  \u2022 {display}", classes="chip-item")
                    else:
                        yield Static("  (none)", classes="empty-chips")
