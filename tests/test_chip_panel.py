"""Tests for the ChipPanel widget."""

import pytest
from textual.app import App, ComposeResult


class ChipPanelTestApp(App):
    """Test app for ChipPanel widget."""

    def __init__(self, model: str, chips: list[dict], rating: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._chips = chips
        self._rating = rating

    def compose(self) -> ComposeResult:
        from tui.widgets.chip_panel import ChipPanel

        yield ChipPanel(model=self._model, chips=self._chips, rating=self._rating)


# === ChipPanel Tests ===


@pytest.mark.asyncio
async def test_chip_panel_displays_model_name():
    """ChipPanel should display the model name in the header."""
    app = ChipPanelTestApp(
        model="anthropic/claude-3-5-sonnet",
        chips=[],
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        # Should display short model name
        assert panel.short_model_name == "claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_chip_panel_displays_model_name_without_slash():
    """ChipPanel should handle model names without provider prefix."""
    app = ChipPanelTestApp(
        model="gpt-4o",
        chips=[],
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        assert panel.short_model_name == "gpt-4o"


@pytest.mark.asyncio
async def test_chip_panel_displays_rating_stars_filled():
    """ChipPanel should display filled stars for rating."""
    app = ChipPanelTestApp(
        model="anthropic/claude-3-5-sonnet",
        chips=[],
        rating=4,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        stars = panel.format_rating_stars()
        assert stars == "★★★★☆"


@pytest.mark.asyncio
async def test_chip_panel_displays_rating_stars_empty():
    """ChipPanel should display empty stars for low rating."""
    app = ChipPanelTestApp(
        model="anthropic/claude-3-5-sonnet",
        chips=[],
        rating=1,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        stars = panel.format_rating_stars()
        assert stars == "★☆☆☆☆"


@pytest.mark.asyncio
async def test_chip_panel_displays_unrated():
    """ChipPanel should display '--' for no rating."""
    app = ChipPanelTestApp(
        model="anthropic/claude-3-5-sonnet",
        chips=[],
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        stars = panel.format_rating_stars()
        assert stars == "--"


@pytest.mark.asyncio
async def test_chip_panel_groups_chips_by_type():
    """ChipPanel should group chips by type."""
    chips = [
        {"key": "chip1", "display": "Chip 1", "type": "situation"},
        {"key": "chip2", "display": "Chip 2", "type": "jargon"},
        {"key": "chip3", "display": "Chip 3", "type": "situation"},
        {"key": "chip4", "display": "Chip 4", "type": "role_task"},
    ]
    app = ChipPanelTestApp(
        model="test-model",
        chips=chips,
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        grouped = panel.group_chips_by_type()

        assert "situation" in grouped
        assert "jargon" in grouped
        assert "role_task" in grouped
        assert len(grouped["situation"]) == 2
        assert len(grouped["jargon"]) == 1
        assert len(grouped["role_task"]) == 1


@pytest.mark.asyncio
async def test_chip_panel_groups_empty_chips():
    """ChipPanel should handle empty chip list."""
    app = ChipPanelTestApp(
        model="test-model",
        chips=[],
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        grouped = panel.group_chips_by_type()

        # All groups should be empty
        for chip_type in ["situation", "jargon", "role_task", "environment"]:
            assert chip_type in grouped
            assert len(grouped[chip_type]) == 0


@pytest.mark.asyncio
async def test_chip_panel_preserves_all_chip_types():
    """ChipPanel should have entries for all chip types even if empty."""
    chips = [
        {"key": "chip1", "display": "Chip 1", "type": "situation"},
    ]
    app = ChipPanelTestApp(
        model="test-model",
        chips=chips,
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        grouped = panel.group_chips_by_type()

        # All chip types should be present
        assert "situation" in grouped
        assert "jargon" in grouped
        assert "role_task" in grouped
        assert "environment" in grouped


@pytest.mark.asyncio
async def test_chip_panel_has_scrollable_content():
    """ChipPanel should have scrollable content area."""
    app = ChipPanelTestApp(
        model="test-model",
        chips=[],
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel
        from textual.containers import VerticalScroll

        panel = app.query_one(ChipPanel)
        # Should have a VerticalScroll container
        scroll = panel.query_one(VerticalScroll)
        assert scroll is not None


@pytest.mark.asyncio
async def test_chip_panel_displays_chip_count_per_type():
    """ChipPanel should display count in type headers."""
    chips = [
        {"key": "chip1", "display": "Chip 1", "type": "situation"},
        {"key": "chip2", "display": "Chip 2", "type": "situation"},
        {"key": "chip3", "display": "Chip 3", "type": "jargon"},
    ]
    app = ChipPanelTestApp(
        model="test-model",
        chips=chips,
        rating=None,
    )
    async with app.run_test() as pilot:
        from tui.widgets.chip_panel import ChipPanel

        panel = app.query_one(ChipPanel)
        grouped = panel.group_chips_by_type()

        # Verify counts
        assert len(grouped["situation"]) == 2
        assert len(grouped["jargon"]) == 1
        assert len(grouped["role_task"]) == 0
        assert len(grouped["environment"]) == 0
