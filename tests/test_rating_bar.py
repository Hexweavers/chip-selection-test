"""Tests for the RatingBar widget."""

import pytest
from textual.app import App, ComposeResult


class RatingBarTestApp(App):
    """Test app for RatingBar widget."""

    def __init__(self, results: list[dict], **kwargs):
        super().__init__(**kwargs)
        self._results = results
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        from tui.widgets.rating_bar import RatingBar

        yield RatingBar(results=self._results)

    def on_rating_bar_rating_changed(self, event) -> None:
        """Capture RatingChanged messages for testing."""
        self.captured_messages.append(event)


# === RatingBar Tests ===


@pytest.mark.asyncio
async def test_rating_bar_initializes_with_results():
    """RatingBar should initialize with results list."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
        {"id": "result-2", "model": "openai/gpt-4o", "rating": 3},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        assert bar.results == results
        assert len(bar.results) == 2


@pytest.mark.asyncio
async def test_rating_bar_shows_current_model():
    """RatingBar should show the current model name."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
        {"id": "result-2", "model": "openai/gpt-4o", "rating": 3},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        # Should start with first model
        assert bar.current_model == 0
        assert "claude-3-5-sonnet" in bar.get_current_model_short()


@pytest.mark.asyncio
async def test_rating_bar_switches_model_right():
    """RatingBar should switch to next model on right arrow."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
        {"id": "result-2", "model": "openai/gpt-4o", "rating": 3},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()
        await pilot.press("right")
        await pilot.pause()

        assert bar.current_model == 1


@pytest.mark.asyncio
async def test_rating_bar_switches_model_left():
    """RatingBar should switch to previous model on left arrow."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
        {"id": "result-2", "model": "openai/gpt-4o", "rating": 3},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()
        # Move right first
        await pilot.press("right")
        await pilot.pause()
        assert bar.current_model == 1

        # Then move left
        await pilot.press("left")
        await pilot.pause()
        assert bar.current_model == 0


@pytest.mark.asyncio
async def test_rating_bar_wraps_model_right():
    """RatingBar should wrap to first model when going past last."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
        {"id": "result-2", "model": "openai/gpt-4o", "rating": 3},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()
        await pilot.press("right")  # Index 1
        await pilot.press("right")  # Should wrap to 0
        await pilot.pause()

        assert bar.current_model == 0


@pytest.mark.asyncio
async def test_rating_bar_wraps_model_left():
    """RatingBar should wrap to last model when going past first."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
        {"id": "result-2", "model": "openai/gpt-4o", "rating": 3},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()
        await pilot.press("left")  # Should wrap to 1
        await pilot.pause()

        assert bar.current_model == 1


@pytest.mark.asyncio
async def test_rating_bar_posts_rating_changed_on_number_key():
    """RatingBar should post RatingChanged message on number key press."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()
        await pilot.press("4")
        await pilot.pause()

        assert len(app.captured_messages) == 1
        assert app.captured_messages[0].result_id == "result-1"
        assert app.captured_messages[0].rating == 4


@pytest.mark.asyncio
async def test_rating_bar_posts_rating_for_current_model():
    """RatingBar should post rating for the currently selected model."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
        {"id": "result-2", "model": "openai/gpt-4o", "rating": None},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()
        # Switch to second model
        await pilot.press("right")
        await pilot.pause()

        # Rate it
        await pilot.press("5")
        await pilot.pause()

        assert len(app.captured_messages) == 1
        assert app.captured_messages[0].result_id == "result-2"
        assert app.captured_messages[0].rating == 5


@pytest.mark.asyncio
async def test_rating_bar_accepts_ratings_1_to_5():
    """RatingBar should accept ratings from 1 to 5."""
    results = [
        {"id": "result-1", "model": "model-1", "rating": None},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()

        for rating in ["1", "2", "3", "4", "5"]:
            await pilot.press(rating)
            await pilot.pause()

        assert len(app.captured_messages) == 5
        for i, msg in enumerate(app.captured_messages):
            assert msg.rating == i + 1


@pytest.mark.asyncio
async def test_rating_bar_ignores_invalid_number_keys():
    """RatingBar should ignore 0, 6-9 keys."""
    results = [
        {"id": "result-1", "model": "model-1", "rating": None},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()

        for key in ["0", "6", "7", "8", "9"]:
            await pilot.press(key)
            await pilot.pause()

        # No messages should be captured
        assert len(app.captured_messages) == 0


@pytest.mark.asyncio
async def test_rating_bar_shows_saved_indicator():
    """RatingBar should show 'Saved' indicator briefly after rating."""
    results = [
        {"id": "result-1", "model": "model-1", "rating": None},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        bar.focus()
        await pilot.press("3")
        await pilot.pause()

        # show_saved should be True immediately after rating
        assert bar.show_saved is True


@pytest.mark.asyncio
async def test_rating_bar_handles_empty_results():
    """RatingBar should handle empty results gracefully."""
    results = []
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar

        bar = app.query_one(RatingBar)
        # Should not crash
        assert bar.results == []


@pytest.mark.asyncio
async def test_rating_bar_displays_label():
    """RatingBar should display 'Rate: <- {model} ->' label."""
    results = [
        {"id": "result-1", "model": "anthropic/claude-3-5-sonnet", "rating": None},
    ]
    app = RatingBarTestApp(results=results)
    async with app.run_test() as pilot:
        from tui.widgets.rating_bar import RatingBar
        from textual.widgets import Static

        bar = app.query_one(RatingBar)
        # Should have label static with model name
        label = bar.query_one("#model-label", Static)
        assert label is not None
