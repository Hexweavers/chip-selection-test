import pytest
from textual.app import App, ComposeResult

from tui.widgets.model_selector import ModelSelector


class ModelSelectorTestApp(App):
    def compose(self) -> ComposeResult:
        yield ModelSelector()


@pytest.mark.asyncio
async def test_model_selector_allows_max_two():
    """ModelSelector should allow selecting at most 2 models."""
    app = ModelSelectorTestApp()
    async with app.run_test() as pilot:
        selector = app.query_one(ModelSelector)
        assert len(selector.selected) == 0
        assert selector.max_selections == 2


@pytest.mark.asyncio
async def test_model_selector_shows_models():
    """ModelSelector should display available models."""
    app = ModelSelectorTestApp()
    async with app.run_test() as pilot:
        selector = app.query_one(ModelSelector)
        # Should have some model options
        assert len(selector.models) > 0
