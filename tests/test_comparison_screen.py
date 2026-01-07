"""Tests for the ComparisonScreen."""

import pytest
from textual.app import App, ComposeResult

from db.schema import init_db
from db.repository import Repository


# === Fixtures ===


@pytest.fixture
def test_db():
    """Create an in-memory test database."""
    conn = init_db(":memory:")
    return conn


@pytest.fixture
def repo(test_db):
    """Create a repository with the test database."""
    return Repository(test_db)


@pytest.fixture
def repo_with_run(repo):
    """Create a repository with a test run and results."""
    # Create run
    run_id = repo.create_run(
        name="Test Run",
        persona="architect",
        prompt_style="terse",
        flow="basic",
        constraint_type="none",
        chip_count=15,
    )

    # Create results with chips
    chips1 = [
        {"key": "chip1", "display": "Working remotely", "type": "situation"},
        {"key": "chip2", "display": "API design", "type": "jargon"},
        {"key": "chip3", "display": "Tech lead", "type": "role_task"},
    ]
    result_id1 = repo.create_result(
        run_id=run_id,
        model="anthropic/claude-3-5-sonnet",
        chips=chips1,
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.01,
        latency_ms=500,
        situation_count=1,
        jargon_count=1,
        role_task_count=1,
        environment_count=0,
    )

    chips2 = [
        {"key": "chip4", "display": "In office", "type": "situation"},
        {"key": "chip5", "display": "Kubernetes", "type": "jargon"},
    ]
    result_id2 = repo.create_result(
        run_id=run_id,
        model="openai/gpt-4o",
        chips=chips2,
        tokens_in=80,
        tokens_out=180,
        cost_usd=0.008,
        latency_ms=400,
        situation_count=1,
        jargon_count=1,
        role_task_count=0,
        environment_count=0,
    )

    return repo, run_id


# === ComparisonScreen Tests ===


@pytest.mark.asyncio
async def test_comparison_screen_initializes_with_run_id(repo_with_run):
    """ComparisonScreen should initialize with run_id and repo."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        # Access the screen directly from the app's screen stack
        screen = app.screen
        assert isinstance(screen, ComparisonScreen)
        assert screen.run_id == run_id
        assert screen.repo is repo


@pytest.mark.asyncio
async def test_comparison_screen_loads_run_data(repo_with_run):
    """ComparisonScreen should load run and results data."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)
        assert screen._run is not None
        assert len(screen._results) == 2


@pytest.mark.asyncio
async def test_comparison_screen_has_header_with_back_button(repo_with_run):
    """ComparisonScreen should have a header with back button."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from textual.widgets import Button

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)
        back_btn = screen.query_one("#back-btn", Button)
        assert back_btn is not None


@pytest.mark.asyncio
async def test_comparison_screen_has_tabbed_content(repo_with_run):
    """ComparisonScreen should have TabbedContent with Chips, Stats, Raw tabs."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from textual.widgets import TabbedContent, TabPane

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)

        tabs = screen.query_one(TabbedContent)
        assert tabs is not None

        # Check tab panes exist
        panes = screen.query(TabPane)
        pane_ids = [p.id for p in panes]
        assert "chips-tab" in pane_ids
        assert "stats-tab" in pane_ids
        assert "raw-tab" in pane_ids


@pytest.mark.asyncio
async def test_comparison_screen_has_chip_panels(repo_with_run):
    """ComparisonScreen should display ChipPanel for each result."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from tui.widgets.chip_panel import ChipPanel

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)
        panels = screen.query(ChipPanel)
        assert len(panels) == 2


@pytest.mark.asyncio
async def test_comparison_screen_has_rating_bar(repo_with_run):
    """ComparisonScreen should have RatingBar docked at bottom."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from tui.widgets.rating_bar import RatingBar

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)
        bar = screen.query_one(RatingBar)
        assert bar is not None


@pytest.mark.asyncio
async def test_comparison_screen_escape_pops_screen(repo_with_run):
    """ComparisonScreen should pop on escape key."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Verify we're on the comparison screen
        assert isinstance(app.screen, ComparisonScreen)

        # Press escape to go back
        await pilot.press("escape")
        await pilot.pause()

        # The comparison screen should no longer be the active screen
        assert not isinstance(app.screen, ComparisonScreen)


@pytest.mark.asyncio
async def test_comparison_screen_backspace_pops_screen(repo_with_run):
    """ComparisonScreen should pop on backspace key."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Verify we're on the comparison screen
        assert isinstance(app.screen, ComparisonScreen)

        # Press backspace to go back
        await pilot.press("backspace")
        await pilot.pause()

        # The comparison screen should no longer be the active screen
        assert not isinstance(app.screen, ComparisonScreen)


@pytest.mark.asyncio
async def test_comparison_screen_back_button_pops_screen(repo_with_run):
    """ComparisonScreen should pop when back button is clicked."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from textual.widgets import Button

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Verify we're on the comparison screen
        assert isinstance(app.screen, ComparisonScreen)

        back_btn = app.screen.query_one("#back-btn", Button)
        await pilot.click(back_btn)
        await pilot.pause()

        # The comparison screen should no longer be the active screen
        assert not isinstance(app.screen, ComparisonScreen)


@pytest.mark.asyncio
async def test_comparison_screen_displays_config_summary(repo_with_run):
    """ComparisonScreen should display config summary in header."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from textual.widgets import Static

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)

        config = screen.query_one("#config-summary", Static)
        # Access the screen's internal state for config content
        content = screen._get_config_summary()
        # Should contain config info
        assert "architect" in content or "terse" in content or "15" in content


@pytest.mark.asyncio
async def test_comparison_screen_handles_rating_changed(repo_with_run):
    """ComparisonScreen should save rating via repo.update_rating on RatingChanged."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from tui.widgets.rating_bar import RatingBar

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)

        # Get the rating bar and focus it
        bar = screen.query_one(RatingBar)
        bar.focus()

        # Rate the first model
        await pilot.press("4")
        await pilot.pause()

        # Check the rating was saved
        results = repo.get_results_for_run(run_id)
        # At least one result should have rating 4
        rated_results = [r for r in results if r.get("rating") == 4]
        assert len(rated_results) >= 1


@pytest.mark.asyncio
async def test_comparison_screen_shows_model_names_in_title(repo_with_run):
    """ComparisonScreen should show model names in the title."""
    repo, run_id = repo_with_run
    from tui.screens.comparison import ComparisonScreen
    from textual.widgets import Static

    class TestApp(App):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.run_id = run_id
            self.repo = repo

        def on_mount(self) -> None:
            self.push_screen(ComparisonScreen(run_id=self.run_id, repo=self.repo))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ComparisonScreen)

        title = screen.query_one("#comparison-title", Static)
        # Access the screen's internal state for model names
        content = screen._get_model_names()
        # Should contain model names (short versions)
        assert "claude-3-5-sonnet" in content or "gpt-4o" in content
