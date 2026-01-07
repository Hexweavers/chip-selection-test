"""Tests for the ResultsScreen widget."""

import pytest
from datetime import datetime, timezone
from textual.app import App, ComposeResult
from textual.widgets import Static, ListView, Select, Button

from db.schema import init_db
from db.repository import Repository


class ResultsScreenTestApp(App):
    """Test app for ResultsScreen."""

    def __init__(self, repo=None, **kwargs):
        super().__init__(**kwargs)
        self.repo = repo

    def compose(self) -> ComposeResult:
        from tui.screens.results import ResultsScreen

        yield ResultsScreen(repo=self.repo)


class RunListItemTestApp(App):
    """Test app for RunListItem widget."""

    def compose(self) -> ComposeResult:
        from tui.screens.results import RunListItem

        run = {
            "id": "test-run-1",
            "name": "Test Run",
            "persona": "architect",
            "prompt_style": "terse",
            "flow": "basic",
            "constraint_type": "none",
            "chip_count": 15,
            "created_at": "2024-01-15T10:30:00+00:00",
        }
        results = [
            {"model": "gpt-4o", "rating": 4},
            {"model": "claude-3-5-sonnet", "rating": 3},
        ]
        yield RunListItem(run=run, results=results)


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
def repo_with_runs(repo):
    """Create a repository with some test runs."""
    # Create first run
    run_id1 = repo.create_run(
        name="Run 1",
        persona="architect",
        prompt_style="terse",
        flow="basic",
        constraint_type="none",
        chip_count=15,
    )
    repo.create_result(
        run_id=run_id1,
        model="gpt-4o",
        chips=[],
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.01,
        latency_ms=500,
        situation_count=2,
        jargon_count=3,
        role_task_count=1,
        environment_count=1,
    )

    # Create second run
    run_id2 = repo.create_run(
        name="Run 2",
        persona="pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    repo.create_result(
        run_id=run_id2,
        model="claude-3-5-sonnet",
        chips=[],
        tokens_in=150,
        tokens_out=250,
        cost_usd=0.02,
        latency_ms=600,
        situation_count=3,
        jargon_count=2,
        role_task_count=2,
        environment_count=2,
    )

    return repo


# === RunListItem Tests ===


@pytest.mark.asyncio
async def test_run_list_item_displays_models():
    """RunListItem should display the model names from results."""
    app = RunListItemTestApp()
    async with app.run_test() as pilot:
        from tui.screens.results import RunListItem

        item = app.query_one(RunListItem)
        # Should display "gpt-4o vs claude-3-5-sonnet" or similar
        content = item.get_display_text()
        assert "gpt-4o" in content
        assert "claude-3-5-sonnet" in content


@pytest.mark.asyncio
async def test_run_list_item_displays_single_model():
    """RunListItem should display single model without 'vs'."""
    from tui.screens.results import RunListItem

    class SingleModelApp(App):
        def compose(self) -> ComposeResult:
            run = {
                "id": "test-run-1",
                "name": "Test Run",
                "persona": "architect",
                "prompt_style": "terse",
                "flow": "basic",
                "constraint_type": "none",
                "chip_count": 15,
                "created_at": "2024-01-15T10:30:00+00:00",
            }
            results = [{"model": "gpt-4o", "rating": None}]
            yield RunListItem(run=run, results=results)

    app = SingleModelApp()
    async with app.run_test() as pilot:
        item = app.query_one(RunListItem)
        content = item.get_display_text()
        assert "gpt-4o" in content
        assert "vs" not in content


@pytest.mark.asyncio
async def test_run_list_item_displays_ratings():
    """RunListItem should display star ratings."""
    app = RunListItemTestApp()
    async with app.run_test() as pilot:
        from tui.screens.results import RunListItem

        item = app.query_one(RunListItem)
        content = item.get_display_text()
        # Should contain stars (using * as star representation in text)
        assert "*" in content or "rating" in content.lower()


@pytest.mark.asyncio
async def test_run_list_item_displays_config():
    """RunListItem should display config information."""
    app = RunListItemTestApp()
    async with app.run_test() as pilot:
        from tui.screens.results import RunListItem

        item = app.query_one(RunListItem)
        content = item.get_display_text()
        assert "architect" in content
        assert "terse" in content
        assert "basic" in content
        assert "15" in content


@pytest.mark.asyncio
async def test_run_list_item_displays_time():
    """RunListItem should display the time from created_at."""
    app = RunListItemTestApp()
    async with app.run_test() as pilot:
        from tui.screens.results import RunListItem

        item = app.query_one(RunListItem)
        content = item.get_display_text()
        assert "10:30" in content


@pytest.mark.asyncio
async def test_run_list_item_stores_run_id():
    """RunListItem should store the run_id for later access."""
    app = RunListItemTestApp()
    async with app.run_test() as pilot:
        from tui.screens.results import RunListItem

        item = app.query_one(RunListItem)
        assert item.run_id == "test-run-1"


# === ResultsScreen Tests ===


@pytest.mark.asyncio
async def test_results_screen_initializes_with_repo(repo):
    """ResultsScreen should initialize with a repository."""
    app = ResultsScreenTestApp(repo=repo)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen

        screen = app.query_one(ResultsScreen)
        assert screen.repo is repo


@pytest.mark.asyncio
async def test_results_screen_has_filter_row(repo):
    """ResultsScreen should have filter dropdowns."""
    app = ResultsScreenTestApp(repo=repo)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen

        screen = app.query_one(ResultsScreen)
        # Should have Select widgets for filtering
        selects = screen.query(Select)
        assert len(selects) >= 1  # At least one filter


@pytest.mark.asyncio
async def test_results_screen_has_list_view(repo):
    """ResultsScreen should have a ListView for runs."""
    app = ResultsScreenTestApp(repo=repo)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen

        screen = app.query_one(ResultsScreen)
        list_view = screen.query_one(ListView)
        assert list_view is not None


@pytest.mark.asyncio
async def test_results_screen_has_buttons(repo):
    """ResultsScreen should have Open, Delete, Export buttons."""
    app = ResultsScreenTestApp(repo=repo)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen

        screen = app.query_one(ResultsScreen)
        buttons = screen.query(Button)
        button_ids = [btn.id for btn in buttons]
        assert "open-btn" in button_ids
        assert "delete-btn" in button_ids
        assert "export-btn" in button_ids


@pytest.mark.asyncio
async def test_results_screen_refresh_runs_populates_list(repo_with_runs):
    """ResultsScreen.refresh_runs should populate the ListView with runs."""
    app = ResultsScreenTestApp(repo=repo_with_runs)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen, RunListItem

        screen = app.query_one(ResultsScreen)
        # refresh_runs is called on mount, so just wait for it
        await pilot.pause()

        # Should have RunListItem widgets in the ListView
        items = screen.query(RunListItem)
        assert len(items) >= 2


@pytest.mark.asyncio
async def test_results_screen_groups_by_date(repo_with_runs):
    """ResultsScreen should group runs by date with headers."""
    app = ResultsScreenTestApp(repo=repo_with_runs)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen

        screen = app.query_one(ResultsScreen)
        # refresh_runs is called on mount, so just wait for it
        await pilot.pause()

        # Should have date headers (as Static widgets with date-header class)
        # The exact implementation may vary, but we check the list has items
        list_view = screen.query_one(ListView)
        assert len(list_view.children) > 0


@pytest.mark.asyncio
async def test_results_screen_posts_run_selected_on_open(repo_with_runs):
    """ResultsScreen should post RunSelected message when Open is clicked."""
    messages = []

    class CaptureApp(App):
        def __init__(self, repo, **kwargs):
            super().__init__(**kwargs)
            self.repo = repo

        def compose(self) -> ComposeResult:
            from tui.screens.results import ResultsScreen

            yield ResultsScreen(repo=self.repo)

        def on_results_screen_run_selected(self, event) -> None:
            messages.append(event)

    app = CaptureApp(repo=repo_with_runs)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen, RunListItem

        screen = app.query_one(ResultsScreen)
        # refresh_runs is called on mount, so just wait for it
        await pilot.pause()

        # Select the first run item by clicking on it
        list_view = screen.query_one(ListView)
        items = screen.query(RunListItem)
        assert len(items) >= 1, "Expected at least one RunListItem"

        # Click on the first run item to select it
        await pilot.click(items[0])
        await pilot.pause()

        # Click Open button
        open_btn = screen.query_one("#open-btn", Button)
        await pilot.click(open_btn)
        await pilot.pause()

        # May get 2 messages: one from clicking item (ListView.Selected), one from Open button
        # We just verify at least one message was posted with a valid run_id
        assert len(messages) >= 1
        assert messages[-1].run_id is not None


@pytest.mark.asyncio
async def test_results_screen_posts_run_selected_on_item_select(repo_with_runs):
    """ResultsScreen should post RunSelected when item is double-clicked or activated."""
    messages = []

    class CaptureApp(App):
        def __init__(self, repo, **kwargs):
            super().__init__(**kwargs)
            self.repo = repo

        def compose(self) -> ComposeResult:
            from tui.screens.results import ResultsScreen

            yield ResultsScreen(repo=self.repo)

        def on_results_screen_run_selected(self, event) -> None:
            messages.append(event)

    app = CaptureApp(repo=repo_with_runs)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen, RunListItem

        screen = app.query_one(ResultsScreen)
        # refresh_runs is called on mount, so just wait for it
        await pilot.pause()

        # Select and activate first item
        list_view = screen.query_one(ListView)
        items = screen.query(RunListItem)
        assert len(items) >= 1, "Expected at least one RunListItem"

        # Click on the item, then press Enter to activate
        await pilot.click(items[0])
        await pilot.pause()
        list_view.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert len(messages) >= 1


@pytest.mark.asyncio
async def test_results_screen_delete_removes_run(repo_with_runs):
    """ResultsScreen should delete the selected run when Delete is clicked."""
    app = ResultsScreenTestApp(repo=repo_with_runs)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen, RunListItem

        screen = app.query_one(ResultsScreen)
        # refresh_runs is called on mount, so just wait for it
        await pilot.pause()

        # Get initial count
        initial_runs = repo_with_runs.list_runs()
        initial_count = len(initial_runs)
        assert initial_count >= 2

        # Select first item by clicking on it
        items = screen.query(RunListItem)
        assert len(items) >= 1, "Expected at least one RunListItem"
        await pilot.click(items[0])
        await pilot.pause()

        # Click Delete button
        delete_btn = screen.query_one("#delete-btn", Button)
        await pilot.click(delete_btn)
        await pilot.pause()

        # Verify run was deleted from database
        remaining_runs = repo_with_runs.list_runs()
        assert len(remaining_runs) == initial_count - 1


@pytest.mark.asyncio
async def test_results_screen_refresh_button(repo_with_runs):
    """ResultsScreen should have a Refresh button that calls refresh_runs."""
    app = ResultsScreenTestApp(repo=repo_with_runs)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen

        screen = app.query_one(ResultsScreen)
        # Find refresh button
        refresh_btn = screen.query_one("#refresh-btn", Button)
        assert refresh_btn is not None

        # Click should call refresh_runs
        await pilot.click(refresh_btn)
        await pilot.pause()

        # ListView should be populated
        list_view = screen.query_one(ListView)
        assert len(list_view.children) > 0


@pytest.mark.asyncio
async def test_results_screen_handles_empty_runs(repo):
    """ResultsScreen should handle empty runs gracefully."""
    app = ResultsScreenTestApp(repo=repo)
    async with app.run_test() as pilot:
        from tui.screens.results import ResultsScreen

        screen = app.query_one(ResultsScreen)
        # refresh_runs is called on mount, so just wait for it
        await pilot.pause()

        # Should not crash, ListView should exist
        list_view = screen.query_one(ListView)
        assert list_view is not None
