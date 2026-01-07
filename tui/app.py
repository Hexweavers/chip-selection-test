"""Main Textual application."""

import asyncio
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static
from textual.binding import Binding

from db.schema import init_db
from db.repository import Repository
from services.runner_async import AsyncRunner, RunConfig, EventType
from tui.screens.configure import ConfigureScreen
from tui.screens.monitor import MonitorScreen


class ChipBenchmarkApp(App):
    """Chip benchmark TUI application."""

    TITLE = "Chip Benchmark"
    CSS = """
    Screen {
        background: $surface;
    }
    TabbedContent {
        height: 100%;
    }
    TabPane {
        padding: 1;
    }
    .placeholder {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
        Binding("1", "switch_tab('configure')", "Configure", show=False),
        Binding("2", "switch_tab('monitor')", "Monitor", show=False),
        Binding("3", "switch_tab('results')", "Results", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.runner = AsyncRunner()
        self.db_conn = init_db()
        self.repo = Repository(self.db_conn)
        self._current_run_id: str | None = None
        self._run_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="configure"):
            with TabPane("Configure", id="configure"):
                yield ConfigureScreen()
            with TabPane("Monitor", id="monitor"):
                yield MonitorScreen()
            with TabPane("Results", id="results"):
                yield Static("Results screen placeholder", classes="placeholder")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        self.query_one(TabbedContent).active = tab_id

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press 1/2/3 to switch tabs, q to quit")

    def on_configure_screen_run_requested(
        self, event: ConfigureScreen.RunRequested
    ) -> None:
        """Handle run request from configure screen."""
        mode = "Dry run" if event.dry_run else "Run"
        self.notify(
            f"{mode}: {len(event.models)} model(s), persona={event.persona_id}, "
            f"style={event.prompt_style}, flow={event.flow}, "
            f"constraint={event.constraint_type}, chips={event.chip_count}"
        )

        # Create run name with timestamp
        run_name = f"Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Create run in database
        self._current_run_id = self.repo.create_run(
            name=run_name,
            persona=event.persona_id,
            prompt_style=event.prompt_style,
            flow=event.flow,
            constraint_type=event.constraint_type,
            chip_count=event.chip_count,
        )

        # Build config summary
        config_str = (
            f"Persona: {event.persona_id} | Style: {event.prompt_style} | "
            f"Flow: {event.flow} | Constraint: {event.constraint_type} | "
            f"Chips: {event.chip_count}"
        )

        # Setup monitor screen
        monitor = self.query_one(MonitorScreen)
        monitor.setup_run(
            models=event.models,
            run_name=run_name,
            config_str=config_str,
        )

        # Switch to monitor tab
        self.query_one(TabbedContent).active = "monitor"

        # Start the test run (unless dry run)
        if not event.dry_run:
            config = RunConfig(
                models=event.models,
                persona_id=event.persona_id,
                prompt_style=event.prompt_style,
                flow=event.flow,
                constraint_type=event.constraint_type,
                chip_count=event.chip_count,
            )
            self._run_task = asyncio.create_task(self._run_test(config))

    async def _run_test(self, config: RunConfig) -> None:
        """Run the test and update the monitor screen with events."""
        monitor = self.query_one(MonitorScreen)
        model_stats: dict[str, dict] = {
            model: {"tokens_in": 0, "tokens_out": 0, "cost": 0.0}
            for model in config.models
        }

        try:
            async for event in self.runner.run(config):
                if event.type == EventType.LOG:
                    monitor.add_log(event.model, event.message)

                elif event.type == EventType.PROGRESS:
                    percent = event.data.get("percent", 0)
                    monitor.update_progress(event.model, percent, event.message)

                elif event.type == EventType.COMPLETE:
                    # Update stats from result data
                    result_data = event.data.get("result", {})
                    tokens = result_data.get("tokens_in", 0) + result_data.get(
                        "tokens_out", 0
                    )
                    cost = result_data.get("cost_usd", 0.0)
                    monitor.update_stats(event.model, tokens, cost)

                    # Save result to database
                    if self._current_run_id:
                        counts = result_data.get("counts", {})
                        self.repo.create_result(
                            run_id=self._current_run_id,
                            model=event.model,
                            chips=[],  # Would be populated from result
                            tokens_in=result_data.get("tokens_in", 0),
                            tokens_out=result_data.get("tokens_out", 0),
                            cost_usd=cost,
                            latency_ms=result_data.get("latency_ms", 0),
                            situation_count=counts.get("situation", 0),
                            jargon_count=counts.get("jargon", 0),
                            role_task_count=counts.get("role_task", 0),
                            environment_count=counts.get("environment", 0),
                        )

                    monitor.add_log(event.model, f"Completed: {event.message}")

                elif event.type == EventType.ERROR:
                    monitor.add_log(event.model, f"[red]Error: {event.message}[/red]")

        except asyncio.CancelledError:
            monitor.add_log("system", "Run cancelled by user")
        except Exception as e:
            monitor.add_log("system", f"[red]Unexpected error: {str(e)}[/red]")
        finally:
            monitor.mark_complete()
            self._run_task = None

    def on_monitor_screen_cancel_requested(
        self, event: MonitorScreen.CancelRequested
    ) -> None:
        """Handle cancel request from monitor screen."""
        self.runner.cancel()
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()

    def on_monitor_screen_view_results_requested(
        self, event: MonitorScreen.ViewResultsRequested
    ) -> None:
        """Handle view results request from monitor screen."""
        self.query_one(TabbedContent).active = "results"


def run():
    """Run the application."""
    app = ChipBenchmarkApp()
    app.run()


if __name__ == "__main__":
    run()
