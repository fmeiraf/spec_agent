"""Rich UI components for terminal output."""

from typing import Optional

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from spec_agent.spec import SubTask

console = Console()


class SchedulerUI:
    """UI components for the scheduler."""

    # Persona icons/emojis
    WORKER_ICON = "👷"
    SUPERVISOR_ICON = "👔"
    TASK_ICON = "📋"
    REVIEW_ICON = "🔍"
    PENDING_ICON = "⏳"
    COMPLETE_ICON = "✅"
    ASSIGN_ICON = "🚀"
    LOADING_ICON = "⚙️"
    INITIALIZING_ICON = "🚀"

    @staticmethod
    def worker_card(task_id: str, task_type: str) -> Panel:
        """Create a worker assignment card."""
        content = Text()
        content.append(f"{SchedulerUI.WORKER_ICON} ", style="bold yellow")
        content.append("Task ", style="bold white")
        content.append(f"{task_id}", style="bold cyan")
        content.append(f"\n{SchedulerUI.TASK_ICON} ", style="dim")
        content.append(f"{task_type}", style="dim italic")
        return Panel(
            Align.left(content),
            title=f"{SchedulerUI.ASSIGN_ICON} Worker Assignment",
            border_style="bright_cyan",
            padding=(1, 2),
            box=ROUNDED,
        )

    @staticmethod
    def worker_complete_card(task_id: str, task_type: str) -> Panel:
        """Create a worker completion card."""
        content = Text()
        content.append(f"{SchedulerUI.COMPLETE_ICON} ", style="bold green")
        content.append("Task ", style="bold white")
        content.append(f"{task_id}", style="bold cyan")
        content.append(" completed", style="green")
        content.append(f"\n{SchedulerUI.TASK_ICON} ", style="dim")
        content.append(f"{task_type}", style="dim italic")
        content.append(f"\n{SchedulerUI.PENDING_ICON} ", style="yellow")
        content.append("Waiting for review", style="yellow")
        return Panel(
            Align.left(content),
            title=f"{SchedulerUI.WORKER_ICON} Worker Complete",
            border_style="green",
            padding=(1, 2),
            box=ROUNDED,
        )

    @staticmethod
    def review_card(task_id: str) -> Panel:
        """Create a review card."""
        content = Text()
        content.append(f"{SchedulerUI.REVIEW_ICON} ", style="bold yellow")
        content.append("Reviewing task ", style="bold white")
        content.append(f"{task_id}", style="bold yellow")
        return Panel(
            Align.left(content),
            title=f"{SchedulerUI.SUPERVISOR_ICON} Supervisor Review",
            border_style="yellow",
            padding=(1, 2),
            box=ROUNDED,
        )

    @staticmethod
    def review_complete_card(task_id: str, followup_count: Optional[int] = None) -> Panel:
        """Create a review complete card."""
        content = Text()
        content.append(f"{SchedulerUI.COMPLETE_ICON} ", style="bold green")
        content.append("Task ", style="bold white")
        content.append(f"{task_id}", style="bold green")
        content.append(" reviewed", style="green")

        if followup_count is not None and followup_count > 0:
            content.append("\n🚀 ", style="bold magenta")
            content.append(f"{followup_count} ", style="bold magenta")
            content.append("follow-up task(s) created", style="magenta")
        else:
            content.append("\n✨ ", style="dim")
            content.append("No follow-ups needed", style="dim")

        return Panel(
            Align.left(content),
            title=f"{SchedulerUI.SUPERVISOR_ICON} Review Complete",
            border_style="green",
            padding=(1, 2),
            box=ROUNDED,
        )

    @staticmethod
    def pending_reviews_card(pending_reviews: dict[str, SubTask]) -> Panel:
        """Create a pending reviews card with table."""
        if not pending_reviews:
            content = Text()
            content.append(f"{SchedulerUI.PENDING_ICON} ", style="dim")
            content.append("No tasks currently waiting for review", style="dim italic")
            return Panel(
                Align.center(content),
                title=f"{SchedulerUI.PENDING_ICON} Pending Reviews",
                border_style="blue",
                padding=(1, 2),
                box=ROUNDED,
            )

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
            show_lines=False,
        )
        table.add_column("Task ID", style="dim", width=12)
        table.add_column("Type", style="magenta", width=20)
        table.add_column("Status", style="yellow", width=15)

        for task in pending_reviews.values():
            table.add_row(str(task.id), task.subitem.type, task.status)

        header = Text()
        header.append(f"{SchedulerUI.PENDING_ICON} ", style="bold blue")
        header.append(f"{len(pending_reviews)} task(s) waiting", style="bold blue")

        content = Group(header, table)

        return Panel(
            content,
            title=f"{SchedulerUI.PENDING_ICON} Tasks Waiting for Review",
            border_style="blue",
            padding=(1, 2),
            box=ROUNDED,
        )

    @staticmethod
    def initializing_status() -> Status:
        """Create a loading status for initial assignment."""
        return Status(
            f"{SchedulerUI.SUPERVISOR_ICON} Initializing tasks...",
            console=console,
            spinner="dots",
            spinner_style="bold cyan",
        )

    @staticmethod
    def initialization_complete_card(task_count: int) -> Panel:
        """Create a card showing initialization complete."""
        content = Text()
        content.append(f"{SchedulerUI.COMPLETE_ICON} ", style="bold green")
        content.append("Initialization complete", style="bold white")
        content.append(f"\n{SchedulerUI.TASK_ICON} ", style="dim")
        content.append(f"{task_count} ", style="bold cyan")
        if task_count == 1:
            content.append("task created", style="cyan")
        else:
            content.append("tasks created", style="cyan")
        return Panel(
            Align.left(content),
            title=f"{SchedulerUI.SUPERVISOR_ICON} Ready to Start",
            border_style="green",
            padding=(1, 2),
            box=ROUNDED,
        )
