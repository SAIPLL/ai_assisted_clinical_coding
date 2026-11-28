import time
from contextlib import contextmanager

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


@contextmanager
def step(description):
    """Show a spinner with the elapsed time while a step runs, then a check mark"""
    start = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}..."),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        yield
    console.print(
        f"[green]✓[/green] {description} ({time.perf_counter() - start:.0f}s)"
    )
