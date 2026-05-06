from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from cartograph import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def version() -> None:
    """Print the Kartograf version."""
    console.print(__version__)


@app.command()
def map(
    target: Path = typer.Option(
        ...,
        "--target",
        help="Path to a Python agent repo",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("kartograf_map.html"),
        "--output",
        help="Where to write the interactive HTML report",
    ),
    alpha: float | None = typer.Option(
        None, "--alpha", help="Override mass-balance α (default: |S|/|E|)"
    ),
    k: int = typer.Option(5, "--k", help="k nearest surfaces per gap region"),
) -> None:
    """Scan a repo's surfaces, plot the semantic map, find gaps, propose BDD eval cases."""
    from cartograph.pipeline import audit_repo

    summary = audit_repo(str(target), str(output), alpha=alpha, k_neighbors=k)
    console.print(f"[bold]coverage:[/bold] {summary['coverage']:.3f}")
    console.print(
        f"surfaces: {summary['surface_count']}, evals: {summary['eval_count']}, "
        f"gap regions: {summary['region_count']}"
    )
    console.print(f"[dim]html:[/dim] {summary['html']}")
    for region in summary["regions"][:3]:
        console.print(f"  {region['id']}: mass={region['mass']:.3f}")


if __name__ == "__main__":
    app()
