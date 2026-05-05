from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cartograph import __version__
from cartograph.agents.common import observe_decisions
from cartograph.agents.root import run_audit_sync
from cartograph.core import storage
from cartograph.core.models import Decision
from cartograph.demo.deep_dive import run_customer_service_deep_dive
from cartograph.demo.prepare import prepare_fleet_benchmark
from cartograph.demo.visualize import generate_visualizations

app = typer.Typer(no_args_is_help=True)
demo_app = typer.Typer(no_args_is_help=True)
corpus_app = typer.Typer(no_args_is_help=True)
app.add_typer(demo_app, name="demo")
app.add_typer(corpus_app, name="corpus")
console = Console()


@app.command()
def version() -> None:
    console.print(__version__)


@app.command()
def audit(
    target: str = typer.Option(..., "--target"),
    corpus: str = typer.Option("bitext", "--corpus"),
    eval_path: str = typer.Option(..., "--eval-path"),
    coverage_threshold: float = typer.Option(0.75, "--coverage-threshold"),
    corpus_limit: int = typer.Option(1000, "--corpus-limit"),
) -> None:
    console.print("decision_log")

    def print_decision(_audit_id: str, decision: Decision) -> None:
        console.print(f"{decision.step}. {decision.action}: {decision.reason}")

    with observe_decisions(print_decision):
        audit_id = run_audit_sync(target, corpus, eval_path, coverage_threshold, corpus_limit)
    report = storage.load_coverage_report(audit_id)
    console.print(f"audit_id: {audit_id}")
    console.print(f"coverage_score: {report.coverage_score:.2f}")


@app.command()
def report(
    audit_id: str | None = typer.Argument(None), latest: bool = typer.Option(False, "--latest")
) -> None:
    if latest or audit_id is None:
        audit_id = storage.latest_audit_id()
    if audit_id is None:
        raise typer.BadParameter("no audit_id provided and no latest audit exists")
    payload = storage.load_coverage_report(audit_id).model_dump(mode="json")
    console.print(json.dumps(payload, indent=2))


@demo_app.command("prepare")
def demo_prepare() -> None:
    payload = prepare_fleet_benchmark()
    table = Table("Agent", "Coverage", "Mode")
    for agent in payload["agents"]:
        table.add_row(agent["target_agent"], f"{agent['coverage_score']:.2f}", agent["corpus_mode"])
    console.print(table)


@demo_app.command("run")
def demo_run(agent: str = typer.Argument("customer-service")) -> None:
    if agent != "customer-service":
        raise typer.BadParameter("v1 live deep-dive only supports customer-service")
    run_customer_service_deep_dive()


@demo_app.command("visualize")
def demo_visualize(audit_id: str) -> None:
    outputs = generate_visualizations(audit_id)
    for output in outputs:
        console.print(str(output))


@corpus_app.command("list")
def corpus_list() -> None:
    console.print("bitext")
    console.print("github")
    console.print("stackexchange")


def write_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    app()
