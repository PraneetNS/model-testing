import typer
import joblib
import pandas as pd
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

# Set up path to allow imports from app
sys.path.append(str(Path(__file__).parent.parent))

from app.domain.services.orchestrator import TestOrchestrator
from app.domain.services.nlp_parser import NLPParser
from app.domain.services.ml_testing.framework.reporters import TestReporter

app = typer.Typer(help="ML Guard Scriptless Testing CLI")

async def run_audit(model_path, train_path, val_path, intent, html_out):
    typer.echo(f"🚀 Initializing ML Guard Audit for intent: '{intent}'...")
    
    # Load artifacts
    try:
        model = joblib.load(model_path)
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    except Exception as e:
        typer.secho(f"❌ Error loading artifacts: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    orchestrator = TestOrchestrator()
    parser = NLPParser()
    
    categories = parser.parse_query(intent)
    
    datasets = {
        "training": train_df,
        "validation": val_df
    }

    result = await orchestrator.run_test_suite(
        project_id="cli-manual-run",
        model_version="1.0.0",
        test_suite_name=f"CLI Audit: {intent}",
        model_artifact=model,
        datasets=datasets,
        categories=categories
    )

    # Print results to console
    typer.echo("\n" + "="*50)
    typer.echo(f"AUDIT SUMMARY: {result.test_suite}")
    typer.echo("="*50)
    
    status_color = typer.colors.GREEN if result.deployment_allowed else typer.colors.RED
    typer.secho(f"QUALITY INDEX: {result.score}%", bold=True)
    typer.secho(f"DEPLOYMENT ALLOWED: {result.deployment_allowed}", fg=status_color, bold=True)
    
    typer.echo("\nTEST DETAILS:")
    for r in result.results:
        symbol = "✅" if r.status == "passed" else "❌"
        typer.echo(f"{symbol} [{r.severity.upper()}] {r.test_name}: {r.message}")

    # Generate Report
    # Note: TestOrchestrator currently returns QualityGateResult, 
    # but the runner inside it generates the MLTestReport.
    # For the CLI, we can re-wrap or modify orchestrator to return the report too.
    # For now, let's just echo success.
    
    typer.echo("\n" + "="*50)
    if not result.deployment_allowed:
        typer.secho("Audit failed due to critical test failures or low quality index.", fg=typer.colors.RED)
        sys.exit(1)

@app.command()
def test(
    model: str = typer.Option(..., help="Path to model .pkl file"),
    train: str = typer.Option(..., help="Path to training .csv file"),
    val: str = typer.Option(..., help="Path to validation .csv file"),
    intent: str = typer.Option("Run full suite", help="Natural language testing intent"),
    html: str = typer.Option("report.html", help="Path to save HTML report")
):
    """Run a structured ML audit via CLI."""
    asyncio.run(run_audit(model, train, val, intent, html))

if __name__ == "__main__":
    app()
