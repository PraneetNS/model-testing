
import typer
from rich.console import Console
from rich.table import Table
import asyncio
import sys
import uuid
import os
from typing import List, Dict

# Hack to find backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.domain.services.orchestrator import TestOrchestrator
from app.domain.models.test_suite import TestSuite, TestConfig
from app.domain.services.report_generator import ReportGenerator

app = typer.Typer()
console = Console()

@app.command()
def scan(
    model_path: str = typer.Argument(..., help="Path to the model file (.pkl, .onnx)"),
    suite: str = typer.Option("default", help="Test suite name"),
    html: bool = typer.Option(True, help="Generate HTML report")
):
    """
    Run a production-readiness scan on a model artifact.
    """
    console.print(f"[bold blue]🛡️ ML Guard - Quality Gate[/bold blue]")
    console.print(f"Scanning model: [yellow]{model_path}[/yellow]")
    console.print(f"Test Suite: [cyan]{suite}[/cyan]")
    
    # Define a default suite (mock)
    default_tests = [
        TestConfig(name="Drift Check (PSI)", category="statistical_stability", type="psi_drift", severity="high", config={"threshold": 0.2}),
        TestConfig(name="Missing Values", category="data_quality", type="missing_values", severity="critical", config={"threshold": 0.05}),
        TestConfig(name="Bias Check (Gender)", category="bias_fairness", type="bias_check", severity="medium", config={"attr": "gender"}),
        TestConfig(name="Robustness Test", category="robustness", type="noise_injection", severity="low", config={})
    ]
    
    test_suite = TestSuite(id="default_suite", name="Production Readiness", description="Standard gate", tests=default_tests)
    
    # Run Orchestrator
    orchestrator = TestOrchestrator()
    
    with console.status("[bold green]Running tests...[/bold green]"):
        # We use asyncio.run because typer is synchronous
        result = asyncio.run(orchestrator.run_suite(test_suite, model_path, "", ""))
    
    # Display Results
    table = Table(title=f"Scan Results (Score: {result.overall_score:.1f})")
    table.add_column("Category", style="cyan")
    table.add_column("Test", style="white")
    table.add_column("Status", style="magenta")
    table.add_column("Message", style="dim")
    
    for r in result.results:
        status_style = "green" if r.status == "passed" else "red"
        table.add_row(
            r.category, 
            r.test_name, 
            f"[{status_style}]{r.status.upper()}[/{status_style}]", 
            r.message
        )
        
    console.print(table)
    
    risk_color = "green" if result.risk_level == "Low" else "red"
    console.print(f"\nRisk Level: [{risk_color}]{result.risk_level}[/{risk_color}]")
    console.print(f"Deployment Allowed: [{'green' if result.deployment_allowed else 'red'}]{result.deployment_allowed}[/]")

    if html:
        report_gen = ReportGenerator()
        # Convert Pydantic models to dicts for the generator
        results_dicts = [r.model_dump() for r in result.results]
        output_file = f"report_{result.run_id}.html"
        report_gen.generate_html(result.run_id, results_dicts, output_file)
        console.print(f"\n[bold]📄 HTML Report generated:[/bold] {output_file}")

if __name__ == "__main__":
    app()
