import click
import yaml
import requests
import sys
import os
from typing import Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def load_policy(path: str) -> Dict:
    """Load mlguard.yaml policy."""
    if not os.path.exists(path):
        console.print(f"[bold red]Error:[/bold red] Policy file '{path}' not found.")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """ML Guard CLI — Enterprise AI Governance & CI/CD Tool."""
    pass

@cli.command()
@click.option('--policy', default='mlguard.yaml', help='Path to mlguard.yaml')
@click.option('--artifact', help='Path to model artifact (.pkl, .joblib, etc.)')
@click.option('--url', help='Inference endpoint URL to probe')
@click.option('--api-url', default='http://localhost:8000/api/v1/gate/evaluate', help='ML Guard API Gateway URL')
def check(policy, artifact, url, api_url):
    """Run a governance gate check against a policy."""
    if not artifact and not url:
        console.print("[bold red]Error:[/bold red] Must provide either --artifact or --url")
        sys.exit(1)

    policy_data = load_policy(policy)
    
    payload = {
        "artifact_path": os.path.abspath(artifact) if artifact else None,
        "inference_url": url,
        "policy": policy_data
    }

    console.print(Panel.fit(
        f"[bold blue]ML Guard v7.2 CI/CD Gate[/bold blue]\n"
        f"Model: {policy_data.get('model_name', 'Unknown')}\n"
        f"Target: {artifact or url}",
        border_style="blue"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Evaluating governance criteria...", total=None)
        try:
            response = requests.post(api_url, json=payload, timeout=65)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            console.print(f"[bold red]API Error:[/bold red] {str(e)}")
            sys.exit(1)

    # Display Results
    status_color = "green" if result["passed"] else "red"
    console.print(f"\n[bold {status_color}]Gate Status: {result['gate_status']}[/bold {status_color}]")
    console.print(f"Score: [bold]{result['score']}/100[/bold]\n")

    table = Table(title="Governance Signal Breakdown", show_header=True, header_style="bold magenta")
    table.add_column("Signal", style="dim")
    table.add_column("Status")
    table.add_column("Message")

    # In a real scenario, the API would return refined check details. 
    # Here we use the failures and details from the response.
    if result["failures"]:
        for failure in result["failures"]:
            table.add_row("Policy Check", "[bold red]FAILED[/bold red]", failure)
    else:
        table.add_row("Governance Signals", "[bold green]PASSED[/bold green]", "All signals within policy thresholds.")

    console.print(table)

    if not result["passed"]:
        console.print(f"\n[bold red]✖ GATE BLOCKED:[/bold red] Model failed governance criteria. Merge prohibited.")
        console.print(f"See Report Card: {result['badge_url']}")
        sys.exit(1)
    else:
        console.print(f"\n[bold green]✔ GATE PASSED:[/bold green] Model meets all compliance standards.")
        console.print(f"Badge: {result['badge_url']}")
        sys.exit(0)

if __name__ == "__main__":
    cli()
