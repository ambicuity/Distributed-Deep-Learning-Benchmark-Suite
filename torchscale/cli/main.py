"""Main CLI application for TorchScale."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from torchscale.core.config import BenchmarkConfig
from torchscale.core.benchmark import BenchmarkRunner
from torchscale.profiling.nsight import ProfilerRunner
from torchscale.reporting.generator import ReportGenerator
from torchscale.utils.validation import SystemValidator

app = typer.Typer(
    name="torchscale",
    help="Benchmarking CLI for PyTorch DDP Clusters.",
    add_completion=False
)

console = Console()

# Create sub-apps for benchmark and report
benchmark_app = typer.Typer(help="Run throughput/latency tests across GPU configs.")
report_app = typer.Typer(help="Generate visual HTML/PDF performance reports.")

app.add_typer(benchmark_app, name="benchmark")
app.add_typer(report_app, name="report")


@benchmark_app.command("run")
def benchmark_run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to benchmark configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        "./results",
        "--output-dir",
        "-o",
        help="Directory to save benchmark results",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Run throughput/latency tests across GPU configs.
    
    This command executes benchmark tests based on the configuration file,
    measuring throughput (samples/sec) and latency (ms/iteration) for
    different model, batch size, and GPU count combinations.
    """
    try:
        # Load configuration
        if verbose:
            console.print(f"[bold blue]Loading configuration from {config}...[/bold blue]")
        
        benchmark_config = BenchmarkConfig.from_yaml(config)
        
        if verbose:
            console.print(f"[green]Experiment: {benchmark_config.experiment_name}[/green]")
            console.print(f"[green]Models: {', '.join(benchmark_config.models)}[/green]")
            console.print(f"[green]Batch sizes: {benchmark_config.batch_sizes}[/green]")
            console.print(f"[green]GPU counts: {benchmark_config.gpu_counts}[/green]")
        
        # Create benchmark runner
        runner = BenchmarkRunner(output_dir=output_dir, verbose=verbose)
        
        # Run benchmarks for all combinations
        total_runs = (
            len(benchmark_config.models) *
            len(benchmark_config.batch_sizes) *
            len(benchmark_config.gpu_counts)
        )
        
        console.print(f"\n[bold]Running {total_runs} benchmark configurations...[/bold]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total_runs)
            
            for model in benchmark_config.models:
                for batch_size in benchmark_config.batch_sizes:
                    for gpu_count in benchmark_config.gpu_counts:
                        progress.update(
                            task,
                            description=f"Running {model} (batch={batch_size}, gpus={gpu_count})"
                        )
                        
                        result = runner.run_benchmark(model, batch_size, gpu_count)
                        
                        progress.advance(task)
        
        # Save results
        runner.save_results()
        
        console.print("\n[bold green]✓ Benchmark complete![/bold green]")
        console.print(f"Results saved to: {output_dir / 'benchmark_results.json'}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def profile(
    gpus: int = typer.Option(
        4,
        "--gpus",
        "-g",
        help="Number of GPUs to profile",
    ),
    duration: int = typer.Option(
        30,
        "--duration",
        "-d",
        help="Duration of profiling session in seconds",
    ),
    target: str = typer.Option(
        "sync_stalls",
        "--target",
        "-t",
        help="Profiling target (e.g., sync_stalls, kernels)",
    ),
    output_dir: Path = typer.Option(
        "./results",
        "--output-dir",
        "-o",
        help="Directory to save profiling results",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Wrap execution in Nsight Systems to find bottlenecks.
    
    This command profiles the training execution using NVIDIA Nsight Systems,
    capturing CUDA kernel timelines and NCCL communication patterns to identify
    synchronization bottlenecks and performance issues.
    """
    try:
        console.print("[bold blue]Starting profiling session...[/bold blue]\n")
        
        profiler = ProfilerRunner(output_dir=output_dir, verbose=verbose)
        
        result = profiler.run_profile(
            gpu_count=gpus,
            duration=duration,
            target=target
        )
        
        console.print("\n[bold green]✓ Profiling complete![/bold green]")
        
        if result.bottlenecks:
            console.print(f"\n[bold yellow]Found {len(result.bottlenecks)} bottleneck(s):[/bold yellow]")
            for i, bottleneck in enumerate(result.bottlenecks, 1):
                console.print(f"\n{i}. [red]{bottleneck['type']}[/red]")
                console.print(f"   {bottleneck['description']}")
                console.print(f"   Impact: {bottleneck['impact']}")
                console.print(f"   [cyan]→ {bottleneck['suggestion']}[/cyan]")
        
        if result.report_file:
            console.print(f"\nDetailed report saved to: {result.report_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@report_app.command("generate")
def report_generate(
    source: Path = typer.Option(
        "./results",
        "--source",
        "-s",
        help="Directory containing benchmark results",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Output format (html or pdf)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <source>/report.<format>)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Generate visual HTML/PDF performance reports.
    
    This command aggregates benchmark and profiling results to create
    a comprehensive performance report with scaling efficiency plots,
    latency histograms, and bottleneck analysis.
    """
    try:
        console.print("[bold blue]Generating performance report...[/bold blue]\n")
        
        generator = ReportGenerator(source_dir=source, verbose=verbose)
        
        # Determine output file
        if output is None:
            output = source / f"report.{format}"
        
        # Generate report based on format
        if format.lower() == "html":
            report_file = generator.generate_html_report(output)
        elif format.lower() == "pdf":
            report_file = generator.generate_pdf_report(output)
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            raise typer.Exit(code=1)
        
        console.print(f"[bold green]✓ Report generated successfully![/bold green]")
        console.print(f"Report saved to: {report_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def validate(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Check if current node meets DDP requirements (NCCL, Drivers).
    
    This command validates that all necessary components are installed
    and properly configured for running distributed deep learning benchmarks,
    including PyTorch, CUDA, NCCL, and NVIDIA drivers.
    """
    try:
        validator = SystemValidator(verbose=verbose)
        checks = validator.validate_all()
        all_passed = validator.print_validation_results(checks)
        
        if not all_passed:
            raise typer.Exit(code=1)
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Disable all output except errors.",
    ),
):
    """
    TorchScale: Benchmarking CLI for PyTorch DDP Clusters.
    
    Automate multi-GPU benchmarks and profiling to quantify scaling efficiency,
    identify synchronization bottlenecks, and generate actionable reports for
    optimizing training throughput on GPU clusters.
    """
    if quiet:
        console.quiet = True


if __name__ == "__main__":
    app()
