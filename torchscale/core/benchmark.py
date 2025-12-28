"""Benchmark execution module for running distributed training benchmarks."""

import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model: str
    batch_size: int
    gpu_count: int
    avg_throughput: float
    iteration_latency: float
    throughput_samples: List[float]
    latency_samples: List[float]
    variance_detected: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkRunner:
    """Runner for distributed training benchmarks."""

    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self, model: str, batch_size: int, gpu_count: int
    ) -> BenchmarkResult:
        """
        Run a single benchmark configuration.

        Args:
            model: Model name (e.g., 'resnet50', 'bert-large')
            batch_size: Batch size per GPU
            gpu_count: Number of GPUs to use

        Returns:
            BenchmarkResult with performance metrics
        """
        if self.verbose:
            print(f"> Running {model} on {gpu_count} GPUs (Batch {batch_size})...")

        # Simulate benchmark execution
        # In a real implementation, this would call torchrun with the training script
        start_time = time.time()

        # Simulate some computation
        time.sleep(0.5)  # Simulate benchmark execution time

        # Generate simulated results based on GPU scaling
        # Real implementation would parse actual training logs
        base_throughput = 400 * gpu_count  # Base scaling
        scaling_efficiency = 0.85 if gpu_count > 4 else 0.95  # Simulate scaling loss
        avg_throughput = base_throughput * scaling_efficiency

        # Latency decreases with more GPUs (ideally)
        iteration_latency = (batch_size * gpu_count) / avg_throughput * 1000  # ms

        # Generate sample data with some variance
        throughput_samples = [
            avg_throughput * (0.95 + 0.1 * (i % 10) / 10) for i in range(10)
        ]
        latency_samples = [
            iteration_latency * (0.95 + 0.1 * (i % 10) / 10) for i in range(10)
        ]

        # Detect high variance in gradient sync
        variance = statistics.stdev(latency_samples) / iteration_latency
        variance_detected = variance > 0.15

        result = BenchmarkResult(
            model=model,
            batch_size=batch_size,
            gpu_count=gpu_count,
            avg_throughput=avg_throughput,
            iteration_latency=iteration_latency,
            throughput_samples=throughput_samples,
            latency_samples=latency_samples,
            variance_detected=variance_detected,
        )

        if self.verbose:
            print(f"> [INFO] Average Throughput: {avg_throughput:,.0f} img/sec")
            print(f"> [INFO] Iteration Latency: {iteration_latency:.0f}ms")
            if variance_detected:
                print("> [WARN] High variance in gradient sync time detected.")

        self.results.append(result)
        return result

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save all benchmark results to JSON file."""
        output_file = self.output_dir / filename
        results_dict = [r.to_dict() for r in self.results]

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        if self.verbose:
            print(f"\n> Results saved to {output_file}")

    def run_torchrun(
        self, script_path: Path, gpu_count: int, args: List[str]
    ) -> Tuple[int, str, str]:
        """
        Execute a training script using torchrun.

        Args:
            script_path: Path to the training script
            gpu_count: Number of GPUs to use
            args: Additional arguments for the script

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        cmd = ["torchrun", f"--nproc_per_node={gpu_count}", str(script_path)] + args

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Benchmark timed out after 5 minutes"
        except Exception as e:
            return -1, "", str(e)
