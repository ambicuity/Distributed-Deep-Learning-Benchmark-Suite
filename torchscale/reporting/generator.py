"""Reporting module for generating performance reports."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class ReportGenerator:
    """Generator for performance analysis reports."""

    def __init__(self, source_dir: Path, verbose: bool = False):
        self.source_dir = source_dir
        self.verbose = verbose

    def load_benchmark_results(self) -> List[Dict]:
        """Load benchmark results from JSON files."""
        results = []
        result_files = list(self.source_dir.glob("benchmark_results*.json"))

        for result_file in result_files:
            with open(result_file, "r") as f:
                data = json.load(f)
                results.extend(data if isinstance(data, list) else [data])

        if self.verbose:
            print(f"> Loaded {len(results)} benchmark results")

        return results

    def load_profiling_results(self) -> List[Dict]:
        """Load profiling results from JSON files."""
        results = []
        profile_files = list(self.source_dir.glob("profile_*.json"))

        for profile_file in profile_files:
            with open(profile_file, "r") as f:
                data = json.load(f)
                results.append(data)

        if self.verbose:
            print(f"> Loaded {len(results)} profiling results")

        return results

    def calculate_scaling_efficiency(self, results: List[Dict]) -> pd.DataFrame:
        """Calculate scaling efficiency from benchmark results."""
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Calculate ideal throughput (perfect linear scaling)
        if "avg_throughput" in df.columns and "gpu_count" in df.columns:
            # Group by model and batch_size
            df["ideal_throughput"] = df.groupby(["model", "batch_size"])[
                "avg_throughput"
            ].transform(
                lambda x: x.iloc[0]
                * df.loc[x.index, "gpu_count"]
                / df.loc[x.index, "gpu_count"].iloc[0]
            )
            df["scaling_efficiency"] = (
                df["avg_throughput"] / df["ideal_throughput"]
            ) * 100

        return df

    def generate_html_report(self, output_file: Path) -> str:
        """Generate HTML performance report."""
        benchmark_results = self.load_benchmark_results()
        profiling_results = self.load_profiling_results()

        if not benchmark_results:
            if self.verbose:
                print("> [WARN] No benchmark results found")
            return ""

        df = self.calculate_scaling_efficiency(benchmark_results)

        html_content = self._create_html_content(df, profiling_results)

        with open(output_file, "w") as f:
            f.write(html_content)

        if self.verbose:
            print(f"> HTML report generated: {output_file}")

        return str(output_file)

    def _create_html_content(
        self, df: pd.DataFrame, profiling_results: List[Dict]
    ) -> str:
        """Create HTML content for the report."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>TorchScale Performance Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 4px;
            min-width: 200px;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }
        .warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }
        .bottleneck {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
        }
        .recommendation {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>TorchScale Performance Report</h1>
"""

        # Add summary metrics
        if not df.empty:
            html += "<h2>Summary Metrics</h2>\n"
            html += '<div class="metrics">\n'

            if "avg_throughput" in df.columns:
                max_throughput = df["avg_throughput"].max()
                html += f'<div class="metric"><div class="metric-label">Max Throughput</div>'
                html += f'<div class="metric-value">{max_throughput:,.0f} img/sec</div></div>\n'

            if "scaling_efficiency" in df.columns:
                avg_efficiency = df["scaling_efficiency"].mean()
                html += f'<div class="metric"><div class="metric-label">Avg Scaling Efficiency</div>'
                html += f'<div class="metric-value">{avg_efficiency:.1f}%</div></div>\n'

            html += "</div>\n"

            # Add benchmark results table
            html += "<h2>Benchmark Results</h2>\n"
            html += df[
                [
                    "model",
                    "batch_size",
                    "gpu_count",
                    "avg_throughput",
                    "iteration_latency",
                    "scaling_efficiency",
                ]
            ].to_html(index=False, float_format=lambda x: f"{x:.2f}", classes="table")

        # Add profiling results
        if profiling_results:
            html += "<h2>Profiling Analysis</h2>\n"

            for result in profiling_results:
                gpu_count = result.get("gpu_count", "N/A")
                sync_stall = result.get("sync_stall_percentage", 0)

                html += f'<div class="warning">'
                html += f"<strong>GPU Configuration: {gpu_count} GPUs</strong><br>"
                html += f"Synchronization stall time: {sync_stall:.1f}%"
                html += "</div>\n"

                bottlenecks = result.get("bottlenecks", [])
                if bottlenecks:
                    html += "<h3>Identified Bottlenecks</h3>\n"
                    for bottleneck in bottlenecks:
                        html += f'<div class="bottleneck">'
                        html += (
                            f'<strong>{bottleneck.get("type", "Unknown")}</strong><br>'
                        )
                        html += f'{bottleneck.get("description", "")}<br>'
                        html += f'<em>Impact: {bottleneck.get("impact", "")}</em>'
                        html += "</div>\n"

                        html += f'<div class="recommendation">'
                        html += f'<strong>Recommendation:</strong> {bottleneck.get("suggestion", "")}'
                        html += "</div>\n"

        html += """
    </div>
</body>
</html>
"""
        return html

    def generate_pdf_report(self, output_file: Path) -> str:
        """Generate PDF performance report."""
        # PDF generation would require additional libraries like reportlab
        # For now, we'll just generate HTML and note that PDF requires conversion
        if self.verbose:
            print("> [INFO] PDF generation requires HTML to PDF conversion")
            print("> [INFO] Generating HTML report instead...")

        html_file = output_file.with_suffix(".html")
        return self.generate_html_report(html_file)
