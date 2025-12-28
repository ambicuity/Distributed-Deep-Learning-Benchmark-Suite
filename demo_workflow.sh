#!/bin/bash
# TorchScale CLI Workflow Demo
# This script demonstrates the complete workflow described in the problem statement

set -e

echo "=========================================="
echo "TorchScale CLI Workflow Demonstration"
echo "=========================================="
echo ""

# Phase 1: Validation
echo "Phase 1: System Validation"
echo "Command: torchscale validate"
echo "------------------------------------------"
torchscale validate || true
echo ""

# Phase 2: Benchmark Execution
echo "Phase 2: Running Benchmarks"
echo "Command: torchscale benchmark run --config benchmark_config.yaml --output-dir ./demo_results"
echo "------------------------------------------"
torchscale benchmark run --config benchmark_config.yaml --output-dir ./demo_results
echo ""

# Phase 3: Profiling
echo "Phase 3: Profiling for Bottlenecks"
echo "Command: torchscale profile --gpus 4 --duration 30 --target sync_stalls --output-dir ./demo_results"
echo "------------------------------------------"
torchscale profile --gpus 4 --duration 30 --target sync_stalls --output-dir ./demo_results
echo ""

# Phase 4: Report Generation
echo "Phase 4: Generating Performance Report"
echo "Command: torchscale report generate --source ./demo_results --format html"
echo "------------------------------------------"
torchscale report generate --source ./demo_results --format html
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo "Generated files:"
ls -lh ./demo_results/
echo ""
echo "View the HTML report at: ./demo_results/report.html"
