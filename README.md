# Distributed Deep Learning Benchmark Suite (TorchScale)

A comprehensive benchmarking CLI for PyTorch Distributed Data Parallel (DDP) clusters. TorchScale automates multi-GPU benchmarks, profiling, and performance analysis to help ML Infrastructure Engineers optimize training throughput on GPU clusters.

## Features

- 🚀 **Automated Benchmarking**: Run throughput and latency tests across multiple GPU configurations
- 🔍 **Profiling Integration**: Deep dive into performance with NVIDIA Nsight Systems integration
- 📊 **Visual Reports**: Generate HTML/PDF reports with scaling efficiency curves and bottleneck analysis
- ✅ **System Validation**: Check DDP requirements (PyTorch, CUDA, NCCL, Drivers)
- 📝 **YAML Configuration**: Define reproducible experiment configurations

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Core Language | Python 3.9+ | The glue code and main logic |
| Deep Learning | PyTorch (DDP) | Distributed training backend |
| CLI Framework | Typer | Robust command-line interface |
| Profiling | NVIDIA Nsight Systems | Capture CUDA kernel and NCCL traces |
| Orchestration | torchrun | Spawn distributed processes |
| Data Analysis | Pandas | Parse logs and calculate statistics |
| Visualization | Plotly/Matplotlib | Generate performance visualizations |
| Configuration | YAML | Define experiment parameters |

## Installation

```bash
# Clone the repository
git clone https://github.com/ambicuity/Distributed-Deep-Learning-Benchmark-Suite.git
cd Distributed-Deep-Learning-Benchmark-Suite

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Validate Your System

Check if your system meets DDP requirements:

```bash
torchscale validate
```

### 2. Create a Configuration File

Create a `benchmark_config.yaml` file (or use the provided example):

```yaml
experiment_name: "resnet50_scaling_test"
models: 
  - "resnet50"
  - "bert-large"
batch_sizes: 
  - 64
  - 128
  - 256
gpu_counts: 
  - 1
  - 2
  - 4
  - 8
profiling:
  enabled: true
  tool: "nsys"
  trigger: "sync_stall_detection"
```

### 3. Run Benchmarks

Execute throughput and latency tests:

```bash
torchscale benchmark --config benchmark_config.yaml --output-dir ./results
```

**Example Output:**
```
> Running ResNet50 on 4 GPUs (Batch 128)...
> [INFO] Average Throughput: 1,450 img/sec
> [INFO] Iteration Latency: 220ms
> [WARN] High variance in gradient sync time detected.
```

### 4. Profile for Bottlenecks

Deep dive into performance issues:

```bash
torchscale profile --gpus 4 --duration 30s --target sync_stalls
```

**Example Output:**
```
> [ANALYSIS] Profiling complete.
> [ANALYSIS] Sync stall time: 15.0%
> [ANALYSIS] Found 2 bottleneck(s):
>   1. NCCL all_reduce: GPU-0 finishing backward pass slower than GPU-3
>   2. Gradient synchronization: High variance in gradient sync time
```

### 5. Generate Reports

Create a visual performance report:

```bash
torchscale report generate --source ./results --format html
```

This generates an HTML report with:
- Scaling efficiency plots (Ideal vs. Actual)
- Throughput and latency statistics
- Bottleneck analysis with recommendations

## CLI Reference

### Main Commands

```
Usage: torchscale [OPTIONS] COMMAND [ARGS]...

  Benchmarking CLI for PyTorch DDP Clusters.

Options:
  --verbose / --quiet  Enable verbose logging.
  --help               Show this message and exit.

Commands:
  benchmark  Run throughput/latency tests across GPU configs.
  profile    Wrap execution in Nsight Systems to find bottlenecks.
  report     Generate visual HTML/PDF performance reports.
  validate   Check if current node meets DDP requirements (NCCL, Drivers).
```

### benchmark

Run throughput/latency tests:

```bash
torchscale benchmark --config <config.yaml> --output-dir <output_dir> [--verbose]
```

**Options:**
- `--config, -c`: Path to benchmark configuration YAML file (required)
- `--output-dir, -o`: Directory to save results (default: ./results)
- `--verbose, -v`: Enable verbose logging

### profile

Profile execution with Nsight Systems:

```bash
torchscale profile --gpus <N> --duration <seconds> --target <target> [--output-dir <dir>]
```

**Options:**
- `--gpus, -g`: Number of GPUs to profile (default: 4)
- `--duration, -d`: Duration in seconds (default: 30)
- `--target, -t`: Profiling target (default: sync_stalls)
- `--output-dir, -o`: Directory to save results (default: ./results)
- `--verbose, -v`: Enable verbose logging

### report

Generate performance reports:

```bash
torchscale report --source <results_dir> --format <html|pdf> [--output <file>]
```

**Options:**
- `--source, -s`: Directory containing benchmark results (default: ./results)
- `--format, -f`: Output format - html or pdf (default: html)
- `--output, -o`: Output file path (default: auto-generated)
- `--verbose, -v`: Enable verbose logging

### validate

Validate system requirements:

```bash
torchscale validate [--verbose]
```

**Options:**
- `--verbose, -v`: Enable verbose logging

## Use Case: Identifying Performance Degradation

### Problem
You notice a 15% performance degradation when scaling from 4 to 8 GPUs.

### Solution Workflow

**Phase 1: Configuration**
```bash
# Create config with GPU scaling matrix
cat > scaling_test.yaml <<EOF
experiment_name: "scaling_analysis"
models: ["resnet50"]
batch_sizes: [128]
gpu_counts: [1, 2, 4, 8]
profiling:
  enabled: true
EOF
```

**Phase 2: Execution**
```bash
torchscale benchmark --config scaling_test.yaml --output-dir ./scaling_results --verbose
```

**Phase 3: Profiling**
```bash
torchscale profile --gpus 8 --duration 60s --target sync_stalls --output-dir ./scaling_results
```

**Phase 4: Analysis**
```bash
torchscale report generate --source ./scaling_results --format html
```

**Result:**
The report identifies that NCCL all_reduce operations are waiting for 15% of cycle time because GPU-0 is finishing its backward pass slower than GPU-7 (system noise or thermal throttling).

## Configuration File Format

```yaml
experiment_name: "my_experiment"  # Experiment identifier

models:                            # List of models to benchmark
  - "resnet50"
  - "bert-large"
  - "gpt2"

batch_sizes:                       # Batch sizes per GPU
  - 32
  - 64
  - 128

gpu_counts:                        # Number of GPUs to test
  - 1
  - 2
  - 4
  - 8

profiling:                         # Profiling configuration
  enabled: true                    # Enable/disable profiling
  tool: "nsys"                     # Profiling tool (nsys)
  trigger: "sync_stall_detection"  # What to profile
```

## Requirements

### Required
- Python 3.9+
- PyTorch 2.0+ with CUDA support
- CUDA-capable GPU(s)
- NCCL library
- NVIDIA Driver

### Optional
- NVIDIA Nsight Systems (`nsys`) for profiling

## Project Structure

```
Distributed-Deep-Learning-Benchmark-Suite/
├── torchscale/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py              # Main CLI application
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   └── benchmark.py         # Benchmark execution
│   ├── profiling/
│   │   ├── __init__.py
│   │   └── nsight.py            # Nsight Systems integration
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── generator.py         # Report generation
│   └── utils/
│       ├── __init__.py
│       └── validation.py        # System validation
├── benchmark_config.yaml        # Example configuration
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on the GitHub repository.