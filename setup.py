from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torchscale",
    version="0.1.0",
    author="ML Infrastructure Team",
    description="Benchmarking CLI for PyTorch DDP Clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "typer>=0.9.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "plotly>=5.0.0",
        "matplotlib>=3.5.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "torchscale=torchscale.cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
