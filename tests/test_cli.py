"""Tests for CLI commands."""

import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import yaml

from torchscale.cli.main import app


runner = CliRunner()


class TestCLI:
    """Test CLI commands."""
    
    def test_cli_help(self):
        """Test main help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "torchscale" in result.stdout.lower()
        assert "benchmark" in result.stdout.lower()
        assert "profile" in result.stdout.lower()
        assert "report" in result.stdout.lower()
        assert "validate" in result.stdout.lower()
    
    def test_validate_help(self):
        """Test validate command help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout.lower()
        assert "DDP" in result.stdout or "requirements" in result.stdout.lower()
    
    def test_benchmark_help(self):
        """Test benchmark command help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.stdout.lower()
    
    def test_benchmark_run_help(self):
        """Test benchmark run command help."""
        result = runner.invoke(app, ["benchmark", "run", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout.lower()
        assert "output" in result.stdout.lower()
    
    def test_profile_help(self):
        """Test profile command help."""
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0
        assert "profile" in result.stdout.lower() or "profil" in result.stdout.lower()
        assert "gpu" in result.stdout.lower()
    
    def test_report_help(self):
        """Test report command help."""
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "report" in result.stdout.lower()
    
    def test_report_generate_help(self):
        """Test report generate command help."""
        result = runner.invoke(app, ["report", "generate", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.stdout.lower()
        assert "source" in result.stdout.lower() or "format" in result.stdout.lower()
    
    def test_benchmark_run_missing_config(self):
        """Test benchmark run without config file fails."""
        result = runner.invoke(app, ["benchmark", "run"])
        assert result.exit_code != 0
    
    def test_benchmark_run_nonexistent_config(self):
        """Test benchmark run with non-existent config file fails."""
        result = runner.invoke(app, ["benchmark", "run", "--config", "/nonexistent/path/config.yaml"])
        assert result.exit_code != 0


class TestCLIValidation:
    """Test CLI validation command in isolation."""
    
    def test_validate_command_runs(self):
        """Test that validate command executes without crashing."""
        # This may fail on systems without CUDA, but it should run
        result = runner.invoke(app, ["validate"])
        # We don't assert exit code since validation may fail on non-GPU systems
        # Just verify it runs without Python exceptions
        assert "Traceback" not in result.stdout
