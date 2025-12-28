"""Tests for configuration management."""

import pytest
import tempfile
from pathlib import Path
import yaml

from torchscale.core.config import BenchmarkConfig, ProfilingConfig


class TestProfilingConfig:
    """Test ProfilingConfig dataclass."""
    
    def test_default_profiling_config(self):
        """Test default ProfilingConfig initialization."""
        config = ProfilingConfig()
        assert config.enabled is False
        assert config.tool == "nsys"
        assert config.trigger == "sync_stall_detection"
    
    def test_custom_profiling_config(self):
        """Test custom ProfilingConfig initialization."""
        config = ProfilingConfig(enabled=True, tool="custom", trigger="custom_trigger")
        assert config.enabled is True
        assert config.tool == "custom"
        assert config.trigger == "custom_trigger"


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""
    
    def test_default_benchmark_config(self):
        """Test default BenchmarkConfig initialization."""
        config = BenchmarkConfig(experiment_name="test")
        assert config.experiment_name == "test"
        assert config.models == []
        assert config.batch_sizes == []
        assert config.gpu_counts == []
        assert isinstance(config.profiling, ProfilingConfig)
    
    def test_custom_benchmark_config(self):
        """Test custom BenchmarkConfig initialization."""
        profiling = ProfilingConfig(enabled=True)
        config = BenchmarkConfig(
            experiment_name="custom_test",
            models=["resnet50"],
            batch_sizes=[32, 64],
            gpu_counts=[1, 2],
            profiling=profiling
        )
        assert config.experiment_name == "custom_test"
        assert config.models == ["resnet50"]
        assert config.batch_sizes == [32, 64]
        assert config.gpu_counts == [1, 2]
        assert config.profiling.enabled is True
    
    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML file
        yaml_content = {
            "experiment_name": "yaml_test",
            "models": ["resnet50", "bert-large"],
            "batch_sizes": [64, 128],
            "gpu_counts": [1, 2, 4],
            "profiling": {
                "enabled": True,
                "tool": "nsys",
                "trigger": "sync_stall_detection"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)
        
        try:
            config = BenchmarkConfig.from_yaml(temp_path)
            assert config.experiment_name == "yaml_test"
            assert config.models == ["resnet50", "bert-large"]
            assert config.batch_sizes == [64, 128]
            assert config.gpu_counts == [1, 2, 4]
            assert config.profiling.enabled is True
            assert config.profiling.tool == "nsys"
            assert config.profiling.trigger == "sync_stall_detection"
        finally:
            temp_path.unlink()
    
    def test_from_yaml_minimal(self):
        """Test loading minimal configuration from YAML."""
        yaml_content = {
            "experiment_name": "minimal_test"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)
        
        try:
            config = BenchmarkConfig.from_yaml(temp_path)
            assert config.experiment_name == "minimal_test"
            assert config.models == []
            assert config.batch_sizes == []
            assert config.gpu_counts == []
            assert config.profiling.enabled is False
        finally:
            temp_path.unlink()
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        profiling = ProfilingConfig(enabled=True, tool="nsys", trigger="sync_stall_detection")
        config = BenchmarkConfig(
            experiment_name="dict_test",
            models=["resnet50"],
            batch_sizes=[64],
            gpu_counts=[1],
            profiling=profiling
        )
        
        result = config.to_dict()
        assert result["experiment_name"] == "dict_test"
        assert result["models"] == ["resnet50"]
        assert result["batch_sizes"] == [64]
        assert result["gpu_counts"] == [1]
        assert result["profiling"]["enabled"] is True
        assert result["profiling"]["tool"] == "nsys"
        assert result["profiling"]["trigger"] == "sync_stall_detection"
