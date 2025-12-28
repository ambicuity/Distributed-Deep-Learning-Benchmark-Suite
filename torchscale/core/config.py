"""Configuration management for benchmark experiments."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from dataclasses import dataclass, field


@dataclass
class ProfilingConfig:
    """Configuration for profiling settings."""
    enabled: bool = False
    tool: str = "nsys"
    trigger: str = "sync_stall_detection"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    experiment_name: str
    models: List[str] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    gpu_counts: List[int] = field(default_factory=list)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        profiling_data = data.get("profiling", {})
        profiling = ProfilingConfig(
            enabled=profiling_data.get("enabled", False),
            tool=profiling_data.get("tool", "nsys"),
            trigger=profiling_data.get("trigger", "sync_stall_detection")
        )
        
        return cls(
            experiment_name=data.get("experiment_name", "unnamed_experiment"),
            models=data.get("models", []),
            batch_sizes=data.get("batch_sizes", []),
            gpu_counts=data.get("gpu_counts", []),
            profiling=profiling
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "models": self.models,
            "batch_sizes": self.batch_sizes,
            "gpu_counts": self.gpu_counts,
            "profiling": {
                "enabled": self.profiling.enabled,
                "tool": self.profiling.tool,
                "trigger": self.profiling.trigger
            }
        }
