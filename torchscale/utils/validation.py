"""Validation utilities for checking system requirements."""

import subprocess
import sys
from typing import Dict, List, Tuple


class SystemValidator:
    """Validator for DDP system requirements."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements (>=3.9)."""
        version = sys.version_info
        meets_req = version.major == 3 and version.minor >= 9
        message = f"Python {version.major}.{version.minor}.{version.micro}"
        return meets_req, message
    
    def check_pytorch(self) -> Tuple[bool, str]:
        """Check if PyTorch is installed and accessible."""
        try:
            import torch
            version = torch.__version__
            return True, f"PyTorch {version}"
        except ImportError:
            return False, "PyTorch not found"
    
    def check_cuda(self) -> Tuple[bool, str]:
        """Check if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return True, f"CUDA available (devices: {torch.cuda.device_count()})"
            else:
                return False, "CUDA not available"
        except Exception as e:
            return False, f"CUDA check failed: {str(e)}"
    
    def check_nccl(self) -> Tuple[bool, str]:
        """Check if NCCL is available for distributed training."""
        try:
            import torch.distributed as dist
            if dist.is_nccl_available():
                return True, "NCCL available"
            else:
                return False, "NCCL not available"
        except Exception as e:
            return False, f"NCCL check failed: {str(e)}"
    
    def check_nvidia_driver(self) -> Tuple[bool, str]:
        """Check NVIDIA driver version using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                driver_version = result.stdout.strip().split('\n')[0]
                return True, f"NVIDIA Driver {driver_version}"
            else:
                return False, "nvidia-smi failed"
        except FileNotFoundError:
            return False, "nvidia-smi not found"
        except Exception as e:
            return False, f"Driver check failed: {str(e)}"
    
    def check_nsys(self) -> Tuple[bool, str]:
        """Check if NVIDIA Nsight Systems is installed."""
        try:
            result = subprocess.run(
                ["nsys", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_line = result.stdout.strip().split('\n')[0]
                return True, f"Nsight Systems: {version_line}"
            else:
                return False, "nsys command failed"
        except FileNotFoundError:
            return False, "nsys not found (optional for profiling)"
        except Exception as e:
            return False, f"nsys check failed: {str(e)}"
    
    def validate_all(self) -> Dict[str, Tuple[bool, str]]:
        """Run all validation checks."""
        checks = {
            "Python Version": self.check_python_version(),
            "PyTorch": self.check_pytorch(),
            "CUDA": self.check_cuda(),
            "NCCL": self.check_nccl(),
            "NVIDIA Driver": self.check_nvidia_driver(),
            "Nsight Systems": self.check_nsys(),
        }
        return checks
    
    def print_validation_results(self, checks: Dict[str, Tuple[bool, str]]) -> bool:
        """Print validation results in a formatted manner."""
        print("\n" + "="*60)
        print("  DDP System Validation Results")
        print("="*60 + "\n")
        
        all_passed = True
        required_checks = ["Python Version", "PyTorch", "CUDA", "NCCL", "NVIDIA Driver"]
        
        for check_name, (passed, message) in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            required = " (Required)" if check_name in required_checks else " (Optional)"
            
            print(f"{status:8} {check_name:20} {required:12} {message}")
            
            if not passed and check_name in required_checks:
                all_passed = False
        
        print("\n" + "="*60)
        
        if all_passed:
            print("✓ All required checks passed. System is ready for DDP benchmarking.")
        else:
            print("✗ Some required checks failed. Please install missing components.")
        
        print("="*60 + "\n")
        
        return all_passed
