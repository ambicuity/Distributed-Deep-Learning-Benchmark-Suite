"""Tests for validation utilities."""

import pytest
import sys

from torchscale.utils.validation import SystemValidator


class TestSystemValidator:
    """Test SystemValidator class."""
    
    def test_init(self):
        """Test SystemValidator initialization."""
        validator = SystemValidator()
        assert validator.verbose is False
        
        validator_verbose = SystemValidator(verbose=True)
        assert validator_verbose.verbose is True
    
    def test_check_python_version(self):
        """Test Python version check."""
        validator = SystemValidator()
        passed, message = validator.check_python_version()
        
        # Should pass since we're running Python 3.9+
        version_info = sys.version_info
        if version_info.major == 3 and version_info.minor >= 9:
            assert passed is True
        
        assert "Python" in message
        assert str(version_info.major) in message
    
    def test_check_pytorch(self):
        """Test PyTorch installation check."""
        validator = SystemValidator()
        passed, message = validator.check_pytorch()
        
        # This should pass as PyTorch is in requirements.txt
        # But we won't assert on the result since it depends on environment
        assert isinstance(passed, bool)
        assert isinstance(message, str)
        
        if passed:
            assert "PyTorch" in message
        else:
            assert "not found" in message
    
    def test_check_cuda(self):
        """Test CUDA availability check."""
        validator = SystemValidator()
        passed, message = validator.check_cuda()
        
        # Result depends on environment
        assert isinstance(passed, bool)
        assert isinstance(message, str)
        
        if passed:
            assert "CUDA" in message
        else:
            assert "CUDA" in message or "not available" in message
    
    def test_check_nccl(self):
        """Test NCCL availability check."""
        validator = SystemValidator()
        passed, message = validator.check_nccl()
        
        # Result depends on environment
        assert isinstance(passed, bool)
        assert isinstance(message, str)
        
        if passed:
            assert "NCCL" in message
        else:
            assert "NCCL" in message or "not available" in message
    
    def test_check_nvidia_driver(self):
        """Test NVIDIA driver check."""
        validator = SystemValidator()
        passed, message = validator.check_nvidia_driver()
        
        # Result depends on environment
        assert isinstance(passed, bool)
        assert isinstance(message, str)
    
    def test_check_nsys(self):
        """Test Nsight Systems check."""
        validator = SystemValidator()
        passed, message = validator.check_nsys()
        
        # Result depends on environment (optional tool)
        assert isinstance(passed, bool)
        assert isinstance(message, str)
        
        if not passed:
            assert "not found" in message or "failed" in message
    
    def test_validate_all(self):
        """Test running all validation checks."""
        validator = SystemValidator()
        checks = validator.validate_all()
        
        # Verify all expected checks are present
        expected_checks = [
            "Python Version",
            "PyTorch",
            "CUDA",
            "NCCL",
            "NVIDIA Driver",
            "Nsight Systems"
        ]
        
        for check_name in expected_checks:
            assert check_name in checks
            passed, message = checks[check_name]
            assert isinstance(passed, bool)
            assert isinstance(message, str)
    
    def test_print_validation_results(self, capsys):
        """Test printing validation results."""
        validator = SystemValidator()
        
        # Create mock check results
        checks = {
            "Python Version": (True, "Python 3.9.0"),
            "PyTorch": (True, "PyTorch 2.0.0"),
            "CUDA": (False, "CUDA not available"),
            "NCCL": (False, "NCCL not available"),
            "NVIDIA Driver": (False, "nvidia-smi not found"),
            "Nsight Systems": (False, "nsys not found (optional)")
        }
        
        result = validator.print_validation_results(checks)
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Verify output contains expected strings
        assert "DDP System Validation Results" in captured.out
        assert "Python Version" in captured.out
        assert "PyTorch" in captured.out
        
        # Result should be False since required checks failed
        assert result is False
    
    def test_print_validation_results_all_pass(self, capsys):
        """Test printing validation results when all pass."""
        validator = SystemValidator()
        
        # Create mock check results where all pass
        checks = {
            "Python Version": (True, "Python 3.9.0"),
            "PyTorch": (True, "PyTorch 2.0.0"),
            "CUDA": (True, "CUDA available"),
            "NCCL": (True, "NCCL available"),
            "NVIDIA Driver": (True, "NVIDIA Driver 525.0"),
            "Nsight Systems": (True, "Nsight Systems 2023.1")
        }
        
        result = validator.print_validation_results(checks)
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Verify output contains success message
        assert "All required checks passed" in captured.out
        
        # Result should be True
        assert result is True
