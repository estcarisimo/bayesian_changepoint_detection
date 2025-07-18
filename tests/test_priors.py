"""
Tests for prior probability distributions.
"""

import pytest
import torch
import numpy as np
from bayesian_changepoint_detection.priors import (
    const_prior,
    geometric_prior,
    negative_binomial_prior
)


class TestConstPrior:
    """Test constant prior function."""
    
    def test_single_timepoint(self):
        """Test constant prior for single time point."""
        log_prob = const_prior(5, p=0.1)
        expected = np.log(0.1)
        assert abs(log_prob - expected) < 1e-6
    
    def test_multiple_timepoints(self):
        """Test constant prior for multiple time points."""
        t = torch.arange(10)
        log_probs = const_prior(t, p=0.2)
        
        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == (10,)
        
        # All values should be the same
        expected = torch.log(torch.tensor(0.2))
        assert torch.allclose(log_probs, expected.expand_as(log_probs))
    
    def test_probability_validation(self):
        """Test probability parameter validation."""
        # Valid probabilities
        const_prior(1, p=0.1)
        const_prior(1, p=1.0)
        
        # Invalid probabilities
        with pytest.raises(ValueError):
            const_prior(1, p=0.0)
        
        with pytest.raises(ValueError):
            const_prior(1, p=1.1)
        
        with pytest.raises(ValueError):
            const_prior(1, p=-0.1)


class TestGeometricPrior:
    """Test geometric prior function."""
    
    def test_single_timepoint(self):
        """Test geometric prior for single time point."""
        log_prob = geometric_prior(3, p=0.1)
        
        # Should be finite and reasonable
        assert torch.isfinite(torch.tensor(log_prob))
        assert log_prob < 0  # Log probability should be negative
    
    def test_multiple_timepoints(self):
        """Test geometric prior for multiple time points."""
        t = torch.arange(1, 11)  # 1 to 10
        log_probs = geometric_prior(t, p=0.2)
        
        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == (10,)
        assert torch.isfinite(log_probs).all()
        
        # Probabilities should generally decrease with time
        # (geometric distribution is decreasing)
        assert log_probs[0] > log_probs[-1]
    
    def test_time_validation(self):
        """Test time parameter validation."""
        # Valid times
        geometric_prior(1, p=0.1)
        geometric_prior(torch.tensor([1, 2, 3]), p=0.1)
        
        # Invalid times
        with pytest.raises(ValueError):
            geometric_prior(0, p=0.1)
        
        with pytest.raises(ValueError):
            geometric_prior(-1, p=0.1)
        
        with pytest.raises(ValueError):
            geometric_prior(torch.tensor([0, 1, 2]), p=0.1)
    
    def test_probability_validation_geometric(self):
        """Test probability validation for geometric prior."""
        with pytest.raises(ValueError):
            geometric_prior(1, p=0.0)
        
        with pytest.raises(ValueError):
            geometric_prior(1, p=1.1)


class TestNegativeBinomialPrior:
    """Test negative binomial prior function."""
    
    def test_single_timepoint(self):
        """Test negative binomial prior for single time point."""
        log_prob = negative_binomial_prior(5, k=2, p=0.1)
        
        assert torch.isfinite(torch.tensor(log_prob))
        assert log_prob < 0  # Log probability should be negative
    
    def test_multiple_timepoints(self):
        """Test negative binomial prior for multiple time points."""
        t = torch.arange(1, 11)
        log_probs = negative_binomial_prior(t, k=2, p=0.2)
        
        assert isinstance(log_probs, torch.Tensor)
        assert log_probs.shape == (10,)
        
        # First k-1 values should be -inf
        assert log_probs[0] == float('-inf')  # t=1, k=2, impossible
        assert torch.isfinite(log_probs[1:]).all()  # t>=2 should be finite
    
    def test_impossible_cases(self):
        """Test handling of impossible cases (t < k)."""
        # Single impossible case
        log_prob = negative_binomial_prior(1, k=2, p=0.1)
        assert log_prob == float('-inf')
        
        # Multiple cases with some impossible
        t = torch.tensor([1, 2, 3, 4])
        log_probs = negative_binomial_prior(t, k=3, p=0.1)
        
        assert log_probs[0] == float('-inf')  # t=1, k=3
        assert log_probs[1] == float('-inf')  # t=2, k=3
        assert torch.isfinite(log_probs[2])   # t=3, k=3, possible
        assert torch.isfinite(log_probs[3])   # t=4, k=3, possible
    
    def test_reduction_to_geometric(self):
        """Test that k=1 negative binomial follows expected pattern."""
        t = torch.arange(1, 6)
        p = 0.3
        
        nb_log_probs = negative_binomial_prior(t, k=1, p=p)
        
        # PyTorch's NegativeBinomial(k, p) counts failures before k successes
        # So NB(t-k, k, p) = C(t-1, k-1) * p^k * (1-p)^(t-k)
        # For k=1: NB(t-1, 1, p) = p * (1-p)^(t-1)
        
        # Check that consecutive differences are constant (geometric property)
        differences = nb_log_probs[1:] - nb_log_probs[:-1]
        
        # All differences should be equal (within numerical tolerance)
        assert torch.allclose(differences, differences[0], atol=1e-5)
        
        # For NB with our parameterization, the difference is log(p)
        expected_diff = torch.log(torch.tensor(p))
        assert torch.allclose(differences[0], expected_diff, atol=1e-5)
    
    def test_parameter_validation_nb(self):
        """Test parameter validation for negative binomial."""
        # Valid parameters
        negative_binomial_prior(5, k=1, p=0.1)
        negative_binomial_prior(5, k=3, p=0.9)
        
        # Invalid k
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=0, p=0.1)
        
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=-1, p=0.1)
        
        # Invalid p
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=1, p=0.0)
        
        with pytest.raises(ValueError):
            negative_binomial_prior(5, k=1, p=1.1)


class TestPriorDeviceHandling:
    """Test device handling for priors."""
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency(self):
        """Test that priors work correctly with different devices."""
        t_cpu = torch.arange(1, 6)
        t_cuda = t_cpu.cuda()
        
        # Test const_prior
        log_probs_cpu = const_prior(t_cpu, p=0.1, device='cpu')
        log_probs_cuda = const_prior(t_cuda, p=0.1, device='cuda')
        
        assert log_probs_cpu.device.type == 'cpu'
        assert log_probs_cuda.device.type == 'cuda'
        assert torch.allclose(log_probs_cpu, log_probs_cuda.cpu())
        
        # Test geometric_prior
        geom_cpu = geometric_prior(t_cpu, p=0.2, device='cpu')
        geom_cuda = geometric_prior(t_cuda, p=0.2, device='cuda')
        
        assert geom_cpu.device.type == 'cpu'
        assert geom_cuda.device.type == 'cuda'
        assert torch.allclose(geom_cpu, geom_cuda.cpu())
        
        # Test negative_binomial_prior
        nb_cpu = negative_binomial_prior(t_cpu, k=2, p=0.3, device='cpu')
        nb_cuda = negative_binomial_prior(t_cuda, k=2, p=0.3, device='cuda')
        
        assert nb_cpu.device.type == 'cpu'
        assert nb_cuda.device.type == 'cuda'
        assert torch.allclose(nb_cpu, nb_cuda.cpu(), equal_nan=True)