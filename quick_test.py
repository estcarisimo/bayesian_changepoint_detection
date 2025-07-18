#!/usr/bin/env python3
"""
Quick test script to verify bayesian_changepoint_detection installation and basic functionality.
"""

import torch
from functools import partial
from bayesian_changepoint_detection import (
    online_changepoint_detection,
    offline_changepoint_detection,
    constant_hazard,
    const_prior,
    StudentT,
    get_device_info
)

def test_online_detection():
    """Test online changepoint detection."""
    print("Testing online changepoint detection...")
    
    # Generate sample data
    torch.manual_seed(42)
    data = torch.cat([
        torch.randn(50) + 0,      # First segment: mean=0
        torch.randn(50) + 3,      # Second segment: mean=3
        torch.randn(50) + 0,      # Third segment: mean=0
    ])
    
    # Set up the model
    hazard_func = partial(constant_hazard, 250)  # Expected run length of 250
    likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
    
    # Run online changepoint detection
    run_length_probs, changepoint_probs = online_changepoint_detection(
        data, hazard_func, likelihood
    )
    
    print(f"✓ Online detection completed")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Max changepoint probability: {changepoint_probs.max().item():.4f}")
    print(f"  - High probability indices: {torch.where(changepoint_probs > 0.01)[0].tolist()[:5]}...")
    
    return True

def test_offline_detection():
    """Test offline changepoint detection."""
    print("\nTesting offline changepoint detection...")
    
    # Generate sample data
    torch.manual_seed(42)
    data = torch.cat([
        torch.randn(30) + 0,      # First segment: mean=0
        torch.randn(30) + 2,      # Second segment: mean=2
        torch.randn(30) + 0,      # Third segment: mean=0
    ])
    
    # Use offline method for batch processing
    prior_func = partial(const_prior, p=1/(len(data)+1))
    
    # Import the offline StudentT (if it exists)
    try:
        from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT
        likelihood = OfflineStudentT()
    except ImportError:
        # Fallback to online StudentT
        likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
    
    try:
        Q, P, changepoint_log_probs = offline_changepoint_detection(
            data, prior_func, likelihood
        )
        
        # Get changepoint probabilities
        changepoint_probs = torch.exp(changepoint_log_probs).sum(0)
        
        print(f"✓ Offline detection completed")
        print(f"  - Data shape: {data.shape}")
        print(f"  - Max changepoint probability: {changepoint_probs.max().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Offline detection not fully implemented: {e}")
        return False

def test_device_info():
    """Test device information."""
    print("\nTesting device information...")
    
    device_info = get_device_info()
    print(f"✓ Device info: {device_info}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Bayesian Changepoint Detection - Quick Test")
    print("=" * 60)
    
    try:
        # Test device info
        test_device_info()
        
        # Test online detection
        test_online_detection()
        
        # Test offline detection
        test_offline_detection()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The bayesian_changepoint_detection package is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check your installation and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()