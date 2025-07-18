"""
Prior probability distributions for Bayesian changepoint detection.

This module provides various prior distributions for modeling the probability
of changepoints in time series data.
"""

import torch
import torch.distributions as dist
from typing import Union, Optional
from .device import ensure_tensor, get_device


def const_prior(
    t: Union[int, torch.Tensor], 
    p: float = 0.25,
    device: Optional[Union[str, torch.device]] = None
) -> Union[float, torch.Tensor]:
    """
    Constant prior probability for changepoints.
    
    Returns the same log probability for all time points, representing
    a uniform prior over changepoint locations.
    
    Parameters
    ----------
    t : int or torch.Tensor
        Time index or tensor of time indices.
    p : float, optional
        Constant probability value (default: 0.25).
        Must be between 0 and 1.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.
        
    Returns
    -------
    float or torch.Tensor
        Log probability value(s).
        
    Examples
    --------
    >>> # Single time point
    >>> log_prob = const_prior(5, p=0.1)
    >>> print(log_prob)  # log(0.1)
    
    >>> # Multiple time points
    >>> t = torch.arange(10)
    >>> log_probs = const_prior(t, p=0.2)
    >>> print(log_probs.shape)  # torch.Size([10])
    
    Notes
    -----
    The constant prior assumes that changepoints are equally likely
    at any time point in the series.
    """
    if not 0 < p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    
    log_p = torch.log(torch.tensor(p, dtype=torch.float32))
    
    if isinstance(t, int):
        return log_p.item()
    else:
        device = get_device(device)
        t_tensor = ensure_tensor(t, device=device)
        return log_p.expand_as(t_tensor)


def geometric_prior(
    t: Union[int, torch.Tensor],
    p: float = 0.25,
    device: Optional[Union[str, torch.device]] = None
) -> Union[float, torch.Tensor]:
    """
    Geometric prior for changepoint detection.
    
    Models the time between changepoints as following a geometric distribution,
    which is the discrete analogue of an exponential distribution.
    
    Parameters
    ----------
    t : int or torch.Tensor
        Time index or tensor of time indices (number of trials).
    p : float, optional
        Probability of success (changepoint) at each trial (default: 0.25).
        Must be between 0 and 1.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.
        
    Returns
    -------
    float or torch.Tensor
        Log probability value(s) from the geometric distribution.
        
    Examples
    --------
    >>> # Single time point
    >>> log_prob = geometric_prior(3, p=0.1)
    
    >>> # Multiple time points
    >>> t = torch.arange(1, 11)  # 1 to 10
    >>> log_probs = geometric_prior(t, p=0.2)
    
    Notes
    -----
    The geometric distribution models the number of trials needed for
    the first success, making it suitable for modeling inter-arrival
    times between changepoints.
    """
    if not 0 < p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    
    device = get_device(device)
    
    if isinstance(t, int):
        if t <= 0:
            raise ValueError("Time index t must be positive for geometric prior")
        geom_dist = dist.Geometric(probs=torch.tensor(p, device=device))
        return geom_dist.log_prob(torch.tensor(t, device=device)).item()
    else:
        t_tensor = ensure_tensor(t, device=device)
        if torch.any(t_tensor <= 0):
            raise ValueError("All time indices must be positive for geometric prior")
        geom_dist = dist.Geometric(probs=torch.tensor(p, device=device))
        return geom_dist.log_prob(t_tensor)


def negative_binomial_prior(
    t: Union[int, torch.Tensor],
    k: int = 1,
    p: float = 0.25,
    device: Optional[Union[str, torch.device]] = None
) -> Union[float, torch.Tensor]:
    """
    Negative binomial prior for changepoint detection.
    
    Models the number of trials needed to achieve k successes (changepoints),
    generalizing the geometric distribution.
    
    Parameters
    ----------
    t : int or torch.Tensor
        Time index or tensor of time indices (number of trials).
    k : int, optional
        Number of successes (changepoints) to achieve (default: 1).
        Must be positive.
    p : float, optional
        Probability of success at each trial (default: 0.25).
        Must be between 0 and 1.
    device : str, torch.device, or None, optional
        Device to place the output tensor on.
        
    Returns
    -------
    float or torch.Tensor
        Log probability value(s) from the negative binomial distribution.
        
    Examples
    --------
    >>> # Single time point
    >>> log_prob = negative_binomial_prior(5, k=2, p=0.1)
    
    >>> # Multiple time points
    >>> t = torch.arange(1, 11)
    >>> log_probs = negative_binomial_prior(t, k=3, p=0.2)
    
    Notes
    -----
    When k=1, the negative binomial distribution reduces to the geometric
    distribution. Higher values of k model scenarios where multiple
    changepoints must occur before the process is considered complete.
    """
    if not 0 < p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    if k <= 0:
        raise ValueError("Number of successes k must be positive")
    
    device = get_device(device)
    
    if isinstance(t, int):
        if t < k:
            return float('-inf')  # Impossible to have k successes in fewer than k trials
        nb_dist = dist.NegativeBinomial(
            total_count=torch.tensor(k, device=device, dtype=torch.float32),
            probs=torch.tensor(p, device=device)
        )
        return nb_dist.log_prob(torch.tensor(t - k, device=device)).item()
    else:
        t_tensor = ensure_tensor(t, device=device)
        # Set impossible cases to -inf
        log_probs = torch.full_like(t_tensor, float('-inf'), dtype=torch.float32)
        valid_mask = t_tensor >= k
        
        if torch.any(valid_mask):
            nb_dist = dist.NegativeBinomial(
                total_count=torch.tensor(k, device=device, dtype=torch.float32),
                probs=torch.tensor(p, device=device)
            )
            log_probs[valid_mask] = nb_dist.log_prob(t_tensor[valid_mask] - k)
        
        return log_probs
