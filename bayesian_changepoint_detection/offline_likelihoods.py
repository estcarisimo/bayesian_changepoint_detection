"""
Offline likelihood functions for Bayesian changepoint detection.

This module provides likelihood functions for offline (batch) changepoint detection
using PyTorch for efficient computation and GPU acceleration.
"""

import torch
import torch.distributions as dist
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Tuple
from .device import ensure_tensor, get_device


class BaseLikelihood(ABC):
    """
    Abstract base class for offline likelihood functions.
    
    This class provides a template for implementing likelihood functions
    for offline Bayesian changepoint detection. Subclasses must implement
    the pdf method.
    
    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on (CPU or GPU).
    cache_enabled : bool, optional
        Whether to enable caching for dynamic programming (default: True).
    """
    
    def __init__(
        self, 
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True
    ):
        self.device = get_device(device)
        self.cache_enabled = cache_enabled
        self._cache: Dict[Tuple[int, int], float] = {}
        self._cached_data = None
    
    @abstractmethod
    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute the log probability density for a data segment.
        
        Parameters
        ----------
        data : torch.Tensor
            The complete time series data.
        t : int
            Start index of the segment (inclusive).
        s : int
            End index of the segment (exclusive).
            
        Returns
        -------
        float
            Log probability density for the segment data[t:s].
        """
        raise NotImplementedError(
            "PDF method must be implemented in subclass."
        )
    
    def _check_cache(self, data: torch.Tensor, t: int, s: int) -> Optional[float]:
        """Check if result is cached and cache is valid."""
        if not self.cache_enabled:
            return None
        
        # Check if data has changed
        if self._cached_data is None or not torch.equal(data, self._cached_data):
            self._cache.clear()
            self._cached_data = data.clone() if self.cache_enabled else None
            return None
        
        return self._cache.get((t, s), None)
    
    def _store_cache(self, t: int, s: int, result: float) -> None:
        """Store result in cache."""
        if self.cache_enabled:
            self._cache[(t, s)] = result


class StudentT(BaseLikelihood):
    """
    Student's t-distribution likelihood for offline changepoint detection.
    
    Uses conjugate Normal-Gamma priors for efficient computation of segment
    probabilities. Suitable for univariate data with unknown mean and variance.
    
    Parameters
    ----------
    alpha0 : float, optional
        Prior shape parameter for precision (default: 1.0).
    beta0 : float, optional
        Prior rate parameter for precision (default: 1.0).
    kappa0 : float, optional
        Prior precision for mean (default: 1.0).
    mu0 : float, optional
        Prior mean (default: 0.0).
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Whether to enable caching (default: True).
        
    Examples
    --------
    >>> import torch
    >>> likelihood = StudentT()
    >>> data = torch.randn(100)
    >>> log_prob = likelihood.pdf(data, 10, 50)  # Segment from 10 to 50
    
    Notes
    -----
    This implementation follows the conjugate prior approach described in
    Murphy, K. "Conjugate Bayesian analysis of the Gaussian distribution" (2007).
    """
    
    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        kappa0: float = 1.0,
        mu0: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True
    ):
        super().__init__(device, cache_enabled)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.kappa0 = kappa0
        self.mu0 = mu0
    
    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute log probability for data segment using Student's t-distribution.
        
        Parameters
        ----------
        data : torch.Tensor
            Complete time series data.
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).
            
        Returns
        -------
        float
            Log probability density for the segment.
        """
        # Check cache first
        cached_result = self._check_cache(data, t, s)
        if cached_result is not None:
            return cached_result
        
        data = ensure_tensor(data, device=self.device)
        
        # Extract segment
        segment = data[t:s]
        n = s - t
        
        if n == 0:
            result = 0.0
            self._store_cache(t, s, result)
            return result
        
        # Compute sufficient statistics
        sample_mean = segment.mean()
        sample_var = segment.var(unbiased=False) if n > 1 else torch.tensor(0.0, device=self.device)
        
        # Update hyperparameters using conjugate prior formulas
        kappa_n = torch.tensor(self.kappa0 + n, device=self.device)
        mu_n = (torch.tensor(self.kappa0, device=self.device) * torch.tensor(self.mu0, device=self.device) + n * sample_mean) / kappa_n
        alpha_n = torch.tensor(self.alpha0 + n / 2, device=self.device)
        
        beta_n = (
            torch.tensor(self.beta0, device=self.device) + 
            0.5 * n * sample_var +
            (torch.tensor(self.kappa0, device=self.device) * n * (sample_mean - torch.tensor(self.mu0, device=self.device)) ** 2) / (2 * kappa_n)
        )
        
        # Student's t parameters for marginal likelihood
        nu_n = (2.0 * alpha_n).detach()
        scale = torch.sqrt(beta_n * kappa_n / (alpha_n * self.kappa0))
        
        # Compute log marginal likelihood using the closed-form formula
        # This is more numerically stable than computing the product of individual densities
        log_prob = torch.tensor(0.0, device=self.device)
        
        for i in range(n):
            x_i = segment[i]
            # For each point, compute its contribution to the log likelihood
            log_prob += (
                torch.lgamma((nu_n + 1) / 2) - torch.lgamma(nu_n / 2) -
                0.5 * torch.log(torch.pi * nu_n) - torch.log(scale) -
                ((nu_n + 1) / 2) * torch.log(1 + ((x_i - mu_n) / scale) ** 2 / nu_n)
            )
        
        result = log_prob.item()
        self._store_cache(t, s, result)
        return result


class IndependentFeaturesLikelihood(BaseLikelihood):
    """
    Independent features likelihood for multivariate data.
    
    Assumes features are independent with unknown means and variances.
    Uses conjugate Normal-Gamma priors for each dimension separately.
    
    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Whether to enable caching (default: True).
        
    Examples
    --------
    >>> import torch
    >>> likelihood = IndependentFeaturesLikelihood()
    >>> data = torch.randn(100, 5)  # 100 time points, 5 dimensions
    >>> log_prob = likelihood.pdf(data, 10, 50)
    
    Notes
    -----
    This model treats each dimension independently, which simplifies computation
    but ignores potential correlations between dimensions. Based on the approach
    in Xiang & Murphy (2007).
    """
    
    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute log probability assuming independent features.
        
        Parameters
        ----------
        data : torch.Tensor
            Complete time series data (shape: [T] or [T, D]).
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).
            
        Returns
        -------
        float
            Log probability density for the segment.
        """
        # Check cache first
        cached_result = self._check_cache(data, t, s)
        if cached_result is not None:
            return cached_result
        
        data = ensure_tensor(data, device=self.device)
        
        # Handle both univariate and multivariate data
        if data.dim() == 1:
            data = data.unsqueeze(1)  # Make it [T, 1]
        
        # Extract segment
        x = data[t:s]
        n, d = x.shape
        
        if n == 0:
            result = 0.0
            self._store_cache(t, s, result)
            return result
        
        # Weakest proper prior
        N0 = d
        V0 = x.var(dim=0, unbiased=False)
        
        # Handle case where variance is 0 (constant data)
        V0 = torch.clamp(V0, min=1e-8)
        
        # Updated parameters
        Vn = V0 + (x ** 2).sum(dim=0)
        
        # Compute log marginal likelihood (Section 3.1 from Xiang & Murphy paper)
        log_prob = d * (
            -(n / 2) * torch.log(torch.tensor(torch.pi, device=self.device)) +
            (N0 / 2) * torch.log(V0).sum() -
            torch.lgamma(torch.tensor(N0 / 2, device=self.device)) +
            torch.lgamma(torch.tensor((N0 + n) / 2, device=self.device))
        ) - ((N0 + n) / 2) * torch.log(Vn).sum()
        
        result = log_prob.item()
        self._store_cache(t, s, result)
        return result


class FullCovarianceLikelihood(BaseLikelihood):
    """
    Full covariance likelihood for multivariate data.
    
    Models the full covariance structure using a Normal-Wishart conjugate prior.
    More flexible than independent features but computationally more expensive.
    
    Parameters
    ----------
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Whether to enable caching (default: True).
        
    Examples
    --------
    >>> import torch
    >>> likelihood = FullCovarianceLikelihood()
    >>> data = torch.randn(100, 3)  # 100 time points, 3 dimensions
    >>> log_prob = likelihood.pdf(data, 10, 50)
    
    Notes
    -----
    This model captures correlations between dimensions but requires more data
    for reliable estimation. Based on the approach in Xiang & Murphy (2007).
    """
    
    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute log probability using full covariance model.
        
        Parameters
        ----------
        data : torch.Tensor
            Complete time series data (shape: [T] or [T, D]).
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).
            
        Returns
        -------
        float
            Log probability density for the segment.
        """
        # Check cache first
        cached_result = self._check_cache(data, t, s)
        if cached_result is not None:
            return cached_result
        
        data = ensure_tensor(data, device=self.device)
        
        # Handle both univariate and multivariate data
        if data.dim() == 1:
            data = data.unsqueeze(1)  # Make it [T, 1]
        
        # Extract segment
        x = data[t:s]
        n, dim = x.shape
        
        if n == 0:
            result = 0.0
            self._store_cache(t, s, result)
            return result
        
        # Weakest proper prior
        N0 = dim
        V0 = x.var(dim=0, unbiased=False).item() * torch.eye(dim, device=self.device)
        
        # Ensure V0 is positive definite
        V0 = V0 + 1e-6 * torch.eye(dim, device=self.device)
        
        # Compute outer product sum efficiently using einsum
        Vn = V0 + torch.einsum('ij,ik->jk', x, x)
        
        # Ensure Vn is positive definite
        try:
            L_V0 = torch.linalg.cholesky(V0)
            L_Vn = torch.linalg.cholesky(Vn)
            logdet_V0 = 2 * torch.diagonal(L_V0).log().sum()
            logdet_Vn = 2 * torch.diagonal(L_Vn).log().sum()
        except RuntimeError:
            # Fallback to eigenvalue decomposition if Cholesky fails
            logdet_V0 = torch.linalg.slogdet(V0)[1]
            logdet_Vn = torch.linalg.slogdet(Vn)[1]
        
        # Multivariate gamma function (log)
        def multigammaln(a: torch.Tensor, p: int) -> torch.Tensor:
            """Multivariate log-gamma function."""
            result = (p * (p - 1) / 4) * torch.log(torch.tensor(torch.pi, device=self.device))
            for j in range(p):
                result += torch.lgamma(a - j / 2)
            return result
        
        # Compute log marginal likelihood (Section 3.2 from Xiang & Murphy paper)
        log_prob = (
            -(dim * n / 2) * torch.log(torch.tensor(torch.pi, device=self.device)) +
            (N0 / 2) * logdet_V0 -
            multigammaln(torch.tensor(N0 / 2, device=self.device), dim) +
            multigammaln(torch.tensor((N0 + n) / 2, device=self.device), dim) -
            ((N0 + n) / 2) * logdet_Vn
        )
        
        result = log_prob.item()
        self._store_cache(t, s, result)
        return result


class MultivariateT(BaseLikelihood):
    """
    Multivariate Student's t-distribution likelihood for offline detection.
    
    Uses Normal-Wishart conjugate priors for modeling multivariate segments
    with unknown mean vector and covariance matrix.
    
    Parameters
    ----------
    dims : int, optional
        Number of dimensions. If None, inferred from data.
    dof0 : float, optional
        Prior degrees of freedom (default: dims + 1).
    kappa0 : float, optional
        Prior precision for mean (default: 1.0).
    mu0 : torch.Tensor, optional
        Prior mean vector (default: zero vector).
    Psi0 : torch.Tensor, optional
        Prior scale matrix (default: identity matrix).
    device : str, torch.device, or None, optional
        Device to place tensors on.
    cache_enabled : bool, optional
        Whether to enable caching (default: True).
        
    Examples
    --------
    >>> import torch
    >>> likelihood = MultivariateT(dims=3)
    >>> data = torch.randn(100, 3)
    >>> log_prob = likelihood.pdf(data, 10, 50)
    
    Notes
    -----
    This is a more principled approach to multivariate modeling than the
    independent features model, as it properly accounts for the covariance
    structure through the multivariate t-distribution.
    """
    
    def __init__(
        self,
        dims: Optional[int] = None,
        dof0: Optional[float] = None,
        kappa0: float = 1.0,
        mu0: Optional[torch.Tensor] = None,
        Psi0: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_enabled: bool = True
    ):
        super().__init__(device, cache_enabled)
        self.dims = dims
        self.kappa0 = kappa0
        
        # Set defaults based on dimensions (will be set when first called if None)
        self.dof0 = dof0
        self.mu0 = mu0
        self.Psi0 = Psi0
    
    def _initialize_params(self, data: torch.Tensor) -> None:
        """Initialize parameters based on data dimensions."""
        if data.dim() == 1:
            data = data.unsqueeze(1)
        
        if self.dims is None:
            self.dims = data.shape[1]
        
        if self.dof0 is None:
            self.dof0 = self.dims + 1
        
        if self.mu0 is None:
            self.mu0 = torch.zeros(self.dims, device=self.device)
        else:
            self.mu0 = ensure_tensor(self.mu0, device=self.device)
        
        if self.Psi0 is None:
            self.Psi0 = torch.eye(self.dims, device=self.device)
        else:
            self.Psi0 = ensure_tensor(self.Psi0, device=self.device)
    
    def pdf(self, data: torch.Tensor, t: int, s: int) -> float:
        """
        Compute log probability using multivariate Student's t-distribution.
        
        Parameters
        ----------
        data : torch.Tensor
            Complete time series data (shape: [T] or [T, D]).
        t : int
            Start index (inclusive).
        s : int
            End index (exclusive).
            
        Returns
        -------
        float
            Log probability density for the segment.
        """
        # Check cache first
        cached_result = self._check_cache(data, t, s)
        if cached_result is not None:
            return cached_result
        
        data = ensure_tensor(data, device=self.device)
        self._initialize_params(data)
        
        # Handle univariate case
        if data.dim() == 1:
            data = data.unsqueeze(1)
        
        # Extract segment
        x = data[t:s]
        n, d = x.shape
        
        if n == 0:
            result = 0.0
            self._store_cache(t, s, result)
            return result
        
        # Update hyperparameters
        sample_mean = x.mean(dim=0)
        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * sample_mean) / kappa_n
        dof_n = self.dof0 + n
        
        # Update scale matrix
        centered = x - sample_mean.unsqueeze(0)
        S = torch.matmul(centered.T, centered)
        
        diff = sample_mean - self.mu0
        Psi_n = (
            self.Psi0 + S + 
            (self.kappa0 * n / kappa_n) * torch.outer(diff, diff)
        )
        
        # Multivariate gamma function (log)
        def multigammaln(a: torch.Tensor, p: int) -> torch.Tensor:
            result = (p * (p - 1) / 4) * torch.log(torch.tensor(torch.pi, device=self.device))
            for j in range(p):
                result += torch.lgamma(a - j / 2)
            return result
        
        # Compute log marginal likelihood for multivariate t-distribution
        try:
            logdet_Psi0 = torch.linalg.slogdet(self.Psi0)[1]
            logdet_Psi_n = torch.linalg.slogdet(Psi_n)[1]
        except RuntimeError:
            # Add regularization if matrices are not positive definite
            Psi0_reg = self.Psi0 + 1e-6 * torch.eye(d, device=self.device)
            Psi_n_reg = Psi_n + 1e-6 * torch.eye(d, device=self.device)
            logdet_Psi0 = torch.linalg.slogdet(Psi0_reg)[1]
            logdet_Psi_n = torch.linalg.slogdet(Psi_n_reg)[1]
        
        log_prob = (
            multigammaln(torch.tensor(dof_n / 2, device=self.device), d) -
            multigammaln(torch.tensor(self.dof0 / 2, device=self.device), d) +
            (self.dof0 / 2) * logdet_Psi0 -
            (dof_n / 2) * logdet_Psi_n +
            (d / 2) * torch.log(torch.tensor(self.kappa0 / kappa_n, device=self.device)) -
            (n * d / 2) * torch.log(torch.tensor(torch.pi, device=self.device))
        )
        
        result = log_prob.item()
        self._store_cache(t, s, result)
        return result