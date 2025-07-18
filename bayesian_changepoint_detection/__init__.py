#!/usr/bin/env python3
"""
Bayesian Changepoint Detection Library.

A PyTorch-based library for Bayesian changepoint detection in time series data.
Implements both online and offline methods with GPU acceleration support.
"""

__version__ = '1.0.0'

from .device import get_device, to_tensor, ensure_tensor, get_device_info
from .bayesian_models import online_changepoint_detection, offline_changepoint_detection
from .hazard_functions import constant_hazard
from .priors import const_prior, geometric_prior, negative_binomial_prior
from .online_likelihoods import StudentT, MultivariateT
from . import online_likelihoods
from . import offline_likelihoods
from . import generate_data

__all__ = [
    'get_device',
    'to_tensor', 
    'ensure_tensor',
    'get_device_info',
    'online_changepoint_detection',
    'offline_changepoint_detection',
    'constant_hazard',
    'const_prior',
    'geometric_prior',
    'negative_binomial_prior',
    'StudentT',
    'MultivariateT',
    'online_likelihoods',
    'offline_likelihoods',
    'generate_data',
]
