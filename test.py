from functools import partial

import numpy as np
from scipy.stats import multivariate_normal, norm

from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.online_likelihoods import StudentT, MultivariateT


def test_multivariate():
    np.random.seed(seed=34)
    # 10-dimensional multivariate normal, that shifts its mean at t=50, 100, and 150
    dataset = np.vstack((
        multivariate_normal.rvs([0] * 10, size=50),
        multivariate_normal.rvs([4] * 10, size=50),
        multivariate_normal.rvs([0] * 10, size=50),
        multivariate_normal.rvs([-4] * 10, size=50)
    ))
    r, maxes = online_changepoint_detection(
        dataset,
        partial(constant_hazard, 50),
        MultivariateT(dims=10)
    )

    # Assert that we detected the mean shifts
    # Check that changepoint probability is higher at breakpoints than nearby points
    for brkpt in [50, 100, 150]:
        # Find the maximum changepoint probability in a window around the breakpoint
        window_start = max(0, brkpt - 5)
        window_end = min(len(maxes) - 1, brkpt + 5)
        window_values = maxes[window_start:window_end+1]
        max_in_window = window_values.max().item()
        max_idx = window_start + window_values.argmax().item()
        # Assert that the detected changepoint is within 5 time steps of the true changepoint
        assert abs(max_idx - brkpt) <= 5


def test_univariate():
    np.random.seed(seed=34)
    # 10-dimensional univariate normal
    dataset = np.hstack((norm.rvs(0, size=50), norm.rvs(2, size=50)))
    r, maxes = online_changepoint_detection(
        dataset,
        partial(constant_hazard, 20),
        StudentT(0.1, .01, 1, 0)
    )
    # Check that there's a significant drop in probability after the changepoint
    # The values are probabilities (0-1), not raw numbers
    assert maxes[50] / maxes[51] > 40  # Ratio test instead of difference
