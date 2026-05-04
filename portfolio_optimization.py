"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)
- Equal risk contribution portfolio (optimization-based)

References:
- Maillard, S., Roncalli, T., & Teïletche, J. (2010).
  "The properties of equally weighted risk contribution portfolios."
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

from typing import Any
import numpy as np
from functools import partial
from scipy.optimize import minimize
from utilities.covariance_utilities import (
    _validate_covariance_matrix,
    risk_contribution,
)


def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum variance portfolio weights given a covariance matrix.
    In particular the weights are given by:
    w = (Σ^(-1) * 1) / (1^T * Σ^(-1) * 1), i.e. the solution of the optimization problem:
    min_w w^T * Σ * w, subject to 1^T * w = 1.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        np.ndarray: Weights of the minimum variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    n = cov_matrix.shape[0]
    ones_vec = np.ones((n, 1))

    min_var_ptf_numerator = None  # !!! COMPLETE AS APPROPRIATE !!!
    min_var_ptf_weights = None  # !!! COMPLETE AS APPROPRIATE !!!

    return min_var_ptf_weights.flatten()


def mean_variance_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Calculate the classic mean-variance portfolio weights given expected returns and a
    covariance matrix.

    In particular the weights solve:
    max_w mu^T * w - (gamma / 2) * w^T * Sigma * w, subject to 1^T * w = 1,
    where mu are the expected returns and gamma is the risk-aversion parameter.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk-aversion parameter gamma. Must be strictly positive.

    Returns:
        np.ndarray: Weights of the mean-variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    expected_returns = np.asarray(expected_returns, dtype=float)
    if expected_returns.ndim == 2 and 1 in expected_returns.shape:
        expected_returns = expected_returns.reshape(-1)
    elif expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be one-dimensional or a single-column vector"
        )

    if expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError(
            "expected_returns and cov_matrix must refer to the same number of assets, "
            f"got {expected_returns.shape[0]} and {cov_matrix.shape[0]}"
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains NaN or Inf values")

    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")

    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be strictly positive, got {risk_aversion}"
        )

    mean_var_ptf_weights = None  # !!! COMPLETE AS APPROPRIATE !!!

    return mean_var_ptf_weights.flatten()


def inverse_volatility_portfolio(covariance: np.ndarray) -> np.ndarray:
    """
    Compute the inverse-volatility (naive risk-parity) portfolio.

    Each weight is proportional to the inverse of the asset's volatility,
    ``w_i ∝ 1 / sigma_i``, with ``sum_i w_i = 1``. Lab notes Question 2 shows
    that this coincides with the ERC solution when the assets are uncorrelated.

    Parameters:
        covariance (np.ndarray): Covariance matrix of asset returns. Only the
            diagonal is used.

    Returns:
        np.ndarray: Inverse-volatility weights summing to 1.
    """

    # !!! COMPLETE AS APPROPRIATE !!!
    pass


def erc_objective_function(weights: np.ndarray, covariance: np.ndarray) -> float:
    """
    Equal risk contribution objective function implemented as the variance of the risk
    contributions. Minimizing this function leads to equal risk contributions across assets.

    Parameters:
        weights (np.ndarray): Portfolio weights.
        covariance (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Objective function value.
    """

    non_normalized_risk_contributions = (
        np.multiply(weights.dot(covariance), weights)
    ).reshape(-1, 1)

    return len(non_normalized_risk_contributions) * np.sum(
        np.square(non_normalized_risk_contributions)
    ) - np.sum(non_normalized_risk_contributions @ non_normalized_risk_contributions.T)


def equal_risk_contribution_portfolio(
    covariance: np.ndarray,
    initial_solution: np.ndarray | None = None,
    options: dict[str, Any] | None = None,
    pcr_tolerance: float = 0.001,
    ignore_objective: bool = False,
) -> np.ndarray:
    """
    Calculate the equal risk contribution portfolio.

    Parameters:
        covariance (np.ndarray): Covariance matrix of assets, must be positive definite.
        initial_solution (np.ndarray | None): Initial solution guess, default to
            None, i.e. to the inverse volatility portfolio.
        options (Dict[str, Any] | None): A dictionary of solver options, see
            scipy.optimize.minimize.
        pcr_tolerance (float): The max allowable tolerance for differences in the percentage
            contribution to risk (pcr) coming from different assets, default to 10bps.

    Returns:
        np.ndarray: Equal risk contribution portfolio.
    """

    # !!! COMPLETE AS APPROPRIATE !!!
    pass
