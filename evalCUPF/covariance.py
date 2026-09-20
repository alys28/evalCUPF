"""Covariance estimators for the forecast-comparison sup-tests.

Every estimator here produces the covariance matrix C-hat that the tests in
``tests.py`` simulate a Gaussian process from. They differ only in how much
they assume about the true win-probability path p(t):

    sigma_avg        no assumptions -- generic sample covariance of the
                      loss-difference paths (equivalent to ``C_cons``).
    sigma_quarter     conservative: uses p(1-p) <= 1/4, valid for any p.
    sigma_risk_bucket  local p(t) estimated via risk-bucket clustering on
                      observed covariates (equivalent to ``C_p_est``).
    sigma_true         exact, given a true/proxy p(t) path.
    sigma_true_v2      exact martingale form of the above, cheaper to compute.
    sigma_cpa          covariance of the instrumented loss differences, for
                      the conditional (CPA) test; no closed form, generic
                      sample covariance in the interleaved instrument layout.

Matrices are (n_timesteps, n_timesteps), except sigma_cpa's, which is
(m * n_timesteps, m * n_timesteps) in the interleaved layout described in
``instruments.py``.
"""

import numpy as np

from .entries import Entries
from .helpers import square_loss, square_loss_linear_slope


def loss_difference_matrix(entries: Entries, loss_fn=square_loss):
    """Per-game, per-timestep loss difference d_i(t) = L(Y,pA) - L(Y,pB)."""
    return loss_fn(entries.Y, entries.p_A) - loss_fn(entries.Y, entries.p_B)


def sigma_avg(entries: Entries, loss_fn=square_loss):
    """Generic sample covariance of the loss-difference paths, 1/n divisor.

    No distributional assumptions; valid for any loss. Used by the
    unconditional (Diebold-Mariano) sup test.
    """
    d_matrix = loss_difference_matrix(entries, loss_fn)
    centered = d_matrix - d_matrix.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / centered.shape[0]


def sigma_quarter(entries: Entries, loss_fn_linear_equiv_slope=square_loss_linear_slope):
    """The conservative "signs" bound.

    C[i,j] = 0.25 * mean(max(0, delta_i * delta_j))

    Uses p(1-p) <= 1/4 and drops negative cross terms, so it is valid without
    knowing the true probability path.
    """
    delta = loss_fn_linear_equiv_slope(entries.p_A) - loss_fn_linear_equiv_slope(entries.p_B)
    n_games = delta.shape[0]
    prod = delta[:, :, None] * delta[:, None, :]
    return 0.25 * np.maximum(prod, 0.0).sum(axis=0) / n_games


def sigma_risk_bucket(entries: Entries, p_est: np.ndarray):
    """Martingale variance with p(1-p) estimated from risk buckets.

    Args:
        p_est: (n_games, T) local estimate of p(t)(1-p(t)) -- already the
            variance, not p itself (see ``Bucketer.add_to_v``), so it is
            bounded by 1/4.

    Same estimator as ``sigma_true_v2``, with the bucketed p(1-p) in place of
    a supplied path: C[i,j] = mean(delta_i * delta_j * p(1-p)(t_max)), where
    t_max = max(i, j). Substituting the bound p(1-p) -> 1/4 recovers
    ``sigma_quarter``.
    """
    X = entries.p_A - entries.p_B
    n_games, n_timesteps = X.shape
    delta_bar = X.mean(axis=0)
    X_centered = X - delta_bar[None, :]

    C = np.zeros((n_timesteps, n_timesteps))
    for m in range(n_timesteps):
        Y = np.sqrt(p_est[:, m])[:, None] * X_centered
        G = Y.T @ Y
        C[m, :m] += G[m, :m]
        C[:m, m] += G[:m, m]
        C[m, m] += G[m, m]

    return C / n_games


def sigma_true_v2(entries: Entries, loss_fn_linear_equiv_slope=square_loss_linear_slope):
    """The exact martingale variance given ``entries.true_p``.

    C[i,j] = mean(delta_i * delta_j * p(t_max) * (1 - p(t_max))),
    where t_max = max(i, j).
    """
    if entries.true_p is None:
        raise ValueError("entries.true_p is not set; call entries.set_true_p(...) first.")
    delta = loss_fn_linear_equiv_slope(entries.p_A) - loss_fn_linear_equiv_slope(entries.p_B)
    n_games, T = delta.shape
    var_p = entries.true_p * (1.0 - entries.true_p)

    idx = np.arange(T)
    max_idx = np.maximum(idx[:, None], idx[None, :])

    prod = delta[:, :, None] * delta[:, None, :]
    weights = var_p[:, max_idx]
    return (prod * weights).sum(axis=0) / n_games


def sigma_true(entries: Entries, loss_fn=square_loss):
    """Covariance of the estimation noise (dhat - d), given ``entries.true_p``.

    ``dhat`` uses the realised outcome Y; ``d`` uses the true probability
    path, so the difference isolates sampling noise.
    """
    if entries.true_p is None:
        raise ValueError("entries.true_p is not set; call entries.set_true_p(...) first.")
    dhat = loss_difference_matrix(entries, loss_fn)
    d = loss_fn(entries.true_p, entries.p_A) - loss_fn(entries.true_p, entries.p_B)
    diffs = dhat - d
    centered = diffs - diffs.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / centered.shape[0]


def sigma_cpa(entries: Entries, loss_fn=square_loss):
    """Covariance of the instrumented loss differences, for the CPA test.

    Builds the (n_games, m*T) matrix of h_k(t) * d(t) in the interleaved
    layout and returns its sample covariance with a 1/n divisor.
    """
    if entries.instrument_matrix is None:
        raise ValueError("entries.instrument_matrix is not set; call entries.set_instruments(...) first.")
    d_matrix = loss_difference_matrix(entries, loss_fn)
    n_games, T = d_matrix.shape
    m = entries.instrument_matrix.shape[1] // T

    weighted = np.empty((n_games, m * T), dtype=float)
    for k in range(m):
        cols = m * np.arange(T) + k
        weighted[:, cols] = entries.instrument_matrix[:, cols] * d_matrix

    centered = weighted - weighted.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / n_games
