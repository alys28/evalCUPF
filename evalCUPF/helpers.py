"""Numeric helpers: loss functions, matrix utilities, and a Brownian-motion
simulation testbed for generating synthetic true-probability paths.
"""

import numpy as np
from scipy.stats import norm


def clip_values(x, lower=0.0, upper=1.0):
    return np.clip(x, lower, upper)


def mid_values(x):
    """Midpoints of consecutive entries, length n-1."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (x[:-1] + x[1:])


def matrix_sqrt(A):
    """Symmetric square root via SVD.

    Returns L with L @ L.T == A for symmetric PSD A, so that L @ z has
    covariance A when z ~ N(0, I).
    """
    U, d, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    return (U * np.sqrt(d)[None, :]) @ Vt


def is_positive_semi_definite(A, tol=1e-12):
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T, atol=1e-10):
        return False
    return np.all(np.linalg.eigvalsh(0.5 * (A + A.T)) >= -tol)


def make_psd(A):
    """Clip negative eigenvalues to zero."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < 0.0] = 0.0
    return (eigvecs * eigvals[None, :]) @ eigvecs.T


def ensure_psd(A, name="Covariance matrix"):
    """Warn-and-repair guard used before every covariance matrix square root."""
    if not is_positive_semi_definite(A):
        import warnings

        warnings.warn(f"{name} not positive definite.", RuntimeWarning)
        return make_psd(A)
    return A


# --------------------------------------------------------------------------
# Loss functions
# --------------------------------------------------------------------------

def square_loss(p, p_hat):
    """Brier-equivalent loss: 0.5 * (p - p_hat)^2."""
    return 0.5 * (np.asarray(p, dtype=float) - np.asarray(p_hat, dtype=float)) ** 2


def square_loss_linear(p, p_hat):
    """The part of square_loss that is linear in p."""
    p_hat = np.asarray(p_hat, dtype=float)
    return -p_hat * np.asarray(p, dtype=float) + 0.5 * p_hat ** 2


def square_loss_linear_slope(p_hat):
    """d/dp of the linear-equivalent loss."""
    return -np.asarray(p_hat, dtype=float)


# --------------------------------------------------------------------------
# Brownian-motion simulation testbed (synthetic ground truth for verification)
# --------------------------------------------------------------------------

def brownian_motion_x(t, mu=0.0, sigma=1.0, X0=0.0, rng=None):
    """One BM path sampled on the grid ``t``."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.asarray(t, dtype=float)
    increments = rng.normal(0.0, sigma * np.sqrt(np.diff(t)))
    return mu * t + np.concatenate([[X0], X0 + np.cumsum(increments)])


def brownian_motion(t, mu=0.0, sigma=1.0, X0=0.0, rng=None):
    t = np.asarray(t, dtype=float)
    return {"t": t, "x": brownian_motion_x(t, mu, sigma, X0, rng)}


def true_p(t, x, mu=0.0, sigma=1.0):
    """P(X_1 < 0 | X_t = x) for BM with drift mu."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    out = norm.cdf(
        np.divide(
            -mu * (1.0 - t) - x,
            sigma * np.sqrt(np.maximum(1.0 - t, 0.0)),
            out=np.full(np.broadcast(t, x).shape, np.inf, dtype=float),
            where=(t < 1.0),
        )
    )
    return np.where(t == 1.0, np.where(x < 0.0, 1.0, 0.0), out)


def true_p_partial(t, x, mu=0.0, sigma=1.0, w=(1.0, 1.0)):
    """True p under partial information with weights w."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    denom = sigma * np.sqrt((1.0 - t) * w[0] ** 2 + w[1] ** 2)
    return norm.cdf((-mu * (1.0 - t) - x) / denom)
