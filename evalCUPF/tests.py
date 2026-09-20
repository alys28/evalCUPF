"""The forecast-comparison sup-tests and confidence sets.

All tests take an ``Entries`` object (two forecast paths + outcomes, plus
optional instruments/true-p for the conditional and exact variants) and
compare forecast A against forecast B under a loss.

    test_h0_avg                 H0: Delta(t) = 0 for all t (unconditional,
                                 Diebold-Mariano extended to a path)
    test_h0_cpa                 H0: E[d(t) | F_t] = 0 (conditional, via
                                 instruments -- the CPA test)
    weighted_integral_test      H0: int w(t) Delta(t) dt = 0
    absolute_supremum_test      H0: sup_t |Delta(t)| <= epsilon
    uniform_confidence_set      joint band for Delta(t)
    weighted_integral_confidence_set   Wald CI for int w(t) Delta(t) dt
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

from . import covariance as cov
from .entries import Entries
from .helpers import ensure_psd, matrix_sqrt, mid_values, square_loss, square_loss_linear_slope

DEFAULT_SIMULATIONS = 10_000


def _unit_weight(t):
    return np.ones_like(np.asarray(t, dtype=float))


def _simulate_sup_abs(covariance, n_sim, rng, chunk_size=2000):
    """Monte-Carlo draws of sup_t |G(t)| for a mean-zero Gaussian process."""
    L = matrix_sqrt(covariance)
    dim = covariance.shape[0]
    out = np.empty(n_sim, dtype=float)
    for start in range(0, n_sim, chunk_size):
        size = min(chunk_size, n_sim - start)
        Z = rng.normal(size=(size, dim))
        out[start:start + size] = np.abs(Z @ L.T).max(axis=1)
    return out


def _simulate_sup_norm(covariance, n_sim, m, T, rng, chunk_size=2000):
    """Monte-Carlo draws of sup_t ||G(t)||_2 for an R^m-valued Gaussian process.

    Squares the draws, sums the m interleaved components at each timestep,
    takes sqrt of the max.
    """
    L = matrix_sqrt(covariance)
    out = np.empty(n_sim, dtype=float)
    time_cols = m * np.arange(T)
    for start in range(0, n_sim, chunk_size):
        size = min(chunk_size, n_sim - start)
        Z = rng.normal(size=(size, m * T))
        G = Z @ L.T
        G2 = G * G
        norm2_by_time = np.zeros((size, T), dtype=float)
        for k in range(m):
            norm2_by_time += G2[:, time_cols + k]
        out[start:start + size] = np.sqrt(norm2_by_time.max(axis=1))
    return out


# --------------------------------------------------------------------------
# 1. Unconditional sup test (Diebold-Mariano, extended to a path)
# --------------------------------------------------------------------------

@dataclass
class AvgTestResult:
    T: np.ndarray
    Gamma_n: np.ndarray
    Cn: np.ndarray
    abs_sup_Gamma_n: float
    sig_levels: float
    crit_values: float
    p_value_approx: float


def test_h0_avg(entries: Entries, loss_fn=square_loss,
                num_simulations=DEFAULT_SIMULATIONS, alpha=0.05, rng=None,
                covariance_matrix=None):
    """Tests H0: Delta(t) = 0 for all t.

    Statistic: sup_t sqrt(n) |Delta_n_hat(t)|, with critical values simulated
    from a mean-zero Gaussian process. By default the covariance is the
    generic sample covariance of the loss-difference paths
    (``covariance.sigma_avg``); pass ``covariance_matrix`` to use a different
    estimator (e.g. ``covariance.sigma_risk_bucket``) for the same statistic.
    """
    rng = np.random.default_rng() if rng is None else rng
    num_games = len(entries)

    d_matrix = cov.loss_difference_matrix(entries, loss_fn)
    Delta_n_hat = d_matrix.mean(axis=0)

    Cn = ensure_psd(cov.sigma_avg(entries, loss_fn) if covariance_matrix is None else covariance_matrix)
    return p_value_from_covariance(entries, Delta_n_hat, Cn, num_simulations=num_simulations,
                                   alpha=alpha, rng=rng)


def p_value_from_covariance(entries: Entries, Delta_n_hat, covariance_matrix,
                            num_simulations=DEFAULT_SIMULATIONS, alpha=0.05, rng=None):
    """The DM-style sup |Gamma_n(t)| test and p-value for an arbitrary
    covariance estimate, given a precomputed Delta_n_hat(t).

    A lower-level building block behind ``test_h0_avg``: useful when the
    covariance comes from a bespoke estimator (e.g. risk buckets) rather than
    one of the ``covariance.sigma_*`` functions.
    """
    rng = np.random.default_rng() if rng is None else rng
    num_games = len(entries)

    Gamma_n = np.sqrt(num_games) * np.asarray(Delta_n_hat, dtype=float)
    abs_sup = float(np.max(np.abs(Gamma_n)))

    Cn = ensure_psd(covariance_matrix)
    sup_draws = _simulate_sup_abs(Cn, num_simulations, rng)

    return AvgTestResult(
        T=entries.T,
        Gamma_n=Gamma_n,
        Cn=Cn,
        abs_sup_Gamma_n=abs_sup,
        sig_levels=1.0 - alpha,
        crit_values=float(np.quantile(sup_draws, 1.0 - alpha)),
        p_value_approx=float(np.mean(abs_sup <= sup_draws)),
    )


# --------------------------------------------------------------------------
# 2. Weighted integral test
# --------------------------------------------------------------------------

@dataclass
class WeightedIntegralResult:
    T: np.ndarray
    Delta_n_hat: np.ndarray
    Delta_n: np.ndarray
    Cn_hat: np.ndarray
    Cn_tilde: np.ndarray
    int_Delta_n_hat: float
    int_Delta_n: float
    overestimated_variance_trace: float
    conservative_CI_trace: np.ndarray
    pval_trace: float
    pval_trace_coverage: float
    overestimated_variance_signed: float
    conservative_CI_signed: np.ndarray
    pval_signed: float
    pval_signed_coverage: float
    variance_cheat: float
    conservative_CI_cheat: np.ndarray
    pval_cheat: float
    pval_cheat_coverage: float


def weighted_integral_test(entries: Entries, weight_fn=_unit_weight,
                           loss_fn=square_loss,
                           loss_fn_linear_equiv_slope=square_loss_linear_slope,
                           alpha=0.05):
    """Tests H0: int w(t) Delta(t) dt = 0.

    Reports three variance estimates:
      * ``trace``  -- (int w sqrt(diag C_hat))^2, the crudest bound
      * ``signed`` -- w' C_hat w, the conservative signs bound
      * ``cheat``  -- w' C_tilde w, exact but requires ``entries.true_p``

    The ``*_coverage`` p-values recentre at Delta_n_hat - Delta_n and are a
    simulation diagnostic only; they are not computable on real data.
    """
    if entries.true_p is None:
        raise ValueError("entries.true_p is not set; call entries.set_true_p(...) first.")
    T = entries.T
    num_games = len(entries)
    T_length = entries.T_length
    tick_length = 1.0 / (T_length - 1)
    w = weight_fn(T)

    d_matrix = cov.loss_difference_matrix(entries, loss_fn)
    Delta_n_hat = d_matrix.mean(axis=0)
    Delta_n = (loss_fn(entries.true_p, entries.p_A) - loss_fn(entries.true_p, entries.p_B)).mean(axis=0)

    int_Delta_n_hat = float(np.sum(tick_length * mid_values(w * Delta_n_hat)))
    int_Delta_n = float(np.sum(tick_length * mid_values(w * Delta_n)))

    Cn_hat = cov.sigma_quarter(entries, loss_fn_linear_equiv_slope)
    Cn_tilde = cov.sigma_true_v2(entries, loss_fn_linear_equiv_slope)

    z_alpha = norm.ppf(1.0 - alpha / 2.0)

    def _summary(variance):
        se = np.sqrt(variance / num_games)
        ci = int_Delta_n_hat + z_alpha * np.array([-1.0, 1.0]) * se
        pval = 2.0 * (1.0 - norm.cdf(abs(int_Delta_n_hat) / se))
        coverage = 2.0 * (1.0 - norm.cdf(abs(int_Delta_n_hat - int_Delta_n) / se))
        return float(variance), ci, float(pval), float(coverage)

    var_trace = float(np.sum(tick_length * mid_values(w * np.sqrt(np.diag(Cn_hat)))) ** 2)
    var_signed = float(tick_length ** 2 * (w @ Cn_hat @ w))
    var_cheat = float(tick_length ** 2 * (w @ Cn_tilde @ w))

    v_tr, ci_tr, p_tr, c_tr = _summary(var_trace)
    v_sg, ci_sg, p_sg, c_sg = _summary(var_signed)
    v_ch, ci_ch, p_ch, c_ch = _summary(var_cheat)

    return WeightedIntegralResult(
        T=T, Delta_n_hat=Delta_n_hat, Delta_n=Delta_n,
        Cn_hat=Cn_hat, Cn_tilde=Cn_tilde,
        int_Delta_n_hat=int_Delta_n_hat, int_Delta_n=int_Delta_n,
        overestimated_variance_trace=v_tr, conservative_CI_trace=ci_tr,
        pval_trace=p_tr, pval_trace_coverage=c_tr,
        overestimated_variance_signed=v_sg, conservative_CI_signed=ci_sg,
        pval_signed=p_sg, pval_signed_coverage=c_sg,
        variance_cheat=v_ch, conservative_CI_cheat=ci_ch,
        pval_cheat=p_ch, pval_cheat_coverage=c_ch,
    )


# --------------------------------------------------------------------------
# 3. Absolute supremum test
# --------------------------------------------------------------------------

@dataclass
class AbsSupResult:
    T: np.ndarray
    Gamma_n: np.ndarray
    Gamma_n_hat: np.ndarray
    Cn_tilde: np.ndarray
    Cn_hat: np.ndarray
    abs_sup_Gamma_n: float
    abs_sup_Gamma_n_hat: float
    sig_levels: float
    crit_values_cheat: float
    p_value_approx_cheat_true: float
    p_value_approx_cheat_feasible: float
    crit_values_overestimate: float
    p_value_approx_overestimate_true: float
    p_value_approx_overestimate_feasible: float


def absolute_supremum_test(entries: Entries, loss_fn=square_loss,
                           loss_fn_linear_equiv_slope=square_loss_linear_slope,
                           H0_epsilon=0.0, alpha=0.05,
                           num_simulations=50_000, rng=None):
    """Tests H0: sup_t |Delta(t)| <= epsilon.

    Critical values come from two covariances: ``cheat`` (exact, needs
    ``entries.true_p``) and ``overestimate`` (the conservative signs bound).
    The shared normal draws are reused across both.
    """
    if entries.true_p is None:
        raise ValueError("entries.true_p is not set; call entries.set_true_p(...) first.")
    rng = np.random.default_rng() if rng is None else rng
    T = entries.T
    num_games = len(entries)

    d_matrix = cov.loss_difference_matrix(entries, loss_fn)
    Delta_n_hat = d_matrix.mean(axis=0)
    Delta_n = (loss_fn(entries.true_p, entries.p_A) - loss_fn(entries.true_p, entries.p_B)).mean(axis=0)

    Gamma_n = np.sqrt(num_games) * (Delta_n_hat - Delta_n)
    Gamma_n_hat = np.sqrt(num_games) * Delta_n_hat

    abs_sup_Gamma_n = float(np.max(np.abs(Gamma_n)))
    abs_sup_Gamma_n_hat = float(np.max(np.abs(Gamma_n_hat)) - H0_epsilon * np.sqrt(num_games))

    Cn_tilde = ensure_psd(cov.sigma_true_v2(entries, loss_fn_linear_equiv_slope))
    Cn_hat = cov.sigma_quarter(entries, loss_fn_linear_equiv_slope)

    Z = rng.normal(size=(num_simulations, Cn_tilde.shape[0]))
    sup_cheat = np.abs(Z @ matrix_sqrt(Cn_tilde).T).max(axis=1)
    sup_over = np.abs(Z @ matrix_sqrt(Cn_hat).T).max(axis=1)

    return AbsSupResult(
        T=T, Gamma_n=Gamma_n, Gamma_n_hat=Gamma_n_hat,
        Cn_tilde=Cn_tilde, Cn_hat=Cn_hat,
        abs_sup_Gamma_n=abs_sup_Gamma_n,
        abs_sup_Gamma_n_hat=abs_sup_Gamma_n_hat,
        sig_levels=1.0 - alpha,
        crit_values_cheat=float(np.quantile(sup_cheat, 1.0 - alpha)),
        p_value_approx_cheat_true=float(np.mean(abs_sup_Gamma_n <= sup_cheat)),
        p_value_approx_cheat_feasible=float(np.mean(abs_sup_Gamma_n_hat <= sup_cheat)),
        crit_values_overestimate=float(np.quantile(sup_over, 1.0 - alpha)),
        p_value_approx_overestimate_true=float(np.mean(abs_sup_Gamma_n <= sup_over)),
        p_value_approx_overestimate_feasible=float(np.mean(abs_sup_Gamma_n_hat <= sup_over)),
    )


# --------------------------------------------------------------------------
# 4. CPA test
# --------------------------------------------------------------------------

@dataclass
class CPATestResult:
    T: np.ndarray
    Gamma_n: np.ndarray            # (m, T)
    norm_Gamma_n: np.ndarray       # (T,)
    Cn: np.ndarray                 # (m*T, m*T)
    sup_norm_Gamma_hat: float
    sig_levels: float
    crit_values: float
    p_value_approx: float


def cpa_delta_n_hat(entries: Entries, loss_fn=square_loss):
    """The (m, T) matrix of instrumented means.

    Delta_n_hat[k, t] = mean_i( h_k,i(t) * d_i(t) )
    """
    if entries.instrument_matrix is None:
        raise ValueError("entries.instrument_matrix is not set; call entries.set_instruments(...) first.")
    d_matrix = cov.loss_difference_matrix(entries, loss_fn)
    n_games, T = d_matrix.shape
    m = entries.instrument_matrix.shape[1] // T
    out = np.empty((m, T), dtype=float)
    for k in range(m):
        cols = m * np.arange(T) + k
        out[k, :] = (entries.instrument_matrix[:, cols] * d_matrix).mean(axis=0)
    return out


def test_h0_cpa(entries: Entries, loss_fn=square_loss,
                num_simulations=DEFAULT_SIMULATIONS, alpha=0.05, rng=None):
    """The Conditional Predictive Ability (CPA) test.

    Tests H0: E[d(t) | F_t] = 0 for all t, against the m instruments attached
    via ``entries.set_instruments(...)``:
        Delta_k(t) = E[h_k(t) d(t)] = 0.

    Statistic: sup_t || sqrt(n) Delta_n_hat(t) ||_2, with critical values
    simulated from a mean-zero R^m-valued Gaussian process.
    """
    if entries.instrument_matrix is None:
        raise ValueError("entries.instrument_matrix is not set; call entries.set_instruments(...) first.")
    rng = np.random.default_rng() if rng is None else rng
    num_games = len(entries)
    T_length = entries.T_length
    m = entries.instrument_matrix.shape[1] // T_length

    if entries.instrument_matrix.shape[1] != m * T_length:
        raise ValueError(
            f"Instrument matrix has {entries.instrument_matrix.shape[1]} columns, "
            f"which is not a multiple of T = {T_length}."
        )

    Delta_n_hat = cpa_delta_n_hat(entries, loss_fn)

    Gamma_n = np.sqrt(num_games) * Delta_n_hat
    norm_Gamma_n = np.linalg.norm(Gamma_n, axis=0)
    sup_norm = float(np.max(norm_Gamma_n))

    Cnh = ensure_psd(cov.sigma_cpa(entries, loss_fn))
    sup_draws = _simulate_sup_norm(Cnh, num_simulations, m, T_length, rng)

    return CPATestResult(
        T=entries.T,
        Gamma_n=Gamma_n,
        norm_Gamma_n=norm_Gamma_n,
        Cn=Cnh,
        sup_norm_Gamma_hat=sup_norm,
        sig_levels=1.0 - alpha,
        crit_values=float(np.quantile(sup_draws, 1.0 - alpha)),
        p_value_approx=float(np.mean(sup_norm <= sup_draws)),
    )


# --------------------------------------------------------------------------
# 5. Confidence sets
# --------------------------------------------------------------------------

@dataclass
class IntegralCIResult:
    Delta_n_hat: np.ndarray
    int_Delta_n_hat: float
    variance_est: float
    CI: np.ndarray
    CI_width: float


def weighted_integral_confidence_set(entries: Entries, Delta_n_hat, Sigma,
                                     weight_fn=_unit_weight, alpha=0.05):
    """Wald CI for int w(t) Delta(t) dt."""
    T = entries.T
    T_length = entries.T_length
    N = len(entries)
    tick_length = 1.0 / (T_length - 1)
    w = weight_fn(T)
    z_alpha = norm.ppf(1.0 - alpha / 2.0)

    int_Delta_n_hat = float(np.sum(tick_length * mid_values(w * Delta_n_hat)))
    variance_est = float(tick_length ** 2 * (w @ Sigma @ w))
    half_width = z_alpha * np.sqrt(variance_est / N)

    return IntegralCIResult(
        Delta_n_hat=Delta_n_hat,
        int_Delta_n_hat=int_Delta_n_hat,
        variance_est=variance_est,
        CI=int_Delta_n_hat + np.array([-half_width, half_width]),
        CI_width=float(2.0 * half_width),
    )


@dataclass
class UniformCIResult:
    T: np.ndarray
    Delta_n_hat: np.ndarray
    Delta_n_hat_lower: np.ndarray
    Delta_n_hat_upper: np.ndarray


def uniform_confidence_set(entries: Entries, Delta_n_hat, Sigma, method="Bonferroni",
                           alpha=0.05, num_simulations=DEFAULT_SIMULATIONS,
                           rng=None):
    """A band covering Delta(t) at all t jointly.

    ``method="Bonferroni"`` splits alpha across the T timesteps;
    ``method="Uniform"`` simulates the sup-norm of the Gaussian process, which
    accounts for dependence across t and gives a narrower band.
    """
    rng = np.random.default_rng() if rng is None else rng
    T = entries.T
    T_length = entries.T_length
    N = len(entries)

    if method == "Bonferroni":
        z = norm.ppf(1.0 - alpha / (2.0 * T_length))
        half = z * np.sqrt(np.diag(Sigma) / N)
    elif method == "Uniform":
        Sigma = ensure_psd(Sigma)
        sup_draws = _simulate_sup_abs(Sigma, num_simulations, rng)
        half = np.full(T_length, np.quantile(sup_draws, 1.0 - alpha) / np.sqrt(N))
    else:
        raise ValueError(f"Unknown method {method!r}; use 'Bonferroni' or 'Uniform'.")

    return UniformCIResult(
        T=T,
        Delta_n_hat=Delta_n_hat,
        Delta_n_hat_lower=Delta_n_hat - half,
        Delta_n_hat_upper=Delta_n_hat + half,
    )
