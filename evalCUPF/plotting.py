"""Plots for the test statistics and pointwise confidence bands.

Plotting is kept out of the test functions, so those stay pure -- pass their
results here.
"""

from dataclasses import dataclass
from typing import Any, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # file output; drop this line for interactive use
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.stats import norm  # noqa: E402

POS_COLOR = "purple"
NEG_COLOR = "navajowhite"


# --------------------------------------------------------------------------
# Sup-statistic / Delta(t) band plots (CPA, DM, uniform confidence set)
# --------------------------------------------------------------------------

def plot_colored_segments(x, y, colors, ax=None, lw=2, xlabel="x", ylabel="y",
                          title=None):
    """Polyline with a per-segment colour."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")
    if len(colors) != len(x) - 1:
        raise ValueError("colors must have length len(x) - 1.")

    ax = plt.gca() if ax is None else ax
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=lw))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(min(0.0, y.min()), y.max() * 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def plot_sup_statistic(T, signed_stat, crit_value, title=None, ylabel=r"$|\Gamma_n(t)|$",
                       alpha=0.05, save_path=None, ax=None):
    """Plot |statistic| against its critical value, coloured by the sign.

    Used for the DM sup test and the CPA test: purple where the statistic is
    positive, pale where negative, with the critical value as a dashed line.
    """
    T = np.asarray(T, dtype=float)
    signed_stat = np.asarray(signed_stat, dtype=float)
    magnitude = np.abs(signed_stat)
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in signed_stat[1:]]

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 5))

    plot_colored_segments(T, magnitude, colors, ax=ax, xlabel="t",
                          ylabel=ylabel, title=title)
    ax.set_ylim(0, max(crit_value, magnitude.max()) * 1.05)
    ax.axhline(crit_value, color="black", ls="--",
               label=f"Significance: {1 - alpha:g}")
    ax.legend(loc="upper left", fontsize=8)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    return ax


def plot_delta_with_band(T, Delta_n_hat, lower, upper, true_delta_n=None,
                         title=None, save_path=None, ax=None):
    """Delta_n_hat(t) with a lower/upper confidence band."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(T, Delta_n_hat, color="black", label=r"$\hat{\Delta}_n(t)$")
    ax.plot(T, lower, color="black", ls="--", label="Confidence band")
    ax.plot(T, upper, color="black", ls="--")
    if true_delta_n is not None:
        ax.plot(T, true_delta_n, color="red", label=r"$\Delta_n(t)$")
    ax.axhline(0.0, color="red", ls=":", lw=1)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Delta_n(t)$")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    return ax


# --------------------------------------------------------------------------
# Pointwise confidence bands (risk-bucket / conservative covariance overlay)
# --------------------------------------------------------------------------

def squared_error(y, p):
    return (y - p) ** 2


@dataclass
class CovBand:
    C: Any        # covariance matrix (np.ndarray or pd.DataFrame)
    label: str    # legend label for the confidence band
    color: str    # fill color for the confidence band


def calc_L_s2(df, covs: List[CovBand], pA="phat_1", pB="phat_2", Y="Y",
             grid="grid", L_func=squared_error):
    """Pointwise mean loss difference and variance for a list of CovBands."""
    df = df.copy()

    L_df = (
        df.groupby(grid)
          .apply(lambda g: np.mean(L_func(g[Y], g[pA]) - L_func(g[Y], g[pB])))
          .rename("L")
          .reset_index()
    )
    n_df = df.groupby(grid).size().rename("n").reset_index()

    grid_vals = L_df[grid].values
    sigma2_data = {grid: grid_vals}
    for cov_band in covs:
        col = f"sigma2_{cov_band.label}"
        C = cov_band.C
        diag = np.diag(C.values) if isinstance(C, pd.DataFrame) else np.diag(C)
        if len(diag) != len(grid_vals):
            raise ValueError(
                f"Covariance matrix diagonal length ({len(diag)}) does not match "
                f"number of unique grid values ({len(grid_vals)}) for '{cov_band.label}'."
            )
        sigma2_data[col] = diag
    sigma2_df = pd.DataFrame(sigma2_data)

    return L_df.merge(sigma2_df, on=grid).merge(n_df, on=grid)


def plot_pcb(df, covs: List[CovBand], grid="grid", L="L", phat_A="phat_A",
            phat_B="phat_B", save_plot=None, pad=None):
    """Pointwise confidence bands using a list of CovBands."""
    z_hi = norm.ppf(0.975)
    z_lo = norm.ppf(0.025)

    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty.")

    required = [grid, L] + [f"sigma2_{cov_band.label}" for cov_band in covs]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    bands = []
    for cov_band in covs:
        col = f"sigma2_{cov_band.label}"
        se = np.sqrt(df[col]) / np.sqrt(n)
        if se.isnull().any():
            raise ValueError(f"Standard error contains NaN values for '{cov_band.label}'.")
        bands.append((cov_band, df[L] + z_lo * se, df[L] + z_hi * se))

    y_top = max(ymax.max(skipna=True) for _, _, ymax in bands)
    y_bot = min(ymin.min(skipna=True) for _, ymin, _ in bands)
    y_range = y_top - y_bot
    if pad is None:
        pad = 0.01 * y_range

    x_vals = df[grid]
    try:
        x_pos = x_vals.max()
    except Exception:
        x_pos = x_vals.iloc[-1]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')
    sns.lineplot(data=df, x=grid, y=L, color="black", label="Mean Loss Difference")
    for cov_band, ymin, ymax in bands:
        plt.fill_between(df[grid], ymin, ymax, color=cov_band.color, alpha=0.2,
                         label=f"95% CI ({cov_band.label})")
    plt.axhline(0, color='black', linewidth=1.25, linestyle="--")
    plt.grid(True, alpha=0.5, linestyle='--', linewidth=0.5, color='black')
    plt.text(x_pos, y_bot - pad, f"{phat_A} favoured", ha='right', va='top', fontsize=12, color="black")
    plt.text(x_pos, y_top + pad, f"{phat_B} favoured", ha='right', va='bottom', fontsize=12, color="black")

    plt.xlabel(grid)
    plt.ylabel(L)
    plt.legend()
    plt.tight_layout()
    if save_plot is not None:
        plt.savefig(save_plot, dpi=300)
    plt.close()
