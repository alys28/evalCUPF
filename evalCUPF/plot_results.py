import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Any

def squared_error(y, p):
    return (y - p) ** 2


@dataclass
class CovBand:
    C: Any        # covariance matrix (np.ndarray or pd.DataFrame)
    label: str    # legend label for the confidence band
    color: str    # fill color for the confidence band


def calc_L_s2(
    df,
    covs: List[CovBand],
    pA="phat_1",
    pB="phat_2",
    Y="Y",
    grid="grid",
    L_func=squared_error
):
    """
    Calculate pointwise mean loss difference and variance
    using a list of CovBand inputs.
    """

    df = df.copy()

    # Pointwise mean loss difference
    L_df = (
        df.groupby(grid)
          .apply(lambda g: np.mean(L_func(g[Y], g[pA]) - L_func(g[Y], g[pB])))
          .rename("L")
          .reset_index()
    )

    # Sample size per grid
    n_df = (
        df.groupby(grid)
          .size()
          .rename("n")
          .reset_index()
    )

    # Extract diagonal of each covariance matrix into its own column
    # Use the grouped grid values (from L_df) to ensure length matches np.diag(C)
    grid_vals = L_df[grid].values
    sigma2_data = {grid: grid_vals}
    for cov in covs:
        col = f"sigma2_{cov.label}"
        C = cov.C
        diag = np.diag(C.values) if isinstance(C, pd.DataFrame) else np.diag(C)
        if len(diag) != len(grid_vals):
            raise ValueError(
                f"Covariance matrix diagonal length ({len(diag)}) does not match "
                f"number of unique grid values ({len(grid_vals)}) for '{cov.label}'."
            )
        sigma2_data[col] = diag
    sigma2_df = pd.DataFrame(sigma2_data)

    # Merge everything
    out = (
        L_df
        .merge(sigma2_df, on=grid)
        .merge(n_df, on=grid)
    )

    return out


def plot_pcb(df, covs: List[CovBand], grid="grid", L="L", phat_A="phat_A", phat_B="phat_B", save_plot=None, pad=None):
    """
    Plotter for point-wise confidence bands using a list of CovBand inputs.
    """
    z_hi = norm.ppf(0.975)
    z_lo = norm.ppf(0.025)

    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty.")

    required = [grid, L] + [f"sigma2_{cov.label}" for cov in covs]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # Compute CI bounds for each CovBand
    bands = []
    for cov in covs:
        col = f"sigma2_{cov.label}"
        se = np.sqrt(df[col]) / np.sqrt(n)
        if se.isnull().any():
            raise ValueError(f"Standard error contains NaN values for '{cov.label}'.")
        bands.append((cov, df[L] + z_lo * se, df[L] + z_hi * se))

    # Compute label positions
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

    # Create plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')
    sns.lineplot(data=df, x=grid, y=L, color="black", label="Mean Loss Difference")
    for cov, ymin, ymax in bands:
        plt.fill_between(df[grid], ymin, ymax, color=cov.color, alpha=0.2, label=f"95% CI ({cov.label})")
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
    plt.show()
