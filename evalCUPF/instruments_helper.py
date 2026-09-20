"""Instrument matrix mechanics for the CPA test.

Layout convention, relied on by ``covariance.sigma_cpa`` and
``tests.cpa_delta_n_hat``: instrument ``k`` at timestep ``j`` lives in column
``m * j + k`` of the (n_games, m * T) instrument matrix. Instruments are
interleaved within each timestep, not blocked by instrument.

What an instrument *is* -- which covariates, which transform -- is user
code; see ``NFL_example/instruments.py`` for that.
"""

import numpy as np


def interleave_instrument_matrix(value_matrices, n_games=None, T=None):
    """Build the interleaved (n_games, m * T) instrument matrix from
    already-computed h_k(t) values -- one (n_games, T) array per instrument.
    """
    m = len(value_matrices)
    if m == 0:
        raise ValueError("value_matrices must contain at least one instrument.")
    if n_games is None or T is None:
        n_games, T = np.shape(value_matrices[0])

    instrument_matrix = np.empty((n_games, m * T), dtype=float)
    for k, values in enumerate(value_matrices):
        values = np.asarray(values, dtype=float)
        if values.shape != (n_games, T):
            raise ValueError(f"Instrument {k} has shape {values.shape}, expected {(n_games, T)}.")
        instrument_matrix[:, m * np.arange(T) + k] = values

    return instrument_matrix
