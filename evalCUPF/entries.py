from typing import Union
import numpy as np
import pandas as pd


class Entries:
    """Holds two forecast paths, outcomes, and optional CPA instruments for a
    panel of games, as (n_games, n_timesteps) numpy arrays.

    This is the single data container every test/estimator in this package
    consumes -- built once via ``load_entries`` (or the loaders in
    ``data_loading.py``), then passed directly to ``evalCUPF.tests`` and
    ``evalCUPF.covariance`` functions.
    """

    def __init__(self, timestep_size: float = 0.005):
        n_exact = 1 / timestep_size
        assert np.isclose(n_exact, round(n_exact)), "Choose a timestep_size that evenly divides the range [0,1]"
        self.timestep_size = timestep_size
        self.ids = []
        self.p_A = None
        self.p_B = None
        self.Y = None
        self.instrument_matrix = None  # (n_games, m * T), set via set_instruments
        self.n_instruments = 0
        self.true_p = None  # optional (n_games, T) proxy/true probability path
        self._n = 0

    @property
    def T_length(self) -> int:
        if self.p_A is None:
            raise ValueError("Load entries before accessing T_length.")
        return self.p_A.shape[1]

    @property
    def T(self) -> np.ndarray:
        """The [0, 1] intra-event time grid, one point per timestep column."""
        return np.linspace(0.0, 1.0, self.T_length)

    def load_entries(self, data: pd.DataFrame, timestep: str, p_A: str, p_B: str,
                      y: str = "Y", id_field: str = "id"):
        """Load entries from a long-format DataFrame, one row per (game, timestep).

        Groups by ``id_field`` and builds (n_games, n_timesteps) arrays for
        p_A, p_B and Y.
        """
        self.ids = pd.unique(data[id_field])
        games = [data[data[id_field] == id_val].reset_index(drop=True) for id_val in self.ids]

        n_timesteps = len(games[0]) if games else 0
        self.p_A = np.zeros((len(games), n_timesteps))
        self.p_B = np.zeros((len(games), n_timesteps))
        self.Y = np.zeros((len(games), n_timesteps))
        self._n = len(games)

        expected_ticks = round(1 / self.timestep_size) + 1
        for i, df in enumerate(games):
            assert len(df) == n_timesteps, (
                "Make sure that all your entries in your dataframe have the same "
                "number of timesteps."
            )
            self.p_A[i, :] = df[p_A].values
            self.p_B[i, :] = df[p_B].values
            self.Y[i, :] = df[y].values
        if n_timesteps != expected_ticks:
            import warnings
            warnings.warn(
                f"Loaded {n_timesteps} timesteps per game, which does not match "
                f"the configured timestep_size={self.timestep_size} "
                f"({expected_ticks} expected). T is derived from the data, so "
                "this is informational only.",
                RuntimeWarning,
            )

    def set_instrument_values(self, value_matrices: list):
        """Attach the CPA instrument matrix from already-computed h(t) values.

        Args:
            value_matrices: list of (n_games, T) arrays, one per instrument
                (e.g. loaded from a CSV via ``data_loading.load_instrument_matrices``).
        """
        from .instruments_helper import interleave_instrument_matrix

        self.instrument_matrix = interleave_instrument_matrix(value_matrices, len(self), self.T_length)
        self.n_instruments = len(value_matrices)
        return self.instrument_matrix

    def set_true_p(self, true_p_matrix: np.ndarray):
        """Attach a (n_games, T) true/proxy probability path, e.g. a
        midpoint-of-forecasts stand-in, for the exact-variance estimators.
        """
        self.true_p = np.asarray(true_p_matrix, dtype=float)

    def get_id(self, i: int):
        return self.ids[i]

    def __len__(self):
        return self._n

    def __getitem__(self, key: Union[tuple, int]):
        """
        Single number as key[i]: get the forecasts of game with idx i
        Tuple[i, j]: get the forecasts of game i and timestep (j * timestep_size)
        Note: Both indexing methods will return forecasts of both A and B
        """
        assert self.p_A is not None, "Load entries before indexing."
        if isinstance(key, tuple):
            i, j = key
            return (self.p_A[i, j], self.p_B[i, j], self.Y[i, j])
        return (self.p_A[key, :], self.p_B[key, :], self.Y[key, :])
