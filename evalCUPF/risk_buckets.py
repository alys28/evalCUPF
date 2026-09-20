from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any
import os
import pandas as pd
import bisect

class Bucketer(ABC):
    """
    Abstract class for a risk bucket strategy. One Bucketer object must be created per timestep.
    """
    def __init__(self, features: List[str], data: np.ndarray | Any, labels: np.ndarray, start, end):
        """
        Note: the order of the features is assumed to be the same as the data's order feature-wise. Order of labels and data is also assumed to be the same.
        """
        self.buckets = {}
        self.start = start
        self.end = end
        self.features = features
        self.v = {} # estimator for each bucket
        self._preprocess_strategy(data, labels)

    @abstractmethod
    def _preprocess_strategy(self, data, labels):
        """
        Strategy to organize given data into bucket groups.
        """
    
    def assign_bucket(self, X: np.ndarray, return_v: bool = True) -> np.ndarray:
        """
        Call the score method, then keep the bucket with max score
        Args:
        X: 2D input array (n_entries, n_features)
        return_v: should return estimator, otherwise return bucket name
        Return: 1D output array (n_entries,), where each entry at idx i is the best fit bucket for the input at idx i.
        """
        scores = self.score(X)
        best_bucket_indices = np.argmax(scores, axis=1)
        bucket_names = list(self.buckets.keys())
        best_buckets = np.array([bucket_names[i] for i in best_bucket_indices])
        if return_v:
            return np.array([self.v[bucket] for bucket in best_buckets])
        return best_buckets
    
    def add_to_v(self, bucket_name: str, y_mean_t, n_j_t):
        """
        Get the unbiased estimate of p(1-p), as described in https://arxiv.org/pdf/1202.5140
        """
        if n_j_t <= 1:
            self.v[bucket_name] = 0.0
        else:
            self.v[bucket_name] = n_j_t / (n_j_t - 1) * y_mean_t * (1 - y_mean_t)

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Get comparison score for each bucket
        Args:
        X: 2D input array
        Return: 2D output array, where each entry at idx i is an array of scores of each bucket for the input at idx i.
        """
    
class BucketContainer:
    def __init__(self):
        self._intervals = [] # (start, end, bucket)

    def add_bucket_interval(self, start: float, end: float, bucketer: Bucketer):
        assert start <= end and 0 <= start and end <= 1, "Intervals must be 0 <= start <= end <= 1."
        # Make sure that the range is non-overlapping with the existing intervals
        self._intervals.sort(key=lambda x: x[0]) # sort by start
        lower_bound_idx = bisect.bisect_left(self._intervals, start, key = lambda x: x[0])
        if lower_bound_idx > 0:
            prev_start, prev_end, _ = self._intervals[lower_bound_idx - 1]
            if start <= prev_end: # Check for left overlap
                raise ValueError(
                    f"Intervals must not overlap. "
                    f"Input [{start}, {end}) overlaps with [{prev_start}, {prev_end})"
                )
        if lower_bound_idx < len(self._intervals):
            if self._intervals[lower_bound_idx][0] <= end:
                raise ValueError(f"Intervals must not overlap. Input [{start}, {end}] overlap with {self._intervals[lower_bound_idx]}")
        self._intervals.insert(lower_bound_idx, (start, end, bucketer))

    def assign_bucket(self, X: np.ndarray, t: float, return_v=True) -> np.ndarray:
        for start, end, bucketer in self._intervals:
            if start <= t <= end:
                return bucketer.assign_bucket(X, return_v)
        raise KeyError(f"No bucket interval contains t={t}")


def create_buckets(
    df_lst: List[pd.DataFrame],
    features: List[str],
    num_bucketers: int,
    BucketerCls: type,
    label_col: str,
    timestep_col="timestep",
    *args,
    **kwargs
):
    """
    Load CSVs, split data into intervals, and create Bucketers per interval.
    """
    assert num_bucketers <= 10000, "Max num_bucketers is 10000."
    dfs = []
    for df in df_lst:
        cols_needed = [timestep_col] + features + [label_col]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns {missing} in {df.head()}")
        dfs.append(df[cols_needed])

    all_df = pd.concat(dfs, ignore_index=True)
    container = BucketContainer()
    EPS = 1e-10
    # Create intervals (e.g., equal-length buckets in [0,1])
    interval_edges = np.linspace(0, 1, num_bucketers + 1)
    for i in range(num_bucketers):
        start = round(interval_edges[i], 5)
        end = round(interval_edges[i + 1], 5) - (EPS if i < num_bucketers - 1 else 0)
        # Filter rows belonging to this interval
        mask = (all_df[timestep_col] >= start) & (all_df[timestep_col] <= end)
        interval_features = all_df.loc[mask, features].to_numpy()
        interval_labels = all_df.loc[mask, [label_col]].to_numpy()
        if interval_features.size == 0:
            continue  # skip empty intervals

        # Create Bucketer WITH labels
        bucketer = BucketerCls(
            features,
            interval_features,
            interval_labels,
            start,
            end,
            *args,
            **kwargs
        )
        container.add_bucket_interval(start, end, bucketer)

    return container
