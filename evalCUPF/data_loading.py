"""Loading and reshaping long-format forecast panels into ``Entries``.

One row per (event, timestep), one file per candidate model, sharing a set
of id/covariate columns used to build CPA instruments. Column names (event
id field, forecast columns, covariate names, etc.) are dataset-specific and
passed in by the caller -- see e.g. ``NFL_example/data_loading.py`` for the
NFL column set.
"""

import os
import warnings

import numpy as np
import pandas as pd

from .entries import Entries


def multi_merge(df_dict, ids):
    """Inner-join model frames on shared id columns, suffixing the rest.

    Each frame's non-key columns get ``.<name>`` appended, so ``phat_B``
    becomes ``phat_B.logistic`` etc. ``ids`` must not include any column
    that differs per model (e.g. the forecast columns), or that model's
    values for it won't be suffixed/distinguished.
    """
    merged = None
    for name, df in df_dict.items():
        renamed = df.rename(
            columns={c: f"{c}.{name}" for c in df.columns if c not in ids}
        )
        merged = renamed if merged is None else merged.merge(renamed, on=ids)
    return merged


def load_model_files(data_dir, model_files, ids):
    """Read each model CSV and merge them on the shared id columns.

    Args:
        data_dir: directory holding the model CSV files.
        model_files: dict mapping short model name -> filename.
        ids: columns shared across every model file (the merge keys).

    Returns:
        The merged long-format DataFrame (one row per event-timestep).
    """
    frames = {}
    for name, filename in model_files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            warnings.warn(f"Missing model file, skipping: {path}", RuntimeWarning)
            continue
        frames[name] = pd.read_csv(path)
    if not frames:
        raise FileNotFoundError(f"No model files found under {data_dir}.")
    return multi_merge(frames, ids)


def _check_balanced(data, id_field):
    """Return (unique_ids, T_length), warning if the panel is unbalanced."""
    counts = data[id_field].value_counts()
    if counts.nunique() != 1:
        warnings.warn("Event ticks not equally spaced", RuntimeWarning)
    unique_ids = pd.unique(data[id_field])
    return unique_ids, int(counts.iloc[0])


def _coerce_bool_strings(col):
    """R-written CSVs store booleans as the strings "True"/"False"."""
    if col.dtype == object:
        return pd.Series(col).map(
            {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}
        ).astype(float).values
    return col


def get_covariate_matrices(data, cov_names, id_field):
    """A dict of (n_events, T) matrices, one per requested covariate."""
    unique_ids, T_length = _check_balanced(data, id_field)
    grouped = data.groupby(id_field, sort=False)

    out = {}
    for name in cov_names:
        mat = np.empty((len(unique_ids), T_length), dtype=float)
        for i, gid in enumerate(unique_ids):
            mat[i, :] = _coerce_bool_strings(grouped.get_group(gid)[name].values[:T_length])
        out[name] = mat
    return out


def entries_from_merged(merged, phat_a_col, phat_b_col, id_field, timestep_col,
                        y_col, timestep_size=0.005):
    """Build an ``Entries`` object from a merged long-format DataFrame.

    Args:
        merged: output of ``load_model_files``/``multi_merge``.
        phat_a_col: merged column name for model A's forecast.
        phat_b_col: merged column name for model B's forecast.
    """
    entries = Entries(timestep_size=timestep_size)
    entries.load_entries(merged, timestep_col, phat_a_col, phat_b_col,
                         y=y_col, id_field=id_field)
    return entries


def load_instrument_matrices(instruments_file, columns, entries, id_field):
    """Load precomputed h(t) instrument columns from a CSV, as (n_events, T)
    matrices aligned to ``entries.ids`` (see ``build_instruments.py``).

    Args:
        instruments_file: long-format CSV with columns [id_field, timestep, *columns].
        columns: names of the instrument columns to load, in instrument order.
        entries: the ``Entries`` this instrument matrix will attach to; used
            for its event order and T_length.

    Returns:
        list of (n_events, T) arrays, one per column, ready for
        ``entries.set_instrument_values``.
    """
    data = pd.read_csv(instruments_file)
    missing_cols = [c for c in columns if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Instruments file is missing columns: {missing_cols}")

    missing_events = set(entries.ids) - set(data[id_field])
    if missing_events:
        raise ValueError(
            f"Instruments file is missing {len(missing_events)} event(s) present in "
            f"entries, e.g. {list(missing_events)[:5]}."
        )

    grouped = data.groupby(id_field, sort=False)
    out = []
    for name in columns:
        mat = np.empty((len(entries.ids), entries.T_length), dtype=float)
        for i, gid in enumerate(entries.ids):
            mat[i, :] = grouped.get_group(gid)[name].values[:entries.T_length]
        out.append(mat)
    return out


def load_entries(data_dir, model_files, phat_a_col, phat_b_col, ids, id_field,
                 timestep_col, y_col, timestep_size=0.005):
    """End-to-end: read + merge model CSVs and build an ``Entries`` comparing
    two forecast columns.

    Args:
        model_files: dict mapping short model name -> filename.
        phat_a_col: merged column name for model A's forecast, e.g.
            ``"phat_A.logistic"``.
        phat_b_col: merged column name for model B's forecast, e.g.
            ``"phat_B.ensemble"``.
        ids: columns shared across every model file (the merge keys).

    Returns:
        (entries, merged_dataframe) -- the merged frame is also returned so
        covariates for CPA instruments can be pulled from it directly.
    """
    merged = load_model_files(data_dir, model_files, ids=ids)
    entries = entries_from_merged(
        merged, phat_a_col, phat_b_col, id_field=id_field,
        timestep_col=timestep_col, y_col=y_col, timestep_size=timestep_size,
    )
    return entries, merged
