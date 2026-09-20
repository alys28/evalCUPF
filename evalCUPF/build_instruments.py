"""Materialize CPA instrument h(t) values into a long-format CSV.
See ``NFL_example/instruments.py`` for a caller that defines its callbacks
and writes the CSV.
"""

import pandas as pd

from .data_loading import get_covariate_matrices, load_model_files


def build_instruments_csv(data_dir, model_files, model, instruments, out_path,
                          covariates, ids, id_field, timestep_col):
    """Compute each instrument callback and write a long-format CSV.

    Args:
        model_files: dict mapping short model name -> filename.
        model: key into ``model_files``; its panel supplies the covariates
            and the (event, timestep) row set.
        instruments: dict mapping output column name -> callback taking the
            dict of (n_events, T) covariate matrices and returning one
            (n_events, T) array of h(t) values.
        covariates: covariate names to make available to the callbacks.
        ids: columns shared across every model file (the merge keys).
    """
    merged = load_model_files(data_dir, {model: model_files[model]}, ids=ids)
    cov_matrices = get_covariate_matrices(merged, covariates, id_field=id_field)

    unique_ids = pd.unique(merged[id_field])
    first_event = merged.groupby(id_field, sort=False).get_group(unique_ids[0])
    T_length = len(first_event)

    out = pd.DataFrame({
        id_field: pd.Series(unique_ids).repeat(T_length).values,
        timestep_col: list(first_event[timestep_col]) * len(unique_ids),
    })
    for name, callback in instruments.items():
        values = callback(cov_matrices)
        if values.shape != (len(unique_ids), T_length):
            raise ValueError(
                f"Instrument {name!r} returned shape {values.shape}, "
                f"expected {(len(unique_ids), T_length)}."
            )
        out[name] = values.reshape(-1)

    out.to_csv(out_path, index=False)
    print(f"Wrote {len(instruments)} instrument column(s) for {len(unique_ids)} events to {out_path}")