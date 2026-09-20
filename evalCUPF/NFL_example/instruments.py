"""CPA instrument definitions, and the script that writes them to a CSV.

Each entry in INSTRUMENTS maps an output column name to a callback taking
the dict of available (n_games, T) covariate matrices (COVARIATES in
main.py lists what's available) and returning one (n_games, T) array of
h(t) values. A callback can read any number of covariates.

Edit INSTRUMENTS, then run:
    python -m evalCUPF.NFL_example.instruments --model logistic --out instruments.csv
"""

import argparse

import numpy as np

from evalCUPF.build_instruments import build_instruments_csv

from .main import COVARIATES, DATA_DIR, ID_FIELD, IDS, MODEL_FILES, TIMESTEP_COL


def end_down_inv1p(cov):
    return 1.0 / (1.0 + np.abs(cov["end.down"]))


def score_diff_sq(cov):
    return cov["score_difference"] ** 2


INSTRUMENTS = {
    "end_down_inv1p": end_down_inv1p,
    "score_diff_sq": score_diff_sq,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--model", choices=list(MODEL_FILES), required=True,
                   help="Which model's panel to pull covariates from.")
    p.add_argument("--out", required=True, help="Path to write the instruments CSV.")
    args = p.parse_args(argv)

    build_instruments_csv(
        args.data_dir, MODEL_FILES, args.model, INSTRUMENTS, args.out,
        covariates=COVARIATES, ids=IDS, id_field=ID_FIELD, timestep_col=TIMESTEP_COL,
    )


if __name__ == "__main__":
    main()
