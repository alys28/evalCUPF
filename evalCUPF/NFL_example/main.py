"""CLI entry point for the NFL forecast-comparison examples.

    cpa       DM and/or CPA sup-test on the merged forecast panel. CPA
              instruments are read from a CSV built by instruments.py.
    bucket    DM sup-test with a risk-bucket covariance, on year-split
              per-game CSVs.
"""

import argparse
import os

import numpy as np
import pandas as pd
from pathlib import Path

from evalCUPF.covariance import sigma_avg, sigma_risk_bucket
from evalCUPF.data_loading import load_entries, load_instrument_matrices
from evalCUPF.entries import Entries
from evalCUPF.plotting import CovBand, calc_L_s2, plot_pcb, plot_sup_statistic
from evalCUPF.risk_buckets import create_buckets
from evalCUPF.tests import p_value_from_covariance, test_h0_avg, test_h0_cpa

from .nfl_heuristic_bucketer import NFLHeuristicBucketer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

ID_FIELD = "game_id"
TIMESTEP_COL = "timestep"
Y_COL = "Y"

# Columns shared across every model file; these are the merge keys. Forecast
# columns (phat_A/phat_B) are deliberately excluded so they get suffixed per
# model on merge.
IDS = [
    "game_id", "timestep", "Y", "game_completed",
    "relative_strength", "score_difference", "home_has_possession",
    "end.down", "end.distance", "end.yardsToEndzone", "home_timeouts_left",
    "away_timeouts_left", "predicted_drive_points_ev",
]

MODEL_FILES = {
    "logistic": "LR_model_combined_data.csv",
    "ensemble": "ensemble_model_combined_data.csv",
    "lstm": "lstm_model_combined_data.csv",
    "nn": "NN_model_combined_data.csv",
    "xgboost": "xgboost_model_combined_data.csv",
    "svm": "svm_model_combined_data.csv",
    "transformer": "transformer_model_combined_data.csv",
    "rf": "random_forest_model_combined_data.csv",
}

# Covariates a CPA instrument callback may read; see instruments.py.
COVARIATES = [
    "score_difference", "predicted_drive_points_ev", "relative_strength",
    "end.down", "end.distance", "end.yardsToEndzone",
    "home_timeouts_left", "away_timeouts_left", "home_has_possession",
]


# --------------------------------------------------------------------------
# cpa subcommand
# --------------------------------------------------------------------------

def add_cpa_parser(subparsers):
    p = subparsers.add_parser("cpa", help="DM/CPA sup-tests on the merged forecast panel.")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--model", choices=list(MODEL_FILES),
                   help="Candidate model to compare against the ESPN benchmark.")
    p.add_argument("--test", choices=["dm", "cpa", "both"], default="both")
    p.add_argument("--instruments-file", help="CSV written by instruments.py. Required if --test includes cpa.")
    p.add_argument("--instrument-columns", nargs="+", default=[],
                   help="Column names to use from --instruments-file, in instrument order.")
    p.add_argument("--num-simulations", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.set_defaults(func=run_cpa)
    return p


def run_cpa(args):
    if not args.model:
        raise SystemExit("cpa: --model is required.")
    if args.test in ("cpa", "both") and not (args.instruments_file and args.instrument_columns):
        raise SystemExit("cpa: --instruments-file and --instrument-columns are required when --test includes cpa.")

    rng = np.random.default_rng(args.seed)

    label = f"ESPN vs {args.model}"
    print(f"Loading data for {label}...")
    entries, merged = load_entries(
        args.data_dir, MODEL_FILES,
        phat_a_col=f"phat_A.{args.model}", phat_b_col=f"phat_B.{args.model}",
        ids=IDS, id_field=ID_FIELD, timestep_col=TIMESTEP_COL, y_col=Y_COL,
    )
    print(f"{len(entries)} games, {entries.T_length} timesteps")

    results = {}

    if args.test in ("dm", "both"):
        dm = test_h0_avg(entries, num_simulations=args.num_simulations, alpha=args.alpha, rng=rng)
        results["dm"] = dm
        print(f"\n[DM sup test]  sup|Gamma_n| = {dm.abs_sup_Gamma_n:.4f}  "
              f"crit = {dm.crit_values:.4f}  p = {dm.p_value_approx:.4f}")

    if args.test in ("cpa", "both"):
        value_matrices = load_instrument_matrices(
            args.instruments_file, args.instrument_columns, entries, id_field=ID_FIELD)
        entries.set_instrument_values(value_matrices)
        print(f"\nInstrument matrix: {entries.instrument_matrix.shape} "
              f"(m = {len(args.instrument_columns)}: {', '.join(args.instrument_columns)})")

        cpa = test_h0_cpa(entries, num_simulations=args.num_simulations, alpha=args.alpha, rng=rng)
        results["cpa"] = cpa
        print(f"[CPA test]     sup||Gamma_n|| = {cpa.sup_norm_Gamma_hat:.4f}  "
              f"crit = {cpa.crit_values:.4f}  p = {cpa.p_value_approx:.4f}")

    if args.plot:
        os.makedirs(args.out_dir, exist_ok=True)
        if "dm" in results:
            plot_sup_statistic(
                entries.T, results["dm"].Gamma_n, results["dm"].crit_values,
                alpha=args.alpha, title=f"DM sup test: {label}",
                save_path=os.path.join(args.out_dir, f"dm_{args.model}.png"),
            )
        if "cpa" in results:
            plot_sup_statistic(
                entries.T, results["cpa"].norm_Gamma_n, results["cpa"].crit_values,
                alpha=args.alpha, title=f"CPA test: {label}",
                ylabel=r"$\|\Gamma_n(t)\|_2$",
                save_path=os.path.join(args.out_dir, f"cpa_{args.model}.png"),
            )

    return results


# --------------------------------------------------------------------------
# bucket subcommand
# --------------------------------------------------------------------------

def add_bucket_parser(subparsers):
    p = subparsers.add_parser("bucket", help="DM sup-test with a risk-bucket covariance.")
    p.add_argument("--data-dir", required=True, help="Root directory holding <year>/game_<id>.csv files.")
    p.add_argument("--forecast-file", required=True, help="Combined-data CSV with phat_A/phat_B/Y per game-timestep.")
    p.add_argument("--train-years", type=int, nargs="+", required=True)
    p.add_argument("--test-years", type=int, nargs="+", required=True)
    p.add_argument("--feature", action="append", default=[], dest="features", required=True,
                   help="Add one bucketing feature column; repeat for multiple.")
    p.add_argument("--num-bucketers", type=int, default=10)
    p.add_argument("--num-buckets", type=int, default=3)
    p.add_argument("--num-simulations", type=int, default=10_000, dest="B")
    p.add_argument("--phat-a-label", default="A")
    p.add_argument("--phat-b-label", default="B")
    p.add_argument("--save-plot", default=None)
    p.add_argument("--save-p-val", default=None)
    p.set_defaults(func=run_bucket)
    return p


def run_bucket(args):
    train_dfs = []
    for year in args.train_years:
        year_dir = os.path.join(args.data_dir, str(year))
        if not os.path.exists(year_dir):
            print(f"Warning: directory {year_dir} does not exist.")
            continue
        for filename in os.listdir(year_dir):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(year_dir, filename))
                df["home_win"] = df.iloc[0]["home_win"]
                train_dfs.append(df.iloc[1:])

    buckets = create_buckets(train_dfs, args.features, args.num_bucketers,
                             NFLHeuristicBucketer, label_col="home_win",
                             n_buckets=args.num_buckets)
    print(f"Loaded {len(train_dfs)} dataframes from train directories.")

    entries = Entries()
    forecast_data = pd.read_csv(args.forecast_file)
    entries.load_entries(forecast_data, "timestep", "phat_A", "phat_B", id_field="game_id")

    n_timesteps = 201
    timestep_size = 0.005
    temp = np.zeros((len(entries), n_timesteps, len(args.features)))
    p_est = np.zeros((n_timesteps, len(entries)))

    print("Loading test files...")
    for i in range(len(entries)):
        game_id = entries.get_id(i)
        file_name = f"game_{game_id}.csv"
        for year in args.test_years:
            year_dir = Path(args.data_dir) / str(year)
            if not year_dir.exists():
                print(f"Warning: Directory {year_dir} does not exist")
                continue
            file_path = year_dir / file_name
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "timestep" not in df.columns:
                    raise ValueError(f"'timestep' column not found in {file_path}")
                missing = [f for f in args.features if f not in df.columns]
                if missing:
                    raise ValueError(f"Missing features {missing} in {file_path}")
                df_subset = df[["timestep"] + args.features].iloc[1:]
                df_subset = df_subset[df_subset["timestep"].duplicated(keep="last") == False]
                for _, row in df_subset.iterrows():
                    t_idx = int(row["timestep"] / timestep_size)
                    if 0 <= t_idx < n_timesteps:
                        temp[i, t_idx] = row[args.features].values
                break

        for t in range(n_timesteps):
            p_est[t] = buckets.assign_bucket(temp[:, t, :], round(timestep_size * t, 3), return_v=True)

    print("Loaded test files, calculating covariance matrix...")
    p_est = p_est.T

    d_matrix = (entries.Y - entries.p_A) ** 2 - (entries.Y - entries.p_B) ** 2
    Delta_n_hat = d_matrix.mean(axis=0)
    risk_bucket_cov = sigma_risk_bucket(entries, p_est)
    result = p_value_from_covariance(entries, Delta_n_hat, risk_bucket_cov, num_simulations=args.B)
    p_val = result.p_value_approx
    print(f"\n[DM sup test, risk-bucket covariance]  p = {p_val:.4f}")

    covs = [
        CovBand(C=risk_bucket_cov, label="Risk Buckets", color="blue"),
        CovBand(C=sigma_avg(entries), label="Conservative", color="black"),
    ]
    df_stats = calc_L_s2(forecast_data, covs, pA="phat_A", pB="phat_B", Y="Y", grid="timestep")
    plot_pcb(df_stats, covs, grid="timestep", L="L", phat_A=args.phat_a_label,
            phat_B=args.phat_b_label, save_plot=args.save_plot)

    if args.save_p_val is not None:
        with open(args.save_p_val, "w") as f:
            f.write(f"{p_val}\n")
        print(f"p-value saved to: {args.save_p_val}")

    return p_val


# --------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the NFL forecast-comparison tests.")
    subparsers = parser.add_subparsers(dest="pipeline", required=True)
    add_cpa_parser(subparsers)
    add_bucket_parser(subparsers)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
