import numpy as np
import pandas as pd
import os
from typing import List
from pathlib import Path
from evalCUPF.risk_buckets import create_buckets
from evalCUPF.C_estimator import estimate_C
from evalCUPF.calculate_p_val import calculate_p_val
from evalCUPF.plot_results import plot_pcb, calc_L_s2, CovBand
from evalCUPF.entries import Entries
from .nfl_bucketer import NFLBucketer
from .nfl_heuristic_bucketer import NFLHeuristicBucketer 
from .combine_data import combine_csv_files



def run_test(dir: str, train_years: List[int], test_years: List[int], forecast_file: str, features: List[str], num_bucketers = 10, num_buckets = 3, B = 10000, phat_A = "A", phat_B = "B", save_plot=None):
    # Load training dataframes
    train_dfs = []
    for year in train_years:
        year_dir = os.path.join(dir, str(year))
        if not os.path.exists(year_dir):
            print(f"Warning: directory {year_dir} does not exist.")
            continue

        # Iterate over all CSV files in the year directory
        for filename in os.listdir(year_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(year_dir, filename)
                df = pd.read_csv(file_path)
                # Extract the value for 'home_win' from the first row
                home_win_value = df.iloc[0]['home_win']
                # Set all rows to have the same value for 'home_win'
                df['home_win'] = home_win_value
                train_dfs.append(df.iloc[1:])
    buckets = create_buckets(train_dfs, features, num_bucketers, NFLHeuristicBucketer, label_col = "home_win", n_buckets=num_buckets)
    print(f"Loaded {len(train_dfs)} dataframes from train directories.")
    entries = Entries()
    forecast_data = pd.read_csv(forecast_file)
    entries.load_entries(forecast_data, "game_completed", "phat_A", "phat_B", id_field="game_id")
    # Bucket the test data -> construct p_est
    n_timesteps = 201
    timestep_size = 0.005
    temp = np.zeros((len(entries), n_timesteps, len(features)))
    p_est = np.zeros((n_timesteps, len(entries))) # 0.005 timestep
    print("Loading test files...")
    for i in range(len(entries)):
        id = entries.get_id(i)
        file_name = "game_{}.csv".format(id)
        for year in test_years:
            year_dir = Path(dir) / str(year)
            if not year_dir.exists():
                print(f"Warning: Directory {year_dir} does not exist")
                continue
            file_path = year_dir / file_name
            if file_path.exists(): # Found the file, so load
                # Load CSV
                df = pd.read_csv(file_path)
                # Keep only relevant features + timestep column
                if 'timestep' not in df.columns:
                    raise ValueError(f"'timestep' column not found in {file_path}")
                missing_features = [f for f in features if f not in df.columns]
                if missing_features:
                    raise ValueError(f"Missing features {missing_features} in {file_path}")
                df_subset = df[['timestep'] + features].iloc[1:] # skip first row
                df_subset = df_subset[df_subset['timestep'].duplicated(keep='last') == False]
                # Place values into temp using timestep as index
                for _, row in df_subset.iterrows():
                    t_idx = int(row['timestep'] / timestep_size)  # Convert timestep to index
                    if 0 <= t_idx < n_timesteps:
                        temp[i, t_idx] = row[features].values
                break  # stop searching other years once found
        
        # For each timestep in temp, figure out the bucket assignment for each entry
        for i in range(n_timesteps):
            # print(f"Assigning buckets for timestep {round(timestep_size * i, 3)}")
            entries_at_t = temp[:, i, :]
            p_est[i] = buckets.assign_bucket(entries_at_t, round(timestep_size * i, 3), return_v=True)
    print("Loaded test files, calculating Covariance Matrix...")
    p_est = p_est.T
    # Run the test
    p_val = calculate_p_val(entries, p_est, B)
    grid_vals = np.sort(forecast_data["game_completed"].unique())
    diag_C3 = np.array([
        ((forecast_data.loc[forecast_data["game_completed"] == g, "Y"] -
          forecast_data.loc[forecast_data["game_completed"] == g, "phat_B"]) ** 2).mean()
        for g in grid_vals
    ])
    C3 = np.diag(diag_C3)
    covs = [
        CovBand(C=estimate_C(entries, p_est), label="Risk Buckets",     color="blue"),
        CovBand(C=estimate_C(entries, None),  label="Conservative",     color="black"),
        # CovBand(C=C3,                         label="True Conservative", color="green"),
    ]
    df_stats = calc_L_s2(forecast_data, covs, pA="phat_A", pB="phat_B", Y="Y", grid="game_completed")
    plot_pcb(df_stats, covs, grid="game_completed", L="L", phat_A=phat_A, phat_B=phat_B, save_plot=save_plot)
    
    return p_val

if __name__ == "__main__":
    forecast_file = "NFL/test_9/xgboost_model_combined_data.csv"
    dir = "NFL/ML/dataset_interpolated_fixed"
    combine_csv_files("xgboost_model", "test_9")
    train_years = [2021, 2022, 2023]
    test_years = [2024, 2025]
    save_plot = "NFL/test_9/plot_ESPN_xgboost_model.png"
    features = ["score_difference", "relative_strength", "end.yardsToEndzone", "end.down", "end.distance"]
    p_val = run_test(dir, train_years, test_years, forecast_file, features, num_bucketers=50, num_buckets=5, phat_A="ESPN", phat_B="XGBoost", save_plot=save_plot)
    print(p_val)