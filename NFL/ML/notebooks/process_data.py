import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from models.Model import Model
from typing import List

def process_csv_file(file_path, history_length, features, label_feature, replace_nan_val, train):
    """
    Process a single CSV file and return the extracted data.
    This function will be run in parallel for each CSV file.
    """
    file_data = []
    try:
        df = pd.read_csv(file_path)
        df.loc[1:, "relative_strength"] = df.iloc[0]["homeWinProbability"]
        df.loc[1:, "away_team_id"] = df.iloc[0]["away_team_id"]
        df.loc[1:, "home_team_id"] = df.iloc[0]["home_team_id"]
        df.loc[1:, "home_win"] = df.iloc[0]["home_win"]
        
        for idx in range(1, len(df)):
            current_row = df.iloc[idx]
            label = current_row[label_feature]
            try:
                label_float = float(label)
                if np.isnan(label_float):
                    print(f"  NaN Label found in file: {file_path}")
                    break
            except (ValueError, TypeError):
                print(f"  Invalid label in file: {file_path}")
                continue
            
            # Check for NaN in current row features (safer method)
            try:
                current_row_features = current_row[features].to_numpy(dtype=np.float32)
                if np.isnan(current_row_features).any():
                    print(f"  NaN found in file: {file_path}")
                    current_row_features = np.nan_to_num(current_row_features, nan=0.0)
            except (ValueError, TypeError):
                print(f"  Invalid features in file: {file_path}")
                continue
                
            current_row_np = current_row_features.reshape(1, -1)
            
            # Check for NaN in history rows (safer method)
            if history_length > 0:
                # Get the indices based on when the sequenceNumber changes
                history_rows = []
                curr_seq_num = -1
                curr_idx = idx
                while curr_idx >= 1:
                    if df.iloc[curr_idx]["sequenceNumber"] != curr_seq_num:
                        history_rows.append(df.iloc[curr_idx][features].to_numpy(dtype=np.float32))
                        curr_seq_num = df.iloc[curr_idx]["sequenceNumber"]
                    curr_idx -= 1
                    if len(history_rows) == history_length:
                        break
                history_rows = np.array(history_rows)
                if np.isnan(history_rows).any():
                    print(f"  NaN found in file: {file_path}")
                    history_rows = np.nan_to_num(history_rows, nan=0.0)
                
                if len(history_rows) < history_length:
                    padding = np.zeros((history_length - len(history_rows), len(features)))
                    history_rows = np.concatenate([padding, history_rows], axis=0)
                
                final_rows_for_timestep = np.concatenate([history_rows, current_row_np], axis=0)
            else:
                final_rows_for_timestep = current_row_np.reshape(-1)
            
            if train:
                file_data.append({
                    "key": round(current_row["model"], 3),
                    "data": {"rows": final_rows_for_timestep, "label": label_float}
                })
            else:
                file_data.append({
                    "key": round(current_row["timestep"], 3),
                    "data": {"rows": final_rows_for_timestep, "label": label_float, "model": round(current_row["model"], 3), "homeWinProbability": current_row["homeWinProbability"]}
                })
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []
    
    return file_data

def load_data(interpolated_dir, years, history_length, features, label_feature, replace_nan_val = 0, train = True, max_workers=None):
    """
    Load data from CSV files with parallel processing for individual files.
    Years are processed sequentially, but CSV files within each year are processed in parallel.
    
    Args:
        max_workers: Number of parallel workers. If None, uses CPU count.
    """
    training_data = defaultdict(list)
    
    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming the system
    
    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        print(f"Loading data for {folder}")
        if os.path.isdir(folder_path):
            if (int(folder) in years):
                # Collect all CSV file paths for this year
                csv_files = []
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        csv_files.append(file_path)
                
                # Process CSV files in parallel
                if csv_files:
                    print(f"  Processing {len(csv_files)} CSV files in parallel with {max_workers} workers...")
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_file = {
                            executor.submit(process_csv_file, file_path, history_length, features, label_feature, replace_nan_val, train): file_path
                            for file_path in csv_files
                        }
                        
                        # Collect results as they complete
                        for future in as_completed(future_to_file):
                            file_path = future_to_file[future]
                            try:
                                file_data = future.result()
                                # Merge the results into training_data
                                for item in file_data:
                                    training_data[item["key"]].append(item["data"])
                            except Exception as e:
                                print(f"Error processing file {file_path}: {e}")
                    
                    print(f"  Completed processing {folder}")
            else: 
                print("skipping ", folder)
    training_data = dict(sorted(training_data.items()))
    return training_data


def process_csv_file_edge_case(file_path, history_length, features, label_feature, replace_nan_val, train, threshold):
    """Process a single CSV for edge-case loader (filters by timestep and score_difference)."""
    file_data = []
    try:
        df = pd.read_csv(file_path)
        
        # Set game-level constants from first row
        df.loc[1:, "relative_strength"] = df.iloc[0]["homeWinProbability"]
        df.loc[1:, "away_team_id"] = df.iloc[0]["away_team_id"]
        df.loc[1:, "home_team_id"] = df.iloc[0]["home_team_id"]
        df.loc[1:, "home_win"] = df.iloc[0]["home_win"]
        
        # ADD CRITICAL DERIVED FEATURES
        # Possession-score interactions
        df["home_winning"] = (df["score_difference"] > 0).astype(int)
        df["away_winning"] = (df["score_difference"] < 0).astype(int)
        df["tied"] = (df["score_difference"] == 0).astype(int)
        
        df["home_winning_with_ball"] = df["home_winning"] * df["home_has_possession"]
        df["home_winning_no_ball"] = df["home_winning"] * (1 - df["home_has_possession"])
        df["away_winning_with_ball"] = df["away_winning"] * (1 - df["home_has_possession"])
        df["away_winning_no_ball"] = df["away_winning"] * df["home_has_possession"]
        
        # Time remaining (more granular than game_completed)
        df["time_remaining_pct"] = 1.0 - df["game_completed"]
        
        # Score-time interaction
        df["score_diff_x_time"] = df["score_difference"] * df["time_remaining_pct"]
        
        # Points needed for trailing team
        df["abs_score_diff"] = df["score_difference"].abs()
        df["needs_fg_only"] = (df["abs_score_diff"] <= 3).astype(int)
        df["needs_td_only"] = ((df["abs_score_diff"] > 3) & (df["abs_score_diff"] <= 7)).astype(int)
        df["needs_multiple_scores"] = (df["abs_score_diff"] > 8).astype(int)
        
        # Field position value (distance to FG range ~37 yards)
        if "end.yardsToEndzone" in df.columns:
            df["yards_to_fg_range"] = (df["end.yardsToEndzone"] - 37).clip(lower=0)
            df["in_fg_range"] = (df["end.yardsToEndzone"] <= 37).astype(int)
            df["in_red_zone"] = (df["end.yardsToEndzone"] <= 20).astype(int)
        
        # Timeout advantage
        df["timeout_diff"] = df["home_timeouts_left"] - df["away_timeouts_left"]
        
        # Can run out clock (leading team with ball and sufficient time)
        # Approximate: each kneel = 40 seconds, defensive timeouts can stop clock
        if "home_has_possession" in df.columns and "home_timeouts_left" in df.columns:
            df["possessing_team_timeouts"] = np.where(
                df["home_has_possession"] == 1,
                df["home_timeouts_left"],
                df["away_timeouts_left"]
            )
            df["defending_team_timeouts"] = np.where(
                df["home_has_possession"] == 1,
                df["away_timeouts_left"],
                df["home_timeouts_left"]
            )
        
        # Update features list to include new derived features
        derived_features = [
            "home_winning_with_ball", "home_winning_no_ball",
            "away_winning_with_ball", "away_winning_no_ball",
            "tied", "time_remaining_pct", "score_diff_x_time",
            "needs_fg_only", "needs_td_only", "needs_multiple_scores",
            "timeout_diff"
        ]
        
        if "end.yardsToEndzone" in df.columns:
            derived_features.extend(["yards_to_fg_range", "in_fg_range", "in_red_zone"])
        
        # Add derived features to feature list
        all_features = features + [f for f in derived_features if f in df.columns]
        
        seen_sequence_numbers = set()
        
        for idx in range(1, len(df)):
            current_row = df.iloc[idx]
            
            # Ensure unique sequenceNumber
            seq_val = current_row["sequenceNumber"]
            seq_key = int(seq_val)
            if seq_key in seen_sequence_numbers:
                continue
            seen_sequence_numbers.add(seq_key)
            
            # Filter by minimum timestep
            try:
                timestep_val = float(current_row["timestep"])
            except Exception:
                print(f"  Invalid timestep in file: {file_path}")
                continue
            
            if timestep_val < threshold:
                continue
            
            # Filter by score_difference (keep only |score_difference| <= 7)
            if "score_difference" in df.columns:
                try:
                    sd = float(current_row["score_difference"])
                    if abs(sd) > 7:
                        continue
                except (ValueError, TypeError):
                    continue
            
            # Get label
            try:
                label_float = float(current_row[label_feature])
                if np.isnan(label_float):
                    print(f"  NaN Label found in file: {file_path}, skipping row")
                    continue
            except (ValueError, TypeError):
                print(f"  Invalid label in file: {file_path}")
                continue
            
            # Get current row features with NaN handling
            try:
                current_row_features = current_row[all_features].to_numpy(dtype=np.float32)
                if np.isnan(current_row_features).any():
                    current_row_features = np.nan_to_num(current_row_features, nan=replace_nan_val)
            except (ValueError, TypeError) as e:
                print(f"  Invalid features in file: {file_path}, error: {e}")
                continue
            
            current_row_np = current_row_features.reshape(1, -1)
            
            # Handle history
            if history_length > 0:
                start_idx = max(1, idx - history_length)
                actual_history_len = idx - start_idx
                
                try:
                    history_rows = df.iloc[start_idx:idx][all_features].to_numpy(dtype=np.float32)
                    if np.isnan(history_rows).any():
                        history_rows = np.nan_to_num(history_rows, nan=replace_nan_val)
                except (ValueError, TypeError) as e:
                    print(f"  Invalid history features in file: {file_path}, error: {e}")
                    continue
                
                # Pad if necessary
                if actual_history_len < history_length:
                    padding = np.full((history_length - actual_history_len, len(all_features)), replace_nan_val)
                    history_rows = np.concatenate([padding, history_rows], axis=0)
                
                final_rows_for_timestep = np.concatenate([history_rows, current_row_np], axis=0)
            else:
                final_rows_for_timestep = current_row_np.reshape(-1)
            
            # Build output dict
            if train:
                file_data.append({
                    "rows": final_rows_for_timestep,
                    "label": label_float
                })
            else:
                try:
                    file_data.append({
                        "rows": final_rows_for_timestep,
                        "label": label_float,
                        "model": round(float(current_row["model"]), 3),
                        "homeWinProbability": float(current_row["homeWinProbability"])
                    })
                except (KeyError, ValueError, TypeError) as e:
                    print(f"  Missing model/homeWinProbability in file: {file_path}, error: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return file_data


def load_edge_case_data(interpolated_dir, years, history_length, features, label_feature, threshold=0.0, replace_nan_val = 0, train = True, max_workers=None):
    """
    Load data from CSV files similar to `load_data` but only include rows whose
    timestep >= threshold and whose `score_difference` is between -7 and 7 (inclusive).

    Args:
        threshold: minimum timestep to include (float between 0 and 1)
        max_workers: Number of parallel workers. If None, uses CPU count (capped at 8).
    """
    training_data = []

    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)

    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        print(f"Loading data for {folder}")
        if os.path.isdir(folder_path):
            if (int(folder) in years):
                # Collect all CSV file paths for this year
                csv_files = []
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        csv_files.append(file_path)

                # Process CSV files in parallel
                if csv_files:
                    print(f"  Processing {len(csv_files)} CSV files in parallel with {max_workers} workers...")
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_file = {
                            executor.submit(process_csv_file_edge_case, file_path, history_length, features, label_feature, replace_nan_val, train, threshold): file_path
                            for file_path in csv_files
                        }

                        # Collect results as they complete
                        for future in as_completed(future_to_file):
                            file_path = future_to_file[future]
                            try:
                                file_data = future.result()
                                # Merge the results into training_data
                                for item in file_data:
                                    training_data.append(item)
                            except Exception as e:
                                print(f"Error processing file {file_path}: {e}")

                    print(f"  Completed processing {folder}")
            else:
                print("skipping ", folder)
    return training_data

def augment_data():
    pass

def feature_selection(data, features, replace_nan_val = 0):
    # Given the features of the data, return data such that each row is an array of the values of the features
    # The data is a dictionary where the key is the timestep and the value is a list of rows
    feature_data = {}
    for timestep in data:
        feature_data[timestep] = []
        for row in data[timestep]:
            new_row = [float(row[feature]) for feature in features]
            # First check if the row has any NaN values
            new_row = [val if not np.isnan(val) else replace_nan_val for val in new_row] 
            feature_data[timestep].append(new_row)
    return feature_data


def setup_models(features_data, MODEL, *args, **kwargs):
    models = {}
    for timestep in features_data:
        X = np.array(features_data[timestep])[:,1:]
        y = np.array(features_data[timestep])[:,0]
        model = MODEL(*args, **kwargs)
        
        print("Training for timestep", timestep)
        model.fit(X, y)

        # Calculate training loss
        y_pred = model.predict_proba(X)[:, 1]  # Get probability predictions
        loss = -np.mean(y * np.log(y_pred + 1e-15) + (1-y) * np.log(1-y_pred + 1e-15))  # Binary cross entropy
        accuracy = model.score(X, y)
        print(f"Timestep {timestep:.2%}: Training Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        models[timestep] = model
    return models

def setup_models_DL(features_data, MODEL, *args, **kwargs):
    '''
    Converts features into torch tensors and trains the model
    '''
    models = {}
    for timestep in features_data:
        X = torch.tensor(np.array(features_data[timestep])[:, 1:], dtype=torch.float32)
        y = torch.tensor(np.array(features_data[timestep])[:, 0])
        model = MODEL(*args, **kwargs)
        print("Training for timestep", timestep)
        model.fit(X, y)
        models[timestep] = model
    return models

# DEPRECATED
def load_test_data(interpolated_dir, test = [2023, 2024]):
    test_folders = {}
    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        print(f"Loading data for {folder}")
        if os.path.isdir(folder_path):
            if int(folder) in test:
                test_data = {}
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        seen_timesteps = set()
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_csv(file_path)
                        for _, row in df.iloc[1:].iterrows():
                            row["relative_strength"] = df.iloc[0]["homeWinProbability"]
                            row["away_team_id"] = df.iloc[0]["away_team_id"]
                            row["home_team_id"] = df.iloc[0]["home_team_id"]
                            row["home_win"] = df.iloc[0]["home_win"]
                            if file not in test_data and row["timestep"] not in seen_timesteps:
                                seen_timesteps.add(row["timestep"])
                                test_data[file] = [row] 
                            elif row["timestep"] not in seen_timesteps:
                                seen_timesteps.add(row["timestep"]) 
                                test_data[file] += [row]
                test_folders[folder] = test_data
    return test_folders

# DEPRECATED
def test_feature_selection(data, features, replace_nan_val = 0):
    # Given the features of the data, return data such that each row is an array of the values of the features
    # The data is a dictionary where the key is the timestep and the value is a list of rows
    feature_data = {}
    for file in data:
        feature_data[file] = []
        for row in data[file]:
            new_row = [float(row[feature]) for feature in features]
            # First check if the row has any NaN values
            new_row = [val if not np.isnan(val) else replace_nan_val for val in new_row] 
            feature_data[file].append(new_row)
    return feature_data

def plot_accuracy(models, test_data, title=""):
    f"""
    Test accuracy of model for each timestep on test data and plot
    Args:
    - models: Dict[float: MODEL_TYPE] where the keys are timesteps between 0 and 1
    - test_data: Dict[float: Dict[str: np.array, str: float, model: float]], coming from load_data (with train = False)
    - (Optional) title: Plot Title
    """
    accuracies = []
    timesteps = []
    for timestep in test_data.keys():
        correct = 0
        total = 0
        for entry in test_data[timestep]:
            X_test = entry["rows"]
            y_test = entry["label"]
            model = models[entry["model"]]
            accuracy = model.score(np.expand_dims(X_test, axis=0), np.expand_dims(y_test, axis=0)) # will be either 0 or 1 (since it's only 1 entry tested at a time)
            total += 1
            correct += accuracy
        accuracies.append(correct / total)
        timesteps.append(timestep)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, accuracies, color='tab:blue', label='Accuracy')
    plt.xlabel('Timestep')
    plt.ylabel('Accuracy')
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(models, test_data, title=""):
    """
    Compute BCE loss and Brier score per timestep and plot.
    - models: dict[name] -> fitted classifier with predict_proba(X) -> (n,2) probs
    - test_data: dict[timestep] -> list of entries, each entry:
        {"rows": X_row (1D array-like), "label": 0 or 1, "model": model_key}
    Requires: every model referenced must implement predict_proba.
    Returns: dict with timesteps, losses, brier_scores.
    """
    eps = 1e-15
    timesteps = sorted(test_data.keys())
    losses = []
    brier_scores = []
    seen = []

    for t in timesteps:
        entries = test_data.get(t, [])
        if not entries:
            # skip empty timesteps
            continue

        # Collect labels in the order entries are listed, and group rows by model
        rows_by_model = defaultdict(list)
        labels_by_model = defaultdict(list)
        for e in entries:
            rows_by_model[e["model"]].append(e["rows"])
            labels_by_model[e["model"]].append(e['label'])
        # Predict probabilities in batches per model (preserve ordering of rows_by_model groups)
        preds = []
        labels = []
        for model_key, rows in rows_by_model.items():
            model = models[model_key]  # KeyError if missing — intentional (no fallbacks)
            X = np.asarray(rows)
            labels.extend(labels_by_model[model_key])
            if X.ndim == 1:
                X = X.reshape(1, -1)
            probs = model.predict_proba(X)  # must exist
            # assume binary classifier: probability of class 1 is column 1
            preds.extend(probs[:, 1].tolist())
        y_preds = np.clip(np.asarray(preds, dtype=float), eps, 1 - eps)
        y_true = np.asarray(labels, dtype=float)

        if y_preds.shape[0] != y_true.shape[0]:
            raise ValueError(f"Prediction/label length mismatch at timestep {t}")

        bce = -np.mean(y_true * np.log(y_preds) + (1 - y_true) * np.log(1 - y_preds))
        brier = np.mean((y_preds - y_true) ** 2)

        losses.append(float(bce))
        brier_scores.append(float(brier))
        seen.append(t)

    # Plot
    plt.figure(figsize=(10, 6))
    try:
        x = [float(x) for x in seen]
        plt.plot(x, losses, label="BCE Loss")
        plt.plot(x, brier_scores, label="Brier Score")
        plt.xlabel("Timestep")
    except Exception:
        positions = range(len(seen))
        plt.plot(positions, losses, label="BCE Loss")
        plt.plot(positions, brier_scores, label="Brier Score")
        plt.xticks(positions, seen, rotation=45)
        plt.xlabel("Timestep")

    plt.ylabel("Loss / Score")
    plt.title(f"{title} - Loss and Brier Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(losses)
    return {"timesteps": seen, "losses": np.array(losses), "brier_scores": np.array(brier_scores)}
# Keep the original plot function for backward compatibility
def plot(models, tests, title=""):
    plot_accuracy(models, tests, title)
    plot_loss(models, tests, title)

def write_predictions(models, interpolated_dir, years, history_length, features, replace_nan_val = 0, phat_b = "phat_b"):
    # Write the predictions to csv file
    for folder in os.listdir(interpolated_dir):
        folder_path = os.path.join(interpolated_dir, folder)
        if os.path.isdir(folder_path):
            if (int(folder) in years):
                print(f"Loading data for {folder}")
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_csv(file_path)
                        df.loc[1:, "relative_strength"] = df.iloc[0]["homeWinProbability"]
                        df.loc[1:, "away_team_id"] = df.iloc[0]["away_team_id"]
                        df.loc[1:, "home_team_id"] = df.iloc[0]["home_team_id"]
                        df.loc[1:, "home_win"] = df.iloc[0]["home_win"]
                        for idx in range(1, len(df)):
                            current_row = df.iloc[idx]
                            # Get column names
                            current_row_features = current_row[features].to_numpy(dtype=np.float32)
                            current_row_features = np.nan_to_num(current_row_features, nan=replace_nan_val)
                            
                            current_row_np = current_row_features.reshape(1, -1)
                            start_idx = max(1, idx - history_length)
                            actual_history_len = idx - start_idx
                            if history_length > 0:
                                history_rows = df.iloc[start_idx:idx][features].to_numpy(dtype=np.float32)
                                history_rows = np.nan_to_num(history_rows, nan=replace_nan_val)
                                if actual_history_len < history_length:
                                    padding = np.zeros((history_length - actual_history_len, len(features)))
                                    history_rows = np.concatenate([padding, history_rows], axis=0)
                                final_rows_for_timestep = np.concatenate([history_rows, current_row_np], axis=0)
                            else:
                                final_rows_for_timestep = current_row_np.reshape(-1)
                            # Do inference
                            model_assigned = round(current_row["model"], 3)
                            model = models[model_assigned]
                            X_test = np.expand_dims(final_rows_for_timestep, axis=0)
                            pred = model.predict_proba(X_test)[0][1]
                            df.at[idx, phat_b] = pred
                        df.to_csv(file_path, index=False)
                        print("Processed file: ", file)


def assess_differences(models: List[Model], data, timestep: float, features: List[str], loss = Model.brier_loss, threshold = 0.01, alt_model = None):
    """Prints the entries that have a loss >= a given threshold
    Args:
    model: Model
    data: Output from load_data
    timestep: 0-1
    loss: loss function
    threshold: 0-1
    """
    entries_above_threshold = []
    model = models[timestep]
    total = 0
    total_diff = 0
    total_diff2 = 0
    for entry in data[timestep]:
        X_test = entry["rows"]
        y_test = entry["label"]
        y_espn = np.array(entry["homeWinProbability"])
        pred = model.predict_proba(np.array([X_test]))[:, 1]
        alt_pred = alt_model.predict_proba(np.array([X_test]))[:, 1]
        loss_val_1 = loss(y_test, y_espn)
        loss_val_2 = loss(y_test, pred)
        loss_val_3 = loss(y_test, alt_pred)
        diff = loss_val_1 - loss_val_2
        if abs(diff) >= threshold:
            diff2 = loss_val_1 - loss_val_3
            total_diff += diff
            total += 1
            total_diff2 += diff2
            entries_above_threshold.append({
                "entry": entry,
                "predicted": pred.item(),
                "alt_predicted": alt_pred.item(),
                "ESPN": y_espn,
                "diff": diff
            })
    
    print(total)
    print(total_diff / total)
    print(total_diff2 / total)
    # Convert entries to a table format
    if entries_above_threshold:
        rows = []
        for i, item in enumerate(entries_above_threshold, 1):
            row_data = []  # Add entry number
            row_data.extend(item["entry"]["rows"].flatten().tolist())
            row_data.extend([item["entry"]["label"], item["predicted"], item["alt_predicted"], item["ESPN"], item["diff"]])
            rows.append(row_data)
        
        columns = features.copy()
        columns.extend(["label", "predicted", "alt_predicted", "ESPN", "diff"])
        
        df_results = pd.DataFrame(rows, columns=columns)
        df_results = df_results.sort_values(by="diff", ascending=True)
        df_results.reset_index(drop=True, inplace=True)
        print(df_results.to_string())
    else:
        print("No entries above threshold")

    
