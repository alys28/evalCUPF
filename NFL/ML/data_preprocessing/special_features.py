import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def run_inference(df: pd.DataFrame, model, features: list, column_name: str, ev_x_values: list = None) -> pd.DataFrame:
    """
    Runs inference on a DataFrame using only the specified feature columns,
    and returns the DataFrame with predictions appended as a new column.

    Parameters
    ----------
    df : pd.DataFrame
    model : fitted sklearn model with a predict() method
    features : list of str
        Column names to extract as input features, in the expected order.
    column_name : str
        Name of the new column to add with the predictions.
    ev_x_values : list of float, optional
        Representative point values per scoring bucket (e.g. [1, 4, 7]).
        If provided, adds a `{column_name}_ev` column with expected points.

    Returns
    -------
    pd.DataFrame with added columns for per-bucket probabilities and optionally expected points.
    """
    df = df.copy()
    X = df[features].astype(float).fillna(0).to_numpy()
    proba = model.predict_proba(X)
    sign = np.where(df["home_has_possession"].astype(bool), 1, -1)
    for i, bucket in enumerate(SCORE_BUCKET_NAMES):
        df[f"{column_name}_{bucket}"] = proba[:, i] * sign
    if ev_x_values is not None:
        df[f"{column_name}_ev"] = expected_points(proba, ev_x_values) * sign
    return df


def load_dataset(directory: str, features: list) -> dict:
    """
    Loads all CSVs from every year subfolder in `directory`, keeps only the
    given feature columns, and returns a dict mapping each year to a numpy array.

    Parameters
    ----------
    directory : str
        Root dataset directory (e.g. '.../dataset_interpolated_fixed').
    features : list of str
        Column names to keep, in the desired output order.

    Returns
    -------
    dict[int, np.ndarray]
        Keys are years (int), values are arrays of shape (n_rows, len(features)).
    """
    year_dirs = sorted(
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d.isdigit()
    )
    if not year_dirs:
        raise FileNotFoundError(f"No year subdirectories found in {directory}")

    result = {}
    for year_str in year_dirs:
        year_dir = os.path.join(directory, year_str)
        dfs = []
        for fname in sorted(os.listdir(year_dir)):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(year_dir, fname), skiprows=[1])
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise KeyError(f"{fname} (year {year_str}) is missing columns: {missing}")
            dfs.append(df[features].astype(float))

        if dfs:
            separator = np.zeros((1, len(features)))
            separated = [arr for df in dfs for arr in (df[features].to_numpy(), separator)]
            result[int(year_str)] = np.concatenate(separated[:-1], axis=0)  # drop trailing separator
    return result


def annotate_possessions(data: dict, possession_index: int, score_difference_index: int) -> dict:
    """
    Adds a column to each year's array indicating how many points the team
    currently in possession scores during that drive.

    A drive is a contiguous sequence of rows sharing the same possession value
    (1 = home, 0 = away). The points scored for the drive is:
        delta = score_difference[last_row] - score_difference[first_row]
    Every row in the drive gets this same delta value.
    Separator rows (all zeros) get 0.

    Parameters
    ----------
    data : dict[int, np.ndarray]
        Output of load_dataset — keys are years, values are 2D arrays.
    possession_index : int
        Column index for the possession indicator (1 = home, 0 = away).
    score_difference_index : int
        Column index for score difference (home_score - away_score).

    Returns
    -------
    dict[int, np.ndarray]
        Same structure with one extra column appended.
    """
    result = {}
    for year, arr in data.items():
        print(f"Annotating possessions for year {year} ...")
        n_rows = arr.shape[0]
        points_scored = np.zeros(n_rows)

        i = 0
        while i < n_rows:
            # Separator row — leave as 0 and skip
            if np.all(arr[i] == 0):
                i += 1
                continue

            possession = arr[i, possession_index]
            drive_start = i
            # Advance until possession changes, a separator is hit, or end of array
            while i < n_rows and not np.all(arr[i] == 0) and arr[i, possession_index] == possession:
                i += 1

            drive_end = i - 1
            delta = arr[drive_end, score_difference_index] - arr[drive_start, score_difference_index]
            points_scored[drive_start:drive_end] = delta  # drive_end (scoring play) stays 0

        result[year] = np.concatenate([arr, points_scored.reshape(-1, 1)], axis=1)

    return result


N_SCORE_CLASSES = 3
SCORE_BUCKET_NAMES = ["prob_0_2pts", "prob_3_5pts", "prob_6plus_pts"]


def verify_sanity_check(df: pd.DataFrame, model, features: list, ev_x_values: list, ground_truth: np.ndarray = None, n_samples: int = 10, random_state: int = 42) -> None:
    """
    Samples random rows from df and prints feature values, per-bucket probabilities, expected points, and ground truth.

    Parameters
    ----------
    ground_truth : np.ndarray of shape (len(df),), optional
        Raw drive points scored for each row (e.g. the last column of the annotated array).
        If provided, prints the actual points and bucket alongside predictions.
    """
    sample = df.sample(n=min(n_samples, len(df)), random_state=random_state)
    sample_idx = sample.index.tolist()
    sample = sample.reset_index(drop=True)
    X = sample[features].astype(float).fillna(0).to_numpy()
    proba = model.predict_proba(X)
    sign = np.where(sample["home_has_possession"].astype(bool), 1, -1)
    ev = expected_points(proba, ev_x_values) * sign

    for i in range(len(sample)):
        print(f"\n--- Sample {i + 1} ---")
        for feat in features:
            print(f"  {feat}: {sample.at[i, feat]}")
        for j, bucket in enumerate(SCORE_BUCKET_NAMES):
            print(f"  {bucket}: {proba[i, j] * sign[i]:.4f}")
        print(f"  expected_points: {ev[i]:.4f}")
        if ground_truth is not None:
            actual_pts = ground_truth[sample_idx[i]]
            actual_bucket = _bucket_labels(np.array([abs(actual_pts)]))[0]
            print(f"  ground_truth_points: {actual_pts:.1f}  (bucket: {SCORE_BUCKET_NAMES[actual_bucket]})")


def expected_points(probabilities: np.ndarray, x_values: list) -> np.ndarray:
    """
    Computes expected points as a weighted sum: EV = sum(P(score=x) * x).

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_samples, n_classes)
        Class probabilities from XGBoost's predict_proba(), one column per bucket.
    x_values : list of float, length n_classes
        Representative point value for each bucket (e.g. [1, 4, 7]).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Expected points for each sample.
    """
    x = np.array(x_values, dtype=float)
    return probabilities @ x


def _bucket_labels(y: np.ndarray) -> np.ndarray:
    """0-2 pts -> 0, 3-5 pts -> 1, 6+ pts -> 2"""
    buckets = np.zeros(len(y), dtype=int)
    buckets[(y >= 3) & (y < 6)] = 1
    buckets[y >= 6] = 2
    return buckets


def train_possession_model(annotated_data: dict, train_years: list, model_type: str = "xgboost", use_smote: bool = False, use_sample_weights: bool = True):
    """
    Trains a classifier on the given years. The last column of each
    array is the drive points label, bucketed into: 0-2pts, 3-5pts, 6+pts.

    Parameters
    ----------
    annotated_data : dict[int, np.ndarray]
        Output of annotate_possessions.
    train_years : list of int
        Years to include in training.
    model_type : str
        One of "xgboost" or "random_forest".
    use_smote : bool
        If True, applies SMOTE to oversample minority classes before training.
        Overrides sample weights (sets them all to 1) since classes are already balanced.
    use_sample_weights : bool
        If True, applies inverse-frequency sample weights to penalize majority class.
        Ignored when use_smote=True.

    Returns
    -------
    model : fitted classifier
    metrics : dict with 'train_accuracy', 'test_accuracy', and 'report'
    """
    arrays = [annotated_data[y] for y in train_years if y in annotated_data]
    if not arrays:
        raise ValueError(f"None of the requested train_years found in data: {train_years}")

    combined = np.concatenate(arrays, axis=0).astype(float)
    n_nan = np.isnan(combined).sum()
    if n_nan > 0:
        print(f"      [train_possession_model] Replacing {n_nan} NaN value(s) with 0 before training.")
    combined = np.nan_to_num(combined, nan=0.0)
    X = combined[:, :-1]
    y = _bucket_labels(combined[:, -1])

    counts = np.bincount(y, minlength=N_SCORE_CLASSES)
    print(f"      Label distribution: { {name: int(c) for name, c in zip(SCORE_BUCKET_NAMES, counts)} }")

    if use_sample_weights and not use_smote:
        sample_weights = np.array([1.0 / counts[label] for label in y])
        sample_weights /= sample_weights.mean()
    else:
        sample_weights = np.ones(len(y))

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

    if use_smote:
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        w_train = np.ones(len(y_train))
        resampled_counts = np.bincount(y_train, minlength=N_SCORE_CLASSES)
        print(f"      After SMOTE: { {name: int(c) for name, c in zip(SCORE_BUCKET_NAMES, resampled_counts)} }")

    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X_train, y_train, sample_weight=w_train)
    elif model_type == "xgboost":
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              eval_metric="mlogloss", random_state=42)
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'xgboost' or 'random_forest'.")
    metrics = {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
        "report": classification_report(y_test, model.predict(X_test),
                                        target_names=SCORE_BUCKET_NAMES, zero_division=0),
    }
    return model, metrics


def run_logistic_regression(X: np.ndarray, y: np.ndarray, test_size: float = 0.05, random_state: int = 42):
    """
    Fits a logistic regression model on X and y.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) — binary or multiclass labels
    test_size : fraction of data held out for evaluation
    random_state : seed for reproducibility

    Returns
    -------
    model : fitted LogisticRegression
    metrics : dict with 'train_accuracy' and 'test_accuracy'
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    metrics = {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
    }
    return model, metrics


def run_linear_regression(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Fits a linear regression model on X and y.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) — continuous target values
    test_size : fraction of data held out for evaluation
    random_state : seed for reproducibility

    Returns
    -------
    model : fitted LinearRegression
    metrics : dict with 'train_r2' and 'test_r2'
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = {
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, model.predict(X_test)),
    }
    return model, metrics


def _infer_and_save(fpath: str, model, features: list, prediction_column: str, ev_x_values: list = None) -> bool:
    full_df = pd.read_csv(fpath)
    skipped_row = full_df.iloc[[0]]       # preserve the first data row
    df = full_df.iloc[1:].reset_index(drop=True)
    had_nan = df[features].astype(float).isna().any().any()
    df = run_inference(df, model, features, prediction_column, ev_x_values)
    # re-attach the skipped row (prediction columns will be NaN for it)
    result = pd.concat([skipped_row, df], ignore_index=True)
    result.to_csv(fpath, index=False)
    return had_nan


def _run_inference_on_year(args):
    directory, year, model, features, prediction_column, ev_x_values = args
    year_dir = os.path.join(directory, str(year))
    if not os.path.isdir(year_dir):
        return year, 0, 0
    csv_files = sorted(f for f in os.listdir(year_dir) if f.endswith(".csv"))
    nan_file_count = 0
    with ThreadPoolExecutor() as io_pool:
        futures = [io_pool.submit(_infer_and_save, os.path.join(year_dir, f), model, features, prediction_column, ev_x_values) for f in csv_files]
        for fut in as_completed(futures):
            if fut.result():
                nan_file_count += 1
    print(f"      Year {year} | {nan_file_count}/{len(csv_files)} files had NaN values (replaced with 0)")
    return year, len(csv_files), nan_file_count


def _loo_train_and_infer(args):
    directory, year, train_years, annotated, features, prediction_column, ev_x_values, model_type, use_smote, use_sample_weights = args
    if year not in annotated:
        return year, None, None, 0
    other_train_years = [y for y in train_years if y != year and y in annotated]
    if not other_train_years:
        return year, None, None, 0
    model, metrics = train_possession_model(annotated, other_train_years, model_type, use_smote, use_sample_weights)
    _, n_files, _ = _run_inference_on_year((directory, year, model, features, prediction_column, ev_x_values))
    return year, other_train_years, metrics, n_files


def run_pipeline(
    directory: str,
    train_years: list,
    test_years: list,
    features: list,
    possession_index: int,
    score_difference_index: int,
    prediction_column: str = "predicted_drive_points",
    ev_x_values: list = None,
    model_type: str = "xgboost",
    use_smote: bool = False,
    use_sample_weights: bool = True,
):
    """
    Full pipeline:

    Train years (leave-one-out):
      For each year Y in train_years, train a model on all other train years,
      then run inference on every CSV in Y and save it in-place with a new column.

    Test years:
      Train a model on all train_years combined, then run inference on every
      CSV in each test year and save it in-place with a new column.

    Parameters
    ----------
    directory : str
        Root dataset directory containing year subfolders.
    train_years : list of int
    test_years : list of int
    features : list of str
        Feature columns used for both training and inference.
    possession_index : int
        Index into `features` for the possession column.
    score_difference_index : int
        Index into `features` for the score difference column.
    prediction_column : str
        Name of the new column written to each CSV.
    ev_x_values : list of float, optional
        Representative point values per scoring bucket (e.g. [1, 4, 7]).
        If provided, writes a `{prediction_column}_ev` column to each CSV.
    model_type : str
        One of "xgboost" or "random_forest".
    use_smote : bool
        If True, applies SMOTE to oversample minority classes before training.
    """
    # --- Load and annotate all train years ---
    print(f"[1/4] Loading dataset from: {directory}")
    all_train_data = load_dataset(directory, features)
    all_train_data = {y: all_train_data[y] for y in train_years if y in all_train_data}
    print(f"      Loaded years: {sorted(all_train_data.keys())} | rows per year: { {y: arr.shape[0] for y, arr in all_train_data.items()} }")

    print(f"[2/4] Annotating possessions ...")
    annotated = annotate_possessions(all_train_data, possession_index, score_difference_index)
    print(f"      Done. Array shape after annotation (sample year {next(iter(annotated))}): {next(iter(annotated.values())).shape}")

    # --- Leave-one-out inference on train years (parallel across years) ---
    print(f"[3/4] Leave-one-out inference on train years: {train_years}")
    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_loo_train_and_infer, (directory, year, train_years, annotated, features, prediction_column, ev_x_values, model_type, use_smote, use_sample_weights)): year for year in train_years}
        for fut in as_completed(futures):
            year, trained_on, metrics, n_files = fut.result()
            if metrics is None:
                print(f"      [SKIP] Train year {year}")
            else:
                print(f"      Year {year} | trained on {trained_on} | train_acc={metrics['train_accuracy']:.4f} | test_acc={metrics['test_accuracy']:.4f} | saved {n_files} files\n{metrics['report']}")

    # --- Train on all train years, run inference on test years (parallel across years) ---
    print(f"[4/4] Training final model on all train years: {sorted(annotated.keys())} ...")
    model, metrics = train_possession_model(annotated, list(annotated.keys()), model_type, use_smote, use_sample_weights)
    print(f"      train_acc={metrics['train_accuracy']:.4f} | test_acc={metrics['test_accuracy']:.4f}\n{metrics['report']}")

    print(f"      Running inference on test years: {test_years}")
    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_run_inference_on_year, (directory, year, model, features, prediction_column, ev_x_values)): year for year in test_years}
        for fut in as_completed(futures):
            year, n_files, _ = fut.result()
            if n_files == 0:
                print(f"      [SKIP] Test year {year}: directory not found")
            else:
                print(f"      Year {year} | saved {n_files} files")

    print("Pipeline complete.")


if __name__ == "__main__":
    DIRECTORY = "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/ML/dataset_interpolated_fixed"
    TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    TEST_YEARS = [2024, 2025]
    FEATURES = ["game_completed", "relative_strength", "score_difference", "home_has_possession", "end.down", "end.distance", "end.yardsToEndzone",  "home_timeouts_left", "away_timeouts_left"]
    POSSESSION_INDEX = FEATURES.index("home_has_possession")
    SCORE_DIFFERENCE_INDEX = FEATURES.index("score_difference")
    EXPECTED_POINTS_X_VALUES = [0, 3, 6]  # representative points for the 0-2, 3-5, and 6+ buckets
    # run_pipeline(
    #     directory=DIRECTORY,
    #     train_years=TRAIN_YEARS,
    #     test_years=TEST_YEARS,
    #     features=FEATURES,
    #     possession_index=POSSESSION_INDEX,
    #     score_difference_index=SCORE_DIFFERENCE_INDEX,
    #     prediction_column="predicted_drive_points",
    #     ev_x_values=EXPECTED_POINTS_X_VALUES,
    #     model_type="xgboost",
    #     use_smote=False,
    #     use_sample_weights=True,
    # )

    print("\n[Sanity Check] Training model on all train years for verification ...")
    all_data = load_dataset(DIRECTORY, FEATURES)
    all_data = {y: all_data[y] for y in TRAIN_YEARS if y in all_data}
    annotated = annotate_possessions(all_data, POSSESSION_INDEX, SCORE_DIFFERENCE_INDEX)
    USE_SMOTE = False
    USE_SAMPLE_WEIGHTS = True
    MODEL_TYPE = "xgboost"
    model, _ = train_possession_model(annotated, list(annotated.keys()), model_type=MODEL_TYPE, use_smote=USE_SMOTE, use_sample_weights=USE_SAMPLE_WEIGHTS)

    sample_year = TRAIN_YEARS[-1]
    sample_year_dir = os.path.join(DIRECTORY, str(sample_year))
    sample_csv = sorted(f for f in os.listdir(sample_year_dir) if f.endswith(".csv"))[0]
    sample_df = pd.read_csv(os.path.join(sample_year_dir, sample_csv), skiprows=[1])
    sample_ground_truth = annotated[sample_year][:len(sample_df), -1]

    verify_sanity_check(sample_df, model, FEATURES, ev_x_values=EXPECTED_POINTS_X_VALUES, ground_truth=sample_ground_truth, n_samples=25)
