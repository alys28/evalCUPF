import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


def SHAP_analysis_timestep(models, timestep, training_data, test_data, plot = True):
    model = models[timestep]
    X_test = []
    X_train = []
    for row in test_data[timestep]:
        X_test.append(row["rows"].reshape(-1))
    for row in training_data[timestep]:
        X_train.append(row["rows"].reshape(-1)) 
    X_test = np.array(X_test)
    X_train = np.array(X_train[:100])
    # y_test = np.array(y_test)
    vals = model.SHAP_analysis(X_test, X_train, plot)
    return vals

def SHAP_analysis(models, training_data, test_data, save_name,
                  save_dir=None, timesteps_size=0.005, num_threads=5):
    """
    Compute SHAP for a subset of timesteps and save outputs, skipping
    timesteps that already have a saved file.

    Args:
        models: dict mapping timestep -> model
        training_data: ...
        test_data: ...
        save_name: base name for files (without extension)
        save_dir: optional directory to save into. If None, uses CWD.
        timesteps_size: fraction of total timesteps to sample, or an int
                        specifying the exact number of timesteps to run.
                        E.g. 0.1 runs 10% of timesteps, 20 runs 20 timesteps.
        num_threads: max number of threads for parallel execution
    """
    # Decide where to save
    base_dir = save_dir if save_dir is not None else os.getcwd()
    os.makedirs(base_dir, exist_ok=True)

    # Find which timesteps are already processed by inspecting filenames
    # Expected pattern: f"{save_name}_{timestep}.npz"
    existing_timestep_strings = set()
    for fname in os.listdir(base_dir):
        if fname.startswith(save_name + "_") and fname.endswith(".npz"):
            ts_str = fname[len(save_name) + 1 : -4]
            existing_timestep_strings.add(ts_str)

    def process_timestep(timestep):
        shap_output = SHAP_analysis_timestep(
            models, timestep, training_data, test_data, plot=False
        )
        filename = save_name + "_" + str(timestep)
        full_path = os.path.join(base_dir, filename)
        save_SHAP_output(shap_output, full_path)
        print(f"Saved {filename}.npz")

    all_timesteps = sorted(models.keys())

    # Validate that 1 / timesteps_size is an integer
    assert abs(round(1 / timesteps_size) - (1 / timesteps_size)) < 1e-9, \
        f"1 / timesteps_size must be an integer, got 1 / {timesteps_size} = {1 / timesteps_size}"

    all_timesteps_arr = np.array(all_timesteps, dtype=float)
    ts_min, ts_max = all_timesteps_arr.min(), all_timesteps_arr.max()

    # Validate that timesteps_size resolution is >= the range of all_timesteps
    ts_range = ts_max - ts_min
    assert timesteps_size <= ts_range or np.isclose(timesteps_size, ts_range), \
        f"timesteps_size {timesteps_size} is coarser than the timestep range {ts_range}"

    # Step from min to max by timesteps_size, picking the nearest actual timestep each time
    grid = np.arange(ts_min, ts_max + timesteps_size * 0.5, timesteps_size)
    print(f"Selected {len(grid)} timesteps on the grid before deduplication.")
    print(grid)
    seen = set()
    selected_timesteps = []
    for val in grid:
        idx = int(np.argmin(np.abs(all_timesteps_arr - val)))
        t = all_timesteps[idx]
        if t not in seen:
            seen.add(t)
            selected_timesteps.append(t)

    # Determine which timesteps still need to be processed
    timesteps_to_run = [
        t for t in selected_timesteps
        if str(t) not in existing_timestep_strings
    ]

    if not timesteps_to_run:
        print("All selected timesteps already processed. Nothing to do.")
        return

    print(f"Processing {len(timesteps_to_run)} timesteps "
          f"(skipping {len(selected_timesteps) - len(timesteps_to_run)} already done, "
          f"{len(all_timesteps) - len(selected_timesteps)} not selected).")

    # Run remaining timesteps in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_timestep, timesteps_to_run)


def save_SHAP_output(shap_output, path):
    """
    Save a shap_output into a single .npz file.
    """
    # Build a dict of NPZ-safe arrays
    arrays_to_save = {}
    arrays_to_save["values"] = shap_output.values
    arrays_to_save["base_values"] = shap_output.base_values
    arrays_to_save["data"] = shap_output.data
    arrays_to_save["feature_names"] = np.array(shap_output.feature_names)
    # Save
    np.savez_compressed(
        (path + ".npz") if not path.endswith(".npz") else path,
        **arrays_to_save
    )