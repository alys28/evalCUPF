import pandas as pd
import numpy as np
import os

# CHANGE FOR EVERY TEST:
# - directory
# - phat_b (below)

# Directory containing CSV files
directory = "test_9/xgboost_model"  # Change this to your actual directory
phat_B = "xgboost_phat_b"  # Change this to the actual column name for phat_B in your CSV files

# Function to process a CSV file
def process_csv(file_path, phat_A, phat_b, interpolate = False, steps = 0.01):
    df = pd.read_csv(file_path)
    
    # Convert clock to minutes remaining
    def parse_clock(clock):
        minutes, seconds = map(int, clock.split(":"))
        return minutes + seconds / 60  # Convert to fraction of a minute
    
    # Compute minutes remaining
    df["minutes_remaining"] = (4 - df["period.number"]) * 15 + df["clock.displayValue"][1:].apply(parse_clock)
    
    # Compute percentage of game completed
    df["game_completed"] = 1 - df["minutes_remaining"] / 60
    
    # Assign phat_B as the first value of home_win_probability
    df["phat_A"] = df["homeWinProbability"].iloc[0]
    df["phat_B"] = df["homeWinProbability"].iloc[0]
    df["Y"] = df["home_win"].iloc[0]
    # Assign phat_A from home_win_probability
    df["phat_A"] = df["homeWinProbability"]
    if interpolate:
        # Interpolation for fixed game_completed values
        new_game_completed = np.arange(0, 1 + steps, steps)
        interpolated_df = pd.DataFrame({
            "game_id": os.path.basename(file_path).split("_")[-1].split(".")[0],  # extract game_id from file_path: "2018_interpolated/updated_game_2018090600.csv",
            "game_completed": new_game_completed,
            "phat_A": np.interp(new_game_completed, df["game_completed"], df[phat_A]),
            "phat_B": np.interp(new_game_completed, df["game_completed"], df[phat_b]),
            "Y": df["Y"].iloc[0]  # Assuming Y remains constant
        })
        
        # Save the interpolated file
        interpolated_file_path = os.path.join(directory, "interpolated_" + os.path.basename(file_path))
        interpolated_df.to_csv(interpolated_file_path, index=False)
        print(f"Processed and saved: {interpolated_file_path}")
    else:
        df = df[df['timestep'].duplicated(keep='last') == False]
        interpolated_df = pd.DataFrame({
            "game_id": os.path.basename(file_path).split("_")[-1].split(".")[0],  # extract game_id from file_path: "2018_interpolated/updated_game_2018090600.csv",
            "game_completed": df["timestep"].iloc[1:],
            "phat_A": df[phat_A].iloc[1:],
            "phat_B": df[phat_b].iloc[1:],
            "Y": df["Y"].iloc[0]  # Assuming Y remains constant
        })
        # Save the file
        interpolated_file_path = os.path.join(directory, os.path.basename(file_path))
        interpolated_df.to_csv(interpolated_file_path, index=False)
        print(f"Processed and saved: {interpolated_file_path}")


# Process all CSV files in the directory
if __name__ == "__main__":
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            print(file_path)
            process_csv(file_path, "homeWinProbability", phat_B, False, 0.005)  # Interpolate with 0.5% steps
    print("Processing complete for all CSV files in the directory.")