import pandas as pd
import glob
import os

def combine_csv_files(name, test_dir, base_dir="/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL"):
    """
    Combines all CSV files in the specified directory into a single DataFrame and saves it as a CSV.

    Parameters:
        name (str): Name of the subdirectory containing CSV files.
        test_num (int or str): Test number to use in the path.
        base_dir (str): Base directory path (default is set for your environment).

    Returns:
        pd.DataFrame: The combined DataFrame.
    """
    csv_dir = os.path.join(base_dir, test_dir, name)
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    game_data_list = [pd.read_csv(file) for file in csv_files]
    combined_data = pd.concat(game_data_list, ignore_index=True)
    output_csv = os.path.join(base_dir, test_dir, f"{name}_combined_data.csv")
    combined_data.to_csv(output_csv, index=False)
    return combined_data

if __name__ == "__main__":
    # Example usage; modify as needed
    combine_csv_files("LR_straw_men", "test_8")