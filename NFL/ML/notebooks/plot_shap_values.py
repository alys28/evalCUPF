# Given all the saved SHAP values, we will plot the importance of each feature over time individually
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass


FEATURES = ["game_completed", "relative_strength", "score_difference", "home_has_possession", "end.down", "end.distance", "end.yardsToEndzone", "home_timeouts_left", "away_timeouts_left", "predicted_drive_points_ev"]

@dataclass
class SHAP_Output:
    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray


def load_SHAP_output(path):
    npz = np.load(path, allow_pickle=True)

    shap_outputs = {}

    # Extract timestep from the file name
    filename = os.path.basename(path)
    shap_outputs = SHAP_Output(
        values = npz[f"values"],
        base_values = npz[f"base_values"],
        data = npz[f"data"]
    )

    return shap_outputs

class SHAP_over_time:
    def __init__(self, features):
        """
        Initialize.
        self.__timesteps: keys = timesteps, values = list[list[SHAP_val for each feature] for each model]
        """
        self.features = features
        self._timesteps = {}

    def add_timestep(self, timestep, shap_output):
        if not(timestep in self._timesteps.keys()):
            self._timesteps[timestep] = []
        mean_abs = np.abs(shap_output.values).mean(axis=0)
        self._timesteps[timestep].append(mean_abs)

    def _convert_to_lines(self):
        lines = {feature: [] for feature in self.features}
        for timestep in sorted(self._timesteps.keys()):
            values = self._timesteps[timestep]
            idx = 0
            for feature in lines.keys():
                lines[feature].append(values[idx])
                idx += 1        
        return lines

    def plot(self, plot_title="Feature importance over time", save_path="SHAP_plot.png", show=True):
        """
        Plot all lines with x-axis from 0 to 1. If series have different lengths,
        they are interpolated onto a common grid for visual comparison.

        Args:
            plot_title: Title for the plot.
            save_path: Optional path to save the figure (e.g., 'shap_over_time.png').
            show: Whether to display the plot (True) or just build it (False).
        """
        if not self._timesteps:
            raise ValueError("No lines to plot. Use self.add_timestep(timestep, shap_output) first.")
        # Choose a common grid size (max length among series)
        ts_sorted = sorted(self._timesteps.keys())
        x = np.array(ts_sorted)
        lines = self._convert_to_lines()
        plt.figure(figsize=(9, 5))
        for name, y in lines.items():
            y_plot = y
            plt.plot(x, y_plot, label=name)

        plt.title(plot_title)
        plt.xlabel("Time")
        plt.ylabel("Feature importance value")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Feature", loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()

    def normalize_timesteps(self):
        for timestep in self._timesteps.keys():
            entry = np.array(self._timesteps[timestep])
            entry = np.mean(entry, axis = 0)
            entry /= np.sum(entry)
            self._timesteps[timestep] = entry


if __name__ == "__main__":
    shap_model_dir = "shap_values/XGBoost"
    shap_over_time = SHAP_over_time(FEATURES) 
    for file in os.listdir(shap_model_dir):
        file_dir = os.path.join(shap_model_dir, file)
        shap_output = load_SHAP_output(file_dir)
        timestep = os.path.splitext(file)[0].split("_")[-1]
        timestep = float(timestep)
        shap_over_time.add_timestep(timestep, shap_output)
    shap_over_time.normalize_timesteps()
    shap_over_time.plot()