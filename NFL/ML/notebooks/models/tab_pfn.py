from tabpfn import TabPFNClassifier
import numpy as np
from .Model import Model


class TabPFN(Model):
    def __init__(self, use_calibration=True, optimize_hyperparams=False, n_trials=50,
                 numeric_features=None, other_features=None, all_features=None, **kwargs):
        super().__init__(use_calibration=use_calibration, optimize_hyperparams=optimize_hyperparams,
                         numeric_features=numeric_features, other_features=other_features,
                         all_features=all_features)
        self.params = {
            'n_estimators': 32,
            'device': 'cpu',
        }
        self.params.update(kwargs)
        self.n_trials = n_trials
        self.best_params = None

    def _define_search_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 4, 64),
        }

    def _fixed_params(self):
        return {
            'device': 'cpu',
        }

    def _train_model(self, X_train, y_train, X_val, y_val, params):
        model = TabPFNClassifier(**params)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        return float(np.mean((probs - y_val) ** 2))

    def fit(self, X, y, val_X=None, val_y=None):
        if val_X is None and val_y is None:
            X_train, X_val, y_train, y_val = self.split_data(X, y, test_size=0.25, random_state=42, stratify=False)
        else:
            X_train, y_train = X, y
            X_val, y_val = val_X, val_y

        X_train_proc = self.fit_transform_X(X_train)
        X_val_proc = self.transform_X(X_val)

        if self.optimize_hyperparams:
            best = self.optimize_hyperparameters(X_train_proc, y_train, X_val_proc, y_val, n_trials=self.n_trials)
            self.params.update(best)
            self.params.update(self._fixed_params())

        self.model = TabPFNClassifier(**self.params)
        self.model.fit(X_train_proc, y_train)

        if self.use_calibration:
            probs_cal = self.model.predict_proba(X_val_proc)[:, 1]
            self.fit_calibrator(probs_cal, y_val)

        y_pred = self.predict_proba(X_train)[:, 1]
        train_loss = self.brier_loss(y_train, y_pred)
        train_accuracy = self.score(X_train, y_train)

        y_val_pred = self.predict_proba(X_val)[:, 1]
        val_loss = Model.brier_loss(y_val, y_val_pred)
        val_accuracy = self.score(X_val, y_val)

        print(f"Training Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        X_proc = self.transform_X(X)
        probs = self.model.predict_proba(X_proc)[:, 1]
        calibrated = self.apply_calibration(probs)
        return np.column_stack([1 - calibrated, calibrated])

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


def setup_tabpfn_models(training_data, validation_data, numeric_features=None, other_features=None,
                        all_features=None, use_calibration=True, optimize_hyperparams=False,
                        n_trials=50, num_models=None):
    """
    Setup TabPFN models for each timestep.

    Args:
        training_data: Dictionary with timestep keys and training data
        validation_data: Dictionary with timestep keys and validation data
        optimize_hyperparams: Whether to perform Bayesian optimization
        n_trials: Number of optimization trials (only used if optimize_hyperparams=True)
        num_models: Number of evenly spaced models to train. If None, uses all timesteps.
    """
    all_timesteps = sorted(training_data.keys())
    models = {}

    if num_models == 1:
        X = np.concatenate([np.array([row["rows"].reshape(-1) for row in training_data[t]]) for t in all_timesteps])
        y = np.concatenate([np.array([row["label"] for row in training_data[t]]) for t in all_timesteps])
        X_val, y_val = None, None
        if validation_data:
            X_val = np.concatenate([np.array([row["rows"].reshape(-1) for row in validation_data[t]]) for t in all_timesteps])
            y_val = np.concatenate([np.array([row["label"] for row in validation_data[t]]) for t in all_timesteps])
        model = TabPFN(
            use_calibration=use_calibration,
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials,
            numeric_features=numeric_features,
            other_features=other_features,
            all_features=all_features
        )
        print("Single model (all timesteps pooled):", end=" ")
        model.fit(X, y, val_X=X_val, val_y=y_val)
        return {t: model for t in all_timesteps}

    if num_models is not None:
        if num_models < 2:
            raise ValueError("num_models must be at least 2.")
        step = 1.0 / (num_models - 1)
        target_timesteps = [round(i * step, 3) for i in range(num_models)]
        missing = [t for t in target_timesteps if t not in training_data]
        if missing:
            raise ValueError(
                f"num_models={num_models} produces timesteps not present in training_data: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Choose a num_models such that 1/(num_models-1) aligns with the data resolution "
                f"(e.g. {1/round(all_timesteps[1]-all_timesteps[0], 5):.0f} for all timesteps)."
            )
        timesteps = target_timesteps
    else:
        timesteps = all_timesteps

    for i, timestep in enumerate(timesteps):
        timestep = round(timestep, 3)

        X = training_data[timestep]
        y = np.array([row["label"] for row in X])
        X = np.array([row["rows"].reshape(-1) for row in X])
        X_val = None
        y_val = None
        if validation_data:
            y_val = np.array([row["label"] for row in validation_data[timestep]])
            X_val = np.array([row["rows"].reshape(-1) for row in validation_data[timestep]])

        model = TabPFN(
            use_calibration=use_calibration,
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials,
            numeric_features=numeric_features,
            other_features=other_features,
            all_features=all_features
        )
        opt_info = (f" (Optimized)" if optimize_hyperparams else "") + "(Calibrated)" if use_calibration else ""
        print(f"Timestep {timestep:.2%}{opt_info}:", end=" ")
        model.fit(X, y, val_X=X_val, val_y=y_val)
        models[timestep] = model

        if (i + 1) % 50 == 0 or i == len(timesteps) - 1:
            print(f"Completed {i + 1}/{len(timesteps)} timesteps")

    # Ensure returned dict has keys at 0.005 intervals, mapped to nearest trained model
    full_timesteps = [round(i * 0.005, 3) for i in range(201)]
    trained_keys = sorted(models.keys())
    result = {}
    for t in full_timesteps:
        nearest = min(trained_keys, key=lambda k: abs(k - t))
        result[t] = models[nearest]
    return result
