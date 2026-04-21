import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import numpy as np
import optuna
from .Model import Model


class LightGBM(Model):
    def __init__(self, use_calibration=True, optimize_hyperparams=False, n_trials=50,
                 numeric_features=None, other_features=None, all_features=None, **kwargs):
        super().__init__(use_calibration=use_calibration, optimize_hyperparams=optimize_hyperparams,
                         numeric_features=numeric_features, other_features=other_features,
                         all_features=all_features)
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 4,
            'learning_rate': 0.015,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_child_samples': 50,
            'min_split_gain': 0.5,
            'max_bin': 255,
            'seed': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
        self.params.update(kwargs)
        self.num_boost_round = 2000
        self.n_trials = n_trials
        self.best_params = None

    def _define_search_space(self, trial):
        """Define the hyperparameter search space for Bayesian optimization."""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'max_bin': trial.suggest_int('max_bin', 100, 500),
        }

    def _fixed_params(self):
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'seed': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }

    def _train_model(self, X_train, y_train, X_val, y_val, params):
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=-1)]
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        probs = model.predict(X_val, num_iteration=model.best_iteration)
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

        dtrain = lgb.Dataset(X_train_proc, label=y_train)
        dval = lgb.Dataset(X_val_proc, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=-1)]
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        if self.use_calibration:
            probs_cal = self.model.predict(X_val_proc, num_iteration=self.model.best_iteration)
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
        probs = self.model.predict(X_proc, num_iteration=self.model.best_iteration)
        calibrated = self.apply_calibration(probs)
        return np.column_stack([1 - calibrated, calibrated])

    def score(self, X, y):
        """Return accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def setup_lightgbm_models(training_data, validation_data, numeric_features=None, other_features=None,
                          all_features=None, use_calibration=True, optimize_hyperparams=False,
                          n_trials=50, num_models=None):
    """
    Setup LightGBM models with optional Bayesian optimization.

    Args:
        training_data: Dictionary with timestep keys and training data
        validation_data: Dictionary with timestep keys and validation data
        optimize_hyperparams: Whether to perform Bayesian optimization
        n_trials: Number of optimization trials (only used if optimize_hyperparams=True)
        num_models: Number of evenly spaced models to train (must divide 1.0 into
                    timesteps that exist in training_data). If None, uses all timesteps.
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
        model = LightGBM(
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

        model = LightGBM(
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
