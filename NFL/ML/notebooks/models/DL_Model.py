import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from abc import ABC, abstractmethod
import shap
import pickle


class NFLDirectDataset(Dataset):
    """
    Common dataset class for all direct prediction models.
    No pairs needed - direct prediction from features to outcomes.
    """
    def __init__(self, data_x, data_y, device='cpu'):
        """
        Direct prediction dataset - no pairs needed
        Args:
            data_x: Game features
            data_y: Game outcomes (score differences)
        """
        self.device = device
        self.data_x = torch.FloatTensor(data_x).to(device)
        
        # Convert score differences to win/loss labels (1 = win, 0 = loss)
        self.data_y = torch.FloatTensor([(1.0 if y > 0 else 0.0) for y in data_y]).to(device)
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]


class BaseDirectClassifier(ABC):
    """
    Abstract base class for direct prediction classifiers.
    Contains common functionality shared across all model types.
    """
    
    def __init__(self, model, epochs, optimizer, criterion, device, features, scheduler=None, use_scaler=True):
        """
        Initialize the base classifier
        
        Args:
            model: PyTorch model (DirectPredictionNetwork, DirectPredictionTransformer, DirectPredictionLSTM, etc.)
            epochs: Number of training epochs
            optimizer: PyTorch optimizer
            criterion: Loss function
            device: PyTorch device
            features: list of features (string), ordered with the data
            scheduler: Learning rate scheduler (optional)
            use_scaler: Whether to use feature scaling
        """
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.scheduler = scheduler
        self.use_scaler = use_scaler
        self.features = features 
        # Initialize scaler if needed
        if self.use_scaler:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler_fitted = False
        else:
            self.scaler = None
            self.scaler_fitted = False
        
        # Track training metrics
        self.final_train_loss = None
        self.final_train_accuracy = None
        self.final_val_loss = None
        self.final_val_accuracy = None
        self.best_epoch = None
        self.epochs_trained = None
    
    @abstractmethod
    def _prepare_data_for_scaling(self, X):
        """
        Prepare data for scaling. Different models may need different preprocessing.
        Should return data in the format expected by the scaler.
        
        Args:
            X: Input data (can be 2D or 3D)
            
        Returns:
            Prepared data for scaling
        """
        pass
    
    @abstractmethod
    def _apply_scaling(self, X, fit=False):
        """
        Apply scaling to data. Different models may handle scaling differently.
        
        Args:
            X: Input data
            fit: Whether to fit the scaler (True) or just transform (False)
            
        Returns:
            Scaled data in original shape
        """
        pass
    
    @abstractmethod
    def _get_training_hooks(self):
        """
        Get training-specific hooks (gradient clipping, label smoothing, etc.)
        Returns a dict with optional hooks:
        - gradient_clip_norm: float or None
        - label_smoothing: float (0.0 to 1.0)
        - patience: int for early stopping
        """
        pass

    @abstractmethod
    def _get_model_config(self):
        """
        Extract model configuration for saving.
        Should return a dict with model architecture parameters.
        """
        pass

    @abstractmethod
    def _recreate_model_from_config(self, model_config):
        """
        Recreate model instance from saved configuration.
        
        Args:
            model_config: Dictionary with model architecture parameters
            
        Returns:
            Recreated model instance
        """
        pass
    
    @abstractmethod
    def _get_model_type_name(self):
        """
        Get the model type name for saving/loading (e.g., 'direct', 'transformer', 'lstm')
        """
        pass
    
    def fit(self, X, y, val_X=None, val_y=None, batch_size=128, verbose=True):
        """
        Train the model with early stopping and validation.
        """
        # Prepare and scale training data
        X_scaled = self._apply_scaling(X, fit=True)
        
        train_dataset = NFLDirectDataset(X_scaled, y, device=self.device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        if val_X is not None and val_y is not None:
            val_X_scaled = self._apply_scaling(val_X, fit=False)
            val_dataset = NFLDirectDataset(val_X_scaled, val_y, device=self.device)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Get training hooks
        hooks = self._get_training_hooks()
        patience = hooks.get('patience', 10)
        gradient_clip_norm = hooks.get('gradient_clip_norm', None)
        label_smoothing = hooks.get('label_smoothing', 0.0)
        
        # Training with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        
        if verbose:
            print(f"Starting training on device: {self.device}")
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (x, y_batch) in enumerate(train_loader):
                x, y_batch = x.to(self.device), y_batch.to(self.device)
                
                # Apply label smoothing if enabled
                y_smooth = y_batch * (1 - label_smoothing) + 0.5 * label_smoothing
                
                output = self.model(x).squeeze(-1)
                
                self.optimizer.zero_grad()
                loss = self.criterion(output, y_smooth)
                
                loss.backward()
                
                # Gradient clipping if enabled
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_norm)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                # Calculate accuracy (use original y, not smoothed)
                with torch.no_grad():
                    predictions = (output > 0.5).float()
                    train_correct += (predictions == y_batch).float().sum().item()
                    train_total += len(predictions)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # Validation
            if val_loader is not None:
                val_loss, val_accuracy = self._evaluate(val_loader)
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch + 1
                    # Store best metrics
                    self.final_val_loss = val_loss
                    self.final_val_accuracy = val_accuracy
                    self.final_train_loss = avg_train_loss
                    self.final_train_accuracy = train_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                            print(f"Best epoch: {best_epoch}, Train Acc: {self.final_train_accuracy:.4f}, "
                                  f"Train Loss: {self.final_train_loss:.4f}, Val Acc: {self.final_val_accuracy:.4f}, "
                                  f"Val Loss: {self.final_val_loss:.4f}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            if verbose:
                                print(f"Restored model from best epoch {best_epoch} with val_loss: {best_val_loss:.6f}")
                        break
            else:
                # Store final metrics for no-validation case
                self.final_train_loss = avg_train_loss
                self.final_train_accuracy = train_accuracy
                best_epoch = epoch + 1
                if best_model_state is None:
                    best_model_state = self.model.state_dict().copy()
        
        # Store final training info
        self.best_epoch = best_epoch
        self.epochs_trained = epoch + 1
    
    def _evaluate(self, data_loader):
        """Helper method for evaluation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        evaluated_batches = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)

                if torch.isnan(x).any() or torch.isnan(y).any():
                    continue

                output = self.model(x).squeeze(-1)

                if torch.isnan(output).any():
                    continue

                loss = self.criterion(output, y)

                if torch.isnan(loss):
                    continue

                total_loss += loss.item()
                evaluated_batches += 1

                predictions = (output > 0.5).float()
                correct += (predictions == y).float().sum().item()
                total_samples += len(predictions)

        avg_loss = total_loss / evaluated_batches if evaluated_batches > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy
    
    @abstractmethod
    def predict(self, x):
        """
        Make predictions on new data.
        Model-specific implementation needed.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, x):
        """
        Return prediction probabilities.
        Should return array of shape (n_samples, 2) with [prob_class_0, prob_class_1].
        """
        pass
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            float: Mean accuracy score
        """
        # Get predictions (scaling is handled internally in predict method)
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(float).flatten()
        
        # Convert y to binary if needed (in case it's score differences)
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        y_true = np.array([(1.0 if label > 0 else 0.0) for label in y]) if y.max() > 1 or y.min() < 0 else y
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        return accuracy
    
    def get_scaler(self):
        """
        Return the fitted scaler if available
        """
        if self.use_scaler and self.scaler_fitted:
            return self.scaler
        else:
            return None
    
    def save_model(self, filepath_prefix, timesteps_range):
        """
        Save the trained model
        """
        val_acc = self.final_val_accuracy if self.final_val_accuracy else 0
        val_loss = self.final_val_loss if self.final_val_loss else 0
        model_type = self._get_model_type_name()
        
        filename = f"{filepath_prefix}_{model_type}_{timesteps_range[0]}-{timesteps_range[1]}.pth"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps_range': timesteps_range,
            'best_epoch': self.best_epoch,
            'epochs_trained': self.epochs_trained,
            'final_train_loss': self.final_train_loss,
            'final_train_accuracy': self.final_train_accuracy,
            'final_val_loss': self.final_val_loss,
            'final_val_accuracy': self.final_val_accuracy,
            'model_config': self._get_model_config(),
            'use_scaler': self.use_scaler,
            'scaler_fitted': self.scaler_fitted
        }
        
        # Save scaler if it exists and is fitted
        if self.use_scaler and self.scaler_fitted:
            checkpoint['scaler'] = pickle.dumps(self.scaler)
        
        torch.save(checkpoint, filename)
        print(f"Direct prediction {model_type} model saved: {filename}")
        return filename
    def predict_proba_single(self, X):
        preds = self.predict_proba(X)
        return preds[:, 1]
    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved model. Must be implemented by subclasses.
        """
        pass

    def SHAP_analysis(self, X_test, X_train, plot = True):
        """
        Model interpretability using SHAP DeepExplainer method
        """
        explainer = shap.Explainer(self.predict_proba_single, X_train[:min(len(X_train), 200)], feature_names=self.features)
        shap_values = explainer(X_test)
        if plot:
            shap.plots.bar(shap_values)
        return shap_values
            