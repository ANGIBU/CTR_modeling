# training.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, f1_score
import optuna
import logging
import time
import gc
import os
import joblib
import torch
import psutil
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from models import ModelFactory, cleanup_memory

class CTRTrainer:
    """CTR model trainer with GPU optimization"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_name = self._get_gpu_name()
        self.rtx_4060_ti_mode = 'RTX 4060 Ti' in str(self.gpu_name)
        
        # Initialize trainer
        self.logger.info(f"GPU training environment: {self.gpu_name}")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"GPU memory: {gpu_memory:.1f}GB")
            
        if self.rtx_4060_ti_mode:
            self.logger.info("RTX 4060 Ti GPU training enabled")
            
        self.logger.info("CTR Trainer GPU initialized (RTX 4060 Ti mode)" if self.rtx_4060_ti_mode else "CTR Trainer initialized")
        
        # Default model parameters
        self.default_params = {
            'logistic': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'liblinear',
                'class_weight': 'balanced'
            },
            'lightgbm': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_estimators': 100
            },
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': 0
            },
            'catboost': {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42,
                'verbose': False
            },
            'deep_ctr': {
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 1024,
                'epochs': 50,
                'early_stopping_patience': 10
            }
        }
        
    def _get_gpu_name(self):
        """Get GPU name"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
            return "No GPU"
        except:
            return "Unknown GPU"
    
    def _get_memory_info(self):
        """Get memory information"""
        memory = psutil.virtual_memory()
        return {
            'available_gb': memory.available / 1024**3,
            'used_percent': memory.percent
        }
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame = None, y_val: pd.Series = None, 
                   custom_params: Dict = None, **kwargs):
        """Train a single model"""
        
        self.logger.info(f"Starting {model_name} model training")
        
        try:
            # Memory check
            memory_info = self._get_memory_info()
            self.logger.info(f"Pre-training memory: {memory_info['available_gb']:.1f}GB available")
            
            # Get model parameters
            params = self.default_params.get(model_name, {}).copy()
            if custom_params:
                params.update(custom_params)
            
            self.logger.info(f"Using default parameters for {model_name}")
            
            # Create model
            model = ModelFactory.create_model(model_name, **params)
            
            # Train model
            self.logger.info(f"Training final {model_name} model")
            start_time = time.time()
            
            model.fit(X_train, y_train, X_val, y_val, **kwargs)
            
            training_time = time.time() - start_time
            
            # Log training results
            self.logger.info(f"{model_name} training completed in {training_time:.2f}s")
            
            if hasattr(model, 'training_history') and model.training_history:
                history = model.training_history
                if 'val_auc' in history:
                    self.logger.info(f"{model_name} validation AUC: {history['val_auc']:.4f}")
                if 'val_logloss' in history:
                    self.logger.info(f"{model_name} validation log loss: {history['val_logloss']:.4f}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"{model_name} model training failed: {e}")
            return None
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int = 5, custom_params: Dict = None):
        """Perform cross-validation for a model"""
        
        self.logger.info(f"Starting {cv_folds}-fold cross-validation for {model_name}")
        
        try:
            # Get model parameters
            params = self.default_params.get(model_name, {}).copy()
            if custom_params:
                params.update(custom_params)
            
            # Cross-validation
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                self.logger.info(f"Training fold {fold + 1}/{cv_folds}")
                
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Create and train model
                model = ModelFactory.create_model(model_name, **params)
                model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                
                # Evaluate
                val_pred = model.predict_proba(X_val_fold)
                fold_auc = roc_auc_score(y_val_fold, val_pred)
                cv_scores.append(fold_auc)
                
                self.logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
                
                # Cleanup
                del model
                cleanup_memory()
            
            mean_auc = np.mean(cv_scores)
            std_auc = np.std(cv_scores)
            
            self.logger.info(f"{model_name} CV results: {mean_auc:.4f} ± {std_auc:.4f}")
            
            return {
                'model_name': model_name,
                'cv_scores': cv_scores,
                'mean_auc': mean_auc,
                'std_auc': std_auc
            }
            
        except Exception as e:
            self.logger.error(f"{model_name} cross-validation failed: {e}")
            return None
    
    def optimize_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                                n_trials: int = 50, cv_folds: int = 3):
        """Optimize hyperparameters using Optuna"""
        
        self.logger.info(f"Starting hyperparameter optimization for {model_name}")
        
        def objective(trial):
            try:
                # Define hyperparameter space based on model
                if model_name == 'logistic':
                    params = {
                        'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                        'max_iter': trial.suggest_int('max_iter', 500, 2000),
                        'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
                    }
                elif model_name == 'lightgbm':
                    params = {
                        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300)
                    }
                elif model_name == 'xgboost':
                    params = {
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                    }
                elif model_name == 'catboost':
                    params = {
                        'depth': trial.suggest_int('depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'iterations': trial.suggest_int('iterations', 50, 300)
                    }
                else:
                    return 0.5  # Default score for unsupported models
                
                # Cross-validation
                cv_result = self.cross_validate_model(model_name, X, y, cv_folds, params)
                
                if cv_result is None:
                    return 0.5
                
                return cv_result['mean_auc']
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.5
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            best_params = study.best_params
            best_score = study.best_value
            
            self.logger.info(f"Best {model_name} parameters: {best_params}")
            self.logger.info(f"Best {model_name} CV score: {best_score:.4f}")
            
            return best_params, best_score
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return None, None
    
    def train_ensemble_models(self, model_names: List[str], X_train: pd.DataFrame, 
                            y_train: pd.Series, X_val: pd.DataFrame = None, 
                            y_val: pd.Series = None, optimize_params: bool = False):
        """Train multiple models for ensemble"""
        
        self.logger.info(f"Training ensemble models: {model_names}")
        
        trained_models = {}
        
        for model_name in model_names:
            self.logger.info(f"Training {model_name} for ensemble")
            
            try:
                params = None
                if optimize_params:
                    self.logger.info(f"Optimizing hyperparameters for {model_name}")
                    params, _ = self.optimize_hyperparameters(model_name, X_train, y_train)
                
                # Train model
                model = self.train_model(model_name, X_train, y_train, X_val, y_val, params)
                
                if model is not None and model.is_trained:
                    trained_models[model_name] = model
                    self.logger.info(f"{model_name} successfully trained for ensemble")
                else:
                    self.logger.warning(f"{model_name} training failed for ensemble")
                    
            except Exception as e:
                self.logger.error(f"Error training {model_name} for ensemble: {e}")
                continue
        
        self.logger.info(f"Ensemble training completed. Successful models: {len(trained_models)}")
        return trained_models
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate model performance"""
        
        try:
            # Predictions
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            
            # Metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            metrics = {
                'auc': auc,
                'logloss': logloss,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            self.logger.info(f"Model evaluation - AUC: {auc:.4f}, Log Loss: {logloss:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return None
    
    def save_model(self, model, filepath: str):
        """Save trained model"""
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            model.save_model(filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        try:
            model = ModelFactory.create_model(model_name)
            model.load_model(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

class BatchTrainer(CTRTrainer):
    """Batch trainer for large datasets"""
    
    def __init__(self, config=None, batch_size: int = 10000):
        super().__init__(config)
        self.batch_size = batch_size
        
    def train_in_batches(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                        validation_split: float = 0.2):
        """Train model in batches for memory efficiency"""
        
        self.logger.info(f"Starting batch training for {model_name}")
        
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Get model parameters
            params = self.default_params.get(model_name, {}).copy()
            
            # For models that support incremental learning
            if model_name in ['logistic']:
                return self._train_incremental(model_name, X_train, y_train, X_val, y_val, params)
            else:
                # For tree-based models, use sampling
                return self._train_with_sampling(model_name, X_train, y_train, X_val, y_val, params)
                
        except Exception as e:
            self.logger.error(f"Batch training failed: {e}")
            return None
    
    def _train_incremental(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series, params: Dict):
        """Train with incremental learning"""
        
        from sklearn.linear_model import SGDClassifier
        
        # Use SGD for incremental learning
        model = SGDClassifier(
            loss='log_loss',
            random_state=42,
            class_weight='balanced'
        )
        
        # Train in batches
        n_batches = len(X_train) // self.batch_size + 1
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(X_train))
            
            if start_idx >= len(X_train):
                break
                
            X_batch = X_train.iloc[start_idx:end_idx]
            y_batch = y_train.iloc[start_idx:end_idx]
            
            if len(X_batch) == 0:
                continue
                
            # Partial fit
            model.partial_fit(X_batch, y_batch, classes=[0, 1])
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed batch {i + 1}/{n_batches}")
        
        # Wrap in our model class
        from models import LogisticModel
        wrapped_model = LogisticModel(name=model_name)
        wrapped_model.model = model
        wrapped_model.is_trained = True
        
        return wrapped_model
    
    def _train_with_sampling(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series, params: Dict):
        """Train with data sampling for memory efficiency"""
        
        # Sample data if too large
        max_samples = 100000
        if len(X_train) > max_samples:
            self.logger.info(f"Sampling {max_samples} samples from {len(X_train)} for training")
            sample_idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_train_sample = X_train.iloc[sample_idx]
            y_train_sample = y_train.iloc[sample_idx]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        # Train normally with sampled data
        return self.train_model(model_name, X_train_sample, y_train_sample, X_val, y_val, params)

# Utility functions
def get_model_memory_usage(model):
    """Estimate model memory usage"""
    try:
        if hasattr(model, 'model'):
            return joblib.dump(model.model, None).__sizeof__() / 1024**2  # MB
        return 0
    except:
        return 0

def cleanup_training_artifacts():
    """Clean up training artifacts"""
    cleanup_memory()
    
    # Additional cleanup for Optuna
    try:
        import optuna
        optuna.delete_study(study_name="default")
    except:
        pass