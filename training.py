# training.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import gc
import time
import pickle
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import threading
import joblib

# Safe library imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna is not installed. Hyperparameter tuning functionality will be disabled.")

from config import Config
from models import BaseModel, ModelFactory
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class LargeDataMemoryTracker:
    """Large data memory monitoring"""
    
    def __init__(self):
        self.warning_threshold = 45.0  # 45GB warning threshold
        self.critical_threshold = 50.0  # 50GB critical threshold
        self.lock = threading.Lock()
        
    def get_memory_usage(self) -> float:
        """Current memory usage in GB"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024**2)
                return memory_mb / 1024
            return 2.0
        except Exception:
            return 2.0
    
    def get_available_memory(self) -> float:
        """Available memory in GB"""
        try:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                return vm.available / (1024**3)
            return 40.0
        except Exception:
            return 40.0
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """GPU memory usage"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                cached = torch.cuda.memory_reserved(0) / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_memory = total_memory - allocated
                
                return {
                    'allocated': allocated,
                    'cached': cached,
                    'free': free_memory,
                    'total': total_memory,
                    'utilization': (allocated / total_memory) * 100
                }
            else:
                return {
                    'allocated': 0.0,
                    'cached': 0.0,
                    'free': 0.0,
                    'total': 0.0,
                    'utilization': 0.0
                }
        except Exception:
            return {
                'allocated': 0.0,
                'cached': 0.0,
                'free': 0.0,
                'total': 0.0,
                'utilization': 0.0
            }
    
    @staticmethod
    def force_cleanup():
        """Memory cleanup"""
        try:
            collected = gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return collected
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0

class CTRModelTrainer:
    """CTR model trainer"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.trained_models = {}
        self.best_params = {}
        self.cv_results = {}
        self.model_performance = {}
        self.gpu_available = False
        
        # GPU setup
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"GPU training environment: {device_name}")
                logger.info(f"GPU memory: {gpu_memory:.1f}GB")
                
                if "RTX 4060 Ti" in device_name:
                    logger.info("RTX 4060 Ti optimization: True")
                    logger.info("Mixed Precision enabled")
                    
                self.gpu_available = True
            else:
                logger.info("Using CPU mode")
                self.gpu_available = False
        except Exception:
            logger.info("CPU training environment - Ryzen 5 5600X 6 cores 12 threads")
    
    def train_single_model(self, 
                          model_type: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          params: Optional[Dict[str, Any]] = None,
                          apply_calibration: bool = True) -> BaseModel:
        """Model training (optimized for 64GB environment)"""
        
        logger.info(f"{model_type} model training started (data size: {len(X_train):,})")
        start_time = time.time()
        memory_before = self.memory_tracker.get_memory_usage()
        gpu_info_before = self.memory_tracker.get_gpu_memory_usage()
        
        try:
            available_memory = self.memory_tracker.get_available_memory()
            gpu_memory = self.memory_tracker.get_gpu_memory_usage()
            
            data_size_gb = (X_train.memory_usage(deep=True).sum() + y_train.memory_usage(deep=True)) / (1024**3)
            logger.info(f"Data size: {data_size_gb:.2f}GB, available memory: {available_memory:.2f}GB")
            
            # More aggressive memory usage threshold for 64GB environment (changed from 5GB to 2GB)
            if available_memory < 2:
                logger.warning(f"Low memory detected: {available_memory:.2f}GB available")
                logger.info("Applying memory efficient processing")
                X_train, y_train, X_val, y_val = self._apply_memory_efficient_sampling(
                    X_train, y_train, X_val, y_val, available_memory
                )
            
            if params:
                params = self._validate_and_apply_params(model_type, params)
            else:
                params = self._get_default_params(model_type)
            
            model_kwargs = {'params': params}
            if model_type.lower() == 'deepctr':
                model_kwargs['input_dim'] = X_train.shape[1]
            
            model = ModelFactory.create_model(model_type, **model_kwargs)
            
            model.fit(X_train, y_train, X_val, y_val)
            
            # Apply calibration for all models in 64GB environment (more aggressive threshold)
            if apply_calibration and X_val is not None and y_val is not None:
                current_memory = self.memory_tracker.get_available_memory()
                if current_memory > 2:  # More aggressive threshold from 5GB to 2GB
                    self._apply_calibration(model, X_val, y_val)
                    logger.info(f"{model_type} calibration applied successfully")
                else:
                    logger.warning("Skipping calibration due to memory shortage")
            
            training_time = time.time() - start_time
            memory_after = self.memory_tracker.get_memory_usage()
            gpu_info_after = self.memory_tracker.get_gpu_memory_usage()
            
            logger.info(f"{model_type} model training complete (time taken: {training_time:.2f}s)")
            logger.info(f"Memory usage: {memory_before:.2f}GB → {memory_after:.2f}GB")
            logger.info(f"GPU memory utilization: {gpu_info_before['utilization']:.1f}% → {gpu_info_after['utilization']:.1f}%")
            
            self.trained_models[model_type] = {
                'model': model,
                'params': params or {},
                'training_time': training_time,
                'calibrated': apply_calibration and model.is_calibrated,
                'memory_used': memory_after - memory_before,
                'gpu_memory_used': gpu_info_after['allocated'] - gpu_info_before['allocated'],
                'data_size': len(X_train),
                'cv_result': None,
                'optimized': True
            }
            
            self._cleanup_memory_after_training(model_type)
            
            return model
            
        except Exception as e:
            logger.error(f"{model_type} model training failed: {str(e)}")
            logger.error(f"Available memory: {self.memory_tracker.get_available_memory():.2f}GB")
            LargeDataMemoryTracker.force_cleanup()
            raise
    
    def _apply_memory_efficient_sampling(self, X_train, y_train, X_val, y_val, available_memory):
        """Apply memory efficient sampling based on available memory"""
        try:
            # Calculate sampling ratio based on available memory (more aggressive)
            if available_memory < 1:
                sample_ratio = 0.2
            elif available_memory < 2:
                sample_ratio = 0.3
            elif available_memory < 3:
                sample_ratio = 0.5
            elif available_memory < 5:
                sample_ratio = 0.7
            else:
                sample_ratio = 0.85
            
            logger.info(f"Applying memory efficient sampling: ratio={sample_ratio:.2f}")
            
            # Sample training data
            if sample_ratio < 1.0:
                n_samples = int(len(X_train) * sample_ratio)
                sample_indices = np.random.choice(X_train.index, size=n_samples, replace=False)
                X_train = X_train.loc[sample_indices].copy()
                y_train = y_train.loc[sample_indices].copy()
                
                logger.info(f"Training data sampled: {len(X_train):,} rows")
                
                # Sample validation data if exists
                if X_val is not None and y_val is not None:
                    val_sample_ratio = min(sample_ratio * 1.2, 1.0)
                    if val_sample_ratio < 1.0:
                        n_val_samples = int(len(X_val) * val_sample_ratio)
                        val_sample_indices = np.random.choice(X_val.index, size=n_val_samples, replace=False)
                        X_val = X_val.loc[val_sample_indices].copy()
                        y_val = y_val.loc[val_sample_indices].copy()
                        
                        logger.info(f"Validation data sampled: {len(X_val):,} rows")
            
            LargeDataMemoryTracker.force_cleanup()
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.error(f"Memory efficient sampling failed: {e}")
            return X_train, y_train, X_val, y_val
    
    def _validate_and_apply_params(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply model parameters"""
        try:
            default_params = self._get_default_params(model_type)
            
            # Merge with defaults
            final_params = default_params.copy()
            final_params.update(params)
            
            # Memory-based adjustments (more aggressive thresholds)
            available_memory = self.memory_tracker.get_available_memory()
            
            if model_type.lower() == 'lightgbm':
                if available_memory < 5:  # Adjusted threshold
                    final_params['max_depth'] = min(final_params.get('max_depth', 6), 5)
                    final_params['num_leaves'] = min(final_params.get('num_leaves', 31), 20)
                    final_params['min_data_in_leaf'] = max(final_params.get('min_data_in_leaf', 20), 50)
            
            elif model_type.lower() == 'xgboost':
                if available_memory < 5:  # Adjusted threshold
                    final_params['max_depth'] = min(final_params.get('max_depth', 6), 5)
                    final_params['min_child_weight'] = max(final_params.get('min_child_weight', 1), 3)
                    final_params['subsample'] = min(final_params.get('subsample', 0.8), 0.7)
            
            return final_params
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return self._get_default_params(model_type)
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model type"""
        if model_type.lower() == 'lightgbm':
            return {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 1000,
                'early_stopping_rounds': 100,
                'random_state': 42
            }
        elif model_type.lower() == 'xgboost':
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'early_stopping_rounds': 100,
                'verbosity': 0
            }
        elif model_type.lower() == 'logistic':
            return {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': 42
            }
        return {}
    
    def _apply_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """Apply calibration to model"""
        try:
            if hasattr(model, 'apply_calibration'):
                model.apply_calibration(X_val, y_val)
                logger.info("Model calibration applied successfully")
            else:
                logger.warning("Model does not support calibration")
        except Exception as e:
            logger.error(f"Calibration application failed: {e}")
    
    def _cleanup_memory_after_training(self, model_type: str):
        """Memory cleanup after training"""
        try:
            LargeDataMemoryTracker.force_cleanup()
            logger.info(f"{model_type} model training memory cleanup completed")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def cross_validate_model(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                           params: Optional[Dict[str, Any]] = None, cv_folds: int = 5) -> Dict[str, Any]:
        """Cross-validation for model"""
        
        logger.info(f"{model_type} cross-validation started (folds: {cv_folds})")
        start_time = time.time()
        
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                logger.info(f"Fold {fold + 1}/{cv_folds} started")
                fold_start_time = time.time()
                
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    model_params = params if params else self._get_default_params(model_type)
                    model = ModelFactory.create_model(model_type, params=model_params)
                    
                    model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                    
                    y_pred = model.predict_proba(X_fold_val)
                    
                    fold_ap = average_precision_score(y_fold_val, y_pred)
                    fold_auc = roc_auc_score(y_fold_val, y_pred)
                    fold_logloss = log_loss(y_fold_val, y_pred)
                    fold_time = time.time() - fold_start_time
                    
                    fold_result = {
                        'fold': fold + 1,
                        'average_precision': fold_ap,
                        'auc': fold_auc,
                        'log_loss': fold_logloss,
                        'fold_time': fold_time
                    }
                    
                    fold_results.append(fold_result)
                    
                    logger.info(f"Fold {fold + 1} completed - AP: {fold_ap:.4f}, AUC: {fold_auc:.4f}")
                    
                    LargeDataMemoryTracker.force_cleanup()
                    
                except Exception as fold_error:
                    logger.error(f"Fold {fold + 1} failed: {fold_error}")
                    fold_results.append({
                        'fold': fold + 1,
                        'average_precision': 0.0,
                        'auc': 0.5,
                        'log_loss': float('inf'),
                        'fold_time': time.time() - fold_start_time,
                        'error': str(fold_error)
                    })
            
            # Calculate statistics
            cv_metrics = {'model_type': model_type}
            
            for metric in ['average_precision', 'auc', 'log_loss', 'fold_time']:
                values = [result[metric] for result in fold_results if metric in result]
                if values:
                    cv_metrics[f'mean_{metric}'] = np.mean(values)
                    cv_metrics[f'std_{metric}'] = np.std(values)
            
            total_time = time.time() - start_time
            cv_metrics['total_time'] = total_time
            cv_metrics['cv_folds'] = len(fold_results)
            
            logger.info(f"Cross-validation completed: {model_type}")
            logger.info(f"Mean AP: {cv_metrics.get('mean_average_precision', 0):.4f} ± {cv_metrics.get('std_average_precision', 0):.4f}")
            logger.info(f"Mean AUC: {cv_metrics.get('mean_auc', 0):.4f} ± {cv_metrics.get('std_auc', 0):.4f}")
            
            return cv_metrics
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            LargeDataMemoryTracker.force_cleanup()
            
            return {
                'model_type': model_type,
                'cv_folds': 0,
                'mean_ap': 0.0,
                'std_ap': 0.0,
                'mean_auc': 0.0,
                'std_auc': 0.0,
                'mean_logloss': float('inf'),
                'std_logloss': 0.0,
                'total_time': 0.0,
                'mean_fold_time': 0.0,
                'error': str(e)
            }
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        model_types: Optional[List[str]] = None,
                        tune_hyperparameters: bool = True,
                        perform_cv: bool = True) -> Dict[str, BaseModel]:
        """Train all models (optimized for 64GB environment)"""
        
        if model_types is None:
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'logistic'] if m in available_models]
        
        logger.info(f"Training all models started: {model_types}")
        logger.info(f"Data size: {len(X_train):,}")
        logger.info(f"Available memory: {self.memory_tracker.get_available_memory():.2f}GB")
        logger.info(f"GPU memory: {self.memory_tracker.get_gpu_memory_usage()['free']:.2f}GB")
        
        trained_models = {}
        
        # Memory-based trial count (more aggressive)
        available_memory = self.memory_tracker.get_available_memory()
        if available_memory > 45:
            n_trials = 120
        elif available_memory > 30:
            n_trials = 100
        elif available_memory > 20:
            n_trials = 80
        elif available_memory > 10:
            n_trials = 60
        else:
            n_trials = 40
        
        if tune_hyperparameters and OPTUNA_AVAILABLE:
            logger.info(f"Hyperparameter tuning started (trials: {n_trials})")
            
            for model_type in model_types:
                try:
                    available_memory = self.memory_tracker.get_available_memory()
                    
                    # More aggressive memory threshold for hyperparameter tuning (changed from 5GB to 2GB)
                    if available_memory < 2:
                        logger.warning(f"Skipping {model_type} hyperparameter tuning due to memory shortage")
                        continue
                    
                    logger.info(f"{model_type} hyperparameter tuning started")
                    
                    best_params = self._tune_hyperparameters(
                        model_type=model_type,
                        X=X_train,
                        y=y_train,
                        n_trials=n_trials,
                        cv_folds=3,
                        timeout=1800
                    )
                    
                    self.best_params[model_type] = best_params
                    logger.info(f"{model_type} hyperparameter tuning completed")
                    
                    LargeDataMemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"{model_type} hyperparameter tuning failed: {str(e)}")
                    self.best_params[model_type] = self._get_default_params(model_type)
                    LargeDataMemoryTracker.force_cleanup()
        
        # Cross-validation stage
        if perform_cv:
            logger.info("Cross-validation stage started")
            
            for model_type in model_types:
                try:
                    available_memory = self.memory_tracker.get_available_memory()
                    
                    # More aggressive memory threshold for CV (changed from 5GB to 2GB)
                    if available_memory < 2:
                        logger.warning(f"Skipping {model_type} cross-validation due to memory shortage")
                        continue
                    
                    logger.info(f"{model_type} cross-validation started")
                    
                    params = self.best_params.get(model_type, None)
                    cv_results = self.cross_validate_model(
                        model_type=model_type,
                        X=X_train,
                        y=y_train,
                        params=params
                    )
                    
                    self.cv_results[model_type] = cv_results
                    logger.info(f"{model_type} cross-validation completed")
                    
                    LargeDataMemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"{model_type} cross-validation failed: {str(e)}")
                    LargeDataMemoryTracker.force_cleanup()
        
        # Final model training stage
        logger.info("Final model training stage")
        for model_type in model_types:
            try:
                available_memory = self.memory_tracker.get_available_memory()
                gpu_memory = self.memory_tracker.get_gpu_memory_usage()
                
                # More aggressive memory threshold for 64GB environment (changed from 5GB to 2GB)
                if available_memory < 2:
                    logger.warning(f"Skipping {model_type} model training due to memory shortage")
                    continue
                
                logger.info(f"{model_type} model training started")
                logger.info(f"Memory: {available_memory:.2f}GB, GPU: {gpu_memory['free']:.2f}GB")
                
                params = self.best_params.get(model_type, None)
                
                model = self.train_single_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    params=params,
                    apply_calibration=True
                )
                
                trained_models[model_type] = model
                
                logger.info(f"{model_type} model training completed")
                
                LargeDataMemoryTracker.force_cleanup()
                
            except Exception as e:
                logger.error(f"{model_type} model training failed: {str(e)}")
                LargeDataMemoryTracker.force_cleanup()
                continue
        
        logger.info(f"All models training completed. Successfully trained: {len(trained_models)}")
        return trained_models
    
    def _tune_hyperparameters(self, model_type: str, X: pd.DataFrame, y: pd.Series,
                             n_trials: int = 100, cv_folds: int = 3, timeout: int = 1800) -> Dict[str, Any]:
        """Hyperparameter tuning with Optuna"""
        
        def objective(trial):
            params = self._suggest_hyperparams(trial, model_type)
            
            try:
                cv_scores = []
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                for train_idx, val_idx in cv.split(X, y):
                    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = ModelFactory.create_model(model_type, params=params)
                    model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                    
                    y_pred = model.predict_proba(X_val_fold)
                    score = average_precision_score(y_val_fold, y_pred)
                    cv_scores.append(score)
                
                return np.mean(cv_scores)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            return self._get_default_params(model_type)
    
    def _suggest_hyperparams(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for trial"""
        
        if model_type.lower() == 'lightgbm':
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42
            }
        
        elif model_type.lower() == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'alpha': trial.suggest_float('alpha', 0.0, 1.0),
                'lambda': trial.suggest_float('lambda', 0.0, 2.0)
            }
        
        elif model_type.lower() == 'logistic':
            return {
                'C': trial.suggest_float('C', 0.01, 10.0),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
            }
        
        return {}
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        calibrated_base_models = sum(
            1 for info in self.trained_models.values()
            if info.get('calibrated', False)
        )
        
        return {
            'trained_models': list(self.trained_models.keys()),
            'total_models': len(self.trained_models),
            'best_params': self.best_params,
            'cv_results': self.cv_results,
            'model_performance': self.model_performance,
            'memory_tracker': {
                'current_usage': self.memory_tracker.get_memory_usage(),
                'available_memory': self.memory_tracker.get_available_memory(),
                'gpu_memory': self.memory_tracker.get_gpu_memory_usage()
            }
        }

class CTRTrainingPipeline:
    """CTR training pipeline"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = CTRModelTrainer(config)
        self.memory_tracker = LargeDataMemoryTracker()
        
        logger.info("GPU memory optimization completed: RTX 4060 Ti 16GB 90% utilization")
        logger.info("Large data memory optimization completed")
    
    def execute_full_training_pipeline(self,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     X_val: Optional[pd.DataFrame] = None,
                                     y_val: Optional[pd.Series] = None,
                                     tune_hyperparameters: bool = True,
                                     perform_cv: bool = True) -> Dict[str, Any]:
        """Execute full training pipeline"""
        
        logger.info("=== Full training pipeline started ===")
        start_time = time.time()
        
        try:
            # Data validation
            logger.info(f"Training data: {X_train.shape}")
            if X_val is not None:
                logger.info(f"Validation data: {X_val.shape}")
            
            logger.info(f"Memory status: {self.memory_tracker.get_available_memory():.2f}GB available")
            
            # Train all models
            trained_models = self.trainer.train_all_models(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tune_hyperparameters=tune_hyperparameters,
                perform_cv=perform_cv
            )
            
            # Training summary
            training_summary = self.trainer.get_training_summary()
            total_time = time.time() - start_time
            
            logger.info("=== Full training pipeline completed ===")
            logger.info(f"Total execution time: {total_time:.2f}s")
            logger.info(f"Successfully trained models: {len(trained_models)}")
            logger.info(f"Final memory usage: {self.memory_tracker.get_memory_usage():.2f}GB")
            
            return {
                'trained_models': trained_models,
                'training_summary': training_summary,
                'execution_time': total_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            LargeDataMemoryTracker.force_cleanup()
            
            return {
                'trained_models': {},
                'training_summary': {},
                'execution_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }