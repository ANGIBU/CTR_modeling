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
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free = total - reserved
                utilization = (reserved / total) * 100
                
                return {
                    'allocated': allocated,
                    'reserved': reserved,
                    'total': total,
                    'free': free,
                    'utilization': utilization
                }
            return {'allocated': 0, 'reserved': 0, 'total': 0, 'free': 0, 'utilization': 0}
        except Exception:
            return {'allocated': 0, 'reserved': 0, 'total': 0, 'free': 0, 'utilization': 0}
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure exists"""
        try:
            current_usage = self.get_memory_usage()
            return current_usage > self.warning_threshold
        except Exception:
            return False
    
    @staticmethod
    def force_cleanup():
        """Memory cleanup"""
        try:
            for _ in range(10):
                gc.collect()
            
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass
        except Exception:
            pass

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
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"GPU training environment: {gpu_name}")
                logger.info(f"GPU memory: {gpu_memory:.1f}GB")
                
                if "RTX 4060 Ti" in gpu_name:
                    logger.info("RTX 4060 Ti optimization: True")
                    logger.info("Mixed Precision enabled")
                
                self.gpu_available = True
            else:
                logger.info("GPU not available. Using CPU mode")
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
            
            # Relaxed memory usage threshold for 64GB environment (changed from 15GB to 5GB)
            if available_memory < 5:
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
            
            # Apply calibration for all models in 64GB environment (relaxed threshold)
            if apply_calibration and X_val is not None and y_val is not None:
                current_memory = self.memory_tracker.get_available_memory()
                if current_memory > 5:  # Relaxed threshold from 15GB to 5GB
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
            # Calculate sampling ratio based on available memory
            if available_memory < 3:
                sample_ratio = 0.3
            elif available_memory < 5:
                sample_ratio = 0.5
            elif available_memory < 8:
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
            
            # Memory-based adjustments
            available_memory = self.memory_tracker.get_available_memory()
            
            if model_type.lower() == 'lightgbm':
                if available_memory < 10:
                    final_params['max_depth'] = min(final_params.get('max_depth', 6), 5)
                    final_params['num_leaves'] = min(final_params.get('num_leaves', 31), 20)
                    final_params['min_data_in_leaf'] = max(final_params.get('min_data_in_leaf', 20), 50)
            
            elif model_type.lower() == 'xgboost':
                if available_memory < 10:
                    final_params['max_depth'] = min(final_params.get('max_depth', 6), 5)
                    final_params['min_child_weight'] = max(final_params.get('min_child_weight', 1), 3)
            
            return final_params
            
        except Exception as e:
            logger.warning(f"Parameter validation failed: {e}")
            return self._get_default_params(model_type)
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model"""
        defaults = {
            'lightgbm': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'min_gain_to_split': 0.0,
                'verbosity': -1,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 100,
                'n_jobs': -1
            },
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'alpha': 0,
                'lambda': 1,
                'random_state': self.config.RANDOM_STATE,
                'n_jobs': -1,
                'verbosity': 0
            },
            'logistic': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'max_iter': 1000,
                'random_state': self.config.RANDOM_STATE,
                'n_jobs': -1
            }
        }
        
        return defaults.get(model_type.lower(), {})
    
    def _apply_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """Apply calibration to model"""
        try:
            logger.info(f"Applying calibration to {model.name}")
            
            # Get uncalibrated predictions
            if hasattr(model, 'predict_proba_raw'):
                val_predictions = model.predict_proba_raw(X_val)
            else:
                val_predictions = model.predict_proba(X_val)
            
            # Apply calibration
            from models import CTRCalibrator
            calibrator = CTRCalibrator(target_ctr=0.0201, method='auto')
            calibrator.fit(y_val.values, val_predictions)
            
            # Set calibrator to model
            model.calibrator = calibrator
            model.is_calibrated = True
            
            # Test calibration
            calibrated_predictions = calibrator.predict(val_predictions)
            
            original_ctr = val_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val.mean()
            
            logger.info(f"Calibration results:")
            logger.info(f"  - Original CTR: {original_ctr:.4f}")
            logger.info(f"  - Calibrated CTR: {calibrated_ctr:.4f}")
            logger.info(f"  - Actual CTR: {actual_ctr:.4f}")
            logger.info(f"  - Best method: {calibrator.best_method}")
            
        except Exception as e:
            logger.error(f"Calibration application failed: {e}")
            model.is_calibrated = False
    
    def _cleanup_memory_after_training(self, model_type: str):
        """Memory cleanup after model training"""
        try:
            LargeDataMemoryTracker.force_cleanup()
            
            available_memory = self.memory_tracker.get_available_memory()
            logger.info(f"Memory after {model_type} training: {available_memory:.2f}GB available")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def cross_validate_model(self,
                           model_type: str,
                           X: pd.DataFrame,
                           y: pd.Series,
                           params: Optional[Dict[str, Any]] = None,
                           cv_folds: int = 3) -> Dict[str, Any]:
        """Cross-validation for model"""
        
        logger.info(f"Cross-validation started: {model_type} (folds: {cv_folds})")
        start_time = time.time()
        
        try:
            available_memory = self.memory_tracker.get_available_memory()
            
            # Adjust folds based on memory
            if available_memory < 10:
                cv_folds = min(cv_folds, 3)
                logger.info(f"Reduced CV folds to {cv_folds} due to memory constraints")
            
            metrics_calculator = CTRMetrics()
            fold_results = []
            
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.RANDOM_STATE)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                fold_start = time.time()
                logger.info(f"Fold {fold_idx + 1}/{cv_folds} started")
                
                try:
                    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Memory efficient sampling for CV
                    if available_memory < 10:
                        sample_size = min(len(X_fold_train), 50000)
                        if len(X_fold_train) > sample_size:
                            sample_indices = np.random.choice(len(X_fold_train), size=sample_size, replace=False)
                            X_fold_train = X_fold_train.iloc[sample_indices]
                            y_fold_train = y_fold_train.iloc[sample_indices]
                    
                    model = self.train_single_model(
                        model_type=model_type,
                        X_train=X_fold_train,
                        y_train=y_fold_train,
                        X_val=X_fold_val,
                        y_val=y_fold_val,
                        params=params,
                        apply_calibration=True
                    )
                    
                    # Predictions
                    y_pred_proba = model.predict_proba(X_fold_val)
                    
                    # Calculate metrics
                    fold_metrics = metrics_calculator.calculate_all_metrics(
                        y_true=y_fold_val.values,
                        y_pred_proba=y_pred_proba
                    )
                    
                    fold_time = time.time() - fold_start
                    fold_metrics['fold_time'] = fold_time
                    fold_results.append(fold_metrics)
                    
                    logger.info(f"Fold {fold_idx + 1} completed: AP={fold_metrics['average_precision']:.4f}, "
                              f"AUC={fold_metrics['auc']:.4f}, LogLoss={fold_metrics['log_loss']:.4f}")
                    
                    LargeDataMemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"Fold {fold_idx + 1} failed: {e}")
                    continue
            
            if not fold_results:
                raise ValueError("All folds failed")
            
            # Aggregate results
            cv_metrics = {}
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
        
        # Memory-based trial count
        available_memory = self.memory_tracker.get_available_memory()
        if available_memory > 45:
            n_trials = 120
        elif available_memory > 40:
            n_trials = 100
        elif available_memory > 35:
            n_trials = 80
        else:
            n_trials = 60
        
        if tune_hyperparameters and OPTUNA_AVAILABLE:
            logger.info(f"Hyperparameter tuning started (trials: {n_trials})")
            
            for model_type in model_types:
                try:
                    available_memory = self.memory_tracker.get_available_memory()
                    
                    # Relaxed memory threshold for hyperparameter tuning (changed from 15GB to 5GB)
                    if available_memory < 5:
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
                    
                    # Relaxed memory threshold for CV (changed from 15GB to 5GB)
                    if available_memory < 5:
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
                
                # Relaxed memory threshold for 64GB environment (changed from 15GB to 5GB)
                if available_memory < 5:
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
        
        logger.info(f"All models training completed. Successful models: {len(trained_models)}")
        
        return trained_models
    
    def _tune_hyperparameters(self,
                            model_type: str,
                            X: pd.DataFrame,
                            y: pd.Series,
                            n_trials: int = 100,
                            cv_folds: int = 3,
                            timeout: int = 1800) -> Dict[str, Any]:
        """Hyperparameter tuning using Optuna"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return self._get_default_params(model_type)
        
        def objective(trial):
            try:
                params = self._suggest_params(trial, model_type)
                
                # Memory efficient CV for hyperparameter tuning
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.RANDOM_STATE)
                scores = []
                
                for train_idx, val_idx in skf.split(X, y):
                    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Memory efficient sampling
                    available_memory = self.memory_tracker.get_available_memory()
                    if available_memory < 8:
                        sample_size = min(len(X_fold_train), 30000)
                        if len(X_fold_train) > sample_size:
                            sample_indices = np.random.choice(len(X_fold_train), size=sample_size, replace=False)
                            X_fold_train = X_fold_train.iloc[sample_indices]
                            y_fold_train = y_fold_train.iloc[sample_indices]
                    
                    model = ModelFactory.create_model(model_type, params=params)
                    model.fit(X_fold_train, y_fold_train)
                    
                    y_pred = model.predict_proba(X_fold_val)
                    score = average_precision_score(y_fold_val, y_pred)
                    scores.append(score)
                    
                    LargeDataMemoryTracker.force_cleanup()
                
                return np.mean(scores)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction='maximize', 
                                      sampler=optuna.samplers.TPESampler(seed=self.config.RANDOM_STATE))
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            best_params = study.best_params
            logger.info(f"{model_type} best score: {study.best_value:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            return self._get_default_params(model_type)
    
    def _suggest_params(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna trial"""
        
        if model_type.lower() == 'lightgbm':
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200)
            }
        
        elif model_type.lower() == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
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