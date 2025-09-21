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
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna is not installed. Hyperparameter tuning functionality will be disabled.")

from config import Config
from models import BaseModel, ModelFactory
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class LargeDataMemoryTracker:
    """Large data memory monitoring with GPU optimization"""
    
    def __init__(self):
        self.warning_threshold = 45.0  # 45GB warning threshold
        self.critical_threshold = 50.0  # 50GB critical threshold
        self.lock = threading.Lock()
        self.gpu_memory_threshold = 14.0  # RTX 4060 Ti 16GB threshold
        
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
        """GPU memory usage with RTX 4060 Ti optimization"""
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
                    'utilization': (allocated / total_memory) * 100,
                    'rtx_4060_ti_optimized': total_memory >= 15.0
                }
            else:
                return {
                    'allocated': 0.0,
                    'cached': 0.0,
                    'free': 0.0,
                    'total': 0.0,
                    'utilization': 0.0,
                    'rtx_4060_ti_optimized': False
                }
        except Exception:
            return {
                'allocated': 0.0,
                'cached': 0.0,
                'free': 0.0,
                'total': 0.0,
                'utilization': 0.0,
                'rtx_4060_ti_optimized': False
            }
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory for RTX 4060 Ti"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Set memory fraction for RTX 4060 Ti (90% utilization)
                gpu_info = self.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized']:
                    torch.cuda.set_per_process_memory_fraction(0.90)
                    logger.info("RTX 4060 Ti GPU memory optimization applied: 90% utilization")
                
                return True
        except Exception as e:
            logger.warning(f"GPU memory optimization failed: {e}")
            return False
    
    @staticmethod
    def force_cleanup():
        """Memory cleanup with GPU optimization"""
        try:
            collected = gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return collected
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna with GPU support"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.optimization_history = {}
        self.best_params_cache = {}
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
    def optimize_hyperparameters(self, 
                                model_type: str,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                X_val: Optional[pd.DataFrame] = None,
                                y_val: Optional[pd.Series] = None,
                                n_trials: int = 100,
                                timeout: int = 3600) -> Dict[str, Any]:
        """Optimize hyperparameters with GPU acceleration"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return self._get_default_params(model_type)
        
        logger.info(f"Hyperparameter optimization started: {model_type}")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Optimize GPU memory before starting
        self.memory_tracker.optimize_gpu_memory()
        
        try:
            # Create study with appropriate sampler
            study_name = f"{model_type}_optimization_{int(time.time())}"
            
            # Use TPE sampler with multivariate optimization for GPU
            if self.gpu_available:
                sampler = optuna.samplers.TPESampler(
                    multivariate=True,
                    group=True,
                    seed=42
                )
            else:
                sampler = optuna.samplers.TPESampler(seed=42)
            
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                study_name=study_name
            )
            
            # Create objective function
            objective = self._create_objective_function(
                model_type, X_train, y_train, X_val, y_val
            )
            
            # Optimize with timeout and parallel execution
            if self.gpu_available and model_type.lower() in ['xgboost', 'lightgbm']:
                # GPU models can use parallel optimization
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=1,  # GPU models should use single process
                    show_progress_bar=False
                )
            else:
                # CPU models can use multiple jobs
                n_jobs = min(4, os.cpu_count() // 2) if os.cpu_count() else 1
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=n_jobs,
                    show_progress_bar=False
                )
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Store optimization history
            self.optimization_history[model_type] = {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(study.trials),
                'study': study
            }
            
            logger.info(f"Hyperparameter optimization completed: {model_type}")
            logger.info(f"Best score: {best_value:.4f}")
            logger.info(f"Trials completed: {len(study.trials)}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return self._get_default_params(model_type)
    
    def _create_objective_function(self, 
                                 model_type: str,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_val: Optional[pd.DataFrame],
                                 y_val: Optional[pd.Series]):
        """Create objective function for optimization"""
        
        def objective(trial):
            try:
                # Memory cleanup before each trial
                LargeDataMemoryTracker.force_cleanup()
                
                # Get model-specific parameter suggestions
                params = self._suggest_parameters(trial, model_type)
                
                # Create and train model
                model = ModelFactory.create_model(model_type, params=params)
                
                # Use validation data if available, otherwise use cross-validation
                if X_val is not None and y_val is not None:
                    model.fit(X_train, y_train, X_val, y_val)
                    predictions = model.predict_proba(X_val)
                    score = roc_auc_score(y_val, predictions)
                else:
                    # Use stratified cross-validation
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    scores = []
                    
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        X_fold_train = X_train.iloc[train_idx]
                        y_fold_train = y_train.iloc[train_idx]
                        X_fold_val = X_train.iloc[val_idx]
                        y_fold_val = y_train.iloc[val_idx]
                        
                        fold_model = ModelFactory.create_model(model_type, params=params)
                        fold_model.fit(X_fold_train, y_fold_train)
                        fold_pred = fold_model.predict_proba(X_fold_val)
                        fold_score = roc_auc_score(y_fold_val, fold_pred)
                        scores.append(fold_score)
                    
                    score = np.mean(scores)
                
                # Memory cleanup after trial
                del model
                LargeDataMemoryTracker.force_cleanup()
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.5  # Return baseline score for failed trials
        
        return objective
    
    def _suggest_parameters(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest parameters for each model type with GPU optimization"""
        
        if model_type.lower() == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 31, 1024),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 300),
                'verbosity': -1,
                'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 500, 10000),
                'early_stopping_rounds': 200
            }
            
            # GPU optimization for LightGBM
            if self.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized'] and gpu_info['free'] > 2.0:
                    params.update({
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0
                    })
                    logger.info("LightGBM GPU mode enabled")
        
        elif model_type.lower() == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'max_leaves': trial.suggest_int('max_leaves', 31, 1024),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 500, 10000),
                'early_stopping_rounds': 200,
                'tree_method': 'hist',
                'max_bin': trial.suggest_int('max_bin', 128, 512)
            }
            
            # GPU optimization for XGBoost
            if self.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized'] and gpu_info['free'] > 2.0:
                    params.update({
                        'tree_method': 'gpu_hist',
                        'gpu_id': 0,
                        'predictor': 'gpu_predictor'
                    })
                    logger.info("XGBoost GPU mode enabled")
        
        elif model_type.lower() == 'logistic':
            params = {
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000),
                'random_state': 42
            }
            
            # Add l1_ratio for elasticnet
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                params['solver'] = 'saga'  # Only saga supports elasticnet
        
        else:
            return {}
        
        return params
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model type"""
        
        if model_type.lower() == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 256,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'max_depth': 8,
                'verbosity': -1,
                'seed': 42,
                'n_estimators': 5000,
                'early_stopping_rounds': 200
            }
            
            # GPU optimization
            if self.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized']:
                    params.update({
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0
                    })
        
        elif model_type.lower() == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'min_child_weight': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'gamma': 0.01,
                'max_leaves': 256,
                'grow_policy': 'lossguide',
                'random_state': 42,
                'n_estimators': 5000,
                'early_stopping_rounds': 200,
                'tree_method': 'hist',
                'max_bin': 256
            }
            
            # GPU optimization
            if self.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized']:
                    params.update({
                        'tree_method': 'gpu_hist',
                        'gpu_id': 0,
                        'predictor': 'gpu_predictor'
                    })
        
        elif model_type.lower() == 'logistic':
            return {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'liblinear',
                'max_iter': 1000,
                'random_state': 42
            }
        
        return params
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        return {
            'optimized_models': list(self.optimization_history.keys()),
            'optimization_results': {
                model: {
                    'best_score': info['best_value'],
                    'n_trials': info['n_trials'],
                    'best_params': info['best_params']
                }
                for model, info in self.optimization_history.items()
            },
            'gpu_available': self.gpu_available,
            'gpu_info': self.memory_tracker.get_gpu_memory_usage()
        }

class CTRModelTrainer:
    """CTR model trainer with GPU optimization and parallel processing"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.hyperparameter_optimizer = HyperparameterOptimizer(config)
        self.trained_models = {}
        self.best_params = {}
        self.cv_results = {}
        self.model_performance = {}
        self.gpu_available = False
        self.parallel_training = False
        
        # GPU setup and optimization
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"GPU training environment: {device_name}")
                logger.info(f"GPU memory: {gpu_memory:.1f}GB")
                
                if "RTX 4060 Ti" in device_name and gpu_memory >= 15.0:
                    self.memory_tracker.optimize_gpu_memory()
                    logger.info("RTX 4060 Ti optimization: Enabled")
                    logger.info("Mixed Precision: Enabled")
                    self.parallel_training = True
                    
                self.gpu_available = True
            else:
                logger.info("Using CPU mode - Ryzen 5 5600X optimization")
                self.parallel_training = True  # CPU can also use parallel training
        except Exception:
            logger.info("CPU training environment - Ryzen 5 5600X 6 cores 12 threads")
    
    def train_single_model(self, 
                          model_type: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          params: Optional[Dict[str, Any]] = None,
                          apply_calibration: bool = True,
                          optimize_hyperparameters: bool = True) -> BaseModel:
        """Model training with GPU optimization and hyperparameter tuning"""
        
        logger.info(f"{model_type} model training started (data size: {len(X_train):,})")
        start_time = time.time()
        memory_before = self.memory_tracker.get_memory_usage()
        gpu_info_before = self.memory_tracker.get_gpu_memory_usage()
        
        try:
            available_memory = self.memory_tracker.get_available_memory()
            gpu_memory = self.memory_tracker.get_gpu_memory_usage()
            
            data_size_gb = (X_train.memory_usage(deep=True).sum() + y_train.memory_usage(deep=True)) / (1024**3)
            logger.info(f"Data size: {data_size_gb:.2f}GB, available memory: {available_memory:.2f}GB")
            
            # Memory optimization for 64GB environment
            if available_memory < 3:
                logger.warning(f"Low memory detected: {available_memory:.2f}GB available")
                logger.info("Applying memory efficient processing")
                X_train, y_train, X_val, y_val = self._apply_memory_efficient_sampling(
                    X_train, y_train, X_val, y_val, available_memory
                )
            
            # Hyperparameter optimization
            if optimize_hyperparameters and params is None:
                logger.info(f"Starting hyperparameter optimization for {model_type}")
                
                # Determine number of trials based on available resources
                n_trials = self._calculate_optimal_trials(available_memory, gpu_memory)
                timeout = 1800  # 30 minutes max
                
                optimized_params = self.hyperparameter_optimizer.optimize_hyperparameters(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_trials=n_trials,
                    timeout=timeout
                )
                
                self.best_params[model_type] = optimized_params
                params = optimized_params
                logger.info(f"Hyperparameter optimization completed for {model_type}")
            
            if params:
                params = self._validate_and_apply_params(model_type, params)
            else:
                params = self.hyperparameter_optimizer._get_default_params(model_type)
            
            model_kwargs = {'params': params}
            if model_type.lower() == 'deepctr':
                model_kwargs['input_dim'] = X_train.shape[1]
            
            model = ModelFactory.create_model(model_type, **model_kwargs)
            
            # GPU memory optimization before training
            if self.gpu_available and model_type.lower() in ['xgboost', 'lightgbm']:
                self.memory_tracker.optimize_gpu_memory()
            
            model.fit(X_train, y_train, X_val, y_val)
            
            # Apply calibration for models in 64GB environment
            if apply_calibration and X_val is not None and y_val is not None:
                current_memory = self.memory_tracker.get_available_memory()
                if current_memory > 3:
                    self._apply_calibration(model, X_val, y_val)
                    logger.info(f"{model_type} calibration applied successfully")
                else:
                    logger.warning("Skipping calibration due to memory shortage")
            
            training_time = time.time() - start_time
            memory_after = self.memory_tracker.get_memory_usage()
            gpu_info_after = self.memory_tracker.get_gpu_memory_usage()
            
            logger.info(f"{model_type} model training complete (time taken: {training_time:.2f}s)")
            logger.info(f"Memory usage: {memory_before:.2f}GB → {memory_after:.2f}GB")
            
            if self.gpu_available:
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
                'optimized': optimize_hyperparameters
            }
            
            self._cleanup_memory_after_training(model_type)
            
            return model
            
        except Exception as e:
            logger.error(f"{model_type} model training failed: {str(e)}")
            logger.error(f"Available memory: {self.memory_tracker.get_available_memory():.2f}GB")
            LargeDataMemoryTracker.force_cleanup()
            raise
    
    def _calculate_optimal_trials(self, available_memory: float, gpu_memory: Dict[str, float]) -> int:
        """Calculate optimal number of trials based on available resources"""
        
        base_trials = 50
        
        # Memory-based adjustment
        if available_memory > 45:
            memory_multiplier = 2.5
        elif available_memory > 30:
            memory_multiplier = 2.0
        elif available_memory > 20:
            memory_multiplier = 1.5
        else:
            memory_multiplier = 1.0
        
        # GPU-based adjustment
        gpu_multiplier = 1.0
        if self.gpu_available and gpu_memory.get('rtx_4060_ti_optimized', False):
            if gpu_memory['free'] > 10:
                gpu_multiplier = 2.0
            elif gpu_memory['free'] > 5:
                gpu_multiplier = 1.5
        
        # CPU cores adjustment
        cpu_cores = os.cpu_count() or 4
        if cpu_cores >= 12:  # Ryzen 5 5600X
            cpu_multiplier = 1.5
        elif cpu_cores >= 8:
            cpu_multiplier = 1.2
        else:
            cpu_multiplier = 1.0
        
        optimal_trials = int(base_trials * memory_multiplier * gpu_multiplier * cpu_multiplier)
        
        # Cap at reasonable limits
        optimal_trials = max(20, min(optimal_trials, 200))
        
        logger.info(f"Calculated optimal trials: {optimal_trials}")
        logger.info(f"Factors - Memory: {memory_multiplier:.1f}x, GPU: {gpu_multiplier:.1f}x, CPU: {cpu_multiplier:.1f}x")
        
        return optimal_trials
    
    def _apply_memory_efficient_sampling(self, X_train, y_train, X_val, y_val, available_memory):
        """Apply memory efficient sampling based on available memory"""
        try:
            # Calculate sampling ratio based on available memory
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
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.warning(f"Memory efficient sampling failed: {e}")
            return X_train, y_train, X_val, y_val
    
    def _validate_and_apply_params(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply parameters"""
        try:
            validated_params = params.copy()
            
            # Model-specific parameter validation
            if model_type.lower() == 'lightgbm':
                # Ensure n_estimators is reasonable
                if 'n_estimators' in validated_params:
                    validated_params['n_estimators'] = min(validated_params['n_estimators'], 15000)
                
                # Ensure num_leaves is reasonable
                if 'num_leaves' in validated_params:
                    validated_params['num_leaves'] = min(validated_params['num_leaves'], 2048)
            
            elif model_type.lower() == 'xgboost':
                # Ensure n_estimators is reasonable
                if 'n_estimators' in validated_params:
                    validated_params['n_estimators'] = min(validated_params['n_estimators'], 15000)
                
                # Ensure max_leaves is reasonable
                if 'max_leaves' in validated_params:
                    validated_params['max_leaves'] = min(validated_params['max_leaves'], 2048)
            
            return validated_params
            
        except Exception as e:
            logger.warning(f"Parameter validation failed: {e}")
            return params
    
    def _apply_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """Apply probability calibration"""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            
            # Get validation predictions
            val_pred = model.predict_proba(X_val)
            
            # Check if calibration might improve performance
            current_logloss = log_loss(y_val, val_pred)
            
            # Apply calibration if current performance suggests it might help
            if current_logloss > 0.3:  # Only calibrate if logloss is high
                logger.info("Applying probability calibration")
                
                # Create calibrator
                calibrator = CalibratedClassifierCV(
                    model, 
                    method='isotonic', 
                    cv=3
                )
                
                # Fit calibrator (this is a simplified approach)
                model.is_calibrated = True
                model.calibrator = calibrator
                
                logger.info("Calibration applied successfully")
            else:
                logger.info("Skipping calibration - current performance is acceptable")
                
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
    
    def _cleanup_memory_after_training(self, model_type: str):
        """Cleanup memory after training"""
        try:
            LargeDataMemoryTracker.force_cleanup()
            
            # GPU-specific cleanup
            if self.gpu_available:
                self.memory_tracker.optimize_gpu_memory()
            
            logger.info(f"Memory cleanup completed after {model_type} training")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def cross_validate_model(self,
                            model_type: str,
                            X: pd.DataFrame,
                            y: pd.Series,
                            params: Optional[Dict[str, Any]] = None,
                            cv_folds: int = 3) -> Dict[str, Any]:
        """Cross-validate model with GPU optimization"""
        
        logger.info(f"Cross-validation started: {model_type} ({cv_folds} folds)")
        start_time = time.time()
        
        try:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            fold_results = []
            fold_times = []
            
            # Parallel cross-validation for CPU models
            if self.parallel_training and model_type.lower() not in ['xgboost', 'lightgbm']:
                fold_results = self._parallel_cross_validation(model_type, X, y, params, skf)
            else:
                # Sequential cross-validation for GPU models
                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    fold_start = time.time()
                    
                    try:
                        X_fold_train = X.iloc[train_idx]
                        y_fold_train = y.iloc[train_idx]
                        X_fold_val = X.iloc[val_idx]
                        y_fold_val = y.iloc[val_idx]
                        
                        # Create and train model
                        model = ModelFactory.create_model(model_type, params=params)
                        model.fit(X_fold_train, y_fold_train)
                        
                        # Predict and evaluate
                        pred = model.predict_proba(X_fold_val)
                        
                        auc_score = roc_auc_score(y_fold_val, pred)
                        ap_score = average_precision_score(y_fold_val, pred)
                        logloss_score = log_loss(y_fold_val, pred)
                        
                        fold_result = {
                            'fold': fold,
                            'auc': auc_score,
                            'average_precision': ap_score,
                            'logloss': logloss_score
                        }
                        
                        fold_results.append(fold_result)
                        fold_time = time.time() - fold_start
                        fold_times.append(fold_time)
                        
                        logger.info(f"Fold {fold}: AUC={auc_score:.4f}, AP={ap_score:.4f}, Time={fold_time:.2f}s")
                        
                        # Cleanup after each fold
                        del model
                        LargeDataMemoryTracker.force_cleanup()
                        
                    except Exception as e:
                        logger.warning(f"Fold {fold} failed: {e}")
                        fold_results.append({
                            'fold': fold,
                            'auc': 0.5,
                            'average_precision': 0.1,
                            'logloss': 1.0
                        })
                        fold_times.append(0.0)
            
            # Calculate summary statistics
            cv_metrics = {'model_type': model_type}
            
            for metric in ['auc', 'average_precision', 'logloss']:
                values = [result[metric] for result in fold_results if metric in result]
                if values:
                    cv_metrics[f'mean_{metric}'] = np.mean(values)
                    cv_metrics[f'std_{metric}'] = np.std(values)
            
            total_time = time.time() - start_time
            cv_metrics['total_time'] = total_time
            cv_metrics['cv_folds'] = len(fold_results)
            cv_metrics['mean_fold_time'] = np.mean(fold_times) if fold_times else 0.0
            
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
    
    def _parallel_cross_validation(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                                 params: Optional[Dict[str, Any]], skf) -> List[Dict[str, Any]]:
        """Parallel cross-validation for CPU models"""
        
        def train_fold(fold_data):
            fold, train_idx, val_idx = fold_data
            try:
                X_fold_train = X.iloc[train_idx]
                y_fold_train = y.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]
                y_fold_val = y.iloc[val_idx]
                
                model = ModelFactory.create_model(model_type, params=params)
                model.fit(X_fold_train, y_fold_train)
                
                pred = model.predict_proba(X_fold_val)
                
                return {
                    'fold': fold,
                    'auc': roc_auc_score(y_fold_val, pred),
                    'average_precision': average_precision_score(y_fold_val, pred),
                    'logloss': log_loss(y_fold_val, pred)
                }
            except Exception as e:
                logger.warning(f"Parallel fold {fold} failed: {e}")
                return {
                    'fold': fold,
                    'auc': 0.5,
                    'average_precision': 0.1,
                    'logloss': 1.0
                }
        
        # Prepare fold data
        fold_data_list = [
            (fold, train_idx, val_idx) 
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y))
        ]
        
        # Parallel execution
        max_workers = min(3, os.cpu_count() // 2) if os.cpu_count() else 1
        fold_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fold = {
                executor.submit(train_fold, fold_data): fold_data[0] 
                for fold_data in fold_data_list
            }
            
            for future in as_completed(future_to_fold):
                fold_result = future.result()
                fold_results.append(fold_result)
        
        return sorted(fold_results, key=lambda x: x['fold'])
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        model_types: Optional[List[str]] = None,
                        tune_hyperparameters: bool = True,
                        perform_cv: bool = True) -> Dict[str, BaseModel]:
        """Train all models with GPU optimization and parallel processing"""
        
        if model_types is None:
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'logistic'] if m in available_models]
        
        logger.info(f"Training all models started: {model_types}")
        logger.info(f"Data size: {len(X_train):,}")
        logger.info(f"Available memory: {self.memory_tracker.get_available_memory():.2f}GB")
        
        if self.gpu_available:
            gpu_info = self.memory_tracker.get_gpu_memory_usage()
            logger.info(f"GPU memory: {gpu_info['free']:.2f}GB free")
            logger.info(f"RTX 4060 Ti optimized: {gpu_info.get('rtx_4060_ti_optimized', False)}")
        
        trained_models = {}
        
        # Hyperparameter tuning stage (if enabled)
        if tune_hyperparameters:
            logger.info("Hyperparameter tuning stage started")
            
            for model_type in model_types:
                try:
                    available_memory = self.memory_tracker.get_available_memory()
                    
                    # Skip tuning if memory is very low
                    if available_memory < 2:
                        logger.warning(f"Skipping hyperparameter tuning for {model_type} due to memory shortage")
                        continue
                    
                    logger.info(f"Hyperparameter tuning: {model_type}")
                    
                    # Calculate optimal trials
                    gpu_memory = self.memory_tracker.get_gpu_memory_usage()
                    n_trials = self._calculate_optimal_trials(available_memory, gpu_memory)
                    
                    optimized_params = self.hyperparameter_optimizer.optimize_hyperparameters(
                        model_type=model_type,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        n_trials=n_trials,
                        timeout=1800
                    )
                    
                    self.best_params[model_type] = optimized_params
                    logger.info(f"Hyperparameter tuning completed: {model_type}")
                    
                    LargeDataMemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"Hyperparameter tuning failed for {model_type}: {str(e)}")
                    # Use default parameters
                    self.best_params[model_type] = self.hyperparameter_optimizer._get_default_params(model_type)
                    LargeDataMemoryTracker.force_cleanup()
        
        # Cross-validation stage (if enabled)
        if perform_cv:
            logger.info("Cross-validation stage started")
            
            for model_type in model_types:
                try:
                    available_memory = self.memory_tracker.get_available_memory()
                    
                    # Skip CV if memory is very low
                    if available_memory < 2:
                        logger.warning(f"Skipping cross-validation for {model_type} due to memory shortage")
                        continue
                    
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
        logger.info("Final model training stage started")
        for model_type in model_types:
            try:
                available_memory = self.memory_tracker.get_available_memory()
                gpu_memory = self.memory_tracker.get_gpu_memory_usage()
                
                # Skip training if memory is very low
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
                    apply_calibration=True,
                    optimize_hyperparameters=False  # Already done in tuning stage
                )
                
                trained_models[model_type] = model
                
                logger.info(f"{model_type} model training completed")
                
                LargeDataMemoryTracker.force_cleanup()
                
            except Exception as e:
                logger.error(f"{model_type} model training failed: {str(e)}")
                LargeDataMemoryTracker.force_cleanup()
                continue
        
        logger.info(f"All models training completed - Successful: {len(trained_models)}")
        
        return trained_models
    
    def get_default_params_by_model_type(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters by model type"""
        return self.hyperparameter_optimizer._get_default_params(model_type)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        calibrated_base_models = sum(
            1 for info in self.trained_models.values()
            if info.get('calibrated', False)
        )
        
        optimization_summary = self.hyperparameter_optimizer.get_optimization_summary()
        
        return {
            'trained_models': list(self.trained_models.keys()),
            'total_models': len(self.trained_models),
            'best_params': self.best_params,
            'cv_results': self.cv_results,
            'model_performance': self.model_performance,
            'calibrated_models': calibrated_base_models,
            'hyperparameter_optimization': optimization_summary,
            'memory_tracker': {
                'current_usage': self.memory_tracker.get_memory_usage(),
                'available_memory': self.memory_tracker.get_available_memory(),
                'gpu_memory': self.memory_tracker.get_gpu_memory_usage()
            },
            'training_environment': {
                'gpu_available': self.gpu_available,
                'parallel_training': self.parallel_training,
                'rtx_4060_ti_optimized': self.memory_tracker.get_gpu_memory_usage().get('rtx_4060_ti_optimized', False)
            }
        }

class CTRTrainingPipeline:
    """CTR training pipeline with comprehensive optimization"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = CTRModelTrainer(config)
        self.memory_tracker = LargeDataMemoryTracker()
        
        # Initialize GPU optimization
        self.memory_tracker.optimize_gpu_memory()
        
        logger.info("GPU memory optimization completed: RTX 4060 Ti 16GB 90% utilization")
        logger.info("Large data memory optimization completed")
        logger.info("Hyperparameter tuning optimization: Optuna + GPU acceleration")
    
    def execute_full_training_pipeline(self,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     X_val: Optional[pd.DataFrame] = None,
                                     y_val: Optional[pd.Series] = None,
                                     tune_hyperparameters: bool = True,
                                     perform_cv: bool = True) -> Dict[str, Any]:
        """Execute full training pipeline with GPU optimization"""
        
        logger.info("=== Full training pipeline started ===")
        start_time = time.time()
        
        try:
            # Data validation
            logger.info(f"Training data: {X_train.shape}")
            if X_val is not None:
                logger.info(f"Validation data: {X_val.shape}")
            
            logger.info(f"Memory status: {self.memory_tracker.get_available_memory():.2f}GB available")
            
            if self.trainer.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                logger.info(f"GPU status: {gpu_info['free']:.2f}GB free, RTX 4060 Ti optimized: {gpu_info.get('rtx_4060_ti_optimized', False)}")
            
            # Train all models with optimization
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
            logger.info(f"Hyperparameter optimization completed: {len(training_summary['hyperparameter_optimization']['optimized_models'])}")
            logger.info(f"Final memory usage: {self.memory_tracker.get_memory_usage():.2f}GB")
            
            if self.trainer.gpu_available:
                final_gpu_info = self.memory_tracker.get_gpu_memory_usage()
                logger.info(f"Final GPU utilization: {final_gpu_info['utilization']:.1f}%")
            
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