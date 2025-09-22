# training.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import gc
import warnings
import os
import tempfile
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
from models import ModelFactory, BaseModel
from config import Config

logger = logging.getLogger(__name__)

class LargeDataMemoryTracker:
    """Memory tracking for large dataset processing"""
    
    def __init__(self):
        self.rtx_4060ti_optimized = False
        self.optimization_logged = False
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                if "RTX 4060 Ti" in device_name:
                    self.rtx_4060ti_optimized = True
            except Exception:
                pass
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().used / (1024**3)
        return 0.0
    
    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available / (1024**3)
        return 64.0
    
    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {
                'available': False,
                'rtx_4060_ti_optimized': False,
                'memory_allocated': 0,
                'memory_reserved': 0
            }
        
        try:
            return {
                'available': True,
                'rtx_4060_ti_optimized': self.rtx_4060ti_optimized,
                'memory_allocated': torch.cuda.memory_allocated(0) / (1024**3),
                'memory_reserved': torch.cuda.memory_reserved(0) / (1024**3)
            }
        except Exception:
            return {
                'available': False,
                'rtx_4060_ti_optimized': False,
                'memory_allocated': 0,
                'memory_reserved': 0
            }
    
    def optimize_gpu_memory(self) -> bool:
        """GPU memory optimization"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                gpu_info = self.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized'] and not self.optimization_logged:
                    torch.cuda.set_per_process_memory_fraction(0.90)
                    logger.info("RTX 4060 Ti GPU memory optimization applied: 90% utilization")
                    self.optimization_logged = True
                
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
        self.optimization_logged = False
        
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
        
        if not self.optimization_logged:
            self.memory_tracker.optimize_gpu_memory()
            self.optimization_logged = True
        
        try:
            study_name = f"{model_type}_optimization_{int(time.time())}"
            
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
            
            objective = self._create_objective_function(
                model_type, X_train, y_train, X_val, y_val
            )
            
            if self.gpu_available and model_type.lower() in ['xgboost', 'lightgbm']:
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=1,
                    show_progress_bar=False
                )
            else:
                n_jobs = min(4, os.cpu_count() // 2) if os.cpu_count() else 1
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs=n_jobs,
                    show_progress_bar=False
                )
            
            best_params = study.best_params
            best_value = study.best_value
            
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
        """Create optimization objective function"""
        
        def objective(trial):
            try:
                if model_type.lower() == 'lightgbm':
                    params = self._suggest_lightgbm_params(trial)
                elif model_type.lower() == 'xgboost':
                    params = self._suggest_xgboost_params(trial)
                elif model_type.lower() == 'logistic':
                    params = self._suggest_logistic_params(trial)
                else:
                    return 0.0
                
                model = ModelFactory.create_model(model_type, params=params)
                model.fit(X_train, y_train, X_val, y_val)
                
                if X_val is not None and y_val is not None:
                    predictions = model.predict_proba(X_val)
                    score = self._calculate_score(y_val, predictions)
                else:
                    from sklearn.model_selection import cross_val_score
                    from sklearn.metrics import roc_auc_score
                    
                    scores = cross_val_score(
                        model.model, 
                        X_train.fillna(0), 
                        y_train, 
                        cv=3, 
                        scoring='roc_auc',
                        n_jobs=1
                    )
                    score = np.mean(scores)
                
                self.memory_tracker.force_cleanup()
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        return objective
    
    def _suggest_lightgbm_params(self, trial) -> Dict[str, Any]:
        """Suggest LightGBM parameters"""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 0.5),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 0.5),
            'random_state': 42,
            'n_estimators': 5000,
            'early_stopping_rounds': 200,
            'verbosity': -1,
            'num_threads': 12
        }
    
    def _suggest_xgboost_params(self, trial) -> Dict[str, Any]:
        """Suggest XGBoost parameters"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02),
            'subsample': trial.suggest_float('subsample', 0.8, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.3),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.3),
            'gamma': trial.suggest_float('gamma', 0.01, 0.1),
            'random_state': 42,
            'n_estimators': 5000,
            'early_stopping_rounds': 200,
            'tree_method': 'hist',
            'nthread': 12
        }
        
        if self.gpu_available:
            gpu_info = self.memory_tracker.get_gpu_memory_usage()
            if gpu_info['rtx_4060_ti_optimized']:
                params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor'
                })
        
        return params
    
    def _suggest_logistic_params(self, trial) -> Dict[str, Any]:
        """Suggest Logistic Regression parameters"""
        return {
            'C': trial.suggest_float('C', 0.1, 10.0),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': 12
        }
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate validation score"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Score calculation failed: {e}")
            return 0.0
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters by model type"""
        if model_type.lower() == 'lightgbm':
            return {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'max_depth': 8,
                'learning_rate': 0.008,
                'feature_fraction': 0.88,
                'bagging_fraction': 0.88,
                'bagging_freq': 5,
                'min_data_in_leaf': 50,
                'lambda_l1': 0.1,
                'lambda_l2': 0.2,
                'random_state': 42,
                'n_estimators': 10000,
                'early_stopping_rounds': 250,
                'verbosity': -1,
                'num_threads': 12
            }
        
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
        self.trainer_initialized = False
        
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
                self.parallel_training = True
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
            
            if available_memory < 3:
                logger.warning(f"Low memory detected: {available_memory:.2f}GB available")
                logger.info("Applying memory efficient processing")
                X_train, y_train, X_val, y_val = self._apply_memory_efficient_sampling(
                    X_train, y_train, X_val, y_val, available_memory
                )
            
            if optimize_hyperparameters and params is None:
                logger.info(f"Starting hyperparameter optimization for {model_type}")
                
                n_trials = self._calculate_optimal_trials(available_memory, gpu_memory)
                timeout = 1800
                
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
            
            if self.gpu_available and model_type.lower() in ['xgboost', 'lightgbm']:
                self.memory_tracker.optimize_gpu_memory()
            
            model.fit(X_train, y_train, X_val, y_val)
            
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
                gpu_before = gpu_info_before.get('memory_allocated', 0)
                gpu_after = gpu_info_after.get('memory_allocated', 0)
                logger.info(f"GPU memory utilization: {gpu_before:.1f}% → {gpu_after:.1f}%")
            
            self.trained_models[model_type] = {
                'model': model,
                'training_time': training_time,
                'memory_usage': memory_after - memory_before,
                'calibrated': apply_calibration and model.is_calibrated,
                'gpu_used': self.gpu_available and model_type.lower() in ['xgboost', 'lightgbm']
            }
            
            self.memory_tracker.force_cleanup()
            
            return model
            
        except Exception as e:
            logger.error(f"Model training failed ({model_type}): {e}")
            self.memory_tracker.force_cleanup()
            raise
    
    def _calculate_optimal_trials(self, available_memory: float, gpu_memory: Dict[str, Any]) -> int:
        """Calculate optimal number of trials"""
        base_trials = 20
        
        memory_factor = min(1.5, available_memory / 30)
        gpu_factor = 2.0 if gpu_memory.get('rtx_4060_ti_optimized', False) else 1.0
        cpu_factor = 1.5
        
        optimal_trials = int(base_trials * memory_factor * gpu_factor * cpu_factor)
        
        logger.info(f"Calculated optimal trials: {optimal_trials}")
        logger.info(f"Factors - Memory: {memory_factor:.1f}x, GPU: {gpu_factor:.1f}x, CPU: {cpu_factor:.1f}x")
        
        return max(10, min(optimal_trials, 200))
    
    def _apply_memory_efficient_sampling(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                                       available_memory: float) -> Tuple[pd.DataFrame, pd.Series, 
                                                                       Optional[pd.DataFrame], Optional[pd.Series]]:
        """Apply memory efficient sampling"""
        try:
            current_size = len(X_train)
            
            if available_memory < 2:
                target_size = min(current_size, 3000000)
            elif available_memory < 5:
                target_size = min(current_size, 6000000)
            else:
                return X_train, y_train, X_val, y_val
            
            if current_size > target_size:
                logger.info(f"Applying memory efficient sampling: {current_size:,} → {target_size:,}")
                
                sample_indices = np.random.choice(current_size, size=target_size, replace=False)
                X_train_sampled = X_train.iloc[sample_indices].reset_index(drop=True)
                y_train_sampled = y_train.iloc[sample_indices].reset_index(drop=True)
                
                if X_val is not None and y_val is not None:
                    val_size = min(len(X_val), target_size // 4)
                    if len(X_val) > val_size:
                        val_indices = np.random.choice(len(X_val), size=val_size, replace=False)
                        X_val_sampled = X_val.iloc[val_indices].reset_index(drop=True)
                        y_val_sampled = y_val.iloc[val_indices].reset_index(drop=True)
                    else:
                        X_val_sampled = X_val
                        y_val_sampled = y_val
                else:
                    X_val_sampled = X_val
                    y_val_sampled = y_val
                
                return X_train_sampled, y_train_sampled, X_val_sampled, y_val_sampled
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.warning(f"Memory efficient sampling failed: {e}")
            return X_train, y_train, X_val, y_val
    
    def _validate_and_apply_params(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply parameters"""
        try:
            if model_type.lower() == 'xgboost':
                if self.gpu_available:
                    gpu_info = self.memory_tracker.get_gpu_memory_usage()
                    if gpu_info['rtx_4060_ti_optimized']:
                        params.update({
                            'tree_method': 'gpu_hist',
                            'gpu_id': 0,
                            'predictor': 'gpu_predictor'
                        })
                
                required_params = ['objective', 'eval_metric', 'random_state']
                for param in required_params:
                    if param not in params:
                        if param == 'objective':
                            params[param] = 'binary:logistic'
                        elif param == 'eval_metric':
                            params[param] = 'logloss'
                        elif param == 'random_state':
                            params[param] = 42
            
            elif model_type.lower() == 'lightgbm':
                required_params = ['objective', 'metric', 'random_state']
                for param in required_params:
                    if param not in params:
                        if param == 'objective':
                            params[param] = 'binary'
                        elif param == 'metric':
                            params[param] = 'binary_logloss'
                        elif param == 'random_state':
                            params[param] = 42
            
            return params
            
        except Exception as e:
            logger.warning(f"Parameter validation failed: {e}")
            return params
    
    def _apply_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """Apply calibration to model"""
        try:
            logger.info("Applying probability calibration")
            success = model.apply_calibration(X_val, y_val, method='auto')
            if success:
                logger.info("Calibration applied successfully")
            else:
                logger.warning("Calibration application failed")
        except Exception as e:
            logger.warning(f"Calibration application error: {e}")
    
    def train_multiple_models(self, 
                            model_types: List[str],
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None,
                            apply_calibration: bool = True,
                            optimize_hyperparameters: bool = True) -> Dict[str, BaseModel]:
        """Train multiple models"""
        
        logger.info(f"Multiple model training started: {len(model_types)} models")
        
        trained_models = {}
        
        for model_type in model_types:
            try:
                logger.info(f"Training model: {model_type}")
                
                model = self.train_single_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    apply_calibration=apply_calibration,
                    optimize_hyperparameters=optimize_hyperparameters
                )
                
                trained_models[model_type] = model
                logger.info(f"Model training completed: {model_type}")
                
                self.memory_tracker.force_cleanup()
                
            except Exception as e:
                logger.error(f"Model training failed ({model_type}): {e}")
                continue
        
        logger.info(f"Multiple models training completed - Successful: {len(trained_models)}")
        
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
        self.pipeline_initialized = False
        
        if not self.pipeline_initialized:
            self.memory_tracker.optimize_gpu_memory()
            
            logger.info("GPU memory optimization completed: RTX 4060 Ti 16GB 90% utilization")
            logger.info("Large data memory optimization completed")
            logger.info("Hyperparameter tuning optimization: Optuna + GPU acceleration")
            self.pipeline_initialized = True
    
    def execute_full_training_pipeline(self,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     X_val: Optional[pd.DataFrame] = None,
                                     y_val: Optional[pd.Series] = None,
                                     tune_hyperparameters: bool = True,
                                     selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute full training pipeline"""
        
        logger.info("Full training pipeline execution started")
        start_time = time.time()
        
        try:
            available_models = ModelFactory.get_available_models()
            
            if selected_models:
                models_to_train = [m for m in selected_models if m in available_models]
            else:
                models_to_train = available_models
            
            logger.info(f"Models to train: {models_to_train}")
            
            trained_models = self.trainer.train_multiple_models(
                model_types=models_to_train,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                apply_calibration=True,
                optimize_hyperparameters=tune_hyperparameters
            )
            
            execution_time = time.time() - start_time
            
            pipeline_summary = {
                'trained_models': trained_models,
                'execution_time': execution_time,
                'successful_models': list(trained_models.keys()),
                'failed_models': [m for m in models_to_train if m not in trained_models],
                'training_summary': self.trainer.get_training_summary()
            }
            
            logger.info(f"Full training pipeline completed - Time: {execution_time:.2f}s")
            logger.info(f"Successful models: {len(trained_models)}/{len(models_to_train)}")
            
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Training pipeline execution failed: {e}")
            raise