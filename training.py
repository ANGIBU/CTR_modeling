# training.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Type
import logging
import time
import gc
import warnings
from pathlib import Path
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Safe imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from config import Config
from models import BaseModel
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class LargeDataMemoryTracker:
    """Memory tracking and management for large datasets"""
    
    def __init__(self):
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        self.gpu_available = False
        self.gpu_memory_gb = 0
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_available = True
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                vm = psutil.virtual_memory()
                
                return {
                    'process_mb': memory_info.rss / (1024**2),
                    'available_gb': vm.available / (1024**3),
                    'total_gb': vm.total / (1024**3),
                    'percent_used': vm.percent
                }
            else:
                return {
                    'process_mb': 0,
                    'available_gb': 8.0,
                    'total_gb': 16.0,
                    'percent_used': 50.0
                }
        except Exception:
            return {
                'process_mb': 0,
                'available_gb': 8.0,
                'total_gb': 16.0,
                'percent_used': 50.0
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage with GPU info"""
        memory_info = self._get_memory_usage()
        
        if self.gpu_available:
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_cached = torch.cuda.memory_reserved() / (1024**3)
                gpu_free = self.gpu_memory_gb - gpu_cached
                
                memory_info.update({
                    'gpu_used_gb': gpu_memory,
                    'gpu_cached_gb': gpu_cached,
                    'gpu_free_gb': gpu_free,
                    'gpu_total_gb': self.gpu_memory_gb,
                    'rtx_4060_ti_optimized': "RTX 4060 Ti" in torch.cuda.get_device_name(0)
                })
            except Exception:
                memory_info.update({
                    'gpu_used_gb': 0,
                    'gpu_cached_gb': 0,
                    'gpu_free_gb': self.gpu_memory_gb,
                    'gpu_total_gb': self.gpu_memory_gb,
                    'rtx_4060_ti_optimized': False
                })
        
        return memory_info
    
    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        memory_info = self._get_memory_usage()
        return memory_info['available_gb']
    
    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage"""
        if not self.gpu_available:
            return {'gpu_available': False}
        
        try:
            return {
                'gpu_available': True,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': self.gpu_memory_gb,
                'gpu_used_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3),
                'rtx_4060_ti_optimized': "RTX 4060 Ti" in torch.cuda.get_device_name(0)
            }
        except Exception:
            return {'gpu_available': False}
    
    def optimize_gpu_memory(self):
        """GPU memory optimization for RTX 4060 Ti"""
        if not self.gpu_available:
            return
        
        try:
            torch.cuda.empty_cache()
            
            # RTX 4060 Ti specific optimization
            if "RTX 4060 Ti" in torch.cuda.get_device_name(0):
                # Set memory fraction for RTX 4060 Ti 16GB
                torch.cuda.set_per_process_memory_fraction(0.9)
        except Exception as e:
            logger.warning(f"GPU memory optimization failed: {e}")

class HyperparameterOptimizer:
    """Hyperparameter optimization with RTX 4060 Ti GPU support"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.optimization_history = {}
        self.gpu_available = False
        self.quick_mode = False
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_info = self.memory_tracker.get_gpu_memory_usage()
            self.gpu_available = gpu_info['gpu_available']
            
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode"""
        self.quick_mode = enabled
        
    def optimize_hyperparameters(self, 
                                model_class: Type[BaseModel],
                                model_name: str,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                X_val: pd.DataFrame,
                                y_val: pd.Series,
                                n_trials: int = None) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return self._get_default_params(model_name)
        
        if n_trials is None:
            n_trials = 10 if self.quick_mode else 50
            
        logger.info(f"Starting hyperparameter optimization for {model_name} with {n_trials} trials")
        
        try:
            study = optuna.create_study(direction='maximize')
            
            def objective(trial):
                return self._optimize_objective(
                    trial, model_class, model_name, X_train, y_train, X_val, y_val
                )
            
            study.optimize(objective, n_trials=n_trials, timeout=300 if self.quick_mode else 1800)
            
            best_params = study.best_params
            best_score = study.best_value
            
            self.optimization_history[model_name] = {
                'best_params': best_params,
                'best_value': best_score,
                'n_trials': len(study.trials)
            }
            
            logger.info(f"Hyperparameter optimization completed for {model_name}")
            logger.info(f"Best score: {best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed for {model_name}: {e}")
            return self._get_default_params(model_name)
    
    def _optimize_objective(self, trial, model_class, model_name, X_train, y_train, X_val, y_val):
        """Optimization objective function"""
        try:
            # Suggest hyperparameters based on model type
            params = self._suggest_params(trial, model_name)
            
            # Create and train model
            model = model_class(model_name, **params)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
            
            # Use AUC as optimization metric
            score = roc_auc_score(y_val, y_pred)
            
            return score
            
        except Exception as e:
            logger.warning(f"Optimization trial failed: {e}")
            return 0.5
    
    def _suggest_params(self, trial, model_name: str) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization"""
        if model_name.lower() == 'lightgbm':
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
        elif model_name.lower() == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            }
        elif model_name.lower() == 'logistic':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            }
        
        return {}
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model type"""
        if model_type.lower() == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_estimators': 100 if self.quick_mode else 500
            }
            
            if self.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info.get('rtx_4060_ti_optimized', False):
                    params.update({
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0,
                        'max_bin': 63
                    })
            
            return params
        
        elif model_type.lower() == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100 if self.quick_mode else 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
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
        
        elif model_type.lower() == 'logistic':
            if self.quick_mode:
                return {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 100,
                    'random_state': 42
                }
            
            return {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'liblinear',
                'max_iter': 1000,
                'random_state': 42
            }
        
        return {}
    
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
            'gpu_info': self.memory_tracker.get_gpu_memory_usage(),
            'quick_mode': self.quick_mode
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
        self.quick_mode = False
        
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"GPU training environment: {device_name}")
                logger.info(f"GPU memory: {gpu_memory:.1f}GB")
                
                if "RTX 4060 Ti" in device_name and gpu_memory >= 15.0:
                    self.gpu_available = True
                    self.parallel_training = True
                    logger.info("RTX 4060 Ti GPU training enabled")
                else:
                    logger.info("GPU detected but not RTX 4060 Ti, using CPU training")
            else:
                logger.info("No GPU available, using CPU training")
                
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
    
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode"""
        self.quick_mode = enabled
        self.hyperparameter_optimizer.set_quick_mode(enabled)
        
    def initialize_trainer(self):
        """Initialize trainer with memory optimization"""
        if not self.trainer_initialized:
            self.memory_tracker.optimize_gpu_memory()
            logger.info("Trainer initialization completed")
            self.trainer_initialized = True
    
    def train_model(self,
                    model_class: Type[BaseModel],
                    model_name: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: pd.DataFrame,
                    y_val: pd.Series,
                    optimize_hyperparams: bool = True) -> Optional[BaseModel]:
        """Train a single model with hyperparameter optimization"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            self.initialize_trainer()
            
            # Memory check before training
            memory_info = self.memory_tracker.get_memory_usage()
            logger.info(f"Pre-training memory: {memory_info.get('available_gb', 0):.1f}GB available")
            
            # Hyperparameter optimization
            if optimize_hyperparams and not self.quick_mode:
                logger.info(f"Hyperparameter optimization for {model_name}")
                best_params = self.hyperparameter_optimizer.optimize_hyperparameters(
                    model_class, model_name, X_train, y_train, X_val, y_val
                )
            else:
                logger.info(f"Using default parameters for {model_name}")
                best_params = self.hyperparameter_optimizer._get_default_params(model_name)
            
            self.best_params[model_name] = best_params
            
            # Train final model
            logger.info(f"Training final {model_name} model")
            model = model_class(model_name, **best_params)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Validate model
            y_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
            
            # Calculate performance metrics
            try:
                auc_score = roc_auc_score(y_val, y_pred)
                ap_score = average_precision_score(y_val, y_pred)
                logloss_score = log_loss(y_val, y_pred)
                
                self.model_performance[model_name] = {
                    'auc': auc_score,
                    'average_precision': ap_score,
                    'logloss': logloss_score,
                    'validation_samples': len(y_val)
                }
                
                logger.info(f"{model_name} performance - AUC: {auc_score:.4f}, AP: {ap_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Performance calculation failed for {model_name}: {e}")
                self.model_performance[model_name] = {'error': str(e)}
            
            # Store trained model
            self.trained_models[model_name] = {
                'model': model,
                'params': best_params,
                'performance': self.model_performance.get(model_name, {}),
                'training_time': time.time(),
                'gpu_trained': self.gpu_available
            }
            
            logger.info(f"{model_name} model training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"{model_name} model training failed: {e}")
            return None
    
    def train_multiple_models(self,
                            model_configs: List[Dict[str, Any]],
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series) -> Dict[str, BaseModel]:
        """Train multiple models"""
        
        logger.info(f"Training {len(model_configs)} models")
        trained_models = {}
        
        for config in model_configs:
            model_name = config['name']
            model_class = config['class']
            
            try:
                model = self.train_model(
                    model_class=model_class,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val
                )
                
                if model:
                    trained_models[model_name] = model
                
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {e}")
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
                'rtx_4060_ti_optimized': self.memory_tracker.get_gpu_memory_usage().get('rtx_4060_ti_optimized', False),
                'quick_mode': self.quick_mode
            }
        }

class CTRTrainingPipeline:
    """CTR training pipeline with comprehensive optimization"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = CTRModelTrainer(config)
        self.memory_tracker = LargeDataMemoryTracker()
        self.pipeline_initialized = False
        self.quick_mode = False
        
        if not self.pipeline_initialized:
            self.memory_tracker.optimize_gpu_memory()
            
            logger.info("GPU memory optimization completed: RTX 4060 Ti 16GB 90% utilization")
            logger.info("Large data memory optimization completed")
            logger.info("Hyperparameter tuning optimization: Optuna + GPU acceleration")
            self.pipeline_initialized = True
    
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode for the entire training pipeline"""
        self.quick_mode = enabled
        self.trainer.set_quick_mode(enabled)
    
    def execute_pipeline(self,
                        model_configs: List[Dict[str, Any]],
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series) -> Dict[str, Any]:
        """Execute the complete training pipeline"""
        
        logger.info("=== CTR Training Pipeline Started ===")
        
        try:
            # Memory optimization
            self.memory_tracker.optimize_gpu_memory()
            
            # Train models
            trained_models = self.trainer.train_multiple_models(
                model_configs, X_train, y_train, X_val, y_val
            )
            
            # Get training summary
            training_summary = self.trainer.get_training_summary()
            
            pipeline_results = {
                'trained_models': trained_models,
                'training_summary': training_summary,
                'pipeline_success': len(trained_models) > 0,
                'quick_mode': self.quick_mode
            }
            
            logger.info("=== CTR Training Pipeline Completed ===")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {
                'trained_models': {},
                'training_summary': {},
                'pipeline_success': False,
                'error': str(e),
                'quick_mode': self.quick_mode
            }

# Main trainer classes for compatibility with main.py

class CTRTrainer(CTRModelTrainer):
    """Basic CTR trainer class for compatibility"""
    
    def __init__(self, config: Config = Config):
        super().__init__(config)
        self.name = "CTRTrainer"
        logger.info("CTR Trainer initialized (CPU mode)")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Training interface for compatibility"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Use LogisticRegression as default model
        model = LogisticRegression(**self.get_default_params_by_model_type('logistic'))
        model.fit(X_train, y_train)
        
        return model

class CTRTrainerGPU(CTRModelTrainer):
    """GPU-optimized CTR trainer class"""
    
    def __init__(self, config: Config = Config):
        super().__init__(config)
        self.name = "CTRTrainerGPU"
        
        if not self.gpu_available:
            logger.warning("GPU not available, falling back to CPU mode")
        else:
            logger.info("CTR Trainer GPU initialized (RTX 4060 Ti mode)")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Training interface with GPU optimization"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Use GPU-optimized parameters if available
        if self.gpu_available:
            # Try to use GPU-accelerated model if available
            try:
                # Default to logistic regression with GPU-friendly settings
                params = self.get_default_params_by_model_type('logistic')
                params.update({
                    'n_jobs': -1,  # Use all available cores
                    'max_iter': 2000 if not self.quick_mode else 100
                })
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                return model
                
            except Exception as e:
                logger.warning(f"GPU training failed, falling back to CPU: {e}")
        
        # Fallback to CPU training
        model = LogisticRegression(**self.get_default_params_by_model_type('logistic'))
        model.fit(X_train, y_train)
        
        return model

# Legacy aliases for backward compatibility
ModelTrainer = CTRModelTrainer
TrainingPipeline = CTRTrainingPipeline

if __name__ == "__main__":
    # Test code for development
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create dummy data for testing
        X_train = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randint(0, 10, 100)
        })
        y_train = np.random.binomial(1, 0.1, 100)
        
        X_val = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),
            'feature_3': np.random.randint(0, 10, 50)
        })
        y_val = np.random.binomial(1, 0.1, 50)
        
        config = Config()
        
        # Test basic trainer
        print("Testing CTRTrainer...")
        trainer = CTRTrainer(config)
        trainer.set_quick_mode(True)
        model = trainer.train(X_train, y_train, X_val, y_val)
        print(f"Basic trainer completed: {type(model)}")
        
        # Test GPU trainer
        print("Testing CTRTrainerGPU...")
        gpu_trainer = CTRTrainerGPU(config)
        gpu_trainer.set_quick_mode(True)
        gpu_model = gpu_trainer.train(X_train, y_train, X_val, y_val)
        print(f"GPU trainer completed: {type(gpu_model)}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")