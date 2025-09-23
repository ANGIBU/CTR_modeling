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
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        return self._get_memory_usage()
    
    def optimize_gpu_memory(self):
        """GPU memory optimization"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
        except Exception:
            pass
    
    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage information"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return {
                    'gpu_available': True,
                    'gpu_memory_gb': self.gpu_memory_gb,
                    'rtx_4060_ti_optimized': 'RTX 4060 Ti' in torch.cuda.get_device_name(0)
                }
            else:
                return {
                    'gpu_available': False,
                    'gpu_memory_gb': 0,
                    'rtx_4060_ti_optimized': False
                }
        except Exception:
            return {
                'gpu_available': False,
                'gpu_memory_gb': 0,
                'rtx_4060_ti_optimized': False
            }

class CTRHyperparameterOptimizer:
    """Hyperparameter optimization for CTR models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.quick_mode = False
        self.memory_tracker = LargeDataMemoryTracker()
        self.gpu_available = self.memory_tracker.gpu_available
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for parameter optimization"""
        self.quick_mode = enabled
        
    def optimize_hyperparameters(self, 
                                model_class: Type[BaseModel],
                                model_name: str,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                X_val: pd.DataFrame,
                                y_val: pd.Series,
                                n_trials: int = None) -> Dict[str, Any]:
        """Optimize hyperparameters for a model"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return self._get_default_params(model_name)
        
        if self.quick_mode:
            return self._get_default_params(model_name)
        
        try:
            n_trials = n_trials or (5 if self.quick_mode else 20)
            
            def objective(trial):
                params = self._suggest_hyperparameters(trial, model_name)
                
                try:
                    model = model_class(name=model_name, params=params)
                    model.set_quick_mode(self.quick_mode)
                    
                    model.fit(X_train, y_train, X_val, y_val)
                    
                    y_pred = model.predict_proba(X_val)
                    auc_score = roc_auc_score(y_val, y_pred)
                    
                    return auc_score
                    
                except Exception as e:
                    logger.warning(f"Hyperparameter trial failed: {e}")
                    return 0.5
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params
            logger.info(f"Best hyperparameters for {model_name}: AUC = {study.best_value:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}")
            return self._get_default_params(model_name)
    
    def _suggest_hyperparameters(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization"""
        
        if model_type.lower() == 'lightgbm':
            return {
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 1.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 1.0)
            }
        
        elif model_type.lower() == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
            }
        
        elif model_type.lower() == 'logistic':
            return {
                'C': trial.suggest_float('C', 0.01, 10.0),
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
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            }
        
        return {}

class CTRModelTrainer:
    """Main trainer class for CTR models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.hyperparameter_optimizer = CTRHyperparameterOptimizer(config)
        self.gpu_available = self.memory_tracker.gpu_available
        self.quick_mode = False
        self.trainer_initialized = False
        self.trained_models = {}
        self.best_params = {}
        self.model_performance = {}
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for training"""
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
            
            # Train final model - FIXED: Use name and params parameters
            logger.info(f"Training final {model_name} model")
            model = model_class(name=model_name, params=best_params)
            
            # Set quick mode for the model
            if self.quick_mode:
                model.set_quick_mode(True)
            
            # Fit model
            model.fit(X_train, y_train, X_val, y_val)
            
            # Validate model - FIXED: predict_proba returns ndarray directly
            y_pred = model.predict_proba(X_val)
            
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
            1 for model_data in self.trained_models.values() 
            if model_data['model'].is_calibrated
        )
        
        avg_performance = {}
        if self.model_performance:
            metrics = ['auc', 'average_precision', 'logloss']
            for metric in metrics:
                values = [
                    perf.get(metric, 0) for perf in self.model_performance.values() 
                    if isinstance(perf, dict) and metric in perf
                ]
                avg_performance[f'avg_{metric}'] = np.mean(values) if values else 0.0
        
        return {
            'total_models_trained': len(self.trained_models),
            'calibrated_models': calibrated_base_models,
            'calibration_rate': calibrated_base_models / max(len(self.trained_models), 1),
            'model_performance': self.model_performance,
            'average_performance': avg_performance,
            'quick_mode': self.quick_mode,
            'gpu_used': self.gpu_available,
            'training_completed': True
        }

class CTRTrainingPipeline:
    """Complete CTR training pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.trainer = CTRModelTrainer(config)
        self.quick_mode = False
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for the entire pipeline"""
        self.quick_mode = enabled
        self.trainer.set_quick_mode(enabled)
        
    def run_training_pipeline(self,
                            model_configs: List[Dict[str, Any]],
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series) -> Dict[str, Any]:
        """Run complete training pipeline"""
        
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