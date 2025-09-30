# training.py
"""
CTR Model Training Module
Training pipeline for Click-Through Rate prediction models
"""

import os
import gc
import json
import time
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Tuple
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Safe imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class LargeDataMemoryTracker:
    """Memory tracking and optimization for large data processing"""
    
    def __init__(self):
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.memory_threshold = 0.85
        self.cleanup_threshold = 0.90
        
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        status = {'available': True, 'usage_gb': 0.0, 'available_gb': 50.0}
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            status.update({
                'usage_gb': (vm.total - vm.available) / (1024**3),
                'available_gb': vm.available / (1024**3),
                'percent': vm.percent
            })
        
        return status
    
    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage information"""
        if not self.gpu_available:
            return {'available': False, 'rtx_4060_ti_optimized': False}
        
        try:
            if TORCH_AVAILABLE:
                allocated = torch.cuda.memory_allocated() / (1024**3)
                cached = torch.cuda.memory_reserved() / (1024**3)
                
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                rtx_4060_ti = "4060 Ti" in gpu_name
                
                return {
                    'available': True,
                    'allocated_gb': allocated,
                    'cached_gb': cached,
                    'rtx_4060_ti_optimized': rtx_4060_ti,
                    'gpu_name': gpu_name
                }
        except Exception:
            pass
        
        return {'available': False, 'rtx_4060_ti_optimized': False}
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
    
    def force_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

class CTRHyperparameterOptimizer:
    """Hyperparameter optimization for CTR models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.quick_mode = False
        self.memory_tracker = LargeDataMemoryTracker()
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for optimization"""
        self.quick_mode = enabled
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for each model type"""
        
        if model_type.lower() == 'lightgbm':
            if self.quick_mode:
                return {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1,
                    'num_iterations': 50
                }
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 64,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'min_child_weight': 0.001,
                'min_split_gain': 0.02,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                'num_iterations': 500
            }
            
            if self.memory_tracker.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized']:
                    params.update({
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0
                    })
            
            return params
        
        elif model_type.lower() == 'xgboost':
            if self.quick_mode:
                return {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'n_estimators': 50
                }
            
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'n_estimators': 500
            }
            
            if self.memory_tracker.gpu_available:
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
                    model_class: Type,
                    model_name: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None,
                    y_val: Optional[pd.Series] = None) -> Optional[Any]:
        """Train individual model with optimization"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            start_time = time.time()
            
            self.initialize_trainer()
            
            self.memory_tracker.force_cleanup()
            logger.info("GPU memory cache cleared")
            
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            if X_val is None or y_val is None:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            else:
                X_train_split, y_train_split = X_train, y_train
                X_val_split, y_val_split = X_val, y_val
            
            best_params = self.hyperparameter_optimizer._get_default_params(model_name)
            if not best_params:
                logger.warning(f"Using default parameters for {model_name}")
                best_params = {}
            else:
                logger.info(f"Using default parameters for {model_name}")
            
            logger.info(f"Training final {model_name} model")
            model = model_class()
            
            if hasattr(model, 'set_quick_mode'):
                model.set_quick_mode(self.quick_mode)
            
            if hasattr(model, 'fit_with_params'):
                model.fit_with_params(X_train_split, y_train_split, X_val_split, y_val_split, **best_params)
            else:
                model.fit(X_train_split, y_train_split, X_val_split, y_val_split)
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val_split)
                    
                    if len(y_pred_proba) == 0:
                        logger.warning(f"{model_name}: Empty prediction array, using default performance")
                        self.model_performance[model_name] = {
                            'auc': 0.5,
                            'average_precision': 0.0,
                            'logloss': 1.0
                        }
                    else:
                        if len(y_pred_proba.shape) > 1:
                            y_pred_proba = y_pred_proba[:, 1]
                        
                        if len(y_pred_proba) != len(y_val_split):
                            logger.warning(f"{model_name}: Prediction length mismatch ({len(y_pred_proba)} vs {len(y_val_split)})")
                            self.model_performance[model_name] = {
                                'auc': 0.5,
                                'average_precision': 0.0,
                                'logloss': 1.0
                            }
                        else:
                            auc = roc_auc_score(y_val_split, y_pred_proba) if len(np.unique(y_val_split)) > 1 else 0.5
                            ap = average_precision_score(y_val_split, y_pred_proba)
                            logloss = log_loss(y_val_split, np.clip(y_pred_proba, 1e-15, 1-1e-15))
                            
                            self.model_performance[model_name] = {
                                'auc': auc,
                                'average_precision': ap,
                                'logloss': logloss
                            }
                            
                            logger.info(f"{model_name} performance - AUC: {auc:.4f}, AP: {ap:.4f}")
                else:
                    y_pred = model.predict(X_val_split)
                    
                    if len(y_pred) == 0:
                        logger.warning(f"{model_name}: Empty prediction array")
                        self.model_performance[model_name] = {'error': 'Empty predictions'}
                    else:
                        self.model_performance[model_name] = {'predictions': 'binary_only'}
                
            except Exception as e:
                logger.warning(f"Performance calculation failed for {model_name}: {e}")
                self.model_performance[model_name] = {'error': str(e)}
            
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
                            y_val: pd.Series) -> Dict[str, Any]:
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
            if hasattr(model_data.get('model', {}), 'is_calibrated') and model_data['model'].is_calibrated
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
            self.memory_tracker.optimize_gpu_memory()
            
            trained_models = self.trainer.train_multiple_models(
                model_configs, X_train, y_train, X_val, y_val
            )
            
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

class CTRTrainer(CTRModelTrainer):
    """Basic CTR trainer class for compatibility"""
    
    def __init__(self, config: Config = Config):
        super().__init__(config)
        self.name = "CTRTrainer"
        logger.info("CTR Trainer initialized (CPU mode)")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available_models = ['logistic']
        
        if LIGHTGBM_AVAILABLE:
            available_models.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        
        return available_models
    
    def enable_gpu_optimization(self):
        """Enable GPU optimization if available"""
        if self.gpu_available:
            self.memory_tracker.optimize_gpu_memory()
            logger.info("GPU optimization enabled")
        else:
            logger.warning("GPU not available, using CPU mode")
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, quick_mode: bool = False) -> Tuple[Any, Dict[str, float]]:
        """Train model with simplified interface for main.py compatibility"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            if quick_mode:
                self.set_quick_mode(True)
            
            self.memory_tracker.force_cleanup()
            logger.info("GPU memory cache cleared")
            logger.info("Trainer initialization completed")
            
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            params = self.get_default_params_by_model_type(model_name)
            logger.info(f"Using default parameters for {model_name}")
            
            logger.info(f"Training final {model_name} model")
            
            if model_name == 'logistic':
                from models import LogisticModel
                model = LogisticModel()
                if hasattr(model, 'set_quick_mode'):
                    model.set_quick_mode(quick_mode)
                logger.info(f"{model_name}: Quick mode enabled - simplified parameters")
                
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                from models import LightGBMModel
                model = LightGBMModel()
                if hasattr(model, 'set_quick_mode'):
                    model.set_quick_mode(quick_mode)
                    
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                from models import XGBoostModel
                model = XGBoostModel()
                if hasattr(model, 'set_quick_mode'):
                    model.set_quick_mode(quick_mode)
            else:
                model = LogisticRegression(**params)
            
            logger.info(f"{model_name} model training started (data: {len(X_train)})")
            logger.info(f"{model_name}: Quick mode parameters applied")
            logger.info(f"{model_name}: Starting training")
            
            if hasattr(model, 'fit_with_params'):
                model.fit_with_params(X_train, y_train, X_val, y_val, **params)
            else:
                model.fit(X_train, y_train, X_val, y_val)
            
            logger.info(f"{model_name}: Training completed successfully")
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)
                    
                    if len(y_pred_proba) == 0:
                        logger.warning(f"{model_name}: Empty prediction array")
                        performance = {'auc': 0.5, 'ap': 0.0}
                    else:
                        if len(y_pred_proba.shape) > 1:
                            y_pred_proba = y_pred_proba[:, 1]
                        
                        if len(y_pred_proba) != len(y_val):
                            logger.warning(f"{model_name}: Prediction length mismatch")
                            performance = {'auc': 0.5, 'ap': 0.0}
                        else:
                            auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                            ap = average_precision_score(y_val, y_pred_proba)
                            
                            performance = {'auc': auc, 'ap': ap}
                            logger.info(f"{model_name} performance - AUC: {auc:.4f}, AP: {ap:.4f}")
                else:
                    y_pred = model.predict(X_val)
                    performance = {'auc': 0.5, 'ap': 0.0}
                
            except Exception as e:
                logger.warning(f"Performance calculation failed: {e}")
                performance = {'auc': 0.5, 'ap': 0.0}
            
            logger.info(f"{model_name} model training completed successfully")
            return model, performance
            
        except Exception as e:
            logger.error(f"{model_name} model training failed: {e}")
            return None, {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Training interface for compatibility"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
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
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available_models = ['logistic']
        
        if LIGHTGBM_AVAILABLE:
            available_models.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        
        return available_models
    
    def enable_gpu_optimization(self):
        """Enable GPU optimization"""
        if self.gpu_available:
            self.memory_tracker.optimize_gpu_memory()
            logger.info("RTX 4060 Ti GPU optimization enabled")
        else:
            logger.warning("GPU not available, using CPU mode")
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, quick_mode: bool = False) -> Tuple[Any, Dict[str, float]]:
        """Train model with GPU optimization"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            self.enable_gpu_optimization()
            
            if quick_mode:
                self.set_quick_mode(True)
            
            return super().train_model(model_name, X_train, y_train, X_val, y_val, quick_mode)
            
        except Exception as e:
            logger.error(f"GPU {model_name} training failed, falling back to CPU: {e}")
            
            params = self.get_default_params_by_model_type(model_name)
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            try:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                ap = average_precision_score(y_val, y_pred_proba)
                performance = {'auc': auc, 'ap': ap}
            except Exception:
                performance = {'auc': 0.5, 'ap': 0.0}
            
            return model, performance
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Training interface with GPU optimization"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        if self.gpu_available:
            try:
                params = self.get_default_params_by_model_type('logistic')
                params.update({
                    'n_jobs': -1,
                    'max_iter': 2000 if not self.quick_mode else 100
                })
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                return model
                
            except Exception as e:
                logger.warning(f"GPU training failed, falling back to CPU: {e}")
        
        model = LogisticRegression(**self.get_default_params_by_model_type('logistic'))
        model.fit(X_train, y_train)
        
        return model

ModelTrainer = CTRModelTrainer
TrainingPipeline = CTRTrainingPipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
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
        
        print("Testing CTRTrainer...")
        trainer = CTRTrainer(config)
        trainer.set_quick_mode(True)
        available_models = trainer.get_available_models()
        print(f"Available models: {available_models}")
        
        model = trainer.train(X_train, y_train, X_val, y_val)
        print(f"Basic trainer completed: {type(model)}")
        
        print("Testing CTRTrainerGPU...")
        gpu_trainer = CTRTrainerGPU(config)
        gpu_trainer.set_quick_mode(True)
        gpu_model = gpu_trainer.train(X_train, y_train, X_val, y_val)
        print(f"GPU trainer completed: {type(gpu_model)}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")