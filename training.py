# training.py

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_GPU_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class LargeDataMemoryTracker:
    """Memory tracking and optimization"""
    
    def __init__(self):
        self.gpu_available = TORCH_GPU_AVAILABLE
        self.memory_threshold = 0.80
        self.cleanup_threshold = 0.85
        
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
        
        if self.gpu_available and TORCH_GPU_AVAILABLE:
            try:
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                status['gpu_usage_gb'] = gpu_mem_allocated
                status['gpu_reserved_gb'] = gpu_mem_reserved
                status['gpu_total_gb'] = gpu_total
                status['gpu_available'] = True
            except:
                status['gpu_available'] = False
        
        return status
    
    def force_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        
        if self.gpu_available and TORCH_GPU_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass

class CTRHyperparameterOptimizer:
    """Hyperparameter optimization for CTR models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.quick_mode = False
        self.memory_tracker = LargeDataMemoryTracker()
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode"""
        self.quick_mode = enabled
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters"""
        
        if model_type.lower() == 'xgboost_gpu':
            params = {
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist' if TORCH_GPU_AVAILABLE else 'hist',
                'max_depth': 9,
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'scale_pos_weight': 51.43,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.05,
                'reg_lambda': 1.5,
                'max_bin': 512,
                'gpu_id': 0 if TORCH_GPU_AVAILABLE else None,
                'predictor': 'gpu_predictor' if TORCH_GPU_AVAILABLE else 'cpu_predictor',
                'verbosity': 0,
                'seed': 42,
                'n_jobs': -1
            }
            
            if self.quick_mode:
                params['learning_rate'] = 0.3
                params['max_depth'] = 4
                params['max_bin'] = 256
            
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
                'C': 0.5,
                'solver': 'saga',
                'max_iter': 100,
                'random_state': 42,
                'n_jobs': 4,
                'tol': 0.001,
                'warm_start': True
            }
        
        return {}

class CTRModelTrainer:
    """Main trainer class for CTR models with memory efficient learning"""
    
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
        """Set quick mode"""
        self.quick_mode = enabled
        self.hyperparameter_optimizer.set_quick_mode(enabled)
        
    def initialize_trainer(self):
        """Initialize trainer"""
        if not self.trainer_initialized:
            logger.info("Trainer initialization completed")
            self.trainer_initialized = True
    
    def _sample_for_memory(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          max_samples: int = 5000000) -> Tuple[pd.DataFrame, pd.Series]:
        """Sample data to fit in memory"""
        memory_status = self.memory_tracker.get_memory_status()
        available_gb = memory_status['available_gb']
        
        if len(X_train) > max_samples and available_gb < 8:
            logger.info(f"Sampling data due to memory constraint: {len(X_train)} -> {max_samples}")
            
            try:
                X_sampled, _, y_sampled, _ = train_test_split(
                    X_train, y_train, 
                    train_size=max_samples, 
                    random_state=42,
                    stratify=y_train
                )
                return X_sampled, y_sampled
            except:
                indices = np.random.choice(len(X_train), size=max_samples, replace=False)
                return X_train.iloc[indices], y_train.iloc[indices]
        
        return X_train, y_train
    
    def train_model(self,
                    model_class: Type,
                    model_name: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None,
                    y_val: Optional[pd.Series] = None) -> Optional[Any]:
        """Train individual model with memory management"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            start_time = time.time()
            
            self.initialize_trainer()
            self.memory_tracker.force_cleanup()
            logger.info("Memory cleanup completed")
            
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            if self.gpu_available:
                logger.info(f"GPU available: {memory_status.get('gpu_total_gb', 0):.1f}GB total")
                logger.info(f"GPU usage: {memory_status.get('gpu_usage_gb', 0):.1f}GB used")
            
            X_train_sampled, y_train_sampled = self._sample_for_memory(X_train, y_train, max_samples=5000000)
            
            if X_val is None or y_val is None:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train_sampled, y_train_sampled, test_size=0.2, random_state=42, stratify=y_train_sampled
                )
            else:
                X_train_split, y_train_split = X_train_sampled, y_train_sampled
                X_val_split, y_val_split = X_val, y_val
            
            best_params = self.hyperparameter_optimizer._get_default_params(model_name)
            if not best_params:
                logger.warning(f"Using default parameters for {model_name}")
                best_params = {}
            else:
                logger.info(f"Using optimized parameters for {model_name}")
                if 'tree_method' in best_params:
                    logger.info(f"XGBoost tree_method: {best_params['tree_method']}")
                    if best_params.get('tree_method') == 'gpu_hist':
                        logger.info(f"GPU acceleration enabled for {model_name}")
            
            logger.info(f"Training {model_name} model with {len(X_train_split)} samples")
            model = model_class()
            
            if hasattr(model, 'set_quick_mode'):
                model.set_quick_mode(self.quick_mode)
            
            if hasattr(model, 'fit_with_params'):
                model.fit_with_params(X_train_split, y_train_split, **best_params)
            else:
                model.fit(X_train_split, y_train_split, X_val_split, y_val_split)
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val_split)
                    if len(y_pred_proba.shape) > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = model.predict(X_val_split)
                
                auc = roc_auc_score(y_val_split, y_pred_proba) if len(np.unique(y_val_split)) > 1 else 0.5
                ap = average_precision_score(y_val_split, y_pred_proba)
                logloss = log_loss(y_val_split, np.clip(y_pred_proba, 1e-15, 1-1e-15))
                
                self.model_performance[model_name] = {
                    'auc': auc,
                    'average_precision': ap,
                    'logloss': logloss
                }
                
                logger.info(f"{model_name} performance - AUC: {auc:.4f}, AP: {ap:.4f}")
                
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
        """Get training summary"""
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

class CTRTrainer(CTRModelTrainer):
    """CTR trainer class"""
    
    def __init__(self, config: Config = Config):
        super().__init__(config)
        self.name = "CTRTrainer"
        logger.info("CTR Trainer initialized")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available_models = ['logistic']
        
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost_gpu')
            if TORCH_GPU_AVAILABLE:
                logger.info("XGBoost GPU mode available")
            else:
                logger.info("XGBoost CPU mode (GPU not available)")
        
        return available_models
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, quick_mode: bool = False) -> Tuple[Any, Dict[str, float]]:
        """Train model with simplified interface"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            if quick_mode:
                self.set_quick_mode(True)
            
            self.memory_tracker.force_cleanup()
            logger.info("Memory cleanup completed")
            
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            if memory_status['available_gb'] < 8:
                logger.warning(f"Low memory detected, sampling data")
                X_train, y_train = self._sample_for_memory(X_train, y_train, max_samples=2000000)
            
            params = self.get_default_params_by_model_type(model_name)
            logger.info(f"Using optimized parameters for {model_name}")
            
            if model_name == 'xgboost_gpu' and 'tree_method' in params:
                logger.info(f"XGBoost tree_method: {params['tree_method']}")
                if params.get('tree_method') == 'gpu_hist':
                    logger.info("GPU hist method enabled")
                    if params.get('predictor'):
                        logger.info(f"GPU predictor: {params.get('predictor')}")
                else:
                    logger.info("CPU hist method (GPU not available)")
            
            logger.info(f"Training {model_name} model with {len(X_train)} samples")
            
            if model_name == 'xgboost_gpu':
                from models import XGBoostGPUModel
                model = XGBoostGPUModel()
                if hasattr(model, 'set_quick_mode'):
                    model.set_quick_mode(quick_mode)
                    
            elif model_name == 'logistic':
                from models import LogisticModel
                model = LogisticModel()
                if hasattr(model, 'set_quick_mode'):
                    model.set_quick_mode(quick_mode)
                logger.info(f"{model_name}: Using memory efficient mode")
            else:
                model = LogisticRegression(**params)
            
            logger.info(f"{model_name} model training started")
            
            if hasattr(model, 'fit_with_params'):
                model.fit_with_params(X_train, y_train, **params)
            else:
                model.fit(X_train, y_train, X_val, y_val)
            
            logger.info(f"{model_name}: Training completed successfully")
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)
                    if len(y_pred_proba.shape) > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = model.predict(X_val)
                
                auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                ap = average_precision_score(y_val, y_pred_proba)
                
                performance = {'auc': auc, 'ap': ap}
                logger.info(f"{model_name} performance - AUC: {auc:.4f}, AP: {ap:.4f}")
                
            except Exception as e:
                logger.warning(f"Performance calculation failed: {e}")
                performance = {'auc': 0.5, 'ap': 0.0}
            
            logger.info(f"{model_name} model training completed successfully")
            return model, performance
            
        except Exception as e:
            logger.error(f"{model_name} model training failed: {e}")
            return None, {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Training interface"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        model = LogisticRegression(**self.get_default_params_by_model_type('logistic'))
        model.fit(X_train, y_train)
        
        return model

ModelTrainer = CTRModelTrainer

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
        print(f"Trainer completed: {type(model)}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")