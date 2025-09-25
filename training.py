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

class ModelWrapper:
    """Unified model wrapper with consistent interface"""
    
    def __init__(self, model, model_type: str, scaler=None):
        self.model = model
        self.model_type = model_type
        self.scaler = scaler
        self.is_fitted = True
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Unified predict_proba interface"""
        try:
            if self.model_type == 'logistic':
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X)
                    proba = self.model.predict_proba(X_scaled)[:, 1]
                else:
                    proba = self.model.predict_proba(X)[:, 1]
            elif self.model_type == 'lightgbm':
                if hasattr(self.model, 'predict'):
                    proba = self.model.predict(X, num_iteration=self.model.best_iteration)
                else:
                    proba = self.model.predict_proba(X)[:, 1]
            elif self.model_type == 'xgboost':
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X)[:, 1]
                else:
                    proba = self.model.predict(X)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Ensure output is 1D array
            proba = np.asarray(proba).flatten()
            return np.clip(proba, 1e-15, 1-1e-15)
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_type}: {e}")
            return np.full(len(X), 0.0191)

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
                
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                rtx_optimized = "4060 Ti" in gpu_name
                
                return {
                    'available': True,
                    'allocated_gb': allocated,
                    'cached_gb': cached,
                    'gpu_name': gpu_name,
                    'rtx_4060_ti_optimized': rtx_optimized
                }
        except Exception as e:
            logger.warning(f"GPU memory check failed: {e}")
        
        return {'available': False, 'rtx_4060_ti_optimized': False}
    
    def force_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"GPU cleanup failed: {e}")
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)
                self.force_cleanup()
                logger.info("GPU memory optimization applied")
            except Exception as e:
                logger.warning(f"GPU optimization failed: {e}")

class CTRValidationStrategy:
    """CTR-specific validation strategy with temporal awareness"""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_ctr = 0.0191
        
    def create_stratified_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified splits maintaining CTR distribution"""
        try:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            splits = []
            for train_idx, val_idx in skf.split(X, y):
                train_ctr = y.iloc[train_idx].mean()
                val_ctr = y.iloc[val_idx].mean()
                
                ctr_diff = abs(train_ctr - val_ctr)
                if ctr_diff < 0.01:
                    splits.append((train_idx, val_idx))
                    logger.info(f"Split created - Train CTR: {train_ctr:.4f}, Val CTR: {val_ctr:.4f}")
                else:
                    logger.warning(f"Split rejected - CTR difference too large: {ctr_diff:.4f}")
            
            if len(splits) == 0:
                logger.warning("Using fallback random split")
                train_idx, val_idx = train_test_split(
                    range(len(X)), test_size=0.2, random_state=self.random_state, 
                    stratify=y
                )
                splits = [(train_idx, val_idx)]
            
            return splits
            
        except Exception as e:
            logger.error(f"Stratified splits creation failed: {e}")
            train_idx, val_idx = train_test_split(
                range(len(X)), test_size=0.2, random_state=self.random_state
            )
            return [(train_idx, val_idx)]

class CTRHyperparameterOptimizer:
    """CTR-focused hyperparameter optimization"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.optimization_history = {}
        
    def get_optimized_params(self, model_name: str, quick_mode: bool = False) -> Dict[str, Any]:
        """Get CTR-optimized parameters for each model"""
        try:
            if model_name == 'logistic':
                if quick_mode:
                    return {
                        'C': 0.1,
                        'penalty': 'l2',
                        'solver': 'liblinear',
                        'max_iter': 500,
                        'random_state': 42
                    }
                else:
                    return {
                        'C': 0.01,
                        'penalty': 'l2',
                        'solver': 'liblinear',
                        'max_iter': 2000,
                        'class_weight': 'balanced',
                        'random_state': 42
                    }
            
            elif model_name == 'lightgbm':
                if quick_mode:
                    return {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'num_leaves': 15,
                        'learning_rate': 0.1,
                        'n_estimators': 50,
                        'verbose': -1,
                        'random_state': 42
                    }
                else:
                    return {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': 63,
                        'learning_rate': 0.02,
                        'n_estimators': 500,
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'min_data_in_leaf': 100,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1,
                        'class_weight': 'balanced',
                        'verbose': -1,
                        'random_state': 42,
                        'early_stopping_rounds': 100
                    }
            
            elif model_name == 'xgboost':
                if quick_mode:
                    return {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'max_depth': 4,
                        'learning_rate': 0.1,
                        'n_estimators': 50,
                        'verbosity': 0,
                        'random_state': 42
                    }
                else:
                    return {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'max_depth': 8,
                        'learning_rate': 0.01,
                        'n_estimators': 800,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'min_child_weight': 10,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1,
                        'gamma': 0.1,
                        'verbosity': 0,
                        'random_state': 42,
                        'early_stopping_rounds': 100
                    }
            
            else:
                logger.warning(f"Unknown model name: {model_name}")
                return {}
                
        except Exception as e:
            logger.error(f"Parameter optimization failed for {model_name}: {e}")
            return {}

class CTRModelTrainer:
    """Main CTR model trainer with validation and calibration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.validation_strategy = CTRValidationStrategy()
        self.param_optimizer = CTRHyperparameterOptimizer()
        
        self.trained_models = {}
        self.model_performance = {}
        self.calibration_history = {}
        self.quick_mode = False
        
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        logger.info(f"Trainer initialized - GPU available: {self.gpu_available}")
        
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode"""
        self.quick_mode = enabled
        logger.info(f"Quick mode {'enabled' if enabled else 'disabled'}")
    
    def get_default_params_by_model_type(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters optimized for CTR prediction"""
        return self.param_optimizer.get_optimized_params(model_name, self.quick_mode)
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train logistic regression model with CTR optimization"""
        try:
            logger.info("Training logistic regression model")
            
            params = self.get_default_params_by_model_type('logistic')
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Evaluate performance
            performance = self._evaluate_model_performance(y_val, val_pred_proba, 'logistic')
            
            # Create wrapped model
            wrapped_model = ModelWrapper(model, 'logistic', scaler)
            
            return wrapped_model, performance
            
        except Exception as e:
            logger.error(f"Logistic regression training failed: {e}")
            return None, {}
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train LightGBM model with CTR optimization"""
        try:
            if not LIGHTGBM_AVAILABLE:
                logger.error("LightGBM not available")
                return None, {}
            
            logger.info("Training LightGBM model")
            
            params = self.get_default_params_by_model_type('lightgbm')
            
            # Prepare datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Predictions
            val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            
            # Evaluate performance
            performance = self._evaluate_model_performance(y_val, val_pred_proba, 'lightgbm')
            
            # Create wrapped model
            wrapped_model = ModelWrapper(model, 'lightgbm')
            
            return wrapped_model, performance
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return None, {}
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train XGBoost model with CTR optimization"""
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost not available")
                return None, {}
            
            logger.info("Training XGBoost model")
            
            params = self.get_default_params_by_model_type('xgboost')
            
            # Calculate scale_pos_weight for class imbalance
            pos_count = (y_train == 1).sum()
            neg_count = (y_train == 0).sum()
            if pos_count > 0:
                params['scale_pos_weight'] = neg_count / pos_count
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            # Predictions
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Evaluate performance
            performance = self._evaluate_model_performance(y_val, val_pred_proba, 'xgboost')
            
            # Create wrapped model
            wrapped_model = ModelWrapper(model, 'xgboost')
            
            return wrapped_model, performance
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None, {}
    
    def _evaluate_model_performance(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                                  model_name: str) -> Dict[str, float]:
        """Comprehensive model performance evaluation"""
        try:
            performance = {}
            
            # Basic classification metrics
            try:
                performance['auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                performance['auc'] = 0.5
            
            try:
                performance['avg_precision'] = average_precision_score(y_true, y_pred_proba)
            except:
                performance['avg_precision'] = y_true.mean()
            
            try:
                performance['log_loss'] = log_loss(y_true, y_pred_proba)
            except:
                performance['log_loss'] = 1.0
            
            # CTR-specific metrics
            actual_ctr = y_true.mean()
            predicted_ctr = y_pred_proba.mean()
            target_ctr = 0.0191
            
            performance.update({
                'actual_ctr': actual_ctr,
                'predicted_ctr': predicted_ctr,
                'ctr_bias': predicted_ctr - actual_ctr,
                'ctr_absolute_error': abs(predicted_ctr - actual_ctr),
                'target_alignment': abs(predicted_ctr - target_ctr),
                'ctr_relative_error': abs(predicted_ctr - actual_ctr) / max(actual_ctr, 0.001)
            })
            
            # Combined score for CTR prediction
            auc_score = performance['auc']
            ap_score = performance['avg_precision']
            ctr_alignment_score = max(0, 1 - performance['target_alignment'] / 0.02)
            log_loss_score = max(0, 1 - performance['log_loss'] / 2.0)
            
            performance['combined_score'] = (
                0.3 * auc_score + 
                0.2 * ap_score + 
                0.4 * ctr_alignment_score + 
                0.1 * log_loss_score
            )
            
            # Binary classification metrics
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            performance['accuracy'] = (y_pred_binary == y_true).mean()
            
            tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
            fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
            fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
            
            performance['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            performance['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            logger.info(f"{model_name} Performance: AUC={performance['auc']:.3f}, "
                       f"AP={performance['avg_precision']:.3f}, "
                       f"CTR_bias={performance['ctr_bias']:.4f}, "
                       f"Combined={performance['combined_score']:.3f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {'combined_score': 0.0, 'auc': 0.5}
    
    def train_multiple_models(self, model_configs: List[Dict[str, Any]], 
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train multiple models with proper validation"""
        try:
            logger.info(f"Training {len(model_configs)} models")
            
            if X_val is None or y_val is None:
                logger.info("Creating validation split")
                splits = self.validation_strategy.create_stratified_splits(X_train, y_train)
                train_idx, val_idx = splits[0]
                
                X_train_split = X_train.iloc[train_idx]
                y_train_split = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]
            else:
                X_train_split = X_train
                y_train_split = y_train
            
            trained_models = {}
            
            for config in model_configs:
                model_name = config.get('name', 'unknown')
                
                try:
                    logger.info(f"Starting {model_name} training")
                    
                    self.memory_tracker.force_cleanup()
                    
                    if model_name == 'logistic':
                        model_result, performance = self.train_logistic_regression(
                            X_train_split, y_train_split, X_val, y_val
                        )
                    elif model_name == 'lightgbm':
                        model_result, performance = self.train_lightgbm(
                            X_train_split, y_train_split, X_val, y_val
                        )
                    elif model_name == 'xgboost':
                        model_result, performance = self.train_xgboost(
                            X_train_split, y_train_split, X_val, y_val
                        )
                    else:
                        logger.warning(f"Unknown model type: {model_name}")
                        continue
                    
                    if model_result is not None:
                        trained_models[model_name] = model_result
                        self.model_performance[model_name] = performance
                        logger.info(f"{model_name} training completed successfully")
                    else:
                        logger.error(f"{model_name} training failed")
                    
                except Exception as e:
                    logger.error(f"{model_name} training failed: {e}")
                    continue
            
            logger.info(f"Model training completed. Successful models: {list(trained_models.keys())}")
            self.trained_models = trained_models
            
            return trained_models
            
        except Exception as e:
            logger.error(f"Multiple model training failed: {e}")
            return {}

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
                model_result, performance = self.train_logistic_regression(
                    X_train, y_train, X_val, y_val
                )
                return model_result, performance
                
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                model_result, performance = self.train_lightgbm(
                    X_train, y_train, X_val, y_val
                )
                return model_result, performance
                
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                model_result, performance = self.train_xgboost(
                    X_train, y_train, X_val, y_val
                )
                return model_result, performance
            
            else:
                logger.error(f"Model {model_name} not available or not supported")
                return None, {}
                
        except Exception as e:
            logger.error(f"{model_name} model training failed: {e}")
            return None, {}

# Export for compatibility
__all__ = [
    'CTRTrainer', 
    'CTRModelTrainer', 
    'ModelWrapper',
    'LargeDataMemoryTracker',
    'CTRValidationStrategy',
    'CTRHyperparameterOptimizer'
]