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
                
                # RTX 4060 Ti optimization check
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
                # Set memory fraction for RTX 4060 Ti
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
            # Use stratified k-fold to maintain class balance
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            splits = []
            for train_idx, val_idx in skf.split(X, y):
                # Validate split quality
                train_ctr = y.iloc[train_idx].mean()
                val_ctr = y.iloc[val_idx].mean()
                
                # Check if CTR distribution is reasonable
                ctr_diff = abs(train_ctr - val_ctr)
                if ctr_diff < 0.01:  # Accept if difference < 1%
                    splits.append((train_idx, val_idx))
                    logger.info(f"Split created - Train CTR: {train_ctr:.4f}, Val CTR: {val_ctr:.4f}")
                else:
                    logger.warning(f"Split rejected - CTR difference too large: {ctr_diff:.4f}")
            
            if len(splits) == 0:
                # Fallback to simple random split
                logger.warning("Using fallback random split")
                train_idx, val_idx = train_test_split(
                    range(len(X)), test_size=0.2, random_state=self.random_state, 
                    stratify=y
                )
                splits = [(train_idx, val_idx)]
            
            return splits
            
        except Exception as e:
            logger.error(f"Stratified splits creation failed: {e}")
            # Simple fallback
            train_idx, val_idx = train_test_split(
                range(len(X)), test_size=0.2, random_state=self.random_state
            )
            return [(train_idx, val_idx)]
    
    def create_temporal_splits(self, X: pd.DataFrame, y: pd.Series, 
                             time_col: Optional[str] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create temporal splits if time column exists"""
        try:
            if time_col and time_col in X.columns:
                # Sort by time and create splits
                sorted_indices = X.sort_values(time_col).index
                
                splits = []
                split_size = len(X) // self.n_splits
                
                for i in range(self.n_splits):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < self.n_splits - 1 else len(X)
                    
                    train_indices = sorted_indices[:start_idx + int(split_size * 0.8)]
                    val_indices = sorted_indices[start_idx + int(split_size * 0.8):end_idx]
                    
                    if len(train_indices) > 0 and len(val_indices) > 0:
                        splits.append((train_indices.values, val_indices.values))
                
                return splits if splits else self.create_stratified_splits(X, y)
            else:
                return self.create_stratified_splits(X, y)
                
        except Exception as e:
            logger.warning(f"Temporal splits creation failed: {e}")
            return self.create_stratified_splits(X, y)
    
    def validate_split_quality(self, y_train: pd.Series, y_val: pd.Series) -> Dict[str, float]:
        """Validate split quality for CTR prediction"""
        try:
            train_ctr = y_train.mean()
            val_ctr = y_val.mean()
            
            quality_metrics = {
                'train_ctr': train_ctr,
                'val_ctr': val_ctr,
                'ctr_difference': abs(train_ctr - val_ctr),
                'train_target_alignment': abs(train_ctr - self.target_ctr),
                'val_target_alignment': abs(val_ctr - self.target_ctr),
                'class_balance_train': min(train_ctr, 1 - train_ctr) * 2,
                'class_balance_val': min(val_ctr, 1 - val_ctr) * 2
            }
            
            # Overall quality score
            ctr_alignment_score = 1 - min(quality_metrics['ctr_difference'] / 0.01, 1.0)
            balance_score = (quality_metrics['class_balance_train'] + quality_metrics['class_balance_val']) / 2
            
            quality_metrics['overall_quality'] = (ctr_alignment_score + balance_score) / 2
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Split quality validation failed: {e}")
            return {'overall_quality': 0.5}

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
                    # CTR-optimized logistic regression parameters
                    return {
                        'C': 0.01,  # Strong regularization for better generalization
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
                    # CTR-optimized LightGBM parameters
                    return {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': 63,  # Increased for more complexity
                        'learning_rate': 0.02,  # Lower for better convergence
                        'n_estimators': 500,
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'min_data_in_leaf': 100,  # Higher to prevent overfitting
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
                    # CTR-optimized XGBoost parameters
                    return {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'max_depth': 8,  # Deeper trees for complex patterns
                        'learning_rate': 0.01,  # Lower for better convergence
                        'n_estimators': 800,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'min_child_weight': 10,  # Higher to prevent overfitting
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1,
                        'gamma': 0.1,  # Minimum loss reduction
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
    
    def optimize_for_ctr_bias(self, params: Dict[str, Any], model_name: str, 
                             ctr_bias: float) -> Dict[str, Any]:
        """Adjust parameters based on CTR bias"""
        try:
            optimized_params = params.copy()
            
            # If over-predicting (positive bias), increase regularization
            if ctr_bias > 0.01:
                if model_name == 'logistic':
                    optimized_params['C'] = min(optimized_params.get('C', 1.0) * 0.5, 0.001)
                elif model_name == 'lightgbm':
                    optimized_params['reg_alpha'] = optimized_params.get('reg_alpha', 0.0) + 0.1
                    optimized_params['reg_lambda'] = optimized_params.get('reg_lambda', 0.0) + 0.1
                    optimized_params['learning_rate'] = optimized_params.get('learning_rate', 0.1) * 0.8
                elif model_name == 'xgboost':
                    optimized_params['reg_alpha'] = optimized_params.get('reg_alpha', 0.0) + 0.1
                    optimized_params['reg_lambda'] = optimized_params.get('reg_lambda', 0.0) + 0.1
                    optimized_params['learning_rate'] = optimized_params.get('learning_rate', 0.1) * 0.8
            
            # If under-predicting (negative bias), reduce regularization
            elif ctr_bias < -0.01:
                if model_name == 'logistic':
                    optimized_params['C'] = optimized_params.get('C', 1.0) * 1.5
                elif model_name == 'lightgbm':
                    optimized_params['reg_alpha'] = max(optimized_params.get('reg_alpha', 0.1) - 0.05, 0.0)
                    optimized_params['reg_lambda'] = max(optimized_params.get('reg_lambda', 0.1) - 0.05, 0.0)
                elif model_name == 'xgboost':
                    optimized_params['reg_alpha'] = max(optimized_params.get('reg_alpha', 0.1) - 0.05, 0.0)
                    optimized_params['reg_lambda'] = max(optimized_params.get('reg_lambda', 0.1) - 0.05, 0.0)
            
            logger.info(f"Parameter optimization for CTR bias {ctr_bias:.4f}: {model_name}")
            
            return optimized_params
            
        except Exception as e:
            logger.warning(f"CTR bias optimization failed: {e}")
            return params

class CTRModelTrainer:
    """Main CTR model trainer with validation and calibration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.validation_strategy = CTRValidationStrategy()
        self.param_optimizer = CTRHyperparameterOptimizer()
        
        # Training state
        self.trained_models = {}
        self.model_performance = {}
        self.calibration_history = {}
        self.quick_mode = False
        
        # Hardware detection
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
            
            # Get optimized parameters
            params = self.get_default_params_by_model_type('logistic')
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train_scaled, y_train)
            
            # Predictions and calibration
            val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # CTR-focused calibration
            calibrated_model = self._calibrate_model_for_ctr(
                model, X_val_scaled, y_val, val_pred_proba
            )
            
            # Evaluate performance
            performance = self._evaluate_model_performance(y_val, val_pred_proba, 'logistic')
            
            # Store scaler with model
            model_package = {
                'model': calibrated_model if calibrated_model else model,
                'scaler': scaler,
                'performance': performance
            }
            
            return model_package, performance
            
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
            
            # Get optimized parameters
            params = self.get_default_params_by_model_type('lightgbm')
            
            # Prepare datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                callbacks=[lgb.log_evaluation(0)]  # Silent training
            )
            
            # Predictions and calibration
            val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            
            # CTR-focused calibration
            calibrated_model = self._calibrate_model_for_ctr(
                model, X_val, y_val, val_pred_proba, model_type='lightgbm'
            )
            
            # Evaluate performance
            performance = self._evaluate_model_performance(y_val, val_pred_proba, 'lightgbm')
            
            model_package = {
                'model': calibrated_model if calibrated_model else model,
                'performance': performance
            }
            
            return model_package, performance
            
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
            
            # Get optimized parameters and adjust for class imbalance
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
            
            # Predictions and calibration
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # CTR-focused calibration
            calibrated_model = self._calibrate_model_for_ctr(
                model, X_val, y_val, val_pred_proba, model_type='xgboost'
            )
            
            # Evaluate performance
            performance = self._evaluate_model_performance(y_val, val_pred_proba, 'xgboost')
            
            model_package = {
                'model': calibrated_model if calibrated_model else model,
                'performance': performance
            }
            
            return model_package, performance
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None, {}
    
    def _calibrate_model_for_ctr(self, model: Any, X_val: np.ndarray, y_val: pd.Series, 
                               val_pred_proba: np.ndarray, model_type: str = 'sklearn') -> Any:
        """CTR-focused model calibration"""
        try:
            # Check if sufficient validation data
            if len(y_val) < 50:
                logger.warning("Insufficient validation data for calibration")
                return None
            
            # Calculate CTR bias
            actual_ctr = y_val.mean()
            predicted_ctr = val_pred_proba.mean()
            ctr_bias = predicted_ctr - actual_ctr
            
            logger.info(f"Pre-calibration CTR - Actual: {actual_ctr:.4f}, Predicted: {predicted_ctr:.4f}, Bias: {ctr_bias:.4f}")
            
            # Apply calibration if bias is significant
            if abs(ctr_bias) > 0.005:  # Calibrate if bias > 0.5%
                try:
                    if model_type == 'sklearn' or model_type == 'xgboost':
                        # Use CalibratedClassifierCV for sklearn-compatible models
                        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
                        
                        # Fit on validation data (in practice, use separate calibration set)
                        calibrated.fit(X_val, y_val)
                        
                        # Test calibration quality
                        calibrated_proba = calibrated.predict_proba(X_val)[:, 1]
                        calibrated_ctr = calibrated_proba.mean()
                        calibrated_bias = calibrated_ctr - actual_ctr
                        
                        logger.info(f"Post-calibration CTR - Predicted: {calibrated_ctr:.4f}, Bias: {calibrated_bias:.4f}")
                        
                        if abs(calibrated_bias) < abs(ctr_bias):
                            logger.info("Calibration improved CTR prediction")
                            return calibrated
                        else:
                            logger.warning("Calibration did not improve CTR prediction")
                            return None
                    
                    else:  # LightGBM or other models
                        # Simple linear calibration
                        correction_factor = actual_ctr / predicted_ctr if predicted_ctr > 0 else 1.0
                        
                        # Create wrapper for calibrated predictions
                        class CalibratedLGBModel:
                            def __init__(self, base_model, correction_factor):
                                self.base_model = base_model
                                self.correction_factor = correction_factor
                                
                            def predict(self, X, **kwargs):
                                raw_pred = self.base_model.predict(X, **kwargs)
                                return np.clip(raw_pred * self.correction_factor, 0, 1)
                        
                        calibrated_model = CalibratedLGBModel(model, correction_factor)
                        logger.info(f"Applied linear calibration with factor: {correction_factor:.4f}")
                        return calibrated_model
                        
                except Exception as e:
                    logger.warning(f"Calibration failed: {e}")
                    return None
            else:
                logger.info("CTR bias within acceptable range, no calibration needed")
                return None
                
        except Exception as e:
            logger.error(f"Model calibration failed: {e}")
            return None
    
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
            
            # Weighted combination emphasizing CTR alignment
            performance['combined_score'] = (
                0.3 * auc_score + 
                0.2 * ap_score + 
                0.4 * ctr_alignment_score + 
                0.1 * log_loss_score
            )
            
            # Binary classification metrics
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            performance['accuracy'] = (y_pred_binary == y_true).mean()
            
            # Precision and Recall
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
            
            # Create validation split if not provided
            if X_val is None or y_val is None:
                logger.info("Creating validation split")
                splits = self.validation_strategy.create_stratified_splits(X_train, y_train)
                train_idx, val_idx = splits[0]  # Use first split
                
                X_train_split = X_train.iloc[train_idx]
                y_train_split = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]
            else:
                X_train_split = X_train
                y_train_split = y_train
            
            # Validate split quality
            split_quality = self.validation_strategy.validate_split_quality(y_train_split, y_val)
            logger.info(f"Split quality score: {split_quality.get('overall_quality', 0):.3f}")
            
            trained_models = {}
            
            # Train each model
            for config in model_configs:
                model_name = config.get('name', 'unknown')
                
                try:
                    logger.info(f"Starting {model_name} training")
                    
                    # Memory cleanup before each model
                    self.memory_tracker.force_cleanup()
                    
                    # Train based on model type
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
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model"""
        try:
            if not self.model_performance:
                return None, None
            
            # Find best model based on combined score
            best_model_name = max(
                self.model_performance.items(),
                key=lambda x: x[1].get('combined_score', 0)
            )[0]
            
            best_model = self.trained_models.get(best_model_name)
            
            logger.info(f"Best model: {best_model_name} "
                       f"(Combined Score: {self.model_performance[best_model_name].get('combined_score', 0):.3f})")
            
            return best_model_name, best_model
            
        except Exception as e:
            logger.error(f"Best model selection failed: {e}")
            return None, None
    
    def save_models(self, output_dir: str = "models") -> bool:
        """Save trained models and performance metrics"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            
            # Save each model
            for model_name, model_data in self.trained_models.items():
                model_file = output_path / f"{model_name}_model.pkl"
                
                try:
                    with open(model_file, 'wb') as f:
                        pickle.dump(model_data, f)
                    saved_count += 1
                    logger.info(f"Saved {model_name} model to {model_file}")
                except Exception as e:
                    logger.error(f"Failed to save {model_name}: {e}")
            
            # Save performance metrics
            performance_file = output_path / "model_performance.json"
            try:
                with open(performance_file, 'w') as f:
                    json.dump(self.model_performance, f, indent=2)
                logger.info(f"Saved performance metrics to {performance_file}")
            except Exception as e:
                logger.error(f"Failed to save performance metrics: {e}")
            
            logger.info(f"Successfully saved {saved_count}/{len(self.trained_models)} models")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        try:
            calibrated_base_models = 0
            for model_name, model_data in self.trained_models.items():
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    if hasattr(model, 'calibrated_classifiers_') or 'Calibrated' in str(type(model)):
                        calibrated_base_models += 1
            
            # Calculate average performance metrics
            avg_performance = {}
            if self.model_performance:
                for metric in ['auc', 'combined_score', 'ctr_absolute_error', 'target_alignment']:
                    values = [
                        perf.get(metric, 0) for perf in self.model_performance.values() 
                        if isinstance(perf, dict) and metric in perf
                    ]
                    avg_performance[f'avg_{metric}'] = np.mean(values) if values else 0.0
            
            return {
                'total_models_trained': len(self.trained_models),
                'calibration_ready_models': calibrated_base_models,
                'calibration_rate': calibrated_base_models / max(len(self.trained_models), 1),
                'model_performance': self.model_performance,
                'average_performance': avg_performance,
                'quick_mode': self.quick_mode,
                'gpu_used': self.gpu_available,
                'training_completed': True
            }
            
        except Exception as e:
            logger.error(f"Training summary generation failed: {e}")
            return {'training_completed': False, 'error': str(e)}

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
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run complete training pipeline with validation"""
        
        logger.info("=== CTR Training Pipeline Started ===")
        
        try:
            # Memory optimization
            self.memory_tracker.optimize_gpu_memory()
            
            # Train models with proper validation
            trained_models = self.trainer.train_multiple_models(
                model_configs, X_train, y_train, X_val, y_val
            )
            
            # Get training summary
            training_summary = self.trainer.get_training_summary()
            
            pipeline_results = {
                'trained_models': trained_models,
                'training_summary': training_summary,
                'pipeline_success': len(trained_models) > 0,
                'quick_mode': self.quick_mode,
                'validation_used': X_val is not None and y_val is not None
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
                'quick_mode': self.quick_mode,
                'validation_used': False
            }

# Main trainer classes for compatibility with main.py

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
            # Set quick mode
            if quick_mode:
                self.set_quick_mode(True)
            
            # Memory cleanup
            self.memory_tracker.force_cleanup()
            logger.info("GPU memory cache cleared")
            logger.info("Trainer initialization completed")
            
            # Memory status
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            # Get model parameters
            params = self.get_default_params_by_model_type(model_name)
            logger.info(f"Using default parameters for {model_name}")
            
            # Train model based on type with proper validation data
            logger.info(f"Training final {model_name} model")
            
            if model_name == 'logistic':
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                model = LogisticRegression(**params)
                model.fit(X_train_scaled, y_train)
                
                # Get predictions
                val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                
                # Evaluate performance
                performance = self._evaluate_model_performance(y_val, val_pred_proba, model_name)
                
                # Package model with scaler
                model_package = {'model': model, 'scaler': scaler}
                
                return model_package, performance
                
            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                # Train LightGBM
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    callbacks=[lgb.log_evaluation(0)]
                )
                
                val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
                performance = self._evaluate_model_performance(y_val, val_pred_proba, model_name)
                
                return model, performance
                
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                # Calculate class weight for imbalanced data
                pos_count = (y_train == 1).sum()
                neg_count = (y_train == 0).sum()
                if pos_count > 0:
                    params['scale_pos_weight'] = neg_count / pos_count
                
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
                
                val_pred_proba = model.predict_proba(X_val)[:, 1]
                performance = self._evaluate_model_performance(y_val, val_pred_proba, model_name)
                
                return model, performance
            
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
    'CTRTrainingPipeline', 
    'LargeDataMemoryTracker',
    'CTRValidationStrategy',
    'CTRHyperparameterOptimizer'
]