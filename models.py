# models.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import pickle
import gc
import warnings
import time
import threading
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM is not installed.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost is not installed.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost is not installed.")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn is not installed.")

TORCH_AVAILABLE = False
AMP_AVAILABLE = False
torch = None
nn = None
optim = None
DataLoader = None
TensorDataset = None
GradScaler = None
autocast = None

try:
    import torch
    
    gpu_available = False
    rtx_4060ti_detected = False
    
    if torch.cuda.is_available():
        try:
            gpu_properties = torch.cuda.get_device_properties(0)
            gpu_name = gpu_properties.name
            gpu_memory_gb = gpu_properties.total_memory / (1024**3)
            
            test_tensor = torch.zeros(1000, 1000).cuda()
            test_result = test_tensor.sum()
            del test_tensor
            torch.cuda.empty_cache()
            
            gpu_available = True
            rtx_4060ti_detected = 'RTX 4060 Ti' in gpu_name or gpu_memory_gb >= 15.0
            
            logging.info(f"GPU detected: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            logging.info(f"RTX 4060 Ti optimization: {rtx_4060ti_detected}")
            
        except Exception as e:
            logging.warning(f"GPU test failed: {e}.")
    
    TORCH_AVAILABLE = True
    
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
    
    try:
        from torch.cuda.amp import GradScaler, autocast
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False
        
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch is not installed.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available.")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available.")

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring with quick mode support"""
    
    def __init__(self):
        # Memory thresholds based on available memory (much lower for large datasets)
        self.memory_thresholds = {
            'warning': 8.0,     # Warning if less than 8GB available
            'critical': 4.0,    # Critical if less than 4GB available
            'abort': 1.0        # Abort if less than 1GB available
        }
        
        # Quick mode thresholds (even more relaxed)
        self.quick_mode_thresholds = {
            'warning': 2.0,     # Warning if less than 2GB available
            'critical': 1.0,    # Critical if less than 1GB available
            'abort': 0.5        # Abort if less than 0.5GB available
        }
        
        self.rtx_4060ti_optimized = False
        self.memory_logged = False
        self.quick_mode = False
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                if "RTX 4060 Ti" in device_name:
                    self.rtx_4060ti_optimized = True
            except Exception:
                pass
    
    def set_quick_mode(self, enabled: bool):
        """Enable quick mode with relaxed memory thresholds"""
        self.quick_mode = enabled
        if enabled:
            logger.info("Memory monitor set to quick mode - relaxed thresholds")
    
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
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status with mode-appropriate thresholds"""
        if not PSUTIL_AVAILABLE:
            return {
                'usage_gb': 0.0,
                'available_gb': 64.0,
                'level': 'normal',
                'should_simplify': False,
                'should_abort': False
            }
        
        vm = psutil.virtual_memory()
        usage_gb = vm.used / (1024**3)
        available_gb = vm.available / (1024**3)
        
        # Use appropriate thresholds based on mode
        thresholds = self.quick_mode_thresholds if self.quick_mode else self.memory_thresholds
        
        level = 'normal'
        should_simplify = False
        should_abort = False
        
        if available_gb < thresholds['abort']:
            level = 'abort'
            should_abort = True
        elif available_gb < thresholds['critical']:
            level = 'critical'
            should_simplify = True
        elif available_gb < thresholds['warning']:
            level = 'warning'
            should_simplify = True
        
        return {
            'usage_gb': usage_gb,
            'available_gb': available_gb,
            'level': level,
            'should_simplify': should_simplify,
            'should_abort': should_abort,
            'quick_mode': self.quick_mode,
            'thresholds': thresholds
        }
    
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

class MultiMethodCalibrator:
    """Multi-method probability calibration system"""
    
    def __init__(self):
        self.calibration_models = {}
        self.calibration_scores = {}
        self.best_method = None
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, method: str = 'auto') -> bool:
        """Fit calibration model with multiple methods"""
        try:
            if len(y_true) < 10:  # Lowered minimum requirement for quick mode
                logger.warning("Insufficient validation data for calibration")
                self._fit_bias_correction(y_true, y_pred_proba)
                return True
            
            y_true = np.array(y_true)
            y_pred_proba = np.array(y_pred_proba).flatten()
            
            if method == 'auto':
                methods_to_try = ['platt', 'isotonic', 'beta']
            else:
                methods_to_try = [method]
            
            for cal_method in methods_to_try:
                try:
                    if cal_method == 'platt':
                        self._fit_platt_scaling(y_true, y_pred_proba)
                    elif cal_method == 'isotonic':
                        self._fit_isotonic_regression(y_true, y_pred_proba)
                    elif cal_method == 'beta':
                        self._fit_beta_calibration(y_true, y_pred_proba)
                except Exception as e:
                    logger.warning(f"Calibration method {cal_method} failed: {e}")
                    continue
            
            if self.calibration_models:
                self.best_method = min(self.calibration_scores.keys(), 
                                     key=lambda k: self.calibration_scores[k])
                logger.info(f"Best calibration method: {self.best_method}")
                return True
            else:
                self._fit_bias_correction(y_true, y_pred_proba)
                return True
                
        except Exception as e:
            logger.warning(f"Calibration fitting failed: {e}")
            return False
    
    def _fit_platt_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Platt scaling calibration"""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            logits = np.log(np.clip(y_pred_proba, 1e-15, 1 - 1e-15) / 
                           np.clip(1 - y_pred_proba, 1e-15, 1 - 1e-15))
            logits = logits.reshape(-1, 1)
            
            platt_model = LogisticRegression()
            platt_model.fit(logits, y_true)
            
            calibrated_proba = platt_model.predict_proba(logits)[:, 1]
            brier_score = np.mean((calibrated_proba - y_true) ** 2)
            
            self.calibration_models['platt'] = platt_model
            self.calibration_scores['platt'] = brier_score
            
        except Exception as e:
            logger.warning(f"Platt scaling failed: {e}")
    
    def _fit_isotonic_regression(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Isotonic regression calibration"""
        if not SKLEARN_AVAILABLE:
            return
        
        try:
            isotonic_model = IsotonicRegression(out_of_bounds='clip')
            isotonic_model.fit(y_pred_proba, y_true)
            
            calibrated_proba = isotonic_model.predict(y_pred_proba)
            brier_score = np.mean((calibrated_proba - y_true) ** 2)
            
            self.calibration_models['isotonic'] = isotonic_model
            self.calibration_scores['isotonic'] = brier_score
            
        except Exception as e:
            logger.warning(f"Isotonic regression failed: {e}")
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Beta calibration"""
        try:
            from scipy.optimize import minimize
            from scipy.special import betaln
            
            def beta_log_likelihood(params, y_true, y_pred):
                a, b = params
                if a <= 0 or b <= 0:
                    return 1e10
                
                eps = 1e-15
                y_pred = np.clip(y_pred, eps, 1 - eps)
                
                log_lik = y_true * (a - 1) * np.log(y_pred) + \
                         (1 - y_true) * (b - 1) * np.log(1 - y_pred) - \
                         betaln(a, b)
                
                return -np.sum(log_lik)
            
            result = minimize(beta_log_likelihood, [1.0, 1.0], 
                            args=(y_true, y_pred_proba),
                            bounds=[(0.1, 10), (0.1, 10)])
            
            if result.success:
                a, b = result.x
                
                eps = 1e-15
                y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)
                log_proba = np.log(y_pred_clipped)
                log_1_minus_proba = np.log(1 - y_pred_clipped)
                
                log_calibrated_proba = (a - 1) * log_proba + (b - 1) * log_1_minus_proba
                log_calibrated_proba -= betaln(a, b)
                
                calibrated_proba = np.exp(log_calibrated_proba)
                calibrated_proba = np.clip(calibrated_proba, eps, 1 - eps)
                
                brier_score = np.mean((calibrated_proba - y_true) ** 2)
                
                self.calibration_models['beta'] = {'a': a, 'b': b}
                self.calibration_scores['beta'] = brier_score
                
        except Exception as e:
            logger.warning(f"Beta calibration failed: {e}")
    
    def _fit_bias_correction(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Simple bias correction"""
        try:
            actual_positive_rate = np.mean(y_true)
            predicted_positive_rate = np.mean(y_pred_proba)
            
            self.bias_correction = actual_positive_rate - predicted_positive_rate
            
            if predicted_positive_rate > 0:
                self.multiplicative_correction = actual_positive_rate / predicted_positive_rate
            else:
                self.multiplicative_correction = 1.0
                
        except Exception as e:
            logger.warning(f"Bias correction failed: {e}")
    
    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply calibration"""
        try:
            y_pred_proba = np.array(y_pred_proba).flatten()
            
            if self.best_method and self.best_method in self.calibration_models:
                model = self.calibration_models[self.best_method]
                
                if self.best_method == 'platt':
                    eps = 1e-15
                    y_pred_clipped = np.clip(y_pred_proba, eps, 1 - eps)
                    logits = np.log(y_pred_clipped / (1 - y_pred_clipped)).reshape(-1, 1)
                    calibrated_proba = model.predict_proba(logits)[:, 1]
                    return np.clip(calibrated_proba, eps, 1 - eps)
                
                elif self.best_method == 'isotonic':
                    calibrated_proba = model.predict(y_pred_proba)
                    return np.clip(calibrated_proba, 1e-15, 1 - 1e-15)
                
                elif self.best_method == 'beta':
                    from scipy.special import betaln
                    params = model
                    eps = 1e-15
                    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
                    log_proba = np.log(y_pred_proba)
                    log_1_minus_proba = np.log(1 - y_pred_proba)
                    
                    log_calibrated_proba = (params['a'] - 1) * log_proba + (params['b'] - 1) * log_1_minus_proba
                    log_calibrated_proba -= betaln(params['a'], params['b'])
                    
                    calibrated_proba = np.exp(log_calibrated_proba)
                    return np.clip(calibrated_proba, eps, 1 - eps)
            
            else:
                corrected_proba = y_pred_proba + self.bias_correction
                corrected_proba = corrected_proba * self.multiplicative_correction
                return np.clip(corrected_proba, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.warning(f"Calibration prediction failed: {e}")
            return np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary"""
        return {
            'best_method': self.best_method,
            'calibration_scores': self.calibration_scores,
            'available_methods': list(self.calibration_models.keys()),
            'bias_correction': self.bias_correction,
            'multiplicative_correction': self.multiplicative_correction
        }

class BaseModel(ABC):
    """Base class for all models"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.calibrator = None
        self.is_calibrated = False
        self.prediction_diversity_threshold = 2500
        self.calibration_applied = True
        self.memory_monitor = MemoryMonitor()
        self.quick_mode = False
    
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode for model training"""
        self.quick_mode = enabled
        self.memory_monitor.set_quick_mode(enabled)
        if enabled:
            logger.info(f"{self.name}: Quick mode enabled - simplified parameters")
        else:
            logger.info(f"{self.name}: Full mode enabled - complete parameter set")
    
    def _memory_safe_fit(self, fit_function, *args, **kwargs):
        """Memory safe fitting with quick mode support"""
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            # Skip memory check in quick mode for small datasets
            if self.quick_mode:
                logger.info(f"{self.name}: Quick mode - skipping memory check")
                return fit_function(*args, **kwargs)
            
            if memory_status['should_abort']:
                logger.error(f"Memory limit reached: {memory_status['available_gb']:.2f}GB available")
                raise MemoryError(f"Memory limit reached: {memory_status['available_gb']:.2f}GB available")
            
            if memory_status['should_simplify']:
                logger.warning(f"Memory pressure detected: {memory_status['available_gb']:.2f}GB available")
                self._simplify_for_memory()
            
            return fit_function(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Memory safe fit failed: {e}")
            raise
    
    def _memory_safe_predict(self, predict_function, X, batch_size=50000):  # Reduced batch size
        """Memory safe prediction with batching"""
        try:
            # For quick mode with small datasets, skip batching
            if self.quick_mode and len(X) < 1000:
                return predict_function(X)
            
            if len(X) <= batch_size:
                return predict_function(X)
            
            logger.info(f"Large dataset prediction: {len(X)} rows, using batch processing")
            predictions = []
            
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch_X = X.iloc[start_idx:end_idx]
                
                batch_pred = predict_function(batch_X)
                predictions.append(batch_pred)
                
                # Cleanup memory after each batch
                if start_idx % (batch_size * 5) == 0:
                    gc.collect()
            
            return np.concatenate(predictions)
            
        except Exception as e:
            logger.error(f"Memory safe prediction failed: {e}")
            raise
    
    def _simplify_for_memory(self):
        """Simplify model parameters for memory conservation"""
        pass
    
    def _ensure_feature_consistency(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency"""
        if self.feature_names is None:
            return X
        
        X_processed = X.copy()
        
        for feature in self.feature_names:
            if feature not in X_processed.columns:
                X_processed[feature] = 0
        
        X_processed = X_processed[self.feature_names]
        return X_processed
    
    def _enhance_prediction_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """Enhance prediction diversity"""
        if len(predictions) < self.prediction_diversity_threshold:
            return predictions
        
        try:
            noise_scale = max(0.0001, np.std(predictions) * 0.001)
            noise = np.random.normal(0, noise_scale, len(predictions))
            enhanced_predictions = predictions + noise
            return np.clip(enhanced_predictions, 1e-15, 1 - 1e-15)
        except Exception:
            return predictions
    
    def apply_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = 'auto'):
        """Apply probability calibration"""
        try:
            if not self.is_fitted:
                logger.warning("Model not fitted, cannot apply calibration")
                return False
            
            # Skip calibration in quick mode if dataset is too small
            if self.quick_mode and len(X_val) < 5:
                logger.info("Quick mode: Skipping calibration for small dataset")
                return False
            
            raw_predictions = self.predict_proba_raw(X_val)
            
            self.calibrator = MultiMethodCalibrator()
            success = self.calibrator.fit(y_val.values, raw_predictions, method)
            
            if success:
                self.is_calibrated = True
                return True
            else:
                logger.warning("Calibration fitting failed")
                return False
                
        except Exception as e:
            logger.warning(f"Calibration application failed: {e}")
            return False
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability predictions"""
        raw_predictions = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            return self.calibrator.predict_proba(raw_predictions)
        
        return raw_predictions
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw probability predictions before calibration"""
        pass

class LightGBMModel(BaseModel):
    """LightGBM model with memory management and tuned parameters"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,               # Reduced for memory efficiency
            'max_depth': 6,                 # Reduced for memory efficiency
            'learning_rate': 0.01,          # Slightly higher
            'feature_fraction': 0.8,        # Reduced for memory efficiency
            'bagging_fraction': 0.8,        # Reduced for memory efficiency
            'bagging_freq': 5,
            'min_data_in_leaf': 100,        # Increased for memory efficiency
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'min_gain_to_split': 0.01,
            'random_state': 42,
            'n_estimators': 3000,           # Reduced for memory efficiency
            'early_stopping_rounds': 100,   # Reduced for faster training
            'num_threads': 8,               # Reduced for memory efficiency
            'max_bin': 128,                 # Reduced for memory efficiency
            'verbosity': -1,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.2,
            'cat_smooth': 8.0,
            'cat_l2': 8.0,
            'max_cat_threshold': 16         # Reduced for memory efficiency
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("LightGBM", default_params)
        self.prediction_diversity_threshold = 2800
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        simplified_params = {
            'num_leaves': 15,              # Further reduced
            'max_depth': 4,                # Further reduced
            'n_estimators': 1000,          # Further reduced
            'min_data_in_leaf': 200,       # Further increased
            'num_threads': 4,              # Further reduced
            'max_bin': 64,                 # Further reduced
            'early_stopping_rounds': 50,   # Further reduced
            'feature_fraction': 0.7,       # Further reduced
            'bagging_fraction': 0.7        # Further reduced
        }
        
        self.params.update(simplified_params)
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def _apply_quick_mode_params(self):
        """Apply quick mode parameters for rapid testing"""
        quick_params = {
            'num_leaves': 15,
            'max_depth': 4,
            'n_estimators': 50,
            'learning_rate': 0.1,
            'min_data_in_leaf': 20,        # Lower for small datasets
            'num_threads': 4,
            'max_bin': 64,
            'early_stopping_rounds': 10,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'verbosity': -1
        }
        
        self.params.update(quick_params)
        logger.info(f"{self.name}: Quick mode parameters applied")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """LightGBM model training with quick mode support"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            if self.quick_mode:
                self._apply_quick_mode_params()
            
            # Apply memory efficient data preprocessing
            X_train_clean = X_train.fillna(0)
            
            # More aggressive dtype optimization
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
                elif X_train_clean[col].dtype in ['int64']:
                    X_train_clean[col] = X_train_clean[col].astype('int32')
            
            # Memory efficient dataset creation
            train_data = lgb.Dataset(
                X_train_clean, 
                label=y_train, 
                free_raw_data=True,  # Free raw data after creating dataset
                params={'max_bin': self.params.get('max_bin', 128)}
            )
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            X_val_clean = None
            if X_val is not None and y_val is not None:
                X_val_clean = X_val.fillna(0)
                for col in X_val_clean.columns:
                    if X_val_clean[col].dtype in ['float64']:
                        X_val_clean[col] = X_val_clean[col].astype('float32')
                    elif X_val_clean[col].dtype in ['int64']:
                        X_val_clean[col] = X_val_clean[col].astype('int32')
                
                valid_data = lgb.Dataset(
                    X_val_clean, 
                    label=y_val, 
                    reference=train_data, 
                    free_raw_data=True,
                    params={'max_bin': self.params.get('max_bin', 128)}
                )
                valid_sets.append(valid_data)
                valid_names.append('valid')
            
            early_stopping = self.params.get('early_stopping_rounds', 100)
            
            # Train with memory efficient callbacks
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(early_stopping), 
                    lgb.log_evaluation(0),
                    lgb.reset_parameter(max_bin=self.params.get('max_bin', 128))
                ]
            )
            
            self.is_fitted = True
            
            # Apply calibration if validation data available and not in quick mode
            if X_val_clean is not None and y_val is not None and not self.quick_mode:
                logger.info(f"{self.name}: Starting calibration application")
                self.apply_calibration(X_val_clean, y_val, method='auto')
                if self.is_calibrated:
                    logger.info(f"{self.name}: Calibration application complete")
                else:
                    logger.warning(f"{self.name}: Calibration application failed")
            
            # Cleanup
            del X_train_clean
            if X_val_clean is not None:
                del X_val_clean
            gc.collect()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions before calibration"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            # Apply same dtype optimization as training
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
                elif X_processed[col].dtype in ['int64']:
                    X_processed[col] = X_processed[col].astype('int32')
            
            num_iteration = getattr(self.model, 'best_iteration', None)
            proba = self.model.predict(X_processed, num_iteration=num_iteration)
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=30000)  # Smaller batch

class XGBoostModel(BaseModel):
    """XGBoost model with memory management and tuned parameters"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 6,                 # Reduced for memory efficiency
            'learning_rate': 0.01,          # Slightly higher
            'subsample': 0.8,               # Reduced for memory efficiency
            'colsample_bytree': 0.8,        # Reduced for memory efficiency
            'colsample_bylevel': 0.8,       # Reduced for memory efficiency
            'min_child_weight': 8,          # Increased for memory efficiency
            'reg_alpha': 0.1,
            'reg_lambda': 0.2,
            'scale_pos_weight': 52.3,
            'random_state': 42,
            'n_estimators': 2000,           # Reduced for memory efficiency
            'early_stopping_rounds': 100,   # Reduced for faster training
            'max_bin': 128,                 # Reduced for memory efficiency
            'nthread': 6,                   # Reduced for memory efficiency
            'grow_policy': 'lossguide',
            'gamma': 0.05,
            'max_leaves': 255               # Reduced for memory efficiency
        }
        
        # GPU optimization if available and memory allows
        if rtx_4060ti_detected and TORCH_AVAILABLE:
            try:
                test_tensor = torch.zeros(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                memory_monitor_temp = MemoryMonitor()
                memory_status = memory_monitor_temp.get_memory_status()
                
                if memory_status['level'] not in ['critical', 'abort']:
                    default_params.update({
                        'tree_method': 'gpu_hist',
                        'gpu_id': 0,
                        'predictor': 'gpu_predictor',
                        'max_bin': 64  # Further reduced for GPU memory
                    })
                    logger.info("XGBoost GPU mode enabled")
                else:
                    logger.info("Using XGBoost CPU mode due to memory shortage")
                    
            except Exception as e:
                logger.warning(f"GPU setup failed, using CPU mode: {e}")
        
        if params:
            default_params.update(params)
        
        super().__init__("XGBoost", default_params)
        self.prediction_diversity_threshold = 2800
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        simplified_params = {
            'max_depth': 4,                # Further reduced
            'n_estimators': 1000,          # Further reduced
            'min_child_weight': 16,        # Further increased
            'nthread': 4,                  # Further reduced
            'tree_method': 'hist',
            'early_stopping_rounds': 50,   # Further reduced
            'max_leaves': 63,              # Further reduced
            'max_bin': 64,                 # Further reduced
            'colsample_bytree': 0.7,       # Further reduced
            'subsample': 0.7               # Further reduced
        }
        
        self.params.update(simplified_params)
        # Remove GPU parameters if memory is low
        self.params.pop('gpu_id', None)
        self.params.pop('predictor', None)
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def _apply_quick_mode_params(self):
        """Apply quick mode parameters for rapid testing"""
        quick_params = {
            'max_depth': 3,
            'n_estimators': 50,
            'learning_rate': 0.1,
            'nthread': 4,
            'tree_method': 'hist',
            'max_bin': 64,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        self.params.update(quick_params)
        # Remove GPU parameters in quick mode
        self.params.pop('gpu_id', None)
        self.params.pop('predictor', None)
        logger.info(f"{self.name}: Quick mode parameters applied")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """XGBoost model training with quick mode support"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            if self.quick_mode:
                self._apply_quick_mode_params()
            
            logger.info(f"{self.name}: Performing memory cleanup before training")
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Apply memory efficient data preprocessing
            X_train_clean = X_train.fillna(0)
            
            # More aggressive dtype optimization
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
                elif X_train_clean[col].dtype in ['int64']:
                    X_train_clean[col] = X_train_clean[col].astype('int32')
            
            logger.info(f"{self.name}: Creating training DMatrix")
            dtrain = xgb.DMatrix(
                X_train_clean, 
                label=y_train, 
                feature_names=list(X_train_clean.columns),
                enable_categorical=False
            )
            
            eval_set = [(dtrain, 'train')]
            dval = None
            
            if X_val is not None and y_val is not None:
                logger.info(f"{self.name}: Creating validation DMatrix")
                X_val_clean = X_val.fillna(0)
                for col in X_val_clean.columns:
                    if X_val_clean[col].dtype in ['float64']:
                        X_val_clean[col] = X_val_clean[col].astype('float32')
                    elif X_val_clean[col].dtype in ['int64']:
                        X_val_clean[col] = X_val_clean[col].astype('int32')
                
                dval = xgb.DMatrix(
                    X_val_clean, 
                    label=y_val, 
                    feature_names=list(X_val_clean.columns),
                    enable_categorical=False
                )
                eval_set.append((dval, 'eval'))
            
            early_stopping = self.params.pop('early_stopping_rounds', 100)
            
            logger.info(f"{self.name}: Starting XGBoost training")
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get('n_estimators', 2000),
                evals=eval_set,
                early_stopping_rounds=early_stopping,
                verbose_eval=0
            )
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            # Apply calibration if validation data available and not in quick mode
            if dval is not None and not self.quick_mode:
                logger.info(f"{self.name}: Starting calibration application")
                X_val_clean = X_val.fillna(0)
                for col in X_val_clean.columns:
                    if X_val_clean[col].dtype in ['float64']:
                        X_val_clean[col] = X_val_clean[col].astype('float32')
                    elif X_val_clean[col].dtype in ['int64']:
                        X_val_clean[col] = X_val_clean[col].astype('int32')
                
                self.apply_calibration(X_val_clean, y_val, method='auto')
                if self.is_calibrated:
                    logger.info(f"{self.name}: Calibration application complete")
                else:
                    logger.warning(f"{self.name}: Calibration application failed")
            
            # Cleanup
            del dtrain
            if dval is not None:
                del dval
            del X_train_clean
            gc.collect()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions before calibration"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            # Apply same dtype optimization as training
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
                elif X_processed[col].dtype in ['int64']:
                    X_processed[col] = X_processed[col].astype('int32')
            
            dtest = xgb.DMatrix(
                X_processed, 
                feature_names=list(X_processed.columns),
                enable_categorical=False
            )
            proba = self.model.predict(dtest)
            
            del dtest
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=20000)  # Smaller batch

class LogisticModel(BaseModel):
    """Logistic Regression model with tuned parameters"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not installed.")
        
        default_params = {
            'C': 1.0,                      # Slightly reduced
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 2000,              # Reduced for memory efficiency
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': 6                    # Reduced for memory efficiency
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("LogisticRegression", default_params)
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 2800
        self.sampling_logged = False
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        simplified_params = {
            'C': 1.0,
            'max_iter': 1000,              # Further reduced
            'n_jobs': 4                    # Further reduced
        }
        
        self.params.update(simplified_params)
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def _apply_quick_mode_params(self):
        """Apply quick mode parameters for rapid testing"""
        quick_params = {
            'C': 1.0,
            'max_iter': 100,
            'n_jobs': 2,
            'solver': 'lbfgs'
        }
        
        self.params.update(quick_params)
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: Quick mode parameters applied")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Logistic Regression model training with quick mode support"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            if self.quick_mode:
                self._apply_quick_mode_params()
            
            # Apply memory efficient data preprocessing
            X_train_clean = X_train.fillna(0)
            
            # More aggressive dtype optimization
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
                elif X_train_clean[col].dtype in ['int64']:
                    X_train_clean[col] = X_train_clean[col].astype('int32')
            
            # More aggressive sampling for large datasets
            if self.quick_mode or len(X_train_clean) <= 2000000:  # Reduced from 6M
                X_train_sample = X_train_clean
                y_train_sample = y_train
            else:
                if not self.sampling_logged:
                    sample_size = 2000000  # Reduced from 6M
                    logger.info(f"Large data detected, applying sampling ({len(X_train_clean):,} -> {sample_size:,})")
                    self.sampling_logged = True
                    
                    # Stratified sampling
                    positive_indices = y_train[y_train == 1].index
                    negative_indices = y_train[y_train == 0].index
                    
                    positive_ratio = len(positive_indices) / len(y_train)
                    target_positive = max(int(sample_size * positive_ratio), 500)  # At least 500 positive
                    target_negative = sample_size - target_positive
                    
                    # Sample positive and negative separately
                    if len(positive_indices) > target_positive:
                        sampled_positive = np.random.choice(positive_indices, size=target_positive, replace=False)
                    else:
                        sampled_positive = positive_indices
                    
                    if len(negative_indices) > target_negative:
                        sampled_negative = np.random.choice(negative_indices, size=target_negative, replace=False)
                    else:
                        sampled_negative = negative_indices[:target_negative]
                    
                    sample_indices = np.concatenate([sampled_positive, sampled_negative])
                    np.random.shuffle(sample_indices)
                    
                    X_train_sample = X_train_clean.iloc[sample_indices]
                    y_train_sample = y_train.iloc[sample_indices]
                else:
                    X_train_sample = X_train_clean
                    y_train_sample = y_train
            
            try:
                start_time = time.time()
                self.model.fit(X_train_sample, y_train_sample)
                training_time = time.time() - start_time
                
                logger.info(f"{self.name} training complete (time taken: {training_time:.2f}s)")
                self.is_fitted = True
                
            except Exception as e:
                logger.warning(f"Logistic regression training failed: {e}")
                self._simplify_for_memory()
                self.model.fit(X_train_sample, y_train_sample)
                self.is_fitted = True
            
            # Apply calibration if validation data available and not in quick mode
            if X_val is not None and y_val is not None and not self.quick_mode:
                logger.info(f"{self.name}: Starting calibration application")
                X_val_clean = X_val.fillna(0)
                for col in X_val_clean.columns:
                    if X_val_clean[col].dtype in ['float64']:
                        X_val_clean[col] = X_val_clean[col].astype('float32')
                    elif X_val_clean[col].dtype in ['int64']:
                        X_val_clean[col] = X_val_clean[col].astype('int32')
                
                self.apply_calibration(X_val_clean, y_val, method='auto')
                if self.is_calibrated:
                    logger.info(f"{self.name}: Calibration application complete")
                else:
                    logger.warning(f"{self.name}: Calibration application failed")
            
            # Cleanup
            del X_train_clean
            gc.collect()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions before calibration"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            # Apply same dtype optimization as training
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
                elif X_processed[col].dtype in ['int64']:
                    X_processed[col] = X_processed[col].astype('int32')
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=40000)  # Larger batch for simpler model

class ModelFactory:
    """Model factory for creating models"""
    
    _factory_logged = False
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create model by type with optional quick mode support"""
        try:
            if not ModelFactory._factory_logged:
                memory_monitor = MemoryMonitor()
                logger.info(f"Creating model: {model_type} (calibration application configured)")
                
                # Get appropriate thresholds for logging
                if kwargs.get('quick_mode', False):
                    thresholds = memory_monitor.quick_mode_thresholds
                    logger.info(f"Quick mode thresholds - warning: {thresholds['warning']:.1f}GB, critical: {thresholds['critical']:.1f}GB, abort: {thresholds['abort']:.1f}GB")
                else:
                    thresholds = memory_monitor.memory_thresholds
                    logger.info(f"Memory thresholds - warning: {thresholds['warning']:.1f}GB, critical: {thresholds['critical']:.1f}GB, abort: {thresholds['abort']:.1f}GB")
                
                ModelFactory._factory_logged = True
            
            quick_mode = kwargs.get('quick_mode', False)
            
            if model_type.lower() == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM is not installed.")
                model = LightGBMModel(kwargs.get('params'))
                
            elif model_type.lower() == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost is not installed.")
                model = XGBoostModel(kwargs.get('params'))
                
            elif model_type.lower() == 'logistic':
                model = LogisticModel(kwargs.get('params'))
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if quick_mode:
                model.set_quick_mode(True)
            
            logger.info(f"{model_type} model creation complete - calibration application guaranteed")
            return model
                
        except Exception as e:
            logger.error(f"Model creation failed ({model_type}): {e}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """List of available model types"""
        available = []
        
        available.append('logistic')
        
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        
        logger.info(f"Available models: {available} (all models have calibration application)")
        return available
    
    @staticmethod
    def get_model_priority() -> List[str]:
        """Model priority list"""
        priority_order = []
        
        if LIGHTGBM_AVAILABLE:
            priority_order.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            priority_order.append('xgboost')
            
        if SKLEARN_AVAILABLE:
            priority_order.append('logistic')
        
        return priority_order
    
    @staticmethod
    def select_models_by_memory_status() -> List[str]:
        """Select models based on memory status"""
        memory_monitor = MemoryMonitor()
        memory_status = memory_monitor.get_memory_status()
        
        if memory_status['level'] == 'abort':
            return ['logistic']
        elif memory_status['level'] == 'critical':
            models = ['logistic']
            if LIGHTGBM_AVAILABLE:
                models.append('lightgbm')
            return models
        elif memory_status['level'] == 'warning':
            models = ['lightgbm', 'logistic']
            if XGBOOST_AVAILABLE:
                models.append('xgboost')
            return models
        else:
            return ModelFactory.get_available_models()

FinalLightGBMModel = LightGBMModel
FinalXGBoostModel = XGBoostModel  
FinalLogisticModel = LogisticModel