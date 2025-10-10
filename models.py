# models.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import gc
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from pathlib import Path
import pickle
import warnings

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring and management"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 10.0,
            'critical': 7.0,  
            'abort': 4.0
        }
        
        self.quick_mode_thresholds = {
            'warning': 4.0,
            'critical': 2.0,
            'abort': 1.0
        }
        
        self.quick_mode = False
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for memory monitoring"""
        self.quick_mode = enabled
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                return {
                    'available_gb': vm.available / (1024**3),
                    'used_gb': vm.used / (1024**3),
                    'total_gb': vm.total / (1024**3),
                    'percent': vm.percent
                }
            else:
                return {
                    'available_gb': 32.0,
                    'used_gb': 16.0,
                    'total_gb': 48.0,
                    'percent': 33.3
                }
        except Exception:
            return {
                'available_gb': 32.0,
                'used_gb': 16.0,
                'total_gb': 48.0,
                'percent': 33.3
            }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Return memory status"""
        usage = self.get_memory_usage()
        available = usage['available_gb']
        
        thresholds = self.quick_mode_thresholds if self.quick_mode else self.memory_thresholds
        
        if available < thresholds['abort']:
            level = 'abort'
        elif available < thresholds['critical']:
            level = 'critical'
        elif available < thresholds['warning']:
            level = 'warning'
        else:
            level = 'normal'
        
        return {
            'level': level,
            'available_gb': available,
            'used_gb': usage['used_gb'],
            'percent': usage['percent'],
            'should_cleanup': level in ['critical', 'abort']
        }
    
    def force_memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        
        if TORCH_GPU_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
    
    def log_memory_status(self, context: str = "", force: bool = False):
        """Log memory status"""
        try:
            status = self.get_memory_status()
            if force or status['level'] != 'normal':
                logger.info(f"Memory status ({context}): {status['percent']:.1f}% used, {status['available_gb']:.1f}GB available")
        except Exception:
            pass

class CTRCalibrator:
    """CTR-specific calibration with Isotonic Regression"""
    
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self.target_ctr = 0.0191
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Fit calibration model"""
        try:
            logger.info(f"Fitting {self.method} calibration")
            
            if self.method == 'isotonic':
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_pred_proba, y_true)
            else:
                logger.warning(f"Unknown calibration method: {self.method}, using isotonic")
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_pred_proba, y_true)
            
            self.is_fitted = True
            logger.info("Calibration fitted successfully")
            
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            self.is_fitted = False
    
    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply calibration"""
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning original predictions")
            return y_pred_proba
        
        try:
            calibrated = self.calibrator.transform(y_pred_proba)
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
            
            original_ctr = np.mean(y_pred_proba)
            calibrated_ctr = np.mean(calibrated)
            
            logger.info(f"Calibration applied: {original_ctr:.4f} -> {calibrated_ctr:.4f}")
            
            return calibrated
            
        except Exception as e:
            logger.error(f"Calibration transform failed: {e}")
            return y_pred_proba
    
    def fit_transform(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(y_true, y_pred_proba)
        return self.transform(y_pred_proba)

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
        self.prediction_diversity_threshold = 2000
        self.calibration_applied = False
        self.memory_monitor = MemoryMonitor()
        self.quick_mode = False
        self.training_time = 0.0
        self.validation_score = 0.0
        self.ctr_calibrator = None
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode"""
        self.quick_mode = enabled
        self.memory_monitor.set_quick_mode(enabled)
        if enabled:
            logger.info(f"{self.name}: Quick mode enabled")
        else:
            logger.info(f"{self.name}: Full mode enabled")
    
    def _memory_safe_fit(self, fit_function, *args, **kwargs):
        """Memory safe fitting"""
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            if memory_status['level'] == 'abort':
                logger.error(f"{self.name}: Insufficient memory for training")
                return None
            elif memory_status['level'] == 'critical':
                logger.warning(f"{self.name}: Memory critical, using reduced mode")
            
            return fit_function(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"{self.name}: Memory safe fitting failed: {e}")
            return None
    
    def _memory_safe_predict(self, predict_function, X: pd.DataFrame, batch_size: int = 10000):
        """Memory safe prediction"""
        try:
            if len(X) <= batch_size:
                return predict_function(X)
            
            results = []
            for i in range(0, len(X), batch_size):
                batch = X.iloc[i:i + batch_size]
                batch_result = predict_function(batch)
                results.append(batch_result)
                
                if i % (batch_size * 5) == 0:
                    gc.collect()
            
            return np.concatenate(results)
            
        except Exception as e:
            logger.error(f"{self.name}: Memory safe prediction failed: {e}")
            return np.array([])
    
    def _ensure_feature_consistency(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency"""
        try:
            if self.feature_names is not None:
                missing_features = set(self.feature_names) - set(X.columns)
                extra_features = set(X.columns) - set(self.feature_names)
                
                if missing_features:
                    for feature in missing_features:
                        X[feature] = 0
                
                if extra_features:
                    X = X.drop(columns=list(extra_features))
                
                X = X[self.feature_names]
            
            return X
            
        except Exception as e:
            logger.warning(f"{self.name}: Feature consistency check failed: {e}")
            return X
    
    def apply_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = 'isotonic'):
        """Apply CTR-specific calibration"""
        try:
            logger.info(f"{self.name}: Applying {method} calibration")
            
            y_pred_proba = self.predict_proba_raw(X_val)
            
            self.ctr_calibrator = CTRCalibrator(method=method)
            self.ctr_calibrator.fit(y_val.values, y_pred_proba)
            
            self.is_calibrated = True
            logger.info(f"{self.name}: Calibration applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Calibration failed: {e}")
            self.is_calibrated = False
            return False
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability predictions with calibration if available"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.ctr_calibrator and self.ctr_calibrator.is_fitted:
            return self.ctr_calibrator.transform(raw_pred)
        
        return raw_pred
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw probability predictions"""
        pass

class XGBoostGPUModel(BaseModel):
    """XGBoost model with GPU acceleration"""
    
    def __init__(self, name: str = "XGBoost_GPU", params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        default_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist' if TORCH_GPU_AVAILABLE else 'hist',
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 15.0,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1.5,
            'max_bin': 256,
            'gpu_id': 0 if TORCH_GPU_AVAILABLE else None,
            'predictor': 'gpu_predictor' if TORCH_GPU_AVAILABLE else 'cpu_predictor',
            'verbosity': 0,
            'seed': 42,
            'n_jobs': -1,
            'max_delta_step': 1
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = None
        self.best_iteration = 0
        self.gpu_enabled = TORCH_GPU_AVAILABLE and default_params.get('tree_method') == 'gpu_hist'
        
        if self.gpu_enabled:
            logger.info(f"{self.name}: GPU mode enabled with gpu_hist")
        else:
            logger.info(f"{self.name}: CPU mode with hist")
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training with GPU acceleration"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['level'] == 'abort':
                logger.error(f"{self.name}: Insufficient memory for training")
                return None
            
            self.feature_names = list(X_train.columns)
            
            logger.info(f"{self.name}: Starting XGBoost training")
            
            if self.gpu_enabled:
                self.memory_monitor.force_memory_cleanup()
                logger.info(f"{self.name}: GPU memory cleared before training")
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            num_boost_round = 100 if self.quick_mode else 200
            early_stopping = 15 if self.quick_mode else 20
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                dval = xgb.DMatrix(X_val, label=y_val)
                
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=early_stopping,
                    verbose_eval=False
                )
                
                self.best_iteration = self.model.best_iteration
                logger.info(f"{self.name}: Best iteration: {self.best_iteration}")
            else:
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_boost_round
                )
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            self.training_time = time.time() - start_time
            
            del dtrain
            if 'dval' in locals():
                del dval
            gc.collect()
            
            if self.gpu_enabled:
                self.memory_monitor.force_memory_cleanup()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions with GPU acceleration"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            
            dtest = xgb.DMatrix(X_processed, feature_names=list(X_processed.columns))
            proba = self.model.predict(dtest)
            
            del dtest
            
            proba = np.clip(proba, 1e-7, 1 - 1e-7)
            return proba
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=50000)

class LogisticModel(BaseModel):
    """Logistic Regression model with memory efficient training"""
    
    def __init__(self, name: str = "LogisticRegression", params: Dict[str, Any] = None):
        default_params = {
            'C': 0.5,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 100,
            'random_state': 42,
            'n_jobs': 4,
            'tol': 0.001,
            'warm_start': True
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 2500
        self.sampling_logged = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training with memory management"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['level'] == 'abort':
                logger.error(f"{self.name}: Insufficient memory for training")
                return None
            
            self.feature_names = list(X_train.columns)
            
            logger.info(f"{self.name}: Starting training")
            self.model.fit(X_train, y_train)
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            self.training_time = time.time() - start_time
            
            gc.collect()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-7, 1 - 1e-7)
            return proba
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=50000)

class ModelFactory:
    """Model factory"""
    
    _factory_logged = False
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create model by type"""
        try:
            if not ModelFactory._factory_logged:
                memory_monitor = MemoryMonitor()
                logger.info(f"Creating model: {model_type}")
                ModelFactory._factory_logged = True
            
            quick_mode = kwargs.get('quick_mode', False)
            
            if model_type.lower() == 'xgboost_gpu':
                model = XGBoostGPUModel(params=kwargs.get('params'))
                
            elif model_type.lower() == 'logistic':
                model = LogisticModel(params=kwargs.get('params'))
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if quick_mode:
                model.set_quick_mode(True)
            
            logger.info(f"{model_type} model creation complete")
            return model
                
        except Exception as e:
            logger.error(f"Model creation failed ({model_type}): {e}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """List of available model types"""
        available = ['logistic']
        
        if XGBOOST_AVAILABLE:
            available.append('xgboost_gpu')
        
        logger.info(f"Available models: {available}")
        return available

FinalXGBoostGPUModel = XGBoostGPUModel
FinalLogisticModel = LogisticModel