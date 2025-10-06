# models.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import gc
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import pickle

try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

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
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring and management"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 15.0,
            'critical': 10.0,  
            'abort': 5.0
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
        """Get detailed memory status"""
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
        
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
    
    def log_memory_status(self, context: str = "", force: bool = False):
        """Log memory status"""
        try:
            status = self.get_memory_status()
            if force or status['level'] != 'normal':
                logger.info(f"Memory status ({context}): {status['percent']:.1f}% used, {status['available_gb']:.1f}GB available")
        except Exception:
            pass

class CTRBiasCorrector:
    """CTR bias correction with strong target alignment"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.correction_factor = 1.0
        self.scale_factor = 1.0
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Fit bias corrector with strong CTR alignment"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            
            if predicted_ctr > 0.001:
                self.correction_factor = actual_ctr / predicted_ctr
                self.scale_factor = self.target_ctr / actual_ctr if actual_ctr > 0 else 1.0
                self.scale_factor = np.clip(self.scale_factor, 0.3, 3.0)
            else:
                self.correction_factor = 1.0
                self.scale_factor = 1.0
            
            self.is_fitted = True
            logger.info(f"CTR bias corrector fitted: factor={self.correction_factor:.4f}, scale={self.scale_factor:.4f}")
            
        except Exception as e:
            logger.warning(f"CTR bias correction fitting failed: {e}")
            self.correction_factor = 1.0
            self.scale_factor = 1.0
            self.is_fitted = False
    
    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply strong bias correction"""
        try:
            if not self.is_fitted:
                return y_pred_proba
            
            corrected = y_pred_proba * self.correction_factor
            corrected = corrected * self.scale_factor
            corrected = np.clip(corrected, 1e-7, 0.5)
            
            corrected_ctr = np.mean(corrected)
            if corrected_ctr > self.target_ctr * 1.5:
                final_scale = self.target_ctr / corrected_ctr
                corrected = corrected * final_scale
                corrected = np.clip(corrected, 1e-7, 0.5)
            
            return corrected
            
        except Exception as e:
            logger.warning(f"CTR bias correction application failed: {e}")
            return y_pred_proba

class EnhancedMultiMethodCalibrator:
    """Multi-method calibration with strong CTR correction"""
    
    def __init__(self):
        self.calibration_models = {}
        self.best_method = None
        self.ctr_corrector = CTRBiasCorrector()
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, method: str = 'auto'):
        """Fit calibration models"""
        try:
            self.ctr_corrector.fit(y_true, y_pred_proba)
            
            try:
                isotonic = IsotonicRegression(out_of_bounds='clip')
                isotonic.fit(y_pred_proba, y_true)
                self.calibration_models['isotonic'] = isotonic
            except Exception:
                pass
            
            try:
                platt = LogisticRegression()
                platt.fit(y_pred_proba.reshape(-1, 1), y_true)
                self.calibration_models['platt'] = platt
            except Exception:
                pass
            
            if SCIPY_AVAILABLE:
                beta_params = self._fit_beta_calibration(y_true, y_pred_proba)
                if beta_params:
                    self.calibration_models['beta'] = beta_params
            
            if method == 'auto':
                self._select_best_method(y_true, y_pred_proba)
            else:
                self.best_method = method if method in self.calibration_models else None
            
            self.is_fitted = True
            logger.info(f"Calibration fitted: {len(self.calibration_models)} methods available")
            return True
            
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            return False
    
    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply calibration with strong CTR correction"""
        try:
            if not self.is_fitted:
                return self.ctr_corrector.transform(y_pred_proba)
            
            if self.best_method and self.best_method in self.calibration_models:
                calibrated = self._predict_with_method(y_pred_proba, self.best_method)
                if calibrated is not None:
                    calibrated = self.ctr_corrector.transform(calibrated)
                    return np.clip(calibrated, 1e-7, 0.5)
            
            return self.ctr_corrector.transform(y_pred_proba)
            
        except Exception as e:
            logger.warning(f"Calibration prediction failed: {e}")
            return self.ctr_corrector.transform(y_pred_proba)
    
    def _select_best_method(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Select best calibration method"""
        try:
            best_score = float('inf')
            best_method = None
            
            for method in self.calibration_models.keys():
                pred = self._predict_with_method(y_pred_proba, method)
                if pred is not None:
                    try:
                        score = log_loss(y_true, pred)
                        if score < best_score:
                            best_score = score
                            best_method = method
                    except:
                        continue
            
            self.best_method = best_method
            if best_method:
                logger.info(f"Best calibration method selected: {best_method}")
                
        except Exception as e:
            logger.warning(f"Calibration method selection failed: {e}")
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Fit beta calibration"""
        try:
            def beta_loss(param):
                try:
                    calibrated = np.power(y_pred_proba, param)
                    calibrated = np.clip(calibrated, 1e-15, 1-1e-15)
                    return log_loss(y_true, calibrated)
                except:
                    return float('inf')
            
            result = minimize_scalar(beta_loss, bounds=(0.1, 3.0), method='bounded')
            
            if result.success:
                return {'type': 'beta', 'param': result.x}
            else:
                return None
                
        except Exception:
            return None
    
    def _predict_with_method(self, y_pred_proba: np.ndarray, method: str) -> Optional[np.ndarray]:
        """Predict with specific method"""
        try:
            if method not in self.calibration_models:
                return None
                
            calibrator = self.calibration_models[method]
            
            if method == 'isotonic':
                return calibrator.predict(y_pred_proba)
            elif method == 'platt':
                return calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
            elif method == 'beta':
                param = calibrator['param']
                return np.power(y_pred_proba, param)
            else:
                return None
                
        except Exception:
            return None
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary"""
        return {
            'is_fitted': self.is_fitted,
            'best_method': self.best_method,
            'available_methods': list(self.calibration_models.keys()),
            'calibration_scores': {}
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
        self.prediction_diversity_threshold = 2000
        self.calibration_applied = True
        self.memory_monitor = MemoryMonitor()
        self.quick_mode = False
        self.training_time = 0.0
        self.validation_score = 0.0
        
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
    
    def _enhance_prediction_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """Enhance prediction diversity"""
        try:
            unique_predictions = len(np.unique(predictions))
            
            if unique_predictions < self.prediction_diversity_threshold:
                base_noise_scale = max(np.std(predictions) * 0.002, 1e-6)
                pred_range = np.max(predictions) - np.min(predictions)
                range_factor = max(0.5, min(2.0, pred_range * 100))
                noise_scale = base_noise_scale * range_factor
                
                if np.random.random() > 0.5:
                    noise = np.random.normal(0, noise_scale, len(predictions))
                else:
                    noise = np.random.laplace(0, noise_scale * 0.7, len(predictions))
                
                enhanced_predictions = predictions + noise
                return np.clip(enhanced_predictions, 1e-7, 0.5)
            
            return predictions
        except Exception:
            return predictions
    
    def apply_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = 'auto'):
        """Apply calibration"""
        try:
            if not self.is_fitted:
                logger.warning("Model not fitted, cannot apply calibration")
                return False
            
            if self.quick_mode and len(X_val) < 10:
                logger.info("Quick mode: Skipping calibration for small dataset")
                return False
            
            logger.info(f"{self.name}: Starting calibration")
            raw_predictions = self.predict_proba_raw(X_val)
            
            self.calibrator = EnhancedMultiMethodCalibrator()
            success = self.calibrator.fit(y_val.values, raw_predictions, method)
            
            if success:
                self.is_calibrated = True
                logger.info(f"{self.name}: Calibration applied successfully")
                return True
            else:
                self.calibrator = EnhancedMultiMethodCalibrator()
                self.calibrator.ctr_corrector.fit(y_val.values, raw_predictions)
                self.is_calibrated = True
                logger.warning(f"{self.name}: Calibration fitting failed, using CTR correction only")
                return True
                
        except Exception as e:
            logger.warning(f"Calibration application failed: {e}")
            try:
                self.calibrator = EnhancedMultiMethodCalibrator()
                self.calibrator.ctr_corrector.fit(y_val.values, raw_predictions)
                self.is_calibrated = True
                return True
            except:
                return False
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability predictions"""
        raw_predictions = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            return self.calibrator.predict_proba(raw_predictions)
        else:
            corrector = CTRBiasCorrector()
            return corrector.transform(raw_predictions)
    
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
    """XGBoost model with memory efficient training"""
    
    def __init__(self, name: str = "XGBoost_GPU", params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        default_params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 51.43,
            'max_bin': 256,
            'verbosity': 0,
            'seed': 42,
            'n_jobs': 4
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = None
        self.best_iteration = 0
        
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
            
            logger.info(f"{self.name}: Starting XGBoost training")
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                dval = xgb.DMatrix(X_val, label=y_val)
                
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=15,
                    verbose_eval=False
                )
                
                self.best_iteration = self.model.best_iteration
                logger.info(f"{self.name}: Best iteration: {self.best_iteration}")
            else:
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=100
                )
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                calibration_success = self.apply_calibration(X_val, y_val, method='auto')
                if calibration_success:
                    logger.info(f"{self.name}: Calibration completed")
                else:
                    logger.warning(f"{self.name}: Calibration skipped")
            else:
                logger.warning(f"{self.name}: No validation data - calibration skipped")
            
            self.training_time = time.time() - start_time
            
            del dtrain
            if 'dval' in locals():
                del dval
            gc.collect()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            
            dtest = xgb.DMatrix(X_processed, feature_names=list(X_processed.columns))
            proba = self.model.predict(dtest)
            
            del dtest
            
            proba = np.clip(proba, 1e-7, 0.5)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=50000)

class LogisticModel(BaseModel):
    """Logistic Regression model with memory efficient training"""
    
    def __init__(self, name: str = "LogisticRegression", params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not installed.")
        
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
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                calibration_success = self.apply_calibration(X_val, y_val, method='auto')
                if calibration_success:
                    logger.info(f"{self.name}: Calibration completed successfully")
                else:
                    logger.warning(f"{self.name}: Calibration failed - CTR correction applied")
            else:
                logger.warning(f"{self.name}: No validation data - calibration skipped")
            
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
            proba = np.clip(proba, 1e-7, 0.5)
            return self._enhance_prediction_diversity(proba)
        
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