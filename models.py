# models.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import gc
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from scipy.special import betaln

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.preprocessing import StandardScaler
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
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        GPU_NAME = "None"
        GPU_MEMORY_GB = 0
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_NAME = "None"
    GPU_MEMORY_GB = 0

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring and management"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 45.0,
            'critical': 50.0,
            'abort': 55.0
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
                    'available_gb': 34.0,
                    'used_gb': 30.0,
                    'total_gb': 64.0,
                    'percent': 47.0
                }
        except Exception:
            return {
                'available_gb': 34.0,
                'used_gb': 30.0,
                'total_gb': 64.0,
                'percent': 47.0
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
        
        if GPU_AVAILABLE and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
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
    """CTR bias correction"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.correction_factor = 1.0
        self.additive_correction = 0.0
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Fit bias corrector"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            
            if predicted_ctr > 0:
                self.correction_factor = actual_ctr / predicted_ctr
                self.additive_correction = actual_ctr - predicted_ctr
            
            self.is_fitted = True
            logger.info(f"CTR bias corrector fitted: factor={self.correction_factor:.4f}, additive={self.additive_correction:.6f}")
            
        except Exception as e:
            logger.warning(f"CTR bias correction fitting failed: {e}")
            self.correction_factor = 1.0
            self.additive_correction = 0.0
            self.is_fitted = False
    
    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply bias correction"""
        try:
            if not self.is_fitted:
                return y_pred_proba
            
            corrected = y_pred_proba * self.correction_factor
            corrected = corrected + self.additive_correction * 0.1
            corrected = np.clip(corrected, 1e-15, 1 - 1e-15)
            
            return corrected
            
        except Exception as e:
            logger.warning(f"CTR bias correction application failed: {e}")
            return y_pred_proba

class EnhancedMultiMethodCalibrator:
    """Multi-method calibration"""
    
    def __init__(self):
        self.calibration_models = {}
        self.best_method = None
        self.ensemble_calibrator = {}
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
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
            
            if len(self.calibration_models) >= 2:
                self._fit_ensemble_calibrator(y_true, y_pred_proba)
            
            self.is_fitted = True
            logger.info(f"Calibration fitted: {len(self.calibration_models)} methods available")
            return True
            
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            return False
    
    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply calibration"""
        try:
            if not self.is_fitted:
                return self.ctr_corrector.transform(y_pred_proba)
            
            if self.ensemble_calibrator and len(self.ensemble_calibrator) > 1:
                calibrated = self._ensemble_predict(y_pred_proba)
            elif self.best_method:
                calibrated = self._predict_with_method(y_pred_proba, self.best_method)
            else:
                calibrated = y_pred_proba
            
            if calibrated is not None:
                calibrated = self.ctr_corrector.transform(calibrated)
            else:
                calibrated = self.ctr_corrector.transform(y_pred_proba)
            
            return np.clip(calibrated, 1e-15, 1 - 1e-15)
            
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
    
    def _fit_ensemble_calibrator(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Fit ensemble calibrator"""
        try:
            method_predictions = {}
            for method in self.calibration_models.keys():
                pred = self._predict_with_method(y_pred_proba, method)
                if pred is not None:
                    method_predictions[method] = pred
            
            if len(method_predictions) < 2:
                return
            
            weights = {}
            total_score = 0
            
            for method, pred in method_predictions.items():
                try:
                    score = -log_loss(y_true, pred)
                    weights[method] = max(0, score)
                    total_score += weights[method]
                except:
                    weights[method] = 0
            
            if total_score > 0:
                for method in weights:
                    weights[method] /= total_score
                
                self.ensemble_calibrator = weights
                logger.info(f"Ensemble calibrator fitted with weights: {weights}")
            
        except Exception as e:
            logger.warning(f"Ensemble calibrator fitting failed: {e}")
    
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
    
    def _ensemble_predict(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Ensemble prediction"""
        try:
            predictions = []
            weights = []
            
            for method, weight in self.ensemble_calibrator.items():
                pred = self._predict_with_method(y_pred_proba, method)
                if pred is not None and weight > 0:
                    predictions.append(pred)
                    weights.append(weight)
            
            if predictions:
                weights = np.array(weights)
                weights = weights / weights.sum()
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                return ensemble_pred
            else:
                return y_pred_proba
                
        except Exception:
            return y_pred_proba
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary"""
        return {
            'is_fitted': self.is_fitted,
            'best_method': self.best_method,
            'available_methods': list(self.calibration_models.keys()),
            'ensemble_weights': self.ensemble_calibrator
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
        self.scaler = StandardScaler()
        self.use_scaling = True
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
                logger.warning(f"{self.name}: Low memory ({memory_status['available_gb']:.1f}GB), attempting to proceed")
            
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
    
    def _safe_data_preprocessing(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Safe data preprocessing with strict validation"""
        try:
            X_processed = X.copy()
            
            # Remove non-numeric columns except expected categoricals
            non_numeric_cols = X_processed.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                logger.warning(f"{self.name}: Removing non-numeric columns: {non_numeric_cols}")
                X_processed = X_processed.drop(columns=non_numeric_cols)
            
            # Verify all columns are numeric
            remaining_types = X_processed.dtypes.unique()
            if not all(np.issubdtype(dtype, np.number) for dtype in remaining_types):
                logger.error(f"{self.name}: Non-numeric data types detected after cleanup: {remaining_types}")
                # Force convert to numeric
                for col in X_processed.columns:
                    try:
                        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"{self.name}: Failed to convert {col} to numeric: {e}")
                        X_processed[col] = 0.0
            
            # Handle missing values
            numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_columns] = X_processed[numeric_columns].fillna(0)
            
            # Handle infinity
            X_processed = X_processed.replace([np.inf, -np.inf], 0)
            
            # Scale if needed
            if self.use_scaling and len(numeric_columns) > 0:
                if fit_scaler:
                    X_processed[numeric_columns] = self.scaler.fit_transform(X_processed[numeric_columns])
                else:
                    X_processed[numeric_columns] = self.scaler.transform(X_processed[numeric_columns])
            
            # Final verification
            if X_processed.empty:
                logger.error(f"{self.name}: Data preprocessing resulted in empty dataframe")
                raise ValueError("Data preprocessing failed - empty result")
            
            # Log final state
            logger.debug(f"{self.name}: Preprocessing complete - shape: {X_processed.shape}, dtypes: {X_processed.dtypes.value_counts().to_dict()}")
            
            return X_processed
            
        except Exception as e:
            logger.warning(f"{self.name}: Data preprocessing failed: {e}")
            # Last resort: return numeric columns only with zero fill
            try:
                X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
                if not X_numeric.empty:
                    return X_numeric
            except:
                pass
            return X
    
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
                return np.clip(enhanced_predictions, 1e-15, 1 - 1e-15)
            
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
    
    def _simplify_for_memory(self):
        """Simplify for memory"""
        pass

class XGBoostModel(BaseModel):
    """XGBoost model with Windows + RTX 4060 Ti GPU optimization"""
    
    def __init__(self, name: str = "XGBoost_GPU", params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'min_child_weight': 10,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'scale_pos_weight': 51.3,
            'random_state': 42,
            'n_jobs': 6
        }
        
        if GPU_AVAILABLE:
            default_params.update({
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
                'max_bin': 256
            })
            logger.info(f"XGBoost GPU mode enabled: {GPU_NAME} ({GPU_MEMORY_GB:.1f}GB)")
        else:
            logger.warning("XGBoost GPU not available, using CPU mode")
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = None
        self.use_scaling = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training with GPU optimization"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            X_train_clean = self._safe_data_preprocessing(X_train, fit_scaler=False)
            
            logger.info(f"{self.name}: Starting XGBoost GPU training")
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                X_val_clean = self._safe_data_preprocessing(X_val, fit_scaler=False)
                self.model = xgb.XGBClassifier(**self.params)
                
                eval_set = [(X_val_clean, y_val)]
                self.model.fit(
                    X_train_clean, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.model = xgb.XGBClassifier(**self.params)
                self.model.fit(X_train_clean, y_train)
            
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
            
            del X_train_clean
            gc.collect()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = self._safe_data_preprocessing(X_processed, fit_scaler=False)
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=25000)

class LightGBMModel(BaseModel):
    """LightGBM model with CPU mode for stability"""
    
    def __init__(self, name: str = "LightGBM", params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 800,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': 6,
            'verbose': -1,
            'device': 'cpu',
            'is_unbalance': True,
            'scale_pos_weight': 51.3
        }
        
        logger.info(f"LightGBM CPU mode enabled for stability")
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = None
        self.use_scaling = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            X_train_clean = self._safe_data_preprocessing(X_train, fit_scaler=False)
            
            logger.info(f"{self.name}: Starting LightGBM training")
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                X_val_clean = self._safe_data_preprocessing(X_val, fit_scaler=False)
                self.model = lgb.LGBMClassifier(**self.params)
                self.model.fit(
                    X_train_clean, y_train,
                    eval_set=[(X_val_clean, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                self.model = lgb.LGBMClassifier(**self.params)
                self.model.fit(X_train_clean, y_train)
            
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
            
            del X_train_clean
            gc.collect()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = self._safe_data_preprocessing(X_processed, fit_scaler=False)
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=50000)

class LogisticModel(BaseModel):
    """Logistic Regression model"""
    
    def __init__(self, name: str = "LogisticRegression", params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not installed.")
        
        default_params = {
            'C': 0.5,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 2000,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': 6,
            'tol': 0.00005
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 2500
        self.sampling_logged = False
        self.use_scaling = True
    
    def _simplify_for_memory(self):
        """Simplify parameters"""
        simplified_params = {
            'C': 0.5,
            'max_iter': 2000,
            'n_jobs': 4,
            'tol': 0.0001
        }
        
        self.params.update(simplified_params)
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: Parameters simplified for memory")
    
    def _apply_quick_mode_params(self):
        """Apply quick mode parameters"""
        quick_params = {
            'C': 0.5,
            'max_iter': 1000,
            'n_jobs': 2,
            'solver': 'saga',
            'tol': 0.001
        }
        
        self.params.update(quick_params)
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: Quick mode parameters applied")
    
    def _safe_sampling(self, X_train: pd.DataFrame, y_train: pd.Series, target_size: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Safe stratified sampling"""
        try:
            current_size = len(X_train)
            
            if current_size <= target_size:
                return X_train, y_train
            
            unique_labels = np.unique(y_train)
            if len(unique_labels) > 1:
                X_sampled, _, y_sampled, _ = train_test_split(
                    X_train, y_train,
                    train_size=target_size,
                    random_state=42,
                    stratify=y_train
                )
            else:
                indices = np.random.choice(current_size, target_size, replace=False)
                X_sampled = X_train.iloc[indices]
                y_sampled = y_train.iloc[indices]
            
            if not self.sampling_logged:
                logger.info(f"{self.name}: Data sampling applied - {current_size} -> {target_size} samples")
                self.sampling_logged = True
            
            return X_sampled, y_sampled
            
        except Exception as e:
            logger.warning(f"{self.name}: Sampling failed: {e}")
            return X_train, y_train
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            if self.quick_mode:
                self._apply_quick_mode_params()
            
            memory_status = self.memory_monitor.get_memory_status()
            
            X_train_sample, y_train_sample = X_train, y_train
            
            X_train_clean = self._safe_data_preprocessing(X_train_sample, fit_scaler=True)
            
            logger.info(f"{self.name}: Starting training")
            self.model.fit(X_train_clean, y_train_sample)
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                try:
                    val_pred = self.predict_proba_raw(X_val)
                    from sklearn.metrics import roc_auc_score
                    self.validation_score = roc_auc_score(y_val, val_pred)
                except:
                    self.validation_score = 0.5
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                calibration_success = self.apply_calibration(X_val, y_val, method='auto')
                if calibration_success:
                    logger.info(f"{self.name}: Calibration completed successfully")
                else:
                    logger.warning(f"{self.name}: Calibration failed - CTR correction applied")
            else:
                logger.warning(f"{self.name}: No validation data - calibration skipped")
            
            self.training_time = time.time() - start_time
            
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
            X_processed = self._safe_data_preprocessing(X_processed, fit_scaler=False)
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
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
                
                if GPU_AVAILABLE:
                    logger.info(f"GPU available: {GPU_NAME} ({GPU_MEMORY_GB:.1f}GB)")
                else:
                    logger.info("GPU not available, using CPU mode")
                
                ModelFactory._factory_logged = True
            
            quick_mode = kwargs.get('quick_mode', False)
            
            if model_type.lower() == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost is not installed.")
                model = XGBoostModel(params=kwargs.get('params'))
                
            elif model_type.lower() == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM is not installed.")
                model = LightGBMModel(params=kwargs.get('params'))
                
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
        available = []
        
        available.append('logistic')
        
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        
        logger.info(f"Available models: {available}")
        return available
    
    @staticmethod
    def get_model_priority() -> List[str]:
        """Model priority list"""
        priority_order = []
        
        if XGBOOST_AVAILABLE:
            priority_order.append('xgboost')
        
        if LIGHTGBM_AVAILABLE:
            priority_order.append('lightgbm')
            
        if SKLEARN_AVAILABLE:
            priority_order.append('logistic')
        
        return priority_order

FinalLightGBMModel = LightGBMModel
FinalXGBoostModel = XGBoostModel  
FinalLogisticModel = LogisticModel