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
from scipy.special import betaln

# Safe imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.preprocessing import StandardScaler
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

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring and management"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 15.0,    # GB
            'critical': 10.0,   # GB  
            'abort': 5.0       # GB
        }
        
        self.quick_mode_thresholds = {
            'warning': 4.0,    # GB
            'critical': 2.0,   # GB
            'abort': 1.0       # GB
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
        """Get memory status with threshold checking"""
        memory_info = self.get_memory_usage()
        available_gb = memory_info['available_gb']
        
        thresholds = self.quick_mode_thresholds if self.quick_mode else self.memory_thresholds
        
        if available_gb < thresholds['abort']:
            level = 'abort'
        elif available_gb < thresholds['critical']:
            level = 'critical'
        elif available_gb < thresholds['warning']:
            level = 'warning'
        else:
            level = 'normal'
        
        return {
            'level': level,
            'available_gb': available_gb,
            'used_gb': memory_info['used_gb'],
            'should_cleanup': level in ['warning', 'critical', 'abort']
        }
    
    def optimize_gpu_memory(self):
        """GPU memory optimization"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
        except Exception:
            pass

class CTRBiasCorrector:
    """Enhanced CTR bias correction for predictions"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.correction_factor = None
        self.scale_factor = 0.85  # Less aggressive than before (was 0.75)
        self.adaptive_correction = True
        self.correction_history = []
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate adaptive correction factor from training data"""
        actual_ctr = np.mean(y_true)
        predicted_ctr = np.mean(y_pred)
        
        if predicted_ctr > 0:
            # Base correction factor
            base_correction = self.target_ctr / predicted_ctr
            
            # Adaptive adjustment based on prediction quality
            prediction_std = np.std(y_pred)
            diversity_factor = min(1.2, max(0.8, prediction_std * 50))  # Reward diversity
            
            self.correction_factor = base_correction * diversity_factor
        else:
            self.correction_factor = 1.0
            
        logger.info(f"CTR bias correction factor: {self.correction_factor:.4f}")
        
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply enhanced CTR bias correction"""
        # Apply scale factor first
        corrected = y_pred * self.scale_factor
        
        # Apply correction factor if fitted
        if self.correction_factor is not None:
            corrected = corrected * self.correction_factor
        
        # Adaptive clipping based on target CTR
        lower_bound = self.target_ctr * 0.05  # 5% of target
        upper_bound = self.target_ctr * 4.0   # 400% of target (less restrictive)
        
        corrected = np.clip(corrected, lower_bound, upper_bound)
        
        return corrected

class EnhancedMultiMethodCalibrator:
    """Enhanced multi-method probability calibrator with advanced techniques"""
    
    def __init__(self):
        self.calibration_models = {}
        self.calibration_scores = {}
        self.best_method = None
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
        self.ctr_corrector = CTRBiasCorrector()
        self.scaler = None
        self.is_fitted = False
        self.ensemble_calibrator = None
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, method: str = 'auto') -> bool:
        """Fit enhanced calibration models with multiple methods"""
        try:
            if len(y_true) < 10:
                logger.warning("Insufficient data for calibration")
                self.is_fitted = False
                return False
            
            # Fit CTR bias correction first
            self.ctr_corrector.fit(y_true, y_pred_proba)
            
            methods_to_try = ['isotonic', 'platt', 'beta'] if method == 'auto' else [method]
            
            for cal_method in methods_to_try:
                try:
                    if cal_method == 'isotonic':
                        calibrator = IsotonicRegression(out_of_bounds='clip')
                        calibrator.fit(y_pred_proba, y_true)
                        self.calibration_models[cal_method] = calibrator
                        
                    elif cal_method == 'platt':
                        from sklearn.calibration import _SigmoidCalibration
                        calibrator = _SigmoidCalibration()
                        calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
                        self.calibration_models[cal_method] = calibrator
                    
                    elif cal_method == 'beta':
                        # Beta calibration for improved performance
                        calibrator = self._fit_beta_calibration(y_pred_proba, y_true)
                        if calibrator is not None:
                            self.calibration_models[cal_method] = calibrator
                    
                    # Calculate calibration score
                    calibrated_proba = self._predict_with_method(y_pred_proba, cal_method)
                    if calibrated_proba is not None:
                        score = -log_loss(y_true, calibrated_proba)
                        self.calibration_scores[cal_method] = score
                    
                except Exception as e:
                    logger.warning(f"Calibration method {cal_method} failed: {e}")
                    continue
            
            if self.calibration_scores:
                self.best_method = max(self.calibration_scores.keys(), 
                                     key=lambda x: self.calibration_scores[x])
                
                # Calculate additional bias correction
                best_calibrated = self._predict_with_method(y_pred_proba, self.best_method)
                if best_calibrated is not None:
                    self.bias_correction = np.mean(y_true) - np.mean(best_calibrated)
                    
                    if np.mean(best_calibrated) > 0:
                        self.multiplicative_correction = np.mean(y_true) / np.mean(best_calibrated)
                
                # Try ensemble calibration if multiple methods work
                if len(self.calibration_models) > 1:
                    self._fit_ensemble_calibrator(y_true, y_pred_proba)
                
                self.is_fitted = True
                logger.info(f"Best calibration method: {self.best_method}")
                return True
            
            self.is_fitted = False
            return False
            
        except Exception as e:
            logger.warning(f"Enhanced calibration fitting failed: {e}")
            self.is_fitted = False
            return False
    
    def _fit_beta_calibration(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
        """Fit beta calibration model"""
        try:
            from scipy.optimize import minimize_scalar
            
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
        """Fit ensemble calibrator using multiple methods"""
        try:
            # Get predictions from all methods
            method_predictions = {}
            for method in self.calibration_models.keys():
                pred = self._predict_with_method(y_pred_proba, method)
                if pred is not None:
                    method_predictions[method] = pred
            
            if len(method_predictions) < 2:
                return
            
            # Calculate weights based on performance
            weights = {}
            total_score = 0
            
            for method, pred in method_predictions.items():
                try:
                    score = -log_loss(y_true, pred)
                    weights[method] = max(0, score)  # Ensure non-negative
                    total_score += weights[method]
                except:
                    weights[method] = 0
            
            # Normalize weights
            if total_score > 0:
                for method in weights:
                    weights[method] /= total_score
                
                self.ensemble_calibrator = weights
                logger.info(f"Ensemble calibrator fitted with weights: {weights}")
            
        except Exception as e:
            logger.warning(f"Ensemble calibrator fitting failed: {e}")
    
    def _predict_with_method(self, y_pred_proba: np.ndarray, method: str) -> Optional[np.ndarray]:
        """Predict with specific calibration method"""
        try:
            if method not in self.calibration_models:
                return None
                
            calibrator = self.calibration_models[method]
            
            if method == 'isotonic':
                return calibrator.predict(y_pred_proba)
            elif method == 'platt':
                return calibrator.predict(y_pred_proba.reshape(-1, 1)).flatten()
            elif method == 'beta':
                param = calibrator['param']
                result = np.power(y_pred_proba, param)
                return np.clip(result, 1e-15, 1-1e-15)
            else:
                return y_pred_proba
        except Exception:
            return None
    
    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply enhanced calibration and CTR correction to predictions"""
        try:
            # Try ensemble calibration first
            if self.ensemble_calibrator and len(self.ensemble_calibrator) > 1:
                ensemble_pred = np.zeros_like(y_pred_proba)
                total_weight = 0
                
                for method, weight in self.ensemble_calibrator.items():
                    method_pred = self._predict_with_method(y_pred_proba, method)
                    if method_pred is not None:
                        ensemble_pred += weight * method_pred
                        total_weight += weight
                
                if total_weight > 0:
                    calibrated = ensemble_pred / total_weight
                else:
                    calibrated = y_pred_proba
            
            # Fallback to best single method
            elif self.is_fitted and self.best_method and self.best_method in self.calibration_models:
                calibrated = self._predict_with_method(y_pred_proba, self.best_method)
                if calibrated is None:
                    calibrated = y_pred_proba
            else:
                calibrated = y_pred_proba
            
            # Apply CTR bias correction
            corrected = self.ctr_corrector.transform(calibrated)
            
            return np.clip(corrected, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.warning(f"Enhanced calibration prediction failed: {e}")
            # Still apply CTR correction even if calibration fails
            return self.ctr_corrector.transform(y_pred_proba)
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get enhanced calibration summary"""
        return {
            'best_method': self.best_method,
            'calibration_scores': self.calibration_scores,
            'available_methods': list(self.calibration_models.keys()),
            'ensemble_weights': self.ensemble_calibrator,
            'bias_correction': self.bias_correction,
            'multiplicative_correction': self.multiplicative_correction,
            'ctr_correction_factor': self.ctr_corrector.correction_factor,
            'is_fitted': self.is_fitted
        }

class BaseModel(ABC):
    """Base class for all models with enhanced calibration"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.calibrator = None
        self.is_calibrated = False
        self.prediction_diversity_threshold = 2000  # Reduced for better diversity
        self.calibration_applied = True
        self.memory_monitor = MemoryMonitor()
        self.quick_mode = False
        self.scaler = StandardScaler()
        self.use_scaling = True
        self.training_time = 0.0
        self.validation_score = 0.0
        
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
            
            if memory_status['level'] == 'abort':
                logger.error(f"{self.name}: Insufficient memory for training")
                return None
            elif memory_status['level'] == 'critical':
                self._simplify_for_memory()
                logger.warning(f"{self.name}: Memory critical - simplified parameters applied")
            
            return fit_function(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"{self.name}: Memory safe fitting failed: {e}")
            return None
    
    def _memory_safe_predict(self, predict_function, X: pd.DataFrame, batch_size: int = 10000):
        """Memory safe prediction with batching"""
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
        """Safe data preprocessing with scaling"""
        try:
            X_processed = X.copy()
            
            # Handle missing values
            numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numeric_columns] = X_processed[numeric_columns].fillna(0)
            
            # Handle categorical columns
            categorical_columns = X_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                X_processed[col] = X_processed[col].fillna('missing')
            
            # Handle infinite values
            X_processed = X_processed.replace([np.inf, -np.inf], 0)
            
            # Apply scaling if enabled
            if self.use_scaling and len(numeric_columns) > 0:
                if fit_scaler:
                    X_processed[numeric_columns] = self.scaler.fit_transform(X_processed[numeric_columns])
                else:
                    X_processed[numeric_columns] = self.scaler.transform(X_processed[numeric_columns])
            
            return X_processed
            
        except Exception as e:
            logger.warning(f"{self.name}: Data preprocessing failed: {e}")
            return X
    
    def _ensure_feature_consistency(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency with training data"""
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
        """Enhanced prediction diversity to prevent convergence"""
        try:
            unique_predictions = len(np.unique(predictions))
            
            if unique_predictions < self.prediction_diversity_threshold:
                # More sophisticated noise injection
                base_noise_scale = max(np.std(predictions) * 0.002, 1e-6)
                
                # Adaptive noise based on prediction range
                pred_range = np.max(predictions) - np.min(predictions)
                range_factor = max(0.5, min(2.0, pred_range * 100))
                
                noise_scale = base_noise_scale * range_factor
                
                # Use different noise distributions
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
        """Apply enhanced probability calibration"""
        try:
            if not self.is_fitted:
                logger.warning("Model not fitted, cannot apply calibration")
                return False
            
            # Skip calibration in quick mode if dataset is too small
            if self.quick_mode and len(X_val) < 10:
                logger.info("Quick mode: Skipping calibration for small dataset")
                return False
            
            logger.info(f"{self.name}: Starting enhanced calibration")
            raw_predictions = self.predict_proba_raw(X_val)
            
            self.calibrator = EnhancedMultiMethodCalibrator()
            success = self.calibrator.fit(y_val.values, raw_predictions, method)
            
            if success:
                self.is_calibrated = True
                logger.info(f"{self.name}: Enhanced calibration applied successfully")
                return True
            else:
                # Even if calibration fails, still apply CTR correction
                self.calibrator = EnhancedMultiMethodCalibrator()
                self.calibrator.ctr_corrector.fit(y_val.values, raw_predictions)
                self.is_calibrated = True
                logger.warning(f"{self.name}: Calibration fitting failed, using CTR correction only")
                return True
                
        except Exception as e:
            logger.warning(f"Enhanced calibration application failed: {e}")
            # Create basic CTR corrector as fallback
            try:
                self.calibrator = EnhancedMultiMethodCalibrator()
                self.calibrator.ctr_corrector.fit(y_val.values, raw_predictions)
                self.is_calibrated = True
                return True
            except:
                return False
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Enhanced calibrated probability predictions with CTR correction"""
        raw_predictions = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            return self.calibrator.predict_proba(raw_predictions)
        else:
            # Apply basic CTR correction even without calibration
            corrector = CTRBiasCorrector()
            return corrector.transform(raw_predictions)
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw probability predictions before calibration"""
        pass

class LogisticModel(BaseModel):
    """Enhanced Logistic Regression model with optimized parameters"""
    
    def __init__(self, name: str = "LogisticRegression", params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not installed.")
        
        default_params = {
            'C': 0.5,  # Increased from 0.3 for less regularization
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 4000,  # Increased from 3000
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': 8,  # Increased from 6
            'tol': 0.00005  # Tighter tolerance
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 2500  # Reduced for better diversity
        self.sampling_logged = False
        self.use_scaling = True
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        simplified_params = {
            'C': 0.5,
            'max_iter': 2000,
            'n_jobs': 4,
            'tol': 0.0001
        }
        
        self.params.update(simplified_params)
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def _apply_quick_mode_params(self):
        """Apply quick mode parameters for rapid testing"""
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
        """Safe stratified sampling with bounds checking"""
        try:
            current_size = len(X_train)
            
            if current_size <= target_size:
                return X_train, y_train
            
            from sklearn.model_selection import train_test_split
            
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
        """Enhanced logistic regression model training with scaling and calibration"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            if self.quick_mode:
                self._apply_quick_mode_params()
            
            # Memory check and potential sampling
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['level'] in ['critical', 'warning'] and len(X_train) > 80000:  # Increased threshold
                target_size = 50000 if memory_status['level'] == 'warning' else 25000
                X_train_sample, y_train_sample = self._safe_sampling(X_train, y_train, target_size)
            else:
                X_train_sample, y_train_sample = X_train, y_train
            
            # Safe data preprocessing with scaling
            X_train_clean = self._safe_data_preprocessing(X_train_sample, fit_scaler=True)
            
            # Fit model
            logger.info(f"{self.name}: Starting training with enhanced parameters")
            self.model.fit(X_train_clean, y_train_sample)
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            # Calculate validation score
            if X_val is not None and y_val is not None and len(X_val) > 0:
                try:
                    val_pred = self.predict_proba_raw(X_val)
                    from sklearn.metrics import roc_auc_score
                    self.validation_score = roc_auc_score(y_val, val_pred)
                except:
                    self.validation_score = 0.5
            
            # Enhanced calibration if validation data available
            if X_val is not None and y_val is not None and len(X_val) > 0:
                calibration_success = self.apply_calibration(X_val, y_val, method='auto')
                if calibration_success:
                    logger.info(f"{self.name}: Enhanced calibration completed successfully")
                else:
                    logger.warning(f"{self.name}: Calibration failed - basic CTR correction applied")
            else:
                logger.warning(f"{self.name}: No validation data - calibration skipped")
            
            # Record training time
            self.training_time = time.time() - start_time
            
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
            X_processed = self._safe_data_preprocessing(X_processed, fit_scaler=False)
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=50000)  # Increased batch size

class LightGBMModel(BaseModel):
    """Enhanced LightGBM model with optimized parameters"""
    
    def __init__(self, name: str = "LightGBM", params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # Increased from 31
            'learning_rate': 0.06,  # Increased from 0.05
            'feature_fraction': 0.85,  # Increased from 0.8
            'bagging_fraction': 0.85,  # Increased from 0.8
            'bagging_freq': 5,
            'min_data_in_leaf': 200,  # Increased from 150
            'lambda_l1': 0.4,  # Reduced from 0.3
            'lambda_l2': 0.4,  # Reduced from 0.3
            'min_gain_to_split': 0.015,  # Increased slightly from 0.01
            'max_depth': 8,  # Increased from 6
            'save_binary': True,
            'seed': 42,
            'feature_fraction_seed': 42,
            'bagging_seed': 42,
            'drop_seed': 42,
            'data_random_seed': 42,
            'verbose': -1,
            'n_estimators': 800,  # Increased from 600
            'early_stopping_rounds': 120  # Increased from 100
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.prediction_diversity_threshold = 2200  # Reduced for better diversity
        self.use_scaling = False
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        simplified_params = {
            'num_leaves': 31,
            'max_depth': 6,
            'n_estimators': 500,
            'min_data_in_leaf': 250,
            'num_threads': 6,
            'max_bin': 128,
            'early_stopping_rounds': 80,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75
        }
        
        self.params.update(simplified_params)
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def _apply_quick_mode_params(self):
        """Apply quick mode parameters for rapid testing"""
        quick_params = {
            'num_leaves': 31,
            'max_depth': 6,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'min_data_in_leaf': 20,
            'num_threads': 4,
            'max_bin': 128,
            'early_stopping_rounds': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'verbosity': -1
        }
        
        self.params.update(quick_params)
        logger.info(f"{self.name}: Quick mode parameters applied")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Enhanced LightGBM model training with optimized parameters"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            if self.quick_mode:
                self._apply_quick_mode_params()
            
            # Safe data preprocessing
            X_train_clean = self._safe_data_preprocessing(X_train)
            
            # Memory efficient dataset creation
            train_params = {
                'max_bin': self.params.get('max_bin', 255),  # Increased from 128
                'verbosity': -1
            }
            
            train_data = lgb.Dataset(
                X_train_clean, 
                label=y_train, 
                free_raw_data=True,
                params=train_params
            )
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            X_val_clean = None
            if X_val is not None and y_val is not None:
                X_val_clean = self._safe_data_preprocessing(X_val)
                
                valid_data = lgb.Dataset(
                    X_val_clean, 
                    label=y_val, 
                    reference=train_data, 
                    free_raw_data=True,
                    params=train_params
                )
                valid_sets.append(valid_data)
                valid_names.append('valid')
            
            # Extract callback parameters from main params
            early_stopping = self.params.pop('early_stopping_rounds', 120)
            n_estimators = self.params.pop('n_estimators', 800)
            
            # Train with enhanced parameters
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(early_stopping), 
                    lgb.log_evaluation(0)
                ]
            )
            
            self.is_fitted = True
            
            # Calculate validation score
            if X_val_clean is not None and y_val is not None and len(y_val) > 0:
                try:
                    val_pred = self.predict_proba_raw(pd.DataFrame(X_val_clean, columns=self.feature_names))
                    from sklearn.metrics import roc_auc_score
                    self.validation_score = roc_auc_score(y_val, val_pred)
                except:
                    self.validation_score = 0.5
            
            # Enhanced calibration
            if X_val_clean is not None and y_val is not None and len(y_val) > 0:
                logger.info(f"{self.name}: Starting enhanced calibration")
                calibration_success = self.apply_calibration(pd.DataFrame(X_val_clean, columns=self.feature_names), y_val, method='auto')
                if calibration_success:
                    logger.info(f"{self.name}: Enhanced calibration completed successfully")
                else:
                    logger.warning(f"{self.name}: Calibration failed - basic CTR correction applied")
            else:
                logger.warning(f"{self.name}: No validation data - calibration skipped")
            
            # Record training time
            self.training_time = time.time() - start_time
            
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
            X_processed = self._safe_data_preprocessing(X_processed)
            
            num_iteration = getattr(self.model, 'best_iteration', None)
            proba = self.model.predict(X_processed, num_iteration=num_iteration)
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=40000)  # Increased batch size

class XGBoostModel(BaseModel):
    """Enhanced XGBoost model with optimized parameters"""
    
    def __init__(self, name: str = "XGBoost", params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 8,  # Increased from 6
            'learning_rate': 0.06,  # Increased from 0.05
            'n_estimators': 700,  # Increased from 500
            'subsample': 0.85,  # Increased from 0.8
            'colsample_bytree': 0.85,  # Increased from 0.8
            'colsample_bylevel': 0.85,  # Increased from 0.8
            'colsample_bynode': 0.85,  # Increased from 0.8
            'reg_alpha': 0.4,  # Reduced from 0.3
            'reg_lambda': 0.4,  # Reduced from 0.3
            'min_child_weight': 6,  # Reduced from 8
            'gamma': 0.03,  # Reduced from 0.05
            'scale_pos_weight': 52.3,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 120  # Increased from 100
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.prediction_diversity_threshold = 2000  # Reduced for better diversity
        self.use_scaling = False
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        simplified_params = {
            'max_depth': 6,
            'n_estimators': 400,
            'learning_rate': 0.1,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'early_stopping_rounds': 80,
            'n_jobs': 6
        }
        
        self.params.update(simplified_params)
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def _apply_quick_mode_params(self):
        """Apply quick mode parameters for rapid testing"""
        quick_params = {
            'max_depth': 6,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 20,
            'n_jobs': 4
        }
        
        self.params.update(quick_params)
        logger.info(f"{self.name}: Quick mode parameters applied")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Enhanced XGBoost model training with optimized parameters"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            if self.quick_mode:
                self._apply_quick_mode_params()
            
            # Safe data preprocessing
            X_train_clean = self._safe_data_preprocessing(X_train)
            
            # Create DMatrix for memory efficiency
            dtrain = xgb.DMatrix(
                X_train_clean, 
                label=y_train, 
                feature_names=list(X_train_clean.columns),
                enable_categorical=False
            )
            
            eval_set = [(dtrain, 'train')]
            dval = None
            
            if X_val is not None and y_val is not None:
                X_val_clean = self._safe_data_preprocessing(X_val)
                dval = xgb.DMatrix(
                    X_val_clean, 
                    label=y_val, 
                    feature_names=list(X_val_clean.columns),
                    enable_categorical=False
                )
                eval_set.append((dval, 'eval'))
            
            early_stopping = self.params.pop('early_stopping_rounds', 120)
            
            logger.info(f"{self.name}: Starting XGBoost training with enhanced parameters")
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get('n_estimators', 700),
                evals=eval_set,
                early_stopping_rounds=early_stopping,
                verbose_eval=0
            )
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            # Calculate validation score
            if dval is not None:
                try:
                    val_pred = self.model.predict(dval)
                    from sklearn.metrics import roc_auc_score
                    self.validation_score = roc_auc_score(y_val, val_pred)
                except:
                    self.validation_score = 0.5
            
            # Enhanced calibration
            if dval is not None:
                logger.info(f"{self.name}: Starting enhanced calibration")
                X_val_clean = self._safe_data_preprocessing(X_val)
                
                calibration_success = self.apply_calibration(pd.DataFrame(X_val_clean, columns=self.feature_names), y_val, method='auto')
                if calibration_success:
                    logger.info(f"{self.name}: Enhanced calibration completed successfully")
                else:
                    logger.warning(f"{self.name}: Calibration failed - basic CTR correction applied")
            else:
                logger.warning(f"{self.name}: No validation data - calibration skipped")
            
            # Record training time
            self.training_time = time.time() - start_time
            
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
            X_processed = self._safe_data_preprocessing(X_processed)
            
            dtest = xgb.DMatrix(
                X_processed, 
                feature_names=list(X_processed.columns),
                enable_categorical=False
            )
            proba = self.model.predict(dtest)
            
            del dtest
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=25000)  # Increased batch size

class ModelFactory:
    """Enhanced model factory for creating models with improved capabilities"""
    
    _factory_logged = False
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create model by type with enhanced calibration support"""
        try:
            if not ModelFactory._factory_logged:
                memory_monitor = MemoryMonitor()
                logger.info(f"Creating enhanced model: {model_type} (improved calibration enabled)")
                
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
                model = LightGBMModel(params=kwargs.get('params'))
                
            elif model_type.lower() == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost is not installed.")
                model = XGBoostModel(params=kwargs.get('params'))
                
            elif model_type.lower() == 'logistic':
                model = LogisticModel(params=kwargs.get('params'))
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if quick_mode:
                model.set_quick_mode(True)
            
            logger.info(f"{model_type} enhanced model creation complete - improved calibration enabled")
            return model
                
        except Exception as e:
            logger.error(f"Enhanced model creation failed ({model_type}): {e}")
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
        
        logger.info(f"Available enhanced models: {available} (all with improved calibration)")
        return available
    
    @staticmethod
    def get_model_priority() -> List[str]:
        """Model priority list for enhanced performance"""
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

# Enhanced model aliases for consistency
FinalLightGBMModel = LightGBMModel
FinalXGBoostModel = XGBoostModel  
FinalLogisticModel = LogisticModel