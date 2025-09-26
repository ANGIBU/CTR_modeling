# models.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import warnings
import pickle
import joblib
from pathlib import Path
from abc import ABC, abstractmethod
import time
import gc
import os
import sys

warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import clone
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
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRBiasCorrector:
    """CTR bias correction for better prediction alignment"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.base_scale_factor = 0.3  # Base reduction factor for over-prediction
        self.correction_factor = None
        self.dynamic_factor = None
        self.is_fitted = False
        self.bias_threshold = 0.002  # Threshold for applying correction
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit correction parameters with dynamic adjustment"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred)
            bias = predicted_ctr - actual_ctr
            
            logger.info(f"CTR Bias Corrector fitting - Actual: {actual_ctr:.4f}, Predicted: {predicted_ctr:.4f}, Bias: {bias:.4f}")
            
            if predicted_ctr > 0:
                # Primary correction factor to align with target CTR
                self.correction_factor = self.target_ctr / predicted_ctr
                
                # Dynamic factor based on bias severity
                if abs(bias) > self.bias_threshold:
                    # Apply more aggressive correction for larger bias
                    bias_severity = min(abs(bias) / 0.01, 5.0)  # Cap at 5x
                    self.dynamic_factor = 1.0 / (1.0 + bias_severity * 0.2)
                    
                    # Additional scaling for severe over-prediction
                    if predicted_ctr > actual_ctr * 1.5:
                        over_prediction_ratio = predicted_ctr / actual_ctr
                        self.dynamic_factor *= (1.0 / over_prediction_ratio) * 1.2
                else:
                    self.dynamic_factor = 1.0
                    
            else:
                self.correction_factor = 1.0
                self.dynamic_factor = 1.0
                
            self.is_fitted = True
            
            logger.info(f"CTR correction factors - Primary: {self.correction_factor:.4f}, Dynamic: {self.dynamic_factor:.4f}")
            
        except Exception as e:
            logger.warning(f"CTR bias correction fitting failed: {e}")
            self.correction_factor = 1.0
            self.dynamic_factor = 1.0
            self.is_fitted = True
        
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply CTR bias correction with dynamic scaling"""
        if not self.is_fitted:
            return y_pred
            
        try:
            # Apply base scaling to reduce over-prediction
            corrected = y_pred * self.base_scale_factor
            
            # Apply primary correction factor
            if self.correction_factor is not None:
                corrected = corrected * self.correction_factor
            
            # Apply dynamic factor for bias severity
            if self.dynamic_factor is not None:
                corrected = corrected * self.dynamic_factor
            
            # Additional target alignment
            current_mean = np.mean(corrected)
            if abs(current_mean - self.target_ctr) > 0.001:
                alignment_factor = self.target_ctr / current_mean if current_mean > 0 else 1.0
                corrected = corrected * alignment_factor
            
            # Clip to reasonable CTR range
            corrected = np.clip(corrected, 0.0001, 0.1)
            
            return corrected
        except Exception as e:
            logger.warning(f"CTR bias correction failed: {e}")
            return y_pred

class CTRMultiMethodCalibrator:
    """Multi-method probability calibrator with CTR correction"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.calibration_models = {}
        self.calibration_scores = {}
        self.best_method = None
        self.ctr_corrector = CTRBiasCorrector(target_ctr)
        self.is_fitted = False
        self.target_ctr = target_ctr
        self.validation_metrics = {}
        
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> bool:
        """Fit calibration models with CTR correction"""
        try:
            if len(y_true) < 20:
                logger.warning(f"Insufficient data for calibration: {len(y_true)} samples")
                self.is_fitted = False
                return False
            
            # Ensure we have both classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("Calibration requires both classes present")
                self.is_fitted = False
                return False
            
            # Fit CTR bias correction first
            self.ctr_corrector.fit(y_true, y_pred_proba)
            
            methods_to_try = ['isotonic', 'platt', 'beta', 'linear_ctr']
            
            for cal_method in methods_to_try:
                try:
                    if cal_method == 'isotonic' and SKLEARN_AVAILABLE:
                        calibrator = IsotonicRegression(out_of_bounds='clip')
                        calibrator.fit(y_pred_proba, y_true)
                        self.calibration_models[cal_method] = calibrator
                        
                    elif cal_method == 'platt' and SKLEARN_AVAILABLE:
                        # Platt scaling using logistic regression
                        calibrator = LogisticRegression()
                        calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
                        self.calibration_models[cal_method] = calibrator
                        
                    elif cal_method == 'beta':
                        # Beta calibration (simple parameter fitting)
                        calibrator = self._fit_beta_calibration(y_true, y_pred_proba)
                        if calibrator is not None:
                            self.calibration_models[cal_method] = calibrator
                        else:
                            continue
                            
                    elif cal_method == 'linear_ctr':
                        # Linear CTR-focused calibration
                        calibrator = self._fit_linear_ctr_calibration(y_true, y_pred_proba)
                        if calibrator is not None:
                            self.calibration_models[cal_method] = calibrator
                        else:
                            continue
                    
                    # Evaluate calibration method
                    calibrated_probs = self._predict_with_method(y_pred_proba, cal_method)
                    score = self._evaluate_calibration_quality(y_true, calibrated_probs)
                    self.calibration_scores[cal_method] = score
                    
                    # Store validation metrics
                    self.validation_metrics[cal_method] = {
                        'ctr_bias': np.mean(calibrated_probs) - np.mean(y_true),
                        'ctr_alignment': abs(np.mean(calibrated_probs) - self.target_ctr)
                    }
                    
                    logger.info(f"Calibration method {cal_method} score: {score:.4f}, CTR bias: {self.validation_metrics[cal_method]['ctr_bias']:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Calibration method {cal_method} failed: {e}")
                    continue
            
            # Select best method
            if self.calibration_scores:
                self.best_method = max(self.calibration_scores.items(), key=lambda x: x[1])[0]
                best_score = self.calibration_scores[self.best_method]
                best_metrics = self.validation_metrics[self.best_method]
                
                logger.info(f"Best calibration method: {self.best_method} (score: {best_score:.4f}, bias: {best_metrics['ctr_bias']:.4f})")
                self.is_fitted = True
                return True
            else:
                logger.warning("No calibration methods succeeded")
                self.is_fitted = False
                return False
                
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            self.is_fitted = False
            return False
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Optional[Dict[str, float]]:
        """Fit simple beta calibration parameters"""
        try:
            # Simple linear transformation: a * x + b
            from sklearn.linear_model import LinearRegression
            
            reg = LinearRegression()
            reg.fit(y_pred_proba.reshape(-1, 1), y_true)
            
            return {
                'a': float(reg.coef_[0]),
                'b': float(reg.intercept_)
            }
        except Exception:
            return None
    
    def _fit_linear_ctr_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Optional[Dict[str, float]]:
        """Fit linear CTR-focused calibration"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            
            # Simple scaling factor to align CTRs
            if predicted_ctr > 0:
                scale_factor = actual_ctr / predicted_ctr
                
                # Add bias term for better alignment
                bias_term = self.target_ctr - (predicted_ctr * scale_factor)
                
                return {
                    'scale': scale_factor,
                    'bias': bias_term,
                    'target_alignment_factor': self.target_ctr / max(actual_ctr, 0.001)
                }
            else:
                return None
        except Exception:
            return None
    
    def _predict_with_method(self, y_pred_proba: np.ndarray, method: str) -> np.ndarray:
        """Predict with specific calibration method"""
        try:
            if method not in self.calibration_models:
                return y_pred_proba
                
            calibrator = self.calibration_models[method]
            
            if method == 'isotonic':
                calibrated = calibrator.predict(y_pred_proba)
            elif method == 'platt':
                calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
            elif method == 'beta':
                calibrated = calibrator['a'] * y_pred_proba + calibrator['b']
                calibrated = np.clip(calibrated, 0, 1)
            elif method == 'linear_ctr':
                calibrated = y_pred_proba * calibrator['scale'] + calibrator['bias']
                # Apply target alignment
                current_mean = np.mean(calibrated)
                if current_mean > 0:
                    alignment_factor = calibrator['target_alignment_factor']
                    calibrated = calibrated * alignment_factor
                calibrated = np.clip(calibrated, 0, 1)
            else:
                calibrated = y_pred_proba
                
            return calibrated
            
        except Exception as e:
            logger.warning(f"Calibration prediction failed for {method}: {e}")
            return y_pred_proba
    
    def _evaluate_calibration_quality(self, y_true: np.ndarray, y_pred_calibrated: np.ndarray) -> float:
        """Evaluate calibration quality with CTR focus"""
        try:
            # CTR alignment score (most important)
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_calibrated)
            target_alignment = 1.0 - abs(predicted_ctr - self.target_ctr) / max(self.target_ctr, 0.01)
            target_alignment = max(0, target_alignment)
            
            # Actual vs predicted alignment
            actual_alignment = 1.0 - abs(actual_ctr - predicted_ctr) / max(actual_ctr, 0.01)
            actual_alignment = max(0, actual_alignment)
            
            # Brier score (reliability)
            brier_score = np.mean((y_pred_calibrated - y_true) ** 2)
            brier_quality = max(0, 1.0 - brier_score * 10)  # Normalize and invert
            
            # Variance penalty (avoid extreme values)
            pred_variance = np.var(y_pred_calibrated)
            variance_penalty = min(0.3, pred_variance * 20)
            
            # Combine scores with CTR focus
            combined_score = (0.4 * target_alignment + 
                            0.3 * actual_alignment + 
                            0.2 * brier_quality - 
                            0.1 * variance_penalty)
            
            return max(0, combined_score)
            
        except Exception as e:
            logger.warning(f"Calibration evaluation failed: {e}")
            return 0.0
    
    def predict(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Apply calibration and CTR correction"""
        if not self.is_fitted:
            return y_pred_proba
            
        try:
            # Apply calibration if available
            if self.best_method and self.best_method in self.calibration_models:
                calibrated = self._predict_with_method(y_pred_proba, self.best_method)
            else:
                calibrated = y_pred_proba
            
            # Apply CTR bias correction
            corrected = self.ctr_corrector.transform(calibrated)
            
            return np.clip(corrected, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.warning(f"Calibration prediction failed: {e}")
            # Still apply CTR correction
            return self.ctr_corrector.transform(y_pred_proba)
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary"""
        return {
            'best_method': self.best_method,
            'calibration_scores': self.calibration_scores,
            'validation_metrics': self.validation_metrics,
            'available_methods': list(self.calibration_models.keys()),
            'ctr_correction_factor': self.ctr_corrector.correction_factor,
            'dynamic_factor': self.ctr_corrector.dynamic_factor,
            'is_fitted': self.is_fitted,
            'target_ctr': self.target_ctr
        }

class BaseModel(ABC):
    """Base class for all models with calibration"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.calibrator = None
        self.is_calibrated = False
        self.memory_monitor = MemoryMonitor()
        self.quick_mode = False
        self.scaler = StandardScaler()
        self.use_scaling = True
        
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode"""
        self.quick_mode = enabled
        self.memory_monitor.set_quick_mode(enabled)
        if enabled:
            logger.info(f"{self.name}: Quick mode enabled - simplified parameters")
        else:
            logger.info(f"{self.name}: Full mode enabled - complete parameter set")
    
    def _memory_safe_fit(self, fit_function, *args, **kwargs):
        """Memory safe fitting"""
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            if memory_status['usage_percent'] > 0.9:
                logger.error(f"{self.name}: Insufficient memory for training")
                return None
            
            return fit_function(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"{self.name}: Memory safe fitting failed: {e}")
            return None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> bool:
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw probabilities without calibration"""
        return self.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary outcomes"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series) -> bool:
        """Calibrate the model predictions"""
        try:
            if not self.is_fitted:
                logger.warning(f"{self.name}: Cannot calibrate unfitted model")
                return False
            
            # Get validation predictions
            val_predictions = self.predict_proba_raw(X_val)
            
            # Initialize and fit calibrator
            self.calibrator = CTRMultiMethodCalibrator()
            calibration_success = self.calibrator.fit(y_val.values, val_predictions)
            
            if calibration_success:
                self.is_calibrated = True
                
                # Log calibration results
                summary = self.calibrator.get_calibration_summary()
                logger.info(f"{self.name}: Calibration successful - Method: {summary['best_method']}")
                logger.info(f"{self.name}: CTR correction - Primary: {summary.get('ctr_correction_factor', 0):.4f}, Dynamic: {summary.get('dynamic_factor', 0):.4f}")
                return True
            else:
                logger.warning(f"{self.name}: Calibration failed")
                return False
                
        except Exception as e:
            logger.error(f"{self.name}: Calibration error: {e}")
            return False
    
    def predict_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibration if available"""
        try:
            # Get raw predictions
            raw_proba = self.predict_proba_raw(X)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                return self.calibrator.predict(raw_proba)
            else:
                return raw_proba
                
        except Exception as e:
            logger.error(f"{self.name}: Calibrated prediction failed: {e}")
            return self.predict_proba_raw(X)
    
    def save_model(self, filepath: str) -> bool:
        """Save model to file"""
        try:
            model_data = {
                'name': self.name,
                'params': self.params,
                'model': self.model,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
                'calibrator': self.calibrator,
                'is_calibrated': self.is_calibrated,
                'scaler': self.scaler,
                'use_scaling': self.use_scaling
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"{self.name}: Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.name = model_data.get('name', self.name)
            self.params = model_data.get('params', {})
            self.model = model_data.get('model')
            self.is_fitted = model_data.get('is_fitted', False)
            self.feature_names = model_data.get('feature_names')
            self.calibrator = model_data.get('calibrator')
            self.is_calibrated = model_data.get('is_calibrated', False)
            self.scaler = model_data.get('scaler', StandardScaler())
            self.use_scaling = model_data.get('use_scaling', True)
            
            logger.info(f"{self.name}: Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Failed to load model: {e}")
            return False

class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for CTR prediction"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'random_state': 42,
            'max_iter': 1000
        }
        if params:
            default_params.update(params)
        
        super().__init__('logistic', default_params)
        self.use_scaling = True
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> bool:
        """Fit logistic regression model"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.error("Scikit-learn not available for LogisticRegression")
                return False
            
            logger.info(f"{self.name}: Starting training with {X.shape[0]} samples")
            
            # CTR-focused parameters
            if self.quick_mode:
                self.params.update({
                    'C': 0.01,  # More regularization for CTR
                    'max_iter': 500
                })
            else:
                # Parameters tuned for CTR prediction
                self.params.update({
                    'C': 0.001,  # Strong regularization to prevent over-fitting
                    'class_weight': 'balanced',  # Handle class imbalance
                    'max_iter': 3000,
                    'tol': 1e-6  # Higher precision
                })
            
            # Initialize and scale data
            X_scaled = X.copy()
            if self.use_scaling:
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X), 
                    columns=X.columns, 
                    index=X.index
                )
            
            # Initialize model
            self.model = LogisticRegression(**self.params)
            
            # Fit model
            fitted_model = self._memory_safe_fit(self.model.fit, X_scaled, y)
            if fitted_model is None:
                return False
            
            self.feature_names = list(X.columns)
            self.is_fitted = True
            
            # Calibrate if validation data provided
            if X_val is not None and y_val is not None and len(y_val) > 10:
                self.calibrate(X_val, y_val)
            
            logger.info(f"{self.name}: Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Training failed: {e}")
            return False
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw probabilities without calibration"""
        try:
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            
            X_scaled = X.copy()
            if self.use_scaling and hasattr(self.scaler, 'transform'):
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X), 
                    columns=X.columns, 
                    index=X.index
                )
            
            proba = self.model.predict_proba(X_scaled)[:, 1]
            return proba
            
        except Exception as e:
            logger.error(f"{self.name}: Raw prediction failed: {e}")
            return np.full(len(X), 0.02)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibration if available"""
        try:
            raw_proba = self.predict_proba_raw(X)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                return self.calibrator.predict(raw_proba)
            else:
                # Apply basic CTR correction even without calibration
                corrector = CTRBiasCorrector()
                if len(raw_proba) > 100:  # Only for reasonably sized datasets
                    # Use median as proxy for typical prediction
                    median_pred = np.median(raw_proba)
                    dummy_y = np.random.binomial(1, 0.019, len(raw_proba))  # Approximate target CTR
                    corrector.fit(dummy_y, raw_proba)
                    return corrector.transform(raw_proba)
                return raw_proba
            
        except Exception as e:
            logger.error(f"{self.name}: Prediction failed: {e}")
            return np.full(len(X), 0.02)

class LightGBMModel(BaseModel):
    """LightGBM model for CTR prediction"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100
        }
        if params:
            default_params.update(params)
        
        super().__init__('lightgbm', default_params)
        self.use_scaling = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> bool:
        """Fit LightGBM model"""
        try:
            if not LIGHTGBM_AVAILABLE:
                logger.error("LightGBM not available")
                return False
            
            logger.info(f"{self.name}: Starting training with {X.shape[0]} samples")
            
            # CTR-focused parameters
            if self.quick_mode:
                self.params.update({
                    'num_leaves': 15,
                    'n_estimators': 50,
                    'learning_rate': 0.1
                })
            else:
                # Parameters tuned for CTR prediction
                self.params.update({
                    'num_leaves': 127,
                    'n_estimators': 300,
                    'learning_rate': 0.01,  # Lower learning rate for stability
                    'min_data_in_leaf': 200,  # Higher min samples
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'max_depth': 10,
                    'class_weight': 'balanced'
                })
            
            # Prepare data
            train_data = lgb.Dataset(X, label=y)
            valid_sets = [train_data]
            
            if X_val is not None and y_val is not None:
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(valid_data)
                self.params['early_stopping_rounds'] = 100
            
            # Train model
            self.model = self._memory_safe_fit(
                lgb.train,
                self.params,
                train_data,
                valid_sets=valid_sets,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            if self.model is None:
                return False
            
            self.feature_names = list(X.columns)
            self.is_fitted = True
            
            # Calibrate if validation data provided
            if X_val is not None and y_val is not None and len(y_val) > 10:
                self.calibrate(X_val, y_val)
            
            logger.info(f"{self.name}: Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Training failed: {e}")
            return False
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw probabilities without calibration"""
        try:
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            
            proba = self.model.predict(X, num_iteration=self.model.best_iteration)
            return proba
            
        except Exception as e:
            logger.error(f"{self.name}: Raw prediction failed: {e}")
            return np.full(len(X), 0.02)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibration if available"""
        try:
            raw_proba = self.predict_proba_raw(X)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                return self.calibrator.predict(raw_proba)
            else:
                # Apply basic CTR correction
                corrector = CTRBiasCorrector()
                if len(raw_proba) > 100:
                    dummy_y = np.random.binomial(1, 0.019, len(raw_proba))
                    corrector.fit(dummy_y, raw_proba)
                    return corrector.transform(raw_proba)
                return raw_proba
            
        except Exception as e:
            logger.error(f"{self.name}: Prediction failed: {e}")
            return np.full(len(X), 0.02)

class XGBoostModel(BaseModel):
    """XGBoost model for CTR prediction"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        if params:
            default_params.update(params)
        
        super().__init__('xgboost', default_params)
        self.use_scaling = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> bool:
        """Fit XGBoost model"""
        try:
            if not XGBOOST_AVAILABLE:
                logger.error("XGBoost not available")
                return False
            
            logger.info(f"{self.name}: Starting training with {X.shape[0]} samples")
            
            # CTR-focused parameters
            if self.quick_mode:
                self.params.update({
                    'max_depth': 4,
                    'n_estimators': 50,
                    'learning_rate': 0.1
                })
            else:
                # Parameters tuned for CTR prediction
                self.params.update({
                    'max_depth': 12,
                    'n_estimators': 400,
                    'learning_rate': 0.005,  # Very low learning rate
                    'min_child_weight': 50,  # Higher min weight
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'gamma': 0.1,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'scale_pos_weight': len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1
                })
            
            # Initialize model
            self.model = xgb.XGBClassifier(**self.params)
            
            # Prepare evaluation set
            eval_set = [(X, y)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
                fit_params = {
                    'eval_set': eval_set,
                    'early_stopping_rounds': 100,
                    'verbose': False
                }
            else:
                fit_params = {}
            
            # Train model
            fitted_model = self._memory_safe_fit(self.model.fit, X, y, **fit_params)
            if fitted_model is None:
                return False
            
            self.feature_names = list(X.columns)
            self.is_fitted = True
            
            # Calibrate if validation data provided
            if X_val is not None and y_val is not None and len(y_val) > 10:
                self.calibrate(X_val, y_val)
            
            logger.info(f"{self.name}: Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Training failed: {e}")
            return False
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw probabilities without calibration"""
        try:
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            
            proba = self.model.predict_proba(X)[:, 1]
            return proba
            
        except Exception as e:
            logger.error(f"{self.name}: Raw prediction failed: {e}")
            return np.full(len(X), 0.02)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibration if available"""
        try:
            raw_proba = self.predict_proba_raw(X)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                return self.calibrator.predict(raw_proba)
            else:
                # Apply basic CTR correction
                corrector = CTRBiasCorrector()
                if len(raw_proba) > 100:
                    dummy_y = np.random.binomial(1, 0.019, len(raw_proba))
                    corrector.fit(dummy_y, raw_proba)
                    return corrector.transform(raw_proba)
                return raw_proba
            
        except Exception as e:
            logger.error(f"{self.name}: Prediction failed: {e}")
            return np.full(len(X), 0.02)

class ModelFactory:
    """Factory for creating CTR prediction models"""
    
    @staticmethod
    def create_model(model_name: str, params: Dict[str, Any] = None) -> Optional[BaseModel]:
        """Create model by name"""
        try:
            model_map = {
                'logistic': LogisticRegressionModel,
                'lightgbm': LightGBMModel,
                'xgboost': XGBoostModel
            }
            
            if model_name not in model_map:
                logger.error(f"Unknown model name: {model_name}")
                return None
            
            model_class = model_map[model_name]
            return model_class(params)
            
        except Exception as e:
            logger.error(f"Model creation failed for {model_name}: {e}")
            return None
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models"""
        available = []
        
        if SKLEARN_AVAILABLE:
            available.append('logistic')
        
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        
        return available
    
    @staticmethod
    def get_model_requirements() -> Dict[str, List[str]]:
        """Get model requirements"""
        return {
            'logistic': ['scikit-learn'],
            'lightgbm': ['lightgbm'],
            'xgboost': ['xgboost']
        }

class CTRModelEvaluator:
    """Evaluate CTR model performance"""
    
    def __init__(self):
        self.target_ctr = 0.0191
        
    def evaluate_model(self, model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        try:
            if not model.is_fitted:
                logger.warning(f"Cannot evaluate unfitted model: {model.name}")
                return {}
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            evaluation_results = {}
            
            # Basic classification metrics
            if METRICS_AVAILABLE:
                try:
                    evaluation_results['auc'] = roc_auc_score(y_test, y_pred_proba)
                except:
                    evaluation_results['auc'] = 0.5
                
                try:
                    evaluation_results['logloss'] = log_loss(y_test, y_pred_proba)
                except:
                    evaluation_results['logloss'] = 1.0
                
                try:
                    evaluation_results['brier_score'] = brier_score_loss(y_test, y_pred_proba)
                except:
                    evaluation_results['brier_score'] = 0.25
            
            # CTR-specific metrics
            actual_ctr = np.mean(y_test)
            predicted_ctr = np.mean(y_pred_proba)
            
            evaluation_results.update({
                'actual_ctr': actual_ctr,
                'predicted_ctr': predicted_ctr,
                'ctr_bias': predicted_ctr - actual_ctr,
                'ctr_absolute_error': abs(predicted_ctr - actual_ctr),
                'ctr_relative_error': abs(predicted_ctr - actual_ctr) / max(actual_ctr, 0.001),
                'target_alignment': abs(predicted_ctr - self.target_ctr)
            })
            
            # Accuracy metrics
            evaluation_results.update({
                'accuracy': np.mean(y_pred == y_test),
                'precision': np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1),
                'recall': np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
            })
            
            # Combined score for CTR prediction with bias penalty
            auc_score = evaluation_results.get('auc', 0.5)
            ctr_alignment = max(0, 1 - evaluation_results['target_alignment'] / 0.02)
            logloss_score = max(0, 1 - evaluation_results.get('logloss', 1.0) / 2.0)
            bias_penalty = min(0.5, abs(evaluation_results['ctr_bias']) * 10)  # Heavy penalty for bias
            
            evaluation_results['combined_score'] = (
                0.3 * auc_score + 
                0.5 * ctr_alignment + 
                0.2 * logloss_score - 
                bias_penalty  # Subtract bias penalty
            )
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}

class CTRTrainer:
    """CTR model trainer with calibration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.evaluator = CTRModelEvaluator()
        self.trained_models = {}
        self.model_performance = {}
        self.quick_mode = False
        
    def set_quick_mode(self, enabled: bool):
        """Enable quick mode"""
        self.quick_mode = enabled
        self.memory_monitor.set_quick_mode(enabled)
        
    def get_available_models(self) -> List[str]:
        """Get available models"""
        return ModelFactory.get_available_models()
    
    def train_model(self, 
                   model_name: str, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None,
                   quick_mode: Optional[bool] = None) -> Tuple[Optional[BaseModel], Dict[str, Any]]:
        """Train single model"""
        try:
            if quick_mode is not None:
                self.set_quick_mode(quick_mode)
            
            logger.info(f"Training {model_name} model with CTR correction")
            
            # Create model
            model = ModelFactory.create_model(model_name)
            if model is None:
                return None, {}
            
            model.set_quick_mode(self.quick_mode)
            
            # Train model
            success = model.fit(X_train, y_train, X_val, y_val)
            if not success:
                logger.error(f"{model_name} training failed")
                return None, {}
            
            # Evaluate model
            performance = {}
            if X_val is not None and y_val is not None:
                performance = self.evaluator.evaluate_model(model, X_val, y_val)
            
            # Store results
            self.trained_models[model_name] = model
            self.model_performance[model_name] = performance
            
            # Log detailed performance
            logger.info(f"{model_name} training completed successfully")
            logger.info(f"Performance: AUC={performance.get('auc', 0):.3f}, "
                       f"CTR Bias={performance.get('ctr_bias', 0):.4f}, "
                       f"Combined Score={performance.get('combined_score', 0):.3f}")
            
            if model.is_calibrated:
                summary = model.calibrator.get_calibration_summary()
                logger.info(f"Calibration: Method={summary['best_method']}, "
                           f"CTR Factor={summary.get('ctr_correction_factor', 0):.3f}")
            
            return model, performance
            
        except Exception as e:
            logger.error(f"{model_name} training failed: {e}")
            return None, {}
    
    def train_all_models(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None) -> Dict[str, BaseModel]:
        """Train all available models"""
        try:
            available_models = self.get_available_models()
            logger.info(f"Training {len(available_models)} models: {available_models}")
            
            trained_models = {}
            
            for model_name in available_models:
                model, performance = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                if model is not None:
                    trained_models[model_name] = model
                
                # Memory cleanup between models
                gc.collect()
            
            logger.info(f"Training completed. Successful models: {list(trained_models.keys())}")
            return trained_models
            
        except Exception as e:
            logger.error(f"Batch model training failed: {e}")
            return {}
    
    def get_best_model(self) -> Optional[BaseModel]:
        """Get best performing model"""
        try:
            if not self.model_performance:
                return None
            
            best_model_name = max(
                self.model_performance.items(),
                key=lambda x: x[1].get('combined_score', 0)
            )[0]
            
            return self.trained_models.get(best_model_name)
            
        except Exception as e:
            logger.error(f"Best model selection failed: {e}")
            return None
    
    def save_models(self, model_dir: str) -> bool:
        """Save all trained models"""
        try:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            for model_name, model in self.trained_models.items():
                filepath = model_dir / f"{model_name}_model.pkl"
                if model.save_model(str(filepath)):
                    saved_count += 1
            
            # Save performance data
            performance_file = model_dir / "model_performance.pkl"
            with open(performance_file, 'wb') as f:
                pickle.dump(self.model_performance, f)
            
            logger.info(f"Saved {saved_count}/{len(self.trained_models)} models to {model_dir}")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            return False
    
    def load_models(self, model_dir: str) -> bool:
        """Load trained models"""
        try:
            model_dir = Path(model_dir)
            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                return False
            
            loaded_count = 0
            available_models = self.get_available_models()
            
            for model_name in available_models:
                filepath = model_dir / f"{model_name}_model.pkl"
                if filepath.exists():
                    model = ModelFactory.create_model(model_name)
                    if model and model.load_model(str(filepath)):
                        self.trained_models[model_name] = model
                        loaded_count += 1
            
            # Load performance data
            performance_file = model_dir / "model_performance.pkl"
            if performance_file.exists():
                with open(performance_file, 'rb') as f:
                    self.model_performance = pickle.load(f)
            
            logger.info(f"Loaded {loaded_count} models from {model_dir}")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        calibrated_models = sum(
            1 for model in self.trained_models.values() 
            if hasattr(model, 'is_calibrated') and model.is_calibrated
        )
        
        avg_performance = {}
        if self.model_performance:
            for metric in ['auc', 'combined_score', 'ctr_absolute_error']:
                values = [
                    perf.get(metric, 0) for perf in self.model_performance.values() 
                    if isinstance(perf, dict) and metric in perf
                ]
                avg_performance[f'avg_{metric}'] = np.mean(values) if values else 0.0
        
        return {
            'total_models_trained': len(self.trained_models),
            'calibrated_models': calibrated_models,
            'calibration_rate': calibrated_models / max(len(self.trained_models), 1),
            'model_performance': self.model_performance,
            'average_performance': avg_performance,
            'quick_mode': self.quick_mode,
            'training_completed': True
        }