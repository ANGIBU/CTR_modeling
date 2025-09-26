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
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRBiasCorrector:
    """CTR bias correction with multiple strategies"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.correction_strategies = {}
        self.best_strategy = None
        self.is_fitted = False
        
        # Correction parameters
        self.bias_threshold = 0.001
        self.severe_bias_threshold = 0.005
        self.max_correction_factor = 0.8
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit multiple correction strategies and select best"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred)
            bias = predicted_ctr - actual_ctr
            
            logger.info(f"CTR Bias Corrector fitting - Actual: {actual_ctr:.4f}, "
                       f"Predicted: {predicted_ctr:.4f}, Bias: {bias:.4f}")
            
            if predicted_ctr <= 0:
                self.correction_strategies['identity'] = {'factor': 1.0, 'offset': 0.0}
                self.best_strategy = 'identity'
                self.is_fitted = True
                return
            
            # Strategy 1: Simple scaling to target CTR
            target_scale_factor = self.target_ctr / predicted_ctr
            self.correction_strategies['target_scale'] = {
                'factor': target_scale_factor,
                'offset': 0.0
            }
            
            # Strategy 2: Actual CTR alignment with target adjustment
            actual_scale_factor = actual_ctr / predicted_ctr
            target_offset = (self.target_ctr - actual_ctr) * 0.3  # Partial adjustment
            self.correction_strategies['actual_align'] = {
                'factor': actual_scale_factor,
                'offset': target_offset
            }
            
            # Strategy 3: Progressive correction based on bias severity
            if abs(bias) <= self.bias_threshold:
                # Small bias - minor adjustment
                progressive_factor = 0.95 + 0.05 * (self.bias_threshold - abs(bias)) / self.bias_threshold
                progressive_offset = (self.target_ctr - predicted_ctr) * 0.1
            elif abs(bias) <= self.severe_bias_threshold:
                # Moderate bias - stronger correction
                progressive_factor = 0.8 + 0.15 * (self.severe_bias_threshold - abs(bias)) / (self.severe_bias_threshold - self.bias_threshold)
                progressive_offset = (self.target_ctr - predicted_ctr) * 0.3
            else:
                # Severe bias - maximum correction
                progressive_factor = max(0.2, 0.8 - min((abs(bias) - self.severe_bias_threshold) * 10, 0.6))
                progressive_offset = (self.target_ctr - predicted_ctr) * 0.5
            
            self.correction_strategies['progressive'] = {
                'factor': progressive_factor,
                'offset': progressive_offset
            }
            
            # Strategy 4: Quantile-based correction
            if len(y_pred) > 100:
                q25 = np.percentile(y_pred, 25)
                q75 = np.percentile(y_pred, 75)
                median = np.median(y_pred)
                
                if median > 0:
                    quantile_factor = self.target_ctr / median
                    quantile_offset = 0.0
                    
                    # Adjust extreme values less aggressively
                    if quantile_factor < 0.5:
                        quantile_factor = max(0.5, quantile_factor)
                    elif quantile_factor > 2.0:
                        quantile_factor = min(2.0, quantile_factor)
                    
                    self.correction_strategies['quantile'] = {
                        'factor': quantile_factor,
                        'offset': quantile_offset
                    }
            
            # Evaluate strategies
            best_score = float('-inf')
            for strategy_name, params in self.correction_strategies.items():
                corrected = self._apply_correction(y_pred, params)
                score = self._evaluate_correction(y_true, corrected)
                
                if score > best_score:
                    best_score = score
                    self.best_strategy = strategy_name
            
            self.is_fitted = True
            
            logger.info(f"Best correction strategy: {self.best_strategy}")
            logger.info(f"Strategy parameters: {self.correction_strategies[self.best_strategy]}")
            
        except Exception as e:
            logger.warning(f"CTR bias correction fitting failed: {e}")
            self.correction_strategies['identity'] = {'factor': 1.0, 'offset': 0.0}
            self.best_strategy = 'identity'
            self.is_fitted = True
    
    def _apply_correction(self, y_pred: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply correction with given parameters"""
        try:
            factor = params.get('factor', 1.0)
            offset = params.get('offset', 0.0)
            
            corrected = y_pred * factor + offset
            corrected = np.clip(corrected, 1e-6, 0.999)
            
            return corrected
        except Exception as e:
            logger.warning(f"Correction application failed: {e}")
            return y_pred
    
    def _evaluate_correction(self, y_true: np.ndarray, y_corrected: np.ndarray) -> float:
        """Evaluate correction quality"""
        try:
            actual_ctr = np.mean(y_true)
            corrected_ctr = np.mean(y_corrected)
            
            # CTR alignment score
            target_alignment = 1.0 - abs(corrected_ctr - self.target_ctr) / max(self.target_ctr, 0.01)
            actual_alignment = 1.0 - abs(corrected_ctr - actual_ctr) / max(actual_ctr, 0.01)
            
            # Variance penalty (avoid extreme corrections)
            variance_penalty = min(0.3, np.var(y_corrected) * 50)
            
            # Combined score
            score = 0.6 * max(0, target_alignment) + 0.4 * max(0, actual_alignment) - variance_penalty
            
            return score
            
        except Exception as e:
            logger.warning(f"Correction evaluation failed: {e}")
            return 0.0
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply best correction strategy"""
        if not self.is_fitted or self.best_strategy is None:
            return y_pred
        
        try:
            params = self.correction_strategies[self.best_strategy]
            return self._apply_correction(y_pred, params)
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
        
        # Calibration quality thresholds
        self.quality_thresholds = {
            'excellent': 0.0002,
            'good': 0.0008,
            'fair': 0.002,
            'poor': float('inf')
        }
        
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
            
            methods_to_try = ['isotonic', 'platt', 'beta', 'linear_ctr', 'sigmoid']
            
            for cal_method in methods_to_try:
                try:
                    if cal_method == 'isotonic' and SKLEARN_AVAILABLE:
                        calibrator = IsotonicRegression(out_of_bounds='clip')
                        calibrator.fit(y_pred_proba, y_true)
                        self.calibration_models[cal_method] = calibrator
                        
                    elif cal_method == 'platt' and SKLEARN_AVAILABLE:
                        # Platt scaling with regularization
                        calibrator = LogisticRegression(C=1.0, random_state=42)
                        calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
                        self.calibration_models[cal_method] = calibrator
                        
                    elif cal_method == 'beta':
                        # Beta calibration
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
                    
                    elif cal_method == 'sigmoid':
                        # Sigmoid calibration
                        calibrator = self._fit_sigmoid_calibration(y_true, y_pred_proba)
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
                        'ctr_alignment': abs(np.mean(calibrated_probs) - self.target_ctr),
                        'brier_score': np.mean((calibrated_probs - y_true) ** 2)
                    }
                    
                    logger.info(f"Calibration method {cal_method} score: {score:.4f}, "
                               f"CTR bias: {self.validation_metrics[cal_method]['ctr_bias']:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Calibration method {cal_method} failed: {e}")
                    continue
            
            # Select best method
            if self.calibration_scores:
                self.best_method = max(self.calibration_scores.items(), key=lambda x: x[1])[0]
                best_score = self.calibration_scores[self.best_method]
                best_metrics = self.validation_metrics[self.best_method]
                
                logger.info(f"Best calibration method: {self.best_method} "
                           f"(score: {best_score:.4f}, bias: {best_metrics['ctr_bias']:.4f})")
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
        """Fit beta calibration parameters"""
        try:
            from sklearn.linear_model import LinearRegression
            
            reg = LinearRegression()
            reg.fit(y_pred_proba.reshape(-1, 1), y_true)
            
            # Add CTR alignment term
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            
            # Adjust intercept for better CTR alignment
            adjusted_intercept = reg.intercept_ + (self.target_ctr - predicted_ctr * reg.coef_[0]) * 0.3
            
            return {
                'a': float(reg.coef_[0]),
                'b': float(adjusted_intercept)
            }
        except Exception:
            return None
    
    def _fit_linear_ctr_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Optional[Dict[str, float]]:
        """Fit linear CTR-focused calibration"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            
            if predicted_ctr <= 0:
                return None
            
            # Primary scaling to actual CTR
            primary_scale = actual_ctr / predicted_ctr
            
            # Secondary adjustment towards target CTR
            target_adjustment = (self.target_ctr - actual_ctr) * 0.5
            
            # Bias correction
            bias_term = target_adjustment / predicted_ctr if predicted_ctr > 0 else 0
            
            return {
                'scale': primary_scale,
                'bias': bias_term,
                'target_weight': 0.3
            }
        except Exception:
            return None
    
    def _fit_sigmoid_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Optional[Dict[str, float]]:
        """Fit sigmoid calibration"""
        try:
            # Logit transformation
            y_pred_logit = np.log(y_pred_proba / (1 - y_pred_proba + 1e-15) + 1e-15)
            
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
            reg.fit(y_pred_logit.reshape(-1, 1), y_true)
            
            return {
                'slope': float(reg.coef_[0]),
                'intercept': float(reg.intercept_)
            }
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
                # Apply target adjustment
                current_ctr = np.mean(calibrated)
                if current_ctr > 0:
                    target_factor = (self.target_ctr * calibrator['target_weight'] + 
                                   current_ctr * (1 - calibrator['target_weight'])) / current_ctr
                    calibrated = calibrated * target_factor
                calibrated = np.clip(calibrated, 0, 1)
            elif method == 'sigmoid':
                y_pred_logit = np.log(y_pred_proba / (1 - y_pred_proba + 1e-15) + 1e-15)
                calibrated_logit = calibrator['slope'] * y_pred_logit + calibrator['intercept']
                calibrated = 1 / (1 + np.exp(-calibrated_logit))
                calibrated = np.clip(calibrated, 1e-6, 1-1e-6)
            else:
                calibrated = y_pred_proba
                
            return calibrated
            
        except Exception as e:
            logger.warning(f"Calibration prediction failed for {method}: {e}")
            return y_pred_proba
    
    def _evaluate_calibration_quality(self, y_true: np.ndarray, y_pred_calibrated: np.ndarray) -> float:
        """Evaluate calibration quality with CTR focus"""
        try:
            # CTR alignment (most important)
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_calibrated)
            
            target_alignment = max(0, 1 - abs(predicted_ctr - self.target_ctr) / max(self.target_ctr, 0.01))
            actual_alignment = max(0, 1 - abs(actual_ctr - predicted_ctr) / max(actual_ctr, 0.01))
            
            # Calibration quality (Brier score)
            brier_score = np.mean((y_pred_calibrated - y_true) ** 2)
            brier_quality = max(0, 1 - brier_score * 50)  # Scale appropriately
            
            # Prediction stability
            pred_variance = np.var(y_pred_calibrated)
            stability_penalty = min(0.3, pred_variance * 30)
            
            # Combined score with heavy CTR focus
            combined_score = (0.5 * target_alignment + 
                            0.3 * actual_alignment + 
                            0.2 * brier_quality - 
                            stability_penalty)
            
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
    
    def get_calibration_quality(self, y_pred_proba: np.ndarray) -> str:
        """Get calibration quality assessment"""
        try:
            if not self.is_fitted:
                return 'UNKNOWN'
            
            corrected = self.predict(y_pred_proba)
            predicted_ctr = np.mean(corrected)
            ctr_bias = abs(predicted_ctr - self.target_ctr)
            
            for quality, threshold in self.quality_thresholds.items():
                if ctr_bias <= threshold:
                    return quality.upper()
            
            return 'POOR'
            
        except Exception:
            return 'UNKNOWN'
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary"""
        return {
            'best_method': self.best_method,
            'calibration_scores': self.calibration_scores,
            'validation_metrics': self.validation_metrics,
            'available_methods': list(self.calibration_models.keys()),
            'ctr_correction_strategy': self.ctr_corrector.best_strategy if self.ctr_corrector.is_fitted else None,
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
        
        # CTR-specific parameters
        self.target_ctr = 0.0191
        self.ctr_weight_factor = None
        
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
    
    def _calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for CTR prediction"""
        try:
            pos_count = (y == 1).sum()
            neg_count = (y == 0).sum()
            total = len(y)
            
            if pos_count == 0 or neg_count == 0:
                return {0: 1.0, 1: 1.0}
            
            # Balance for CTR prediction
            pos_weight = total / (2.0 * pos_count)
            neg_weight = total / (2.0 * neg_count)
            
            # Adjust for extreme imbalance
            if pos_weight > 100:
                pos_weight = min(pos_weight, 50)  # Cap positive weight
                neg_weight = total / (pos_weight * pos_count) - pos_weight
            
            return {0: neg_weight, 1: pos_weight}
            
        except Exception as e:
            logger.warning(f"Class weight calculation failed: {e}")
            return {0: 1.0, 1: 1.0}
    
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
            self.calibrator = CTRMultiMethodCalibrator(self.target_ctr)
            calibration_success = self.calibrator.fit(y_val.values, val_predictions)
            
            if calibration_success:
                self.is_calibrated = True
                
                # Log calibration results
                summary = self.calibrator.get_calibration_summary()
                logger.info(f"{self.name}: Calibration successful - Method: {summary['best_method']}")
                
                # Test calibration quality
                test_corrected = self.calibrator.predict(val_predictions)
                quality = self.calibrator.get_calibration_quality(test_corrected)
                logger.info(f"{self.name}: Calibration quality: {quality}")
                
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
                'use_scaling': self.use_scaling,
                'target_ctr': self.target_ctr
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
            self.target_ctr = model_data.get('target_ctr', 0.0191)
            
            logger.info(f"{self.name}: Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Failed to load model: {e}")
            return False

class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for CTR prediction"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'C': 0.01,  # Strong regularization for CTR
            'penalty': 'l2',
            'solver': 'liblinear',
            'random_state': 42,
            'max_iter': 2000,
            'class_weight': 'balanced'
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
                    'C': 0.1,
                    'max_iter': 1000
                })
            else:
                # Calculate class weights
                class_weights = self._calculate_class_weights(y)
                
                self.params.update({
                    'C': 0.005,  # Even stronger regularization
                    'class_weight': class_weights,
                    'max_iter': 3000,
                    'tol': 1e-6
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
            return np.full(len(X), self.target_ctr)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibration if available"""
        try:
            raw_proba = self.predict_proba_raw(X)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                return self.calibrator.predict(raw_proba)
            else:
                return raw_proba
            
        except Exception as e:
            logger.error(f"{self.name}: Prediction failed: {e}")
            return np.full(len(X), self.target_ctr)

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
                # Calculate class weights
                pos_count = (y == 1).sum()
                neg_count = (y == 0).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                
                self.params.update({
                    'num_leaves': 63,
                    'n_estimators': 400,
                    'learning_rate': 0.02,
                    'min_data_in_leaf': 200,  # Reduce overfitting
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'max_depth': 8,
                    'scale_pos_weight': min(scale_pos_weight, 30)  # Cap extreme weights
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
            return np.full(len(X), self.target_ctr)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibration if available"""
        try:
            raw_proba = self.predict_proba_raw(X)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                return self.calibrator.predict(raw_proba)
            else:
                return raw_proba
            
        except Exception as e:
            logger.error(f"{self.name}: Prediction failed: {e}")
            return np.full(len(X), self.target_ctr)

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
                # Calculate scale_pos_weight
                pos_count = (y == 1).sum()
                neg_count = (y == 0).sum()
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                
                self.params.update({
                    'max_depth': 10,
                    'n_estimators': 600,
                    'learning_rate': 0.01,
                    'min_child_weight': 100,  # Reduce overfitting
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'gamma': 0.1,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'scale_pos_weight': min(scale_pos_weight, 25)  # Cap extreme weights
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
            return np.full(len(X), self.target_ctr)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities with calibration if available"""
        try:
            raw_proba = self.predict_proba_raw(X)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                return self.calibrator.predict(raw_proba)
            else:
                return raw_proba
            
        except Exception as e:
            logger.error(f"{self.name}: Prediction failed: {e}")
            return np.full(len(X), self.target_ctr)

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
            
            # CTR-focused combined score with severe bias penalty
            auc_score = evaluation_results.get('auc', 0.5)
            ctr_alignment = max(0, 1 - evaluation_results['target_alignment'] / 0.02)
            logloss_score = max(0, 1 - evaluation_results.get('logloss', 1.0) / 2.0)
            
            # Severe bias penalty
            bias_magnitude = abs(evaluation_results['ctr_bias'])
            if bias_magnitude <= 0.001:
                bias_penalty = 0.0
            elif bias_magnitude <= 0.005:
                bias_penalty = (bias_magnitude - 0.001) / 0.004 * 0.3
            else:
                bias_penalty = 0.3 + min(0.6, (bias_magnitude - 0.005) / 0.01 * 0.6)
            
            evaluation_results['combined_score'] = max(0, (
                0.2 * auc_score + 
                0.6 * ctr_alignment + 
                0.2 * logloss_score - 
                bias_penalty
            ))
            
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
                logger.info(f"Calibration: Method={summary['best_method']}")
            
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