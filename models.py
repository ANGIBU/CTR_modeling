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
            logging.warning(f"GPU test failed: {e}. CPU only mode")
            gpu_available = False
    
    TORCH_AVAILABLE = True
    
    if TORCH_AVAILABLE:
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        
        try:
            if gpu_available and hasattr(torch.cuda, 'amp'):
                from torch.cuda.amp import GradScaler, autocast
                AMP_AVAILABLE = True
                logging.info("Mixed Precision enabled")
            else:
                AMP_AVAILABLE = False
        except (ImportError, AttributeError):
            AMP_AVAILABLE = False
            
except ImportError:
    TORCH_AVAILABLE = False
    AMP_AVAILABLE = False
    rtx_4060ti_detected = False
    logging.warning("PyTorch is not installed. DeepCTR models will not be available.")

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
            'warning': 38.5,
            'critical': 44.0,
            'abort': 49.5
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss / (1024**3)
            return 2.0
        except Exception:
            return 2.0
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get memory status"""
        try:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                usage_gb = (vm.total - vm.available) / (1024**3)
                available_gb = vm.available / (1024**3)
            else:
                usage_gb = 20.0
                available_gb = 40.0
            
            if usage_gb >= self.memory_thresholds['abort']:
                level = "abort"
            elif usage_gb >= self.memory_thresholds['critical']:
                level = "critical"
            elif usage_gb >= self.memory_thresholds['warning']:
                level = "warning"
            else:
                level = "normal"
            
            return {
                'usage_gb': usage_gb,
                'available_gb': available_gb,
                'level': level,
                'should_cleanup': level in ['warning', 'critical'],
                'should_abort': level == 'abort',
                'should_simplify': level in ['critical', 'abort']
            }
        except:
            return {
                'usage_gb': 20.0,
                'available_gb': 40.0,
                'level': 'normal',
                'should_cleanup': False,
                'should_abort': False,
                'should_simplify': False
            }
    
    def force_memory_cleanup(self, intensive: bool = False):
        """Memory cleanup"""
        try:
            initial_memory = self.get_memory_usage()
            
            cleanup_rounds = 20 if intensive else 15
            sleep_time = 0.2 if intensive else 0.1
            
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                if i % 5 == 0:
                    current_memory = self.get_memory_usage()
                    if initial_memory - current_memory > 5.0:
                        break
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                    if intensive:
                        time.sleep(0.5)
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass
            
            final_memory = self.get_memory_usage()
            memory_freed = initial_memory - final_memory
            
            if memory_freed > 0.1:
                logger.info(f"Memory cleanup: {memory_freed:.2f}GB freed ({cleanup_rounds} rounds)")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0.0

class CTRCalibrator:
    """CTR calibration system"""
    
    def __init__(self):
        self.calibration_models = {}
        self.calibration_scores = {}
        self.best_method = None
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
    
    def fit_platt_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Platt Scaling calibration"""
        try:
            if SKLEARN_AVAILABLE:
                logger.info("Platt Scaling calibration training started")
                
                from sklearn.linear_model import LogisticRegression
                platt_model = LogisticRegression()
                platt_model.fit(y_pred_proba.reshape(-1, 1), y_true)
                
                self.calibration_models['platt_scaling'] = platt_model
                
                calibrated_proba = platt_model.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
                score = self._calculate_calibration_score(y_true, calibrated_proba)
                self.calibration_scores['platt_scaling'] = score
                
                logger.info(f"Platt Scaling training complete - score: {score:.4f}")
            
        except Exception as e:
            logger.error(f"Platt Scaling calibration failed: {e}")
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Isotonic Regression calibration"""
        try:
            if SKLEARN_AVAILABLE:
                logger.info("Isotonic Regression calibration training started")
                
                isotonic_model = IsotonicRegression(out_of_bounds='clip')
                isotonic_model.fit(y_pred_proba, y_true)
                
                self.calibration_models['isotonic_regression'] = isotonic_model
                
                calibrated_proba = isotonic_model.predict(y_pred_proba)
                score = self._calculate_calibration_score(y_true, calibrated_proba)
                self.calibration_scores['isotonic_regression'] = score
                
                logger.info(f"Isotonic Regression training complete - score: {score:.4f}")
            
        except Exception as e:
            logger.error(f"Isotonic Regression calibration failed: {e}")
    
    def fit_temperature_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Temperature Scaling calibration"""
        try:
            logger.info("Temperature Scaling calibration training started")
            
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(temperature, y_true, y_pred_logits, bias=0.0):
                scaled_logits = y_pred_logits / temperature + bias
                scaled_proba = 1 / (1 + np.exp(-scaled_logits))
                scaled_proba = np.clip(scaled_proba, 1e-15, 1 - 1e-15)
                return -np.mean(y_true * np.log(scaled_proba) + (1 - y_true) * np.log(1 - scaled_proba))
            
            y_pred_logits = np.log(y_pred_proba / (1 - y_pred_proba + 1e-15) + 1e-15)
            
            result_temp = minimize_scalar(
                lambda t: temperature_loss(t, y_true, y_pred_logits),
                bounds=(0.1, 10.0),
                method='bounded'
            )
            optimal_temperature = result_temp.x
            
            result_bias = minimize_scalar(
                lambda b: temperature_loss(optimal_temperature, y_true, y_pred_logits, b),
                bounds=(-5.0, 5.0),
                method='bounded'
            )
            optimal_bias = result_bias.x
            
            self.calibration_models['temperature_scaling'] = {
                'temperature': optimal_temperature,
                'bias': optimal_bias
            }
            
            scaled_logits = y_pred_logits / optimal_temperature + optimal_bias
            calibrated_proba = 1 / (1 + np.exp(-scaled_logits))
            calibrated_proba = np.clip(calibrated_proba, 1e-15, 1 - 1e-15)
            
            score = self._calculate_calibration_score(y_true, calibrated_proba)
            self.calibration_scores['temperature_scaling'] = score
            
            logger.info(f"Temperature Scaling training complete - temperature: {optimal_temperature:.3f}, bias: {optimal_bias:.3f}")
            logger.info(f"Temperature Scaling score: {score:.4f}")
            
        except Exception as e:
            logger.error(f"Temperature Scaling calibration failed: {e}")
    
    def fit_beta_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Beta calibration"""
        try:
            logger.info("Beta calibration training started")
            
            from scipy.optimize import minimize
            from scipy.special import betaln
            
            def beta_loss(params, y_true, y_pred_proba):
                a, b = np.exp(params)
                eps = 1e-12
                
                log_proba = np.log(y_pred_proba + eps)
                log_1_minus_proba = np.log(1 - y_pred_proba + eps)
                
                beta_proba = np.exp(betaln(a, b) + (a - 1) * log_proba + (b - 1) * log_1_minus_proba)
                beta_proba = np.clip(beta_proba, eps, 1 - eps)
                
                return -np.mean(y_true * np.log(beta_proba) + (1 - y_true) * np.log(1 - beta_proba))
            
            result = minimize(
                lambda params: beta_loss(params, y_true, y_pred_proba),
                x0=[0.0, 0.0],
                method='BFGS'
            )
            
            optimal_a, optimal_b = np.exp(result.x)
            
            self.calibration_models['beta_calibration'] = {
                'a': optimal_a,
                'b': optimal_b
            }
            
            eps = 1e-12
            log_proba = np.log(y_pred_proba + eps)
            log_1_minus_proba = np.log(1 - y_pred_proba + eps)
            calibrated_proba = np.exp(betaln(optimal_a, optimal_b) + 
                                    (optimal_a - 1) * log_proba + 
                                    (optimal_b - 1) * log_1_minus_proba)
            calibrated_proba = np.clip(calibrated_proba, eps, 1 - eps)
            
            score = self._calculate_calibration_score(y_true, calibrated_proba)
            self.calibration_scores['beta_calibration'] = score
            
            logger.info(f"Beta calibration training complete - score: {score:.4f}")
            
        except Exception as e:
            logger.error(f"Beta calibration failed: {e}")
    
    def fit_bias_correction(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Bias correction"""
        try:
            logger.info("Bias correction training started")
            
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            
            self.bias_correction = actual_ctr - predicted_ctr
            
            if predicted_ctr > 0:
                self.multiplicative_correction = actual_ctr / predicted_ctr
            else:
                self.multiplicative_correction = 1.0
            
            logger.info(f"Bias correction complete - additive: {self.bias_correction:.4f}, multiplicative: {self.multiplicative_correction:.4f}")
            
        except Exception as e:
            logger.error(f"Bias correction training failed: {e}")
    
    def _calculate_calibration_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate calibration score"""
        try:
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_error = abs(predicted_ctr - actual_ctr)
            ctr_score = max(0, 1 - ctr_error * 1000)
            
            if SKLEARN_AVAILABLE:
                from sklearn.metrics import brier_score_loss
                brier_score = brier_score_loss(y_true, y_pred_proba)
                brier_score_normalized = max(0, 1 - brier_score * 4)
            else:
                brier_score_normalized = 0.5
            
            return (ctr_score * 0.7) + (brier_score_normalized * 0.3)
            
        except Exception:
            return 0.0
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, methods: List[str] = None):
        """Complete calibration training"""
        logger.info("CTR calibration training started")
        
        if methods is None:
            methods = ['platt_scaling', 'isotonic_regression', 'temperature_scaling', 'beta_calibration', 'bias_correction']
        
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        
        if len(y_true) != len(y_pred_proba):
            raise ValueError("Length mismatch between y_true and y_pred_proba")
        
        if len(y_true) == 0:
            raise ValueError("Empty array cannot be processed")
        
        if 'platt_scaling' in methods:
            self.fit_platt_scaling(y_true, y_pred_proba)
        
        if 'isotonic_regression' in methods:
            self.fit_isotonic_regression(y_true, y_pred_proba)
        
        if 'temperature_scaling' in methods:
            self.fit_temperature_scaling(y_true, y_pred_proba)
        
        if 'beta_calibration' in methods:
            self.fit_beta_calibration(y_true, y_pred_proba)
        
        if 'bias_correction' in methods:
            self.fit_bias_correction(y_true, y_pred_proba)
        
        if self.calibration_scores:
            self.best_method = max(self.calibration_scores, key=self.calibration_scores.get)
            logger.info(f"Best calibration method: {self.best_method} (score: {self.calibration_scores[self.best_method]:.4f})")
        else:
            self.best_method = 'bias_correction'
            logger.warning("All calibration methods failed, using bias correction")
        
        logger.info("CTR calibration training complete")
    
    def predict(self, y_pred_proba: np.ndarray, method: str = None) -> np.ndarray:
        """Apply calibration"""
        try:
            if method is None:
                method = self.best_method
            
            if method not in self.calibration_models and method != 'bias_correction':
                method = 'bias_correction'
            
            y_pred_proba = np.asarray(y_pred_proba)
            
            if method == 'platt_scaling' and method in self.calibration_models:
                model = self.calibration_models[method]
                return model.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
            
            elif method == 'isotonic_regression' and method in self.calibration_models:
                model = self.calibration_models[method]
                return np.clip(model.predict(y_pred_proba), 1e-15, 1 - 1e-15)
            
            elif method == 'temperature_scaling' and method in self.calibration_models:
                params = self.calibration_models[method]
                y_pred_logits = np.log(y_pred_proba / (1 - y_pred_proba + 1e-15) + 1e-15)
                scaled_logits = y_pred_logits / params['temperature'] + params['bias']
                calibrated_proba = 1 / (1 + np.exp(-scaled_logits))
                return np.clip(calibrated_proba, 1e-15, 1 - 1e-15)
            
            elif method == 'beta_calibration' and method in self.calibration_models:
                params = self.calibration_models[method]
                from scipy.special import betaln
                eps = 1e-12
                log_proba = np.log(y_pred_proba + eps)
                log_1_minus_proba = np.log(1 - y_pred_proba + eps)
                calibrated_proba = np.exp(betaln(params['a'], params['b']) + 
                                        (params['a'] - 1) * log_proba + 
                                        (params['b'] - 1) * log_1_minus_proba)
                return np.clip(calibrated_proba, eps, 1 - eps)
            
            else:
                corrected_proba = y_pred_proba + self.bias_correction
                corrected_proba = corrected_proba * self.multiplicative_correction
                return np.clip(corrected_proba, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.warning(f"Calibration prediction failed: {e}")
            return np.clip(y_pred_proba, 1e-15, 1 - 1e-15)

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
    
    def _memory_safe_fit(self, fit_function, *args, **kwargs):
        """Memory safe fitting"""
        try:
            memory_status = self.memory_monitor.get_memory_status()
            logger.info(f"Memory thresholds - warning: {self.memory_monitor.memory_thresholds['warning']:.1f}GB, critical: {self.memory_monitor.memory_thresholds['critical']:.1f}GB, abort: {self.memory_monitor.memory_thresholds['abort']:.1f}GB")
            
            if memory_status['should_abort']:
                logger.error(f"Memory limit reached, aborting fit: {memory_status['usage_gb']:.2f}GB")
                raise MemoryError(f"Memory limit reached: {memory_status['usage_gb']:.2f}GB")
            
            if memory_status['should_simplify']:
                logger.warning(f"Memory pressure detected, simplifying parameters: {memory_status['usage_gb']:.2f}GB")
                self._simplify_for_memory()
            
            if memory_status['should_cleanup']:
                logger.info("Memory cleanup before training")
                self.memory_monitor.force_memory_cleanup()
            
            result = fit_function(*args, **kwargs)
            
            if memory_status['should_cleanup']:
                logger.info("Memory cleanup after training")
                self.memory_monitor.force_memory_cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Memory safe fit failed: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            raise
    
    def _memory_safe_predict(self, predict_function, X: pd.DataFrame) -> np.ndarray:
        """Memory safe prediction"""
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            if memory_status['should_cleanup']:
                self.memory_monitor.force_memory_cleanup()
            
            if len(X) > 500000:
                logger.info(f"Large dataset prediction: {len(X)} rows, using batch processing")
                batch_size = 100000
                predictions = []
                
                for i in range(0, len(X), batch_size):
                    batch = X.iloc[i:i + batch_size]
                    batch_pred = predict_function(batch)
                    predictions.append(batch_pred)
                    
                    if i % (batch_size * 5) == 0:
                        self.memory_monitor.force_memory_cleanup()
                
                return np.concatenate(predictions)
            else:
                return predict_function(X)
            
        except Exception as e:
            logger.error(f"Memory safe prediction failed: {e}")
            self.memory_monitor.force_memory_cleanup()
            raise
    
    def _enhance_prediction_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """Enhance prediction diversity"""
        try:
            if len(predictions) < self.prediction_diversity_threshold:
                return predictions
            
            unique_count = len(np.unique(predictions))
            if unique_count < len(predictions) * 0.1:
                noise = np.random.normal(0, 0.001, len(predictions))
                predictions = predictions + noise
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            
            return predictions
        except Exception:
            return predictions
    
    def _ensure_feature_consistency(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency"""
        try:
            if self.feature_names is None:
                return X
            
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                for feature in missing_features:
                    X[feature] = 0.0
            
            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                X = X.drop(columns=list(extra_features))
            
            X = X[self.feature_names]
            return X
            
        except Exception as e:
            logger.warning(f"Feature consistency check failed: {e}")
            return X
    
    def apply_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, method: str = 'auto'):
        """Apply calibration to model"""
        try:
            logger.info(f"Force applying calibration to {self.name} model: {method}")
            
            if not self.is_fitted:
                logger.warning("Model is not fitted, skipping calibration")
                return
            
            raw_predictions = self.predict_proba_raw(X_val)
            
            self.calibrator = CTRCalibrator()
            self.calibrator.fit(y_val.values, raw_predictions)
            
            original_ctr = np.mean(raw_predictions)
            calibrated_predictions = self.calibrator.predict(raw_predictions)
            calibrated_ctr = np.mean(calibrated_predictions)
            actual_ctr = np.mean(y_val)
            
            self.is_calibrated = True
            
            logger.info("Forced calibration application complete")
            logger.info(f"  - Original CTR: {original_ctr:.4f}")
            logger.info(f"  - Calibrated CTR: {calibrated_ctr:.4f}")
            logger.info(f"  - Actual CTR: {actual_ctr:.4f}")
            logger.info(f"  - Best method: {self.calibrator.best_method}")
            
        except Exception as e:
            logger.error(f"Calibration application failed: {e}")
            self.is_calibrated = False
    
    def _simplify_for_memory(self):
        """Simplify parameters for memory conservation (to be overridden)"""
        pass
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions before calibration"""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability prediction"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict(raw_pred)
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except Exception as e:
                logger.warning(f"{self.name} calibration prediction failed: {e}")
        
        return raw_pred
    
    def save(self, path: str):
        """Save model"""
        try:
            save_data = {
                'model': self.model,
                'params': self.params,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
                'calibrator': self.calibrator,
                'is_calibrated': self.is_calibrated
            }
            
            with open(path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"{self.name} model saved: {path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    def load(self, path: str):
        """Load model"""
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.model = save_data['model']
            self.params = save_data['params']
            self.is_fitted = save_data['is_fitted']
            self.feature_names = save_data['feature_names']
            self.calibrator = save_data.get('calibrator')
            self.is_calibrated = save_data.get('is_calibrated', False)
            
            logger.info(f"{self.name} model loaded: {path}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")

class LightGBMModel(BaseModel):
    """LightGBM model"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'max_depth': 15,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 8000,
            'early_stopping_rounds': 800,
            'random_state': 42,
            'force_col_wise': True,
            'min_child_samples': 25,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'scale_pos_weight': 52.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("LightGBM", default_params)
        self.prediction_diversity_threshold = 2500
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        self.params.update({
            'num_leaves': 127,
            'max_depth': 10,
            'n_estimators': 5000,
            'min_child_samples': 50,
            'early_stopping_rounds': 500
        })
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """LightGBM model training"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0)
            if X_val is not None:
                X_val_clean = X_val.fillna(0)
            else:
                X_val_clean = None
            
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
                if X_val_clean is not None and col in X_val_clean.columns and X_val_clean[col].dtype in ['float64']:
                    X_val_clean[col] = X_val_clean[col].astype('float32')
            
            train_data = lgb.Dataset(X_train_clean, label=y_train)
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val_clean is not None and y_val is not None:
                valid_data = lgb.Dataset(X_val_clean, label=y_val, reference=train_data)
                valid_sets.append(valid_data)
                valid_names.append('valid')
            
            early_stopping = self.params.get('early_stopping_rounds', 800)
            
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)]
            )
            
            self.is_fitted = True
            
            if X_val_clean is not None and y_val is not None:
                logger.info(f"{self.name}: Starting forced calibration application")
                self.apply_calibration(X_val_clean, y_val, method='auto')
                if self.is_calibrated:
                    logger.info(f"{self.name}: Forced calibration application complete")
                else:
                    logger.warning(f"{self.name}: Forced calibration application failed")
            
            del X_train_clean
            if X_val_clean is not None:
                del X_val_clean
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions before calibration"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
            
            num_iteration = getattr(self.model, 'best_iteration', None)
            proba = self.model.predict(X_processed, num_iteration=num_iteration)
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X)

class XGBoostModel(BaseModel):
    """XGBoost model with memory management"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        default_params = {
            'objective': 'binary:logistic',
            'max_depth': 12,
            'learning_rate': 0.008,
            'subsample': 0.85,
            'colsample_bytree': 0.95,
            'colsample_bylevel': 0.85,
            'min_child_weight': 8,
            'reg_alpha': 2.5,
            'reg_lambda': 2.5,
            'scale_pos_weight': 52.0,
            'random_state': 42,
            'n_estimators': 8000,
            'early_stopping_rounds': 800,
            'max_bin': 255,
            'nthread': 12,
            'grow_policy': 'lossguide',
            'gamma': 0.0,
            'max_leaves': 2047,
            'tree_method': 'hist'
        }
        
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
                        'predictor': 'gpu_predictor'
                    })
                    logger.info("XGBoost GPU mode enabled")
                else:
                    logger.info("Using XGBoost CPU mode due to memory shortage")
                    
            except Exception as e:
                logger.warning(f"GPU setup failed, using CPU mode: {e}")
        
        if params:
            default_params.update(params)
        
        super().__init__("XGBoost", default_params)
        self.prediction_diversity_threshold = 2500
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        self.params.update({
            'max_depth': 8,
            'n_estimators': 3000,
            'min_child_weight': 15,
            'nthread': 6,
            'tree_method': 'hist',
            'early_stopping_rounds': 300,
            'max_leaves': 511,
            'max_bin': 128,
            'colsample_bytree': 0.8,
            'subsample': 0.8
        })
        self.params.pop('gpu_id', None)
        self.params.pop('predictor', None)
        logger.info(f"{self.name}: Parameters simplified for memory conservation and switched to CPU mode")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """XGBoost model training with GPU fallback"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        
        def _fit_internal():
            logger.info(f"{self.name}: Performing memory cleanup before training")
            
            self.memory_monitor.force_memory_cleanup(intensive=True)
            time.sleep(2)
            
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_abort']:
                raise MemoryError(f"Insufficient memory even after cleanup: {memory_status['usage_gb']:.2f}GB")
            
            if memory_status['should_simplify']:
                logger.warning(f"Memory pressure detected, applying simplified parameters")
                self._simplify_for_memory()
            
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0).copy()
            if X_val is not None:
                X_val_clean = X_val.fillna(0).copy()
            else:
                X_val_clean = None
            
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
                elif X_train_clean[col].dtype in ['int64']:
                    X_train_clean[col] = X_train_clean[col].astype('int32')
            
            if X_val_clean is not None:
                for col in X_val_clean.columns:
                    if col in X_train_clean.columns:
                        if X_val_clean[col].dtype in ['float64']:
                            X_val_clean[col] = X_val_clean[col].astype('float32')
                        elif X_val_clean[col].dtype in ['int64']:
                            X_val_clean[col] = X_val_clean[col].astype('int32')
            
            self.memory_monitor.force_memory_cleanup()
            time.sleep(1)
            
            gpu_success = False
            if 'gpu_id' in self.params or self.params.get('tree_method') == 'gpu_hist':
                try:
                    logger.info(f"{self.name}: Attempting GPU training")
                    gpu_success = self._try_gpu_training(X_train_clean, y_train, X_val_clean, y_val)
                except Exception as gpu_error:
                    logger.warning(f"{self.name}: GPU training failed: {gpu_error}")
                    if "cudaErrorMemoryAllocation" in str(gpu_error) or "out of memory" in str(gpu_error):
                        logger.info(f"{self.name}: GPU memory error detected, switching to CPU mode")
                        self._force_cpu_mode()
                    else:
                        logger.info(f"{self.name}: GPU error detected, switching to CPU mode")
                        self._force_cpu_mode()
            
            if not gpu_success:
                try:
                    logger.info(f"{self.name}: Starting CPU training")
                    self._train_cpu_mode(X_train_clean, y_train, X_val_clean, y_val)
                    
                    if X_val is not None and y_val is not None:
                        logger.info(f"{self.name}: Starting forced calibration application")
                        X_val_for_calibration = X_val.fillna(0).copy()
                        for col in X_val_for_calibration.columns:
                            if X_val_for_calibration[col].dtype in ['float64']:
                                X_val_for_calibration[col] = X_val_for_calibration[col].astype('float32')
                            elif X_val_for_calibration[col].dtype in ['int64']:
                                X_val_for_calibration[col] = X_val_for_calibration[col].astype('int32')
                        
                        self.apply_calibration(X_val_for_calibration, y_val, method='auto')
                        if self.is_calibrated:
                            logger.info(f"{self.name}: Forced calibration application complete")
                        else:
                            logger.warning(f"{self.name}: Forced calibration application failed")
                        
                        del X_val_for_calibration
                    
                except Exception as cpu_error:
                    logger.error(f"{self.name}: CPU training also failed: {cpu_error}")
                    
                    self._cleanup_training_data(locals())
                    
                    cleanup_rounds = 20
                    for i in range(cleanup_rounds):
                        collected = gc.collect()
                        time.sleep(0.1)
                        if i % 5 == 0:
                            memory_freed = self.memory_monitor.force_memory_cleanup(intensive=True)
                            logger.info(f"Memory cleanup: {memory_freed:.2f}GB freed (round {i+1})")
                    
                    raise
            
            self._cleanup_training_data(locals())
            self.memory_monitor.force_memory_cleanup()
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def _try_gpu_training(self, X_train_clean: pd.DataFrame, y_train: pd.Series, 
                         X_val_clean: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> bool:
        """Try GPU training"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_memory_gb = free_memory / (1024**3)
                
                if free_memory_gb < 6.0:
                    logger.warning(f"{self.name}: Insufficient GPU memory ({free_memory_gb:.2f}GB), switching to CPU")
                    return False
            
            logger.info(f"{self.name}: Creating XGBoost model with GPU parameters")
            
            # XGBoost 모델 파라미터에서 eval_metric 제거
            train_params = self.params.copy()
            train_params.pop('early_stopping_rounds', None)
            
            self.model = xgb.XGBClassifier(**train_params)
            
            eval_set = None
            if X_val_clean is not None and y_val is not None:
                eval_set = [(X_val_clean, y_val)]
            
            early_stopping_rounds = self.params.get('early_stopping_rounds', 800)
            
            logger.info(f"{self.name}: Starting XGBoost GPU training")
            self.model.fit(
                X_train_clean, 
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            
            self.is_fitted = True
            
            logger.info(f"{self.name}: GPU training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: GPU training failed: {e}")
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return False
    
    def _train_cpu_mode(self, X_train_clean: pd.DataFrame, y_train: pd.Series, 
                       X_val_clean: Optional[pd.DataFrame], y_val: Optional[pd.Series]):
        """Train in CPU mode"""
        try:
            logger.info(f"{self.name}: Creating XGBoost model with CPU parameters")
            
            # XGBoost 모델 파라미터에서 eval_metric 제거
            train_params = self.params.copy()
            train_params.pop('early_stopping_rounds', None)
            
            self.model = xgb.XGBClassifier(**train_params)
            
            eval_set = None
            if X_val_clean is not None and y_val is not None:
                eval_set = [(X_val_clean, y_val)]
            
            early_stopping_rounds = self.params.get('early_stopping_rounds', 800)
            
            logger.info(f"{self.name}: Starting XGBoost CPU training")
            self.model.fit(
                X_train_clean, 
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            
            self.is_fitted = True
            
            logger.info(f"{self.name}: CPU training completed successfully")
            
        except Exception as e:
            logger.error(f"{self.name}: CPU training failed: {e}")
            raise
    
    def _force_cpu_mode(self):
        """Force switch to CPU mode"""
        logger.info(f"{self.name}: Forcing CPU mode")
        
        self.params.pop('gpu_id', None)
        self.params.pop('predictor', None)
        
        self.params['tree_method'] = 'hist'
        
        if 'nthread' not in self.params:
            self.params['nthread'] = 8
        
        logger.info(f"{self.name}: Switched to CPU mode with optimized parameters")
    
    def _cleanup_training_data(self, local_vars: dict):
        """Cleanup training data"""
        try:
            cleanup_vars = ['X_train_clean', 'X_val_clean']
            for var_name in cleanup_vars:
                if var_name in local_vars and local_vars[var_name] is not None:
                    del local_vars[var_name]
        except Exception:
            pass
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions before calibration"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
                elif X_processed[col].dtype in ['int64']:
                    X_processed[col] = X_processed[col].astype('int32')
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X)

class LogisticModel(BaseModel):
    """Logistic Regression model"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not installed.")
        
        default_params = {
            'C': 0.1,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 3000,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': 12
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("LogisticRegression", default_params)
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 2500
    
    def _simplify_for_memory(self):
        """Simplify parameters when memory is low"""
        self.params.update({
            'C': 1.0,
            'max_iter': 1000,
            'n_jobs': 6
        })
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: Parameters simplified for memory conservation")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Logistic Regression model training"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0)
            
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
            
            if len(X_train_clean) > 4000000:
                sample_size = 4000000
                logger.info(f"Large data detected, applying sampling ({len(X_train_clean):,} -> {sample_size:,})")
                
                sample_indices = np.random.choice(len(X_train_clean), size=sample_size, replace=False)
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
            
            if X_val is not None and y_val is not None:
                logger.info(f"{self.name}: Starting forced calibration application")
                X_val_clean = X_val.fillna(0)
                for col in X_val_clean.columns:
                    if X_val_clean[col].dtype in ['float64']:
                        X_val_clean[col] = X_val_clean[col].astype('float32')
                
                self.apply_calibration(X_val_clean, y_val, method='auto')
                if self.is_calibrated:
                    logger.info(f"{self.name}: Forced calibration application complete")
                else:
                    logger.warning(f"{self.name}: Forced calibration application failed")
            
            del X_train_clean
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw predictions before calibration"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X)

class ModelFactory:
    """Model factory for creating models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create model by type"""
        try:
            memory_monitor = MemoryMonitor()
            logger.info(f"Creating model: {model_type} (forced calibration application configured)")
            logger.info(f"Memory thresholds - warning: {memory_monitor.memory_thresholds['warning']:.1f}GB, critical: {memory_monitor.memory_thresholds['critical']:.1f}GB, abort: {memory_monitor.memory_thresholds['abort']:.1f}GB")
            
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
            
            logger.info(f"{model_type} model creation complete - forced calibration application guaranteed")
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
        
        logger.info(f"Available models: {available} (all models have forced calibration application)")
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