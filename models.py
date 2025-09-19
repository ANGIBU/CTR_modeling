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
    logging.warning("LightGBM이 설치되지 않았습니다.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost가 설치되지 않았습니다.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost가 설치되지 않았습니다.")

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
            
            logging.info(f"GPU 감지: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            logging.info(f"RTX 4060 Ti 최적화: {rtx_4060ti_detected}")
            
        except Exception as e:
            logging.warning(f"GPU 테스트 실패: {e}. CPU 전용 모드")
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
                logging.info("Mixed Precision 활성화")
            else:
                AMP_AVAILABLE = False
        except (ImportError, AttributeError):
            AMP_AVAILABLE = False
            
except ImportError:
    TORCH_AVAILABLE = False
    AMP_AVAILABLE = False
    rtx_4060ti_detected = False
    logging.warning("PyTorch가 설치되지 않았습니다. DeepCTR 모델을 사용할 수 없습니다.")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.model_selection import cross_val_predict
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn이 설치되지 않았습니다.")

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
    logging.warning("scipy가 설치되지 않았습니다. 일부 캘리브레이션 기능이 제한됩니다.")

from config import Config

logger = logging.getLogger(__name__)

class CTRCalibrator:
    """CTR 예측 전용 캘리브레이션 클래스"""
    
    def __init__(self, target_ctr: float = 0.0191, method: str = 'auto'):
        self.target_ctr = target_ctr
        self.method = method
        self.calibrators = {}
        self.is_fitted = False
        self.best_method = None
        self.calibration_curve = None
        
        self.platt_scaler = None
        self.isotonic_regressor = None
        self.temperature = 1.0
        self.temperature_bias = 0.0
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
        self.calibration_scores = {}
        self.beta_calibrator = None
        self.spline_calibrator = None
        
    def fit_platt_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Platt Scaling 캘리브레이션 학습"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn을 사용할 수 없어 Platt Scaling을 건너뜁니다")
                return
                
            logger.info("Platt Scaling 캘리브레이션 학습 시작")
            
            self.platt_scaler = LogisticRegression()
            
            pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            logits = np.log(pred_clipped / (1 - pred_clipped))
            
            self.platt_scaler.fit(logits.reshape(-1, 1), y_true)
            
            calibrated_probs = self.apply_platt_scaling(y_pred_proba)
            self.calibration_scores['platt_scaling'] = self._evaluate_calibration(y_true, calibrated_probs)
            
            logger.info(f"Platt Scaling 학습 완료 - 점수: {self.calibration_scores['platt_scaling']:.4f}")
            
        except Exception as e:
            logger.error(f"Platt Scaling 학습 실패: {e}")
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Isotonic Regression 캘리브레이션 학습"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn을 사용할 수 없어 Isotonic Regression을 건너뜁니다")
                return
                
            logger.info("Isotonic Regression 캘리브레이션 학습 시작")
            
            self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor.fit(y_pred_proba, y_true)
            
            calibrated_probs = self.apply_isotonic_regression(y_pred_proba)
            self.calibration_scores['isotonic_regression'] = self._evaluate_calibration(y_true, calibrated_probs)
            
            logger.info(f"Isotonic Regression 학습 완료 - 점수: {self.calibration_scores['isotonic_regression']:.4f}")
            
        except Exception as e:
            logger.error(f"Isotonic Regression 학습 실패: {e}")
    
    def fit_temperature_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Temperature Scaling 캘리브레이션 학습"""
        try:
            logger.info("Temperature Scaling 캘리브레이션 학습 시작")
            
            def temperature_loss(params):
                temp, bias = params
                if temp <= 0:
                    return float('inf')
                
                pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                
                adjusted_logits = (logits + bias) / temp
                calibrated_probs = 1 / (1 + np.exp(-adjusted_logits))
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                log_loss_val = -np.mean(y_true * np.log(calibrated_probs) + (1 - y_true) * np.log(1 - calibrated_probs))
                ctr_bias = abs(calibrated_probs.mean() - y_true.mean()) * 1200
                diversity_loss = -calibrated_probs.std() * 10
                
                return log_loss_val + ctr_bias + diversity_loss
            
            if SCIPY_AVAILABLE:
                from scipy.optimize import minimize
                result = minimize(
                    temperature_loss, 
                    x0=[1.0, 0.0], 
                    bounds=[(0.2, 12.0), (-2.5, 2.5)],
                    method='L-BFGS-B'
                )
                self.temperature = result.x[0]
                self.temperature_bias = result.x[1]
            else:
                best_loss = float('inf')
                best_temp = 1.0
                best_bias = 0.0
                
                for temp in np.logspace(-0.7, 1.1, 30):
                    for bias in np.linspace(-2.5, 2.5, 30):
                        loss = temperature_loss([temp, bias])
                        if loss < best_loss:
                            best_loss = loss
                            best_temp = temp
                            best_bias = bias
                
                self.temperature = best_temp
                self.temperature_bias = best_bias
            
            calibrated_probs = self.apply_temperature_scaling(y_pred_proba)
            self.calibration_scores['temperature_scaling'] = self._evaluate_calibration(y_true, calibrated_probs)
            
            logger.info(f"Temperature Scaling 학습 완료 - 온도: {self.temperature:.3f}, 편향: {self.temperature_bias:.3f}")
            logger.info(f"Temperature Scaling 점수: {self.calibration_scores['temperature_scaling']:.4f}")
            
        except Exception as e:
            logger.error(f"Temperature Scaling 학습 실패: {e}")
            self.temperature = 1.0
            self.temperature_bias = 0.0
    
    def fit_beta_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Beta 캘리브레이션 학습"""
        try:
            logger.info("Beta 캘리브레이션 학습 시작")
            
            def beta_loss(params):
                a, b = params
                if a <= 0 or b <= 0:
                    return float('inf')
                
                from scipy.stats import beta
                calibrated_probs = beta.cdf(y_pred_proba, a, b)
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                log_loss_val = -np.mean(y_true * np.log(calibrated_probs) + (1 - y_true) * np.log(1 - calibrated_probs))
                ctr_bias = abs(calibrated_probs.mean() - y_true.mean()) * 1000
                
                return log_loss_val + ctr_bias
            
            if SCIPY_AVAILABLE:
                from scipy.optimize import minimize
                result = minimize(
                    beta_loss,
                    x0=[1.0, 1.0],
                    bounds=[(0.1, 10.0), (0.1, 10.0)],
                    method='L-BFGS-B'
                )
                
                self.beta_params = result.x
                
                calibrated_probs = self.apply_beta_calibration(y_pred_proba)
                self.calibration_scores['beta_calibration'] = self._evaluate_calibration(y_true, calibrated_probs)
                
                logger.info(f"Beta 캘리브레이션 학습 완료 - 점수: {self.calibration_scores['beta_calibration']:.4f}")
            
        except Exception as e:
            logger.warning(f"Beta 캘리브레이션 학습 실패: {e}")
            self.beta_params = None
    
    def fit_bias_correction(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """편향 보정 학습"""
        try:
            logger.info("편향 보정 학습 시작")
            
            actual_ctr = y_true.mean()
            predicted_ctr = y_pred_proba.mean()
            
            self.bias_correction = actual_ctr - predicted_ctr
            
            if predicted_ctr > 0:
                self.multiplicative_correction = actual_ctr / predicted_ctr
            else:
                self.multiplicative_correction = 1.0
            
            logger.info(f"편향 보정 완료 - 가법: {self.bias_correction:.4f}, 승법: {self.multiplicative_correction:.4f}")
            
        except Exception as e:
            logger.error(f"편향 보정 학습 실패: {e}")
    
    def fit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, methods: List[str] = None):
        """전체 캘리브레이션 학습"""
        logger.info("CTR 캘리브레이션 학습 시작")
        
        if methods is None:
            methods = ['platt_scaling', 'isotonic_regression', 'temperature_scaling', 'beta_calibration', 'bias_correction']
        
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        
        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true와 y_pred_proba의 길이가 다릅니다")
        
        if len(y_true) == 0:
            raise ValueError("빈 배열은 처리할 수 없습니다")
        
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
            logger.info(f"최적 캘리브레이션 방법: {self.best_method} (점수: {self.calibration_scores[self.best_method]:.4f})")
        else:
            self.best_method = 'bias_correction'
            logger.warning("모든 캘리브레이션 방법이 실패했습니다. 편향 보정만 사용")
        
        self.is_fitted = True
        logger.info("CTR 캘리브레이션 학습 완료")
    
    def apply_platt_scaling(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Platt Scaling 적용"""
        if self.platt_scaler is None:
            return y_pred_proba
        
        try:
            pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            logits = np.log(pred_clipped / (1 - pred_clipped))
            calibrated_probs = self.platt_scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
            return np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
        except Exception as e:
            logger.warning(f"Platt Scaling 적용 실패: {e}")
            return y_pred_proba
    
    def apply_isotonic_regression(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Isotonic Regression 적용"""
        if self.isotonic_regressor is None:
            return y_pred_proba
        
        try:
            calibrated_probs = self.isotonic_regressor.predict(y_pred_proba)
            return np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
        except Exception as e:
            logger.warning(f"Isotonic Regression 적용 실패: {e}")
            return y_pred_proba
    
    def apply_temperature_scaling(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Temperature Scaling 적용"""
        try:
            pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            logits = np.log(pred_clipped / (1 - pred_clipped))
            
            adjusted_logits = (logits + self.temperature_bias) / self.temperature
            calibrated_probs = 1 / (1 + np.exp(-adjusted_logits))
            
            return np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
        except Exception as e:
            logger.warning(f"Temperature Scaling 적용 실패: {e}")
            return y_pred_proba
    
    def apply_beta_calibration(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """Beta 캘리브레이션 적용"""
        if not hasattr(self, 'beta_params') or self.beta_params is None:
            return y_pred_proba
        
        try:
            from scipy.stats import beta
            a, b = self.beta_params
            calibrated_probs = beta.cdf(y_pred_proba, a, b)
            return np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
        except Exception as e:
            logger.warning(f"Beta 캘리브레이션 적용 실패: {e}")
            return y_pred_proba
    
    def apply_bias_correction(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """편향 보정 적용"""
        try:
            corrected = y_pred_proba * self.multiplicative_correction + self.bias_correction
            return np.clip(corrected, 1e-15, 1 - 1e-15)
        except Exception as e:
            logger.warning(f"편향 보정 적용 실패: {e}")
            return y_pred_proba
    
    def predict(self, y_pred_proba: np.ndarray, method: str = None) -> np.ndarray:
        """캘리브레이션된 예측 반환"""
        if not self.is_fitted:
            logger.warning("캘리브레이션이 학습되지 않았습니다. 원본 예측을 반환합니다.")
            return y_pred_proba
        
        if method is None:
            method = self.best_method
        
        y_pred_proba = np.asarray(y_pred_proba)
        
        if method == 'platt_scaling':
            return self.apply_platt_scaling(y_pred_proba)
        elif method == 'isotonic_regression':
            return self.apply_isotonic_regression(y_pred_proba)
        elif method == 'temperature_scaling':
            return self.apply_temperature_scaling(y_pred_proba)
        elif method == 'beta_calibration':
            return self.apply_beta_calibration(y_pred_proba)
        elif method == 'bias_correction':
            return self.apply_bias_correction(y_pred_proba)
        else:
            logger.warning(f"알 수 없는 캘리브레이션 방법: {method}. 편향 보정 사용")
            return self.apply_bias_correction(y_pred_proba)
    
    def _evaluate_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """캘리브레이션 품질 평가"""
        try:
            ece = self._calculate_ece(y_true, y_pred_proba)
            
            actual_ctr = y_true.mean()
            predicted_ctr = y_pred_proba.mean()
            ctr_alignment = 1.0 - abs(actual_ctr - predicted_ctr) / max(actual_ctr, 0.001)
            
            reliability = self._calculate_reliability(y_true, y_pred_proba)
            
            calibration_score = 0.4 * (1.0 - ece) + 0.4 * ctr_alignment + 0.2 * reliability
            
            return max(0.0, calibration_score)
            
        except Exception as e:
            logger.warning(f"캘리브레이션 평가 실패: {e}")
            return 0.0
    
    def _calculate_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 15) -> float:
        """Expected Calibration Error 계산"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
            
        except Exception as e:
            logger.warning(f"ECE 계산 실패: {e}")
            return 1.0
    
    def _calculate_reliability(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """신뢰도 계산"""
        try:
            sorted_indices = np.argsort(y_pred_proba)
            sorted_true = y_true[sorted_indices]
            sorted_pred = y_pred_proba[sorted_indices]
            
            bin_size = len(y_true) // 10
            reliability_scores = []
            
            for i in range(0, len(y_true), bin_size):
                end_idx = min(i + bin_size, len(y_true))
                bin_true = sorted_true[i:end_idx]
                bin_pred = sorted_pred[i:end_idx]
                
                if len(bin_true) > 0:
                    actual_positive_rate = bin_true.mean()
                    predicted_positive_rate = bin_pred.mean()
                    
                    if predicted_positive_rate > 0:
                        reliability = 1 - abs(actual_positive_rate - predicted_positive_rate) / predicted_positive_rate
                        reliability_scores.append(max(0, reliability))
            
            return np.mean(reliability_scores) if reliability_scores else 0.0
            
        except Exception as e:
            logger.warning(f"신뢰도 계산 실패: {e}")
            return 0.0
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """캘리브레이션 요약 정보"""
        return {
            'is_fitted': self.is_fitted,
            'best_method': self.best_method,
            'calibration_scores': self.calibration_scores.copy(),
            'temperature': self.temperature,
            'temperature_bias': self.temperature_bias,
            'bias_correction': self.bias_correction,
            'multiplicative_correction': self.multiplicative_correction,
            'available_methods': list(self.calibration_scores.keys()),
            'beta_params': getattr(self, 'beta_params', None)
        }

class MemoryMonitor:
    """64GB RAM 환경 메모리 모니터링"""
    
    def __init__(self, max_memory_gb: float = 55.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.max_memory_gb = max_memory_gb
        self.lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 2.0
        
        self.warning_threshold = max_memory_gb * 0.65
        self.critical_threshold = max_memory_gb * 0.75
        self.abort_threshold = max_memory_gb * 0.85
        
        logger.info(f"메모리 임계값 - 경고: {self.warning_threshold:.1f}GB, "
                   f"위험: {self.critical_threshold:.1f}GB, 중단: {self.abort_threshold:.1f}GB")
        
    def get_memory_usage(self) -> float:
        """메모리 사용량 (GB)"""
        if not self.monitoring_enabled:
            return 2.0
        
        try:
            with self.lock:
                current_time = time.time()
                if current_time - self._last_check_time < self._check_interval:
                    return getattr(self, '_cached_memory', 2.0)
                
                process = psutil.Process()
                memory_gb = process.memory_info().rss / (1024**3)
                self._cached_memory = memory_gb
                self._last_check_time = current_time
                return memory_gb
        except Exception:
            return 2.0
    
    def get_available_memory(self) -> float:
        """사용 가능한 메모리 (GB)"""
        if not self.monitoring_enabled:
            return 45.0
        
        try:
            with self.lock:
                return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 45.0
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            return usage > self.critical_threshold or available < 15.0
        except Exception:
            return False
    
    def get_memory_status(self) -> Dict[str, Any]:
        """상세한 메모리 상태 반환"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            if usage > self.abort_threshold or available < 8:
                level = "abort"
            elif usage > self.critical_threshold or available < 15:
                level = "critical"
            elif usage > self.warning_threshold or available < 25:
                level = "warning"
            else:
                level = "normal"
            
            return {
                'usage_gb': usage,
                'available_gb': available,
                'level': level,
                'should_cleanup': level in ['warning', 'critical', 'abort'],
                'should_simplify': level in ['critical', 'abort'],
                'should_reduce_batch': level in ['warning', 'critical', 'abort']
            }
        except Exception:
            return {
                'usage_gb': 2.0,
                'available_gb': 45.0,
                'level': 'normal',
                'should_cleanup': False,
                'should_simplify': False,
                'should_reduce_batch': False
            }
    
    def force_memory_cleanup(self, intensive: bool = False):
        """메모리 정리"""
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
                logger.info(f"메모리 정리: {memory_freed:.2f}GB 해제 ({cleanup_rounds}라운드)")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")
            return 0.0

class BaseModel(ABC):
    """모든 모델의 기본 클래스"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.calibrator = None
        self.is_calibrated = False
        self.prediction_diversity_threshold = 2000
        
        self.memory_monitor = MemoryMonitor()
        
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """모델 학습"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """이진 예측"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def apply_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, 
                         method: str = 'auto', cv_folds: int = 5):
        """캘리브레이션 적용"""
        try:
            logger.info(f"{self.name} 모델에 캘리브레이션 적용: {method}")
            
            if not self.is_fitted:
                logger.warning("모델이 학습되지 않아 캘리브레이션을 건너뜁니다")
                return
            
            raw_predictions = self.predict_proba_raw(X_val)
            
            self.calibrator = CTRCalibrator()
            self.calibrator.fit(y_val.values, raw_predictions)
            
            self.is_calibrated = True
            
            calibrated_predictions = self.calibrator.predict(raw_predictions)
            
            original_ctr = raw_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val.mean()
            
            logger.info(f"캘리브레이션 결과 - 원본 CTR: {original_ctr:.4f}, "
                       f"캘리브레이션 CTR: {calibrated_ctr:.4f}, 실제 CTR: {actual_ctr:.4f}")
            
        except Exception as e:
            logger.error(f"캘리브레이션 적용 실패: {e}")
            self.is_calibrated = False
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션 이전 원본 예측"""
        return self.predict_proba(X)
    
    def _memory_safe_fit(self, fit_function, *args, **kwargs):
        """메모리 안전 학습 래퍼"""
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            if memory_status['should_cleanup']:
                logger.info(f"{self.name}: 학습 전 메모리 정리 수행")
                self.memory_monitor.force_memory_cleanup()
            
            if memory_status['should_simplify']:
                logger.warning(f"{self.name}: 메모리 부족으로 단순화 모드 활성화")
                self._simplify_for_memory()
            
            result = fit_function(*args, **kwargs)
            
            self.memory_monitor.force_memory_cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"{self.name} 메모리 안전 학습 실패: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            raise
    
    def _memory_safe_predict(self, predict_function, X: pd.DataFrame, batch_size: int = None) -> np.ndarray:
        """메모리 안전 예측 래퍼"""
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            if batch_size is None:
                if memory_status['level'] == 'abort':
                    batch_size = 3000
                elif memory_status['level'] == 'critical':
                    batch_size = 10000
                elif memory_status['level'] == 'warning':
                    batch_size = 25000
                else:
                    batch_size = 100000
            
            n_samples = len(X)
            predictions = []
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X.iloc[i:batch_end]
                
                try:
                    batch_pred = predict_function(batch_X)
                    predictions.append(batch_pred)
                    
                    if (i // batch_size) % 3 == 0:
                        gc.collect()
                    
                    if self.memory_monitor.check_memory_pressure():
                        logger.warning(f"{self.name}: 예측 중 메모리 압박 감지, 정리 수행")
                        self.memory_monitor.force_memory_cleanup()
                        
                except Exception as e:
                    logger.warning(f"{self.name}: 배치 {i}-{batch_end} 예측 실패: {e}")
                    batch_size_actual = batch_end - i
                    predictions.append(np.full(batch_size_actual, 0.0191))
            
            return np.concatenate(predictions) if predictions else np.array([])
            
        except Exception as e:
            logger.error(f"{self.name} 메모리 안전 예측 실패: {e}")
            return np.full(len(X), 0.0191)
    
    def _simplify_for_memory(self):
        """메모리 부족 시 모델 파라미터 단순화"""
        pass
    
    def _ensure_feature_consistency(self, X: pd.DataFrame) -> pd.DataFrame:
        """피처 일관성 보장"""
        if self.feature_names is None:
            return X
        
        try:
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                for feature in missing_features:
                    X[feature] = 0.0
            
            return X[self.feature_names]
            
        except Exception as e:
            logger.warning(f"피처 일관성 보장 실패: {e}")
            return X
    
    def _enhance_prediction_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """예측 다양성 향상"""
        try:
            unique_predictions = len(np.unique(predictions))
            
            if unique_predictions < self.prediction_diversity_threshold:
                noise_scale = max(predictions.std() * 0.005, 1e-6)
                noise = np.random.normal(0, noise_scale, len(predictions))
                
                predictions = predictions + noise
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"예측 다양성 향상 실패: {e}")
            return predictions

class FinalLightGBMModel(BaseModel):
    """최종 완성 LightGBM 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM이 설치되지 않았습니다.")
            
        final_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 2047,
            'learning_rate': 0.008,
            'feature_fraction': 0.95,
            'bagging_fraction': 0.85,
            'bagging_freq': 3,
            'min_child_samples': 80,
            'min_child_weight': 3,
            'lambda_l1': 2.5,
            'lambda_l2': 2.5,
            'max_depth': 20,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 8000,
            'early_stopping_rounds': 800,
            'scale_pos_weight': 52.0,
            'force_row_wise': True,
            'max_bin': 255,
            'num_threads': 12,
            'device_type': 'cpu',
            'extra_trees': False,
            'path_smooth': 1.0
        }
        
        if params:
            final_params.update(params)
        
        super().__init__("FinalLightGBM", final_params)
        self.prediction_diversity_threshold = 2500
    
    def _simplify_for_memory(self):
        """메모리 부족 시 파라미터 단순화"""
        self.params.update({
            'num_leaves': 1023,
            'max_depth': 15,
            'n_estimators': 5000,
            'min_child_samples': 150,
            'num_threads': 8,
            'early_stopping_rounds': 500
        })
        logger.info(f"{self.name}: 메모리 절약을 위해 파라미터 단순화")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """최종 완성 LightGBM 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
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
            
            train_data = lgb.Dataset(
                X_train_clean, 
                label=y_train, 
                free_raw_data=True,
                feature_name=list(X_train_clean.columns)
            )
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val_clean is not None and y_val is not None:
                val_data = lgb.Dataset(
                    X_val_clean, 
                    label=y_val, 
                    reference=train_data, 
                    free_raw_data=True,
                    feature_name=list(X_val_clean.columns)
                )
                valid_sets.append(val_data)
                valid_names.append('valid')
            
            callbacks = []
            early_stopping = self.params.get('early_stopping_rounds', 800)
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
            
            def memory_callback(env):
                if env.iteration % 300 == 0:
                    if self.memory_monitor.check_memory_pressure():
                        logger.warning("학습 중 메모리 압박 감지")
                        self.memory_monitor.force_memory_cleanup()
            
            callbacks.append(memory_callback)
            
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            
            self.is_fitted = True
            
            del train_data
            if 'val_data' in locals():
                del val_data
            del X_train_clean
            if X_val_clean is not None:
                del X_val_clean
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션 이전 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
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
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict(raw_pred)
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class FinalXGBoostModel(BaseModel):
    """최종 완성 XGBoost 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost가 설치되지 않았습니다.")
        
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
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
            'max_leaves': 2047
        }
        
        if rtx_4060ti_detected and TORCH_AVAILABLE:
            try:
                test_tensor = torch.zeros(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                memory_monitor_temp = MemoryMonitor()
                memory_status = memory_monitor_temp.get_memory_status()
                
                if memory_status['level'] not in ['critical', 'abort']:
                    final_params.update({
                        'tree_method': 'gpu_hist',
                        'gpu_id': 0,
                        'predictor': 'gpu_predictor'
                    })
                    logger.info("XGBoost GPU 모드 활성화")
                else:
                    logger.info("메모리 부족으로 XGBoost CPU 모드 사용")
                    
            except Exception as e:
                logger.warning(f"GPU 설정 실패, CPU 모드 사용: {e}")
        
        if params:
            final_params.update(params)
        
        super().__init__("FinalXGBoost", final_params)
        self.prediction_diversity_threshold = 2500
    
    def _simplify_for_memory(self):
        """메모리 부족 시 파라미터 단순화"""
        self.params.update({
            'max_depth': 10,
            'n_estimators': 5000,
            'min_child_weight': 12,
            'nthread': 8,
            'tree_method': 'hist',
            'early_stopping_rounds': 500,
            'max_leaves': 1023
        })
        self.params.pop('gpu_id', None)
        self.params.pop('predictor', None)
        logger.info(f"{self.name}: 메모리 절약을 위해 파라미터 단순화 및 CPU 모드 전환")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """최종 완성 XGBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
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
            
            dtrain = xgb.DMatrix(
                X_train_clean, 
                label=y_train, 
                enable_categorical=False,
                feature_names=list(X_train_clean.columns)
            )
            
            evals = [(dtrain, 'train')]
            dval = None
            
            if X_val_clean is not None and y_val is not None:
                dval = xgb.DMatrix(
                    X_val_clean, 
                    label=y_val, 
                    enable_categorical=False,
                    feature_names=list(X_val_clean.columns)
                )
                evals.append((dval, 'valid'))
            
            early_stopping = self.params.get('early_stopping_rounds', 800)
            
            self.model = xgb.train(
                self.params,
                dtrain,
                evals=evals,
                early_stopping_rounds=early_stopping,
                verbose_eval=False
            )
            
            self.is_fitted = True
            
            del dtrain
            if dval is not None:
                del dval
            del X_train_clean
            if X_val_clean is not None:
                del X_val_clean
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션 이전 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
            
            dtest = xgb.DMatrix(
                X_processed, 
                enable_categorical=False,
                feature_names=list(X_processed.columns)
            )
            
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
                proba = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
            else:
                proba = self.model.predict(dtest)
            
            del dtest
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict(raw_pred)
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class FinalLogisticModel(BaseModel):
    """최종 완성 로지스틱 회귀 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn이 설치되지 않았습니다.")
            
        final_params = {
            'C': 0.03,
            'max_iter': 1200,
            'random_state': 42,
            'class_weight': 'balanced',
            'solver': 'liblinear',
            'penalty': 'l2',
            'tol': 1e-5,
            'fit_intercept': True,
            'warm_start': False,
            'dual': False
        }
        
        if params:
            final_params.update(params)
        
        super().__init__("FinalLogisticRegression", final_params)
        
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 2000
    
    def _simplify_for_memory(self):
        """메모리 부족 시 파라미터 단순화"""
        self.params.update({
            'C': 0.05,
            'max_iter': 800,
            'tol': 1e-4,
            'solver': 'liblinear'
        })
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: 메모리 절약을 위해 파라미터 단순화")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """최종 완성 로지스틱 회귀 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0)
            
            if len(X_train_clean) > 4000000:
                logger.info(f"대용량 데이터 감지, 샘플링 적용 ({len(X_train_clean):,} -> 4,000,000)")
                
                pos_indices = np.where(y_train == 1)[0]
                neg_indices = np.where(y_train == 0)[0]
                
                n_pos = len(pos_indices)
                n_neg_target = min(4000000 - n_pos, len(neg_indices))
                
                selected_neg = np.random.choice(neg_indices, n_neg_target, replace=False)
                selected_indices = np.concatenate([pos_indices, selected_neg])
                np.random.shuffle(selected_indices)
                
                X_train_clean = X_train_clean.iloc[selected_indices]
                y_train_sample = y_train.iloc[selected_indices]
            else:
                y_train_sample = y_train
            
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
            
            try:
                start_time = time.time()
                self.model.fit(X_train_clean, y_train_sample)
                training_time = time.time() - start_time
                
                logger.info(f"{self.name} 학습 완료 (소요시간: {training_time:.2f}초)")
                self.is_fitted = True
                
            except Exception as e:
                logger.warning(f"로지스틱 회귀 학습 실패: {e}")
                self._simplify_for_memory()
                self.model.fit(X_train_clean, y_train_sample)
                self.is_fitted = True
            
            del X_train_clean
            
            return self
        
        return self._memory_safe_fit(_fit_internal)
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션 이전 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        def _predict_internal(batch_X):
            X_processed = self._ensure_feature_consistency(batch_X)
            X_processed = X_processed.fillna(0)
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
            
            proba = self.model.predict_proba(X_processed)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba)
        
        return self._memory_safe_predict(_predict_internal, X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict(raw_pred)
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class ModelFactory:
    """최종 완성 CTR 모델 팩토리"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """최종 완성 모델 인스턴스 생성"""
        
        try:
            if model_type.lower() == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM이 설치되지 않았습니다.")
                return FinalLightGBMModel(kwargs.get('params'))
            
            elif model_type.lower() == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost가 설치되지 않았습니다.")
                return FinalXGBoostModel(kwargs.get('params'))
            
            elif model_type.lower() == 'logistic':
                return FinalLogisticModel(kwargs.get('params'))
            
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
                
        except Exception as e:
            logger.error(f"최종 완성 모델 생성 실패 ({model_type}): {e}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """사용 가능한 모델 타입 리스트"""
        available = []
        
        available.append('logistic')
        
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        
        logger.info(f"사용 가능한 모델: {available}")
        return available
    
    @staticmethod
    def get_final_models() -> List[str]:
        """최종 완성 모델 우선순위 리스트"""
        final_order = []
        
        if LIGHTGBM_AVAILABLE:
            final_order.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            final_order.append('xgboost')
            
        if SKLEARN_AVAILABLE:
            final_order.append('logistic')
        
        return final_order
    
    @staticmethod
    def select_models_by_memory_status() -> List[str]:
        """메모리 상태에 따른 모델 선택"""
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

LightGBMModel = FinalLightGBMModel
XGBoostModel = FinalXGBoostModel
LogisticModel = FinalLogisticModel