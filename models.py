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

# 안전한 PyTorch import 및 RTX 4060 Ti 최적화
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

# Psutil import 안전 처리
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
    """CTR 예측 전용 고급 캘리브레이션 클래스 - Combined Score 0.30+ 목표"""
    
    def __init__(self, target_ctr: float = 0.0201, method: str = 'auto'):
        self.target_ctr = target_ctr
        self.method = method
        self.calibrators = {}
        self.is_fitted = False
        self.best_method = None
        self.calibration_curve = None
        
        # Platt Scaling
        self.platt_scaler = None
        
        # Isotonic Regression
        self.isotonic_regressor = None
        
        # Temperature Scaling
        self.temperature = 1.0
        self.temperature_bias = 0.0
        
        # Bias Correction
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
        
        # 성능 지표
        self.calibration_scores = {}
        
    def fit_platt_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Platt Scaling 캘리브레이션 학습"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn을 사용할 수 없어 Platt Scaling을 건너뜁니다")
                return
                
            logger.info("Platt Scaling 캘리브레이션 학습 시작")
            
            # 로지스틱 회귀를 사용한 Platt Scaling
            self.platt_scaler = LogisticRegression()
            
            # 예측값을 로지트 공간으로 변환
            pred_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            logits = np.log(pred_clipped / (1 - pred_clipped))
            
            self.platt_scaler.fit(logits.reshape(-1, 1), y_true)
            
            # 성능 평가
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
            
            # 성능 평가
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
                
                # 로그 손실 계산
                log_loss_val = -np.mean(y_true * np.log(calibrated_probs) + (1 - y_true) * np.log(1 - calibrated_probs))
                
                # CTR 정렬 보정
                ctr_bias = abs(calibrated_probs.mean() - y_true.mean()) * 1000
                
                return log_loss_val + ctr_bias
            
            # 최적 온도 및 편향 찾기
            if SCIPY_AVAILABLE:
                from scipy.optimize import minimize
                result = minimize(
                    temperature_loss, 
                    x0=[1.0, 0.0], 
                    bounds=[(0.1, 10.0), (-2.0, 2.0)],
                    method='L-BFGS-B'
                )
                self.temperature = result.x[0]
                self.temperature_bias = result.x[1]
            else:
                # 간단한 그리드 서치
                best_loss = float('inf')
                best_temp = 1.0
                best_bias = 0.0
                
                for temp in np.logspace(-1, 1, 20):  # 0.1 to 10
                    for bias in np.linspace(-2, 2, 20):
                        loss = temperature_loss([temp, bias])
                        if loss < best_loss:
                            best_loss = loss
                            best_temp = temp
                            best_bias = bias
                
                self.temperature = best_temp
                self.temperature_bias = best_bias
            
            # 성능 평가
            calibrated_probs = self.apply_temperature_scaling(y_pred_proba)
            self.calibration_scores['temperature_scaling'] = self._evaluate_calibration(y_true, calibrated_probs)
            
            logger.info(f"Temperature Scaling 학습 완료 - 온도: {self.temperature:.3f}, 편향: {self.temperature_bias:.3f}")
            logger.info(f"Temperature Scaling 점수: {self.calibration_scores['temperature_scaling']:.4f}")
            
        except Exception as e:
            logger.error(f"Temperature Scaling 학습 실패: {e}")
            self.temperature = 1.0
            self.temperature_bias = 0.0
    
    def fit_bias_correction(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """편향 보정 학습"""
        try:
            logger.info("편향 보정 학습 시작")
            
            actual_ctr = y_true.mean()
            predicted_ctr = y_pred_proba.mean()
            
            # 가법적 편향 보정
            self.bias_correction = actual_ctr - predicted_ctr
            
            # 승법적 편향 보정
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
            methods = ['platt_scaling', 'isotonic_regression', 'temperature_scaling', 'bias_correction']
        
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        
        # 입력 검증
        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true와 y_pred_proba의 길이가 다릅니다")
        
        if len(y_true) == 0:
            raise ValueError("빈 배열은 처리할 수 없습니다")
        
        # 각 방법별 학습
        if 'platt_scaling' in methods:
            self.fit_platt_scaling(y_true, y_pred_proba)
        
        if 'isotonic_regression' in methods:
            self.fit_isotonic_regression(y_true, y_pred_proba)
        
        if 'temperature_scaling' in methods:
            self.fit_temperature_scaling(y_true, y_pred_proba)
        
        if 'bias_correction' in methods:
            self.fit_bias_correction(y_true, y_pred_proba)
        
        # 최적 방법 선택
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
    
    def apply_bias_correction(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """편향 보정 적용"""
        try:
            # 승법적 보정 후 가법적 보정
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
        elif method == 'bias_correction':
            return self.apply_bias_correction(y_pred_proba)
        else:
            logger.warning(f"알 수 없는 캘리브레이션 방법: {method}. 편향 보정 사용")
            return self.apply_bias_correction(y_pred_proba)
    
    def _evaluate_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """캘리브레이션 품질 평가"""
        try:
            # Expected Calibration Error (ECE) 계산
            ece = self._calculate_ece(y_true, y_pred_proba)
            
            # CTR 정렬도
            actual_ctr = y_true.mean()
            predicted_ctr = y_pred_proba.mean()
            ctr_alignment = 1.0 - abs(actual_ctr - predicted_ctr) / max(actual_ctr, 0.001)
            
            # 종합 점수 (낮은 ECE와 높은 CTR 정렬도가 좋음)
            calibration_score = 0.6 * (1.0 - ece) + 0.4 * ctr_alignment
            
            return max(0.0, calibration_score)
            
        except Exception as e:
            logger.warning(f"캘리브레이션 평가 실패: {e}")
            return 0.0
    
    def _calculate_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
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
            'available_methods': list(self.calibration_scores.keys())
        }

class MemoryMonitor:
    """메모리 모니터링 클래스 - 64GB RAM 환경 최적화"""
    
    def __init__(self, max_memory_gb: float = 50.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.max_memory_gb = max_memory_gb
        self.lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 3.0
        
        # RTX 4060 Ti + 64GB RAM 환경에 최적화된 메모리 임계값
        self.warning_threshold = max_memory_gb * 0.70
        self.critical_threshold = max_memory_gb * 0.80
        self.abort_threshold = max_memory_gb * 0.90
        
        logger.info(f"메모리 임계값 - 경고: {self.warning_threshold:.1f}GB, "
                   f"위험: {self.critical_threshold:.1f}GB, 중단: {self.abort_threshold:.1f}GB")
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
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
            return 40.0
        
        try:
            with self.lock:
                return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 40.0
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            return usage > self.critical_threshold or available < 12.0
        except Exception:
            return False
    
    def get_memory_status(self) -> Dict[str, Any]:
        """상세한 메모리 상태 반환"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            if usage > self.abort_threshold or available < 5:
                level = "abort"
            elif usage > self.critical_threshold or available < 12:
                level = "critical"
            elif usage > self.warning_threshold or available < 20:
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
                'available_gb': 40.0,
                'level': 'normal',
                'should_cleanup': False,
                'should_simplify': False,
                'should_reduce_batch': False
            }
    
    def force_memory_cleanup(self, intensive: bool = False):
        """강화된 메모리 정리"""
        try:
            initial_memory = self.get_memory_usage()
            
            cleanup_rounds = 12 if intensive else 8
            sleep_time = 0.3 if intensive else 0.2
            
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                if i % 4 == 0:
                    current_memory = self.get_memory_usage()
                    if initial_memory - current_memory > 3.0:
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
    """모든 모델의 기본 클래스 - 캘리브레이션 지원"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.calibrator = None
        self.is_calibrated = False
        self.prediction_diversity_threshold = 1000
        
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
                         method: str = 'auto', cv_folds: int = 3):
        """캘리브레이션 적용"""
        try:
            logger.info(f"{self.name} 모델에 캘리브레이션 적용: {method}")
            
            if not self.is_fitted:
                logger.warning("모델이 학습되지 않아 캘리브레이션을 건너뜁니다")
                return
            
            # 검증 데이터에 대한 예측
            raw_predictions = self.predict_proba_raw(X_val)
            
            # 캘리브레이터 생성 및 학습
            self.calibrator = CTRCalibrator()
            self.calibrator.fit(y_val.values, raw_predictions)
            
            self.is_calibrated = True
            
            # 캘리브레이션 효과 확인
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
        """캘리브레이션 이전 원본 예측 - 하위클래스에서 구현"""
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
                    batch_size = 1000
                elif memory_status['level'] == 'critical':
                    batch_size = 5000
                elif memory_status['level'] == 'warning':
                    batch_size = 15000
                else:
                    batch_size = 50000
            
            n_samples = len(X)
            predictions = []
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X.iloc[i:batch_end]
                
                try:
                    batch_pred = predict_function(batch_X)
                    predictions.append(batch_pred)
                    
                    if (i // batch_size) % 5 == 0:
                        gc.collect()
                    
                    if self.memory_monitor.check_memory_pressure():
                        logger.warning(f"{self.name}: 예측 중 메모리 압박 감지, 정리 수행")
                        self.memory_monitor.force_memory_cleanup()
                        
                except Exception as e:
                    logger.warning(f"{self.name}: 배치 {i}-{batch_end} 예측 실패: {e}")
                    batch_size_actual = batch_end - i
                    predictions.append(np.full(batch_size_actual, 0.0201))
            
            return np.concatenate(predictions) if predictions else np.array([])
            
        except Exception as e:
            logger.error(f"{self.name} 메모리 안전 예측 실패: {e}")
            return np.full(len(X), 0.0201)
    
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

class HighPerformanceLightGBMModel(BaseModel):
    """메모리 최적화된 고성능 LightGBM 모델 - 캘리브레이션 지원"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM이 설치되지 않았습니다.")
            
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'min_child_samples': 200,
            'min_child_weight': 10,
            'lambda_l1': 2.0,
            'lambda_l2': 2.0,
            'max_depth': 12,
            'verbose': -1,
            'random_state': Config.RANDOM_STATE,
            'n_estimators': 2000,
            'early_stopping_rounds': 150,
            'scale_pos_weight': 49.0,
            'force_row_wise': True,
            'max_bin': 255,
            'num_threads': 8,
            'device_type': 'cpu',
            'min_data_in_leaf': 100,
            'feature_fraction_bynode': 0.8
        }
        
        if params:
            default_params.update(params)
        
        super().__init__("MemoryOptimizedLightGBM", default_params)
        self.prediction_diversity_threshold = 1500
    
    def _simplify_for_memory(self):
        """메모리 부족 시 파라미터 단순화"""
        self.params.update({
            'num_leaves': 127,
            'max_depth': 8,
            'n_estimators': 1000,
            'min_child_samples': 300,
            'num_threads': 4
        })
        logger.info(f"{self.name}: 메모리 절약을 위해 파라미터 단순화")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """메모리 최적화된 LightGBM 모델 학습"""
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
            early_stopping = self.params.get('early_stopping_rounds', 150)
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
            
            def memory_callback(env):
                if env.iteration % 50 == 0:
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

# 다른 모델들도 동일한 패턴으로 구현 (XGBoost, CatBoost, DeepCTR, Logistic)
class HighPerformanceXGBoostModel(BaseModel):
    """메모리 최적화된 고성능 XGBoost 모델 - 캘리브레이션 지원"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost가 설치되지 않았습니다.")
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'colsample_bynode': 0.8,
            'min_child_weight': 15,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'scale_pos_weight': 49.0,
            'random_state': Config.RANDOM_STATE,
            'n_estimators': 2000,
            'early_stopping_rounds': 150,
            'max_bin': 255,
            'nthread': 8,
            'grow_policy': 'lossguide',
            'max_leaves': 255,
            'gamma': 0.1
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
                    logger.info("XGBoost GPU 모드 활성화")
                else:
                    logger.info("메모리 부족으로 XGBoost CPU 모드 사용")
                    
            except Exception as e:
                logger.warning(f"GPU 설정 실패, CPU 모드 사용: {e}")
        
        if params:
            default_params.update(params)
        
        super().__init__("MemoryOptimizedXGBoost", default_params)
        self.prediction_diversity_threshold = 1500
    
    def _simplify_for_memory(self):
        """메모리 부족 시 파라미터 단순화"""
        self.params.update({
            'max_depth': 6,
            'max_leaves': 127,
            'n_estimators': 1000,
            'min_child_weight': 20,
            'nthread': 4,
            'tree_method': 'hist',
        })
        self.params.pop('gpu_id', None)
        self.params.pop('predictor', None)
        logger.info(f"{self.name}: 메모리 절약을 위해 파라미터 단순화 및 CPU 모드 전환")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """메모리 최적화된 XGBoost 모델 학습"""
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
            
            early_stopping = self.params.get('early_stopping_rounds', 150)
            
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

class HighPerformanceCatBoostModel(BaseModel):
    """메모리 최적화된 고성능 CatBoost 모델 - 캘리브레이션 지원"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost가 설치되지 않았습니다.")
        
        default_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'task_type': 'CPU',
            'depth': 8,
            'learning_rate': 0.03,
            'l2_leaf_reg': 10,
            'iterations': 2000,
            'random_seed': Config.RANDOM_STATE,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'max_ctr_complexity': 2,
            'thread_count': 8,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1.0,
            'leaf_estimation_iterations': 8,
            'leaf_estimation_method': 'Newton',
            'grow_policy': 'Lossguide',
            'max_leaves': 255,
            'min_data_in_leaf': 80,
            'od_wait': 150,
            'od_type': 'IncToDec'
        }
        
        if rtx_4060ti_detected and TORCH_AVAILABLE:
            try:
                memory_monitor_temp = MemoryMonitor()
                memory_status = memory_monitor_temp.get_memory_status()
                
                if memory_status['level'] == 'normal':
                    default_params.update({
                        'task_type': 'GPU',
                        'devices': '0'
                    })
                    logger.info("CatBoost GPU 모드 활성화")
                else:
                    logger.info("메모리 부족으로 CatBoost CPU 모드 사용")
                    
            except Exception as e:
                logger.warning(f"GPU 설정 실패, CPU 모드 사용: {e}")
        
        if params:
            default_params.update(params)
        
        super().__init__("MemoryOptimizedCatBoost", default_params)
        self.prediction_diversity_threshold = 1500
        
        init_params = {k: v for k, v in self.params.items() 
                      if k not in ['early_stopping_rounds', 'use_best_model', 'eval_set', 'od_wait', 'od_type']}
        
        try:
            self.model = CatBoostClassifier(**init_params)
        except Exception as e:
            logger.error(f"CatBoost 모델 초기화 실패: {e}")
            if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                logger.info("GPU 초기화 실패, CPU로 재시도")
                self.params['task_type'] = 'CPU'
                self.params.pop('devices', None)
                init_params = {k: v for k, v in self.params.items() 
                              if k not in ['early_stopping_rounds', 'use_best_model', 'eval_set', 'od_wait', 'od_type']}
                self.model = CatBoostClassifier(**init_params)
            else:
                raise
    
    def _simplify_for_memory(self):
        """메모리 부족 시 파라미터 단순화"""
        self.params.update({
            'depth': 6,
            'max_leaves': 127,
            'iterations': 1000,
            'min_data_in_leaf': 100,
            'thread_count': 4,
            'task_type': 'CPU',
            'max_ctr_complexity': 1
        })
        self.params.pop('devices', None)
        
        init_params = {k: v for k, v in self.params.items() 
                      if k not in ['early_stopping_rounds', 'use_best_model', 'eval_set', 'od_wait', 'od_type']}
        self.model = CatBoostClassifier(**init_params)
        logger.info(f"{self.name}: 메모리 절약을 위해 파라미터 단순화 및 CPU 모드 전환")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """메모리 최적화된 CatBoost 모델 학습"""
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
            
            fit_params = {
                'X': X_train_clean,
                'y': y_train,
                'verbose': False,
                'plot': False
            }
            
            if X_val_clean is not None and y_val is not None:
                fit_params['eval_set'] = (X_val_clean, y_val)
                fit_params['use_best_model'] = True
            
            try:
                self.model.fit(**fit_params)
                self.is_fitted = True
            except Exception as e:
                logger.error(f"CatBoost 학습 실패: {e}")
                
                if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                    logger.info("GPU 학습 실패, CPU로 재시도")
                    self._simplify_for_memory()
                    self.model.fit(**fit_params)
                    self.is_fitted = True
                else:
                    logger.info("단순화된 학습 시도")
                    simple_fit_params = {
                        'X': X_train_clean,
                        'y': y_train,
                        'verbose': False
                    }
                    self.model.fit(**simple_fit_params)
                    self.is_fitted = True
            
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

# DeepCTR과 Logistic 모델도 유사하게 구현...
class HighPerformanceLogisticModel(BaseModel):
    """메모리 최적화된 고성능 로지스틱 회귀 모델 - 캘리브레이션 지원"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn이 설치되지 않았습니다.")
            
        default_params = {
            'C': 0.1,
            'max_iter': 2000,
            'random_state': Config.RANDOM_STATE,
            'class_weight': 'balanced',
            'solver': 'liblinear',
            'penalty': 'l2',
            'tol': 1e-4,
            'fit_intercept': True
        }
        if params:
            default_params.update(params)
        super().__init__("MemoryOptimizedLogisticRegression", default_params)
        
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 1000
    
    def _simplify_for_memory(self):
        """메모리 부족 시 파라미터 단순화"""
        self.params.update({
            'C': 1.0,
            'max_iter': 500,
            'tol': 1e-3,
            'solver': 'liblinear'
        })
        self.model = LogisticRegression(**self.params)
        logger.info(f"{self.name}: 메모리 절약을 위해 파라미터 단순화")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """메모리 최적화된 로지스틱 회귀 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0)
            
            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64']:
                    X_train_clean[col] = X_train_clean[col].astype('float32')
            
            try:
                self.model.fit(X_train_clean, y_train)
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"로지스틱 회귀 학습 실패: {e}")
                self._simplify_for_memory()
                self.model.fit(X_train_clean, y_train)
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

# RTX4060Ti 최적화 DeepCTR 모델도 구현 (기존과 동일하되 캘리브레이션 지원 추가)
class RTX4060TiOptimizedDeepCTRModel(BaseModel):
    """RTX 4060 Ti 16GB + 64GB RAM 최적화 DeepCTR 모델 - 캘리브레이션 지원"""
    
    def __init__(self, input_dim: int, params: Dict[str, Any] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 설치되지 않았습니다.")
            
        BaseModel.__init__(self, "RTX4060TiOptimizedDeepCTR", params)
        
        memory_monitor_temp = MemoryMonitor()
        memory_status = memory_monitor_temp.get_memory_status()
        
        if memory_status['level'] in ['critical', 'abort']:
            default_params = {
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 512,
                'epochs': 30,
                'patience': 10,
                'use_batch_norm': True,
                'activation': 'relu',
                'use_residual': False,
                'use_attention': False,
                'use_focal_loss': False
            }
        elif memory_status['level'] == 'warning':
            default_params = {
                'hidden_dims': [512, 256, 128, 64],
                'dropout_rate': 0.25,
                'learning_rate': 0.0008,
                'batch_size': 1024,
                'epochs': 50,
                'patience': 15,
                'use_batch_norm': True,
                'activation': 'gelu',
                'use_residual': True,
                'use_attention': False,
                'use_focal_loss': True
            }
        else:
            default_params = {
                'hidden_dims': [1024, 512, 256, 128, 64] if rtx_4060ti_detected else [512, 256, 128, 64],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 2048 if rtx_4060ti_detected else 1024,
                'epochs': 80 if rtx_4060ti_detected else 50,
                'patience': 25,
                'use_batch_norm': True,
                'activation': 'gelu',
                'use_residual': True,
                'use_attention': rtx_4060ti_detected,
                'use_focal_loss': True,
                'focal_loss_alpha': 0.3,
                'focal_loss_gamma': 2.0,
                'gradient_clip_val': 1.0,
                'weight_decay': 1e-5
            }
        
        if params:
            default_params.update(params)
        self.params = default_params
        
        self.input_dim = input_dim
        self.device = 'cpu'
        self.gpu_available = False
        self.rtx_optimized = False
        
        if TORCH_AVAILABLE and memory_status['level'] not in ['abort']:
            try:
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(1000, 1000).cuda()
                    test_result = test_tensor.sum().item()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    torch.cuda.set_per_process_memory_fraction(0.8)
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
                    self.device = 'cuda:0'
                    self.gpu_available = True
                    self.rtx_optimized = rtx_4060ti_detected
                    
                    logger.info(f"DeepCTR RTX 4060 Ti 최적화 GPU 설정 완료")
                else:
                    logger.info("DeepCTR CUDA 사용 불가능, CPU 모드")
            except Exception as e:
                logger.warning(f"DeepCTR GPU 설정 실패, CPU 사용: {e}")
                self.device = 'cpu'
                self.gpu_available = False
        
        try:
            self.network = self._build_memory_optimized_network()
            self.optimizer = None
            
            self.scaler = None
            if AMP_AVAILABLE and self.gpu_available and memory_status['level'] == 'normal':
                try:
                    self.scaler = GradScaler()
                    logger.info("DeepCTR Mixed Precision 활성화")
                except:
                    self.scaler = None
            
            if TORCH_AVAILABLE:
                pos_weight = torch.tensor([49.0], device=self.device)
                self.criterion = self._get_memory_efficient_loss(pos_weight)
                self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.2)
                self.to(self.device)
                
        except Exception as e:
            logger.error(f"DeepCTR 모델 초기화 실패: {e}")
            if self.gpu_available:
                logger.info("DeepCTR CPU 모드로 재시도")
                self.device = 'cpu'
                self.gpu_available = False
                self.rtx_optimized = False
                self.network = self._build_memory_optimized_network()
                pos_weight = torch.tensor([49.0])
                self.criterion = self._get_memory_efficient_loss(pos_weight)
                self.temperature = nn.Parameter(torch.ones(1) * 1.2)
                self.to(self.device)
            else:
                raise
    
    def _build_memory_optimized_network(self):
        """메모리 최적화된 네트워크 구조"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
        
        try:
            hidden_dims = self.params['hidden_dims']
            dropout_rate = self.params['dropout_rate']
            use_batch_norm = self.params.get('use_batch_norm', True)
            activation = self.params.get('activation', 'gelu')
            use_residual = self.params.get('use_residual', True)
            use_attention = self.params.get('use_attention', False)
            
            layers = []
            prev_dim = self.input_dim
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(prev_dim))
            
            for i, hidden_dim in enumerate(hidden_dims):
                linear = nn.Linear(prev_dim, hidden_dim)
                
                if self.rtx_optimized:
                    nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                
                layers.append(linear)
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'swish':
                    layers.append(nn.SiLU(inplace=True))
                
                if use_residual and i > 0 and prev_dim == hidden_dim and hidden_dim >= 128:
                    layers.append(ResidualConnection(hidden_dim))
                
                if use_attention and i == len(hidden_dims) // 2 and hidden_dim >= 256:
                    layers.append(SelfAttentionLayer(hidden_dim))
                
                if i < len(hidden_dims) - 1:
                    layers.append(nn.Dropout(dropout_rate))
                
                prev_dim = hidden_dim
            
            output_layer = nn.Linear(prev_dim, 1)
            if self.rtx_optimized:
                nn.init.kaiming_uniform_(output_layer.weight, gain=0.1)
            else:
                nn.init.xavier_uniform_(output_layer.weight, gain=0.1)
            nn.init.zeros_(output_layer.bias)
            
            layers.append(output_layer)
            
            network = nn.Sequential(*layers)
            
            if self.rtx_optimized and self.memory_monitor.get_memory_status()['level'] == 'normal':
                try:
                    network = torch.jit.script(network)
                    logger.info("DeepCTR JIT 컴파일 활성화")
                except Exception as e:
                    logger.warning(f"DeepCTR JIT 컴파일 실패: {e}")
            
            return network
            
        except Exception as e:
            logger.error(f"DeepCTR 네트워크 구조 생성 실패: {e}")
            raise
    
    def _get_memory_efficient_loss(self, pos_weight):
        """메모리 효율적인 손실함수"""
        if self.params.get('use_focal_loss', False):
            return self._focal_loss_optimized
        else:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def _focal_loss_optimized(self, inputs, targets):
        """최적화된 Focal Loss"""
        alpha = self.params.get('focal_loss_alpha', 0.3)
        gamma = self.params.get('focal_loss_gamma', 2.0)
        
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        
        return focal_loss.mean()
    
    def _simplify_for_memory(self):
        """메모리 부족 시 모델 단순화"""
        self.params.update({
            'hidden_dims': [256, 128, 64],
            'batch_size': 256,
            'epochs': 20,
            'use_attention': False,
            'use_residual': False,
            'dropout_rate': 0.4
        })
        
        if self.gpu_available:
            self.device = 'cpu'
            self.gpu_available = False
            self.rtx_optimized = False
            self.scaler = None
        
        self.network = self._build_memory_optimized_network()
        pos_weight = torch.tensor([49.0], device=self.device)
        self.criterion = self._get_memory_efficient_loss(pos_weight)
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.2)
        self.to(self.device)
        
        logger.info(f"{self.name}: 메모리 절약을 위해 모델 단순화")
    
    def to(self, device):
        """디바이스 이동"""
        if TORCH_AVAILABLE:
            try:
                self.network = self.network.to(device)
                if hasattr(self, 'temperature'):
                    self.temperature = self.temperature.to(device)
                if hasattr(self, 'criterion') and hasattr(self.criterion, 'to'):
                    self.criterion = self.criterion.to(device)
                self.device = device
            except Exception as e:
                logger.warning(f"DeepCTR 디바이스 이동 실패: {e}")
                self.device = 'cpu'
                self.gpu_available = False
                self.rtx_optimized = False
    
    def train(self, mode=True):
        """학습 모드 설정"""
        if TORCH_AVAILABLE and hasattr(self, 'network'):
            self.network.train(mode)
    
    def eval(self):
        """평가 모드 설정"""
        if TORCH_AVAILABLE and hasattr(self, 'network'):
            self.network.eval()
    
    def parameters(self):
        """모델 파라미터 반환"""
        if TORCH_AVAILABLE and hasattr(self, 'network'):
            params = list(self.network.parameters())
            if hasattr(self, 'temperature'):
                params.append(self.temperature)
            return params
        return []
    
    def forward(self, x):
        """순전파"""
        if TORCH_AVAILABLE and hasattr(self, 'network'):
            return self.network(x).squeeze(-1)
        else:
            raise RuntimeError("PyTorch가 사용 불가능합니다")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """메모리 최적화된 DeepCTR 모델 학습"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
            
        logger.info(f"{self.name} 모델 학습 시작 (Device: {self.device}, RTX최적화: {self.rtx_optimized})")
        
        def _fit_internal():
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0)
            if X_val is not None:
                X_val_clean = X_val.fillna(0)
            else:
                X_val_clean = None
            
            X_train_values = X_train_clean.values.astype('float32')
            if X_val_clean is not None:
                X_val_values = X_val_clean.values.astype('float32')
            
            mean = X_train_values.mean(axis=0, keepdims=True)
            std = X_train_values.std(axis=0, keepdims=True) + 1e-8
            X_train_values = (X_train_values - mean) / std
            if X_val_clean is not None:
                X_val_values = (X_val_values - mean) / std
            
            self.normalization_params = {'mean': mean, 'std': std}
            
            self.optimizer = optim.AdamW(
                self.parameters(), 
                lr=self.params['learning_rate'],
                weight_decay=self.params.get('weight_decay', 1e-5),
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
            total_steps = len(X_train) // self.params['batch_size'] * self.params['epochs']
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.params['learning_rate'],
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
            batch_size = self.params['batch_size']
            
            X_train_tensor = torch.FloatTensor(X_train_values)
            y_train_tensor = torch.FloatTensor(y_train.values)
            
            try:
                X_train_tensor = X_train_tensor.to(self.device)
                y_train_tensor = y_train_tensor.to(self.device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning("GPU 메모리 부족, CPU 모드로 전환")
                    self.device = 'cpu'
                    self.gpu_available = False
                    self.rtx_optimized = False
                    self.scaler = None
                    self.to('cpu')
                    X_train_tensor = X_train_tensor.to(self.device)
                    y_train_tensor = y_train_tensor.to(self.device)
                else:
                    raise
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = TorchDataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,
                pin_memory=self.gpu_available and batch_size <= 1024,
                drop_last=True
            )
            
            val_loader = None
            if X_val_clean is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val_values).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = TorchDataLoader(
                    val_dataset, 
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=False
                )
            
            best_val_loss = float('inf')
            patience_counter = 0
            max_epochs = self.params['epochs']
            
            for epoch in range(max_epochs):
                memory_status = self.memory_monitor.get_memory_status()
                if memory_status['should_cleanup']:
                    self.memory_monitor.force_memory_cleanup()
                
                if memory_status['should_simplify'] and not hasattr(self, '_already_simplified'):
                    logger.warning("학습 중 메모리 부족 감지, 모델 단순화")
                    self._simplify_for_memory()
                    self._already_simplified = True
                    break
                
                self.train()
                train_loss = 0.0
                batch_count = 0
                
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    try:
                        self.optimizer.zero_grad()
                        
                        if self.scaler is not None and AMP_AVAILABLE and self.rtx_optimized:
                            with autocast():
                                logits = self.forward(batch_X)
                                loss = self.criterion(logits, batch_y)
                            
                            self.scaler.scale(loss).backward()
                            
                            if self.params.get('gradient_clip_val', 0) > 0:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.params['gradient_clip_val'])
                            
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            logits = self.forward(batch_X)
                            loss = self.criterion(logits, batch_y)
                            
                            loss.backward()
                            
                            if self.params.get('gradient_clip_val', 0) > 0:
                                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.params['gradient_clip_val'])
                            
                            self.optimizer.step()
                        
                        scheduler.step()
                        
                        train_loss += loss.item()
                        batch_count += 1
                        
                        if self.rtx_optimized and batch_idx % 50 == 0:
                            torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            logger.warning("배치 처리 중 메모리 부족")
                            torch.cuda.empty_cache() if self.gpu_available else None
                            gc.collect()
                            continue
                        else:
                            logger.warning(f"배치 학습 실패: {e}")
                            continue
                
                if batch_count == 0:
                    logger.error("모든 배치 학습이 실패했습니다")
                    break
                    
                train_loss /= batch_count
                
                val_loss = train_loss
                if val_loader is not None:
                    self.eval()
                    val_loss = 0.0
                    val_batch_count = 0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            try:
                                if self.scaler is not None and AMP_AVAILABLE and self.rtx_optimized:
                                    with autocast():
                                        logits = self.forward(batch_X)
                                        loss = self.criterion(logits, batch_y)
                                else:
                                    logits = self.forward(batch_X)
                                    loss = self.criterion(logits, batch_y)
                                
                                val_loss += loss.item()
                                val_batch_count += 1
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    torch.cuda.empty_cache() if self.gpu_available else None
                                    continue
                                else:
                                    logger.warning(f"검증 배치 실패: {e}")
                                    continue
                    
                    if val_batch_count > 0:
                        val_loss /= val_batch_count
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= self.params['patience']:
                            logger.info(f"DeepCTR 조기 종료: epoch {epoch + 1}")
                            break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"DeepCTR Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                if self.rtx_optimized and (epoch + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                elif (epoch + 1) % 10 == 0:
                    gc.collect()
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
            del X_train_tensor, y_train_tensor
            if val_loader is not None:
                del X_val_tensor, y_val_tensor
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
            
            X_values = X_processed.values.astype('float32')
            
            if hasattr(self, 'normalization_params'):
                X_values = (X_values - self.normalization_params['mean']) / self.normalization_params['std']
            
            self.eval()
            
            try:
                X_tensor = torch.FloatTensor(X_values).to(self.device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning("예측 중 GPU 메모리 부족, CPU 사용")
                    X_tensor = torch.FloatTensor(X_values).to('cpu')
                    self.network = self.network.to('cpu')
                    self.temperature = self.temperature.to('cpu')
                else:
                    raise
            
            with torch.no_grad():
                try:
                    if self.scaler is not None and AMP_AVAILABLE and self.rtx_optimized:
                        with autocast():
                            logits = self.forward(X_tensor)
                            proba = torch.sigmoid(logits / self.temperature)
                    else:
                        logits = self.forward(X_tensor)
                        proba = torch.sigmoid(logits / self.temperature)
                    
                    proba_np = proba.cpu().numpy()
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning("예측 중 메모리 부족, 기본값 반환")
                        return np.full(len(batch_X), 0.0201)
                    else:
                        raise
            
            if self.device != 'cpu':
                try:
                    self.network = self.network.to(self.device)
                    self.temperature = self.temperature.to(self.device)
                except:
                    pass
            
            proba_np = np.clip(proba_np, 1e-15, 1 - 1e-15)
            return self._enhance_prediction_diversity(proba_np)
        
        return self._memory_safe_predict(_predict_internal, X, batch_size=1024)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """캘리브레이션이 적용된 확률 예측"""
        try:
            raw_pred = self.predict_proba_raw(X)
            
            if self.is_calibrated and self.calibrator is not None:
                try:
                    calibrated_pred = self.calibrator.predict(raw_pred)
                    return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
                except:
                    pass
            
            return raw_pred
            
        finally:
            if self.rtx_optimized:
                torch.cuda.empty_cache()

# PyTorch 보조 클래스들
if TORCH_AVAILABLE:
    class ResidualConnection(nn.Module):
        """잔차 연결 레이어"""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            
        def forward(self, x):
            return x
    
    class SelfAttentionLayer(nn.Module):
        """셀프 어텐션 레이어"""
        def __init__(self, dim, num_heads=4):
            super().__init__()
            self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.layer_norm = nn.LayerNorm(dim)
            
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            try:
                attn_out, _ = self.attention(x, x, x)
                x = self.layer_norm(x + attn_out)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    pass
                else:
                    raise
            
            if x.size(1) == 1:
                x = x.squeeze(1)
            
            return x

class ModelFactory:
    """메모리 최적화된 CTR 특화 모델 팩토리"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """모델 타입에 따라 메모리 최적화된 모델 인스턴스 생성"""
        
        try:
            if model_type.lower() == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM이 설치되지 않았습니다.")
                return HighPerformanceLightGBMModel(kwargs.get('params'))
            
            elif model_type.lower() == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost가 설치되지 않았습니다.")
                return HighPerformanceXGBoostModel(kwargs.get('params'))
            
            elif model_type.lower() == 'catboost':
                if not CATBOOST_AVAILABLE:
                    raise ImportError("CatBoost가 설치되지 않았습니다.")
                return HighPerformanceCatBoostModel(kwargs.get('params'))
            
            elif model_type.lower() == 'deepctr':
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch가 설치되지 않았습니다.")
                input_dim = kwargs.get('input_dim')
                if input_dim is None:
                    raise ValueError("DeepCTR 모델에는 input_dim이 필요합니다.")
                return RTX4060TiOptimizedDeepCTRModel(input_dim, kwargs.get('params'))
            
            elif model_type.lower() == 'logistic':
                return HighPerformanceLogisticModel(kwargs.get('params'))
            
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
                
        except Exception as e:
            logger.error(f"메모리 최적화 모델 생성 실패 ({model_type}): {e}")
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
        if CATBOOST_AVAILABLE:
            available.append('catboost')
        if TORCH_AVAILABLE:
            available.append('deepctr')
        
        if not available:
            logger.error("사용 가능한 모델이 없습니다.")
            available = ['logistic']
        
        logger.info(f"사용 가능한 모델: {available}")
        return available
    
    @staticmethod
    def get_memory_optimized_models() -> List[str]:
        """메모리 최적화된 모델 우선순위 리스트"""
        memory_efficient_order = []
        
        if SKLEARN_AVAILABLE:
            memory_efficient_order.append('logistic')
        
        if LIGHTGBM_AVAILABLE:
            memory_efficient_order.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            memory_efficient_order.append('xgboost')
        
        if CATBOOST_AVAILABLE:
            memory_efficient_order.append('catboost')
        
        if TORCH_AVAILABLE:
            memory_efficient_order.append('deepctr')
        
        return memory_efficient_order
    
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
            models = ['logistic']
            if LIGHTGBM_AVAILABLE:
                models.append('lightgbm')
            if XGBOOST_AVAILABLE:
                models.append('xgboost')
            return models
        else:
            return ModelFactory.get_available_models()

# 기존 코드와의 호환성을 위한 별칭
LightGBMModel = HighPerformanceLightGBMModel
XGBoostModel = HighPerformanceXGBoostModel
CatBoostModel = HighPerformanceCatBoostModel
DeepCTRModel = RTX4060TiOptimizedDeepCTRModel
LogisticModel = HighPerformanceLogisticModel