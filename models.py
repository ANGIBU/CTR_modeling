# models.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import pickle
import gc

# 트리 기반 모델 - 안전한 import
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

# 신경망 모델 - 안전한 import 강화
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
    
    # GPU 사용 가능성 엄격한 테스트
    gpu_available = False
    if torch.cuda.is_available():
        try:
            # 실제 GPU 메모리 할당 테스트
            test_tensor = torch.zeros(1000, 1000).cuda()
            test_result = test_tensor.sum()
            del test_tensor
            torch.cuda.empty_cache()
            gpu_available = True
        except Exception as e:
            logging.warning(f"GPU 테스트 실패: {e}. CPU 전용 모드")
            gpu_available = False
    
    TORCH_AVAILABLE = True
    
    if TORCH_AVAILABLE:
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
        
        # Mixed Precision 모듈 - 더 안전하게
        try:
            if gpu_available and hasattr(torch.cuda, 'amp'):
                from torch.cuda.amp import GradScaler, autocast
                AMP_AVAILABLE = True
            else:
                AMP_AVAILABLE = False
        except (ImportError, AttributeError):
            AMP_AVAILABLE = False
            
except ImportError:
    TORCH_AVAILABLE = False
    AMP_AVAILABLE = False
    logging.warning("PyTorch가 설치되지 않았습니다. DeepCTR 모델을 사용할 수 없습니다.")

# Calibration 모듈
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn이 설치되지 않았습니다.")

# 기타 모델
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, log_loss
except ImportError:
    pass

from config import Config

logger = logging.getLogger(__name__)

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
    
    def apply_calibration(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         method: str = 'platt', cv_folds: int = 3):
        """예측 확률 보정 적용"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn이 없어 calibration을 적용할 수 없습니다")
            return
            
        try:
            # 기본 예측 수행
            train_pred = self.predict_proba_raw(X_train)
            
            # Calibration 방법 선택
            if method == 'platt':
                calibrator = CalibratedClassifierCV(
                    estimator=None, 
                    method='sigmoid', 
                    cv=cv_folds
                )
            elif method == 'isotonic':
                calibrator = CalibratedClassifierCV(
                    estimator=None, 
                    method='isotonic', 
                    cv=cv_folds
                )
            else:
                logger.warning(f"지원하지 않는 calibration 방법: {method}")
                return
            
            # Calibration 학습
            calibrator.fit(train_pred.reshape(-1, 1), y_train)
            self.calibrator = calibrator
            self.is_calibrated = True
            
            logger.info(f"{self.name} {method} calibration 적용 완료")
            
        except Exception as e:
            logger.error(f"Calibration 적용 실패 ({self.name}): {str(e)}")
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 확률 예측"""
        return self.predict_proba(X)
    
    def _ensure_feature_consistency(self, X: pd.DataFrame) -> pd.DataFrame:
        """피처 일관성 보장"""
        if self.feature_names is None:
            return X
        
        try:
            # 누락된 피처 추가
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0.0
            
            # 필요한 피처만 선택
            X = X[self.feature_names]
            return X
        except Exception as e:
            logger.warning(f"피처 일관성 보장 실패: {str(e)}")
            return X

class LightGBMModel(BaseModel):
    """LightGBM 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM이 설치되지 않았습니다.")
            
        default_params = Config.LGBM_PARAMS.copy()
        if params:
            default_params.update(params)
        
        # 파라미터 안전성 검증
        default_params = self._validate_params(default_params)
        
        super().__init__("LightGBM", default_params)
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """LightGBM 파라미터 검증"""
        safe_params = params.copy()
        
        # 필수 파라미터 확인
        if 'objective' not in safe_params:
            safe_params['objective'] = 'binary'
        if 'metric' not in safe_params:
            safe_params['metric'] = 'binary_logloss'
        if 'verbose' not in safe_params:
            safe_params['verbose'] = -1
        
        # 충돌하는 파라미터 제거
        if 'is_unbalance' in safe_params and 'scale_pos_weight' in safe_params:
            safe_params.pop('is_unbalance', None)
        
        # 메모리 안전성을 위한 파라미터 조정
        safe_params['num_leaves'] = min(safe_params.get('num_leaves', 63), 127)
        safe_params['max_bin'] = min(safe_params.get('max_bin', 255), 255)
        safe_params['num_threads'] = min(safe_params.get('num_threads', 4), 6)
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """LightGBM 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        try:
            self.feature_names = list(X_train.columns)
            
            # 데이터 검증
            if X_train.isnull().sum().sum() > 0:
                logger.warning("학습 데이터에 결측치가 있습니다. 0으로 대체합니다.")
                X_train = X_train.fillna(0)
            
            if X_val is not None and X_val.isnull().sum().sum() > 0:
                logger.warning("검증 데이터에 결측치가 있습니다. 0으로 대체합니다.")
                X_val = X_val.fillna(0)
            
            # LightGBM 데이터셋 생성
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
                valid_sets.append(val_data)
                valid_names.append('valid')
            
            # 콜백 설정
            callbacks = []
            early_stopping = self.params.get('early_stopping_rounds', 100)
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
            
            # 학습 실행
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
            # 메모리 정리
            del train_data
            if 'val_data' in locals():
                del val_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"LightGBM 학습 실패: {str(e)}")
            # 메모리 정리
            gc.collect()
            raise
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            
            # 결측치 처리
            X_processed = X_processed.fillna(0)
            
            # 예측 수행
            num_iteration = getattr(self.model, 'best_iteration', None)
            proba = self.model.predict(X_processed, num_iteration=num_iteration)
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return proba
        except Exception as e:
            logger.error(f"LightGBM 예측 실패: {str(e)}")
            return np.full(len(X), Config.CALIBRATION_CONFIG['target_ctr'])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """calibration이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict_proba(raw_pred.reshape(-1, 1))[:, 1]
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class XGBoostModel(BaseModel):
    """XGBoost 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost가 설치되지 않았습니다.")
            
        default_params = Config.XGB_PARAMS.copy()
        if params:
            default_params.update(params)
        
        # GPU 사용 가능성 재확인
        gpu_available = False
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(100, 100).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    gpu_available = True
            except:
                gpu_available = False
        
        # GPU 파라미터 안전하게 설정
        if gpu_available:
            default_params['tree_method'] = 'gpu_hist'
            default_params['gpu_id'] = 0
        else:
            default_params['tree_method'] = 'hist'
            default_params.pop('gpu_id', None)
        
        # 파라미터 검증
        default_params = self._validate_params(default_params)
        
        super().__init__("XGBoost", default_params)
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """XGBoost 파라미터 검증"""
        safe_params = params.copy()
        
        # 필수 파라미터 확인
        if 'objective' not in safe_params:
            safe_params['objective'] = 'binary:logistic'
        if 'eval_metric' not in safe_params:
            safe_params['eval_metric'] = 'logloss'
        
        # 메모리 안전성을 위한 파라미터 조정
        safe_params['max_depth'] = min(safe_params.get('max_depth', 6), 8)
        safe_params['max_bin'] = min(safe_params.get('max_bin', 256), 256)
        safe_params['nthread'] = min(safe_params.get('nthread', 4), 6)
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """XGBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        try:
            self.feature_names = list(X_train.columns)
            
            # 데이터 검증 및 전처리
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            # XGBoost 데이터셋 생성
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
            
            evals = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=False)
                evals.append((dval, 'valid'))
            
            # 학습 실행
            early_stopping = self.params.get('early_stopping_rounds', 100)
            
            self.model = xgb.train(
                self.params,
                dtrain,
                evals=evals,
                early_stopping_rounds=early_stopping,
                verbose_eval=False
            )
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
            # 메모리 정리
            del dtrain
            if 'dval' in locals():
                del dval
            gc.collect()
            
        except Exception as e:
            logger.error(f"XGBoost 학습 실패: {str(e)}")
            # GPU 에러 시 CPU로 재시도
            if 'gpu' in str(e).lower() and self.params.get('tree_method') == 'gpu_hist':
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.params['tree_method'] = 'hist'
                self.params.pop('gpu_id', None)
                return self.fit(X_train, y_train, X_val, y_val)
            
            gc.collect()
            raise
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            X_processed = X_processed.fillna(0)
            
            dtest = xgb.DMatrix(X_processed, enable_categorical=False)
            
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
                proba = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
            else:
                proba = self.model.predict(dtest)
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            # 메모리 정리
            del dtest
            
            return proba
        except Exception as e:
            logger.error(f"XGBoost 예측 실패: {str(e)}")
            return np.full(len(X), Config.CALIBRATION_CONFIG['target_ctr'])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """calibration이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict_proba(raw_pred.reshape(-1, 1))[:, 1]
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class CatBoostModel(BaseModel):
    """CatBoost 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost가 설치되지 않았습니다.")
            
        default_params = Config.CAT_PARAMS.copy()
        if params:
            default_params.update(params)
        
        # GPU 사용 가능성 재확인
        gpu_available = False
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(100, 100).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    gpu_available = True
            except:
                gpu_available = False
        
        # GPU 파라미터 안전하게 설정
        if gpu_available:
            default_params['task_type'] = 'GPU'
            default_params['devices'] = '0'
        else:
            default_params['task_type'] = 'CPU'
            default_params.pop('devices', None)
        
        # 파라미터 검증
        default_params = self._validate_params(default_params)
        
        super().__init__("CatBoost", default_params)
        
        try:
            self.model = CatBoostClassifier(**self.params)
        except Exception as e:
            logger.error(f"CatBoost 모델 초기화 실패: {e}")
            # GPU 실패시 CPU로 재시도
            if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                logger.info("GPU 초기화 실패, CPU로 재시도")
                self.params['task_type'] = 'CPU'
                self.params.pop('devices', None)
                self.model = CatBoostClassifier(**self.params)
            else:
                raise
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """CatBoost 파라미터 검증"""
        safe_params = params.copy()
        
        # 필수 파라미터 확인
        if 'loss_function' not in safe_params:
            safe_params['loss_function'] = 'Logloss'
        if 'verbose' not in safe_params:
            safe_params['verbose'] = False
        
        # 메모리 안전성을 위한 파라미터 조정
        safe_params['depth'] = min(safe_params.get('depth', 6), 8)
        safe_params['thread_count'] = min(safe_params.get('thread_count', 4), 6)
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """CatBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        try:
            self.feature_names = list(X_train.columns)
            
            # 데이터 전처리
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)
            
            # 학습 실행
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                use_best_model=True if eval_set is not None else False,
                plot=False,
                verbose=False
            )
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"CatBoost 학습 실패: {str(e)}")
            
            # GPU 에러 시 CPU로 재시도
            if ('gpu' in str(e).lower() or 'cuda' in str(e).lower()) and self.params.get('task_type') == 'GPU':
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.params['task_type'] = 'CPU'
                self.params.pop('devices', None)
                self.model = CatBoostClassifier(**self.params)
                return self.fit(X_train, y_train, X_val, y_val)
            
            # 단순화된 학습 시도
            try:
                logger.info("단순화된 CatBoost 학습 시도")
                self.model.fit(X_train, y_train, verbose=False)
                self.is_fitted = True
                logger.info("단순화된 CatBoost 학습 완료")
            except Exception as e2:
                logger.error(f"단순화된 CatBoost 학습도 실패: {str(e2)}")
                raise
        
        # 메모리 정리
        gc.collect()
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            X_processed = X_processed.fillna(0)
            
            proba = self.model.predict_proba(X_processed)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return proba
        except Exception as e:
            logger.error(f"CatBoost 예측 실패: {str(e)}")
            return np.full(len(X), Config.CALIBRATION_CONFIG['target_ctr'])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """calibration이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict_proba(raw_pred.reshape(-1, 1))[:, 1]
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class DeepCTRModel(BaseModel):
    """GPU 기반 딥러닝 CTR 모델"""
    
    def __init__(self, input_dim: int, params: Dict[str, Any] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 설치되지 않았습니다.")
            
        BaseModel.__init__(self, "DeepCTR", params)
        
        default_params = Config.NN_PARAMS.copy()
        if params:
            default_params.update(params)
        self.params = default_params
        
        self.input_dim = input_dim
        
        # 디바이스 안전하게 설정
        self.device = 'cpu'
        self.gpu_available = False
        
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    # 엄격한 GPU 테스트
                    test_tensor = torch.zeros(1000, 1000).cuda()
                    test_result = test_tensor.sum().item()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    # 메모리 할당 테스트
                    torch.cuda.set_per_process_memory_fraction(0.5)  # 50%로 제한
                    
                    self.device = 'cuda:0'
                    self.gpu_available = True
                    logger.info("GPU 디바이스 사용 설정 완료")
                else:
                    logger.info("CUDA 사용 불가능, CPU 모드")
            except Exception as e:
                logger.warning(f"GPU 설정 실패, CPU 사용: {e}")
                self.device = 'cpu'
                self.gpu_available = False
        
        # 네트워크 구조 정의
        try:
            self.network = self._build_network()
            self.optimizer = None
            
            # Mixed Precision
            self.scaler = None
            if AMP_AVAILABLE and self.gpu_available:
                try:
                    self.scaler = GradScaler()
                    logger.info("Mixed Precision 활성화")
                except:
                    self.scaler = None
                    logger.info("Mixed Precision 비활성화")
            
            # CTR 특화 손실함수
            if TORCH_AVAILABLE:
                pos_weight = torch.tensor([49.0], device=self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                # Temperature scaling
                self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
                
                # 모델을 디바이스로 이동
                self.to(self.device)
                
        except Exception as e:
            logger.error(f"DeepCTR 모델 초기화 실패: {e}")
            # CPU 모드로 재시도
            if self.gpu_available:
                logger.info("CPU 모드로 재시도")
                self.device = 'cpu'
                self.gpu_available = False
                self.network = self._build_network()
                pos_weight = torch.tensor([49.0])
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                self.temperature = nn.Parameter(torch.ones(1) * 1.5)
                self.to(self.device)
            else:
                raise
    
    def _build_network(self):
        """네트워크 구조 생성"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
            
        try:
            hidden_dims = self.params['hidden_dims']
            dropout_rate = self.params['dropout_rate']
            use_batch_norm = self.params.get('use_batch_norm', True)
            activation = self.params.get('activation', 'relu')
            
            layers = []
            prev_dim = self.input_dim
            
            # 입력층 정규화
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(prev_dim))
            
            # 은닉층
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            # 출력층
            layers.append(nn.Linear(prev_dim, 1))
            
            return nn.Sequential(*layers)
        except Exception as e:
            logger.error(f"네트워크 구조 생성 실패: {e}")
            raise
    
    def to(self, device):
        """모델을 디바이스로 이동"""
        if TORCH_AVAILABLE:
            try:
                self.network = self.network.to(device)
                if hasattr(self, 'temperature'):
                    self.temperature = self.temperature.to(device)
                if hasattr(self, 'criterion'):
                    self.criterion = self.criterion.to(device)
                self.device = device
            except Exception as e:
                logger.warning(f"디바이스 이동 실패: {e}")
                self.device = 'cpu'
                self.gpu_available = False
    
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
        """딥러닝 모델 학습"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
            
        logger.info(f"{self.name} 모델 학습 시작 (Device: {self.device})")
        
        try:
            self.feature_names = list(X_train.columns)
            
            # 데이터 전처리
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            # 옵티마이저 초기화
            self.optimizer = optim.AdamW(
                self.parameters(), 
                lr=self.params['learning_rate'],
                weight_decay=self.params.get('weight_decay', 1e-5)
            )
            
            # 스케줄러 설정
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.7,
                patience=8,
                min_lr=1e-6
            )
            
            # 배치 크기 조정 (메모리 절약)
            batch_size = min(self.params['batch_size'], 1024) if self.gpu_available else 512
            
            # 데이터 텐서 변환
            X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = TorchDataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,
                pin_memory=False  # 안정성을 위해 비활성화
            )
            
            # 검증 데이터 준비
            val_loader = None
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = TorchDataLoader(
                    val_dataset, 
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=False
                )
            
            # 학습 루프
            best_val_loss = float('inf')
            patience_counter = 0
            max_epochs = min(self.params['epochs'], 30)  # 최대 에포크 제한
            
            for epoch in range(max_epochs):
                # 학습 모드
                self.train()
                train_loss = 0.0
                batch_count = 0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    
                    try:
                        if self.scaler is not None and AMP_AVAILABLE:
                            with autocast():
                                logits = self.forward(batch_X)
                                loss = self.criterion(logits, batch_y)
                            
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            logits = self.forward(batch_X)
                            loss = self.criterion(logits, batch_y)
                            
                            loss.backward()
                            self.optimizer.step()
                        
                        train_loss += loss.item()
                        batch_count += 1
                        
                        # 배치별 메모리 정리
                        if batch_count % 10 == 0 and self.gpu_available:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.warning(f"배치 학습 실패: {e}")
                        continue
                
                if batch_count == 0:
                    logger.error("모든 배치 학습이 실패했습니다")
                    break
                    
                train_loss /= batch_count
                
                # 검증
                val_loss = train_loss
                if val_loader is not None:
                    self.eval()
                    val_loss = 0.0
                    val_batch_count = 0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            try:
                                if self.scaler is not None and AMP_AVAILABLE:
                                    with autocast():
                                        logits = self.forward(batch_X)
                                        loss = self.criterion(logits, batch_y)
                                else:
                                    logits = self.forward(batch_X)
                                    loss = self.criterion(logits, batch_y)
                                
                                val_loss += loss.item()
                                val_batch_count += 1
                            except Exception as e:
                                logger.warning(f"검증 배치 실패: {e}")
                                continue
                    
                    if val_batch_count > 0:
                        val_loss /= val_batch_count
                        scheduler.step(val_loss)
                        
                        # 조기 종료 확인
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= self.params['patience']:
                            logger.info(f"조기 종료: epoch {epoch + 1}")
                            break
                
                # 진행 상황 로깅
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # GPU 메모리 정리
                if self.gpu_available:
                    torch.cuda.empty_cache()
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"DeepCTR 학습 실패: {str(e)}")
            # GPU 에러시 CPU로 재시도
            if self.gpu_available and ('cuda' in str(e).lower() or 'gpu' in str(e).lower()):
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.device = 'cpu'
                self.gpu_available = False
                self.to('cpu')
                return self.fit(X_train, y_train, X_val, y_val)
            raise
        finally:
            # 메모리 정리
            if self.gpu_available:
                torch.cuda.empty_cache()
            gc.collect()
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            X_processed = X_processed.fillna(0)
            
            self.eval()
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            
            predictions = []
            batch_size = min(self.params['batch_size'], 512)
            
            with torch.no_grad():
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i + batch_size]
                    
                    try:
                        if self.scaler is not None and AMP_AVAILABLE:
                            with autocast():
                                logits = self.forward(batch)
                                proba = torch.sigmoid(logits)
                        else:
                            logits = self.forward(batch)
                            proba = torch.sigmoid(logits)
                        
                        predictions.append(proba.cpu().numpy())
                        
                        # 배치별 메모리 정리
                        if self.gpu_available and i % (batch_size * 5) == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.warning(f"예측 배치 실패: {e}")
                        batch_size_actual = len(batch)
                        predictions.append(np.full(batch_size_actual, Config.CALIBRATION_CONFIG['target_ctr']))
            
            proba = np.concatenate(predictions)
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
        except Exception as e:
            logger.error(f"DeepCTR 예측 실패: {str(e)}")
            return np.full(len(X), Config.CALIBRATION_CONFIG['target_ctr'])
        finally:
            # GPU 메모리 정리
            if self.gpu_available:
                torch.cuda.empty_cache()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """calibration이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict_proba(raw_pred.reshape(-1, 1))[:, 1]
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class LogisticModel(BaseModel):
    """로지스틱 회귀 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn이 설치되지 않았습니다.")
            
        default_params = {
            'C': 0.1,
            'max_iter': 2000,
            'random_state': Config.RANDOM_STATE,
            'class_weight': 'balanced',
            'solver': 'lbfgs'
        }
        if params:
            default_params.update(params)
        super().__init__("LogisticRegression", default_params)
        
        self.model = LogisticRegression(**self.params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """로지스틱 회귀 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        try:
            self.feature_names = list(X_train.columns)
            
            # 데이터 전처리
            X_train = X_train.fillna(0)
            
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
        except Exception as e:
            logger.error(f"Logistic 학습 실패: {str(e)}")
            raise
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            X_processed = X_processed.fillna(0)
            
            proba = self.model.predict_proba(X_processed)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            return proba
        except Exception as e:
            logger.error(f"Logistic 예측 실패: {str(e)}")
            return np.full(len(X), Config.CALIBRATION_CONFIG['target_ctr'])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """calibration이 적용된 확률 예측"""
        raw_pred = self.predict_proba_raw(X)
        
        if self.is_calibrated and self.calibrator is not None:
            try:
                calibrated_pred = self.calibrator.predict_proba(raw_pred.reshape(-1, 1))[:, 1]
                return np.clip(calibrated_pred, 1e-15, 1 - 1e-15)
            except:
                pass
        
        return raw_pred

class CTRCalibrator:
    """CTR 특화 확률 보정 클래스"""
    
    def __init__(self, target_ctr: float = 0.0201):
        self.target_ctr = target_ctr
        self.platt_scaler = None
        self.isotonic_regressor = None
        self.bias_correction = 0.0
        
    def fit_platt_scaling(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Platt Scaling 학습"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn이 없어 Platt scaling을 사용할 수 없습니다")
            return
            
        try:
            self.platt_scaler = LogisticRegression()
            self.platt_scaler.fit(y_pred.reshape(-1, 1), y_true)
            logger.info("Platt scaling 학습 완료")
        except Exception as e:
            logger.error(f"Platt scaling 학습 실패: {str(e)}")
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Isotonic Regression 학습"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn이 없어 Isotonic regression을 사용할 수 없습니다")
            return
            
        try:
            self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor.fit(y_pred, y_true)
            logger.info("Isotonic regression 학습 완료")
        except Exception as e:
            logger.error(f"Isotonic regression 학습 실패: {str(e)}")
    
    def fit_bias_correction(self, y_true: np.ndarray, y_pred: np.ndarray):
        """단순 편향 보정 학습"""
        try:
            predicted_ctr = y_pred.mean()
            actual_ctr = y_true.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            logger.info(f"편향 보정 학습 완료: {self.bias_correction:.4f}")
        except Exception as e:
            logger.error(f"편향 보정 학습 실패: {str(e)}")

class ModelFactory:
    """모델 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """모델 타입에 따라 모델 인스턴스 생성"""
        
        try:
            if model_type.lower() == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM이 설치되지 않았습니다.")
                return LightGBMModel(kwargs.get('params'))
            
            elif model_type.lower() == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost가 설치되지 않았습니다.")
                return XGBoostModel(kwargs.get('params'))
            
            elif model_type.lower() == 'catboost':
                if not CATBOOST_AVAILABLE:
                    raise ImportError("CatBoost가 설치되지 않았습니다.")
                return CatBoostModel(kwargs.get('params'))
            
            elif model_type.lower() == 'deepctr':
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch가 설치되지 않았습니다.")
                input_dim = kwargs.get('input_dim')
                if input_dim is None:
                    raise ValueError("DeepCTR 모델에는 input_dim이 필요합니다.")
                return DeepCTRModel(input_dim, kwargs.get('params'))
            
            elif model_type.lower() == 'logistic':
                return LogisticModel(kwargs.get('params'))
            
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
                
        except Exception as e:
            logger.error(f"모델 생성 실패 ({model_type}): {str(e)}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """사용 가능한 모델 타입 리스트"""
        available = []
        
        # 기본 모델들
        if SKLEARN_AVAILABLE:
            available.append('logistic')
        
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        if CATBOOST_AVAILABLE:
            available.append('catboost')
        if TORCH_AVAILABLE:
            available.append('deepctr')
        
        # 최소한 하나의 모델은 사용 가능해야 함
        if not available:
            logger.error("사용 가능한 모델이 없습니다. 필요한 라이브러리를 설치해주세요.")
            available = ['logistic']  # 기본 fallback
            
        return available

class ModelEvaluator:
    """모델 평가 클래스"""
    
    @staticmethod
    def evaluate_model(model: BaseModel, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """모델 평가 수행"""
        
        try:
            y_pred_proba = model.predict_proba(X_test)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            metrics = {}
            
            # sklearn이 사용 가능한 경우에만 고급 지표 계산
            if SKLEARN_AVAILABLE:
                try:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
                    metrics['logloss'] = log_loss(y_test, y_pred_proba)
                except:
                    metrics['auc'] = 0.5
                    metrics['logloss'] = 1.0
                
                from sklearn.metrics import precision_score, recall_score, f1_score
                try:
                    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
                except:
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['f1'] = 0.0
            else:
                # 기본 지표만 계산
                metrics['auc'] = 0.5
                metrics['logloss'] = 1.0
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
            
            metrics['accuracy'] = (y_test == y_pred).mean()
            
            metrics['ctr_actual'] = y_test.mean()
            metrics['ctr_predicted'] = y_pred_proba.mean()
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            
        except Exception as e:
            logger.error(f"평가 지표 계산 중 오류: {str(e)}")
            metrics = {
                'auc': 0.5,
                'logloss': 1.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ctr_actual': 0.0201,
                'ctr_predicted': 0.0201,
                'ctr_bias': 0.0
            }
        
        return metrics