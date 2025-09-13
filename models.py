# models.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import pickle
import gc

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
    if torch.cuda.is_available():
        try:
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

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn이 설치되지 않았습니다.")

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
            train_pred = self.predict_proba_raw(X_train)
            
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
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0.0
            
            X = X[self.feature_names]
            return X
        except Exception as e:
            logger.warning(f"피처 일관성 보장 실패: {str(e)}")
            return X

class LightGBMModel(BaseModel):
    """CTR 특화 LightGBM 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM이 설치되지 않았습니다.")
            
        # CTR 특화 기본 파라미터
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 200,
            'min_child_weight': 10,
            'lambda_l1': 2.0,
            'lambda_l2': 2.0,
            'verbose': -1,
            'random_state': Config.RANDOM_STATE,
            'n_estimators': 3000,
            'early_stopping_rounds': 200,
            'scale_pos_weight': 49.0,
            'force_row_wise': True,
            'max_bin': 255,
            'num_threads': 6,
            'device_type': 'cpu',
            'min_data_in_leaf': 100,
            'max_depth': 12,
            'feature_fraction_bynode': 0.8,
            'extra_trees': True,
            'path_smooth': 1.0
        }
        
        if params:
            default_params.update(params)
        
        default_params = self._validate_params(default_params)
        
        super().__init__("LightGBM", default_params)
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """LightGBM 파라미터 검증"""
        safe_params = params.copy()
        
        if 'objective' not in safe_params:
            safe_params['objective'] = 'binary'
        if 'metric' not in safe_params:
            safe_params['metric'] = 'binary_logloss'
        if 'verbose' not in safe_params:
            safe_params['verbose'] = -1
        
        if 'is_unbalance' in safe_params and 'scale_pos_weight' in safe_params:
            safe_params.pop('is_unbalance', None)
        
        safe_params['num_leaves'] = min(safe_params.get('num_leaves', 63), 511)
        safe_params['max_bin'] = min(safe_params.get('max_bin', 255), 255)
        safe_params['num_threads'] = min(safe_params.get('num_threads', 4), 6)
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """CTR 특화 LightGBM 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        try:
            self.feature_names = list(X_train.columns)
            
            if X_train.isnull().sum().sum() > 0:
                logger.warning("학습 데이터에 결측치가 있습니다. 0으로 대체합니다.")
                X_train = X_train.fillna(0)
            
            if X_val is not None and X_val.isnull().sum().sum() > 0:
                logger.warning("검증 데이터에 결측치가 있습니다. 0으로 대체합니다.")
                X_val = X_val.fillna(0)
            
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
                valid_sets.append(val_data)
                valid_names.append('valid')
            
            callbacks = []
            early_stopping = self.params.get('early_stopping_rounds', 100)
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
            
            # CTR 특화 학습
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
            del train_data
            if 'val_data' in locals():
                del val_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"LightGBM 학습 실패: {str(e)}")
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
    """CTR 특화 XGBoost 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost가 설치되지 않았습니다.")
        
        # CTR 특화 기본 파라미터
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
            'n_estimators': 3000,
            'early_stopping_rounds': 200,
            'max_bin': 255,
            'nthread': 6,
            'grow_policy': 'lossguide',
            'max_leaves': 255,
            'gamma': 0.1
        }
        
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
        
        if gpu_available:
            default_params['tree_method'] = 'gpu_hist'
            default_params['gpu_id'] = 0
        else:
            default_params['tree_method'] = 'hist'
            default_params.pop('gpu_id', None)
        
        if params:
            default_params.update(params)
        
        default_params = self._validate_params(default_params)
        
        super().__init__("XGBoost", default_params)
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """XGBoost 파라미터 검증"""
        safe_params = params.copy()
        
        if 'objective' not in safe_params:
            safe_params['objective'] = 'binary:logistic'
        if 'eval_metric' not in safe_params:
            safe_params['eval_metric'] = 'logloss'
        
        safe_params['max_depth'] = min(safe_params.get('max_depth', 6), 12)
        safe_params['max_bin'] = min(safe_params.get('max_bin', 256), 256)
        safe_params['nthread'] = min(safe_params.get('nthread', 4), 6)
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """CTR 특화 XGBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
            
            evals = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=False)
                evals.append((dval, 'valid'))
            
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
            
            del dtrain
            if 'dval' in locals():
                del dval
            gc.collect()
            
        except Exception as e:
            logger.error(f"XGBoost 학습 실패: {str(e)}")
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
    """CTR 특화 CatBoost 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost가 설치되지 않았습니다.")
        
        # CTR 특화 기본 파라미터 (충돌 방지)
        default_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'task_type': 'CPU',
            'depth': 8,
            'learning_rate': 0.03,
            'l2_leaf_reg': 10,
            'iterations': 3000,
            'random_seed': Config.RANDOM_STATE,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'max_ctr_complexity': 2,
            'thread_count': 6,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1.0,
            'od_type': 'IncToDec',
            'od_wait': 200,  # early_stopping_rounds 대신 od_wait 사용
            'leaf_estimation_iterations': 10,
            'leaf_estimation_method': 'Newton',
            'grow_policy': 'Lossguide',
            'max_leaves': 255,
            'min_data_in_leaf': 100
        }
        
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
        
        if gpu_available:
            default_params['task_type'] = 'GPU'
            default_params['devices'] = '0'
        else:
            default_params['task_type'] = 'CPU'
            default_params.pop('devices', None)
        
        if params:
            default_params.update(params)
        
        default_params = self._validate_params(default_params)
        
        super().__init__("CatBoost", default_params)
        
        try:
            self.model = CatBoostClassifier(**self.params)
        except Exception as e:
            logger.error(f"CatBoost 모델 초기화 실패: {e}")
            if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                logger.info("GPU 초기화 실패, CPU로 재시도")
                self.params['task_type'] = 'CPU'
                self.params.pop('devices', None)
                self.model = CatBoostClassifier(**self.params)
            else:
                raise
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """CatBoost 파라미터 검증 (충돌 방지)"""
        safe_params = params.copy()
        
        if 'loss_function' not in safe_params:
            safe_params['loss_function'] = 'Logloss'
        if 'verbose' not in safe_params:
            safe_params['verbose'] = False
        
        # early_stopping_rounds와 od_wait 충돌 방지
        if 'early_stopping_rounds' in safe_params and 'od_wait' in safe_params:
            logger.warning("CatBoost: early_stopping_rounds와 od_wait 동시 설정 방지. od_wait 사용")
            safe_params.pop('early_stopping_rounds', None)
        
        # early_stopping_rounds만 있는 경우 od_wait로 변경
        if 'early_stopping_rounds' in safe_params and 'od_wait' not in safe_params:
            early_stop_value = safe_params.pop('early_stopping_rounds')
            safe_params['od_wait'] = early_stop_value
            safe_params['od_type'] = 'IncToDec'
        
        safe_params['depth'] = min(safe_params.get('depth', 6), 10)
        safe_params['thread_count'] = min(safe_params.get('thread_count', 4), 6)
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """CTR 특화 CatBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)
            
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
            
            if ('gpu' in str(e).lower() or 'cuda' in str(e).lower()) and self.params.get('task_type') == 'GPU':
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.params['task_type'] = 'CPU'
                self.params.pop('devices', None)
                self.model = CatBoostClassifier(**self.params)
                return self.fit(X_train, y_train, X_val, y_val)
            
            try:
                logger.info("단순화된 CatBoost 학습 시도")
                simplified_params = {
                    'loss_function': 'Logloss',
                    'task_type': 'CPU',
                    'depth': 6,
                    'learning_rate': 0.1,
                    'iterations': 1000,
                    'verbose': False,
                    'random_seed': self.params.get('random_seed', 42)
                }
                self.model = CatBoostClassifier(**simplified_params)
                self.model.fit(X_train, y_train, verbose=False)
                self.is_fitted = True
                logger.info("단순화된 CatBoost 학습 완료")
            except Exception as e2:
                logger.error(f"단순화된 CatBoost 학습도 실패: {str(e2)}")
                raise
        
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
    """CTR 특화 딥러닝 모델"""
    
    def __init__(self, input_dim: int, params: Dict[str, Any] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 설치되지 않았습니다.")
            
        BaseModel.__init__(self, "DeepCTR", params)
        
        # CTR 특화 기본 파라미터
        default_params = {
            'hidden_dims': [512, 256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 1024,
            'epochs': 50,
            'patience': 15,
            'use_batch_norm': True,
            'activation': 'relu',
            'use_residual': True,
            'use_attention': False,
            'focal_loss_alpha': 0.25,
            'focal_loss_gamma': 2.0
        }
        
        if params:
            default_params.update(params)
        self.params = default_params
        
        self.input_dim = input_dim
        
        self.device = 'cpu'
        self.gpu_available = False
        
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(1000, 1000).cuda()
                    test_result = test_tensor.sum().item()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    torch.cuda.set_per_process_memory_fraction(0.6)
                    
                    self.device = 'cuda:0'
                    self.gpu_available = True
                    logger.info("GPU 디바이스 사용 설정 완료")
                else:
                    logger.info("CUDA 사용 불가능, CPU 모드")
            except Exception as e:
                logger.warning(f"GPU 설정 실패, CPU 사용: {e}")
                self.device = 'cpu'
                self.gpu_available = False
        
        try:
            self.network = self._build_ctr_network()
            self.optimizer = None
            
            self.scaler = None
            if AMP_AVAILABLE and self.gpu_available:
                try:
                    self.scaler = GradScaler()
                    logger.info("Mixed Precision 활성화")
                except:
                    self.scaler = None
                    logger.info("Mixed Precision 비활성화")
            
            if TORCH_AVAILABLE:
                pos_weight = torch.tensor([49.0], device=self.device)
                self.criterion = self._get_ctr_loss(pos_weight)
                
                self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
                
                self.to(self.device)
                
        except Exception as e:
            logger.error(f"DeepCTR 모델 초기화 실패: {e}")
            if self.gpu_available:
                logger.info("CPU 모드로 재시도")
                self.device = 'cpu'
                self.gpu_available = False
                self.network = self._build_ctr_network()
                pos_weight = torch.tensor([49.0])
                self.criterion = self._get_ctr_loss(pos_weight)
                self.temperature = nn.Parameter(torch.ones(1) * 1.5)
                self.to(self.device)
            else:
                raise
    
    def _build_ctr_network(self):
        """CTR 특화 네트워크 구조 생성"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
            
        try:
            hidden_dims = self.params['hidden_dims']
            dropout_rate = self.params['dropout_rate']
            use_batch_norm = self.params.get('use_batch_norm', True)
            activation = self.params.get('activation', 'relu')
            use_residual = self.params.get('use_residual', False)
            
            layers = []
            prev_dim = self.input_dim
            
            # 입력층 정규화
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(prev_dim))
            
            # 은닉층
            for i, hidden_dim in enumerate(hidden_dims):
                linear = nn.Linear(prev_dim, hidden_dim)
                
                # Xavier 초기화
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                
                layers.append(linear)
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'swish':
                    layers.append(nn.SiLU())
                
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            # 출력층
            output_layer = nn.Linear(prev_dim, 1)
            nn.init.xavier_uniform_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            layers.append(output_layer)
            
            return nn.Sequential(*layers)
        except Exception as e:
            logger.error(f"네트워크 구조 생성 실패: {e}")
            raise
    
    def _get_ctr_loss(self, pos_weight):
        """CTR 특화 손실함수"""
        if self.params.get('use_focal_loss', False):
            return self._focal_loss
        else:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def _focal_loss(self, inputs, targets):
        """Focal Loss 구현"""
        alpha = self.params['focal_loss_alpha']
        gamma = self.params['focal_loss_gamma']
        
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        
        return focal_loss.mean()
    
    def to(self, device):
        """모델을 디바이스로 이동"""
        if TORCH_AVAILABLE:
            try:
                self.network = self.network.to(device)
                if hasattr(self, 'temperature'):
                    self.temperature = self.temperature.to(device)
                if hasattr(self, 'criterion') and hasattr(self.criterion, 'to'):
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
        """CTR 특화 딥러닝 모델 학습"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
            
        logger.info(f"{self.name} 모델 학습 시작 (Device: {self.device})")
        
        try:
            self.feature_names = list(X_train.columns)
            
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
                factor=0.8,
                patience=10,
                min_lr=1e-6
            )
            
            # 배치 크기 조정
            batch_size = min(self.params['batch_size'], 2048) if self.gpu_available else 1024
            
            X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = TorchDataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
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
            max_epochs = min(self.params['epochs'], 50)
            
            for epoch in range(max_epochs):
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
                        
                        if batch_count % 10 == 0 and self.gpu_available:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
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
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= self.params['patience']:
                            logger.info(f"조기 종료: epoch {epoch + 1}")
                            break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                if self.gpu_available:
                    torch.cuda.empty_cache()
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"DeepCTR 학습 실패: {str(e)}")
            if self.gpu_available and ('cuda' in str(e).lower() or 'gpu' in str(e).lower()):
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.device = 'cpu'
                self.gpu_available = False
                self.to('cpu')
                return self.fit(X_train, y_train, X_val, y_val)
            raise
        finally:
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
            batch_size = min(self.params['batch_size'], 1024)
            
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
        self.temperature_scaler = None
        
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
    
    def fit_temperature_scaling(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Temperature Scaling 학습"""
        try:
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(temperature):
                calibrated = 1 / (1 + np.exp(-y_pred / temperature))
                return -np.sum(y_true * np.log(calibrated + 1e-15) + (1 - y_true) * np.log(1 - calibrated + 1e-15))
            
            result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature_scaler = result.x
            logger.info(f"Temperature scaling 학습 완료: T={self.temperature_scaler:.3f}")
        except Exception as e:
            logger.error(f"Temperature scaling 학습 실패: {str(e)}")
    
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
    """CTR 특화 모델 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """모델 타입에 따라 CTR 특화 모델 인스턴스 생성"""
        
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
        
        if not available:
            logger.error("사용 가능한 모델이 없습니다. 필요한 라이브러리를 설치해주세요.")
            available = ['logistic']
            
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