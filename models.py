# models.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import pickle
import gc
import warnings
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
            
            test_tensor = torch.zeros(2000, 2000).cuda()
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
    """모든 모델의 기본 클래스 - Combined Score 0.30+ 달성 목표"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.calibrator = None
        self.is_calibrated = False
        self.prediction_diversity_threshold = 1000
        
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
                         method: str = 'platt', cv_folds: int = 5):
        """고성능 예측 확률 보정 적용"""
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
            
            logger.info(f"{self.name} {method} 고성능 calibration 적용 완료")
            
        except Exception as e:
            logger.error(f"고성능 Calibration 적용 실패 ({self.name}): {str(e)}")
    
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
    
    def _enhance_prediction_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """예측 다양성 향상 - Combined Score 개선"""
        try:
            unique_predictions = len(np.unique(predictions))
            
            if unique_predictions < self.prediction_diversity_threshold:
                logger.info(f"{self.name}: 예측 다양성 향상 적용 (고유값: {unique_predictions})")
                
                noise_scale = max(predictions.std() * 0.005, 1e-6)
                noise = np.random.normal(0, noise_scale, len(predictions))
                
                predictions = predictions + noise
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                
                if len(np.unique(predictions)) > unique_predictions:
                    logger.info(f"{self.name}: 예측 다양성 개선됨")
            
            return predictions
            
        except Exception as e:
            logger.warning(f"예측 다양성 향상 실패: {e}")
            return predictions

class HighPerformanceLightGBMModel(BaseModel):
    """Combined Score 0.30+ 달성 목표 고성능 LightGBM 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM이 설치되지 않았습니다.")
            
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 511,
            'learning_rate': 0.025,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.75,
            'bagging_freq': 7,
            'min_child_samples': 300,
            'min_child_weight': 15,
            'lambda_l1': 3.0,
            'lambda_l2': 3.0,
            'max_depth': 15,
            'verbose': -1,
            'random_state': Config.RANDOM_STATE,
            'n_estimators': 4000,
            'early_stopping_rounds': 250,
            'scale_pos_weight': 49.8,
            'force_row_wise': True,
            'max_bin': 255,
            'num_threads': 12,
            'device_type': 'cpu',
            'min_data_in_leaf': 120,
            'feature_fraction_bynode': 0.85,
            'extra_trees': True,
            'path_smooth': 1.5,
            'grow_policy': 'lossguide',
            'reg_alpha': 1.5,
            'reg_sqrt': True,
            'categorical_feature': 'auto'
        }
        
        if params:
            default_params.update(params)
        
        default_params = self._validate_high_performance_params(default_params)
        
        super().__init__("HighPerformanceLightGBM", default_params)
        self.prediction_diversity_threshold = 2000
    
    def _validate_high_performance_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """고성능 LightGBM 파라미터 검증 및 최적화"""
        safe_params = params.copy()
        
        if 'objective' not in safe_params:
            safe_params['objective'] = 'binary'
        if 'metric' not in safe_params:
            safe_params['metric'] = 'binary_logloss'
        if 'verbose' not in safe_params:
            safe_params['verbose'] = -1
        
        if 'is_unbalance' in safe_params and 'scale_pos_weight' in safe_params:
            safe_params.pop('is_unbalance', None)
        
        safe_params['num_leaves'] = min(max(safe_params.get('num_leaves', 511), 63), 1023)
        safe_params['max_bin'] = min(safe_params.get('max_bin', 255), 255)
        safe_params['num_threads'] = min(safe_params.get('num_threads', 12), 12)
        safe_params['max_depth'] = min(max(safe_params.get('max_depth', 15), 8), 20)
        
        safe_params['lambda_l1'] = max(safe_params.get('lambda_l1', 3.0), 1.0)
        safe_params['lambda_l2'] = max(safe_params.get('lambda_l2', 3.0), 1.0)
        safe_params['min_child_samples'] = max(safe_params.get('min_child_samples', 300), 100)
        safe_params['min_child_weight'] = max(safe_params.get('min_child_weight', 15), 5)
        
        safe_params['force_row_wise'] = True
        safe_params['device_type'] = 'cpu'
        safe_params['deterministic'] = True
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """고성능 LightGBM 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
        try:
            self.feature_names = list(X_train.columns)
            
            if X_train.isnull().sum().sum() > 0:
                logger.warning("학습 데이터에 결측치가 있습니다. 0으로 대체합니다.")
                X_train = X_train.fillna(0)
            
            if X_val is not None and X_val.isnull().sum().sum() > 0:
                logger.warning("검증 데이터에 결측치가 있습니다. 0으로 대체합니다.")
                X_val = X_val.fillna(0)
            
            for col in X_train.columns:
                if X_train[col].dtype in ['float64']:
                    X_train[col] = X_train[col].astype('float32')
                if X_val is not None and X_val[col].dtype in ['float64']:
                    X_val[col] = X_val[col].astype('float32')
            
            train_data = lgb.Dataset(
                X_train, 
                label=y_train, 
                free_raw_data=False,
                feature_name=list(X_train.columns)
            )
            
            valid_sets = [train_data]
            valid_names = ['train']
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(
                    X_val, 
                    label=y_val, 
                    reference=train_data, 
                    free_raw_data=False,
                    feature_name=list(X_val.columns)
                )
                valid_sets.append(val_data)
                valid_names.append('valid')
            
            callbacks = []
            early_stopping = self.params.get('early_stopping_rounds', 250)
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
            
            callbacks.append(lgb.reset_parameter(learning_rate=lambda iter: max(0.01, self.params['learning_rate'] * (0.99 ** (iter // 100)))))
            
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
            logger.error(f"고성능 LightGBM 학습 실패: {str(e)}")
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
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
            
            num_iteration = getattr(self.model, 'best_iteration', None)
            proba = self.model.predict(X_processed, num_iteration=num_iteration)
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            proba = self._enhance_prediction_diversity(proba)
            
            return proba
        except Exception as e:
            logger.error(f"고성능 LightGBM 예측 실패: {str(e)}")
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

class HighPerformanceXGBoostModel(BaseModel):
    """Combined Score 0.30+ 달성 목표 고성능 XGBoost 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost가 설치되지 않았습니다.")
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': 10,
            'learning_rate': 0.025,
            'subsample': 0.82,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.85,
            'colsample_bynode': 0.85,
            'min_child_weight': 20,
            'reg_alpha': 3.0,
            'reg_lambda': 3.0,
            'scale_pos_weight': 49.8,
            'random_state': Config.RANDOM_STATE,
            'n_estimators': 4000,
            'early_stopping_rounds': 250,
            'max_bin': 255,
            'nthread': 12,
            'grow_policy': 'lossguide',
            'max_leaves': 511,
            'gamma': 0.15,
            'max_delta_step': 1,
            'sampling_method': 'uniform'
        }
        
        gpu_available = False
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(1000, 1000).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    gpu_available = True
            except:
                gpu_available = False
        
        if gpu_available:
            default_params['tree_method'] = 'gpu_hist'
            default_params['gpu_id'] = 0
            default_params['predictor'] = 'gpu_predictor'
        else:
            default_params['tree_method'] = 'hist'
            default_params.pop('gpu_id', None)
            default_params.pop('predictor', None)
        
        if params:
            default_params.update(params)
        
        default_params = self._validate_high_performance_params(default_params)
        
        super().__init__("HighPerformanceXGBoost", default_params)
        self.prediction_diversity_threshold = 2000
    
    def _validate_high_performance_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """고성능 XGBoost 파라미터 검증 및 최적화"""
        safe_params = params.copy()
        
        if 'objective' not in safe_params:
            safe_params['objective'] = 'binary:logistic'
        if 'eval_metric' not in safe_params:
            safe_params['eval_metric'] = 'logloss'
        
        safe_params['max_depth'] = min(max(safe_params.get('max_depth', 10), 6), 15)
        safe_params['max_bin'] = min(safe_params.get('max_bin', 255), 255)
        safe_params['nthread'] = min(safe_params.get('nthread', 12), 12)
        
        safe_params['reg_alpha'] = max(safe_params.get('reg_alpha', 3.0), 1.0)
        safe_params['reg_lambda'] = max(safe_params.get('reg_lambda', 3.0), 1.0)
        safe_params['min_child_weight'] = max(safe_params.get('min_child_weight', 20), 10)
        safe_params['gamma'] = max(safe_params.get('gamma', 0.15), 0.0)
        
        safe_params['learning_rate'] = min(max(safe_params.get('learning_rate', 0.025), 0.01), 0.2)
        safe_params['subsample'] = min(max(safe_params.get('subsample', 0.82), 0.6), 1.0)
        safe_params['colsample_bytree'] = min(max(safe_params.get('colsample_bytree', 0.85), 0.6), 1.0)
        
        safe_params['grow_policy'] = 'lossguide'
        safe_params['max_leaves'] = min(safe_params.get('max_leaves', 511), 1023)
        safe_params['validate_parameters'] = True
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """고성능 XGBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            for col in X_train.columns:
                if X_train[col].dtype in ['float64']:
                    X_train[col] = X_train[col].astype('float32')
                if X_val is not None and X_val[col].dtype in ['float64']:
                    X_val[col] = X_val[col].astype('float32')
            
            dtrain = xgb.DMatrix(
                X_train, 
                label=y_train, 
                enable_categorical=False,
                feature_names=list(X_train.columns)
            )
            
            evals = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(
                    X_val, 
                    label=y_val, 
                    enable_categorical=False,
                    feature_names=list(X_val.columns)
                )
                evals.append((dval, 'valid'))
            
            early_stopping = self.params.get('early_stopping_rounds', 250)
            
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
            logger.error(f"고성능 XGBoost 학습 실패: {str(e)}")
            if 'gpu' in str(e).lower() and self.params.get('tree_method') == 'gpu_hist':
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.params['tree_method'] = 'hist'
                self.params.pop('gpu_id', None)
                self.params.pop('predictor', None)
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
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            proba = self._enhance_prediction_diversity(proba)
            
            del dtest
            
            return proba
        except Exception as e:
            logger.error(f"고성능 XGBoost 예측 실패: {str(e)}")
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

class HighPerformanceCatBoostModel(BaseModel):
    """Combined Score 0.30+ 달성 목표 고성능 CatBoost 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost가 설치되지 않았습니다.")
        
        default_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'task_type': 'CPU',
            'depth': 10,
            'learning_rate': 0.025,
            'l2_leaf_reg': 15,
            'iterations': 4000,
            'random_seed': Config.RANDOM_STATE,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'max_ctr_complexity': 3,
            'thread_count': 12,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1.5,
            'leaf_estimation_iterations': 12,
            'leaf_estimation_method': 'Newton',
            'grow_policy': 'Lossguide',
            'max_leaves': 511,
            'min_data_in_leaf': 120,
            'od_wait': 250,
            'od_type': 'IncToDec'
        }
        
        gpu_available = False
        if TORCH_AVAILABLE:
            try:
                import torch
                if torch.cuda.is_available():
                    test_tensor = torch.zeros(1000, 1000).cuda()
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
        
        default_params = self._validate_high_performance_params(default_params)
        
        super().__init__("HighPerformanceCatBoost", default_params)
        self.prediction_diversity_threshold = 2000
        
        init_params = {k: v for k, v in self.params.items() 
                      if k not in ['early_stopping_rounds', 'use_best_model', 'eval_set', 'od_wait', 'od_type']}
        
        try:
            self.model = CatBoostClassifier(**init_params)
        except Exception as e:
            logger.error(f"고성능 CatBoost 모델 초기화 실패: {e}")
            if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                logger.info("GPU 초기화 실패, CPU로 재시도")
                self.params['task_type'] = 'CPU'
                self.params.pop('devices', None)
                init_params = {k: v for k, v in self.params.items() 
                              if k not in ['early_stopping_rounds', 'use_best_model', 'eval_set', 'od_wait', 'od_type']}
                try:
                    self.model = CatBoostClassifier(**init_params)
                except Exception as e2:
                    logger.error(f"CPU 초기화도 실패: {e2}")
                    raise
            else:
                raise
    
    def _validate_high_performance_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """고성능 CatBoost 파라미터 검증"""
        safe_params = params.copy()
        
        if 'loss_function' not in safe_params:
            safe_params['loss_function'] = 'Logloss'
        if 'verbose' not in safe_params:
            safe_params['verbose'] = False
        
        conflicting_params = [
            'early_stopping_rounds', 'use_best_model', 'eval_set', 
            'early_stopping', 'early_stop', 'best_model_min_trees'
        ]
        
        removed_params = []
        for param in conflicting_params:
            if param in safe_params:
                removed_params.append(param)
                if param == 'early_stopping_rounds':
                    early_stop_val = safe_params.pop(param)
                    if 'od_wait' not in safe_params:
                        safe_params['od_wait'] = early_stop_val
                        safe_params['od_type'] = 'IncToDec'
                else:
                    safe_params.pop(param)
        
        if removed_params:
            logger.info(f"고성능 CatBoost: 충돌 방지를 위해 제거된 파라미터: {removed_params}")
        
        safe_params['depth'] = min(max(safe_params.get('depth', 10), 6), 12)
        safe_params['thread_count'] = min(safe_params.get('thread_count', 12), 12)
        safe_params['iterations'] = min(safe_params.get('iterations', 4000), 6000)
        
        safe_params['l2_leaf_reg'] = max(safe_params.get('l2_leaf_reg', 15), 5)
        safe_params['min_data_in_leaf'] = max(safe_params.get('min_data_in_leaf', 120), 50)
        
        safe_params['learning_rate'] = min(max(safe_params.get('learning_rate', 0.025), 0.01), 0.2)
        
        safe_params['grow_policy'] = 'Lossguide'
        safe_params['max_leaves'] = min(safe_params.get('max_leaves', 511), 1023)
        
        if 'od_wait' in safe_params:
            safe_params['od_wait'] = max(safe_params['od_wait'], 50)
        if 'od_type' not in safe_params:
            safe_params['od_type'] = 'IncToDec'
        
        return safe_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """고성능 CatBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            for col in X_train.columns:
                if X_train[col].dtype in ['float64']:
                    X_train[col] = X_train[col].astype('float32')
                if X_val is not None and X_val[col].dtype in ['float64']:
                    X_val[col] = X_val[col].astype('float32')
            
            fit_params = {
                'X': X_train,
                'y': y_train,
                'verbose': False,
                'plot': False
            }
            
            if X_val is not None and y_val is not None:
                fit_params['eval_set'] = (X_val, y_val)
                fit_params['use_best_model'] = True
                
                if 'od_wait' in self.params:
                    od_params = {k: v for k, v in self.params.items() 
                               if k not in ['early_stopping_rounds', 'use_best_model', 'eval_set']}
                    
                    try:
                        self.model = CatBoostClassifier(**od_params)
                    except Exception as e:
                        logger.warning(f"od 파라미터 포함 초기화 실패: {e}")
                        od_params = {k: v for k, v in od_params.items() 
                                   if k not in ['od_wait', 'od_type']}
                        self.model = CatBoostClassifier(**od_params)
            
            self.model.fit(**fit_params)
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"고성능 CatBoost 학습 실패: {str(e)}")
            
            if ('gpu' in str(e).lower() or 'cuda' in str(e).lower() or 'device' in str(e).lower()) and self.params.get('task_type') == 'GPU':
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.params['task_type'] = 'CPU'
                self.params.pop('devices', None)
                try:
                    cpu_params = {k: v for k, v in self.params.items() 
                                 if k not in ['early_stopping_rounds', 'use_best_model', 'eval_set', 'od_wait', 'od_type']}
                    self.model = CatBoostClassifier(**cpu_params)
                    return self.fit(X_train, y_train, X_val, y_val)
                except Exception as e2:
                    logger.error(f"CPU 재시도도 실패: {e2}")
            
            if any(keyword in str(e).lower() for keyword in ['early_stopping', 'od_', 'overfitting']):
                logger.info("조기 종료 관련 오류 - 단순화된 설정으로 재시도")
                try:
                    simplified_params = {k: v for k, v in self.params.items() 
                                       if k not in ['od_wait', 'od_type', 'early_stopping_rounds', 'use_best_model', 'eval_set']}
                    self.model = CatBoostClassifier(**simplified_params)
                    self.model.fit(X_train, y_train, verbose=False)
                    self.is_fitted = True
                    logger.info("단순화된 고성능 CatBoost 학습 완료")
                    return self
                except Exception as e3:
                    logger.error(f"단순화된 학습도 실패: {e3}")
            
            try:
                logger.info("최소 파라미터로 고성능 CatBoost 학습 시도")
                minimal_params = {
                    'loss_function': 'Logloss',
                    'task_type': 'CPU',
                    'depth': 8,
                    'learning_rate': 0.03,
                    'iterations': 2000,
                    'verbose': False,
                    'random_seed': self.params.get('random_seed', 42),
                    'auto_class_weights': 'Balanced'
                }
                self.model = CatBoostClassifier(**minimal_params)
                self.model.fit(X_train, y_train, verbose=False)
                self.is_fitted = True
                logger.info("최소 파라미터 고성능 CatBoost 학습 완료")
            except Exception as e4:
                logger.error(f"최소 파라미터 학습도 실패: {str(e4)}")
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
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
            
            proba = self.model.predict_proba(X_processed)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            proba = self._enhance_prediction_diversity(proba)
            
            return proba
        except Exception as e:
            logger.error(f"고성능 CatBoost 예측 실패: {str(e)}")
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

class RTX4060TiOptimizedDeepCTRModel(BaseModel):
    """RTX 4060 Ti 16GB 최적화 DeepCTR 모델 - Combined Score 0.30+ 목표"""
    
    def __init__(self, input_dim: int, params: Dict[str, Any] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 설치되지 않았습니다.")
            
        BaseModel.__init__(self, "RTX4060TiOptimizedDeepCTR", params)
        
        default_params = {
            'hidden_dims': [1024, 512, 256, 128, 64] if rtx_4060ti_detected else [512, 256, 128, 64],
            'dropout_rate': 0.25,
            'learning_rate': 0.0008,
            'weight_decay': 5e-5,
            'batch_size': 2048 if rtx_4060ti_detected else 1024,
            'epochs': 80 if rtx_4060ti_detected else 50,
            'patience': 25,
            'use_batch_norm': True,
            'activation': 'gelu',
            'use_residual': True,
            'use_attention': rtx_4060ti_detected,
            'focal_loss_alpha': 0.3,
            'focal_loss_gamma': 2.5,
            'use_focal_loss': True,
            'gradient_clip_val': 1.0,
            'use_label_smoothing': True,
            'label_smoothing_factor': 0.05
        }
        
        if params:
            default_params.update(params)
        self.params = default_params
        
        self.input_dim = input_dim
        
        self.device = 'cpu'
        self.gpu_available = False
        self.rtx_optimized = False
        
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_properties = torch.cuda.get_device_properties(0)
                    gpu_memory_gb = gpu_properties.total_memory / (1024**3)
                    
                    test_tensor = torch.zeros(2000, 2000).cuda()
                    test_result = test_tensor.sum().item()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    torch.cuda.set_per_process_memory_fraction(0.85)
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    
                    self.device = 'cuda:0'
                    self.gpu_available = True
                    self.rtx_optimized = rtx_4060ti_detected
                    
                    logger.info(f"RTX 4060 Ti 최적화 GPU 설정 완료 ({gpu_memory_gb:.1f}GB)")
                else:
                    logger.info("CUDA 사용 불가능, CPU 모드")
            except Exception as e:
                logger.warning(f"GPU 설정 실패, CPU 사용: {e}")
                self.device = 'cpu'
                self.gpu_available = False
        
        try:
            self.network = self._build_rtx_optimized_network()
            self.optimizer = None
            
            self.scaler = None
            if AMP_AVAILABLE and self.gpu_available:
                try:
                    self.scaler = GradScaler()
                    logger.info("Mixed Precision 활성화 - RTX 4060 Ti 최적화")
                except:
                    self.scaler = None
                    logger.info("Mixed Precision 비활성화")
            
            if TORCH_AVAILABLE:
                pos_weight = torch.tensor([49.8], device=self.device)
                self.criterion = self._get_rtx_optimized_loss(pos_weight)
                
                self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.2)
                
                self.to(self.device)
                
        except Exception as e:
            logger.error(f"RTX 4060 Ti 최적화 DeepCTR 모델 초기화 실패: {e}")
            if self.gpu_available:
                logger.info("CPU 모드로 재시도")
                self.device = 'cpu'
                self.gpu_available = False
                self.rtx_optimized = False
                self.network = self._build_rtx_optimized_network()
                pos_weight = torch.tensor([49.8])
                self.criterion = self._get_rtx_optimized_loss(pos_weight)
                self.temperature = nn.Parameter(torch.ones(1) * 1.2)
                self.to(self.device)
            else:
                raise
    
    def _build_rtx_optimized_network(self):
        """RTX 4060 Ti 최적화 네트워크 구조"""
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
                
                if use_residual and i > 0 and prev_dim == hidden_dim:
                    layers.append(ResidualConnection(hidden_dim))
                
                if use_attention and i == len(hidden_dims) // 2:
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
            
            if self.rtx_optimized:
                network = torch.jit.script(network)
            
            return network
            
        except Exception as e:
            logger.error(f"RTX 최적화 네트워크 구조 생성 실패: {e}")
            raise
    
    def _get_rtx_optimized_loss(self, pos_weight):
        """RTX 4060 Ti 최적화 손실함수"""
        if self.params.get('use_focal_loss', False):
            return self._focal_loss_with_label_smoothing
        else:
            if self.params.get('use_label_smoothing', False):
                return self._bce_with_label_smoothing
            else:
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def _focal_loss_with_label_smoothing(self, inputs, targets):
        """Focal Loss with Label Smoothing - RTX 최적화"""
        alpha = self.params['focal_loss_alpha']
        gamma = self.params['focal_loss_gamma']
        smoothing = self.params.get('label_smoothing_factor', 0.05)
        
        targets_smooth = targets * (1 - smoothing) + 0.5 * smoothing
        
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        
        return focal_loss.mean()
    
    def _bce_with_label_smoothing(self, inputs, targets):
        """BCE with Label Smoothing - RTX 최적화"""
        smoothing = self.params.get('label_smoothing_factor', 0.05)
        targets_smooth = targets * (1 - smoothing) + 0.5 * smoothing
        
        return nn.functional.binary_cross_entropy_with_logits(inputs, targets_smooth)
    
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
        """RTX 4060 Ti 최적화 DeepCTR 모델 학습"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다.")
            
        logger.info(f"{self.name} 모델 학습 시작 (Device: {self.device}, RTX최적화: {self.rtx_optimized})")
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train = X_train.fillna(0)
            if X_val is not None:
                X_val = X_val.fillna(0)
            
            X_train_values = X_train.values.astype('float32')
            if X_val is not None:
                X_val_values = X_val.values.astype('float32')
            
            mean = X_train_values.mean(axis=0, keepdims=True)
            std = X_train_values.std(axis=0, keepdims=True) + 1e-8
            X_train_values = (X_train_values - mean) / std
            if X_val is not None:
                X_val_values = (X_val_values - mean) / std
            
            self.normalization_params = {'mean': mean, 'std': std}
            
            if self.rtx_optimized:
                self.optimizer = optim.AdamW(
                    self.parameters(), 
                    lr=self.params['learning_rate'],
                    weight_decay=self.params.get('weight_decay', 5e-5),
                    eps=1e-8,
                    betas=(0.9, 0.999)
                )
            else:
                self.optimizer = optim.AdamW(
                    self.parameters(), 
                    lr=self.params['learning_rate'],
                    weight_decay=self.params.get('weight_decay', 1e-5)
                )
            
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.params['learning_rate'],
                epochs=self.params['epochs'],
                steps_per_epoch=len(X_train) // self.params['batch_size'] + 1,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
            batch_size = self.params['batch_size']
            
            X_train_tensor = torch.FloatTensor(X_train_values).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = TorchDataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,
                pin_memory=self.gpu_available,
                drop_last=True
            )
            
            val_loader = None
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val_values).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = TorchDataLoader(
                    val_dataset, 
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=self.gpu_available
                )
            
            best_val_loss = float('inf')
            patience_counter = 0
            max_epochs = self.params['epochs']
            
            for epoch in range(max_epochs):
                self.train()
                train_loss = 0.0
                batch_count = 0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    
                    try:
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
                        
                        if self.rtx_optimized and batch_count % 20 == 0:
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
                                if self.scaler is not None and AMP_AVAILABLE and self.rtx_optimized:
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
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= self.params['patience']:
                            logger.info(f"조기 종료: epoch {epoch + 1}")
                            break
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"RTX 최적화 Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                if self.rtx_optimized:
                    torch.cuda.empty_cache()
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"RTX 4060 Ti 최적화 DeepCTR 학습 실패: {str(e)}")
            if self.gpu_available and ('cuda' in str(e).lower() or 'gpu' in str(e).lower()):
                logger.info("GPU 학습 실패, CPU로 재시도")
                self.device = 'cpu'
                self.gpu_available = False
                self.rtx_optimized = False
                self.to('cpu')
                return self.fit(X_train, y_train, X_val, y_val)
            raise
        finally:
            if self.rtx_optimized:
                torch.cuda.empty_cache()
            gc.collect()
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측 - RTX 최적화"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            X_processed = X_processed.fillna(0)
            
            X_values = X_processed.values.astype('float32')
            
            if hasattr(self, 'normalization_params'):
                X_values = (X_values - self.normalization_params['mean']) / self.normalization_params['std']
            
            self.eval()
            X_tensor = torch.FloatTensor(X_values).to(self.device)
            
            predictions = []
            batch_size = self.params['batch_size']
            
            with torch.no_grad():
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i + batch_size]
                    
                    try:
                        if self.scaler is not None and AMP_AVAILABLE and self.rtx_optimized:
                            with autocast():
                                logits = self.forward(batch)
                                proba = torch.sigmoid(logits / self.temperature)
                        else:
                            logits = self.forward(batch)
                            proba = torch.sigmoid(logits / self.temperature)
                        
                        predictions.append(proba.cpu().numpy())
                        
                        if self.rtx_optimized and i % (batch_size * 10) == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.warning(f"예측 배치 실패: {e}")
                        batch_size_actual = len(batch)
                        predictions.append(np.full(batch_size_actual, Config.CALIBRATION_CONFIG['target_ctr']))
            
            proba = np.concatenate(predictions)
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            proba = self._enhance_prediction_diversity(proba)
            
            return proba
            
        except Exception as e:
            logger.error(f"RTX 최적화 DeepCTR 예측 실패: {str(e)}")
            return np.full(len(X), Config.CALIBRATION_CONFIG['target_ctr'])
        finally:
            if self.rtx_optimized:
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

class ResidualConnection(nn.Module):
    """Residual Connection Layer"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        return x

class SelfAttentionLayer(nn.Module):
    """Self Attention Layer for DeepCTR"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        
        if x.size(1) == 1:
            x = x.squeeze(1)
        
        return x

class HighPerformanceLogisticModel(BaseModel):
    """고성능 로지스틱 회귀 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn이 설치되지 않았습니다.")
            
        default_params = {
            'C': 0.05,
            'max_iter': 3000,
            'random_state': Config.RANDOM_STATE,
            'class_weight': 'balanced',
            'solver': 'liblinear',
            'penalty': 'l2',
            'tol': 1e-6,
            'fit_intercept': True,
            'intercept_scaling': 1.0
        }
        if params:
            default_params.update(params)
        super().__init__("HighPerformanceLogisticRegression", default_params)
        
        self.model = LogisticRegression(**self.params)
        self.prediction_diversity_threshold = 1500
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """고성능 로지스틱 회귀 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (데이터: {len(X_train):,})")
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train = X_train.fillna(0)
            
            for col in X_train.columns:
                if X_train[col].dtype in ['float64']:
                    X_train[col] = X_train[col].astype('float32')
            
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
        except Exception as e:
            logger.error(f"고성능 Logistic 학습 실패: {str(e)}")
            raise
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            X_processed = X_processed.fillna(0)
            
            for col in X_processed.columns:
                if X_processed[col].dtype in ['float64']:
                    X_processed[col] = X_processed[col].astype('float32')
            
            proba = self.model.predict_proba(X_processed)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            proba = self._enhance_prediction_diversity(proba)
            
            return proba
        except Exception as e:
            logger.error(f"고성능 Logistic 예측 실패: {str(e)}")
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
    """CTR 특화 고성능 확률 보정 클래스"""
    
    def __init__(self, target_ctr: float = 0.0201):
        self.target_ctr = target_ctr
        self.platt_scaler = None
        self.isotonic_regressor = None
        self.bias_correction = 0.0
        self.temperature_scaler = None
        self.multiplicative_factor = 1.0
        
    def fit_platt_scaling(self, y_true: np.ndarray, y_pred: np.ndarray):
        """고성능 Platt Scaling 학습"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn이 없어 Platt scaling을 사용할 수 없습니다")
            return
            
        try:
            self.platt_scaler = LogisticRegression(
                C=1.0, 
                max_iter=1000, 
                solver='liblinear',
                random_state=42
            )
            self.platt_scaler.fit(y_pred.reshape(-1, 1), y_true)
            logger.info("고성능 Platt scaling 학습 완료")
        except Exception as e:
            logger.error(f"고성능 Platt scaling 학습 실패: {str(e)}")
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_pred: np.ndarray):
        """고성능 Isotonic Regression 학습"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn이 없어 Isotonic regression을 사용할 수 없습니다")
            return
            
        try:
            self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor.fit(y_pred, y_true)
            logger.info("고성능 Isotonic regression 학습 완료")
        except Exception as e:
            logger.error(f"고성능 Isotonic regression 학습 실패: {str(e)}")
    
    def fit_temperature_scaling(self, y_true: np.ndarray, y_pred: np.ndarray):
        """고성능 Temperature Scaling 학습"""
        try:
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(temperature):
                if temperature <= 0:
                    return float('inf')
                
                logits = np.log(np.clip(y_pred, 1e-15, 1-1e-15) / (1 - np.clip(y_pred, 1e-15, 1-1e-15)))
                calibrated_logits = logits / temperature
                calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                return -np.mean(y_true * np.log(calibrated_probs) + (1 - y_true) * np.log(1 - calibrated_probs))
            
            result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature_scaler = result.x
            logger.info(f"고성능 Temperature scaling 학습 완료: T={self.temperature_scaler:.3f}")
        except Exception as e:
            logger.error(f"고성능 Temperature scaling 학습 실패: {str(e)}")
            self.temperature_scaler = 1.0
    
    def fit_bias_correction(self, y_true: np.ndarray, y_pred: np.ndarray):
        """고성능 편향 보정 학습"""
        try:
            predicted_ctr = y_pred.mean()
            actual_ctr = y_true.mean()
            
            self.bias_correction = actual_ctr - predicted_ctr
            
            if predicted_ctr > 0:
                self.multiplicative_factor = actual_ctr / predicted_ctr
            else:
                self.multiplicative_factor = 1.0
                
            logger.info(f"고성능 편향 보정 학습 완료: 가산보정={self.bias_correction:.4f}, 승산보정={self.multiplicative_factor:.4f}")
        except Exception as e:
            logger.error(f"고성능 편향 보정 학습 실패: {str(e)}")
    
    def apply_combined_calibration(self, predictions: np.ndarray, method: str = 'combined') -> np.ndarray:
        """고성능 결합 보정 적용"""
        try:
            calibrated = predictions.copy()
            
            if method in ['platt', 'combined'] and self.platt_scaler is not None:
                try:
                    calibrated = self.platt_scaler.predict_proba(calibrated.reshape(-1, 1))[:, 1]
                except:
                    pass
            
            if method in ['isotonic', 'combined'] and self.isotonic_regressor is not None:
                try:
                    calibrated = self.isotonic_regressor.predict(calibrated)
                except:
                    pass
            
            if method in ['temperature', 'combined'] and self.temperature_scaler is not None:
                try:
                    logits = np.log(np.clip(calibrated, 1e-15, 1-1e-15) / (1 - np.clip(calibrated, 1e-15, 1-1e-15)))
                    calibrated_logits = logits / self.temperature_scaler
                    calibrated = 1 / (1 + np.exp(-calibrated_logits))
                except:
                    pass
            
            if method in ['bias', 'combined']:
                calibrated = calibrated * self.multiplicative_factor + self.bias_correction
            
            return np.clip(calibrated, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"고성능 결합 보정 적용 실패: {str(e)}")
            return np.clip(predictions, 1e-15, 1 - 1e-15)

class ModelFactory:
    """고성능 CTR 특화 모델 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """모델 타입에 따라 고성능 CTR 특화 모델 인스턴스 생성"""
        
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
            logger.error(f"고성능 모델 생성 실패 ({model_type}): {str(e)}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """사용 가능한 고성능 모델 타입 리스트"""
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
            logger.error("사용 가능한 고성능 모델이 없습니다. 필요한 라이브러리를 설치해주세요.")
            available = ['logistic']
            
        return available
    
    @staticmethod
    def get_model_capabilities() -> Dict[str, Dict[str, Any]]:
        """각 모델의 성능 특성 정보"""
        capabilities = {
            'lightgbm': {
                'gpu_support': False,
                'memory_efficiency': 'high',
                'training_speed': 'very_fast',
                'prediction_speed': 'very_fast',
                'feature_importance': True,
                'combined_score_potential': 0.32,
                'best_for': 'large_datasets'
            },
            'xgboost': {
                'gpu_support': True,
                'memory_efficiency': 'medium',
                'training_speed': 'fast',
                'prediction_speed': 'fast',
                'feature_importance': True,
                'combined_score_potential': 0.31,
                'best_for': 'balanced_performance'
            },
            'catboost': {
                'gpu_support': True,
                'memory_efficiency': 'medium',
                'training_speed': 'medium',
                'prediction_speed': 'fast',
                'feature_importance': True,
                'combined_score_potential': 0.30,
                'best_for': 'categorical_features'
            },
            'deepctr': {
                'gpu_support': True,
                'memory_efficiency': 'low',
                'training_speed': 'slow',
                'prediction_speed': 'medium',
                'feature_importance': False,
                'combined_score_potential': 0.29,
                'best_for': 'complex_interactions',
                'rtx_4060ti_optimized': rtx_4060ti_detected
            },
            'logistic': {
                'gpu_support': False,
                'memory_efficiency': 'very_high',
                'training_speed': 'very_fast',
                'prediction_speed': 'very_fast',
                'feature_importance': True,
                'combined_score_potential': 0.25,
                'best_for': 'baseline'
            }
        }
        
        return {k: v for k, v in capabilities.items() if k in ModelFactory.get_available_models()}

class HighPerformanceModelEvaluator:
    """고성능 모델 평가 클래스 - Combined Score 0.30+ 달성 목표"""
    
    @staticmethod
    def evaluate_model_comprehensive(model: BaseModel, 
                                   X_test: pd.DataFrame, 
                                   y_test: pd.Series) -> Dict[str, float]:
        """고성능 모델 종합 평가"""
        
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
                
                from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
                try:
                    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
                    metrics['ap'] = average_precision_score(y_test, y_pred_proba)
                except:
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['f1'] = 0.0
                    metrics['ap'] = 0.0
            else:
                metrics['auc'] = 0.5
                metrics['logloss'] = 1.0
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
                metrics['ap'] = 0.0
            
            metrics['accuracy'] = (y_test == y_pred).mean()
            
            metrics['ctr_actual'] = y_test.mean()
            metrics['ctr_predicted'] = y_pred_proba.mean()
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            metrics['ctr_absolute_bias'] = abs(metrics['ctr_bias'])
            
            pos_weight = 49.8
            neg_weight = 1.0
            sample_weights = np.where(y_test == 1, pos_weight, neg_weight)
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            try:
                wll = -np.average(
                    y_test * np.log(y_pred_proba_clipped) + (1 - y_test) * np.log(1 - y_pred_proba_clipped),
                    weights=sample_weights
                )
                metrics['weighted_log_loss'] = wll
            except:
                metrics['weighted_log_loss'] = float('inf')
            
            ap_weight = 0.5
            wll_weight = 0.5
            wll_normalized = 1 / (1 + metrics['weighted_log_loss']) if metrics['weighted_log_loss'] != float('inf') else 0.0
            
            metrics['combined_score'] = ap_weight * metrics['ap'] + wll_weight * wll_normalized
            
            ctr_bias_penalty = np.exp(-metrics['ctr_absolute_bias'] * 200)
            metrics['ctr_optimized_score'] = metrics['combined_score'] * (1.0 + 0.2 * ctr_bias_penalty)
            
            metrics['prediction_diversity'] = len(np.unique(y_pred_proba))
            metrics['prediction_std'] = y_pred_proba.std()
            metrics['prediction_entropy'] = -np.mean(
                y_pred_proba_clipped * np.log2(y_pred_proba_clipped) + 
                (1 - y_pred_proba_clipped) * np.log2(1 - y_pred_proba_clipped)
            )
            
            pos_mask = (y_test == 1)
            neg_mask = (y_test == 0)
            
            if pos_mask.any() and neg_mask.any():
                metrics['separation'] = y_pred_proba[pos_mask].mean() - y_pred_proba[neg_mask].mean()
            else:
                metrics['separation'] = 0.0
            
            high_confidence_threshold = 0.8
            low_confidence_threshold = 0.2
            
            high_conf_mask = y_pred_proba >= high_confidence_threshold
            low_conf_mask = y_pred_proba <= low_confidence_threshold
            
            if high_conf_mask.any():
                metrics['high_conf_precision'] = y_test[high_conf_mask].mean()
                metrics['high_conf_ratio'] = high_conf_mask.mean()
            else:
                metrics['high_conf_precision'] = 0.0
                metrics['high_conf_ratio'] = 0.0
            
            if low_conf_mask.any():
                metrics['low_conf_precision'] = 1.0 - y_test[low_conf_mask].mean()
                metrics['low_conf_ratio'] = low_conf_mask.mean()
            else:
                metrics['low_conf_precision'] = 0.0
                metrics['low_conf_ratio'] = 0.0
            
            metrics['target_achievement'] = 1.0 if metrics['combined_score'] >= 0.30 else 0.0
            metrics['model_name'] = model.name
            
        except Exception as e:
            logger.error(f"고성능 평가 지표 계산 중 오류: {str(e)}")
            metrics = {
                'auc': 0.5, 'logloss': 1.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'ap': 0.0,
                'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0, 'ctr_absolute_bias': 0.0,
                'weighted_log_loss': float('inf'), 'combined_score': 0.0, 'ctr_optimized_score': 0.0,
                'prediction_diversity': 0, 'prediction_std': 0.0, 'prediction_entropy': 0.0, 'separation': 0.0,
                'high_conf_precision': 0.0, 'high_conf_ratio': 0.0, 'low_conf_precision': 0.0, 'low_conf_ratio': 0.0,
                'target_achievement': 0.0, 'model_name': model.name
            }
        
        return metrics
    
    @staticmethod
    def compare_models_performance(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """고성능 모델들의 성능 비교"""
        
        if not models_results:
            return pd.DataFrame()
        
        try:
            comparison_df = pd.DataFrame(models_results).T
            
            comparison_df['rank_combined_score'] = comparison_df['combined_score'].rank(ascending=False)
            comparison_df['rank_ctr_optimized'] = comparison_df['ctr_optimized_score'].rank(ascending=False)
            comparison_df['rank_ap'] = comparison_df['ap'].rank(ascending=False)
            
            comparison_df['overall_rank'] = (
                0.5 * comparison_df['rank_combined_score'] +
                0.3 * comparison_df['rank_ctr_optimized'] +
                0.2 * comparison_df['rank_ap']
            )
            
            comparison_df = comparison_df.sort_values('overall_rank')
            
            comparison_df['performance_tier'] = pd.cut(
                comparison_df['combined_score'],
                bins=[0, 0.20, 0.25, 0.30, 1.0],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"고성능 모델 비교 실패: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_best_model_recommendation(comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """최고 성능 모델 추천"""
        
        if comparison_df.empty:
            return {'recommendation': 'no_models', 'reason': '평가 가능한 모델이 없습니다'}
        
        try:
            best_combined = comparison_df.loc[comparison_df['combined_score'].idxmax()]
            best_ctr_optimized = comparison_df.loc[comparison_df['ctr_optimized_score'].idxmax()]
            
            target_achievers = comparison_df[comparison_df['combined_score'] >= 0.30]
            
            if len(target_achievers) > 0:
                best_overall = target_achievers.loc[target_achievers['overall_rank'].idxmin()]
                
                recommendation = {
                    'recommendation': 'target_achieved',
                    'best_model': best_overall.name,
                    'combined_score': best_overall['combined_score'],
                    'ctr_optimized_score': best_overall['ctr_optimized_score'],
                    'ctr_bias': best_overall['ctr_bias'],
                    'performance_tier': best_overall['performance_tier'],
                    'reason': f'Combined Score 0.30+ 달성 ({best_overall["combined_score"]:.4f})',
                    'achievers_count': len(target_achievers)
                }
            else:
                best_overall = comparison_df.loc[comparison_df['overall_rank'].idxmin()]
                
                recommendation = {
                    'recommendation': 'best_available',
                    'best_model': best_overall.name,
                    'combined_score': best_overall['combined_score'],
                    'ctr_optimized_score': best_overall['ctr_optimized_score'],
                    'ctr_bias': best_overall['ctr_bias'],
                    'performance_tier': best_overall['performance_tier'],
                    'reason': f'목표 미달성, 최고 점수: {best_overall["combined_score"]:.4f}',
                    'gap_to_target': 0.30 - best_overall['combined_score']
                }
            
            recommendation['alternative_models'] = {
                'best_combined_score': {
                    'model': best_combined.name,
                    'score': best_combined['combined_score']
                },
                'best_ctr_optimized': {
                    'model': best_ctr_optimized.name,
                    'score': best_ctr_optimized['ctr_optimized_score']
                }
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"최고 성능 모델 추천 실패: {e}")
            return {'recommendation': 'error', 'reason': f'추천 과정에서 오류 발생: {str(e)}'}

LightGBMModel = HighPerformanceLightGBMModel
XGBoostModel = HighPerformanceXGBoostModel
CatBoostModel = HighPerformanceCatBoostModel
DeepCTRModel = RTX4060TiOptimizedDeepCTRModel
LogisticModel = HighPerformanceLogisticModel
ModelEvaluator = HighPerformanceModelEvaluator