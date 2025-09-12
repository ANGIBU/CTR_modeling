# models.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod
import pickle

# 트리 기반 모델
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# 신경망 모델
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

# Calibration 모듈
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# 기타 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss

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
        try:
            # 기본 예측 수행
            train_pred = self.predict_proba_raw(X_train)
            
            # Calibration 방법 선택
            if method == 'platt':
                calibrator = CalibratedClassifierCV(
                    base_estimator=None, 
                    method='sigmoid', 
                    cv=cv_folds
                )
            elif method == 'isotonic':
                calibrator = CalibratedClassifierCV(
                    base_estimator=None, 
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
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0.0
            
            X = X[self.feature_names]
            return X
        except Exception as e:
            logger.warning(f"피처 일관성 보장 실패: {str(e)}")
            return X

class LightGBMModel(BaseModel):
    """LightGBM 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = Config.LGBM_PARAMS.copy()
        if params:
            default_params.update(params)
        super().__init__("LightGBM", default_params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """LightGBM 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        self.feature_names = list(X_train.columns)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        callbacks = []
        if self.params.get('early_stopping_rounds'):
            callbacks.append(lgb.early_stopping(self.params.get('early_stopping_rounds', 200)))
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.is_fitted = True
        logger.info(f"{self.name} 모델 학습 완료")
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
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
        default_params = Config.XGB_PARAMS.copy()
        if params:
            default_params.update(params)
        super().__init__("XGBoost", default_params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """XGBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        self.feature_names = list(X_train.columns)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'valid'))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            evals=evals,
            early_stopping_rounds=self.params.get('early_stopping_rounds', 200),
            verbose_eval=False
        )
        
        self.is_fitted = True
        logger.info(f"{self.name} 모델 학습 완료")
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            dtest = xgb.DMatrix(X_processed)
            
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
                proba = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
            else:
                proba = self.model.predict(dtest)
            
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
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
        default_params = Config.CAT_PARAMS.copy()
        if params:
            default_params.update(params)
        super().__init__("CatBoost", default_params)
        
        self.model = CatBoostClassifier(**self.params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """CatBoost 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        self.feature_names = list(X_train.columns)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        try:
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
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        
        return self
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            X_array = X_processed.values
            
            proba = self.model.predict_proba(X_array)
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

class DeepCTRModel(BaseModel, nn.Module):
    """GPU 기반 딥러닝 CTR 모델"""
    
    def __init__(self, input_dim: int, params: Dict[str, Any] = None):
        BaseModel.__init__(self, "DeepCTR", params)
        nn.Module.__init__(self)
        
        default_params = Config.NN_PARAMS.copy()
        if params:
            default_params.update(params)
        self.params = default_params
        
        self.input_dim = input_dim
        self.device = Config.DEVICE
        
        # 네트워크 구조 정의
        self.network = self._build_network()
        
        # 옵티마이저 및 손실함수
        self.optimizer = None
        self.scaler = GradScaler() if Config.USE_MIXED_PRECISION else None
        
        # CTR 특화 가중 손실함수
        pos_weight = torch.tensor([49.0], device=self.device)  # 1/0.0201 - 1
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.to(self.device)
        
        # Calibration 관련
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def _build_network(self):
        """네트워크 구조 생성"""
        hidden_dims = self.params['hidden_dims']
        dropout_rate = self.params['dropout_rate']
        use_batch_norm = self.params.get('use_batch_norm', True)
        activation = self.params.get('activation', 'relu')
        
        layers = []
        prev_dim = self.input_dim
        
        # 입력층 정규화
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
    
    def forward(self, x):
        """순전파"""
        return self.network(x).squeeze(-1)
    
    def forward_with_temperature(self, x):
        """Temperature scaling이 적용된 순전파"""
        logits = self.forward(x)
        return logits / self.temperature
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """딥러닝 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작 (GPU: {torch.cuda.is_available()})")
        
        self.feature_names = list(X_train.columns)
        
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
            factor=0.5, 
            patience=10,
            min_lr=1e-6
        )
        
        # 데이터 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 검증 데이터 준비
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.params['batch_size'],
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.params['epochs']):
            # 학습 모드
            self.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
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
            
            train_loss /= len(train_loader)
            
            # 검증
            val_loss = train_loss
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        if self.scaler is not None:
                            with autocast():
                                logits = self.forward(batch_X)
                                loss = self.criterion(logits, batch_y)
                        else:
                            logits = self.forward(batch_X)
                            loss = self.criterion(logits, batch_y)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 스케줄러 업데이트
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
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Temperature scaling 보정
        if val_loader is not None:
            self._calibrate_temperature(val_loader)
        
        self.is_fitted = True
        logger.info(f"{self.name} 모델 학습 완료")
        
        return self
    
    def _calibrate_temperature(self, val_loader):
        """Temperature scaling으로 보정"""
        self.eval()
        
        # Temperature 파라미터만 학습
        temp_optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def temp_loss():
            temp_optimizer.zero_grad()
            loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = self.forward(batch_X)
                    
            for batch_X, batch_y in val_loader:
                scaled_logits = self.forward_with_temperature(batch_X)
                loss += self.criterion(scaled_logits, batch_y)
            
            loss.backward()
            return loss
        
        temp_optimizer.step(temp_loss)
        logger.info(f"Temperature scaling 완료: {self.temperature.item():.3f}")
    
    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """보정되지 않은 원본 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            X_processed = self._ensure_feature_consistency(X)
            
            self.eval()
            X_tensor = torch.FloatTensor(X_processed.values).to(self.device)
            
            predictions = []
            batch_size = self.params['batch_size']
            
            with torch.no_grad():
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i + batch_size]
                    
                    if self.scaler is not None:
                        with autocast():
                            logits = self.forward(batch)
                            proba = torch.sigmoid(logits)
                    else:
                        logits = self.forward(batch)
                        proba = torch.sigmoid(logits)
                    
                    predictions.append(proba.cpu().numpy())
            
            proba = np.concatenate(predictions)
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
        except Exception as e:
            logger.error(f"DeepCTR 예측 실패: {str(e)}")
            return np.full(len(X), Config.CALIBRATION_CONFIG['target_ctr'])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Temperature scaling이 적용된 확률 예측"""
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
        
        self.feature_names = list(X_train.columns)
        
        try:
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
        try:
            self.platt_scaler = LogisticRegression()
            self.platt_scaler.fit(y_pred.reshape(-1, 1), y_true)
            logger.info("Platt scaling 학습 완료")
        except Exception as e:
            logger.error(f"Platt scaling 학습 실패: {str(e)}")
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Isotonic Regression 학습"""
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
    
    def apply_platt_scaling(self, y_pred: np.ndarray) -> np.ndarray:
        """Platt Scaling 적용"""
        if self.platt_scaler is None:
            return y_pred
        
        try:
            calibrated = self.platt_scaler.predict_proba(y_pred.reshape(-1, 1))[:, 1]
            return np.clip(calibrated, 1e-15, 1 - 1e-15)
        except:
            return y_pred
    
    def apply_isotonic_regression(self, y_pred: np.ndarray) -> np.ndarray:
        """Isotonic Regression 적용"""
        if self.isotonic_regressor is None:
            return y_pred
        
        try:
            calibrated = self.isotonic_regressor.predict(y_pred)
            return np.clip(calibrated, 1e-15, 1 - 1e-15)
        except:
            return y_pred
    
    def apply_bias_correction(self, y_pred: np.ndarray) -> np.ndarray:
        """편향 보정 적용"""
        try:
            corrected = y_pred + self.bias_correction
            return np.clip(corrected, 1e-15, 1 - 1e-15)
        except:
            return y_pred

class ModelFactory:
    """모델 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """모델 타입에 따라 모델 인스턴스 생성"""
        
        if model_type.lower() == 'lightgbm':
            return LightGBMModel(kwargs.get('params'))
        
        elif model_type.lower() == 'xgboost':
            return XGBoostModel(kwargs.get('params'))
        
        elif model_type.lower() == 'catboost':
            return CatBoostModel(kwargs.get('params'))
        
        elif model_type.lower() == 'deepctr':
            input_dim = kwargs.get('input_dim')
            if input_dim is None:
                raise ValueError("DeepCTR 모델에는 input_dim이 필요합니다.")
            return DeepCTRModel(input_dim, kwargs.get('params'))
        
        elif model_type.lower() == 'logistic':
            return LogisticModel(kwargs.get('params'))
        
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """사용 가능한 모델 타입 리스트"""
        return ['lightgbm', 'xgboost', 'catboost', 'deepctr', 'logistic']

class ModelEvaluator:
    """모델 평가 클래스"""
    
    @staticmethod
    def evaluate_model(model: BaseModel, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """모델 평가 수행"""
        
        y_pred_proba = model.predict_proba(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {}
        
        try:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['logloss'] = log_loss(y_test, y_pred_proba)
            
            metrics['accuracy'] = (y_test == y_pred).mean()
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
            
            metrics['ctr_actual'] = y_test.mean()
            metrics['ctr_predicted'] = y_pred_proba.mean()
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            
        except Exception as e:
            logger.warning(f"평가 지표 계산 중 오류: {str(e)}")
        
        return metrics