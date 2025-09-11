# models.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import ABC, abstractmethod

# 트리 기반 모델
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# 신경망 모델
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neural_network import MLPClassifier

# 기타 모델
from sklearn.linear_model import LogisticRegression
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
        
        # 학습 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # 검증 데이터가 있는 경우
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # 콜백 설정
        callbacks = []
        if self.params.get('early_stopping_rounds'):
            callbacks.append(lgb.early_stopping(self.params.get('early_stopping_rounds', 100)))
        
        # 모델 학습
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
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            # best_iteration이 있으면 사용
            num_iteration = getattr(self.model, 'best_iteration', None)
            proba = self.model.predict(X, num_iteration=num_iteration)
            
            # 확률값 클리핑
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
        except Exception as e:
            logger.error(f"LightGBM 예측 실패: {str(e)}")
            # 기본값 반환
            return np.full(len(X), 0.0191)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        
        return dict(zip(feature_names, importance))

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
        
        # 학습 데이터셋 생성
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # 검증 데이터가 있는 경우
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'valid'))
        
        # 모델 학습
        self.model = xgb.train(
            self.params,
            dtrain,
            evals=evals,
            early_stopping_rounds=self.params.get('early_stopping_rounds', 100),
            verbose_eval=False
        )
        
        self.is_fitted = True
        logger.info(f"{self.name} 모델 학습 완료")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            dtest = xgb.DMatrix(X)
            
            # best_iteration이 있으면 사용
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
                proba = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
            else:
                proba = self.model.predict(dtest)
            
            # 확률값 클리핑
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
        except Exception as e:
            logger.error(f"XGBoost 예측 실패: {str(e)}")
            # 기본값 반환
            return np.full(len(X), 0.0191)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.model.get_score(importance_type='gain')

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
        
        # 검증 데이터 설정
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        try:
            # 모델 학습
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                use_best_model=True if eval_set is not None else False,
                plot=False
            )
            
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
            
        except Exception as e:
            logger.error(f"CatBoost 학습 실패: {str(e)}")
            # 기본 모델로 학습
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            proba = self.model.predict_proba(X)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            # 확률값 클리핑
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
        except Exception as e:
            logger.error(f"CatBoost 예측 실패: {str(e)}")
            # 기본값 반환
            return np.full(len(X), 0.0191)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            importance = self.model.get_feature_importance()
            feature_names = self.model.feature_names_
            
            return dict(zip(feature_names, importance))
        except:
            return {}

class DeepCTRModel(BaseModel, nn.Module):
    """딥러닝 기반 CTR 모델"""
    
    def __init__(self, input_dim: int, params: Dict[str, Any] = None):
        BaseModel.__init__(self, "DeepCTR", params)
        nn.Module.__init__(self)
        
        default_params = Config.NN_PARAMS.copy()
        if params:
            default_params.update(params)
        self.params = default_params
        
        self.input_dim = input_dim
        hidden_dims = self.params['hidden_dims']
        dropout_rate = self.params['dropout_rate']
        
        # 네트워크 구조 정의
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 출력층
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # 옵티마이저 및 손실함수
        self.optimizer = None
        
        # CTR 특화 손실함수 (가중 BCE)
        pos_weight = torch.tensor([51.2])  # 1/0.0191 - 1
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        """순전파"""
        return self.network(x).squeeze()
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """딥러닝 모델 학습"""
        logger.info(f"{self.name} 모델 학습 시작")
        
        # 옵티마이저 초기화
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        
        # 데이터 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train.values).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True
        )
        
        # 검증 데이터 준비
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'])
        
        # 학습 루프
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.params['epochs']):
            # 학습 모드
            self.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Logits 출력 (Sigmoid 제거)
                logits = self.network[:-1](batch_X).squeeze()
                loss = self.criterion(logits, batch_y)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 검증
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        logits = self.network[:-1](batch_X).squeeze()
                        loss = self.criterion(logits, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 조기 종료 확인
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
        
        self.is_fitted = True
        logger.info(f"{self.name} 모델 학습 완료")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            self.eval()
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            
            with torch.no_grad():
                outputs = self.forward(X_tensor)
                proba = outputs.cpu().numpy()
            
            # 확률값 클리핑
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
        except Exception as e:
            logger.error(f"DeepCTR 예측 실패: {str(e)}")
            # 기본값 반환
            return np.full(len(X), 0.0191)

class LogisticModel(BaseModel):
    """로지스틱 회귀 모델"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': Config.RANDOM_STATE,
            'class_weight': 'balanced'  # CTR 불균형 처리
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
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            logger.info(f"{self.name} 모델 학습 완료")
        except Exception as e:
            logger.error(f"Logistic 학습 실패: {str(e)}")
            raise
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        try:
            proba = self.model.predict_proba(X)
            if proba.ndim == 2:
                proba = proba[:, 1]
            
            # 확률값 클리핑
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
        except Exception as e:
            logger.error(f"Logistic 예측 실패: {str(e)}")
            # 기본값 반환
            return np.full(len(X), 0.0191)

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
        
        # 확률 예측
        y_pred_proba = model.predict_proba(X_test)
        
        # 이진 예측
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # 평가 지표 계산
        metrics = {}
        
        try:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['logloss'] = log_loss(y_test, y_pred_proba)
            
            # 정확도
            metrics['accuracy'] = (y_test == y_pred).mean()
            
            # 정밀도, 재현율, F1
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
            
            # CTR 관련 지표
            metrics['ctr_actual'] = y_test.mean()
            metrics['ctr_predicted'] = y_pred_proba.mean()
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            
        except Exception as e:
            logger.warning(f"평가 지표 계산 중 오류: {str(e)}")
        
        return metrics