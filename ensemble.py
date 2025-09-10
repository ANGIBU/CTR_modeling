# ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna

from config import Config
from models import BaseModel
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """앙상블 모델 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """앙상블 모델 학습"""
        pass
    
    @abstractmethod
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """앙상블 예측"""
        pass
    
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
    
    def get_base_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """모든 기본 모델의 예측 수집"""
        predictions = {}
        
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                predictions[name] = np.zeros(len(X))
        
        return predictions

class WeightedBlending(BaseEnsemble):
    """가중 블렌딩 앙상블"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("WeightedBlending")
        self.weights = weights or {}
        self.optimized_weights = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """가중치 최적화"""
        logger.info("가중 블렌딩 앙상블 학습 시작")
        
        # 실제 사용 가능한 모델명으로 가중치 매핑
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models}")
        
        if self.weights:
            # 사전 정의된 가중치를 실제 모델명으로 매핑
            mapped_weights = {}
            for model_name in available_models:
                if 'lightgbm' in model_name or 'lgb' in model_name:
                    mapped_weights[model_name] = self.weights.get('lgbm', 0.4)
                elif 'xgboost' in model_name or 'xgb' in model_name:
                    mapped_weights[model_name] = self.weights.get('xgb', 0.3)
                elif 'catboost' in model_name or 'cat' in model_name:
                    mapped_weights[model_name] = self.weights.get('cat', 0.2)
                elif 'deepctr' in model_name or 'nn' in model_name:
                    mapped_weights[model_name] = self.weights.get('nn', 0.1)
                else:
                    # 기본값으로 균등 가중치
                    mapped_weights[model_name] = 1.0 / len(available_models)
            
            self.optimized_weights = mapped_weights
        else:
            # 가중치 최적화
            self.optimized_weights = self._optimize_weights(base_predictions, y)
        
        # 가중치 정규화
        total_weight = sum(self.optimized_weights.values())
        if total_weight > 0:
            self.optimized_weights = {k: v/total_weight for k, v in self.optimized_weights.items()}
        else:
            # 모든 가중치가 0인 경우 균등 가중치
            self.optimized_weights = {k: 1.0/len(available_models) for k in available_models}
        
        self.is_fitted = True
        logger.info(f"최적화된 가중치: {self.optimized_weights}")
    
    def _optimize_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Optuna를 사용한 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        metrics_calc = CTRMetrics()
        
        def objective(trial):
            weights = {}
            for name in model_names:
                weights[name] = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
            
            # 가중치 정규화
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # 앙상블 예측
            ensemble_pred = np.zeros(len(y))
            for name, weight in weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
            
            # 성능 평가
            score = metrics_calc.combined_score(y, ensemble_pred)
            return score
        
        # 최적화 실행
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        
        # 결과를 실제 모델명으로 매핑
        optimized_weights = {}
        for param_name, weight in study.best_params.items():
            model_name = param_name.replace('weight_', '')
            optimized_weights[model_name] = weight
        
        return optimized_weights
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 블렌딩 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다.")
        
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.optimized_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred

class StackingEnsemble(BaseEnsemble):
    """스태킹 앙상블"""
    
    def __init__(self, meta_model_type: str = 'logistic', cv_folds: int = 5):
        super().__init__("StackingEnsemble")
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.meta_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Optional[Dict[str, np.ndarray]] = None):
        """스태킹 앙상블 학습"""
        logger.info("스태킹 앙상블 학습 시작")
        
        # Out-of-fold 예측 생성
        oof_predictions = self._generate_oof_predictions(X, y)
        
        # 메타 모델 학습
        self._train_meta_model(oof_predictions, y)
        
        self.is_fitted = True
        logger.info("스태킹 앙상블 학습 완료")
    
    def _generate_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Out-of-fold 예측 생성"""
        
        oof_predictions = pd.DataFrame(index=X.index)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in self.base_models.items():
            oof_pred = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                # 폴드별 학습
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                
                # 모델 복사 및 학습
                fold_model = self._clone_model(model)
                fold_model.fit(X_train_fold, y_train_fold)
                
                # 검증 세트 예측
                oof_pred[val_idx] = fold_model.predict_proba(X_val_fold)
            
            oof_predictions[model_name] = oof_pred
        
        return oof_predictions
    
    def _clone_model(self, model: BaseModel) -> BaseModel:
        """모델 복사"""
        # 간단한 방법으로 모델의 타입과 파라미터를 이용해 새 인스턴스 생성
        from models import ModelFactory
        
        model_type = model.name.lower()
        return ModelFactory.create_model(model_type, params=model.params)
    
    def _train_meta_model(self, oof_predictions: pd.DataFrame, y: pd.Series):
        """메타 모델 학습"""
        
        if self.meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.meta_model_type == 'rf':
            self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"지원하지 않는 메타 모델: {self.meta_model_type}")
        
        self.meta_model.fit(oof_predictions, y)
        
        # 메타 모델 성능 확인
        meta_pred = self.meta_model.predict_proba(oof_predictions)[:, 1]
        auc_score = roc_auc_score(y, meta_pred)
        logger.info(f"메타 모델 AUC: {auc_score:.4f}")
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """스태킹 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다.")
        
        # 기본 모델 예측을 DataFrame으로 변환
        pred_df = pd.DataFrame(base_predictions)
        
        # 메타 모델로 최종 예측
        ensemble_pred = self.meta_model.predict_proba(pred_df)[:, 1]
        
        return ensemble_pred

class RankAveraging(BaseEnsemble):
    """순위 평균 앙상블"""
    
    def __init__(self):
        super().__init__("RankAveraging")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """순위 평균은 별도 학습이 필요 없음"""
        self.is_fitted = True
        logger.info("순위 평균 앙상블 준비 완료")
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """순위 평균 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다.")
        
        # 각 모델의 예측을 순위로 변환
        ranks = []
        for pred in base_predictions.values():
            rank = pd.Series(pred).rank(pct=True).values  # 백분위 순위
            ranks.append(rank)
        
        # 순위 평균
        ensemble_rank = np.mean(ranks, axis=0)
        
        return ensemble_rank

class DynamicEnsemble(BaseEnsemble):
    """동적 앙상블 (샘플별 최적 모델 선택)"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__("DynamicEnsemble")
        self.confidence_threshold = confidence_threshold
        self.model_performance = {}
        self.routing_model = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """동적 앙상블 학습"""
        logger.info("동적 앙상블 학습 시작")
        
        # 각 모델의 성능 계산
        metrics_calc = CTRMetrics()
        
        for name, pred in base_predictions.items():
            score = metrics_calc.combined_score(y, pred)
            self.model_performance[name] = score
        
        # 라우팅 모델 학습 (어떤 모델을 선택할지 결정)
        self._train_routing_model(X, y, base_predictions)
        
        self.is_fitted = True
        logger.info("동적 앙상별 학습 완료")
    
    def _train_routing_model(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """라우팅 모델 학습"""
        
        # 각 샘플에 대해 최고 성능 모델 라벨 생성
        model_names = list(base_predictions.keys())
        best_models = []
        
        for i in range(len(y)):
            sample_scores = {}
            
            for name, pred in base_predictions.items():
                # 개별 샘플에 대한 모델 신뢰도 계산 (단순화)
                confidence = abs(pred[i] - 0.5) * 2
                sample_scores[name] = confidence * self.model_performance[name]
            
            best_model = max(sample_scores, key=sample_scores.get)
            best_models.append(model_names.index(best_model))
        
        # 라우팅 모델 학습
        self.routing_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # 피처가 부족한 경우 기본 통계 피처 추가
        if X.shape[1] < 5:
            X_extended = X.copy()
            X_extended['row_sum'] = X.sum(axis=1)
            X_extended['row_mean'] = X.mean(axis=1)
            X_extended['row_std'] = X.std(axis=1)
            X = X_extended
        
        self.routing_model.fit(X, best_models)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """동적 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다.")
        
        # 단순화된 동적 선택 (실제로는 입력 피처가 필요)
        # 여기서는 신뢰도 기반 가중 평균 사용
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        total_weight = 0
        
        for name, pred in base_predictions.items():
            model_weight = self.model_performance.get(name, 0.1)
            confidence = np.abs(pred - 0.5) * 2  # 신뢰도
            
            weight = model_weight * np.mean(confidence)
            ensemble_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred

class EnsembleManager:
    """앙상블 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
    
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        logger.info(f"기본 모델 추가: {name}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """앙상블 생성"""
        
        if ensemble_type == 'weighted':
            # 실제 모델명을 기반으로 가중치 설정
            default_weights = {
                'lgbm': 0.4,
                'xgb': 0.3, 
                'cat': 0.2,
                'nn': 0.1
            }
            weights = kwargs.get('weights', default_weights)
            ensemble = WeightedBlending(weights)
        
        elif ensemble_type == 'stacking':
            meta_model = kwargs.get('meta_model', self.config.ENSEMBLE_CONFIG['meta_model'])
            cv_folds = kwargs.get('cv_folds', 5)
            ensemble = StackingEnsemble(meta_model, cv_folds)
        
        elif ensemble_type == 'rank':
            ensemble = RankAveraging()
        
        elif ensemble_type == 'dynamic':
            threshold = kwargs.get('confidence_threshold', 0.7)
            ensemble = DynamicEnsemble(threshold)
        
        else:
            raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
        
        # 기본 모델들 추가
        for name, model in self.base_models.items():
            ensemble.add_base_model(name, model)
        
        self.ensembles[ensemble_type] = ensemble
        logger.info(f"앙상블 생성: {ensemble_type}")
        
        return ensemble
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """모든 앙상블 학습"""
        logger.info("모든 앙상블 학습 시작")
        
        # 기본 모델들의 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                base_predictions[name] = pred
                logger.info(f"{name} 모델 예측 완료")
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
        
        # 각 앙상블 학습
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                ensemble.fit(X, y, base_predictions)
                logger.info(f"{ensemble_type} 앙상블 학습 완료")
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 학습 실패: {str(e)}")
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """앙상블 성능 평가"""
        logger.info("앙상블 성능 평가 시작")
        
        metrics_calc = CTRMetrics()
        results = {}
        
        # 기본 모델들의 검증 예측
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                base_predictions[name] = pred
                
                # 기본 모델 성능도 기록
                score = metrics_calc.combined_score(y_val, pred)
                results[f"base_{name}"] = score
                
            except Exception as e:
                logger.error(f"{name} 모델 검증 예측 실패: {str(e)}")
        
        # 앙상블 성능 평가
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    ensemble_pred = ensemble.predict_proba(base_predictions)
                    score = metrics_calc.combined_score(y_val, ensemble_pred)
                    results[f"ensemble_{ensemble_type}"] = score
                    
                    logger.info(f"{ensemble_type} 앙상블 점수: {score:.4f}")
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 평가 실패: {str(e)}")
        
        # 최고 성능 앙상블 선택
        if results:
            best_ensemble_name = max(results, key=results.get)
            best_score = results[best_ensemble_name]
            
            if best_ensemble_name.startswith('ensemble_'):
                ensemble_type = best_ensemble_name.replace('ensemble_', '')
                self.best_ensemble = self.ensembles[ensemble_type]
                logger.info(f"최고 성능 앙상블: {ensemble_type} (점수: {best_score:.4f})")
            else:
                # 기본 모델이 더 좋은 경우 앙상블을 사용하지 않음
                logger.info(f"기본 모델이 더 우수함: {best_ensemble_name} (점수: {best_score:.4f})")
                self.best_ensemble = None
        
        self.ensemble_results = results
        return results
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """최고 성능 앙상블로 예측"""
        if self.best_ensemble is None:
            # 앙상블이 없으면 최고 성능 기본 모델 사용
            best_model_name = None
            best_score = 0
            
            # 기본 모델 중 최고 성능 찾기
            for result_name, score in self.ensemble_results.items():
                if result_name.startswith('base_') and score > best_score:
                    best_score = score
                    best_model_name = result_name.replace('base_', '')
            
            if best_model_name and best_model_name in self.base_models:
                logger.info(f"최고 성능 기본 모델 사용: {best_model_name}")
                return self.base_models[best_model_name].predict_proba(X)
            else:
                raise ValueError("사용 가능한 모델이 없습니다.")
        
        # 기본 모델들의 예측
        base_predictions = {}
        for name, model in self.base_models.items():
            pred = model.predict_proba(X)
            base_predictions[name] = pred
        
        # 앙상블 예측
        return self.best_ensemble.predict_proba(base_predictions)
    
    def save_ensembles(self, output_dir: Path = None):
        """앙상블 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 앙상블 저장
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                ensemble_path = output_dir / f"ensemble_{ensemble_type}.pkl"
                
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(ensemble, f)
                
                logger.info(f"{ensemble_type} 앙상블 저장: {ensemble_path}")
        
        # 최고 성능 앙상블 정보 저장
        best_info = {
            'best_ensemble_type': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results
        }
        
        import json
        info_path = output_dir / "best_ensemble_info.json"
        with open(info_path, 'w') as f:
            json.dump(best_info, f, indent=2)
    
    def load_ensembles(self, input_dir: Path = None):
        """앙상블 로딩"""
        if input_dir is None:
            input_dir = self.config.MODEL_DIR
        
        input_dir = Path(input_dir)
        
        # 앙상블 파일 찾기
        ensemble_files = list(input_dir.glob("ensemble_*.pkl"))
        
        for ensemble_file in ensemble_files:
            try:
                ensemble_type = ensemble_file.stem.replace('ensemble_', '')
                
                with open(ensemble_file, 'rb') as f:
                    ensemble = pickle.load(f)
                
                self.ensembles[ensemble_type] = ensemble
                logger.info(f"{ensemble_type} 앙상블 로딩 완료")
                
            except Exception as e:
                logger.error(f"{ensemble_file} 앙상블 로딩 실패: {str(e)}")
        
        # 최고 성능 앙상블 정보 로딩
        info_path = input_dir / "best_ensemble_info.json"
        if info_path.exists():
            try:
                import json
                with open(info_path, 'r') as f:
                    best_info = json.load(f)
                
                best_type = best_info['best_ensemble_type']
                if best_type and best_type in self.ensembles:
                    self.best_ensemble = self.ensembles[best_type]
                    self.ensemble_results = best_info.get('ensemble_results', {})
                
            except Exception as e:
                logger.error(f"최고 앙상블 정보 로딩 실패: {str(e)}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """앙상블 요약 정보"""
        return {
            'total_ensembles': len(self.ensembles),
            'fitted_ensembles': sum(1 for e in self.ensembles.values() if e.is_fitted),
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results,
            'base_models_count': len(self.base_models)
        }