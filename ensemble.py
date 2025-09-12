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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
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
                predictions[name] = np.full(len(X), 0.0191)
        
        return predictions

class CTRCalibratedEnsemble(BaseEnsemble):
    """CTR 보정 앙상블 클래스"""
    
    def __init__(self, target_ctr: float = 0.0201, calibration_method: str = 'platt'):
        super().__init__("CTRCalibratedEnsemble")
        self.target_ctr = target_ctr
        self.calibration_method = calibration_method
        self.weights = {}
        self.calibrator = None
        self.bias_correction = 0.0
        self.metrics_calculator = CTRMetrics()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """보정된 앙상블 학습"""
        logger.info("CTR 보정 앙상블 학습 시작")
        
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models}")
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # 1단계: Combined Score 기준 가중치 최적화
        self.weights = self._optimize_weights_for_combined_score(base_predictions, y)
        
        # 2단계: 가중 앙상블 생성
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 3단계: CTR 보정 적용
        self._apply_ctr_calibration(ensemble_pred, y)
        
        self.is_fitted = True
        logger.info("CTR 보정 앙상블 학습 완료")
    
    def _optimize_weights_for_combined_score(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Combined Score 최적화를 위한 가중치 튜닝"""
        
        model_names = list(base_predictions.keys())
        
        def objective(trial):
            weights = {}
            for name in model_names:
                weights[name] = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
            
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            weights = {k: v/total_weight for k, v in weights.items()}
            
            ensemble_pred = np.zeros(len(y))
            for name, weight in weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
            
            score = self.metrics_calculator.combined_score(y, ensemble_pred)
            return score
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=200, show_progress_bar=False)
            
            optimized_weights = {}
            for param_name, weight in study.best_params.items():
                model_name = param_name.replace('weight_', '')
                optimized_weights[model_name] = weight
            
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
            
            logger.info(f"최적화 완료 - 최고 Combined Score: {study.best_value:.4f}")
            logger.info(f"최적 가중치: {optimized_weights}")
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"가중치 최적화 실패: {str(e)}")
            return {name: 1.0/len(model_names) for name in model_names}
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred
    
    def _apply_ctr_calibration(self, predictions: np.ndarray, y: pd.Series):
        """CTR 보정 적용"""
        logger.info(f"CTR 보정 적용 시작 - 방법: {self.calibration_method}")
        
        try:
            # 편향 계산
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            
            logger.info(f"CTR 편향: 예측 {predicted_ctr:.4f} vs 실제 {actual_ctr:.4f} = 보정 {self.bias_correction:.4f}")
            
            # Calibration 방법에 따른 보정
            if self.calibration_method == 'platt':
                self.calibrator = LogisticRegression()
                self.calibrator.fit(predictions.reshape(-1, 1), y)
                
            elif self.calibration_method == 'isotonic':
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(predictions, y)
                
            elif self.calibration_method == 'bias_only':
                self.calibrator = None
                
            else:
                raise ValueError(f"지원하지 않는 보정 방법: {self.calibration_method}")
            
            # 보정 후 검증
            calibrated_pred = self._calibrate_predictions(predictions)
            calibrated_ctr = calibrated_pred.mean()
            
            logger.info(f"보정 결과: {predicted_ctr:.4f} → {calibrated_ctr:.4f} (목표: {actual_ctr:.4f})")
            
        except Exception as e:
            logger.error(f"CTR 보정 실패: {str(e)}")
            self.calibrator = None
            self.bias_correction = 0.0
    
    def _calibrate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """예측값 보정 적용"""
        try:
            if self.calibrator is not None:
                if self.calibration_method == 'platt':
                    calibrated = self.calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
                elif self.calibration_method == 'isotonic':
                    calibrated = self.calibrator.predict(predictions)
                else:
                    calibrated = predictions
            else:
                calibrated = predictions
            
            # 편향 보정 적용
            calibrated = calibrated + self.bias_correction
            
            # CTR 범위 클리핑
            calibrated = np.clip(calibrated, 0.001, 0.999)
            
            return calibrated
            
        except Exception as e:
            logger.warning(f"보정 적용 실패: {str(e)}")
            return np.clip(predictions + self.bias_correction, 0.001, 0.999)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """보정된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        # 가중 앙상블
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 보정 적용
        calibrated_pred = self._calibrate_predictions(ensemble_pred)
        
        return calibrated_pred

class CTRWeightedBlending(BaseEnsemble):
    """CTR 예측에 특화된 가중 블렌딩 앙상블"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("CTRWeightedBlending")
        self.weights = weights or {}
        self.optimized_weights = {}
        self.metrics_calculator = CTRMetrics()
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Combined Score 기준 가중치 최적화"""
        logger.info("CTR 가중 블렌딩 앙상블 학습 시작")
        
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models}")
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.optimized_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # Combined Score 기준 가중치 최적화
        self.optimized_weights = self._optimize_weights_for_combined_score(base_predictions, y)
        
        # 가중치 정규화
        total_weight = sum(self.optimized_weights.values())
        if total_weight > 0:
            self.optimized_weights = {k: v/total_weight for k, v in self.optimized_weights.items()}
        else:
            self.optimized_weights = {k: 1.0/len(available_models) for k in available_models}
        
        self.is_fitted = True
        logger.info(f"Combined Score 기준 최적화된 가중치: {self.optimized_weights}")
    
    def _optimize_weights_for_combined_score(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Combined Score 최적화를 위한 가중치 튜닝"""
        
        model_names = list(base_predictions.keys())
        
        def objective(trial):
            weights = {}
            for name in model_names:
                weights[name] = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
            
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            weights = {k: v/total_weight for k, v in weights.items()}
            
            ensemble_pred = np.zeros(len(y))
            for name, weight in weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
            
            score = self.metrics_calculator.combined_score(y, ensemble_pred)
            return score
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=200, show_progress_bar=False)
            
            optimized_weights = {}
            for param_name, weight in study.best_params.items():
                model_name = param_name.replace('weight_', '')
                optimized_weights[model_name] = weight
            
            logger.info(f"최적화 완료 - 최고 Combined Score: {study.best_value:.4f}")
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"가중치 최적화 실패: {str(e)}")
            return self._performance_based_weights(base_predictions, y)
    
    def _performance_based_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """개별 모델 성능 기반 가중치 계산"""
        
        weights = {}
        
        for name, pred in base_predictions.items():
            try:
                score = self.metrics_calculator.combined_score(y, pred)
                weights[name] = max(score, 0.01)
            except:
                weights[name] = 0.01
        
        logger.info(f"성능 기반 가중치: {weights}")
        return weights
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 블렌딩 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        total_weight = 0
        
        for name, weight in self.optimized_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
        else:
            ensemble_pred = np.mean(list(base_predictions.values()), axis=0)
        
        return ensemble_pred

class CTRStackingEnsemble(BaseEnsemble):
    """CTR 예측에 특화된 스태킹 앙상블"""
    
    def __init__(self, meta_model_type: str = 'logistic', cv_folds: int = 3):
        super().__init__("CTRStackingEnsemble")
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.meta_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Optional[Dict[str, np.ndarray]] = None):
        """시간적 순서를 고려한 스태킹 앙상블 학습"""
        logger.info("CTR 스태킹 앙상블 학습 시작")
        
        oof_predictions = self._generate_temporal_oof_predictions(X, y)
        self._train_meta_model(oof_predictions, y)
        
        self.is_fitted = True
        logger.info("CTR 스태킹 앙상블 학습 완료")
    
    def _generate_temporal_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """시간적 순서를 고려한 Out-of-fold 예측 생성"""
        
        oof_predictions = pd.DataFrame(index=X.index)
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        for model_name, model in self.base_models.items():
            oof_pred = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    
                    fold_model = self._clone_model(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    oof_pred[val_idx] = fold_model.predict_proba(X_val_fold)
                    
                except Exception as e:
                    logger.error(f"{model_name} 폴드 {fold} 학습 실패: {str(e)}")
                    oof_pred[val_idx] = 0.0191
            
            oof_predictions[model_name] = oof_pred
        
        return oof_predictions
    
    def _clone_model(self, model: BaseModel) -> BaseModel:
        """모델 복사"""
        from models import ModelFactory
        
        try:
            model_type = model.name.lower()
            return ModelFactory.create_model(model_type, params=model.params)
        except:
            return model
    
    def _train_meta_model(self, oof_predictions: pd.DataFrame, y: pd.Series):
        """CTR 특화 메타 모델 학습"""
        
        if self.meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.meta_model_type == 'rf':
            self.meta_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"지원하지 않는 메타 모델: {self.meta_model_type}")
        
        try:
            self.meta_model.fit(oof_predictions, y)
            
            meta_pred = self.meta_model.predict_proba(oof_predictions)[:, 1]
            metrics_calc = CTRMetrics()
            combined_score = metrics_calc.combined_score(y, meta_pred)
            logger.info(f"메타 모델 Combined Score: {combined_score:.4f}")
            
        except Exception as e:
            logger.error(f"메타 모델 학습 실패: {str(e)}")
            raise
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """스태킹 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        try:
            pred_df = pd.DataFrame(base_predictions)
            ensemble_pred = self.meta_model.predict_proba(pred_df)[:, 1]
            
            return ensemble_pred
        except Exception as e:
            logger.error(f"스태킹 예측 실패: {str(e)}")
            return np.mean(list(base_predictions.values()), axis=0)

class CTRRankAveraging(BaseEnsemble):
    """CTR 예측에 특화된 순위 평균 앙상블"""
    
    def __init__(self):
        super().__init__("CTRRankAveraging")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """순위 평균은 별도 학습이 필요 없음"""
        self.is_fitted = True
        logger.info("CTR 순위 평균 앙상블 준비 완료")
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """순위 평균 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        try:
            ranks = []
            for pred in base_predictions.values():
                rank = pd.Series(pred).rank(pct=True).values
                ranks.append(rank)
            
            ensemble_rank = np.mean(ranks, axis=0)
            
            return ensemble_rank
        except Exception as e:
            logger.error(f"순위 평균 예측 실패: {str(e)}")
            return np.mean(list(base_predictions.values()), axis=0)

class CTREnsembleManager:
    """CTR 특화 앙상블 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.calibrated_ensemble = None
    
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        logger.info(f"기본 모델 추가: {name}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """CTR 특화 앙상블 생성"""
        
        if ensemble_type == 'weighted':
            weights = kwargs.get('weights', None)
            ensemble = CTRWeightedBlending(weights)
        
        elif ensemble_type == 'stacking':
            meta_model = kwargs.get('meta_model', 'logistic')
            cv_folds = kwargs.get('cv_folds', 3)
            ensemble = CTRStackingEnsemble(meta_model, cv_folds)
        
        elif ensemble_type == 'rank':
            ensemble = CTRRankAveraging()
        
        elif ensemble_type == 'calibrated':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            calibration_method = kwargs.get('calibration_method', 'platt')
            ensemble = CTRCalibratedEnsemble(target_ctr, calibration_method)
            self.calibrated_ensemble = ensemble
        
        else:
            raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
        
        for name, model in self.base_models.items():
            ensemble.add_base_model(name, model)
        
        self.ensembles[ensemble_type] = ensemble
        logger.info(f"앙상블 생성: {ensemble_type}")
        
        return ensemble
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """모든 앙상블 학습"""
        logger.info("모든 앙상블 학습 시작")
        
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                base_predictions[name] = pred
                logger.info(f"{name} 모델 예측 완료")
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0191)
        
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                ensemble.fit(X, y, base_predictions)
                logger.info(f"{ensemble_type} 앙상블 학습 완료")
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 학습 실패: {str(e)}")
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """앙상블 성능 평가 (Combined Score 기준)"""
        logger.info("앙상블 성능 평가 시작")
        
        results = {}
        
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                base_predictions[name] = pred
                
                score = self.metrics_calculator.combined_score(y_val, pred)
                results[f"base_{name}"] = score
                
            except Exception as e:
                logger.error(f"{name} 모델 검증 예측 실패: {str(e)}")
                results[f"base_{name}"] = 0.0
        
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    ensemble_pred = ensemble.predict_proba(base_predictions)
                    score = self.metrics_calculator.combined_score(y_val, ensemble_pred)
                    results[f"ensemble_{ensemble_type}"] = score
                    
                    logger.info(f"{ensemble_type} 앙상블 Combined Score: {score:.4f}")
                    
                    if ensemble_type == 'calibrated':
                        predicted_ctr = ensemble_pred.mean()
                        actual_ctr = y_val.mean()
                        logger.info(f"보정 앙상블 CTR: 예측 {predicted_ctr:.4f} vs 실제 {actual_ctr:.4f}")
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 평가 실패: {str(e)}")
                    results[f"ensemble_{ensemble_type}"] = 0.0
        
        if results:
            best_name = max(results, key=results.get)
            best_score = results[best_name]
            
            if best_name.startswith('ensemble_'):
                ensemble_type = best_name.replace('ensemble_', '')
                self.best_ensemble = self.ensembles[ensemble_type]
                logger.info(f"최고 성능 앙상블: {ensemble_type} (Combined Score: {best_score:.4f})")
            else:
                logger.info(f"기본 모델이 우수함: {best_name} (Combined Score: {best_score:.4f})")
                self.best_ensemble = None
        
        self.ensemble_results = results
        return results
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """최고 성능 앙상블로 예측"""
        if self.best_ensemble is None:
            best_model_name = None
            best_score = 0
            
            for result_name, score in self.ensemble_results.items():
                if result_name.startswith('base_') and score > best_score:
                    best_score = score
                    best_model_name = result_name.replace('base_', '')
            
            if best_model_name and best_model_name in self.base_models:
                logger.info(f"최고 성능 기본 모델 사용: {best_model_name}")
                return self.base_models[best_model_name].predict_proba(X)
            else:
                raise ValueError("사용 가능한 모델이 없습니다")
        
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                base_predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0191)
        
        return self.best_ensemble.predict_proba(base_predictions)
    
    def save_ensembles(self, output_dir: Path = None):
        """앙상블 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                ensemble_path = output_dir / f"ensemble_{ensemble_type}.pkl"
                
                try:
                    with open(ensemble_path, 'wb') as f:
                        pickle.dump(ensemble, f)
                    
                    logger.info(f"{ensemble_type} 앙상블 저장: {ensemble_path}")
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 저장 실패: {str(e)}")
        
        best_info = {
            'best_ensemble_type': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results
        }
        
        try:
            import json
            info_path = output_dir / "best_ensemble_info.json"
            with open(info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
        except Exception as e:
            logger.error(f"앙상블 정보 저장 실패: {str(e)}")
    
    def load_ensembles(self, input_dir: Path = None):
        """앙상블 로딩"""
        if input_dir is None:
            input_dir = self.config.MODEL_DIR
        
        input_dir = Path(input_dir)
        
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
                logger.error(f"앙상블 정보 로딩 실패: {str(e)}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """앙상블 요약 정보"""
        return {
            'total_ensembles': len(self.ensembles),
            'fitted_ensembles': sum(1 for e in self.ensembles.values() if e.is_fitted),
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results,
            'base_models_count': len(self.base_models),
            'calibrated_ensemble_available': self.calibrated_ensemble is not None
        }

EnsembleManager = CTREnsembleManager