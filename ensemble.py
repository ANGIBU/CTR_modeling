# ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import gc

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna가 설치되지 않았습니다. 하이퍼파라미터 튜닝 기능이 비활성화됩니다.")

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
                predictions[name] = np.full(len(X), 0.0201)
        
        return predictions

class CTROptimalEnsemble(BaseEnsemble):
    """CTR 예측 최적화 앙상블"""
    
    def __init__(self, target_ctr: float = 0.0201, optimization_method: str = 'combined'):
        super().__init__("CTROptimalEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.final_weights = {}
        self.ctr_calibrator = None
        self.metrics_calculator = CTRMetrics()
        self.temperature = 1.0
        self.bias_correction = 0.0
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """CTR 최적화 앙상블 학습"""
        logger.info(f"CTR 최적화 앙상블 학습 시작 - 방법: {self.optimization_method}")
        
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models}")
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # 1단계: 다단계 가중치 최적화
        self.final_weights = self._multi_stage_optimization(base_predictions, y)
        
        # 2단계: CTR 특화 후처리 최적화
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        self._optimize_ctr_postprocessing(ensemble_pred, y)
        
        self.is_fitted = True
        logger.info("CTR 최적화 앙상블 학습 완료")
    
    def _multi_stage_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """다단계 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        
        # 1단계: 개별 모델 성능 평가
        individual_scores = {}
        for name, pred in base_predictions.items():
            score = self.metrics_calculator.combined_score(y, pred)
            individual_scores[name] = score
        
        logger.info(f"개별 모델 성능: {individual_scores}")
        
        # 2단계: 초기 가중치 설정 (성능 기반)
        total_score = sum(individual_scores.values())
        if total_score > 0:
            initial_weights = {name: score/total_score for name, score in individual_scores.items()}
        else:
            initial_weights = {name: 1.0/len(model_names) for name in model_names}
        
        # 3단계: 그리드 서치 기반 세밀 조정
        optimized_weights = self._grid_search_optimization(base_predictions, y, initial_weights)
        
        # 4단계: Optuna 기반 정밀 최적화
        if OPTUNA_AVAILABLE and len(model_names) <= 4:
            try:
                final_weights = self._optuna_optimization(base_predictions, y, optimized_weights)
            except Exception as e:
                logger.warning(f"Optuna 최적화 실패: {e}")
                final_weights = optimized_weights
        else:
            final_weights = optimized_weights
        
        logger.info(f"최종 가중치: {final_weights}")
        return final_weights
    
    def _grid_search_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series, 
                                initial_weights: Dict[str, float]) -> Dict[str, float]:
        """그리드 서치 기반 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        best_weights = initial_weights.copy()
        best_score = self._evaluate_ensemble(base_predictions, y, best_weights)
        
        # 가중치 조정 단계
        adjustment_steps = [0.1, 0.05, 0.02, 0.01]
        
        for step in adjustment_steps:
            improved = True
            iteration = 0
            
            while improved and iteration < 20:
                improved = False
                iteration += 1
                
                for target_model in model_names:
                    for direction in [-1, 1]:
                        # 가중치 조정 시도
                        test_weights = best_weights.copy()
                        test_weights[target_model] += direction * step
                        
                        # 음수 방지
                        if test_weights[target_model] < 0:
                            continue
                        
                        # 정규화
                        total_weight = sum(test_weights.values())
                        if total_weight > 0:
                            test_weights = {k: v/total_weight for k, v in test_weights.items()}
                        else:
                            continue
                        
                        # 평가
                        score = self._evaluate_ensemble(base_predictions, y, test_weights)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = test_weights
                            improved = True
        
        logger.info(f"그리드 서치 완료 - 점수: {best_score:.4f}")
        return best_weights
    
    def _optuna_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series,
                           initial_weights: Dict[str, float]) -> Dict[str, float]:
        """Optuna 기반 정밀 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        
        def objective(trial):
            weights = {}
            
            # 가중치 범위를 초기값 기준으로 제한
            for name in model_names:
                initial_val = initial_weights.get(name, 1.0/len(model_names))
                low_bound = max(0.01, initial_val - 0.3)
                high_bound = min(0.99, initial_val + 0.3)
                weights[name] = trial.suggest_float(f'weight_{name}', low_bound, high_bound)
            
            # 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                return 0.0
            
            return self._evaluate_ensemble(base_predictions, y, weights)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        
        optimized_weights = {}
        for param_name, weight in study.best_params.items():
            model_name = param_name.replace('weight_', '')
            optimized_weights[model_name] = weight
        
        # 정규화
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
        
        logger.info(f"Optuna 최적화 완료 - 점수: {study.best_value:.4f}")
        return optimized_weights
    
    def _evaluate_ensemble(self, base_predictions: Dict[str, np.ndarray], y: pd.Series, 
                         weights: Dict[str, float]) -> float:
        """앙상블 성능 평가"""
        try:
            ensemble_pred = np.zeros(len(y))
            for name, weight in weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
            
            if self.optimization_method == 'combined':
                return self.metrics_calculator.combined_score(y, ensemble_pred)
            elif self.optimization_method == 'ap':
                return self.metrics_calculator.average_precision(y, ensemble_pred)
            elif self.optimization_method == 'wll':
                wll = self.metrics_calculator.weighted_log_loss(y, ensemble_pred)
                return 1.0 / (1.0 + wll) if wll != float('inf') else 0.0
            else:
                return self.metrics_calculator.combined_score(y, ensemble_pred)
        except Exception as e:
            logger.warning(f"앙상블 평가 실패: {e}")
            return 0.0
    
    def _optimize_ctr_postprocessing(self, predictions: np.ndarray, y: pd.Series):
        """CTR 특화 후처리 최적화"""
        logger.info("CTR 후처리 최적화 시작")
        
        try:
            # 1. 편향 보정
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            
            # 2. Temperature scaling 최적화
            self._optimize_temperature_scaling(predictions, y)
            
            # 3. CTR 분포 매칭
            self._optimize_distribution_matching(predictions, y)
            
            logger.info(f"CTR 후처리 최적화 완료 - 편향: {self.bias_correction:.4f}, Temperature: {self.temperature:.3f}")
            
        except Exception as e:
            logger.error(f"CTR 후처리 최적화 실패: {e}")
            self.bias_correction = 0.0
            self.temperature = 1.0
    
    def _optimize_temperature_scaling(self, predictions: np.ndarray, y: pd.Series):
        """Temperature scaling 최적화"""
        try:
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(temp):
                if temp <= 0:
                    return float('inf')
                
                # Logit 변환
                pred_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                
                # Temperature 적용
                calibrated_logits = logits / temp
                calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                # Log loss 계산
                return -np.mean(y * np.log(calibrated_probs) + (1 - y) * np.log(1 - calibrated_probs))
            
            result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
            
        except Exception as e:
            logger.warning(f"Temperature scaling 최적화 실패: {e}")
            self.temperature = 1.0
    
    def _optimize_distribution_matching(self, predictions: np.ndarray, y: pd.Series):
        """분포 매칭 최적화"""
        try:
            # CTR 분포 특성 분석
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            
            # 분위수 기반 매칭
            pred_quantiles = np.percentile(predictions, [25, 50, 75, 90, 95, 99])
            
            # 고CTR 구간 보정
            high_ctr_threshold = np.percentile(predictions, 95)
            high_ctr_mask = predictions >= high_ctr_threshold
            
            if high_ctr_mask.sum() > 0:
                high_ctr_actual_rate = y[high_ctr_mask].mean()
                high_ctr_pred_rate = predictions[high_ctr_mask].mean()
                
                # 고CTR 구간 보정 계수
                if high_ctr_pred_rate > 0:
                    self.high_ctr_correction = high_ctr_actual_rate / high_ctr_pred_rate
                else:
                    self.high_ctr_correction = 1.0
            else:
                self.high_ctr_correction = 1.0
            
            logger.info(f"분포 매칭 완료 - 고CTR 보정: {self.high_ctr_correction:.3f}")
            
        except Exception as e:
            logger.warning(f"분포 매칭 최적화 실패: {e}")
            self.high_ctr_correction = 1.0
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred
    
    def _apply_postprocessing(self, predictions: np.ndarray) -> np.ndarray:
        """후처리 적용"""
        try:
            # 1. Temperature scaling
            if self.temperature != 1.0:
                pred_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                calibrated_logits = logits / self.temperature
                predictions = 1 / (1 + np.exp(-calibrated_logits))
            
            # 2. 편향 보정
            predictions = predictions + self.bias_correction
            
            # 3. 고CTR 구간 보정
            if hasattr(self, 'high_ctr_correction'):
                high_ctr_threshold = np.percentile(predictions, 95)
                high_ctr_mask = predictions >= high_ctr_threshold
                predictions[high_ctr_mask] *= self.high_ctr_correction
            
            # 4. 범위 클리핑
            predictions = np.clip(predictions, 0.001, 0.999)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"후처리 적용 실패: {e}")
            return np.clip(predictions, 0.001, 0.999)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """최적화된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        # 가중 앙상블
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 후처리 적용
        calibrated_pred = self._apply_postprocessing(ensemble_pred)
        
        return calibrated_pred

class CTRStabilizedEnsemble(BaseEnsemble):
    """CTR 예측 안정화 앙상블"""
    
    def __init__(self, diversification_method: str = 'rank_weighted'):
        super().__init__("CTRStabilizedEnsemble")
        self.diversification_method = diversification_method
        self.model_weights = {}
        self.diversity_weights = {}
        self.final_weights = {}
        self.metrics_calculator = CTRMetrics()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """안정화 앙상블 학습"""
        logger.info(f"CTR 안정화 앙상블 학습 시작 - 방법: {self.diversification_method}")
        
        available_models = list(base_predictions.keys())
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # 1. 개별 모델 성능 평가
        self.model_weights = self._evaluate_individual_performance(base_predictions, y)
        
        # 2. 다양성 가중치 계산
        self.diversity_weights = self._calculate_diversity_weights(base_predictions)
        
        # 3. 최종 가중치 결합
        self.final_weights = self._combine_weights()
        
        self.is_fitted = True
        logger.info(f"CTR 안정화 앙상블 학습 완료 - 최종 가중치: {self.final_weights}")
    
    def _evaluate_individual_performance(self, base_predictions: Dict[str, np.ndarray], 
                                       y: pd.Series) -> Dict[str, float]:
        """개별 모델 성능 평가"""
        
        performance_weights = {}
        
        for name, pred in base_predictions.items():
            try:
                # Combined Score 기반 성능
                combined_score = self.metrics_calculator.combined_score(y, pred)
                
                # CTR 편향 패널티
                predicted_ctr = pred.mean()
                actual_ctr = y.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_penalty = np.exp(-ctr_bias * 50)  # CTR 편향에 따른 패널티
                
                # 최종 성능 가중치
                performance_score = combined_score * ctr_penalty
                performance_weights[name] = max(performance_score, 0.01)
                
                logger.info(f"{name} - Combined: {combined_score:.4f}, CTR편향: {ctr_bias:.4f}, 최종: {performance_score:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 성능 평가 실패: {e}")
                performance_weights[name] = 0.01
        
        return performance_weights
    
    def _calculate_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """다양성 가중치 계산"""
        
        model_names = list(base_predictions.keys())
        diversity_weights = {}
        
        if self.diversification_method == 'correlation_based':
            # 상관관계 기반 다양성
            correlation_matrix = self._calculate_correlation_matrix(base_predictions)
            
            for name in model_names:
                avg_correlation = np.mean([abs(correlation_matrix[name][other]) 
                                         for other in model_names if other != name])
                diversity_score = 1.0 - avg_correlation  # 낮은 상관관계 = 높은 다양성
                diversity_weights[name] = max(diversity_score, 0.1)
        
        elif self.diversification_method == 'rank_weighted':
            # 순위 기반 다양성
            rank_matrix = {}
            for name, pred in base_predictions.items():
                rank_matrix[name] = pd.Series(pred).rank(pct=True).values
            
            diversity_scores = {}
            for name in model_names:
                rank_differences = []
                for other in model_names:
                    if other != name:
                        rank_diff = np.mean(np.abs(rank_matrix[name] - rank_matrix[other]))
                        rank_differences.append(rank_diff)
                
                diversity_score = np.mean(rank_differences) if rank_differences else 0.5
                diversity_weights[name] = max(diversity_score, 0.1)
        
        elif self.diversification_method == 'prediction_spread':
            # 예측 분산 기반 다양성
            for name, pred in base_predictions.items():
                pred_std = np.std(pred)
                pred_entropy = self._calculate_prediction_entropy(pred)
                diversity_score = pred_std * pred_entropy
                diversity_weights[name] = max(diversity_score, 0.1)
        
        else:
            # 균등 다양성
            diversity_weights = {name: 1.0 for name in model_names}
        
        logger.info(f"다양성 가중치: {diversity_weights}")
        return diversity_weights
    
    def _calculate_correlation_matrix(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """예측간 상관관계 행렬 계산"""
        
        model_names = list(base_predictions.keys())
        correlation_matrix = {}
        
        for name1 in model_names:
            correlation_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    correlation_matrix[name1][name2] = 1.0
                else:
                    try:
                        corr = np.corrcoef(base_predictions[name1], base_predictions[name2])[0, 1]
                        correlation_matrix[name1][name2] = corr if not np.isnan(corr) else 0.0
                    except:
                        correlation_matrix[name1][name2] = 0.0
        
        return correlation_matrix
    
    def _calculate_prediction_entropy(self, predictions: np.ndarray) -> float:
        """예측 엔트로피 계산"""
        try:
            p = np.clip(predictions, 1e-15, 1 - 1e-15)
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            return entropy
        except:
            return 0.5
    
    def _combine_weights(self) -> Dict[str, float]:
        """성능과 다양성 가중치 결합"""
        
        combined_weights = {}
        model_names = list(self.model_weights.keys())
        
        # 가중치 정규화
        performance_sum = sum(self.model_weights.values())
        diversity_sum = sum(self.diversity_weights.values())
        
        if performance_sum > 0 and diversity_sum > 0:
            for name in model_names:
                perf_weight = self.model_weights[name] / performance_sum
                div_weight = self.diversity_weights[name] / diversity_sum
                
                # 성능 70%, 다양성 30% 비율로 결합
                combined_weight = 0.7 * perf_weight + 0.3 * div_weight
                combined_weights[name] = combined_weight
        else:
            # 균등 가중치
            combined_weights = {name: 1.0/len(model_names) for name in model_names}
        
        # 최종 정규화
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        return combined_weights
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """안정화된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred

class CTRMetaLearning(BaseEnsemble):
    """CTR 예측 메타 학습 앙상블"""
    
    def __init__(self, meta_model_type: str = 'ridge', use_meta_features: bool = True):
        super().__init__("CTRMetaLearning")
        self.meta_model_type = meta_model_type
        self.use_meta_features = use_meta_features
        self.meta_model = None
        self.feature_scaler = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Optional[Dict[str, np.ndarray]] = None):
        """메타 학습 앙상블 학습"""
        logger.info(f"CTR 메타 학습 앙상블 학습 시작 - 메타모델: {self.meta_model_type}")
        
        # Out-of-fold 예측 생성
        oof_predictions = self._generate_oof_predictions(X, y)
        
        # 메타 피처 생성
        if self.use_meta_features:
            meta_features = self._create_meta_features(oof_predictions, X)
        else:
            meta_features = oof_predictions
        
        # 메타 모델 학습
        self._train_meta_model(meta_features, y)
        
        self.is_fitted = True
        logger.info("CTR 메타 학습 앙상블 학습 완료")
    
    def _generate_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Out-of-fold 예측 생성"""
        
        oof_predictions = pd.DataFrame(index=X.index)
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model_name, model in self.base_models.items():
            oof_pred = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    
                    # 모델 복사 및 학습
                    fold_model = self._clone_model(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Out-of-fold 예측
                    oof_pred[val_idx] = fold_model.predict_proba(X_val_fold)
                    
                except Exception as e:
                    logger.error(f"{model_name} 폴드 {fold} 학습 실패: {str(e)}")
                    oof_pred[val_idx] = 0.0201
            
            oof_predictions[model_name] = oof_pred
        
        return oof_predictions
    
    def _clone_model(self, model: BaseModel) -> BaseModel:
        """모델 복사"""
        from models import ModelFactory
        
        try:
            model_type = model.name.lower()
            if model_type == 'lightgbm':
                model_type = 'lightgbm'
            elif model_type == 'xgboost':
                model_type = 'xgboost'
            elif model_type == 'catboost':
                model_type = 'catboost'
            elif 'deepctr' in model_type:
                # DeepCTR는 복사하기 복잡하므로 기본 파라미터로 새로 생성
                return ModelFactory.create_model('deepctr', input_dim=100, params=model.params)
            else:
                model_type = 'logistic'
            
            return ModelFactory.create_model(model_type, params=model.params)
        except:
            return model
    
    def _create_meta_features(self, oof_predictions: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """메타 피처 생성"""
        
        meta_features = oof_predictions.copy()
        
        try:
            # 기본 통계 피처
            meta_features['pred_mean'] = oof_predictions.mean(axis=1)
            meta_features['pred_std'] = oof_predictions.std(axis=1)
            meta_features['pred_min'] = oof_predictions.min(axis=1)
            meta_features['pred_max'] = oof_predictions.max(axis=1)
            meta_features['pred_median'] = oof_predictions.median(axis=1)
            
            # 순위 기반 피처
            for col in oof_predictions.columns:
                meta_features[f'{col}_rank'] = oof_predictions[col].rank(pct=True)
            
            # 모델간 차이 피처
            model_cols = oof_predictions.columns.tolist()
            for i, col1 in enumerate(model_cols):
                for col2 in model_cols[i+1:]:
                    meta_features[f'{col1}_{col2}_diff'] = oof_predictions[col1] - oof_predictions[col2]
                    meta_features[f'{col1}_{col2}_ratio'] = oof_predictions[col1] / (oof_predictions[col2] + 1e-8)
            
            # 신뢰도 피처
            meta_features['prediction_confidence'] = 1 - meta_features['pred_std']
            meta_features['consensus_strength'] = np.exp(-meta_features['pred_std'])
            
            # 원본 데이터 요약 피처 (선택적)
            if len(X.columns) <= 50:  # 피처 수가 적을 때만
                try:
                    numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]
                    if len(numeric_cols) > 0:
                        meta_features['x_mean'] = X[numeric_cols].mean(axis=1)
                        meta_features['x_std'] = X[numeric_cols].std(axis=1)
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"메타 피처 생성 중 오류: {e}")
        
        return meta_features
    
    def _train_meta_model(self, meta_features: pd.DataFrame, y: pd.Series):
        """메타 모델 학습"""
        
        # 피처 전처리
        meta_features_clean = meta_features.fillna(0)
        meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 스케일링
        self.feature_scaler = StandardScaler()
        meta_features_scaled = self.feature_scaler.fit_transform(meta_features_clean)
        
        # 메타 모델 선택 및 학습
        if self.meta_model_type == 'ridge':
            self.meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3)
        elif self.meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.meta_model_type == 'mlp':
            self.meta_model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        else:
            self.meta_model = RidgeCV(cv=3)
        
        try:
            if self.meta_model_type == 'logistic':
                self.meta_model.fit(meta_features_scaled, y)
            else:
                self.meta_model.fit(meta_features_scaled, y)
            
            # 성능 평가
            meta_pred = self.meta_model.predict(meta_features_scaled)
            if self.meta_model_type == 'logistic':
                meta_pred = self.meta_model.predict_proba(meta_features_scaled)[:, 1]
            
            metrics_calc = CTRMetrics()
            combined_score = metrics_calc.combined_score(y, meta_pred)
            logger.info(f"메타 모델 Combined Score: {combined_score:.4f}")
            
        except Exception as e:
            logger.error(f"메타 모델 학습 실패: {str(e)}")
            raise
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """메타 학습 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        try:
            # 기본 예측을 DataFrame으로 변환
            pred_df = pd.DataFrame(base_predictions)
            
            # 메타 피처 생성
            if self.use_meta_features:
                meta_features = self._create_inference_meta_features(pred_df)
            else:
                meta_features = pred_df
            
            # 전처리
            meta_features_clean = meta_features.fillna(0)
            meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # 스케일링
            meta_features_scaled = self.feature_scaler.transform(meta_features_clean)
            
            # 예측
            if self.meta_model_type == 'logistic':
                ensemble_pred = self.meta_model.predict_proba(meta_features_scaled)[:, 1]
            else:
                ensemble_pred = self.meta_model.predict(meta_features_scaled)
                ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"메타 학습 예측 실패: {str(e)}")
            return np.mean(list(base_predictions.values()), axis=0)
    
    def _create_inference_meta_features(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """추론용 메타 피처 생성"""
        
        meta_features = pred_df.copy()
        
        try:
            # 기본 통계 피처
            meta_features['pred_mean'] = pred_df.mean(axis=1)
            meta_features['pred_std'] = pred_df.std(axis=1)
            meta_features['pred_min'] = pred_df.min(axis=1)
            meta_features['pred_max'] = pred_df.max(axis=1)
            meta_features['pred_median'] = pred_df.median(axis=1)
            
            # 순위 기반 피처
            for col in pred_df.columns:
                meta_features[f'{col}_rank'] = pred_df[col].rank(pct=True)
            
            # 모델간 차이 피처
            model_cols = pred_df.columns.tolist()
            for i, col1 in enumerate(model_cols):
                for col2 in model_cols[i+1:]:
                    meta_features[f'{col1}_{col2}_diff'] = pred_df[col1] - pred_df[col2]
                    meta_features[f'{col1}_{col2}_ratio'] = pred_df[col1] / (pred_df[col2] + 1e-8)
            
            # 신뢰도 피처
            meta_features['prediction_confidence'] = 1 - meta_features['pred_std']
            meta_features['consensus_strength'] = np.exp(-meta_features['pred_std'])
            
        except Exception as e:
            logger.warning(f"추론용 메타 피처 생성 중 오류: {e}")
        
        return meta_features

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
        self.optimal_ensemble = None
        
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        logger.info(f"기본 모델 추가: {name}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """CTR 특화 앙상블 생성"""
        
        if ensemble_type == 'optimal':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            optimization_method = kwargs.get('optimization_method', 'combined')
            ensemble = CTROptimalEnsemble(target_ctr, optimization_method)
            self.optimal_ensemble = ensemble
        
        elif ensemble_type == 'stabilized':
            diversification_method = kwargs.get('diversification_method', 'rank_weighted')
            ensemble = CTRStabilizedEnsemble(diversification_method)
        
        elif ensemble_type == 'meta':
            meta_model_type = kwargs.get('meta_model_type', 'ridge')
            use_meta_features = kwargs.get('use_meta_features', True)
            ensemble = CTRMetaLearning(meta_model_type, use_meta_features)
        
        elif ensemble_type == 'weighted':
            weights = kwargs.get('weights', None)
            from ensemble import CTRWeightedBlending
            ensemble = CTRWeightedBlending(weights)
        
        elif ensemble_type == 'calibrated':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            calibration_method = kwargs.get('calibration_method', 'platt')
            from ensemble import CTRCalibratedEnsemble
            ensemble = CTRCalibratedEnsemble(target_ctr, calibration_method)
            self.calibrated_ensemble = ensemble
        
        else:
            raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
        
        # 기본 모델 추가
        for name, model in self.base_models.items():
            ensemble.add_base_model(name, model)
        
        self.ensembles[ensemble_type] = ensemble
        logger.info(f"앙상블 생성: {ensemble_type}")
        
        return ensemble
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """모든 앙상블 학습"""
        logger.info("모든 앙상블 학습 시작")
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                base_predictions[name] = pred
                logger.info(f"{name} 모델 예측 완료")
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
        
        # 각 앙상블 학습
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                ensemble.fit(X, y, base_predictions)
                logger.info(f"{ensemble_type} 앙상블 학습 완료")
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 학습 실패: {str(e)}")
        
        # 메모리 정리
        gc.collect()
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """앙상블 성능 평가"""
        logger.info("앙상블 성능 평가 시작")
        
        results = {}
        
        # 기본 모델 예측 수집
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
        
        # 앙상블 성능 평가
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    ensemble_pred = ensemble.predict_proba(base_predictions)
                    score = self.metrics_calculator.combined_score(y_val, ensemble_pred)
                    results[f"ensemble_{ensemble_type}"] = score
                    
                    logger.info(f"{ensemble_type} 앙상블 Combined Score: {score:.4f}")
                    
                    # CTR 분석
                    predicted_ctr = ensemble_pred.mean()
                    actual_ctr = y_val.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    logger.info(f"{ensemble_type} CTR: 예측 {predicted_ctr:.4f} vs 실제 {actual_ctr:.4f} (편향: {ctr_bias:.4f})")
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 평가 실패: {str(e)}")
                    results[f"ensemble_{ensemble_type}"] = 0.0
        
        # 최고 성능 앙상블 선택
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
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                base_predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
        
        if self.best_ensemble is None:
            # 기본 모델 중 최고 성능 선택
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
                # 평균 앙상블
                return np.mean(list(base_predictions.values()), axis=0)
        
        return self.best_ensemble.predict_proba(base_predictions)
    
    def save_ensembles(self, output_dir: Path = None):
        """앙상블 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 개별 앙상블 저장
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                ensemble_path = output_dir / f"ensemble_{ensemble_type}.pkl"
                
                try:
                    with open(ensemble_path, 'wb') as f:
                        pickle.dump(ensemble, f)
                    
                    logger.info(f"{ensemble_type} 앙상블 저장: {ensemble_path}")
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 저장 실패: {str(e)}")
        
        # 최고 앙상블 정보 저장
        best_info = {
            'best_ensemble_type': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results,
            'calibrated_available': self.calibrated_ensemble is not None,
            'optimal_available': self.optimal_ensemble is not None
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
                
                # 특수 앙상블 참조 설정
                if ensemble_type == 'calibrated':
                    self.calibrated_ensemble = ensemble
                elif ensemble_type == 'optimal':
                    self.optimal_ensemble = ensemble
                
                logger.info(f"{ensemble_type} 앙상블 로딩 완료")
                
            except Exception as e:
                logger.error(f"{ensemble_file} 앙상블 로딩 실패: {str(e)}")
        
        # 최고 앙상블 정보 로딩
        info_path = input_dir / "best_ensemble_info.json"
        if info_path.exists():
            try:
                import json
                with open(info_path, 'r') as f:
                    best_info = json.load(f)
                
                best_type = best_info.get('best_ensemble_type')
                if best_type:
                    for ensemble in self.ensembles.values():
                        if ensemble.name == best_type:
                            self.best_ensemble = ensemble
                            break
                
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
            'calibrated_ensemble_available': self.calibrated_ensemble is not None and self.calibrated_ensemble.is_fitted,
            'optimal_ensemble_available': self.optimal_ensemble is not None and self.optimal_ensemble.is_fitted,
            'ensemble_types': list(self.ensembles.keys())
        }

# 기존 앙상블 클래스들 (하위 호환성)
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
        
        if len(available_models) < 2:
            if available_models:
                self.weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # Combined Score 기준 가중치 최적화
        self.weights = self._optimize_weights_for_combined_score(base_predictions, y)
        
        # 가중 앙상블 생성
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # CTR 보정 적용
        self._apply_ctr_calibration(ensemble_pred, y)
        
        self.is_fitted = True
        logger.info("CTR 보정 앙상블 학습 완료")
    
    def _optimize_weights_for_combined_score(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Combined Score 최적화를 위한 가중치 튜닝"""
        
        model_names = list(base_predictions.keys())
        
        if not OPTUNA_AVAILABLE:
            # 성능 기반 균등 가중치
            weights = {}
            for name in model_names:
                score = self.metrics_calculator.combined_score(y, base_predictions[name])
                weights[name] = max(score, 0.01)
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            return weights
        
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
            study.optimize(objective, n_trials=100, show_progress_bar=False)
            
            optimized_weights = {}
            for param_name, weight in study.best_params.items():
                model_name = param_name.replace('weight_', '')
                optimized_weights[model_name] = weight
            
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
            
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
        try:
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            
            if self.calibration_method == 'platt':
                self.calibrator = LogisticRegression()
                self.calibrator.fit(predictions.reshape(-1, 1), y)
            elif self.calibration_method == 'isotonic':
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(predictions, y)
                
        except Exception as e:
            logger.error(f"CTR 보정 실패: {str(e)}")
            self.calibrator = None
            self.bias_correction = 0.0
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """보정된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        try:
            if self.calibrator is not None:
                if self.calibration_method == 'platt':
                    calibrated = self.calibrator.predict_proba(ensemble_pred.reshape(-1, 1))[:, 1]
                elif self.calibration_method == 'isotonic':
                    calibrated = self.calibrator.predict(ensemble_pred)
                else:
                    calibrated = ensemble_pred
            else:
                calibrated = ensemble_pred
            
            calibrated = calibrated + self.bias_correction
            calibrated = np.clip(calibrated, 0.001, 0.999)
            
            return calibrated
            
        except Exception as e:
            logger.warning(f"보정 적용 실패: {str(e)}")
            return np.clip(ensemble_pred + self.bias_correction, 0.001, 0.999)

class CTRWeightedBlending(BaseEnsemble):
    """CTR 예측에 특화된 가중 블렌딩 앙상블"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("CTRWeightedBlending")
        self.weights = weights or {}
        self.optimized_weights = {}
        self.metrics_calculator = CTRMetrics()
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Combined Score 기준 가중치 최적화"""
        available_models = list(base_predictions.keys())
        
        if len(available_models) < 2:
            if available_models:
                self.optimized_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        self.optimized_weights = self._performance_based_weights(base_predictions, y)
        
        total_weight = sum(self.optimized_weights.values())
        if total_weight > 0:
            self.optimized_weights = {k: v/total_weight for k, v in self.optimized_weights.items()}
        
        self.is_fitted = True
    
    def _performance_based_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """개별 모델 성능 기반 가중치 계산"""
        weights = {}
        
        for name, pred in base_predictions.items():
            try:
                score = self.metrics_calculator.combined_score(y, pred)
                weights[name] = max(score, 0.01)
            except:
                weights[name] = 0.01
        
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

# 기존 클래스명 유지 (하위 호환성)
EnsembleManager = CTREnsembleManager