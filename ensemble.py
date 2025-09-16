# ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import gc
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna가 설치되지 않았습니다. 하이퍼파라미터 튜닝이 제한됩니다.")

try:
    import scipy.stats as stats
    from scipy.optimize import minimize, minimize_scalar, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Scipy가 설치되지 않았습니다. 일부 최적화 기능이 제한됩니다.")

from config import Config
from models import BaseModel
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """앙상블 모델 기본 클래스 - 고성능 최적화"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.performance_history = []
        self.optimization_results = {}
        
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
        logger.debug(f"기본 모델 추가: {name}")
    
    def get_base_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """모든 기본 모델의 예측 수집"""
        predictions = {}
        
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                predictions[name] = pred
                logger.debug(f"{name} 예측 완료: shape={pred.shape}, range=[{pred.min():.4f}, {pred.max():.4f}]")
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                predictions[name] = np.full(len(X), 0.0201)
        
        return predictions

class UltraHighPerformanceEnsemble(BaseEnsemble):
    """울트라 고성능 앙상블 - Combined Score 0.32+ 목표"""
    
    def __init__(self, target_ctr: float = 0.0201, optimization_method: str = 'ultra_combined'):
        super().__init__("UltraHighPerformanceEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.final_weights = {}
        self.advanced_calibrator = None
        self.metrics_calculator = CTRMetrics()
        self.temperature = 1.0
        self.bias_correction = 0.0
        self.distribution_transformer = None
        self.ensemble_confidence = 0.0
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """울트라 고성능 앙상블 학습"""
        logger.info(f"울트라 고성능 앙상블 학습 시작 - 목표: Combined Score 0.32+")
        
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models} ({len(available_models)}개)")
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # 1단계: 고급 모델 성능 분석
        model_analysis = self._comprehensive_model_analysis(base_predictions, y)
        
        # 2단계: 다층 가중치 최적화
        self.final_weights = self._multi_layer_optimization(base_predictions, y, model_analysis)
        
        # 3단계: 울트라 고성능 후처리 최적화
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        self._ultra_postprocessing_optimization(ensemble_pred, y)
        
        # 4단계: 신뢰도 기반 동적 조정
        self._confidence_based_adjustment(base_predictions, y)
        
        self.is_fitted = True
        logger.info("울트라 고성능 앙상블 학습 완료")
    
    def _comprehensive_model_analysis(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, Dict]:
        """종합적인 모델 성능 분석"""
        
        analysis = {}
        
        for name, pred in base_predictions.items():
            try:
                # 기본 성능 지표
                combined_score = self.metrics_calculator.combined_score(y, pred)
                ap_score = self.metrics_calculator.average_precision(y, pred)
                wll_score = self.metrics_calculator.weighted_log_loss(y, pred)
                
                # CTR 분석
                predicted_ctr = pred.mean()
                actual_ctr = y.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_ratio = predicted_ctr / actual_ctr if actual_ctr > 0 else 1.0
                
                # 예측 품질 분석
                pred_std = pred.std()
                pred_entropy = -np.mean(pred * np.log(pred + 1e-15) + (1 - pred) * np.log(1 - pred + 1e-15))
                pred_diversity = len(np.unique(np.round(pred, 6))) / len(pred)
                
                # 분위수 분석
                quantiles = np.percentile(pred, [10, 25, 50, 75, 90, 95, 99])
                high_conf_precision = self._calculate_high_confidence_precision(y, pred)
                
                # 안정성 분석
                stability_score = self._calculate_stability_score(y, pred)
                
                analysis[name] = {
                    'combined_score': combined_score,
                    'ap_score': ap_score,
                    'wll_score': wll_score,
                    'ctr_bias': ctr_bias,
                    'ctr_ratio': ctr_ratio,
                    'pred_std': pred_std,
                    'pred_entropy': pred_entropy,
                    'pred_diversity': pred_diversity,
                    'quantiles': quantiles,
                    'high_conf_precision': high_conf_precision,
                    'stability_score': stability_score,
                    'quality_score': combined_score * (1 - ctr_bias * 1000) * pred_diversity * stability_score
                }
                
                logger.debug(f"{name} 분석 완료 - Quality: {analysis[name]['quality_score']:.4f}")
                
            except Exception as e:
                logger.error(f"{name} 모델 분석 실패: {e}")
                analysis[name] = {'quality_score': 0.0}
        
        return analysis
    
    def _calculate_high_confidence_precision(self, y: pd.Series, pred: np.ndarray) -> float:
        """고신뢰도 구간에서의 정밀도"""
        try:
            top_5_percent_threshold = np.percentile(pred, 95)
            top_mask = pred >= top_5_percent_threshold
            
            if top_mask.sum() > 0:
                return y[top_mask].mean()
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_stability_score(self, y: pd.Series, pred: np.ndarray) -> float:
        """예측 안정성 점수"""
        try:
            # 부트스트래핑을 통한 안정성 측정
            n_bootstrap = 10
            scores = []
            
            for _ in range(n_bootstrap):
                sample_indices = np.random.choice(len(y), size=min(10000, len(y)), replace=True)
                sample_y = y.iloc[sample_indices]
                sample_pred = pred[sample_indices]
                
                score = self.metrics_calculator.combined_score(sample_y, sample_pred)
                if score > 0:
                    scores.append(score)
            
            if len(scores) > 3:
                stability = 1.0 - (np.std(scores) / np.mean(scores))
                return max(0.0, stability)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"안정성 점수 계산 실패: {e}")
            return 0.5
    
    def _multi_layer_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series, 
                                 model_analysis: Dict[str, Dict]) -> Dict[str, float]:
        """다층 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        
        # 1층: 성능 기반 초기 가중치
        quality_weights = {}
        total_quality = sum(analysis.get('quality_score', 0) for analysis in model_analysis.values())
        
        if total_quality > 0:
            for name in model_names:
                quality_score = model_analysis.get(name, {}).get('quality_score', 0)
                quality_weights[name] = quality_score / total_quality
        else:
            quality_weights = {name: 1.0/len(model_names) for name in model_names}
        
        logger.info(f"1층 품질 기반 가중치: {quality_weights}")
        
        # 2층: 다양성 기반 조정
        diversity_weights = self._calculate_diversity_weights(base_predictions)
        
        # 3층: 고급 최적화
        if OPTUNA_AVAILABLE and len(model_names) <= 5:
            try:
                optimized_weights = self._ultra_optuna_optimization(base_predictions, y, quality_weights)
            except Exception as e:
                logger.warning(f"Optuna 최적화 실패: {e}")
                optimized_weights = quality_weights
        else:
            optimized_weights = self._advanced_grid_optimization(base_predictions, y, quality_weights)
        
        # 4층: 안정성 검증 및 미세 조정
        final_weights = self._stability_validation_adjustment(base_predictions, y, optimized_weights)
        
        logger.info(f"최종 다층 최적화 가중치: {final_weights}")
        return final_weights
    
    def _calculate_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """다양성 가중치 계산"""
        
        model_names = list(base_predictions.keys())
        diversity_weights = {}
        
        if len(model_names) < 2:
            return {name: 1.0 for name in model_names}
        
        # 상관관계 매트릭스 계산
        correlation_matrix = {}
        for name1 in model_names:
            correlation_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    correlation_matrix[name1][name2] = 1.0
                else:
                    try:
                        corr = np.corrcoef(base_predictions[name1], base_predictions[name2])[0, 1]
                        correlation_matrix[name1][name2] = abs(corr) if not np.isnan(corr) else 0.0
                    except:
                        correlation_matrix[name1][name2] = 0.0
        
        # 다양성 점수 계산 (낮은 상관관계 = 높은 다양성)
        for name in model_names:
            avg_correlation = np.mean([correlation_matrix[name][other] 
                                     for other in model_names if other != name])
            diversity_score = 1.0 - avg_correlation
            diversity_weights[name] = max(0.1, diversity_score)
        
        # 정규화
        total_diversity = sum(diversity_weights.values())
        if total_diversity > 0:
            diversity_weights = {k: v/total_diversity for k, v in diversity_weights.items()}
        
        logger.debug(f"다양성 가중치: {diversity_weights}")
        return diversity_weights
    
    def _ultra_optuna_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series,
                                  initial_weights: Dict[str, float]) -> Dict[str, float]:
        """울트라 Optuna 최적화"""
        
        model_names = list(base_predictions.keys())
        
        def ultra_objective(trial):
            try:
                weights = {}
                
                # 초기 가중치 기준 범위 설정
                for name in model_names:
                    initial_val = initial_weights.get(name, 1.0/len(model_names))
                    # 더 좁은 범위로 정밀 최적화
                    low_bound = max(0.01, initial_val - 0.2)
                    high_bound = min(0.99, initial_val + 0.2)
                    weights[name] = trial.suggest_float(f'weight_{name}', low_bound, high_bound)
                
                # 정규화
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}
                else:
                    return 0.0
                
                # 앙상블 예측 생성
                ensemble_pred = np.zeros(len(y))
                for name, weight in weights.items():
                    if name in base_predictions:
                        ensemble_pred += weight * base_predictions[name]
                
                # 울트라 고성능 목표 함수
                combined_score = self.metrics_calculator.combined_score(y, ensemble_pred)
                
                # CTR 편향 패널티 (강화)
                predicted_ctr = ensemble_pred.mean()
                actual_ctr = y.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_penalty = np.exp(-ctr_bias * 2000) if ctr_bias < 0.005 else 0.3
                
                # 예측 다양성 보너스
                pred_diversity = len(np.unique(np.round(ensemble_pred, 6))) / len(ensemble_pred)
                diversity_bonus = min(1.3, 1.0 + pred_diversity * 1.0)
                
                # 최종 울트라 점수
                ultra_score = combined_score * ctr_penalty * diversity_bonus
                
                return ultra_score if ultra_score > 0 else 0.0
                
            except Exception as e:
                logger.debug(f"Ultra objective 실행 실패: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(
                    seed=42,
                    n_startup_trials=20,
                    n_ei_candidates=64,
                    multivariate=True,
                    group=True
                ),
                pruner=HyperbandPruner(
                    min_resource=5,
                    max_resource=50,
                    reduction_factor=3
                )
            )
            
            study.optimize(
                ultra_objective,
                n_trials=150,  # 더 많은 시도
                timeout=1800,  # 30분 제한
                show_progress_bar=False,
                gc_after_trial=True
            )
            
            if study.best_value and study.best_value > 0:
                optimized_weights = {}
                for param_name, weight in study.best_params.items():
                    model_name = param_name.replace('weight_', '')
                    optimized_weights[model_name] = weight
                
                # 정규화
                total_weight = sum(optimized_weights.values())
                if total_weight > 0:
                    optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
                
                logger.info(f"울트라 Optuna 최적화 완료 - 점수: {study.best_value:.4f}")
                return optimized_weights
            
        except Exception as e:
            logger.error(f"울트라 Optuna 최적화 실패: {e}")
        
        return initial_weights
    
    def _advanced_grid_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series,
                                   initial_weights: Dict[str, float]) -> Dict[str, float]:
        """고급 그리드 최적화"""
        
        model_names = list(base_predictions.keys())
        best_weights = initial_weights.copy()
        best_score = self._evaluate_ensemble_ultra(base_predictions, y, best_weights)
        
        # 다층 그리드 서치
        adjustment_steps = [0.05, 0.02, 0.01, 0.005]
        
        for step in adjustment_steps:
            improved = True
            iteration = 0
            
            while improved and iteration < 30:
                improved = False
                iteration += 1
                
                for target_model in model_names:
                    for direction in [-1, 1]:
                        for multiplier in [0.5, 1.0, 1.5, 2.0]:
                            test_weights = best_weights.copy()
                            adjustment = direction * step * multiplier
                            test_weights[target_model] += adjustment
                            
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
                            score = self._evaluate_ensemble_ultra(base_predictions, y, test_weights)
                            
                            if score > best_score * 1.001:  # 더 엄격한 개선 기준
                                best_score = score
                                best_weights = test_weights
                                improved = True
        
        logger.info(f"고급 그리드 최적화 완료 - 점수: {best_score:.4f}")
        return best_weights
    
    def _stability_validation_adjustment(self, base_predictions: Dict[str, np.ndarray], y: pd.Series,
                                       weights: Dict[str, float]) -> Dict[str, float]:
        """안정성 검증 및 미세 조정"""
        
        # K-폴드 교차검증을 통한 안정성 검증
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in kf.split(y):
            try:
                val_y = y.iloc[val_idx]
                val_predictions = {}
                
                for name in base_predictions.keys():
                    val_predictions[name] = base_predictions[name][val_idx]
                
                fold_score = self._evaluate_ensemble_ultra(val_predictions, val_y, weights)
                fold_scores.append(fold_score)
                
            except Exception as e:
                logger.warning(f"안정성 검증 폴드 실패: {e}")
                fold_scores.append(0.0)
        
        # 안정성 지표
        if len(fold_scores) > 2:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            cv_score = std_score / mean_score if mean_score > 0 else 1.0
            
            logger.info(f"안정성 검증 - 평균: {mean_score:.4f}, 표준편차: {std_score:.4f}, CV: {cv_score:.4f}")
            
            # 불안정한 경우 보수적 조정
            if cv_score > 0.1:
                logger.info("불안정성 감지, 보수적 조정 적용")
                adjusted_weights = {}
                for name, weight in weights.items():
                    # 극단값 완화
                    adjusted_weight = weight * 0.8 + (1.0 / len(weights)) * 0.2
                    adjusted_weights[name] = adjusted_weight
                
                # 정규화
                total_weight = sum(adjusted_weights.values())
                adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
                
                return adjusted_weights
        
        return weights
    
    def _evaluate_ensemble_ultra(self, base_predictions: Dict[str, np.ndarray], y: pd.Series, 
                                weights: Dict[str, float]) -> float:
        """울트라 앙상블 평가"""
        try:
            ensemble_pred = np.zeros(len(y))
            for name, weight in weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
            
            # 기본 Combined Score
            combined_score = self.metrics_calculator.combined_score(y, ensemble_pred)
            
            # CTR 정확도 보너스
            predicted_ctr = ensemble_pred.mean()
            actual_ctr = y.mean()
            ctr_accuracy = 1.0 - min(abs(predicted_ctr - actual_ctr) * 2000, 1.0)
            
            # 예측 품질 보너스
            pred_diversity = len(np.unique(np.round(ensemble_pred, 6))) / len(ensemble_pred)
            quality_bonus = min(1.2, 1.0 + pred_diversity * 0.5)
            
            # 분포 매칭 보너스
            distribution_bonus = self._calculate_distribution_matching_bonus(y, ensemble_pred)
            
            # 울트라 점수
            ultra_score = combined_score * ctr_accuracy * quality_bonus * distribution_bonus
            
            return ultra_score if ultra_score > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"울트라 앙상블 평가 실패: {e}")
            return 0.0
    
    def _calculate_distribution_matching_bonus(self, y: pd.Series, pred: np.ndarray) -> float:
        """분포 매칭 보너스 계산"""
        try:
            # 분위수별 매칭 확인
            pred_quantiles = np.percentile(pred, [25, 50, 75, 90, 95])
            
            matching_scores = []
            for q_value in [25, 50, 75, 90, 95]:
                threshold = np.percentile(pred, q_value)
                mask = pred >= threshold
                
                if mask.sum() > 0:
                    actual_rate = y[mask].mean()
                    expected_rate = pred[mask].mean()
                    
                    if expected_rate > 0:
                        ratio = min(actual_rate, expected_rate) / max(actual_rate, expected_rate)
                        matching_scores.append(ratio)
            
            if matching_scores:
                distribution_bonus = np.mean(matching_scores)
                return min(1.3, max(0.7, distribution_bonus))
            else:
                return 1.0
                
        except Exception as e:
            logger.debug(f"분포 매칭 보너스 계산 실패: {e}")
            return 1.0
    
    def _ultra_postprocessing_optimization(self, predictions: np.ndarray, y: pd.Series):
        """울트라 후처리 최적화"""
        logger.info("울트라 후처리 최적화 시작")
        
        try:
            # 1. 고급 편향 보정
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            
            # 2. 정밀 Temperature scaling
            self._optimize_precision_temperature_scaling(predictions, y)
            
            # 3. 분포 변환 최적화
            self._optimize_distribution_transformation(predictions, y)
            
            # 4. 동적 보정 시스템
            self._setup_dynamic_calibration_system(predictions, y)
            
            logger.info(f"울트라 후처리 완료 - 편향: {self.bias_correction:.4f}, Temperature: {self.temperature:.3f}")
            
        except Exception as e:
            logger.error(f"울트라 후처리 최적화 실패: {e}")
            self.bias_correction = 0.0
            self.temperature = 1.0
    
    def _optimize_precision_temperature_scaling(self, predictions: np.ndarray, y: pd.Series):
        """정밀 Temperature scaling 최적화"""
        try:
            if not SCIPY_AVAILABLE:
                self.temperature = 1.0
                return
            
            def precision_temperature_loss(temp):
                if temp <= 0.1 or temp > 10:
                    return float('inf')
                
                pred_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                
                calibrated_logits = logits / temp
                calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                # 다중 목표 최적화
                log_loss = -np.mean(y * np.log(calibrated_probs) + (1 - y) * np.log(1 - calibrated_probs))
                
                # CTR 편향 패널티 (강화)
                ctr_bias = abs(calibrated_probs.mean() - y.mean())
                ctr_penalty = ctr_bias * 3000
                
                # 예측 품질 패널티
                pred_diversity = len(np.unique(np.round(calibrated_probs, 6))) / len(calibrated_probs)
                diversity_penalty = max(0, (0.01 - pred_diversity) * 100)
                
                return log_loss + ctr_penalty + diversity_penalty
            
            result = minimize_scalar(
                precision_temperature_loss, 
                bounds=(0.3, 5.0), 
                method='bounded'
            )
            
            self.temperature = result.x
            logger.debug(f"정밀 Temperature scaling: {self.temperature:.3f}")
            
        except Exception as e:
            logger.warning(f"Temperature scaling 최적화 실패: {e}")
            self.temperature = 1.0
    
    def _optimize_distribution_transformation(self, predictions: np.ndarray, y: pd.Series):
        """분포 변환 최적화"""
        try:
            # 분위수 기반 변환 최적화
            pred_quantiles = np.percentile(predictions, np.arange(5, 100, 5))
            target_rates = []
            
            for q in pred_quantiles:
                mask = predictions <= q
                if mask.sum() > 0:
                    actual_rate = y[mask].mean()
                    target_rates.append(actual_rate)
                else:
                    target_rates.append(self.target_ctr)
            
            self.distribution_transformer = {
                'quantiles': pred_quantiles,
                'target_rates': np.array(target_rates)
            }
            
            logger.debug("분포 변환 최적화 완료")
            
        except Exception as e:
            logger.warning(f"분포 변환 최적화 실패: {e}")
            self.distribution_transformer = None
    
    def _setup_dynamic_calibration_system(self, predictions: np.ndarray, y: pd.Series):
        """동적 보정 시스템 설정"""
        try:
            # 구간별 보정 계수 계산
            n_bins = 20
            bin_edges = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
            
            self.dynamic_calibration = {
                'bin_edges': bin_edges,
                'bin_corrections': []
            }
            
            for i in range(n_bins):
                if i == 0:
                    mask = predictions <= bin_edges[i + 1]
                elif i == n_bins - 1:
                    mask = predictions > bin_edges[i]
                else:
                    mask = (predictions > bin_edges[i]) & (predictions <= bin_edges[i + 1])
                
                if mask.sum() > 0:
                    bin_actual = y[mask].mean()
                    bin_predicted = predictions[mask].mean()
                    correction = bin_actual - bin_predicted
                    self.dynamic_calibration['bin_corrections'].append(correction)
                else:
                    self.dynamic_calibration['bin_corrections'].append(0.0)
            
            logger.debug("동적 보정 시스템 설정 완료")
            
        except Exception as e:
            logger.warning(f"동적 보정 시스템 실패: {e}")
            self.dynamic_calibration = None
    
    def _confidence_based_adjustment(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """신뢰도 기반 동적 조정"""
        try:
            # 예측 신뢰도 계산
            ensemble_pred = self._create_weighted_ensemble(base_predictions)
            
            # 모델간 일치도
            predictions_array = np.array(list(base_predictions.values()))
            agreement_std = np.std(predictions_array, axis=0)
            
            # 신뢰도 점수 (낮은 표준편차 = 높은 신뢰도)
            confidence_scores = 1.0 / (1.0 + agreement_std * 10)
            self.ensemble_confidence = confidence_scores.mean()
            
            logger.info(f"앙상블 신뢰도: {self.ensemble_confidence:.3f}")
            
        except Exception as e:
            logger.warning(f"신뢰도 기반 조정 실패: {e}")
            self.ensemble_confidence = 0.8
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred
    
    def _apply_ultra_postprocessing(self, predictions: np.ndarray) -> np.ndarray:
        """울트라 후처리 적용"""
        try:
            processed_pred = predictions.copy()
            
            # 1. Temperature scaling
            if self.temperature != 1.0:
                pred_clipped = np.clip(processed_pred, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                calibrated_logits = logits / self.temperature
                processed_pred = 1 / (1 + np.exp(-calibrated_logits))
            
            # 2. 편향 보정
            processed_pred = processed_pred + self.bias_correction
            
            # 3. 분포 변환
            if self.distribution_transformer is not None:
                try:
                    quantiles = self.distribution_transformer['quantiles']
                    target_rates = self.distribution_transformer['target_rates']
                    
                    # 분위수 기반 매핑
                    for i, (q, target) in enumerate(zip(quantiles, target_rates)):
                        mask = processed_pred <= q
                        if mask.sum() > 0:
                            current_rate = processed_pred[mask].mean()
                            if current_rate > 0:
                                adjustment_ratio = target / current_rate
                                processed_pred[mask] *= adjustment_ratio
                except Exception as e:
                    logger.debug(f"분포 변환 적용 실패: {e}")
            
            # 4. 동적 보정
            if hasattr(self, 'dynamic_calibration') and self.dynamic_calibration is not None:
                try:
                    bin_edges = self.dynamic_calibration['bin_edges']
                    corrections = self.dynamic_calibration['bin_corrections']
                    
                    for i, correction in enumerate(corrections):
                        if i == 0:
                            mask = processed_pred <= bin_edges[i + 1]
                        elif i == len(corrections) - 1:
                            mask = processed_pred > bin_edges[i]
                        else:
                            mask = (processed_pred > bin_edges[i]) & (processed_pred <= bin_edges[i + 1])
                        
                        processed_pred[mask] += correction
                except Exception as e:
                    logger.debug(f"동적 보정 적용 실패: {e}")
            
            # 5. 신뢰도 기반 조정
            confidence_factor = min(1.1, max(0.9, self.ensemble_confidence))
            processed_pred *= confidence_factor
            
            # 6. 최종 범위 클리핑
            processed_pred = np.clip(processed_pred, 0.0005, 0.9995)
            
            return processed_pred
            
        except Exception as e:
            logger.warning(f"울트라 후처리 적용 실패: {e}")
            return np.clip(predictions, 0.001, 0.999)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """울트라 고성능 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        # 가중 앙상블 생성
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 울트라 후처리 적용
        ultra_pred = self._apply_ultra_postprocessing(ensemble_pred)
        
        return ultra_pred

class AdaptiveStackingEnsemble(BaseEnsemble):
    """적응형 스태킹 앙상블 - Meta Learning"""
    
    def __init__(self, meta_model_type: str = 'ridge_advanced', use_advanced_features: bool = True):
        super().__init__("AdaptiveStackingEnsemble")
        self.meta_model_type = meta_model_type
        self.use_advanced_features = use_advanced_features
        self.meta_model = None
        self.feature_scaler = None
        self.feature_selector = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Optional[Dict[str, np.ndarray]] = None):
        """적응형 스태킹 앙상블 학습"""
        logger.info(f"적응형 스태킹 앙상블 학습 시작 - Meta Model: {self.meta_model_type}")
        
        # Out-of-fold 예측 생성
        oof_predictions = self._generate_advanced_oof_predictions(X, y)
        
        # 고급 메타 피처 생성
        if self.use_advanced_features:
            meta_features = self._create_advanced_meta_features(oof_predictions, X)
        else:
            meta_features = oof_predictions
        
        # 피처 선택 및 스케일링
        meta_features_processed = self._preprocess_meta_features(meta_features, y)
        
        # 고급 메타 모델 학습
        self._train_advanced_meta_model(meta_features_processed, y)
        
        self.is_fitted = True
        logger.info("적응형 스태킹 앙상블 학습 완료")
    
    def _generate_advanced_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """고급 Out-of-fold 예측 생성"""
        
        oof_predictions = pd.DataFrame(index=X.index)
        
        # TimeSeriesSplit + StratifiedKFold 조합
        tscv = TimeSeriesSplit(n_splits=5)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 두 가지 폴딩 방법 모두 사용
        cv_methods = [('time', tscv), ('stratified', skf)]
        
        for cv_name, cv_splitter in cv_methods:
            for model_name, model in self.base_models.items():
                oof_pred = np.zeros(len(X))
                
                for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
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
                        logger.error(f"{model_name} {cv_name} 폴드 {fold} 실패: {str(e)}")
                        oof_pred[val_idx] = 0.0201
                
                column_name = f'{model_name}_{cv_name}'
                oof_predictions[column_name] = oof_pred
        
        return oof_predictions
    
    def _clone_model(self, model: BaseModel) -> BaseModel:
        """모델 복사"""
        try:
            from models import HighPerformanceModelFactory
            
            model_type_mapping = {
                'HighPerformanceLightGBM': 'lightgbm',
                'LightGBM': 'lightgbm',
                'HighPerformanceXGBoost': 'xgboost', 
                'XGBoost': 'xgboost',
                'HighPerformanceCatBoost': 'catboost',
                'CatBoost': 'catboost',
                'RTX4060TiDeepCTR': 'deepctr',
                'DeepCTR': 'deepctr',
                'HighPerformanceLogistic': 'logistic',
                'LogisticRegression': 'logistic'
            }
            
            model_type = model_type_mapping.get(model.name, 'logistic')
            
            clone_kwargs = {'params': model.params.copy()}
            if 'deepctr' in model_type.lower():
                clone_kwargs['input_dim'] = getattr(model, 'input_dim', 100)
            
            return HighPerformanceModelFactory.create_high_performance_model(model_type, **clone_kwargs)
            
        except Exception as e:
            logger.warning(f"모델 복사 실패: {e}")
            return model
    
    def _create_advanced_meta_features(self, oof_predictions: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """고급 메타 피처 생성"""
        
        meta_features = oof_predictions.copy()
        
        try:
            # 기본 통계 피처
            meta_features['pred_mean'] = oof_predictions.mean(axis=1)
            meta_features['pred_std'] = oof_predictions.std(axis=1)
            meta_features['pred_min'] = oof_predictions.min(axis=1)
            meta_features['pred_max'] = oof_predictions.max(axis=1)
            meta_features['pred_median'] = oof_predictions.median(axis=1)
            meta_features['pred_q25'] = oof_predictions.quantile(0.25, axis=1)
            meta_features['pred_q75'] = oof_predictions.quantile(0.75, axis=1)
            
            # 고급 통계 피처
            meta_features['pred_skew'] = oof_predictions.skew(axis=1)
            meta_features['pred_kurt'] = oof_predictions.kurtosis(axis=1)
            meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
            meta_features['pred_iqr'] = meta_features['pred_q75'] - meta_features['pred_q25']
            meta_features['pred_cv'] = meta_features['pred_std'] / (meta_features['pred_mean'] + 1e-8)
            
            # 순위 기반 피처
            for col in oof_predictions.columns:
                meta_features[f'{col}_rank'] = oof_predictions[col].rank(pct=True)
                meta_features[f'{col}_rank_norm'] = (oof_predictions[col].rank() - 1) / (len(oof_predictions) - 1)
            
            # 모델간 관계 피처
            model_cols = oof_predictions.columns.tolist()
            for i, col1 in enumerate(model_cols):
                for col2 in model_cols[i+1:]:
                    meta_features[f'{col1}_{col2}_diff'] = oof_predictions[col1] - oof_predictions[col2]
                    meta_features[f'{col1}_{col2}_ratio'] = oof_predictions[col1] / (oof_predictions[col2] + 1e-8)
                    meta_features[f'{col1}_{col2}_avg'] = (oof_predictions[col1] + oof_predictions[col2]) / 2
                    meta_features[f'{col1}_{col2}_max'] = np.maximum(oof_predictions[col1], oof_predictions[col2])
                    meta_features[f'{col1}_{col2}_min'] = np.minimum(oof_predictions[col1], oof_predictions[col2])
            
            # 신뢰도/일치도 피처
            meta_features['prediction_consensus'] = 1 - meta_features['pred_std']
            meta_features['prediction_confidence'] = np.exp(-meta_features['pred_std'])
            meta_features['agreement_strength'] = 1 / (1 + meta_features['pred_cv'])
            
            # 분포 기반 피처
            for percentile in [10, 25, 50, 75, 90]:
                threshold = oof_predictions.quantile(percentile/100, axis=0).mean()
                meta_features[f'above_p{percentile}'] = (oof_predictions > threshold).sum(axis=1)
            
            # 원본 데이터 요약 피처 (선택적)
            if len(X.columns) <= 100:
                try:
                    numeric_cols = X.select_dtypes(include=[np.number]).columns[:20]
                    if len(numeric_cols) > 0:
                        meta_features['x_mean'] = X[numeric_cols].mean(axis=1)
                        meta_features['x_std'] = X[numeric_cols].std(axis=1)
                        meta_features['x_median'] = X[numeric_cols].median(axis=1)
                        meta_features['x_max'] = X[numeric_cols].max(axis=1)
                        meta_features['x_min'] = X[numeric_cols].min(axis=1)
                except:
                    pass
            
            # 고차 상호작용 피처
            try:
                # 2차 다항식 피처 (주요 피처만)
                important_features = ['pred_mean', 'pred_std', 'pred_median']
                for feat in important_features:
                    if feat in meta_features.columns:
                        meta_features[f'{feat}_squared'] = meta_features[feat] ** 2
                        meta_features[f'{feat}_log'] = np.log(meta_features[feat] + 1e-8)
                        meta_features[f'{feat}_sqrt'] = np.sqrt(np.abs(meta_features[feat]))
            except Exception as e:
                logger.debug(f"고차 피처 생성 실패: {e}")
            
        except Exception as e:
            logger.warning(f"고급 메타 피처 생성 중 오류: {e}")
        
        return meta_features
    
    def _preprocess_meta_features(self, meta_features: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """메타 피처 전처리"""
        
        # 결측치 및 무한값 처리
        meta_features_clean = meta_features.fillna(0)
        meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 피처 선택 (상관관계 기반)
        try:
            from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
            
            # 너무 많은 피처가 있는 경우 선택
            if meta_features_clean.shape[1] > 100:
                # 상호정보량 기반 선택
                selector = SelectKBest(score_func=mutual_info_regression, k=min(80, meta_features_clean.shape[1]))
                meta_features_selected = selector.fit_transform(meta_features_clean, y)
                selected_features = meta_features_clean.columns[selector.get_support()]
                meta_features_clean = pd.DataFrame(meta_features_selected, 
                                                  columns=selected_features,
                                                  index=meta_features_clean.index)
                
                self.feature_selector = selector
                logger.info(f"피처 선택 완료: {len(selected_features)}개 선택")
        
        except Exception as e:
            logger.warning(f"피처 선택 실패: {e}")
        
        # 스케일링
        self.feature_scaler = RobustScaler()
        meta_features_scaled = self.feature_scaler.fit_transform(meta_features_clean)
        meta_features_processed = pd.DataFrame(
            meta_features_scaled, 
            columns=meta_features_clean.columns,
            index=meta_features_clean.index
        )
        
        return meta_features_processed
    
    def _train_advanced_meta_model(self, meta_features: pd.DataFrame, y: pd.Series):
        """고급 메타 모델 학습"""
        
        if self.meta_model_type == 'ridge_advanced':
            # 고급 Ridge with CV
            alphas = np.logspace(-3, 3, 50)
            self.meta_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
            
        elif self.meta_model_type == 'elastic_net_advanced':
            # ElasticNet with CV
            self.meta_model = ElasticNetCV(
                alphas=np.logspace(-3, 1, 20),
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                cv=5,
                max_iter=2000
            )
            
        elif self.meta_model_type == 'mlp_advanced':
            # 고급 MLP
            self.meta_model = MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                max_iter=2000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                alpha=0.001,
                learning_rate='adaptive'
            )
            
        else:
            # 기본 Ridge
            self.meta_model = RidgeCV(cv=5)
        
        try:
            self.meta_model.fit(meta_features, y)
            
            # 성능 평가
            meta_pred = self.meta_model.predict(meta_features)
            meta_pred = np.clip(meta_pred, 0.001, 0.999)
            
            metrics_calc = CTRMetrics()
            combined_score = metrics_calc.combined_score(y, meta_pred)
            logger.info(f"고급 메타 모델 Combined Score: {combined_score:.4f}")
            
        except Exception as e:
            logger.error(f"고급 메타 모델 학습 실패: {str(e)}")
            raise
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """적응형 스태킹 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        try:
            # 기본 예측을 DataFrame으로 변환
            pred_df = pd.DataFrame(base_predictions)
            
            # 고급 메타 피처 생성 (추론용)
            if self.use_advanced_features:
                meta_features = self._create_inference_meta_features(pred_df)
            else:
                meta_features = pred_df
            
            # 전처리 적용
            meta_features_clean = meta_features.fillna(0)
            meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # 피처 선택 적용
            if self.feature_selector is not None:
                try:
                    meta_features_selected = self.feature_selector.transform(meta_features_clean)
                    selected_columns = meta_features_clean.columns[self.feature_selector.get_support()]
                    meta_features_clean = pd.DataFrame(meta_features_selected, columns=selected_columns)
                except Exception as e:
                    logger.warning(f"피처 선택 적용 실패: {e}")
            
            # 스케일링 적용
            meta_features_scaled = self.feature_scaler.transform(meta_features_clean)
            
            # 예측
            ensemble_pred = self.meta_model.predict(meta_features_scaled)
            ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"적응형 스태킹 예측 실패: {str(e)}")
            return np.mean(list(base_predictions.values()), axis=0)
    
    def _create_inference_meta_features(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """추론용 메타 피처 생성"""
        
        meta_features = pred_df.copy()
        
        try:
            # 기본 통계 피처 (학습 시와 동일)
            meta_features['pred_mean'] = pred_df.mean(axis=1)
            meta_features['pred_std'] = pred_df.std(axis=1)
            meta_features['pred_min'] = pred_df.min(axis=1)
            meta_features['pred_max'] = pred_df.max(axis=1)
            meta_features['pred_median'] = pred_df.median(axis=1)
            meta_features['pred_q25'] = pred_df.quantile(0.25, axis=1)
            meta_features['pred_q75'] = pred_df.quantile(0.75, axis=1)
            
            # 고급 통계 피처
            meta_features['pred_skew'] = pred_df.skew(axis=1)
            meta_features['pred_kurt'] = pred_df.kurtosis(axis=1)
            meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
            meta_features['pred_iqr'] = meta_features['pred_q75'] - meta_features['pred_q25']
            meta_features['pred_cv'] = meta_features['pred_std'] / (meta_features['pred_mean'] + 1e-8)
            
            # 순위 기반 피처
            for col in pred_df.columns:
                meta_features[f'{col}_rank'] = pred_df[col].rank(pct=True)
                meta_features[f'{col}_rank_norm'] = (pred_df[col].rank() - 1) / (len(pred_df) - 1)
            
            # 모델간 관계 피처
            model_cols = pred_df.columns.tolist()
            for i, col1 in enumerate(model_cols):
                for col2 in model_cols[i+1:]:
                    meta_features[f'{col1}_{col2}_diff'] = pred_df[col1] - pred_df[col2]
                    meta_features[f'{col1}_{col2}_ratio'] = pred_df[col1] / (pred_df[col2] + 1e-8)
                    meta_features[f'{col1}_{col2}_avg'] = (pred_df[col1] + pred_df[col2]) / 2
                    meta_features[f'{col1}_{col2}_max'] = np.maximum(pred_df[col1], pred_df[col2])
                    meta_features[f'{col1}_{col2}_min'] = np.minimum(pred_df[col1], pred_df[col2])
            
            # 신뢰도 피처
            meta_features['prediction_consensus'] = 1 - meta_features['pred_std']
            meta_features['prediction_confidence'] = np.exp(-meta_features['pred_std'])
            meta_features['agreement_strength'] = 1 / (1 + meta_features['pred_cv'])
            
            # 분포 기반 피처
            for percentile in [10, 25, 50, 75, 90]:
                threshold = pred_df.quantile(percentile/100, axis=0).mean()
                meta_features[f'above_p{percentile}'] = (pred_df > threshold).sum(axis=1)
            
            # 고차 상호작용 피처
            important_features = ['pred_mean', 'pred_std', 'pred_median']
            for feat in important_features:
                if feat in meta_features.columns:
                    meta_features[f'{feat}_squared'] = meta_features[feat] ** 2
                    meta_features[f'{feat}_log'] = np.log(meta_features[feat] + 1e-8)
                    meta_features[f'{feat}_sqrt'] = np.sqrt(np.abs(meta_features[feat]))
            
        except Exception as e:
            logger.warning(f"추론용 메타 피처 생성 중 오류: {e}")
        
        return meta_features

class UltraHighPerformanceEnsembleManager:
    """울트라 고성능 앙상블 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.performance_history = []
        
        # 특화 앙상블 참조
        self.ultra_ensemble = None
        self.stacking_ensemble = None
        
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        logger.info(f"고성능 기본 모델 추가: {name}")
    
    def create_ultra_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """울트라 고성능 앙상블 생성"""
        
        if ensemble_type == 'ultra_performance':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            optimization_method = kwargs.get('optimization_method', 'ultra_combined')
            ensemble = UltraHighPerformanceEnsemble(target_ctr, optimization_method)
            self.ultra_ensemble = ensemble
        
        elif ensemble_type == 'adaptive_stacking':
            meta_model_type = kwargs.get('meta_model_type', 'ridge_advanced')
            use_advanced_features = kwargs.get('use_advanced_features', True)
            ensemble = AdaptiveStackingEnsemble(meta_model_type, use_advanced_features)
            self.stacking_ensemble = ensemble
        
        else:
            raise ValueError(f"지원하지 않는 울트라 앙상블 타입: {ensemble_type}")
        
        # 기본 모델 추가
        for name, model in self.base_models.items():
            ensemble.add_base_model(name, model)
        
        self.ensembles[ensemble_type] = ensemble
        logger.info(f"울트라 앙상블 생성: {ensemble_type}")
        
        return ensemble
    
    def train_all_ultra_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """모든 울트라 앙상블 학습"""
        logger.info("울트라 고성능 앙상블 학습 시작")
        logger.info(f"목표: Combined Score 0.32+, CTR 편향 0.001 이하")
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                base_predictions[name] = pred
                
                # 개별 모델 성능 로깅
                combined_score = self.metrics_calculator.combined_score(y, pred)
                ctr_bias = abs(pred.mean() - y.mean())
                logger.info(f"{name}: Combined={combined_score:.4f}, CTR편향={ctr_bias:.4f}")
                
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
        
        # 각 울트라 앙상블 학습
        training_results = {}
        
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                start_time = time.time()
                
                ensemble.fit(X, y, base_predictions)
                
                training_time = time.time() - start_time
                training_results[ensemble_type] = {
                    'success': True,
                    'training_time': training_time
                }
                
                logger.info(f"{ensemble_type} 울트라 앙상블 학습 완료 (시간: {training_time:.2f}초)")
                
            except Exception as e:
                logger.error(f"{ensemble_type} 울트라 앙상블 학습 실패: {str(e)}")
                training_results[ensemble_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 메모리 정리
        gc.collect()
        
        logger.info(f"울트라 앙상블 학습 완료: {len(training_results)}개 중 {sum(1 for r in training_results.values() if r['success'])}개 성공")
    
    def evaluate_ultra_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """울트라 앙상블 성능 평가"""
        logger.info("울트라 고성능 앙상블 평가 시작")
        
        results = {}
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                base_predictions[name] = pred
                
                # 기본 모델 점수
                score = self.metrics_calculator.combined_score(y_val, pred)
                results[f"base_{name}"] = score
                
            except Exception as e:
                logger.error(f"{name} 검증 예측 실패: {str(e)}")
                results[f"base_{name}"] = 0.0
        
        # 울트라 앙상블 성능 평가
        best_ensemble_name = None
        best_score = 0.0
        
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    ensemble_pred = ensemble.predict_proba(base_predictions)
                    
                    # 종합 평가
                    combined_score = self.metrics_calculator.combined_score(y_val, ensemble_pred)
                    ap_score = self.metrics_calculator.average_precision(y_val, ensemble_pred)
                    wll_score = self.metrics_calculator.weighted_log_loss(y_val, ensemble_pred)
                    
                    results[f"ensemble_{ensemble_type}"] = combined_score
                    results[f"ensemble_{ensemble_type}_ap"] = ap_score
                    results[f"ensemble_{ensemble_type}_wll"] = wll_score
                    
                    # CTR 분석
                    predicted_ctr = ensemble_pred.mean()
                    actual_ctr = y_val.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    ctr_ratio = predicted_ctr / actual_ctr if actual_ctr > 0 else 1.0
                    
                    results[f"ensemble_{ensemble_type}_ctr_bias"] = ctr_bias
                    results[f"ensemble_{ensemble_type}_ctr_ratio"] = ctr_ratio
                    
                    # 예측 품질 분석
                    pred_diversity = len(np.unique(np.round(ensemble_pred, 6))) / len(ensemble_pred)
                    results[f"ensemble_{ensemble_type}_diversity"] = pred_diversity
                    
                    # 로깅
                    logger.info(f"{ensemble_type} 울트라 앙상블:")
                    logger.info(f"  Combined Score: {combined_score:.4f}")
                    logger.info(f"  CTR 편향: {ctr_bias:.4f} (목표: <0.001)")
                    logger.info(f"  CTR 비율: {ctr_ratio:.3f}")
                    logger.info(f"  예측 다양성: {pred_diversity:.4f}")
                    
                    # 목표 달성 여부 확인
                    if combined_score >= 0.32 and ctr_bias <= 0.001:
                        logger.info(f"🎯 {ensemble_type}: 울트라 목표 달성! (Combined≥0.32, CTR편향≤0.001)")
                    elif combined_score >= 0.30:
                        logger.info(f"✅ {ensemble_type}: 고성능 목표 달성! (Combined≥0.30)")
                    
                    # 최고 성능 앙상블 선택
                    if combined_score > best_score:
                        best_score = combined_score
                        best_ensemble_name = ensemble_type
                        self.best_ensemble = ensemble
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} 울트라 앙상블 평가 실패: {str(e)}")
                    results[f"ensemble_{ensemble_type}"] = 0.0
        
        # 최고 성능 결과 요약
        if best_ensemble_name:
            logger.info(f"🏆 최고 성능 울트라 앙상블: {best_ensemble_name} (Combined Score: {best_score:.4f})")
        
        self.ensemble_results = results
        return results
    
    def predict_with_best_ultra_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """최고 성능 울트라 앙상블로 예측"""
        
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
            if self.ensemble_results:
                best_base_name = None
                best_base_score = 0
                
                for result_name, score in self.ensemble_results.items():
                    if result_name.startswith('base_') and score > best_base_score:
                        best_base_score = score
                        best_base_name = result_name.replace('base_', '')
                
                if best_base_name and best_base_name in self.base_models:
                    logger.info(f"최고 성능 기본 모델 사용: {best_base_name}")
                    return self.base_models[best_base_name].predict_proba(X)
            
            # 평균 앙상블 사용
            logger.info("평균 앙상블 사용")
            return np.mean(list(base_predictions.values()), axis=0)
        
        # 최고 성능 울트라 앙상블 사용
        logger.info(f"최고 성능 울트라 앙상블 사용: {self.best_ensemble.name}")
        return self.best_ensemble.predict_proba(base_predictions)
    
    def save_ultra_ensembles(self, output_dir: Path = None):
        """울트라 앙상블 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 개별 앙상블 저장
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                ensemble_path = output_dir / f"ultra_ensemble_{ensemble_type}.pkl"
                
                try:
                    with open(ensemble_path, 'wb') as f:
                        pickle.dump(ensemble, f)
                    
                    logger.info(f"{ensemble_type} 울트라 앙상블 저장: {ensemble_path}")
                except Exception as e:
                    logger.error(f"{ensemble_type} 울트라 앙상블 저장 실패: {str(e)}")
        
        # 성능 결과 저장
        if self.ensemble_results:
            try:
                import json
                results_path = output_dir / "ultra_ensemble_results.json"
                with open(results_path, 'w') as f:
                    json.dump(self.ensemble_results, f, indent=2, default=str)
                logger.info("울트라 앙상블 결과 저장 완료")
            except Exception as e:
                logger.error(f"앙상블 결과 저장 실패: {str(e)}")
        
        # 최고 앙상블 정보 저장
        best_info = {
            'best_ensemble_type': self.best_ensemble.name if self.best_ensemble else None,
            'ultra_performance_available': self.ultra_ensemble is not None and self.ultra_ensemble.is_fitted,
            'adaptive_stacking_available': self.stacking_ensemble is not None and self.stacking_ensemble.is_fitted,
            'target_achieved': any(
                self.ensemble_results.get(f'ensemble_{etype}', 0) >= 0.32 and
                self.ensemble_results.get(f'ensemble_{etype}_ctr_bias', 1) <= 0.001
                for etype in self.ensembles.keys()
            ),
            'high_performance_achieved': any(
                self.ensemble_results.get(f'ensemble_{etype}', 0) >= 0.30
                for etype in self.ensembles.keys()
            )
        }
        
        try:
            import json
            info_path = output_dir / "ultra_ensemble_info.json"
            with open(info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
        except Exception as e:
            logger.error(f"울트라 앙상블 정보 저장 실패: {str(e)}")
    
    def get_ultra_summary(self) -> Dict[str, Any]:
        """울트라 앙상블 요약 정보"""
        return {
            'total_ensembles': len(self.ensembles),
            'fitted_ensembles': sum(1 for e in self.ensembles.values() if e.is_fitted),
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results,
            'base_models_count': len(self.base_models),
            'ultra_ensemble_available': self.ultra_ensemble is not None and self.ultra_ensemble.is_fitted,
            'stacking_ensemble_available': self.stacking_ensemble is not None and self.stacking_ensemble.is_fitted,
            'ultra_target_achieved': any(
                self.ensemble_results.get(f'ensemble_{etype}', 0) >= 0.32 and
                self.ensemble_results.get(f'ensemble_{etype}_ctr_bias', 1) <= 0.001
                for etype in self.ensembles.keys()
            ),
            'high_performance_achieved': any(
                self.ensemble_results.get(f'ensemble_{etype}', 0) >= 0.30
                for etype in self.ensembles.keys()
            ),
            'ensemble_types': list(self.ensembles.keys()),
            'performance_history': self.performance_history
        }

# 호환성을 위한 alias
CTREnsembleManager = UltraHighPerformanceEnsembleManager
EnsembleManager = UltraHighPerformanceEnsembleManager
CTROptimalEnsemble = UltraHighPerformanceEnsemble
CTRMetaLearning = AdaptiveStackingEnsemble