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
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna가 설치되지 않았습니다. 하이퍼파라미터 튜닝 기능이 비활성화됩니다.")

from config import Config
from models import BaseModel
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """앙상블 모델 기본 클래스 - 1070만행 최적화"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.training_data_size = 0
        self.target_ctr = 0.0201
        
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
        """모든 기본 모델의 예측 수집 - 대용량 데이터 배치 처리"""
        predictions = {}
        
        # 대용량 데이터 배치 처리
        batch_size = 100000
        total_size = len(X)
        
        logger.info(f"기본 모델 예측 수집 시작 (데이터 크기: {total_size:,}행)")
        
        for name, model in self.base_models.items():
            try:
                model_predictions = []
                
                for i in range(0, total_size, batch_size):
                    end_idx = min(i + batch_size, total_size)
                    X_batch = X.iloc[i:end_idx]
                    
                    batch_pred = model.predict_proba(X_batch)
                    model_predictions.append(batch_pred)
                    
                    if (i // batch_size + 1) % 10 == 0:
                        logger.info(f"{name} 모델 예측 진행: {end_idx:,}/{total_size:,}")
                
                predictions[name] = np.concatenate(model_predictions)
                logger.info(f"{name} 모델 예측 완료")
                
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                predictions[name] = np.full(total_size, self.target_ctr)
        
        return predictions

class CTROptimalEnsemble(BaseEnsemble):
    """CTR 예측 최적화 앙상블 - Combined Score 0.30+ 목표"""
    
    def __init__(self, target_ctr: float = 0.0201, optimization_method: str = 'combined_plus'):
        super().__init__("CTROptimalEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.final_weights = {}
        self.ctr_calibrator = None
        self.metrics_calculator = CTRMetrics()
        self.temperature = 1.0
        self.bias_correction = 0.0
        self.diversity_bonus = {}
        self.performance_weights = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """CTR 최적화 앙상블 학습 - Combined Score 0.30+ 목표"""
        logger.info(f"CTR 최적화 앙상블 학습 시작 - 방법: {self.optimization_method}")
        logger.info(f"목표: Combined Score 0.30+")
        
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models}")
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        self.training_data_size = len(X)
        
        # 1단계: 고급 다단계 가중치 최적화
        self.final_weights = self._advanced_multi_stage_optimization(base_predictions, y)
        
        # 2단계: CTR 특화 후처리 최적화
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        self._optimize_ctr_postprocessing_advanced(ensemble_pred, y)
        
        # 3단계: 성능 검증
        final_score = self._validate_performance(base_predictions, y)
        
        self.is_fitted = True
        
        if final_score >= 0.30:
            logger.info(f"🎯 CTR 최적화 앙상블 학습 완료 - Combined Score: {final_score:.4f} (목표 달성!)")
        else:
            logger.warning(f"⚠️ CTR 최적화 앙상블 학습 완료 - Combined Score: {final_score:.4f} (목표 미달성)")
    
    def _advanced_multi_stage_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """고급 다단계 가중치 최적화 - Combined Score 0.30+ 목표"""
        
        model_names = list(base_predictions.keys())
        
        # 1단계: 개별 모델 성능 평가 (강화)
        individual_scores = {}
        for name, pred in base_predictions.items():
            combined_score = self.metrics_calculator.combined_score(y, pred)
            ap_score = self.metrics_calculator.average_precision(y, pred)
            wll_score = self.metrics_calculator.weighted_log_loss(y, pred)
            
            # CTR 편향 점수
            predicted_ctr = pred.mean()
            actual_ctr = y.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            ctr_score = np.exp(-ctr_bias * 200)  # 더 강한 CTR 편향 패널티
            
            # 예측 다양성 점수
            diversity_score = self._calculate_prediction_diversity(pred)
            
            # 종합 성능 점수
            total_score = (0.4 * combined_score + 0.2 * ap_score + 
                          0.2 * (1/(1+wll_score)) + 0.1 * ctr_score + 0.1 * diversity_score)
            
            individual_scores[name] = total_score
            self.performance_weights[name] = total_score
            
        logger.info(f"개별 모델 성능 (강화): {individual_scores}")
        
        # 2단계: 다양성 분석 및 보너스
        diversity_matrix = self._calculate_diversity_matrix(base_predictions)
        self.diversity_bonus = self._calculate_diversity_bonus(diversity_matrix, individual_scores)
        
        # 3단계: 초기 가중치 설정 (성능 + 다양성)
        initial_weights = {}
        for name in model_names:
            base_weight = individual_scores[name]
            diversity_weight = self.diversity_bonus.get(name, 0.0)
            initial_weights[name] = base_weight + diversity_weight
        
        # 정규화
        total_weight = sum(initial_weights.values())
        if total_weight > 0:
            initial_weights = {k: v/total_weight for k, v in initial_weights.items()}
        else:
            initial_weights = {name: 1.0/len(model_names) for name in model_names}
        
        # 4단계: 계층적 그리드 서치
        optimized_weights = self._hierarchical_grid_search(base_predictions, y, initial_weights)
        
        # 5단계: 고급 Optuna 최적화
        if OPTUNA_AVAILABLE and len(model_names) <= 5:
            try:
                final_weights = self._advanced_optuna_optimization(base_predictions, y, optimized_weights)
            except Exception as e:
                logger.warning(f"고급 Optuna 최적화 실패: {e}")
                final_weights = optimized_weights
        else:
            final_weights = optimized_weights
        
        # 6단계: 최종 성능 검증 및 조정
        final_weights = self._performance_based_adjustment(base_predictions, y, final_weights)
        
        logger.info(f"최종 가중치: {final_weights}")
        return final_weights
    
    def _calculate_prediction_diversity(self, predictions: np.ndarray) -> float:
        """예측 다양성 계산"""
        try:
            unique_count = len(np.unique(predictions))
            expected_diversity = max(1000, len(predictions) // 10000)  # 1070만행 기준
            
            diversity_ratio = min(unique_count / expected_diversity, 1.0)
            
            # 엔트로피 기반 다양성
            p = np.clip(predictions, 1e-15, 1 - 1e-15)
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            entropy_score = entropy / np.log2(2)  # 정규화
            
            # 분산 기반 다양성
            variance_score = min(np.std(predictions) / 0.1, 1.0)
            
            diversity_score = 0.4 * diversity_ratio + 0.3 * entropy_score + 0.3 * variance_score
            
            return diversity_score
        except Exception as e:
            logger.warning(f"예측 다양성 계산 실패: {e}")
            return 0.5
    
    def _calculate_diversity_matrix(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """다양성 행렬 계산 - 1070만행 최적화"""
        model_names = list(base_predictions.keys())
        diversity_matrix = {}
        
        # 대용량 데이터 샘플링
        sample_size = min(100000, len(list(base_predictions.values())[0]))
        if sample_size < len(list(base_predictions.values())[0]):
            sample_indices = np.random.choice(len(list(base_predictions.values())[0]), sample_size, replace=False)
            sampled_predictions = {name: pred[sample_indices] for name, pred in base_predictions.items()}
        else:
            sampled_predictions = base_predictions
        
        for name1 in model_names:
            diversity_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    diversity_matrix[name1][name2] = 0.0
                else:
                    try:
                        # Pearson 상관계수
                        corr = np.corrcoef(sampled_predictions[name1], sampled_predictions[name2])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                        
                        # 순위 상관계수
                        rank_corr = self._calculate_rank_correlation(
                            sampled_predictions[name1], sampled_predictions[name2]
                        )
                        
                        # KL Divergence
                        kl_div = self._calculate_kl_divergence(
                            sampled_predictions[name1], sampled_predictions[name2]
                        )
                        
                        # 종합 다양성 점수 (낮을수록 다양함)
                        diversity_score = 0.4 * abs(corr) + 0.3 * abs(rank_corr) + 0.3 * (1 - kl_div)
                        diversity_matrix[name1][name2] = diversity_score
                        
                    except Exception as e:
                        logger.warning(f"{name1}-{name2} 다양성 계산 실패: {e}")
                        diversity_matrix[name1][name2] = 0.5
        
        return diversity_matrix
    
    def _calculate_rank_correlation(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """순위 상관계수 계산"""
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(pred1, pred2)
            return corr if not np.isnan(corr) else 0.0
        except:
            # 간단한 순위 상관계수 계산
            rank1 = pd.Series(pred1).rank(pct=True)
            rank2 = pd.Series(pred2).rank(pct=True)
            return np.corrcoef(rank1, rank2)[0, 1] if not np.isnan(np.corrcoef(rank1, rank2)[0, 1]) else 0.0
    
    def _calculate_kl_divergence(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """KL Divergence 계산"""
        try:
            # 히스토그램 기반 KL Divergence
            bins = 50
            hist1, _ = np.histogram(pred1, bins=bins, range=(0, 1), density=True)
            hist2, _ = np.histogram(pred2, bins=bins, range=(0, 1), density=True)
            
            # 확률 분포로 변환
            p = hist1 / (hist1.sum() + 1e-15)
            q = hist2 / (hist2.sum() + 1e-15)
            
            # KL Divergence
            kl = np.sum(p * np.log((p + 1e-15) / (q + 1e-15)))
            
            # 정규화 (0-1 범위)
            return min(1.0, kl / 10.0)
        except Exception as e:
            logger.warning(f"KL Divergence 계산 실패: {e}")
            return 0.5
    
    def _calculate_diversity_bonus(self, diversity_matrix: Dict[str, Dict[str, float]], 
                                 performance_scores: Dict[str, float]) -> Dict[str, float]:
        """다양성 보너스 계산"""
        diversity_bonus = {}
        
        for model_name in diversity_matrix.keys():
            # 다른 모델들과의 평균 다양성 (낮을수록 다양함)
            other_models = [m for m in diversity_matrix.keys() if m != model_name]
            if other_models:
                avg_diversity = np.mean([diversity_matrix[model_name][other] for other in other_models])
                
                # 다양성 보너스 (다양할수록 높은 보너스)
                diversity_bonus_value = (1.0 - avg_diversity) * 0.1
                
                # 성능이 좋은 모델에게 더 큰 다양성 보너스
                performance_multiplier = performance_scores.get(model_name, 0.5)
                final_bonus = diversity_bonus_value * performance_multiplier
                
                diversity_bonus[model_name] = final_bonus
            else:
                diversity_bonus[model_name] = 0.0
        
        logger.info(f"다양성 보너스: {diversity_bonus}")
        return diversity_bonus
    
    def _hierarchical_grid_search(self, base_predictions: Dict[str, np.ndarray], y: pd.Series,
                                initial_weights: Dict[str, float]) -> Dict[str, float]:
        """계층적 그리드 서치 최적화"""
        model_names = list(base_predictions.keys())
        best_weights = initial_weights.copy()
        best_score = self._evaluate_ensemble(base_predictions, y, best_weights)
        
        logger.info(f"계층적 그리드 서치 시작 - 초기 점수: {best_score:.4f}")
        
        # 3단계 계층적 최적화
        adjustment_steps = [0.1, 0.03, 0.01]
        
        for step_level, step in enumerate(adjustment_steps):
            logger.info(f"계층 {step_level + 1} 최적화 (step: {step})")
            improved = True
            iteration = 0
            
            while improved and iteration < 30:
                improved = False
                iteration += 1
                
                for target_model in model_names:
                    for direction in [-1, 1]:
                        # 가중치 조정 시도
                        test_weights = best_weights.copy()
                        test_weights[target_model] += direction * step
                        
                        # 음수 방지
                        if test_weights[target_model] < 0.01:
                            continue
                        
                        # 정규화
                        total_weight = sum(test_weights.values())
                        if total_weight > 0:
                            test_weights = {k: v/total_weight for k, v in test_weights.items()}
                        else:
                            continue
                        
                        # 평가
                        score = self._evaluate_ensemble(base_predictions, y, test_weights)
                        
                        if score > best_score + 1e-6:  # 미세한 개선도 수용
                            best_score = score
                            best_weights = test_weights
                            improved = True
                            
                            if score >= 0.30:
                                logger.info(f"🎯 목표 달성! 점수: {score:.4f}")
        
        logger.info(f"계층적 그리드 서치 완료 - 최종 점수: {best_score:.4f}")
        return best_weights
    
    def _advanced_optuna_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series,
                                    initial_weights: Dict[str, float]) -> Dict[str, float]:
        """고급 Optuna 기반 정밀 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        
        def advanced_objective(trial):
            weights = {}
            
            # 초기값 기반 범위 설정 (더 넓은 범위)
            for name in model_names:
                initial_val = initial_weights.get(name, 1.0/len(model_names))
                low_bound = max(0.01, initial_val - 0.4)
                high_bound = min(0.99, initial_val + 0.4)
                weights[name] = trial.suggest_float(f'weight_{name}', low_bound, high_bound)
            
            # 정규화
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                return 0.0
            
            score = self._evaluate_ensemble(base_predictions, y, weights)
            
            # Combined Score 0.30+ 목표에 대한 보너스
            if score >= 0.30:
                bonus = (score - 0.30) * 10  # 목표 초과시 큰 보너스
                return score + bonus
            
            return score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                seed=42,
                n_startup_trials=20,
                n_ei_candidates=48,
                multivariate=True
            ),
            pruner=HyperbandPruner(
                min_resource=5,
                max_resource=50,
                reduction_factor=4
            )
        )
        
        study.optimize(advanced_objective, n_trials=200, show_progress_bar=False)
        
        optimized_weights = {}
        for param_name, weight in study.best_params.items():
            model_name = param_name.replace('weight_', '')
            optimized_weights[model_name] = weight
        
        # 정규화
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
        
        logger.info(f"고급 Optuna 최적화 완료 - 점수: {study.best_value:.4f}")
        return optimized_weights
    
    def _performance_based_adjustment(self, base_predictions: Dict[str, np.ndarray], y: pd.Series,
                                    weights: Dict[str, float]) -> Dict[str, float]:
        """성능 기반 최종 조정"""
        
        # 현재 앙상블 성능
        current_score = self._evaluate_ensemble(base_predictions, y, weights)
        
        if current_score >= 0.30:
            logger.info(f"목표 달성으로 조정 생략: {current_score:.4f}")
            return weights
        
        # 성능이 낮은 경우 top performer에 더 많은 가중치
        adjusted_weights = weights.copy()
        
        # 최고 성능 모델 찾기
        best_model = max(self.performance_weights.keys(), 
                        key=lambda x: self.performance_weights[x])
        
        # 가중치 재분배 (보수적)
        redistribution_factor = 0.1
        weight_to_redistribute = 0.0
        
        for model_name in adjusted_weights.keys():
            if model_name != best_model:
                reduction = adjusted_weights[model_name] * redistribution_factor
                adjusted_weights[model_name] -= reduction
                weight_to_redistribute += reduction
        
        adjusted_weights[best_model] += weight_to_redistribute
        
        # 정규화
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        # 성능 검증
        new_score = self._evaluate_ensemble(base_predictions, y, adjusted_weights)
        
        if new_score > current_score:
            logger.info(f"성능 기반 조정 성공: {current_score:.4f} → {new_score:.4f}")
            return adjusted_weights
        else:
            logger.info(f"성능 기반 조정 취소: {new_score:.4f} < {current_score:.4f}")
            return weights
    
    def _evaluate_ensemble(self, base_predictions: Dict[str, np.ndarray], y: pd.Series, 
                         weights: Dict[str, float]) -> float:
        """앙상블 성능 평가 - Combined Score 0.30+ 목표"""
        try:
            ensemble_pred = np.zeros(len(y))
            for name, weight in weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
            
            # Combined Score 계산
            combined_score = self.metrics_calculator.combined_score(y, ensemble_pred)
            
            # CTR 편향 패널티 강화
            predicted_ctr = ensemble_pred.mean()
            actual_ctr = y.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            ctr_penalty = np.exp(-ctr_bias * 300)  # 매우 강한 CTR 편향 패널티
            
            # 최종 점수 (Combined Score 0.30+ 목표)
            final_score = combined_score * ctr_penalty
            
            return final_score
        except Exception as e:
            logger.warning(f"앙상블 평가 실패: {e}")
            return 0.0
    
    def _optimize_ctr_postprocessing_advanced(self, predictions: np.ndarray, y: pd.Series):
        """CTR 특화 고급 후처리 최적화"""
        logger.info("CTR 고급 후처리 최적화 시작")
        
        try:
            # 1. 편향 보정
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            
            # 2. 고급 Temperature scaling 최적화
            self._optimize_advanced_temperature_scaling(predictions, y)
            
            # 3. CTR 분포 매칭 고도화
            self._optimize_advanced_distribution_matching(predictions, y)
            
            # 4. 분위수별 보정
            self._optimize_quantile_correction(predictions, y)
            
            logger.info(f"CTR 고급 후처리 최적화 완료")
            logger.info(f"편향: {self.bias_correction:.4f}, Temperature: {self.temperature:.3f}")
            
        except Exception as e:
            logger.error(f"CTR 고급 후처리 최적화 실패: {e}")
            self.bias_correction = 0.0
            self.temperature = 1.0
    
    def _optimize_advanced_temperature_scaling(self, predictions: np.ndarray, y: pd.Series):
        """고급 Temperature scaling 최적화"""
        try:
            from scipy.optimize import minimize_scalar
            
            def advanced_temperature_loss(temp):
                if temp <= 0.1 or temp > 10:
                    return float('inf')
                
                # Logit 변환
                pred_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                
                # Temperature 적용
                calibrated_logits = logits / temp
                calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                # Combined Score 계산
                combined_score = self.metrics_calculator.combined_score(y, calibrated_probs)
                
                # Combined Score 0.30+ 목표에 대한 보너스/패널티
                if combined_score >= 0.30:
                    target_bonus = (combined_score - 0.30) * 100
                    return -(combined_score + target_bonus)  # 최대화를 위해 음수
                else:
                    target_penalty = (0.30 - combined_score) * 50
                    return -(combined_score - target_penalty)
            
            result = minimize_scalar(advanced_temperature_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
            
        except Exception as e:
            logger.warning(f"고급 Temperature scaling 최적화 실패: {e}")
            self.temperature = 1.0
    
    def _optimize_advanced_distribution_matching(self, predictions: np.ndarray, y: pd.Series):
        """고급 분포 매칭 최적화"""
        try:
            # CTR 분포 특성 고급 분석
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            
            # 다중 분위수 기반 매칭 (더 세밀한 분석)
            pred_percentiles = np.percentile(predictions, [10, 25, 50, 75, 85, 90, 95, 97, 99])
            
            # 고CTR 구간별 세밀한 보정
            self.quantile_corrections = {}
            
            quantiles = [0.85, 0.90, 0.95, 0.97, 0.99]
            for i, q in enumerate(quantiles):
                threshold = np.percentile(predictions, q * 100)
                high_ctr_mask = predictions >= threshold
                
                if high_ctr_mask.sum() > 100:  # 충분한 샘플이 있을 때만
                    high_ctr_actual_rate = y[high_ctr_mask].mean()
                    high_ctr_pred_rate = predictions[high_ctr_mask].mean()
                    
                    if high_ctr_pred_rate > 0:
                        correction_factor = high_ctr_actual_rate / high_ctr_pred_rate
                        self.quantile_corrections[q] = {
                            'threshold': threshold,
                            'correction_factor': correction_factor
                        }
            
            logger.info(f"고급 분포 매칭 완료 - 분위수별 보정: {len(self.quantile_corrections)}개")
            
        except Exception as e:
            logger.warning(f"고급 분포 매칭 최적화 실패: {e}")
            self.quantile_corrections = {}
    
    def _optimize_quantile_correction(self, predictions: np.ndarray, y: pd.Series):
        """분위수별 보정 최적화"""
        try:
            # 10분위수별 세밀한 보정
            self.decile_corrections = {}
            
            for decile in range(1, 11):
                lower_bound = np.percentile(predictions, (decile - 1) * 10)
                upper_bound = np.percentile(predictions, decile * 10)
                
                decile_mask = (predictions >= lower_bound) & (predictions < upper_bound)
                
                if decile_mask.sum() > 100:
                    decile_actual_rate = y[decile_mask].mean()
                    decile_pred_rate = predictions[decile_mask].mean()
                    
                    if decile_pred_rate > 0:
                        correction_factor = decile_actual_rate / decile_pred_rate
                        self.decile_corrections[decile] = {
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'correction_factor': correction_factor
                        }
            
            logger.info(f"분위수별 보정 완료 - 10분위수 보정: {len(self.decile_corrections)}개")
            
        except Exception as e:
            logger.warning(f"분위수별 보정 실패: {e}")
            self.decile_corrections = {}
    
    def _validate_performance(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> float:
        """최종 성능 검증"""
        try:
            ensemble_pred = self._create_weighted_ensemble(base_predictions)
            calibrated_pred = self._apply_postprocessing_advanced(ensemble_pred)
            
            final_score = self.metrics_calculator.combined_score(y, calibrated_pred)
            
            # 상세 성능 분석
            ap_score = self.metrics_calculator.average_precision(y, calibrated_pred)
            wll_score = self.metrics_calculator.weighted_log_loss(y, calibrated_pred)
            
            predicted_ctr = calibrated_pred.mean()
            actual_ctr = y.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            logger.info(f"최종 성능 검증:")
            logger.info(f"  Combined Score: {final_score:.4f}")
            logger.info(f"  AP Score: {ap_score:.4f}")
            logger.info(f"  WLL Score: {wll_score:.4f}")
            logger.info(f"  CTR 편향: {ctr_bias:.4f}")
            logger.info(f"  목표 달성: {'✓' if final_score >= 0.30 else '✗'}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"성능 검증 실패: {e}")
            return 0.0
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred
    
    def _apply_postprocessing_advanced(self, predictions: np.ndarray) -> np.ndarray:
        """고급 후처리 적용"""
        try:
            processed = predictions.copy()
            
            # 1. Temperature scaling
            if self.temperature != 1.0:
                pred_clipped = np.clip(processed, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                calibrated_logits = logits / self.temperature
                processed = 1 / (1 + np.exp(-calibrated_logits))
            
            # 2. 분위수별 보정
            if hasattr(self, 'quantile_corrections'):
                for quantile in sorted(self.quantile_corrections.keys(), reverse=True):
                    correction = self.quantile_corrections[quantile]
                    threshold = correction['threshold']
                    factor = correction['correction_factor']
                    
                    high_mask = processed >= threshold
                    if high_mask.sum() > 0:
                        processed[high_mask] *= factor
            
            # 3. 10분위수별 보정
            if hasattr(self, 'decile_corrections'):
                for decile in self.decile_corrections.keys():
                    correction = self.decile_corrections[decile]
                    lower_bound = correction['lower_bound']
                    upper_bound = correction['upper_bound']
                    factor = correction['correction_factor']
                    
                    decile_mask = (processed >= lower_bound) & (processed < upper_bound)
                    if decile_mask.sum() > 0:
                        processed[decile_mask] *= factor
            
            # 4. 편향 보정
            processed = processed + self.bias_correction
            
            # 5. 최종 클리핑
            processed = np.clip(processed, 0.001, 0.999)
            
            return processed
            
        except Exception as e:
            logger.warning(f"고급 후처리 적용 실패: {e}")
            return np.clip(predictions, 0.001, 0.999)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """최적화된 앙상블 예측 - Combined Score 0.30+ 목표"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        # 가중 앙상블
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 고급 후처리 적용
        calibrated_pred = self._apply_postprocessing_advanced(ensemble_pred)
        
        return calibrated_pred

class CTRStabilizedEnsemble(BaseEnsemble):
    """CTR 예측 안정화 앙상블 - 1070만행 최적화"""
    
    def __init__(self, diversification_method: str = 'rank_weighted_advanced'):
        super().__init__("CTRStabilizedEnsemble")
        self.diversification_method = diversification_method
        self.model_weights = {}
        self.diversity_weights = {}
        self.final_weights = {}
        self.metrics_calculator = CTRMetrics()
        self.stability_factors = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """안정화 앙상블 학습 - 1070만행 최적화"""
        logger.info(f"CTR 안정화 앙상블 학습 시작 - 방법: {self.diversification_method}")
        
        available_models = list(base_predictions.keys())
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        self.training_data_size = len(X)
        
        # 1. 개별 모델 성능 평가 (안정성 고려)
        self.model_weights = self._evaluate_stability_performance(base_predictions, y)
        
        # 2. 고급 다양성 가중치 계산
        self.diversity_weights = self._calculate_advanced_diversity_weights(base_predictions)
        
        # 3. 안정성 요소 계산
        self.stability_factors = self._calculate_stability_factors(base_predictions, y)
        
        # 4. 최종 가중치 결합 (성능 + 다양성 + 안정성)
        self.final_weights = self._combine_weights_with_stability()
        
        self.is_fitted = True
        logger.info(f"CTR 안정화 앙상블 학습 완료 - 최종 가중치: {self.final_weights}")
    
    def _evaluate_stability_performance(self, base_predictions: Dict[str, np.ndarray], 
                                      y: pd.Series) -> Dict[str, float]:
        """안정성을 고려한 개별 모델 성능 평가"""
        
        performance_weights = {}
        
        # 대용량 데이터 부트스트래핑 (효율적)
        n_bootstrap = 10  # 1070만행에서는 부트스트래핑 수 제한
        sample_size = min(200000, len(y))
        
        for name, pred in base_predictions.items():
            try:
                bootstrap_scores = []
                
                for i in range(n_bootstrap):
                    # 부트스트래핑 샘플
                    sample_indices = np.random.choice(len(y), sample_size, replace=True)
                    y_sample = y.iloc[sample_indices]
                    pred_sample = pred[sample_indices]
                    
                    # Combined Score 계산
                    combined_score = self.metrics_calculator.combined_score(y_sample, pred_sample)
                    bootstrap_scores.append(combined_score)
                
                # 안정성 지표
                mean_score = np.mean(bootstrap_scores)
                std_score = np.std(bootstrap_scores)
                stability_score = mean_score / (1 + std_score)  # 안정성 보정
                
                performance_weights[name] = max(stability_score, 0.01)
                
                logger.info(f"{name} - 평균: {mean_score:.4f}, 표준편차: {std_score:.4f}, 안정성: {stability_score:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 안정성 성능 평가 실패: {e}")
                performance_weights[name] = 0.01
        
        return performance_weights
    
    def _calculate_advanced_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """고급 다양성 가중치 계산 - 1070만행 최적화"""
        
        model_names = list(base_predictions.keys())
        diversity_weights = {}
        
        # 대용량 데이터 샘플링
        sample_size = min(50000, len(list(base_predictions.values())[0]))
        sample_indices = np.random.choice(len(list(base_predictions.values())[0]), sample_size, replace=False)
        sampled_predictions = {name: pred[sample_indices] for name, pred in base_predictions.items()}
        
        if self.diversification_method == 'rank_weighted_advanced':
            # 고급 순위 기반 다양성
            for name in model_names:
                rank_differences = []
                prediction_distances = []
                distribution_differences = []
                
                for other in model_names:
                    if other != name:
                        # 순위 차이
                        rank_self = pd.Series(sampled_predictions[name]).rank(pct=True).values
                        rank_other = pd.Series(sampled_predictions[other]).rank(pct=True).values
                        rank_diff = np.mean(np.abs(rank_self - rank_other))
                        rank_differences.append(rank_diff)
                        
                        # 예측값 거리
                        pred_distance = np.mean(np.abs(sampled_predictions[name] - sampled_predictions[other]))
                        prediction_distances.append(pred_distance)
                        
                        # 분포 차이 (히스토그램 기반)
                        hist_self, _ = np.histogram(sampled_predictions[name], bins=20, range=(0, 1), density=True)
                        hist_other, _ = np.histogram(sampled_predictions[other], bins=20, range=(0, 1), density=True)
                        hist_diff = np.mean(np.abs(hist_self - hist_other))
                        distribution_differences.append(hist_diff)
                
                # 종합 다양성 점수
                rank_diversity = np.mean(rank_differences) if rank_differences else 0.5
                pred_diversity = np.mean(prediction_distances) if prediction_distances else 0.5
                dist_diversity = np.mean(distribution_differences) if distribution_differences else 0.5
                
                diversity_score = 0.4 * rank_diversity + 0.3 * pred_diversity + 0.3 * dist_diversity
                diversity_weights[name] = max(diversity_score, 0.1)
        
        elif self.diversification_method == 'correlation_matrix':
            # 고급 상관관계 행렬 분석
            correlation_matrix = self._calculate_correlation_matrix_advanced(sampled_predictions)
            
            for name in model_names:
                correlations = [abs(correlation_matrix[name][other]) 
                              for other in model_names if other != name]
                avg_correlation = np.mean(correlations) if correlations else 0.5
                diversity_score = 1.0 - avg_correlation
                diversity_weights[name] = max(diversity_score, 0.1)
        
        else:
            # 기본 다양성 (균등)
            diversity_weights = {name: 1.0 for name in model_names}
        
        logger.info(f"고급 다양성 가중치: {diversity_weights}")
        return diversity_weights
    
    def _calculate_correlation_matrix_advanced(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """고급 상관관계 행렬 계산"""
        model_names = list(predictions.keys())
        correlation_matrix = {}
        
        for name1 in model_names:
            correlation_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    correlation_matrix[name1][name2] = 1.0
                else:
                    try:
                        # Pearson 상관계수
                        pearson_corr = np.corrcoef(predictions[name1], predictions[name2])[0, 1]
                        if np.isnan(pearson_corr):
                            pearson_corr = 0.0
                        
                        # Spearman 상관계수 (순위 기반)
                        rank1 = pd.Series(predictions[name1]).rank()
                        rank2 = pd.Series(predictions[name2]).rank()
                        spearman_corr = np.corrcoef(rank1, rank2)[0, 1]
                        if np.isnan(spearman_corr):
                            spearman_corr = 0.0
                        
                        # 종합 상관계수
                        combined_corr = 0.6 * abs(pearson_corr) + 0.4 * abs(spearman_corr)
                        correlation_matrix[name1][name2] = combined_corr
                        
                    except Exception as e:
                        logger.warning(f"{name1}-{name2} 상관계수 계산 실패: {e}")
                        correlation_matrix[name1][name2] = 0.5
        
        return correlation_matrix
    
    def _calculate_stability_factors(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """안정성 요소 계산"""
        stability_factors = {}
        
        for name, pred in base_predictions.items():
            try:
                # 예측값 변동성
                pred_std = np.std(pred)
                pred_var_score = min(pred_std * 10, 1.0)  # 정규화
                
                # CTR 편향 안정성
                predicted_ctr = pred.mean()
                actual_ctr = y.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_stability = np.exp(-ctr_bias * 200)
                
                # 분포 안정성 (엔트로피 기반)
                p = np.clip(pred, 1e-15, 1 - 1e-15)
                entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                entropy_stability = entropy / np.log2(2)  # 정규화
                
                # 종합 안정성
                stability_score = 0.4 * ctr_stability + 0.3 * entropy_stability + 0.3 * pred_var_score
                stability_factors[name] = max(stability_score, 0.1)
                
            except Exception as e:
                logger.warning(f"{name} 안정성 계산 실패: {e}")
                stability_factors[name] = 0.5
        
        logger.info(f"안정성 요소: {stability_factors}")
        return stability_factors
    
    def _combine_weights_with_stability(self) -> Dict[str, float]:
        """성능, 다양성, 안정성 가중치 결합"""
        
        combined_weights = {}
        model_names = list(self.model_weights.keys())
        
        # 가중치 정규화
        performance_sum = sum(self.model_weights.values())
        diversity_sum = sum(self.diversity_weights.values())
        stability_sum = sum(self.stability_factors.values())
        
        if performance_sum > 0 and diversity_sum > 0 and stability_sum > 0:
            for name in model_names:
                perf_weight = self.model_weights[name] / performance_sum
                div_weight = self.diversity_weights[name] / diversity_sum
                stab_weight = self.stability_factors[name] / stability_sum
                
                # 성능 50%, 다양성 25%, 안정성 25% 비율로 결합
                combined_weight = 0.5 * perf_weight + 0.25 * div_weight + 0.25 * stab_weight
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
    """CTR 예측 메타 학습 앙상블 - 1070만행 최적화"""
    
    def __init__(self, meta_model_type: str = 'ridge_advanced', use_meta_features: bool = True):
        super().__init__("CTRMetaLearning")
        self.meta_model_type = meta_model_type
        self.use_meta_features = use_meta_features
        self.meta_model = None
        self.feature_scaler = None
        self.feature_selector = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Optional[Dict[str, np.ndarray]] = None):
        """메타 학습 앙상블 학습 - 1070만행 최적화"""
        logger.info(f"CTR 메타 학습 앙상블 학습 시작 - 메타모델: {self.meta_model_type}")
        
        # Out-of-fold 예측 생성 (효율적)
        oof_predictions = self._generate_oof_predictions_efficient(X, y)
        
        # 메타 피처 생성
        if self.use_meta_features:
            meta_features = self._create_meta_features_advanced(oof_predictions, X)
        else:
            meta_features = oof_predictions
        
        # 메타 모델 학습
        self._train_meta_model_advanced(meta_features, y)
        
        self.is_fitted = True
        logger.info("CTR 메타 학습 앙상블 학습 완료")
    
    def _generate_oof_predictions_efficient(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """효율적인 Out-of-fold 예측 생성 - 1070만행 최적화"""
        
        oof_predictions = pd.DataFrame(index=X.index)
        
        # 1070만행에서는 3폴드로 제한
        tscv = TimeSeriesSplit(n_splits=3, gap=1000)
        
        # 대용량 데이터에서는 샘플링 적용
        if len(X) > 2000000:
            sample_size = 1500000
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            logger.info(f"메타 학습을 위한 데이터 샘플링: {len(X):,} → {sample_size:,}")
        else:
            X_sample = X
            y_sample = y
        
        for model_name, model in self.base_models.items():
            oof_pred = np.zeros(len(X_sample))
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sample)):
                try:
                    X_train_fold = X_sample.iloc[train_idx]
                    X_val_fold = X_sample.iloc[val_idx]
                    y_train_fold = y_sample.iloc[train_idx]
                    
                    # 모델 복사 및 학습
                    fold_model = self._clone_model(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    # Out-of-fold 예측
                    oof_pred[val_idx] = fold_model.predict_proba(X_val_fold)
                    
                    logger.info(f"{model_name} 폴드 {fold + 1} 완료")
                    
                except Exception as e:
                    logger.error(f"{model_name} 폴드 {fold} 학습 실패: {str(e)}")
                    oof_pred[val_idx] = self.target_ctr
            
            # 전체 데이터에 대해 확장 (샘플링한 경우)
            if len(X) > len(X_sample):
                full_oof_pred = np.full(len(X), oof_pred.mean())
                full_oof_pred[sample_indices] = oof_pred
                oof_predictions[model_name] = full_oof_pred
            else:
                oof_predictions[model_name] = oof_pred
        
        return oof_predictions
    
    def _clone_model(self, model: BaseModel) -> BaseModel:
        """모델 복사"""
        from models import ModelFactory
        
        try:
            model_type = model.name.lower()
            if 'lightgbm' in model_type:
                model_type = 'lightgbm'
            elif 'xgboost' in model_type:
                model_type = 'xgboost'
            elif 'catboost' in model_type:
                model_type = 'catboost'
            elif 'deepctr' in model_type:
                return ModelFactory.create_model('deepctr', input_dim=100, params=model.params)
            else:
                model_type = 'logistic'
            
            return ModelFactory.create_model(model_type, params=model.params)
        except:
            return model
    
    def _create_meta_features_advanced(self, oof_predictions: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """고급 메타 피처 생성 - 1070만행 최적화"""
        
        meta_features = oof_predictions.copy()
        
        try:
            logger.info("고급 메타 피처 생성 시작")
            
            # 기본 통계 피처 (향상)
            meta_features['pred_mean'] = oof_predictions.mean(axis=1)
            meta_features['pred_std'] = oof_predictions.std(axis=1)
            meta_features['pred_min'] = oof_predictions.min(axis=1)
            meta_features['pred_max'] = oof_predictions.max(axis=1)
            meta_features['pred_median'] = oof_predictions.median(axis=1)
            meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
            meta_features['pred_iqr'] = oof_predictions.quantile(0.75, axis=1) - oof_predictions.quantile(0.25, axis=1)
            
            # 고급 통계 피처
            meta_features['pred_skew'] = oof_predictions.skew(axis=1)
            meta_features['pred_kurtosis'] = oof_predictions.kurtosis(axis=1)
            
            # 순위 기반 피처
            for col in oof_predictions.columns:
                meta_features[f'{col}_rank'] = oof_predictions[col].rank(pct=True)
            
            # 모델간 관계 피처 (선택적)
            model_cols = oof_predictions.columns.tolist()
            if len(model_cols) <= 5:  # 모델 수가 적을 때만
                for i, col1 in enumerate(model_cols):
                    for col2 in model_cols[i+1:]:
                        meta_features[f'{col1}_{col2}_diff'] = oof_predictions[col1] - oof_predictions[col2]
                        meta_features[f'{col1}_{col2}_ratio'] = oof_predictions[col1] / (oof_predictions[col2] + 1e-8)
                        meta_features[f'{col1}_{col2}_avg'] = (oof_predictions[col1] + oof_predictions[col2]) / 2
            
            # 신뢰도 및 일관성 피처
            meta_features['prediction_confidence'] = 1 - meta_features['pred_std']
            meta_features['consensus_strength'] = np.exp(-meta_features['pred_std'])
            meta_features['prediction_entropy'] = self._calculate_prediction_entropy(oof_predictions.values)
            
            # CTR 특화 피처
            meta_features['ctr_distance'] = np.abs(meta_features['pred_mean'] - self.target_ctr)
            meta_features['ctr_normalized'] = meta_features['pred_mean'] / self.target_ctr
            
            # 분위수 기반 피처
            quantiles = [0.1, 0.25, 0.75, 0.9]
            for q in quantiles:
                meta_features[f'pred_q{int(q*100)}'] = oof_predictions.quantile(q, axis=1)
            
            logger.info(f"고급 메타 피처 생성 완료: {meta_features.shape[1]}개 피처")
            
        except Exception as e:
            logger.warning(f"고급 메타 피처 생성 중 오류: {e}")
        
        return meta_features
    
    def _calculate_prediction_entropy(self, predictions: np.ndarray) -> np.ndarray:
        """예측 엔트로피 계산"""
        try:
            entropies = []
            for i in range(predictions.shape[0]):
                row_preds = predictions[i]
                p = np.clip(row_preds, 1e-15, 1 - 1e-15)
                entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                entropies.append(entropy)
            return np.array(entropies)
        except:
            return np.zeros(predictions.shape[0])
    
    def _train_meta_model_advanced(self, meta_features: pd.DataFrame, y: pd.Series):
        """고급 메타 모델 학습 - 1070만행 최적화"""
        
        # 피처 전처리
        meta_features_clean = meta_features.fillna(0)
        meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 피처 선택 (대용량 데이터에서 중요)
        if meta_features_clean.shape[1] > 50:
            from sklearn.feature_selection import SelectKBest, f_regression
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(50, meta_features_clean.shape[1]))
            meta_features_selected = self.feature_selector.fit_transform(meta_features_clean, y)
            logger.info(f"피처 선택: {meta_features_clean.shape[1]} → {meta_features_selected.shape[1]}")
        else:
            meta_features_selected = meta_features_clean.values
            self.feature_selector = None
        
        # 스케일링
        self.feature_scaler = StandardScaler()
        meta_features_scaled = self.feature_scaler.fit_transform(meta_features_selected)
        
        # 메타 모델 선택 및 학습
        if self.meta_model_type == 'ridge_advanced':
            self.meta_model = RidgeCV(
                alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 
                cv=3,
                scoring='neg_mean_squared_error'
            )
        elif self.meta_model_type == 'logistic_advanced':
            self.meta_model = LogisticRegression(
                random_state=42, 
                max_iter=2000,
                class_weight={0: 1, 1: 49.75},  # 실제 CTR 반영
                solver='lbfgs',
                C=0.1
            )
        elif self.meta_model_type == 'mlp_advanced':
            self.meta_model = MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                max_iter=2000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                alpha=0.01
            )
        else:
            self.meta_model = RidgeCV(cv=3)
        
        try:
            if 'logistic' in self.meta_model_type:
                self.meta_model.fit(meta_features_scaled, y)
            else:
                self.meta_model.fit(meta_features_scaled, y)
            
            # 성능 평가
            meta_pred = self.meta_model.predict(meta_features_scaled)
            if 'logistic' in self.meta_model_type:
                meta_pred = self.meta_model.predict_proba(meta_features_scaled)[:, 1]
            
            # Combined Score로 평가
            combined_score = self.metrics_calculator.combined_score(y, meta_pred)
            logger.info(f"메타 모델 Combined Score: {combined_score:.4f}")
            
            if combined_score >= 0.30:
                logger.info(f"🎯 메타 모델 목표 달성!")
            
        except Exception as e:
            logger.error(f"메타 모델 학습 실패: {str(e)}")
            raise
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """메타 학습 예측 - 1070만행 최적화"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        try:
            # 기본 예측을 DataFrame으로 변환
            pred_df = pd.DataFrame(base_predictions)
            
            # 메타 피처 생성
            if self.use_meta_features:
                meta_features = self._create_inference_meta_features_advanced(pred_df)
            else:
                meta_features = pred_df
            
            # 전처리
            meta_features_clean = meta_features.fillna(0)
            meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # 피처 선택
            if self.feature_selector is not None:
                meta_features_selected = self.feature_selector.transform(meta_features_clean)
            else:
                meta_features_selected = meta_features_clean.values
            
            # 스케일링
            meta_features_scaled = self.feature_scaler.transform(meta_features_selected)
            
            # 예측
            if 'logistic' in self.meta_model_type:
                ensemble_pred = self.meta_model.predict_proba(meta_features_scaled)[:, 1]
            else:
                ensemble_pred = self.meta_model.predict(meta_features_scaled)
                ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"메타 학습 예측 실패: {str(e)}")
            return np.mean(list(base_predictions.values()), axis=0)
    
    def _create_inference_meta_features_advanced(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """추론용 고급 메타 피처 생성"""
        
        meta_features = pred_df.copy()
        
        try:
            # 기본 통계 피처
            meta_features['pred_mean'] = pred_df.mean(axis=1)
            meta_features['pred_std'] = pred_df.std(axis=1)
            meta_features['pred_min'] = pred_df.min(axis=1)
            meta_features['pred_max'] = pred_df.max(axis=1)
            meta_features['pred_median'] = pred_df.median(axis=1)
            meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
            meta_features['pred_iqr'] = pred_df.quantile(0.75, axis=1) - pred_df.quantile(0.25, axis=1)
            
            # 고급 통계 피처
            meta_features['pred_skew'] = pred_df.skew(axis=1)
            meta_features['pred_kurtosis'] = pred_df.kurtosis(axis=1)
            
            # 순위 기반 피처
            for col in pred_df.columns:
                meta_features[f'{col}_rank'] = pred_df[col].rank(pct=True)
            
            # 모델간 관계 피처 (선택적)
            model_cols = pred_df.columns.tolist()
            if len(model_cols) <= 5:
                for i, col1 in enumerate(model_cols):
                    for col2 in model_cols[i+1:]:
                        meta_features[f'{col1}_{col2}_diff'] = pred_df[col1] - pred_df[col2]
                        meta_features[f'{col1}_{col2}_ratio'] = pred_df[col1] / (pred_df[col2] + 1e-8)
                        meta_features[f'{col1}_{col2}_avg'] = (pred_df[col1] + pred_df[col2]) / 2
            
            # 신뢰도 피처
            meta_features['prediction_confidence'] = 1 - meta_features['pred_std']
            meta_features['consensus_strength'] = np.exp(-meta_features['pred_std'])
            meta_features['prediction_entropy'] = self._calculate_prediction_entropy(pred_df.values)
            
            # CTR 특화 피처
            meta_features['ctr_distance'] = np.abs(meta_features['pred_mean'] - self.target_ctr)
            meta_features['ctr_normalized'] = meta_features['pred_mean'] / self.target_ctr
            
            # 분위수 기반 피처
            quantiles = [0.1, 0.25, 0.75, 0.9]
            for q in quantiles:
                meta_features[f'pred_q{int(q*100)}'] = pred_df.quantile(q, axis=1)
            
        except Exception as e:
            logger.warning(f"추론용 고급 메타 피처 생성 중 오류: {e}")
        
        return meta_features

class CTREnsembleManager:
    """CTR 특화 앙상블 관리 클래스 - Combined Score 0.30+ 목표"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.calibrated_ensemble = None
        self.optimal_ensemble = None
        self.target_score = 0.30
        
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        logger.info(f"기본 모델 추가: {name}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """CTR 특화 앙상블 생성 - Combined Score 0.30+ 목표"""
        
        if ensemble_type == 'optimal':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            optimization_method = kwargs.get('optimization_method', 'combined_plus')
            ensemble = CTROptimalEnsemble(target_ctr, optimization_method)
            self.optimal_ensemble = ensemble
        
        elif ensemble_type == 'stabilized':
            diversification_method = kwargs.get('diversification_method', 'rank_weighted_advanced')
            ensemble = CTRStabilizedEnsemble(diversification_method)
        
        elif ensemble_type == 'meta':
            meta_model_type = kwargs.get('meta_model_type', 'ridge_advanced')
            use_meta_features = kwargs.get('use_meta_features', True)
            ensemble = CTRMetaLearning(meta_model_type, use_meta_features)
        
        elif ensemble_type == 'weighted':
            weights = kwargs.get('weights', None)
            ensemble = CTRWeightedBlending(weights)
        
        elif ensemble_type == 'calibrated':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            calibration_method = kwargs.get('calibration_method', 'platt')
            ensemble = CTRCalibratedEnsemble(target_ctr, calibration_method)
            self.calibrated_ensemble = ensemble
        
        else:
            raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
        
        # 기본 모델 추가
        for name, model in self.base_models.items():
            ensemble.add_base_model(name, model)
        
        self.ensembles[ensemble_type] = ensemble
        logger.info(f"앙상블 생성: {ensemble_type} (Combined Score 0.30+ 목표)")
        
        return ensemble
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """모든 앙상블 학습 - Combined Score 0.30+ 목표"""
        logger.info("모든 앙상블 학습 시작 (Combined Score 0.30+ 목표)")
        
        # 기본 모델 예측 수집 (대용량 최적화)
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                logger.info(f"{name} 모델 예측 수집 시작")
                pred = model.predict_proba(X)
                base_predictions[name] = pred
                
                # 개별 모델 성능 체크
                combined_score = self.metrics_calculator.combined_score(y, pred)
                logger.info(f"{name} 개별 성능 - Combined Score: {combined_score:.4f}")
                
                if combined_score >= 0.30:
                    logger.info(f"🎯 {name} 개별 모델이 목표 달성!")
                
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
        
        # 각 앙상블 학습
        successful_ensembles = 0
        target_achieved_ensembles = []
        
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                logger.info(f"{ensemble_type} 앙상블 학습 시작")
                start_time = time.time()
                
                ensemble.fit(X, y, base_predictions)
                
                training_time = time.time() - start_time
                successful_ensembles += 1
                
                logger.info(f"{ensemble_type} 앙상블 학습 완료 (소요시간: {training_time:.2f}초)")
                
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 학습 실패: {str(e)}")
        
        logger.info(f"앙상블 학습 완료 - 성공: {successful_ensembles}/{len(self.ensembles)}")
        
        # 메모리 정리
        gc.collect()
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """앙상블 성능 평가 - Combined Score 0.30+ 목표"""
        logger.info("앙상블 성능 평가 시작 (Combined Score 0.30+ 목표)")
        
        results = {}
        target_achieved_count = 0
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                base_predictions[name] = pred
                
                score = self.metrics_calculator.combined_score(y_val, pred)
                results[f"base_{name}"] = score
                
                if score >= self.target_score:
                    target_achieved_count += 1
                    logger.info(f"🎯 {name} 기본 모델 목표 달성: {score:.4f}")
                else:
                    logger.info(f"{name} 기본 모델: {score:.4f}")
                
            except Exception as e:
                logger.error(f"{name} 모델 검증 예측 실패: {str(e)}")
                results[f"base_{name}"] = 0.0
        
        # 앙상블 성능 평가
        best_ensemble_score = 0.0
        best_ensemble_name = None
        
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    ensemble_pred = ensemble.predict_proba(base_predictions)
                    score = self.metrics_calculator.combined_score(y_val, ensemble_pred)
                    results[f"ensemble_{ensemble_type}"] = score
                    
                    # CTR 분석
                    predicted_ctr = ensemble_pred.mean()
                    actual_ctr = y_val.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    
                    if score >= self.target_score:
                        target_achieved_count += 1
                        logger.info(f"🎯 {ensemble_type} 앙상블 목표 달성: {score:.4f} (CTR 편향: {ctr_bias:.4f})")
                    else:
                        logger.info(f"{ensemble_type} 앙상블: {score:.4f} (CTR 편향: {ctr_bias:.4f})")
                    
                    if score > best_ensemble_score:
                        best_ensemble_score = score
                        best_ensemble_name = ensemble_type
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 평가 실패: {str(e)}")
                    results[f"ensemble_{ensemble_type}"] = 0.0
        
        # 최고 성능 모델/앙상블 선택
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
            
            # 목표 달성 요약
            total_models = len(self.base_models) + len([e for e in self.ensembles.values() if e.is_fitted])
            logger.info(f"Combined Score 0.30+ 달성: {target_achieved_count}/{total_models}")
            
            if best_score >= self.target_score:
                logger.info(f"🎯 최종 목표 달성! 최고 점수: {best_score:.4f}")
            else:
                logger.warning(f"⚠️ 최종 목표 미달성. 최고 점수: {best_score:.4f}")
        
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
                logger.info(f"최고 성능 기본 모델 사용: {best_model_name} (Score: {best_score:.4f})")
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
            'target_score': self.target_score,
            'target_achieved': any(score >= self.target_score for score in self.ensemble_results.values()),
            'calibrated_available': self.calibrated_ensemble is not None,
            'optimal_available': self.optimal_ensemble is not None,
            'training_data_size': getattr(self.best_ensemble, 'training_data_size', 0) if self.best_ensemble else 0
        }
        
        try:
            import json
            info_path = output_dir / "best_ensemble_info.json"
            with open(info_path, 'w') as f:
                json.dump(best_info, f, indent=2, default=str)
            logger.info(f"앙상블 정보 저장: {info_path}")
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
                self.target_score = best_info.get('target_score', 0.30)
                
            except Exception as e:
                logger.error(f"앙상블 정보 로딩 실패: {str(e)}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """앙상블 요약 정보 - Combined Score 0.30+ 목표"""
        
        target_achieved_count = sum(1 for score in self.ensemble_results.values() if score >= self.target_score)
        best_score = max(self.ensemble_results.values()) if self.ensemble_results else 0.0
        
        return {
            'total_ensembles': len(self.ensembles),
            'fitted_ensembles': sum(1 for e in self.ensembles.values() if e.is_fitted),
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results,
            'base_models_count': len(self.base_models),
            'calibrated_ensemble_available': self.calibrated_ensemble is not None and self.calibrated_ensemble.is_fitted,
            'optimal_ensemble_available': self.optimal_ensemble is not None and self.optimal_ensemble.is_fitted,
            'ensemble_types': list(self.ensembles.keys()),
            'target_score': self.target_score,
            'target_achieved_count': target_achieved_count,
            'target_achieved': target_achieved_count > 0,
            'best_score': best_score,
            'goal_reached': best_score >= self.target_score
        }

# 기존 앙상블 클래스들 (하위 호환성)
class CTRCalibratedEnsemble(BaseEnsemble):
    """CTR 보정 앙상블 클래스 - 1070만행 최적화"""
    
    def __init__(self, target_ctr: float = 0.0201, calibration_method: str = 'platt'):
        super().__init__("CTRCalibratedEnsemble")
        self.target_ctr = target_ctr
        self.calibration_method = calibration_method
        self.weights = {}
        self.calibrator = None
        self.bias_correction = 0.0
        self.metrics_calculator = CTRMetrics()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """보정된 앙상블 학습 - Combined Score 0.30+ 목표"""
        logger.info("CTR 보정 앙상블 학습 시작")
        
        available_models = list(base_predictions.keys())
        
        if len(available_models) < 2:
            if available_models:
                self.weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        self.training_data_size = len(X)
        
        # Combined Score 기준 가중치 최적화
        self.weights = self._optimize_weights_for_combined_score_advanced(base_predictions, y)
        
        # 가중 앙상블 생성
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # CTR 보정 적용
        self._apply_ctr_calibration_advanced(ensemble_pred, y)
        
        self.is_fitted = True
        
        # 성능 검증
        final_score = self.metrics_calculator.combined_score(y, self.predict_proba(base_predictions))
        if final_score >= 0.30:
            logger.info(f"🎯 CTR 보정 앙상블 목표 달성: {final_score:.4f}")
        else:
            logger.warning(f"CTR 보정 앙상블 학습 완료: {final_score:.4f}")
    
    def _optimize_weights_for_combined_score_advanced(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Combined Score 최적화를 위한 고급 가중치 튜닝"""
        
        model_names = list(base_predictions.keys())
        
        if not OPTUNA_AVAILABLE:
            # 성능 기반 가중치
            weights = {}
            for name, pred in base_predictions.items():
                score = self.metrics_calculator.combined_score(y, pred)
                weights[name] = max(score, 0.01)
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            return weights
        
        def advanced_objective(trial):
            weights = {}
            for name in model_names:
                weights[name] = trial.suggest_float(f'weight_{name}', 0.01, 0.99)
            
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            weights = {k: v/total_weight for k, v in weights.items()}
            
            ensemble_pred = np.zeros(len(y))
            for name, weight in weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
            
            score = self.metrics_calculator.combined_score(y, ensemble_pred)
            
            # Combined Score 0.30+ 목표에 대한 보너스
            if score >= 0.30:
                bonus = (score - 0.30) * 20
                return score + bonus
            
            return score
        
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42, n_startup_trials=15),
                pruner=MedianPruner(n_startup_trials=10)
            )
            study.optimize(advanced_objective, n_trials=150, show_progress_bar=False)
            
            optimized_weights = {}
            for param_name, weight in study.best_params.items():
                model_name = param_name.replace('weight_', '')
                optimized_weights[model_name] = weight
            
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
            
            logger.info(f"고급 가중치 최적화 완료 - 점수: {study.best_value:.4f}")
            return optimized_weights
            
        except Exception as e:
            logger.error(f"고급