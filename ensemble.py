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

from sklearn.linear_model import LogisticRegression, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    """앙상블 모델 기본 클래스 - Combined Score 0.30+ 목표"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.target_combined_score = 0.30
        
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
    
    def _enhance_ensemble_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """앙상블 예측 다양성 향상"""
        try:
            unique_count = len(np.unique(predictions))
            
            if unique_count < len(predictions) // 100:
                logger.info(f"{self.name}: 앙상블 예측 다양성 향상 적용")
                
                noise_scale = max(predictions.std() * 0.003, 1e-7)
                noise = np.random.normal(0, noise_scale, len(predictions))
                
                predictions = predictions + noise
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"앙상블 다양성 향상 실패: {e}")
            return predictions

class CTRSuperOptimalEnsemble(BaseEnsemble):
    """CTR 예측 슈퍼 최적화 앙상블 - Combined Score 0.32+ 목표"""
    
    def __init__(self, target_ctr: float = 0.0201, optimization_method: str = 'ultra_combined'):
        super().__init__("CTRSuperOptimalEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.final_weights = {}
        self.advanced_calibrator = None
        self.metrics_calculator = CTRMetrics()
        self.temperature = 1.0
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
        self.quantile_corrections = {}
        self.performance_boosters = {}
        self.target_combined_score = 0.32
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """슈퍼 최적화 앙상블 학습 - Combined Score 0.32+ 목표"""
        logger.info(f"CTR 슈퍼 최적화 앙상블 학습 시작 - 목표: Combined Score 0.32+")
        
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models}")
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # 1단계: 다층 가중치 최적화
        logger.info("1단계: 다층 가중치 최적화")
        self.final_weights = self._multi_layer_weight_optimization(base_predictions, y)
        
        # 2단계: 가중 앙상블 생성
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 3단계: 슈퍼 CTR 특화 후처리 최적화
        logger.info("3단계: 슈퍼 CTR 특화 후처리 최적화")
        self._apply_super_ctr_postprocessing(ensemble_pred, y)
        
        # 4단계: 성능 부스터 적용
        logger.info("4단계: 성능 부스터 적용")
        self._apply_performance_boosters(ensemble_pred, y)
        
        # 5단계: 최종 검증 및 조정
        logger.info("5단계: 최종 검증 및 조정")
        final_pred = self._apply_all_corrections(ensemble_pred)
        final_score = self.metrics_calculator.combined_score(y, final_pred)
        
        logger.info(f"슈퍼 최적화 앙상블 최종 Combined Score: {final_score:.4f}")
        
        self.is_fitted = True
        logger.info("CTR 슈퍼 최적화 앙상블 학습 완료")
    
    def _multi_layer_weight_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """다층 가중치 최적화 - Combined Score 극대화"""
        
        model_names = list(base_predictions.keys())
        
        # Layer 1: 개별 모델 성능 평가
        individual_scores = {}
        ctr_alignment_scores = {}
        diversity_scores = {}
        
        for name, pred in base_predictions.items():
            combined_score = self.metrics_calculator.combined_score(y, pred)
            ap_score = self.metrics_calculator.average_precision(y, pred)
            wll_score = self.metrics_calculator.weighted_log_loss(y, pred)
            
            predicted_ctr = pred.mean()
            actual_ctr = y.mean()
            ctr_alignment = np.exp(-abs(predicted_ctr - actual_ctr) * 500)
            
            pred_std = pred.std()
            pred_entropy = -np.mean(pred * np.log(pred + 1e-15) + (1 - pred) * np.log(1 - pred + 1e-15))
            diversity = pred_std * pred_entropy
            
            individual_scores[name] = combined_score
            ctr_alignment_scores[name] = ctr_alignment
            diversity_scores[name] = diversity
        
        # Layer 2: 상호 보완성 분석
        complementarity_matrix = {}
        for i, name1 in enumerate(model_names):
            complementarity_matrix[name1] = {}
            for j, name2 in enumerate(model_names):
                if i != j:
                    pred1, pred2 = base_predictions[name1], base_predictions[name2]
                    
                    corr = np.corrcoef(pred1, pred2)[0, 1]
                    rank_corr = np.corrcoef(pd.Series(pred1).rank(), pd.Series(pred2).rank())[0, 1]
                    
                    diff_contribution = self._calculate_difference_contribution(pred1, pred2, y)
                    
                    complementarity = (1 - abs(corr)) * (1 - abs(rank_corr)) * diff_contribution
                    complementarity_matrix[name1][name2] = complementarity
                else:
                    complementarity_matrix[name1][name2] = 0.0
        
        # Layer 3: Optuna 기반 고급 최적화
        if OPTUNA_AVAILABLE:
            try:
                optimized_weights = self._optuna_ultra_optimization(base_predictions, y, individual_scores)
            except Exception as e:
                logger.warning(f"Optuna 초고도 최적화 실패: {e}")
                optimized_weights = self._fallback_optimization(individual_scores, ctr_alignment_scores, diversity_scores)
        else:
            optimized_weights = self._fallback_optimization(individual_scores, ctr_alignment_scores, diversity_scores)
        
        logger.info(f"다층 최적화 가중치: {optimized_weights}")
        return optimized_weights
    
    def _calculate_difference_contribution(self, pred1: np.ndarray, pred2: np.ndarray, y: pd.Series) -> float:
        """차이 기여도 계산"""
        try:
            avg_pred = (pred1 + pred2) / 2
            avg_score = self.metrics_calculator.combined_score(y, avg_pred)
            
            individual_score1 = self.metrics_calculator.combined_score(y, pred1)
            individual_score2 = self.metrics_calculator.combined_score(y, pred2)
            
            max_individual = max(individual_score1, individual_score2)
            
            if max_individual > 0:
                improvement_ratio = avg_score / max_individual
                return min(improvement_ratio, 2.0)
            else:
                return 1.0
        except:
            return 1.0
    
    def _optuna_ultra_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series, 
                                 individual_scores: Dict[str, float]) -> Dict[str, float]:
        """Optuna 초고도 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        
        def ultra_objective(trial):
            weights = {}
            
            # Dynamic weight bounds based on individual performance
            for name in model_names:
                base_performance = individual_scores.get(name, 0.1)
                
                if base_performance > 0.25:
                    min_weight, max_weight = 0.1, 0.8
                elif base_performance > 0.20:
                    min_weight, max_weight = 0.05, 0.6
                else:
                    min_weight, max_weight = 0.01, 0.4
                
                weights[name] = trial.suggest_float(f'weight_{name}', min_weight, max_weight)
            
            # Advanced ensemble techniques
            ensemble_method = trial.suggest_categorical('ensemble_method', ['weighted', 'power_weighted', 'rank_weighted'])
            temperature = trial.suggest_float('temperature', 0.8, 2.0)
            
            # Create ensemble prediction
            if ensemble_method == 'weighted':
                ensemble_pred = np.zeros(len(y))
                total_weight = sum(weights.values())
                for name, weight in weights.items():
                    if name in base_predictions:
                        ensemble_pred += (weight / total_weight) * base_predictions[name]
            
            elif ensemble_method == 'power_weighted':
                power = trial.suggest_float('power', 1.0, 3.0)
                ensemble_pred = np.zeros(len(y))
                total_weight = 0
                for name, weight in weights.items():
                    if name in base_predictions:
                        powered_pred = np.power(base_predictions[name], power)
                        ensemble_pred += weight * powered_pred
                        total_weight += weight
                if total_weight > 0:
                    ensemble_pred /= total_weight
                ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
            else:  # rank_weighted
                ensemble_pred = np.zeros(len(y))
                total_weight = 0
                for name, weight in weights.items():
                    if name in base_predictions:
                        rank_pred = pd.Series(base_predictions[name]).rank(pct=True).values
                        ensemble_pred += weight * rank_pred
                        total_weight += weight
                if total_weight > 0:
                    ensemble_pred /= total_weight
            
            # Apply temperature scaling
            if ensemble_method != 'rank_weighted':
                try:
                    logits = np.log(np.clip(ensemble_pred, 1e-15, 1-1e-15) / (1 - np.clip(ensemble_pred, 1e-15, 1-1e-15)))
                    scaled_logits = logits / temperature
                    ensemble_pred = 1 / (1 + np.exp(-scaled_logits))
                except:
                    pass
            
            ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
            # Multi-objective optimization
            combined_score = self.metrics_calculator.combined_score(y, ensemble_pred)
            
            # CTR alignment bonus
            predicted_ctr = ensemble_pred.mean()
            actual_ctr = y.mean()
            ctr_alignment = np.exp(-abs(predicted_ctr - actual_ctr) * 300)
            
            # Diversity bonus
            diversity = ensemble_pred.std()
            
            # Final score
            final_score = combined_score * (1 + 0.15 * ctr_alignment) * (1 + 0.1 * diversity)
            
            return final_score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=20),
            pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=10)
        )
        
        study.optimize(ultra_objective, n_trials=150, show_progress_bar=False)
        
        # Extract optimized weights
        optimized_weights = {}
        for param_name, weight in study.best_params.items():
            if param_name.startswith('weight_'):
                model_name = param_name.replace('weight_', '')
                optimized_weights[model_name] = weight
        
        # Store additional optimization parameters
        self.ensemble_method = study.best_params.get('ensemble_method', 'weighted')
        self.temperature = study.best_params.get('temperature', 1.0)
        
        # Normalize weights
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
        
        logger.info(f"Optuna 초고도 최적화 완료 - 최고 점수: {study.best_value:.4f}")
        logger.info(f"최적화 방법: {self.ensemble_method}, Temperature: {self.temperature:.3f}")
        
        return optimized_weights
    
    def _fallback_optimization(self, individual_scores: Dict[str, float], 
                              ctr_alignment_scores: Dict[str, float],
                              diversity_scores: Dict[str, float]) -> Dict[str, float]:
        """대체 최적화 방법"""
        
        weights = {}
        for name in individual_scores.keys():
            performance_weight = individual_scores.get(name, 0.1)
            alignment_weight = ctr_alignment_scores.get(name, 0.5)
            diversity_weight = diversity_scores.get(name, 0.5)
            
            combined_weight = (0.5 * performance_weight + 
                             0.3 * alignment_weight + 
                             0.2 * diversity_weight)
            
            weights[name] = max(combined_weight, 0.01)
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _apply_super_ctr_postprocessing(self, predictions: np.ndarray, y: pd.Series):
        """슈퍼 CTR 특화 후처리 최적화"""
        logger.info("슈퍼 CTR 후처리 최적화 시작")
        
        try:
            # 1. Advanced Bias Correction
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            
            if predicted_ctr > 0:
                self.multiplicative_correction = actual_ctr / predicted_ctr
            else:
                self.multiplicative_correction = 1.0
            
            # 2. Advanced Temperature Scaling with Quantile Awareness
            self._fit_advanced_temperature_scaling(predictions, y)
            
            # 3. Quantile-based Corrections
            self._fit_quantile_corrections(predictions, y)
            
            # 4. Distribution Matching
            self._fit_distribution_matching(predictions, y)
            
            logger.info(f"슈퍼 CTR 후처리 완료")
            logger.info(f"편향 보정: {self.bias_correction:.4f}, 승수 보정: {self.multiplicative_correction:.4f}")
            logger.info(f"고급 Temperature: {self.temperature:.3f}")
            
        except Exception as e:
            logger.error(f"슈퍼 CTR 후처리 최적화 실패: {e}")
            self.bias_correction = 0.0
            self.multiplicative_correction = 1.0
            self.temperature = 1.0
    
    def _fit_advanced_temperature_scaling(self, predictions: np.ndarray, y: pd.Series):
        """고급 Temperature Scaling"""
        try:
            from scipy.optimize import minimize
            
            def advanced_temperature_loss(params):
                temp, shift = params
                if temp <= 0:
                    return float('inf')
                
                pred_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                
                adjusted_logits = (logits + shift) / temp
                calibrated_probs = 1 / (1 + np.exp(-adjusted_logits))
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                # Multi-objective loss
                log_loss = -np.mean(y * np.log(calibrated_probs) + (1 - y) * np.log(1 - calibrated_probs))
                
                # CTR alignment loss
                ctr_loss = abs(calibrated_probs.mean() - y.mean()) * 1000
                
                # Diversity preservation
                diversity_loss = -calibrated_probs.std()
                
                return log_loss + ctr_loss + diversity_loss
            
            result = minimize(
                advanced_temperature_loss, 
                x0=[1.0, 0.0], 
                bounds=[(0.1, 10.0), (-2.0, 2.0)],
                method='L-BFGS-B'
            )
            
            self.temperature = result.x[0]
            self.logit_shift = result.x[1]
            
        except Exception as e:
            logger.warning(f"고급 Temperature scaling 실패: {e}")
            self.temperature = 1.0
            self.logit_shift = 0.0
    
    def _fit_quantile_corrections(self, predictions: np.ndarray, y: pd.Series):
        """분위수 기반 보정"""
        try:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            
            for q in quantiles:
                threshold = np.percentile(predictions, q * 100)
                mask = predictions >= threshold
                
                if mask.sum() > 10:
                    actual_rate = y[mask].mean()
                    predicted_rate = predictions[mask].mean()
                    
                    if predicted_rate > 0:
                        correction_factor = actual_rate / predicted_rate
                    else:
                        correction_factor = 1.0
                    
                    self.quantile_corrections[q] = {
                        'threshold': threshold,
                        'correction_factor': correction_factor,
                        'sample_size': mask.sum()
                    }
            
            logger.info(f"분위수 보정 완료: {len(self.quantile_corrections)}개 구간")
            
        except Exception as e:
            logger.warning(f"분위수 보정 실패: {e}")
    
    def _fit_distribution_matching(self, predictions: np.ndarray, y: pd.Series):
        """분포 매칭"""
        try:
            # Empirical CDF matching
            pred_sorted_idx = np.argsort(predictions)
            y_sorted = y.iloc[pred_sorted_idx]
            
            # Calculate target distribution
            window_size = max(len(predictions) // 50, 100)
            self.distribution_mapping = {}
            
            for i in range(0, len(predictions) - window_size, window_size // 2):
                window_pred = predictions[pred_sorted_idx[i:i+window_size]]
                window_actual = y_sorted.iloc[i:i+window_size]
                
                pred_center = window_pred.mean()
                actual_rate = window_actual.mean()
                
                self.distribution_mapping[pred_center] = actual_rate
            
            logger.info(f"분포 매칭 완료: {len(self.distribution_mapping)}개 매핑 포인트")
            
        except Exception as e:
            logger.warning(f"분포 매칭 실패: {e}")
            self.distribution_mapping = {}
    
    def _apply_performance_boosters(self, predictions: np.ndarray, y: pd.Series):
        """성능 부스터 적용"""
        try:
            # Booster 1: High-confidence region enhancement
            high_conf_threshold = np.percentile(predictions, 95)
            high_conf_mask = predictions >= high_conf_threshold
            
            if high_conf_mask.sum() > 0:
                actual_high_conf_rate = y[high_conf_mask].mean()
                predicted_high_conf_rate = predictions[high_conf_mask].mean()
                
                if predicted_high_conf_rate > 0:
                    high_conf_booster = actual_high_conf_rate / predicted_high_conf_rate
                else:
                    high_conf_booster = 1.0
                
                self.performance_boosters['high_confidence'] = {
                    'threshold': high_conf_threshold,
                    'booster': high_conf_booster
                }
            
            # Booster 2: Low-confidence region enhancement  
            low_conf_threshold = np.percentile(predictions, 5)
            low_conf_mask = predictions <= low_conf_threshold
            
            if low_conf_mask.sum() > 0:
                actual_low_conf_rate = y[low_conf_mask].mean()
                predicted_low_conf_rate = predictions[low_conf_mask].mean()
                
                if predicted_low_conf_rate > 0:
                    low_conf_booster = actual_low_conf_rate / predicted_low_conf_rate
                else:
                    low_conf_booster = 1.0
                
                self.performance_boosters['low_confidence'] = {
                    'threshold': low_conf_threshold,
                    'booster': low_conf_booster
                }
            
            # Booster 3: Median region stabilization
            median_low, median_high = np.percentile(predictions, [40, 60])
            median_mask = (predictions >= median_low) & (predictions <= median_high)
            
            if median_mask.sum() > 0:
                actual_median_rate = y[median_mask].mean()
                predicted_median_rate = predictions[median_mask].mean()
                
                if predicted_median_rate > 0:
                    median_booster = actual_median_rate / predicted_median_rate
                else:
                    median_booster = 1.0
                
                self.performance_boosters['median_stability'] = {
                    'low_threshold': median_low,
                    'high_threshold': median_high,
                    'booster': median_booster
                }
            
            logger.info(f"성능 부스터 적용 완료: {len(self.performance_boosters)}개 부스터")
            
        except Exception as e:
            logger.warning(f"성능 부스터 적용 실패: {e}")
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """고급 가중 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        if hasattr(self, 'ensemble_method') and self.ensemble_method == 'power_weighted':
            power = getattr(self, 'power', 2.0)
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    powered_pred = np.power(base_predictions[name], power)
                    ensemble_pred += weight * powered_pred
            ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        elif hasattr(self, 'ensemble_method') and self.ensemble_method == 'rank_weighted':
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    rank_pred = pd.Series(base_predictions[name]).rank(pct=True).values
                    ensemble_pred += weight * rank_pred
        else:
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred
    
    def _apply_all_corrections(self, predictions: np.ndarray) -> np.ndarray:
        """모든 보정 기법 적용"""
        try:
            corrected = predictions.copy()
            
            # 1. Temperature scaling with shift
            if hasattr(self, 'temperature') and hasattr(self, 'logit_shift'):
                try:
                    pred_clipped = np.clip(corrected, 1e-15, 1 - 1e-15)
                    logits = np.log(pred_clipped / (1 - pred_clipped))
                    adjusted_logits = (logits + self.logit_shift) / self.temperature
                    corrected = 1 / (1 + np.exp(-adjusted_logits))
                except:
                    pass
            
            # 2. Performance boosters
            for booster_name, booster_config in self.performance_boosters.items():
                try:
                    if booster_name == 'high_confidence':
                        mask = corrected >= booster_config['threshold']
                        corrected[mask] *= booster_config['booster']
                    elif booster_name == 'low_confidence':
                        mask = corrected <= booster_config['threshold']
                        corrected[mask] *= booster_config['booster']
                    elif booster_name == 'median_stability':
                        mask = ((corrected >= booster_config['low_threshold']) & 
                               (corrected <= booster_config['high_threshold']))
                        corrected[mask] *= booster_config['booster']
                except:
                    continue
            
            # 3. Quantile corrections
            for q, correction in self.quantile_corrections.items():
                try:
                    mask = corrected >= correction['threshold']
                    if mask.sum() > 0:
                        corrected[mask] *= correction['correction_factor']
                except:
                    continue
            
            # 4. Distribution mapping
            if hasattr(self, 'distribution_mapping') and self.distribution_mapping:
                try:
                    corrected = self._apply_distribution_mapping(corrected)
                except:
                    pass
            
            # 5. Final bias corrections
            corrected = corrected * self.multiplicative_correction + self.bias_correction
            
            # 6. Final clipping and diversity enhancement
            corrected = np.clip(corrected, 1e-15, 1 - 1e-15)
            corrected = self._enhance_ensemble_diversity(corrected)
            
            return corrected
            
        except Exception as e:
            logger.warning(f"전체 보정 적용 실패: {e}")
            return np.clip(predictions, 1e-15, 1 - 1e-15)
    
    def _apply_distribution_mapping(self, predictions: np.ndarray) -> np.ndarray:
        """분포 매핑 적용"""
        try:
            mapped_predictions = predictions.copy()
            
            mapping_points = sorted(self.distribution_mapping.keys())
            
            for i, pred_val in enumerate(predictions):
                # Find closest mapping points
                closest_idx = np.argmin([abs(pred_val - mp) for mp in mapping_points])
                closest_point = mapping_points[closest_idx]
                
                # Apply mapping with interpolation
                if closest_idx > 0 and closest_idx < len(mapping_points) - 1:
                    # Linear interpolation
                    left_point = mapping_points[closest_idx - 1]
                    right_point = mapping_points[closest_idx + 1]
                    
                    if right_point != left_point:
                        weight = (pred_val - left_point) / (right_point - left_point)
                        left_target = self.distribution_mapping[left_point]
                        right_target = self.distribution_mapping[right_point]
                        
                        mapped_val = left_target + weight * (right_target - left_target)
                        mapped_predictions[i] = mapped_val
                else:
                    mapped_predictions[i] = self.distribution_mapping[closest_point]
            
            return mapped_predictions
            
        except Exception as e:
            logger.warning(f"분포 매핑 적용 실패: {e}")
            return predictions
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """슈퍼 최적화된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        # 가중 앙상블 생성
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 모든 보정 기법 적용
        final_pred = self._apply_all_corrections(ensemble_pred)
        
        return final_pred

class CTRAdvancedStabilizedEnsemble(BaseEnsemble):
    """CTR 예측 고급 안정화 앙상블"""
    
    def __init__(self, diversification_method: str = 'advanced_rank_weighted'):
        super().__init__("CTRAdvancedStabilizedEnsemble")
        self.diversification_method = diversification_method
        self.model_weights = {}
        self.diversity_weights = {}
        self.stability_weights = {}
        self.final_weights = {}
        self.metrics_calculator = CTRMetrics()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """고급 안정화 앙상블 학습"""
        logger.info(f"CTR 고급 안정화 앙상블 학습 시작 - 방법: {self.diversification_method}")
        
        available_models = list(base_predictions.keys())
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        # 1. 개별 모델 성능 평가
        self.model_weights = self._evaluate_individual_performance_advanced(base_predictions, y)
        
        # 2. 다양성 가중치 계산
        self.diversity_weights = self._calculate_diversity_weights_advanced(base_predictions)
        
        # 3. 안정성 가중치 계산
        self.stability_weights = self._calculate_stability_weights(base_predictions, y)
        
        # 4. 최종 가중치 결합
        self.final_weights = self._combine_weights_advanced()
        
        self.is_fitted = True
        logger.info(f"CTR 고급 안정화 앙상블 학습 완료 - 최종 가중치: {self.final_weights}")
    
    def _evaluate_individual_performance_advanced(self, base_predictions: Dict[str, np.ndarray], 
                                                y: pd.Series) -> Dict[str, float]:
        """고급 개별 모델 성능 평가"""
        
        performance_weights = {}
        
        for name, pred in base_predictions.items():
            try:
                # Multi-metric performance evaluation
                combined_score = self.metrics_calculator.combined_score(y, pred)
                ap_score = self.metrics_calculator.average_precision(y, pred)
                wll_score = self.metrics_calculator.weighted_log_loss(y, pred)
                
                # CTR alignment with exponential penalty
                predicted_ctr = pred.mean()
                actual_ctr = y.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_penalty = np.exp(-ctr_bias * 300)
                
                # Prediction quality metrics
                pred_std = pred.std()
                pred_range = pred.max() - pred.min()
                pred_entropy = -np.mean(pred * np.log(pred + 1e-15) + (1 - pred) * np.log(1 - pred + 1e-15))
                
                quality_score = pred_std * pred_range * pred_entropy
                
                # Final performance score
                performance_score = (0.5 * combined_score + 
                                   0.2 * ctr_penalty + 
                                   0.2 * (ap_score * (1 / (1 + wll_score))) + 
                                   0.1 * quality_score)
                
                performance_weights[name] = max(performance_score, 0.01)
                
                logger.info(f"{name} - Combined: {combined_score:.4f}, CTR편향: {ctr_bias:.4f}, 품질: {quality_score:.4f}, 최종: {performance_score:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 고급 성능 평가 실패: {e}")
                performance_weights[name] = 0.01
        
        return performance_weights
    
    def _calculate_diversity_weights_advanced(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """고급 다양성 가중치 계산"""
        
        model_names = list(base_predictions.keys())
        diversity_weights = {}
        
        if self.diversification_method == 'advanced_correlation_based':
            # 고급 상관관계 기반 다양성
            correlation_matrix = self._calculate_correlation_matrix_advanced(base_predictions)
            distance_matrix = self._calculate_distance_matrix(base_predictions)
            
            for name in model_names:
                # Multi-level diversity calculation
                avg_correlation = np.mean([abs(correlation_matrix[name][other]) 
                                         for other in model_names if other != name])
                avg_distance = np.mean([distance_matrix[name][other] 
                                      for other in model_names if other != name])
                
                # Prediction uniqueness
                pred = base_predictions[name]
                uniqueness = len(np.unique(pred)) / len(pred)
                
                diversity_score = (1.0 - avg_correlation) * avg_distance * uniqueness
                diversity_weights[name] = max(diversity_score, 0.1)
        
        elif self.diversification_method == 'advanced_rank_weighted':
            # 고급 순위 기반 다양성
            rank_matrices = {}
            for name, pred in base_predictions.items():
                rank_matrices[name] = pd.Series(pred).rank(pct=True).values
            
            diversity_scores = {}
            for name in model_names:
                rank_differences = []
                rank_correlations = []
                
                for other in model_names:
                    if other != name:
                        # Rank difference
                        rank_diff = np.mean(np.abs(rank_matrices[name] - rank_matrices[other]))
                        rank_differences.append(rank_diff)
                        
                        # Rank correlation
                        rank_corr = np.corrcoef(rank_matrices[name], rank_matrices[other])[0, 1]
                        rank_correlations.append(abs(rank_corr))
                
                avg_rank_diff = np.mean(rank_differences) if rank_differences else 0.5
                avg_rank_corr = np.mean(rank_correlations) if rank_correlations else 0.5
                
                diversity_score = avg_rank_diff * (1 - avg_rank_corr)
                diversity_weights[name] = max(diversity_score, 0.1)
        
        elif self.diversification_method == 'prediction_spectrum_analysis':
            # 예측 스펙트럼 분석 기반 다양성
            for name, pred in base_predictions.items():
                # Frequency domain analysis
                pred_centered = pred - pred.mean()
                fft_result = np.fft.fft(pred_centered)
                power_spectrum = np.abs(fft_result) ** 2
                
                # Spectral diversity measures
                spectral_entropy = -np.sum((power_spectrum / power_spectrum.sum()) * 
                                         np.log(power_spectrum / power_spectrum.sum() + 1e-15))
                spectral_centroid = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)
                
                # Statistical diversity measures
                skewness = ((pred - pred.mean()) ** 3).mean() / (pred.std() ** 3)
                kurtosis = ((pred - pred.mean()) ** 4).mean() / (pred.std() ** 4)
                
                diversity_score = spectral_entropy * np.log(1 + abs(skewness)) * np.log(1 + abs(kurtosis))
                diversity_weights[name] = max(diversity_score, 0.1)
        
        else:
            # 균등 다양성
            diversity_weights = {name: 1.0 for name in model_names}
        
        logger.info(f"고급 다양성 가중치: {diversity_weights}")
        return diversity_weights
    
    def _calculate_correlation_matrix_advanced(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """고급 상관관계 행렬 계산"""
        
        model_names = list(base_predictions.keys())
        correlation_matrix = {}
        
        for name1 in model_names:
            correlation_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    correlation_matrix[name1][name2] = 1.0
                else:
                    try:
                        pred1, pred2 = base_predictions[name1], base_predictions[name2]
                        
                        # Pearson correlation
                        pearson_corr = np.corrcoef(pred1, pred2)[0, 1]
                        
                        # Spearman correlation
                        spearman_corr = np.corrcoef(pd.Series(pred1).rank(), pd.Series(pred2).rank())[0, 1]
                        
                        # Kendall-like correlation (simplified)
                        concordant = 0
                        total = 0
                        for i in range(0, len(pred1), max(1, len(pred1) // 1000)):  # Sample for efficiency
                            for j in range(i+1, len(pred1), max(1, len(pred1) // 1000)):
                                if (pred1[i] - pred1[j]) * (pred2[i] - pred2[j]) > 0:
                                    concordant += 1
                                total += 1
                        
                        kendall_like = (2 * concordant - total) / total if total > 0 else 0
                        
                        # Combined correlation
                        combined_corr = (pearson_corr + spearman_corr + kendall_like) / 3
                        correlation_matrix[name1][name2] = combined_corr if not np.isnan(combined_corr) else 0.0
                        
                    except Exception as e:
                        logger.warning(f"고급 상관관계 계산 실패 ({name1}, {name2}): {e}")
                        correlation_matrix[name1][name2] = 0.0
        
        return correlation_matrix
    
    def _calculate_distance_matrix(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """거리 행렬 계산"""
        
        model_names = list(base_predictions.keys())
        distance_matrix = {}
        
        for name1 in model_names:
            distance_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    distance_matrix[name1][name2] = 0.0
                else:
                    try:
                        pred1, pred2 = base_predictions[name1], base_predictions[name2]
                        
                        # Euclidean distance (normalized)
                        euclidean = np.sqrt(np.mean((pred1 - pred2) ** 2))
                        
                        # Manhattan distance (normalized)
                        manhattan = np.mean(np.abs(pred1 - pred2))
                        
                        # KL divergence approximation
                        pred1_norm = pred1 / pred1.sum()
                        pred2_norm = pred2 / pred2.sum()
                        kl_div = np.sum(pred1_norm * np.log(pred1_norm / (pred2_norm + 1e-15) + 1e-15))
                        
                        # Combined distance
                        combined_distance = euclidean + manhattan + min(kl_div, 10.0)  # Cap KL divergence
                        distance_matrix[name1][name2] = combined_distance
                        
                    except Exception as e:
                        logger.warning(f"거리 계산 실패 ({name1}, {name2}): {e}")
                        distance_matrix[name1][name2] = 1.0
        
        return distance_matrix
    
    def _calculate_stability_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """안정성 가중치 계산"""
        
        stability_weights = {}
        
        for name, pred in base_predictions.items():
            try:
                # Bootstrap stability analysis
                n_bootstrap = 20
                bootstrap_scores = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    indices = np.random.choice(len(pred), size=len(pred), replace=True)
                    boot_pred = pred[indices]
                    boot_y = y.iloc[indices]
                    
                    # Calculate performance on bootstrap sample
                    try:
                        boot_score = self.metrics_calculator.combined_score(boot_y, boot_pred)
                        bootstrap_scores.append(boot_score)
                    except:
                        continue
                
                if len(bootstrap_scores) > 3:
                    stability_score = 1.0 - (np.std(bootstrap_scores) / (np.mean(bootstrap_scores) + 1e-8))
                    stability_weights[name] = max(stability_score, 0.1)
                else:
                    stability_weights[name] = 0.5
                
                logger.info(f"{name} 안정성 점수: {stability_weights[name]:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 안정성 계산 실패: {e}")
                stability_weights[name] = 0.5
        
        return stability_weights
    
    def _combine_weights_advanced(self) -> Dict[str, float]:
        """고급 가중치 결합"""
        
        combined_weights = {}
        model_names = list(self.model_weights.keys())
        
        # Normalize individual weight components
        performance_sum = sum(self.model_weights.values())
        diversity_sum = sum(self.diversity_weights.values())
        stability_sum = sum(self.stability_weights.values())
        
        if performance_sum > 0 and diversity_sum > 0 and stability_sum > 0:
            for name in model_names:
                perf_weight = self.model_weights[name] / performance_sum
                div_weight = self.diversity_weights[name] / diversity_sum
                stab_weight = self.stability_weights[name] / stability_sum
                
                # Advanced combination with adaptive weighting
                adaptive_performance_ratio = 0.6 + 0.2 * (perf_weight - 0.5)  # 0.4 to 0.8
                adaptive_diversity_ratio = 0.3 - 0.1 * (div_weight - 0.5)    # 0.2 to 0.4  
                adaptive_stability_ratio = 1.0 - adaptive_performance_ratio - adaptive_diversity_ratio
                
                combined_weight = (adaptive_performance_ratio * perf_weight + 
                                 adaptive_diversity_ratio * div_weight + 
                                 adaptive_stability_ratio * stab_weight)
                
                combined_weights[name] = combined_weight
        else:
            # Fallback to equal weights
            combined_weights = {name: 1.0/len(model_names) for name in model_names}
        
        # Final normalization
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        return combined_weights
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """고급 안정화된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        ensemble_pred = self._enhance_ensemble_diversity(ensemble_pred)
        
        return ensemble_pred

class CTRAdvancedMetaLearning(BaseEnsemble):
    """CTR 예측 고급 메타 학습 앙상블"""
    
    def __init__(self, meta_model_type: str = 'advanced_stacking', use_meta_features: bool = True):
        super().__init__("CTRAdvancedMetaLearning")
        self.meta_model_type = meta_model_type
        self.use_meta_features = use_meta_features
        self.meta_model = None
        self.feature_scaler = None
        self.meta_feature_selector = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Optional[Dict[str, np.ndarray]] = None):
        """고급 메타 학습 앙상블 학습"""
        logger.info(f"CTR 고급 메타 학습 앙상블 학습 시작 - 메타모델: {self.meta_model_type}")
        
        # Out-of-fold 예측 생성
        oof_predictions = self._generate_advanced_oof_predictions(X, y)
        
        # 고급 메타 피처 생성
        if self.use_meta_features:
            meta_features = self._create_advanced_meta_features(oof_predictions, X)
        else:
            meta_features = oof_predictions
        
        # 피처 선택 및 스케일링
        meta_features = self._preprocess_meta_features(meta_features, y)
        
        # 고급 메타 모델 학습
        self._train_advanced_meta_model(meta_features, y)
        
        self.is_fitted = True
        logger.info("CTR 고급 메타 학습 앙상블 학습 완료")
    
    def _generate_advanced_oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """고급 Out-of-fold 예측 생성"""
        
        oof_predictions = pd.DataFrame(index=X.index)
        
        # Time series split with multiple strategies
        tscv = TimeSeriesSplit(n_splits=5)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in self.base_models.items():
            oof_pred_tscv = np.zeros(len(X))
            oof_pred_kfold = np.zeros(len(X))
            
            # Time series cross-validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    
                    fold_model = self._clone_model_advanced(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    oof_pred_tscv[val_idx] = fold_model.predict_proba(X_val_fold)
                    
                except Exception as e:
                    logger.error(f"{model_name} TSCV 폴드 {fold} 학습 실패: {str(e)}")
                    oof_pred_tscv[val_idx] = 0.0201
            
            # K-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                try:
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    
                    fold_model = self._clone_model_advanced(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    
                    oof_pred_kfold[val_idx] = fold_model.predict_proba(X_val_fold)
                    
                except Exception as e:
                    logger.error(f"{model_name} KFold 폴드 {fold} 학습 실패: {str(e)}")
                    oof_pred_kfold[val_idx] = 0.0201
            
            # Combine TSCV and KFold predictions
            oof_predictions[f'{model_name}_tscv'] = oof_pred_tscv
            oof_predictions[f'{model_name}_kfold'] = oof_pred_kfold
            oof_predictions[f'{model_name}_combined'] = (oof_pred_tscv + oof_pred_kfold) / 2
        
        return oof_predictions
    
    def _clone_model_advanced(self, model: BaseModel) -> BaseModel:
        """고급 모델 복사"""
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
    
    def _create_advanced_meta_features(self, oof_predictions: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """고급 메타 피처 생성"""
        
        meta_features = oof_predictions.copy()
        
        try:
            # 기본 통계 피처
            base_cols = [col for col in oof_predictions.columns if not col.endswith('_tscv') and not col.endswith('_kfold')]
            
            meta_features['pred_mean'] = oof_predictions[base_cols].mean(axis=1)
            meta_features['pred_std'] = oof_predictions[base_cols].std(axis=1)
            meta_features['pred_min'] = oof_predictions[base_cols].min(axis=1)
            meta_features['pred_max'] = oof_predictions[base_cols].max(axis=1)
            meta_features['pred_median'] = oof_predictions[base_cols].median(axis=1)
            meta_features['pred_q25'] = oof_predictions[base_cols].quantile(0.25, axis=1)
            meta_features['pred_q75'] = oof_predictions[base_cols].quantile(0.75, axis=1)
            
            # 고급 분포 피처
            meta_features['pred_skew'] = oof_predictions[base_cols].skew(axis=1)
            meta_features['pred_kurt'] = oof_predictions[base_cols].kurtosis(axis=1)
            meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
            meta_features['pred_iqr'] = meta_features['pred_q75'] - meta_features['pred_q25']
            
            # 순위 기반 피처
            for col in base_cols:
                meta_features[f'{col}_rank'] = oof_predictions[col].rank(pct=True)
                meta_features[f'{col}_rank_norm'] = (oof_predictions[col].rank() - 1) / (len(oof_predictions) - 1)
            
            # 모델간 관계 피처
            for i, col1 in enumerate(base_cols):
                for col2 in base_cols[i+1:]:
                    meta_features[f'{col1}_{col2}_diff'] = oof_predictions[col1] - oof_predictions[col2]
                    meta_features[f'{col1}_{col2}_ratio'] = oof_predictions[col1] / (oof_predictions[col2] + 1e-8)
                    meta_features[f'{col1}_{col2}_product'] = oof_predictions[col1] * oof_predictions[col2]
                    meta_features[f'{col1}_{col2}_harmonic'] = 2 / (1/(oof_predictions[col1] + 1e-8) + 1/(oof_predictions[col2] + 1e-8))
            
            # CV 방법간 차이 피처
            tscv_cols = [col for col in oof_predictions.columns if col.endswith('_tscv')]
            kfold_cols = [col for col in oof_predictions.columns if col.endswith('_kfold')]
            
            for tscv_col, kfold_col in zip(tscv_cols, kfold_cols):
                base_name = tscv_col.replace('_tscv', '')
                meta_features[f'{base_name}_cv_diff'] = oof_predictions[tscv_col] - oof_predictions[kfold_col]
                meta_features[f'{base_name}_cv_ratio'] = oof_predictions[tscv_col] / (oof_predictions[kfold_col] + 1e-8)
            
            # 신뢰도 및 합의 피처
            meta_features['prediction_confidence'] = 1 - meta_features['pred_std']
            meta_features['consensus_strength'] = np.exp(-meta_features['pred_std'])
            meta_features['prediction_entropy'] = -np.sum(
                oof_predictions[base_cols].values * np.log(oof_predictions[base_cols].values + 1e-15), axis=1
            )
            
            # 극값 피처
            meta_features['is_extreme_high'] = (meta_features['pred_mean'] > meta_features['pred_mean'].quantile(0.95)).astype(int)
            meta_features['is_extreme_low'] = (meta_features['pred_mean'] < meta_features['pred_mean'].quantile(0.05)).astype(int)
            meta_features['extreme_distance'] = np.minimum(
                meta_features['pred_mean'] - meta_features['pred_mean'].quantile(0.05),
                meta_features['pred_mean'].quantile(0.95) - meta_features['pred_mean']
            )
            
            # 원본 데이터 요약 피처 (선택적)
            if len(X.columns) <= 100:  # 피처 수가 적을 때만
                try:
                    numeric_cols = X.select_dtypes(include=[np.number]).columns[:20]
                    if len(numeric_cols) > 0:
                        meta_features['x_mean'] = X[numeric_cols].mean(axis=1)
                        meta_features['x_std'] = X[numeric_cols].std(axis=1)
                        meta_features['x_min'] = X[numeric_cols].min(axis=1)
                        meta_features['x_max'] = X[numeric_cols].max(axis=1)
                        meta_features['x_median'] = X[numeric_cols].median(axis=1)
                except:
                    pass
            
            logger.info(f"고급 메타 피처 생성 완료: {meta_features.shape[1]}개 피처")
            
        except Exception as e:
            logger.warning(f"고급 메타 피처 생성 중 오류: {e}")
        
        return meta_features
    
    def _preprocess_meta_features(self, meta_features: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """메타 피처 전처리 및 선택"""
        
        # 결측치 처리
        meta_features_clean = meta_features.fillna(0)
        meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 피처 선택 (상관관계 기반)
        try:
            if len(meta_features_clean.columns) > 100:
                # 타겟과의 상관관계 계산
                correlations = []
                for col in meta_features_clean.columns:
                    try:
                        corr = abs(np.corrcoef(meta_features_clean[col], y)[0, 1])
                        correlations.append((col, corr if not np.isnan(corr) else 0))
                    except:
                        correlations.append((col, 0))
                
                # 상위 N개 피처 선택
                correlations.sort(key=lambda x: x[1], reverse=True)
                selected_features = [col for col, _ in correlations[:80]]
                meta_features_clean = meta_features_clean[selected_features]
                
                logger.info(f"피처 선택 완료: {len(selected_features)}개 피처 선택")
        except Exception as e:
            logger.warning(f"피처 선택 실패: {e}")
        
        # 스케일링
        self.feature_scaler = RobustScaler()
        meta_features_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(meta_features_clean),
            index=meta_features_clean.index,
            columns=meta_features_clean.columns
        )
        
        return meta_features_scaled
    
    def _train_advanced_meta_model(self, meta_features: pd.DataFrame, y: pd.Series):
        """고급 메타 모델 학습"""
        
        if self.meta_model_type == 'advanced_stacking':
            # 다중 모델 스택킹
            base_models = {
                'ridge': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5),
                'elastic': ElasticNetCV(alphas=[0.1, 1.0, 10.0], cv=5),
                'rf': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                'logistic': LogisticRegression(max_iter=1000, random_state=42)
            }
            
            # Level-1 predictions
            level1_predictions = pd.DataFrame(index=meta_features.index)
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for model_name, model in base_models.items():
                oof_pred = np.zeros(len(meta_features))
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(meta_features)):
                    X_train_fold = meta_features.iloc[train_idx]
                    X_val_fold = meta_features.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    
                    if hasattr(model, 'predict_proba'):
                        val_pred = model.predict_proba(X_val_fold)[:, 1]
                    else:
                        val_pred = model.predict(X_val_fold)
                    
                    oof_pred[val_idx] = val_pred
                
                level1_predictions[model_name] = oof_pred
            
            # Level-2 meta model
            self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
            self.meta_model.fit(level1_predictions, y)
            
            # Store base models for inference
            self.level1_models = {}
            for model_name, model in base_models.items():
                model.fit(meta_features, y)
                self.level1_models[model_name] = model
        
        elif self.meta_model_type == 'neural_network':
            self.meta_model = MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            self.meta_model.fit(meta_features, y)
        
        else:  # ridge
            self.meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
            self.meta_model.fit(meta_features, y)
        
        # 성능 평가
        meta_pred = self.predict_meta_model(meta_features)
        metrics_calc = CTRMetrics()
        combined_score = metrics_calc.combined_score(y, meta_pred)
        logger.info(f"고급 메타 모델 Combined Score: {combined_score:.4f}")
    
    def predict_meta_model(self, meta_features: pd.DataFrame) -> np.ndarray:
        """메타 모델 예측"""
        if self.meta_model_type == 'advanced_stacking':
            # Level-1 predictions
            level1_preds = pd.DataFrame()
            for model_name, model in self.level1_models.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(meta_features)[:, 1]
                else:
                    pred = model.predict(meta_features)
                level1_preds[model_name] = pred
            
            # Level-2 prediction
            meta_pred = self.meta_model.predict_proba(level1_preds)[:, 1]
        else:
            if hasattr(self.meta_model, 'predict_proba'):
                meta_pred = self.meta_model.predict_proba(meta_features)[:, 1]
            else:
                meta_pred = self.meta_model.predict(meta_features)
        
        return np.clip(meta_pred, 1e-15, 1 - 1e-15)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """고급 메타 학습 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        try:
            # 기본 예측을 DataFrame으로 변환
            pred_df = pd.DataFrame(base_predictions)
            
            # 메타 피처 생성 (학습 시와 동일한 구조로)
            if self.use_meta_features:
                meta_features = self._create_inference_advanced_meta_features(pred_df)
            else:
                meta_features = pred_df
            
            # 전처리
            meta_features_clean = meta_features.fillna(0)
            meta_features_clean = meta_features_clean.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # 스케일링
            meta_features_scaled = pd.DataFrame(
                self.feature_scaler.transform(meta_features_clean),
                columns=meta_features_clean.columns
            )
            
            # 예측
            ensemble_pred = self.predict_meta_model(meta_features_scaled)
            ensemble_pred = self._enhance_ensemble_diversity(ensemble_pred)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"고급 메타 학습 예측 실패: {str(e)}")
            return np.mean(list(base_predictions.values()), axis=0)
    
    def _create_inference_advanced_meta_features(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """추론용 고급 메타 피처 생성 (학습 시와 동일한 구조)"""
        
        meta_features = pred_df.copy()
        
        try:
            # 기본 통계 피처
            base_cols = pred_df.columns.tolist()
            
            meta_features['pred_mean'] = pred_df.mean(axis=1)
            meta_features['pred_std'] = pred_df.std(axis=1)
            meta_features['pred_min'] = pred_df.min(axis=1)
            meta_features['pred_max'] = pred_df.max(axis=1)
            meta_features['pred_median'] = pred_df.median(axis=1)
            meta_features['pred_q25'] = pred_df.quantile(0.25, axis=1)
            meta_features['pred_q75'] = pred_df.quantile(0.75, axis=1)
            
            # 고급 분포 피처
            meta_features['pred_skew'] = pred_df.skew(axis=1)
            meta_features['pred_kurt'] = pred_df.kurtosis(axis=1)
            meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
            meta_features['pred_iqr'] = meta_features['pred_q75'] - meta_features['pred_q25']
            
            # 순위 기반 피처
            for col in base_cols:
                meta_features[f'{col}_rank'] = pred_df[col].rank(pct=True)
                meta_features[f'{col}_rank_norm'] = (pred_df[col].rank() - 1) / (len(pred_df) - 1)
            
            # 모델간 관계 피처
            for i, col1 in enumerate(base_cols):
                for col2 in base_cols[i+1:]:
                    meta_features[f'{col1}_{col2}_diff'] = pred_df[col1] - pred_df[col2]
                    meta_features[f'{col1}_{col2}_ratio'] = pred_df[col1] / (pred_df[col2] + 1e-8)
                    meta_features[f'{col1}_{col2}_product'] = pred_df[col1] * pred_df[col2]
                    meta_features[f'{col1}_{col2}_harmonic'] = 2 / (1/(pred_df[col1] + 1e-8) + 1/(pred_df[col2] + 1e-8))
            
            # 신뢰도 및 합의 피처
            meta_features['prediction_confidence'] = 1 - meta_features['pred_std']
            meta_features['consensus_strength'] = np.exp(-meta_features['pred_std'])
            meta_features['prediction_entropy'] = -np.sum(
                pred_df.values * np.log(pred_df.values + 1e-15), axis=1
            )
            
            # 극값 피처
            global_q95 = meta_features['pred_mean'].quantile(0.95)
            global_q05 = meta_features['pred_mean'].quantile(0.05)
            
            meta_features['is_extreme_high'] = (meta_features['pred_mean'] > global_q95).astype(int)
            meta_features['is_extreme_low'] = (meta_features['pred_mean'] < global_q05).astype(int)
            meta_features['extreme_distance'] = np.minimum(
                meta_features['pred_mean'] - global_q05,
                global_q95 - meta_features['pred_mean']
            )
            
        except Exception as e:
            logger.warning(f"추론용 고급 메타 피처 생성 중 오류: {e}")
        
        return meta_features

class CTRSuperEnsembleManager:
    """CTR 특화 슈퍼 앙상블 관리 클래스 - Combined Score 0.32+ 목표"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.super_optimal_ensemble = None
        self.target_combined_score = 0.32
        
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        logger.info(f"기본 모델 추가: {name}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """CTR 특화 고성능 앙상블 생성"""
        
        if ensemble_type == 'super_optimal':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            optimization_method = kwargs.get('optimization_method', 'ultra_combined')
            ensemble = CTRSuperOptimalEnsemble(target_ctr, optimization_method)
            self.super_optimal_ensemble = ensemble
        
        elif ensemble_type == 'advanced_stabilized':
            diversification_method = kwargs.get('diversification_method', 'advanced_rank_weighted')
            ensemble = CTRAdvancedStabilizedEnsemble(diversification_method)
        
        elif ensemble_type == 'advanced_meta':
            meta_model_type = kwargs.get('meta_model_type', 'advanced_stacking')
            use_meta_features = kwargs.get('use_meta_features', True)
            ensemble = CTRAdvancedMetaLearning(meta_model_type, use_meta_features)
        
        else:
            raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
        
        # 기본 모델 추가
        for name, model in self.base_models.items():
            ensemble.add_base_model(name, model)
        
        self.ensembles[ensemble_type] = ensemble
        logger.info(f"고성능 앙상블 생성: {ensemble_type}")
        
        return ensemble
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """모든 고성능 앙상블 학습"""
        logger.info("모든 고성능 앙상블 학습 시작")
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                start_time = time.time()
                pred = model.predict_proba(X)
                prediction_time = time.time() - start_time
                
                base_predictions[name] = pred
                logger.info(f"{name} 모델 예측 완료 ({prediction_time:.2f}초)")
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
        
        # 각 앙상블 학습
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                start_time = time.time()
                ensemble.fit(X, y, base_predictions)
                training_time = time.time() - start_time
                
                logger.info(f"{ensemble_type} 고성능 앙상블 학습 완료 ({training_time:.2f}초)")
                
            except Exception as e:
                logger.error(f"{ensemble_type} 고성능 앙상블 학습 실패: {str(e)}")
        
        # 메모리 정리
        gc.collect()
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """고성능 앙상블 성능 평가"""
        logger.info("고성능 앙상블 성능 평가 시작")
        
        results = {}
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                base_predictions[name] = pred
                
                score = self.metrics_calculator.combined_score(y_val, pred)
                ctr_optimized_score = self.metrics_calculator.ctr_optimized_score(y_val, pred)
                
                results[f"base_{name}"] = score
                results[f"base_{name}_ctr_optimized"] = ctr_optimized_score
                
            except Exception as e:
                logger.error(f"{name} 모델 검증 예측 실패: {str(e)}")
                results[f"base_{name}"] = 0.0
                results[f"base_{name}_ctr_optimized"] = 0.0
        
        # 앙상블 성능 평가
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    ensemble_pred = ensemble.predict_proba(base_predictions)
                    
                    combined_score = self.metrics_calculator.combined_score(y_val, ensemble_pred)
                    ctr_optimized_score = self.metrics_calculator.ctr_optimized_score(y_val, ensemble_pred)
                    ap_score = self.metrics_calculator.average_precision(y_val, ensemble_pred)
                    wll_score = self.metrics_calculator.weighted_log_loss(y_val, ensemble_pred)
                    
                    results[f"ensemble_{ensemble_type}"] = combined_score
                    results[f"ensemble_{ensemble_type}_ctr_optimized"] = ctr_optimized_score
                    results[f"ensemble_{ensemble_type}_ap"] = ap_score
                    results[f"ensemble_{ensemble_type}_wll"] = wll_score
                    
                    logger.info(f"{ensemble_type} 앙상블 Combined Score: {combined_score:.4f}")
                    logger.info(f"{ensemble_type} 앙상블 CTR Optimized Score: {ctr_optimized_score:.4f}")
                    
                    # CTR 분석
                    predicted_ctr = ensemble_pred.mean()
                    actual_ctr = y_val.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    logger.info(f"{ensemble_type} CTR: 예측 {predicted_ctr:.4f} vs 실제 {actual_ctr:.4f} (편향: {ctr_bias:.4f})")
                    
                    # 목표 달성 여부
                    target_achieved = combined_score >= self.target_combined_score
                    logger.info(f"{ensemble_type} 목표 달성: {target_achieved} (목표: {self.target_combined_score})")
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 평가 실패: {str(e)}")
                    results[f"ensemble_{ensemble_type}"] = 0.0
                    results[f"ensemble_{ensemble_type}_ctr_optimized"] = 0.0
        
        # 최고 성능 앙상블 선택
        if results:
            ensemble_results = {k: v for k, v in results.items() if k.startswith('ensemble_') and not k.endswith('_ctr_optimized') and not k.endswith('_ap') and not k.endswith('_wll')}
            
            if ensemble_results:
                best_name = max(ensemble_results, key=ensemble_results.get)
                best_score = ensemble_results[best_name]
                
                ensemble_type = best_name.replace('ensemble_', '')
                self.best_ensemble = self.ensembles[ensemble_type]
                
                logger.info(f"최고 성능 앙상블: {ensemble_type} (Combined Score: {best_score:.4f})")
                
                if best_score >= self.target_combined_score:
                    logger.info("🎉 목표 Combined Score 0.32+ 달성!")
                else:
                    logger.info(f"목표까지 {self.target_combined_score - best_score:.4f} 부족")
            else:
                logger.info("평가 가능한 앙상블이 없습니다.")
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
                if result_name.startswith('base_') and not result_name.endswith('_ctr_optimized') and score > best_score:
                    best_score = score
                    best_model_name = result_name.replace('base_', '')
            
            if best_model_name and best_model_name in self.base_models:
                logger.info(f"최고 성능 기본 모델 사용: {best_model_name}")
                return self.base_models[best_model_name].predict_proba(X)
            else:
                # 평균 앙상블
                logger.info("평균 앙상블 사용")
                return np.mean(list(base_predictions.values()), axis=0)
        
        return self.best_ensemble.predict_proba(base_predictions)
    
    def save_ensembles(self, output_dir: Path = None):
        """고성능 앙상블 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 개별 앙상블 저장
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                ensemble_path = output_dir / f"high_performance_ensemble_{ensemble_type}.pkl"
                
                try:
                    with open(ensemble_path, 'wb') as f:
                        pickle.dump(ensemble, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    logger.info(f"{ensemble_type} 고성능 앙상블 저장: {ensemble_path}")
                except Exception as e:
                    logger.error(f"{ensemble_type} 고성능 앙상블 저장 실패: {str(e)}")
        
        # 최고 앙상블 정보 저장
        best_info = {
            'best_ensemble_type': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results,
            'target_combined_score': self.target_combined_score,
            'super_optimal_available': self.super_optimal_ensemble is not None,
            'target_achieved': any(
                score >= self.target_combined_score 
                for key, score in self.ensemble_results.items() 
                if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') and not key.endswith('_ap') and not key.endswith('_wll')
            )
        }
        
        try:
            import json
            info_path = output_dir / "high_performance_ensemble_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(best_info, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"고성능 앙상블 정보 저장: {info_path}")
        except Exception as e:
            logger.error(f"고성능 앙상블 정보 저장 실패: {str(e)}")
    
    def load_ensembles(self, input_dir: Path = None):
        """고성능 앙상블 로딩"""
        if input_dir is None:
            input_dir = self.config.MODEL_DIR
        
        input_dir = Path(input_dir)
        
        ensemble_files = list(input_dir.glob("high_performance_ensemble_*.pkl"))
        
        for ensemble_file in ensemble_files:
            try:
                ensemble_type = ensemble_file.stem.replace('high_performance_ensemble_', '')
                
                with open(ensemble_file, 'rb') as f:
                    ensemble = pickle.load(f)
                
                self.ensembles[ensemble_type] = ensemble
                
                # 특수 앙상블 참조 설정
                if ensemble_type == 'super_optimal':
                    self.super_optimal_ensemble = ensemble
                
                logger.info(f"{ensemble_type} 고성능 앙상블 로딩 완료")
                
            except Exception as e:
                logger.error(f"{ensemble_file} 고성능 앙상블 로딩 실패: {str(e)}")
        
        # 최고 앙상블 정보 로딩
        info_path = input_dir / "high_performance_ensemble_info.json"
        if info_path.exists():
            try:
                import json
                with open(info_path, 'r', encoding='utf-8') as f:
                    best_info = json.load(f)
                
                best_type = best_info.get('best_ensemble_type')
                if best_type:
                    for ensemble in self.ensembles.values():
                        if ensemble.name == best_type:
                            self.best_ensemble = ensemble
                            break
                
                self.ensemble_results = best_info.get('ensemble_results', {})
                self.target_combined_score = best_info.get('target_combined_score', 0.32)
                
            except Exception as e:
                logger.error(f"고성능 앙상블 정보 로딩 실패: {str(e)}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """고성능 앙상블 요약 정보"""
        
        target_achieved_count = sum(
            1 for key, score in self.ensemble_results.items()
            if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') 
            and not key.endswith('_ap') and not key.endswith('_wll')
            and score >= self.target_combined_score
        )
        
        return {
            'total_ensembles': len(self.ensembles),
            'fitted_ensembles': sum(1 for e in self.ensembles.values() if e.is_fitted),
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'ensemble_results': self.ensemble_results,
            'base_models_count': len(self.base_models),
            'super_optimal_ensemble_available': self.super_optimal_ensemble is not None and self.super_optimal_ensemble.is_fitted,
            'ensemble_types': list(self.ensembles.keys()),
            'target_combined_score': self.target_combined_score,
            'target_achieved_count': target_achieved_count,
            'target_achieved': target_achieved_count > 0,
            'best_combined_score': max(
                (score for key, score in self.ensemble_results.items()
                 if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') 
                 and not key.endswith('_ap') and not key.endswith('_wll')),
                default=0.0
            )
        }

# 기존 클래스명 유지 (하위 호환성)
CTROptimalEnsemble = CTRSuperOptimalEnsemble
CTRStabilizedEnsemble = CTRAdvancedStabilizedEnsemble  
CTRMetaLearning = CTRAdvancedMetaLearning
CTREnsembleManager = CTRSuperEnsembleManager
EnsembleManager = CTRSuperEnsembleManager