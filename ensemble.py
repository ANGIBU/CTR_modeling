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
from models import BaseModel, CTRCalibrator
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """앙상블 모델 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.target_combined_score = 0.30
        self.ensemble_calibrator = None
        self.is_calibrated = False
        
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
                # 모델이 캘리브레이션되어 있으면 캘리브레이션된 예측 사용
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    pred = model.predict_proba(X)
                    logger.info(f"{name} 모델: 캘리브레이션 적용된 예측 사용")
                else:
                    pred = model.predict_proba(X)
                    
                predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                predictions[name] = np.full(len(X), 0.0201)
        
        return predictions
    
    def apply_ensemble_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, 
                                 ensemble_predictions: np.ndarray, method: str = 'auto'):
        """앙상블 수준 캘리브레이션 적용"""
        try:
            logger.info(f"{self.name} 앙상블에 캘리브레이션 적용: {method}")
            
            if len(ensemble_predictions) != len(y_val):
                logger.warning("앙상블 캘리브레이션: 크기 불일치")
                return
            
            # CTRCalibrator 생성 및 학습
            self.ensemble_calibrator = CTRCalibrator(target_ctr=0.0201, method=method)
            self.ensemble_calibrator.fit(y_val.values, ensemble_predictions)
            
            self.is_calibrated = True
            
            # 캘리브레이션 효과 확인
            calibrated_predictions = self.ensemble_calibrator.predict(ensemble_predictions)
            
            original_ctr = ensemble_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val.mean()
            
            logger.info(f"앙상블 캘리브레이션 결과:")
            logger.info(f"  - 원본 CTR: {original_ctr:.4f}")
            logger.info(f"  - 캘리브레이션 CTR: {calibrated_ctr:.4f}")
            logger.info(f"  - 실제 CTR: {actual_ctr:.4f}")
            logger.info(f"  - 최적 방법: {self.ensemble_calibrator.best_method}")
            
        except Exception as e:
            logger.error(f"앙상블 캘리브레이션 적용 실패: {e}")
            self.is_calibrated = False
    
    def predict_proba_calibrated(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """캘리브레이션이 적용된 앙상블 예측"""
        # 기본 앙상블 예측
        raw_prediction = self.predict_proba(base_predictions)
        
        # 앙상블 수준 캘리브레이션 적용
        if self.is_calibrated and self.ensemble_calibrator is not None:
            try:
                calibrated_prediction = self.ensemble_calibrator.predict(raw_prediction)
                return np.clip(calibrated_prediction, 1e-15, 1 - 1e-15)
            except Exception as e:
                logger.warning(f"앙상블 캘리브레이션 예측 실패: {e}")
        
        return raw_prediction
    
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
    """CTR 예측 최적 앙상블"""
    
    def __init__(self, target_ctr: float = 0.0201, optimization_method: str = 'ultra_combined'):
        super().__init__("CTRSuperOptimalEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.final_weights = {}
        self.metrics_calculator = CTRMetrics()
        self.temperature = 1.2  # 1.0 → 1.2로 증가
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
        self.quantile_corrections = {}
        self.performance_boosters = {}
        self.target_combined_score = 0.32  # 0.30 → 0.32로 증가
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """최적 앙상블 학습"""
        logger.info(f"CTR 최적 앙상블 학습 시작 - 목표: Combined Score 0.32+")
        
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
        
        # 3단계: CTR 특화 후처리 최적화
        logger.info("3단계: CTR 특화 후처리 최적화")
        self._apply_super_ctr_postprocessing(ensemble_pred, y)
        
        # 4단계: 성능 부스터 적용
        logger.info("4단계: 성능 부스터 적용")
        self._apply_performance_boosters(ensemble_pred, y)
        
        # 5단계: 앙상블 캘리브레이션 적용
        logger.info("5단계: 앙상블 캘리브레이션 적용")
        try:
            # 검증 데이터 분할 (캘리브레이션용)
            split_idx = int(len(X) * 0.8)
            X_cal = X.iloc[split_idx:]
            y_cal = y.iloc[split_idx:]
            
            # 캘리브레이션용 앙상블 예측
            cal_base_predictions = {}
            for name in available_models:
                if name in base_predictions:
                    cal_base_predictions[name] = base_predictions[name][split_idx:]
            
            cal_ensemble_pred = self._create_weighted_ensemble(cal_base_predictions)
            cal_ensemble_pred = self._apply_all_corrections(cal_ensemble_pred)
            
            # 앙상블 캘리브레이션 적용
            self.apply_ensemble_calibration(X_cal, y_cal, cal_ensemble_pred, method='auto')
            
        except Exception as e:
            logger.warning(f"앙상블 캘리브레이션 적용 실패: {e}")
        
        # 6단계: 최종 검증 및 조정
        logger.info("6단계: 최종 검증 및 조정")
        final_pred = self._apply_all_corrections(ensemble_pred)
        
        if self.is_calibrated and self.ensemble_calibrator:
            try:
                final_pred = self.ensemble_calibrator.predict(final_pred)
            except:
                pass
        
        final_score = self.metrics_calculator.combined_score(y, final_pred)
        
        logger.info(f"최적 앙상블 최종 Combined Score: {final_score:.4f}")
        logger.info(f"앙상블 캘리브레이션 적용: {'Yes' if self.is_calibrated else 'No'}")
        
        self.is_fitted = True
        logger.info("CTR 최적 앙상블 학습 완료")
    
    def _multi_layer_weight_optimization(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """다층 가중치 최적화"""
        
        model_names = list(base_predictions.keys())
        
        # Layer 1: 개별 모델 성능 평가
        individual_scores = {}
        ctr_alignment_scores = {}
        diversity_scores = {}
        calibration_scores = {}
        
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
            
            # 캘리브레이션 품질 평가
            model = self.base_models.get(name)
            calibration_quality = 0.5  # 기본값
            if model and hasattr(model, 'is_calibrated') and model.is_calibrated:
                calibration_quality = 0.8
                if hasattr(model, 'calibrator') and model.calibrator:
                    calibration_summary = model.calibrator.get_calibration_summary()
                    if calibration_summary['calibration_scores']:
                        calibration_quality = max(calibration_summary['calibration_scores'].values())
            
            individual_scores[name] = combined_score
            ctr_alignment_scores[name] = ctr_alignment
            diversity_scores[name] = diversity
            calibration_scores[name] = calibration_quality
        
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
        
        # Layer 3: Optuna 기반 최적화
        if OPTUNA_AVAILABLE:
            try:
                optimized_weights = self._optuna_ultra_optimization_with_calibration(
                    base_predictions, y, individual_scores, calibration_scores
                )
            except Exception as e:
                logger.warning(f"Optuna 최적화 실패: {e}")
                optimized_weights = self._fallback_optimization_with_calibration(
                    individual_scores, ctr_alignment_scores, diversity_scores, calibration_scores
                )
        else:
            optimized_weights = self._fallback_optimization_with_calibration(
                individual_scores, ctr_alignment_scores, diversity_scores, calibration_scores
            )
        
        logger.info(f"다층 최적화 가중치: {optimized_weights}")
        return optimized_weights
    
    def _optuna_ultra_optimization_with_calibration(self, base_predictions: Dict[str, np.ndarray], 
                                                   y: pd.Series, 
                                                   individual_scores: Dict[str, float],
                                                   calibration_scores: Dict[str, float]) -> Dict[str, float]:
        """Optuna 최적화"""
        
        model_names = list(base_predictions.keys())
        
        def ultra_objective_with_calibration(trial):
            weights = {}
            
            # Dynamic weight bounds based on individual performance and calibration
            for name in model_names:
                base_performance = individual_scores.get(name, 0.1)
                calibration_quality = calibration_scores.get(name, 0.5)
                
                # 캘리브레이션 품질이 높은 모델에 더 높은 가중치 범위 부여
                performance_factor = base_performance + 0.4 * calibration_quality  # 0.3 → 0.4로 증가
                
                if performance_factor > 0.65:  # 0.6 → 0.65로 증가
                    min_weight, max_weight = 0.15, 0.85  # 0.1, 0.8 → 0.15, 0.85로 증가
                elif performance_factor > 0.45:  # 0.4 → 0.45로 증가
                    min_weight, max_weight = 0.08, 0.65  # 0.05, 0.6 → 0.08, 0.65로 증가
                else:
                    min_weight, max_weight = 0.02, 0.45  # 0.01, 0.4 → 0.02, 0.45로 증가
                
                weights[name] = trial.suggest_float(f'weight_{name}', min_weight, max_weight)
            
            # Advanced ensemble techniques
            ensemble_method = trial.suggest_categorical('ensemble_method', ['weighted', 'power_weighted', 'rank_weighted'])
            temperature = trial.suggest_float('temperature', 0.9, 2.2)  # 0.8, 2.0 → 0.9, 2.2로 증가
            
            # Create ensemble prediction
            if ensemble_method == 'weighted':
                ensemble_pred = np.zeros(len(y))
                total_weight = sum(weights.values())
                for name, weight in weights.items():
                    if name in base_predictions:
                        ensemble_pred += (weight / total_weight) * base_predictions[name]
            
            elif ensemble_method == 'power_weighted':
                power = trial.suggest_float('power', 1.2, 3.5)  # 1.0, 3.0 → 1.2, 3.5로 증가
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
            ctr_alignment = np.exp(-abs(predicted_ctr - actual_ctr) * 350)  # 300 → 350으로 증가
            
            # Diversity bonus
            diversity = ensemble_pred.std()
            
            # Calibration bonus
            calibration_bonus = sum(weights[name] * calibration_scores.get(name, 0.5) 
                                  for name in weights if name in calibration_scores)
            calibration_bonus /= sum(weights.values()) if sum(weights.values()) > 0 else 1.0
            
            # Final score with enhanced bonuses
            final_score = combined_score * (1 + 0.2 * ctr_alignment) * (1 + 0.15 * diversity) * (1 + 0.25 * calibration_bonus)  # 0.15, 0.1, 0.2 → 0.2, 0.15, 0.25로 증가
            
            return final_score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=25),  # 20 → 25로 증가
            pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=15)  # 15, 10 → 20, 15로 증가
        )
        
        study.optimize(ultra_objective_with_calibration, n_trials=200, show_progress_bar=False)  # 150 → 200으로 증가
        
        # Extract optimized weights
        optimized_weights = {}
        for param_name, weight in study.best_params.items():
            if param_name.startswith('weight_'):
                model_name = param_name.replace('weight_', '')
                optimized_weights[model_name] = weight
        
        # Store additional optimization parameters
        self.ensemble_method = study.best_params.get('ensemble_method', 'weighted')
        self.temperature = study.best_params.get('temperature', 1.2)  # 1.0 → 1.2로 증가
        
        # Normalize weights
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
        
        logger.info(f"Optuna 최적화 완료 - 최고 점수: {study.best_value:.4f}")
        logger.info(f"최적화 방법: {self.ensemble_method}, Temperature: {self.temperature:.3f}")
        
        return optimized_weights
    
    def _fallback_optimization_with_calibration(self, individual_scores: Dict[str, float], 
                                              ctr_alignment_scores: Dict[str, float],
                                              diversity_scores: Dict[str, float],
                                              calibration_scores: Dict[str, float]) -> Dict[str, float]:
        """대체 최적화 방법"""
        
        weights = {}
        for name in individual_scores.keys():
            performance_weight = individual_scores.get(name, 0.1)
            alignment_weight = ctr_alignment_scores.get(name, 0.5)
            diversity_weight = diversity_scores.get(name, 0.5)
            calibration_weight = calibration_scores.get(name, 0.5)
            
            # 캘리브레이션 가중치 강화
            combined_weight = (0.35 * performance_weight +   # 0.4 → 0.35로 조정
                             0.25 * alignment_weight +       # 0.25 유지
                             0.15 * diversity_weight +       # 0.15 유지
                             0.25 * calibration_weight)      # 0.2 → 0.25로 증가
            
            weights[name] = max(combined_weight, 0.02)  # 0.01 → 0.02로 증가
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
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
                return min(improvement_ratio, 2.2)  # 2.0 → 2.2로 증가
            else:
                return 1.0
        except:
            return 1.0
    
    def _apply_super_ctr_postprocessing(self, predictions: np.ndarray, y: pd.Series):
        """CTR 특화 후처리 최적화"""
        logger.info("CTR 후처리 최적화 시작")
        
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
            
            logger.info(f"CTR 후처리 완료")
            logger.info(f"편향 보정: {self.bias_correction:.4f}, 승수 보정: {self.multiplicative_correction:.4f}")
            logger.info(f"Temperature: {self.temperature:.3f}")
            
        except Exception as e:
            logger.error(f"CTR 후처리 최적화 실패: {e}")
            self.bias_correction = 0.0
            self.multiplicative_correction = 1.0
            self.temperature = 1.2  # 1.0 → 1.2로 증가
    
    def _fit_advanced_temperature_scaling(self, predictions: np.ndarray, y: pd.Series):
        """Temperature Scaling"""
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
                ctr_loss = abs(calibrated_probs.mean() - y.mean()) * 1200  # 1000 → 1200으로 증가
                
                # Diversity preservation
                diversity_loss = -calibrated_probs.std()
                
                return log_loss + ctr_loss + diversity_loss
            
            result = minimize(
                advanced_temperature_loss, 
                x0=[1.2, 0.0],  # 1.0 → 1.2로 증가
                bounds=[(0.2, 12.0), (-2.5, 2.5)],  # (0.1, 10.0), (-2.0, 2.0) → (0.2, 12.0), (-2.5, 2.5)로 확장
                method='L-BFGS-B'
            )
            
            self.temperature = result.x[0]
            self.logit_shift = result.x[1]
            
        except Exception as e:
            logger.warning(f"Temperature scaling 실패: {e}")
            self.temperature = 1.2  # 1.0 → 1.2로 증가
            self.logit_shift = 0.0
    
    def _fit_quantile_corrections(self, predictions: np.ndarray, y: pd.Series):
        """분위수 기반 보정"""
        try:
            quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]  # 분위수 추가
            
            for q in quantiles:
                threshold = np.percentile(predictions, q * 100)
                mask = predictions >= threshold
                
                if mask.sum() > 8:  # 10 → 8로 감소
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
            window_size = max(len(predictions) // 60, 80)  # 50, 100 → 60, 80으로 조정
            self.distribution_mapping = {}
            
            for i in range(0, len(predictions) - window_size, window_size // 3):  # window_size // 2 → window_size // 3으로 증가
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
            high_conf_threshold = np.percentile(predictions, 96)  # 95 → 96으로 증가
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
            low_conf_threshold = np.percentile(predictions, 4)  # 5 → 4로 감소
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
            median_low, median_high = np.percentile(predictions, [38, 62])  # [40, 60] → [38, 62]로 확장
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
        """가중 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        if hasattr(self, 'ensemble_method') and self.ensemble_method == 'power_weighted':
            power = getattr(self, 'power', 2.2)  # 2.0 → 2.2로 증가
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
        """최적화된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        # 가중 앙상블 생성
        ensemble_pred = self._create_weighted_ensemble(base_predictions)
        
        # 모든 보정 기법 적용
        final_pred = self._apply_all_corrections(ensemble_pred)
        
        # 앙상블 수준 캘리브레이션 적용
        if self.is_calibrated and self.ensemble_calibrator is not None:
            try:
                final_pred = self.ensemble_calibrator.predict(final_pred)
                final_pred = np.clip(final_pred, 1e-15, 1 - 1e-15)
            except Exception as e:
                logger.warning(f"앙상블 캘리브레이션 예측 실패: {e}")
        
        return final_pred

class CTRAdvancedStabilizedEnsemble(BaseEnsemble):
    """CTR 예측 안정화 앙상블"""
    
    def __init__(self, diversification_method: str = 'advanced_rank_weighted'):
        super().__init__("CTRAdvancedStabilizedEnsemble")
        self.diversification_method = diversification_method
        self.model_weights = {}
        self.diversity_weights = {}
        self.stability_weights = {}
        self.calibration_weights = {}
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
        self.model_weights = self._evaluate_individual_performance_advanced(base_predictions, y)
        
        # 2. 다양성 가중치 계산
        self.diversity_weights = self._calculate_diversity_weights_advanced(base_predictions)
        
        # 3. 안정성 가중치 계산
        self.stability_weights = self._calculate_stability_weights(base_predictions, y)
        
        # 4. 캘리브레이션 가중치 계산
        self.calibration_weights = self._calculate_calibration_weights()
        
        # 5. 최종 가중치 결합
        self.final_weights = self._combine_weights_advanced_with_calibration()
        
        # 6. 앙상블 캘리브레이션 적용
        try:
            ensemble_pred = self._create_stabilized_ensemble(base_predictions)
            
            # 검증 데이터 분할
            split_idx = int(len(X) * 0.8)
            X_cal = X.iloc[split_idx:]
            y_cal = y.iloc[split_idx:]
            
            cal_base_predictions = {}
            for name in available_models:
                if name in base_predictions:
                    cal_base_predictions[name] = base_predictions[name][split_idx:]
            
            cal_ensemble_pred = self._create_stabilized_ensemble(cal_base_predictions)
            
            self.apply_ensemble_calibration(X_cal, y_cal, cal_ensemble_pred, method='auto')
            
        except Exception as e:
            logger.warning(f"안정화 앙상블 캘리브레이션 적용 실패: {e}")
        
        self.is_fitted = True
        logger.info(f"CTR 안정화 앙상블 학습 완료 - 최종 가중치: {self.final_weights}")
        logger.info(f"앙상블 캘리브레이션 적용: {'Yes' if self.is_calibrated else 'No'}")
    
    def _evaluate_individual_performance_advanced(self, base_predictions: Dict[str, np.ndarray], 
                                                y: pd.Series) -> Dict[str, float]:
        """개별 모델 성능 평가"""
        
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
                ctr_penalty = np.exp(-ctr_bias * 350)  # 300 → 350으로 증가
                
                # Prediction quality metrics
                pred_std = pred.std()
                pred_range = pred.max() - pred.min()
                pred_entropy = -np.mean(pred * np.log(pred + 1e-15) + (1 - pred) * np.log(1 - pred + 1e-15))
                
                quality_score = pred_std * pred_range * pred_entropy
                
                # Calibration bonus
                model = self.base_models.get(name)
                calibration_bonus = 1.0  # 기본값
                if model and hasattr(model, 'is_calibrated') and model.is_calibrated:
                    calibration_bonus = 1.25  # 1.2 → 1.25로 증가
                    if hasattr(model, 'calibrator') and model.calibrator:
                        calibration_summary = model.calibrator.get_calibration_summary()
                        if calibration_summary['calibration_scores']:
                            calibration_quality = max(calibration_summary['calibration_scores'].values())
                            calibration_bonus = 1.0 + 0.35 * calibration_quality  # 0.3 → 0.35로 증가
                
                # Final performance score with calibration bonus
                performance_score = (0.5 * combined_score + 
                                   0.22 * ctr_penalty +      # 0.2 → 0.22로 증가
                                   0.18 * (ap_score * (1 / (1 + wll_score))) +  # 0.2 → 0.18로 조정
                                   0.1 * quality_score) * calibration_bonus
                
                performance_weights[name] = max(performance_score, 0.015)  # 0.01 → 0.015로 증가
                
                logger.info(f"{name} - Combined: {combined_score:.4f}, CTR편향: {ctr_bias:.4f}, "
                          f"품질: {quality_score:.4f}, 캘리브레이션 보너스: {calibration_bonus:.2f}x, "
                          f"최종: {performance_score:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 성능 평가 실패: {e}")
                performance_weights[name] = 0.015  # 0.01 → 0.015로 증가
        
        return performance_weights
    
    def _calculate_calibration_weights(self) -> Dict[str, float]:
        """캘리브레이션 가중치 계산"""
        calibration_weights = {}
        
        for name, model in self.base_models.items():
            try:
                calibration_weight = 0.5  # 기본값
                
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    calibration_weight = 0.85  # 0.8 → 0.85로 증가
                    
                    if hasattr(model, 'calibrator') and model.calibrator:
                        calibration_summary = model.calibrator.get_calibration_summary()
                        if calibration_summary['calibration_scores']:
                            # 캘리브레이션 품질에 따른 가중치
                            calibration_quality = max(calibration_summary['calibration_scores'].values())
                            calibration_weight = 0.5 + 0.6 * calibration_quality  # 0.5 → 0.6으로 증가
                            
                            logger.info(f"{name} 캘리브레이션 품질: {calibration_quality:.4f}, "
                                      f"가중치: {calibration_weight:.4f}")
                
                calibration_weights[name] = calibration_weight
                
            except Exception as e:
                logger.warning(f"{name} 캘리브레이션 가중치 계산 실패: {e}")
                calibration_weights[name] = 0.5
        
        return calibration_weights
    
    def _calculate_diversity_weights_advanced(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """다양성 가중치 계산"""
        
        model_names = list(base_predictions.keys())
        diversity_weights = {}
        
        if self.diversification_method == 'advanced_correlation_based':
            # 상관관계 기반 다양성
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
                diversity_weights[name] = max(diversity_score, 0.12)  # 0.1 → 0.12로 증가
        
        elif self.diversification_method == 'advanced_rank_weighted':
            # 순위 기반 다양성
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
                diversity_weights[name] = max(diversity_score, 0.12)  # 0.1 → 0.12로 증가
        
        else:
            # 균등 다양성
            diversity_weights = {name: 1.0 for name in model_names}
        
        logger.info(f"다양성 가중치: {diversity_weights}")
        return diversity_weights
    
    def _calculate_correlation_matrix_advanced(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """상관관계 행렬 계산"""
        
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
                        for i in range(0, len(pred1), max(1, len(pred1) // 1000)):
                            for j in range(i+1, len(pred1), max(1, len(pred1) // 1000)):
                                if (pred1[i] - pred1[j]) * (pred2[i] - pred2[j]) > 0:
                                    concordant += 1
                                total += 1
                        
                        kendall_like = (2 * concordant - total) / total if total > 0 else 0
                        
                        # Combined correlation
                        combined_corr = (pearson_corr + spearman_corr + kendall_like) / 3
                        correlation_matrix[name1][name2] = combined_corr if not np.isnan(combined_corr) else 0.0
                        
                    except Exception as e:
                        logger.warning(f"상관관계 계산 실패 ({name1}, {name2}): {e}")
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
                        combined_distance = euclidean + manhattan + min(kl_div, 12.0)  # 10.0 → 12.0으로 증가
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
                n_bootstrap = 25  # 20 → 25로 증가
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
                    stability_weights[name] = max(stability_score, 0.12)  # 0.1 → 0.12로 증가
                else:
                    stability_weights[name] = 0.5
                
                logger.info(f"{name} 안정성 점수: {stability_weights[name]:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 안정성 계산 실패: {e}")
                stability_weights[name] = 0.5
        
        return stability_weights
    
    def _combine_weights_advanced_with_calibration(self) -> Dict[str, float]:
        """가중치 결합"""
        
        combined_weights = {}
        model_names = list(self.model_weights.keys())
        
        # Normalize individual weight components
        performance_sum = sum(self.model_weights.values())
        diversity_sum = sum(self.diversity_weights.values())
        stability_sum = sum(self.stability_weights.values())
        calibration_sum = sum(self.calibration_weights.values())
        
        if performance_sum > 0 and diversity_sum > 0 and stability_sum > 0 and calibration_sum > 0:
            for name in model_names:
                perf_weight = self.model_weights[name] / performance_sum
                div_weight = self.diversity_weights[name] / diversity_sum
                stab_weight = self.stability_weights[name] / stability_sum
                cal_weight = self.calibration_weights[name] / calibration_sum
                
                # Advanced combination with adaptive weighting (Calibration 강화)
                adaptive_performance_ratio = 0.45 + 0.15 * (perf_weight - 0.5)  # 0.5 → 0.45로 조정
                adaptive_diversity_ratio = 0.23 - 0.05 * (div_weight - 0.5)      # 0.25 → 0.23으로 조정
                adaptive_stability_ratio = 0.15 - 0.05 * (stab_weight - 0.5)    # 유지
                adaptive_calibration_ratio = 0.17 + 0.05 * (cal_weight - 0.5)   # 0.1 → 0.17로 증가
                
                # 비율 정규화
                total_ratio = (adaptive_performance_ratio + adaptive_diversity_ratio + 
                             adaptive_stability_ratio + adaptive_calibration_ratio)
                
                adaptive_performance_ratio /= total_ratio
                adaptive_diversity_ratio /= total_ratio
                adaptive_stability_ratio /= total_ratio
                adaptive_calibration_ratio /= total_ratio
                
                combined_weight = (adaptive_performance_ratio * perf_weight + 
                                 adaptive_diversity_ratio * div_weight + 
                                 adaptive_stability_ratio * stab_weight +
                                 adaptive_calibration_ratio * cal_weight)
                
                combined_weights[name] = combined_weight
        else:
            # Fallback to equal weights
            combined_weights = {name: 1.0/len(model_names) for name in model_names}
        
        # Final normalization
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        return combined_weights
    
    def _create_stabilized_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """안정화된 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        ensemble_pred = self._enhance_ensemble_diversity(ensemble_pred)
        
        return ensemble_pred
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """안정화된 앙상블 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블 모델이 학습되지 않았습니다")
        
        # 안정화된 앙상블 예측
        ensemble_pred = self._create_stabilized_ensemble(base_predictions)
        
        # 앙상블 수준 캘리브레이션 적용
        if self.is_calibrated and self.ensemble_calibrator is not None:
            try:
                ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            except Exception as e:
                logger.warning(f"안정화 앙상블 캘리브레이션 예측 실패: {e}")
        
        return ensemble_pred

class CTRSuperEnsembleManager:
    """CTR 특화 앙상블 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.super_optimal_ensemble = None
        self.target_combined_score = 0.32  # 0.30 → 0.32로 증가
        self.calibration_manager = {}
        
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        
        # 캘리브레이션 상태 로깅
        calibration_status = "No"
        if hasattr(model, 'is_calibrated') and model.is_calibrated:
            calibration_status = "Yes"
            if hasattr(model, 'calibrator') and model.calibrator:
                calibration_method = getattr(model.calibrator, 'best_method', 'unknown')
                calibration_status = f"Yes ({calibration_method})"
        
        logger.info(f"기본 모델 추가: {name} - Calibration: {calibration_status}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """CTR 특화 앙상블 생성"""
        
        if ensemble_type == 'super_optimal':
            target_ctr = kwargs.get('target_ctr', 0.0201)
            optimization_method = kwargs.get('optimization_method', 'ultra_combined')
            ensemble = CTRSuperOptimalEnsemble(target_ctr, optimization_method)
            self.super_optimal_ensemble = ensemble
        
        elif ensemble_type == 'advanced_stabilized':
            diversification_method = kwargs.get('diversification_method', 'advanced_rank_weighted')
            ensemble = CTRAdvancedStabilizedEnsemble(diversification_method)
        
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
        calibration_info = {}
        
        for name, model in self.base_models.items():
            try:
                start_time = time.time()
                
                # 캘리브레이션이 적용된 예측 사용
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    pred = model.predict_proba(X)
                    calibration_info[name] = {'calibrated': True, 'method': getattr(model.calibrator, 'best_method', 'unknown') if hasattr(model, 'calibrator') and model.calibrator else 'unknown'}
                else:
                    pred = model.predict_proba(X)
                    calibration_info[name] = {'calibrated': False, 'method': 'none'}
                
                prediction_time = time.time() - start_time
                
                base_predictions[name] = pred
                logger.info(f"{name} 모델 예측 완료 ({prediction_time:.2f}초) - "
                          f"Calibration: {'Yes' if calibration_info[name]['calibrated'] else 'No'}")
                
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
                calibration_info[name] = {'calibrated': False, 'method': 'error'}
        
        # 각 앙상블 학습
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                start_time = time.time()
                ensemble.fit(X, y, base_predictions)
                training_time = time.time() - start_time
                
                calibration_status = "Yes" if ensemble.is_calibrated else "No"
                logger.info(f"{ensemble_type} 앙상블 학습 완료 ({training_time:.2f}초) - "
                          f"Ensemble Calibration: {calibration_status}")
                
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 학습 실패: {str(e)}")
        
        # 캘리브레이션 정보 저장
        self.calibration_manager = calibration_info
        
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
                # 캘리브레이션된 예측 사용
                pred = model.predict_proba(X_val)
                base_predictions[name] = pred
                
                score = self.metrics_calculator.combined_score(y_val, pred)
                ctr_optimized_score = self.metrics_calculator.ctr_optimized_score(y_val, pred)
                
                results[f"base_{name}"] = score
                results[f"base_{name}_ctr_optimized"] = ctr_optimized_score
                
                # 캘리브레이션 효과 분석
                calibration_status = self.calibration_manager.get(name, {})
                if calibration_status.get('calibrated', False):
                    # 원본 예측과 비교 (가능한 경우)
                    if hasattr(model, 'predict_proba_raw'):
                        try:
                            raw_pred = model.predict_proba_raw(X_val)
                            raw_score = self.metrics_calculator.combined_score(y_val, raw_pred)
                            calibration_improvement = score - raw_score
                            results[f"base_{name}_calibration_improvement"] = calibration_improvement
                            logger.info(f"{name} 캘리브레이션 효과: {calibration_improvement:+.4f}")
                        except:
                            pass
                
            except Exception as e:
                logger.error(f"{name} 모델 검증 예측 실패: {str(e)}")
                results[f"base_{name}"] = 0.0
                results[f"base_{name}_ctr_optimized"] = 0.0
        
        # 앙상블 성능 평가
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    # 원본 앙상블 예측
                    raw_ensemble_pred = ensemble.predict_proba(base_predictions)
                    
                    # 캘리브레이션 적용 예측 (있는 경우)
                    if ensemble.is_calibrated:
                        calibrated_ensemble_pred = ensemble.predict_proba_calibrated(base_predictions)
                    else:
                        calibrated_ensemble_pred = raw_ensemble_pred
                    
                    # 성능 지표 계산
                    combined_score = self.metrics_calculator.combined_score(y_val, calibrated_ensemble_pred)
                    ctr_optimized_score = self.metrics_calculator.ctr_optimized_score(y_val, calibrated_ensemble_pred)
                    ap_score = self.metrics_calculator.average_precision(y_val, calibrated_ensemble_pred)
                    wll_score = self.metrics_calculator.weighted_log_loss(y_val, calibrated_ensemble_pred)
                    
                    results[f"ensemble_{ensemble_type}"] = combined_score
                    results[f"ensemble_{ensemble_type}_ctr_optimized"] = ctr_optimized_score
                    results[f"ensemble_{ensemble_type}_ap"] = ap_score
                    results[f"ensemble_{ensemble_type}_wll"] = wll_score
                    
                    # 캘리브레이션 효과 분석
                    if ensemble.is_calibrated:
                        try:
                            raw_combined_score = self.metrics_calculator.combined_score(y_val, raw_ensemble_pred)
                            calibration_improvement = combined_score - raw_combined_score
                            results[f"ensemble_{ensemble_type}_calibration_improvement"] = calibration_improvement
                            logger.info(f"{ensemble_type} 앙상블 캘리브레이션 효과: {calibration_improvement:+.4f}")
                        except:
                            pass
                    
                    logger.info(f"{ensemble_type} 앙상블 Combined Score: {combined_score:.4f}")
                    logger.info(f"{ensemble_type} 앙상블 CTR Optimized Score: {ctr_optimized_score:.4f}")
                    logger.info(f"{ensemble_type} 앙상블 Calibration: {'Yes' if ensemble.is_calibrated else 'No'}")
                    
                    # CTR 분석
                    predicted_ctr = calibrated_ensemble_pred.mean()
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
            ensemble_results = {k: v for k, v in results.items() 
                              if k.startswith('ensemble_') and not k.endswith('_ctr_optimized') 
                              and not k.endswith('_ap') and not k.endswith('_wll') 
                              and not k.endswith('_calibration_improvement')}
            
            if ensemble_results:
                best_name = max(ensemble_results, key=ensemble_results.get)
                best_score = ensemble_results[best_name]
                
                ensemble_type = best_name.replace('ensemble_', '')
                self.best_ensemble = self.ensembles[ensemble_type]
                
                logger.info(f"최고 성능 앙상블: {ensemble_type} (Combined Score: {best_score:.4f})")
                
                if best_score >= self.target_combined_score:
                    logger.info(f"목표 Combined Score {self.target_combined_score}+ 달성!")
                else:
                    logger.info(f"목표까지 {self.target_combined_score - best_score:.4f} 부족")
                    
                # 캘리브레이션 상태 확인
                best_ensemble_obj = self.ensembles[ensemble_type]
                if best_ensemble_obj.is_calibrated:
                    logger.info("최고 앙상블에 캘리브레이션 적용됨")
            else:
                logger.info("평가 가능한 앙상블이 없습니다.")
                self.best_ensemble = None
        
        self.ensemble_results = results
        
        # 캘리브레이션 효과 요약
        calibration_improvements = [v for k, v in results.items() if k.endswith('_calibration_improvement')]
        if calibration_improvements:
            avg_improvement = np.mean(calibration_improvements)
            logger.info(f"평균 캘리브레이션 효과: {avg_improvement:+.4f}")
        
        return results
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """최고 성능 앙상블로 예측"""
        
        # 기본 모델 예측 수집
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)  # 캘리브레이션된 예측 사용
                base_predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
        
        if self.best_ensemble is None:
            # 기본 모델 중 최고 성능 선택
            best_model_name = None
            best_score = 0
            
            for result_name, score in self.ensemble_results.items():
                if (result_name.startswith('base_') and 
                    not result_name.endswith('_ctr_optimized') and 
                    not result_name.endswith('_calibration_improvement') and 
                    score > best_score):
                    best_score = score
                    best_model_name = result_name.replace('base_', '')
            
            if best_model_name and best_model_name in self.base_models:
                logger.info(f"최고 성능 기본 모델 사용: {best_model_name}")
                return self.base_models[best_model_name].predict_proba(X)
            else:
                # 평균 앙상블
                logger.info("평균 앙상블 사용")
                return np.mean(list(base_predictions.values()), axis=0)
        
        # 최고 앙상블 예측
        if self.best_ensemble.is_calibrated:
            return self.best_ensemble.predict_proba_calibrated(base_predictions)
        else:
            return self.best_ensemble.predict_proba(base_predictions)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """앙상블 요약 정보"""
        
        target_achieved_count = sum(
            1 for key, score in self.ensemble_results.items()
            if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') 
            and not key.endswith('_ap') and not key.endswith('_wll')
            and not key.endswith('_calibration_improvement')
            and score >= self.target_combined_score
        )
        
        # 캘리브레이션 분석
        calibrated_base_models = sum(
            1 for info in self.calibration_manager.values()
            if info.get('calibrated', False)
        )
        
        calibrated_ensembles = sum(
            1 for ensemble in self.ensembles.values()
            if hasattr(ensemble, 'is_calibrated') and ensemble.is_calibrated
        )
        
        calibration_improvements = [
            v for k, v in self.ensemble_results.items()
            if k.endswith('_calibration_improvement') and v > 0
        ]
        
        return {
            'total_ensembles': len(self.ensembles),
            'fitted_ensembles': sum(1 for e in self.ensembles.values() if e.is_fitted),
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'best_ensemble_calibrated': self.best_ensemble.is_calibrated if self.best_ensemble else False,
            'ensemble_results': self.ensemble_results,
            'base_models_count': len(self.base_models),
            'calibrated_base_models': calibrated_base_models,
            'calibrated_ensembles': calibrated_ensembles,
            'super_optimal_ensemble_available': self.super_optimal_ensemble is not None and self.super_optimal_ensemble.is_fitted,
            'ensemble_types': list(self.ensembles.keys()),
            'target_combined_score': self.target_combined_score,
            'target_achieved_count': target_achieved_count,
            'target_achieved': target_achieved_count > 0,
            'best_combined_score': max(
                (score for key, score in self.ensemble_results.items()
                 if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') 
                 and not key.endswith('_ap') and not key.endswith('_wll')
                 and not key.endswith('_calibration_improvement')),
                default=0.0
            ),
            'calibration_analysis': {
                'base_models_calibration_rate': calibrated_base_models / max(len(self.base_models), 1),
                'ensemble_calibration_rate': calibrated_ensembles / max(len(self.ensembles), 1),
                'positive_calibration_improvements': len(calibration_improvements),
                'avg_calibration_improvement': np.mean(calibration_improvements) if calibration_improvements else 0.0,
                'calibration_methods_used': {
                    info.get('method', 'unknown'): 1 
                    for info in self.calibration_manager.values() 
                    if info.get('calibrated', False)
                }
            }
        }

# 기존 클래스명 유지 (하위 호환성)
CTROptimalEnsemble = CTRSuperOptimalEnsemble
CTRStabilizedEnsemble = CTRAdvancedStabilizedEnsemble  
CTREnsembleManager = CTRSuperEnsembleManager
EnsembleManager = CTRSuperEnsembleManager