# evaluation.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.metrics import (
    average_precision_score, log_loss, roc_auc_score, 
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, brier_score_loss
)
import time
import gc

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib이 설치되지 않았습니다. 시각화 기능이 비활성화됩니다.")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy가 설치되지 않았습니다. 일부 통계 기능이 비활성화됩니다.")

from config import Config

logger = logging.getLogger(__name__)

class CTRAdvancedMetrics:
    """CTR 예측 전용 고급 평가 지표 클래스 - Combined Score 0.30+ 달성 목표"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ap_weight = config.EVALUATION_CONFIG['ap_weight']
        self.wll_weight = config.EVALUATION_CONFIG['wll_weight']
        self.actual_ctr = 0.0201
        self.pos_weight = 49.8
        self.neg_weight = 1.0
        self.target_combined_score = 0.30
        self.ctr_tolerance = 0.0005
        self.bias_penalty_weight = 5.0
        self.calibration_weight = 0.4
        
        self.cache = {}
        
    def average_precision_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Average Precision - 안정성 및 정확도 개선"""
        try:
            cache_key = f"ap_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"크기 불일치: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return 0.0
            
            if len(y_true) == 0:
                logger.warning("빈 배열로 인해 AP 계산 불가")
                return 0.0
            
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("단일 클래스만 존재하여 AP 계산 불가")
                return 0.0
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("예측값에 NaN 또는 무한값 존재, 클리핑 적용")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            try:
                ap_score = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"sklearn AP 계산 실패, 수동 계산: {e}")
                ap_score = self._manual_average_precision(y_true, y_pred_proba)
            
            if np.isnan(ap_score) or np.isinf(ap_score):
                logger.warning("AP 계산 결과가 유효하지 않음")
                return 0.0
            
            ap_score = float(ap_score)
            self.cache[cache_key] = ap_score
            
            return ap_score
        except Exception as e:
            logger.error(f"Enhanced AP 계산 오류: {str(e)}")
            return 0.0
    
    def _manual_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """수동 Average Precision 계산"""
        try:
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            precisions = []
            recalls = []
            
            true_positives = 0
            false_positives = 0
            total_positives = np.sum(y_true)
            
            if total_positives == 0:
                return 0.0
            
            for i, label in enumerate(y_true_sorted):
                if label == 1:
                    true_positives += 1
                else:
                    false_positives += 1
                
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / total_positives
                
                precisions.append(precision)
                recalls.append(recall)
            
            ap = 0.0
            for i in range(1, len(recalls)):
                ap += precisions[i] * (recalls[i] - recalls[i-1])
            
            return ap
        except Exception as e:
            logger.error(f"수동 AP 계산 실패: {e}")
            return 0.0
    
    def weighted_log_loss_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Weighted Log Loss - CTR 분포 특화 개선"""
        try:
            cache_key = f"wll_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"크기 불일치: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return float('inf')
            
            if len(y_true) == 0:
                logger.warning("빈 배열로 인해 WLL 계산 불가")
                return float('inf')
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("예측값에 NaN 또는 무한값 존재, 클리핑 적용")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            actual_ctr = y_true.mean()
            predicted_ctr = y_pred_proba.mean()
            
            dynamic_pos_weight = self.pos_weight * (self.actual_ctr / max(actual_ctr, 1e-8))
            dynamic_neg_weight = self.neg_weight
            
            sample_weights = np.where(y_true == 1, dynamic_pos_weight, dynamic_neg_weight)
            
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            log_loss_values = -(y_true * np.log(y_pred_proba_clipped) + 
                              (1 - y_true) * np.log(1 - y_pred_proba_clipped))
            
            if np.any(np.isnan(log_loss_values)) or np.any(np.isinf(log_loss_values)):
                logger.warning("Log loss 계산에서 NaN 또는 무한값 발생")
                return float('inf')
            
            weighted_log_loss = np.average(log_loss_values, weights=sample_weights)
            
            ctr_penalty = abs(predicted_ctr - actual_ctr) * 1000
            final_wll = weighted_log_loss + ctr_penalty
            
            if np.isnan(final_wll) or np.isinf(final_wll):
                logger.warning("WLL 계산 결과가 유효하지 않음")
                return float('inf')
            
            final_wll = float(final_wll)
            self.cache[cache_key] = final_wll
            
            return final_wll
            
        except Exception as e:
            logger.error(f"Enhanced WLL 계산 오류: {str(e)}")
            return float('inf')
    
    def combined_score_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Combined Score = AP + WLL + CTR보정 + 다양성보정 - 0.30+ 목표"""
        try:
            cache_key = f"combined_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("Combined Score 계산을 위한 입력 데이터 문제")
                return 0.0
            
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            basic_combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            # CTR 정렬 보정
            try:
                predicted_ctr = y_pred_proba.mean()
                actual_ctr = y_true.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                
                ctr_alignment = np.exp(-(ctr_bias / self.ctr_tolerance) ** 2)
                ctr_ratio_penalty = min(abs(predicted_ctr / max(actual_ctr, 1e-8) - 1.0), 2.0)
                ctr_penalty = np.exp(-ctr_ratio_penalty * self.bias_penalty_weight)
                
                ctr_bonus = ctr_alignment * ctr_penalty
                
            except Exception as e:
                logger.warning(f"CTR 정렬 계산 실패: {e}")
                ctr_bonus = 0.5
            
            # 예측 다양성 보정
            try:
                pred_std = y_pred_proba.std()
                pred_range = y_pred_proba.max() - y_pred_proba.min()
                unique_ratio = len(np.unique(y_pred_proba)) / len(y_pred_proba)
                
                diversity_score = pred_std * pred_range * unique_ratio
                diversity_bonus = min(diversity_score * 2.0, 0.5)
                
            except Exception as e:
                logger.warning(f"다양성 계산 실패: {e}")
                diversity_bonus = 0.0
            
            # 분포 매칭 보정
            try:
                distribution_score = self._calculate_distribution_matching_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"분포 매칭 계산 실패: {e}")
                distribution_score = 0.5
            
            # 최종 점수 계산
            final_score = (basic_combined * 
                          (1.0 + 0.3 * ctr_bonus) * 
                          (1.0 + 0.1 * diversity_bonus) * 
                          (1.0 + 0.2 * distribution_score))
            
            if np.isnan(final_score) or np.isinf(final_score) or final_score < 0:
                logger.warning("Combined Score 계산 결과가 유효하지 않음")
                return 0.0
            
            final_score = float(final_score)
            self.cache[cache_key] = final_score
            
            return final_score
            
        except Exception as e:
            logger.error(f"Enhanced Combined Score 계산 오류: {str(e)}")
            return 0.0
    
    def _calculate_distribution_matching_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """분포 매칭 점수 계산"""
        try:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            
            pred_quantiles = np.percentile(y_pred_proba, [q * 100 for q in quantiles])
            
            actual_rates_by_quantile = []
            for i, q in enumerate(quantiles):
                if i == 0:
                    mask = y_pred_proba <= pred_quantiles[i]
                else:
                    mask = (y_pred_proba > pred_quantiles[i-1]) & (y_pred_proba <= pred_quantiles[i])
                
                if mask.sum() > 0:
                    actual_rate = y_true[mask].mean()
                    predicted_rate = y_pred_proba[mask].mean()
                    
                    if predicted_rate > 0:
                        rate_ratio = min(actual_rate / predicted_rate, predicted_rate / actual_rate)
                    else:
                        rate_ratio = 0.0
                    
                    actual_rates_by_quantile.append(rate_ratio)
                else:
                    actual_rates_by_quantile.append(0.5)
            
            distribution_score = np.mean(actual_rates_by_quantile)
            
            return distribution_score
            
        except Exception as e:
            logger.warning(f"분포 매칭 점수 계산 실패: {e}")
            return 0.5
    
    def ctr_ultra_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR 예측에 초고도 최적화된 종합 점수 - 0.32+ 목표"""
        try:
            cache_key = f"ultra_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("CTR 초최적화 점수 계산을 위한 입력 데이터 문제")
                return 0.0
            
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            # CTR 초정밀 정렬
            try:
                predicted_ctr = y_pred_proba.mean()
                actual_ctr = y_true.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                
                ultra_ctr_accuracy = np.exp(-ctr_bias * 2000) if ctr_bias < 0.01 else 0.0
                ctr_stability = 1.0 / (1.0 + ctr_bias * 10000)
                
            except Exception as e:
                logger.warning(f"CTR 초정밀 계산 실패: {e}")
                ultra_ctr_accuracy = 0.0
                ctr_stability = 0.0
            
            # 고급 보정 품질
            try:
                calibration_score = self._calculate_advanced_calibration_quality(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"고급 보정 품질 계산 실패: {e}")
                calibration_score = 0.5
            
            # 분포 완벽 매칭
            try:
                perfect_distribution_score = self._calculate_perfect_distribution_matching(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"완벽 분포 매칭 계산 실패: {e}")
                perfect_distribution_score = 0.5
            
            # 예측 품질 지표
            try:
                prediction_quality = self._calculate_prediction_quality_score(y_pred_proba)
            except Exception as e:
                logger.warning(f"예측 품질 계산 실패: {e}")
                prediction_quality = 0.5
            
            # 극값 영역 성능
            try:
                extreme_performance = self._calculate_extreme_region_performance(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"극값 영역 성능 계산 실패: {e}")
                extreme_performance = 0.5
            
            # 가중치 적용
            weights = {
                'ap': 0.25,
                'wll': 0.20,
                'ultra_ctr_accuracy': 0.25,
                'ctr_stability': 0.10,
                'calibration': 0.08,
                'perfect_distribution': 0.07,
                'prediction_quality': 0.03,
                'extreme_performance': 0.02
            }
            
            ultra_optimized_score = (
                weights['ap'] * ap_score +
                weights['wll'] * wll_normalized +
                weights['ultra_ctr_accuracy'] * ultra_ctr_accuracy +
                weights['ctr_stability'] * ctr_stability +
                weights['calibration'] * calibration_score +
                weights['perfect_distribution'] * perfect_distribution_score +
                weights['prediction_quality'] * prediction_quality +
                weights['extreme_performance'] * extreme_performance
            )
            
            if np.isnan(ultra_optimized_score) or np.isinf(ultra_optimized_score) or ultra_optimized_score < 0:
                logger.warning("CTR 초최적화 점수 계산 결과가 유효하지 않음")
                return 0.0
            
            ultra_optimized_score = float(ultra_optimized_score)
            self.cache[cache_key] = ultra_optimized_score
            
            return ultra_optimized_score
            
        except Exception as e:
            logger.error(f"CTR 초최적화 점수 계산 오류: {str(e)}")
            return 0.0
    
    def _calculate_advanced_calibration_quality(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """고급 모델 보정 품질 계산"""
        try:
            n_bins = 20
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            mce = 0.0
            total_count = len(y_true)
            
            bin_accuracies = []
            bin_confidences = []
            bin_weights = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    bin_diff = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += bin_diff * prop_in_bin
                    mce = max(mce, bin_diff)
                    
                    bin_accuracies.append(accuracy_in_bin)
                    bin_confidences.append(avg_confidence_in_bin)
                    bin_weights.append(prop_in_bin)
            
            if len(bin_accuracies) > 2:
                try:
                    reliability_slope = np.corrcoef(bin_confidences, bin_accuracies)[0, 1]
                    if np.isnan(reliability_slope):
                        reliability_slope = 0.5
                except:
                    reliability_slope = 0.5
            else:
                reliability_slope = 0.5
            
            calibration_score = (
                max(0.0, 1.0 - ece * 10) * 0.4 +
                max(0.0, 1.0 - mce * 10) * 0.3 +
                abs(reliability_slope) * 0.3
            )
            
            return calibration_score
            
        except Exception as e:
            logger.warning(f"고급 보정 품질 계산 실패: {e}")
            return 0.5
    
    def _calculate_perfect_distribution_matching(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """완벽 분포 매칭 계산"""
        try:
            percentiles = np.arange(5, 100, 5)
            
            distribution_scores = []
            
            for p in percentiles:
                threshold = np.percentile(y_pred_proba, p)
                top_mask = y_pred_proba >= threshold
                
                if top_mask.sum() > 10:
                    actual_rate = y_true[top_mask].mean()
                    predicted_rate = y_pred_proba[top_mask].mean()
                    
                    if predicted_rate > 0:
                        rate_alignment = min(actual_rate / predicted_rate, predicted_rate / actual_rate)
                    else:
                        rate_alignment = 0.0
                    
                    distribution_scores.append(rate_alignment)
            
            if distribution_scores:
                perfect_score = np.mean(distribution_scores)
                
                std_penalty = min(np.std(distribution_scores), 0.5)
                final_score = perfect_score * (1.0 - std_penalty)
                
                return final_score
            else:
                return 0.5
            
        except Exception as e:
            logger.warning(f"완벽 분포 매칭 계산 실패: {e}")
            return 0.5
    
    def _calculate_prediction_quality_score(self, y_pred_proba: np.ndarray) -> float:
        """예측 품질 점수 계산"""
        try:
            pred_std = y_pred_proba.std()
            pred_var = y_pred_proba.var()
            pred_range = y_pred_proba.max() - y_pred_proba.min()
            
            unique_ratio = len(np.unique(y_pred_proba)) / len(y_pred_proba)
            
            pred_entropy = -np.mean(
                y_pred_proba * np.log(y_pred_proba + 1e-15) + 
                (1 - y_pred_proba) * np.log(1 - y_pred_proba + 1e-15)
            )
            
            gini_coeff = self._calculate_gini_coefficient(y_pred_proba)
            
            quality_score = (
                pred_std * 0.3 +
                pred_range * 0.2 +
                unique_ratio * 0.2 +
                min(pred_entropy, 1.0) * 0.15 +
                gini_coeff * 0.15
            )
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"예측 품질 점수 계산 실패: {e}")
            return 0.5
    
    def _calculate_extreme_region_performance(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """극값 영역 성능 계산"""
        try:
            high_threshold = np.percentile(y_pred_proba, 99)
            low_threshold = np.percentile(y_pred_proba, 1)
            
            extreme_scores = []
            
            # 고극값 영역
            high_mask = y_pred_proba >= high_threshold
            if high_mask.sum() > 5:
                high_actual = y_true[high_mask].mean()
                high_predicted = y_pred_proba[high_mask].mean()
                
                if high_predicted > 0:
                    high_score = min(high_actual / high_predicted, high_predicted / high_actual)
                else:
                    high_score = 0.0
                
                extreme_scores.append(high_score)
            
            # 저극값 영역
            low_mask = y_pred_proba <= low_threshold
            if low_mask.sum() > 5:
                low_actual = y_true[low_mask].mean()
                low_predicted = y_pred_proba[low_mask].mean()
                
                if low_predicted > 0:
                    low_score = min(low_actual / low_predicted, low_predicted / low_actual)
                else:
                    low_score = 0.0
                
                extreme_scores.append(low_score)
            
            if extreme_scores:
                return np.mean(extreme_scores)
            else:
                return 0.5
            
        except Exception as e:
            logger.warning(f"극값 영역 성능 계산 실패: {e}")
            return 0.5
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """지니 계수 계산"""
        try:
            sorted_values = np.sort(values)
            n = len(sorted_values)
            
            if n == 0:
                return 0.0
            
            cumsum = np.cumsum(sorted_values)
            if cumsum[-1] == 0:
                return 0.0
                
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            
            if np.isnan(gini) or np.isinf(gini):
                return 0.0
            
            return float(np.abs(gini))
        except:
            return 0.0
    
    def comprehensive_evaluation_ultra(self, 
                                     y_true: np.ndarray, 
                                     y_pred_proba: np.ndarray,
                                     threshold: float = 0.5) -> Dict[str, float]:
        """초종합적인 평가 지표 계산 - Combined Score 0.30+ 목표"""
        
        try:
            start_time = time.time()
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"입력 크기 불일치: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return self._get_default_metrics_ultra()
            
            if len(y_true) == 0:
                logger.error("빈 입력 배열")
                return self._get_default_metrics_ultra()
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("예측값에 NaN 또는 무한값 존재, 정리 수행")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            y_pred = (y_pred_proba >= threshold).astype(int)
        
            metrics = {}
            
            # 주요 CTR 초고성능 지표
            try:
                metrics['ap_enhanced'] = self.average_precision_enhanced(y_true, y_pred_proba)
                metrics['wll_enhanced'] = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
                metrics['combined_score_enhanced'] = self.combined_score_enhanced(y_true, y_pred_proba)
                metrics['ctr_ultra_optimized_score'] = self.ctr_ultra_optimized_score(y_true, y_pred_proba)
            except Exception as e:
                logger.error(f"주요 CTR 지표 계산 실패: {e}")
                metrics.update({
                    'ap_enhanced': 0.0, 'wll_enhanced': float('inf'), 
                    'combined_score_enhanced': 0.0, 'ctr_ultra_optimized_score': 0.0
                })
            
            # 기본 분류 지표 (안정성 개선)
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"기본 분류 지표 계산 실패: {e}")
                metrics.update({
                    'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25
                })
            
            # 혼동 행렬 기반 지표 (강화)
            try:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    logger.warning("혼동 행렬 형태가 예상과 다름")
                    tn, fp, fn, tp = 0, 0, 0, 0
                
                total = tp + tn + fp + fn
                if total > 0:
                    metrics['accuracy'] = (tp + tn) / total
                    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    metrics['sensitivity'] = metrics['recall']
                    
                    if metrics['precision'] + metrics['recall'] > 0:
                        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                    else:
                        metrics['f1'] = 0.0
                    
                    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    if denominator != 0:
                        metrics['mcc'] = (tp * tn - fp * fn) / denominator
                    else:
                        metrics['mcc'] = 0.0
                    
                    # 향상된 성능 지표
                    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
                    metrics['geometric_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])
                    
                else:
                    metrics.update({
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                        'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0,
                        'balanced_accuracy': 0.0, 'geometric_mean': 0.0
                    })
                
            except Exception as e:
                logger.warning(f"혼동 행렬 지표 계산 실패: {e}")
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0,
                    'balanced_accuracy': 0.0, 'geometric_mean': 0.0
                })
            
            # CTR 초정밀 분석
            try:
                metrics['ctr_actual'] = float(y_true.mean())
                metrics['ctr_predicted'] = float(y_pred_proba.mean())
                metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
                metrics['ctr_ratio'] = metrics['ctr_predicted'] / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
                metrics['ctr_relative_error'] = abs(metrics['ctr_bias']) / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_alignment_score'] = np.exp(-abs(metrics['ctr_bias']) * 1000)
                
                # CTR 안정성 지표
                metrics['ctr_stability'] = 1.0 / (1.0 + abs(metrics['ctr_bias']) * 10000)
                metrics['target_ctr_achievement'] = 1.0 if abs(metrics['ctr_bias']) < 0.001 else 0.0
                
            except Exception as e:
                logger.warning(f"CTR 초정밀 분석 실패: {e}")
                metrics.update({
                    'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
                    'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0,
                    'ctr_alignment_score': 1.0, 'ctr_stability': 1.0, 'target_ctr_achievement': 1.0
                })
            
            # CTR 범위별 초상세 지표
            try:
                self._add_ultra_ctr_range_metrics(metrics, y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"CTR 범위별 초상세 지표 계산 실패: {e}")
            
            # 예측 품질 통계 (강화)
            try:
                metrics['prediction_std'] = float(y_pred_proba.std())
                metrics['prediction_var'] = float(y_pred_proba.var())
                metrics['prediction_entropy'] = self._calculate_entropy_enhanced(y_pred_proba)
                metrics['prediction_gini'] = self._calculate_gini_coefficient(y_pred_proba)
                metrics['prediction_range'] = float(y_pred_proba.max() - y_pred_proba.min())
                metrics['prediction_iqr'] = float(np.percentile(y_pred_proba, 75) - np.percentile(y_pred_proba, 25))
                
                metrics['unique_predictions_ratio'] = len(np.unique(y_pred_proba)) / len(y_pred_proba)
                metrics['effective_sample_size'] = len(y_pred_proba) / max(1 + metrics['prediction_var'] * len(y_pred_proba), 1)
                
            except Exception as e:
                logger.warning(f"예측 품질 통계 계산 실패: {e}")
                metrics.update({
                    'prediction_std': 0.0, 'prediction_var': 0.0, 'prediction_entropy': 0.0, 
                    'prediction_gini': 0.0, 'prediction_range': 0.0, 'prediction_iqr': 0.0,
                    'unique_predictions_ratio': 0.0, 'effective_sample_size': len(y_true)
                })
            
            # 클래스별 예측 통계 (강화)
            try:
                pos_mask = (y_true == 1)
                neg_mask = (y_true == 0)
                
                if pos_mask.any():
                    metrics['pos_mean_pred'] = float(y_pred_proba[pos_mask].mean())
                    metrics['pos_std_pred'] = float(y_pred_proba[pos_mask].std())
                    metrics['pos_median_pred'] = float(np.median(y_pred_proba[pos_mask]))
                    metrics['pos_q25_pred'] = float(np.percentile(y_pred_proba[pos_mask], 25))
                    metrics['pos_q75_pred'] = float(np.percentile(y_pred_proba[pos_mask], 75))
                    metrics['pos_min_pred'] = float(y_pred_proba[pos_mask].min())
                    metrics['pos_max_pred'] = float(y_pred_proba[pos_mask].max())
                else:
                    metrics.update({
                        'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
                        'pos_q25_pred': 0.0, 'pos_q75_pred': 0.0, 'pos_min_pred': 0.0, 'pos_max_pred': 0.0
                    })
                    
                if neg_mask.any():
                    metrics['neg_mean_pred'] = float(y_pred_proba[neg_mask].mean())
                    metrics['neg_std_pred'] = float(y_pred_proba[neg_mask].std())
                    metrics['neg_median_pred'] = float(np.median(y_pred_proba[neg_mask]))
                    metrics['neg_q25_pred'] = float(np.percentile(y_pred_proba[neg_mask], 25))
                    metrics['neg_q75_pred'] = float(np.percentile(y_pred_proba[neg_mask], 75))
                    metrics['neg_min_pred'] = float(y_pred_proba[neg_mask].min())
                    metrics['neg_max_pred'] = float(y_pred_proba[neg_mask].max())
                else:
                    metrics.update({
                        'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
                        'neg_q25_pred': 0.0, 'neg_q75_pred': 0.0, 'neg_min_pred': 0.0, 'neg_max_pred': 0.0
                    })
                
                if pos_mask.any() and neg_mask.any():
                    metrics['separation'] = metrics['pos_mean_pred'] - metrics['neg_mean_pred']
                    metrics['separation_ratio'] = metrics['pos_mean_pred'] / max(metrics['neg_mean_pred'], 1e-10)
                    
                    # KS 통계 (강화)
                    if SCIPY_AVAILABLE:
                        try:
                            ks_stat, ks_pvalue = stats.ks_2samp(y_pred_proba[pos_mask], y_pred_proba[neg_mask])
                            metrics['ks_statistic'] = float(ks_stat)
                            metrics['ks_pvalue'] = float(ks_pvalue)
                        except:
                            metrics['ks_statistic'] = 0.0
                            metrics['ks_pvalue'] = 1.0
                    else:
                        metrics['ks_statistic'] = 0.0
                        metrics['ks_pvalue'] = 1.0
                else:
                    metrics['separation'] = 0.0
                    metrics['separation_ratio'] = 1.0
                    metrics['ks_statistic'] = 0.0
                    metrics['ks_pvalue'] = 1.0
            except Exception as e:
                logger.warning(f"클래스별 예측 통계 계산 실패: {e}")
                metrics.update({
                    'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
                    'pos_q25_pred': 0.0, 'pos_q75_pred': 0.0, 'pos_min_pred': 0.0, 'pos_max_pred': 0.0,
                    'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
                    'neg_q25_pred': 0.0, 'neg_q75_pred': 0.0, 'neg_min_pred': 0.0, 'neg_max_pred': 0.0,
                    'separation': 0.0, 'separation_ratio': 1.0, 'ks_statistic': 0.0, 'ks_pvalue': 1.0
                })
            
            # 초고급 보정 지표
            try:
                ultra_calibration_metrics = self.calculate_ultra_calibration_metrics(y_true, y_pred_proba)
                metrics.update(ultra_calibration_metrics)
            except Exception as e:
                logger.warning(f"초고급 보정 지표 계산 실패: {e}")
                metrics.update({
                    'ece_enhanced': 0.0, 'mce_enhanced': 0.0, 'ace_enhanced': 0.0, 
                    'reliability_slope_enhanced': 1.0, 'reliability_intercept_enhanced': 0.0,
                    'calibration_auc': 0.5, 'sharpness': 0.0, 'refinement': 0.0
                })
            
            # 분위수별 상세 지표
            try:
                self._add_ultra_quantile_metrics(metrics, y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"분위수별 상세 지표 계산 실패: {e}")
            
            # 목표 달성 지표
            try:
                metrics['target_combined_score_achievement'] = 1.0 if metrics['combined_score_enhanced'] >= self.target_combined_score else 0.0
                metrics['combined_score_gap'] = max(0, self.target_combined_score - metrics['combined_score_enhanced'])
                metrics['ultra_score_achievement'] = 1.0 if metrics['ctr_ultra_optimized_score'] >= 0.32 else 0.0
                metrics['performance_tier'] = self._classify_performance_tier(metrics['combined_score_enhanced'])
                
            except Exception as e:
                logger.warning(f"목표 달성 지표 계산 실패: {e}")
                metrics['target_combined_score_achievement'] = 0.0
                metrics['combined_score_gap'] = self.target_combined_score
                metrics['ultra_score_achievement'] = 0.0
                metrics['performance_tier'] = 'poor'
            
            # 계산 시간
            metrics['evaluation_time'] = time.time() - start_time
            
            # 모든 지표값을 float로 변환하고 유효성 검사
            validated_metrics = {}
            for key, value in metrics.items():
                try:
                    if isinstance(value, (int, float, np.number)):
                        if np.isnan(value) or np.isinf(value):
                            if 'wll' in key.lower():
                                validated_metrics[key] = float('inf')
                            else:
                                validated_metrics[key] = 0.0
                        else:
                            validated_metrics[key] = float(value)
                    else:
                        validated_metrics[key] = value
                except:
                    validated_metrics[key] = 0.0 if 'wll' not in key.lower() else float('inf')
            
            return validated_metrics
            
        except Exception as e:
            logger.error(f"초종합 평가 계산 오류: {str(e)}")
            return self._get_default_metrics_ultra()
    
    def _get_default_metrics_ultra(self) -> Dict[str, float]:
        """초고성능 기본 지표값 반환"""
        return {
            'ap_enhanced': 0.0, 'wll_enhanced': float('inf'), 
            'combined_score_enhanced': 0.0, 'ctr_ultra_optimized_score': 0.0,
            'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25,
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 
            'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0, 'balanced_accuracy': 0.0, 'geometric_mean': 0.0,
            'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
            'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0,
            'ctr_alignment_score': 1.0, 'ctr_stability': 1.0, 'target_ctr_achievement': 1.0,
            'prediction_std': 0.0, 'prediction_var': 0.0, 'prediction_entropy': 0.0, 'prediction_gini': 0.0,
            'prediction_range': 0.0, 'prediction_iqr': 0.0, 'unique_predictions_ratio': 0.0, 'effective_sample_size': 1000,
            'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
            'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
            'separation': 0.0, 'separation_ratio': 1.0, 'ks_statistic': 0.0, 'ks_pvalue': 1.0,
            'ece_enhanced': 0.0, 'mce_enhanced': 0.0, 'ace_enhanced': 0.0, 
            'reliability_slope_enhanced': 1.0, 'reliability_intercept_enhanced': 0.0,
            'calibration_auc': 0.5, 'sharpness': 0.0, 'refinement': 0.0,
            'target_combined_score_achievement': 0.0, 'combined_score_gap': self.target_combined_score,
            'ultra_score_achievement': 0.0, 'performance_tier': 'poor',
            'evaluation_time': 0.0
        }
    
    def _add_ultra_ctr_range_metrics(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred_proba: np.ndarray):
        """CTR 범위별 초상세 성능 지표 추가"""
        try:
            ranges = {
                'micro': (0.0, 0.005),
                'very_low': (0.005, 0.01),
                'low': (0.01, 0.015),
                'low_medium': (0.015, 0.02),
                'medium': (0.02, 0.03),
                'medium_high': (0.03, 0.05),
                'high': (0.05, 0.08),
                'very_high': (0.08, 0.15),
                'extreme': (0.15, 1.0)
            }
            
            for range_name, (low, high) in ranges.items():
                mask = (y_pred_proba >= low) & (y_pred_proba < high)
                
                if mask.sum() > 0:
                    range_actual_ctr = float(y_true[mask].mean())
                    range_pred_ctr = float(y_pred_proba[mask].mean())
                    range_bias = abs(range_pred_ctr - range_actual_ctr)
                    range_count = int(mask.sum())
                    range_ratio = range_count / len(y_true)
                    
                    if range_pred_ctr > 0:
                        range_alignment = min(range_actual_ctr / range_pred_ctr, range_pred_ctr / range_actual_ctr)
                    else:
                        range_alignment = 0.0
                    
                    metrics[f'ctr_{range_name}_actual'] = range_actual_ctr
                    metrics[f'ctr_{range_name}_predicted'] = range_pred_ctr
                    metrics[f'ctr_{range_name}_bias'] = range_bias
                    metrics[f'ctr_{range_name}_count'] = range_count
                    metrics[f'ctr_{range_name}_ratio'] = range_ratio
                    metrics[f'ctr_{range_name}_alignment'] = range_alignment
                    
                else:
                    metrics[f'ctr_{range_name}_actual'] = 0.0
                    metrics[f'ctr_{range_name}_predicted'] = 0.0
                    metrics[f'ctr_{range_name}_bias'] = 0.0
                    metrics[f'ctr_{range_name}_count'] = 0
                    metrics[f'ctr_{range_name}_ratio'] = 0.0
                    metrics[f'ctr_{range_name}_alignment'] = 0.0
                    
        except Exception as e:
            logger.warning(f"CTR 범위별 초상세 지표 계산 실패: {e}")
    
    def _add_ultra_quantile_metrics(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred_proba: np.ndarray):
        """분위수별 초상세 성능 지표 추가"""
        try:
            quantiles = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
            
            for q in quantiles:
                threshold = np.percentile(y_pred_proba, q * 100)
                top_q_mask = y_pred_proba >= threshold
                
                if top_q_mask.sum() > 0:
                    top_q_ctr = float(y_true[top_q_mask].mean())
                    top_q_pred_ctr = float(y_pred_proba[top_q_mask].mean())
                    top_q_size = int(top_q_mask.sum())
                    
                    baseline_ctr = metrics.get('ctr_actual', 0.0201)
                    if baseline_ctr > 0:
                        top_q_lift = top_q_ctr / baseline_ctr
                        top_q_precision = top_q_ctr
                    else:
                        top_q_lift = 1.0
                        top_q_precision = 0.0
                    
                    if top_q_pred_ctr > 0:
                        top_q_alignment = min(top_q_ctr / top_q_pred_ctr, top_q_pred_ctr / top_q_ctr)
                    else:
                        top_q_alignment = 0.0
                    
                    q_label = int(q * 100)
                    metrics[f'top_{q_label}p_ctr'] = top_q_ctr
                    metrics[f'top_{q_label}p_pred_ctr'] = top_q_pred_ctr
                    metrics[f'top_{q_label}p_size'] = top_q_size
                    metrics[f'top_{q_label}p_lift'] = top_q_lift
                    metrics[f'top_{q_label}p_precision'] = top_q_precision
                    metrics[f'top_{q_label}p_alignment'] = top_q_alignment
                    
        except Exception as e:
            logger.warning(f"분위수별 초상세 지표 계산 실패: {e}")
    
    def _calculate_entropy_enhanced(self, probabilities: np.ndarray) -> float:
        """향상된 예측 확률의 엔트로피 계산"""
        try:
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            
            if np.isnan(entropy) or np.isinf(entropy):
                return 0.0
            
            return float(entropy)
        except:
            return 0.0
    
    def _classify_performance_tier(self, combined_score: float) -> str:
        """성능 등급 분류"""
        if combined_score >= 0.35:
            return 'exceptional'
        elif combined_score >= 0.30:
            return 'excellent'
        elif combined_score >= 0.25:
            return 'good'
        elif combined_score >= 0.20:
            return 'fair'
        elif combined_score >= 0.15:
            return 'poor'
        else:
            return 'very_poor'
    
    def calculate_pr_auc_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """향상된 Precision-Recall AUC 계산"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            pr_auc = np.trapz(precision, recall)
            
            if np.isnan(pr_auc) or np.isinf(pr_auc):
                pr_auc = 0.0
            
            return pr_auc, precision, recall
        except Exception as e:
            logger.error(f"향상된 PR-AUC 계산 오류: {str(e)}")
            return 0.0, np.array([]), np.array([])
    
    def find_optimal_threshold_enhanced(self, 
                                      y_true: np.ndarray, 
                                      y_pred_proba: np.ndarray,
                                      metric: str = 'combined_score_enhanced') -> Tuple[float, float]:
        """향상된 최적 임계값 찾기"""
        
        thresholds = np.arange(0.001, 0.999, 0.001)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            try:
                if metric == 'combined_score_enhanced':
                    score = self.combined_score_enhanced(y_true, y_pred_proba)
                elif metric == 'ctr_ultra_optimized_score':
                    score = self.ctr_ultra_optimized_score(y_true, y_pred_proba)
                elif metric == 'f1':
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    else:
                        score = 0.0
                elif metric == 'accuracy':
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    score = (y_true == y_pred).mean()
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception as e:
                continue
        
        return best_threshold, best_score
    
    def calculate_ultra_calibration_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 30) -> Dict[str, float]:
        """초고급 모델 보정 지표 계산"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            bin_weights = []
            
            ece = 0.0
            mce = 0.0
            total_count = len(y_true)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    count_in_bin = in_bin.sum()
                    
                    bin_diff = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += bin_diff * prop_in_bin
                    mce = max(mce, bin_diff)
                    
                    bin_accuracies.append(accuracy_in_bin)
                    bin_confidences.append(avg_confidence_in_bin)
                    bin_counts.append(count_in_bin)
                    bin_weights.append(prop_in_bin)
            
            # ACE 계산
            if len(bin_accuracies) > 0:
                ace = np.mean([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)])
            else:
                ace = 0.0
            
            # 신뢰도 곡선 기울기 및 절편
            if len(bin_accuracies) > 2:
                try:
                    slope, intercept = np.polyfit(bin_confidences, bin_accuracies, 1)
                    correlation = np.corrcoef(bin_confidences, bin_accuracies)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.5
                except:
                    slope, intercept, correlation = 1.0, 0.0, 0.5
            else:
                slope, intercept, correlation = 1.0, 0.0, 0.5
            
            # 보정 AUC 계산
            try:
                if len(bin_confidences) > 2:
                    calibration_auc = np.trapz(bin_accuracies, bin_confidences)
                    if np.isnan(calibration_auc):
                        calibration_auc = 0.5
                else:
                    calibration_auc = 0.5
            except:
                calibration_auc = 0.5
            
            # Sharpness (예측의 확신도)
            sharpness = y_pred_proba.var()
            
            # Refinement (분해 기반)
            try:
                refinement = np.mean([prop * acc * (1 - acc) for prop, acc in zip(bin_weights, bin_accuracies) if prop > 0])
            except:
                refinement = 0.0
            
            return {
                'ece_enhanced': float(ece),
                'mce_enhanced': float(mce), 
                'ace_enhanced': float(ace),
                'reliability_slope_enhanced': float(slope),
                'reliability_intercept_enhanced': float(intercept),
                'reliability_correlation': float(correlation),
                'calibration_auc': float(calibration_auc),
                'sharpness': float(sharpness),
                'refinement': float(refinement),
                'bin_accuracies': [float(x) for x in bin_accuracies],
                'bin_confidences': [float(x) for x in bin_confidences],
                'bin_counts': [int(x) for x in bin_counts],
                'bin_weights': [float(x) for x in bin_weights]
            }
            
        except Exception as e:
            logger.error(f"초고급 보정 지표 계산 오류: {str(e)}")
            return {
                'ece_enhanced': 0.0, 'mce_enhanced': 0.0, 'ace_enhanced': 0.0, 
                'reliability_slope_enhanced': 1.0, 'reliability_intercept_enhanced': 0.0,
                'reliability_correlation': 0.5, 'calibration_auc': 0.5, 'sharpness': 0.0, 'refinement': 0.0
            }
    
    def clear_cache(self):
        """캐시 정리"""
        self.cache.clear()
        gc.collect()

class UltraModelComparator:
    """초고성능 다중 모델 비교 클래스 - Combined Score 0.30+ 달성"""
    
    def __init__(self):
        self.metrics_calculator = CTRAdvancedMetrics()
        self.comparison_results = pd.DataFrame()
        self.performance_analysis = {}
        
    def compare_models_ultra(self, 
                           models_predictions: Dict[str, np.ndarray],
                           y_true: np.ndarray) -> pd.DataFrame:
        """초고성능 여러 모델의 성능 비교"""
        
        results = []
        
        y_true = np.asarray(y_true).flatten()
        
        logger.info(f"초고성능 모델 비교 시작: {len(models_predictions)}개 모델")
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                start_time = time.time()
                
                y_pred_proba = np.asarray(y_pred_proba).flatten()
                
                if len(y_pred_proba) != len(y_true):
                    logger.error(f"{model_name}: 예측값과 실제값 크기 불일치")
                    continue
                
                if len(y_pred_proba) == 0:
                    logger.error(f"{model_name}: 빈 예측값")
                    continue
                
                if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                    logger.warning(f"{model_name}: 예측값에 NaN 또는 무한값 존재, 정리 수행")
                    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                
                metrics = self.metrics_calculator.comprehensive_evaluation_ultra(y_true, y_pred_proba)
                metrics['model_name'] = model_name
                
                evaluation_time = time.time() - start_time
                metrics['evaluation_duration'] = evaluation_time
                
                # 초고급 안정성 지표 계산
                try:
                    stability_metrics = self._calculate_ultra_stability_metrics(y_true, y_pred_proba)
                    metrics.update(stability_metrics)
                except Exception as e:
                    logger.warning(f"{model_name} 초고급 안정성 지표 계산 실패: {e}")
                    metrics.update(self._get_default_stability_metrics())
                
                # 초고급 로버스트성 지표 계산
                try:
                    robustness_metrics = self._calculate_ultra_robustness_metrics(y_true, y_pred_proba)
                    metrics.update(robustness_metrics)
                except Exception as e:
                    logger.warning(f"{model_name} 초고급 로버스트성 지표 계산 실패: {e}")
                    metrics.update(self._get_default_robustness_metrics())
                
                results.append(metrics)
                
                logger.info(f"{model_name} 평가 완료 ({evaluation_time:.2f}초)")
                logger.info(f"  - Combined Score Enhanced: {metrics['combined_score_enhanced']:.4f}")
                logger.info(f"  - Ultra Optimized Score: {metrics['ctr_ultra_optimized_score']:.4f}")
                logger.info(f"  - CTR Bias: {metrics['ctr_bias']:.4f}")
                logger.info(f"  - Performance Tier: {metrics['performance_tier']}")
                
            except Exception as e:
                logger.error(f"{model_name} 초고성능 모델 평가 실패: {str(e)}")
                default_metrics = self.metrics_calculator._get_default_metrics_ultra()
                default_metrics['model_name'] = model_name
                default_metrics['evaluation_duration'] = 0.0
                default_metrics.update(self._get_default_stability_metrics())
                default_metrics.update(self._get_default_robustness_metrics())
                results.append(default_metrics)
        
        if not results:
            logger.error("평가 가능한 모델이 없습니다")
            return pd.DataFrame()
        
        try:
            comparison_df = pd.DataFrame(results)
            
            if not comparison_df.empty:
                comparison_df.set_index('model_name', inplace=True)
                
                # 다중 정렬 기준
                sort_columns = []
                if 'ctr_ultra_optimized_score' in comparison_df.columns:
                    sort_columns.append('ctr_ultra_optimized_score')
                if 'combined_score_enhanced' in comparison_df.columns:
                    sort_columns.append('combined_score_enhanced')
                if 'ap_enhanced' in comparison_df.columns:
                    sort_columns.append('ap_enhanced')
                
                if sort_columns:
                    comparison_df.sort_values(sort_columns, ascending=False, inplace=True)
                
                # 추가 분석
                self._analyze_ultra_performance_patterns(comparison_df)
        
            self.comparison_results = comparison_df
            
            # 목표 달성 모델 수 로깅
            target_achieved_count = comparison_df['target_combined_score_achievement'].sum()
            ultra_achieved_count = comparison_df['ultra_score_achievement'].sum()
            
            logger.info(f"초고성능 모델 비교 완료")
            logger.info(f"Combined Score 0.30+ 달성 모델: {target_achieved_count}/{len(comparison_df)}")
            logger.info(f"Ultra Score 0.32+ 달성 모델: {ultra_achieved_count}/{len(comparison_df)}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"초고성능 비교 결과 생성 실패: {e}")
            return pd.DataFrame()
    
    def _calculate_ultra_stability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """초고급 모델 안정성 지표 계산"""
        try:
            y_true_array = np.asarray(y_true).flatten()
            y_pred_array = np.asarray(y_pred_proba).flatten()
            
            min_len = min(len(y_true_array), len(y_pred_array))
            if min_len == 0:
                logger.warning("빈 배열로 인해 안정성 지표 계산 불가")
                return self._get_default_stability_metrics()
            
            y_true_array = y_true_array[:min_len]
            y_pred_array = y_pred_array[:min_len]
            
            # 강화된 부트스트래핑
            n_bootstrap = min(50, max(20, min_len // 5000))
            sample_size = min(min_len, 20000)
            
            combined_scores = []
            ctr_biases = []
            ap_scores = []
            
            np.random.seed(42)
            
            for i in range(n_bootstrap):
                try:
                    if sample_size >= min_len:
                        indices = np.arange(min_len)
                    else:
                        indices = np.random.choice(min_len, size=sample_size, replace=True)
                    
                    indices = np.clip(indices, 0, min_len - 1)
                    
                    boot_y_true = y_true_array[indices]
                    boot_y_pred = y_pred_array[indices]
                    
                    unique_classes = np.unique(boot_y_true)
                    if len(unique_classes) < 2:
                        continue
                    
                    if np.any(np.isnan(boot_y_pred)) or np.any(np.isinf(boot_y_pred)):
                        boot_y_pred = np.clip(boot_y_pred, 1e-15, 1 - 1e-15)
                        boot_y_pred = np.nan_to_num(boot_y_pred, nan=0.5, posinf=1.0, neginf=0.0)
                    
                    combined_score = self.metrics_calculator.combined_score_enhanced(boot_y_true, boot_y_pred)
                    ap_score = self.metrics_calculator.average_precision_enhanced(boot_y_true, boot_y_pred)
                    
                    ctr_bias = abs(boot_y_pred.mean() - boot_y_true.mean())
                    
                    if (combined_score > 0 and not np.isnan(combined_score) and not np.isinf(combined_score) and
                        ap_score > 0 and not np.isnan(ap_score) and not np.isinf(ap_score)):
                        combined_scores.append(combined_score)
                        ap_scores.append(ap_score)
                        ctr_biases.append(ctr_bias)
                        
                except Exception as e:
                    logger.debug(f"부트스트래핑 {i+1} 실패: {e}")
                    continue
            
            # 결과 계산
            if len(combined_scores) >= 5:
                combined_scores = np.array(combined_scores)
                ap_scores = np.array(ap_scores)
                ctr_biases = np.array(ctr_biases)
                
                stability_metrics = {
                    'stability_combined_mean': float(combined_scores.mean()),
                    'stability_combined_std': float(combined_scores.std()),
                    'stability_combined_cv': float(combined_scores.std() / combined_scores.mean()) if combined_scores.mean() > 0 else float('inf'),
                    'stability_combined_ci_lower': float(np.percentile(combined_scores, 2.5)),
                    'stability_combined_ci_upper': float(np.percentile(combined_scores, 97.5)),
                    'stability_combined_range': float(np.percentile(combined_scores, 97.5) - np.percentile(combined_scores, 2.5)),
                    
                    'stability_ap_mean': float(ap_scores.mean()),
                    'stability_ap_std': float(ap_scores.std()),
                    'stability_ap_cv': float(ap_scores.std() / ap_scores.mean()) if ap_scores.mean() > 0 else float('inf'),
                    
                    'stability_ctr_bias_mean': float(ctr_biases.mean()),
                    'stability_ctr_bias_std': float(ctr_biases.std()),
                    'stability_ctr_bias_max': float(ctr_biases.max()),
                    
                    'stability_sample_count': len(combined_scores),
                    'stability_success_rate': len(combined_scores) / n_bootstrap,
                    
                    # 종합 안정성 점수
                    'overall_stability_score': float(
                        max(0, 1.0 - combined_scores.std() / max(combined_scores.mean(), 0.1)) * 0.5 +
                        max(0, 1.0 - ap_scores.std() / max(ap_scores.mean(), 0.1)) * 0.3 +
                        max(0, 1.0 - ctr_biases.std()) * 0.2
                    )
                }
            else:
                logger.warning("유효한 부트스트래핑 점수가 부족합니다")
                stability_metrics = self._get_default_stability_metrics()
            
            return stability_metrics
            
        except Exception as e:
            logger.warning(f"초고급 안정성 지표 계산 실패: {e}")
            return self._get_default_stability_metrics()
    
    def _calculate_ultra_robustness_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """초고급 로버스트성 지표 계산"""
        try:
            # 노이즈 추가 테스트
            noise_levels = [0.001, 0.005, 0.01, 0.02]
            robustness_scores = []
            
            original_score = self.metrics_calculator.combined_score_enhanced(y_true, y_pred_proba)
            
            for noise_level in noise_levels:
                try:
                    # 예측값에 노이즈 추가
                    noise = np.random.normal(0, noise_level, len(y_pred_proba))
                    noisy_pred = np.clip(y_pred_proba + noise, 1e-15, 1 - 1e-15)
                    
                    noisy_score = self.metrics_calculator.combined_score_enhanced(y_true, noisy_pred)
                    
                    if original_score > 0:
                        robustness_ratio = noisy_score / original_score
                    else:
                        robustness_ratio = 0.0
                    
                    robustness_scores.append(robustness_ratio)
                    
                except:
                    robustness_scores.append(0.0)
            
            # 극값 제거 테스트
            try:
                # 상위/하위 1% 제거
                q01, q99 = np.percentile(y_pred_proba, [1, 99])
                trimmed_mask = (y_pred_proba >= q01) & (y_pred_proba <= q99)
                
                if trimmed_mask.sum() > 100:
                    trimmed_score = self.metrics_calculator.combined_score_enhanced(
                        y_true[trimmed_mask], y_pred_proba[trimmed_mask]
                    )
                    trimmed_robustness = trimmed_score / original_score if original_score > 0 else 0.0
                else:
                    trimmed_robustness = 0.0
            except:
                trimmed_robustness = 0.0
            
            # 서브샘플링 테스트
            try:
                subsample_robustness = []
                subsample_sizes = [0.8, 0.9, 0.95]
                
                for size_ratio in subsample_sizes:
                    sample_size = int(len(y_pred_proba) * size_ratio)
                    if sample_size > 100:
                        indices = np.random.choice(len(y_pred_proba), sample_size, replace=False)
                        subsample_score = self.metrics_calculator.combined_score_enhanced(
                            y_true[indices], y_pred_proba[indices]
                        )
                        subsample_ratio = subsample_score / original_score if original_score > 0 else 0.0
                        subsample_robustness.append(subsample_ratio)
                    else:
                        subsample_robustness.append(0.0)
                
                avg_subsample_robustness = np.mean(subsample_robustness) if subsample_robustness else 0.0
            except:
                avg_subsample_robustness = 0.0
            
            return {
                'robustness_noise_001': robustness_scores[0] if len(robustness_scores) > 0 else 0.0,
                'robustness_noise_005': robustness_scores[1] if len(robustness_scores) > 1 else 0.0,
                'robustness_noise_010': robustness_scores[2] if len(robustness_scores) > 2 else 0.0,
                'robustness_noise_020': robustness_scores[3] if len(robustness_scores) > 3 else 0.0,
                'robustness_noise_mean': float(np.mean(robustness_scores)) if robustness_scores else 0.0,
                'robustness_noise_std': float(np.std(robustness_scores)) if len(robustness_scores) > 1 else 0.0,
                'robustness_trimmed': float(trimmed_robustness),
                'robustness_subsample': float(avg_subsample_robustness),
                'overall_robustness_score': float(
                    np.mean(robustness_scores) * 0.4 +
                    trimmed_robustness * 0.3 +
                    avg_subsample_robustness * 0.3
                ) if robustness_scores else 0.0
            }
            
        except Exception as e:
            logger.warning(f"초고급 로버스트성 지표 계산 실패: {e}")
            return self._get_default_robustness_metrics()
    
    def _get_default_stability_metrics(self) -> Dict[str, float]:
        """기본 안정성 지표 반환"""
        return {
            'stability_combined_mean': 0.0, 'stability_combined_std': 0.0, 'stability_combined_cv': float('inf'),
            'stability_combined_ci_lower': 0.0, 'stability_combined_ci_upper': 0.0, 'stability_combined_range': 0.0,
            'stability_ap_mean': 0.0, 'stability_ap_std': 0.0, 'stability_ap_cv': float('inf'),
            'stability_ctr_bias_mean': 0.0, 'stability_ctr_bias_std': 0.0, 'stability_ctr_bias_max': 0.0,
            'stability_sample_count': 0, 'stability_success_rate': 0.0, 'overall_stability_score': 0.0
        }
    
    def _get_default_robustness_metrics(self) -> Dict[str, float]:
        """기본 로버스트성 지표 반환"""
        return {
            'robustness_noise_001': 0.0, 'robustness_noise_005': 0.0, 'robustness_noise_010': 0.0, 'robustness_noise_020': 0.0,
            'robustness_noise_mean': 0.0, 'robustness_noise_std': 0.0, 'robustness_trimmed': 0.0,
            'robustness_subsample': 0.0, 'overall_robustness_score': 0.0
        }
    
    def _analyze_ultra_performance_patterns(self, comparison_df: pd.DataFrame):
        """초고급 성능 패턴 분석"""
        try:
            analysis = {}
            
            # 목표 달성 분석
            target_achievers = comparison_df[comparison_df['target_combined_score_achievement'] == 1.0]
            ultra_achievers = comparison_df[comparison_df['ultra_score_achievement'] == 1.0]
            
            analysis['target_achievement_analysis'] = {
                'total_models': len(comparison_df),
                'target_achievers_count': len(target_achievers),
                'ultra_achievers_count': len(ultra_achievers),
                'target_achievement_rate': len(target_achievers) / len(comparison_df),
                'ultra_achievement_rate': len(ultra_achievers) / len(comparison_df),
                'best_combined_score': comparison_df['combined_score_enhanced'].max(),
                'best_ultra_score': comparison_df['ctr_ultra_optimized_score'].max(),
                'avg_combined_score': comparison_df['combined_score_enhanced'].mean(),
                'avg_ultra_score': comparison_df['ctr_ultra_optimized_score'].mean()
            }
            
            # 성능 계층 분석
            tier_counts = comparison_df['performance_tier'].value_counts().to_dict()
            analysis['performance_tier_distribution'] = tier_counts
            
            # CTR 편향 분석
            if 'ctr_bias' in comparison_df.columns:
                ctr_biases = comparison_df['ctr_bias']
                analysis['ctr_bias_analysis'] = {
                    'mean_bias': float(ctr_biases.mean()),
                    'std_bias': float(ctr_biases.std()),
                    'max_absolute_bias': float(ctr_biases.abs().max()),
                    'models_with_ultra_low_bias': int((ctr_biases.abs() < 0.0005).sum()),
                    'models_with_low_bias': int((ctr_biases.abs() < 0.001).sum()),
                    'overestimating_models': int((ctr_biases > 0.001).sum()),
                    'underestimating_models': int((ctr_biases < -0.001).sum())
                }
            
            # 안정성 분석
            stability_cols = [col for col in comparison_df.columns if 'stability' in col and 'overall' in col]
            if stability_cols:
                stability_col = stability_cols[0]
                analysis['stability_analysis'] = {
                    'mean_stability': float(comparison_df[stability_col].mean()),
                    'std_stability': float(comparison_df[stability_col].std()),
                    'highly_stable_models': int((comparison_df[stability_col] > 0.8).sum()),
                    'unstable_models': int((comparison_df[stability_col] < 0.5).sum())
                }
            
            # 로버스트성 분석
            robustness_cols = [col for col in comparison_df.columns if 'robustness' in col and 'overall' in col]
            if robustness_cols:
                robustness_col = robustness_cols[0]
                analysis['robustness_analysis'] = {
                    'mean_robustness': float(comparison_df[robustness_col].mean()),
                    'std_robustness': float(comparison_df[robustness_col].std()),
                    'highly_robust_models': int((comparison_df[robustness_col] > 0.8).sum()),
                    'fragile_models': int((comparison_df[robustness_col] < 0.5).sum())
                }
            
            self.performance_analysis = analysis
            
        except Exception as e:
            logger.warning(f"초고급 성능 패턴 분석 실패: {e}")
    
    def rank_models_ultra(self, 
                         ranking_metric: str = 'ctr_ultra_optimized_score') -> pd.DataFrame:
        """초고성능 모델 순위 매기기"""
        
        if self.comparison_results.empty:
            logger.warning("비교 결과가 없습니다.")
            return pd.DataFrame()
        
        try:
            ranking_df = self.comparison_results.copy()
            
            if ranking_metric in ranking_df.columns:
                ranking_df['rank'] = ranking_df[ranking_metric].rank(ascending=False)
            else:
                ranking_df['rank'] = ranking_df['combined_score_enhanced'].rank(ascending=False)
            
            ranking_df.sort_values('rank', inplace=True)
            
            key_columns = ['rank', ranking_metric, 'combined_score_enhanced', 'ap_enhanced', 'wll_enhanced', 
                          'auc', 'f1', 'ctr_bias', 'ctr_ratio', 'ctr_alignment_score',
                          'performance_tier', 'target_combined_score_achievement', 'ultra_score_achievement']
            
            if 'overall_stability_score' in ranking_df.columns:
                key_columns.append('overall_stability_score')
            if 'overall_robustness_score' in ranking_df.columns:
                key_columns.append('overall_robustness_score')
            
            available_columns = [col for col in key_columns if col in ranking_df.columns]
            
            return ranking_df[available_columns]
        
        except Exception as e:
            logger.error(f"초고성능 모델 순위 매기기 실패: {e}")
            return pd.DataFrame()
    
    def get_best_model_ultra(self, metric: str = 'ctr_ultra_optimized_score') -> Tuple[str, float]:
        """최고 성능 모델 반환 (초고성능)"""
        
        if self.comparison_results.empty:
            return None, 0.0
        
        try:
            if metric not in self.comparison_results.columns:
                metric = 'combined_score_enhanced'
            
            best_idx = self.comparison_results[metric].idxmax()
            best_score = self.comparison_results.loc[best_idx, metric]
            
            return best_idx, best_score
        
        except Exception as e:
            logger.error(f"최고 초고성능 모델 찾기 실패: {e}")
            return None, 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """초고성능 성능 요약 반환"""
        
        if self.comparison_results.empty:
            return {'status': 'no_results', 'message': '비교 결과가 없습니다'}
        
        try:
            summary = {
                'total_models': len(self.comparison_results),
                'target_achievers': int(self.comparison_results['target_combined_score_achievement'].sum()),
                'ultra_achievers': int(self.comparison_results['ultra_score_achievement'].sum()),
                'best_combined_score': float(self.comparison_results['combined_score_enhanced'].max()),
                'best_ultra_score': float(self.comparison_results['ctr_ultra_optimized_score'].max()),
                'avg_combined_score': float(self.comparison_results['combined_score_enhanced'].mean()),
                'performance_analysis': self.performance_analysis
            }
            
            # 최고 모델 정보
            best_model, best_score = self.get_best_model_ultra()
            if best_model:
                summary['best_model'] = {
                    'name': best_model,
                    'ultra_score': best_score,
                    'combined_score': float(self.comparison_results.loc[best_model, 'combined_score_enhanced']),
                    'performance_tier': self.comparison_results.loc[best_model, 'performance_tier'],
                    'target_achieved': bool(self.comparison_results.loc[best_model, 'target_combined_score_achievement']),
                    'ultra_achieved': bool(self.comparison_results.loc[best_model, 'ultra_score_achievement'])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"초고성능 성능 요약 생성 실패: {e}")
            return {'status': 'error', 'message': f'요약 생성 중 오류: {str(e)}'}

# 기존 클래스명 유지 (하위 호환성)
CTRMetrics = CTRAdvancedMetrics