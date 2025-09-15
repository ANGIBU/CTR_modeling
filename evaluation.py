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

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib이 설치되지 않았습니다. 시각화 기능이 비활성화됩니다.")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy가 설치되지 않았습니다. 일부 통계 기능이 비활성화됩니다.")

from config import Config

logger = logging.getLogger(__name__)

class CTRMetrics:
    """CTR 예측 전용 평가 지표 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ap_weight = config.EVALUATION_CONFIG['ap_weight']
        self.wll_weight = config.EVALUATION_CONFIG['wll_weight']
        self.actual_ctr = config.EVALUATION_CONFIG['actual_ctr']
        self.pos_weight = config.EVALUATION_CONFIG['pos_weight']
        self.neg_weight = config.EVALUATION_CONFIG['neg_weight']
        self.target_score = config.EVALUATION_CONFIG['target_score']
        self.ctr_tolerance = config.EVALUATION_CONFIG.get('ctr_tolerance', 0.001)
        self.bias_penalty_weight = config.EVALUATION_CONFIG.get('bias_penalty_weight', 2.0)
        self.calibration_weight = config.EVALUATION_CONFIG.get('calibration_weight', 0.3)
    
    def average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Average Precision (AP) 계산 - 안정성 개선"""
        try:
            # 입력 검증
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
            
            # NaN 및 무한값 처리
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("예측값에 NaN 또는 무한값 존재, 클리핑 적용")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            ap_score = average_precision_score(y_true, y_pred_proba)
            
            # 결과 검증
            if np.isnan(ap_score) or np.isinf(ap_score):
                logger.warning("AP 계산 결과가 유효하지 않음")
                return 0.0
            
            return float(ap_score)
        except Exception as e:
            logger.error(f"AP 계산 오류: {str(e)}")
            return 0.0
    
    def weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """실제 CTR 분포를 반영한 Weighted Log Loss - 안정성 개선"""
        try:
            # 입력 검증
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"크기 불일치: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return float('inf')
            
            if len(y_true) == 0:
                logger.warning("빈 배열로 인해 WLL 계산 불가")
                return float('inf')
            
            # NaN 및 무한값 처리
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("예측값에 NaN 또는 무한값 존재, 클리핑 적용")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            pos_weight = self.pos_weight
            neg_weight = self.neg_weight
            
            sample_weights = np.where(y_true == 1, pos_weight, neg_weight)
            
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            log_loss_values = -(y_true * np.log(y_pred_proba_clipped) + 
                              (1 - y_true) * np.log(1 - y_pred_proba_clipped))
            
            # NaN 체크
            if np.any(np.isnan(log_loss_values)) or np.any(np.isinf(log_loss_values)):
                logger.warning("Log loss 계산에서 NaN 또는 무한값 발생")
                return float('inf')
            
            weighted_log_loss = np.average(log_loss_values, weights=sample_weights)
            
            # 결과 검증
            if np.isnan(weighted_log_loss) or np.isinf(weighted_log_loss):
                logger.warning("WLL 계산 결과가 유효하지 않음")
                return float('inf')
            
            return float(weighted_log_loss)
            
        except Exception as e:
            logger.error(f"WLL 계산 오류: {str(e)}")
            return float('inf')
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR 특화 Combined Score = 0.5 * AP + 0.5 * (1/(1+WLL)) + CTR편향보정"""
        try:
            # 입력 검증
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("Combined Score 계산을 위한 입력 데이터 문제")
                return 0.0
            
            ap_score = self.average_precision(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss(y_true, y_pred_proba)
            
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            basic_combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            # CTR 편향 보정
            try:
                predicted_ctr = y_pred_proba.mean()
                actual_ctr = y_true.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                
                if ctr_bias > 0:
                    ctr_penalty = np.exp(-(ctr_bias / self.ctr_tolerance) ** 2)
                else:
                    ctr_penalty = 1.0
                
                final_score = basic_combined * (1.0 + 0.1 * ctr_penalty)
            except Exception as e:
                logger.warning(f"CTR 편향 계산 실패: {e}")
                final_score = basic_combined
            
            # 결과 검증
            if np.isnan(final_score) or np.isinf(final_score) or final_score < 0:
                logger.warning("Combined Score 계산 결과가 유효하지 않음")
                return 0.0
            
            return float(final_score)
            
        except Exception as e:
            logger.error(f"Combined Score 계산 오류: {str(e)}")
            return 0.0
    
    def ctr_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR 예측에 최적화된 종합 점수"""
        try:
            # 입력 검증
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("CTR 최적화 점수 계산을 위한 입력 데이터 문제")
                return 0.0
            
            ap_score = self.average_precision(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss(y_true, y_pred_proba)
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            # CTR 정확도 계산
            try:
                predicted_ctr = y_pred_proba.mean()
                actual_ctr = y_true.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_ratio = predicted_ctr / actual_ctr if actual_ctr > 0 else 1.0
                
                ctr_accuracy = np.exp(-ctr_bias * 100) if ctr_bias < 0.1 else 0.0
            except Exception as e:
                logger.warning(f"CTR 정확도 계산 실패: {e}")
                ctr_accuracy = 0.0
            
            # 보정 품질 계산
            try:
                calibration_score = self._calculate_calibration_quality(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"보정 품질 계산 실패: {e}")
                calibration_score = 0.5
            
            # 분포 매칭 계산
            try:
                distribution_score = self._calculate_distribution_matching(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"분포 매칭 계산 실패: {e}")
                distribution_score = 0.5
            
            # 가중치 적용
            weights = {
                'ap': 0.35,
                'wll': 0.25,
                'ctr_accuracy': 0.20,
                'calibration': 0.10,
                'distribution': 0.10
            }
            
            optimized_score = (
                weights['ap'] * ap_score +
                weights['wll'] * wll_normalized +
                weights['ctr_accuracy'] * ctr_accuracy +
                weights['calibration'] * calibration_score +
                weights['distribution'] * distribution_score
            )
            
            # 결과 검증
            if np.isnan(optimized_score) or np.isinf(optimized_score) or optimized_score < 0:
                logger.warning("CTR 최적화 점수 계산 결과가 유효하지 않음")
                return 0.0
            
            return float(optimized_score)
            
        except Exception as e:
            logger.error(f"CTR 최적화 점수 계산 오류: {str(e)}")
            return 0.0
    
    def _calculate_calibration_quality(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """모델 보정 품질 계산"""
        try:
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            total_count = len(y_true)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_score = max(0.0, 1.0 - ece * 5)
            
            return calibration_score
            
        except Exception as e:
            logger.warning(f"보정 품질 계산 실패: {e}")
            return 0.5
    
    def _calculate_distribution_matching(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """예측 분포와 실제 분포 매칭 점수"""
        try:
            actual_percentiles = np.percentile(y_true, [10, 25, 50, 75, 90])
            pred_percentiles = np.percentile(y_pred_proba, [10, 25, 50, 75, 90])
            
            percentile_diff = np.mean(np.abs(actual_percentiles - pred_percentiles))
            distribution_score = np.exp(-percentile_diff * 10)
            
            actual_std = np.std(y_true)
            pred_std = np.std(y_pred_proba)
            std_ratio = min(actual_std, pred_std) / max(actual_std, pred_std) if max(actual_std, pred_std) > 0 else 1.0
            
            final_score = 0.7 * distribution_score + 0.3 * std_ratio
            
            return final_score
            
        except Exception as e:
            logger.warning(f"분포 매칭 점수 계산 실패: {e}")
            return 0.5
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray, 
                               y_pred_proba: np.ndarray,
                               threshold: float = 0.5) -> Dict[str, float]:
        """종합적인 평가 지표 계산 - 안정성 개선"""
        
        try:
            # 입력 검증 및 전처리
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"입력 크기 불일치: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return self._get_default_metrics()
            
            if len(y_true) == 0:
                logger.error("빈 입력 배열")
                return self._get_default_metrics()
            
            # NaN 및 무한값 처리
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("예측값에 NaN 또는 무한값 존재, 정리 수행")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 이진 예측 생성
            y_pred = (y_pred_proba >= threshold).astype(int)
        
            metrics = {}
            
            # 주요 CTR 지표
            try:
                metrics['ap'] = self.average_precision(y_true, y_pred_proba)
                metrics['wll'] = self.weighted_log_loss(y_true, y_pred_proba)
                metrics['combined_score'] = self.combined_score(y_true, y_pred_proba)
                metrics['ctr_optimized_score'] = self.ctr_optimized_score(y_true, y_pred_proba)
            except Exception as e:
                logger.error(f"주요 CTR 지표 계산 실패: {e}")
                metrics.update({
                    'ap': 0.0, 'wll': float('inf'), 'combined_score': 0.0, 'ctr_optimized_score': 0.0
                })
            
            # 기본 분류 지표
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"기본 분류 지표 계산 실패: {e}")
                metrics.update({
                    'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25
                })
            
            # 혼동 행렬 기반 지표
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
                    
                    # MCC 계산
                    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    if denominator != 0:
                        metrics['mcc'] = (tp * tn - fp * fn) / denominator
                    else:
                        metrics['mcc'] = 0.0
                else:
                    metrics.update({
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                        'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0
                    })
                
            except Exception as e:
                logger.warning(f"혼동 행렬 지표 계산 실패: {e}")
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0
                })
            
            # CTR 분석
            try:
                metrics['ctr_actual'] = float(y_true.mean())
                metrics['ctr_predicted'] = float(y_pred_proba.mean())
                metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
                metrics['ctr_ratio'] = metrics['ctr_predicted'] / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
            except Exception as e:
                logger.warning(f"CTR 분석 실패: {e}")
                metrics.update({
                    'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
                    'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0
                })
            
            # CTR 범위별 지표
            try:
                self._add_ctr_range_metrics(metrics, y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"CTR 범위별 지표 계산 실패: {e}")
            
            # 예측 통계
            try:
                metrics['prediction_std'] = float(y_pred_proba.std())
                metrics['prediction_var'] = float(y_pred_proba.var())
                metrics['prediction_entropy'] = self._calculate_entropy(y_pred_proba)
                metrics['prediction_gini'] = self._calculate_gini_coefficient(y_pred_proba)
            except Exception as e:
                logger.warning(f"예측 통계 계산 실패: {e}")
                metrics.update({
                    'prediction_std': 0.0, 'prediction_var': 0.0,
                    'prediction_entropy': 0.0, 'prediction_gini': 0.0
                })
            
            # 클래스별 예측 통계
            try:
                pos_mask = (y_true == 1)
                neg_mask = (y_true == 0)
                
                if pos_mask.any():
                    metrics['pos_mean_pred'] = float(y_pred_proba[pos_mask].mean())
                    metrics['pos_std_pred'] = float(y_pred_proba[pos_mask].std())
                    metrics['pos_median_pred'] = float(np.median(y_pred_proba[pos_mask]))
                else:
                    metrics['pos_mean_pred'] = 0.0
                    metrics['pos_std_pred'] = 0.0
                    metrics['pos_median_pred'] = 0.0
                    
                if neg_mask.any():
                    metrics['neg_mean_pred'] = float(y_pred_proba[neg_mask].mean())
                    metrics['neg_std_pred'] = float(y_pred_proba[neg_mask].std())
                    metrics['neg_median_pred'] = float(np.median(y_pred_proba[neg_mask]))
                else:
                    metrics['neg_mean_pred'] = 0.0
                    metrics['neg_std_pred'] = 0.0
                    metrics['neg_median_pred'] = 0.0
                
                if pos_mask.any() and neg_mask.any():
                    metrics['separation'] = metrics['pos_mean_pred'] - metrics['neg_mean_pred']
                    
                    # KS 통계
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
                    metrics['ks_statistic'] = 0.0
                    metrics['ks_pvalue'] = 1.0
            except Exception as e:
                logger.warning(f"클래스별 예측 통계 계산 실패: {e}")
                metrics.update({
                    'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
                    'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
                    'separation': 0.0, 'ks_statistic': 0.0, 'ks_pvalue': 1.0
                })
            
            # 보정 지표
            try:
                calibration_metrics = self.calculate_calibration_metrics(y_true, y_pred_proba)
                metrics.update(calibration_metrics)
            except Exception as e:
                logger.warning(f"보정 지표 계산 실패: {e}")
                metrics.update({
                    'ece': 0.0, 'mce': 0.0, 'ace': 0.0, 
                    'reliability_slope': 1.0, 'reliability_intercept': 0.0
                })
            
            # 분위수별 지표
            try:
                self._add_quantile_metrics(metrics, y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"분위수별 지표 계산 실패: {e}")
            
            # 최적 임계값
            try:
                optimal_threshold, optimal_f1 = self.find_optimal_threshold(y_true, y_pred_proba, 'f1')
                metrics['optimal_threshold'] = float(optimal_threshold)
                metrics['optimal_f1'] = float(optimal_f1)
            except Exception as e:
                logger.warning(f"최적 임계값 계산 실패: {e}")
                metrics['optimal_threshold'] = 0.5
                metrics['optimal_f1'] = metrics.get('f1', 0.0)
            
            # 모든 지표값을 float로 변환하고 유효성 검사
            validated_metrics = {}
            for key, value in metrics.items():
                try:
                    if isinstance(value, (int, float, np.number)):
                        if np.isnan(value) or np.isinf(value):
                            validated_metrics[key] = 0.0
                        else:
                            validated_metrics[key] = float(value)
                    else:
                        validated_metrics[key] = value
                except:
                    validated_metrics[key] = 0.0
            
            return validated_metrics
            
        except Exception as e:
            logger.error(f"종합 평가 계산 오류: {str(e)}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """기본 지표값 반환"""
        return {
            'ap': 0.0, 'wll': float('inf'), 'combined_score': 0.0, 'ctr_optimized_score': 0.0,
            'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25,
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 
            'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0,
            'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
            'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0,
            'prediction_std': 0.0, 'prediction_var': 0.0, 'prediction_entropy': 0.0, 'prediction_gini': 0.0,
            'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
            'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
            'separation': 0.0, 'ks_statistic': 0.0, 'ks_pvalue': 1.0,
            'ece': 0.0, 'mce': 0.0, 'ace': 0.0, 'reliability_slope': 1.0, 'reliability_intercept': 0.0,
            'optimal_threshold': 0.5, 'optimal_f1': 0.0
        }
    
    def _add_ctr_range_metrics(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred_proba: np.ndarray):
        """CTR 범위별 성능 지표 추가"""
        try:
            ranges = {
                'very_low': (0.0, 0.01),
                'low': (0.01, 0.02),
                'medium': (0.02, 0.05),
                'high': (0.05, 0.1),
                'very_high': (0.1, 1.0)
            }
            
            for range_name, (low, high) in ranges.items():
                mask = (y_pred_proba >= low) & (y_pred_proba < high)
                
                if mask.sum() > 0:
                    range_actual_ctr = float(y_true[mask].mean())
                    range_pred_ctr = float(y_pred_proba[mask].mean())
                    range_bias = abs(range_pred_ctr - range_actual_ctr)
                    range_count = int(mask.sum())
                    
                    metrics[f'ctr_{range_name}_actual'] = range_actual_ctr
                    metrics[f'ctr_{range_name}_predicted'] = range_pred_ctr
                    metrics[f'ctr_{range_name}_bias'] = range_bias
                    metrics[f'ctr_{range_name}_count'] = range_count
                    metrics[f'ctr_{range_name}_ratio'] = range_count / len(y_true)
                else:
                    metrics[f'ctr_{range_name}_actual'] = 0.0
                    metrics[f'ctr_{range_name}_predicted'] = 0.0
                    metrics[f'ctr_{range_name}_bias'] = 0.0
                    metrics[f'ctr_{range_name}_count'] = 0
                    metrics[f'ctr_{range_name}_ratio'] = 0.0
                    
        except Exception as e:
            logger.warning(f"CTR 범위별 지표 계산 실패: {e}")
    
    def _add_quantile_metrics(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred_proba: np.ndarray):
        """분위수별 성능 지표 추가"""
        try:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            
            for q in quantiles:
                threshold = np.percentile(y_pred_proba, q * 100)
                top_q_mask = y_pred_proba >= threshold
                
                if top_q_mask.sum() > 0:
                    top_q_ctr = float(y_true[top_q_mask].mean())
                    top_q_size = int(top_q_mask.sum())
                    
                    metrics[f'top_{int(q*100)}p_ctr'] = top_q_ctr
                    metrics[f'top_{int(q*100)}p_size'] = top_q_size
                    metrics[f'top_{int(q*100)}p_lift'] = top_q_ctr / metrics['ctr_actual'] if metrics['ctr_actual'] > 0 else 1.0
                    
        except Exception as e:
            logger.warning(f"분위수별 지표 계산 실패: {e}")
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """예측 확률의 엔트로피 계산"""
        try:
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            
            if np.isnan(entropy) or np.isinf(entropy):
                return 0.0
            
            return float(entropy)
        except:
            return 0.0
    
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
            
            return float(gini)
        except:
            return 0.0
    
    def calculate_pr_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Precision-Recall AUC 계산"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = np.trapz(precision, recall)
            
            return pr_auc, precision, recall
        except Exception as e:
            logger.error(f"PR-AUC 계산 오류: {str(e)}")
            return 0.0, np.array([]), np.array([])
    
    def find_optimal_threshold(self, 
                             y_true: np.ndarray, 
                             y_pred_proba: np.ndarray,
                             metric: str = 'f1') -> Tuple[float, float]:
        """최적 임계값 찾기"""
        
        thresholds = np.arange(0.01, 0.99, 0.01)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            try:
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                if metric == 'f1':
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    else:
                        score = 0.0
                
                elif metric == 'accuracy':
                    score = (y_true == y_pred).mean()
                
                elif metric == 'combined':
                    score = self.combined_score(y_true, y_pred_proba)
                
                elif metric == 'ctr_optimized':
                    score = self.ctr_optimized_score(y_true, y_pred_proba)
                
                else:
                    continue
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception as e:
                continue
        
        return best_threshold, best_score
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """모델 보정 지표 계산"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    bin_accuracies.append(accuracy_in_bin)
                    bin_confidences.append(avg_confidence_in_bin)
                    bin_counts.append(in_bin.sum())
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append(0)
                    bin_counts.append(0)
            
            bin_accuracies = np.array(bin_accuracies)
            bin_confidences = np.array(bin_confidences)
            bin_counts = np.array(bin_counts)
            
            # ECE 계산
            ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(y_true)
            
            # MCE 계산
            mce = np.max(np.abs(bin_accuracies - bin_confidences))
            
            # ACE 계산
            non_empty_bins = bin_counts > 0
            if non_empty_bins.any():
                ace = np.mean(np.abs(bin_accuracies[non_empty_bins] - bin_confidences[non_empty_bins]))
            else:
                ace = 0.0
            
            # 신뢰도 기울기 계산
            try:
                if len(bin_confidences[non_empty_bins]) > 1:
                    slope, intercept = np.polyfit(bin_confidences[non_empty_bins], bin_accuracies[non_empty_bins], 1)
                else:
                    slope, intercept = 1.0, 0.0
            except:
                slope, intercept = 1.0, 0.0
            
            return {
                'ece': float(ece),
                'mce': float(mce),
                'ace': float(ace),
                'reliability_slope': float(slope),
                'reliability_intercept': float(intercept),
                'bin_accuracies': bin_accuracies.tolist(),
                'bin_confidences': bin_confidences.tolist(),
                'bin_counts': bin_counts.tolist()
            }
            
        except Exception as e:
            logger.error(f"보정 지표 계산 오류: {str(e)}")
            return {'ece': 0.0, 'mce': 0.0, 'ace': 0.0, 'reliability_slope': 1.0, 'reliability_intercept': 0.0}

class ModelComparator:
    """다중 모델 비교 클래스 - 안정성 개선"""
    
    def __init__(self):
        self.metrics_calculator = CTRMetrics()
        self.comparison_results = pd.DataFrame()
    
    def compare_models(self, 
                      models_predictions: Dict[str, np.ndarray],
                      y_true: np.ndarray) -> pd.DataFrame:
        """여러 모델의 성능 비교"""
        
        results = []
        
        # 입력 검증
        y_true = np.asarray(y_true).flatten()
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                # 예측값 검증 및 전처리
                y_pred_proba = np.asarray(y_pred_proba).flatten()
                
                if len(y_pred_proba) != len(y_true):
                    logger.error(f"{model_name}: 예측값과 실제값 크기 불일치")
                    continue
                
                if len(y_pred_proba) == 0:
                    logger.error(f"{model_name}: 빈 예측값")
                    continue
                
                # NaN 및 무한값 처리
                if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                    logger.warning(f"{model_name}: 예측값에 NaN 또는 무한값 존재, 정리 수행")
                    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                
                # 평가 수행
                metrics = self.metrics_calculator.comprehensive_evaluation(y_true, y_pred_proba)
                metrics['model_name'] = model_name
                
                # 안정성 지표 계산
                try:
                    stability_metrics = self._calculate_stability_metrics(y_true, y_pred_proba)
                    metrics.update(stability_metrics)
                except Exception as e:
                    logger.warning(f"{model_name} 안정성 지표 계산 실패: {e}")
                    metrics.update(self._get_default_stability_metrics())
                
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"{model_name} 모델 평가 실패: {str(e)}")
                # 기본값으로 추가
                default_metrics = self.metrics_calculator._get_default_metrics()
                default_metrics['model_name'] = model_name
                default_metrics.update(self._get_default_stability_metrics())
                results.append(default_metrics)
        
        if not results:
            logger.error("평가 가능한 모델이 없습니다")
            return pd.DataFrame()
        
        try:
            comparison_df = pd.DataFrame(results)
            
            if not comparison_df.empty:
                comparison_df.set_index('model_name', inplace=True)
                
                # 정렬 기준 결정
                sort_column = 'ctr_optimized_score'
                if sort_column not in comparison_df.columns:
                    sort_column = 'combined_score'
                    if sort_column not in comparison_df.columns:
                        sort_column = 'ap'
                
                if sort_column in comparison_df.columns:
                    comparison_df.sort_values(sort_column, ascending=False, inplace=True)
        
            self.comparison_results = comparison_df
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"비교 결과 생성 실패: {e}")
            return pd.DataFrame()
    
    def _calculate_stability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """모델 안정성 지표 계산 - 개선된 버전"""
        try:
            # 입력 검증
            y_true_array = np.asarray(y_true).flatten()
            y_pred_array = np.asarray(y_pred_proba).flatten()
            
            min_len = min(len(y_true_array), len(y_pred_array))
            if min_len == 0:
                logger.warning("빈 배열로 인해 안정성 지표 계산 불가")
                return self._get_default_stability_metrics()
            
            # 배열 크기 통일
            y_true_array = y_true_array[:min_len]
            y_pred_array = y_pred_array[:min_len]
            
            # 부트스트래핑 설정
            n_bootstrap = min(30, max(10, min_len // 1000))  # 데이터 크기에 따라 조정
            sample_size = min(min_len, 10000)
            scores = []
            
            # 랜덤 시드 설정
            np.random.seed(42)
            
            for i in range(n_bootstrap):
                try:
                    # 안전한 인덱스 생성
                    if sample_size >= min_len:
                        indices = np.arange(min_len)
                    else:
                        indices = np.random.choice(min_len, size=sample_size, replace=True)
                    
                    # 인덱스 범위 검증
                    indices = np.clip(indices, 0, min_len - 1)
                    
                    boot_y_true = y_true_array[indices]
                    boot_y_pred = y_pred_array[indices]
                    
                    # 클래스 분포 확인
                    unique_classes = np.unique(boot_y_true)
                    if len(unique_classes) < 2:
                        continue
                    
                    # 예측값 유효성 검증
                    if np.any(np.isnan(boot_y_pred)) or np.any(np.isinf(boot_y_pred)):
                        boot_y_pred = np.clip(boot_y_pred, 1e-15, 1 - 1e-15)
                        boot_y_pred = np.nan_to_num(boot_y_pred, nan=0.5, posinf=1.0, neginf=0.0)
                    
                    score = self.metrics_calculator.combined_score(boot_y_true, boot_y_pred)
                    
                    # 점수 유효성 검증
                    if score > 0 and not np.isnan(score) and not np.isinf(score):
                        scores.append(score)
                        
                except Exception as e:
                    logger.debug(f"부트스트래핑 {i+1} 실패: {e}")
                    continue
            
            # 결과 계산
            if len(scores) >= 3:
                scores = np.array(scores)
                
                stability_metrics = {
                    'stability_mean': float(scores.mean()),
                    'stability_std': float(scores.std()),
                    'stability_cv': float(scores.std() / scores.mean()) if scores.mean() > 0 else float('inf'),
                    'stability_ci_lower': float(np.percentile(scores, 2.5)),
                    'stability_ci_upper': float(np.percentile(scores, 97.5)),
                    'stability_range': float(np.percentile(scores, 97.5) - np.percentile(scores, 2.5)),
                    'stability_sample_count': len(scores)
                }
            else:
                logger.warning("유효한 부트스트래핑 점수가 부족합니다")
                stability_metrics = self._get_default_stability_metrics()
            
            return stability_metrics
            
        except Exception as e:
            logger.warning(f"안정성 지표 계산 실패: {e}")
            return self._get_default_stability_metrics()
    
    def _get_default_stability_metrics(self) -> Dict[str, float]:
        """기본 안정성 지표 반환"""
        return {
            'stability_mean': 0.0,
            'stability_std': 0.0,
            'stability_cv': float('inf'),
            'stability_ci_lower': 0.0,
            'stability_ci_upper': 0.0,
            'stability_range': 0.0,
            'stability_sample_count': 0
        }
    
    def rank_models(self, 
                   ranking_metric: str = 'ctr_optimized_score') -> pd.DataFrame:
        """모델 순위 매기기"""
        
        if self.comparison_results.empty:
            logger.warning("비교 결과가 없습니다.")
            return pd.DataFrame()
        
        try:
            ranking_df = self.comparison_results.copy()
            
            if ranking_metric in ranking_df.columns:
                ranking_df['rank'] = ranking_df[ranking_metric].rank(ascending=False)
            else:
                ranking_df['rank'] = ranking_df['combined_score'].rank(ascending=False)
            
            ranking_df.sort_values('rank', inplace=True)
            
            key_columns = ['rank', ranking_metric, 'combined_score', 'ap', 'wll', 'auc', 'f1', 
                          'ctr_bias', 'ctr_ratio', 'stability_mean', 'stability_std']
            available_columns = [col for col in key_columns if col in ranking_df.columns]
            
            return ranking_df[available_columns]
        
        except Exception as e:
            logger.error(f"모델 순위 매기기 실패: {e}")
            return pd.DataFrame()
    
    def get_best_model(self, metric: str = 'ctr_optimized_score') -> Tuple[str, float]:
        """최고 성능 모델 반환"""
        
        if self.comparison_results.empty:
            return None, 0.0
        
        try:
            if metric not in self.comparison_results.columns:
                metric = 'combined_score'
            
            best_idx = self.comparison_results[metric].idxmax()
            best_score = self.comparison_results.loc[best_idx, metric]
            
            return best_idx, best_score
        
        except Exception as e:
            logger.error(f"최고 모델 찾기 실패: {e}")
            return None, 0.0

class EvaluationVisualizer:
    """평가 결과 시각화 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        
        if self.matplotlib_available:
            try:
                plt.style.available
                available_styles = plt.style.available
                
                if 'seaborn-v0_8' in available_styles:
                    plt.style.use('seaborn-v0_8')
                elif 'seaborn' in available_styles:
                    plt.style.use('seaborn')
                else:
                    plt.style.use('default')
            except Exception:
                pass
        else:
            logger.warning("matplotlib을 사용할 수 없습니다. 시각화 기능이 비활성화됩니다.")
    
    def plot_comprehensive_evaluation(self,
                                    models_predictions: Dict[str, np.ndarray],
                                    y_true: np.ndarray,
                                    save_path: Optional[str] = None):
        """종합 평가 시각화"""
        
        if not self.matplotlib_available:
            logger.warning("matplotlib을 사용할 수 없어 시각화를 할 수 없습니다.")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()
            
            self._plot_roc_curves_subplot(axes[0], models_predictions, y_true)
            self._plot_pr_curves_subplot(axes[1], models_predictions, y_true)
            self._plot_ctr_bias_subplot(axes[2], models_predictions, y_true)
            self._plot_prediction_distributions_subplot(axes[3], models_predictions, y_true)
            self._plot_calibration_subplot(axes[4], models_predictions, y_true)
            self._plot_performance_radar_subplot(axes[5], models_predictions, y_true)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"종합 평가 시각화 실패: {str(e)}")
    
    def _plot_roc_curves_subplot(self, ax, models_predictions, y_true):
        """ROC 곡선 서브플롯"""
        try:
            for model_name, y_pred_proba in models_predictions.items():
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                ax.plot(fpr, tpr, label=f'{model_name} (AUC: {auc_score:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'ROC 오류: {str(e)}', ha='center', va='center')
    
    def _plot_pr_curves_subplot(self, ax, models_predictions, y_true):
        """PR 곡선 서브플롯"""
        try:
            metrics_calc = CTRMetrics()
            
            for model_name, y_pred_proba in models_predictions.items():
                pr_auc, precision, recall = metrics_calc.calculate_pr_auc(y_true, y_pred_proba)
                ax.plot(recall, precision, label=f'{model_name} (PR-AUC: {pr_auc:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'PR 오류: {str(e)}', ha='center', va='center')
    
    def _plot_ctr_bias_subplot(self, ax, models_predictions, y_true):
        """CTR 편향 비교 서브플롯"""
        try:
            actual_ctr = y_true.mean()
            model_names = []
            ctr_biases = []
            
            for model_name, y_pred_proba in models_predictions.items():
                predicted_ctr = y_pred_proba.mean()
                bias = predicted_ctr - actual_ctr
                
                model_names.append(model_name)
                ctr_biases.append(bias)
            
            colors = ['red' if bias > 0 else 'blue' for bias in ctr_biases]
            bars = ax.bar(model_names, ctr_biases, color=colors, alpha=0.7)
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_ylabel('CTR Bias')
            ax.set_title(f'CTR Bias (Actual CTR: {actual_ctr:.4f})')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            for bar, bias in zip(bars, ctr_biases):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.0001 if height >= 0 else -0.0001),
                       f'{bias:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
        except Exception as e:
            ax.text(0.5, 0.5, f'CTR 편향 오류: {str(e)}', ha='center', va='center')
    
    def _plot_prediction_distributions_subplot(self, ax, models_predictions, y_true):
        """예측 분포 서브플롯"""
        try:
            for model_name, y_pred_proba in models_predictions.items():
                ax.hist(y_pred_proba, bins=50, alpha=0.5, label=model_name, density=True)
            
            actual_ctr = y_true.mean()
            ax.axvline(actual_ctr, color='red', linestyle='--', 
                      label=f'Actual CTR: {actual_ctr:.4f}')
            
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Density')
            ax.set_title('Prediction Distributions')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'분포 오류: {str(e)}', ha='center', va='center')
    
    def _plot_calibration_subplot(self, ax, models_predictions, y_true):
        """보정 다이어그램 서브플롯"""
        try:
            metrics_calc = CTRMetrics()
            
            for model_name, y_pred_proba in models_predictions.items():
                calibration_metrics = metrics_calc.calculate_calibration_metrics(y_true, y_pred_proba)
                
                bin_confidences = calibration_metrics['bin_confidences']
                bin_accuracies = calibration_metrics['bin_accuracies']
                
                ax.plot(bin_confidences, bin_accuracies, 'o-', label=f'{model_name}')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'보정 오류: {str(e)}', ha='center', va='center')
    
    def _plot_performance_radar_subplot(self, ax, models_predictions, y_true):
        """성능 레이더 차트 서브플롯"""
        try:
            metrics_calc = CTRMetrics()
            
            metric_names = ['AP', 'AUC', 'F1', 'CTR Accuracy', 'Calibration']
            
            for model_name, y_pred_proba in models_predictions.items():
                metrics = metrics_calc.comprehensive_evaluation(y_true, y_pred_proba)
                
                values = [
                    metrics.get('ap', 0.0),
                    metrics.get('auc', 0.0),
                    metrics.get('f1', 0.0),
                    1.0 - min(metrics.get('ctr_absolute_error', 1.0), 1.0),
                    1.0 - min(metrics.get('ece', 1.0), 1.0)
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
                values += values[:1]
                angles = np.concatenate((angles, [angles[0]]))
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names)
            ax.set_ylim(0, 1)
            ax.set_title('Performance Radar Chart')
            ax.legend()
            ax.grid(True)
        except Exception as e:
            ax.text(0.5, 0.5, f'레이더 차트 오류: {str(e)}', ha='center', va='center')

class EvaluationReporter:
    """평가 보고서 생성 클래스"""
    
    def __init__(self):
        self.metrics_calculator = CTRMetrics()
        self.visualizer = EvaluationVisualizer()
    
    def generate_comprehensive_report(self,
                                    models_predictions: Dict[str, np.ndarray],
                                    y_true: np.ndarray,
                                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """종합 평가 보고서 생성"""
        
        logger.info("종합 평가 보고서 생성 시작")
        
        try:
            comparator = ModelComparator()
            comparison_df = comparator.compare_models(models_predictions, y_true)
            
            best_model, best_score = comparator.get_best_model('ctr_optimized_score')
            
            # 안정성 분석은 간소화
            stability_results = {}
            for model_name, y_pred_proba in models_predictions.items():
                try:
                    stability_metrics = comparator._calculate_stability_metrics(y_true, y_pred_proba)
                    stability_results[model_name] = stability_metrics
                except Exception as e:
                    logger.warning(f"{model_name} 안정성 분석 실패: {e}")
                    stability_results[model_name] = comparator._get_default_stability_metrics()
            
            # 다양성 분석은 간소화
            diversity_results = self._analyze_model_diversity_simple(models_predictions)
            
            # 비즈니스 지표
            business_metrics = {}
            if best_model and best_model in models_predictions:
                try:
                    business_metrics = self.metrics_calculator.calculate_business_metrics(
                        y_true, models_predictions[best_model]
                    )
                except Exception as e:
                    logger.warning(f"비즈니스 지표 계산 실패: {e}")
            
            report = {
                'summary': {
                    'total_models': len(models_predictions),
                    'best_model': best_model,
                    'best_score': best_score,
                    'data_size': len(y_true),
                    'actual_ctr': float(y_true.mean()),
                    'target_score': self.metrics_calculator.target_score,
                    'evaluation_timestamp': pd.Timestamp.now().isoformat()
                },
                'detailed_comparison': comparison_df.to_dict() if not comparison_df.empty else {},
                'model_rankings': comparator.rank_models('ctr_optimized_score').to_dict() if not comparison_df.empty else {},
                'stability_analysis': stability_results,
                'diversity_analysis': diversity_results,
                'business_metrics': business_metrics
            }
            
            if not comparison_df.empty:
                try:
                    report['performance_analysis'] = self._analyze_performance_patterns(comparison_df)
                    report['recommendations'] = self._generate_recommendations(comparison_df, stability_results)
                except Exception as e:
                    logger.warning(f"성능 분석 실패: {e}")
            
            if output_dir and MATPLOTLIB_AVAILABLE:
                try:
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    
                    self.visualizer.plot_comprehensive_evaluation(
                        models_predictions, y_true,
                        save_path=f"{output_dir}/comprehensive_evaluation.png"
                    )
                    
                    report['visualizations'] = {
                        'comprehensive_evaluation': f"{output_dir}/comprehensive_evaluation.png"
                    }
                    
                except Exception as e:
                    logger.warning(f"시각화 생성 중 오류: {str(e)}")
            
            logger.info("종합 평가 보고서 생성 완료")
            
            return report
            
        except Exception as e:
            logger.error(f"종합 평가 보고서 생성 실패: {e}")
            return {'error': str(e)}
    
    def _analyze_model_diversity_simple(self, models_predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """간소화된 모델 다양성 분석"""
        try:
            model_names = list(models_predictions.keys())
            n_models = len(model_names)
            
            if n_models < 2:
                return {'diversity_score': 0.0, 'correlation_matrix': {}}
            
            correlations = []
            
            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i < j:
                        try:
                            pred1 = np.asarray(models_predictions[name1]).flatten()
                            pred2 = np.asarray(models_predictions[name2]).flatten()
                            
                            min_len = min(len(pred1), len(pred2))
                            if min_len > 0:
                                pred1 = pred1[:min_len]
                                pred2 = pred2[:min_len]
                                corr = np.corrcoef(pred1, pred2)[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                        except:
                            continue
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            diversity_score = 1.0 - avg_correlation
            
            return {
                'diversity_score': float(diversity_score),
                'average_correlation': float(avg_correlation),
                'model_count': n_models
            }
            
        except Exception as e:
            logger.error(f"모델 다양성 분석 실패: {e}")
            return {'diversity_score': 0.0, 'correlation_matrix': {}}
    
    def _analyze_performance_patterns(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """성능 패턴 분석"""
        try:
            analysis = {}
            
            correlation_cols = ['ctr_optimized_score', 'combined_score', 'ap', 'auc', 'f1']
            available_cols = [col for col in correlation_cols if col in comparison_df.columns]
            
            if len(available_cols) > 1:
                correlation_matrix = comparison_df[available_cols].corr()
                analysis['metric_correlations'] = correlation_matrix.to_dict()
            
            if 'ctr_bias' in comparison_df.columns:
                ctr_biases = comparison_df['ctr_bias']
                analysis['ctr_bias_patterns'] = {
                    'mean_bias': float(ctr_biases.mean()),
                    'std_bias': float(ctr_biases.std()),
                    'max_absolute_bias': float(ctr_biases.abs().max()),
                    'models_with_low_bias': int((ctr_biases.abs() < 0.001).sum()),
                    'overestimating_models': int((ctr_biases > 0.001).sum()),
                    'underestimating_models': int((ctr_biases < -0.001).sum())
                }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"성능 패턴 분석 실패: {e}")
            return {}
    
    def _generate_recommendations(self, comparison_df: pd.DataFrame, stability_results: Dict) -> List[str]:
        """모델 권장사항 생성"""
        recommendations = []
        
        try:
            if 'ctr_bias' in comparison_df.columns:
                high_bias_models = comparison_df[comparison_df['ctr_bias'].abs() > 0.002]
                if not high_bias_models.empty:
                    recommendations.append(
                        f"CTR 편향이 큰 모델들({', '.join(high_bias_models.index)})에 대해 "
                        "Calibration 기법 적용을 권장합니다."
                    )
            
            unstable_models = []
            for model_name, stability in stability_results.items():
                if stability.get('stability_cv', float('inf')) > 0.1:
                    unstable_models.append(model_name)
            
            if unstable_models:
                recommendations.append(
                    f"안정성이 낮은 모델들({', '.join(unstable_models)})에 대해 "
                    "앙상블 기법 적용을 권장합니다."
                )
            
            if 'ctr_optimized_score' in comparison_df.columns:
                low_performance_models = comparison_df[comparison_df['ctr_optimized_score'] < 0.3]
                if not low_performance_models.empty:
                    recommendations.append(
                        f"성능이 낮은 모델들({', '.join(low_performance_models.index)})에 대해 "
                        "하이퍼파라미터 재튜닝을 권장합니다."
                    )
            
            if len(comparison_df) > 1:
                recommendations.append("다중 모델 앙상블을 통해 전체적인 성능 향상을 기대할 수 있습니다.")
            
            if not recommendations:
                recommendations.append("모든 모델이 양호한 성능을 보이고 있습니다.")
            
        except Exception as e:
            logger.warning(f"권장사항 생성 실패: {e}")
            recommendations.append("권장사항 생성 중 오류가 발생했습니다.")
        
        return recommendations