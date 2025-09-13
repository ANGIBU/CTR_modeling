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

# matplotlib import 처리
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib이 설치되지 않았습니다. 시각화 기능이 비활성화됩니다.")

# scipy import 처리
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
        """Average Precision (AP) 계산"""
        try:
            if len(np.unique(y_true)) < 2:
                logger.warning("단일 클래스만 존재하여 AP 계산 불가")
                return 0.0
            
            ap_score = average_precision_score(y_true, y_pred_proba)
            return ap_score
        except Exception as e:
            logger.error(f"AP 계산 오류: {str(e)}")
            return 0.0
    
    def weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """실제 CTR 분포를 반영한 Weighted Log Loss"""
        try:
            # 클래스 가중치 설정
            pos_weight = self.pos_weight
            neg_weight = self.neg_weight
            
            # 각 샘플별 가중치 적용
            sample_weights = np.where(y_true == 1, pos_weight, neg_weight)
            
            # 확률 클리핑으로 수치 안정성 확보
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # 가중 로그 손실 직접 계산
            log_loss_values = -(y_true * np.log(y_pred_proba_clipped) + 
                              (1 - y_true) * np.log(1 - y_pred_proba_clipped))
            
            weighted_log_loss = np.average(log_loss_values, weights=sample_weights)
            
            return weighted_log_loss
            
        except Exception as e:
            logger.error(f"WLL 계산 오류: {str(e)}")
            return float('inf')
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR 특화 Combined Score = 0.5 * AP + 0.5 * (1/(1+WLL)) + CTR편향보정"""
        try:
            ap_score = self.average_precision(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss(y_true, y_pred_proba)
            
            # WLL을 0-1 스케일로 변환
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            # 기본 Combined Score
            basic_combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            # CTR 편향 패널티 계산
            predicted_ctr = y_pred_proba.mean()
            actual_ctr = y_true.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # CTR 편향에 따른 패널티 (가우시안 형태)
            ctr_penalty = np.exp(-(ctr_bias / self.ctr_tolerance) ** 2)
            
            # 최종 점수 (CTR 편향 고려)
            final_score = basic_combined * (1.0 + 0.1 * ctr_penalty)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Combined Score 계산 오류: {str(e)}")
            return 0.0
    
    def ctr_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR 예측에 최적화된 종합 점수"""
        try:
            # 기본 성능 지표
            ap_score = self.average_precision(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss(y_true, y_pred_proba)
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            # CTR 관련 지표
            predicted_ctr = y_pred_proba.mean()
            actual_ctr = y_true.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            ctr_ratio = predicted_ctr / actual_ctr if actual_ctr > 0 else 1.0
            
            # Calibration 품질
            calibration_score = self._calculate_calibration_quality(y_true, y_pred_proba)
            
            # 분포 매칭 점수
            distribution_score = self._calculate_distribution_matching(y_true, y_pred_proba)
            
            # 클릭률 정확도 점수 (CTR 편향 기반)
            ctr_accuracy = np.exp(-ctr_bias * 100) if ctr_bias < 0.1 else 0.0
            
            # 가중 합산
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
            
            return optimized_score
            
        except Exception as e:
            logger.error(f"CTR 최적화 점수 계산 오류: {str(e)}")
            return 0.0
    
    def _calculate_calibration_quality(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """모델 보정 품질 계산"""
        try:
            # ECE (Expected Calibration Error) 계산
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
            
            # ECE를 0-1 점수로 변환 (낮은 ECE = 높은 점수)
            calibration_score = max(0.0, 1.0 - ece * 5)
            
            return calibration_score
            
        except Exception as e:
            logger.warning(f"보정 품질 계산 실패: {e}")
            return 0.5
    
    def _calculate_distribution_matching(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """예측 분포와 실제 분포 매칭 점수"""
        try:
            # 분위수 기반 분포 비교
            actual_percentiles = np.percentile(y_true, [10, 25, 50, 75, 90])
            pred_percentiles = np.percentile(y_pred_proba, [10, 25, 50, 75, 90])
            
            # KL divergence 또는 Wasserstein distance 대신 단순 차이 사용
            percentile_diff = np.mean(np.abs(actual_percentiles - pred_percentiles))
            distribution_score = np.exp(-percentile_diff * 10)
            
            # 추가: 분산 비교
            actual_std = np.std(y_true)
            pred_std = np.std(y_pred_proba)
            std_ratio = min(actual_std, pred_std) / max(actual_std, pred_std) if max(actual_std, pred_std) > 0 else 1.0
            
            # 최종 분포 매칭 점수
            final_score = 0.7 * distribution_score + 0.3 * std_ratio
            
            return final_score
            
        except Exception as e:
            logger.warning(f"분포 매칭 점수 계산 실패: {e}")
            return 0.5
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray, 
                               y_pred_proba: np.ndarray,
                               threshold: float = 0.5) -> Dict[str, float]:
        """종합적인 평가 지표 계산"""
        
        # 이진 예측
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {}
        
        try:
            # 대회 평가 지표
            metrics['ap'] = self.average_precision(y_true, y_pred_proba)
            metrics['wll'] = self.weighted_log_loss(y_true, y_pred_proba)
            metrics['combined_score'] = self.combined_score(y_true, y_pred_proba)
            metrics['ctr_optimized_score'] = self.ctr_optimized_score(y_true, y_pred_proba)
            
            # 기본 분류 지표
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            except:
                metrics['auc'] = 0.5
                metrics['log_loss'] = 1.0
                metrics['brier_score'] = 0.25
            
            # 혼동 행렬 기반 지표
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                
                metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['sensitivity'] = metrics['recall']
                
                # F1 Score
                if metrics['precision'] + metrics['recall'] > 0:
                    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                else:
                    metrics['f1'] = 0.0
                
                # Matthews Correlation Coefficient
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                if denominator != 0:
                    metrics['mcc'] = (tp * tn - fp * fn) / denominator
                else:
                    metrics['mcc'] = 0.0
                
            except Exception as e:
                logger.warning(f"혼동 행렬 지표 계산 실패: {e}")
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0
                })
            
            # CTR 관련 지표
            metrics['ctr_actual'] = y_true.mean()
            metrics['ctr_predicted'] = y_pred_proba.mean()
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            metrics['ctr_ratio'] = metrics['ctr_predicted'] / max(metrics['ctr_actual'], 1e-10)
            metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
            
            # CTR 범위별 성능
            self._add_ctr_range_metrics(metrics, y_true, y_pred_proba)
            
            # 분포 관련 지표
            metrics['prediction_std'] = y_pred_proba.std()
            metrics['prediction_var'] = y_pred_proba.var()
            metrics['prediction_entropy'] = self._calculate_entropy(y_pred_proba)
            metrics['prediction_gini'] = self._calculate_gini_coefficient(y_pred_proba)
            
            # 클래스별 예측 품질
            pos_mask = (y_true == 1)
            neg_mask = (y_true == 0)
            
            if pos_mask.any():
                metrics['pos_mean_pred'] = y_pred_proba[pos_mask].mean()
                metrics['pos_std_pred'] = y_pred_proba[pos_mask].std()
                metrics['pos_median_pred'] = np.median(y_pred_proba[pos_mask])
            else:
                metrics['pos_mean_pred'] = 0.0
                metrics['pos_std_pred'] = 0.0
                metrics['pos_median_pred'] = 0.0
                
            if neg_mask.any():
                metrics['neg_mean_pred'] = y_pred_proba[neg_mask].mean()
                metrics['neg_std_pred'] = y_pred_proba[neg_mask].std()
                metrics['neg_median_pred'] = np.median(y_pred_proba[neg_mask])
            else:
                metrics['neg_mean_pred'] = 0.0
                metrics['neg_std_pred'] = 0.0
                metrics['neg_median_pred'] = 0.0
            
            # 분리도 지표
            if pos_mask.any() and neg_mask.any():
                metrics['separation'] = metrics['pos_mean_pred'] - metrics['neg_mean_pred']
                
                # KS (Kolmogorov-Smirnov) 통계량
                if SCIPY_AVAILABLE:
                    try:
                        ks_stat, ks_pvalue = stats.ks_2samp(y_pred_proba[pos_mask], y_pred_proba[neg_mask])
                        metrics['ks_statistic'] = ks_stat
                        metrics['ks_pvalue'] = ks_pvalue
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
            
            # 보정 지표
            calibration_metrics = self.calculate_calibration_metrics(y_true, y_pred_proba)
            metrics.update(calibration_metrics)
            
            # 분위수별 성능
            self._add_quantile_metrics(metrics, y_true, y_pred_proba)
            
            # 임계값 최적화
            optimal_threshold, optimal_f1 = self.find_optimal_threshold(y_true, y_pred_proba, 'f1')
            metrics['optimal_threshold'] = optimal_threshold
            metrics['optimal_f1'] = optimal_f1
            
        except Exception as e:
            logger.error(f"종합 평가 계산 오류: {str(e)}")
        
        return metrics
    
    def _add_ctr_range_metrics(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred_proba: np.ndarray):
        """CTR 범위별 성능 지표 추가"""
        try:
            # CTR 구간 정의
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
                    range_actual_ctr = y_true[mask].mean()
                    range_pred_ctr = y_pred_proba[mask].mean()
                    range_bias = abs(range_pred_ctr - range_actual_ctr)
                    range_count = mask.sum()
                    
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
                    top_q_ctr = y_true[top_q_mask].mean()
                    top_q_size = top_q_mask.sum()
                    
                    metrics[f'top_{int(q*100)}p_ctr'] = top_q_ctr
                    metrics[f'top_{int(q*100)}p_size'] = top_q_size
                    metrics[f'top_{int(q*100)}p_lift'] = top_q_ctr / metrics['ctr_actual'] if metrics['ctr_actual'] > 0 else 1.0
                    
        except Exception as e:
            logger.warning(f"분위수별 지표 계산 실패: {e}")
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """예측 확률의 엔트로피 계산"""
        try:
            # 확률 클립핑
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            
            # 엔트로피 계산
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            
            return entropy
        except:
            return 0.0
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """지니 계수 계산"""
        try:
            # 값 정렬
            sorted_values = np.sort(values)
            n = len(sorted_values)
            
            # 지니 계수 계산
            cumsum = np.cumsum(sorted_values)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            
            return gini
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
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            try:
                if metric == 'f1':
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
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
            # 빈 경계 설정
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # 빈에 속하는 샘플 찾기
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
            
            # ECE (Expected Calibration Error) 계산
            bin_accuracies = np.array(bin_accuracies)
            bin_confidences = np.array(bin_confidences)
            bin_counts = np.array(bin_counts)
            
            ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(y_true)
            
            # MCE (Maximum Calibration Error) 계산
            mce = np.max(np.abs(bin_accuracies - bin_confidences))
            
            # ACE (Average Calibration Error) 계산
            non_empty_bins = bin_counts > 0
            if non_empty_bins.any():
                ace = np.mean(np.abs(bin_accuracies[non_empty_bins] - bin_confidences[non_empty_bins]))
            else:
                ace = 0.0
            
            # Reliability diagram의 기울기
            try:
                if len(bin_confidences[non_empty_bins]) > 1:
                    slope, intercept = np.polyfit(bin_confidences[non_empty_bins], bin_accuracies[non_empty_bins], 1)
                else:
                    slope, intercept = 1.0, 0.0
            except:
                slope, intercept = 1.0, 0.0
            
            return {
                'ece': ece,
                'mce': mce,
                'ace': ace,
                'reliability_slope': slope,
                'reliability_intercept': intercept,
                'bin_accuracies': bin_accuracies.tolist(),
                'bin_confidences': bin_confidences.tolist(),
                'bin_counts': bin_counts.tolist()
            }
            
        except Exception as e:
            logger.error(f"보정 지표 계산 오류: {str(e)}")
            return {'ece': 0.0, 'mce': 0.0, 'ace': 0.0, 'reliability_slope': 1.0, 'reliability_intercept': 0.0}
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                 cost_per_click: float = 1.0, revenue_per_conversion: float = 20.0) -> Dict[str, float]:
        """비즈니스 관련 지표 계산"""
        try:
            # 다양한 임계값에서의 비즈니스 지표
            thresholds = np.arange(0.01, 0.5, 0.01)
            business_metrics = {}
            
            for threshold in thresholds:
                predictions = (y_pred_proba >= threshold).astype(int)
                
                # 혼동 행렬
                tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
                
                # 비즈니스 지표
                total_clicks = tp + fp
                total_conversions = tp
                
                if total_clicks > 0:
                    conversion_rate = total_conversions / total_clicks
                    cost = total_clicks * cost_per_click
                    revenue = total_conversions * revenue_per_conversion
                    profit = revenue - cost
                    roi = (revenue - cost) / cost if cost > 0 else 0.0
                    
                    business_metrics[f'threshold_{threshold:.2f}'] = {
                        'clicks': total_clicks,
                        'conversions': total_conversions,
                        'conversion_rate': conversion_rate,
                        'cost': cost,
                        'revenue': revenue,
                        'profit': profit,
                        'roi': roi
                    }
            
            # 최적 임계값 (수익 기준)
            if business_metrics:
                best_threshold = max(business_metrics.keys(), 
                                   key=lambda x: business_metrics[x]['profit'])
                business_metrics['optimal_threshold'] = best_threshold
                business_metrics['optimal_metrics'] = business_metrics[best_threshold]
            
            return business_metrics
            
        except Exception as e:
            logger.error(f"비즈니스 지표 계산 오류: {str(e)}")
            return {}

class ModelComparator:
    """다중 모델 비교 클래스"""
    
    def __init__(self):
        self.metrics_calculator = CTRMetrics()
        self.comparison_results = {}
    
    def compare_models(self, 
                      models_predictions: Dict[str, np.ndarray],
                      y_true: np.ndarray) -> pd.DataFrame:
        """여러 모델의 성능 비교"""
        
        results = []
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                # 기본 성능 지표
                metrics = self.metrics_calculator.comprehensive_evaluation(y_true, y_pred_proba)
                metrics['model_name'] = model_name
                
                # 안정성 지표 추가
                stability_metrics = self._calculate_stability_metrics(y_true, y_pred_proba)
                metrics.update(stability_metrics)
                
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"{model_name} 모델 평가 실패: {str(e)}")
        
        # DataFrame으로 변환
        comparison_df = pd.DataFrame(results)
        
        if not comparison_df.empty:
            # 모델명을 인덱스로 설정
            comparison_df.set_index('model_name', inplace=True)
            
            # CTR 최적화 점수로 정렬
            if 'ctr_optimized_score' in comparison_df.columns:
                comparison_df.sort_values('ctr_optimized_score', ascending=False, inplace=True)
            else:
                comparison_df.sort_values('combined_score', ascending=False, inplace=True)
        
        self.comparison_results = comparison_df
        
        return comparison_df
    
    def _calculate_stability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """모델 안정성 지표 계산"""
        try:
            # 배열 타입 변환 및 길이 확인
            y_true_array = np.asarray(y_true).flatten()
            y_pred_array = np.asarray(y_pred_proba).flatten()
            
            # 배열 길이 통일
            min_len = min(len(y_true_array), len(y_pred_array))
            if min_len == 0:
                logger.warning("빈 배열로 인해 안정성 지표 계산 불가")
                return self._get_default_stability_metrics()
            
            y_true_array = y_true_array[:min_len]
            y_pred_array = y_pred_array[:min_len]
            
            # 부트스트래핑 기반 안정성 계산
            n_bootstrap = 30  # 메모리 및 성능 최적화
            scores = []
            
            sample_size = min(min_len, 10000)  # 샘플 크기 제한
            
            for i in range(n_bootstrap):
                try:
                    # 부트스트래핑 샘플링
                    indices = np.random.choice(min_len, size=sample_size, replace=True)
                    boot_y_true = y_true_array[indices]
                    boot_y_pred = y_pred_array[indices]
                    
                    # 유효성 검사
                    if len(np.unique(boot_y_true)) < 2:
                        continue
                    
                    score = self.metrics_calculator.combined_score(boot_y_true, boot_y_pred)
                    if score > 0 and not np.isnan(score) and not np.isinf(score):
                        scores.append(score)
                        
                except Exception as e:
                    logger.debug(f"부트스트래핑 {i+1} 실패: {e}")
                    continue
            
            # 결과 계산
            if len(scores) >= 3:  # 최소한의 유효한 점수 필요
                scores = np.array(scores)
                
                stability_metrics = {
                    'stability_mean': float(scores.mean()),
                    'stability_std': float(scores.std()),
                    'stability_cv': float(scores.std() / scores.mean()) if scores.mean() > 0 else float('inf'),
                    'stability_ci_lower': float(np.percentile(scores, 2.5)),
                    'stability_ci_upper': float(np.percentile(scores, 97.5)),
                    'stability_range': float(np.percentile(scores, 97.5) - np.percentile(scores, 2.5))
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
            'stability_range': 0.0
        }
    
    def rank_models(self, 
                   ranking_metric: str = 'ctr_optimized_score') -> pd.DataFrame:
        """모델 순위 매기기"""
        
        if self.comparison_results.empty:
            logger.warning("비교 결과가 없습니다.")
            return pd.DataFrame()
        
        ranking_df = self.comparison_results.copy()
        
        # 순위 지정
        if ranking_metric in ranking_df.columns:
            ranking_df['rank'] = ranking_df[ranking_metric].rank(ascending=False)
        else:
            ranking_df['rank'] = ranking_df['combined_score'].rank(ascending=False)
        
        ranking_df.sort_values('rank', inplace=True)
        
        # 주요 지표만 선택
        key_columns = ['rank', ranking_metric, 'combined_score', 'ap', 'wll', 'auc', 'f1', 
                      'ctr_bias', 'ctr_ratio', 'stability_mean', 'stability_std']
        available_columns = [col for col in key_columns if col in ranking_df.columns]
        
        return ranking_df[available_columns]
    
    def get_best_model(self, metric: str = 'ctr_optimized_score') -> Tuple[str, float]:
        """최고 성능 모델 반환"""
        
        if self.comparison_results.empty:
            return None, 0.0
        
        if metric not in self.comparison_results.columns:
            metric = 'combined_score'
        
        best_idx = self.comparison_results[metric].idxmax()
        best_score = self.comparison_results.loc[best_idx, metric]
        
        return best_idx, best_score
    
    def analyze_model_diversity(self, models_predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """모델 다양성 분석"""
        try:
            model_names = list(models_predictions.keys())
            n_models = len(model_names)
            
            if n_models < 2:
                return {'diversity_score': 0.0, 'correlation_matrix': {}}
            
            # 상관관계 행렬 계산
            correlation_matrix = {}
            correlations = []
            
            for i, name1 in enumerate(model_names):
                correlation_matrix[name1] = {}
                for j, name2 in enumerate(model_names):
                    if i == j:
                        corr = 1.0
                    else:
                        try:
                            pred1 = np.asarray(models_predictions[name1]).flatten()
                            pred2 = np.asarray(models_predictions[name2]).flatten()
                            
                            # 배열 길이 통일
                            min_len = min(len(pred1), len(pred2))
                            if min_len > 0:
                                pred1 = pred1[:min_len]
                                pred2 = pred2[:min_len]
                                corr = np.corrcoef(pred1, pred2)[0, 1]
                                if np.isnan(corr):
                                    corr = 0.0
                            else:
                                corr = 0.0
                        except:
                            corr = 0.0
                    
                    correlation_matrix[name1][name2] = float(corr)
                    
                    if i < j:  # 중복 제거
                        correlations.append(abs(corr))
            
            # 다양성 점수 (낮은 상관관계 = 높은 다양성)
            avg_correlation = np.mean(correlations) if correlations else 0.0
            diversity_score = 1.0 - avg_correlation
            
            # 추가 다양성 지표
            diversity_metrics = {
                'diversity_score': float(diversity_score),
                'average_correlation': float(avg_correlation),
                'max_correlation': float(max(correlations)) if correlations else 0.0,
                'min_correlation': float(min(correlations)) if correlations else 0.0,
                'correlation_std': float(np.std(correlations)) if correlations else 0.0,
                'correlation_matrix': correlation_matrix
            }
            
            return diversity_metrics
            
        except Exception as e:
            logger.error(f"모델 다양성 분석 실패: {e}")
            return {'diversity_score': 0.0, 'correlation_matrix': {}}
    
    def analyze_model_stability(self, 
                              models_predictions: Dict[str, np.ndarray],
                              y_true: np.ndarray,
                              n_bootstrap: int = 30) -> Dict[str, Dict[str, float]]:
        """모델 안정성 분석 (부트스트래핑)"""
        
        stability_results = {}
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                # 배열 타입 변환 및 길이 확인
                y_true_array = np.asarray(y_true).flatten()
                y_pred_array = np.asarray(y_pred_proba).flatten()
                
                # 배열 길이 통일
                min_len = min(len(y_true_array), len(y_pred_array))
                if min_len == 0:
                    logger.warning(f"{model_name}: 빈 배열로 인해 안정성 분석 불가")
                    stability_results[model_name] = self._get_default_model_stability()
                    continue
                
                y_true_array = y_true_array[:min_len]
                y_pred_array = y_pred_array[:min_len]
                
                scores = []
                ctr_biases = []
                
                sample_size = min(min_len, 10000)  # 메모리 절약
                
                for i in range(n_bootstrap):
                    try:
                        # 부트스트래핑 샘플링
                        indices = np.random.choice(min_len, size=sample_size, replace=True)
                        y_true_bootstrap = y_true_array[indices]
                        y_pred_bootstrap = y_pred_array[indices]
                        
                        # 유효성 검사
                        if len(np.unique(y_true_bootstrap)) < 2:
                            continue
                        
                        # 점수 및 CTR 편향 계산
                        score = self.metrics_calculator.ctr_optimized_score(y_true_bootstrap, y_pred_bootstrap)
                        ctr_bias = abs(y_pred_bootstrap.mean() - y_true_bootstrap.mean())
                        
                        if score > 0 and not np.isnan(score) and not np.isinf(score):
                            scores.append(score)
                            ctr_biases.append(ctr_bias)
                            
                    except Exception as e:
                        logger.debug(f"{model_name} 부트스트래핑 {i+1} 실패: {e}")
                        continue
                
                if len(scores) >= 3:
                    scores = np.array(scores)
                    ctr_biases = np.array(ctr_biases)
                    
                    stability_results[model_name] = {
                        'mean_score': float(scores.mean()),
                        'std_score': float(scores.std()),
                        'cv_score': float(scores.std() / scores.mean()) if scores.mean() > 0 else float('inf'),
                        'ci_lower': float(np.percentile(scores, 2.5)),
                        'ci_upper': float(np.percentile(scores, 97.5)),
                        'score_range': float(np.percentile(scores, 97.5) - np.percentile(scores, 2.5)),
                        'mean_ctr_bias': float(ctr_biases.mean()),
                        'std_ctr_bias': float(ctr_biases.std()),
                        'max_ctr_bias': float(ctr_biases.max()),
                        'stability_ratio': float(1.0 / (1.0 + scores.std())) if scores.std() > 0 else 1.0
                    }
                else:
                    logger.warning(f"{model_name}: 유효한 부트스트래핑 점수가 부족")
                    stability_results[model_name] = self._get_default_model_stability()
                
            except Exception as e:
                logger.error(f"{model_name} 안정성 분석 실패: {str(e)}")
                stability_results[model_name] = self._get_default_model_stability()
        
        return stability_results
    
    def _get_default_model_stability(self) -> Dict[str, float]:
        """기본 모델 안정성 지표 반환"""
        return {
            'mean_score': 0.0, 
            'std_score': 0.0, 
            'cv_score': float('inf'),
            'ci_lower': 0.0, 
            'ci_upper': 0.0, 
            'score_range': 0.0,
            'mean_ctr_bias': 0.0, 
            'std_ctr_bias': 0.0, 
            'max_ctr_bias': 0.0,
            'stability_ratio': 0.0
        }

class EvaluationVisualizer:
    """평가 결과 시각화 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        
        if self.matplotlib_available:
            # matplotlib 스타일 안전하게 설정
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
            
            # 1. ROC 곡선
            self._plot_roc_curves_subplot(axes[0], models_predictions, y_true)
            
            # 2. Precision-Recall 곡선
            self._plot_pr_curves_subplot(axes[1], models_predictions, y_true)
            
            # 3. CTR 편향 비교
            self._plot_ctr_bias_subplot(axes[2], models_predictions, y_true)
            
            # 4. 예측 분포
            self._plot_prediction_distributions_subplot(axes[3], models_predictions, y_true)
            
            # 5. 보정 다이어그램
            self._plot_calibration_subplot(axes[4], models_predictions, y_true)
            
            # 6. 성능 레이더 차트
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
            
            # 수치 표시
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
            
            # 실제 CTR 표시
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
            
            # 성능 지표 선택
            metric_names = ['AP', 'AUC', 'F1', 'CTR Accuracy', 'Calibration']
            
            for model_name, y_pred_proba in models_predictions.items():
                metrics = metrics_calc.comprehensive_evaluation(y_true, y_pred_proba)
                
                # 지표 정규화 (0-1 스케일)
                values = [
                    metrics.get('ap', 0.0),
                    metrics.get('auc', 0.0),
                    metrics.get('f1', 0.0),
                    1.0 - min(metrics.get('ctr_absolute_error', 1.0), 1.0),  # CTR 정확도
                    1.0 - min(metrics.get('ece', 1.0), 1.0)  # 보정 품질
                ]
                
                # 레이더 차트 각도
                angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
                values += values[:1]  # 닫힌 도형
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
        
        # 모델 비교
        comparator = ModelComparator()
        comparison_df = comparator.compare_models(models_predictions, y_true)
        
        # 최고 성능 모델
        best_model, best_score = comparator.get_best_model('ctr_optimized_score')
        
        # 안정성 분석
        stability_results = comparator.analyze_model_stability(models_predictions, y_true, n_bootstrap=30)
        
        # 다양성 분석
        diversity_results = comparator.analyze_model_diversity(models_predictions)
        
        # 비즈니스 지표 (대표 모델)
        business_metrics = {}
        if best_model and best_model in models_predictions:
            business_metrics = self.metrics_calculator.calculate_business_metrics(
                y_true, models_predictions[best_model]
            )
        
        # 보고서 데이터 구성
        report = {
            'summary': {
                'total_models': len(models_predictions),
                'best_model': best_model,
                'best_score': best_score,
                'data_size': len(y_true),
                'actual_ctr': y_true.mean(),
                'target_score': self.metrics_calculator.target_score,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'detailed_comparison': comparison_df.to_dict() if not comparison_df.empty else {},
            'model_rankings': comparator.rank_models('ctr_optimized_score').to_dict() if not comparison_df.empty else {},
            'stability_analysis': stability_results,
            'diversity_analysis': diversity_results,
            'business_metrics': business_metrics
        }
        
        # 상세 분석 추가
        if not comparison_df.empty:
            report['performance_analysis'] = self._analyze_performance_patterns(comparison_df)
            report['recommendations'] = self._generate_recommendations(comparison_df, stability_results)
        
        # 시각화 생성 (출력 디렉터리가 지정된 경우)
        if output_dir and MATPLOTLIB_AVAILABLE:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # 종합 평가 시각화
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
    
    def _analyze_performance_patterns(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """성능 패턴 분석"""
        try:
            analysis = {}
            
            # 성능 지표 간 상관관계
            correlation_cols = ['ctr_optimized_score', 'combined_score', 'ap', 'auc', 'f1']
            available_cols = [col for col in correlation_cols if col in comparison_df.columns]
            
            if len(available_cols) > 1:
                correlation_matrix = comparison_df[available_cols].corr()
                analysis['metric_correlations'] = correlation_matrix.to_dict()
            
            # CTR 편향 패턴
            if 'ctr_bias' in comparison_df.columns:
                ctr_biases = comparison_df['ctr_bias']
                analysis['ctr_bias_patterns'] = {
                    'mean_bias': ctr_biases.mean(),
                    'std_bias': ctr_biases.std(),
                    'max_absolute_bias': ctr_biases.abs().max(),
                    'models_with_low_bias': (ctr_biases.abs() < 0.001).sum(),
                    'overestimating_models': (ctr_biases > 0.001).sum(),
                    'underestimating_models': (ctr_biases < -0.001).sum()
                }
            
            # 안정성 패턴
            stability_cols = [col for col in comparison_df.columns if 'stability' in col]
            if stability_cols:
                analysis['stability_patterns'] = {
                    col: {
                        'mean': comparison_df[col].mean(),
                        'std': comparison_df[col].std(),
                        'best_model': comparison_df[col].idxmax(),
                        'worst_model': comparison_df[col].idxmin()
                    } for col in stability_cols
                }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"성능 패턴 분석 실패: {e}")
            return {}
    
    def _generate_recommendations(self, comparison_df: pd.DataFrame, stability_results: Dict) -> List[str]:
        """모델 권장사항 생성"""
        recommendations = []
        
        try:
            # CTR 편향 기반 권장사항
            if 'ctr_bias' in comparison_df.columns:
                high_bias_models = comparison_df[comparison_df['ctr_bias'].abs() > 0.002]
                if not high_bias_models.empty:
                    recommendations.append(
                        f"CTR 편향이 큰 모델들({', '.join(high_bias_models.index)})에 대해 "
                        "Calibration 기법 적용을 권장합니다."
                    )
            
            # 안정성 기반 권장사항
            unstable_models = []
            for model_name, stability in stability_results.items():
                if stability.get('cv_score', float('inf')) > 0.1:
                    unstable_models.append(model_name)
            
            if unstable_models:
                recommendations.append(
                    f"안정성이 낮은 모델들({', '.join(unstable_models)})에 대해 "
                    "앙상블 기법 적용을 권장합니다."
                )
            
            # 성능 기반 권장사항
            if 'ctr_optimized_score' in comparison_df.columns:
                low_performance_models = comparison_df[comparison_df['ctr_optimized_score'] < 0.3]
                if not low_performance_models.empty:
                    recommendations.append(
                        f"성능이 낮은 모델들({', '.join(low_performance_models.index)})에 대해 "
                        "하이퍼파라미터 재튜닝을 권장합니다."
                    )
            
            # 일반적인 권장사항
            if len(comparison_df) > 1:
                recommendations.append("다중 모델 앙상블을 통해 전체적인 성능 향상을 기대할 수 있습니다.")
            
            if not recommendations:
                recommendations.append("모든 모델이 양호한 성능을 보이고 있습니다.")
            
        except Exception as e:
            logger.warning(f"권장사항 생성 실패: {e}")
            recommendations.append("권장사항 생성 중 오류가 발생했습니다.")
        
        return recommendations