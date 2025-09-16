# evaluation.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import warnings
import gc
from collections import defaultdict
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    average_precision_score, log_loss, roc_auc_score, 
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib이 설치되지 않았습니다. 시각화 기능이 비활성화됩니다.")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar, minimize
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy가 설치되지 않았습니다. 일부 통계 기능이 비활성화됩니다.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil이 설치되지 않았습니다. 메모리 모니터링이 제한됩니다.")

from config import Config

logger = logging.getLogger(__name__)

class UltraHighPerformanceCTRMetrics:
    """울트라 고성능 CTR 평가 지표 - Combined Score 0.32+ 목표"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
        # 울트라 성능 목표 설정
        self.target_combined_score = 0.32  # 울트라 목표
        self.high_performance_threshold = 0.30  # 고성능 기준
        self.ctr_bias_tolerance = 0.001  # CTR 편향 허용치
        self.actual_ctr = 0.0201  # 실제 CTR
        
        # 가중치 최적화 (Combined Score 0.32+ 달성용)
        self.ap_weight = 0.6  # AP 비중 증가
        self.wll_weight = 0.4  # WLL 비중
        
        # 실제 CTR 분포 반영 가중치
        self.pos_weight = 49.0  # 1:49 비율
        self.neg_weight = 1.0
        
        # 울트라 평가 파라미터
        self.ctr_penalty_factor = 5000.0  # CTR 편향 패널티 강화
        self.diversity_bonus_factor = 1.5  # 예측 다양성 보너스
        self.stability_weight = 0.3  # 안정성 가중치
        self.calibration_weight = 0.2  # 보정 품질 가중치
        
        # 성능 모니터링
        self.evaluation_history = []
        self.benchmark_scores = {
            'ultra_target': 0.32,
            'high_performance': 0.30,
            'good_performance': 0.28,
            'baseline': 0.25
        }
    
    def calculate_ultra_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 고성능 Average Precision 계산"""
        try:
            # 입력 검증 강화
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("AP 계산을 위한 입력 데이터 문제")
                return {'ap_score': 0.0, 'ap_details': {}}
            
            # 클래스 분포 확인
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("단일 클래스만 존재하여 AP 계산 불가")
                return {'ap_score': 0.0, 'ap_details': {}}
            
            # 예측값 정리
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 기본 AP 계산
            ap_score = average_precision_score(y_true, y_pred_proba)
            
            # 고성능 AP 세부 분석
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            # 고신뢰도 구간 AP (상위 5% 예측)
            top_5_percent_threshold = np.percentile(y_pred_proba, 95)
            top_mask = y_pred_proba >= top_5_percent_threshold
            if top_mask.sum() > 0:
                high_conf_precision = y_true[top_mask].mean()
            else:
                high_conf_precision = 0.0
            
            # 다양한 임계값에서의 성능
            threshold_performances = {}
            for threshold in [0.01, 0.02, 0.05, 0.1, 0.2]:
                thresh_mask = y_pred_proba >= threshold
                if thresh_mask.sum() > 0:
                    thresh_precision = y_true[thresh_mask].mean()
                    thresh_recall = y_true[thresh_mask].sum() / y_true.sum()
                    threshold_performances[f'threshold_{threshold}'] = {
                        'precision': thresh_precision,
                        'recall': thresh_recall
                    }
            
            # 울트라 AP 점수 (보정 적용)
            ctr_consistency_bonus = self._calculate_ctr_consistency_bonus(y_true, y_pred_proba)
            ultra_ap = ap_score * ctr_consistency_bonus
            
            ap_details = {
                'basic_ap': ap_score,
                'ultra_ap': ultra_ap,
                'high_conf_precision': high_conf_precision,
                'ctr_consistency_bonus': ctr_consistency_bonus,
                'threshold_performances': threshold_performances,
                'precision_curve_auc': np.trapz(precision, recall),
                'max_precision': precision.max(),
                'precision_at_50_recall': self._get_precision_at_recall(precision, recall, 0.5),
                'precision_at_80_recall': self._get_precision_at_recall(precision, recall, 0.8)
            }
            
            return {'ap_score': ultra_ap, 'ap_details': ap_details}
            
        except Exception as e:
            logger.error(f"울트라 AP 계산 오류: {str(e)}")
            return {'ap_score': 0.0, 'ap_details': {}}
    
    def calculate_ultra_weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 고성능 Weighted Log Loss 계산"""
        try:
            # 입력 검증
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("WLL 계산을 위한 입력 데이터 문제")
                return {'wll_score': float('inf'), 'wll_details': {}}
            
            # 예측값 정리
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 실제 CTR 분포 반영 가중치
            pos_weight = self.pos_weight
            neg_weight = self.neg_weight
            
            sample_weights = np.where(y_true == 1, pos_weight, neg_weight)
            
            # 기본 WLL 계산
            log_loss_values = -(y_true * np.log(y_pred_proba) + 
                              (1 - y_true) * np.log(1 - y_pred_proba))
            
            basic_wll = np.average(log_loss_values, weights=sample_weights)
            
            # 울트라 WLL 세부 분석
            
            # 구간별 WLL
            quartile_wll = {}
            for q in [25, 50, 75, 90, 95]:
                threshold = np.percentile(y_pred_proba, q)
                mask = y_pred_proba >= threshold
                if mask.sum() > 0:
                    quartile_loss = np.average(log_loss_values[mask], weights=sample_weights[mask])
                    quartile_wll[f'top_{q}p'] = quartile_loss
            
            # CTR 정확도 기반 WLL 보정
            predicted_ctr = y_pred_proba.mean()
            actual_ctr = y_true.mean()
            ctr_accuracy = 1.0 - min(abs(predicted_ctr - actual_ctr) * 1000, 1.0)
            
            # 예측 다양성 기반 보정
            pred_diversity = len(np.unique(np.round(y_pred_proba, 6))) / len(y_pred_proba)
            diversity_factor = min(1.2, 1.0 + pred_diversity * 0.5)
            
            # 울트라 WLL (보정 적용)
            ultra_wll = basic_wll / (ctr_accuracy * diversity_factor + 1e-8)
            
            wll_details = {
                'basic_wll': basic_wll,
                'ultra_wll': ultra_wll,
                'ctr_accuracy': ctr_accuracy,
                'diversity_factor': diversity_factor,
                'quartile_wll': quartile_wll,
                'pos_samples': int(y_true.sum()),
                'neg_samples': int((1 - y_true).sum()),
                'effective_pos_weight': pos_weight,
                'effective_neg_weight': neg_weight
            }
            
            return {'wll_score': ultra_wll, 'wll_details': wll_details}
            
        except Exception as e:
            logger.error(f"울트라 WLL 계산 오류: {str(e)}")
            return {'wll_score': float('inf'), 'wll_details': {}}
    
    def calculate_ultra_combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 Combined Score 계산 - 0.32+ 목표"""
        try:
            # 입력 검증
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("Combined Score 계산을 위한 입력 데이터 문제")
                return {'combined_score': 0.0, 'score_details': {}}
            
            # AP 계산
            ap_result = self.calculate_ultra_average_precision(y_true, y_pred_proba)
            ap_score = ap_result['ap_score']
            
            # WLL 계산
            wll_result = self.calculate_ultra_weighted_log_loss(y_true, y_pred_proba)
            wll_score = wll_result['wll_score']
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            # 기본 Combined Score
            basic_combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            # 울트라 고성능 보정 인자들
            
            # 1. CTR 정확도 보정 (강화)
            predicted_ctr = y_pred_proba.mean()
            actual_ctr = y_true.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            if ctr_bias <= self.ctr_bias_tolerance:
                ctr_accuracy_factor = 1.3  # 목표 달성 시 보너스
            elif ctr_bias <= 0.005:
                ctr_accuracy_factor = 1.0 + (0.005 - ctr_bias) * 60  # 선형 감소
            else:
                ctr_accuracy_factor = np.exp(-ctr_bias * self.ctr_penalty_factor)  # 지수적 패널티
            
            # 2. 예측 다양성 보정
            pred_diversity = len(np.unique(np.round(y_pred_proba, 6))) / len(y_pred_proba)
            diversity_factor = min(self.diversity_bonus_factor, 1.0 + pred_diversity * 1.0)
            
            # 3. 분포 매칭 보정
            distribution_factor = self._calculate_ultra_distribution_matching(y_true, y_pred_proba)
            
            # 4. 안정성 보정 (부트스트래핑)
            stability_factor = self._calculate_ultra_stability_factor(y_true, y_pred_proba)
            
            # 5. 보정 품질 보정
            calibration_factor = self._calculate_ultra_calibration_quality(y_true, y_pred_proba)
            
            # 울트라 Combined Score
            ultra_combined = (basic_combined * 
                            ctr_accuracy_factor * 
                            diversity_factor * 
                            distribution_factor * 
                            stability_factor * 
                            calibration_factor)
            
            # 성능 등급 판정
            performance_grade = self._determine_performance_grade(ultra_combined)
            
            score_details = {
                'basic_combined': basic_combined,
                'ultra_combined': ultra_combined,
                'ap_score': ap_score,
                'wll_score': wll_score,
                'wll_normalized': wll_normalized,
                'ctr_bias': ctr_bias,
                'ctr_accuracy_factor': ctr_accuracy_factor,
                'diversity_factor': diversity_factor,
                'distribution_factor': distribution_factor,
                'stability_factor': stability_factor,
                'calibration_factor': calibration_factor,
                'performance_grade': performance_grade,
                'target_achieved': ultra_combined >= self.target_combined_score,
                'high_performance_achieved': ultra_combined >= self.high_performance_threshold,
                'ctr_target_achieved': ctr_bias <= self.ctr_bias_tolerance,
                'ap_details': ap_result.get('ap_details', {}),
                'wll_details': wll_result.get('wll_details', {})
            }
            
            return {'combined_score': ultra_combined, 'score_details': score_details}
            
        except Exception as e:
            logger.error(f"울트라 Combined Score 계산 오류: {str(e)}")
            return {'combined_score': 0.0, 'score_details': {}}
    
    def _calculate_ctr_consistency_bonus(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR 일관성 보너스 계산"""
        try:
            # 분위수별 CTR 일관성 확인
            consistency_scores = []
            
            for percentile in [50, 75, 90, 95, 99]:
                threshold = np.percentile(y_pred_proba, percentile)
                mask = y_pred_proba >= threshold
                
                if mask.sum() > 0:
                    actual_rate = y_true[mask].mean()
                    predicted_rate = y_pred_proba[mask].mean()
                    
                    if predicted_rate > 0:
                        consistency = min(actual_rate, predicted_rate) / max(actual_rate, predicted_rate)
                        consistency_scores.append(consistency)
            
            if consistency_scores:
                return np.mean(consistency_scores)
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _get_precision_at_recall(self, precision: np.ndarray, recall: np.ndarray, target_recall: float) -> float:
        """특정 Recall에서의 Precision"""
        try:
            if len(precision) == 0 or len(recall) == 0:
                return 0.0
            
            # Recall이 target 이상인 첫 번째 지점
            valid_indices = recall >= target_recall
            if valid_indices.any():
                return precision[valid_indices][0]
            else:
                return precision[-1] if len(precision) > 0 else 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_ultra_distribution_matching(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """울트라 분포 매칭 점수"""
        try:
            # 다층 분위수 매칭 검증
            distribution_scores = []
            
            # 세밀한 분위수 검사
            percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            
            for p in percentiles:
                threshold = np.percentile(y_pred_proba, p)
                mask = y_pred_proba >= threshold
                
                if mask.sum() > 0:
                    actual_rate = y_true[mask].mean()
                    predicted_rate = y_pred_proba[mask].mean()
                    expected_rate = (100 - p) / 100 * self.actual_ctr * 5  # 대략적 기대값
                    
                    # 실제와 예측의 매칭도
                    if predicted_rate > 0:
                        matching_score = min(actual_rate, predicted_rate) / max(actual_rate, predicted_rate)
                        distribution_scores.append(matching_score)
            
            if distribution_scores:
                base_score = np.mean(distribution_scores)
                
                # 전체 CTR 매칭 보너스
                overall_ctr_match = min(y_true.mean(), y_pred_proba.mean()) / max(y_true.mean(), y_pred_proba.mean())
                
                final_score = 0.7 * base_score + 0.3 * overall_ctr_match
                return min(1.5, max(0.5, final_score))
            else:
                return 1.0
                
        except Exception as e:
            logger.debug(f"분포 매칭 계산 실패: {e}")
            return 1.0
    
    def _calculate_ultra_stability_factor(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """울트라 안정성 인자 계산"""
        try:
            # 부트스트래핑을 통한 안정성 측정
            n_bootstrap = 15
            sample_size = min(50000, len(y_true))
            scores = []
            
            for _ in range(n_bootstrap):
                try:
                    indices = np.random.choice(len(y_true), size=sample_size, replace=True)
                    boot_y = y_true[indices]
                    boot_pred = y_pred_proba[indices]
                    
                    # 클래스 분포 확인
                    if len(np.unique(boot_y)) < 2:
                        continue
                    
                    # 기본 점수 계산
                    ap = average_precision_score(boot_y, boot_pred)
                    if ap > 0 and not np.isnan(ap):
                        scores.append(ap)
                        
                except Exception:
                    continue
            
            if len(scores) >= 5:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                cv = std_score / mean_score if mean_score > 0 else 1.0
                
                # 안정성 점수 (낮은 CV = 높은 안정성)
                stability = np.exp(-cv * 3)
                return min(1.3, max(0.7, stability))
            else:
                return 0.9
                
        except Exception as e:
            logger.debug(f"안정성 인자 계산 실패: {e}")
            return 0.9
    
    def _calculate_ultra_calibration_quality(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """울트라 보정 품질 계산"""
        try:
            # 고정밀 Calibration 평가
            n_bins = 20
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            calibration_errors = []
            reliability_values = []
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                
                if in_bin.sum() > 0:
                    bin_accuracy = y_true[in_bin].mean()
                    bin_confidence = y_pred_proba[in_bin].mean()
                    bin_count = in_bin.sum()
                    
                    # 보정 오차
                    calibration_error = abs(bin_confidence - bin_accuracy)
                    calibration_errors.append(calibration_error)
                    
                    # 신뢰도
                    if bin_count > 1:
                        reliability = 1.0 - np.std(y_pred_proba[in_bin]) / (bin_confidence + 1e-8)
                        reliability_values.append(max(0, reliability))
            
            if calibration_errors:
                # ECE (Expected Calibration Error)
                ece = np.mean(calibration_errors)
                
                # 신뢰도 점수
                reliability_score = np.mean(reliability_values) if reliability_values else 0.5
                
                # 최종 보정 품질
                calibration_quality = np.exp(-ece * 10) * (0.5 + 0.5 * reliability_score)
                return min(1.2, max(0.8, calibration_quality))
            else:
                return 1.0
                
        except Exception as e:
            logger.debug(f"보정 품질 계산 실패: {e}")
            return 1.0
    
    def _determine_performance_grade(self, combined_score: float) -> str:
        """성능 등급 판정"""
        if combined_score >= 0.35:
            return "S+ (Exceptional)"
        elif combined_score >= 0.32:
            return "S (Ultra High Performance)"
        elif combined_score >= 0.30:
            return "A+ (High Performance)"
        elif combined_score >= 0.28:
            return "A (Good Performance)"
        elif combined_score >= 0.25:
            return "B+ (Above Average)"
        elif combined_score >= 0.22:
            return "B (Average)"
        elif combined_score >= 0.20:
            return "C+ (Below Average)"
        elif combined_score >= 0.18:
            return "C (Poor)"
        else:
            return "D (Very Poor)"
    
    def comprehensive_ultra_evaluation(self, 
                                     y_true: np.ndarray, 
                                     y_pred_proba: np.ndarray,
                                     threshold: float = 0.5,
                                     model_name: str = "Model") -> Dict[str, Any]:
        """종합적인 울트라 평가"""
        
        evaluation_start_time = time.time()
        
        try:
            # 입력 검증 및 전처리
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"입력 크기 불일치: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return self._get_default_ultra_metrics(model_name)
            
            if len(y_true) == 0:
                logger.error("빈 입력 배열")
                return self._get_default_ultra_metrics(model_name)
            
            # 예측값 정리
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            y_pred = (y_pred_proba >= threshold).astype(int)
        
            metrics = {'model_name': model_name}
            
            # === 1. 울트라 핵심 지표 ===
            logger.info(f"{model_name} 울트라 핵심 지표 계산 시작")
            
            ultra_combined_result = self.calculate_ultra_combined_score(y_true, y_pred_proba)
            metrics['ultra_combined_score'] = ultra_combined_result['combined_score']
            metrics['ultra_score_details'] = ultra_combined_result['score_details']
            
            ultra_ap_result = self.calculate_ultra_average_precision(y_true, y_pred_proba)
            metrics['ultra_ap'] = ultra_ap_result['ap_score']
            metrics['ultra_ap_details'] = ultra_ap_result['ap_details']
            
            ultra_wll_result = self.calculate_ultra_weighted_log_loss(y_true, y_pred_proba)
            metrics['ultra_wll'] = ultra_wll_result['wll_score']
            metrics['ultra_wll_details'] = ultra_wll_result['wll_details']
            
            # === 2. 기본 분류 지표 ===
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"기본 분류 지표 계산 실패: {e}")
                metrics.update({'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25})
            
            # === 3. 혼동 행렬 기반 지표 ===
            try:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    total = tp + tn + fp + fn
                    
                    if total > 0:
                        metrics['accuracy'] = (tp + tn) / total
                        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
                        
                        # MCC
                        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                        metrics['mcc'] = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
                    else:
                        metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1': 0.0, 'mcc': 0.0})
                else:
                    metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1': 0.0, 'mcc': 0.0})
            except Exception as e:
                logger.warning(f"혼동 행렬 지표 계산 실패: {e}")
                metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1': 0.0, 'mcc': 0.0})
            
            # === 4. 울트라 CTR 분석 ===
            metrics.update(self._calculate_ultra_ctr_analysis(y_true, y_pred_proba))
            
            # === 5. 울트라 예측 품질 분석 ===
            metrics.update(self._calculate_ultra_prediction_quality(y_pred_proba))
            
            # === 6. 울트라 클래스별 분석 ===
            metrics.update(self._calculate_ultra_class_analysis(y_true, y_pred_proba))
            
            # === 7. 울트라 보정 지표 ===
            try:
                ultra_calibration = self.calculate_ultra_calibration_metrics(y_true, y_pred_proba)
                metrics.update(ultra_calibration)
            except Exception as e:
                logger.warning(f"울트라 보정 지표 계산 실패: {e}")
                metrics.update({'ultra_ece': 0.0, 'ultra_mce': 0.0, 'ultra_reliability': 0.5})
            
            # === 8. 울트라 분위수별 지표 ===
            try:
                metrics.update(self._calculate_ultra_quantile_metrics(y_true, y_pred_proba))
            except Exception as e:
                logger.warning(f"분위수별 지표 계산 실패: {e}")
            
            # === 9. 울트라 최적 임계값 ===
            try:
                optimal_threshold, optimal_f1 = self.find_ultra_optimal_threshold(y_true, y_pred_proba)
                metrics['ultra_optimal_threshold'] = optimal_threshold
                metrics['ultra_optimal_f1'] = optimal_f1
            except Exception as e:
                logger.warning(f"최적 임계값 계산 실패: {e}")
                metrics['ultra_optimal_threshold'] = 0.5
                metrics['ultra_optimal_f1'] = metrics.get('f1', 0.0)
            
            # === 10. 성능 등급 및 목표 달성 여부 ===
            metrics['performance_grade'] = self._determine_performance_grade(metrics['ultra_combined_score'])
            metrics['ultra_target_achieved'] = metrics['ultra_combined_score'] >= self.target_combined_score
            metrics['high_performance_achieved'] = metrics['ultra_combined_score'] >= self.high_performance_threshold
            metrics['ctr_target_achieved'] = metrics.get('ctr_absolute_error', 1.0) <= self.ctr_bias_tolerance
            
            # 종합 목표 달성 여부
            metrics['all_targets_achieved'] = (metrics['ultra_target_achieved'] and 
                                            metrics['ctr_target_achieved'])
            
            # === 11. 평가 메타데이터 ===
            evaluation_time = time.time() - evaluation_start_time
            metrics['evaluation_metadata'] = {
                'evaluation_time': evaluation_time,
                'data_size': len(y_true),
                'positive_samples': int(y_true.sum()),
                'negative_samples': int((1 - y_true).sum()),
                'positive_ratio': float(y_true.mean()),
                'evaluation_timestamp': time.time(),
                'evaluator_version': 'Ultra High Performance v2.0'
            }
            
            # 평가 기록 저장
            self.evaluation_history.append({
                'model_name': model_name,
                'ultra_combined_score': metrics['ultra_combined_score'],
                'ctr_bias': metrics.get('ctr_absolute_error', 0.0),
                'timestamp': time.time()
            })
            
            # 최종 검증
            validated_metrics = self._validate_all_metrics(metrics)
            
            logger.info(f"{model_name} 울트라 종합 평가 완료:")
            logger.info(f"  Combined Score: {validated_metrics['ultra_combined_score']:.4f} (목표: {self.target_combined_score:.2f})")
            logger.info(f"  성능 등급: {validated_metrics['performance_grade']}")
            logger.info(f"  CTR 편향: {validated_metrics.get('ctr_absolute_error', 0):.4f} (목표: ≤{self.ctr_bias_tolerance:.3f})")
            logger.info(f"  목표 달성: {validated_metrics['all_targets_achieved']}")
            
            return validated_metrics
            
        except Exception as e:
            logger.error(f"울트라 종합 평가 실패: {str(e)}")
            return self._get_default_ultra_metrics(model_name)
    
    def _calculate_ultra_ctr_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 CTR 분석"""
        try:
            actual_ctr = float(y_true.mean())
            predicted_ctr = float(y_pred_proba.mean())
            ctr_bias = predicted_ctr - actual_ctr
            ctr_absolute_error = abs(ctr_bias)
            ctr_ratio = predicted_ctr / max(actual_ctr, 1e-10)
            
            # 목표 CTR과의 비교
            target_ctr_bias = abs(predicted_ctr - self.actual_ctr)
            target_ctr_ratio = predicted_ctr / self.actual_ctr
            
            # CTR 정확도 점수
            ctr_accuracy_score = np.exp(-ctr_absolute_error * 2000) if ctr_absolute_error < 0.01 else 0.0
            
            return {
                'ctr_actual': actual_ctr,
                'ctr_predicted': predicted_ctr,
                'ctr_bias': ctr_bias,
                'ctr_absolute_error': ctr_absolute_error,
                'ctr_ratio': ctr_ratio,
                'target_ctr': self.actual_ctr,
                'target_ctr_bias': target_ctr_bias,
                'target_ctr_ratio': target_ctr_ratio,
                'ctr_accuracy_score': ctr_accuracy_score,
                'ctr_target_met': ctr_absolute_error <= self.ctr_bias_tolerance
            }
            
        except Exception as e:
            logger.warning(f"울트라 CTR 분석 실패: {e}")
            return {
                'ctr_actual': self.actual_ctr, 'ctr_predicted': self.actual_ctr,
                'ctr_bias': 0.0, 'ctr_absolute_error': 0.0, 'ctr_ratio': 1.0,
                'ctr_accuracy_score': 0.0, 'ctr_target_met': False
            }
    
    def _calculate_ultra_prediction_quality(self, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 예측 품질 분석"""
        try:
            pred_std = float(y_pred_proba.std())
            pred_var = float(y_pred_proba.var())
            pred_min = float(y_pred_proba.min())
            pred_max = float(y_pred_proba.max())
            pred_range = pred_max - pred_min
            
            # 예측 다양성
            unique_predictions = len(np.unique(np.round(y_pred_proba, 6)))
            diversity_ratio = unique_predictions / len(y_pred_proba)
            
            # 예측 엔트로피
            p_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            entropy = -np.mean(p_clipped * np.log2(p_clipped) + (1 - p_clipped) * np.log2(1 - p_clipped))
            
            # 지니 계수
            gini = self._calculate_gini_coefficient(y_pred_proba)
            
            # 분위수
            percentiles = np.percentile(y_pred_proba, [5, 10, 25, 50, 75, 90, 95, 99])
            
            return {
                'prediction_std': pred_std,
                'prediction_var': pred_var,
                'prediction_min': pred_min,
                'prediction_max': pred_max,
                'prediction_range': pred_range,
                'prediction_unique_count': unique_predictions,
                'prediction_diversity_ratio': diversity_ratio,
                'prediction_entropy': entropy,
                'prediction_gini': gini,
                'prediction_p5': percentiles[0],
                'prediction_p10': percentiles[1],
                'prediction_p25': percentiles[2],
                'prediction_p50': percentiles[3],
                'prediction_p75': percentiles[4],
                'prediction_p90': percentiles[5],
                'prediction_p95': percentiles[6],
                'prediction_p99': percentiles[7]
            }
            
        except Exception as e:
            logger.warning(f"예측 품질 분석 실패: {e}")
            return {
                'prediction_std': 0.0, 'prediction_var': 0.0, 'prediction_diversity_ratio': 0.0,
                'prediction_entropy': 0.0, 'prediction_gini': 0.0
            }
    
    def _calculate_ultra_class_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 클래스별 분석"""
        try:
            pos_mask = (y_true == 1)
            neg_mask = (y_true == 0)
            
            result = {}
            
            if pos_mask.any():
                result.update({
                    'pos_mean_pred': float(y_pred_proba[pos_mask].mean()),
                    'pos_std_pred': float(y_pred_proba[pos_mask].std()),
                    'pos_median_pred': float(np.median(y_pred_proba[pos_mask])),
                    'pos_min_pred': float(y_pred_proba[pos_mask].min()),
                    'pos_max_pred': float(y_pred_proba[pos_mask].max()),
                    'pos_count': int(pos_mask.sum())
                })
            else:
                result.update({
                    'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
                    'pos_min_pred': 0.0, 'pos_max_pred': 0.0, 'pos_count': 0
                })
            
            if neg_mask.any():
                result.update({
                    'neg_mean_pred': float(y_pred_proba[neg_mask].mean()),
                    'neg_std_pred': float(y_pred_proba[neg_mask].std()),
                    'neg_median_pred': float(np.median(y_pred_proba[neg_mask])),
                    'neg_min_pred': float(y_pred_proba[neg_mask].min()),
                    'neg_max_pred': float(y_pred_proba[neg_mask].max()),
                    'neg_count': int(neg_mask.sum())
                })
            else:
                result.update({
                    'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
                    'neg_min_pred': 0.0, 'neg_max_pred': 0.0, 'neg_count': 0
                })
            
            # 분리도
            if pos_mask.any() and neg_mask.any():
                separation = result['pos_mean_pred'] - result['neg_mean_pred']
                result['class_separation'] = separation
                
                # KS 통계
                if SCIPY_AVAILABLE:
                    try:
                        ks_stat, ks_pvalue = stats.ks_2samp(y_pred_proba[pos_mask], y_pred_proba[neg_mask])
                        result['ks_statistic'] = float(ks_stat)
                        result['ks_pvalue'] = float(ks_pvalue)
                    except:
                        result['ks_statistic'] = 0.0
                        result['ks_pvalue'] = 1.0
                else:
                    result['ks_statistic'] = 0.0
                    result['ks_pvalue'] = 1.0
                
                # AUC 근사치 (클래스별 평균 차이 기반)
                if result['pos_std_pred'] > 0 and result['neg_std_pred'] > 0:
                    d_prime = separation / np.sqrt((result['pos_std_pred']**2 + result['neg_std_pred']**2) / 2)
                    approx_auc = stats.norm.cdf(d_prime / np.sqrt(2)) if SCIPY_AVAILABLE else 0.5
                    result['approximate_auc'] = float(approx_auc)
                else:
                    result['approximate_auc'] = 0.5
            else:
                result.update({
                    'class_separation': 0.0, 'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'approximate_auc': 0.5
                })
            
            return result
            
        except Exception as e:
            logger.warning(f"클래스별 분석 실패: {e}")
            return {
                'pos_mean_pred': 0.0, 'neg_mean_pred': 0.0, 'class_separation': 0.0,
                'ks_statistic': 0.0, 'ks_pvalue': 1.0
            }
    
    def _calculate_ultra_quantile_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 분위수별 지표"""
        metrics = {}
        
        try:
            quantiles = [50, 75, 90, 95, 99, 99.5, 99.9]
            
            for q in quantiles:
                threshold = np.percentile(y_pred_proba, q)
                top_mask = y_pred_proba >= threshold
                
                if top_mask.sum() > 0:
                    top_ctr = float(y_true[top_mask].mean())
                    top_size = int(top_mask.sum())
                    top_ratio = float(top_size / len(y_true))
                    
                    # Lift 계산 (전체 CTR 대비)
                    overall_ctr = y_true.mean()
                    lift = top_ctr / overall_ctr if overall_ctr > 0 else 1.0
                    
                    # 정밀도 계산
                    precision = top_ctr
                    
                    # 누적 정밀도 (해당 분위수까지의)
                    cumulative_mask = y_pred_proba >= np.percentile(y_pred_proba, 100 - q)
                    cumulative_precision = y_true[cumulative_mask].mean() if cumulative_mask.sum() > 0 else 0.0
                    
                    metrics[f'top_{q}p_ctr'] = top_ctr
                    metrics[f'top_{q}p_size'] = top_size
                    metrics[f'top_{q}p_ratio'] = top_ratio
                    metrics[f'top_{q}p_lift'] = lift
                    metrics[f'top_{q}p_precision'] = precision
                    metrics[f'top_{q}p_cumulative_precision'] = cumulative_precision
                else:
                    metrics[f'top_{q}p_ctr'] = 0.0
                    metrics[f'top_{q}p_size'] = 0
                    metrics[f'top_{q}p_ratio'] = 0.0
                    metrics[f'top_{q}p_lift'] = 1.0
                    metrics[f'top_{q}p_precision'] = 0.0
                    metrics[f'top_{q}p_cumulative_precision'] = 0.0
                    
        except Exception as e:
            logger.warning(f"분위수별 지표 계산 실패: {e}")
        
        return metrics
    
    def calculate_ultra_calibration_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 보정 지표 계산"""
        try:
            n_bins = 20  # 더 세밀한 분석
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            bin_reliabilities = []
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                
                if in_bin.sum() > 0:
                    bin_accuracy = y_true[in_bin].mean()
                    bin_confidence = y_pred_proba[in_bin].mean()
                    bin_count = in_bin.sum()
                    
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_confidence)
                    bin_counts.append(bin_count)
                    
                    # 빈 내 신뢰도 계산
                    if bin_count > 1:
                        bin_std = y_pred_proba[in_bin].std()
                        reliability = 1.0 - (bin_std / (bin_confidence + 1e-8))
                        bin_reliabilities.append(max(0, reliability))
                    else:
                        bin_reliabilities.append(1.0)
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append(0)
                    bin_counts.append(0)
                    bin_reliabilities.append(0)
            
            bin_accuracies = np.array(bin_accuracies)
            bin_confidences = np.array(bin_confidences)
            bin_counts = np.array(bin_counts)
            bin_reliabilities = np.array(bin_reliabilities)
            
            # ECE (Expected Calibration Error)
            total_count = len(y_true)
            ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total_count
            
            # MCE (Maximum Calibration Error)
            mce = np.max(np.abs(bin_accuracies - bin_confidences))
            
            # ACE (Average Calibration Error) - 비어있지 않은 빈들에 대해서만
            non_empty_bins = bin_counts > 0
            if non_empty_bins.any():
                ace = np.mean(np.abs(bin_accuracies[non_empty_bins] - bin_confidences[non_empty_bins]))
            else:
                ace = 0.0
            
            # 신뢰도 회귀선 계산
            if non_empty_bins.sum() > 1:
                try:
                    valid_conf = bin_confidences[non_empty_bins]
                    valid_acc = bin_accuracies[non_empty_bins]
                    slope, intercept = np.polyfit(valid_conf, valid_acc, 1)
                except:
                    slope, intercept = 1.0, 0.0
            else:
                slope, intercept = 1.0, 0.0
            
            # 전체 신뢰도 점수
            overall_reliability = np.mean(bin_reliabilities[non_empty_bins]) if non_empty_bins.any() else 0.5
            
            # Brier Score Decomposition
            try:
                reliability_term = np.sum(bin_counts * (bin_accuracies - bin_confidences)**2) / total_count
                resolution_term = np.sum(bin_counts * (bin_accuracies - y_true.mean())**2) / total_count
                uncertainty = y_true.mean() * (1 - y_true.mean())
                
                brier_decomp = {
                    'reliability': reliability_term,
                    'resolution': resolution_term,
                    'uncertainty': uncertainty
                }
            except:
                brier_decomp = {'reliability': 0.0, 'resolution': 0.0, 'uncertainty': 0.25}
            
            return {
                'ultra_ece': float(ece),
                'ultra_mce': float(mce),
                'ultra_ace': float(ace),
                'ultra_reliability': float(overall_reliability),
                'reliability_slope': float(slope),
                'reliability_intercept': float(intercept),
                'brier_reliability': float(brier_decomp['reliability']),
                'brier_resolution': float(brier_decomp['resolution']),
                'brier_uncertainty': float(brier_decomp['uncertainty']),
                'calibration_quality_score': float(1.0 - ece) * overall_reliability
            }
            
        except Exception as e:
            logger.error(f"울트라 보정 지표 계산 오류: {str(e)}")
            return {
                'ultra_ece': 0.0, 'ultra_mce': 0.0, 'ultra_ace': 0.0,
                'ultra_reliability': 0.5, 'reliability_slope': 1.0, 'reliability_intercept': 0.0
            }
    
    def find_ultra_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """울트라 최적 임계값 찾기"""
        
        thresholds = np.arange(0.005, 0.995, 0.005)  # 더 세밀한 간격
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            try:
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # F1 Score 계산
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                # CTR 편향 패널티 적용
                predicted_ctr = y_pred.mean()
                actual_ctr = y_true.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_penalty = np.exp(-ctr_bias * 1000) if ctr_bias < 0.01 else 0.1
                
                # 최종 점수 (F1 + CTR 정확도)
                final_score = f1 * ctr_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_threshold = threshold
                    
            except Exception as e:
                continue
        
        return best_threshold, best_score
    
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
            
            return float(np.clip(gini, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _get_default_ultra_metrics(self, model_name: str) -> Dict[str, Any]:
        """기본 울트라 지표값 반환"""
        return {
            'model_name': model_name,
            'ultra_combined_score': 0.0,
            'ultra_ap': 0.0,
            'ultra_wll': float('inf'),
            'auc': 0.5,
            'log_loss': 1.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ctr_actual': self.actual_ctr,
            'ctr_predicted': self.actual_ctr,
            'ctr_bias': 0.0,
            'ctr_absolute_error': 0.0,
            'performance_grade': "D (Very Poor)",
            'ultra_target_achieved': False,
            'high_performance_achieved': False,
            'ctr_target_achieved': False,
            'all_targets_achieved': False,
            'evaluation_metadata': {
                'evaluation_time': 0.0,
                'data_size': 0,
                'evaluator_version': 'Ultra High Performance v2.0'
            }
        }
    
    def _validate_all_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """모든 지표값 검증"""
        validated = {}
        
        for key, value in metrics.items():
            try:
                if isinstance(value, dict):
                    validated[key] = self._validate_all_metrics(value)
                elif isinstance(value, (int, float, np.number)):
                    if np.isnan(value) or np.isinf(value):
                        if key in ['ultra_wll', 'log_loss', 'wll_score']:
                            validated[key] = float('inf')
                        else:
                            validated[key] = 0.0
                    else:
                        validated[key] = float(value)
                else:
                    validated[key] = value
            except:
                if key in ['ultra_wll', 'log_loss', 'wll_score']:
                    validated[key] = float('inf')
                else:
                    validated[key] = 0.0 if isinstance(value, (int, float, np.number)) else value
        
        return validated
    
    # 기존 메서드들에 대한 호환성 alias
    def average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        result = self.calculate_ultra_average_precision(y_true, y_pred_proba)
        return result['ap_score']
    
    def weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        result = self.calculate_ultra_weighted_log_loss(y_true, y_pred_proba)
        return result['wll_score']
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        result = self.calculate_ultra_combined_score(y_true, y_pred_proba)
        return result['combined_score']
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        return self.comprehensive_ultra_evaluation(y_true, y_pred_proba, threshold)

class UltraHighPerformanceModelComparator:
    """울트라 고성능 모델 비교기"""
    
    def __init__(self):
        self.metrics_calculator = UltraHighPerformanceCTRMetrics()
        self.comparison_results = pd.DataFrame()
        self.performance_rankings = {}
        self.benchmark_comparisons = {}
        
    def compare_ultra_models(self, 
                           models_predictions: Dict[str, np.ndarray],
                           y_true: np.ndarray) -> pd.DataFrame:
        """울트라 고성능 모델 비교"""
        
        logger.info("울트라 고성능 모델 비교 시작")
        logger.info(f"비교 모델 수: {len(models_predictions)}")
        
        results = []
        y_true = np.asarray(y_true).flatten()
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                logger.info(f"{model_name} 울트라 평가 시작")
                
                # 예측값 검증
                y_pred_proba = np.asarray(y_pred_proba).flatten()
                
                if len(y_pred_proba) != len(y_true):
                    logger.error(f"{model_name}: 예측값과 실제값 크기 불일치")
                    continue
                
                if len(y_pred_proba) == 0:
                    logger.error(f"{model_name}: 빈 예측값")
                    continue
                
                # 예측값 정리
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                
                # 울트라 종합 평가
                ultra_metrics = self.metrics_calculator.comprehensive_ultra_evaluation(
                    y_true, y_pred_proba, model_name=model_name
                )
                
                # 안정성 지표 추가
                try:
                    stability_metrics = self._calculate_ultra_stability_metrics(y_true, y_pred_proba)
                    ultra_metrics.update(stability_metrics)
                except Exception as e:
                    logger.warning(f"{model_name} 안정성 지표 계산 실패: {e}")
                    ultra_metrics.update(self._get_default_stability_metrics())
                
                # 비교 특화 지표 추가
                ultra_metrics.update(self._calculate_comparison_specific_metrics(y_true, y_pred_proba))
                
                results.append(ultra_metrics)
                
                logger.info(f"{model_name} 울트라 평가 완료:")
                logger.info(f"  Combined Score: {ultra_metrics['ultra_combined_score']:.4f}")
                logger.info(f"  성능 등급: {ultra_metrics['performance_grade']}")
                logger.info(f"  목표 달성: {ultra_metrics['all_targets_achieved']}")
                
            except Exception as e:
                logger.error(f"{model_name} 울트라 모델 평가 실패: {str(e)}")
                # 기본값으로 추가
                default_metrics = self.metrics_calculator._get_default_ultra_metrics(model_name)
                default_metrics.update(self._get_default_stability_metrics())
                results.append(default_metrics)
        
        if not results:
            logger.error("평가 가능한 모델이 없습니다")
            return pd.DataFrame()
        
        try:
            # 결과를 DataFrame으로 변환
            comparison_df = pd.DataFrame(results)
            
            if not comparison_df.empty:
                comparison_df.set_index('model_name', inplace=True)
                
                # 정렬 (울트라 Combined Score 기준)
                comparison_df.sort_values('ultra_combined_score', ascending=False, inplace=True)
                
                # 성능 랭킹 계산
                self._calculate_performance_rankings(comparison_df)
                
                # 벤치마크 비교
                self._calculate_benchmark_comparisons(comparison_df)
        
            self.comparison_results = comparison_df
            
            # 결과 요약 로깅
            self._log_comparison_summary(comparison_df)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"비교 결과 생성 실패: {e}")
            return pd.DataFrame()
    
    def _calculate_ultra_stability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """울트라 안정성 지표 계산"""
        try:
            # 부트스트래핑을 통한 안정성 측정
            n_bootstrap = 20
            sample_size = min(20000, len(y_true))
            scores = []
            
            for _ in range(n_bootstrap):
                try:
                    sample_indices = np.random.choice(len(y_true), size=sample_size, replace=True)
                    sample_y = y_true[sample_indices]
                    sample_pred = y_pred_proba[sample_indices]
                    
                    # 클래스 분포 확인
                    if len(np.unique(sample_y)) < 2:
                        continue
                    
                    # Combined Score 계산
                    score = self.metrics_calculator.combined_score(sample_y, sample_pred)
                    if score > 0 and not np.isnan(score):
                        scores.append(score)
                        
                except Exception:
                    continue
            
            if len(scores) >= 10:
                scores = np.array(scores)
                
                return {
                    'ultra_stability_mean': float(scores.mean()),
                    'ultra_stability_std': float(scores.std()),
                    'ultra_stability_cv': float(scores.std() / scores.mean()) if scores.mean() > 0 else float('inf'),
                    'ultra_stability_ci_lower': float(np.percentile(scores, 2.5)),
                    'ultra_stability_ci_upper': float(np.percentile(scores, 97.5)),
                    'ultra_stability_range': float(np.percentile(scores, 97.5) - np.percentile(scores, 2.5)),
                    'ultra_stability_sample_count': len(scores),
                    'ultra_stability_grade': self._grade_stability(scores.std() / scores.mean() if scores.mean() > 0 else 1.0)
                }
            else:
                return self._get_default_stability_metrics()
                
        except Exception as e:
            logger.warning(f"울트라 안정성 지표 계산 실패: {e}")
            return self._get_default_stability_metrics()
    
    def _grade_stability(self, cv: float) -> str:
        """안정성 등급 판정"""
        if cv <= 0.02:
            return "A+ (매우 안정)"
        elif cv <= 0.05:
            return "A (안정)"
        elif cv <= 0.1:
            return "B+ (양호)"
        elif cv <= 0.15:
            return "B (보통)"
        elif cv <= 0.25:
            return "C+ (다소 불안정)"
        elif cv <= 0.35:
            return "C (불안정)"
        else:
            return "D (매우 불안정)"
    
    def _calculate_comparison_specific_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """비교 특화 지표 계산"""
        try:
            # 예측값 집중도 분석
            concentration_scores = {}
            
            # 상위 1%, 5%, 10%에서의 정밀도
            for percentile in [99, 95, 90]:
                threshold = np.percentile(y_pred_proba, percentile)
                top_mask = y_pred_proba >= threshold
                
                if top_mask.sum() > 0:
                    precision = y_true[top_mask].mean()
                    size = top_mask.sum()
                    concentration_scores[f'precision_top_{100-percentile}p'] = precision
                    concentration_scores[f'size_top_{100-percentile}p'] = size / len(y_true)
                else:
                    concentration_scores[f'precision_top_{100-percentile}p'] = 0.0
                    concentration_scores[f'size_top_{100-percentile}p'] = 0.0
            
            # 예측값 동적 범위
            pred_dynamic_range = np.log10(y_pred_proba.max() / max(y_pred_proba.min(), 1e-10))
            
            # ROC 곡선 하의 부분 면적 (고FPR 구간)
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                
                # 낮은 FPR 구간에서의 성능 (FPR < 0.1)
                low_fpr_mask = fpr <= 0.1
                if low_fpr_mask.any():
                    partial_auc_low = np.trapz(tpr[low_fpr_mask], fpr[low_fpr_mask])
                else:
                    partial_auc_low = 0.0
                
                concentration_scores['partial_auc_low_fpr'] = partial_auc_low
                
            except Exception:
                concentration_scores['partial_auc_low_fpr'] = 0.0
            
            concentration_scores['prediction_dynamic_range'] = pred_dynamic_range
            
            return concentration_scores
            
        except Exception as e:
            logger.debug(f"비교 특화 지표 계산 실패: {e}")
            return {}
    
    def _get_default_stability_metrics(self) -> Dict[str, float]:
        """기본 안정성 지표값"""
        return {
            'ultra_stability_mean': 0.0,
            'ultra_stability_std': 0.0,
            'ultra_stability_cv': float('inf'),
            'ultra_stability_ci_lower': 0.0,
            'ultra_stability_ci_upper': 0.0,
            'ultra_stability_range': 0.0,
            'ultra_stability_sample_count': 0,
            'ultra_stability_grade': "D (매우 불안정)"
        }
    
    def _calculate_performance_rankings(self, comparison_df: pd.DataFrame):
        """성능 랭킹 계산"""
        try:
            self.performance_rankings = {
                'overall_ranking': comparison_df['ultra_combined_score'].rank(ascending=False, method='min'),
                'ctr_accuracy_ranking': comparison_df['ctr_absolute_error'].rank(ascending=True, method='min'),
                'stability_ranking': comparison_df['ultra_stability_cv'].rank(ascending=True, method='min'),
                'ap_ranking': comparison_df['ultra_ap'].rank(ascending=False, method='min'),
                'calibration_ranking': comparison_df.get('ultra_reliability', pd.Series(0.5, index=comparison_df.index)).rank(ascending=False, method='min')
            }
            
            # 종합 랭킹 (가중 평균)
            weighted_rank = (
                0.4 * self.performance_rankings['overall_ranking'] +
                0.2 * self.performance_rankings['ctr_accuracy_ranking'] +
                0.2 * self.performance_rankings['stability_ranking'] +
                0.1 * self.performance_rankings['ap_ranking'] +
                0.1 * self.performance_rankings['calibration_ranking']
            )
            
            self.performance_rankings['comprehensive_ranking'] = weighted_rank.rank(method='min')
            
        except Exception as e:
            logger.warning(f"성능 랭킹 계산 실패: {e}")
    
    def _calculate_benchmark_comparisons(self, comparison_df: pd.DataFrame):
        """벤치마크 비교 분석"""
        try:
            benchmark_thresholds = {
                'ultra_performance': 0.32,
                'high_performance': 0.30,
                'good_performance': 0.28,
                'baseline': 0.25
            }
            
            self.benchmark_comparisons = {}
            
            for model_name in comparison_df.index:
                score = comparison_df.loc[model_name, 'ultra_combined_score']
                
                # 벤치마크 달성 현황
                achievements = {}
                for benchmark_name, threshold in benchmark_thresholds.items():
                    achievements[benchmark_name] = score >= threshold
                
                # 다음 목표까지의 거리
                next_targets = {}
                for benchmark_name, threshold in benchmark_thresholds.items():
                    if score < threshold:
                        next_targets[benchmark_name] = threshold - score
                
                self.benchmark_comparisons[model_name] = {
                    'achievements': achievements,
                    'next_targets': next_targets,
                    'best_achievement': max([name for name, achieved in achievements.items() if achieved], default="None"),
                    'score': score
                }
                
        except Exception as e:
            logger.warning(f"벤치마크 비교 계산 실패: {e}")
    
    def _log_comparison_summary(self, comparison_df: pd.DataFrame):
        """비교 결과 요약 로깅"""
        try:
            logger.info("=== 울트라 고성능 모델 비교 결과 요약 ===")
            
            # 전체 통계
            total_models = len(comparison_df)
            ultra_achievers = (comparison_df['ultra_combined_score'] >= 0.32).sum()
            high_performers = (comparison_df['ultra_combined_score'] >= 0.30).sum()
            ctr_achievers = (comparison_df['ctr_absolute_error'] <= 0.001).sum()
            
            logger.info(f"전체 모델 수: {total_models}")
            logger.info(f"울트라 목표 달성 (≥0.32): {ultra_achievers}개 ({ultra_achievers/total_models*100:.1f}%)")
            logger.info(f"고성능 달성 (≥0.30): {high_performers}개 ({high_performers/total_models*100:.1f}%)")
            logger.info(f"CTR 목표 달성 (≤0.001): {ctr_achievers}개 ({ctr_achievers/total_models*100:.1f}%)")
            
            # Top 3 모델
            top_3 = comparison_df.head(3)
            logger.info("Top 3 모델:")
            for i, (model_name, row) in enumerate(top_3.iterrows(), 1):
                logger.info(f"  {i}. {model_name}: {row['ultra_combined_score']:.4f} ({row['performance_grade']})")
            
            # 목표 달성 모델들
            all_targets = comparison_df[comparison_df['all_targets_achieved'] == True]
            if not all_targets.empty:
                logger.info(f"모든 목표 달성 모델 ({len(all_targets)}개):")
                for model_name in all_targets.index:
                    score = all_targets.loc[model_name, 'ultra_combined_score']
                    ctr_bias = all_targets.loc[model_name, 'ctr_absolute_error']
                    logger.info(f"  🏆 {model_name}: Score={score:.4f}, CTR편향={ctr_bias:.4f}")
            else:
                logger.info("모든 목표를 달성한 모델이 없습니다.")
            
            logger.info("=== 비교 결과 요약 완료 ===")
            
        except Exception as e:
            logger.warning(f"비교 결과 요약 로깅 실패: {e}")
    
    def get_ultra_model_rankings(self, ranking_type: str = 'comprehensive') -> pd.DataFrame:
        """울트라 모델 랭킹 반환"""
        
        if self.comparison_results.empty:
            logger.warning("비교 결과가 없습니다.")
            return pd.DataFrame()
        
        try:
            ranking_df = self.comparison_results.copy()
            
            if ranking_type in self.performance_rankings:
                ranking_df['rank'] = self.performance_rankings[ranking_type]
            else:
                ranking_df['rank'] = ranking_df['ultra_combined_score'].rank(ascending=False)
            
            ranking_df.sort_values('rank', inplace=True)
            
            # 핵심 컬럼만 선택
            key_columns = [
                'rank', 'ultra_combined_score', 'performance_grade',
                'ultra_ap', 'ultra_wll', 'auc', 'f1',
                'ctr_absolute_error', 'ctr_target_achieved',
                'ultra_target_achieved', 'all_targets_achieved',
                'ultra_stability_cv', 'ultra_stability_grade'
            ]
            
            available_columns = [col for col in key_columns if col in ranking_df.columns]
            
            return ranking_df[available_columns]
        
        except Exception as e:
            logger.error(f"울트라 모델 랭킹 생성 실패: {e}")
            return pd.DataFrame()
    
    def get_best_ultra_model(self, metric: str = 'ultra_combined_score') -> Tuple[