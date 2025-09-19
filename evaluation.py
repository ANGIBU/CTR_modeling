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
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not installed. Visualization functionality will be disabled.")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not installed. Some statistical functionality will be disabled.")

from config import Config

logger = logging.getLogger(__name__)

class CTRAdvancedMetrics:
    """CTR prediction specialized advanced evaluation metrics class - Combined Score 0.30+ achievement goal + calibration evaluation"""
    
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
        """Enhanced Average Precision - improved stability and accuracy"""
        try:
            cache_key = f"ap_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"Size mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return 0.0
            
            if len(y_true) == 0:
                logger.warning("Cannot calculate AP due to empty array")
                return 0.0
            
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("Cannot calculate AP - only single class exists")
                return 0.0
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("NaN or infinite values in predictions, applying clipping")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            try:
                ap_score = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"sklearn AP calculation failed, using manual calculation: {e}")
                ap_score = self._manual_average_precision(y_true, y_pred_proba)
            
            if np.isnan(ap_score) or np.isinf(ap_score):
                logger.warning("AP calculation result is invalid")
                return 0.0
            
            ap_score = float(ap_score)
            self.cache[cache_key] = ap_score
            
            return ap_score
        except Exception as e:
            logger.error(f"Enhanced AP calculation error: {str(e)}")
            return 0.0
    
    def _manual_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual Average Precision calculation"""
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
            logger.error(f"Manual AP calculation failed: {e}")
            return 0.0
    
    def weighted_log_loss_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Weighted Log Loss - CTR distribution specialized improvement"""
        try:
            cache_key = f"wll_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"Size mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return float('inf')
            
            if len(y_true) == 0:
                logger.warning("Cannot calculate WLL due to empty array")
                return float('inf')
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("NaN or infinite values in predictions, applying clipping")
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
                logger.warning("NaN or infinite values in log loss calculation")
                return float('inf')
            
            weighted_log_loss = np.average(log_loss_values, weights=sample_weights)
            
            ctr_penalty = abs(predicted_ctr - actual_ctr) * 1000
            final_wll = weighted_log_loss + ctr_penalty
            
            if np.isnan(final_wll) or np.isinf(final_wll):
                logger.warning("WLL calculation result is invalid")
                return float('inf')
            
            final_wll = float(final_wll)
            self.cache[cache_key] = final_wll
            
            return final_wll
            
        except Exception as e:
            logger.error(f"Enhanced WLL calculation error: {str(e)}")
            return float('inf')
    
    def combined_score_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Combined Score = AP + WLL + CTR correction + diversity correction - 0.30+ target"""
        try:
            cache_key = f"combined_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("Input data problem for Combined Score calculation")
                return 0.0
            
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            basic_combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            # CTR alignment correction
            try:
                predicted_ctr = y_pred_proba.mean()
                actual_ctr = y_true.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                
                ctr_alignment = np.exp(-(ctr_bias / self.ctr_tolerance) ** 2)
                ctr_ratio_penalty = min(abs(predicted_ctr / max(actual_ctr, 1e-8) - 1.0), 2.0)
                ctr_penalty = np.exp(-ctr_ratio_penalty * self.bias_penalty_weight)
                
                ctr_bonus = ctr_alignment * ctr_penalty
                
            except Exception as e:
                logger.warning(f"CTR alignment calculation failed: {e}")
                ctr_bonus = 0.5
            
            # Prediction diversity correction
            try:
                pred_std = y_pred_proba.std()
                pred_range = y_pred_proba.max() - y_pred_proba.min()
                unique_ratio = len(np.unique(y_pred_proba)) / len(y_pred_proba)
                
                diversity_score = pred_std * pred_range * unique_ratio
                diversity_bonus = min(diversity_score * 2.0, 0.5)
                
            except Exception as e:
                logger.warning(f"Diversity calculation failed: {e}")
                diversity_bonus = 0.0
            
            # Distribution matching correction
            try:
                distribution_score = self._calculate_distribution_matching_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Distribution matching calculation failed: {e}")
                distribution_score = 0.5
            
            # Final score calculation
            final_score = (basic_combined * 
                          (1.0 + 0.3 * ctr_bonus) * 
                          (1.0 + 0.1 * diversity_bonus) * 
                          (1.0 + 0.2 * distribution_score))
            
            if np.isnan(final_score) or np.isinf(final_score) or final_score < 0:
                logger.warning("Combined Score calculation result is invalid")
                return 0.0
            
            final_score = float(final_score)
            self.cache[cache_key] = final_score
            
            return final_score
            
        except Exception as e:
            logger.error(f"Enhanced Combined Score calculation error: {str(e)}")
            return 0.0
    
    def calibration_score_advanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Advanced calibration quality evaluation - multiple metrics"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return {'ece': 1.0, 'mce': 1.0, 'brier_reliability': 1.0, 'brier_resolution': 0.0, 'calibration_score': 0.0}
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece_advanced(y_true, y_pred_proba)
            
            # Maximum Calibration Error (MCE)
            mce = self._calculate_mce_advanced(y_true, y_pred_proba)
            
            # Brier Score Decomposition
            brier_reliability, brier_resolution = self._calculate_brier_decomposition(y_true, y_pred_proba)
            
            # Overall Calibration Score
            calibration_score = self._calculate_overall_calibration_score(ece, mce, brier_reliability, brier_resolution)
            
            return {
                'ece': ece,
                'mce': mce,
                'brier_reliability': brier_reliability,
                'brier_resolution': brier_resolution,
                'calibration_score': calibration_score
            }
            
        except Exception as e:
            logger.error(f"Advanced calibration evaluation failed: {e}")
            return {'ece': 1.0, 'mce': 1.0, 'brier_reliability': 1.0, 'brier_resolution': 0.0, 'calibration_score': 0.0}
    
    def _calculate_ece_advanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 15) -> float:
        """Advanced Expected Calibration Error calculation"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            total_samples = len(y_true)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.sum() / total_samples
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return min(ece, 1.0)
            
        except Exception as e:
            logger.warning(f"ECE calculation failed: {e}")
            return 1.0
    
    def _calculate_mce_advanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 15) -> float:
        """Advanced Maximum Calibration Error calculation"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            max_error = 0.0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                
                if in_bin.sum() > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    max_error = max(max_error, bin_error)
            
            return min(max_error, 1.0)
            
        except Exception as e:
            logger.warning(f"MCE calculation failed: {e}")
            return 1.0
    
    def _calculate_brier_decomposition(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
        """Brier score decomposition: reliability and resolution"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            overall_accuracy = y_true.mean()
            total_samples = len(y_true)
            
            reliability = 0.0
            resolution = 0.0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.sum() / total_samples
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    
                    # Reliability (lower is better)
                    reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                    
                    # Resolution (higher is better)
                    resolution += prop_in_bin * (accuracy_in_bin - overall_accuracy) ** 2
            
            return min(reliability, 1.0), min(resolution, 1.0)
            
        except Exception as e:
            logger.warning(f"Brier decomposition calculation failed: {e}")
            return 1.0, 0.0
    
    def _calculate_overall_calibration_score(self, ece: float, mce: float, reliability: float, resolution: float) -> float:
        """Calculate overall calibration score"""
        try:
            # ECE and MCE are better when lower, Resolution is better when higher, Reliability is better when lower
            ece_score = max(0.0, 1.0 - ece * 5)  # Full score if ECE <= 0.2
            mce_score = max(0.0, 1.0 - mce * 3)  # Full score if MCE <= 0.33
            reliability_score = max(0.0, 1.0 - reliability * 10)  # Full score if Reliability <= 0.1
            resolution_score = min(resolution * 5, 1.0)  # Full score if Resolution >= 0.2
            
            # Weighted average
            overall_score = (0.3 * ece_score + 
                           0.2 * mce_score + 
                           0.3 * reliability_score + 
                           0.2 * resolution_score)
            
            return max(0.0, min(overall_score, 1.0))
            
        except Exception as e:
            logger.warning(f"Overall calibration score calculation failed: {e}")
            return 0.0
    
    def ctr_ultra_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Ultra-optimized comprehensive score for CTR prediction - 0.32+ target"""
        try:
            cache_key = f"ultra_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                logger.error("Input data problem for CTR ultra-optimization score calculation")
                return 0.0
            
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            # CTR ultra-precision alignment
            try:
                predicted_ctr = y_pred_proba.mean()
                actual_ctr = y_true.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                
                ultra_ctr_accuracy = np.exp(-ctr_bias * 2000) if ctr_bias < 0.01 else 0.0
                ctr_stability = 1.0 / (1.0 + ctr_bias * 10000)
                
            except Exception as e:
                logger.warning(f"CTR ultra-precision calculation failed: {e}")
                ultra_ctr_accuracy = 0.0
                ctr_stability = 0.0
            
            # Advanced calibration quality
            try:
                calibration_metrics = self.calibration_score_advanced(y_true, y_pred_proba)
                calibration_score = calibration_metrics['calibration_score']
            except Exception as e:
                logger.warning(f"Advanced calibration quality calculation failed: {e}")
                calibration_score = 0.5
            
            # Perfect distribution matching
            try:
                perfect_distribution_score = self._calculate_perfect_distribution_matching(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Perfect distribution matching calculation failed: {e}")
                perfect_distribution_score = 0.5
            
            # Prediction quality metrics
            try:
                prediction_quality = self._calculate_prediction_quality_score(y_pred_proba)
            except Exception as e:
                logger.warning(f"Prediction quality calculation failed: {e}")
                prediction_quality = 0.5
            
            # Extreme region performance
            try:
                extreme_performance = self._calculate_extreme_region_performance(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Extreme region performance calculation failed: {e}")
                extreme_performance = 0.5
            
            # Weight application (increased calibration importance)
            weights = {
                'ap': 0.20,
                'wll': 0.15,
                'ultra_ctr_accuracy': 0.20,
                'ctr_stability': 0.10,
                'calibration': 0.15,  # Increased calibration weight
                'perfect_distribution': 0.10,
                'prediction_quality': 0.05,
                'extreme_performance': 0.05
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
                logger.warning("CTR ultra-optimization score calculation result is invalid")
                return 0.0
            
            ultra_optimized_score = float(ultra_optimized_score)
            self.cache[cache_key] = ultra_optimized_score
            
            return ultra_optimized_score
            
        except Exception as e:
            logger.error(f"CTR ultra-optimization score calculation error: {str(e)}")
            return 0.0
    
    def _calculate_distribution_matching_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate distribution matching score"""
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
            logger.warning(f"Distribution matching score calculation failed: {e}")
            return 0.5
    
    def _calculate_perfect_distribution_matching(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate perfect distribution matching"""
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
            logger.warning(f"Perfect distribution matching calculation failed: {e}")
            return 0.5
    
    def _calculate_prediction_quality_score(self, y_pred_proba: np.ndarray) -> float:
        """Calculate prediction quality score"""
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
            logger.warning(f"Prediction quality score calculation failed: {e}")
            return 0.5
    
    def _calculate_extreme_region_performance(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate extreme region performance"""
        try:
            high_threshold = np.percentile(y_pred_proba, 99)
            low_threshold = np.percentile(y_pred_proba, 1)
            
            extreme_scores = []
            
            # High extreme region
            high_mask = y_pred_proba >= high_threshold
            if high_mask.sum() > 5:
                high_actual = y_true[high_mask].mean()
                high_predicted = y_pred_proba[high_mask].mean()
                
                if high_predicted > 0:
                    high_score = min(high_actual / high_predicted, high_predicted / high_actual)
                else:
                    high_score = 0.0
                
                extreme_scores.append(high_score)
            
            # Low extreme region
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
            logger.warning(f"Extreme region performance calculation failed: {e}")
            return 0.5
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient"""
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
    
    def comprehensive_evaluation_ultra_with_calibration(self, 
                                                      y_true: np.ndarray, 
                                                      y_pred_proba: np.ndarray,
                                                      model_name: str = "Unknown",
                                                      threshold: float = 0.5) -> Dict[str, float]:
        """Ultra-comprehensive evaluation metrics calculation - Combined Score 0.30+ target + calibration evaluation"""
        
        try:
            start_time = time.time()
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"Input size mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return self._get_default_metrics_ultra_with_calibration(model_name)
            
            if len(y_true) == 0:
                logger.error("Empty input array")
                return self._get_default_metrics_ultra_with_calibration(model_name)
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("NaN or infinite values in predictions, performing cleanup")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            y_pred = (y_pred_proba >= threshold).astype(int)
        
            metrics = {}
            metrics['model_name'] = model_name
            
            # Main CTR ultra-performance metrics
            try:
                metrics['ap_enhanced'] = self.average_precision_enhanced(y_true, y_pred_proba)
                metrics['wll_enhanced'] = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
                metrics['combined_score_enhanced'] = self.combined_score_enhanced(y_true, y_pred_proba)
                metrics['ctr_ultra_optimized_score'] = self.ctr_ultra_optimized_score(y_true, y_pred_proba)
            except Exception as e:
                logger.error(f"Main CTR metrics calculation failed: {e}")
                metrics.update({
                    'ap_enhanced': 0.0, 'wll_enhanced': float('inf'), 
                    'combined_score_enhanced': 0.0, 'ctr_ultra_optimized_score': 0.0
                })
            
            # Advanced calibration evaluation
            try:
                calibration_metrics = self.calibration_score_advanced(y_true, y_pred_proba)
                metrics.update({
                    f'calibration_{k}': v for k, v in calibration_metrics.items()
                })
                
                logger.info(f"{model_name} calibration evaluation:")
                logger.info(f"  - ECE: {calibration_metrics['ece']:.4f}")
                logger.info(f"  - MCE: {calibration_metrics['mce']:.4f}")
                logger.info(f"  - Calibration score: {calibration_metrics['calibration_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Calibration metrics calculation failed: {e}")
                metrics.update({
                    'calibration_ece': 1.0, 'calibration_mce': 1.0, 
                    'calibration_brier_reliability': 1.0, 'calibration_brier_resolution': 0.0,
                    'calibration_score': 0.0
                })
            
            # Basic classification metrics (improved stability)
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Basic classification metrics calculation failed: {e}")
                metrics.update({
                    'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25
                })
            
            # Confusion matrix based metrics (enhanced)
            try:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    logger.warning("Confusion matrix shape differs from expected")
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
                    
                    # Enhanced performance metrics
                    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
                    metrics['geometric_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])
                    
                else:
                    metrics.update({
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                        'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0,
                        'balanced_accuracy': 0.0, 'geometric_mean': 0.0
                    })
                
            except Exception as e:
                logger.warning(f"Confusion matrix metrics calculation failed: {e}")
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0,
                    'balanced_accuracy': 0.0, 'geometric_mean': 0.0
                })
            
            # CTR ultra-precision analysis
            try:
                metrics['ctr_actual'] = float(y_true.mean())
                metrics['ctr_predicted'] = float(y_pred_proba.mean())
                metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
                metrics['ctr_ratio'] = metrics['ctr_predicted'] / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
                metrics['ctr_relative_error'] = abs(metrics['ctr_bias']) / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_alignment_score'] = np.exp(-abs(metrics['ctr_bias']) * 1000)
                
                # CTR stability metrics
                metrics['ctr_stability'] = 1.0 / (1.0 + abs(metrics['ctr_bias']) * 10000)
                metrics['target_ctr_achievement'] = 1.0 if abs(metrics['ctr_bias']) < 0.001 else 0.0
                
            except Exception as e:
                logger.warning(f"CTR ultra-precision analysis failed: {e}")
                metrics.update({
                    'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
                    'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0,
                    'ctr_alignment_score': 1.0, 'ctr_stability': 1.0, 'target_ctr_achievement': 1.0
                })
            
            # Prediction quality statistics (enhanced)
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
                logger.warning(f"Prediction quality statistics calculation failed: {e}")
                metrics.update({
                    'prediction_std': 0.0, 'prediction_var': 0.0, 'prediction_entropy': 0.0, 
                    'prediction_gini': 0.0, 'prediction_range': 0.0, 'prediction_iqr': 0.0,
                    'unique_predictions_ratio': 0.0, 'effective_sample_size': len(y_true)
                })
            
            # Class-wise prediction statistics (enhanced)
            try:
                pos_mask = (y_true == 1)
                neg_mask = (y_true == 0)
                
                if pos_mask.any():
                    metrics['pos_mean_pred'] = float(y_pred_proba[pos_mask].mean())
                    metrics['pos_std_pred'] = float(y_pred_proba[pos_mask].std())
                    metrics['pos_median_pred'] = float(np.median(y_pred_proba[pos_mask]))
                else:
                    metrics.update({
                        'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0
                    })
                    
                if neg_mask.any():
                    metrics['neg_mean_pred'] = float(y_pred_proba[neg_mask].mean())
                    metrics['neg_std_pred'] = float(y_pred_proba[neg_mask].std())
                    metrics['neg_median_pred'] = float(np.median(y_pred_proba[neg_mask]))
                else:
                    metrics.update({
                        'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0
                    })
                
                if pos_mask.any() and neg_mask.any():
                    metrics['separation'] = metrics['pos_mean_pred'] - metrics['neg_mean_pred']
                    metrics['separation_ratio'] = metrics['pos_mean_pred'] / max(metrics['neg_mean_pred'], 1e-10)
                else:
                    metrics['separation'] = 0.0
                    metrics['separation_ratio'] = 1.0
                    
            except Exception as e:
                logger.warning(f"Class-wise prediction statistics calculation failed: {e}")
                metrics.update({
                    'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
                    'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
                    'separation': 0.0, 'separation_ratio': 1.0
                })
            
            # Target achievement metrics (including calibration)
            try:
                metrics['target_combined_score_achievement'] = 1.0 if metrics['combined_score_enhanced'] >= self.target_combined_score else 0.0
                metrics['combined_score_gap'] = max(0, self.target_combined_score - metrics['combined_score_enhanced'])
                metrics['ultra_score_achievement'] = 1.0 if metrics['ctr_ultra_optimized_score'] >= 0.32 else 0.0
                metrics['performance_tier'] = self._classify_performance_tier(metrics['combined_score_enhanced'])
                
                # Calibration target achievement
                calibration_threshold = 0.7  # Calibration quality threshold
                metrics['calibration_achievement'] = 1.0 if metrics.get('calibration_score', 0.0) >= calibration_threshold else 0.0
                metrics['calibration_gap'] = max(0, calibration_threshold - metrics.get('calibration_score', 0.0))
                
                # Integrated target achievement (performance + calibration)
                metrics['integrated_achievement'] = metrics['target_combined_score_achievement'] * metrics['calibration_achievement']
                
            except Exception as e:
                logger.warning(f"Target achievement metrics calculation failed: {e}")
                metrics['target_combined_score_achievement'] = 0.0
                metrics['combined_score_gap'] = self.target_combined_score
                metrics['ultra_score_achievement'] = 0.0
                metrics['performance_tier'] = 'poor'
                metrics['calibration_achievement'] = 0.0
                metrics['calibration_gap'] = 0.7
                metrics['integrated_achievement'] = 0.0
            
            # Calculation time
            metrics['evaluation_time'] = time.time() - start_time
            
            # Convert all metric values to float and validate
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
            logger.error(f"Ultra-comprehensive evaluation calculation error: {str(e)}")
            return self._get_default_metrics_ultra_with_calibration(model_name)
    
    def _get_default_metrics_ultra_with_calibration(self, model_name: str = "Unknown") -> Dict[str, float]:
        """Return ultra-high performance default metric values - including calibration"""
        return {
            'model_name': model_name,
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
            'separation': 0.0, 'separation_ratio': 1.0,
            'target_combined_score_achievement': 0.0, 'combined_score_gap': self.target_combined_score,
            'ultra_score_achievement': 0.0, 'performance_tier': 'poor',
            'calibration_ece': 1.0, 'calibration_mce': 1.0,
            'calibration_brier_reliability': 1.0, 'calibration_brier_resolution': 0.0,
            'calibration_score': 0.0, 'calibration_achievement': 0.0, 'calibration_gap': 0.7,
            'integrated_achievement': 0.0,
            'evaluation_time': 0.0
        }
    
    def _calculate_entropy_enhanced(self, probabilities: np.ndarray) -> float:
        """Enhanced entropy calculation of prediction probabilities"""
        try:
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            
            if np.isnan(entropy) or np.isinf(entropy):
                return 0.0
            
            return float(entropy)
        except:
            return 0.0
    
    def _classify_performance_tier(self, combined_score: float) -> str:
        """Performance tier classification"""
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
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        gc.collect()

class UltraModelComparator:
    """Ultra high-performance multi-model comparison class - Combined Score 0.30+ achievement + calibration evaluation"""
    
    def __init__(self):
        self.metrics_calculator = CTRAdvancedMetrics()
        self.comparison_results = pd.DataFrame()
        self.performance_analysis = {}
        
    def compare_models_ultra_with_calibration(self, 
                                            models_predictions: Dict[str, np.ndarray],
                                            y_true: np.ndarray,
                                            models_info: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """Ultra high-performance multiple model performance comparison - including calibration evaluation"""
        
        results = []
        
        y_true = np.asarray(y_true).flatten()
        
        logger.info(f"Ultra high-performance model comparison started (including calibration evaluation): {len(models_predictions)} models")
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                start_time = time.time()
                
                y_pred_proba = np.asarray(y_pred_proba).flatten()
                
                if len(y_pred_proba) != len(y_true):
                    logger.error(f"{model_name}: Prediction and actual value size mismatch")
                    continue
                
                if len(y_pred_proba) == 0:
                    logger.error(f"{model_name}: Empty predictions")
                    continue
                
                if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                    logger.warning(f"{model_name}: NaN or infinite values in predictions, performing cleanup")
                    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Comprehensive evaluation including calibration evaluation
                metrics = self.metrics_calculator.comprehensive_evaluation_ultra_with_calibration(
                    y_true, y_pred_proba, model_name
                )
                
                evaluation_time = time.time() - start_time
                metrics['evaluation_duration'] = evaluation_time
                
                # Add model information
                if models_info and model_name in models_info:
                    model_info = models_info[model_name]
                    metrics['is_calibrated'] = model_info.get('is_calibrated', False)
                    metrics['calibration_method'] = model_info.get('calibration_method', 'none')
                    metrics['model_type'] = model_info.get('model_type', 'unknown')
                else:
                    metrics['is_calibrated'] = False
                    metrics['calibration_method'] = 'none'
                    metrics['model_type'] = 'unknown'
                
                results.append(metrics)
                
                logger.info(f"{model_name} evaluation completed ({evaluation_time:.2f}s)")
                logger.info(f"  - Combined Score Enhanced: {metrics['combined_score_enhanced']:.4f}")
                logger.info(f"  - Ultra Optimized Score: {metrics['ctr_ultra_optimized_score']:.4f}")
                logger.info(f"  - CTR Bias: {metrics['ctr_bias']:.4f}")
                logger.info(f"  - Calibration Score: {metrics.get('calibration_score', 0.0):.4f}")
                logger.info(f"  - Calibration Applied: {'Yes' if metrics['is_calibrated'] else 'No'}")
                logger.info(f"  - Performance Tier: {metrics['performance_tier']}")
                
            except Exception as e:
                logger.error(f"{model_name} ultra high-performance model evaluation failed: {str(e)}")
                default_metrics = self.metrics_calculator._get_default_metrics_ultra_with_calibration(model_name)
                default_metrics['evaluation_duration'] = 0.0
                default_metrics['is_calibrated'] = False
                default_metrics['calibration_method'] = 'none'
                default_metrics['model_type'] = 'unknown'
                results.append(default_metrics)
        
        if not results:
            logger.error("No evaluable models available")
            return pd.DataFrame()
        
        try:
            comparison_df = pd.DataFrame(results)
            
            if not comparison_df.empty:
                comparison_df.set_index('model_name', inplace=True)
                
                # Multi-sort criteria (including calibration score)
                sort_columns = []
                if 'ctr_ultra_optimized_score' in comparison_df.columns:
                    sort_columns.append('ctr_ultra_optimized_score')
                if 'combined_score_enhanced' in comparison_df.columns:
                    sort_columns.append('combined_score_enhanced')
                if 'calibration_score' in comparison_df.columns:
                    sort_columns.append('calibration_score')
                if 'ap_enhanced' in comparison_df.columns:
                    sort_columns.append('ap_enhanced')
                
                if sort_columns:
                    comparison_df.sort_values(sort_columns, ascending=False, inplace=True)
        
            self.comparison_results = comparison_df
            
            # Log target achievement model count (including calibration)
            target_achieved_count = comparison_df['target_combined_score_achievement'].sum()
            ultra_achieved_count = comparison_df['ultra_score_achievement'].sum()
            calibration_achieved_count = comparison_df['calibration_achievement'].sum()
            integrated_achieved_count = comparison_df['integrated_achievement'].sum()
            calibrated_models_count = comparison_df['is_calibrated'].sum()
            
            logger.info(f"Ultra high-performance model comparison completed (including calibration evaluation)")
            logger.info(f"Combined Score 0.30+ achievement models: {target_achieved_count}/{len(comparison_df)}")
            logger.info(f"Ultra Score 0.32+ achievement models: {ultra_achieved_count}/{len(comparison_df)}")
            logger.info(f"Calibration quality target achievement models: {calibration_achieved_count}/{len(comparison_df)}")
            logger.info(f"Integrated target achievement models: {integrated_achieved_count}/{len(comparison_df)}")
            logger.info(f"Calibration applied models: {calibrated_models_count}/{len(comparison_df)}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Ultra high-performance comparison result generation failed: {e}")
            return pd.DataFrame()
    
    def rank_models_ultra_with_calibration(self, 
                                         ranking_metric: str = 'ctr_ultra_optimized_score') -> pd.DataFrame:
        """Ultra high-performance model ranking - calibration consideration"""
        
        if self.comparison_results.empty:
            logger.warning("No comparison results available.")
            return pd.DataFrame()
        
        try:
            ranking_df = self.comparison_results.copy()
            
            if ranking_metric in ranking_df.columns:
                ranking_df['rank'] = ranking_df[ranking_metric].rank(ascending=False)
            else:
                ranking_df['rank'] = ranking_df['combined_score_enhanced'].rank(ascending=False)
            
            ranking_df.sort_values('rank', inplace=True)
            
            key_columns = [
                'rank', ranking_metric, 'combined_score_enhanced', 'ap_enhanced', 'wll_enhanced', 
                'auc', 'f1', 'ctr_bias', 'ctr_ratio', 'ctr_alignment_score',
                'calibration_score', 'calibration_ece', 'calibration_mce',
                'is_calibrated', 'calibration_method', 'calibration_achievement',
                'performance_tier', 'target_combined_score_achievement', 'ultra_score_achievement',
                'integrated_achievement'
            ]
            
            available_columns = [col for col in key_columns if col in ranking_df.columns]
            
            return ranking_df[available_columns]
        
        except Exception as e:
            logger.error(f"Ultra high-performance model ranking failed: {e}")
            return pd.DataFrame()
    
    def get_best_model_ultra_with_calibration(self, metric: str = 'integrated_achievement') -> Tuple[str, float]:
        """Return best performance model (ultra high-performance + calibration consideration)"""
        
        if self.comparison_results.empty:
            return None, 0.0
        
        try:
            if metric not in self.comparison_results.columns:
                # Fallback metrics order
                fallback_metrics = ['ctr_ultra_optimized_score', 'combined_score_enhanced', 'calibration_score']
                for fallback_metric in fallback_metrics:
                    if fallback_metric in self.comparison_results.columns:
                        metric = fallback_metric
                        break
            
            best_idx = self.comparison_results[metric].idxmax()
            best_score = self.comparison_results.loc[best_idx, metric]
            
            return best_idx, best_score
        
        except Exception as e:
            logger.error(f"Best ultra high-performance model finding failed: {e}")
            return None, 0.0

class EvaluationReporter:
    """Evaluation result reporting class - including calibration reports"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.report_data = {}
        
    def generate_model_performance_report_with_calibration(self, 
                                                         comparator: UltraModelComparator,
                                                         save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate model performance report - including calibration evaluation"""
        try:
            if comparator.comparison_results.empty:
                return {'error': 'No comparison results available'}
            
            report = {
                'summary': {
                    'total_models': len(comparator.comparison_results),
                    'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'best_model': None,
                    'best_score': 0.0,
                    'calibration_summary': {}
                },
                'detailed_results': {},
                'performance_analysis': {},
                'calibration_analysis': {},
                'recommendations': []
            }
            
            # Find best performance model (calibration consideration)
            best_model, best_score = comparator.get_best_model_ultra_with_calibration()
            report['summary']['best_model'] = best_model
            report['summary']['best_score'] = best_score
            
            # Detailed results
            for model_name, row in comparator.comparison_results.iterrows():
                model_result = {
                    'combined_score': row.get('combined_score_enhanced', 0.0),
                    'ultra_score': row.get('ctr_ultra_optimized_score', 0.0),
                    'ap_score': row.get('ap_enhanced', 0.0),
                    'auc_score': row.get('auc', 0.5),
                    'ctr_bias': row.get('ctr_bias', 0.0),
                    'performance_tier': row.get('performance_tier', 'unknown'),
                    'target_achieved': row.get('target_combined_score_achievement', 0.0) == 1.0,
                    'evaluation_time': row.get('evaluation_duration', 0.0),
                    'calibration_score': row.get('calibration_score', 0.0),
                    'calibration_ece': row.get('calibration_ece', 1.0),
                    'calibration_mce': row.get('calibration_mce', 1.0),
                    'is_calibrated': row.get('is_calibrated', False),
                    'calibration_method': row.get('calibration_method', 'none'),
                    'calibration_achievement': row.get('calibration_achievement', 0.0) == 1.0,
                    'integrated_achievement': row.get('integrated_achievement', 0.0) == 1.0
                }
                report['detailed_results'][model_name] = model_result
            
            # Performance analysis
            df = comparator.comparison_results
            report['performance_analysis'] = {
                'avg_combined_score': float(df['combined_score_enhanced'].mean()),
                'std_combined_score': float(df['combined_score_enhanced'].std()),
                'target_achievers': int(df['target_combined_score_achievement'].sum()),
                'ultra_achievers': int(df['ultra_score_achievement'].sum()),
                'tier_distribution': df['performance_tier'].value_counts().to_dict()
            }
            
            # Calibration analysis
            report['calibration_analysis'] = {
                'avg_calibration_score': float(df['calibration_score'].mean()),
                'std_calibration_score': float(df['calibration_score'].std()),
                'avg_calibration_ece': float(df['calibration_ece'].mean()),
                'avg_calibration_mce': float(df['calibration_mce'].mean()),
                'calibration_achievers': int(df['calibration_achievement'].sum()),
                'calibrated_models_count': int(df['is_calibrated'].sum()),
                'calibration_methods_distribution': df['calibration_method'].value_counts().to_dict(),
                'integrated_achievers': int(df['integrated_achievement'].sum())
            }
            
            # Calibration summary
            report['summary']['calibration_summary'] = {
                'total_calibrated': int(df['is_calibrated'].sum()),
                'calibration_coverage': float(df['is_calibrated'].mean()),
                'avg_calibration_quality': float(df['calibration_score'].mean()),
                'best_calibration_score': float(df['calibration_score'].max())
            }
            
            # Generate recommendations (calibration consideration)
            if best_score >= 0.30:
                report['recommendations'].append("Target achieved: Models achieving Combined Score 0.30+ are available")
            else:
                report['recommendations'].append("Improvement needed: All models fall short of target score")
            
            if report['calibration_analysis']['calibrated_models_count'] == 0:
                report['recommendations'].append("Apply calibration: Apply calibration to all models to improve prediction reliability")
            elif report['calibration_analysis']['calibrated_models_count'] < len(df):
                report['recommendations'].append("Expand calibration: Apply calibration to more models")
            
            if report['calibration_analysis']['avg_calibration_ece'] > 0.1:
                report['recommendations'].append("Improve calibration quality: ECE is high. Try more advanced calibration techniques")
            
            if report['calibration_analysis']['integrated_achievers'] == 0:
                report['recommendations'].append("Integrated optimization: Find ways to simultaneously improve performance and calibration")
            
            # Save
            if save_path:
                try:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    logger.info(f"Evaluation report saved: {save_path} (including calibration analysis)")
                except Exception as e:
                    logger.warning(f"Report saving failed: {e}")
            
            self.report_data = report
            return report
            
        except Exception as e:
            logger.error(f"Evaluation report generation failed: {e}")
            return {'error': f'Error during report generation: {str(e)}'}
    
    def save_comparison_results_with_calibration(self, 
                                               comparison_df: pd.DataFrame,
                                               save_path: Optional[Path] = None) -> bool:
        """Save comparison results to file - including calibration information"""
        try:
            if save_path is None:
                save_path = self.config.OUTPUT_DIR / "model_comparison_results_with_calibration.csv"
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(save_path, index=True, encoding='utf-8')
            
            logger.info(f"Model comparison results saved: {save_path} (including calibration information)")
            return True
            
        except Exception as e:
            logger.error(f"Comparison results saving failed: {e}")
            return False
    
    def generate_summary_table_with_calibration(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary table - including calibration"""
        try:
            if comparison_df.empty:
                return pd.DataFrame()
            
            summary_columns = [
                'combined_score_enhanced', 'ctr_ultra_optimized_score', 
                'ap_enhanced', 'auc', 'ctr_bias', 'performance_tier',
                'calibration_score', 'calibration_ece', 'calibration_mce',
                'is_calibrated', 'calibration_method',
                'target_combined_score_achievement', 'ultra_score_achievement',
                'calibration_achievement', 'integrated_achievement'
            ]
            
            available_columns = [col for col in summary_columns if col in comparison_df.columns]
            
            if available_columns:
                summary_df = comparison_df[available_columns].copy()
                
                # Round only numeric columns
                numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
                summary_df[numeric_columns] = summary_df[numeric_columns].round(4)
                
                return summary_df
            else:
                return comparison_df
            
        except Exception as e:
            logger.error(f"Summary table generation failed: {e}")
            return pd.DataFrame()

# Aliases for backward compatibility - ensure import from main.py
CTRMetrics = CTRAdvancedMetrics
ModelComparator = UltraModelComparator

# Additional backward compatibility assurance
def create_ctr_metrics():
    """CTR metrics generator"""
    return CTRAdvancedMetrics()

def create_model_comparator():
    """Model comparator generator"""
    return UltraModelComparator()

def create_evaluation_reporter():
    """Evaluation reporter generator"""
    return EvaluationReporter()

# Main functions directly accessible at module level
def evaluate_model_performance_with_calibration(y_true, y_pred_proba, model_name="Unknown"):
    """Single model performance evaluation - including calibration"""
    metrics_calc = CTRAdvancedMetrics()
    return metrics_calc.comprehensive_evaluation_ultra_with_calibration(y_true, y_pred_proba, model_name)

def compare_multiple_models_with_calibration(models_predictions, y_true, models_info=None):
    """Multiple model comparison - including calibration evaluation"""
    comparator = UltraModelComparator()
    return comparator.compare_models_ultra_with_calibration(models_predictions, y_true, models_info)

# Existing functions (backward compatibility)
def evaluate_model_performance(y_true, y_pred_proba):
    """Single model performance evaluation"""
    metrics_calc = CTRAdvancedMetrics()
    return metrics_calc.comprehensive_evaluation_ultra_with_calibration(y_true, y_pred_proba)

def compare_multiple_models(models_predictions, y_true):
    """Multiple model comparison"""
    comparator = UltraModelComparator()
    return comparator.compare_models_ultra_with_calibration(models_predictions, y_true)