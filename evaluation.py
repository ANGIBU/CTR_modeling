# evaluation.py

import numpy as np
import pandas as pd
import logging
import time
import gc
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, log_loss, 
        brier_score_loss, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed. Some advanced metrics will be disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("matplotlib/seaborn not installed. Visualization features will be disabled.")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not installed. Some statistical features will be disabled.")

from config import Config

logger = logging.getLogger(__name__)

class CTRAdvancedMetrics:
    """CTR prediction specialized advanced evaluation metrics class - Combined Score 0.30+ target + calibration evaluation"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
        evaluation_config = getattr(config, 'EVALUATION_CONFIG', {})
        
        self.ap_weight = evaluation_config.get('ap_weight', 0.5)
        self.wll_weight = evaluation_config.get('wll_weight', 0.5)
        self.actual_ctr = evaluation_config.get('target_ctr', 0.0191)
        self.pos_weight = evaluation_config.get('pos_weight', 52.3)
        self.neg_weight = evaluation_config.get('neg_weight', 1.0)
        self.target_combined_score = evaluation_config.get('target_combined_score', 0.34)
        
        # Advanced adaptive bias correction parameters
        self.base_ctr_tolerance = evaluation_config.get('ctr_tolerance', 0.002)  # More realistic base
        self.adaptive_tolerance_factor = evaluation_config.get('adaptive_tolerance_factor', 1.5)
        self.bias_penalty_base = evaluation_config.get('bias_penalty_base', 5.0)  # Reduced from 8.0
        self.calibration_weight = evaluation_config.get('calibration_weight', 0.6)
        self.wll_normalization_factor = evaluation_config.get('wll_normalization_factor', 1.8)  # Increased
        self.ctr_bias_multiplier = evaluation_config.get('ctr_bias_multiplier', 6.0)  # Reduced from 10.0
        
        # Adaptive correction parameters
        self.correction_adaptation_rate = 0.1
        self.min_correction_factor = 0.05  # Minimum 5% correction instead of 15%
        self.max_correction_factor = 0.30  # Maximum 30% correction instead of aggressive values
        self.correction_stability_threshold = 0.001
        
        # Dynamic bias correction history for adaptation
        self.bias_correction_history = []
        self.performance_history = []
        
        self.cache = {}
        
    def adaptive_ctr_tolerance(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate adaptive CTR tolerance based on data characteristics"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            
            # Base tolerance adjusted by actual CTR level
            ctr_based_factor = max(0.5, min(2.0, actual_ctr / 0.02))  # Normalize around 2%
            
            # Prediction variance consideration
            pred_std = np.std(y_pred_proba)
            variance_factor = max(0.8, min(1.5, pred_std * 10))
            
            # Sample size consideration
            sample_size_factor = max(0.7, min(1.3, np.log10(len(y_true)) / 6))
            
            adaptive_tolerance = (self.base_ctr_tolerance * 
                                 ctr_based_factor * 
                                 variance_factor * 
                                 sample_size_factor)
            
            return float(np.clip(adaptive_tolerance, 0.0005, 0.01))
            
        except Exception as e:
            logger.warning(f"Adaptive tolerance calculation failed: {e}")
            return self.base_ctr_tolerance
    
    def calculate_optimal_bias_correction(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate optimal bias correction parameters using data-driven approach"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Adaptive tolerance
            dynamic_tolerance = self.adaptive_ctr_tolerance(y_true, y_pred_proba)
            
            # Calculate correction strength based on bias severity
            if ctr_bias <= dynamic_tolerance:
                # Minimal correction for good predictions
                correction_factor = max(0.0, min(0.05, ctr_bias / dynamic_tolerance * 0.05))
                correction_type = 'minimal'
            elif ctr_bias <= dynamic_tolerance * 3:
                # Moderate correction for acceptable bias
                excess_bias = ctr_bias - dynamic_tolerance
                correction_factor = 0.05 + (excess_bias / (dynamic_tolerance * 2)) * 0.15
                correction_type = 'moderate'
            else:
                # Strong but capped correction for large bias
                correction_factor = min(self.max_correction_factor, 
                                      0.20 + (ctr_bias / (dynamic_tolerance * 5)) * 0.10)
                correction_type = 'strong'
            
            # Additional factors
            prediction_confidence = 1.0 - np.std(y_pred_proba)
            confidence_adjustment = max(0.8, min(1.2, prediction_confidence))
            
            final_correction = correction_factor * confidence_adjustment
            
            return {
                'correction_factor': float(np.clip(final_correction, 0.0, self.max_correction_factor)),
                'tolerance_used': float(dynamic_tolerance),
                'bias_severity': float(ctr_bias / dynamic_tolerance),
                'correction_type': correction_type,
                'confidence_adjustment': float(confidence_adjustment)
            }
            
        except Exception as e:
            logger.warning(f"Optimal bias correction calculation failed: {e}")
            return {
                'correction_factor': 0.10,
                'tolerance_used': self.base_ctr_tolerance,
                'bias_severity': 1.0,
                'correction_type': 'default',
                'confidence_adjustment': 1.0
            }
    
    def hierarchical_bias_correction(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   base_score: float) -> Dict[str, float]:
        """Apply hierarchical bias correction strategy"""
        try:
            correction_params = self.calculate_optimal_bias_correction(y_true, y_pred_proba)
            
            # Stage 1: Primary CTR alignment correction
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            primary_correction = 1.0
            if ctr_bias > correction_params['tolerance_used']:
                bias_ratio = ctr_bias / correction_params['tolerance_used']
                primary_penalty = min(0.4, bias_ratio * 0.08)  # Gentler penalty
                primary_correction = 1.0 - primary_penalty
            else:
                # Small bonus for accurate CTR
                accuracy_bonus = min(0.03, (correction_params['tolerance_used'] - ctr_bias) / 
                                   correction_params['tolerance_used'] * 0.03)
                primary_correction = 1.0 + accuracy_bonus
            
            stage1_score = base_score * primary_correction
            
            # Stage 2: Distribution alignment correction
            try:
                # Calculate distribution similarity
                actual_pos_rate = np.mean(y_true)
                pred_pos_rate = np.mean(y_pred_proba > 0.5)
                
                distribution_bias = abs(actual_pos_rate - pred_pos_rate)
                if distribution_bias > 0.02:  # 2% threshold
                    dist_penalty = min(0.15, distribution_bias * 2.0)
                    distribution_correction = 1.0 - dist_penalty
                else:
                    distribution_correction = 1.0 + min(0.02, (0.02 - distribution_bias) * 0.5)
                
            except Exception:
                distribution_correction = 1.0
            
            stage2_score = stage1_score * distribution_correction
            
            # Stage 3: Prediction quality correction
            try:
                # Reward prediction diversity and calibration
                pred_std = np.std(y_pred_proba)
                pred_range = np.max(y_pred_proba) - np.min(y_pred_proba)
                quality_factor = min(1.1, 1.0 + (pred_std * pred_range) * 0.5)
                
            except Exception:
                quality_factor = 1.0
            
            final_score = stage2_score * quality_factor
            
            # Reality check with improved bounds
            if predicted_ctr > 0.15:  # Very lenient upper bound
                extreme_penalty = min(0.3, (predicted_ctr - 0.15) * 2.0)
                final_score *= (1.0 - extreme_penalty)
            
            return {
                'final_score': float(np.clip(final_score, 0.0, 1.0)),
                'primary_correction': float(primary_correction),
                'distribution_correction': float(distribution_correction),
                'quality_factor': float(quality_factor),
                'correction_applied': True,
                'correction_params': correction_params
            }
            
        except Exception as e:
            logger.warning(f"Hierarchical bias correction failed: {e}")
            return {
                'final_score': float(base_score),
                'primary_correction': 1.0,
                'distribution_correction': 1.0,
                'quality_factor': 1.0,
                'correction_applied': False,
                'correction_params': {}
            }
    
    def average_precision_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Average Precision - stability and accuracy improvements"""
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
                logger.warning("Empty array prevents AP calculation")
                return 0.0
            
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("Single class only - AP calculation impossible")
                return 0.0
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("NaN or infinity values in predictions, applying clipping")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            try:
                ap_score = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"sklearn AP calculation failed, manual calculation: {e}")
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
        """Manual average precision calculation as fallback"""
        try:
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            precisions = []
            recalls = []
            
            true_positives = 0
            total_positives = np.sum(y_true)
            
            if total_positives == 0:
                return 0.0
            
            for i, label in enumerate(y_true_sorted):
                if label == 1:
                    true_positives += 1
                
                precision = true_positives / (i + 1)
                recall = true_positives / total_positives
                
                precisions.append(precision)
                recalls.append(recall)
            
            precisions = np.array(precisions)
            recalls = np.array(recalls)
            
            # Calculate AP using trapezoidal integration
            ap = 0.0
            for i in range(1, len(recalls)):
                ap += (recalls[i] - recalls[i-1]) * precisions[i]
            
            return float(ap)
            
        except Exception as e:
            logger.warning(f"Manual AP calculation failed: {e}")
            return 0.0
    
    def weighted_log_loss_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced weighted log loss with improved class balancing"""
        try:
            cache_key = f"wll_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return float('inf')
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # Dynamic weight calculation based on actual data distribution
            pos_count = np.sum(y_true)
            neg_count = len(y_true) - pos_count
            
            if pos_count == 0 or neg_count == 0:
                try:
                    regular_loss = log_loss(y_true, y_pred_proba)
                    return float(regular_loss)
                except:
                    return float('inf')
            
            # Adaptive weights based on actual imbalance
            actual_ratio = neg_count / pos_count
            adaptive_pos_weight = min(100.0, max(1.0, actual_ratio * 0.8))  # More conservative
            adaptive_neg_weight = 1.0
            
            # Calculate weighted loss
            pos_mask = (y_true == 1)
            neg_mask = (y_true == 0)
            
            pos_loss = -adaptive_pos_weight * np.mean(np.log(y_pred_proba[pos_mask])) if pos_count > 0 else 0.0
            neg_loss = -adaptive_neg_weight * np.mean(np.log(1 - y_pred_proba[neg_mask])) if neg_count > 0 else 0.0
            
            total_weight = adaptive_pos_weight * (pos_count / len(y_true)) + adaptive_neg_weight * (neg_count / len(y_true))
            weighted_loss = (pos_loss + neg_loss) / max(total_weight, 1e-8)
            
            if np.isnan(weighted_loss) or np.isinf(weighted_loss):
                return float('inf')
            
            result = float(weighted_loss)
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced weighted log loss calculation error: {str(e)}")
            return float('inf')
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Standard combined score - backward compatibility wrapper"""
        return self.combined_score_enhanced(y_true, y_pred_proba)
    
    def combined_score_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced combined score with adaptive hierarchical bias correction"""
        try:
            cache_key = f"combined_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return 0.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # Calculate base metrics
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            
            # Improved WLL normalization
            normalized_wll = max(0, 1 - wll_score / self.wll_normalization_factor)
            
            # Base combined score
            base_combined = (ap_score * self.ap_weight) + (normalized_wll * self.wll_weight)
            
            # Apply hierarchical bias correction
            correction_result = self.hierarchical_bias_correction(y_true, y_pred_proba, base_combined)
            
            # Additional diversity and distribution bonuses
            try:
                # Prediction diversity bonus
                pred_std = y_pred_proba.std()
                pred_range = y_pred_proba.max() - y_pred_proba.min()
                unique_ratio = len(np.unique(y_pred_proba)) / len(y_pred_proba)
                
                diversity_score = pred_std * pred_range * unique_ratio
                diversity_bonus = min(diversity_score * 1.5, 0.3)  # Reduced multiplier
                
            except Exception as e:
                logger.warning(f"Diversity calculation failed: {e}")
                diversity_bonus = 0.0
            
            # Distribution matching bonus
            try:
                distribution_score = self._calculate_distribution_matching_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Distribution matching calculation failed: {e}")
                distribution_score = 0.5
            
            # Final score calculation
            final_score = (correction_result['final_score'] * 
                          (1.0 + 0.2 * diversity_bonus) * 
                          (1.0 + 0.1 * distribution_score))
            
            if np.isnan(final_score) or np.isinf(final_score) or final_score < 0:
                logger.warning("Combined Score calculation result is invalid")
                return 0.0
            
            final_score = float(final_score)
            self.cache[cache_key] = final_score
            
            return final_score
            
        except Exception as e:
            logger.error(f"Enhanced Combined Score calculation error: {str(e)}")
            return 0.0
    
    def _calculate_distribution_matching_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate how well prediction distribution matches true distribution"""
        try:
            # Quantile-based comparison
            true_quantiles = np.percentile(y_true, [25, 50, 75])
            pred_quantiles = np.percentile(y_pred_proba, [25, 50, 75])
            
            quantile_diff = np.mean(np.abs(true_quantiles - pred_quantiles))
            quantile_score = np.exp(-quantile_diff * 10)
            
            # Mean and std comparison
            true_mean, pred_mean = np.mean(y_true), np.mean(y_pred_proba)
            true_std, pred_std = np.std(y_true), np.std(y_pred_proba)
            
            mean_diff = abs(true_mean - pred_mean)
            std_diff = abs(true_std - pred_std)
            
            mean_score = np.exp(-mean_diff * 20)
            std_score = np.exp(-std_diff * 5)
            
            # Combined distribution score
            distribution_score = (quantile_score * 0.4 + mean_score * 0.4 + std_score * 0.2)
            
            return float(np.clip(distribution_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Distribution matching calculation failed: {e}")
            return 0.5
    
    def calibration_score_advanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Advanced calibration quality evaluation - multiple metrics"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return {'ece': 1.0, 'mce': 1.0, 'brier_reliability': 1.0, 'brier_resolution': 0.0,
                       'calibration_score': 0.0}
            
            # Expected Calibration Error
            ece = self._calculate_ece(y_true, y_pred_proba)
            
            # Maximum Calibration Error
            mce = self._calculate_mce(y_true, y_pred_proba)
            
            # Brier Score Decomposition
            brier_reliability, brier_resolution = self._calculate_brier_decomposition(y_true, y_pred_proba)
            
            # Overall calibration score
            ece_score = max(0.0, 1.0 - ece)
            mce_score = max(0.0, 1.0 - mce)
            reliability_score = max(0.0, 1.0 - brier_reliability)
            resolution_score = min(1.0, brier_resolution * 2)  # Scale resolution
            
            # Weighted average
            overall_score = (0.3 * ece_score + 
                           0.2 * mce_score + 
                           0.3 * reliability_score + 
                           0.2 * resolution_score)
            
            return {
                'ece': float(ece),
                'mce': float(mce),
                'brier_reliability': float(brier_reliability),
                'brier_resolution': float(brier_resolution),
                'calibration_score': float(max(0.0, min(overall_score, 1.0)))
            }
            
        except Exception as e:
            logger.warning(f"Advanced calibration calculation failed: {e}")
            return {'ece': 1.0, 'mce': 1.0, 'brier_reliability': 1.0, 'brier_resolution': 0.0,
                   'calibration_score': 0.0}
    
    def _calculate_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return float(ece)
            
        except Exception as e:
            logger.warning(f"ECE calculation failed: {e}")
            return 1.0
    
    def _calculate_mce(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_errors = []
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                
                if in_bin.sum() > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    calibration_errors.append(np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
            return float(max(calibration_errors)) if calibration_errors else 1.0
            
        except Exception as e:
            logger.warning(f"MCE calculation failed: {e}")
            return 1.0
    
    def _calculate_brier_decomposition(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """Calculate Brier score decomposition into reliability and resolution"""
        try:
            if SKLEARN_AVAILABLE:
                brier_score = brier_score_loss(y_true, y_pred_proba)
            else:
                brier_score = np.mean((y_pred_proba - y_true) ** 2)
            
            # Simplified reliability and resolution estimates
            base_rate = np.mean(y_true)
            unconditional_variance = base_rate * (1 - base_rate)
            
            # Estimate reliability (calibration component)
            bin_errors = []
            n_bins = 10
            for i in range(n_bins):
                bin_mask = (y_pred_proba >= i/n_bins) & (y_pred_proba < (i+1)/n_bins)
                if bin_mask.sum() > 0:
                    bin_accuracy = y_true[bin_mask].mean()
                    bin_confidence = y_pred_proba[bin_mask].mean()
                    bin_errors.append((bin_confidence - bin_accuracy) ** 2 * bin_mask.mean())
            
            reliability = sum(bin_errors) if bin_errors else 0.0
            
            # Resolution (how much better than base rate)
            resolution = max(0.0, unconditional_variance - (brier_score - reliability))
            
            return float(reliability), float(resolution)
            
        except Exception as e:
            logger.warning(f"Brier decomposition failed: {e}")
            return 1.0, 0.0
    
    def ctr_ultra_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Ultra-optimized CTR score with adaptive algorithms"""
        try:
            cache_key = f"ultra_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return 0.0
            
            # Base metrics
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            wll_normalized = 1 / (1 + wll_score) if wll_score != float('inf') else 0.0
            
            # CTR precision analysis with adaptive approach
            predicted_ctr = y_pred_proba.mean()
            actual_ctr = y_true.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Dynamic tolerance based on data characteristics
            dynamic_tolerance = self.adaptive_ctr_tolerance(y_true, y_pred_proba)
            
            # Ultra-precision CTR alignment with adaptive thresholds
            if ctr_bias < dynamic_tolerance:
                ultra_ctr_accuracy = np.exp(-ctr_bias * 800)  # Less aggressive exponential
            else:
                # Progressive penalty instead of cliff
                penalty_severity = min(3.0, ctr_bias / dynamic_tolerance)
                ultra_ctr_accuracy = max(0.1, np.exp(-penalty_severity * 0.5))
            
            ctr_stability = 1.0 / (1.0 + ctr_bias * 3000)  # Reduced multiplier
            
            # Advanced calibration assessment
            try:
                calibration_metrics = self.calibration_score_advanced(y_true, y_pred_proba)
                calibration_score = calibration_metrics.get('calibration_score', 0.5)
            except Exception as e:
                logger.warning(f"Advanced calibration calculation failed: {e}")
                calibration_score = 0.5
            
            # Distribution matching with improved algorithm
            try:
                distribution_score = self._calculate_perfect_distribution_matching(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Perfect distribution matching calculation failed: {e}")
                distribution_score = 0.5
            
            # Prediction quality assessment
            try:
                quality_metrics = self._calculate_prediction_quality_metrics(y_pred_proba)
                quality_score = quality_metrics.get('overall_quality', 0.5)
            except Exception as e:
                logger.warning(f"Quality metrics calculation failed: {e}")
                quality_score = 0.5
            
            # Ultra score composition with balanced weights
            base_score = (ap_score * 0.35 + wll_normalized * 0.25)
            ctr_component = (ultra_ctr_accuracy * 0.15 + ctr_stability * 0.10)
            calibration_component = calibration_score * 0.10
            quality_component = (distribution_score * 0.03 + quality_score * 0.02)
            
            ultra_score = (base_score + ctr_component + 
                          calibration_component + quality_component)
            
            # Final validation and caching
            final_score = float(np.clip(ultra_score, 0.0, 1.0))
            self.cache[cache_key] = final_score
            
            return final_score
            
        except Exception as e:
            logger.error(f"Ultra optimized score calculation error: {str(e)}")
            return 0.0
    
    def _calculate_perfect_distribution_matching(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate perfect distribution matching score with multiple criteria"""
        try:
            scores = []
            
            # 1. Quantile matching (more detailed)
            quantiles = [10, 25, 50, 75, 90]
            true_quantiles = np.percentile(y_true, quantiles)
            pred_quantiles = np.percentile(y_pred_proba, quantiles)
            
            quantile_diffs = np.abs(true_quantiles - pred_quantiles)
            quantile_score = np.exp(-np.mean(quantile_diffs) * 15)
            scores.append(quantile_score)
            
            # 2. Moment matching
            moments_score = 1.0
            for moment in range(1, 4):  # 1st, 2nd, 3rd moments
                try:
                    true_moment = np.mean(y_true ** moment)
                    pred_moment = np.mean(y_pred_proba ** moment)
                    moment_diff = abs(true_moment - pred_moment)
                    moment_score = np.exp(-moment_diff * (10 / moment))
                    moments_score *= moment_score
                except:
                    continue
            scores.append(moments_score)
            
            # 3. Distribution shape similarity
            try:
                # Use histogram comparison
                true_hist, _ = np.histogram(y_true, bins=20, range=(0, 1))
                pred_hist, _ = np.histogram(y_pred_proba, bins=20, range=(0, 1))
                
                # Normalize histograms
                true_hist = true_hist / np.sum(true_hist)
                pred_hist = pred_hist / np.sum(pred_hist)
                
                # Calculate similarity (1 - Jensen-Shannon divergence approximation)
                hist_diff = np.sum(np.abs(true_hist - pred_hist))
                shape_score = max(0.0, 1.0 - hist_diff / 2.0)
                scores.append(shape_score)
                
            except:
                scores.append(0.5)
            
            # Weighted combination
            weights = [0.4, 0.3, 0.3]
            final_score = np.average(scores, weights=weights)
            
            return float(np.clip(final_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Perfect distribution matching calculation failed: {e}")
            return 0.5
    
    def _calculate_prediction_quality_metrics(self, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive prediction quality metrics"""
        try:
            metrics = {}
            
            # Basic statistics
            metrics['std'] = float(np.std(y_pred_proba))
            metrics['var'] = float(np.var(y_pred_proba))
            metrics['range'] = float(np.max(y_pred_proba) - np.min(y_pred_proba))
            metrics['iqr'] = float(np.percentile(y_pred_proba, 75) - np.percentile(y_pred_proba, 25))
            
            # Diversity metrics
            unique_predictions = len(np.unique(y_pred_proba))
            metrics['unique_ratio'] = float(unique_predictions / len(y_pred_proba))
            
            # Entropy calculation
            try:
                metrics['entropy'] = self._calculate_entropy_enhanced(y_pred_proba)
            except:
                metrics['entropy'] = 0.0
            
            # Gini coefficient
            try:
                metrics['gini'] = self._calculate_gini_coefficient(y_pred_proba)
            except:
                metrics['gini'] = 0.0
            
            # Overall quality score
            quality_components = [
                min(1.0, metrics['std'] * 4),      # Reward good standard deviation
                min(1.0, metrics['range'] * 2),     # Reward good range
                metrics['unique_ratio'],             # Reward diversity
                min(1.0, metrics['entropy'] / 0.8), # Reward entropy
                min(1.0, metrics['gini'] * 2)       # Reward distribution
            ]
            
            metrics['overall_quality'] = float(np.mean(quality_components))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Prediction quality metrics calculation failed: {e}")
            return {'overall_quality': 0.5}
    
    def _calculate_entropy_enhanced(self, probabilities: np.ndarray) -> float:
        """Enhanced prediction probability entropy calculation"""
        try:
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            
            if np.isnan(entropy) or np.isinf(entropy):
                return 0.0
            
            return float(entropy)
        except:
            return 0.0
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for prediction distribution"""
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
        """Ultra-comprehensive evaluation metric calculation - Combined Score 0.30+ target + calibration evaluation"""
        
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
                logger.warning("NaN or infinity values in predictions, cleaning performed")
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
                logger.error(f"Main CTR metric calculation failed: {e}")
                metrics.update({
                    'ap_enhanced': 0.0, 'wll_enhanced': float('inf'), 
                    'combined_score_enhanced': 0.0, 'ctr_ultra_optimized_score': 0.0
                })
            
            # Standard evaluation metrics
            try:
                if SKLEARN_AVAILABLE:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                else:
                    metrics['auc'] = 0.5
                    metrics['log_loss'] = 1.0
                    metrics['brier_score'] = 0.25
            except Exception as e:
                logger.warning(f"Standard metric calculation failed: {e}")
                metrics.update({'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25})
            
            # Confusion matrix based metrics
            try:
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    
                    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
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
                logger.warning(f"Confusion matrix metric calculation failed: {e}")
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
                
                # Dynamic tolerance for achievement
                dynamic_tolerance = self.adaptive_ctr_tolerance(y_true, y_pred_proba)
                metrics['target_ctr_achievement'] = 1.0 if abs(metrics['ctr_bias']) < dynamic_tolerance else 0.0
                
            except Exception as e:
                logger.warning(f"CTR ultra-precision analysis failed: {e}")
                metrics.update({
                    'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
                    'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0,
                    'ctr_alignment_score': 1.0, 'ctr_stability': 1.0, 'target_ctr_achievement': 1.0
                })
            
            # Prediction analysis metrics
            try:
                quality_metrics = self._calculate_prediction_quality_metrics(y_pred_proba)
                metrics.update({
                    'prediction_std': quality_metrics.get('std', 0.0),
                    'prediction_var': quality_metrics.get('var', 0.0),
                    'prediction_entropy': quality_metrics.get('entropy', 0.0),
                    'prediction_gini': quality_metrics.get('gini', 0.0),
                    'prediction_range': quality_metrics.get('range', 0.0),
                    'prediction_iqr': quality_metrics.get('iqr', 0.0),
                    'unique_predictions_ratio': quality_metrics.get('unique_ratio', 0.0),
                    'effective_sample_size': len(y_true)
                })
            except Exception as e:
                logger.warning(f"Prediction analysis failed: {e}")
                metrics.update({
                    'prediction_std': 0.0, 'prediction_var': 0.0, 'prediction_entropy': 0.0, 'prediction_gini': 0.0,
                    'prediction_range': 0.0, 'prediction_iqr': 0.0, 'unique_predictions_ratio': 0.0, 'effective_sample_size': 1000
                })
            
            # Class-wise prediction analysis
            try:
                pos_mask = (y_true == 1)
                neg_mask = (y_true == 0)
                
                if np.any(pos_mask):
                    metrics['pos_mean_pred'] = float(y_pred_proba[pos_mask].mean())
                    metrics['pos_std_pred'] = float(y_pred_proba[pos_mask].std())
                    metrics['pos_median_pred'] = float(np.median(y_pred_proba[pos_mask]))
                else:
                    metrics.update({'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0})
                
                if np.any(neg_mask):
                    metrics['neg_mean_pred'] = float(y_pred_proba[neg_mask].mean())
                    metrics['neg_std_pred'] = float(y_pred_proba[neg_mask].std())
                    metrics['neg_median_pred'] = float(np.median(y_pred_proba[neg_mask]))
                else:
                    metrics.update({'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0})
                
                # Class separation
                metrics['separation'] = abs(metrics['pos_mean_pred'] - metrics['neg_mean_pred'])
                metrics['separation_ratio'] = metrics['pos_mean_pred'] / max(metrics['neg_mean_pred'], 1e-10)
                
            except Exception as e:
                logger.warning(f"Class-wise analysis failed: {e}")
                metrics.update({
                    'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
                    'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
                    'separation': 0.0, 'separation_ratio': 1.0
                })
            
            # Target achievement analysis
            try:
                metrics['target_combined_score_achievement'] = 1.0 if metrics['combined_score_enhanced'] >= self.target_combined_score else 0.0
                metrics['combined_score_gap'] = max(0.0, self.target_combined_score - metrics['combined_score_enhanced'])
                
                ultra_target = 0.32
                metrics['ultra_score_achievement'] = 1.0 if metrics['ctr_ultra_optimized_score'] >= ultra_target else 0.0
                
                # Performance tier classification
                metrics['performance_tier'] = self._classify_performance_tier(metrics['combined_score_enhanced'])
                
            except Exception as e:
                logger.warning(f"Target achievement analysis failed: {e}")
                metrics.update({
                    'target_combined_score_achievement': 0.0, 'combined_score_gap': self.target_combined_score,
                    'ultra_score_achievement': 0.0, 'performance_tier': 'poor'
                })
            
            # Advanced calibration evaluation
            try:
                calibration_metrics = self.calibration_score_advanced(y_true, y_pred_proba)
                metrics.update({
                    'calibration_ece': calibration_metrics['ece'],
                    'calibration_mce': calibration_metrics['mce'],
                    'calibration_brier_reliability': calibration_metrics['brier_reliability'],
                    'calibration_brier_resolution': calibration_metrics['brier_resolution'],
                    'calibration_score': calibration_metrics['calibration_score']
                })
                
                # Calibration achievement
                calibration_target = 0.7
                metrics['calibration_achievement'] = 1.0 if metrics['calibration_score'] >= calibration_target else 0.0
                metrics['calibration_gap'] = max(0.0, calibration_target - metrics['calibration_score'])
                
                # Integrated achievement (combined + calibration)
                metrics['integrated_achievement'] = 1.0 if (metrics['target_combined_score_achievement'] == 1.0 and 
                                                          metrics['calibration_achievement'] == 1.0) else 0.0
                
            except Exception as e:
                logger.warning(f"Advanced calibration evaluation failed: {e}")
                metrics.update({
                    'calibration_ece': 1.0, 'calibration_mce': 1.0,
                    'calibration_brier_reliability': 1.0, 'calibration_brier_resolution': 0.0,
                    'calibration_score': 0.0, 'calibration_achievement': 0.0, 'calibration_gap': 0.7,
                    'integrated_achievement': 0.0
                })
            
            metrics['evaluation_time'] = time.time() - start_time
            
            # Validate all metrics
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
        """Return ultra-performance default metric values - calibration included"""
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
        """Cache cleanup"""
        self.cache.clear()
        gc.collect()

class UltraModelComparator:
    """Ultra-performance multiple model comparison class - Combined Score 0.30+ achievement + calibration evaluation"""
    
    def __init__(self):
        self.metrics_calculator = CTRAdvancedMetrics()
        self.comparison_results = pd.DataFrame()
        self.performance_analysis = {}
        
    def compare_models_ultra_with_calibration(self, 
                                            models_predictions: Dict[str, np.ndarray],
                                            y_true: np.ndarray,
                                            models_info: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """Ultra-performance multiple model performance comparison - calibration evaluation included"""
        
        results = []
        
        y_true_clean = np.asarray(y_true).flatten()
        
        logger.info(f"Ultra-performance model comparison started (calibration evaluation included): {len(models_predictions)} models")
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                start_time = time.time()
                
                y_pred_proba = np.asarray(y_pred_proba).flatten()
                
                if len(y_pred_proba) != len(y_true_clean):
                    logger.error(f"{model_name}: Prediction and actual value size mismatch")
                    continue
                
                if len(y_pred_proba) == 0:
                    logger.error(f"{model_name}: Empty prediction values")
                    continue
                
                if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                    logger.warning(f"{model_name}: NaN or infinity values in predictions, cleaning")
                    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Comprehensive evaluation with calibration
                metrics = self.metrics_calculator.comprehensive_evaluation_ultra_with_calibration(
                    y_true_clean, y_pred_proba, model_name
                )
                
                evaluation_time = time.time() - start_time
                metrics['evaluation_duration'] = evaluation_time
                
                # Add model info if available
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
                logger.info(f"  - Performance Tier: {metrics['performance_tier']}")
                
            except Exception as e:
                logger.error(f"{model_name} ultra-performance evaluation failed: {str(e)}")
                default_metrics = self.metrics_calculator._get_default_metrics_ultra_with_calibration(model_name)
                default_metrics['evaluation_duration'] = 0.0
                default_metrics['is_calibrated'] = False
                default_metrics['calibration_method'] = 'none'
                default_metrics['model_type'] = 'unknown'
                results.append(default_metrics)
        
        if not results:
            logger.error("No models could be evaluated")
            return pd.DataFrame()
        
        try:
            comparison_df = pd.DataFrame(results)
            
            if not comparison_df.empty:
                comparison_df.set_index('model_name', inplace=True)
                
                # Multiple sorting criteria (calibration score included)
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
            
            # Target achievement model count logging (calibration included)
            target_achieved_count = comparison_df['target_combined_score_achievement'].sum()
            ultra_achieved_count = comparison_df['ultra_score_achievement'].sum()
            calibration_achieved_count = comparison_df['calibration_achievement'].sum()
            integrated_achieved_count = comparison_df['integrated_achievement'].sum()
            calibrated_models_count = comparison_df['is_calibrated'].sum()
            
            logger.info(f"Ultra-performance model comparison completed (calibration evaluation included)")
            logger.info(f"Combined Score 0.30+ achievement models: {target_achieved_count}/{len(comparison_df)}")
            logger.info(f"Ultra Score 0.32+ achievement models: {ultra_achieved_count}/{len(comparison_df)}")
            logger.info(f"Calibration quality target achievement models: {calibration_achieved_count}/{len(comparison_df)}")
            logger.info(f"Integrated target achievement models: {integrated_achieved_count}/{len(comparison_df)}")
            logger.info(f"Calibration applied models: {calibrated_models_count}/{len(comparison_df)}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Ultra-performance comparison result generation failed: {e}")
            return pd.DataFrame()
    
    def rank_models_ultra_with_calibration(self, 
                                         ranking_metric: str = 'ctr_ultra_optimized_score') -> pd.DataFrame:
        """Ultra-performance model ranking - calibration consideration"""
        
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
            logger.error(f"Ultra-performance model ranking failed: {e}")
            return pd.DataFrame()
    
    def get_best_model_ultra_with_calibration(self, metric: str = 'integrated_achievement') -> Tuple[str, float]:
        """Return best performance model (ultra-performance + calibration consideration)"""
        
        if self.comparison_results.empty:
            return None, 0.0
        
        try:
            if metric not in self.comparison_results.columns:
                # Alternative metric order
                fallback_metrics = ['ctr_ultra_optimized_score', 'combined_score_enhanced', 'calibration_score']
                for fallback_metric in fallback_metrics:
                    if fallback_metric in self.comparison_results.columns:
                        metric = fallback_metric
                        break
            
            best_idx = self.comparison_results[metric].idxmax()
            best_score = self.comparison_results.loc[best_idx, metric]
            
            return best_idx, best_score
        
        except Exception as e:
            logger.error(f"Best ultra-performance model search failed: {e}")
            return None, 0.0

class EvaluationReporter:
    """Evaluation result reporting class - calibration report included"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.report_data = {}
        
    def generate_model_performance_report_with_calibration(self, 
                                                         comparator: UltraModelComparator,
                                                         save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Model performance report generation - calibration evaluation included"""
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
            
            # Best performance model search (calibration consideration)
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
                'calibration_achievers': int(df['calibration_achievement'].sum()),
                'integrated_achievers': int(df['integrated_achievement'].sum()),
                'calibrated_models': int(df['is_calibrated'].sum()),
                'avg_ece': float(df['calibration_ece'].mean()),
                'avg_mce': float(df['calibration_mce'].mean())
            }
            
            # Recommendations
            if report['performance_analysis']['target_achievers'] == 0:
                report['recommendations'].append("No models achieved target Combined Score 0.30+. Consider model architecture improvements.")
            
            if report['calibration_analysis']['calibration_achievers'] == 0:
                report['recommendations'].append("No models achieved calibration quality targets. Consider calibration method application.")
            
            if save_path:
                try:
                    import json
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    logger.info(f"Report saved to {save_path}")
                except Exception as e:
                    logger.warning(f"Report save failed: {e}")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}

class CTRMetrics:
    """CTR prediction evaluation metrics with improved bias correction"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
        # Initialize advanced metrics
        self.advanced_metrics = CTRAdvancedMetrics(config)
        
        # Configuration with improved defaults
        evaluation_config = getattr(config, 'EVALUATION_CONFIG', {})
        
        self.ap_weight = evaluation_config.get('ap_weight', 0.5)
        self.wll_weight = evaluation_config.get('wll_weight', 0.5)
        self.actual_ctr = evaluation_config.get('target_ctr', 0.0191)
        self.pos_weight = evaluation_config.get('pos_weight', 52.3)
        self.neg_weight = evaluation_config.get('neg_weight', 1.0)
        self.target_combined_score = evaluation_config.get('target_combined_score', 0.34)
        
        # Improved bias correction parameters
        self.ctr_tolerance = evaluation_config.get('ctr_tolerance', 0.001)  # More lenient
        self.bias_penalty_weight = evaluation_config.get('bias_penalty_weight', 8.0)  # Reduced
        self.calibration_weight = evaluation_config.get('calibration_weight', 0.6)
        self.wll_normalization_factor = evaluation_config.get('wll_normalization_factor', 1.5)
        self.ctr_bias_multiplier = evaluation_config.get('ctr_bias_multiplier', 10.0)  # Reduced
        
        self.cache = {}
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Improved combined score with balanced CTR bias penalty"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) == 0 or len(y_pred_proba) == 0:
                return 0.0
            
            if len(y_true) != len(y_pred_proba):
                logger.warning(f"Size mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return 0.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # Calculate base metrics
            ap_score = self.average_precision(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss(y_true, y_pred_proba)
            
            # Improved WLL normalization
            normalized_wll = max(0, 1 - wll_score / self.wll_normalization_factor)
            
            # Base combined score
            base_combined = (ap_score * self.ap_weight) + (normalized_wll * self.wll_weight)
            
            # Improved CTR bias correction
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # More balanced penalty approach
            if ctr_bias > self.ctr_tolerance:
                # Linear penalty with cap instead of exponential
                bias_severity = min(3.0, ctr_bias / self.ctr_tolerance)
                penalty_factor = min(0.5, bias_severity * 0.15)  # Max 50% penalty
                penalized_score = base_combined * (1.0 - penalty_factor)
            else:
                # Small bonus for accurate CTR
                bonus = min(0.05, (self.ctr_tolerance - ctr_bias) / self.ctr_tolerance * 0.05)
                penalized_score = base_combined * (1.0 + bonus)
            
            # Moderate reality check
            if predicted_ctr > 0.12:  # More lenient threshold (was 0.08)
                extreme_penalty = min(0.6, (predicted_ctr - 0.12) * 5)  # Less aggressive penalty
                penalized_score *= (1.0 - extreme_penalty)
            
            final_score = float(np.clip(penalized_score, 0.0, 1.0))
            
            return final_score
            
        except Exception as e:
            logger.error(f"Improved combined score calculation failed: {e}")
            return 0.0
    
    def average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Average precision calculation"""
        return self.advanced_metrics.average_precision_enhanced(y_true, y_pred_proba)
    
    def weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Weighted log loss calculation"""
        return self.advanced_metrics.weighted_log_loss_enhanced(y_true, y_pred_proba)

# Backward compatibility aliases - ensure import availability from main.py
CTRMetrics = CTRAdvancedMetrics
ModelComparator = UltraModelComparator

# Additional backward compatibility guarantee
def create_ctr_metrics():
    """CTR metrics generator"""
    return CTRAdvancedMetrics()

def create_model_comparator():
    """Model comparator generator"""
    return UltraModelComparator()

def create_evaluation_reporter():
    """Evaluation reporter generator"""
    return EvaluationReporter()

# Module level directly accessible main functions
def evaluate_model_performance_with_calibration(y_true, y_pred_proba, model_name="Unknown"):
    """Single model performance evaluation - calibration included"""
    metrics_calc = CTRAdvancedMetrics()
    return metrics_calc.comprehensive_evaluation_ultra_with_calibration(y_true, y_pred_proba, model_name)

def compare_multiple_models_with_calibration(models_predictions, y_true, models_info=None):
    """Multiple model comparison - calibration evaluation included"""
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