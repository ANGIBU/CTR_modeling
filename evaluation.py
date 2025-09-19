# evaluation.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
import warnings
from pathlib import Path
import json
import gc
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, log_loss, brier_score_loss,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_curve, precision_recall_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed. Visualization features will be disabled.")

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
        
        # Safe config access with fallback values
        evaluation_config = getattr(config, 'EVALUATION_CONFIG', {
            'ap_weight': 0.6,
            'wll_weight': 0.4,
            'target_combined_score': 0.34,
            'target_ctr': 0.0191,
            'ctr_tolerance': 0.0005,
            'bias_penalty_weight': 5.0,
            'calibration_weight': 0.4,
            'pos_weight': 49.8,
            'neg_weight': 1.0
        })
        
        self.ap_weight = evaluation_config.get('ap_weight', 0.6)
        self.wll_weight = evaluation_config.get('wll_weight', 0.4)
        self.actual_ctr = evaluation_config.get('target_ctr', 0.0201)
        self.pos_weight = evaluation_config.get('pos_weight', 49.8)
        self.neg_weight = evaluation_config.get('neg_weight', 1.0)
        self.target_combined_score = evaluation_config.get('target_combined_score', 0.30)
        self.ctr_tolerance = evaluation_config.get('ctr_tolerance', 0.0005)
        self.bias_penalty_weight = evaluation_config.get('bias_penalty_weight', 5.0)
        self.calibration_weight = evaluation_config.get('calibration_weight', 0.4)
        
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
                if SKLEARN_AVAILABLE:
                    ap_score = average_precision_score(y_true, y_pred_proba)
                else:
                    ap_score = self._manual_average_precision(y_true, y_pred_proba)
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
            logger.error(f"AP calculation failed: {e}")
            return 0.0
    
    def _manual_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual Average Precision calculation"""
        try:
            indices = np.argsort(y_pred_proba)[::-1]
            y_true_sorted = y_true[indices]
            
            tp_cumsum = np.cumsum(y_true_sorted)
            fp_cumsum = np.cumsum(1 - y_true_sorted)
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-15)
            recall = tp_cumsum / (np.sum(y_true) + 1e-15)
            
            recall_diff = np.diff(np.concatenate(([0], recall)))
            ap = np.sum(precision[1:] * recall_diff)
            
            return float(ap)
        except:
            return 0.0
    
    def weighted_log_loss_ctr(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR specific weighted log loss"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"Size mismatch in WLL calculation")
                return 100.0
            
            if len(y_true) == 0:
                return 100.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            pos_mask = (y_true == 1)
            neg_mask = (y_true == 0)
            
            if not np.any(pos_mask) or not np.any(neg_mask):
                if SKLEARN_AVAILABLE:
                    return log_loss(y_true, y_pred_proba)
                else:
                    return self._manual_log_loss(y_true, y_pred_proba)
            
            pos_loss = -np.mean(np.log(y_pred_proba[pos_mask])) * self.pos_weight
            neg_loss = -np.mean(np.log(1 - y_pred_proba[neg_mask])) * self.neg_weight
            
            total_weight = self.pos_weight + self.neg_weight
            weighted_loss = (pos_loss + neg_loss) / total_weight
            
            return float(weighted_loss)
            
        except Exception as e:
            logger.error(f"WLL calculation failed: {e}")
            return 100.0
    
    def weighted_log_loss_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced weighted log loss with additional stability"""
        try:
            base_wll = self.weighted_log_loss_ctr(y_true, y_pred_proba)
            
            # Additional stability measures
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Apply CTR bias penalty
            if ctr_bias > self.ctr_tolerance:
                bias_penalty = ctr_bias * self.bias_penalty_weight
                enhanced_wll = base_wll + bias_penalty
            else:
                enhanced_wll = base_wll
            
            return float(enhanced_wll)
            
        except Exception as e:
            logger.error(f"Enhanced WLL calculation failed: {e}")
            return 100.0
    
    def _manual_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual log loss calculation"""
        try:
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        except:
            return 100.0
    
    def combined_score_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced combined score calculation"""
        try:
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            
            # Normalize WLL to 0-1 range (lower is better)
            wll_normalized = max(0, 1 - wll_score / 10.0)
            
            # Basic combined score
            combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            # CTR alignment bonus
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            if ctr_bias <= self.ctr_tolerance:
                ctr_bonus = 0.05  # Bonus for good CTR alignment
                combined += ctr_bonus
            
            # Calibration bonus
            try:
                calibration_metrics = self.ctr_calibration_score_enhanced(y_true, y_pred_proba)
                calibration_score = calibration_metrics.get('calibration_score', 0.0)
                calibration_bonus = calibration_score * self.calibration_weight * 0.1
                combined += calibration_bonus
            except:
                pass
            
            return float(np.clip(combined, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Enhanced combined score calculation failed: {e}")
            return 0.0
    
    def ctr_ultra_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR ultra optimized score for high performance achievement"""
        try:
            # Core metrics
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            
            # CTR specific metrics
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            ctr_alignment = max(0, 1 - ctr_bias / self.ctr_tolerance)
            
            # Prediction quality metrics
            pred_std = np.std(y_pred_proba)
            pred_range = np.max(y_pred_proba) - np.min(y_pred_proba)
            diversity_score = min(1.0, pred_std * 5)  # Encourage prediction diversity
            
            # Calibration score
            try:
                calibration_metrics = self.ctr_calibration_score_enhanced(y_true, y_pred_proba)
                calibration_score = calibration_metrics.get('calibration_score', 0.0)
            except:
                calibration_score = 0.0
            
            # Ultra optimized scoring
            base_score = (ap_score * 0.4) + (max(0, 1 - wll_score/5.0) * 0.3)
            ctr_score = ctr_alignment * 0.15
            quality_score = diversity_score * 0.05
            calibration_contribution = calibration_score * 0.1
            
            ultra_score = base_score + ctr_score + quality_score + calibration_contribution
            
            # Performance tier multiplier
            if ultra_score >= 0.35:
                ultra_score *= 1.1  # Exceptional performance bonus
            elif ultra_score >= 0.30:
                ultra_score *= 1.05  # Excellent performance bonus
            
            return float(np.clip(ultra_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Ultra optimized score calculation failed: {e}")
            return 0.0
    
    def ctr_calibration_score_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """Enhanced CTR calibration evaluation"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0  # Expected Calibration Error
            mce = 0.0  # Maximum Calibration Error
            total_samples = len(y_true)
            reliability = 0.0
            resolution = 0.0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(y_true[in_bin])
                    avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                    
                    bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += bin_error * prop_in_bin
                    mce = max(mce, bin_error)
                    
                    # Brier score decomposition components
                    reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                    resolution += prop_in_bin * (accuracy_in_bin - np.mean(y_true)) ** 2
            
            # Brier score calculation
            if SKLEARN_AVAILABLE:
                brier_score = brier_score_loss(y_true, y_pred_proba)
            else:
                brier_score = np.mean((y_pred_proba - y_true) ** 2)
            
            uncertainty = np.mean(y_true) * (1 - np.mean(y_true))
            
            # Calibration slope (ideal = 1.0)
            try:
                if SCIPY_AVAILABLE:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred_proba, y_true)
                    calibration_slope = slope
                else:
                    calibration_slope = 1.0
            except:
                calibration_slope = 1.0
            
            # Overall calibration score (0-1, higher is better)
            calibration_overall = max(0, 1 - ece - mce * 0.5)
            
            # Enhanced calibration score considering CTR specifics
            ctr_predicted = np.mean(y_pred_proba)
            ctr_actual = np.mean(y_true)
            ctr_calibration_bonus = max(0, 1 - abs(ctr_predicted - ctr_actual) / 0.01)
            
            enhanced_calibration_score = (calibration_overall * 0.7 + ctr_calibration_bonus * 0.3)
            
            calibration_metrics = {
                'calibration_ece': float(ece),
                'calibration_mce': float(mce),
                'calibration_brier_reliability': float(reliability),
                'calibration_brier_resolution': float(resolution),
                'calibration_brier_uncertainty': float(uncertainty),
                'calibration_slope': float(calibration_slope),
                'calibration_score': float(enhanced_calibration_score)
            }
            
            return calibration_metrics
            
        except Exception as e:
            logger.error(f"Enhanced calibration evaluation failed: {e}")
            return {
                'calibration_ece': 1.0,
                'calibration_mce': 1.0,
                'calibration_brier_reliability': 1.0,
                'calibration_brier_resolution': 0.0,
                'calibration_brier_uncertainty': 0.25,
                'calibration_slope': 1.0,
                'calibration_score': 0.0
            }
    
    def _compute_ks_statistic(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic"""
        try:
            if not SCIPY_AVAILABLE:
                return 0.0
            
            pos_scores = y_pred_proba[y_true == 1]
            neg_scores = y_pred_proba[y_true == 0]
            
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                return 0.0
            
            ks_stat, _ = stats.ks_2samp(pos_scores, neg_scores)
            return float(ks_stat)
            
        except:
            return 0.0
    
    def _compute_gini_coefficient(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute Gini coefficient"""
        try:
            if SKLEARN_AVAILABLE:
                auc_score = roc_auc_score(y_true, y_pred_proba)
                gini = 2 * auc_score - 1
            else:
                gini = 0.0
            
            return float(gini)
            
        except:
            return 0.0
    
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
    
    def comprehensive_evaluation_ultra_with_calibration(self, 
                                                       y_true: np.ndarray, 
                                                       y_pred_proba: np.ndarray, 
                                                       model_name: str = "Unknown",
                                                       threshold: float = 0.5) -> Dict[str, Any]:
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
                    'ap_enhanced': 0.0, 'wll_enhanced': 100.0, 
                    'combined_score_enhanced': -100.0, 'ctr_ultra_optimized_score': 0.0
                })
            
            # Standard sklearn metrics
            try:
                if SKLEARN_AVAILABLE:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                else:
                    metrics['auc'] = 0.5
                    metrics['log_loss'] = self._manual_log_loss(y_true, y_pred_proba)
                    metrics['brier_score'] = np.mean((y_pred_proba - y_true) ** 2)
            except Exception as e:
                logger.warning(f"Standard metrics calculation failed: {e}")
                metrics.update({'auc': 0.5, 'log_loss': 100.0, 'brier_score': 1.0})
            
            # Classification metrics
            try:
                if SKLEARN_AVAILABLE:
                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        
                        metrics['accuracy'] = accuracy_score(y_true, y_pred)
                        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
                        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
                        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        metrics['sensitivity'] = metrics['recall']
                        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
                        
                        # Matthews Correlation Coefficient
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
                else:
                    # Manual calculation fallback
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    tn = np.sum((y_true == 0) & (y_pred == 0))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))
                    
                    metrics['accuracy'] = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
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
                
            except Exception as e:
                logger.warning(f"Classification metrics calculation failed: {e}")
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0,
                    'balanced_accuracy': 0.0, 'geometric_mean': 0.0
                })
            
            # CTR ultra precision analysis
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
                logger.warning(f"CTR ultra precision analysis failed: {e}")
                metrics.update({
                    'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
                    'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0,
                    'ctr_alignment_score': 1.0, 'ctr_stability': 1.0, 'target_ctr_achievement': 1.0
                })
            
            # Prediction distribution analysis
            try:
                metrics['prediction_std'] = float(np.std(y_pred_proba))
                metrics['prediction_var'] = float(np.var(y_pred_proba))
                metrics['prediction_entropy'] = self._calculate_entropy_enhanced(y_pred_proba)
                metrics['prediction_gini'] = self._compute_gini_coefficient(y_true, y_pred_proba)
                metrics['prediction_range'] = float(np.max(y_pred_proba) - np.min(y_pred_proba))
                
                pred_q25, pred_q75 = np.percentile(y_pred_proba, [25, 75])
                metrics['prediction_iqr'] = float(pred_q75 - pred_q25)
                
                unique_predictions = len(np.unique(y_pred_proba))
                metrics['unique_predictions_ratio'] = unique_predictions / len(y_pred_proba)
                metrics['effective_sample_size'] = len(y_pred_proba)
                
            except Exception as e:
                logger.warning(f"Prediction distribution analysis failed: {e}")
                metrics.update({
                    'prediction_std': 0.0, 'prediction_var': 0.0, 'prediction_entropy': 0.0, 'prediction_gini': 0.0,
                    'prediction_range': 0.0, 'prediction_iqr': 0.0, 'unique_predictions_ratio': 0.0, 'effective_sample_size': 1000
                })
            
            # Positive/Negative class prediction analysis
            try:
                pos_mask = (y_true == 1)
                neg_mask = (y_true == 0)
                
                if np.any(pos_mask):
                    metrics['pos_mean_pred'] = float(np.mean(y_pred_proba[pos_mask]))
                    metrics['pos_std_pred'] = float(np.std(y_pred_proba[pos_mask]))
                    metrics['pos_median_pred'] = float(np.median(y_pred_proba[pos_mask]))
                else:
                    metrics['pos_mean_pred'] = 0.0
                    metrics['pos_std_pred'] = 0.0
                    metrics['pos_median_pred'] = 0.0
                
                if np.any(neg_mask):
                    metrics['neg_mean_pred'] = float(np.mean(y_pred_proba[neg_mask]))
                    metrics['neg_std_pred'] = float(np.std(y_pred_proba[neg_mask]))
                    metrics['neg_median_pred'] = float(np.median(y_pred_proba[neg_mask]))
                else:
                    metrics['neg_mean_pred'] = 0.0
                    metrics['neg_std_pred'] = 0.0
                    metrics['neg_median_pred'] = 0.0
                
                metrics['separation'] = metrics['pos_mean_pred'] - metrics['neg_mean_pred']
                metrics['separation_ratio'] = metrics['pos_mean_pred'] / max(metrics['neg_mean_pred'], 1e-10)
                
            except Exception as e:
                logger.warning(f"Class prediction analysis failed: {e}")
                metrics.update({
                    'pos_mean_pred': 0.0, 'pos_std_pred': 0.0, 'pos_median_pred': 0.0,
                    'neg_mean_pred': 0.0, 'neg_std_pred': 0.0, 'neg_median_pred': 0.0,
                    'separation': 0.0, 'separation_ratio': 1.0
                })
            
            # Target achievement analysis
            try:
                target_combined_score = self.target_combined_score
                current_combined_score = metrics.get('combined_score_enhanced', 0.0)
                
                metrics['target_combined_score_achievement'] = 1.0 if current_combined_score >= target_combined_score else 0.0
                metrics['combined_score_gap'] = max(0, target_combined_score - current_combined_score)
                
                ultra_score = metrics.get('ctr_ultra_optimized_score', 0.0)
                metrics['ultra_score_achievement'] = 1.0 if ultra_score >= 0.30 else 0.0
                
                performance_tier = self._classify_performance_tier(current_combined_score)
                metrics['performance_tier'] = performance_tier
                
            except Exception as e:
                logger.warning(f"Target achievement analysis failed: {e}")
                metrics.update({
                    'target_combined_score_achievement': 0.0, 'combined_score_gap': self.target_combined_score,
                    'ultra_score_achievement': 0.0, 'performance_tier': 'poor'
                })
            
            # Calibration evaluation
            try:
                calibration_results = self.ctr_calibration_score_enhanced(y_true, y_pred_proba)
                metrics.update(calibration_results)
                
                # Calibration achievement
                calibration_score = calibration_results.get('calibration_score', 0.0)
                metrics['calibration_achievement'] = 1.0 if calibration_score >= 0.7 else 0.0
                metrics['calibration_gap'] = max(0, 0.7 - calibration_score)
                
            except Exception as e:
                logger.warning(f"Calibration evaluation failed: {e}")
                metrics.update({
                    'calibration_ece': 1.0, 'calibration_mce': 1.0,
                    'calibration_brier_reliability': 1.0, 'calibration_brier_resolution': 0.0,
                    'calibration_score': 0.0, 'calibration_achievement': 0.0, 'calibration_gap': 0.7
                })
            
            # Integrated achievement score
            try:
                target_achievement = metrics.get('target_combined_score_achievement', 0.0)
                calibration_achievement = metrics.get('calibration_achievement', 0.0)
                ctr_achievement = metrics.get('target_ctr_achievement', 0.0)
                
                metrics['integrated_achievement'] = (target_achievement + calibration_achievement + ctr_achievement) / 3.0
                
            except Exception as e:
                logger.warning(f"Integrated achievement calculation failed: {e}")
                metrics['integrated_achievement'] = 0.0
            
            # Evaluation time
            metrics['evaluation_time'] = time.time() - start_time
            
            # Validate all numeric values
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
            logger.error(f"Ultra comprehensive evaluation calculation error: {str(e)}")
            return self._get_default_metrics_ultra_with_calibration(model_name)
    
    def _get_default_metrics_ultra_with_calibration(self, model_name: str = "Unknown") -> Dict[str, float]:
        """Ultra high-performance default metric values - including calibration"""
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
                
                # Comprehensive evaluation including calibration
                metrics = self.metrics_calculator.comprehensive_evaluation_ultra_with_calibration(
                    y_true, y_pred_proba, model_name
                )
                
                evaluation_time = time.time() - start_time
                metrics['evaluation_duration'] = evaluation_time
                
                # Add model info
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
            logger.error("No models available for evaluation")
            return pd.DataFrame()
        
        try:
            comparison_df = pd.DataFrame(results)
            
            # Sort by enhanced combined score
            comparison_df = comparison_df.sort_values('combined_score_enhanced', ascending=False)
            comparison_df.reset_index(drop=True, inplace=True)
            
            self.comparison_results = comparison_df
            
            # Performance analysis
            self._ultra_performance_analysis_with_calibration()
            
            logger.info(f"Ultra high-performance model comparison completed - {len(comparison_df)} models evaluated")
            logger.info(f"Best model: {comparison_df.iloc[0]['model_name']} (Score: {comparison_df.iloc[0]['combined_score_enhanced']:.4f})")
            
            return self.comparison_results
            
        except Exception as e:
            logger.error(f"Ultra high-performance comparison result processing failed: {str(e)}")
            return pd.DataFrame()
    
    def _ultra_performance_analysis_with_calibration(self):
        """Ultra high-performance analysis - including calibration evaluation"""
        try:
            if self.comparison_results.empty:
                return
            
            # Basic statistics
            self.performance_analysis = {
                'total_models': len(self.comparison_results),
                'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'best_model_info': {},
                'target_achievers': {},
                'calibration_analysis': {},
                'performance_distribution': {},
                'recommendations': []
            }
            
            # Best model information
            best_row = self.comparison_results.iloc[0]
            self.performance_analysis['best_model_info'] = {
                'name': best_row['model_name'],
                'combined_score_enhanced': float(best_row['combined_score_enhanced']),
                'ultra_optimized_score': float(best_row['ctr_ultra_optimized_score']),
                'ctr_bias': float(best_row['ctr_bias']),
                'calibration_score': float(best_row.get('calibration_score', 0.0)),
                'is_calibrated': bool(best_row.get('is_calibrated', False)),
                'performance_tier': best_row['performance_tier']
            }
            
            # Target achievers analysis
            target_achievers_count = int((self.comparison_results['combined_score_enhanced'] >= 0.30).sum())
            calibration_achievers_count = int((self.comparison_results['calibration_score'] >= 0.7).sum())
            ultra_achievers_count = int((self.comparison_results['ctr_ultra_optimized_score'] >= 0.30).sum())
            
            self.performance_analysis['target_achievers'] = {
                'combined_score_achievers': target_achievers_count,
                'calibration_achievers': calibration_achievers_count, 
                'ultra_score_achievers': ultra_achievers_count,
                'integrated_achievers': int((self.comparison_results['integrated_achievement'] >= 0.7).sum()),
                'achievement_rates': {
                    'combined_score_rate': float(target_achievers_count / len(self.comparison_results)),
                    'calibration_rate': float(calibration_achievers_count / len(self.comparison_results)),
                    'ultra_score_rate': float(ultra_achievers_count / len(self.comparison_results))
                }
            }
            
            # Calibration analysis
            calibration_scores = self.comparison_results['calibration_score']
            self.performance_analysis['calibration_analysis'] = {
                'mean_calibration_score': float(calibration_scores.mean()),
                'std_calibration_score': float(calibration_scores.std()),
                'min_calibration_score': float(calibration_scores.min()),
                'max_calibration_score': float(calibration_scores.max()),
                'models_well_calibrated': int((calibration_scores >= 0.7).sum()),
                'calibration_achievement_rate': float((calibration_scores >= 0.7).mean()),
                'calibrated_models_count': int(self.comparison_results['is_calibrated'].sum())
            }
            
            # Performance tier distribution
            tier_counts = self.comparison_results['performance_tier'].value_counts()
            self.performance_analysis['performance_distribution'] = dict(tier_counts)
            
            # Generate recommendations
            self._generate_ultra_recommendations()
            
        except Exception as e:
            logger.error(f"Ultra performance analysis failed: {e}")
            self.performance_analysis = {}
    
    def _generate_ultra_recommendations(self):
        """Generate ultra high-performance recommendations"""
        try:
            recommendations = []
            
            if self.comparison_results.empty:
                return
            
            # Best model recommendation
            best_model = self.performance_analysis['best_model_info']
            if best_model['combined_score_enhanced'] >= 0.30:
                recommendations.append(f"Use {best_model['name']} as primary model (Score: {best_model['combined_score_enhanced']:.4f})")
            else:
                recommendations.append(f"Best model {best_model['name']} below target 0.30 (Score: {best_model['combined_score_enhanced']:.4f}) - consider ensemble")
            
            # Calibration recommendations
            calibration_analysis = self.performance_analysis['calibration_analysis']
            if calibration_analysis['calibration_achievement_rate'] < 0.5:
                recommendations.append("Apply calibration to more models - low calibration achievement rate")
            
            # Model diversity recommendations
            if len(self.comparison_results) < 3:
                recommendations.append("Train additional models for ensemble diversity")
            
            # Performance improvement recommendations
            poor_performers = (self.comparison_results['combined_score_enhanced'] < 0.15).sum()
            if poor_performers > len(self.comparison_results) * 0.5:
                recommendations.append("Review feature engineering and model hyperparameters - many poor performers")
            
            self.performance_analysis['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Ultra recommendation generation failed: {e}")
            self.performance_analysis['recommendations'] = []
    
    def get_best_model_ultra_with_calibration(self, metric: str = 'combined_score_enhanced') -> Tuple[Optional[str], float]:
        """Get ultra high-performance best model - considering calibration"""
        try:
            if self.comparison_results.empty:
                logger.warning("No comparison results available")
                return None, 0.0
            
            if metric not in self.comparison_results.columns:
                logger.warning(f"Metric {metric} not found, using combined_score_enhanced")
                metric = 'combined_score_enhanced'
            
            best_idx = self.comparison_results[metric].idxmax()
            best_model = self.comparison_results.loc[best_idx, 'model_name']
            best_score = self.comparison_results.loc[best_idx, metric]
            
            return best_model, float(best_score)
            
        except Exception as e:
            logger.error(f"Ultra high-performance best model finding failed: {e}")
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
            
            # Find best model (considering calibration)
            best_model, best_score = comparator.get_best_model_ultra_with_calibration()
            report['summary']['best_model'] = best_model
            report['summary']['best_score'] = best_score
            
            # Detailed results
            for idx, row in comparator.comparison_results.iterrows():
                model_data = row.to_dict()
                report['detailed_results'][model_data['model_name']] = model_data
            
            # Performance analysis
            report['performance_analysis'] = comparator.performance_analysis
            
            # Calibration specific analysis
            calibration_scores = comparator.comparison_results['calibration_score']
            report['calibration_analysis'] = {
                'mean_calibration_score': float(calibration_scores.mean()),
                'std_calibration_score': float(calibration_scores.std()),
                'min_calibration_score': float(calibration_scores.min()),
                'max_calibration_score': float(calibration_scores.max()),
                'models_well_calibrated': int((calibration_scores >= 0.7).sum()),
                'calibration_achievement_rate': float((calibration_scores >= 0.7).mean()),
                'target_achievers_count': int(comparator.performance_analysis.get('target_achievers_count', 0))
            }
            
            # Recommendations
            report['recommendations'] = comparator.performance_analysis.get('recommendations', [])
            
            # Save report
            if save_path:
                try:
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                    
                    logger.info(f"Report saved: {save_path}")
                    
                except Exception as e:
                    logger.error(f"Report saving failed: {e}")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': f'Report generation failed: {str(e)}'}

def create_ultra_comparison_report():
    """Create ultra comparison report"""
    try:
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Ultra comparison report creation failed: {e}")
        return pd.DataFrame()

# Backward compatibility aliases - ensure import from main.py
CTRMetrics = CTRAdvancedMetrics
CTRMetricsCalculator = CTRAdvancedMetrics  # Added missing alias
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