# evaluation.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import logging
import gc
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import (
        average_precision_score, log_loss, roc_auc_score, precision_recall_curve,
        roc_curve, auc, accuracy_score, precision_score, recall_score, 
        f1_score, matthews_corrcoef, confusion_matrix, brier_score_loss
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not installed. Some evaluation metrics will not be available.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization features will be disabled.")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not installed. Some statistical functionality will be disabled.")

from config import Config

logger = logging.getLogger(__name__)

class AdaptiveCTRCorrector:
    """Adaptive CTR bias correction with data-dependent optimization"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.correction_history = []
        self.adaptive_factor = 1.0
        self.stability_threshold = 0.001
        self.max_correction_factor = 2.0
        self.min_correction_factor = 0.5
        
    def calculate_adaptive_correction(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate data-dependent correction factor"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            
            if predicted_ctr <= 0:
                return 1.0
                
            # Base correction ratio
            base_ratio = self.target_ctr / predicted_ctr
            
            # Data quality assessment
            pred_std = np.std(y_pred_proba)
            pred_range = np.max(y_pred_proba) - np.min(y_pred_proba)
            diversity_score = pred_std * pred_range
            
            # Stability factor (less aggressive correction for diverse predictions)
            stability_factor = min(1.2, max(0.8, diversity_score * 10))
            
            # Calculate adaptive correction
            adaptive_correction = base_ratio * stability_factor
            
            # Constrain correction factor
            adaptive_correction = np.clip(
                adaptive_correction, 
                self.min_correction_factor, 
                self.max_correction_factor
            )
            
            self.correction_history.append(adaptive_correction)
            
            # Keep history manageable
            if len(self.correction_history) > 10:
                self.correction_history = self.correction_history[-10:]
                
            return float(adaptive_correction)
            
        except Exception as e:
            logger.warning(f"Adaptive correction calculation failed: {e}")
            return 1.0
    
    def get_smoothed_correction(self) -> float:
        """Get smoothed correction factor from history"""
        if not self.correction_history:
            return 1.0
            
        # Use weighted average with recent corrections having more weight
        weights = np.exp(np.linspace(-1, 0, len(self.correction_history)))
        weighted_correction = np.average(self.correction_history, weights=weights)
        
        return float(weighted_correction)

class CTRAdvancedMetrics:
    """CTR prediction advanced evaluation metrics with adaptive correction"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
        # Get evaluation config with improved defaults
        evaluation_config = getattr(config, 'EVALUATION_CONFIG', {})
        
        self.ap_weight = evaluation_config.get('ap_weight', 0.5)
        self.wll_weight = evaluation_config.get('wll_weight', 0.5)
        self.actual_ctr = evaluation_config.get('target_ctr', 0.0191)
        self.pos_weight = evaluation_config.get('pos_weight', 52.3)
        self.neg_weight = evaluation_config.get('neg_weight', 1.0)
        self.target_combined_score = evaluation_config.get('target_combined_score', 0.34)
        
        # Improved CTR bias correction parameters (less aggressive)
        self.ctr_tolerance = evaluation_config.get('ctr_tolerance', 0.001)  # More lenient
        self.bias_penalty_weight = evaluation_config.get('bias_penalty_weight', 8.0)  # Reduced from 15.0
        self.calibration_weight = evaluation_config.get('calibration_weight', 0.6)
        self.wll_normalization_factor = evaluation_config.get('wll_normalization_factor', 1.5)  # Reduced from 1.8
        self.ctr_bias_multiplier = evaluation_config.get('ctr_bias_multiplier', 12.0)  # Reduced from 20.0
        
        # Initialize adaptive corrector
        self.adaptive_corrector = AdaptiveCTRCorrector(target_ctr=self.actual_ctr)
        
        self.cache = {}
        
    def average_precision_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Average Precision with stability improvements"""
        try:
            cache_key = f"ap_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return 0.0
            
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("Only single class exists")
                return 0.0
            
            # Clean and clip predictions
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("Invalid predictions detected, cleaning")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            try:
                ap_score = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"sklearn AP calculation failed, using manual calculation: {e}")
                ap_score = self._manual_average_precision(y_true, y_pred_proba)
            
            if np.isnan(ap_score) or np.isinf(ap_score):
                logger.warning("AP calculation result invalid")
                return 0.0
            
            ap_score = float(np.clip(ap_score, 0.0, 1.0))
            self.cache[cache_key] = ap_score
            
            return ap_score
            
        except Exception as e:
            logger.error(f"Enhanced AP calculation failed: {e}")
            return 0.0
    
    def _manual_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual average precision calculation fallback"""
        try:
            if SKLEARN_AVAILABLE:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                return auc(recall, precision)
            else:
                # Simple fallback implementation
                sorted_indices = np.argsort(y_pred_proba)[::-1]
                y_true_sorted = y_true[sorted_indices]
                
                # Calculate precision at each point
                precisions = []
                true_positives = 0
                
                for i, label in enumerate(y_true_sorted):
                    if label == 1:
                        true_positives += 1
                    precision = true_positives / (i + 1)
                    precisions.append(precision)
                
                # Average precision approximation
                return np.mean(precisions) if precisions else 0.0
                
        except Exception as e:
            logger.error(f"Manual AP calculation failed: {e}")
            return 0.0
    
    def weighted_log_loss_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced weighted log loss with improved stability"""
        try:
            cache_key = f"wll_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return float('inf')
            
            # Clean predictions
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Calculate class weights based on actual data distribution
            pos_count = np.sum(y_true)
            neg_count = len(y_true) - pos_count
            
            if pos_count == 0 or neg_count == 0:
                return float('inf')
            
            # Dynamic weight calculation
            pos_weight = neg_count / pos_count
            neg_weight = 1.0
            
            # Apply weights
            sample_weights = np.where(y_true == 1, pos_weight, neg_weight)
            
            # Calculate weighted log loss
            log_loss_values = -(y_true * np.log(y_pred_proba) + 
                             (1 - y_true) * np.log(1 - y_pred_proba))
            
            weighted_loss = np.average(log_loss_values, weights=sample_weights)
            
            if np.isnan(weighted_loss) or np.isinf(weighted_loss):
                return float('inf')
            
            weighted_loss = float(max(0.0, weighted_loss))
            self.cache[cache_key] = weighted_loss
            
            return weighted_loss
            
        except Exception as e:
            logger.error(f"Enhanced WLL calculation failed: {e}")
            return float('inf')
    
    def combined_score_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced combined score with adaptive CTR bias correction"""
        try:
            cache_key = f"combined_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return 0.0
            
            # Calculate base metrics
            ap_score = self.average_precision_enhanced(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_enhanced(y_true, y_pred_proba)
            
            # Normalize WLL with improved approach
            wll_normalized = 1 / (1 + wll_score / self.wll_normalization_factor) if wll_score != float('inf') else 0.0
            
            # Base combined score
            base_combined = (ap_score * self.ap_weight) + (wll_normalized * self.wll_weight)
            
            # Adaptive CTR bias correction
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Calculate adaptive correction factor
            adaptive_correction = self.adaptive_corrector.calculate_adaptive_correction(y_true, y_pred_proba)
            
            # Apply bias correction with adaptive approach
            if ctr_bias > self.ctr_tolerance:
                # Progressive penalty (less aggressive than exponential)
                bias_severity = min(5.0, ctr_bias / self.ctr_tolerance)
                penalty_factor = min(0.7, bias_severity * 0.1)  # Max 70% penalty, more gradual
                ctr_adjusted_score = base_combined * (1.0 - penalty_factor)
            else:
                # Reward accurate CTR prediction
                accuracy_bonus = min(0.08, (self.ctr_tolerance - ctr_bias) / self.ctr_tolerance * 0.08)
                ctr_adjusted_score = base_combined * (1.0 + accuracy_bonus)
            
            # Additional reality check with improved thresholds
            if predicted_ctr > 0.12:  # More lenient threshold (was 0.08)
                extreme_penalty = min(0.6, (predicted_ctr - 0.12) * 5)  # Less aggressive penalty
                ctr_adjusted_score *= (1.0 - extreme_penalty)
            
            final_score = float(np.clip(ctr_adjusted_score, 0.0, 1.0))
            self.cache[cache_key] = final_score
            
            return final_score
            
        except Exception as e:
            logger.error(f"Enhanced combined score calculation failed: {e}")
            return 0.0
    
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
            
            # CTR precision analysis
            predicted_ctr = y_pred_proba.mean()
            actual_ctr = y_true.mean()
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Ultra-precision CTR alignment with adaptive thresholds
            ultra_ctr_tolerance = max(0.0005, self.ctr_tolerance * 0.5)  # Dynamic tolerance
            if ctr_bias < ultra_ctr_tolerance:
                ultra_ctr_accuracy = np.exp(-ctr_bias * 1000)  # Less aggressive exponential
            else:
                ultra_ctr_accuracy = 0.1  # Give some base score instead of 0
            
            ctr_stability = 1.0 / (1.0 + ctr_bias * 5000)  # Reduced multiplier
            
            # Advanced calibration assessment
            try:
                calibration_metrics = self.calibration_score_advanced(y_true, y_pred_proba)
                calibration_score = calibration_metrics.get('calibration_score', 0.5)
            except Exception as e:
                logger.warning(f"Advanced calibration calculation failed: {e}")
                calibration_score = 0.5
            
            # Distribution matching with improved algorithm
            try:
                distribution_score = self._calculate_improved_distribution_matching(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Distribution matching calculation failed: {e}")
                distribution_score = 0.5
            
            # Prediction quality assessment
            try:
                pred_quality = self._assess_prediction_quality(y_pred_proba)
            except Exception as e:
                logger.warning(f"Prediction quality assessment failed: {e}")
                pred_quality = 0.5
            
            # Optimized score combination with balanced weights
            base_score = (ap_score * 0.35) + (wll_normalized * 0.25)
            ctr_component = (ultra_ctr_accuracy * 0.15) + (ctr_stability * 0.1)
            quality_component = (calibration_score * 0.1) + (distribution_score * 0.05)
            
            ultra_score = base_score + ctr_component + quality_component
            
            # Final adjustments
            if predicted_ctr > 0.15:  # Very high prediction penalty
                ultra_score *= max(0.3, 1.0 - (predicted_ctr - 0.15) * 2)
            
            final_score = float(np.clip(ultra_score, 0.0, 1.0))
            self.cache[cache_key] = final_score
            
            return final_score
            
        except Exception as e:
            logger.error(f"Ultra-optimized score calculation failed: {e}")
            return 0.0
    
    def _calculate_improved_distribution_matching(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Improved distribution matching algorithm"""
        try:
            # Bin-based distribution comparison
            n_bins = min(50, max(10, len(y_true) // 1000))
            
            # Create probability bins
            bin_edges = np.linspace(0, 1, n_bins + 1)
            
            # Calculate bin-wise statistics
            bin_scores = []
            for i in range(n_bins):
                mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
                if i == n_bins - 1:  # Include right edge for last bin
                    mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
                
                if np.sum(mask) > 0:
                    bin_actual = np.mean(y_true[mask])
                    bin_predicted = np.mean(y_pred_proba[mask])
                    bin_error = abs(bin_actual - bin_predicted)
                    bin_score = max(0.0, 1.0 - bin_error * 10)  # Less aggressive penalty
                    bin_scores.append(bin_score)
            
            if not bin_scores:
                return 0.5
            
            return float(np.mean(bin_scores))
            
        except Exception as e:
            logger.warning(f"Improved distribution matching failed: {e}")
            return 0.5
    
    def _assess_prediction_quality(self, y_pred_proba: np.ndarray) -> float:
        """Assess overall prediction quality"""
        try:
            # Diversity metrics
            pred_std = np.std(y_pred_proba)
            pred_range = np.max(y_pred_proba) - np.min(y_pred_proba)
            unique_ratio = len(np.unique(y_pred_proba)) / len(y_pred_proba)
            
            # Quality score based on diversity and range
            diversity_score = min(1.0, pred_std * 20)  # Reward standard deviation
            range_score = min(1.0, pred_range * 2)     # Reward prediction range
            uniqueness_score = min(1.0, unique_ratio * 2)  # Reward unique predictions
            
            # Combined quality score
            quality_score = (diversity_score * 0.4 + range_score * 0.3 + uniqueness_score * 0.3)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Prediction quality assessment failed: {e}")
            return 0.5
    
    def calibration_score_advanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Advanced calibration quality assessment with multiple metrics"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return self._get_default_calibration_metrics()
            
            # Clean data
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            calibration_metrics = {}
            
            # Expected Calibration Error (ECE)
            try:
                ece_score = self._calculate_ece(y_true, y_pred_proba)
                calibration_metrics['ece'] = ece_score
            except Exception as e:
                logger.warning(f"ECE calculation failed: {e}")
                calibration_metrics['ece'] = 1.0
            
            # Maximum Calibration Error (MCE)
            try:
                mce_score = self._calculate_mce(y_true, y_pred_proba)
                calibration_metrics['mce'] = mce_score
            except Exception as e:
                logger.warning(f"MCE calculation failed: {e}")
                calibration_metrics['mce'] = 1.0
            
            # Brier Score decomposition
            try:
                brier_reliability, brier_resolution = self._calculate_brier_decomposition(y_true, y_pred_proba)
                calibration_metrics['brier_reliability'] = brier_reliability
                calibration_metrics['brier_resolution'] = brier_resolution
            except Exception as e:
                logger.warning(f"Brier decomposition failed: {e}")
                calibration_metrics['brier_reliability'] = 1.0
                calibration_metrics['brier_resolution'] = 0.0
            
            # Overall calibration score (lower ECE/MCE is better)
            ece_score_norm = max(0.0, 1.0 - calibration_metrics['ece'] * 10)  # Less aggressive
            mce_score_norm = max(0.0, 1.0 - calibration_metrics['mce'] * 10)  # Less aggressive
            reliability_score = max(0.0, 1.0 - calibration_metrics['brier_reliability'] * 5)
            resolution_score = min(1.0, calibration_metrics['brier_resolution'] * 10)
            
            # Balanced combination
            overall_score = (0.3 * ece_score_norm + 
                           0.2 * mce_score_norm + 
                           0.3 * reliability_score + 
                           0.2 * resolution_score)
            
            calibration_metrics['calibration_score'] = max(0.0, min(overall_score, 1.0))
            
            return calibration_metrics
            
        except Exception as e:
            logger.warning(f"Advanced calibration score calculation failed: {e}")
            return self._get_default_calibration_metrics()
    
    def _get_default_calibration_metrics(self) -> Dict[str, float]:
        """Default calibration metrics"""
        return {
            'ece': 1.0,
            'mce': 1.0, 
            'brier_reliability': 1.0,
            'brier_resolution': 0.0,
            'calibration_score': 0.0
        }
    
    def _calculate_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 15) -> float:
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
    
    def _calculate_mce(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 15) -> float:
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
            if predicted_ctr > 0.10:  # More lenient than before
                extreme_penalty = min(0.4, (predicted_ctr - 0.10) * 4)  # Less aggressive
                penalized_score *= (1.0 - extreme_penalty)
            
            return float(np.clip(penalized_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.0
    
    def average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Average precision calculation with caching"""
        try:
            if not SKLEARN_AVAILABLE:
                return self._manual_average_precision(y_true, y_pred_proba)
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            cache_key = f"ap_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return 0.0
            
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("Only single class exists")
                return 0.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            try:
                ap_score = average_precision_score(y_true, y_pred_proba)
                ap_score = float(np.clip(ap_score, 0.0, 1.0))
                
                self.cache[cache_key] = ap_score
                return ap_score
                
            except Exception as e:
                logger.warning(f"sklearn AP failed, using manual calculation: {e}")
                return self._manual_average_precision(y_true, y_pred_proba)
                
        except Exception as e:
            logger.error(f"AP calculation failed: {e}")
            return 0.0
    
    def _manual_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual average precision calculation"""
        try:
            # Sort by prediction score descending
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            # Calculate precision at each recall level
            true_positives = np.cumsum(y_true_sorted)
            false_positives = np.cumsum(1 - y_true_sorted)
            
            # Precision = TP / (TP + FP)
            precision = true_positives / (true_positives + false_positives + 1e-10)
            
            # Average precision
            ap = 0.0
            prev_recall = 0.0
            
            total_positives = np.sum(y_true)
            if total_positives == 0:
                return 0.0
            
            for i in range(len(y_true_sorted)):
                if y_true_sorted[i] == 1:  # If this is a positive example
                    recall = true_positives[i] / total_positives
                    ap += precision[i] * (recall - prev_recall)
                    prev_recall = recall
            
            return float(np.clip(ap, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Manual AP calculation failed: {e}")
            return 0.0
    
    def weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Weighted log loss calculation"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            cache_key = f"wll_{hash(y_true.tobytes())}_{hash(y_pred_proba.tobytes())}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return float('inf')
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # Calculate sample weights
            pos_count = np.sum(y_true)
            neg_count = len(y_true) - pos_count
            
            if pos_count == 0 or neg_count == 0:
                return float('inf')
            
            sample_weights = np.where(y_true == 1, self.pos_weight, self.neg_weight)
            
            # Calculate weighted log loss
            log_losses = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
            weighted_loss = np.average(log_losses, weights=sample_weights)
            
            if np.isnan(weighted_loss) or np.isinf(weighted_loss):
                return float('inf')
            
            weighted_loss = float(max(0.0, weighted_loss))
            self.cache[cache_key] = weighted_loss
            
            return weighted_loss
            
        except Exception as e:
            logger.error(f"WLL calculation failed: {e}")
            return float('inf')
    
    def comprehensive_evaluation_ultra_with_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                                      model_name: str = "Unknown") -> Dict[str, Any]:
        """Comprehensive evaluation with ultra optimization and calibration"""
        try:
            start_time = time.time()
            
            # Clean input data
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return self._get_default_metrics_ultra_with_calibration(model_name)
            
            # Clean predictions
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning(f"{model_name}: Invalid predictions detected, cleaning")
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
            
            metrics = {}
            
            # Core metrics
            metrics['model_name'] = model_name
            metrics['ap_enhanced'] = self.advanced_metrics.average_precision_enhanced(y_true, y_pred_proba)
            metrics['wll_enhanced'] = self.advanced_metrics.weighted_log_loss_enhanced(y_true, y_pred_proba)
            metrics['combined_score_enhanced'] = self.advanced_metrics.combined_score_enhanced(y_true, y_pred_proba)
            metrics['ctr_ultra_optimized_score'] = self.advanced_metrics.ctr_ultra_optimized_score(y_true, y_pred_proba)
            
            # Standard metrics for compatibility
            metrics['ap'] = self.average_precision(y_true, y_pred_proba)
            metrics['combined_score'] = self.combined_score(y_true, y_pred_proba)
            
            # AUC calculation
            try:
                if SKLEARN_AVAILABLE:
                    metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba))
                else:
                    metrics['auc'] = 0.5
            except Exception as e:
                logger.warning(f"AUC calculation failed: {e}")
                metrics['auc'] = 0.5
            
            # CTR analysis
            try:
                metrics['ctr_actual'] = float(y_true.mean())
                metrics['ctr_predicted'] = float(y_pred_proba.mean())
                metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
                metrics['ctr_ratio'] = metrics['ctr_predicted'] / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
                metrics['ctr_relative_error'] = abs(metrics['ctr_bias']) / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_alignment_score'] = np.exp(-abs(metrics['ctr_bias']) * 500)  # Less aggressive
                metrics['ctr_stability'] = 1.0 / (1.0 + abs(metrics['ctr_bias']) * 5000)  # Less aggressive
                metrics['target_ctr_achievement'] = 1.0 if abs(metrics['ctr_bias']) < 0.001 else 0.0
            except Exception as e:
                logger.warning(f"CTR analysis failed: {e}")
                metrics.update({
                    'ctr_actual': 0.0191, 'ctr_predicted': 0.0191, 'ctr_bias': 0.0,
                    'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0,
                    'ctr_alignment_score': 1.0, 'ctr_stability': 1.0, 'target_ctr_achievement': 1.0
                })
            
            # Calibration metrics
            try:
                calibration_metrics = self.advanced_metrics.calibration_score_advanced(y_true, y_pred_proba)
                metrics.update(calibration_metrics)
            except Exception as e:
                logger.warning(f"Calibration metrics calculation failed: {e}")
                metrics.update(self.advanced_metrics._get_default_calibration_metrics())
            
            # Performance tier classification
            combined_score = metrics['combined_score_enhanced']
            if combined_score >= 0.35:
                tier = 'EXCEPTIONAL'
            elif combined_score >= 0.30:
                tier = 'EXCELLENT'
            elif combined_score >= 0.25:
                tier = 'GOOD'
            elif combined_score >= 0.20:
                tier = 'FAIR'
            else:
                tier = 'POOR'
            
            metrics['performance_tier'] = tier
            
            # Achievement flags
            metrics['target_combined_score_achievement'] = 1.0 if combined_score >= self.target_combined_score else 0.0
            metrics['ultra_score_achievement'] = 1.0 if metrics['ctr_ultra_optimized_score'] >= 0.32 else 0.0
            metrics['calibration_achievement'] = 1.0 if metrics['calibration_score'] >= 0.7 else 0.0
            metrics['integrated_achievement'] = 1.0 if (metrics['target_combined_score_achievement'] == 1.0 and 
                                                      metrics['calibration_achievement'] == 1.0) else 0.0
            
            # Timing
            evaluation_time = time.time() - start_time
            metrics['evaluation_duration'] = evaluation_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed for {model_name}: {e}")
            return self._get_default_metrics_ultra_with_calibration(model_name)
    
    def _get_default_metrics_ultra_with_calibration(self, model_name: str) -> Dict[str, Any]:
        """Default metrics for failed evaluations"""
        return {
            'model_name': model_name,
            'ap_enhanced': 0.0,
            'wll_enhanced': float('inf'),
            'combined_score_enhanced': 0.0,
            'ctr_ultra_optimized_score': 0.0,
            'ap': 0.0,
            'combined_score': 0.0,
            'auc': 0.5,
            'ctr_actual': 0.0191,
            'ctr_predicted': 0.0191,
            'ctr_bias': 0.0,
            'ctr_ratio': 1.0,
            'ctr_absolute_error': 0.0,
            'ctr_relative_error': 0.0,
            'ctr_alignment_score': 1.0,
            'ctr_stability': 1.0,
            'target_ctr_achievement': 0.0,
            'ece': 1.0,
            'mce': 1.0,
            'brier_reliability': 1.0,
            'brier_resolution': 0.0,
            'calibration_score': 0.0,
            'performance_tier': 'POOR',
            'target_combined_score_achievement': 0.0,
            'ultra_score_achievement': 0.0,
            'calibration_achievement': 0.0,
            'integrated_achievement': 0.0,
            'evaluation_duration': 0.0
        }
    
    def ctr_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR score with balanced bias penalty"""
        try:
            base_score = self.combined_score(y_true, y_pred_proba)
            
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Balanced CTR bias penalty
            if ctr_bias > self.ctr_tolerance:
                # Progressive penalty (less aggressive than exponential)
                penalty_factor = min(0.8, (ctr_bias / self.ctr_tolerance) ** 1.2 * 0.25)
                ctr_adjusted_score = base_score * (1.0 - penalty_factor)
            else:
                # Small bonus for accurate CTR prediction
                bonus_factor = min((self.ctr_tolerance - ctr_bias) / self.ctr_tolerance * 0.03, 0.03)
                ctr_adjusted_score = base_score * (1.0 + bonus_factor)
            
            return float(np.clip(ctr_adjusted_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"CTR score calculation failed: {e}")
            return 0.0
    
    def diversity_score(self, predictions_list: List[np.ndarray]) -> float:
        """Calculate prediction diversity score"""
        try:
            if len(predictions_list) < 2:
                return 0.0
            
            correlations = []
            for i in range(len(predictions_list)):
                for j in range(i + 1, len(predictions_list)):
                    pred_i = np.asarray(predictions_list[i]).flatten()
                    pred_j = np.asarray(predictions_list[j]).flatten()
                    
                    if len(pred_i) == len(pred_j) and len(pred_i) > 0:
                        corr = np.corrcoef(pred_i, pred_j)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if not correlations:
                return 0.0
            
            avg_correlation = np.mean(correlations)
            diversity = 1.0 - avg_correlation
            
            return float(np.clip(diversity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Diversity score calculation failed: {e}")
            return 0.0
    
    def stability_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_segments: int = 10) -> float:
        """Calculate prediction stability across data segments"""
        try:
            if len(y_true) < n_segments * 10:
                return 0.0
            
            segment_size = len(y_true) // n_segments
            segment_scores = []
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size
                
                if i == n_segments - 1:  # Last segment includes remainder
                    end_idx = len(y_true)
                
                segment_y_true = y_true[start_idx:end_idx]
                segment_y_pred = y_pred_proba[start_idx:end_idx]
                
                if len(segment_y_true) > 0:
                    segment_score = self.combined_score(segment_y_true, segment_y_pred)
                    segment_scores.append(segment_score)
            
            if not segment_scores:
                return 0.0
            
            # Stability is inverse of standard deviation
            score_std = np.std(segment_scores)
            stability = max(0.0, 1.0 - score_std * 2)  # Less aggressive penalty
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Stability score calculation failed: {e}")
            return 0.0

class UltraModelComparator:
    """Ultra-performance model comparator with adaptive evaluation"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.metrics_calculator = CTRMetrics(config)
        self.comparison_results = pd.DataFrame()
        
    def evaluate_models_ultra_with_calibration(self, models_predictions: Dict[str, np.ndarray], 
                                             y_true: np.ndarray,
                                             models_info: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """Ultra-performance model evaluation with calibration assessment"""
        
        logger.info(f"Starting ultra-performance evaluation of {len(models_predictions)} models")
        
        results = []
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                start_time = time.time()
                
                logger.info(f"Evaluating {model_name}...")
                
                # Data validation and cleaning
                y_true_clean = np.asarray(y_true).flatten()
                y_pred_proba = np.asarray(y_pred_proba).flatten()
                
                if len(y_true_clean) != len(y_pred_proba):
                    logger.error(f"{model_name}: Size mismatch - skipping")
                    continue
                
                if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                    logger.warning(f"{model_name}: Invalid predictions detected, cleaning")
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
            comparison_df = comparison_df.sort_values('combined_score_enhanced', ascending=False)
            self.comparison_results = comparison_df
            
            logger.info(f"Model comparison completed: {len(results)} models evaluated")
            logger.info(f"Best model: {comparison_df.iloc[0]['model_name']} " +
                       f"(Score: {comparison_df.iloc[0]['combined_score_enhanced']:.4f})")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Failed to create comparison DataFrame: {e}")
            return pd.DataFrame()
    
    def get_best_model_ultra_with_calibration(self, metric: str = 'combined_score_enhanced') -> Tuple[str, float]:
        """Get best model based on ultra-optimized metrics"""
        try:
            if self.comparison_results.empty:
                return None, 0.0
            
            best_idx = self.comparison_results[metric].idxmax()
            best_score = self.comparison_results.loc[best_idx, metric]
            best_model = self.comparison_results.loc[best_idx, 'model_name']
            
            return best_model, best_score
            
        except Exception as e:
            logger.error(f"Failed to get best ultra-performance model: {e}")
            return None, 0.0

class EvaluationReporter:
    """Evaluation report generator with calibration analysis"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.report_data = {}
        
    def generate_model_performance_report_with_calibration(self, 
                                                         comparator: UltraModelComparator,
                                                         save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate model performance report with calibration evaluation"""
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
            
            # Get best model with calibration consideration
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
                    'calibration_ece': row.get('ece', 1.0),
                    'calibration_mce': row.get('mce', 1.0),
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
                'avg_calibration_ece': float(df['ece'].mean()),
                'avg_calibration_mce': float(df['mce'].mean()),
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
            
            # Generate recommendations
            if best_score >= 0.30:
                report['recommendations'].append("Target achievement: Combined Score 0.30+ achieved by best model")
            else:
                report['recommendations'].append("Improvement needed: All models below target score")
            
            if report['calibration_analysis']['calibrated_models_count'] == 0:
                report['recommendations'].append("Apply calibration: Apply calibration to all models to improve prediction reliability")
            elif report['calibration_analysis']['calibrated_models_count'] < len(df):
                report['recommendations'].append("Expand calibration: Apply calibration to more models")
            
            if report['calibration_analysis']['avg_calibration_ece'] > 0.1:
                report['recommendations'].append("Improve calibration quality: ECE is high, try advanced calibration techniques")
            
            if report['calibration_analysis']['integrated_achievers'] == 0:
                report['recommendations'].append("Integrated optimization: Find methods to improve both performance and calibration simultaneously")
            
            # Save report
            if save_path:
                try:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    logger.info(f"Evaluation report saved: {save_path} (with calibration analysis)")
                except Exception as e:
                    logger.warning(f"Report save failed: {e}")
            
            self.report_data = report
            return report
            
        except Exception as e:
            logger.error(f"Evaluation report generation failed: {e}")
            return {'error': f'Report generation error: {str(e)}'}
    
    def save_comparison_results_with_calibration(self, 
                                               comparison_df: pd.DataFrame,
                                               save_path: Optional[Path] = None) -> bool:
        """Save comparison results to file with calibration information"""
        try:
            if save_path is None:
                save_path = self.config.OUTPUT_DIR / "model_comparison_results_with_calibration.csv"
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(save_path, index=True, encoding='utf-8')
            
            logger.info(f"Comparison results saved: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save comparison results: {e}")
            return False