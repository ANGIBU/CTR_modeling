# analysis.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import warnings
from pathlib import Path
import json
warnings.filterwarnings('ignore')

try:
    from sklearn.metrics import (
        roc_curve, precision_recall_curve, auc, calibration_curve,
        confusion_matrix, classification_report, roc_auc_score,
        average_precision_score, log_loss, brier_score_loss
    )
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available for analysis")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available for statistical analysis")

from evaluation import CTRMetrics
from config import Config

logger = logging.getLogger(__name__)

class CTRPerformanceAnalyzer:
    """Comprehensive CTR model performance analyzer"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.metrics_calculator = CTRMetrics(config)
        self.analysis_results = {}
        
        # CTR specific thresholds
        self.target_ctr = 0.0191
        self.ctr_tolerance = 0.0002
        self.performance_thresholds = {
            'excellent': 0.34,
            'good': 0.30,
            'fair': 0.25,
            'poor': 0.20
        }
    
    def full_performance_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                model_name: str = "Model", quick_mode: bool = False) -> Dict[str, Any]:
        """Complete performance analysis for CTR model"""
        logger.info(f"Starting full performance analysis for {model_name}")
        
        start_time = time.time()
        
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return self._get_empty_analysis(model_name)
            
            analysis = {
                'model_name': model_name,
                'analysis_type': 'quick' if quick_mode else 'full',
                'data_size': len(y_true),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Core metrics
            analysis['core_metrics'] = self._calculate_core_metrics(y_true, y_pred_proba)
            
            # CTR specific analysis
            analysis['ctr_analysis'] = self._analyze_ctr_performance(y_true, y_pred_proba)
            
            # Prediction distribution analysis
            analysis['prediction_distribution'] = self._analyze_prediction_distribution(y_pred_proba)
            
            # Calibration analysis
            if not quick_mode and len(y_true) > 100:
                analysis['calibration_analysis'] = self._analyze_calibration(y_true, y_pred_proba)
            else:
                analysis['calibration_analysis'] = self._basic_calibration_check(y_true, y_pred_proba)
            
            # Performance segmentation
            if not quick_mode and len(y_true) > 1000:
                analysis['segment_analysis'] = self._analyze_performance_segments(y_true, y_pred_proba)
            else:
                analysis['segment_analysis'] = {'status': 'insufficient_data'}
            
            # Business impact analysis
            analysis['business_impact'] = self._calculate_business_impact(y_true, y_pred_proba)
            
            # Model reliability
            analysis['reliability_metrics'] = self._calculate_reliability_metrics(y_true, y_pred_proba)
            
            # Overall assessment
            analysis['overall_assessment'] = self._generate_overall_assessment(analysis)
            
            analysis['analysis_duration'] = time.time() - start_time
            
            self.analysis_results[model_name] = analysis
            
            logger.info(f"Performance analysis completed for {model_name} in {analysis['analysis_duration']:.2f}s")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed for {model_name}: {e}")
            return self._get_empty_analysis(model_name, error=str(e))
    
    def _calculate_core_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate core CTR metrics"""
        try:
            core_metrics = {}
            
            # Basic classification metrics
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # CTR specific metrics
            core_metrics['ap'] = self.metrics_calculator.average_precision(y_true, y_pred_proba)
            core_metrics['wll'] = self.metrics_calculator.weighted_log_loss(y_true, y_pred_proba)
            core_metrics['combined_score'] = self.metrics_calculator.combined_score(y_true, y_pred_proba)
            core_metrics['ctr_score'] = self.metrics_calculator.ctr_score(y_true, y_pred_proba)
            
            if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
                core_metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                core_metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                core_metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                
                # Classification metrics at 0.5 threshold
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                core_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                core_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                core_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                core_metrics['f1_score'] = 2 * core_metrics['precision'] * core_metrics['recall'] / (core_metrics['precision'] + core_metrics['recall']) if (core_metrics['precision'] + core_metrics['recall']) > 0 else 0.0
            else:
                core_metrics.update({
                    'auc': 0.5, 'log_loss': 1.0, 'brier_score': 1.0,
                    'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1_score': 0.0
                })
            
            return core_metrics
            
        except Exception as e:
            logger.error(f"Core metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_ctr_performance(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze CTR specific performance"""
        try:
            ctr_analysis = {}
            
            # Basic CTR statistics
            actual_ctr = y_true.mean()
            predicted_ctr = y_pred_proba.mean()
            ctr_bias = predicted_ctr - actual_ctr
            ctr_absolute_error = abs(ctr_bias)
            
            ctr_analysis['actual_ctr'] = float(actual_ctr)
            ctr_analysis['predicted_ctr'] = float(predicted_ctr)
            ctr_analysis['ctr_bias'] = float(ctr_bias)
            ctr_analysis['ctr_absolute_error'] = float(ctr_absolute_error)
            ctr_analysis['ctr_relative_error'] = float(ctr_absolute_error / actual_ctr) if actual_ctr > 0 else float('inf')
            
            # Target alignment
            ctr_analysis['target_ctr'] = self.target_ctr
            ctr_analysis['target_alignment'] = {
                'actual_vs_target_error': abs(actual_ctr - self.target_ctr),
                'predicted_vs_target_error': abs(predicted_ctr - self.target_ctr),
                'within_tolerance': ctr_absolute_error <= self.ctr_tolerance,
                'tolerance_level': self.ctr_tolerance
            }
            
            # CTR distribution analysis
            ctr_analysis['ctr_distribution'] = {
                'prediction_std': float(np.std(y_pred_proba)),
                'prediction_min': float(np.min(y_pred_proba)),
                'prediction_max': float(np.max(y_pred_proba)),
                'prediction_range': float(np.max(y_pred_proba) - np.min(y_pred_proba)),
                'prediction_median': float(np.median(y_pred_proba)),
                'prediction_q25': float(np.percentile(y_pred_proba, 25)),
                'prediction_q75': float(np.percentile(y_pred_proba, 75))
            }
            
            # CTR quality assessment
            if ctr_absolute_error <= 0.0001:
                ctr_quality = 'excellent'
            elif ctr_absolute_error <= 0.0002:
                ctr_quality = 'good'
            elif ctr_absolute_error <= 0.0005:
                ctr_quality = 'fair'
            else:
                ctr_quality = 'poor'
            
            ctr_analysis['ctr_quality'] = ctr_quality
            
            return ctr_analysis
            
        except Exception as e:
            logger.error(f"CTR analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_prediction_distribution(self, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction distribution patterns"""
        try:
            dist_analysis = {}
            
            # Basic distribution statistics
            dist_analysis['statistics'] = {
                'mean': float(np.mean(y_pred_proba)),
                'std': float(np.std(y_pred_proba)),
                'min': float(np.min(y_pred_proba)),
                'max': float(np.max(y_pred_proba)),
                'median': float(np.median(y_pred_proba)),
                'skewness': float(stats.skew(y_pred_proba)) if SCIPY_AVAILABLE else 0.0,
                'kurtosis': float(stats.kurtosis(y_pred_proba)) if SCIPY_AVAILABLE else 0.0
            }
            
            # Percentile analysis
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            dist_analysis['percentiles'] = {
                f'p{p}': float(np.percentile(y_pred_proba, p)) for p in percentiles
            }
            
            # Histogram data
            hist, bin_edges = np.histogram(y_pred_proba, bins=20)
            dist_analysis['histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
            
            # Prediction concentration analysis
            unique_predictions = len(np.unique(y_pred_proba))
            total_predictions = len(y_pred_proba)
            
            dist_analysis['concentration'] = {
                'unique_predictions': unique_predictions,
                'total_predictions': total_predictions,
                'diversity_ratio': unique_predictions / total_predictions,
                'concentration_level': 'low' if unique_predictions / total_predictions > 0.8 else 'medium' if unique_predictions / total_predictions > 0.5 else 'high'
            }
            
            # Outlier detection
            q1 = np.percentile(y_pred_proba, 25)
            q3 = np.percentile(y_pred_proba, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.sum((y_pred_proba < lower_bound) | (y_pred_proba > upper_bound))
            
            dist_analysis['outliers'] = {
                'count': int(outliers),
                'percentage': float(outliers / len(y_pred_proba) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            return dist_analysis
            
        except Exception as e:
            logger.error(f"Prediction distribution analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Detailed calibration analysis"""
        try:
            if not SKLEARN_AVAILABLE:
                return self._basic_calibration_check(y_true, y_pred_proba)
            
            calibration_analysis = {}
            
            # Reliability diagram data
            try:
                prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
                
                calibration_analysis['reliability_diagram'] = {
                    'prob_true': prob_true.tolist(),
                    'prob_pred': prob_pred.tolist(),
                    'perfect_calibration': [prob_pred[i] for i in range(len(prob_pred))]
                }
                
                # Calibration error calculation
                calibration_error = np.mean(np.abs(prob_true - prob_pred))
                calibration_analysis['calibration_error'] = float(calibration_error)
                
                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_true[in_bin].mean()
                        avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                calibration_analysis['expected_calibration_error'] = float(ece)
                
                # Calibration quality assessment
                if ece < 0.01:
                    calibration_quality = 'excellent'
                elif ece < 0.02:
                    calibration_quality = 'good'
                elif ece < 0.05:
                    calibration_quality = 'fair'
                else:
                    calibration_quality = 'poor'
                
                calibration_analysis['calibration_quality'] = calibration_quality
                
            except Exception as e:
                logger.warning(f"Detailed calibration analysis failed: {e}")
                return self._basic_calibration_check(y_true, y_pred_proba)
            
            return calibration_analysis
            
        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")
            return self._basic_calibration_check(y_true, y_pred_proba)
    
    def _basic_calibration_check(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Basic calibration check for small datasets"""
        try:
            # Simple binning approach
            n_bins = min(5, len(y_true) // 10) if len(y_true) > 50 else 3
            
            if n_bins < 2:
                return {
                    'status': 'insufficient_data',
                    'calibration_quality': 'unknown',
                    'calibration_error': float('nan')
                }
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = []
            bin_accuracies = []
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
                if i == n_bins - 1:  # Include upper boundary in last bin
                    in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba <= bin_upper)
                
                if np.sum(in_bin) > 0:
                    bin_center = y_pred_proba[in_bin].mean()
                    bin_accuracy = y_true[in_bin].mean()
                    
                    bin_centers.append(bin_center)
                    bin_accuracies.append(bin_accuracy)
            
            if len(bin_centers) >= 2:
                calibration_error = np.mean([abs(center - accuracy) for center, accuracy in zip(bin_centers, bin_accuracies)])
                
                calibration_quality = 'good' if calibration_error < 0.02 else 'fair' if calibration_error < 0.05 else 'poor'
                
                return {
                    'status': 'basic_analysis',
                    'n_bins': n_bins,
                    'bin_centers': bin_centers,
                    'bin_accuracies': bin_accuracies,
                    'calibration_error': float(calibration_error),
                    'calibration_quality': calibration_quality
                }
            else:
                return {
                    'status': 'insufficient_bins',
                    'calibration_quality': 'unknown',
                    'calibration_error': float('nan')
                }
                
        except Exception as e:
            logger.error(f"Basic calibration check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'calibration_quality': 'unknown'
            }
    
    def _analyze_performance_segments(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze performance across different data segments"""
        try:
            segment_analysis = {}
            
            # Prediction score segments
            score_segments = ['0.0-0.01', '0.01-0.02', '0.02-0.05', '0.05-0.1', '0.1+']
            score_thresholds = [0.0, 0.01, 0.02, 0.05, 0.1, 1.0]
            
            segment_performance = {}
            
            for i, segment_name in enumerate(score_segments):
                lower = score_thresholds[i]
                upper = score_thresholds[i + 1]
                
                in_segment = (y_pred_proba >= lower) & (y_pred_proba < upper)
                if i == len(score_segments) - 1:  # Last segment includes upper bound
                    in_segment = (y_pred_proba >= lower) & (y_pred_proba <= upper)
                
                segment_size = np.sum(in_segment)
                
                if segment_size > 0:
                    segment_true = y_true[in_segment]
                    segment_pred = y_pred_proba[in_segment]
                    
                    segment_performance[segment_name] = {
                        'size': int(segment_size),
                        'percentage': float(segment_size / len(y_true) * 100),
                        'actual_ctr': float(segment_true.mean()),
                        'predicted_ctr': float(segment_pred.mean()),
                        'ctr_bias': float(segment_pred.mean() - segment_true.mean())
                    }
                    
                    # Calculate segment-specific metrics if enough data
                    if segment_size > 10 and len(np.unique(segment_true)) > 1:
                        try:
                            segment_ap = self.metrics_calculator.average_precision(segment_true, segment_pred)
                            segment_performance[segment_name]['ap'] = float(segment_ap)
                        except Exception:
                            segment_performance[segment_name]['ap'] = float('nan')
            
            segment_analysis['score_segments'] = segment_performance
            
            # Decile analysis
            if len(y_pred_proba) >= 100:
                decile_analysis = {}
                
                # Sort by prediction score
                sorted_indices = np.argsort(y_pred_proba)[::-1]
                decile_size = len(y_pred_proba) // 10
                
                for decile in range(10):
                    start_idx = decile * decile_size
                    end_idx = (decile + 1) * decile_size if decile < 9 else len(y_pred_proba)
                    
                    decile_indices = sorted_indices[start_idx:end_idx]
                    decile_true = y_true[decile_indices]
                    decile_pred = y_pred_proba[decile_indices]
                    
                    decile_analysis[f'decile_{decile + 1}'] = {
                        'size': len(decile_indices),
                        'actual_ctr': float(decile_true.mean()),
                        'predicted_ctr': float(decile_pred.mean()),
                        'min_score': float(decile_pred.min()),
                        'max_score': float(decile_pred.max()),
                        'lift': float(decile_true.mean() / y_true.mean()) if y_true.mean() > 0 else float('inf')
                    }
                
                segment_analysis['decile_analysis'] = decile_analysis
            
            return segment_analysis
            
        except Exception as e:
            logger.error(f"Performance segment analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_business_impact(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate business impact metrics"""
        try:
            business_impact = {}
            
            # Confusion matrix at different thresholds
            thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
            threshold_analysis = {}
            
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                
                if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_thresh)) > 1:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    threshold_analysis[f'threshold_{threshold}'] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'specificity': float(specificity),
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'true_negatives': int(tn),
                        'false_negatives': int(fn)
                    }
            
            business_impact['threshold_analysis'] = threshold_analysis
            
            # Lift calculation
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            sorted_true = y_true[sorted_indices]
            
            # Calculate lift for top percentiles
            percentiles = [1, 5, 10, 20, 50]
            lift_analysis = {}
            
            baseline_ctr = y_true.mean()
            
            for percentile in percentiles:
                top_n = int(len(y_true) * percentile / 100)
                if top_n > 0:
                    top_ctr = sorted_true[:top_n].mean()
                    lift = top_ctr / baseline_ctr if baseline_ctr > 0 else float('inf')
                    
                    lift_analysis[f'top_{percentile}%'] = {
                        'ctr': float(top_ctr),
                        'lift': float(lift),
                        'sample_size': top_n
                    }
            
            business_impact['lift_analysis'] = lift_analysis
            
            # Revenue impact estimation (assuming unit values)
            business_impact['revenue_estimation'] = {
                'baseline_revenue': float(baseline_ctr * len(y_true)),
                'predicted_revenue': float(y_pred_proba.sum()),
                'revenue_difference': float(y_pred_proba.sum() - baseline_ctr * len(y_true)),
                'revenue_accuracy': float(abs(y_pred_proba.sum() - y_true.sum()) / y_true.sum()) if y_true.sum() > 0 else float('inf')
            }
            
            return business_impact
            
        except Exception as e:
            logger.error(f"Business impact calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_reliability_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate model reliability metrics"""
        try:
            reliability = {}
            
            # Prediction consistency
            prediction_std = np.std(y_pred_proba)
            prediction_range = np.max(y_pred_proba) - np.min(y_pred_proba)
            
            reliability['prediction_consistency'] = {
                'standard_deviation': float(prediction_std),
                'coefficient_of_variation': float(prediction_std / np.mean(y_pred_proba)) if np.mean(y_pred_proba) > 0 else float('inf'),
                'prediction_range': float(prediction_range),
                'interquartile_range': float(np.percentile(y_pred_proba, 75) - np.percentile(y_pred_proba, 25))
            }
            
            # Stability across data segments
            if len(y_true) > 100:
                n_segments = 5
                segment_size = len(y_true) // n_segments
                segment_performances = []
                
                for i in range(n_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(y_true)
                    
                    segment_true = y_true[start_idx:end_idx]
                    segment_pred = y_pred_proba[start_idx:end_idx]
                    
                    if len(segment_true) > 0 and len(np.unique(segment_true)) > 1:
                        try:
                            segment_score = self.metrics_calculator.combined_score(segment_true, segment_pred)
                            segment_performances.append(segment_score)
                        except Exception:
                            continue
                
                if len(segment_performances) > 1:
                    stability_score = 1.0 - (np.std(segment_performances) / np.mean(segment_performances))
                    reliability['stability'] = {
                        'segment_scores': segment_performances,
                        'score_std': float(np.std(segment_performances)),
                        'score_mean': float(np.mean(segment_performances)),
                        'stability_score': float(max(0, stability_score))
                    }
            
            # Confidence calibration
            confidence_levels = [0.8, 0.9, 0.95, 0.99]
            confidence_analysis = {}
            
            for confidence_level in confidence_levels:
                threshold = 1 - confidence_level
                high_confidence = y_pred_proba >= (1 - threshold/2)
                low_confidence = y_pred_proba <= threshold/2
                
                if np.sum(high_confidence) > 0:
                    high_conf_accuracy = y_true[high_confidence].mean()
                    confidence_analysis[f'high_confidence_{confidence_level}'] = {
                        'count': int(np.sum(high_confidence)),
                        'accuracy': float(high_conf_accuracy),
                        'expected_accuracy': confidence_level
                    }
                
                if np.sum(low_confidence) > 0:
                    low_conf_accuracy = y_true[low_confidence].mean()
                    confidence_analysis[f'low_confidence_{confidence_level}'] = {
                        'count': int(np.sum(low_confidence)),
                        'accuracy': float(low_conf_accuracy),
                        'expected_accuracy': 1 - confidence_level
                    }
            
            reliability['confidence_analysis'] = confidence_analysis
            
            return reliability
            
        except Exception as e:
            logger.error(f"Reliability metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _generate_overall_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall model assessment"""
        try:
            assessment = {}
            
            # Extract key metrics
            combined_score = analysis['core_metrics'].get('combined_score', 0.0)
            ctr_absolute_error = analysis['ctr_analysis'].get('ctr_absolute_error', 1.0)
            calibration_quality = analysis.get('calibration_analysis', {}).get('calibration_quality', 'unknown')
            
            # Performance tier
            if combined_score >= self.performance_thresholds['excellent']:
                performance_tier = 'excellent'
            elif combined_score >= self.performance_thresholds['good']:
                performance_tier = 'good'
            elif combined_score >= self.performance_thresholds['fair']:
                performance_tier = 'fair'
            else:
                performance_tier = 'poor'
            
            assessment['performance_tier'] = performance_tier
            assessment['combined_score'] = float(combined_score)
            assessment['target_achievement'] = combined_score >= self.performance_thresholds['excellent']
            
            # CTR quality assessment
            ctr_quality = analysis['ctr_analysis'].get('ctr_quality', 'unknown')
            assessment['ctr_quality'] = ctr_quality
            assessment['ctr_bias_acceptable'] = ctr_absolute_error <= self.ctr_tolerance
            
            # Overall model readiness
            readiness_criteria = {
                'performance_acceptable': combined_score >= self.performance_thresholds['fair'],
                'ctr_bias_acceptable': ctr_absolute_error <= self.ctr_tolerance,
                'calibration_acceptable': calibration_quality in ['excellent', 'good'],
                'no_critical_errors': 'error' not in analysis['core_metrics']
            }
            
            readiness_score = sum(readiness_criteria.values()) / len(readiness_criteria)
            assessment['readiness_criteria'] = readiness_criteria
            assessment['readiness_score'] = float(readiness_score)
            
            if readiness_score >= 0.75:
                assessment['deployment_recommendation'] = 'ready'
            elif readiness_score >= 0.5:
                assessment['deployment_recommendation'] = 'needs_improvement'
            else:
                assessment['deployment_recommendation'] = 'not_ready'
            
            # Key recommendations
            recommendations = []
            
            if combined_score < self.performance_thresholds['good']:
                recommendations.append("Improve model performance - consider ensemble or hyperparameter tuning")
            
            if ctr_absolute_error > self.ctr_tolerance:
                recommendations.append("Reduce CTR bias - apply calibration or adjust prediction thresholds")
            
            if calibration_quality in ['poor', 'unknown']:
                recommendations.append("Improve calibration - apply probability calibration techniques")
            
            assessment['recommendations'] = recommendations
            
            return assessment
            
        except Exception as e:
            logger.error(f"Overall assessment generation failed: {e}")
            return {
                'performance_tier': 'unknown',
                'deployment_recommendation': 'not_ready',
                'error': str(e)
            }
    
    def _get_empty_analysis(self, model_name: str, error: str = None) -> Dict[str, Any]:
        """Return empty analysis structure for error cases"""
        return {
            'model_name': model_name,
            'analysis_type': 'error',
            'data_size': 0,
            'error': error,
            'core_metrics': {},
            'ctr_analysis': {},
            'prediction_distribution': {},
            'calibration_analysis': {},
            'segment_analysis': {},
            'business_impact': {},
            'reliability_metrics': {},
            'overall_assessment': {
                'performance_tier': 'unknown',
                'deployment_recommendation': 'not_ready'
            }
        }
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model analysis results"""
        logger.info(f"Comparing {len(model_results)} models")
        
        try:
            comparison = {
                'models_compared': list(model_results.keys()),
                'comparison_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'best_models': {},
                'performance_ranking': [],
                'detailed_comparison': {}
            }
            
            # Extract key metrics for comparison
            metrics_for_comparison = [
                'combined_score', 'ctr_absolute_error', 'ap', 'auc', 'calibration_quality'
            ]
            
            model_scores = {}
            
            for model_name, analysis in model_results.items():
                score_dict = {}
                
                # Combined score
                score_dict['combined_score'] = analysis.get('core_metrics', {}).get('combined_score', 0.0)
                
                # CTR error (lower is better)
                ctr_error = analysis.get('ctr_analysis', {}).get('ctr_absolute_error', 1.0)
                score_dict['ctr_score'] = max(0, 1 - (ctr_error / 0.01))  # Normalize to 0-1
                
                # AP score
                score_dict['ap'] = analysis.get('core_metrics', {}).get('ap', 0.0)
                
                # AUC score
                score_dict['auc'] = analysis.get('core_metrics', {}).get('auc', 0.5)
                
                # Calibration quality score
                cal_quality = analysis.get('calibration_analysis', {}).get('calibration_quality', 'poor')
                cal_score = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4, 'unknown': 0.0}.get(cal_quality, 0.0)
                score_dict['calibration_score'] = cal_score
                
                # Overall composite score
                composite_score = (
                    0.4 * score_dict['combined_score'] +
                    0.2 * score_dict['ctr_score'] +
                    0.2 * score_dict['ap'] +
                    0.1 * (score_dict['auc'] - 0.5) * 2 +  # Normalize AUC to 0-1
                    0.1 * score_dict['calibration_score']
                )
                
                score_dict['composite_score'] = composite_score
                model_scores[model_name] = score_dict
            
            # Rank models
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
            comparison['performance_ranking'] = [{'model': name, 'composite_score': scores['composite_score']} for name, scores in ranked_models]
            
            # Identify best models for each metric
            for metric in ['combined_score', 'ctr_score', 'ap', 'auc', 'calibration_score']:
                best_model = max(model_scores.items(), key=lambda x: x[1][metric])
                comparison['best_models'][metric] = {
                    'model': best_model[0],
                    'score': best_model[1][metric]
                }
            
            # Detailed comparison matrix
            comparison_matrix = {}
            for model_name in model_results.keys():
                comparison_matrix[model_name] = model_scores[model_name]
            
            comparison['detailed_comparison'] = comparison_matrix
            
            return comparison
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {
                'error': str(e),
                'models_compared': list(model_results.keys()) if model_results else []
            }
    
    def save_analysis_report(self, analysis: Dict[str, Any], output_path: str = None) -> str:
        """Save detailed analysis report"""
        try:
            if output_path is None:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                model_name = analysis.get('model_name', 'unknown').replace(' ', '_')
                output_path = f"analysis_report_{model_name}_{timestamp}.json"
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Analysis report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
            return ""

def analyze_model_performance(y_true, y_pred_proba, model_name="Model", quick_mode=False):
    """Standalone function for model performance analysis"""
    analyzer = CTRPerformanceAnalyzer()
    return analyzer.full_performance_analysis(y_true, y_pred_proba, model_name, quick_mode)

def compare_model_performances(model_results):
    """Standalone function for comparing multiple models"""
    analyzer = CTRPerformanceAnalyzer()
    return analyzer.compare_models(model_results)