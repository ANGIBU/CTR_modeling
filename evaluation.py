# evaluation.py

import logging
import time
import gc
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization functionality will be disabled.")

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
                return log_loss(y_true, y_pred_proba)
            
            pos_loss = -np.mean(np.log(y_pred_proba[pos_mask])) * self.pos_weight
            neg_loss = -np.mean(np.log(1 - y_pred_proba[neg_mask])) * self.neg_weight
            
            total_weight = self.pos_weight + self.neg_weight
            weighted_loss = (pos_loss + neg_loss) / total_weight
            
            return float(weighted_loss)
            
        except Exception as e:
            logger.error(f"WLL calculation failed: {e}")
            return 100.0
    
    def ctr_calibration_score_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                     n_bins: int = 10) -> Dict[str, float]:
        """Enhanced CTR calibration evaluation"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return {'calibration_score': 0.0, 'expected_calibration_error': 1.0, 
                       'reliability_score': 0.0, 'calibration_slope': 0.0}
            
            # Bin creation and validation
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_boundaries[-1] = 1.0001  # Ensure last bin includes 1.0
            
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_errors = []
            bin_scores = []
            
            total_samples = len(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(y_true[in_bin])
                    avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                    
                    bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                    calibration_errors.append(bin_error * prop_in_bin)
                    
                    # Bin quality score (lower error = higher score)
                    bin_score = max(0, 1 - bin_error)
                    bin_scores.append(bin_score * prop_in_bin)
            
            expected_calibration_error = np.sum(calibration_errors)
            reliability_score = np.sum(bin_scores)
            
            # CTR bias assessment
            ctr_bias = abs(predicted_ctr - actual_ctr)
            ctr_bias_penalty = min(ctr_bias / self.ctr_tolerance, 1.0)
            
            # Calibration slope calculation
            try:
                if SCIPY_AVAILABLE:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred_proba, y_true)
                    calibration_slope = abs(slope - 1.0)  # Perfect calibration slope = 1
                else:
                    calibration_slope = ctr_bias_penalty
            except:
                calibration_slope = ctr_bias_penalty
            
            # Integrated calibration score
            base_calibration_score = 1 - expected_calibration_error
            bias_adjusted_score = base_calibration_score * (1 - ctr_bias_penalty)
            reliability_adjusted_score = bias_adjusted_score * reliability_score
            
            calibration_score = max(0, min(1, reliability_adjusted_score))
            
            return {
                'calibration_score': float(calibration_score),
                'expected_calibration_error': float(expected_calibration_error),
                'reliability_score': float(reliability_score),
                'calibration_slope': float(calibration_slope),
                'ctr_bias': float(ctr_bias),
                'predicted_ctr': float(predicted_ctr),
                'actual_ctr': float(actual_ctr)
            }
            
        except Exception as e:
            logger.error(f"CTR calibration evaluation failed: {e}")
            return {'calibration_score': 0.0, 'expected_calibration_error': 1.0, 
                   'reliability_score': 0.0, 'calibration_slope': 1.0}
    
    def ctr_ultra_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR ultra optimized combined score calculation"""
        try:
            # Basic metrics
            ap = self.average_precision_enhanced(y_true, y_pred_proba)
            wll = self.weighted_log_loss_ctr(y_true, y_pred_proba)
            
            # Avoid extreme WLL values
            wll = min(wll, 10.0)
            
            # Calibration assessment
            calibration_metrics = self.ctr_calibration_score_enhanced(y_true, y_pred_proba)
            calibration_score = calibration_metrics['calibration_score']
            
            # CTR specific adjustments
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Bias penalty
            if ctr_bias <= self.ctr_tolerance:
                bias_penalty = 0.0
            else:
                bias_penalty = min((ctr_bias - self.ctr_tolerance) * self.bias_penalty_weight, 0.2)
            
            # Base combined score
            base_score = (ap * self.ap_weight) - (wll * self.wll_weight)
            
            # Apply bias penalty
            bias_adjusted_score = base_score - bias_penalty
            
            # Apply calibration enhancement
            calibration_enhanced_score = bias_adjusted_score + (calibration_score * self.calibration_weight)
            
            # Distribution quality bonus
            pred_entropy = self._calculate_entropy_enhanced(y_pred_proba)
            entropy_bonus = min(pred_entropy * 0.05, 0.03)
            
            final_score = calibration_enhanced_score + entropy_bonus
            
            return max(0, float(final_score))
            
        except Exception as e:
            logger.error(f"Ultra optimized score calculation failed: {e}")
            return 0.0
    
    def comprehensive_evaluation_ultra_with_calibration(self, 
                                                      y_true: np.ndarray, 
                                                      y_pred_proba: np.ndarray,
                                                      model_name: str = "Unknown") -> Dict[str, Any]:
        """Comprehensive evaluation including calibration assessment"""
        
        start_time = time.time()
        
        y_true = np.asarray(y_true).flatten()
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        
        logger.info(f"Comprehensive evaluation started for {model_name} (including calibration)")
        
        if len(y_true) != len(y_pred_proba):
            logger.error(f"Size mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
            return {'error': 'Size mismatch between true and predicted values'}
        
        if len(y_true) == 0:
            logger.error("Empty arrays provided")
            return {'error': 'Empty arrays provided'}
        
        metrics = {}
        
        try:
            # Basic evaluation metrics
            metrics['model_name'] = model_name
            metrics['sample_count'] = len(y_true)
            metrics['positive_ratio'] = float(np.mean(y_true))
            metrics['predicted_ctr'] = float(np.mean(y_pred_proba))
            metrics['actual_ctr'] = metrics['positive_ratio']
            
            # Core performance metrics
            try:
                metrics['average_precision'] = self.average_precision_enhanced(y_true, y_pred_proba)
                metrics['weighted_log_loss'] = self.weighted_log_loss_ctr(y_true, y_pred_proba)
                metrics['combined_score_enhanced'] = (metrics['average_precision'] * self.ap_weight) - (metrics['weighted_log_loss'] * self.wll_weight)
                metrics['ctr_ultra_optimized_score'] = self.ctr_ultra_optimized_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Core metrics calculation failed: {e}")
                metrics.update({
                    'average_precision': 0.0, 'weighted_log_loss': 100.0, 
                    'combined_score_enhanced': -100.0, 'ctr_ultra_optimized_score': 0.0
                })
            
            # Calibration evaluation
            try:
                calibration_results = self.ctr_calibration_score_enhanced(y_true, y_pred_proba)
                metrics.update(calibration_results)
            except Exception as e:
                logger.warning(f"Calibration evaluation failed: {e}")
                metrics['calibration_score'] = 0.0
                metrics['expected_calibration_error'] = 1.0
                metrics['reliability_score'] = 0.0
                metrics['calibration_slope'] = 1.0
            
            # Additional sklearn metrics
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Additional sklearn metrics calculation failed: {e}")
                metrics.update({'roc_auc': 0.5, 'log_loss': 100.0, 'brier_score': 1.0})
            
            # CTR specific metrics
            try:
                ctr_bias = abs(metrics['predicted_ctr'] - metrics['actual_ctr'])
                metrics['ctr_bias'] = ctr_bias
                metrics['ctr_bias_ratio'] = ctr_bias / max(metrics['actual_ctr'], 1e-10)
                metrics['ctr_accuracy'] = 1.0 - min(ctr_bias / self.ctr_tolerance, 1.0)
            except Exception as e:
                logger.warning(f"CTR specific metrics calculation failed: {e}")
                metrics.update({'ctr_bias': 1.0, 'ctr_bias_ratio': 50.0, 'ctr_accuracy': 0.0})
            
            # Prediction distribution analysis
            try:
                metrics['pred_mean'] = float(np.mean(y_pred_proba))
                metrics['pred_std'] = float(np.std(y_pred_proba))
                metrics['pred_min'] = float(np.min(y_pred_proba))
                metrics['pred_max'] = float(np.max(y_pred_proba))
                metrics['pred_median'] = float(np.median(y_pred_proba))
                
                pred_q25, pred_q75 = np.percentile(y_pred_proba, [25, 75])
                metrics['pred_q25'] = float(pred_q25)
                metrics['pred_q75'] = float(pred_q75)
                metrics['pred_iqr'] = float(pred_q75 - pred_q25)
                
                metrics['pred_entropy'] = self._calculate_entropy_enhanced(y_pred_proba)
                
            except Exception as e:
                logger.warning(f"Prediction distribution analysis failed: {e}")
                metrics.update({
                    'pred_mean': 0.0, 'pred_std': 0.0, 'pred_min': 0.0, 'pred_max': 1.0,
                    'pred_median': 0.0, 'pred_q25': 0.0, 'pred_q75': 1.0, 'pred_iqr': 1.0,
                    'pred_entropy': 0.0
                })
            
            # Class-wise prediction statistics
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
                
                # Class separation metrics
                if metrics['pos_mean_pred'] > metrics['neg_mean_pred']:
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
                            validated_metrics[key] = 0.0
                        else:
                            validated_metrics[key] = float(value)
                    else:
                        validated_metrics[key] = value
                except:
                    validated_metrics[key] = 0.0 if key not in ['model_name', 'performance_tier'] else str(value)
            
            logger.info(f"Evaluation completed for {model_name}: Combined Score = {validated_metrics['combined_score_enhanced']:.4f}, Calibration Score = {validated_metrics.get('calibration_score', 0.0):.4f}")
            
            return validated_metrics
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed for {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'sample_count': len(y_true),
                'evaluation_time': time.time() - start_time
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
                    logger.warning(f"{model_name}: Invalid values in predictions, applying corrections")
                    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Comprehensive evaluation
                metrics = self.metrics_calculator.comprehensive_evaluation_ultra_with_calibration(
                    y_true, y_pred_proba, model_name
                )
                
                if 'error' not in metrics:
                    # Additional model info integration
                    if models_info and model_name in models_info:
                        info = models_info[model_name]
                        metrics.update({
                            'model_type': info.get('type', 'unknown'),
                            'hyperparameters': str(info.get('params', {})),
                            'training_time': info.get('training_time', 0.0)
                        })
                    
                    metrics['comparison_time'] = time.time() - start_time
                    results.append(metrics)
                    
                    logger.info(f"{model_name} evaluation completed: Combined Score = {metrics['combined_score_enhanced']:.4f}")
                
            except Exception as e:
                logger.error(f"{model_name} evaluation failed: {e}")
                continue
        
        if not results:
            logger.error("No models were successfully evaluated")
            return pd.DataFrame()
        
        self.comparison_results = pd.DataFrame(results)
        
        # Performance analysis
        self._perform_ultra_analysis_with_calibration()
        
        logger.info(f"Model comparison completed: {len(self.comparison_results)} models evaluated")
        
        return self.comparison_results
    
    def _perform_ultra_analysis_with_calibration(self):
        """Ultra performance analysis including calibration assessment"""
        try:
            if self.comparison_results.empty:
                return
            
            analysis = {}
            
            # Performance statistics
            numeric_columns = self.comparison_results.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                analysis[f'{col}_mean'] = self.comparison_results[col].mean()
                analysis[f'{col}_std'] = self.comparison_results[col].std()
                analysis[f'{col}_min'] = self.comparison_results[col].min()
                analysis[f'{col}_max'] = self.comparison_results[col].max()
                analysis[f'{col}_median'] = self.comparison_results[col].median()
            
            # Target achievement analysis
            target_achievers = self.comparison_results[
                self.comparison_results['target_combined_score_achievement'] == 1.0
            ]
            analysis['target_achievement_rate'] = len(target_achievers) / len(self.comparison_results)
            analysis['target_achievers_count'] = len(target_achievers)
            
            # Calibration analysis
            calibrated_models = self.comparison_results[
                self.comparison_results['calibration_achievement'] == 1.0
            ]
            analysis['calibration_achievement_rate'] = len(calibrated_models) / len(self.comparison_results)
            analysis['calibrated_models_count'] = len(calibrated_models)
            
            # Integrated achievement analysis
            integrated_achievers = self.comparison_results[
                self.comparison_results['integrated_achievement'] == 1.0
            ]
            analysis['integrated_achievement_rate'] = len(integrated_achievers) / len(self.comparison_results)
            analysis['integrated_achievers_count'] = len(integrated_achievers)
            
            # Performance tier distribution
            tier_counts = self.comparison_results['performance_tier'].value_counts()
            for tier, count in tier_counts.items():
                analysis[f'tier_{tier}_count'] = count
                analysis[f'tier_{tier}_rate'] = count / len(self.comparison_results)
            
            self.performance_analysis = analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            self.performance_analysis = {}
    
    def get_best_model_ultra_with_calibration(self, 
                                            metric: str = 'ctr_ultra_optimized_score',
                                            calibration_threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """Find best ultra high-performance model - calibration consideration"""
        try:
            if self.comparison_results.empty:
                return None, 0.0
            
            if metric not in self.comparison_results.columns:
                logger.warning(f"Metric {metric} not found, using combined_score_enhanced")
                metric = 'combined_score_enhanced'
            
            # Filter models meeting calibration threshold
            calibrated_models = self.comparison_results[
                self.comparison_results['calibration_score'] >= calibration_threshold
            ]
            
            if not calibrated_models.empty:
                # Find best among calibrated models
                best_idx = calibrated_models[metric].idxmax()
                best_score = calibrated_models.loc[best_idx, metric]
                best_model = calibrated_models.loc[best_idx, 'model_name']
                logger.info(f"Best calibrated model: {best_model} (score: {best_score:.4f})")
                return best_model, best_score
            else:
                # If no calibrated models, find best overall
                logger.warning("No models meet calibration threshold, selecting best overall performance")
                for attempt_metric in [metric, 'combined_score_enhanced', 'average_precision']:
                    if attempt_metric in self.comparison_results.columns:
                        metric = attempt_metric
                        break
            
            best_idx = self.comparison_results[metric].idxmax()
            best_score = self.comparison_results.loc[best_idx, metric]
            best_model = self.comparison_results.loc[best_idx, 'model_name']
            
            return best_model, best_score
        
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
                'calibration_achievement_rate': float((calibration_scores >= 0.7).mean())
            }
            
            # Recommendations
            if report['summary']['best_score'] >= 0.30:
                report['recommendations'].append("Target combined score achieved")
            else:
                report['recommendations'].append("Performance improvement needed")
            
            if report['calibration_analysis']['calibration_achievement_rate'] >= 0.5:
                report['recommendations'].append("Good model calibration achieved")
            else:
                report['recommendations'].append("Calibration improvement recommended")
            
            # Save report
            if save_path:
                try:
                    import json
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                    
                    logger.info(f"Performance report saved: {save_path}")
                except Exception as e:
                    logger.warning(f"Report saving failed: {e}")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}
    
    def generate_calibration_plots(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                 model_name: str, save_dir: Optional[Path] = None) -> bool:
        """Generate calibration visualization plots"""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization not available, skipping plot generation")
            return False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Model Calibration Analysis: {model_name}', fontsize=14)
            
            # Calibration plot
            ax1 = axes[0, 0]
            fraction_of_positives, mean_predicted_value = self._compute_calibration_curve(y_true, y_pred_proba)
            
            ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax1.set_xlabel('Mean Predicted Probability')
            ax1.set_ylabel('Fraction of Positives')
            ax1.set_title('Calibration Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Prediction histogram
            ax2 = axes[0, 1]
            ax2.hist(y_pred_proba, bins=30, alpha=0.7, density=True)
            ax2.axvline(np.mean(y_pred_proba), color='red', linestyle='--', label=f'Mean: {np.mean(y_pred_proba):.4f}')
            ax2.axvline(np.mean(y_true), color='green', linestyle='--', label=f'Actual CTR: {np.mean(y_true):.4f}')
            ax2.set_xlabel('Predicted Probability')
            ax2.set_ylabel('Density')
            ax2.set_title('Prediction Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Reliability diagram
            ax3 = axes[1, 0]
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = []
            bin_accuracies = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                if np.any(in_bin):
                    bin_centers.append((bin_lower + bin_upper) / 2)
                    bin_accuracies.append(np.mean(y_true[in_bin]))
                    bin_counts.append(np.sum(in_bin))
            
            if bin_centers:
                ax3.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, edgecolor='black')
                ax3.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                ax3.set_xlabel('Bin Center')
                ax3.set_ylabel('Accuracy')
                ax3.set_title('Reliability Diagram')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Calibration error by bin
            ax4 = axes[1, 1]
            if bin_centers and bin_accuracies:
                errors = [abs(center - accuracy) for center, accuracy in zip(bin_centers, bin_accuracies)]
                ax4.bar(bin_centers, errors, width=0.08, alpha=0.7, color='red', edgecolor='black')
                ax4.set_xlabel('Bin Center')
                ax4.set_ylabel('Calibration Error')
                ax4.set_title('Calibration Error by Bin')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f'{model_name}_calibration_analysis.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Calibration plot saved: {save_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Calibration plot generation failed: {e}")
            return False
    
    def _compute_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
        """Compute calibration curve"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = (bin_lowers + bin_uppers) / 2
        bin_true_rates = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            if np.any(in_bin):
                bin_true_rates.append(np.mean(y_true[in_bin]))
            else:
                bin_true_rates.append(0.0)
        
        return np.array(bin_true_rates), bin_centers

def create_ultra_comparison_report():
    """Create ultra comparison report"""
    try:
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Ultra comparison report creation failed: {e}")
        return pd.DataFrame()

# Aliases for backward compatibility - ensure import from main.py
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