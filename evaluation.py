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

class CTRMetrics:
    """CTR prediction evaluation metrics"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
        evaluation_config = getattr(config, 'EVALUATION_CONFIG', {
            'ap_weight': 0.6,
            'wll_weight': 0.4,
            'target_combined_score': 0.34,
            'target_ctr': 0.0191,
            'ctr_tolerance': 0.0002,
            'bias_penalty_weight': 8.0,
            'calibration_weight': 0.5,
            'pos_weight': 52.3,
            'neg_weight': 1.0,
            'wll_normalization_factor': 2.6
        })
        
        self.ap_weight = evaluation_config.get('ap_weight', 0.6)
        self.wll_weight = evaluation_config.get('wll_weight', 0.4)
        self.actual_ctr = evaluation_config.get('target_ctr', 0.0191)
        self.pos_weight = evaluation_config.get('pos_weight', 52.3)
        self.neg_weight = evaluation_config.get('neg_weight', 1.0)
        self.target_combined_score = evaluation_config.get('target_combined_score', 0.34)
        self.ctr_tolerance = evaluation_config.get('ctr_tolerance', 0.0002)
        self.bias_penalty_weight = evaluation_config.get('bias_penalty_weight', 8.0)
        self.calibration_weight = evaluation_config.get('calibration_weight', 0.5)
        self.wll_normalization_factor = evaluation_config.get('wll_normalization_factor', 2.6)
        
        self.cache = {}
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Standard combined score calculation with corrected normalization"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) == 0 or len(y_pred_proba) == 0:
                return 0.0
            
            if len(y_true) != len(y_pred_proba):
                logger.warning(f"Size mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
                return 0.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            ap_score = self.average_precision(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss(y_true, y_pred_proba)
            
            # Corrected WLL normalization with tuned factor
            normalized_wll = max(0, 1 - wll_score / self.wll_normalization_factor)
            combined = (ap_score * self.ap_weight) + (normalized_wll * self.wll_weight)
            
            return float(np.clip(combined, 0.0, 1.0))
            
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
                logger.warning(f"sklearn AP calculation failed: {e}")
                return self._manual_average_precision(y_true, y_pred_proba)
            
        except Exception as e:
            logger.error(f"Average precision calculation failed: {e}")
            return 0.0
    
    def weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Weighted log loss calculation with tuned weights"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return 100.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # Use updated pos_weight from config
            sample_weights = np.where(y_true == 1, self.pos_weight, self.neg_weight)
            
            log_loss_values = -(y_true * np.log(y_pred_proba) + 
                              (1 - y_true) * np.log(1 - y_pred_proba))
            
            weighted_loss = np.average(log_loss_values, weights=sample_weights)
            
            return float(np.clip(weighted_loss, 0.0, 100.0))
            
        except Exception as e:
            logger.error(f"Weighted log loss calculation failed: {e}")
            return 100.0
    
    def _manual_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual average precision calculation"""
        try:
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            precisions = []
            recalls = []
            
            tp = 0
            fp = 0
            total_positives = np.sum(y_true)
            
            if total_positives == 0:
                return 0.0
            
            for i, label in enumerate(y_true_sorted):
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / total_positives
                
                precisions.append(precision)
                recalls.append(recall)
            
            ap = 0.0
            for i in range(1, len(recalls)):
                ap += precisions[i] * (recalls[i] - recalls[i-1])
            
            return float(np.clip(ap, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def ctr_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR score calculation with bias penalty"""
        try:
            base_score = self.combined_score(y_true, y_pred_proba)
            
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Tuned bias penalty with tighter tolerance
            if ctr_bias > self.ctr_tolerance:
                penalty_factor = min((ctr_bias / self.ctr_tolerance) * 0.12, 0.35)
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
                end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(y_true)
                
                segment_y = y_true[start_idx:end_idx]
                segment_pred = y_pred_proba[start_idx:end_idx]
                
                if len(segment_y) > 0 and len(np.unique(segment_y)) > 1:
                    segment_score = self.combined_score(segment_y, segment_pred)
                    segment_scores.append(segment_score)
            
            if len(segment_scores) < 2:
                return 0.0
            
            # Stability is inverse of coefficient of variation
            mean_score = np.mean(segment_scores)
            std_score = np.std(segment_scores)
            
            if mean_score == 0:
                return 0.0
            
            cv = std_score / mean_score
            stability = 1.0 / (1.0 + cv)
            
            return float(np.clip(stability, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Stability score calculation failed: {e}")
            return 0.0
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               model_name: str = "Unknown", threshold: float = 0.5) -> Dict[str, Any]:
        """Comprehensive evaluation metrics"""
        try:
            start_time = time.time()
            
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return self._get_default_metrics(model_name)
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            metrics = {'model_name': model_name}
            
            # Core metrics with tuned parameters
            metrics['ap'] = self.average_precision(y_true, y_pred_proba)
            metrics['wll'] = self.weighted_log_loss(y_true, y_pred_proba)
            metrics['combined_score'] = self.combined_score(y_true, y_pred_proba)
            metrics['ctr_score'] = self.ctr_score(y_true, y_pred_proba)
            
            # CTR analysis with corrected target
            metrics['ctr_actual'] = float(y_true.mean())
            metrics['ctr_predicted'] = float(y_pred_proba.mean())
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
            
            # Target alignment check
            metrics['ctr_target_aligned'] = abs(metrics['ctr_actual'] - self.actual_ctr) < 0.001
            metrics['ctr_prediction_aligned'] = abs(metrics['ctr_predicted'] - self.actual_ctr) < self.ctr_tolerance
            
            # Stability metrics
            metrics['stability_score'] = self.stability_score(y_true, y_pred_proba)
            
            # Classification metrics
            if SKLEARN_AVAILABLE and len(np.unique(y_true)) >= 2:
                try:
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
                    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                    
                    # Brier score for calibration assessment
                    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                except Exception:
                    metrics.update({
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                        'f1': 0.0, 'roc_auc': 0.5, 'brier_score': 1.0
                    })
            else:
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'f1': 0.0, 'roc_auc': 0.5, 'brier_score': 1.0
                })
            
            # Performance tier assessment with adjusted thresholds
            combined_score = metrics['combined_score']
            if combined_score >= 0.34:
                metrics['performance_tier'] = 'target_achieved'
            elif combined_score >= 0.30:
                metrics['performance_tier'] = 'excellent'
            elif combined_score >= 0.25:
                metrics['performance_tier'] = 'good'
            elif combined_score >= 0.20:
                metrics['performance_tier'] = 'fair'
            else:
                metrics['performance_tier'] = 'poor'
            
            # Prediction quality metrics
            pred_std = np.std(y_pred_proba)
            pred_entropy = -np.mean(y_pred_proba * np.log(y_pred_proba + 1e-15) + 
                                  (1 - y_pred_proba) * np.log(1 - y_pred_proba + 1e-15))
            
            metrics['prediction_std'] = float(pred_std)
            metrics['prediction_entropy'] = float(pred_entropy)
            metrics['prediction_range'] = float(y_pred_proba.max() - y_pred_proba.min())
            
            # Calibration quality assessment
            if metrics['brier_score'] < 0.02:
                metrics['calibration_quality'] = 'excellent'
            elif metrics['brier_score'] < 0.025:
                metrics['calibration_quality'] = 'good'
            elif metrics['brier_score'] < 0.03:
                metrics['calibration_quality'] = 'fair'
            else:
                metrics['calibration_quality'] = 'poor'
            
            metrics['evaluation_time'] = time.time() - start_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return self._get_default_metrics(model_name)
    
    def _get_default_metrics(self, model_name: str = "Unknown") -> Dict[str, Any]:
        """Default metrics for error cases"""
        return {
            'model_name': model_name,
            'ap': 0.0, 'wll': 100.0, 'combined_score': 0.0, 'ctr_score': 0.0,
            'ctr_actual': self.actual_ctr, 'ctr_predicted': self.actual_ctr, 'ctr_bias': 0.0,
            'ctr_absolute_error': 0.0, 'ctr_target_aligned': True, 'ctr_prediction_aligned': True,
            'stability_score': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1': 0.0, 'roc_auc': 0.5, 'brier_score': 1.0, 'performance_tier': 'poor',
            'prediction_std': 0.0, 'prediction_entropy': 0.0, 'prediction_range': 0.0,
            'calibration_quality': 'poor', 'evaluation_time': 0.0
        }
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        gc.collect()

class ModelComparator:
    """Model comparison utility"""
    
    def __init__(self):
        self.metrics_calculator = CTRMetrics()
        self.comparison_results = pd.DataFrame()
    
    def compare_models(self, models_predictions: Dict[str, np.ndarray], 
                      y_true: np.ndarray, models_info: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """Compare multiple models"""
        results = []
        y_true = np.asarray(y_true).flatten()
        
        logger.info(f"Model comparison started: {len(models_predictions)} models")
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                start_time = time.time()
                
                y_pred_proba = np.asarray(y_pred_proba).flatten()
                
                if len(y_pred_proba) != len(y_true):
                    logger.error(f"{model_name}: Size mismatch")
                    continue
                
                if len(y_pred_proba) == 0:
                    logger.error(f"{model_name}: Empty predictions")
                    continue
                
                y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                
                metrics = self.metrics_calculator.comprehensive_evaluation(
                    y_true, y_pred_proba, model_name
                )
                
                evaluation_time = time.time() - start_time
                metrics['evaluation_duration'] = evaluation_time
                
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
                logger.info(f"  - Combined Score: {metrics['combined_score']:.4f}")
                logger.info(f"  - CTR Score: {metrics['ctr_score']:.4f}")
                logger.info(f"  - CTR Bias: {metrics['ctr_bias']:.4f}")
                logger.info(f"  - Performance Tier: {metrics['performance_tier']}")
                logger.info(f"  - Stability Score: {metrics['stability_score']:.4f}")
                
            except Exception as e:
                logger.error(f"{model_name} evaluation failed: {str(e)}")
                default_metrics = self.metrics_calculator._get_default_metrics(model_name)
                default_metrics['evaluation_duration'] = 0.0
                default_metrics['is_calibrated'] = False
                default_metrics['calibration_method'] = 'none'
                default_metrics['model_type'] = 'unknown'
                results.append(default_metrics)
        
        if results:
            comparison_df = pd.DataFrame(results)
            comparison_df = comparison_df.sort_values('combined_score', ascending=False)
            self.comparison_results = comparison_df
            
            logger.info(f"Model comparison completed: {len(results)} models evaluated")
            return comparison_df
        else:
            logger.error("No models could be evaluated")
            return pd.DataFrame()

class EvaluationReporter:
    """Evaluation report generator"""
    
    def __init__(self):
        self.metrics_calculator = CTRMetrics()
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate evaluation report"""
        try:
            report = {
                'report_metadata': {
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'evaluation_framework': 'CTR Metrics',
                    'target_combined_score': self.metrics_calculator.target_combined_score,
                    'wll_normalization_factor': self.metrics_calculator.wll_normalization_factor,
                    'target_ctr': self.metrics_calculator.actual_ctr,
                    'pos_weight': self.metrics_calculator.pos_weight
                },
                'summary': evaluation_results,
                'performance_analysis': {
                    'target_achievement': evaluation_results.get('combined_score', 0.0) >= self.metrics_calculator.target_combined_score,
                    'performance_tier': evaluation_results.get('performance_tier', 'poor'),
                    'ctr_bias_analysis': {
                        'bias': evaluation_results.get('ctr_bias', 0.0),
                        'acceptable': abs(evaluation_results.get('ctr_bias', 1.0)) <= self.metrics_calculator.ctr_tolerance,
                        'target_aligned': evaluation_results.get('ctr_target_aligned', False)
                    },
                    'calibration_quality': {
                        'brier_score': evaluation_results.get('brier_score', 1.0),
                        'prediction_diversity': evaluation_results.get('prediction_std', 0.0),
                        'calibration_assessment': evaluation_results.get('calibration_quality', 'poor')
                    },
                    'stability_assessment': {
                        'stability_score': evaluation_results.get('stability_score', 0.0),
                        'prediction_range': evaluation_results.get('prediction_range', 0.0)
                    }
                }
            }
            
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

# Backward compatibility
CTRAdvancedMetrics = CTRMetrics
CTRMetricsCalculator = CTRMetrics
UltraModelComparator = ModelComparator

def create_ctr_metrics():
    """CTR metrics generator"""
    return CTRMetrics()

def create_model_comparator():
    """Model comparator generator"""
    return ModelComparator()

def create_evaluation_reporter():
    """Evaluation reporter generator"""
    return EvaluationReporter()

def evaluate_model_performance(y_true, y_pred_proba, model_name="Unknown"):
    """Single model performance evaluation"""
    metrics_calc = CTRMetrics()
    return metrics_calc.comprehensive_evaluation(y_true, y_pred_proba, model_name)

def compare_multiple_models(models_predictions, y_true, models_info=None):
    """Multiple model comparison"""
    comparator = ModelComparator()
    return comparator.compare_models(models_predictions, y_true, models_info)