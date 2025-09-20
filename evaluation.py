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

class CTRAdvancedMetrics:
    """CTR prediction evaluation metrics"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
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
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Standard combined score calculation"""
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
            
            normalized_wll = max(0, 1 - wll_score / 5.0)
            combined = (ap_score * self.ap_weight) + (normalized_wll * self.wll_weight)
            
            return float(np.clip(combined, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.0
    
    def average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Average precision calculation"""
        try:
            if not SKLEARN_AVAILABLE:
                return self._manual_average_precision(y_true, y_pred_proba)
            
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
        """Weighted log loss calculation"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba) or len(y_true) == 0:
                return 100.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
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
    
    def ctr_optimized_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR optimized score calculation"""
        try:
            base_score = self.combined_score(y_true, y_pred_proba)
            
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            ctr_penalty = min(ctr_bias / self.ctr_tolerance, 1.0)
            ctr_adjusted_score = base_score * (1.0 - ctr_penalty * 0.2)
            
            return float(np.clip(ctr_adjusted_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"CTR optimized score calculation failed: {e}")
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
            
            # Core metrics
            metrics['ap'] = self.average_precision(y_true, y_pred_proba)
            metrics['wll'] = self.weighted_log_loss(y_true, y_pred_proba)
            metrics['combined_score'] = self.combined_score(y_true, y_pred_proba)
            metrics['ctr_optimized_score'] = self.ctr_optimized_score(y_true, y_pred_proba)
            
            # CTR analysis
            metrics['ctr_actual'] = float(y_true.mean())
            metrics['ctr_predicted'] = float(y_pred_proba.mean())
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
            
            # Classification metrics
            if SKLEARN_AVAILABLE and len(np.unique(y_true)) >= 2:
                try:
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
                    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                except Exception:
                    metrics.update({
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                        'f1': 0.0, 'roc_auc': 0.5
                    })
            else:
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'f1': 0.0, 'roc_auc': 0.5
                })
            
            # Performance tier
            combined_score = metrics['combined_score']
            if combined_score >= 0.35:
                metrics['performance_tier'] = 'exceptional'
            elif combined_score >= 0.30:
                metrics['performance_tier'] = 'excellent'
            elif combined_score >= 0.25:
                metrics['performance_tier'] = 'good'
            elif combined_score >= 0.20:
                metrics['performance_tier'] = 'fair'
            else:
                metrics['performance_tier'] = 'poor'
            
            metrics['evaluation_time'] = time.time() - start_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return self._get_default_metrics(model_name)
    
    def _get_default_metrics(self, model_name: str = "Unknown") -> Dict[str, Any]:
        """Default metrics for error cases"""
        return {
            'model_name': model_name,
            'ap': 0.0, 'wll': 100.0, 'combined_score': 0.0, 'ctr_optimized_score': 0.0,
            'ctr_actual': self.actual_ctr, 'ctr_predicted': self.actual_ctr, 'ctr_bias': 0.0,
            'ctr_absolute_error': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1': 0.0, 'roc_auc': 0.5, 'performance_tier': 'poor', 'evaluation_time': 0.0
        }
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        gc.collect()

class ModelComparator:
    """Model comparison utility"""
    
    def __init__(self):
        self.metrics_calculator = CTRAdvancedMetrics()
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
                logger.info(f"  - CTR Bias: {metrics['ctr_bias']:.4f}")
                logger.info(f"  - Performance Tier: {metrics['performance_tier']}")
                
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
        self.metrics_calculator = CTRAdvancedMetrics()
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate evaluation report"""
        try:
            report = {
                'report_metadata': {
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'evaluation_framework': 'CTR Advanced Metrics',
                    'target_combined_score': self.metrics_calculator.target_combined_score
                },
                'summary': evaluation_results,
                'performance_analysis': {
                    'target_achievement': evaluation_results.get('combined_score', 0.0) >= self.metrics_calculator.target_combined_score,
                    'performance_tier': evaluation_results.get('performance_tier', 'poor'),
                    'ctr_bias_analysis': {
                        'bias': evaluation_results.get('ctr_bias', 0.0),
                        'acceptable': abs(evaluation_results.get('ctr_bias', 1.0)) <= self.metrics_calculator.ctr_tolerance
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
CTRMetrics = CTRAdvancedMetrics
CTRMetricsCalculator = CTRAdvancedMetrics
UltraModelComparator = ModelComparator

def create_ctr_metrics():
    """CTR metrics generator"""
    return CTRAdvancedMetrics()

def create_model_comparator():
    """Model comparator generator"""
    return ModelComparator()

def create_evaluation_reporter():
    """Evaluation reporter generator"""
    return EvaluationReporter()

def evaluate_model_performance(y_true, y_pred_proba, model_name="Unknown"):
    """Single model performance evaluation"""
    metrics_calc = CTRAdvancedMetrics()
    return metrics_calc.comprehensive_evaluation(y_true, y_pred_proba, model_name)

def compare_multiple_models(models_predictions, y_true, models_info=None):
    """Multiple model comparison"""
    comparator = ModelComparator()
    return comparator.compare_models(models_predictions, y_true, models_info)