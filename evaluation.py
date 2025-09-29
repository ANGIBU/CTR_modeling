# evaluation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
import warnings
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("matplotlib/seaborn not installed. Visualization features disabled.")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not installed. Some statistical features disabled.")

from config import Config

logger = logging.getLogger(__name__)

class CTRAdvancedMetrics:
    """CTR prediction specialized metrics class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ap_weight = config.EVALUATION_CONFIG['ap_weight']
        self.wll_weight = config.EVALUATION_CONFIG['wll_weight']
        self.actual_ctr = 0.0201
        self.pos_weight = 49.8
        self.neg_weight = 1.0
        self.target_combined_score = 0.34
        self.ctr_tolerance = 0.001
        self.bias_penalty_weight = 5.0
        self.calibration_weight = 0.4
        
        self.cache = {}
        
    def average_precision_enhanced(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Enhanced Average Precision calculation"""
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
                logger.warning("Empty array, AP calculation not possible")
                return 0.0
            
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                logger.warning("Single class only, AP calculation not possible")
                return 0.0
            
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                logger.warning("NaN or infinite values in predictions, clipping applied")
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
            
            ap_score = float(ap_score)
            self.cache[cache_key] = ap_score
            
            return ap_score
            
        except Exception as e:
            logger.error(f"AP calculation failed: {e}")
            return 0.0
    
    def _manual_average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual Average Precision calculation"""
        try:
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            y_true_sorted = y_true[sorted_indices]
            
            tp_sum = 0
            ap_sum = 0
            
            for i, label in enumerate(y_true_sorted):
                if label == 1:
                    tp_sum += 1
                    precision = tp_sum / (i + 1)
                    ap_sum += precision
            
            total_positives = np.sum(y_true)
            if total_positives > 0:
                return ap_sum / total_positives
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def weighted_logloss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Weighted Log Loss calculation"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"Size mismatch in WLL calculation")
                return 100.0
            
            if len(y_true) == 0:
                return 100.0
            
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            weights = np.where(y_true == 1, self.pos_weight, self.neg_weight)
            
            log_losses = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
            weighted_loss = np.average(log_losses, weights=weights)
            
            return float(weighted_loss)
            
        except Exception as e:
            logger.error(f"WLL calculation failed: {e}")
            return 100.0
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Combined Score calculation"""
        try:
            ap = self.average_precision_enhanced(y_true, y_pred_proba)
            wll = self.weighted_logloss(y_true, y_pred_proba)
            
            if wll > 10:
                wll = 10
            
            normalized_wll = 1.0 - (wll / 10.0)
            combined = (self.ap_weight * ap) + (self.wll_weight * normalized_wll)
            
            return float(combined)
            
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.0
    
    def ctr_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR alignment score calculation"""
        try:
            actual_ctr = np.mean(y_true)
            predicted_ctr = np.mean(y_pred_proba)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # CTR alignment score based on bias
            if ctr_bias < 0.0001:
                score = 1.0
            elif ctr_bias < 0.0005:
                score = 0.9
            elif ctr_bias < 0.001:
                score = 0.7
            elif ctr_bias < 0.002:
                score = 0.5
            else:
                score = max(0.0, 1.0 - (ctr_bias * 100))
            
            return float(score)
            
        except Exception as e:
            logger.error(f"CTR score calculation failed: {e}")
            return 0.0
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                model_name: str = "Unknown") -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        try:
            metrics = {'model_name': model_name}
            
            # Core metrics
            try:
                unique_classes = np.unique(y_true)
                if len(unique_classes) >= 2:
                    auc = roc_auc_score(y_true, y_pred_proba)
                else:
                    auc = float('nan')
                
                ap = self.average_precision_enhanced(y_true, y_pred_proba)
                
                y_pred_proba_safe = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                log_loss_val = log_loss(y_true, y_pred_proba_safe)
                
                combined_score = self.combined_score(y_true, y_pred_proba)
                
                # Binary classification metrics
                y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                
                # Specificity
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
            except Exception as e:
                auc = float('nan')
                ap = 0.0
                log_loss_val = 100.0
                combined_score = 0.0
                accuracy = precision = recall = f1 = specificity = 0.0
                logger.warning(f"{model_name}: Core metrics calculation failed: {e}")
            
            metrics.update({
                'auc': float(auc) if not np.isnan(auc) else float('nan'),
                'ap': float(ap),
                'log_loss': float(log_loss_val),
                'combined_score': float(combined_score),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1)
            })
            
            # CTR analysis
            try:
                actual_ctr = float(np.mean(y_true))
                predicted_ctr = float(np.mean(y_pred_proba))
                ctr_bias = predicted_ctr - actual_ctr
                ctr_absolute_error = abs(ctr_bias)
                
                metrics.update({
                    'actual_ctr': actual_ctr,
                    'predicted_ctr': predicted_ctr,
                    'ctr_bias': ctr_bias,
                    'ctr_absolute_error': ctr_absolute_error,
                    'ctr_relative_error': float(ctr_absolute_error / actual_ctr) if actual_ctr > 0 else float('inf'),
                    'ctr_score': self.ctr_score(y_true, y_pred_proba)
                })
                
            except Exception as e:
                logger.warning(f"{model_name}: CTR analysis failed: {e}")
                metrics.update({
                    'actual_ctr': 0.0201,
                    'predicted_ctr': 0.0201,
                    'ctr_bias': 0.0,
                    'ctr_absolute_error': 0.0,
                    'ctr_relative_error': 0.0,
                    'ctr_score': 0.0
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return {'error': str(e), 'model_name': model_name}

class UltraModelComparator:
    """Model comparison class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.metrics_calculator = CTRAdvancedMetrics(config)
        self.comparison_results = pd.DataFrame()
        
    def compare_models(self, models_predictions: Dict[str, np.ndarray], 
                      y_true: np.ndarray,
                      models_info: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """Compare multiple models"""
        try:
            logger.info(f"Starting model comparison: {len(models_predictions)} models")
            
            results = []
            
            for model_name, y_pred_proba in models_predictions.items():
                try:
                    start_time = time.time()
                    
                    # Input validation
                    y_true = np.asarray(y_true).flatten()
                    y_pred_proba = np.asarray(y_pred_proba).flatten()
                    
                    if len(y_true) != len(y_pred_proba):
                        logger.error(f"{model_name}: Size mismatch")
                        continue
                    
                    if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                        logger.warning(f"{model_name}: NaN or infinite values in predictions, cleaning")
                        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
                        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                    
                    # Comprehensive evaluation
                    metrics = self.metrics_calculator.comprehensive_evaluation(
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
                    logger.info(f"  - Combined Score: {metrics['combined_score']:.4f}")
                    logger.info(f"  - CTR Bias: {metrics['ctr_bias']:.4f}")
                    logger.info(f"  - CTR Score: {metrics['ctr_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"{model_name} evaluation failed: {str(e)}")
                    default_metrics = {
                        'model_name': model_name,
                        'combined_score': 0.0,
                        'auc': float('nan'),
                        'ap': 0.0,
                        'ctr_score': 0.0,
                        'evaluation_duration': 0.0,
                        'is_calibrated': False,
                        'calibration_method': 'none',
                        'model_type': 'unknown'
                    }
                    results.append(default_metrics)
            
            if not results:
                logger.error("No models could be evaluated")
                return pd.DataFrame()
            
            try:
                comparison_df = pd.DataFrame(results)
                comparison_df = comparison_df.set_index('model_name')
                comparison_df = comparison_df.sort_values('combined_score', ascending=False)
                
                self.comparison_results = comparison_df
                
                logger.info("Model comparison completed")
                logger.info(f"Best model: {comparison_df.index[0]} (Combined Score: {comparison_df.iloc[0]['combined_score']:.4f})")
                
                return comparison_df
                
            except Exception as e:
                logger.error(f"Comparison result processing failed: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return pd.DataFrame()
    
    def get_best_model(self, metric: str = 'combined_score') -> Tuple[Optional[str], float]:
        """Get best performing model"""
        try:
            if self.comparison_results.empty:
                logger.error("No comparison results available")
                return None, 0.0
            
            if metric not in self.comparison_results.columns:
                logger.error(f"Metric {metric} not found")
                return None, 0.0
            
            best_idx = self.comparison_results[metric].idxmax()
            best_score = self.comparison_results.loc[best_idx, metric]
            
            return best_idx, best_score
        
        except Exception as e:
            logger.error(f"Best model search failed: {e}")
            return None, 0.0

class EvaluationReporter:
    """Evaluation result reporting class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.report_data = {}
        
    def generate_model_performance_report(self, 
                                         comparator: UltraModelComparator,
                                         save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate model performance report"""
        try:
            if comparator.comparison_results.empty:
                return {'error': 'No comparison results available'}
            
            report = {
                'summary': {
                    'total_models': len(comparator.comparison_results),
                    'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'best_model': None,
                    'best_score': 0.0
                },
                'detailed_results': {},
                'performance_analysis': {},
                'recommendations': []
            }
            
            # Find best performing model
            best_model, best_score = comparator.get_best_model()
            report['summary']['best_model'] = best_model
            report['summary']['best_score'] = best_score
            
            # Detailed results
            for model_name, row in comparator.comparison_results.iterrows():
                report['detailed_results'][model_name] = {
                    'combined_score': float(row['combined_score']),
                    'auc': float(row['auc']) if not np.isnan(row['auc']) else None,
                    'ap': float(row['ap']),
                    'ctr_bias': float(row.get('ctr_bias', 0.0)),
                    'ctr_score': float(row.get('ctr_score', 0.0)),
                    'is_calibrated': bool(row.get('is_calibrated', False)),
                    'calibration_method': str(row.get('calibration_method', 'none'))
                }
            
            # Performance analysis
            df = comparator.comparison_results
            report['performance_analysis'] = {
                'avg_combined_score': float(df['combined_score'].mean()),
                'std_combined_score': float(df['combined_score'].std()),
                'avg_ctr_bias': float(df['ctr_bias'].mean()) if 'ctr_bias' in df.columns else 0.0,
                'target_achievers': int((df['combined_score'] >= 0.34).sum())
            }
            
            # Recommendations
            if best_score >= 0.34:
                report['recommendations'].append("Target achieved: Model(s) achieving Combined Score 0.34+")
            else:
                report['recommendations'].append("Improvement needed: All models below target score")
            
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

# Backward compatibility aliases
CTRMetrics = CTRAdvancedMetrics
ModelComparator = UltraModelComparator

# Additional backward compatibility
def create_ctr_metrics():
    """CTR metrics generator"""
    return CTRAdvancedMetrics()

def create_model_comparator():
    """Model comparator generator"""
    return UltraModelComparator()

def create_evaluation_reporter():
    """Evaluation reporter generator"""
    return EvaluationReporter()

# Module level functions
def evaluate_model_performance(y_true, y_pred_proba, model_name="Unknown"):
    """Single model performance evaluation"""
    metrics_calc = CTRAdvancedMetrics()
    return metrics_calc.comprehensive_evaluation(y_true, y_pred_proba, model_name)

def compare_multiple_models(models_predictions, y_true, models_info=None):
    """Multiple model comparison"""
    comparator = UltraModelComparator()
    return comparator.compare_models(models_predictions, y_true, models_info)