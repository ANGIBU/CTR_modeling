# evaluation.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
import warnings
from pathlib import Path
import json
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

class CTRMetrics:
    """CTR prediction specialized evaluation metrics class"""
    
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
        
    def average_precision_score_safe(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Safe Average Precision calculation"""
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
    
    def _manual_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Manual log loss calculation"""
        try:
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        except:
            return 100.0
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate combined score"""
        try:
            ap_score = self.average_precision_score_safe(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss_ctr(y_true, y_pred_proba)
            
            # Normalize WLL to 0-1 range (lower is better)
            wll_normalized = max(0, 1 - wll_score / 10.0)
            
            combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            # Apply CTR bias penalty
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            if ctr_bias > self.ctr_tolerance:
                bias_penalty = min(0.5, ctr_bias * self.bias_penalty_weight)
                combined = max(0, combined - bias_penalty)
            
            return float(combined)
            
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.0
    
    def calibration_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """Calculate calibration metrics"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0  # Expected Calibration Error
            mce = 0.0  # Maximum Calibration Error
            total_samples = len(y_true)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(y_true[in_bin])
                    avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                    
                    bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += bin_error * prop_in_bin
                    mce = max(mce, bin_error)
            
            # Brier score decomposition
            reliability = 0.0
            resolution = 0.0
            uncertainty = np.mean(y_true) * (1 - np.mean(y_true))
            
            if SKLEARN_AVAILABLE:
                brier_score = brier_score_loss(y_true, y_pred_proba)
            else:
                brier_score = np.mean((y_pred_proba - y_true) ** 2)
            
            calibration_metrics = {
                'ece': float(ece),
                'mce': float(mce),
                'brier_score': float(brier_score),
                'reliability': float(reliability),
                'resolution': float(resolution),
                'uncertainty': float(uncertainty)
            }
            
            # Overall calibration score (0-1, higher is better)
            calibration_overall = max(0, 1 - ece - mce * 0.5)
            calibration_metrics['calibration_score'] = float(calibration_overall)
            
            return calibration_metrics
            
        except Exception as e:
            logger.error(f"Calibration score calculation failed: {e}")
            return {
                'ece': 1.0, 'mce': 1.0, 'brier_score': 0.25,
                'reliability': 1.0, 'resolution': 0.0, 'uncertainty': 0.25,
                'calibration_score': 0.0
            }
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               model_name: str = "Unknown") -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                logger.error(f"Size mismatch in comprehensive evaluation")
                return self._get_default_metrics(model_name)
            
            metrics = {'model_name': model_name}
            
            # Core metrics
            try:
                metrics['ap'] = self.average_precision_score_safe(y_true, y_pred_proba)
                metrics['wll'] = self.weighted_log_loss_ctr(y_true, y_pred_proba)
                metrics['combined_score'] = self.combined_score(y_true, y_pred_proba)
                
                if SKLEARN_AVAILABLE:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                else:
                    metrics['auc'] = 0.5
                    metrics['log_loss'] = self._manual_log_loss(y_true, y_pred_proba)
                    metrics['brier_score'] = np.mean((y_pred_proba - y_true) ** 2)
                
            except Exception as e:
                logger.warning(f"Core metrics calculation failed: {e}")
                metrics.update({
                    'ap': 0.0, 'wll': 100.0, 'combined_score': 0.0,
                    'auc': 0.5, 'log_loss': 100.0, 'brier_score': 0.25
                })
            
            # Classification metrics
            try:
                y_pred_binary = (y_pred_proba > 0.5).astype(int)
                
                if SKLEARN_AVAILABLE:
                    cm = confusion_matrix(y_true, y_pred_binary)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        
                        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
                        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
                        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
                        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        metrics['sensitivity'] = metrics['recall']
                        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
                        
                        # Matthews Correlation Coefficient
                        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                        if denominator != 0:
                            metrics['mcc'] = (tp * tn - fp * fn) / denominator
                        else:
                            metrics['mcc'] = 0.0
                    else:
                        metrics.update({
                            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                            'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0
                        })
                else:
                    # Manual calculation fallback
                    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
                    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
                    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
                    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
                    
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
                
            except Exception as e:
                logger.warning(f"Classification metrics calculation failed: {e}")
                metrics.update({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'specificity': 0.0, 'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0
                })
            
            # CTR specific metrics
            try:
                metrics['ctr_actual'] = float(y_true.mean())
                metrics['ctr_predicted'] = float(y_pred_proba.mean())
                metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
                metrics['ctr_ratio'] = metrics['ctr_predicted'] / max(metrics['ctr_actual'], 1e-10)
                metrics['ctr_absolute_error'] = abs(metrics['ctr_bias'])
                metrics['ctr_relative_error'] = abs(metrics['ctr_bias']) / max(metrics['ctr_actual'], 1e-10)
                
            except Exception as e:
                logger.warning(f"CTR specific metrics calculation failed: {e}")
                metrics.update({
                    'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
                    'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0
                })
            
            # Calibration metrics
            try:
                calibration_metrics = self.calibration_score(y_true, y_pred_proba)
                metrics.update(calibration_metrics)
            except Exception as e:
                logger.warning(f"Calibration evaluation failed: {e}")
                metrics.update({
                    'ece': 1.0, 'mce': 1.0, 'brier_score': 0.25,
                    'reliability': 1.0, 'resolution': 0.0, 'uncertainty': 0.25,
                    'calibration_score': 0.0
                })
            
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
                
            except Exception as e:
                logger.warning(f"Prediction distribution analysis failed: {e}")
                metrics.update({
                    'pred_mean': 0.0, 'pred_std': 0.0, 'pred_min': 0.0, 'pred_max': 1.0,
                    'pred_median': 0.0, 'pred_q25': 0.0, 'pred_q75': 1.0, 'pred_iqr': 1.0
                })
            
            # Performance tier assessment
            combined_score = metrics.get('combined_score', 0.0)
            if combined_score >= 0.30:
                metrics['performance_tier'] = 'excellent'
            elif combined_score >= 0.20:
                metrics['performance_tier'] = 'good'
            elif combined_score >= 0.10:
                metrics['performance_tier'] = 'fair'
            else:
                metrics['performance_tier'] = 'poor'
            
            # Validate all metrics
            validated_metrics = {}
            for key, value in metrics.items():
                try:
                    if isinstance(value, (int, float, np.number)):
                        if np.isnan(value) or np.isinf(value):
                            if 'wll' in key.lower() or 'loss' in key.lower():
                                validated_metrics[key] = float('inf')
                            else:
                                validated_metrics[key] = 0.0
                        else:
                            validated_metrics[key] = float(value)
                    else:
                        validated_metrics[key] = value
                except:
                    validated_metrics[key] = 0.0 if 'wll' not in key.lower() and 'loss' not in key.lower() else float('inf')
            
            return validated_metrics
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation calculation error: {str(e)}")
            return self._get_default_metrics(model_name)
    
    def _get_default_metrics(self, model_name: str = "Unknown") -> Dict[str, Any]:
        """Return default metric values"""
        return {
            'model_name': model_name,
            'ap': 0.0, 'wll': float('inf'), 'combined_score': 0.0,
            'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25,
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0,
            'sensitivity': 0.0, 'f1': 0.0, 'mcc': 0.0,
            'ctr_actual': 0.0201, 'ctr_predicted': 0.0201, 'ctr_bias': 0.0,
            'ctr_ratio': 1.0, 'ctr_absolute_error': 0.0, 'ctr_relative_error': 0.0,
            'ece': 1.0, 'mce': 1.0, 'reliability': 1.0, 'resolution': 0.0,
            'uncertainty': 0.25, 'calibration_score': 0.0,
            'pred_mean': 0.0, 'pred_std': 0.0, 'pred_min': 0.0, 'pred_max': 1.0,
            'pred_median': 0.0, 'pred_q25': 0.0, 'pred_q75': 1.0, 'pred_iqr': 1.0,
            'performance_tier': 'poor'
        }

class ModelComparator:
    """Model comparison class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.metrics_calculator = CTRMetrics(config)
        self.comparison_results = pd.DataFrame()
        self.performance_analysis = {}
        
    def compare_models(self, models_predictions: Dict[str, np.ndarray], 
                      y_true: np.ndarray, models_info: Optional[Dict] = None) -> pd.DataFrame:
        """Compare multiple models"""
        try:
            logger.info(f"Model comparison started - {len(models_predictions)} models")
            
            results = []
            
            for model_name, predictions in models_predictions.items():
                try:
                    logger.info(f"Evaluating model: {model_name}")
                    
                    metrics = self.metrics_calculator.comprehensive_evaluation(
                        y_true, predictions, model_name
                    )
                    
                    # Add model info if available
                    if models_info and model_name in models_info:
                        metrics.update(models_info[model_name])
                    
                    results.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Model evaluation failed for {model_name}: {e}")
                    default_metrics = self.metrics_calculator._get_default_metrics(model_name)
                    results.append(default_metrics)
            
            self.comparison_results = pd.DataFrame(results)
            
            if not self.comparison_results.empty:
                self.comparison_results = self.comparison_results.sort_values(
                    'combined_score', ascending=False
                ).reset_index(drop=True)
                
                # Performance analysis
                self._analyze_performance()
            
            logger.info(f"Model comparison completed - {len(self.comparison_results)} models evaluated")
            
            return self.comparison_results
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return pd.DataFrame()
    
    def _analyze_performance(self):
        """Analyze model performance"""
        try:
            if self.comparison_results.empty:
                return
            
            self.performance_analysis = {
                'total_models': len(self.comparison_results),
                'best_model': self.comparison_results.iloc[0]['model_name'],
                'best_combined_score': float(self.comparison_results.iloc[0]['combined_score']),
                'avg_combined_score': float(self.comparison_results['combined_score'].mean()),
                'std_combined_score': float(self.comparison_results['combined_score'].std()),
                'target_achievers': int((self.comparison_results['combined_score'] >= 0.30).sum()),
                'performance_distribution': dict(self.comparison_results['performance_tier'].value_counts())
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            self.performance_analysis = {}
    
    def get_best_model(self, metric: str = 'combined_score') -> Tuple[Optional[str], float]:
        """Get best performing model"""
        try:
            if self.comparison_results.empty:
                return None, 0.0
            
            if metric not in self.comparison_results.columns:
                logger.warning(f"Metric {metric} not found, using combined_score")
                metric = 'combined_score'
            
            best_idx = self.comparison_results[metric].idxmax()
            best_model = self.comparison_results.loc[best_idx, 'model_name']
            best_score = self.comparison_results.loc[best_idx, metric]
            
            return best_model, float(best_score)
            
        except Exception as e:
            logger.error(f"Best model finding failed: {e}")
            return None, 0.0

class EvaluationReporter:
    """Evaluation result reporting class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.report_data = {}
        
    def generate_model_performance_report(self, comparator: ModelComparator,
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
            
            # Find best model
            best_model, best_score = comparator.get_best_model()
            report['summary']['best_model'] = best_model
            report['summary']['best_score'] = best_score
            
            # Detailed results
            for idx, row in comparator.comparison_results.iterrows():
                model_data = row.to_dict()
                report['detailed_results'][model_data['model_name']] = model_data
            
            # Performance analysis
            report['performance_analysis'] = comparator.performance_analysis
            
            # Save report if path provided
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

# Aliases for backward compatibility
CTRAdvancedMetrics = CTRMetrics
UltraModelComparator = ModelComparator

# Factory functions
def create_ctr_metrics(config: Config = Config):
    """CTR metrics generator"""
    return CTRMetrics(config)

def create_model_comparator(config: Config = Config):
    """Model comparator generator"""
    return ModelComparator(config)

def create_evaluation_reporter(config: Config = Config):
    """Evaluation reporter generator"""
    return EvaluationReporter(config)

# Main functions directly accessible at module level
def evaluate_model_performance(y_true, y_pred_proba, model_name="Unknown"):
    """Single model performance evaluation"""
    metrics_calc = CTRMetrics()
    return metrics_calc.comprehensive_evaluation(y_true, y_pred_proba, model_name)

def compare_multiple_models(models_predictions, y_true, models_info=None):
    """Multiple model comparison"""
    comparator = ModelComparator()
    return comparator.compare_models(models_predictions, y_true, models_info)

def evaluate_model_performance_with_calibration(y_true, y_pred_proba, model_name="Unknown"):
    """Single model performance evaluation with calibration"""
    return evaluate_model_performance(y_true, y_pred_proba, model_name)

def compare_multiple_models_with_calibration(models_predictions, y_true, models_info=None):
    """Multiple model comparison with calibration evaluation"""
    return compare_multiple_models(models_predictions, y_true, models_info)