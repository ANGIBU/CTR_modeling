# analysis.py
"""
CTR Performance Analysis Module
Advanced performance analysis for Click-Through Rate prediction models
"""

import os
import gc
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Essential imports
import numpy as np
import pandas as pd

# Try advanced imports with fallbacks
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, log_loss, 
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, brier_score_loss
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available for analysis")

logger = logging.getLogger(__name__)

class CTRPerformanceAnalyzer:
    """
    Comprehensive CTR performance analyzer with advanced metrics
    """
    
    def __init__(self, config):
        self.config = config
        self.target_ctr = getattr(config, 'TARGET_CTR', 0.0191)
        self.ctr_tolerance = getattr(config, 'CTR_TOLERANCE', 0.0002)
        self.target_combined_score = getattr(config, 'TARGET_COMBINED_SCORE', 0.34)
        
        # Performance thresholds
        self.performance_tiers = {
            'EXCEPTIONAL': 0.35,
            'EXCELLENT': 0.30,
            'GOOD': 0.25,
            'FAIR': 0.20,
            'POOR': 0.15
        }
        
        # Cache for computations
        self.cache = {}
        
        logger.info("CTR Performance Analyzer initialized")
    
    def full_performance_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                model_name: str = "Unknown", quick_mode: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model being analyzed
            quick_mode: Whether to run in quick mode
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            logger.info(f"Starting full performance analysis for {model_name}")
            start_time = time.time()
            
            # Validate inputs
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            if len(y_true) != len(y_pred_proba):
                raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}")
            
            if len(y_true) == 0:
                raise ValueError("Empty input arrays")
            
            # Core performance metrics
            core_metrics = self._calculate_core_metrics(y_true, y_pred_proba, model_name)
            
            # CTR-specific analysis
            ctr_analysis = self._analyze_ctr_performance(y_true, y_pred_proba)
            
            # Execution metrics
            execution_time = time.time() - start_time
            execution_metrics = {
                'execution_time': execution_time,
                'data_size': len(y_true),
                'memory_usage': self._get_memory_usage(),
                'gpu_utilization': self._get_gpu_utilization() if not quick_mode else 0.0,
                'analysis_mode': 'QUICK' if quick_mode else 'FULL'
            }
            
            # Overall assessment
            overall_assessment = self._generate_overall_assessment(core_metrics, ctr_analysis)
            
            # Performance recommendations
            recommendations = self._generate_recommendations(core_metrics, ctr_analysis, overall_assessment)
            
            analysis_result = {
                'model_name': model_name,
                'core_metrics': core_metrics,
                'ctr_analysis': ctr_analysis,
                'execution_metrics': execution_metrics,
                'overall_assessment': overall_assessment,
                'recommendations': recommendations,
                'analysis_timestamp': time.time(),
                'quick_mode': quick_mode
            }
            
            logger.info(f"Performance analysis completed for {model_name} in {execution_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Performance analysis failed for {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'analysis_timestamp': time.time(),
                'quick_mode': quick_mode
            }
    
    def _calculate_core_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              model_name: str) -> Dict[str, Any]:
        """Calculate core performance metrics"""
        try:
            metrics = {'model_name': model_name}
            
            if not SKLEARN_AVAILABLE:
                logger.warning("Sklearn not available, using basic metrics")
                metrics.update({
                    'auc': 0.5, 'ap': 0.0, 'log_loss': 1.0, 'combined_score': 0.0,
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1_score': 0.0
                })
                return metrics
            
            # AUC calculation with error handling
            try:
                if len(np.unique(y_true)) == 1:
                    auc = float('nan')
                    logger.warning(f"{model_name}: AUC calculation failed - only one class present")
                else:
                    auc = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                auc = float('nan')
                logger.warning(f"{model_name}: AUC calculation failed: {e}")
            
            # Average Precision
            try:
                ap = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                ap = 0.0
                logger.warning(f"{model_name}: AP calculation failed: {e}")
            
            # Log Loss
            try:
                # Clip predictions to avoid log(0)
                y_pred_clipped = np.clip(y_pred_proba, 1e-15, 1-1e-15)
                log_loss_val = log_loss(y_true, y_pred_clipped)
            except Exception as e:
                log_loss_val = 1.0
                logger.warning(f"{model_name}: Log loss calculation failed: {e}")
            
            # Combined score calculation
            if not np.isnan(auc):
                combined_score = (auc * 0.6) + (ap * 0.4)
            else:
                combined_score = ap * 0.4  # Use only AP if AUC is invalid
            
            # Binary classification metrics
            try:
                y_pred_binary = (y_pred_proba >= 0.5).astype(int)
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, zero_division=0)
                recall = recall_score(y_true, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                
                # Specificity calculation
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
            except Exception as e:
                accuracy = precision = recall = f1 = specificity = 0.0
                logger.warning(f"{model_name}: Binary metrics calculation failed: {e}")
            
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
            
            return metrics
            
        except Exception as e:
            logger.error(f"Core metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_ctr_performance(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze CTR specific performance"""
        try:
            ctr_analysis = {}
            
            # Basic CTR statistics
            actual_ctr = float(np.mean(y_true))
            predicted_ctr = float(np.mean(y_pred_proba))
            ctr_bias = predicted_ctr - actual_ctr
            ctr_absolute_error = abs(ctr_bias)
            
            ctr_analysis.update({
                'actual_ctr': actual_ctr,
                'predicted_ctr': predicted_ctr,
                'ctr_bias': ctr_bias,
                'ctr_absolute_error': ctr_absolute_error,
                'ctr_relative_error': float(ctr_absolute_error / actual_ctr) if actual_ctr > 0 else float('inf')
            })
            
            # Target alignment
            ctr_analysis['target_alignment'] = {
                'target_ctr': self.target_ctr,
                'actual_vs_target_error': abs(actual_ctr - self.target_ctr),
                'predicted_vs_target_error': abs(predicted_ctr - self.target_ctr),
                'within_tolerance': ctr_absolute_error <= self.ctr_tolerance
            }
            
            # CTR distribution analysis
            ctr_analysis['distribution'] = {
                'prediction_std': float(np.std(y_pred_proba)),
                'prediction_min': float(np.min(y_pred_proba)),
                'prediction_max': float(np.max(y_pred_proba)),
                'prediction_median': float(np.median(y_pred_proba)),
                'prediction_q25': float(np.percentile(y_pred_proba, 25)),
                'prediction_q75': float(np.percentile(y_pred_proba, 75))
            }
            
            # CTR quality assessment
            if ctr_absolute_error <= 0.0001:
                ctr_quality = 'EXCELLENT'
            elif ctr_absolute_error <= 0.0002:
                ctr_quality = 'GOOD'
            elif ctr_absolute_error <= 0.0005:
                ctr_quality = 'FAIR'
            else:
                ctr_quality = 'POOR'
            
            ctr_analysis['ctr_quality'] = ctr_quality
            
            return ctr_analysis
            
        except Exception as e:
            logger.error(f"CTR analysis failed: {e}")
            return {'error': str(e)}
    
    def _classify_performance_tier(self, combined_score: float) -> str:
        """Classify model performance tier"""
        for tier, threshold in self.performance_tiers.items():
            if combined_score >= threshold:
                return tier
        return 'POOR'
    
    def _generate_overall_assessment(self, core_metrics: Dict[str, Any], 
                                   ctr_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall performance assessment"""
        try:
            combined_score = core_metrics.get('combined_score', 0.0)
            performance_tier = self._classify_performance_tier(combined_score)
            
            # Deployment readiness assessment
            auc = core_metrics.get('auc', 0.5)
            deployment_ready = (
                combined_score >= 0.30 and
                not np.isnan(auc) and
                auc >= 0.75 and
                ctr_analysis.get('ctr_quality', 'POOR') in ['EXCELLENT', 'GOOD']
            )
            
            # Target achievement
            target_achievement = combined_score >= self.target_combined_score
            
            # Overall score calculation
            ctr_quality_score = {
                'EXCELLENT': 1.0, 'GOOD': 0.8, 'FAIR': 0.6, 'POOR': 0.3
            }.get(ctr_analysis.get('ctr_quality', 'POOR'), 0.3)
            
            overall_score = (combined_score * 0.7) + (ctr_quality_score * 0.3)
            
            return {
                'performance_tier': performance_tier,
                'deployment_ready': deployment_ready,
                'target_achievement': target_achievement,
                'overall_score': float(overall_score)
            }
            
        except Exception as e:
            logger.error(f"Overall assessment failed: {e}")
            return {
                'performance_tier': 'POOR',
                'deployment_ready': False,
                'target_achievement': False,
                'overall_score': 0.0
            }
    
    def _generate_recommendations(self, core_metrics: Dict[str, Any], 
                                ctr_analysis: Dict[str, Any],
                                overall_assessment: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        try:
            combined_score = core_metrics.get('combined_score', 0.0)
            ctr_bias = ctr_analysis.get('ctr_bias', 0.0)
            performance_tier = overall_assessment.get('performance_tier', 'POOR')
            
            # Performance-based recommendations
            if combined_score < 0.20:
                recommendations.append("Model performance is below acceptable threshold - consider complete redesign")
            elif combined_score < 0.30:
                recommendations.append("Model shows potential but needs improvement in feature engineering")
            
            # CTR bias recommendations
            if abs(ctr_bias) > 0.002:
                if ctr_bias > 0:
                    recommendations.append("Model over-predicts CTR - apply calibration or adjust threshold")
                else:
                    recommendations.append("Model under-predicts CTR - consider boosting positive examples")
            
            # Tier-specific recommendations
            if performance_tier == 'POOR':
                recommendations.append("Focus on data quality and feature selection improvements")
            elif performance_tier == 'FAIR':
                recommendations.append("Consider ensemble methods or hyperparameter optimization")
            elif performance_tier in ['GOOD', 'EXCELLENT']:
                recommendations.append("Model ready for production with monitoring")
                
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("Manual review recommended due to analysis errors")
        
        return recommendations
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024**3)  # Convert to GB
        except ImportError:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
            return 0.0
        except (ImportError, IndexError):
            return 0.0
    
    def save_analysis_report(self, analysis_result: Dict[str, Any], output_path: str) -> Optional[str]:
        """Save analysis report to JSON file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            serializable_result = convert_numpy_types(analysis_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
            return None
    
    def create_summary_csv(self, analysis_results: Dict[str, Any]) -> bool:
        """Create comprehensive summary CSV file"""
        try:
            logger.info("Creating summary CSV file")
            
            summary_data = []
            
            for model_name, analysis in analysis_results.items():
                if 'error' in analysis:
                    continue
                
                core_metrics = analysis.get('core_metrics', {})
                ctr_analysis = analysis.get('ctr_analysis', {})
                execution_metrics = analysis.get('execution_metrics', {})
                assessment = analysis.get('overall_assessment', {})
                
                row = {
                    'model_name': model_name,
                    'combined_score': round(core_metrics.get('combined_score', 0.0), 6),
                    'auc_score': round(core_metrics.get('auc', 0.5), 6),
                    'ap_score': round(core_metrics.get('ap', 0.0), 6),
                    'log_loss': round(core_metrics.get('log_loss', 1.0), 6),
                    'actual_ctr': round(ctr_analysis.get('actual_ctr', 0.0), 6),
                    'predicted_ctr': round(ctr_analysis.get('predicted_ctr', 0.0), 6),
                    'ctr_bias': round(ctr_analysis.get('ctr_bias', 0.0), 6),
                    'ctr_absolute_error': round(ctr_analysis.get('ctr_absolute_error', 0.0), 6),
                    'ctr_quality': ctr_analysis.get('ctr_quality', 'UNKNOWN'),
                    'execution_time_sec': round(execution_metrics.get('execution_time', 0.0), 2),
                    'memory_peak_gb': round(execution_metrics.get('memory_usage', 0.0), 2),
                    'gpu_utilization_pct': round(execution_metrics.get('gpu_utilization', 0.0), 1),
                    'performance_tier': assessment.get('performance_tier', 'POOR'),
                    'deployment_ready': assessment.get('deployment_ready', False),
                    'target_achievement': assessment.get('target_achievement', False),
                    'overall_score': round(assessment.get('overall_score', 0.0), 6)
                }
                
                summary_data.append(row)
            
            if not summary_data:
                logger.warning("No valid analysis data to create summary CSV")
                return False
            
            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Sort by combined score descending
            summary_df = summary_df.sort_values('combined_score', ascending=False)
            
            # Save to CSV with proper path
            csv_path = "results/summary.csv"
            os.makedirs("results", exist_ok=True)
            summary_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"Summary CSV created: {csv_path}")
            logger.info(f"Summary contains {len(summary_df)} models")
            
            return True
            
        except Exception as e:
            logger.error(f"Summary CSV creation failed: {e}")
            return False

def compare_model_performances(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare performance across multiple models"""
    try:
        if len(analysis_results) < 2:
            return {}
        
        logger.info("Comparing model performances")
        
        model_scores = {}
        for model_name, analysis in analysis_results.items():
            if 'error' not in analysis:
                core_metrics = analysis.get('core_metrics', {})
                model_scores[model_name] = {
                    'combined_score': core_metrics.get('combined_score', 0.0),
                    'auc': core_metrics.get('auc', 0.5),
                    'ap': core_metrics.get('ap', 0.0)
                }
        
        # Find best models by each metric
        best_models = {}
        for metric in ['combined_score', 'auc', 'ap']:
            valid_models = {name: scores[metric] for name, scores in model_scores.items() 
                          if not np.isnan(scores[metric])}
            if valid_models:
                best_model = max(valid_models.items(), key=lambda x: x[1])
                best_models[metric] = {
                    'model': best_model[0],
                    'score': best_model[1]
                }
        
        comparison_result = {
            'model_count': len(model_scores),
            'best_models': best_models,
            'model_scores': model_scores
        }
        
        logger.info(f"Model comparison completed for {len(model_scores)} models")
        return comparison_result
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        return {}