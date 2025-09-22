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
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
                })
                return metrics
            
            # Primary CTR metrics
            try:
                auc_score = roc_auc_score(y_true, y_pred_proba)
                metrics['auc'] = float(auc_score)
            except Exception:
                metrics['auc'] = 0.5
            
            try:
                ap_score = average_precision_score(y_true, y_pred_proba)
                metrics['ap'] = float(ap_score)
            except Exception:
                metrics['ap'] = 0.0
            
            try:
                ll_score = log_loss(y_true, y_pred_proba, eps=1e-15)
                metrics['log_loss'] = float(ll_score)
            except Exception:
                metrics['log_loss'] = 1.0
            
            # Combined score calculation (optimized for CTR)
            if metrics['ap'] > 0 and metrics['auc'] > 0.5:
                metrics['combined_score'] = (metrics['ap'] * 0.7) + ((metrics['auc'] - 0.5) * 0.6)
            else:
                metrics['combined_score'] = 0.0
            
            # Classification metrics
            try:
                y_pred = (y_pred_proba > 0.5).astype(int)
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    total = tp + tn + fp + fn
                    if total > 0:
                        metrics['accuracy'] = (tp + tn) / total
                        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        
                        if metrics['precision'] + metrics['recall'] > 0:
                            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
                        else:
                            metrics['f1_score'] = 0.0
                    else:
                        metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1_score': 0.0})
                else:
                    metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1_score': 0.0})
            except Exception as e:
                logger.warning(f"Classification metrics calculation failed: {e}")
                metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'f1_score': 0.0})
            
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
    
    def _generate_overall_assessment(self, core_metrics: Dict, ctr_analysis: Dict) -> Dict[str, Any]:
        """Generate overall model assessment"""
        try:
            assessment = {}
            
            combined_score = core_metrics.get('combined_score', 0.0)
            ctr_quality = ctr_analysis.get('ctr_quality', 'POOR')
            ctr_bias = abs(ctr_analysis.get('ctr_bias', 1.0))
            
            # Performance tier
            performance_tier = self._classify_performance_tier(combined_score)
            assessment['performance_tier'] = performance_tier
            
            # Deployment readiness
            deployment_ready = (
                combined_score >= self.target_combined_score and 
                ctr_bias <= self.ctr_tolerance and
                ctr_quality in ['EXCELLENT', 'GOOD']
            )
            assessment['deployment_ready'] = deployment_ready
            
            # Target achievement
            target_achievement = combined_score >= self.target_combined_score
            assessment['target_achievement'] = target_achievement
            
            # Overall score
            ctr_score = 1.0 - min(ctr_bias / 0.001, 1.0)  # Normalize CTR bias
            overall_score = (combined_score * 0.8) + (ctr_score * 0.2)
            assessment['overall_score'] = float(overall_score)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Overall assessment failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, core_metrics: Dict, ctr_analysis: Dict, 
                                assessment: Dict) -> List[str]:
        """Generate performance-based recommendations"""
        recommendations = []
        
        try:
            combined_score = core_metrics.get('combined_score', 0.0)
            ctr_bias = abs(ctr_analysis.get('ctr_bias', 1.0))
            performance_tier = assessment.get('performance_tier', 'POOR')
            
            # Score-based recommendations
            if combined_score < 0.15:
                recommendations.append("Model performance is below acceptable threshold - consider complete redesign")
            elif combined_score < 0.25:
                recommendations.append("Model needs improvement - try different algorithms or feature engineering")
            elif combined_score < 0.30:
                recommendations.append("Model showing progress - fine-tune hyperparameters")
            else:
                recommendations.append("Model performance is strong - ready for production consideration")
            
            # CTR-specific recommendations
            if ctr_bias > self.ctr_tolerance:
                if ctr_analysis.get('ctr_bias', 0) > 0:
                    recommendations.append("Model over-predicts CTR - apply calibration or adjust threshold")
                else:
                    recommendations.append("Model under-predicts CTR - review feature engineering")
            
            # Deployment recommendations
            if combined_score >= self.target_combined_score and ctr_bias <= self.ctr_tolerance:
                recommendations.append("Model ready for deployment - monitor performance in production")
            
            if not recommendations:
                recommendations.append("Model performance acceptable - continue monitoring")
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            recommendations.append("Manual performance review required")
        
        return recommendations
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except:
            return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.utilization() / 100.0
        except:
            return 0.0
        return 0.0
    
    def save_analysis_report(self, analysis_result: Dict[str, Any], 
                           output_path: str) -> Optional[str]:
        """Save analysis report to JSON file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
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
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                summary_path = "output/performance_summary.csv"
                os.makedirs("output", exist_ok=True)
                df.to_csv(summary_path, index=False)
                logger.info(f"Summary CSV created: {summary_path}")
                return True
            else:
                logger.warning("No valid analysis results to create summary")
                return False
                
        except Exception as e:
            logger.error(f"Summary CSV creation failed: {e}")
            return False


def compare_model_performances(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare performance across multiple models
    
    Args:
        analysis_results: Dictionary containing analysis results for each model
        
    Returns:
        Dictionary containing comparison results and rankings
    """
    try:
        logger.info("Starting model performance comparison")
        
        if not analysis_results:
            logger.warning("No analysis results provided for comparison")
            return {}
        
        # Filter out error results
        valid_results = {
            name: result for name, result in analysis_results.items() 
            if 'error' not in result
        }
        
        if not valid_results:
            logger.warning("No valid analysis results for comparison")
            return {}
        
        if len(valid_results) < 2:
            logger.info("Only one model available, no comparison needed")
            return {
                'comparison_type': 'single_model',
                'models_count': len(valid_results),
                'single_model': list(valid_results.keys())[0]
            }
        
        logger.info(f"Comparing {len(valid_results)} models")
        
        # Extract key metrics for comparison
        comparison_data = []
        
        for model_name, analysis in valid_results.items():
            core_metrics = analysis.get('core_metrics', {})
            ctr_analysis = analysis.get('ctr_analysis', {})
            assessment = analysis.get('overall_assessment', {})
            
            model_data = {
                'model_name': model_name,
                'combined_score': core_metrics.get('combined_score', 0.0),
                'auc_score': core_metrics.get('auc', 0.5),
                'ap_score': core_metrics.get('ap', 0.0),
                'log_loss': core_metrics.get('log_loss', 1.0),
                'ctr_absolute_error': ctr_analysis.get('ctr_absolute_error', 1.0),
                'ctr_quality': ctr_analysis.get('ctr_quality', 'POOR'),
                'performance_tier': assessment.get('performance_tier', 'POOR'),
                'deployment_ready': assessment.get('deployment_ready', False),
                'overall_score': assessment.get('overall_score', 0.0)
            }
            
            comparison_data.append(model_data)
        
        # Create comparison dataframe
        df = pd.DataFrame(comparison_data)
        
        # Rankings
        rankings = {}
        
        # Combined score ranking (higher is better)
        df_combined = df.sort_values('combined_score', ascending=False)
        rankings['combined_score'] = df_combined[['model_name', 'combined_score']].to_dict('records')
        
        # AUC ranking (higher is better)
        df_auc = df.sort_values('auc_score', ascending=False)
        rankings['auc_score'] = df_auc[['model_name', 'auc_score']].to_dict('records')
        
        # AP ranking (higher is better)
        df_ap = df.sort_values('ap_score', ascending=False)
        rankings['ap_score'] = df_ap[['model_name', 'ap_score']].to_dict('records')
        
        # CTR error ranking (lower is better)
        df_ctr = df.sort_values('ctr_absolute_error', ascending=True)
        rankings['ctr_accuracy'] = df_ctr[['model_name', 'ctr_absolute_error']].to_dict('records')
        
        # Overall score ranking (higher is better)
        df_overall = df.sort_values('overall_score', ascending=False)
        rankings['overall_score'] = df_overall[['model_name', 'overall_score']].to_dict('records')
        
        # Best models by metric
        best_models = {
            'combined_score': {
                'model': df_combined.iloc[0]['model_name'],
                'score': df_combined.iloc[0]['combined_score']
            },
            'auc_score': {
                'model': df_auc.iloc[0]['model_name'],
                'score': df_auc.iloc[0]['auc_score']
            },
            'ap_score': {
                'model': df_ap.iloc[0]['model_name'],
                'score': df_ap.iloc[0]['ap_score']
            },
            'ctr_accuracy': {
                'model': df_ctr.iloc[0]['model_name'],
                'score': df_ctr.iloc[0]['ctr_absolute_error']
            },
            'overall_score': {
                'model': df_overall.iloc[0]['model_name'],
                'score': df_overall.iloc[0]['overall_score']
            }
        }
        
        # Performance statistics
        performance_stats = {
            'models_count': len(valid_results),
            'deployment_ready_count': sum(1 for data in comparison_data if data['deployment_ready']),
            'avg_combined_score': float(df['combined_score'].mean()),
            'max_combined_score': float(df['combined_score'].max()),
            'min_combined_score': float(df['combined_score'].min()),
            'avg_ctr_error': float(df['ctr_absolute_error'].mean()),
            'performance_tier_distribution': df['performance_tier'].value_counts().to_dict()
        }
        
        # Model recommendations
        recommendations = []
        
        best_overall = df_overall.iloc[0]
        if best_overall['deployment_ready']:
            recommendations.append(f"Recommended for deployment: {best_overall['model_name']} (Overall Score: {best_overall['overall_score']:.4f})")
        else:
            recommendations.append(f"Best performing model: {best_overall['model_name']} - requires further optimization before deployment")
        
        if performance_stats['deployment_ready_count'] > 1:
            recommendations.append(f"{performance_stats['deployment_ready_count']} models are deployment-ready - consider ensemble approach")
        elif performance_stats['deployment_ready_count'] == 0:
            recommendations.append("No models currently meet deployment criteria - continue optimization")
        
        comparison_result = {
            'comparison_type': 'multi_model',
            'models_count': len(valid_results),
            'model_names': list(valid_results.keys()),
            'rankings': rankings,
            'best_models': best_models,
            'performance_stats': performance_stats,
            'recommendations': recommendations,
            'comparison_dataframe': df.to_dict('records'),
            'comparison_timestamp': time.time()
        }
        
        logger.info(f"Model comparison completed for {len(valid_results)} models")
        logger.info(f"Best overall model: {best_models['overall_score']['model']} (Score: {best_models['overall_score']['score']:.4f})")
        
        return comparison_result
        
    except Exception as e:
        logger.error(f"Model performance comparison failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return {
            'comparison_type': 'error',
            'error': str(e),
            'comparison_timestamp': time.time()
        }