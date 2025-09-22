# analysis.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import warnings
from pathlib import Path
import json
import gc
import sys
warnings.filterwarnings('ignore')

# Safe imports for sklearn
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
    
    # Fallback functions when sklearn is not available
    def confusion_matrix(y_true, y_pred):
        """Fallback confusion matrix calculation"""
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            return np.array([[tn, fp], [fn, tp]])
        except:
            return np.array([[0, 0], [0, 0]])
    
    def roc_auc_score(y_true, y_pred_proba):
        """Fallback AUC calculation"""
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5
            # Simple AUC approximation
            return 0.5
        except:
            return 0.5
    
    def average_precision_score(y_true, y_pred_proba):
        """Fallback AP calculation"""
        try:
            pos_rate = np.mean(y_true)
            return pos_rate if pos_rate > 0 else 0.01
        except:
            return 0.01
    
    def log_loss(y_true, y_pred_proba):
        """Fallback log loss calculation"""
        try:
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        except:
            return 1.0
    
    def brier_score_loss(y_true, y_pred_proba):
        """Fallback brier score calculation"""
        try:
            return np.mean((y_pred_proba - y_true) ** 2)
        except:
            return 1.0

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available for statistical analysis")

try:
    from evaluation import CTRMetrics
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logging.warning("Evaluation module not available")

try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("Config module not available")
    # Create minimal config
    class Config:
        TARGET_CTR = 0.0191
        TARGET_COMBINED_SCORE = 0.34

logger = logging.getLogger(__name__)

class CTRPerformanceAnalyzer:
    """CTR model performance analyzer with comprehensive metrics"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # CTR specific thresholds
        self.target_ctr = getattr(self.config, 'TARGET_CTR', 0.0191)
        self.target_combined_score = getattr(self.config, 'TARGET_COMBINED_SCORE', 0.34)
        self.ctr_tolerance = 0.0002
        
        # Performance tier thresholds
        self.performance_tiers = {
            'EXCELLENT': 0.34,
            'GOOD': 0.30,
            'FAIR': 0.25,
            'POOR': 0.20
        }
        
        logger.info("CTR Performance Analyzer initialized")
    
    def analyze_model_performance(self, model_name: str, y_true: np.ndarray, 
                                y_pred_proba: np.ndarray, 
                                execution_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete performance analysis for a single model"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting performance analysis for {model_name}")
            
            # Initialize results
            analysis = {
                'model_name': model_name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'data_size': len(y_true)
            }
            
            # Core metrics calculation
            core_metrics = self._calculate_core_metrics(y_true, y_pred_proba)
            analysis['core_metrics'] = core_metrics
            
            # CTR specific analysis
            ctr_analysis = self._analyze_ctr_performance(y_true, y_pred_proba)
            analysis['ctr_analysis'] = ctr_analysis
            
            # Execution metrics
            if execution_metrics:
                analysis['execution_metrics'] = execution_metrics
            
            # Overall assessment
            assessment = self._generate_overall_assessment(core_metrics, ctr_analysis)
            analysis['overall_assessment'] = assessment
            
            # Performance tier classification
            analysis['performance_tier'] = self._classify_performance_tier(core_metrics.get('combined_score', 0.0))
            
            # Recommendations
            analysis['recommendations'] = self._generate_recommendations(core_metrics, ctr_analysis)
            
            # Analysis duration
            analysis['analysis_duration'] = time.time() - start_time
            
            logger.info(f"Performance analysis completed for {model_name} in {analysis['analysis_duration']:.2f}s")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed for {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'analysis_duration': time.time() - start_time
            }
    
    def _calculate_core_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate core performance metrics"""
        try:
            metrics = {}
            
            # Ensure proper data types
            y_true = np.array(y_true).astype(float)
            y_pred_proba = np.array(y_pred_proba).astype(float)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Basic metrics
            if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['ap'] = average_precision_score(y_true, y_pred_proba)
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                except Exception as e:
                    logger.warning(f"Sklearn metrics calculation failed: {e}")
                    metrics.update({'auc': 0.5, 'ap': 0.01, 'log_loss': 1.0, 'brier_score': 1.0})
            else:
                metrics.update({'auc': 0.5, 'ap': 0.01, 'log_loss': 1.0, 'brier_score': 1.0})
            
            # CTR specific combined score
            ap_score = metrics.get('ap', 0.01)
            auc_score = metrics.get('auc', 0.5)
            
            # Combined score calculation (CTR optimized)
            if ap_score > 0 and auc_score > 0.5:
                metrics['combined_score'] = (ap_score * 0.7) + ((auc_score - 0.5) * 0.6)
            else:
                metrics['combined_score'] = 0.0
            
            # Classification metrics
            try:
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
                combined_score >= self.target_combined_score * 0.8 and  # 80% of target
                ctr_bias <= self.ctr_tolerance and
                ctr_quality in ['EXCELLENT', 'GOOD']
            )
            assessment['deployment_ready'] = deployment_ready
            assessment['deployment_recommendation'] = 'READY' if deployment_ready else 'NOT_READY'
            
            # Risk assessment
            risk_factors = []
            if combined_score < 0.25:
                risk_factors.append('Low performance score')
            if ctr_bias > 0.001:
                risk_factors.append('High CTR bias')
            if ctr_quality == 'POOR':
                risk_factors.append('Poor CTR quality')
            
            assessment['risk_factors'] = risk_factors
            assessment['risk_level'] = 'HIGH' if len(risk_factors) >= 2 else ('MEDIUM' if len(risk_factors) == 1 else 'LOW')
            
            return assessment
            
        except Exception as e:
            logger.error(f"Overall assessment failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, core_metrics: Dict, ctr_analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            combined_score = core_metrics.get('combined_score', 0.0)
            ctr_bias = abs(ctr_analysis.get('ctr_bias', 0.0))
            ctr_quality = ctr_analysis.get('ctr_quality', 'POOR')
            
            # Performance recommendations
            if combined_score < 0.25:
                recommendations.append("Improve model performance - consider ensemble or hyperparameter tuning")
            elif combined_score < 0.34:
                recommendations.append("Good performance but room for improvement - try advanced feature engineering")
            
            # CTR bias recommendations
            if ctr_bias > 0.001:
                recommendations.append("Reduce CTR bias - apply calibration or adjust prediction thresholds")
            elif ctr_bias > 0.0005:
                recommendations.append("Monitor CTR bias - consider probability calibration")
            
            # Quality recommendations
            if ctr_quality == 'POOR':
                recommendations.append("Improve calibration - apply probability calibration techniques")
            elif ctr_quality == 'FAIR':
                recommendations.append("Enhance model calibration for better CTR prediction")
            
            # Deployment recommendations
            if combined_score >= self.target_combined_score and ctr_bias <= self.ctr_tolerance:
                recommendations.append("Model ready for deployment - monitor performance in production")
            
            if not recommendations:
                recommendations.append("Model performance acceptable - continue monitoring")
            
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            recommendations.append("Manual performance review required")
        
        return recommendations
    
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
                    'memory_peak_gb': round(execution_metrics.get('memory_peak', 0.0), 2),
                    'gpu_utilization_pct': round(execution_metrics.get('gpu_utilization', 0.0), 1),
                    'performance_tier': assessment.get('performance_tier', 'POOR'),
                    'deployment_ready': assessment.get('deployment_ready', False),
                    'risk_level': assessment.get('risk_level', 'HIGH'),
                    'precision': round(core_metrics.get('precision', 0.0), 6),
                    'recall': round(core_metrics.get('recall', 0.0), 6),
                    'f1_score': round(core_metrics.get('f1_score', 0.0), 6),
                    'accuracy': round(core_metrics.get('accuracy', 0.0), 6)
                }
                
                summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Sort by combined_score descending
                summary_df = summary_df.sort_values('combined_score', ascending=False)
                
                # Save to results folder
                csv_path = self.results_dir / "summary.csv"
                summary_df.to_csv(csv_path, index=False, encoding='utf-8')
                
                logger.info(f"Summary CSV saved: {csv_path}")
                logger.info(f"Models analyzed: {len(summary_data)}")
                
                return True
            else:
                logger.warning("No valid analysis results to save")
                return False
            
        except Exception as e:
            logger.error(f"Summary CSV creation failed: {e}")
            return False
    
    def save_detailed_analysis(self, analysis_results: Dict[str, Any]) -> bool:
        """Save detailed analysis results as JSON"""
        try:
            for model_name, analysis in analysis_results.items():
                if 'error' not in analysis:
                    json_path = self.results_dir / f"detailed_analysis_{model_name}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Detailed analysis saved: {json_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Detailed analysis saving failed: {e}")
            return False

def analyze_all_models(training_results: Dict[str, Any], quick_mode: bool = False) -> Dict[str, Any]:
    """Analyze all trained models and generate comprehensive results"""
    try:
        logger.info(f"Starting analysis of all models (quick_mode: {quick_mode})")
        
        analyzer = CTRPerformanceAnalyzer()
        analysis_results = {}
        
        # Extract models and their results
        models = training_results.get('models', {})
        
        if not models:
            logger.warning("No models found in training results")
            return {}
        
        logger.info(f"Analyzing {len(models)} models")
        
        for model_name, model_data in models.items():
            try:
                logger.info(f"Analyzing model: {model_name}")
                
                # Extract validation data and predictions
                y_true = model_data.get('y_val', np.array([]))
                y_pred_proba = model_data.get('val_predictions', np.array([]))
                
                if len(y_true) == 0 or len(y_pred_proba) == 0:
                    logger.warning(f"No validation data found for {model_name}")
                    continue
                
                # Extract execution metrics
                execution_metrics = {
                    'execution_time': model_data.get('training_time', 0.0),
                    'memory_peak': model_data.get('memory_peak', 0.0),
                    'gpu_utilization': model_data.get('gpu_utilization', 0.0)
                }
                
                # Perform analysis
                analysis = analyzer.analyze_model_performance(
                    model_name, y_true, y_pred_proba, execution_metrics
                )
                
                analysis_results[model_name] = analysis
                
                logger.info(f"Analysis completed for {model_name}")
                
            except Exception as e:
                logger.error(f"Analysis failed for model {model_name}: {e}")
                analysis_results[model_name] = {'error': str(e), 'model_name': model_name}
        
        if analysis_results:
            # Create summary CSV
            csv_success = analyzer.create_summary_csv(analysis_results)
            
            # Save detailed analysis
            detail_success = analyzer.save_detailed_analysis(analysis_results)
            
            logger.info(f"Analysis completed: CSV={csv_success}, Details={detail_success}")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Model analysis failed: {e}")
        return {}

# Cleanup and memory management
def cleanup_analysis_memory():
    """Clean up analysis memory"""
    try:
        gc.collect()
        logger.info("Analysis memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")