# visualization.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import warnings
from pathlib import Path
import io
import base64
warnings.filterwarnings('ignore')

# Safe imports for visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available for visualization")

try:
    import seaborn as sns
    if MATPLOTLIB_AVAILABLE:
        sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logging.warning("seaborn not available for enhanced visualization")

try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("plotly not available for interactive visualization")

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available for curve calculations")

from analysis import CTRPerformanceAnalyzer
from config import Config

logger = logging.getLogger(__name__)

class CTRVisualizationEngine:
    """CTR model performance visualization engine"""
    
    def __init__(self, config: Config = Config, style: str = 'default'):
        self.config = config
        self.style = style
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes for CTR visualization
        self.color_schemes = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4A261',
            'neutral': '#6C757D'
        }
        
        # Configure matplotlib if available
        if MATPLOTLIB_AVAILABLE:
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.size'] = 10
    
    def create_performance_dashboard(self, analysis_result: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create comprehensive performance dashboard"""
        logger.info(f"Creating performance dashboard for {analysis_result.get('model_name', 'Unknown')}")
        
        dashboard_data = {
            'model_name': analysis_result.get('model_name', 'Unknown'),
            'charts': {},
            'summary': {},
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            # 1. Core metrics summary chart
            dashboard_data['charts']['metrics_summary'] = self._create_metrics_summary_chart(analysis_result)
            
            # 2. CTR analysis charts
            dashboard_data['charts']['ctr_analysis'] = self._create_ctr_analysis_charts(analysis_result)
            
            # 3. Prediction distribution
            dashboard_data['charts']['prediction_distribution'] = self._create_prediction_distribution_chart(analysis_result)
            
            # 4. Calibration plot
            dashboard_data['charts']['calibration'] = self._create_calibration_plot(analysis_result)
            
            # 5. Performance curves (ROC, PR)
            if 'core_metrics' in analysis_result and len(analysis_result.get('y_true', [])) > 0:
                dashboard_data['charts']['performance_curves'] = self._create_performance_curves(analysis_result)
            
            # 6. Business impact charts
            if 'business_impact' in analysis_result:
                dashboard_data['charts']['business_impact'] = self._create_business_impact_charts(analysis_result)
            
            # 7. Segment analysis
            if 'segment_analysis' in analysis_result:
                dashboard_data['charts']['segment_analysis'] = self._create_segment_analysis_charts(analysis_result)
            
            # Generate summary
            dashboard_data['summary'] = self._generate_dashboard_summary(analysis_result)
            
            # Save dashboard if path provided
            if save_path:
                self._save_dashboard_html(dashboard_data, save_path)
            
            logger.info("Performance dashboard created successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return {
                'error': str(e),
                'model_name': analysis_result.get('model_name', 'Unknown')
            }
    
    def _create_metrics_summary_chart(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create metrics summary visualization"""
        try:
            core_metrics = analysis_result.get('core_metrics', {})
            
            metrics_data = {
                'Combined Score': core_metrics.get('combined_score', 0.0),
                'AP Score': core_metrics.get('ap', 0.0),
                'AUC': core_metrics.get('auc', 0.5),
                'CTR Score': analysis_result.get('ctr_analysis', {}).get('ctr_quality', 'poor')
            }
            
            # Convert CTR quality to numeric
            ctr_quality_map = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4, 'unknown': 0.0}
            if isinstance(metrics_data['CTR Score'], str):
                metrics_data['CTR Score'] = ctr_quality_map.get(metrics_data['CTR Score'], 0.0)
            
            chart_data = {'chart_type': 'metrics_summary', 'data': metrics_data}
            
            if MATPLOTLIB_AVAILABLE:
                chart_data['matplotlib_chart'] = self._create_matplotlib_metrics_summary(metrics_data)
            
            if PLOTLY_AVAILABLE:
                chart_data['plotly_chart'] = self._create_plotly_metrics_summary(metrics_data)
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Metrics summary chart creation failed: {e}")
            return {'error': str(e)}
    
    def _create_ctr_analysis_charts(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create CTR analysis visualization"""
        try:
            ctr_analysis = analysis_result.get('ctr_analysis', {})
            
            ctr_data = {
                'actual_ctr': ctr_analysis.get('actual_ctr', 0.0),
                'predicted_ctr': ctr_analysis.get('predicted_ctr', 0.0),
                'target_ctr': ctr_analysis.get('target_ctr', 0.0191),
                'ctr_bias': ctr_analysis.get('ctr_bias', 0.0),
                'ctr_absolute_error': ctr_analysis.get('ctr_absolute_error', 0.0)
            }
            
            chart_data = {'chart_type': 'ctr_analysis', 'data': ctr_data}
            
            if MATPLOTLIB_AVAILABLE:
                chart_data['matplotlib_chart'] = self._create_matplotlib_ctr_analysis(ctr_data)
            
            if PLOTLY_AVAILABLE:
                chart_data['plotly_chart'] = self._create_plotly_ctr_analysis(ctr_data)
            
            return chart_data
            
        except Exception as e:
            logger.error(f"CTR analysis chart creation failed: {e}")
            return {'error': str(e)}
    
    def _create_prediction_distribution_chart(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create prediction distribution visualization"""
        try:
            pred_dist = analysis_result.get('prediction_distribution', {})
            
            if 'histogram' in pred_dist:
                hist_data = pred_dist['histogram']
                chart_data = {
                    'chart_type': 'prediction_distribution',
                    'data': {
                        'counts': hist_data.get('counts', []),
                        'bin_edges': hist_data.get('bin_edges', []),
                        'statistics': pred_dist.get('statistics', {})
                    }
                }
                
                if MATPLOTLIB_AVAILABLE:
                    chart_data['matplotlib_chart'] = self._create_matplotlib_distribution(chart_data['data'])
                
                if PLOTLY_AVAILABLE:
                    chart_data['plotly_chart'] = self._create_plotly_distribution(chart_data['data'])
                
                return chart_data
            else:
                return {'error': 'No histogram data available'}
                
        except Exception as e:
            logger.error(f"Prediction distribution chart creation failed: {e}")
            return {'error': str(e)}
    
    def _create_calibration_plot(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create calibration plot"""
        try:
            calibration = analysis_result.get('calibration_analysis', {})
            
            if 'reliability_diagram' in calibration:
                reliability_data = calibration['reliability_diagram']
                chart_data = {
                    'chart_type': 'calibration',
                    'data': {
                        'prob_true': reliability_data.get('prob_true', []),
                        'prob_pred': reliability_data.get('prob_pred', []),
                        'perfect_calibration': reliability_data.get('perfect_calibration', []),
                        'calibration_error': calibration.get('calibration_error', 0.0)
                    }
                }
                
                if MATPLOTLIB_AVAILABLE:
                    chart_data['matplotlib_chart'] = self._create_matplotlib_calibration(chart_data['data'])
                
                if PLOTLY_AVAILABLE:
                    chart_data['plotly_chart'] = self._create_plotly_calibration(chart_data['data'])
                
                return chart_data
            else:
                # Basic calibration data
                basic_data = {
                    'bin_centers': calibration.get('bin_centers', []),
                    'bin_accuracies': calibration.get('bin_accuracies', []),
                    'calibration_error': calibration.get('calibration_error', float('nan'))
                }
                
                if basic_data['bin_centers']:
                    chart_data = {'chart_type': 'basic_calibration', 'data': basic_data}
                    
                    if MATPLOTLIB_AVAILABLE:
                        chart_data['matplotlib_chart'] = self._create_matplotlib_basic_calibration(basic_data)
                    
                    return chart_data
                else:
                    return {'error': 'Insufficient calibration data'}
                
        except Exception as e:
            logger.error(f"Calibration plot creation failed: {e}")
            return {'error': str(e)}
    
    def _create_performance_curves(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create ROC and PR curves"""
        try:
            # This would need y_true and y_pred_proba from the original data
            # For now, return placeholder
            return {
                'chart_type': 'performance_curves',
                'status': 'requires_original_data',
                'note': 'ROC and PR curves require original prediction data'
            }
            
        except Exception as e:
            logger.error(f"Performance curves creation failed: {e}")
            return {'error': str(e)}
    
    def _create_business_impact_charts(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create business impact visualization"""
        try:
            business_impact = analysis_result.get('business_impact', {})
            
            charts = {}
            
            # Lift analysis chart
            if 'lift_analysis' in business_impact:
                lift_data = business_impact['lift_analysis']
                charts['lift_chart'] = {
                    'data': lift_data,
                    'chart_type': 'lift_analysis'
                }
                
                if MATPLOTLIB_AVAILABLE:
                    charts['lift_chart']['matplotlib_chart'] = self._create_matplotlib_lift_chart(lift_data)
            
            # Threshold analysis
            if 'threshold_analysis' in business_impact:
                threshold_data = business_impact['threshold_analysis']
                charts['threshold_chart'] = {
                    'data': threshold_data,
                    'chart_type': 'threshold_analysis'
                }
                
                if MATPLOTLIB_AVAILABLE:
                    charts['threshold_chart']['matplotlib_chart'] = self._create_matplotlib_threshold_chart(threshold_data)
            
            return {'chart_type': 'business_impact', 'charts': charts}
            
        except Exception as e:
            logger.error(f"Business impact charts creation failed: {e}")
            return {'error': str(e)}
    
    def _create_segment_analysis_charts(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create segment analysis visualization"""
        try:
            segment_analysis = analysis_result.get('segment_analysis', {})
            
            charts = {}
            
            # Decile analysis
            if 'decile_analysis' in segment_analysis:
                decile_data = segment_analysis['decile_analysis']
                charts['decile_chart'] = {
                    'data': decile_data,
                    'chart_type': 'decile_analysis'
                }
                
                if MATPLOTLIB_AVAILABLE:
                    charts['decile_chart']['matplotlib_chart'] = self._create_matplotlib_decile_chart(decile_data)
            
            # Score segments
            if 'score_segments' in segment_analysis:
                segment_data = segment_analysis['score_segments']
                charts['segment_chart'] = {
                    'data': segment_data,
                    'chart_type': 'score_segments'
                }
                
                if MATPLOTLIB_AVAILABLE:
                    charts['segment_chart']['matplotlib_chart'] = self._create_matplotlib_segment_chart(segment_data)
            
            return {'chart_type': 'segment_analysis', 'charts': charts}
            
        except Exception as e:
            logger.error(f"Segment analysis charts creation failed: {e}")
            return {'error': str(e)}
    
    # Matplotlib chart creation methods
    def _create_matplotlib_metrics_summary(self, metrics_data: Dict[str, float]) -> str:
        """Create matplotlib metrics summary chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart
            metrics = list(metrics_data.keys())
            values = list(metrics_data.values())
            
            bars = ax1.bar(metrics, values, color=[self.color_schemes['primary'], 
                                                  self.color_schemes['secondary'],
                                                  self.color_schemes['accent'],
                                                  self.color_schemes['success']])
            
            ax1.set_title('Performance Metrics Summary')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values_radar = values + [values[0]]  # Complete the circle
            angles += angles[:1]
            
            ax2 = plt.subplot(122, projection='polar')
            ax2.plot(angles, values_radar, 'o-', linewidth=2, color=self.color_schemes['primary'])
            ax2.fill(angles, values_radar, alpha=0.25, color=self.color_schemes['primary'])
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(metrics)
            ax2.set_ylim(0, 1)
            ax2.set_title('Performance Radar Chart')
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib metrics summary creation failed: {e}")
            return ""
    
    def _create_matplotlib_ctr_analysis(self, ctr_data: Dict[str, float]) -> str:
        """Create matplotlib CTR analysis chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # CTR comparison bar chart
            ctr_values = [ctr_data['actual_ctr'], ctr_data['predicted_ctr'], ctr_data['target_ctr']]
            ctr_labels = ['Actual', 'Predicted', 'Target']
            colors = [self.color_schemes['primary'], self.color_schemes['secondary'], self.color_schemes['neutral']]
            
            bars = ax1.bar(ctr_labels, ctr_values, color=colors)
            ax1.set_title('CTR Comparison')
            ax1.set_ylabel('CTR Value')
            
            # Add value labels
            for bar, value in zip(bars, ctr_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom')
            
            # Bias analysis
            bias_value = ctr_data['ctr_bias']
            abs_error = ctr_data['ctr_absolute_error']
            
            ax2.barh(['CTR Bias', 'Absolute Error'], [bias_value, abs_error], 
                    color=[self.color_schemes['warning'] if bias_value != 0 else self.color_schemes['success'],
                           self.color_schemes['accent']])
            ax2.set_title('CTR Error Analysis')
            ax2.set_xlabel('Error Value')
            
            # Add tolerance line
            tolerance = 0.0002
            ax2.axvline(x=tolerance, color='red', linestyle='--', alpha=0.7, label=f'Tolerance ({tolerance})')
            ax2.axvline(x=-tolerance, color='red', linestyle='--', alpha=0.7)
            ax2.legend()
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib CTR analysis creation failed: {e}")
            return ""
    
    def _create_matplotlib_distribution(self, dist_data: Dict[str, Any]) -> str:
        """Create matplotlib prediction distribution chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            counts = dist_data['counts']
            bin_edges = dist_data['bin_edges']
            
            if len(counts) > 0 and len(bin_edges) > 1:
                bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                
                ax1.bar(bin_centers, counts, width=np.diff(bin_edges), 
                       alpha=0.7, color=self.color_schemes['primary'], edgecolor='black')
                ax1.set_title('Prediction Distribution')
                ax1.set_xlabel('Prediction Value')
                ax1.set_ylabel('Count')
            
            # Statistics summary
            stats = dist_data.get('statistics', {})
            if stats:
                stats_text = f"""Statistics:
Mean: {stats.get('mean', 0):.4f}
Std: {stats.get('std', 0):.4f}
Min: {stats.get('min', 0):.4f}
Max: {stats.get('max', 0):.4f}
Median: {stats.get('median', 0):.4f}"""
                
                ax2.text(0.1, 0.7, stats_text, transform=ax2.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax2.set_title('Distribution Statistics')
                ax2.axis('off')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib distribution creation failed: {e}")
            return ""
    
    def _create_matplotlib_calibration(self, cal_data: Dict[str, Any]) -> str:
        """Create matplotlib calibration plot"""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            prob_true = cal_data['prob_true']
            prob_pred = cal_data['prob_pred']
            
            if len(prob_true) > 0 and len(prob_pred) > 0:
                # Plot calibration curve
                ax.plot(prob_pred, prob_true, 'o-', color=self.color_schemes['primary'], 
                       linewidth=2, markersize=6, label='Model Calibration')
                
                # Perfect calibration line
                ax.plot([0, 1], [0, 1], '--', color=self.color_schemes['neutral'], 
                       linewidth=1, label='Perfect Calibration')
                
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title('Calibration Plot (Reliability Diagram)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                
                # Add calibration error text
                cal_error = cal_data.get('calibration_error', 0)
                ax.text(0.05, 0.95, f'Calibration Error: {cal_error:.4f}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib calibration creation failed: {e}")
            return ""
    
    def _create_matplotlib_basic_calibration(self, cal_data: Dict[str, Any]) -> str:
        """Create basic matplotlib calibration plot"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            bin_centers = cal_data['bin_centers']
            bin_accuracies = cal_data['bin_accuracies']
            
            if len(bin_centers) > 0 and len(bin_accuracies) > 0:
                ax.scatter(bin_centers, bin_accuracies, color=self.color_schemes['primary'], 
                          s=100, alpha=0.7, label='Observed')
                ax.plot([0, 1], [0, 1], '--', color=self.color_schemes['neutral'], 
                       linewidth=1, label='Perfect Calibration')
                
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Observed Frequency')
                ax.set_title('Basic Calibration Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                cal_error = cal_data.get('calibration_error', float('nan'))
                if not np.isnan(cal_error):
                    ax.text(0.05, 0.95, f'Calibration Error: {cal_error:.4f}', 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib basic calibration creation failed: {e}")
            return ""
    
    def _create_matplotlib_lift_chart(self, lift_data: Dict[str, Any]) -> str:
        """Create matplotlib lift chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            percentiles = []
            lifts = []
            ctrs = []
            
            for key, data in lift_data.items():
                if 'top_' in key and '%' in key:
                    percentile = key.replace('top_', '').replace('%', '')
                    percentiles.append(float(percentile))
                    lifts.append(data['lift'])
                    ctrs.append(data['ctr'])
            
            if percentiles:
                # Sort by percentile
                sorted_data = sorted(zip(percentiles, lifts, ctrs))
                percentiles, lifts, ctrs = zip(*sorted_data)
                
                # Lift chart
                ax.bar(range(len(percentiles)), lifts, color=self.color_schemes['accent'], alpha=0.7)
                ax.set_xlabel('Top Percentile')
                ax.set_ylabel('Lift')
                ax.set_title('Lift Analysis by Percentile')
                ax.set_xticks(range(len(percentiles)))
                ax.set_xticklabels([f'Top {p}%' for p in percentiles], rotation=45)
                
                # Add horizontal line at lift = 1
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (Lift=1)')
                ax.legend()
                
                # Add value labels
                for i, (lift, ctr) in enumerate(zip(lifts, ctrs)):
                    ax.text(i, lift + 0.05, f'{lift:.2f}\n({ctr:.3f})', 
                           ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib lift chart creation failed: {e}")
            return ""
    
    def _create_matplotlib_threshold_chart(self, threshold_data: Dict[str, Any]) -> str:
        """Create matplotlib threshold analysis chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            thresholds = []
            precisions = []
            recalls = []
            
            for key, data in threshold_data.items():
                if 'threshold_' in key:
                    threshold = float(key.replace('threshold_', ''))
                    thresholds.append(threshold)
                    precisions.append(data['precision'])
                    recalls.append(data['recall'])
            
            if thresholds:
                # Sort by threshold
                sorted_data = sorted(zip(thresholds, precisions, recalls))
                thresholds, precisions, recalls = zip(*sorted_data)
                
                # Precision-Recall vs Threshold
                ax1.plot(thresholds, precisions, 'o-', color=self.color_schemes['primary'], label='Precision')
                ax1.plot(thresholds, recalls, 's-', color=self.color_schemes['secondary'], label='Recall')
                ax1.set_xlabel('Threshold')
                ax1.set_ylabel('Score')
                ax1.set_title('Precision & Recall vs Threshold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_xscale('log')
                
                # F1 Score
                f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
                ax2.plot(thresholds, f1_scores, 'o-', color=self.color_schemes['accent'], linewidth=2)
                ax2.set_xlabel('Threshold')
                ax2.set_ylabel('F1 Score')
                ax2.set_title('F1 Score vs Threshold')
                ax2.grid(True, alpha=0.3)
                ax2.set_xscale('log')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib threshold chart creation failed: {e}")
            return ""
    
    def _create_matplotlib_decile_chart(self, decile_data: Dict[str, Any]) -> str:
        """Create matplotlib decile analysis chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            deciles = []
            actual_ctrs = []
            predicted_ctrs = []
            lifts = []
            
            for key, data in decile_data.items():
                if 'decile_' in key:
                    decile_num = int(key.replace('decile_', ''))
                    deciles.append(decile_num)
                    actual_ctrs.append(data['actual_ctr'])
                    predicted_ctrs.append(data['predicted_ctr'])
                    lifts.append(data['lift'])
            
            if deciles:
                # Sort by decile
                sorted_data = sorted(zip(deciles, actual_ctrs, predicted_ctrs, lifts))
                deciles, actual_ctrs, predicted_ctrs, lifts = zip(*sorted_data)
                
                # CTR by Decile
                x_pos = np.arange(len(deciles))
                width = 0.35
                
                ax1.bar(x_pos - width/2, actual_ctrs, width, label='Actual CTR', 
                       color=self.color_schemes['primary'], alpha=0.7)
                ax1.bar(x_pos + width/2, predicted_ctrs, width, label='Predicted CTR', 
                       color=self.color_schemes['secondary'], alpha=0.7)
                
                ax1.set_xlabel('Decile')
                ax1.set_ylabel('CTR')
                ax1.set_title('CTR by Decile')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([f'D{d}' for d in deciles])
                ax1.legend()
                
                # Lift by Decile
                ax2.bar(x_pos, lifts, color=self.color_schemes['accent'], alpha=0.7)
                ax2.set_xlabel('Decile')
                ax2.set_ylabel('Lift')
                ax2.set_title('Lift by Decile')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([f'D{d}' for d in deciles])
                ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
                ax2.legend()
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib decile chart creation failed: {e}")
            return ""
    
    def _create_matplotlib_segment_chart(self, segment_data: Dict[str, Any]) -> str:
        """Create matplotlib segment analysis chart"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            segments = []
            actual_ctrs = []
            predicted_ctrs = []
            sizes = []
            
            for segment, data in segment_data.items():
                segments.append(segment)
                actual_ctrs.append(data['actual_ctr'])
                predicted_ctrs.append(data['predicted_ctr'])
                sizes.append(data['size'])
            
            if segments:
                x_pos = np.arange(len(segments))
                width = 0.35
                
                ax.bar(x_pos - width/2, actual_ctrs, width, label='Actual CTR', 
                      color=self.color_schemes['primary'], alpha=0.7)
                ax.bar(x_pos + width/2, predicted_ctrs, width, label='Predicted CTR', 
                      color=self.color_schemes['secondary'], alpha=0.7)
                
                ax.set_xlabel('Score Segment')
                ax.set_ylabel('CTR')
                ax.set_title('CTR by Score Segment')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(segments, rotation=45)
                ax.legend()
                
                # Add size annotations
                for i, size in enumerate(sizes):
                    ax.text(i, max(actual_ctrs[i], predicted_ctrs[i]) + 0.001, 
                           f'n={size}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Matplotlib segment chart creation failed: {e}")
            return ""
    
    # Plotly chart creation methods (placeholders)
    def _create_plotly_metrics_summary(self, metrics_data: Dict[str, float]) -> Dict[str, Any]:
        """Create plotly metrics summary chart"""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available'}
        
        try:
            # Create plotly chart data structure
            return {
                'data': metrics_data,
                'chart_type': 'plotly_metrics_summary',
                'status': 'plotly_available'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _create_plotly_ctr_analysis(self, ctr_data: Dict[str, float]) -> Dict[str, Any]:
        """Create plotly CTR analysis chart"""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available'}
        
        try:
            return {
                'data': ctr_data,
                'chart_type': 'plotly_ctr_analysis',
                'status': 'plotly_available'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _create_plotly_distribution(self, dist_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create plotly distribution chart"""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available'}
        
        try:
            return {
                'data': dist_data,
                'chart_type': 'plotly_distribution',
                'status': 'plotly_available'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _create_plotly_calibration(self, cal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create plotly calibration chart"""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not available'}
        
        try:
            return {
                'data': cal_data,
                'chart_type': 'plotly_calibration',
                'status': 'plotly_available'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_dashboard_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard summary"""
        try:
            summary = {
                'overall_performance': analysis_result.get('overall_assessment', {}).get('performance_tier', 'unknown'),
                'combined_score': analysis_result.get('core_metrics', {}).get('combined_score', 0.0),
                'ctr_quality': analysis_result.get('ctr_analysis', {}).get('ctr_quality', 'unknown'),
                'deployment_ready': analysis_result.get('overall_assessment', {}).get('deployment_recommendation', 'not_ready') == 'ready',
                'key_strengths': [],
                'improvement_areas': []
            }
            
            # Identify strengths and areas for improvement
            combined_score = summary['combined_score']
            if combined_score >= 0.34:
                summary['key_strengths'].append('Excellent combined score performance')
            elif combined_score >= 0.30:
                summary['key_strengths'].append('Good combined score performance')
            
            ctr_error = analysis_result.get('ctr_analysis', {}).get('ctr_absolute_error', 1.0)
            if ctr_error <= 0.0002:
                summary['key_strengths'].append('Excellent CTR accuracy')
            elif ctr_error > 0.001:
                summary['improvement_areas'].append('CTR bias needs reduction')
            
            calibration_quality = analysis_result.get('calibration_analysis', {}).get('calibration_quality', 'unknown')
            if calibration_quality == 'excellent':
                summary['key_strengths'].append('Excellent probability calibration')
            elif calibration_quality in ['poor', 'unknown']:
                summary['improvement_areas'].append('Probability calibration needs improvement')
            
            return summary
            
        except Exception as e:
            logger.error(f"Dashboard summary generation failed: {e}")
            return {'error': str(e)}
    
    def _save_dashboard_html(self, dashboard_data: Dict[str, Any], save_path: str):
        """Save dashboard as HTML file"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>CTR Model Performance Dashboard - {dashboard_data['model_name']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                    .chart-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                    .summary {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>CTR Model Performance Dashboard</h1>
                    <h2>Model: {dashboard_data['model_name']}</h2>
                    <p>Generated: {dashboard_data['creation_timestamp']}</p>
                </div>
                
                <div class="summary">
                    <h3>Summary</h3>
                    <p>Performance Tier: {dashboard_data.get('summary', {}).get('overall_performance', 'Unknown')}</p>
                    <p>Combined Score: {dashboard_data.get('summary', {}).get('combined_score', 0.0):.4f}</p>
                    <p>CTR Quality: {dashboard_data.get('summary', {}).get('ctr_quality', 'Unknown')}</p>
                    <p>Deployment Ready: {dashboard_data.get('summary', {}).get('deployment_ready', False)}</p>
                </div>
            """
            
            # Add charts
            for chart_name, chart_data in dashboard_data.get('charts', {}).items():
                if isinstance(chart_data, dict) and 'matplotlib_chart' in chart_data:
                    html_content += f"""
                    <div class="chart-section">
                        <div class="chart-title">{chart_name.replace('_', ' ').title()}</div>
                        <img src="data:image/png;base64,{chart_data['matplotlib_chart']}" alt="{chart_name}">
                    </div>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Dashboard HTML saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Dashboard HTML saving failed: {e}")
    
    def create_model_comparison_chart(self, comparison_result: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> str:
        """Create model comparison visualization"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return ""
            
            models = comparison_result.get('models_compared', [])
            detailed_comparison = comparison_result.get('detailed_comparison', {})
            
            if not models or not detailed_comparison:
                return ""
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Combined scores comparison
            combined_scores = [detailed_comparison[model]['combined_score'] for model in models]
            ax1.bar(models, combined_scores, color=self.color_schemes['primary'], alpha=0.7)
            ax1.set_title('Combined Score Comparison')
            ax1.set_ylabel('Combined Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # CTR scores comparison
            ctr_scores = [detailed_comparison[model]['ctr_score'] for model in models]
            ax2.bar(models, ctr_scores, color=self.color_schemes['secondary'], alpha=0.7)
            ax2.set_title('CTR Score Comparison')
            ax2.set_ylabel('CTR Score')
            ax2.tick_params(axis='x', rotation=45)
            
            # AP scores comparison
            ap_scores = [detailed_comparison[model]['ap'] for model in models]
            ax3.bar(models, ap_scores, color=self.color_schemes['accent'], alpha=0.7)
            ax3.set_title('AP Score Comparison')
            ax3.set_ylabel('AP Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # Composite scores radar chart
            metrics = ['combined_score', 'ctr_score', 'ap', 'auc', 'calibration_score']
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            ax4 = plt.subplot(224, projection='polar')
            
            colors = [self.color_schemes['primary'], self.color_schemes['secondary'], 
                     self.color_schemes['accent'], self.color_schemes['success']]
            
            for i, model in enumerate(models[:4]):  # Limit to 4 models for readability
                values = [detailed_comparison[model].get(metric, 0) for metric in metrics]
                values += values[:1]
                
                color = colors[i % len(colors)]
                ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
                ax4.fill(angles, values, alpha=0.25, color=color)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics)
            ax4.set_ylim(0, 1)
            ax4.set_title('Model Performance Radar')
            ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model comparison chart saved: {save_path}")
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Model comparison chart creation failed: {e}")
            return ""

def create_performance_dashboard(analysis_result, save_path=None):
    """Standalone function to create performance dashboard"""
    visualizer = CTRVisualizationEngine()
    return visualizer.create_performance_dashboard(analysis_result, save_path)

def create_model_comparison_chart(comparison_result, save_path=None):
    """Standalone function to create model comparison chart"""
    visualizer = CTRVisualizationEngine()
    return visualizer.create_model_comparison_chart(comparison_result, save_path)