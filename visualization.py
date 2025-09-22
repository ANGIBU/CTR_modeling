# visualization.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import warnings
from pathlib import Path
import json
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Safe imports for visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.font_manager as fm
    
    # Korean font setup for better text rendering
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available for visualization")

try:
    import seaborn as sns
    if MATPLOTLIB_AVAILABLE:
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logging.warning("seaborn not available for enhanced visualization")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not available for PDF generation")

logger = logging.getLogger(__name__)

class CTRVisualizationEngine:
    """CTR model performance visualization engine with publication quality charts"""
    
    def __init__(self, style: str = 'professional'):
        self.style = style
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Color schemes optimized for CTR analysis
        self.colors = {
            'primary': '#4472C4',      # Professional blue (matching sample)
            'secondary': '#70AD47',     # Success green
            'accent': '#E15759',        # Warning red
            'neutral': '#A5A5A5',       # Neutral gray
            'background': '#F8F9FA',    # Light background
            'text': '#333333'           # Dark text
        }
        
        # Configure matplotlib for high quality output
        if MATPLOTLIB_AVAILABLE:
            plt.rcParams.update({
                'figure.figsize': (12, 8),
                'figure.dpi': 100,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.facecolor': 'white',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'font.size': 11,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10
            })
        
        logger.info("CTR Visualization Engine initialized")
    
    def create_model_performance_chart(self, summary_df: pd.DataFrame, save_path: Optional[str] = None) -> bool:
        """Create model performance overview chart (Feature Importance style)"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available, skipping chart creation")
                return False
            
            logger.info("Creating model performance chart")
            
            # Prepare data
            models = summary_df['model_name'].tolist()
            scores = summary_df['combined_score'].tolist()
            
            # Sort by score for better visualization
            sorted_data = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
            models_sorted, scores_sorted = zip(*sorted_data) if sorted_data else ([], [])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.8)))
            
            # Create horizontal bars (matching sample style)
            bars = ax.barh(models_sorted, scores_sorted, 
                          color=self.colors['primary'], 
                          alpha=0.8, 
                          height=0.6)
            
            # Add value labels on the right side of bars (matching sample)
            for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
                ax.text(bar.get_width() + max(scores_sorted) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{score:.4f}', 
                       va='center', ha='left',
                       fontsize=11, fontweight='bold')
            
            # Styling (matching sample clean style)
            ax.set_title('Model Performance Overview - Combined Score', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Combined Score', fontsize=12)
            ax.set_xlim(0, max(scores_sorted) * 1.15 if scores_sorted else 1)
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add performance tier indicators
            if scores_sorted:
                excellent_line = 0.34
                good_line = 0.30
                
                if max(scores_sorted) > excellent_line:
                    ax.axvline(x=excellent_line, color=self.colors['secondary'], 
                              linestyle='--', alpha=0.7, label='Excellent (0.34+)')
                
                if max(scores_sorted) > good_line:
                    ax.axvline(x=good_line, color=self.colors['accent'], 
                              linestyle='--', alpha=0.7, label='Good (0.30+)')
                
                ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.results_dir / "model_performance.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Model performance chart saved: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model performance chart creation failed: {e}")
            return False
    
    def create_ctr_analysis_chart(self, summary_df: pd.DataFrame, save_path: Optional[str] = None) -> bool:
        """Create CTR analysis chart showing bias and quality"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available, skipping chart creation")
                return False
            
            logger.info("Creating CTR analysis chart")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('CTR Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # Chart 1: CTR Bias Analysis
            models = summary_df['model_name']
            ctr_bias = summary_df['ctr_bias']
            colors_bias = [self.colors['accent'] if abs(bias) > 0.001 else self.colors['primary'] for bias in ctr_bias]
            
            bars1 = ax1.barh(models, ctr_bias, color=colors_bias, alpha=0.8)
            ax1.set_title('CTR Bias by Model', fontsize=12, fontweight='bold')
            ax1.set_xlabel('CTR Bias (Predicted - Actual)')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.axvline(x=0.001, color=self.colors['accent'], linestyle='--', alpha=0.7, label='Warning (+0.001)')
            ax1.axvline(x=-0.001, color=self.colors['accent'], linestyle='--', alpha=0.7, label='Warning (-0.001)')
            ax1.legend()
            
            # Add bias values
            for bar, bias in zip(bars1, ctr_bias):
                ax1.text(bias + (max(ctr_bias) - min(ctr_bias)) * 0.02 if bias >= 0 else bias - (max(ctr_bias) - min(ctr_bias)) * 0.02,
                        bar.get_y() + bar.get_height()/2,
                        f'{bias:.6f}', va='center', ha='left' if bias >= 0 else 'right', fontsize=9)
            
            # Chart 2: CTR Quality Distribution
            quality_counts = summary_df['ctr_quality'].value_counts()
            colors_quality = [self.colors['secondary'] if q in ['EXCELLENT', 'GOOD'] else 
                             (self.colors['primary'] if q == 'FAIR' else self.colors['accent']) 
                             for q in quality_counts.index]
            
            wedges, texts, autotexts = ax2.pie(quality_counts.values, labels=quality_counts.index, 
                                              colors=colors_quality, autopct='%1.1f%%', startangle=90)
            ax2.set_title('CTR Quality Distribution', fontsize=12, fontweight='bold')
            
            # Chart 3: Actual vs Predicted CTR
            ax3.scatter(summary_df['actual_ctr'], summary_df['predicted_ctr'], 
                       c=self.colors['primary'], alpha=0.7, s=100)
            
            # Perfect prediction line
            min_ctr = min(min(summary_df['actual_ctr']), min(summary_df['predicted_ctr']))
            max_ctr = max(max(summary_df['actual_ctr']), max(summary_df['predicted_ctr']))
            ax3.plot([min_ctr, max_ctr], [min_ctr, max_ctr], 'r--', alpha=0.7, label='Perfect Prediction')
            
            ax3.set_xlabel('Actual CTR')
            ax3.set_ylabel('Predicted CTR')
            ax3.set_title('Actual vs Predicted CTR', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add model labels
            for i, model in enumerate(summary_df['model_name']):
                ax3.annotate(model, (summary_df['actual_ctr'].iloc[i], summary_df['predicted_ctr'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
            
            # Chart 4: Performance Tier Summary
            tier_counts = summary_df['performance_tier'].value_counts()
            colors_tier = [self.colors['secondary'] if tier == 'EXCELLENT' else
                          (self.colors['primary'] if tier == 'GOOD' else
                           (self.colors['neutral'] if tier == 'FAIR' else self.colors['accent']))
                          for tier in tier_counts.index]
            
            bars4 = ax4.bar(tier_counts.index, tier_counts.values, color=colors_tier, alpha=0.8)
            ax4.set_title('Performance Tier Distribution', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Number of Models')
            
            # Add count labels on bars
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.results_dir / "ctr_analysis.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"CTR analysis chart saved: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"CTR analysis chart creation failed: {e}")
            return False
    
    def create_execution_summary_chart(self, summary_df: pd.DataFrame, save_path: Optional[str] = None) -> bool:
        """Create execution performance summary chart"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available, skipping chart creation")
                return False
            
            logger.info("Creating execution summary chart")
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Execution Performance Summary', fontsize=16, fontweight='bold')
            
            models = summary_df['model_name']
            
            # Chart 1: Execution Time (Feature Importance style)
            exec_times = summary_df['execution_time_sec']
            bars1 = ax1.barh(models, exec_times, color=self.colors['primary'], alpha=0.8)
            ax1.set_title('Training Time by Model', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Execution Time (seconds)')
            
            # Add time labels
            for bar, time_val in zip(bars1, exec_times):
                ax1.text(bar.get_width() + max(exec_times) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{time_val:.1f}s', va='center', ha='left', fontweight='bold')
            
            # Chart 2: Memory Usage
            memory_usage = summary_df['memory_peak_gb']
            bars2 = ax2.barh(models, memory_usage, color=self.colors['secondary'], alpha=0.8)
            ax2.set_title('Peak Memory Usage by Model', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Memory Usage (GB)')
            
            # Add memory labels
            for bar, mem_val in zip(bars2, memory_usage):
                ax2.text(bar.get_width() + max(memory_usage) * 0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{mem_val:.1f}GB', va='center', ha='left', fontweight='bold')
            
            # Chart 3: GPU Utilization
            gpu_usage = summary_df['gpu_utilization_pct']
            colors_gpu = [self.colors['secondary'] if gpu > 50 else self.colors['accent'] for gpu in gpu_usage]
            bars3 = ax3.barh(models, gpu_usage, color=colors_gpu, alpha=0.8)
            ax3.set_title('GPU Utilization by Model', fontsize=12, fontweight='bold')
            ax3.set_xlabel('GPU Utilization (%)')
            ax3.set_xlim(0, 100)
            
            # Add GPU labels
            for bar, gpu_val in zip(bars3, gpu_usage):
                ax3.text(bar.get_width() + 2,
                        bar.get_y() + bar.get_height()/2,
                        f'{gpu_val:.1f}%', va='center', ha='left', fontweight='bold')
            
            # Style all charts
            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                save_path = self.results_dir / "execution_summary.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Execution summary chart saved: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Execution summary chart creation failed: {e}")
            return False
    
    def create_comprehensive_pdf_report(self, summary_df: pd.DataFrame, analysis_results: Dict[str, Any]) -> bool:
        """Create comprehensive PDF report with all analysis results"""
        try:
            if not REPORTLAB_AVAILABLE:
                logger.warning("ReportLab not available, creating simple HTML report instead")
                return self._create_html_report(summary_df, analysis_results)
            
            logger.info("Creating comprehensive PDF report")
            
            pdf_path = self.results_dir / "comprehensive_report.pdf"
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            
            # Build story
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("CTR Modeling Performance Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            
            # Create summary table
            summary_table_data = [['Model', 'Combined Score', 'CTR Quality', 'Performance Tier', 'Deployment Ready']]
            
            for _, row in summary_df.iterrows():
                summary_table_data.append([
                    str(row['model_name']),
                    f"{row['combined_score']:.4f}",
                    str(row['ctr_quality']),
                    str(row['performance_tier']),
                    'Yes' if row['deployment_ready'] else 'No'
                ])
            
            summary_table = Table(summary_table_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Key Findings
            story.append(Paragraph("Key Findings", styles['Heading2']))
            
            best_model = summary_df.loc[summary_df['combined_score'].idxmax()]
            worst_model = summary_df.loc[summary_df['combined_score'].idxmin()]
            
            findings = [
                f"• Best performing model: {best_model['model_name']} (Score: {best_model['combined_score']:.4f})",
                f"• Lowest performing model: {worst_model['model_name']} (Score: {worst_model['combined_score']:.4f})",
                f"• Models ready for deployment: {len(summary_df[summary_df['deployment_ready']])} out of {len(summary_df)}",
                f"• Average CTR bias: {summary_df['ctr_bias'].abs().mean():.6f}",
                f"• Models with excellent CTR quality: {len(summary_df[summary_df['ctr_quality'] == 'EXCELLENT'])}"
            ]
            
            for finding in findings:
                story.append(Paragraph(finding, styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Recommendations", styles['Heading2']))
            
            recommendations = [
                "• Focus on the best performing model for production deployment",
                "• Monitor CTR bias closely to ensure prediction accuracy",
                "• Consider ensemble methods to improve overall performance",
                "• Implement regular model retraining schedule",
                "• Set up monitoring for model drift detection"
            ]
            
            for recommendation in recommendations:
                story.append(Paragraph(recommendation, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Comprehensive PDF report saved: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF report creation failed: {e}")
            return self._create_html_report(summary_df, analysis_results)
    
    def _create_html_report(self, summary_df: pd.DataFrame, analysis_results: Dict[str, Any]) -> bool:
        """Create HTML report as fallback"""
        try:
            logger.info("Creating HTML report as fallback")
            
            html_path = self.results_dir / "comprehensive_report.html"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>CTR Modeling Performance Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #2c3e50; text-align: center; }}
                    h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .highlight {{ background-color: #e8f5e8; }}
                    .warning {{ background-color: #fff3cd; }}
                    .danger {{ background-color: #f8d7da; }}
                    ul {{ margin-left: 20px; }}
                    li {{ margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>CTR Modeling Performance Analysis Report</h1>
                
                <h2>Executive Summary</h2>
                {summary_df.to_html(index=False, classes='summary-table', escape=False)}
                
                <h2>Key Findings</h2>
                <ul>
                    <li>Total models analyzed: {len(summary_df)}</li>
                    <li>Best performing model: {summary_df.loc[summary_df['combined_score'].idxmax(), 'model_name']} 
                        (Score: {summary_df['combined_score'].max():.4f})</li>
                    <li>Models ready for deployment: {len(summary_df[summary_df['deployment_ready']])} out of {len(summary_df)}</li>
                    <li>Average CTR bias: {summary_df['ctr_bias'].abs().mean():.6f}</li>
                </ul>
                
                <h2>Recommendations</h2>
                <ul>
                    <li>Focus on the best performing model for production deployment</li>
                    <li>Monitor CTR bias closely to ensure prediction accuracy</li>
                    <li>Consider ensemble methods to improve overall performance</li>
                    <li>Implement regular model retraining schedule</li>
                </ul>
                
                <h2>Generated Charts</h2>
                <p>The following visualization files have been generated:</p>
                <ul>
                    <li><strong>model_performance.png</strong> - Model performance comparison</li>
                    <li><strong>ctr_analysis.png</strong> - CTR bias and quality analysis</li>
                    <li><strong>execution_summary.png</strong> - Training time and resource usage</li>
                </ul>
                
                <p><em>Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </body>
            </html>
            """
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved: {html_path}")
            return True
            
        except Exception as e:
            logger.error(f"HTML report creation failed: {e}")
            return False
    
    def generate_all_visualizations(self, analysis_results: Dict[str, Any]) -> bool:
        """Generate all visualizations from analysis results"""
        try:
            logger.info("Generating all visualizations")
            
            # Load summary CSV
            csv_path = self.results_dir / "summary.csv"
            if not csv_path.exists():
                logger.error("Summary CSV not found, cannot create visualizations")
                return False
            
            summary_df = pd.read_csv(csv_path)
            
            success_count = 0
            
            # Create model performance chart
            if self.create_model_performance_chart(summary_df):
                success_count += 1
            
            # Create CTR analysis chart
            if self.create_ctr_analysis_chart(summary_df):
                success_count += 1
            
            # Create execution summary chart
            if self.create_execution_summary_chart(summary_df):
                success_count += 1
            
            # Create comprehensive report
            if self.create_comprehensive_pdf_report(summary_df, analysis_results):
                success_count += 1
            
            logger.info(f"Visualization generation completed: {success_count}/4 successful")
            
            return success_count >= 3  # At least 3 out of 4 should succeed
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return False

def create_all_visualizations(analysis_results: Dict[str, Any]) -> bool:
    """Create all visualizations for CTR analysis results"""
    try:
        logger.info("Starting visualization creation process")
        
        visualizer = CTRVisualizationEngine()
        success = visualizer.generate_all_visualizations(analysis_results)
        
        if success:
            logger.info("All visualizations created successfully")
            logger.info("Results saved in 'results' folder:")
            logger.info("  - summary.csv (comprehensive metrics)")
            logger.info("  - model_performance.png (performance comparison)")
            logger.info("  - ctr_analysis.png (CTR bias and quality)")
            logger.info("  - execution_summary.png (resource usage)")
            logger.info("  - comprehensive_report.pdf/html (full report)")
        else:
            logger.warning("Some visualizations failed to create")
        
        return success
        
    except Exception as e:
        logger.error(f"Visualization creation process failed: {e}")
        return False