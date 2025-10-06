# visualization.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import warnings
from pathlib import Path
from datetime import datetime
import gc
import os
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
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

logger = logging.getLogger(__name__)

class CTRVisualizationEngine:
    """CTR model performance visualization engine with professional quality output"""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.figures = {}
        self.performance_data = []
        self.created_png_files = []
        
        if MATPLOTLIB_AVAILABLE:
            self.colors = {
                'primary_gradient': plt.cm.Blues(np.linspace(0.4, 0.9, 10)),
                'secondary_gradient': plt.cm.Oranges(np.linspace(0.4, 0.9, 10)),
                'success': '#27ae60',
                'warning': '#f39c12', 
                'danger': '#e74c3c',
                'info': '#3498db',
                'neutral': '#95a5a6'
            }
        else:
            self.colors = {}
        
        logger.info("CTR Visualization Engine initialized with professional styling")
    
    def load_summary_data(self) -> Optional[pd.DataFrame]:
        """Load summary CSV data with fallback"""
        try:
            csv_path = self.results_dir / "summary.csv"
            if not csv_path.exists():
                logger.warning("Summary CSV not found, will create basic visualizations")
                return None
            
            summary_df = pd.read_csv(csv_path)
            self.performance_data = summary_df.to_dict('records')
            logger.info(f"Loaded {len(summary_df)} models from summary CSV")
            return summary_df
            
        except Exception as e:
            logger.error(f"Failed to load summary data: {e}")
            return None
    
    def create_model_performance_chart(self, summary_df: Optional[pd.DataFrame] = None) -> bool:
        """Create model performance comparison chart"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available for performance chart")
                return False
            
            if summary_df is None or len(summary_df) == 0:
                logger.warning("No summary data available for performance chart")
                return False
            
            logger.info("Creating model performance comparison chart")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            models = summary_df['model_name'].tolist()
            scores = summary_df['combined_score'].tolist()
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
            
            bars = ax1.bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax1.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
            ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Combined Score', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ctr_quality_counts = summary_df['ctr_quality'].value_counts()
            colors_pie = ['#27ae60', '#f39c12', '#e74c3c', '#95a5a6'][:len(ctr_quality_counts)]
            
            wedges, texts, autotexts = ax2.pie(ctr_quality_counts.values, 
                                             labels=ctr_quality_counts.index,
                                             colors=colors_pie, autopct='%1.1f%%',
                                             startangle=90, explode=[0.05]*len(ctr_quality_counts))
            
            ax2.set_title('CTR Quality Distribution', fontsize=16, fontweight='bold', pad=20)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            
            plt.tight_layout()
            
            chart_path = self.results_dir / "model_performance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            self.created_png_files.append(chart_path)
            logger.info(f"Model performance chart saved: {chart_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model performance chart creation failed: {e}")
            return False
    
    def create_ctr_analysis_dashboard(self, summary_df: Optional[pd.DataFrame] = None) -> bool:
        """Create CTR analysis dashboard with 4 panels"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available for CTR dashboard")
                return False
            
            if summary_df is None or len(summary_df) == 0:
                logger.warning("No summary data available for CTR dashboard")
                return False
            
            logger.info("Creating CTR analysis dashboard")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('CTR Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
            
            models = summary_df['model_name'].tolist()
            ctr_bias = summary_df['ctr_bias'].tolist()
            colors = ['#e74c3c' if bias > 0 else '#27ae60' for bias in ctr_bias]
            
            bars = ax1.barh(models, ctr_bias, color=colors, alpha=0.7)
            ax1.set_title('CTR Bias by Model', fontsize=14, fontweight='bold')
            ax1.set_xlabel('CTR Bias', fontsize=12)
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)
            
            actual_ctr = summary_df['actual_ctr'].tolist()
            predicted_ctr = summary_df['predicted_ctr'].tolist()
            
            ax2.scatter(actual_ctr, predicted_ctr, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
            
            min_ctr = min(min(actual_ctr), min(predicted_ctr))
            max_ctr = max(max(actual_ctr), max(predicted_ctr))
            ax2.plot([min_ctr, max_ctr], [min_ctr, max_ctr], 'r--', alpha=0.8, linewidth=2)
            
            ax2.set_title('Actual vs Predicted CTR', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Actual CTR', fontsize=12)
            ax2.set_ylabel('Predicted CTR', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            tier_counts = summary_df['performance_tier'].value_counts()
            tier_colors = {'EXCELLENT': '#27ae60', 'GOOD': '#2ecc71', 'FAIR': '#f39c12', 'POOR': '#e74c3c'}
            colors = [tier_colors.get(tier, '#95a5a6') for tier in tier_counts.index]
            
            ax3.bar(tier_counts.index, tier_counts.values, color=colors, alpha=0.8)
            ax3.set_title('Performance Tier Distribution', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Number of Models', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            exec_time = summary_df['execution_time_sec'].tolist()
            combined_scores = summary_df['combined_score'].tolist()
            
            scatter = ax4.scatter(exec_time, combined_scores, s=100, alpha=0.7, 
                                c=range(len(models)), cmap='plasma')
            
            ax4.set_title('Execution Time vs Performance', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Execution Time (seconds)', fontsize=12)
            ax4.set_ylabel('Combined Score', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            for i, model in enumerate(models):
                ax4.annotate(model, (exec_time[i], combined_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            
            dashboard_path = self.results_dir / "ctr_analysis.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            self.created_png_files.append(dashboard_path)
            logger.info(f"CTR analysis dashboard saved: {dashboard_path}")
            return True
            
        except Exception as e:
            logger.error(f"CTR dashboard creation failed: {e}")
            return False
    
    def create_execution_summary_chart(self, summary_df: Optional[pd.DataFrame] = None) -> bool:
        """Create execution summary and resource usage chart"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available for execution summary")
                return False
            
            if summary_df is None or len(summary_df) == 0:
                logger.warning("No summary data available for execution summary")
                return False
            
            logger.info("Creating execution summary chart")
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Execution Summary & Resource Usage', fontsize=16, fontweight='bold')
            
            models = summary_df['model_name'].tolist()
            
            exec_times = summary_df['execution_time_sec'].tolist()
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(models)))
            
            bars1 = ax1.bar(models, exec_times, color=colors, alpha=0.8)
            ax1.set_title('Training Time by Model', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Time (seconds)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, time_val in zip(bars1, exec_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(exec_times)*0.01,
                        f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
            
            memory_usage = summary_df['memory_peak_gb'].tolist()
            colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(models)))
            
            bars2 = ax2.bar(models, memory_usage, color=colors2, alpha=0.8)
            ax2.set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Memory (GB)', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, mem_val in zip(bars2, memory_usage):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(memory_usage)*0.01,
                        f'{mem_val:.1f}GB', ha='center', va='bottom', fontweight='bold')
            
            gpu_util = summary_df['gpu_utilization_pct'].tolist()
            colors3 = plt.cm.Greens(np.linspace(0.4, 0.9, len(models)))
            
            bars3 = ax3.bar(models, gpu_util, color=colors3, alpha=0.8)
            ax3.set_title('GPU Utilization', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Utilization (%)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, gpu_val in zip(bars3, gpu_util):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(gpu_util)*0.01,
                        f'{gpu_val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax3.legend(loc='lower right')
            
            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = self.results_dir / "execution_summary.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            self.created_png_files.append(chart_path)
            logger.info(f"Execution summary chart saved: {chart_path}")
            return True
            
        except Exception as e:
            logger.error(f"Execution summary chart creation failed: {e}")
            return False
    
    def create_comprehensive_pdf_report(self, summary_df: Optional[pd.DataFrame] = None, 
                                      analysis_results: Optional[Dict[str, Any]] = None) -> bool:
        """Create comprehensive PDF report using matplotlib backend"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available for PDF creation")
                return False
            
            if summary_df is None or len(summary_df) == 0:
                logger.warning("No summary data available for PDF report")
                return False
            
            logger.info("Creating comprehensive PDF report")
            
            pdf_path = self.results_dir / "comprehensive_report.pdf"
            
            with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
                fig_cover = plt.figure(figsize=(8.27, 11.69))
                fig_cover.text(0.5, 0.8, 'CTR Model Performance Analysis Report', 
                              ha='center', fontsize=24, fontweight='bold')
                fig_cover.text(0.5, 0.7, 'Comprehensive Analysis and Recommendations', 
                              ha='center', fontsize=16, color='#34495e')
                fig_cover.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                              ha='center', fontsize=12, color='#7f8c8d')
                
                best_model = summary_df.loc[summary_df['combined_score'].idxmax()]
                deployment_ready_count = len(summary_df[summary_df['deployment_ready']])
                avg_ctr_bias = summary_df['ctr_bias'].abs().mean()
                
                fig_cover.text(0.5, 0.45, 'Executive Summary', ha='center', fontsize=18, fontweight='bold')
                fig_cover.text(0.5, 0.4, f'Best Model: {best_model["model_name"]}', ha='center', fontsize=14)
                fig_cover.text(0.5, 0.37, f'Combined Score: {best_model["combined_score"]:.4f}', ha='center', fontsize=12)
                fig_cover.text(0.5, 0.34, f'Deployment Ready Models: {deployment_ready_count}', ha='center', fontsize=12)
                fig_cover.text(0.5, 0.31, f'Average CTR Bias: {avg_ctr_bias:.6f}', ha='center', fontsize=12)
                
                plt.axis('off')
                pdf.savefig(fig_cover, bbox_inches='tight')
                plt.close(fig_cover)
                
                charts = ['model_performance.png', 'ctr_analysis.png', 'execution_summary.png']
                for chart_name in charts:
                    chart_path = self.results_dir / chart_name
                    if chart_path.exists():
                        fig = plt.figure(figsize=(11.69, 8.27))
                        img = plt.imread(str(chart_path))
                        plt.imshow(img)
                        plt.axis('off')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            
            logger.info(f"Comprehensive PDF report saved: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF report creation failed: {e}")
            return False
    
    def cleanup_png_files(self):
        """Delete PNG files after PDF creation"""
        try:
            deleted_count = 0
            for png_path in self.created_png_files:
                if png_path.exists():
                    try:
                        png_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted PNG file: {png_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {png_path.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"PNG cleanup completed: {deleted_count} files deleted")
            else:
                logger.info("No PNG files to cleanup")
                
            self.created_png_files.clear()
            
        except Exception as e:
            logger.warning(f"PNG cleanup failed: {e}")
    
    def print_console_summary(self, summary_df: Optional[pd.DataFrame] = None):
        """Print performance summary to console"""
        try:
            if summary_df is None or len(summary_df) == 0:
                print("\n" + "="*85)
                print("CTR PERFORMANCE ANALYSIS SUMMARY")
                print("="*85)
                print("No performance data available")
                print("="*85)
                return
            
            print("\n" + "="*85)
            print("CTR PERFORMANCE ANALYSIS SUMMARY")
            print("="*85)
            
            top_models = summary_df.head(5)
            print(f"{'Model Name':<15} {'Combined Score':<15} {'CTR Quality':<12} {'Tier':<10}")
            print("-" * 85)
            
            max_score = top_models['combined_score'].max() if len(top_models) > 0 else 1
            
            for _, model in top_models.iterrows():
                name = str(model['model_name'])[:15]
                score = model['combined_score']
                quality = str(model['ctr_quality'])[:12]
                tier = str(model['performance_tier'])[:10]
                
                print(f"{name:<15} {score:<15.4f} {quality:<12} {tier:<10}")
            
            print("-" * 85)
            
            best_model = summary_df.loc[summary_df['combined_score'].idxmax()]
            deployment_ready_count = len(summary_df[summary_df['deployment_ready']])
            excellent_count = len(summary_df[summary_df['performance_tier'] == 'EXCELLENT'])
            avg_ctr_bias = summary_df['ctr_bias'].abs().mean()
            
            print(f"\nBest Performing Model: {best_model['model_name']}")
            print(f"Combined Score: {best_model['combined_score']:.4f}")
            print(f"CTR Quality: {best_model['ctr_quality']}")
            print(f"Training Time: {best_model['execution_time_sec']:.1f}s")
            print(f"Deployment Ready: {'Yes' if best_model['deployment_ready'] else 'No'}")
            print(f"\nSystem Overview:")
            print(f"   • Total Models: {len(summary_df)}")
            print(f"   • Deployment Ready: {deployment_ready_count}")
            print(f"   • Excellent Tier: {excellent_count}")
            print(f"   • Avg CTR Bias: {avg_ctr_bias:.6f}")
            
            print(f"\nAnalysis Results Location: {self.results_dir}")
            print("="*85)
            
        except Exception as e:
            logger.error(f"Console summary printing failed: {e}")
    
    def generate_all_visualizations(self, analysis_results: Dict[str, Any]) -> bool:
        """Generate all visualizations and reports"""
        try:
            logger.info("Starting comprehensive visualization generation")
            
            summary_df = self.load_summary_data()
            
            success_count = 0
            total_tasks = 4
            created_files = []
            
            if self.create_model_performance_chart(summary_df):
                success_count += 1
                created_files.append("model_performance.png")
                logger.info("Model performance chart created")
            
            if self.create_ctr_analysis_dashboard(summary_df):
                success_count += 1
                created_files.append("ctr_analysis.png")
                logger.info("CTR analysis dashboard created")
            
            if self.create_execution_summary_chart(summary_df):
                success_count += 1
                created_files.append("execution_summary.png")
                logger.info("Execution summary chart created")
            
            pdf_created = False
            if self.create_comprehensive_pdf_report(summary_df, analysis_results):
                success_count += 1
                created_files.append("comprehensive_report.pdf")
                pdf_created = True
                logger.info("Comprehensive PDF report created")
            
            if pdf_created:
                logger.info("PDF creation successful, cleaning up PNG files")
                self.cleanup_png_files()
            else:
                logger.warning("PDF creation failed, keeping PNG files")
            
            self.print_console_summary(summary_df)
            
            logger.info(f"Visualization generation completed: {success_count}/{total_tasks} successful")
            if created_files:
                final_files = [f for f in created_files if f.endswith('.pdf') or f.endswith('.csv') or f.endswith('.json')]
                if final_files:
                    logger.info(f"Final files in results folder: PDF, JSON reports, CSV summary")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Visualization generation process failed: {e}")
            return False
    
    def clear_visualizations(self):
        """Clear all visualization data and figures"""
        try:
            if MATPLOTLIB_AVAILABLE:
                plt.close('all')
            
            self.figures.clear()
            self.performance_data.clear()
            self.created_png_files.clear()
            gc.collect()
            
            logger.info("Visualization data cleared")
            
        except Exception as e:
            logger.warning(f"Visualization cleanup failed: {e}")

def create_all_visualizations(analysis_results: Dict[str, Any]) -> bool:
    """Main function to create all CTR visualizations and reports"""
    try:
        logger.info("Initializing CTR visualization system")
        
        visualizer = CTRVisualizationEngine()
        success = visualizer.generate_all_visualizations(analysis_results)
        
        visualizer.clear_visualizations()
        
        return success
        
    except Exception as e:
        logger.error(f"CTR visualization system failed: {e}")
        return False