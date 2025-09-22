# visualization.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import warnings
from pathlib import Path
from datetime import datetime
import gc
warnings.filterwarnings('ignore')

# Safe imports for visualization libraries
try:
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
        
        # Professional color schemes
        self.colors = {
            'primary_gradient': plt.cm.Blues(np.linspace(0.4, 0.9, 10)) if MATPLOTLIB_AVAILABLE else None,
            'secondary_gradient': plt.cm.Oranges(np.linspace(0.4, 0.9, 10)) if MATPLOTLIB_AVAILABLE else None,
            'success': '#27ae60',
            'warning': '#f39c12', 
            'danger': '#e74c3c',
            'info': '#3498db',
            'neutral': '#95a5a6'
        }
        
        logger.info("CTR Visualization Engine initialized with professional styling")
    
    def load_summary_data(self) -> Optional[pd.DataFrame]:
        """Load summary CSV data"""
        try:
            csv_path = self.results_dir / "summary.csv"
            if not csv_path.exists():
                logger.error("Summary CSV not found")
                return None
            
            summary_df = pd.read_csv(csv_path)
            self.performance_data = summary_df.to_dict('records')
            logger.info(f"Loaded {len(summary_df)} model records")
            return summary_df
            
        except Exception as e:
            logger.error(f"Failed to load summary data: {e}")
            return None
    
    def create_model_performance_chart(self, summary_df: pd.DataFrame) -> bool:
        """Create professional model performance comparison chart"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available")
                return False
            
            logger.info("Creating model performance chart")
            
            # Sort by combined_score for better visualization
            sorted_df = summary_df.sort_values('combined_score', ascending=True)
            models = sorted_df['model_name'].tolist()
            scores = sorted_df['combined_score'].tolist()
            
            fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.8)))
            
            # Create color gradient based on performance
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(models)))
            
            # Create horizontal bars with improved styling
            bars = ax.barh(models, scores, color=colors, height=0.6, alpha=0.8)
            
            # Add value labels on the right (Feature Importance style)
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(bar.get_width() + max(scores) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{score:.4f}', 
                       va='center', ha='left', fontweight='bold', fontsize=12)
            
            # Add performance tier reference lines
            target_excellent = 0.34
            target_good = 0.30
            
            if max(scores) > target_excellent:
                ax.axvline(x=target_excellent, color=self.colors['success'], 
                          linestyle='--', alpha=0.7, linewidth=2, label='Excellent (0.34+)')
            
            if max(scores) > target_good:
                ax.axvline(x=target_good, color=self.colors['warning'], 
                          linestyle='--', alpha=0.7, linewidth=2, label='Good (0.30+)')
            
            # Professional styling
            ax.set_xlabel('Combined Score', fontsize=14, fontweight='bold')
            ax.set_title('CTR Model Performance Comparison - Combined Score Analysis', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlim(0, max(max(scores) * 1.15, 0.4))
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if max(scores) > target_good:
                ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.results_dir / "model_performance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            self.figures['model_performance'] = fig
            logger.info(f"Model performance chart saved: {chart_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model performance chart creation failed: {e}")
            return False
    
    def create_ctr_analysis_dashboard(self, summary_df: pd.DataFrame) -> bool:
        """Create comprehensive CTR analysis dashboard"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available")
                return False
            
            logger.info("Creating CTR analysis dashboard")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('CTR Analysis Dashboard - Comprehensive Performance Overview', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Chart 1: CTR Bias Analysis (Horizontal bars)
            models = summary_df['model_name']
            ctr_bias = summary_df['ctr_bias']
            colors_bias = [self.colors['danger'] if abs(bias) > 0.001 else self.colors['info'] for bias in ctr_bias]
            
            bars1 = ax1.barh(models, ctr_bias, color=colors_bias, alpha=0.8, height=0.6)
            ax1.set_title('CTR Bias Analysis by Model', fontsize=14, fontweight='bold')
            ax1.set_xlabel('CTR Bias (Predicted - Actual)')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax1.axvline(x=0.001, color=self.colors['danger'], linestyle='--', alpha=0.7, label='Warning (+0.001)')
            ax1.axvline(x=-0.001, color=self.colors['danger'], linestyle='--', alpha=0.7, label='Warning (-0.001)')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(axis='x', alpha=0.3)
            
            # Add bias values
            for bar, bias in zip(bars1, ctr_bias):
                x_pos = bias + (max(ctr_bias) - min(ctr_bias)) * 0.02 if bias >= 0 else bias - (max(ctr_bias) - min(ctr_bias)) * 0.02
                ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                        f'{bias:.6f}', va='center', ha='left' if bias >= 0 else 'right', fontsize=10)
            
            # Chart 2: Performance Tier Distribution
            tier_counts = summary_df['performance_tier'].value_counts()
            tier_colors = {'EXCELLENT': self.colors['success'], 'GOOD': self.colors['info'], 
                          'FAIR': self.colors['warning'], 'POOR': self.colors['danger']}
            colors_pie = [tier_colors.get(tier, self.colors['neutral']) for tier in tier_counts.index]
            
            wedges, texts, autotexts = ax2.pie(tier_counts.values, labels=tier_counts.index, 
                                              colors=colors_pie, autopct='%1.1f%%', 
                                              startangle=90, textprops={'fontsize': 11})
            ax2.set_title('Performance Tier Distribution', fontsize=14, fontweight='bold')
            
            # Chart 3: Actual vs Predicted CTR Scatter Plot
            ax3.scatter(summary_df['actual_ctr'], summary_df['predicted_ctr'], 
                       s=120, alpha=0.7, c=self.colors['info'], edgecolors='black')
            
            # Perfect prediction line
            min_ctr = min(min(summary_df['actual_ctr']), min(summary_df['predicted_ctr']))
            max_ctr = max(max(summary_df['actual_ctr']), max(summary_df['predicted_ctr']))
            ax3.plot([min_ctr, max_ctr], [min_ctr, max_ctr], 
                    color=self.colors['success'], linestyle='--', alpha=0.8, 
                    linewidth=2, label='Perfect Prediction')
            
            ax3.set_xlabel('Actual CTR', fontweight='bold')
            ax3.set_ylabel('Predicted CTR', fontweight='bold')
            ax3.set_title('Actual vs Predicted CTR Analysis', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add model labels
            for i, model in enumerate(summary_df['model_name']):
                ax3.annotate(model, (summary_df['actual_ctr'].iloc[i], summary_df['predicted_ctr'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
            
            # Chart 4: Deployment Readiness Analysis
            deployment_counts = summary_df['deployment_ready'].value_counts()
            deployment_labels = ['Ready' if x else 'Not Ready' for x in deployment_counts.index]
            deployment_colors = [self.colors['success'] if x else self.colors['danger'] for x in deployment_counts.index]
            
            bars4 = ax4.bar(deployment_labels, deployment_counts.values, 
                           color=deployment_colors, alpha=0.8, width=0.6)
            ax4.set_title('Model Deployment Readiness', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Number of Models', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.results_dir / "ctr_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            self.figures['ctr_analysis'] = fig
            logger.info(f"CTR analysis dashboard saved: {chart_path}")
            return True
            
        except Exception as e:
            logger.error(f"CTR analysis dashboard creation failed: {e}")
            return False
    
    def create_execution_summary_chart(self, summary_df: pd.DataFrame) -> bool:
        """Create execution performance summary chart"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available")
                return False
            
            logger.info("Creating execution summary chart")
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
            fig.suptitle('System Execution Performance Analysis', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            models = summary_df['model_name']
            
            # Chart 1: Training Time Analysis (Horizontal bars)
            exec_times = summary_df['execution_time_sec']
            colors1 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(models)))
            
            bars1 = ax1.barh(models, exec_times, color=colors1, alpha=0.8, height=0.6)
            ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Execution Time (seconds)', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # Add time labels
            for bar, time_val in zip(bars1, exec_times):
                ax1.text(bar.get_width() + max(exec_times) * 0.02,
                        bar.get_y() + bar.get_height()/2,
                        f'{time_val:.1f}s', va='center', ha='left', fontweight='bold')
            
            # Chart 2: Memory Usage Analysis (Horizontal bars)
            memory_usage = summary_df['memory_peak_gb']
            colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(models)))
            
            bars2 = ax2.barh(models, memory_usage, color=colors2, alpha=0.8, height=0.6)
            ax2.set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Memory Usage (GB)', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            # Add memory labels
            for bar, mem_val in zip(bars2, memory_usage):
                ax2.text(bar.get_width() + max(memory_usage) * 0.02,
                        bar.get_y() + bar.get_height()/2,
                        f'{mem_val:.1f}GB', va='center', ha='left', fontweight='bold')
            
            # Chart 3: GPU Utilization Analysis (Horizontal bars)
            gpu_usage = summary_df['gpu_utilization_pct']
            colors_gpu = [self.colors['success'] if gpu > 50 else self.colors['warning'] for gpu in gpu_usage]
            
            bars3 = ax3.barh(models, gpu_usage, color=colors_gpu, alpha=0.8, height=0.6)
            ax3.set_title('GPU Utilization Analysis', fontsize=14, fontweight='bold')
            ax3.set_xlabel('GPU Utilization (%)', fontweight='bold')
            ax3.set_xlim(0, 100)
            ax3.grid(axis='x', alpha=0.3)
            
            # Add GPU labels and reference line
            ax3.axvline(x=50, color=self.colors['info'], linestyle='--', alpha=0.7, label='Target (50%)')
            for bar, gpu_val in zip(bars3, gpu_usage):
                ax3.text(bar.get_width() + 2,
                        bar.get_y() + bar.get_height()/2,
                        f'{gpu_val:.1f}%', va='center', ha='left', fontweight='bold')
            
            ax3.legend(loc='lower right')
            
            # Style all charts
            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.results_dir / "execution_summary.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            self.figures['execution_summary'] = fig
            logger.info(f"Execution summary chart saved: {chart_path}")
            return True
            
        except Exception as e:
            logger.error(f"Execution summary chart creation failed: {e}")
            return False
    
    def create_comprehensive_pdf_report(self, summary_df: pd.DataFrame, analysis_results: Dict[str, Any]) -> bool:
        """Create comprehensive PDF report using matplotlib backend"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available for PDF creation")
                return False
            
            logger.info("Creating comprehensive PDF report")
            
            pdf_path = self.results_dir / "comprehensive_report.pdf"
            
            with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
                # Cover page
                fig_cover = plt.figure(figsize=(8.27, 11.69))  # A4 size
                fig_cover.text(0.5, 0.8, 'CTR Model Performance Analysis Report', 
                              ha='center', fontsize=24, fontweight='bold')
                fig_cover.text(0.5, 0.7, 'Comprehensive Analysis and Recommendations', 
                              ha='center', fontsize=16, color='#34495e')
                fig_cover.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                              ha='center', fontsize=12, color='#7f8c8d')
                
                # Executive summary
                best_model = summary_df.loc[summary_df['combined_score'].idxmax()]
                deployment_ready_count = len(summary_df[summary_df['deployment_ready']])
                avg_ctr_bias = summary_df['ctr_bias'].abs().mean()
                
                fig_cover.text(0.5, 0.45, 'Executive Summary', ha='center', fontsize=16, fontweight='bold')
                fig_cover.text(0.5, 0.38, f"Best Model: {best_model['model_name']}", ha='center', fontsize=12)
                fig_cover.text(0.5, 0.34, f"Best Combined Score: {best_model['combined_score']:.4f}", ha='center', fontsize=12)
                fig_cover.text(0.5, 0.30, f"Models Deployment Ready: {deployment_ready_count}/{len(summary_df)}", ha='center', fontsize=12)
                fig_cover.text(0.5, 0.26, f"Average CTR Bias: {avg_ctr_bias:.6f}", ha='center', fontsize=12)
                fig_cover.text(0.5, 0.22, f"Total Models Analyzed: {len(summary_df)}", ha='center', fontsize=12)
                
                # Key recommendations
                fig_cover.text(0.5, 0.15, 'Key Recommendations', ha='center', fontsize=16, fontweight='bold')
                fig_cover.text(0.5, 0.08, f"• Deploy {best_model['model_name']} for production", ha='center', fontsize=11)
                fig_cover.text(0.5, 0.05, "• Implement CTR bias monitoring system", ha='center', fontsize=11)
                fig_cover.text(0.5, 0.02, "• Consider ensemble methods for improved performance", ha='center', fontsize=11)
                
                pdf.savefig(fig_cover, bbox_inches='tight')
                plt.close(fig_cover)
                
                # Add all generated charts to PDF
                for fig_name, fig in self.figures.items():
                    if fig is not None:
                        pdf.savefig(fig, bbox_inches='tight')
                        logger.info(f"Added {fig_name} to PDF report")
            
            logger.info(f"Comprehensive PDF report saved: {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF report creation failed: {e}")
            return False
    
    def print_console_summary(self, summary_df: pd.DataFrame):
        """Print professional ASCII summary to console"""
        try:
            print("\n" + "="*85)
            print("CTR MODEL PERFORMANCE ANALYSIS - EXECUTION SUMMARY")
            print("="*85)
            
            # Sort by combined_score for display
            top_models = summary_df.sort_values('combined_score', ascending=False).head(5)
            
            print("\nTop 5 Model Performance (Combined Score):")
            print("┌" + "─"*75 + "┐")
            print("│" + " "*27 + "Model Performance Overview" + " "*22 + "│")
            print("├" + "─"*75 + "┤")
            
            max_score = top_models['combined_score'].max() if len(top_models) > 0 else 1
            
            for _, model in top_models.iterrows():
                name = str(model['model_name'])[:15]
                score = model['combined_score']
                bar_length = int((score / max_score) * 35) if max_score > 0 else 0
                bar = "█" * bar_length
                
                print(f"│ {name:<15} {bar:<35} {score:.4f} │")
            
            print("└" + "─"*75 + "┘")
            
            # Key metrics summary
            best_model = summary_df.loc[summary_df['combined_score'].idxmax()]
            deployment_ready_count = len(summary_df[summary_df['deployment_ready']])
            excellent_count = len(summary_df[summary_df['performance_tier'] == 'EXCELLENT'])
            avg_ctr_bias = summary_df['ctr_bias'].abs().mean()
            
            print(f"\n🎯 Best Performing Model: {best_model['model_name']}")
            print(f"📊 Combined Score: {best_model['combined_score']:.4f}")
            print(f"🎯 CTR Quality: {best_model['ctr_quality']}")
            print(f"⏱️  Training Time: {best_model['execution_time_sec']:.1f}s")
            print(f"✅ Deployment Ready: {'Yes' if best_model['deployment_ready'] else 'No'}")
            print(f"\n📈 System Overview:")
            print(f"   • Total Models: {len(summary_df)}")
            print(f"   • Deployment Ready: {deployment_ready_count}")
            print(f"   • Excellent Tier: {excellent_count}")
            print(f"   • Avg CTR Bias: {avg_ctr_bias:.6f}")
            
            print(f"\n📁 Analysis Results Location: {self.results_dir}")
            print("="*85)
            
        except Exception as e:
            logger.error(f"Console summary printing failed: {e}")
    
    def generate_all_visualizations(self, analysis_results: Dict[str, Any]) -> bool:
        """Generate all visualizations and reports"""
        try:
            logger.info("Starting comprehensive visualization generation")
            
            # Load summary data
            summary_df = self.load_summary_data()
            if summary_df is None:
                return False
            
            success_count = 0
            total_tasks = 4
            
            # 1. Create model performance chart
            if self.create_model_performance_chart(summary_df):
                success_count += 1
                logger.info("✓ Model performance chart created")
            
            # 2. Create CTR analysis dashboard
            if self.create_ctr_analysis_dashboard(summary_df):
                success_count += 1
                logger.info("✓ CTR analysis dashboard created")
            
            # 3. Create execution summary chart
            if self.create_execution_summary_chart(summary_df):
                success_count += 1
                logger.info("✓ Execution summary chart created")
            
            # 4. Create comprehensive PDF report
            if self.create_comprehensive_pdf_report(summary_df, analysis_results):
                success_count += 1
                logger.info("✓ Comprehensive PDF report created")
            
            # Print console summary
            self.print_console_summary(summary_df)
            
            # Final status
            logger.info(f"Visualization generation completed: {success_count}/{total_tasks} successful")
            
            if success_count >= 3:
                logger.info("Results saved in 'results' folder:")
                logger.info("  - summary.csv (comprehensive metrics)")
                logger.info("  - model_performance.png (performance comparison)")
                logger.info("  - ctr_analysis.png (4-panel CTR dashboard)")
                logger.info("  - execution_summary.png (resource usage)")
                logger.info("  - comprehensive_report.pdf (complete analysis)")
                return True
            else:
                logger.warning("Some visualizations failed to create")
                return False
            
        except Exception as e:
            logger.error(f"Visualization generation process failed: {e}")
            return False
    
    def clear_visualizations(self):
        """Clear all visualization data and figures"""
        try:
            # Close all matplotlib figures
            for fig in self.figures.values():
                try:
                    plt.close(fig)
                except:
                    pass
            
            self.figures.clear()
            self.performance_data.clear()
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
        
        # Cleanup
        visualizer.clear_visualizations()
        
        return success
        
    except Exception as e:
        logger.error(f"CTR visualization system failed: {e}")
        return False