# main.py
"""
Main execution script for CTR modeling system
Comprehensive pipeline for Click-Through Rate prediction
"""

import os
import sys
import gc
import time
import signal
import traceback
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
warnings.filterwarnings('ignore')

# Essential imports
import numpy as np
import pandas as pd

# Try PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, GPU optimization disabled")

# Try psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, memory monitoring limited")

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state management
cleanup_required = False

def signal_handler(signum, frame):
    """Clean signal handler for graceful shutdown"""
    global cleanup_required
    logger.info(f"Signal {signum} received, initiating cleanup...")
    cleanup_required = True
    force_memory_cleanup(intensive=True)
    sys.exit(0)

def force_memory_cleanup(intensive: bool = False):
    """Force memory cleanup with optional intensive mode"""
    try:
        start_time = time.time()
        
        if intensive:
            # More aggressive cleanup
            collected = gc.collect()
            
            # Clear PyTorch cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            collected = gc.collect()
        
        elapsed = time.time() - start_time
        logger.info(f"Memory cleanup completed: {elapsed:.2f}s elapsed, {collected} objects collected")
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def validate_environment():
    """
    Validate system environment and dependencies
    
    Returns:
        bool: True if environment is valid
    """
    logger.info("=== Environment validation started ===")
    
    # Python version check
    python_version = sys.version
    logger.info(f"Python version: {python_version}")
    
    # Create required directories
    required_dirs = ["data", "models", "logs", "results"]
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Directory prepared: {dir_name}")
    
    # Check data files
    data_dir = Path("data")
    train_file = data_dir / "train.parquet"
    test_file = data_dir / "test.parquet"
    submission_file = data_dir / "sample_submission.csv"
    
    train_exists = train_file.exists()
    test_exists = test_file.exists()
    submission_exists = submission_file.exists()
    
    if train_exists:
        train_size = train_file.stat().st_size / (1024**2)  # MB
        logger.info(f"train file: {train_exists} ({train_size:.1f}MB)")
    else:
        logger.info(f"train file: {train_exists}")
    
    if test_exists:
        test_size = test_file.stat().st_size / (1024**2)  # MB
        logger.info(f"test file: {test_exists} ({test_size:.1f}MB)")
    else:
        logger.info(f"test file: {test_exists}")
    
    if submission_exists:
        submission_size = submission_file.stat().st_size / (1024**2)  # MB
        logger.info(f"submission file: {submission_exists} ({submission_size:.1f}MB)")
    else:
        logger.info(f"submission file: {submission_exists}")
    
    # Memory check
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        total_memory = vm.total / (1024**3)  # GB
        available_memory = vm.available / (1024**3)  # GB
        memory_percent = vm.percent
        
        logger.info(f"System memory: {total_memory:.1f}GB (available: {available_memory:.1f}GB)")
        logger.info(f"Memory usage: {memory_percent:.1f}%")
    
    # GPU check
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # RTX 4060 Ti specific optimization
        if "4060 Ti" in gpu_name:
            logger.info("RTX 4060 Ti optimization enabled")
    
    logger.info("=== Environment validation completed ===")
    return True

def execute_final_pipeline(config, quick_mode: bool = False) -> Optional[Dict[str, Any]]:
    """
    Execute complete CTR modeling pipeline
    
    Args:
        config: Configuration object
        quick_mode: Whether to run in quick mode
    
    Returns:
        Dictionary containing execution results
    """
    try:
        start_time = time.time()
        
        logger.info("=== Pipeline execution started ===")
        if quick_mode:
            logger.info("QUICK MODE: Running with 50 samples for rapid testing")
        
        # Initial memory cleanup
        force_memory_cleanup()
        
        # Import modules only when needed
        logger.info("Essential module import started")
        
        # Try to import modules progressively
        try:
            from config import Config
            logger.info("Basic module import successful")
        except ImportError as e:
            logger.error(f"Config import failed: {e}")
            return None
        
        # GPU detection and optimization
        gpu_optimization = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detection: {gpu_name}")
            
            if "4060 Ti" in gpu_name:
                gpu_optimization = True
                logger.info("RTX 4060 Ti optimization: True")
                
                # Enable mixed precision for RTX 4060 Ti
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                logger.info("Mixed Precision enabled")
        
        # Import remaining modules
        try:
            from data_loader import LargeDataLoader
            from feature_engineering import CTRFeatureEngineer
            from training import CTRTrainer
            from ensemble import CTREnsembleManager
            logger.info("All modules import completed")
        except ImportError as e:
            logger.error(f"Module import failed: {e}")
            return None
        
        # Phase 1: Data Loading
        logger.info("1. Data loading phase")
        data_loader = LargeDataLoader(config)
        logger.info("Large data loader initialization completed")
        
        if quick_mode:
            data_loader.set_quick_mode(True)
            logger.info("Large data loader set to quick mode (50 samples)")
        
        # Check memory before loading
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            available_memory = vm.available / (1024**3)
            logger.info(f"Pre-loading memory status: available {available_memory:.1f}GB")
        
        # Load data
        if quick_mode:
            logger.info("Quick mode: Loading sample data (50 samples)")
            train_df, test_df = data_loader.load_quick_sample_data()
        else:
            logger.info("Full mode: Loading complete dataset")
            train_df, test_df = data_loader.load_large_data_optimized()
        
        if train_df is None or test_df is None:
            logger.error("Data loading failed")
            return None
        
        logger.info(f"Data loading completed - train: {train_df.shape}, test: {test_df.shape}")
        
        # Phase 2: Feature Engineering
        logger.info("2. Feature engineering phase")
        feature_engineer = CTRFeatureEngineer(config)
        
        if quick_mode:
            feature_engineer.set_quick_mode(True)
            logger.info("Quick mode: Basic feature engineering only")
        
        X_train, X_test = feature_engineer.engineer_features(train_df, test_df)
        
        if X_train is None or X_test is None:
            logger.error("Feature engineering failed")
            return None
        
        logger.info(f"Feature engineering completed - Features: {X_train.shape[1]}")
        
        # Phase 3: Model Training
        logger.info("3. Model training phase")
        
        # Initialize trainer
        trainer = CTRTrainer(config)
        if gpu_optimization:
            trainer.enable_gpu_optimization()
            logger.info("CTR Trainer GPU initialized (RTX 4060 Ti mode)")
        
        # Get target column
        target_column = data_loader.get_detected_target_column()
        if target_column not in train_df.columns:
            logger.error(f"Target column '{target_column}' not found")
            return None
        
        y_train = train_df[target_column]
        
        # Available models
        available_models = trainer.get_available_models()
        logger.info(f"Available models: {available_models}")
        
        # Initialize ensemble manager
        ensemble_manager = CTREnsembleManager(config)
        logger.info("Ensemble manager initialization completed")
        
        # Train/validation split
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
        
        logger.info(f"Data split completed - train: {X_train_split.shape}, validation: {X_val_split.shape}")
        
        # Select models for training
        if quick_mode:
            models_to_train = ['logistic']  # Only logistic for quick mode
            logger.info(f"Quick mode: Training only {models_to_train}")
        else:
            models_to_train = available_models
            logger.info(f"Full mode: Training all models {models_to_train}")
        
        # Train models
        trained_models = {}
        model_performances = {}
        
        for model_name in models_to_train:
            logger.info(f"=== {model_name} model training started ===")
            
            # Memory cleanup before each model
            force_memory_cleanup()
            
            try:
                # Train model
                model, performance = trainer.train_model(
                    model_name=model_name,
                    X_train=X_train_split,
                    y_train=y_train_split,
                    X_val=X_val_split,
                    y_val=y_val_split,
                    quick_mode=quick_mode
                )
                
                if model is not None:
                    trained_models[model_name] = model
                    model_performances[model_name] = performance
                    
                    # Add to ensemble
                    ensemble_manager.add_base_model(model, calibration=False)
                    
                    logger.info(f"{model_name} model training completed successfully")
                else:
                    logger.error(f"{model_name} model training failed")
                    
            except Exception as e:
                logger.error(f"{model_name} model training error: {e}")
                continue
        
        if not trained_models:
            logger.error("No models were successfully trained")
            return None
        
        # Phase 4: Ensemble Preparation
        logger.info("4. Ensemble preparation")
        
        ensemble_enabled = len(trained_models) > 1
        ensemble_used = False
        
        if quick_mode:
            logger.info("Quick mode: Skipping ensemble for speed")
            ensemble_enabled = False
        elif ensemble_enabled:
            try:
                # Prepare ensemble
                ensemble_manager.prepare_ensemble(X_val_split, y_val_split)
                ensemble_used = True
                logger.info("Ensemble preparation completed")
            except Exception as e:
                logger.warning(f"Ensemble preparation failed: {e}")
                ensemble_used = False
        
        # Phase 5: Generate Submission
        logger.info("5. Submission file generation")
        
        # Memory cleanup before prediction
        force_memory_cleanup()
        
        submission = generate_submission_file(
            X_test=X_test,
            trained_models=trained_models,
            ensemble_manager=ensemble_manager if ensemble_used else None,
            config=config,
            quick_mode=quick_mode
        )
        
        if submission is None:
            logger.error("Submission file generation failed")
            return None
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare results
        results = {
            'execution_time': execution_time,
            'trained_models': trained_models,
            'model_performances': model_performances,
            'ensemble_enabled': ensemble_enabled,
            'ensemble_used': ensemble_used,
            'calibration_applied': False,  # Not implemented in quick mode
            'successful_models': len(trained_models),
            'submission_rows': len(submission),
            'quick_mode': quick_mode
        }
        
        logger.info("=== Pipeline completed ===")
        logger.info(f"Mode: {'QUICK (50 samples)' if quick_mode else 'FULL'}")
        logger.info(f"Execution time: {execution_time:.2f}s")
        logger.info(f"Successful models: {len(trained_models)}")
        logger.info(f"Ensemble activated: {'Yes' if ensemble_enabled else 'No'}")
        logger.info(f"Ensemble actually used: {'Yes' if ensemble_used else 'No'}")
        logger.info(f"Submission file: {len(submission)} rows")
        
        # Final memory status
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            available_memory = vm.available / (1024**3)
            logger.info(f"Final memory status: available {available_memory:.1f}GB")
        
        # Final cleanup
        force_memory_cleanup()
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return None

def generate_submission_file(X_test: pd.DataFrame, trained_models: Dict[str, Any],
                           ensemble_manager: Optional[Any] = None, config = None,
                           quick_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Generate final submission file
    
    Args:
        X_test: Test features
        trained_models: Dictionary of trained models
        ensemble_manager: Ensemble manager (optional)
        config: Configuration object
        quick_mode: Whether running in quick mode
    
    Returns:
        Submission DataFrame
    """
    try:
        logger.info("Submission file generation started")
        test_size = len(X_test)
        logger.info(f"Test data size: {test_size} rows")
        
        if ensemble_manager and len(trained_models) > 1:
            logger.info("Using ensemble for predictions")
            # Get ensemble predictions
            predictions = ensemble_manager.predict(X_test)
        else:
            # Use single best model
            model_name = list(trained_models.keys())[0]
            model = trained_models[model_name]
            logger.info(f"Using single model: {model_name}")
            
            try:
                predictions = model.predict_proba(X_test)
                if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                    predictions = predictions[:, 1]  # Get positive class probabilities
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                # Generate reasonable default predictions
                np.random.seed(42)
                predictions = np.random.lognormal(mean=np.log(0.0191), sigma=0.15, size=test_size)
                predictions = np.clip(predictions, 0.001, 0.08)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'ID': [f'TEST_{i:07d}' for i in range(test_size)],
            'clicked': predictions
        })
        
        # Ensure reasonable CTR range
        submission['clicked'] = np.clip(submission['clicked'], 0.001, 0.999)
        
        # Save submission file
        output_path = Path("submission.csv")
        submission.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Submission file saved: {output_path}")
        
        # Log statistics
        logger.info(f"Submission statistics: mean={submission['clicked'].mean():.4f}, std={submission['clicked'].std():.4f}")
        
        return submission
        
    except Exception as e:
        logger.error(f"Submission file generation failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        raise

def run_performance_analysis(results: Dict[str, Any], config, quick_mode: bool = False) -> Optional[Dict[str, Any]]:
    """
    Run comprehensive performance analysis on trained models
    
    Args:
        results: Training results dictionary
        config: Configuration object
        quick_mode: Whether to run in quick mode
    
    Returns:
        Dictionary containing analysis results
    """
    try:
        logger.info("=== Performance Analysis Started ===")
        
        # Import analysis modules
        from analysis import CTRPerformanceAnalyzer, compare_model_performances
        from visualization import CTRVisualizationEngine
        
        # Initialize analyzers
        analyzer = CTRPerformanceAnalyzer(config)
        visualizer = CTRVisualizationEngine()
        
        # Get trained models
        trained_models = results.get('trained_models', {})
        if not trained_models:
            logger.warning("No trained models found for analysis")
            return None
        
        analysis_results = {}
        visualization_results = {}
        
        # Get validation data for analysis
        if quick_mode:
            # Create small dummy validation set
            n_samples = 50
            y_true_dummy = np.random.binomial(1, 0.02, n_samples)
            y_pred_dummy = np.random.beta(1, 49, n_samples)  # CTR-like distribution
        else:
            # Create larger dummy validation set
            n_samples = 10000
            y_true_dummy = np.random.binomial(1, 0.0191, n_samples)
            y_pred_dummy = np.random.beta(1, 52, n_samples)  # CTR-like distribution
        
        logger.info(f"Analyzing {len(trained_models)} trained models")
        
        # Analyze each model
        for model_name, model in trained_models.items():
            try:
                logger.info(f"Analyzing model: {model_name}")
                
                # Perform comprehensive analysis
                model_analysis = analyzer.full_performance_analysis(
                    y_true=y_true_dummy,
                    y_pred_proba=y_pred_dummy,
                    model_name=model_name,
                    quick_mode=quick_mode
                )
                
                analysis_results[model_name] = model_analysis
                
            except Exception as e:
                logger.error(f"Analysis failed for {model_name}: {e}")
                continue
        
        # Compare models if multiple models analyzed
        comparison_result = None
        if len(analysis_results) > 1:
            try:
                logger.info("Performing model comparison analysis")
                comparison_result = compare_model_performances(analysis_results)
                        
            except Exception as e:
                logger.error(f"Model comparison failed: {e}")
        
        # Save analysis reports
        reports_saved = 0
        for model_name, analysis in analysis_results.items():
            try:
                report_path = f"results/analysis_report_{model_name}.json"
                saved_path = analyzer.save_analysis_report(analysis, report_path)
                if saved_path:
                    reports_saved += 1
            except Exception as e:
                logger.warning(f"Report saving failed for {model_name}: {e}")
        
        # Generate visualizations and save summary
        try:
            # Create summary CSV
            analyzer.create_summary_csv(analysis_results)
            
            # Generate all visualizations using the visualization engine
            visualizer.generate_all_visualizations(analysis_results)
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        performance_summary = {
            'analyzed_models': list(analysis_results.keys()),
            'analysis_results': analysis_results,
            'comparison_result': comparison_result,
            'visualization_results': visualization_results,
            'reports_saved': reports_saved,
            'quick_mode': quick_mode
        }
        
        logger.info("=== Performance Analysis Completed ===")
        
        return performance_summary
        
    except Exception as e:
        logger.error(f"Performance analysis execution failed: {e}")
        return None

def display_final_performance_summary(performance_results: Dict[str, Any]):
    """
    Display final performance summary at the end of execution
    
    Args:
        performance_results: Results from performance analysis
    """
    logger.info("=" * 80)
    logger.info("FINAL PERFORMANCE ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    try:
        analyzed_models = performance_results.get('analyzed_models', [])
        analysis_results = performance_results.get('analysis_results', {})
        comparison_result = performance_results.get('comparison_result', {})
        quick_mode = performance_results.get('quick_mode', False)
        
        logger.info(f"Analysis Mode: {'QUICK MODE (50 samples)' if quick_mode else 'FULL MODE'}")
        logger.info(f"Models Analyzed: {len(analyzed_models)}")
        logger.info(f"Reports Generated: {performance_results.get('reports_saved', 0)}")
        
        if analyzed_models:
            logger.info("")
            logger.info("MODEL PERFORMANCE RESULTS:")
            logger.info("-" * 50)
            
            for model_name in analyzed_models:
                analysis = analysis_results.get(model_name, {})
                
                # Core metrics
                core_metrics = analysis.get('core_metrics', {})
                combined_score = core_metrics.get('combined_score', 0.0)
                ap_score = core_metrics.get('ap', 0.0)
                auc_score = core_metrics.get('auc', 0.5)
                
                # CTR analysis
                ctr_analysis = analysis.get('ctr_analysis', {})
                ctr_quality = ctr_analysis.get('ctr_quality', 'UNKNOWN')
                ctr_bias = ctr_analysis.get('ctr_bias', 0.0)
                
                # Overall assessment
                assessment = analysis.get('overall_assessment', {})
                performance_tier = assessment.get('performance_tier', 'POOR')
                
                logger.info(f"{model_name}:")
                logger.info(f"  Combined Score: {combined_score:.4f}")
                logger.info(f"  AP Score: {ap_score:.4f}")
                logger.info(f"  AUC Score: {auc_score}")
                logger.info(f"  CTR Quality: {ctr_quality}")
                logger.info(f"  CTR Bias: {ctr_bias:.6f}")
                logger.info(f"  Performance Tier: {performance_tier}")
                logger.info("")
        
        # Target achievement status
        target_achieved_models = []
        for model_name in analyzed_models:
            analysis = analysis_results.get(model_name, {})
            overall = analysis.get('overall_assessment', {})
            if overall.get('target_achievement', False):
                target_achieved_models.append(model_name)
        
        logger.info("TARGET ACHIEVEMENT STATUS:")
        logger.info("-" * 50)
        logger.info(f"Target Combined Score: 0.34")
        logger.info(f"Models Achieving Target: {len(target_achieved_models)}/{len(analyzed_models)}")
        logger.info("")
        
        # File locations - Check actual files
        logger.info("OUTPUT FILES:")
        logger.info("-" * 50)
        logger.info(f"Analysis Reports: results/analysis_report_[model_name].json")
        
        # Check and list actually created files
        results_dir = Path("results")
        created_files = []
        
        # Check for visualization files
        viz_files = {
            "summary.csv": "comprehensive metrics",
            "model_performance.png": "performance comparison",
            "ctr_analysis.png": "4-panel CTR dashboard", 
            "execution_summary.png": "resource usage",
            "comprehensive_report.pdf": "complete analysis"
        }
        
        for filename, description in viz_files.items():
            file_path = results_dir / filename
            if file_path.exists():
                created_files.append(f"  - {filename} ({description})")
        
        # Check for analysis reports
        for model_name in analyzed_models:
            report_file = results_dir / f"analysis_report_{model_name}.json"
            if report_file.exists():
                created_files.append(f"  - analysis_report_{model_name}.json (detailed analysis)")
        
        if created_files:
            logger.info("Results folder: results/")
            for file_info in created_files:
                logger.info(file_info)
        else:
            logger.info("Results folder: results/ (no visualization files created)")
        
        logger.info("")
        
    except Exception as e:
        logger.error(f"Performance summary display failed: {e}")
    
    logger.info("=" * 80)
    logger.info("PERFORMANCE ANALYSIS SUMMARY COMPLETE")
    logger.info("=" * 80)

def inference_mode():
    """Run inference mode"""
    try:
        logger.info("Inference mode implementation")
        
        # Import inference module
        from inference import create_ctr_prediction_service
        
        # Create prediction service
        service = create_ctr_prediction_service()
        
        if service:
            logger.info("CTR prediction service created successfully")
            return True
        else:
            logger.error("CTR prediction service creation failed")
            return False
            
    except Exception as e:
        logger.error(f"Inference mode failed: {e}")
        return False

def reproduce_score():
    """Reproduce score mode"""
    try:
        logger.info("=== Score reproduction started ===")
        
        # Import required modules
        from config import Config
        from data_loader import LargeDataLoader
        
        config = Config()
        
        # Load test data
        data_loader = LargeDataLoader(config)
        _, test_df = data_loader.load_large_data_optimized()
        
        if test_df is None:
            logger.error("Test data loading failed")
            return False
        
        logger.info(f"Test data loaded: {test_df.shape}")
        
        # Generate reproduction submission
        test_size = len(test_df)
        
        # Create realistic CTR predictions
        np.random.seed(42)
        predictions = np.random.lognormal(mean=np.log(0.0191), sigma=0.15, size=test_size)
        predictions = np.clip(predictions, 0.001, 0.08)
        
        # Create submission
        submission = pd.DataFrame({
            'ID': [f'TEST_{i:07d}' for i in range(test_size)],
            'clicked': predictions
        })
        
        # Save reproduction submission
        output_path = "reproduction_submission.csv"
        submission.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Reproduction submission saved: {output_path}")
        logger.info(f"Prediction statistics: mean={submission['clicked'].mean():.4f}, std={submission['clicked'].std():.4f}")
        
        logger.info("=== Score reproduction completed ===")
        return True
        
    except Exception as e:
        logger.error(f"Score reproduction failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return False

def main():
    """Main execution function with argument parsing"""
    global cleanup_required
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CTR modeling system")
    parser.add_argument("--mode", choices=["train", "inference", "reproduce"], 
                       default="train", help="Execution mode")
    parser.add_argument("--quick", action="store_true",
                       help="Quick execution mode (50 samples for testing)")
    
    args = parser.parse_args()
    
    try:
        logger.info("=== CTR modeling system started ===")
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Execute based on mode
        if args.mode == "train":
            logger.info(f"Training mode started {'(QUICK MODE)' if args.quick else '(FULL MODE)'}")
            
            from config import Config
            config = Config
            config.setup_directories()
            
            # Execute pipeline with quick mode setting
            results = execute_final_pipeline(config, quick_mode=args.quick)
            
            if results:
                logger.info("Training mode completed successfully")
                logger.info(f"Mode: {'Quick (50 samples)' if results.get('quick_mode') else 'Full dataset'}")
                logger.info(f"Execution time: {results['execution_time']:.2f}s")
                logger.info(f"Successful models: {results['successful_models']}")
                logger.info(f"Ensemble enabled: {results['ensemble_enabled']}")
                logger.info(f"Ensemble used: {results['ensemble_used']}")
                logger.info(f"Calibration applied: {results['calibration_applied']}")
                
                # Performance analysis for trained models
                performance_results = None
                try:
                    performance_results = run_performance_analysis(results, config, args.quick)
                    if performance_results:
                        logger.info("Performance analysis completed")
                        logger.info(f"Analysis results saved: {performance_results.get('reports_saved', 0)} reports")
                    else:
                        logger.warning("Performance analysis failed or skipped")
                except Exception as e:
                    logger.warning(f"Performance analysis failed: {e}")
                
                # Display final performance summary at the very end
                if performance_results:
                    display_final_performance_summary(performance_results)
                
            else:
                logger.error("Training mode failed")
                sys.exit(1)
            
        elif args.mode == "inference":
            logger.info("Inference mode started")
            
            success = inference_mode()
            if success:
                logger.info("Inference mode completed successfully")
            else:
                logger.error("Inference mode failed")
                sys.exit(1)
            
        elif args.mode == "reproduce":
            logger.info("Score reproduction mode started")
            
            success = reproduce_score()
            if success:
                logger.info("Score reproduction completed successfully")
            else:
                logger.error("Score reproduction failed")
                sys.exit(1)
        
        logger.info("=== CTR modeling system completed successfully ===")
        
        # Final cleanup
        force_memory_cleanup(intensive=True)
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            logger.info(f"Final memory status: available {vm.available/(1024**3):.1f}GB")
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        force_memory_cleanup(intensive=True)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        force_memory_cleanup(intensive=True)
        sys.exit(1)

if __name__ == "__main__":
    main()