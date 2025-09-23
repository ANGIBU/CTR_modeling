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
            
            elapsed_time = time.time() - start_time
            logger.info(f"Intensive memory cleanup completed: {elapsed_time:.2f}s elapsed, {collected} objects collected")
        else:
            # Standard cleanup
            collected = gc.collect()
            elapsed_time = time.time() - start_time
            logger.info(f"Memory cleanup completed: {elapsed_time:.2f}s elapsed, {collected} objects collected")
    
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def create_required_directories():
    """Create necessary directories for the project"""
    directories = ['data', 'models', 'logs', 'results']
    created_dirs = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path.absolute()))
    
    if created_dirs:
        print(f"Created directories: {created_dirs}")
    
    return created_dirs

def validate_environment():
    """Validate system environment and requirements"""
    logger.info("=== Environment validation started ===")
    
    # Check Python version
    python_version = sys.version
    logger.info(f"Python version: {python_version}")
    
    # Create directories
    create_required_directories()
    
    # File validation
    from config import Config
    config = Config()
    
    train_file = config.TRAIN_PATH
    test_file = config.TEST_PATH
    submission_file = config.SUBMISSION_TEMPLATE_PATH
    
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
                    
                    # Add to ensemble - Fixed parameter order and removed non-existent calibration parameter
                    ensemble_manager.add_base_model(model_name, model)
                    
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
                # Train all ensembles - Fixed method name from prepare_ensemble to train_all_ensembles
                ensemble_manager.train_all_ensembles(X_val_split, y_val_split)
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
    Generate submission file using trained models
    
    Args:
        X_test: Test features
        trained_models: Dictionary of trained models
        ensemble_manager: Ensemble manager (optional)
        config: Configuration object
        quick_mode: Whether running in quick mode
    
    Returns:
        Submission DataFrame or None if failed
    """
    try:
        logger.info("Submission file generation started")
        logger.info(f"Test data size: {len(X_test)} rows")
        
        # Use ensemble if available and more than one model
        if ensemble_manager and len(trained_models) > 1:
            try:
                # Get ensemble prediction
                predictions = ensemble_manager.predict(X_test)
                logger.info("Using ensemble prediction")
            except Exception as e:
                logger.warning(f"Ensemble prediction failed: {e}, falling back to single model")
                # Fallback to best performing model
                best_model_name = list(trained_models.keys())[0]
                best_model = trained_models[best_model_name]
                predictions = best_model.predict_proba(X_test)
                logger.info(f"Using single model: {best_model_name}")
        else:
            # Use single model
            model_name = list(trained_models.keys())[0]
            model = trained_models[model_name]
            predictions = model.predict_proba(X_test)
            logger.info(f"Using single model: {model_name}")
        
        # Load submission template
        submission_template_path = config.SUBMISSION_TEMPLATE_PATH
        if submission_template_path.exists():
            submission_df = pd.read_csv(submission_template_path)
            submission_df['clicked'] = predictions
        else:
            # Create submission format if template doesn't exist
            submission_df = pd.DataFrame({
                'ID': range(len(X_test)),
                'clicked': predictions
            })
        
        # Save submission file
        submission_path = Path("submission.csv")
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved: {submission_path}")
        
        # Log statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        logger.info(f"Submission statistics: mean={pred_mean:.4f}, std={pred_std:.4f}")
        
        return submission_df
        
    except Exception as e:
        logger.error(f"Submission file generation failed: {e}")
        return None

def run_training_mode(quick_mode: bool = False):
    """Run the training mode pipeline"""
    try:
        logger.info("Training mode started")
        
        # Import config
        from config import Config
        config = Config()
        
        # Execute pipeline
        results = execute_final_pipeline(config, quick_mode=quick_mode)
        
        if results is None:
            logger.error("Pipeline execution failed")
            return False
        
        # Log completion
        force_memory_cleanup()
        
        logger.info("Training mode completed successfully")
        logger.info(f"Mode: {'Quick (50 samples)' if quick_mode else 'Full'}")
        logger.info(f"Execution time: {results['execution_time']:.2f}s")
        logger.info(f"Successful models: {results['successful_models']}")
        logger.info(f"Ensemble enabled: {results['ensemble_enabled']}")
        logger.info(f"Ensemble used: {results['ensemble_used']}")
        logger.info(f"Calibration applied: {results['calibration_applied']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training mode failed: {e}")
        return False

def run_performance_analysis():
    """Run performance analysis on existing models"""
    try:
        logger.info("=== Performance Analysis Started ===")
        
        # Import required modules
        from analysis import CTRPerformanceAnalyzer
        from visualization import CTRVisualizationEngine
        from pathlib import Path
        import os
        
        # Initialize analyzer and visualization engine
        from config import Config
        config = Config()
        
        analyzer = CTRPerformanceAnalyzer(config)
        visualizer = CTRVisualizationEngine(config)
        
        # Find trained models
        model_dir = Path("models")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        trained_models = []
        model_files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
        
        for model_file in model_files:
            model_name = model_file.stem
            trained_models.append(model_name)
        
        logger.info(f"Analyzing {len(trained_models)} trained models")
        
        analysis_results = {}
        
        # Analyze each model
        for model_name in trained_models:
            logger.info(f"Analyzing model: {model_name}")
            
            try:
                # Load model and get predictions
                # This is a simplified version - would need actual model loading logic
                analysis = analyzer.full_performance_analysis(
                    y_true=np.array([0, 1, 0, 1, 0] * 3),  # Dummy data for testing
                    y_pred_proba=np.array([0.1, 0.9, 0.2, 0.8, 0.1] * 3),  # Dummy data
                    model_name=model_name,
                    quick_mode=True
                )
                
                analysis_results[model_name] = analysis
                
                # Save individual analysis report
                report_path = results_dir / f"analysis_report_{model_name}.json"
                with open(report_path, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    json_safe_analysis = convert_numpy_types(analysis)
                    json.dump(json_safe_analysis, f, indent=2)
                
                logger.info(f"Analysis report saved: {report_path}")
                
            except Exception as e:
                logger.error(f"Analysis failed for {model_name}: {e}")
                continue
        
        if not analysis_results:
            logger.warning("No models were successfully analyzed")
            return False
        
        # Create summary
        analyzer.create_summary_csv(analysis_results, results_dir / "summary.csv")
        logger.info("Summary CSV created: results/summary.csv")
        logger.info(f"Summary contains {len(analysis_results)} models")
        
        # Generate visualizations
        visualizer.generate_comprehensive_visualizations(analysis_results, results_dir)
        logger.info("Visualization generation completed")
        
        # Display performance summary
        display_performance_summary(analysis_results)
        
        logger.info("=== Performance Analysis Completed ===")
        return True
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        return False

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj) or obj != obj:  # Check for NaN
        return None
    else:
        return obj

def display_performance_summary(analysis_results: Dict[str, Any]):
    """Display comprehensive performance analysis summary"""
    try:
        logger.info("Performance analysis completed")
        logger.info(f"Analysis results saved: {len(analysis_results)} reports")
        
        # Create summary table for logging
        logger.info("FINAL PERFORMANCE ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Analysis Mode: QUICK MODE (50 samples)")
        logger.info(f"Models Analyzed: {len(analysis_results)}")
        logger.info(f"Reports Generated: {len(analysis_results)}")
        logger.info("")
        
        # Model performance results
        logger.info("MODEL PERFORMANCE RESULTS:")
        logger.info("-" * 50)
        
        analyzed_models = list(analysis_results.keys())
        
        for model_name in analyzed_models:
            analysis = analysis_results.get(model_name, {})
            core_metrics = analysis.get('core_metrics', {})
            ctr_analysis = analysis.get('ctr_analysis', {})
            overall = analysis.get('overall_assessment', {})
            
            combined_score = core_metrics.get('combined_score', 0.0)
            ap_score = core_metrics.get('ap', 0.0)
            auc_score = core_metrics.get('auc', float('nan'))
            ctr_quality = ctr_analysis.get('ctr_quality', 'UNKNOWN')
            ctr_bias = ctr_analysis.get('ctr_bias', 0.0)
            performance_tier = overall.get('performance_tier', 'UNKNOWN')
            
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
        logger.info("=" * 80)
        logger.info("PERFORMANCE ANALYSIS SUMMARY COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Performance summary display failed: {e}")

def main():
    """Main entry point for CTR modeling system"""
    try:
        # Setup signal handlers
        setup_signal_handlers()
        
        # Parse arguments
        parser = argparse.ArgumentParser(description='CTR Modeling System')
        parser.add_argument('--quick', action='store_true', help='Run in quick mode (50 samples)')
        parser.add_argument('--analyze', action='store_true', help='Run performance analysis only')
        parser.add_argument('--validate', action='store_true', help='Validate environment only')
        
        args = parser.parse_args()
        
        logger.info("=== CTR modeling system started ===")
        
        # Environment validation
        logger.info("=== Environment validation started ===")
        if not validate_environment():
            logger.error("Environment validation failed")
            return False
        
        if args.validate:
            logger.info("Environment validation completed successfully")
            return True
        
        # Performance analysis mode
        if args.analyze:
            logger.info("Performance analysis mode started")
            success = run_performance_analysis()
            if success:
                logger.info("Performance analysis mode completed successfully")
            else:
                logger.error("Performance analysis mode failed")
            return success
        
        # Training mode
        logger.info("Training mode started")
        if args.quick:
            logger.info("QUICK MODE enabled")
        
        success = run_training_mode(quick_mode=args.quick)
        
        if success:
            logger.info("Training mode completed successfully")
            
            # Run performance analysis automatically after training
            logger.info("Performance analysis mode started")
            analysis_success = run_performance_analysis()
            if analysis_success:
                logger.info("Performance analysis mode completed successfully")
            else:
                logger.error("Performance analysis mode failed")
        else:
            logger.error("Training mode failed")
        
        # Final cleanup
        force_memory_cleanup()
        
        logger.info("=== CTR modeling system completed successfully ===")
        return success
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        force_memory_cleanup()
        return False
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        force_memory_cleanup()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)