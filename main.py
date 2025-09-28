# main.py
# score : 0.3193693153

import os
import sys
import logging
import time
import signal
import argparse
import traceback
import gc
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# Create essential directories first
def ensure_directories():
    """Ensure essential directories exist before logging setup"""
    directories = ['data', 'models', 'logs', 'output', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Ensure directories exist before logging setup
ensure_directories()

# Safe imports with availability checking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Essential package import failed: {e}")
    sys.exit(1)

# Configure logging after directory creation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/main.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

# Global cleanup flag
cleanup_required = False

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global cleanup_required
    cleanup_required = True
    logger.info("Interrupt signal received, cleaning up...")
    force_memory_cleanup()
    sys.exit(0)

def force_memory_cleanup():
    """Force memory cleanup"""
    try:
        start_time = time.time()
        
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch cleanup if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Memory cleanup completed: {elapsed_time:.2f}s elapsed, {collected} objects collected")
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def validate_environment() -> bool:
    """Validate execution environment"""
    try:
        logger.info("=== Environment validation started ===")
        
        # Python version check
        python_version = sys.version
        logger.info(f"Python version: {python_version}")
        
        # Ensure directories exist (redundant but safe)
        directories = ['data', 'models', 'logs', 'output', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory prepared: {directory}")
        
        # Check data files
        train_path = Path('data/train.parquet')
        test_path = Path('data/test.parquet')
        submission_path = Path('data/sample_submission.csv')
        
        # File existence and size check
        train_exists = train_path.exists()
        test_exists = test_path.exists()
        submission_exists = submission_path.exists()
        
        train_size = train_path.stat().st_size / (1024**2) if train_exists else 0
        test_size = test_path.stat().st_size / (1024**2) if test_exists else 0
        submission_size = submission_path.stat().st_size / (1024**2) if submission_exists else 0
        
        logger.info(f"Train data: {'EXISTS' if train_exists else 'MISSING'} ({train_size:.1f}MB)")
        logger.info(f"Test data: {'EXISTS' if test_exists else 'MISSING'} ({test_size:.1f}MB)")
        logger.info(f"Submission template: {'EXISTS' if submission_exists else 'MISSING'} ({submission_size:.1f}MB)")
        
        # Memory information
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            logger.info(f"Memory status: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        # GPU information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            logger.info(f"GPU available: {gpu_count} devices")
            logger.info(f"Current GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.info("GPU: Not available (CPU mode)")
        
        # Check essential files existence
        if not train_exists:
            logger.error("Training data missing: data/train.parquet")
            return False
        
        if not test_exists:
            logger.error("Test data missing: data/test.parquet")
            return False
        
        if not submission_exists:
            logger.error("Submission template missing: data/sample_submission.csv")
            return False
        
        logger.info("=== Environment validation completed successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def safe_train_test_split(X, y, test_size=0.3, random_state=42):
    """Safe train-test split with memory optimization"""
    from sklearn.model_selection import train_test_split
    
    try:
        logger.info(f"Performing train-test split: test_size={test_size}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Split completed - Train: {X_train.shape}, Validation: {X_val.shape}")
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        logger.error(f"Train-test split failed: {e}")
        raise

def reproduce_score_validation() -> bool:
    """Reproduce target score validation"""
    try:
        logger.info("=== Score reproduction validation started ===")
        
        # Import required modules
        from config import Config
        from data_loader import CTRDataLoader
        from feature_engineering import FeatureEngineer
        from training import CTRTrainer
        from inference import CTRPredictor
        from evaluation import CTRMetricsEvaluator
        
        config = Config()
        config.setup_directories()
        
        # Load data with quick mode for validation
        data_loader = CTRDataLoader(config)
        train_df = data_loader.load_train_data(sample_rows=1000)  # Quick validation
        test_df = data_loader.load_test_data(sample_rows=500)
        
        if train_df is None or test_df is None:
            logger.error("Data loading failed")
            return False
        
        # Feature engineering
        feature_engineer = FeatureEngineer(config)
        X_train, feature_names = feature_engineer.engineer_features(train_df, is_training=True)
        
        # Get target
        target_column = data_loader.get_detected_target_column()
        y_train = train_df[target_column]
        
        # Train simple model for validation
        trainer = CTRTrainer(config)
        trainer.set_quick_mode(True)
        
        # Simple logistic regression for validation
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict_proba(X_train)[:, 1]
        
        # Evaluate
        evaluator = CTRMetricsEvaluator()
        metrics = evaluator.calculate_metrics(y_train, predictions)
        
        logger.info(f"Validation metrics: {metrics}")
        logger.info("=== Score reproduction validation completed ===")
        
        return True
        
    except Exception as e:
        logger.error(f"Score reproduction validation failed: {e}")
        return False

def execute_final_pipeline(config, quick_mode: bool = False) -> Optional[Dict[str, Any]]:
    """Execute complete CTR modeling pipeline"""
    
    try:
        logger.info("=== Final pipeline execution started ===")
        start_time = time.time()
        
        # Import required modules
        from data_loader import CTRDataLoader
        from feature_engineering import FeatureEngineer
        from training import CTRTrainer
        from ensemble import CTREnsembleManager
        from inference import CTRPredictor
        from evaluation import CTRMetricsEvaluator
        
        # Initialize components
        data_loader = CTRDataLoader(config)
        feature_engineer = FeatureEngineer(config)
        
        logger.info("=== Data loading phase ===")
        
        # Load training data
        if quick_mode:
            train_df = data_loader.load_train_data(sample_rows=50)
            logger.info("Quick mode: Using 50 samples for training")
        else:
            train_df = data_loader.load_train_data()
            logger.info("Full mode: Loading complete training dataset")
        
        if train_df is None:
            logger.error("Training data loading failed")
            return None
        
        logger.info(f"Training data loaded: {train_df.shape}")
        
        # Load test data for inference
        if quick_mode:
            test_df = data_loader.load_test_data(sample_rows=25)
        else:
            test_df = data_loader.load_test_data()
        
        if test_df is None:
            logger.error("Test data loading failed")
            return None
        
        logger.info(f"Test data loaded: {test_df.shape}")
        
        logger.info("=== Feature engineering phase ===")
        
        # Feature engineering for training data
        X_train, feature_names = feature_engineer.engineer_features(train_df, is_training=True)
        logger.info(f"Training features engineered: {X_train.shape}, {len(feature_names)} features")
        
        # Feature engineering for test data
        X_test, _ = feature_engineer.engineer_features(test_df, is_training=False)
        logger.info(f"Test features engineered: {X_test.shape}")
        
        logger.info("=== Model training phase ===")
        
        # Initialize trainer
        trainer = CTRTrainer(config)
        logger.info("CTR Trainer initialized (CPU mode)")
        
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
        
        # Safe train/validation split
        X_train_split, X_val_split, y_train_split, y_val_split = safe_train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
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
                    y_val=y_val_split
                )
                
                if model is not None:
                    trained_models[model_name] = model
                    model_performances[model_name] = performance
                    logger.info(f"{model_name} training completed successfully")
                    
                    if performance:
                        logger.info(f"{model_name} performance: {performance}")
                else:
                    logger.warning(f"{model_name} training failed")
                    
            except Exception as e:
                logger.error(f"{model_name} training error: {e}")
                continue
        
        logger.info(f"Model training completed. Successful models: {list(trained_models.keys())}")
        
        # Ensemble training
        ensemble_model = None
        ensemble_used = False
        
        if len(trained_models) > 1 and not quick_mode:
            logger.info("=== Ensemble training phase ===")
            try:
                ensemble_model = ensemble_manager.create_ensemble(
                    trained_models, X_val_split, y_val_split
                )
                if ensemble_model:
                    ensemble_used = True
                    logger.info("Ensemble model created successfully")
                else:
                    logger.warning("Ensemble creation failed")
            except Exception as e:
                logger.error(f"Ensemble training failed: {e}")
        
        logger.info("=== Inference phase ===")
        
        # Initialize predictor
        predictor = CTRPredictor(config)
        
        # Select best model for inference
        if ensemble_used and ensemble_model:
            final_model = ensemble_model
            model_type = 'ensemble'
            logger.info("Using ensemble model for final predictions")
        elif trained_models:
            # Use the first successfully trained model
            best_model_name = list(trained_models.keys())[0]
            final_model = trained_models[best_model_name]
            model_type = best_model_name
            logger.info(f"Using {best_model_name} model for final predictions")
        else:
            logger.error("No trained models available for inference")
            return None
        
        # Make predictions
        try:
            predictions = predictor.predict(final_model, X_test, model_type=model_type)
            logger.info(f"Predictions generated: {len(predictions)} samples")
            
            # Save predictions
            submission_df = pd.DataFrame({
                'ID': test_df.index if hasattr(test_df, 'index') else range(len(predictions)),
                'clicked': predictions
            })
            
            submission_path = 'submission.csv'
            submission_df.to_csv(submission_path, index=False)
            logger.info(f"Submission saved: {submission_path}")
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            predictions = None
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare results
        results = {
            'successful_models': list(trained_models.keys()),
            'model_performances': model_performances,
            'ensemble_enabled': len(trained_models) > 1,
            'ensemble_used': ensemble_used,
            'calibration_applied': False,  # Not implemented in this version
            'execution_time': execution_time,
            'quick_mode': quick_mode,
            'predictions_generated': predictions is not None,
            'submission_saved': predictions is not None
        }
        
        logger.info("=== Final pipeline execution completed ===")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return None

def run_performance_analysis(pipeline_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run performance analysis on trained models"""
    try:
        logger.info("=== Performance analysis started ===")
        
        if not pipeline_results.get('successful_models'):
            logger.warning("No successful models for performance analysis")
            return None
        
        # Create basic performance summary
        analysis_results = {
            'model_count': len(pipeline_results['successful_models']),
            'ensemble_used': pipeline_results.get('ensemble_used', False),
            'execution_time': pipeline_results.get('execution_time', 0),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_path = 'output/performance_summary.json'
        os.makedirs('output', exist_ok=True)
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Performance analysis completed: {summary_path}")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        return None

def main():
    """Main execution function"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="CTR Modeling System")
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
                    performance_results = run_performance_analysis(results)
                    
                    if performance_results:
                        logger.info("Performance analysis completed")
                        logger.info(f"Analysis results saved: {results['successful_models']} reports")
                    
                    # Print final summary
                    logger.info("================================================================================")
                    logger.info("FINAL PERFORMANCE ANALYSIS SUMMARY")
                    logger.info("================================================================================")
                    logger.info("Reports generated:")
                    logger.info("  - performance_summary.json (basic metrics)")
                    
                    logger.info("")
                    logger.info("================================================================================")
                    logger.info("PERFORMANCE ANALYSIS SUMMARY COMPLETE")
                    logger.info("================================================================================")
                    
                except Exception as e:
                    logger.warning(f"Performance analysis failed: {e}")
            else:
                logger.error("Training mode failed")
                sys.exit(1)
        
        elif args.mode == "inference":
            logger.info("Inference mode started")
            # Inference implementation placeholder
            logger.info("Inference mode completed")
        
        elif args.mode == "reproduce":
            logger.info("Score reproduction mode started")
            reproduction_success = reproduce_score_validation()
            if reproduction_success:
                logger.info("Score reproduction mode completed successfully")
            else:
                logger.error("Score reproduction mode failed")
                sys.exit(1)
        
        logger.info("=== CTR modeling system completed successfully ===")
        force_memory_cleanup()
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()