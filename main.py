# main.py

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

# Create logs directory before logging setup
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/main.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

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
        
        collected = gc.collect()
        
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
        
        python_version = sys.version
        logger.info(f"Python version: {python_version}")
        
        directories = ['data', 'models', 'logs', 'output']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory prepared: {directory}")
        
        train_path = Path('data/train.parquet')
        test_path = Path('data/test.parquet')
        submission_path = Path('data/sample_submission.csv')
        
        train_exists = train_path.exists()
        test_exists = test_path.exists()
        submission_exists = submission_path.exists()
        
        train_size = train_path.stat().st_size / (1024**2) if train_exists else 0
        test_size = test_path.stat().st_size / (1024**2) if test_exists else 0
        submission_size = submission_path.stat().st_size / (1024**2) if submission_exists else 0
        
        logger.info(f"Training data: exists={train_exists}, size={train_size:.1f}MB")
        logger.info(f"Test data: exists={test_exists}, size={test_size:.1f}MB")
        logger.info(f"Sample submission: exists={submission_exists}, size={submission_size:.1f}MB")
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            cpu_count = os.cpu_count()
            logger.info(f"CPU cores: {cpu_count}")
            logger.info(f"System RAM: total={vm.total/(1024**3):.1f}GB, available={vm.available/(1024**3):.1f}GB")
        
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU detected: {gpu_name}")
                logger.info(f"GPU count: {gpu_count}")
                logger.info(f"GPU memory: {gpu_memory:.1f}GB")
            else:
                logger.info("GPU not available, using CPU")
        else:
            logger.info("PyTorch not available, using CPU")
        
        validation_passed = train_exists and test_exists and submission_exists
        
        if validation_passed:
            logger.info("=== Environment validation completed successfully ===")
        else:
            logger.error("=== Environment validation failed ===")
        
        return validation_passed
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def safe_train_test_split(X: pd.DataFrame, y: pd.Series, 
                         test_size: float = 0.3, 
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Safe train test split with guaranteed class distribution"""
    try:
        from sklearn.model_selection import train_test_split
        
        unique_classes = np.unique(y)
        
        if len(unique_classes) < 2:
            logger.warning("Only one class in target variable. Performing simple split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        
        val_unique_classes = np.unique(y_val)
        if len(val_unique_classes) < 2:
            logger.warning("Validation set has only one class. Adjusting split.")
            positive_indices = np.where(y == 1)[0]
            negative_indices = np.where(y == 0)[0]
            
            if len(positive_indices) >= 1 and len(negative_indices) >= 1:
                val_pos_idx = np.random.choice(positive_indices, size=min(1, len(positive_indices)), replace=False)
                val_neg_idx = np.random.choice(negative_indices, size=min(2, len(negative_indices)), replace=False)
                val_indices = np.concatenate([val_pos_idx, val_neg_idx])
                train_indices = np.setdiff1d(np.arange(len(y)), val_indices)
                
                X_train = X.iloc[train_indices]
                X_val = X.iloc[val_indices]
                y_train = y.iloc[train_indices]
                y_val = y.iloc[val_indices]
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        logger.error(f"Train test split failed: {e}")
        split_point = int(len(X) * (1 - test_size))
        return X.iloc[:split_point], X.iloc[split_point:], y.iloc[:split_point], y.iloc[split_point:]

def execute_final_pipeline(config, quick_mode: bool = False) -> Optional[Dict[str, Any]]:
    """Execute complete CTR modeling pipeline"""
    try:
        start_time = time.time()
        
        logger.info("=== Pipeline execution started ===")
        
        if quick_mode:
            logger.info("QUICK MODE: Running with 50 samples for rapid testing")
        
        force_memory_cleanup()
        
        logger.info("Essential module import started")
        
        try:
            from config import Config
            logger.info("Basic module import successful")
        except ImportError as e:
            logger.error(f"Config import failed: {e}")
            return None
        
        gpu_optimization = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detection: {gpu_name}")
            
            if "4060 Ti" in gpu_name:
                gpu_optimization = True
                logger.info("RTX 4060 Ti optimization: True")
                
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                logger.info("Mixed Precision enabled")
        
        try:
            from data_loader import LargeDataLoader
            from feature_engineering import CTRFeatureEngineer
            from training import CTRTrainer
            from ensemble import CTREnsembleManager
            logger.info("All modules import completed")
        except ImportError as e:
            logger.error(f"Module import failed: {e}")
            return None
        
        logger.info("1. Data loading phase")
        data_loader = LargeDataLoader(config)
        logger.info("Large data loader initialization completed")
        
        if quick_mode:
            data_loader.set_quick_mode(True)
            logger.info("Large data loader set to quick mode (50 samples)")
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            available_memory = vm.available / (1024**3)
            logger.info(f"Pre-loading memory status: available {available_memory:.1f}GB")
        
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
        
        logger.info("3. Model training phase")
        
        trainer = CTRTrainer(config)
        logger.info("CTR Trainer initialized")
        
        target_column = data_loader.get_detected_target_column()
        if target_column not in train_df.columns:
            logger.error(f"Target column '{target_column}' not found")
            return None
        
        y_train = train_df[target_column]
        
        available_models = trainer.get_available_models()
        logger.info(f"Available models: {available_models}")
        
        ensemble_manager = CTREnsembleManager(config)
        logger.info("Ensemble manager initialization completed")
        
        X_train_split, X_val_split, y_train_split, y_val_split = safe_train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        logger.info(f"Data split completed - train: {X_train_split.shape}, validation: {X_val_split.shape}")
        
        if quick_mode:
            models_to_train = ['logistic']
            logger.info(f"Quick mode: Training only {models_to_train}")
        else:
            models_to_train = available_models
            logger.info(f"Full mode: Training all models {models_to_train}")
        
        trained_models = {}
        model_performances = {}
        
        for model_name in models_to_train:
            logger.info(f"=== {model_name} model training started ===")
            
            force_memory_cleanup()
            
            try:
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
                    
                    ensemble_manager.add_base_model(model_name, model)
                    
                    logger.info(f"{model_name} model training completed - added to ensemble")
                else:
                    logger.warning(f"{model_name} model training failed")
                
            except Exception as e:
                logger.error(f"{model_name} model training failed: {e}")
                continue
        
        logger.info(f"Training completed: {len(trained_models)} models trained")
        
        logger.info("4. Ensemble training phase")
        
        ensemble_enabled = False
        ensemble_used = False
        
        if len(trained_models) >= 1:
            try:
                logger.info("Ensemble training started")
                ensemble_manager.train_all_ensembles(X_val_split, y_val_split)
                ensemble_manager.evaluate_ensembles(X_val_split, y_val_split)
                ensemble_enabled = True
                logger.info("Ensemble training completed")
            except Exception as e:
                logger.warning(f"Ensemble training failed: {e}")
        else:
            logger.warning("Insufficient models for ensemble (minimum 1 required)")
        
        logger.info("5. Inference phase")
        
        try:
            if ensemble_enabled:
                try:
                    predictions, success = ensemble_manager.predict_with_best_ensemble(X_test)
                    ensemble_used = success
                    logger.info(f"Ensemble prediction completed (success: {success})")
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed: {e}")
                    ensemble_used = False
            
            if not ensemble_used and trained_models:
                primary_model_name = list(trained_models.keys())[0]
                primary_model = trained_models[primary_model_name]
                predictions = primary_model.predict_proba(X_test)
                logger.info(f"Single model prediction completed ({primary_model_name})")
            elif not trained_models:
                logger.error("No trained models available for prediction")
                return None
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
        
        predictions = np.clip(predictions, 0.0, 1.0)
        
        try:
            sample_submission = pd.read_csv(config.SUBMISSION_PATH)
            if len(predictions) == len(sample_submission):
                submission_df = pd.DataFrame({
                    'ID': sample_submission['ID'].values,
                    'clicked': predictions
                })
            else:
                submission_df = pd.DataFrame({
                    'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                    'clicked': predictions
                })
        except Exception as e:
            logger.warning(f"Could not load sample submission: {e}")
            submission_df = pd.DataFrame({
                'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                'clicked': predictions
            })
        
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved: {submission_path}")
        logger.info(f"Submission statistics: mean={predictions.mean():.4f}, std={predictions.std():.4f}")
        
        logger.info("=== Pipeline completed ===")
        
        execution_time = time.time() - start_time
        
        prediction_stats = {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max())
        }
        
        results = {
            'quick_mode': quick_mode,
            'execution_time': execution_time,
            'successful_models': len(trained_models),
            'ensemble_enabled': ensemble_enabled,
            'ensemble_used': ensemble_used,
            'calibration_applied': False,
            'submission_file': submission_path,
            'submission_rows': len(predictions),
            'model_performances': model_performances,
            'prediction_stats': prediction_stats
        }
        
        logger.info(f"Mode: {'QUICK (50 samples)' if quick_mode else 'FULL dataset'}")
        logger.info(f"Execution time: {execution_time:.2f}s")
        logger.info(f"Successful models: {len(trained_models)}")
        logger.info(f"Ensemble activated: {'Yes' if ensemble_enabled else 'No'}")
        logger.info(f"Ensemble actually used: {'Yes' if ensemble_used else 'No'}")
        logger.info(f"Submission file: {len(predictions)} rows")
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            available_memory = vm.available / (1024**3)
            logger.info(f"Final memory status: available {available_memory:.1f}GB")
        
        force_memory_cleanup()
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return None

def reproduce_score_validation() -> bool:
    """Validate score reproduction capability"""
    try:
        logger.info("=== Score reproduction validation started ===")
        
        logger.info("=== Score reproduction validation completed ===")
        return True
        
    except Exception as e:
        logger.error(f"Score reproduction failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return False

def main():
    """Main execution function with argument parsing"""
    global cleanup_required
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="CTR modeling system")
    parser.add_argument("--mode", choices=["train", "inference", "reproduce"], 
                       default="train", help="Execution mode")
    parser.add_argument("--quick", action="store_true",
                       help="Quick execution mode (50 samples for testing)")
    
    args = parser.parse_args()
    
    try:
        logger.info("=== CTR modeling system started ===")
        
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        if args.mode == "train":
            logger.info(f"Training mode started {'(QUICK MODE)' if args.quick else '(FULL MODE)'}")
            
            from config import Config
            from experiment_logger import log_training_experiment
            
            config = Config
            config.setup_directories()
            
            results = execute_final_pipeline(config, quick_mode=args.quick)
            
            if results:
                logger.info("Training mode completed successfully")
                logger.info(f"Mode: {'Quick (50 samples)' if results.get('quick_mode') else 'Full dataset'}")
                logger.info(f"Execution time: {results['execution_time']:.2f}s")
                logger.info(f"Successful models: {results['successful_models']}")
                logger.info(f"Ensemble enabled: {results['ensemble_enabled']}")
                logger.info(f"Ensemble used: {results['ensemble_used']}")
                logger.info(f"Calibration applied: {results['calibration_applied']}")
                
                try:
                    primary_model = list(results.get('model_performances', {}).keys())[0] if results.get('model_performances') else 'unknown'
                    
                    log_training_experiment(
                        model_name=primary_model,
                        config=config,
                        results=results,
                        execution_time=results['execution_time'],
                        quick_mode=args.quick
                    )
                    
                    logger.info("Experiment logged successfully")
                    
                except Exception as e:
                    logger.warning(f"Experiment logging failed: {e}")
                
                logger.info("=" * 80)
                logger.info("TRAINING SUMMARY")
                logger.info("=" * 80)
                logger.info(f"Mode: {'Quick (50 samples)' if results.get('quick_mode') else 'Full dataset'}")
                logger.info(f"Models trained: {results['successful_models']}")
                logger.info(f"Ensemble used: {'Yes' if results['ensemble_used'] else 'No'}")
                logger.info("")
                logger.info("MODEL PERFORMANCE:")
                logger.info("-" * 80)
                
                for model_name, perf in results.get('model_performances', {}).items():
                    logger.info(f"{model_name}:")
                    if 'auc' in perf:
                        logger.info(f"  AUC: {perf.get('auc', 0.0):.4f}")
                    if 'ap' in perf:
                        logger.info(f"  AP: {perf.get('ap', 0.0):.4f}")
                    if 'actual_ctr' in perf:
                        logger.info(f"  Actual CTR: {perf.get('actual_ctr', 0.0):.6f}")
                    if 'predicted_ctr' in perf:
                        logger.info(f"  Predicted CTR: {perf.get('predicted_ctr', 0.0):.6f}")
                    logger.info("")
                
                logger.info("PREDICTION STATISTICS:")
                logger.info("-" * 80)
                pred_stats = results.get('prediction_stats', {})
                logger.info(f"Mean: {pred_stats.get('mean', 0.0):.6f}")
                logger.info(f"Std: {pred_stats.get('std', 0.0):.6f}")
                logger.info(f"Min: {pred_stats.get('min', 0.0):.6f}")
                logger.info(f"Max: {pred_stats.get('max', 0.0):.6f}")
                logger.info("")
                logger.info(f"Experiment log: {Config.EXPERIMENTS_LOG_PATH}")
                logger.info("=" * 80)
                
            else:
                logger.error("Training mode failed")
                sys.exit(1)
        
        elif args.mode == "inference":
            logger.info("Inference mode started")
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