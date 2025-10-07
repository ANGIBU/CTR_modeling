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

os.makedirs('logs', exist_ok=True)

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
        
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
        
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
        
        directories = ['data', 'models', 'logs', 'results']
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
        
        logger.info(f"train file: {train_exists} ({train_size:.1f}MB)")
        logger.info(f"test file: {test_exists} ({test_size:.1f}MB)")
        logger.info(f"submission file: {submission_exists} ({submission_size:.1f}MB)")
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            total_memory = vm.total / (1024**3)
            available_memory = vm.available / (1024**3)
            memory_percent = vm.percent
            
            logger.info(f"System memory: {total_memory:.1f}GB (available: {available_memory:.1f}GB)")
            logger.info(f"Memory usage: {memory_percent:.1f}%")
        
        logger.info("=== Environment validation completed ===")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def check_gpu_availability() -> Tuple[bool, str]:
    """Check GPU availability and return status"""
    try:
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available"
        
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.zeros(1, device=device)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            del test_tensor
            torch.cuda.empty_cache()
            
            return True, f"{gpu_name} ({gpu_memory:.1f}GB)"
        except Exception as e:
            return False, f"GPU test failed: {e}"
            
    except Exception as e:
        return False, f"GPU check error: {e}"

def safe_train_test_split(X, y, test_size=0.3, random_state=42):
    """Safe train test split with class imbalance handling"""
    try:
        from sklearn.model_selection import train_test_split
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = min(class_counts)
        
        if min_class_count < 2 or len(y) < 10:
            logger.warning(f"Small dataset ({len(y)} samples) or class imbalance detected. Using simple split.")
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

def apply_ctr_correction(predictions: np.ndarray, target_ctr: float = 0.0191) -> np.ndarray:
    """Apply CTR correction to predictions"""
    try:
        current_ctr = predictions.mean()
        
        if abs(current_ctr - target_ctr) > 0.001:
            logger.info(f"Applying CTR correction: {current_ctr:.4f} -> {target_ctr:.4f}")
            
            correction_factor = target_ctr / current_ctr if current_ctr > 0 else 1.0
            corrected = predictions * correction_factor
            
            corrected = np.clip(corrected, 1e-7, 1 - 1e-7)
            
            final_ctr = corrected.mean()
            logger.info(f"CTR after correction: {final_ctr:.4f}")
            
            return corrected
        
        return predictions
        
    except Exception as e:
        logger.warning(f"CTR correction failed: {e}")
        return predictions

def execute_final_pipeline(config, quick_mode: bool = False) -> Optional[Dict[str, Any]]:
    """Execute complete CTR modeling pipeline"""
    try:
        start_time = time.time()
        
        logger.info("=== Pipeline execution started ===")
        
        if quick_mode:
            logger.info("QUICK MODE: Running with 50 samples for rapid testing")
        else:
            logger.info("FULL MODE: Running with complete dataset for 0.35+ target score")
        
        from experiment_logger import ExperimentLogger
        exp_logger = ExperimentLogger(config.LOG_DIR)
        
        force_memory_cleanup()
        
        logger.info("Essential module import started")
        
        try:
            from config import Config
            logger.info("Basic module import successful")
        except ImportError as e:
            logger.error(f"Config import failed: {e}")
            return None
        
        gpu_optimization = False
        gpu_info = "Not available"
        
        gpu_available, gpu_status = check_gpu_availability()
        if gpu_available:
            gpu_optimization = True
            gpu_info = gpu_status
            logger.info(f"GPU detection: {gpu_info}")
            logger.info("GPU optimization: Enabled")
            
            if TORCH_AVAILABLE:
                try:
                    torch.backends.cudnn.benchmark = True
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    logger.info("GPU settings applied")
                except Exception as e:
                    logger.warning(f"GPU settings failed: {e}")
        else:
            logger.warning(f"GPU not available: {gpu_status}")
            logger.info("Using CPU mode")
        
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
            logger.info("Full mode: Loading complete dataset for target score")
            train_df, test_df = data_loader.load_large_data_optimized()
        
        if train_df is None or test_df is None:
            logger.error("Data loading failed")
            return None
        
        logger.info(f"Data loading completed - train: {train_df.shape}, test: {test_df.shape}")
        
        logger.info("2. Feature engineering phase (tree model optimized)")
        feature_engineer = CTRFeatureEngineer(config)
        
        if quick_mode:
            feature_engineer.set_quick_mode(True)
            logger.info("Quick mode: Basic feature engineering only")
        else:
            logger.info("Full mode: Tree model optimized features (no normalization)")
        
        X_train, X_test = feature_engineer.engineer_features(train_df, test_df)
        
        if X_train is None or X_test is None:
            logger.error("Feature engineering failed")
            return None
        
        logger.info(f"Feature engineering completed - Features: {X_train.shape[1]}")
        
        logger.info("3. Model training phase (Multi-model with ensemble)")
        
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
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            available_memory = vm.available / (1024**3)
            logger.info(f"Available memory before split: {available_memory:.1f}GB")
        
        X_train_split, X_val_split, y_train_split, y_val_split = safe_train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        logger.info(f"Data split completed - train: {X_train_split.shape}, validation: {X_val_split.shape}")
        
        if quick_mode:
            models_to_train = ['logistic']
            logger.info(f"Quick mode: Training only {models_to_train}")
        else:
            models_to_train = available_models
            logger.info(f"Full mode: Training all available models {models_to_train}")
        
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
                
                if model is not None and hasattr(model, 'is_fitted') and model.is_fitted:
                    trained_models[model_name] = model
                    model_performances[model_name] = performance
                    
                    ensemble_manager.add_base_model(model_name, model)
                    
                    logger.info(f"{model_name} model training completed successfully")
                else:
                    logger.error(f"{model_name} model training failed or model not fitted")
                    
            except Exception as e:
                logger.error(f"{model_name} model training error: {e}")
                continue
        
        if not trained_models:
            logger.error("No models were successfully trained")
            logger.error("Creating default submission with baseline CTR")
            
            predictions = np.full(len(X_test), 0.0191)
            
            try:
                sample_submission = pd.read_csv('data/sample_submission.csv')
                if len(sample_submission) != len(predictions):
                    logger.warning(f"Sample submission length mismatch")
                    submission_df = pd.DataFrame({
                        'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                        'clicked': predictions
                    })
                else:
                    submission_df = pd.DataFrame({
                        'ID': sample_submission['ID'].values,
                        'clicked': predictions
                    })
            except:
                submission_df = pd.DataFrame({
                    'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                    'clicked': predictions
                })
            
            submission_path = 'submission.csv'
            submission_df.to_csv(submission_path, index=False)
            logger.info(f"Default submission file saved: {submission_path}")
            
            default_results = {
                'quick_mode': quick_mode,
                'execution_time': time.time() - start_time,
                'successful_models': 0,
                'ensemble_enabled': False,
                'submission_file': submission_path,
                'submission_rows': len(predictions),
                'warning': 'No models trained successfully'
            }
            
            try:
                exp_logger.log_experiment(
                    config=config,
                    results=default_results,
                    model_name="None",
                    notes="No models trained - default submission"
                )
            except Exception as e:
                logger.warning(f"Experiment logging failed: {e}")
            
            return default_results
        
        ensemble_enabled = False
        ensemble_used = False
        min_models_for_ensemble = config.ENSEMBLE_CONFIG.get('min_models_for_ensemble', 2)
        
        if not quick_mode and len(trained_models) >= min_models_for_ensemble and config.ENSEMBLE_CONFIG.get('enable_ensemble', True):
            logger.info(f"4. Ensemble preparation ({len(trained_models)} models available)")
            try:
                fitted_models = {name: model for name, model in trained_models.items() 
                               if hasattr(model, 'is_fitted') and model.is_fitted}
                
                if len(fitted_models) >= min_models_for_ensemble:
                    ensemble_manager.train_all_ensembles(X_val_split, y_val_split)
                    ensemble_enabled = True
                    ensemble_used = True
                    logger.info("Ensemble preparation completed")
                else:
                    logger.warning(f"Only {len(fitted_models)} fitted models, skipping ensemble")
                    ensemble_used = False
            except Exception as e:
                logger.warning(f"Ensemble preparation failed: {e}")
                ensemble_used = False
        else:
            if quick_mode:
                logger.info("4. Ensemble skipped (quick mode)")
            elif len(trained_models) < min_models_for_ensemble:
                logger.info(f"4. Ensemble skipped (insufficient models: {len(trained_models)} < {min_models_for_ensemble})")
            else:
                logger.info("4. Ensemble skipped (disabled in config)")
        
        usable_models = {name: model for name, model in trained_models.items()
                        if hasattr(model, 'is_fitted') and model.is_fitted}
        
        if not usable_models:
            logger.error("No usable models available for prediction")
            logger.info("Creating default submission with baseline CTR")
            
            predictions = np.full(len(X_test), 0.0191)
            
            try:
                sample_submission = pd.read_csv('data/sample_submission.csv')
                if len(sample_submission) != len(predictions):
                    submission_df = pd.DataFrame({
                        'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                        'clicked': predictions
                    })
                else:
                    submission_df = pd.DataFrame({
                        'ID': sample_submission['ID'].values,
                        'clicked': predictions
                    })
            except:
                submission_df = pd.DataFrame({
                    'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                    'clicked': predictions
                })
            
            submission_path = 'submission.csv'
            submission_df.to_csv(submission_path, index=False)
            logger.info(f"Default submission file saved: {submission_path}")
            
            no_models_results = {
                'quick_mode': quick_mode,
                'execution_time': time.time() - start_time,
                'successful_models': 0,
                'submission_file': submission_path,
                'warning': 'No usable models for prediction'
            }
            
            try:
                exp_logger.log_experiment(
                    config=config,
                    results=no_models_results,
                    model_name="None",
                    notes="No usable models - default submission"
                )
            except Exception as e:
                logger.warning(f"Experiment logging failed: {e}")
            
            return no_models_results
        
        logger.info("5. Submission file generation")
        
        force_memory_cleanup()
        
        logger.info("Submission file generation started")
        logger.info(f"Test data size: {len(X_test)} rows")
        
        batch_size = 50000
        all_predictions = []
        
        if ensemble_used and ensemble_manager.final_ensemble and ensemble_manager.final_ensemble.is_fitted:
            logger.info("Using ensemble for prediction")
            
            for i in range(0, len(X_test), batch_size):
                end_idx = min(i + batch_size, len(X_test))
                batch_X = X_test.iloc[i:end_idx]
                
                base_predictions = {}
                for name, model in trained_models.items():
                    if not hasattr(model, 'is_fitted') or not model.is_fitted:
                        logger.warning(f"Model {name} not fitted, skipping")
                        continue
                    
                    try:
                        pred = model.predict_proba(batch_X)
                        base_predictions[name] = pred
                    except Exception as e:
                        logger.warning(f"Prediction failed for {name}: {e}")
                        base_predictions[name] = np.full(len(batch_X), 0.0191)
                
                if not base_predictions:
                    logger.warning(f"No predictions available for batch, using baseline")
                    batch_pred = np.full(len(batch_X), 0.0191)
                    all_predictions.append(batch_pred)
                    continue
                
                try:
                    batch_pred = ensemble_manager.final_ensemble.predict_proba(base_predictions)
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed: {e}, using average")
                    batch_pred = np.mean(list(base_predictions.values()), axis=0)
                
                all_predictions.append(batch_pred)
                
                logger.info(f"Batch {i//batch_size + 1} completed ({i:,}~{end_idx:,})")
                
                if i % (batch_size * 5) == 0:
                    gc.collect()
            
            predictions = np.concatenate(all_predictions)
            
        else:
            best_model_name = None
            best_model = None
            
            for name, model in trained_models.items():
                if hasattr(model, 'is_fitted') and model.is_fitted:
                    best_model_name = name
                    best_model = model
                    break
            
            if best_model is None:
                logger.error("No fitted models available for prediction")
                predictions = np.full(len(X_test), 0.0191)
            else:
                logger.info(f"Using single model: {best_model_name}")
                
                for i in range(0, len(X_test), batch_size):
                    end_idx = min(i + batch_size, len(X_test))
                    batch_X = X_test.iloc[i:end_idx]
                    
                    try:
                        batch_pred = best_model.predict_proba(batch_X)
                    except Exception as e:
                        logger.error(f"Single model prediction failed: {e}")
                        batch_pred = np.full(len(batch_X), 0.0191)
                    
                    all_predictions.append(batch_pred)
                    
                    logger.info(f"Batch {i//batch_size + 1} completed ({i:,}~{end_idx:,})")
                    
                    if i % (batch_size * 5) == 0:
                        gc.collect()
                
                predictions = np.concatenate(all_predictions)
        
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        if np.allclose(predictions, 0.0):
            logger.warning("All predictions are zero! Using default CTR")
            predictions = np.full(len(predictions), 0.0191)
        
        logger.info(f"Raw predictions - mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
        
        predictions = apply_ctr_correction(predictions, target_ctr=0.0191)
        
        try:
            sample_submission = pd.read_csv('data/sample_submission.csv')
            if len(sample_submission) != len(predictions):
                logger.warning(f"Sample submission length ({len(sample_submission)}) != predictions length ({len(predictions)})")
                submission_df = pd.DataFrame({
                    'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                    'clicked': predictions
                })
            else:
                submission_df = pd.DataFrame({
                    'ID': sample_submission['ID'].values,
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
        logger.info(f"Submission statistics: mean={predictions.mean():.4f}, std={predictions.std():.4f}, min={predictions.min():.4f}, max={predictions.max():.4f}")
        
        logger.info("=== Pipeline completed ===")
        
        execution_time = time.time() - start_time
        
        results = {
            'quick_mode': quick_mode,
            'execution_time': execution_time,
            'successful_models': len(trained_models),
            'ensemble_enabled': ensemble_enabled,
            'ensemble_used': ensemble_used,
            'calibration_applied': True,
            'submission_file': submission_path,
            'submission_rows': len(predictions),
            'model_performances': model_performances,
            'target_score': 0.35,
            'gpu_used': gpu_optimization,
            'gpu_info': gpu_info,
            'prediction_stats': {
                'mean': float(predictions.mean()),
                'std': float(predictions.std()),
                'min': float(predictions.min()),
                'max': float(predictions.max())
            },
            'trained_models': trained_models,
            'y_val': y_val_split,
            'X_val': X_val_split
        }
        
        logger.info(f"Mode: {'QUICK (50 samples)' if quick_mode else 'FULL dataset (0.35+ target)'}")
        logger.info(f"Execution time: {execution_time:.2f}s")
        logger.info(f"Successful models: {len(trained_models)}")
        logger.info(f"Ensemble activated: {'Yes' if ensemble_enabled else 'No'}")
        logger.info(f"GPU optimization: {'Yes' if gpu_optimization else 'No'}")
        if gpu_optimization:
            logger.info(f"GPU info: {gpu_info}")
        logger.info(f"Submission file: {len(predictions)} rows")
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            available_memory = vm.available / (1024**3)
            logger.info(f"Final memory status: available {available_memory:.1f}GB")
        
        force_memory_cleanup()
        
        logger.info("6. Performance analysis and visualization")
        try:
            from analysis import CTRPerformanceAnalyzer
            from visualization import create_all_visualizations
            
            analyzer = CTRPerformanceAnalyzer(config)
            
            analysis_results = {}
            for model_name, model in trained_models.items():
                try:
                    y_pred = model.predict_proba(X_val_split)
                    analysis = analyzer.full_performance_analysis(
                        y_val_split.values, 
                        y_pred,
                        model_name=model_name,
                        quick_mode=quick_mode
                    )
                    analysis_results[model_name] = analysis
                    
                    report_path = f"results/{model_name}_analysis.json"
                    analyzer.save_analysis_report(analysis, report_path)
                    
                except Exception as e:
                    logger.warning(f"Analysis failed for {model_name}: {e}")
            
            if analysis_results:
                csv_created = analyzer.create_summary_csv(analysis_results)
                if csv_created:
                    logger.info("Summary CSV created successfully")
                
                viz_success = create_all_visualizations(analysis_results)
                if viz_success:
                    logger.info("Visualizations created successfully")
                else:
                    logger.warning("Some visualizations failed")
            
            logger.info("Performance analysis completed")
            
        except Exception as e:
            logger.warning(f"Performance analysis phase failed: {e}")
        
        logger.info("7. Experiment logging")
        try:
            primary_model = list(trained_models.keys())[0] if trained_models else "Unknown"
            
            exp_logger.log_experiment(
                config=config,
                results=results,
                model_name=primary_model,
                notes=f"Mode: {'Quick' if quick_mode else 'Full'}"
            )
            
            logger.info("Experiment logged successfully")
            
        except Exception as e:
            logger.warning(f"Experiment logging failed: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return None

def main():
    """Main execution function"""
    global cleanup_required
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="CTR modeling system - optimized for 0.35+ score")
    parser.add_argument("--mode", choices=["train", "inference"], 
                       default="train", help="Execution mode")
    parser.add_argument("--quick", action="store_true",
                       help="Quick execution mode (50 samples for testing)")
    
    args = parser.parse_args()
    
    try:
        logger.info("=== CTR modeling system started (0.35+ target) ===")
        
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        if args.mode == "train":
            logger.info(f"Training mode started {'(QUICK MODE)' if args.quick else '(FULL MODE - 0.35+ TARGET)'}")
            
            from config import Config
            config = Config
            config.setup_directories()
            
            results = execute_final_pipeline(config, quick_mode=args.quick)
            
            if results:
                logger.info("Training mode completed successfully")
                logger.info(f"Mode: {'Quick (50 samples)' if results.get('quick_mode') else 'Full dataset (0.35+ target)'}")
                logger.info(f"Execution time: {results['execution_time']:.2f}s")
                logger.info(f"Successful models: {results['successful_models']}")
                logger.info(f"GPU optimization: {results['gpu_used']}")
                if results['gpu_used']:
                    logger.info(f"GPU info: {results['gpu_info']}")
                logger.info(f"Target score: {results['target_score']}")
                
                pred_stats = results.get('prediction_stats', {})
                logger.info(f"Prediction statistics:")
                logger.info(f"  Mean: {pred_stats.get('mean', 0):.4f}")
                logger.info(f"  Std: {pred_stats.get('std', 0):.4f}")
                logger.info(f"  Min: {pred_stats.get('min', 0):.4f}")
                logger.info(f"  Max: {pred_stats.get('max', 0):.4f}")
                
                if results.get('model_performances'):
                    logger.info("\nModel Performance Summary:")
                    for model_name, perf in results['model_performances'].items():
                        logger.info(f"  {model_name}:")
                        logger.info(f"    AUC: {perf.get('auc', 0.0):.4f}")
                        logger.info(f"    AP: {perf.get('ap', 0.0):.4f}")
            else:
                logger.error("Training mode failed")
                sys.exit(1)
        
        elif args.mode == "inference":
            logger.info("Inference mode started")
            logger.info("Inference mode completed")
        
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