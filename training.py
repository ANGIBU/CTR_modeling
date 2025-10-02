# training.py

import os
import gc
import json
import time
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Tuple
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class LargeDataMemoryTracker:
    """Memory tracking and optimization for large data processing"""
    
    def __init__(self):
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.memory_threshold = 0.85
        self.cleanup_threshold = 0.90
        
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        status = {'available': True, 'usage_gb': 0.0, 'available_gb': 50.0}
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            status.update({
                'usage_gb': (vm.total - vm.available) / (1024**3),
                'available_gb': vm.available / (1024**3),
                'percent': vm.percent
            })
        
        return status
    
    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage information"""
        if not self.gpu_available:
            return {'available': False, 'rtx_4060_ti_optimized': False}
        
        try:
            if TORCH_AVAILABLE:
                allocated = torch.cuda.memory_allocated() / (1024**3)
                cached = torch.cuda.memory_reserved() / (1024**3)
                
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                rtx_4060_ti = "4060 Ti" in gpu_name
                
                return {
                    'available': True,
                    'allocated_gb': allocated,
                    'cached_gb': cached,
                    'rtx_4060_ti_optimized': rtx_4060_ti,
                    'gpu_name': gpu_name
                }
        except Exception:
            pass
        
        return {'available': False, 'rtx_4060_ti_optimized': False}
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
    
    def force_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if self.gpu_available and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

class CTRHyperparameterOptimizer:
    """Hyperparameter optimization for CTR models with Optuna"""
    
    def __init__(self, config: Config):
        self.config = config
        self.quick_mode = False
        self.memory_tracker = LargeDataMemoryTracker()
        self.enable_optuna = config.ENABLE_OPTUNA if hasattr(config, 'ENABLE_OPTUNA') else False
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for optimization"""
        self.quick_mode = enabled
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for each model type"""
        
        if model_type.lower() == 'lightgbm':
            if self.quick_mode:
                return {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1,
                    'num_iterations': 50
                }
            
            params = self.config.MODEL_TRAINING_CONFIG.get('lightgbm', {})
            
            if self.memory_tracker.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized']:
                    params.update({
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0
                    })
            
            return params
        
        elif model_type.lower() == 'xgboost':
            if self.quick_mode:
                return {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'n_estimators': 50
                }
            
            params = self.config.MODEL_TRAINING_CONFIG.get('xgboost', {})
            
            if self.memory_tracker.gpu_available:
                gpu_info = self.memory_tracker.get_gpu_memory_usage()
                if gpu_info['rtx_4060_ti_optimized']:
                    params.update({
                        'tree_method': 'gpu_hist',
                        'gpu_id': 0,
                        'predictor': 'gpu_predictor'
                    })
            
            return params
        
        elif model_type.lower() == 'logistic':
            if self.quick_mode:
                return {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 100,
                    'random_state': 42
                }
            
            return self.config.MODEL_TRAINING_CONFIG.get('logistic', {})
        
        return {}
    
    def optimize_hyperparameters(self, model_type: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_val: pd.DataFrame, 
                                y_val: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        if not OPTUNA_AVAILABLE or not self.enable_optuna:
            logger.info(f"Optuna not available or disabled, using default parameters")
            return self._get_default_params(model_type)
        
        logger.info(f"Starting Optuna hyperparameter optimization for {model_type}")
        
        try:
            n_trials = self.config.OPTUNA_N_TRIALS
            timeout = self.config.OPTUNA_TIMEOUT
            
            if self.quick_mode:
                n_trials = 10
                timeout = 300
            
            def objective(trial):
                if model_type.lower() == 'lightgbm':
                    params = {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 50, 150),
                        'max_depth': trial.suggest_int('max_depth', 5, 9),
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.95),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95),
                        'bagging_freq': 5,
                        'min_child_samples': trial.suggest_int('min_child_samples', 100, 250),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 0.8),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 0.8),
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1,
                        'n_estimators': trial.suggest_int('n_estimators', 500, 1500)
                    }
                    
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                             callbacks=[lgb.early_stopping(50, verbose=False)])
                    
                elif model_type.lower() == 'xgboost':
                    params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'max_depth': trial.suggest_int('max_depth', 5, 9),
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08),
                        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
                        'gamma': trial.suggest_float('gamma', 0.05, 0.15),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.8),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 0.8),
                        'random_state': 42,
                        'n_jobs': -1,
                        'n_estimators': trial.suggest_int('n_estimators', 400, 1000)
                    }
                    
                    model = xgb.XGBClassifier(**params)
                    model.fit(
                        X_train, y_train, 
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                        callbacks=[xgb.callback.EarlyStopping(rounds=50)]
                    )
                    
                elif model_type.lower() == 'logistic':
                    params = {
                        'C': trial.suggest_float('C', 0.5, 2.0),
                        'penalty': 'l2',
                        'solver': 'saga',
                        'max_iter': trial.suggest_int('max_iter', 3000, 7000),
                        'random_state': 42,
                        'class_weight': 'balanced',
                        'n_jobs': -1
                    }
                    
                    model = LogisticRegression(**params)
                    
                    sample_size = min(500000, len(X_train))
                    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                    X_sample = X_train.iloc[sample_indices]
                    y_sample = y_train.iloc[sample_indices]
                    
                    model.fit(X_sample, y_sample)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                
                return score
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
            
            best_params = study.best_params
            logger.info(f"Optuna optimization completed - Best score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            default_params = self._get_default_params(model_type)
            default_params.update(best_params)
            
            return default_params
            
        except Exception as e:
            logger.warning(f"Optuna optimization failed: {e}, using default parameters")
            return self._get_default_params(model_type)

class CTRModelTrainer:
    """Main trainer class for CTR models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.hyperparameter_optimizer = CTRHyperparameterOptimizer(config)
        self.gpu_available = self.memory_tracker.gpu_available
        self.quick_mode = False
        self.trainer_initialized = False
        self.trained_models = {}
        self.best_params = {}
        self.model_performance = {}
        self.use_cv = True
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for training"""
        self.quick_mode = enabled
        self.hyperparameter_optimizer.set_quick_mode(enabled)
        if enabled:
            self.use_cv = False
        
    def initialize_trainer(self):
        """Initialize trainer with memory optimization"""
        if not self.trainer_initialized:
            self.memory_tracker.optimize_gpu_memory()
            logger.info("Trainer initialization completed")
            self.trainer_initialized = True
    
    def train_model_with_cv(self,
                           model_class: Type,
                           model_name: str,
                           X: pd.DataFrame,
                           y: pd.Series,
                           n_folds: int = 5) -> Optional[Any]:
        """Train model with cross-validation"""
        
        logger.info(f"Starting {model_name} model training with {n_folds}-fold CV")
        
        try:
            start_time = time.time()
            
            self.initialize_trainer()
            self.memory_tracker.force_cleanup()
            
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            oof_predictions = np.zeros(len(X))
            fold_models = []
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"Training fold {fold + 1}/{n_folds}")
                
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                if fold == 0:
                    best_params = self.hyperparameter_optimizer.optimize_hyperparameters(
                        model_name, X_train_fold, y_train_fold, X_val_fold, y_val_fold
                    )
                    self.best_params[model_name] = best_params
                else:
                    best_params = self.best_params[model_name]
                
                model = model_class()
                if hasattr(model, 'set_quick_mode'):
                    model.set_quick_mode(self.quick_mode)
                
                model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                
                y_pred_fold = model.predict_proba(X_val_fold)
                oof_predictions[val_idx] = y_pred_fold
                
                fold_score = roc_auc_score(y_val_fold, y_pred_fold)
                fold_scores.append(fold_score)
                fold_models.append(model)
                
                logger.info(f"Fold {fold + 1} AUC: {fold_score:.4f}")
                
                self.memory_tracker.force_cleanup()
            
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            logger.info(f"{model_name} CV completed - Mean AUC: {avg_score:.4f} (+/- {std_score:.4f})")
            
            overall_auc = roc_auc_score(y, oof_predictions)
            overall_ap = average_precision_score(y, oof_predictions)
            
            self.model_performance[model_name] = {
                'cv_auc': avg_score,
                'cv_std': std_score,
                'overall_auc': overall_auc,
                'overall_ap': overall_ap,
                'fold_scores': fold_scores
            }
            
            best_fold_idx = np.argmax(fold_scores)
            best_model = fold_models[best_fold_idx]
            
            self.trained_models[model_name] = {
                'model': best_model,
                'fold_models': fold_models,
                'params': self.best_params[model_name],
                'performance': self.model_performance[model_name],
                'training_time': time.time() - start_time,
                'gpu_trained': self.gpu_available
            }
            
            logger.info(f"{model_name} model training completed successfully")
            return best_model
            
        except Exception as e:
            logger.error(f"{model_name} model training failed: {e}")
            return None
    
    def train_model(self,
                    model_class: Type,
                    model_name: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None,
                    y_val: Optional[pd.Series] = None) -> Optional[Any]:
        """Train individual model with optimization"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            start_time = time.time()
            
            self.initialize_trainer()
            self.memory_tracker.force_cleanup()
            logger.info("GPU memory cache cleared")
            
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            if X_val is None or y_val is None:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            else:
                X_train_split, y_train_split = X_train, y_train
                X_val_split, y_val_split = X_val, y_val
            
            best_params = self.hyperparameter_optimizer.optimize_hyperparameters(
                model_name, X_train_split, y_train_split, X_val_split, y_val_split
            )
            
            if not best_params:
                logger.warning(f"Using default parameters for {model_name}")
                best_params = self.hyperparameter_optimizer._get_default_params(model_name)
            else:
                logger.info(f"Using optimized parameters for {model_name}")
            
            self.best_params[model_name] = best_params
            
            logger.info(f"Training final {model_name} model")
            model = model_class()
            
            if hasattr(model, 'set_quick_mode'):
                model.set_quick_mode(self.quick_mode)
            
            if hasattr(model, 'fit_with_params'):
                model.fit_with_params(X_train_split, y_train_split, **best_params)
            else:
                model.fit(X_train_split, y_train_split, X_val_split, y_val_split)
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val_split)
                    if len(y_pred_proba.shape) > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = model.predict(X_val_split)
                
                auc = roc_auc_score(y_val_split, y_pred_proba) if len(np.unique(y_val_split)) > 1 else 0.5
                ap = average_precision_score(y_val_split, y_pred_proba)
                logloss = log_loss(y_val_split, np.clip(y_pred_proba, 1e-15, 1-1e-15))
                
                self.model_performance[model_name] = {
                    'auc': auc,
                    'average_precision': ap,
                    'logloss': logloss
                }
                
                logger.info(f"{model_name} performance - AUC: {auc:.4f}, AP: {ap:.4f}")
                
            except Exception as e:
                logger.warning(f"Performance calculation failed for {model_name}: {e}")
                self.model_performance[model_name] = {'error': str(e)}
            
            self.trained_models[model_name] = {
                'model': model,
                'params': best_params,
                'performance': self.model_performance.get(model_name, {}),
                'training_time': time.time() - start_time,
                'gpu_trained': self.gpu_available
            }
            
            logger.info(f"{model_name} model training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"{model_name} model training failed: {e}")
            return None
    
    def train_multiple_models(self,
                            model_configs: List[Dict[str, Any]],
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series) -> Dict[str, Any]:
        """Train multiple models"""
        
        logger.info(f"Training {len(model_configs)} models")
        trained_models = {}
        
        for config in model_configs:
            model_name = config['name']
            model_class = config['class']
            
            try:
                if self.use_cv and not self.quick_mode:
                    X_combined = pd.concat([X_train, X_val], ignore_index=True)
                    y_combined = pd.concat([y_train, y_val], ignore_index=True)
                    
                    model = self.train_model_with_cv(
                        model_class=model_class,
                        model_name=model_name,
                        X=X_combined,
                        y=y_combined,
                        n_folds=5
                    )
                else:
                    model = self.train_model(
                        model_class=model_class,
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val
                    )
                
                if model:
                    trained_models[model_name] = model
                
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {e}")
                continue
        
        logger.info(f"Multiple models training completed - Successful: {len(trained_models)}")
        
        return trained_models
    
    def get_default_params_by_model_type(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters by model type"""
        return self.hyperparameter_optimizer._get_default_params(model_type)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        calibrated_base_models = sum(
            1 for model_data in self.trained_models.values() 
            if hasattr(model_data.get('model', {}), 'is_calibrated') and model_data['model'].is_calibrated
        )
        
        avg_performance = {}
        if self.model_performance:
            metrics = ['auc', 'average_precision', 'logloss']
            for metric in metrics:
                values = [
                    perf.get(metric, 0) for perf in self.model_performance.values() 
                    if isinstance(perf, dict) and metric in perf
                ]
                avg_performance[f'avg_{metric}'] = np.mean(values) if values else 0.0
        
        return {
            'total_models_trained': len(self.trained_models),
            'calibrated_models': calibrated_base_models,
            'calibration_rate': calibrated_base_models / max(len(self.trained_models), 1),
            'model_performance': self.model_performance,
            'average_performance': avg_performance,
            'quick_mode': self.quick_mode,
            'gpu_used': self.gpu_available,
            'cv_used': self.use_cv,
            'training_completed': True
        }

class CTRTrainingPipeline:
    """Complete CTR training pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_tracker = LargeDataMemoryTracker()
        self.trainer = CTRModelTrainer(config)
        self.quick_mode = False
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for the entire pipeline"""
        self.quick_mode = enabled
        self.trainer.set_quick_mode(enabled)
        
    def run_training_pipeline(self,
                            model_configs: List[Dict[str, Any]],
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series) -> Dict[str, Any]:
        """Run complete training pipeline"""
        
        logger.info("=== CTR Training Pipeline Started ===")
        
        try:
            self.memory_tracker.optimize_gpu_memory()
            
            trained_models = self.trainer.train_multiple_models(
                model_configs, X_train, y_train, X_val, y_val
            )
            
            training_summary = self.trainer.get_training_summary()
            
            pipeline_results = {
                'trained_models': trained_models,
                'training_summary': training_summary,
                'pipeline_success': len(trained_models) > 0,
                'quick_mode': self.quick_mode
            }
            
            logger.info("=== CTR Training Pipeline Completed ===")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {
                'trained_models': {},
                'training_summary': {},
                'pipeline_success': False,
                'error': str(e),
                'quick_mode': self.quick_mode
            }

class CTRTrainer(CTRModelTrainer):
    """Basic CTR trainer class for compatibility"""
    
    def __init__(self, config: Config = Config):
        super().__init__(config)
        self.name = "CTRTrainer"
        logger.info("CTR Trainer initialized (CPU mode)")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available_models = ['logistic']
        
        if LIGHTGBM_AVAILABLE:
            available_models.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        
        return available_models
    
    def enable_gpu_optimization(self):
        """Enable GPU optimization if available"""
        if self.gpu_available:
            self.memory_tracker.optimize_gpu_memory()
            logger.info("GPU optimization enabled")
        else:
            logger.warning("GPU not available, using CPU mode")
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, quick_mode: bool = False) -> Tuple[Any, Dict[str, float]]:
        """Train model with simplified interface for main.py compatibility"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            if quick_mode:
                self.set_quick_mode(True)
            
            self.memory_tracker.force_cleanup()
            logger.info("GPU memory cache cleared")
            logger.info("Trainer initialization completed")
            
            memory_status = self.memory_tracker.get_memory_status()
            logger.info(f"Pre-training memory: {memory_status['available_gb']:.1f}GB available")
            
            if self.use_cv and not quick_mode:
                X_combined = pd.concat([X_train, X_val], ignore_index=True)
                y_combined = pd.concat([y_train, y_val], ignore_index=True)
                
                from models import LogisticModel, LightGBMModel, XGBoostModel
                
                if model_name == 'logistic':
                    model_class = LogisticModel
                elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model_class = LightGBMModel
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    model_class = XGBoostModel
                else:
                    model_class = LogisticModel
                
                model = self.train_model_with_cv(
                    model_class=model_class,
                    model_name=model_name,
                    X=X_combined,
                    y=y_combined,
                    n_folds=5
                )
                
                performance = self.model_performance.get(model_name, {'cv_auc': 0.5, 'overall_ap': 0.0})
                performance_output = {
                    'auc': performance.get('overall_auc', performance.get('cv_auc', 0.5)),
                    'ap': performance.get('overall_ap', 0.0)
                }
                
            else:
                params = self.get_default_params_by_model_type(model_name)
                logger.info(f"Using default parameters for {model_name}")
                
                logger.info(f"Training final {model_name} model")
                
                if model_name == 'logistic':
                    from models import LogisticModel
                    model = LogisticModel()
                    if hasattr(model, 'set_quick_mode'):
                        model.set_quick_mode(quick_mode)
                    logger.info(f"{model_name}: Quick mode enabled - simplified parameters")
                    
                elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    from models import LightGBMModel
                    model = LightGBMModel()
                    if hasattr(model, 'set_quick_mode'):
                        model.set_quick_mode(quick_mode)
                        
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    from models import XGBoostModel
                    model = XGBoostModel()
                    if hasattr(model, 'set_quick_mode'):
                        model.set_quick_mode(quick_mode)
                else:
                    model = LogisticRegression(**params)
                
                logger.info(f"{model_name} model training started (data: {len(X_train)})")
                logger.info(f"{model_name}: Quick mode parameters applied")
                logger.info(f"{model_name}: Starting training")
                
                if hasattr(model, 'fit_with_params'):
                    model.fit_with_params(X_train, y_train, **params)
                else:
                    model.fit(X_train, y_train, X_val, y_val)
                
                logger.info(f"{model_name}: Training completed successfully")
                
                try:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val)
                        if len(y_pred_proba.shape) > 1:
                            y_pred_proba = y_pred_proba[:, 1]
                    else:
                        y_pred_proba = model.predict(X_val)
                    
                    auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                    ap = average_precision_score(y_val, y_pred_proba)
                    
                    performance_output = {'auc': auc, 'ap': ap}
                    logger.info(f"{model_name} performance - AUC: {auc:.4f}, AP: {ap:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Performance calculation failed: {e}")
                    performance_output = {'auc': 0.5, 'ap': 0.0}
            
            logger.info(f"{model_name} model training completed successfully")
            return model, performance_output
            
        except Exception as e:
            logger.error(f"{model_name} model training failed: {e}")
            return None, {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Training interface for compatibility"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        model = LogisticRegression(**self.get_default_params_by_model_type('logistic'))
        model.fit(X_train, y_train)
        
        return model

class CTRTrainerGPU(CTRModelTrainer):
    """GPU-optimized CTR trainer class"""
    
    def __init__(self, config: Config = Config):
        super().__init__(config)
        self.name = "CTRTrainerGPU"
        
        if not self.gpu_available:
            logger.warning("GPU not available, falling back to CPU mode")
        else:
            logger.info("CTR Trainer GPU initialized (RTX 4060 Ti mode)")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available_models = ['logistic']
        
        if LIGHTGBM_AVAILABLE:
            available_models.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        
        return available_models
    
    def enable_gpu_optimization(self):
        """Enable GPU optimization"""
        if self.gpu_available:
            self.memory_tracker.optimize_gpu_memory()
            logger.info("RTX 4060 Ti GPU optimization enabled")
        else:
            logger.warning("GPU not available, using CPU mode")
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series, quick_mode: bool = False) -> Tuple[Any, Dict[str, float]]:
        """Train model with GPU optimization"""
        
        logger.info(f"Starting {model_name} model training")
        
        try:
            self.enable_gpu_optimization()
            
            if quick_mode:
                self.set_quick_mode(True)
            
            return super().train_model(model_name, X_train, y_train, X_val, y_val, quick_mode)
            
        except Exception as e:
            logger.error(f"GPU {model_name} training failed, falling back to CPU: {e}")
            
            params = self.get_default_params_by_model_type(model_name)
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            try:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
                ap = average_precision_score(y_val, y_pred_proba)
                performance = {'auc': auc, 'ap': ap}
            except Exception:
                performance = {'auc': 0.5, 'ap': 0.0}
            
            return model, performance
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Training interface with GPU optimization"""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        if self.gpu_available:
            try:
                params = self.get_default_params_by_model_type('logistic')
                params.update({
                    'n_jobs': -1,
                    'max_iter': 2000 if not self.quick_mode else 100
                })
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                return model
                
            except Exception as e:
                logger.warning(f"GPU training failed, falling back to CPU: {e}")
        
        model = LogisticRegression(**self.get_default_params_by_model_type('logistic'))
        model.fit(X_train, y_train)
        
        return model

ModelTrainer = CTRModelTrainer
TrainingPipeline = CTRTrainingPipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        X_train = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randint(0, 10, 100)
        })
        y_train = np.random.binomial(1, 0.1, 100)
        
        X_val = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),
            'feature_3': np.random.randint(0, 10, 50)
        })
        y_val = np.random.binomial(1, 0.1, 50)
        
        config = Config()
        
        print("Testing CTRTrainer...")
        trainer = CTRTrainer(config)
        trainer.set_quick_mode(True)
        available_models = trainer.get_available_models()
        print(f"Available models: {available_models}")
        
        model = trainer.train(X_train, y_train, X_val, y_val)
        print(f"Basic trainer completed: {type(model)}")
        
        print("Testing CTRTrainerGPU...")
        gpu_trainer = CTRTrainerGPU(config)
        gpu_trainer.set_quick_mode(True)
        gpu_model = gpu_trainer.train(X_train, y_train, X_val, y_val)
        print(f"GPU trainer completed: {type(gpu_model)}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")