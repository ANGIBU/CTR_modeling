# config.py

import os
import logging
from pathlib import Path
from typing import Dict, Any, List

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_COUNT = torch.cuda.device_count()
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        GPU_COUNT = 0
        GPU_MEMORY = 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    GPU_MEMORY = 0

try:
    import psutil
    TOTAL_MEMORY_GB = psutil.virtual_memory().total / (1024**3)
    CPU_COUNT = psutil.cpu_count()
except ImportError:
    TOTAL_MEMORY_GB = 32
    CPU_COUNT = 8

class Config:
    # Basic Configuration
    DEBUG = False
    RANDOM_STATE = 42
    
    # Hardware Configuration  
    NUM_WORKERS = min(CPU_COUNT, 12)
    GPU_AVAILABLE = GPU_AVAILABLE
    GPU_COUNT = GPU_COUNT
    GPU_MEMORY_GB = GPU_MEMORY
    
    # Memory Configuration
    MAX_MEMORY_GB = min(int(TOTAL_MEMORY_GB * 0.7), 45)
    CHUNK_SIZE = 50000
    BATCH_SIZE = 10000
    MEMORY_THRESHOLD = 0.85
    
    # File paths
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    
    # Output directories
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Target column candidates
    TARGET_COLUMN_CANDIDATES = ['clicked', 'target', 'label', 'y']
    
    # Model configuration
    CV_SPLITS = 3
    TEST_SIZE = 0.2
    
    # LightGBM Parameters
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 200,
        'min_child_weight': 20,
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,
        'early_stopping_rounds': 150,
        'scale_pos_weight': 49.0,
        'force_row_wise': True,
        'max_bin': 255,
        'num_threads': NUM_WORKERS,
        'device_type': 'cpu',
        'min_data_in_leaf': 100,
        'max_depth': 12,
        'feature_fraction_bynode': 0.8
    }
    
    # XGBoost Parameters
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 20,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,
        'early_stopping_rounds': 150,
        'max_bin': 255,
        'nthread': NUM_WORKERS,
        'grow_policy': 'depthwise',
        'max_leaves': 127,
        'gamma': 0.1
    }
    
    # CatBoost Parameters
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 10,
        'learning_rate': 0.05,
        'l2_leaf_reg': 15,
        'iterations': 1500,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 3,
        'thread_count': NUM_WORKERS,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'od_type': 'IncToDec',
        'od_wait': 150,
        'leaf_estimation_iterations': 15,
        'grow_policy': 'SymmetricTree',
        'max_leaves': 127,
        'min_data_in_leaf': 100,
        'rsm': 0.8
    }
    
    # Deep Learning Parameters
    NN_PARAMS = {
        'embedding_dim': 8,
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'batch_norm': True,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 1024,
        'epochs': 100,
        'early_stopping': 10,
        'device': 'cuda' if GPU_AVAILABLE else 'cpu'
    }
    
    # Feature Engineering Configuration
    FEATURE_CONFIG = {
        'max_categories': 50,
        'min_frequency': 5,
        'target_encoding': True,
        'interaction_features': True,
        'polynomial_features': False,
        'statistical_features': True,
        'temporal_features': True,
        'frequency_encoding': True,
        'numerical_binning': True,
        'outlier_treatment': True,
        'missing_value_treatment': True
    }
    
    # Ensemble Configuration
    ENSEMBLE_CONFIG = {
        'methods': ['voting', 'stacking', 'blending'],
        'diversity_threshold': 0.1,
        'correlation_threshold': 0.95,
        'max_models': 10,
        'stacking_cv_folds': 3,
        'blending_holdout': 0.2
    }
    
    # Calibration Configuration
    CALIBRATION_CONFIG = {
        'methods': ['platt', 'isotonic', 'beta', 'sigmoid'],
        'cv_folds': 5,
        'test_size': 0.3,
        'enable_auto_calibration': True,
        'target_reliability': 0.95
    }
    
    # Parallel Processing Configuration
    PARALLEL_CONFIG = {
        'enable_multiprocessing': True,
        'parallel_data_loading': True,
        'parallel_feature_engineering': True,
        'parallel_model_training': True,
        'parallel_inference': True,
        'parallel_evaluation': True,
        'thread_local_storage': True,
        'numa_optimization': False,
        'cpu_affinity': False,
        'priority_scheduling': False,
        'load_balancing': True,
        'work_stealing': True,
        'dynamic_scheduling': True,
        'parallel_io': True,
        'async_processing': True
    }
    
    # Data Processing Configuration
    DATA_CONFIG = {
        'use_pyarrow': True,
        'compression': 'snappy',
        'memory_map': False,
        'lazy_loading': True,
        'batch_processing': True,
        'streaming_processing': True,
        'data_validation': True,
        'schema_validation': True,
        'type_optimization': True,
        'categorical_optimization': True,
        'string_optimization': True,
        'datetime_optimization': True,
        'numeric_optimization': True,
        'memory_efficient_dtypes': True,
        'sparse_arrays': False,
        'columnar_storage': True,
        'indexed_access': True,
        'cached_operations': True,
        'vectorized_operations': True,
        'broadcast_operations': True,
        'parallel_reading': True,
        'async_io': True,
        'prefetch_batches': True,
        'chunk_size': CHUNK_SIZE,
        'max_memory_usage': MAX_MEMORY_GB,
        'large_data_optimization': True,
        'target_column_detection': True,
        'validate_target_column': True,
        'target_column_fallback': 'clicked'
    }
    
    # Model I/O Configuration
    MODEL_IO_CONFIG = {
        'compression_level': 6,
        'pickle_protocol': 5,
        'joblib_compression': 'lz4',
        'model_versioning': True,
        'incremental_saving': True,
        'checkpoint_frequency': 200,
        'backup_models': True,
        'model_metadata': True,
        'model_signature': True,
        'model_validation': True,
        'lazy_model_loading': True,
        'model_caching': True,
        'model_pooling': True,
        'distributed_storage': False,
        'cloud_storage': False,
        'local_storage_optimization': True,
        'model_compression': True,
        'large_model_handling': True
    }
    
    @classmethod
    def get_target_column_config(cls):
        """Get target column configuration"""
        return {
            'candidates': cls.TARGET_COLUMN_CANDIDATES,
            'detection_config': {
                'binary_threshold': 0.9,
                'unique_ratio_threshold': 0.1,
                'typical_ctr_range': (0.001, 0.1),
                'min_positive_samples': 1000,
                'check_naming_patterns': True,
                'validate_distribution': True
            }
        }
    
    @classmethod
    def setup_logging(cls, level=logging.INFO):
        """Setup logging configuration"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_DIR / 'ctr_modeling.log'),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_gpu_environment(cls):
        """Setup GPU environment if available"""
        if cls.GPU_AVAILABLE:
            try:
                import torch
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                logging.info(f"GPU environment setup completed. Available GPUs: {cls.GPU_COUNT}")
            except Exception as e:
                logging.warning(f"Failed to setup GPU environment: {e}")