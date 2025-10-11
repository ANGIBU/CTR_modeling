# config.py

import os
from pathlib import Path
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. GPU functions will be disabled.")

class Config:
    """Project-wide configuration management for Windows environment"""
    
    # Basic path settings
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Data file paths
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_TEMPLATE_PATH = DATA_DIR / "sample_submission.csv"
    
    # Experiment log path
    EXPERIMENTS_LOG_PATH = LOG_DIR / "experiments.log"
    
    # Target column settings
    TARGET_COLUMN_CANDIDATES = [
        'clicked', 'click', 'is_click', 'target', 'label', 'y',
        'ctr', 'response', 'conversion', 'action'
    ]
    
    # Target column detection settings
    TARGET_DETECTION_CONFIG = {
        'binary_values': {0, 1},
        'min_ctr': 0.001,
        'max_ctr': 0.1,
        'prefer_low_ctr': True,
        'typical_ctr_range': (0.005, 0.05)
    }
    
    # GPU and hardware settings
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        GPU_AVAILABLE = torch.cuda.is_available()
    else:
        DEVICE = 'cpu'
        GPU_AVAILABLE = False
    
    GPU_MEMORY_LIMIT = 16
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    GPU_OPTIMIZATION_LEVEL = 3
    FORCE_GPU_XGBOOST = True
    
    # Windows environment - NVTabular not available
    USE_NVTABULAR = False
    NVTABULAR_PARTITION_SIZE = None
    NVTABULAR_OUT_FILES_PER_PROC = None
    NVTABULAR_SHUFFLE = False
    
    # Memory settings for Windows
    MAX_MEMORY_GB = 45
    CHUNK_SIZE = 100000
    BATCH_SIZE_GPU = 32768
    BATCH_SIZE_CPU = 8192
    PREFETCH_FACTOR = 4
    NUM_WORKERS = 6
    
    # Memory thresholds
    MEMORY_WARNING_THRESHOLD = 50
    MEMORY_CRITICAL_THRESHOLD = 55
    MEMORY_ABORT_THRESHOLD = 60
    
    # Data size limits
    MAX_TRAIN_SIZE = 10704179
    MAX_TEST_SIZE = 1527298
    MAX_INTERACTION_FEATURES = 0
    
    # Model training settings
    MODEL_TRAINING_CONFIG = {
        'xgboost': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
            'gpu_id': 0 if GPU_AVAILABLE else None,
            'predictor': 'gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 51.43,
            'early_stopping_rounds': 20,
            'verbosity': 0,
            'seed': 42
        },
        'lightgbm': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'early_stopping_rounds': 50,
            'verbosity': -1,
            'device': 'cpu',
            'n_jobs': 6,
            'is_unbalance': True,
            'scale_pos_weight': 51.43
        },
        'logistic': {
            'C': 0.5,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 1000,
            'n_jobs': -1,
            'verbose': 0,
            'class_weight': 'balanced',
            'tol': 0.0001
        }
    }
    
    # Feature engineering settings
    FEATURE_ENGINEERING_CONFIG = {
        'target_feature_count': 117,
        'use_feature_selection': False,
        'feature_selection_method': 'mutual_info',
        'cleanup_after_each_step': True,
        'max_categorical_for_encoding': 5,
        'max_numeric_for_interaction': 0,
        'interaction_max_features': 0,
        'disable_normalization_for_trees': True,
        'use_original_features_only': True,
        'preserve_inventory_id': True,
        'categorical_columns': ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour'],
        'continuous_columns': (
            [f'feat_a_{i}' for i in range(1, 19)] +
            [f'feat_b_{i}' for i in range(1, 7)] +
            [f'feat_c_{i}' for i in range(1, 9)] +
            [f'feat_d_{i}' for i in range(1, 7)] +
            [f'feat_e_{i}' for i in range(1, 11)] +
            [f'history_a_{i}' for i in range(1, 8)] +
            [f'history_b_{i}' for i in range(1, 31)] +
            [f'l_feat_{i}' for i in range(1, 28)]
        )
    }
    
    # Training and evaluation settings
    CV_FOLDS = 5
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Calibration settings
    CALIBRATION_CONFIG = {
        'enabled': False,
        'methods': ['isotonic'],
        'cv_folds': 3
    }
    
    # Ensemble settings
    ENSEMBLE_CONFIG = {
        'enabled': False,
        'stacking_enabled': False,
        'voting_enabled': False,
        'cv_folds': 3,
        'meta_learners': ['logistic'],
        'voting_weights': None,
        'target_combined_score': 0.35
    }
    
    # Evaluation settings
    EVALUATION_CONFIG = {
        'metrics': ['auc', 'ap', 'log_loss'],
        'ctr_validation_enabled': True,
        'ctr_tolerance': 0.001,
        'ap_weight': 0.5,
        'wll_weight': 0.5
    }
    
    # Target metrics
    TARGET_COMBINED_SCORE = 0.35
    TARGET_CTR = 0.0191
    
    # Logging settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024
    LOG_FILE_BACKUP_COUNT = 5
    
    # Performance settings
    ENABLE_PARALLEL_PROCESSING = True
    ENABLE_MEMORY_MAPPING = False
    ENABLE_CACHING = True
    CACHE_SIZE_MB = 4096
    
    # Large dataset specific settings
    LARGE_DATASET_MODE = True
    MEMORY_EFFICIENT_SAMPLING = False
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.95
    MIN_SAMPLE_SIZE = 10000000
    MAX_SAMPLE_SIZE = 10704179
    
    # RTX 4060 Ti specific settings
    RTX_4060_TI_OPTIMIZATION = True
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR, 
            cls.MODEL_DIR, 
            cls.LOG_DIR, 
            cls.OUTPUT_DIR
        ]
        created_dirs = []
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(directory))
            except Exception as e:
                print(f"Directory creation failed {directory}: {e}")
        
        if created_dirs:
            print(f"Created directories: {created_dirs}")
        
        return created_dirs
    
    @classmethod
    def verify_paths(cls):
        """Path validation"""
        print("=== Path validation ===")
        
        paths_to_check = {
            'BASE_DIR': cls.BASE_DIR,
            'DATA_DIR': cls.DATA_DIR,
            'MODEL_DIR': cls.MODEL_DIR,
            'LOG_DIR': cls.LOG_DIR,
            'TRAIN_PATH': cls.TRAIN_PATH,
            'TEST_PATH': cls.TEST_PATH,
            'SUBMISSION_PATH': cls.SUBMISSION_PATH
        }
        
        for name, path in paths_to_check.items():
            exists = path.exists()
            size_info = ""
            if exists and path.is_file():
                size_mb = path.stat().st_size / (1024**2)
                size_info = f", size: {size_mb:.1f}MB"
            
            print(f"{name}: {path} (exists: {exists}{size_info})")
        
        print("=== Validation completed ===")
    
    @classmethod
    def verify_data_requirements(cls):
        """Data requirements validation"""
        print("=== Data requirements validation ===")
        
        requirements = {
            'train_file_exists': cls.TRAIN_PATH.exists(),
            'test_file_exists': cls.TEST_PATH.exists(),
            'train_file_size_mb': cls.TRAIN_PATH.stat().st_size / (1024**2) if cls.TRAIN_PATH.exists() else 0,
            'test_file_size_mb': cls.TEST_PATH.stat().st_size / (1024**2) if cls.TEST_PATH.exists() else 0,
            'memory_available': cls.MAX_MEMORY_GB,
            'chunk_size': cls.CHUNK_SIZE,
            'expected_train_size': cls.MAX_TRAIN_SIZE,
            'expected_test_size': cls.MAX_TEST_SIZE,
            'nvtabular_available': False,
            'backend': 'pandas',
            'gpu_available': cls.GPU_AVAILABLE
        }
        
        requirements['train_size_adequate'] = requirements['train_file_size_mb'] > 1000
        requirements['test_size_adequate'] = requirements['test_file_size_mb'] > 100
        requirements['memory_adequate'] = requirements['memory_available'] > 30
        
        for key, value in requirements.items():
            print(f"  {key}: {value}")
        
        all_met = all([
            requirements['train_file_exists'],
            requirements['test_file_exists'], 
            requirements['train_size_adequate'],
            requirements['test_size_adequate'],
            requirements['memory_adequate']
        ])
        
        print(f"\nAll requirements met: {all_met}")
        if all_met:
            print("System ready for processing!")
            print(f"Data loading backend: {requirements['backend']}")
            print(f"GPU available: {requirements['gpu_available']}")
        else:
            print("Requirements not met. Check data files and system resources.")
        
        return requirements
    
    @classmethod
    def get_memory_efficient_config(cls):
        """Get memory efficient configuration"""
        return {
            'chunk_size': cls.CHUNK_SIZE,
            'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
            'max_train_size': cls.MAX_TRAIN_SIZE,
            'max_test_size': cls.MAX_TEST_SIZE,
            'memory_thresholds': {
                'warning': cls.MEMORY_WARNING_THRESHOLD,
                'critical': cls.MEMORY_CRITICAL_THRESHOLD,
                'abort': cls.MEMORY_ABORT_THRESHOLD
            },
            'feature_engineering': {
                'target_feature_count': cls.FEATURE_ENGINEERING_CONFIG['target_feature_count'],
                'use_feature_selection': cls.FEATURE_ENGINEERING_CONFIG['use_feature_selection'],
                'cleanup_after_each_step': cls.FEATURE_ENGINEERING_CONFIG['cleanup_after_each_step']
            },
            'use_nvtabular': False,
            'backend': 'pandas'
        }
    
    @classmethod
    def get_optimization_summary(cls):
        """Get optimization settings summary"""
        return {
            'gpu_optimizations': {
                'gpu_available': cls.GPU_AVAILABLE,
                'force_gpu_xgboost': cls.FORCE_GPU_XGBOOST,
                'rtx_4060_ti_optimized': cls.RTX_4060_TI_OPTIMIZATION,
                'gpu_memory_limit': cls.GPU_MEMORY_LIMIT
            },
            'memory_optimizations': {
                'target_features': cls.FEATURE_ENGINEERING_CONFIG['target_feature_count'],
                'feature_selection_enabled': cls.FEATURE_ENGINEERING_CONFIG['use_feature_selection'],
                'step_cleanup_enabled': cls.FEATURE_ENGINEERING_CONFIG['cleanup_after_each_step'],
                'max_categorical_encoding': cls.FEATURE_ENGINEERING_CONFIG['max_categorical_for_encoding'],
                'max_numeric_interaction': cls.FEATURE_ENGINEERING_CONFIG['max_numeric_for_interaction']
            },
            'performance_targets': {
                'target_combined_score': cls.TARGET_COMBINED_SCORE,
                'target_ctr': cls.TARGET_CTR,
                'ctr_tolerance': cls.EVALUATION_CONFIG['ctr_tolerance']
            },
            'data_loading': {
                'use_nvtabular': False,
                'backend': 'pandas',
                'chunk_size': cls.CHUNK_SIZE
            }
        }

def get_config():
    """Get configuration instance"""
    return Config()

if __name__ == "__main__":
    config = Config()
    
    print("=== Configuration Test ===")
    config.setup_directories()
    config.verify_paths()
    config.verify_data_requirements()
    
    print("\n=== Optimization Summary ===")
    opt_summary = config.get_optimization_summary()
    for category, settings in opt_summary.items():
        print(f"\n{category}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("\n=== Memory Efficient Config ===")
    mem_config = config.get_memory_efficient_config()
    for key, value in mem_config.items():
        print(f"{key}: {value}")