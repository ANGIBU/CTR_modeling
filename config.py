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
    """Project-wide configuration management - optimized for performance"""
    
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
    
    GPU_MEMORY_LIMIT = 14
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    GPU_OPTIMIZATION_LEVEL = 3
    
    # Memory settings - OPTIMIZED for 64GB RAM system
    MAX_MEMORY_GB = 58
    CHUNK_SIZE = 150000
    BATCH_SIZE_GPU = 20480
    BATCH_SIZE_CPU = 6144
    PREFETCH_FACTOR = 6
    NUM_WORKERS = 10
    
    # Memory thresholds - adjusted for better performance
    MEMORY_WARNING_THRESHOLD = 48
    MEMORY_CRITICAL_THRESHOLD = 53
    MEMORY_ABORT_THRESHOLD = 58
    
    # Data size limits
    MAX_TRAIN_SIZE = 15000000
    MAX_TEST_SIZE = 2500000
    MAX_INTERACTION_FEATURES = 60
    
    # Model training settings
    MODEL_TRAINING_CONFIG = {
        'lightgbm': {
            'max_depth': 6,
            'num_leaves': 63,
            'min_data_in_leaf': 200,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'early_stopping_rounds': 10,
            'verbosity': -1
        },
        'xgboost': {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 5,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'early_stopping_rounds': 10,
            'verbosity': 0
        },
        'catboost': {
            'depth': 6,
            'learning_rate': 0.05,
            'iterations': 200,
            'l2_leaf_reg': 3,
            'border_count': 128,
            'verbose': 0,
            'early_stopping_rounds': 10
        },
        'logistic': {
            'C': 0.5,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 100,
            'n_jobs': -1,
            'verbose': 0
        }
    }
    
    # Feature engineering settings
    FEATURE_ENGINEERING_CONFIG = {
        'target_feature_count': 100,
        'use_feature_selection': True,
        'feature_selection_method': 'mutual_info',
        'cleanup_after_each_step': True,
        'max_categorical_for_encoding': 20,
        'max_numeric_for_interaction': 15,
        'interaction_max_features': 30
    }
    
    # Training and evaluation settings
    CV_FOLDS = 5
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Calibration settings
    CALIBRATION_CONFIG = {
        'enabled': False,
        'methods': ['isotonic', 'sigmoid'],
        'cv_folds': 3
    }
    
    # Ensemble settings
    ENSEMBLE_CONFIG = {
        'enabled': True,
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
        'ctr_tolerance': 0.001
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
    CACHE_SIZE_MB = 2048
    
    # Large dataset specific settings - OPTIMIZED
    LARGE_DATASET_MODE = True
    MEMORY_EFFICIENT_SAMPLING = True
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.65
    MIN_SAMPLE_SIZE = 100000
    MAX_SAMPLE_SIZE = 1000000
    
    # NEW: Logistic regression sampling configuration
    LOGISTIC_SAMPLING_CONFIG = {
        'normal_size': 1000000,
        'warning_size': 750000,
        'critical_size': 500000,
        'abort_size': 100000,
        'enable_dynamic_sizing': True
    }
    
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
        }
        
        requirements['train_size_adequate'] = requirements['train_file_size_mb'] > 1000
        requirements['test_size_adequate'] = requirements['test_file_size_mb'] > 100
        requirements['memory_adequate'] = requirements['memory_available'] > 40
        
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
        else:
            print("Requirements not met. Check data files and system resources.")
        
        return requirements
    
    @classmethod
    def get_memory_efficient_config(cls):
        """Get memory efficient configuration"""
        return {
            'chunk_size': cls.CHUNK_SIZE,
            'batch_size': cls.BATCH_SIZE_CPU,
            'max_train_size': cls.MAX_TRAIN_SIZE,
            'max_test_size': cls.MAX_TEST_SIZE,
            'memory_thresholds': {
                'warning': cls.MEMORY_WARNING_THRESHOLD,
                'critical': cls.MEMORY_CRITICAL_THRESHOLD,
                'abort': cls.MEMORY_ABORT_THRESHOLD
            },
            'sampling_config': {
                'aggressive_threshold': cls.AGGRESSIVE_SAMPLING_THRESHOLD,
                'min_sample_size': cls.MIN_SAMPLE_SIZE,
                'max_sample_size': cls.MAX_SAMPLE_SIZE
            },
            'logistic_sampling': cls.LOGISTIC_SAMPLING_CONFIG,
            'feature_engineering': {
                'target_feature_count': cls.FEATURE_ENGINEERING_CONFIG['target_feature_count'],
                'use_feature_selection': cls.FEATURE_ENGINEERING_CONFIG['use_feature_selection'],
                'cleanup_after_each_step': cls.FEATURE_ENGINEERING_CONFIG['cleanup_after_each_step']
            }
        }
    
    @classmethod
    def get_optimization_summary(cls):
        """Get optimization settings summary"""
        return {
            'memory_optimizations': {
                'target_features': cls.FEATURE_ENGINEERING_CONFIG['target_feature_count'],
                'feature_selection_enabled': cls.FEATURE_ENGINEERING_CONFIG['use_feature_selection'],
                'step_cleanup_enabled': cls.FEATURE_ENGINEERING_CONFIG['cleanup_after_each_step'],
                'max_categorical_encoding': cls.FEATURE_ENGINEERING_CONFIG['max_categorical_for_encoding'],
                'max_numeric_interaction': cls.FEATURE_ENGINEERING_CONFIG['max_numeric_for_interaction']
            },
            'sampling_optimizations': {
                'logistic_normal': cls.LOGISTIC_SAMPLING_CONFIG['normal_size'],
                'logistic_warning': cls.LOGISTIC_SAMPLING_CONFIG['warning_size'],
                'logistic_critical': cls.LOGISTIC_SAMPLING_CONFIG['critical_size'],
                'dynamic_sizing': cls.LOGISTIC_SAMPLING_CONFIG['enable_dynamic_sizing']
            },
            'performance_targets': {
                'target_combined_score': cls.TARGET_COMBINED_SCORE,
                'target_ctr': cls.TARGET_CTR,
                'ctr_tolerance': cls.EVALUATION_CONFIG['ctr_tolerance']
            }
        }

# Backward compatibility
def get_config():
    """Get configuration instance"""
    return Config()

if __name__ == "__main__":
    # Test configuration
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