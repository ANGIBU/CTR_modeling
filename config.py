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
    """Project-wide configuration management"""
    
    # Basic path settings
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Data file paths
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_TEMPLATE_PATH = DATA_DIR / "sample_submission.csv"
    
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
    
    # Memory settings - optimized for 64GB RAM
    MAX_MEMORY_GB = 60
    CHUNK_SIZE = 200000
    BATCH_SIZE_GPU = 25600
    BATCH_SIZE_CPU = 8192
    PREFETCH_FACTOR = 8
    NUM_WORKERS = 12
    
    # Memory thresholds - relaxed for better performance
    MEMORY_WARNING_THRESHOLD = 52
    MEMORY_CRITICAL_THRESHOLD = 56
    MEMORY_ABORT_THRESHOLD = 60
    
    # Data size limits
    MAX_TRAIN_SIZE = 15000000
    MAX_TEST_SIZE = 2500000
    MAX_INTERACTION_FEATURES = 80
    
    # Model training settings
    MODEL_TRAINING_CONFIG = {
        'lightgbm': {
            'max_depth': 7,
            'num_leaves': 95,
            'min_data_in_leaf': 150,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'lambda_l1': 0.3,
            'lambda_l2': 0.3,
            'min_gain_to_split': 0.015,
            'max_cat_threshold': 48,
            'cat_smooth': 15.0,
            'cat_l2': 15.0,
            'learning_rate': 0.04,
            'num_iterations': 1200,
            'scale_pos_weight': 52.3,
            'is_unbalance': True
        },
        'xgboost': {
            'max_depth': 7,
            'learning_rate': 0.04,
            'n_estimators': 800,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 8,
            'gamma': 0.08,
            'alpha': 0.3,
            'lambda': 0.3,
            'scale_pos_weight': 52.3,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3
        },
        'logistic': {
            'C': 0.8,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 5000,
            'class_weight': 'balanced',
            'random_state': 42,
            'tol': 0.00003,
            'n_jobs': 10
        }
    }
    
    # Feature engineering settings - expanded
    FEATURE_ENGINEERING_CONFIG = {
        'enable_interaction_features': True,
        'enable_polynomial_features': False,
        'enable_binning': True,
        'enable_target_encoding': True,
        'enable_frequency_encoding': True,
        'enable_statistical_features': True,
        'enable_cross_validation_encoding': True,
        'max_interaction_degree': 2,
        'binning_strategy': 'quantile',
        'n_bins': 8,
        'min_frequency': 8,
        'target_encoding_smoothing': 25.0,
        
        # Increased feature count
        'target_feature_count': 350,
        'use_feature_selection': True,
        'feature_selection_method': 'f_classif',
        'feature_importance_threshold': 0.008,
        
        # Expanded limits
        'max_categorical_for_encoding': 20,
        'max_numeric_for_interaction': 15,
        'max_cross_combinations': 12,
        'max_statistical_features': 12,
        
        'cleanup_after_each_step': True,
        'intermediate_storage': False
    }
    
    # Cross-validation settings
    CV_FOLDS = 5
    CV_SHUFFLE = True
    RANDOM_STATE = 42
    
    # Early stopping settings
    EARLY_STOPPING_ROUNDS = 200
    EARLY_STOPPING_TOLERANCE = 1e-5
    
    # Hyperparameter tuning settings - activated
    OPTUNA_N_TRIALS = 80
    OPTUNA_TIMEOUT = 7200
    OPTUNA_N_JOBS = 2
    OPTUNA_VERBOSITY = 1
    ENABLE_OPTUNA = True
    
    # Ensemble settings
    ENSEMBLE_CONFIG = {
        'voting_weights': {'lightgbm': 0.42, 'xgboost': 0.38, 'logistic': 0.2},
        'stacking_cv_folds': 5,
        'blending_ratio': 0.8,
        'diversity_threshold': 0.025,
        'performance_threshold': 0.18,
        'enable_meta_features': True,
        'use_simple_average': False
    }
    
    # Calibration settings
    CALIBRATION_METHOD = 'isotonic'
    CALIBRATION_CV_FOLDS = 5
    CALIBRATION_MANDATORY = True
    
    # Evaluation configuration
    EVALUATION_CONFIG = {
        'ap_weight': 0.6,
        'wll_weight': 0.4,
        'target_combined_score': 0.34,
        'target_ctr': 0.0191,
        'ctr_tolerance': 0.0008,
        'bias_penalty_weight': 4.5,
        'calibration_weight': 0.4,
        'pos_weight': 52.3,
        'neg_weight': 1.0,
        'wll_normalization_factor': 1.8,
        'ctr_bias_multiplier': 6.0
    }
    
    # CTR bias correction settings
    CTR_BIAS_CORRECTION = {
        'enable': True,
        'target_ctr': 0.0191,
        'correction_factor': 0.22,
        'post_processing': True,
        'clip_range': (0.0004, 0.12),
        'bias_threshold': 0.0002,
        'calibration_strength': 1.4,
        'prediction_scaling': 0.48
    }
    
    # Evaluation metrics
    PRIMARY_METRIC = 'combined_score'
    SECONDARY_METRICS = ['ap', 'auc', 'log_loss', 'ctr_bias', 'ctr_score']
    TARGET_COMBINED_SCORE = 0.34
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
    CACHE_SIZE_MB = 3072
    
    # Large dataset specific settings
    LARGE_DATASET_MODE = True
    MEMORY_EFFICIENT_SAMPLING = True
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.72
    MIN_SAMPLE_SIZE = 300000
    MAX_SAMPLE_SIZE = 3000000
    
    # Logistic regression sampling configuration - increased
    LOGISTIC_SAMPLING_CONFIG = {
        'normal_size': 3000000,
        'warning_size': 2000000,
        'critical_size': 1200000,
        'abort_size': 300000,
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
            cls.OUTPUT_DIR,
            cls.RESULTS_DIR
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