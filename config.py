# config.py

import os
from pathlib import Path
import logging

# PyTorch import safe handling
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
    
    # Data file paths
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_TEMPLATE_PATH = DATA_DIR / "sample_submission.csv"
    
    # Target column settings
    TARGET_COLUMN_CANDIDATES = [
        'clicked',      # Most common CTR target column name
        'click',        # Abbreviated form
        'is_click',     # Boolean form
        'target',       # General target name
        'label',        # Label
        'y',            # Mathematical expression
        'ctr',          # Direct CTR expression
        'response',     # Response
        'conversion',   # Conversion
        'action'        # Action
    ]
    
    # Target column detection settings
    TARGET_DETECTION_CONFIG = {
        'binary_values': {0, 1},           # Binary classification values
        'min_ctr': 0.001,                  # Minimum CTR (0.1%)
        'max_ctr': 0.1,                    # Maximum CTR (10%)
        'prefer_low_ctr': True,            # Prefer low CTR (CTR characteristic)
        'typical_ctr_range': (0.005, 0.05) # Typical CTR range (0.5%-5%)
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
    
    # Memory settings - optimized for performance
    MAX_MEMORY_GB = 58  # Increased from 55 for better performance
    CHUNK_SIZE = 150000  # Increased from 100000 for efficiency
    BATCH_SIZE_GPU = 20480  # Increased for better GPU utilization
    BATCH_SIZE_CPU = 6144  # Increased from 4096
    PREFETCH_FACTOR = 6  # Increased from 4
    NUM_WORKERS = 10  # Increased from 8
    
    # Memory thresholds - adjusted for performance
    MEMORY_WARNING_THRESHOLD = 48  # Increased threshold
    MEMORY_CRITICAL_THRESHOLD = 53  # Increased threshold
    MEMORY_ABORT_THRESHOLD = 60
    
    # Data size limits - optimized for full dataset
    MAX_TRAIN_SIZE = 15000000  # Increased from 12000000
    MAX_TEST_SIZE = 2500000  # Increased from 2000000
    MAX_INTERACTION_FEATURES = 80  # Increased from 50
    
    # Model training settings - optimized for performance
    MODEL_TRAINING_CONFIG = {
        'lightgbm': {
            'max_depth': 6,  # Increased from 4 for better capacity
            'num_leaves': 63,  # Increased from 15 for more complexity
            'min_data_in_leaf': 200,  # Reduced from 300 for better learning
            'feature_fraction': 0.8,  # Increased from 0.6
            'bagging_fraction': 0.8,  # Increased from 0.6
            'bagging_freq': 5,
            'lambda_l1': 0.5,  # Reduced from 1.0 for less regularization
            'lambda_l2': 0.5,  # Reduced from 1.0
            'min_gain_to_split': 0.02,  # Reduced from 0.05
            'max_cat_threshold': 32,  # Increased from 16
            'cat_smooth': 10.0,  # Reduced from 20.0
            'cat_l2': 10.0,  # Reduced from 20.0
            'learning_rate': 0.05,  # Increased from 0.02
            'num_iterations': 800,  # Increased from 500
            'scale_pos_weight': 52.3,
            'is_unbalance': True
        },
        'xgboost': {
            'max_depth': 6,  # Increased from 4
            'learning_rate': 0.05,  # Increased from 0.03
            'n_estimators': 600,  # Increased from 400
            'subsample': 0.8,  # Increased from 0.6
            'colsample_bytree': 0.8,  # Increased from 0.6
            'min_child_weight': 10,  # Reduced from 15
            'gamma': 0.1,  # Reduced from 0.2
            'alpha': 0.5,  # Reduced from 1.0
            'lambda': 0.5,  # Reduced from 1.0
            'scale_pos_weight': 52.3,
            'reg_alpha': 0.5,  # Reduced from 1.0
            'reg_lambda': 0.5  # Reduced from 1.0
        },
        'logistic': {
            'C': 0.1,  # Increased from 0.01 for less regularization
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 5000,  # Increased from 3000
            'class_weight': 'balanced',
            'random_state': 42,
            'tol': 0.00001  # Tighter tolerance
        }
    }
    
    # Feature engineering settings - expanded for better features
    FEATURE_ENGINEERING_CONFIG = {
        'enable_interaction_features': True,
        'enable_polynomial_features': True,  # Re-enabled
        'enable_binning': True,
        'enable_target_encoding': True,
        'enable_frequency_encoding': True,
        'enable_statistical_features': True,
        'max_interaction_degree': 3,  # Increased from 2
        'binning_strategy': 'quantile',  # Changed back to quantile
        'n_bins': 8,  # Increased from 5
        'min_frequency': 10,  # Reduced from 20
        'target_encoding_smoothing': 20.0,  # Reduced from 50.0
        'enable_cross_validation_encoding': True  # Re-enabled
    }
    
    # Cross-validation settings
    CV_FOLDS = 5  # Increased from 3
    CV_SHUFFLE = True
    RANDOM_STATE = 42
    
    # Early stopping settings
    EARLY_STOPPING_ROUNDS = 150  # Increased from 100
    EARLY_STOPPING_TOLERANCE = 1e-5  # Tighter tolerance
    
    # Hyperparameter tuning settings - significantly increased
    OPTUNA_N_TRIALS = 150  # Increased from 30
    OPTUNA_TIMEOUT = 3600  # Increased from 1200
    OPTUNA_N_JOBS = 2  # Increased from 1
    OPTUNA_VERBOSITY = 1
    
    # Ensemble settings - enhanced strategy
    ENSEMBLE_CONFIG = {
        'voting_weights': {'lightgbm': 0.45, 'xgboost': 0.35, 'logistic': 0.2},  # Adjusted weights
        'stacking_cv_folds': 5,  # Increased from 3
        'blending_ratio': 0.8,  # Increased from 0.7
        'diversity_threshold': 0.03,  # Reduced from 0.05
        'performance_threshold': 0.20,  # Reduced threshold
        'enable_meta_features': True,  # Re-enabled
        'use_simple_average': False  # Disabled for better ensembling
    }
    
    # Calibration settings - optimized
    CALIBRATION_METHOD = 'isotonic'
    CALIBRATION_CV_FOLDS = 5  # Increased from 3
    CALIBRATION_MANDATORY = True
    
    # Evaluation configuration - balanced for performance
    EVALUATION_CONFIG = {
        'ap_weight': 0.6,  # Increased AP weight for better CTR prediction
        'wll_weight': 0.4,  # Reduced WLL weight
        'target_combined_score': 0.34,
        'target_ctr': 0.0191,
        'ctr_tolerance': 0.0003,  # Tighter tolerance
        'bias_penalty_weight': 8.0,  # Reduced from 15.0 for less aggressive penalty
        'calibration_weight': 0.6,  # Reduced from 0.7
        'pos_weight': 52.3,
        'neg_weight': 1.0,
        'wll_normalization_factor': 2.2,  # Increased from 1.8
        'ctr_bias_multiplier': 12.0  # Reduced from 20.0
    }
    
    # CTR bias correction settings - less aggressive
    CTR_BIAS_CORRECTION = {
        'enable': True,
        'target_ctr': 0.0191,
        'correction_factor': 0.25,  # Less aggressive (was 0.15)
        'post_processing': True,
        'clip_range': (0.0005, 0.1),  # Wider range
        'bias_threshold': 0.0003,  # Tighter threshold
        'calibration_strength': 1.5,  # Reduced from 2.0
        'prediction_scaling': 0.5  # Less aggressive scaling
    }
    
    # Evaluation metrics
    PRIMARY_METRIC = 'combined_score'
    SECONDARY_METRICS = ['ap', 'auc', 'log_loss', 'ctr_bias', 'ctr_quality']
    TARGET_COMBINED_SCORE = 0.34
    TARGET_CTR = 0.0191
    
    # Logging settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5
    
    # Performance settings
    ENABLE_PARALLEL_PROCESSING = True
    ENABLE_MEMORY_MAPPING = False
    ENABLE_CACHING = True  # Re-enabled for performance
    CACHE_SIZE_MB = 2048  # Increased from 1024
    
    # Large dataset specific settings
    LARGE_DATASET_MODE = True
    MEMORY_EFFICIENT_SAMPLING = True
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.6  # Increased from 0.5
    MIN_SAMPLE_SIZE = 2000000  # Increased from 1000000
    MAX_SAMPLE_SIZE = 8000000  # Increased from 5000000
    
    # RTX 4060 Ti specific settings
    RTX_4060_TI_OPTIMIZATION = True
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR]
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
            'SUBMISSION_PATH': cls.SUBMISSION_PATH,
            'SUBMISSION_TEMPLATE_PATH': cls.SUBMISSION_TEMPLATE_PATH
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
        """Large data requirements validation"""
        print("=== Large data requirements validation ===")
        
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
        
        # Additional validations
        requirements['train_size_adequate'] = requirements['train_file_size_mb'] > 1000  # > 1GB
        requirements['test_size_adequate'] = requirements['test_file_size_mb'] > 100    # > 100MB
        requirements['memory_adequate'] = requirements['memory_available'] > 40         # > 40GB
        
        # Display results
        for key, value in requirements.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        
        all_met = all([
            requirements['train_file_exists'],
            requirements['test_file_exists'], 
            requirements['train_size_adequate'],
            requirements['test_size_adequate'],
            requirements['memory_adequate']
        ])
        
        print(f"\nAll requirements met: {'✓' if all_met else '✗'}")
        if all_met:
            print("Large data processing ready!")
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
            }
        }