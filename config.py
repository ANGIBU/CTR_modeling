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
        'target',       # General target name
        'label',        # Label
        'y',            # Mathematical expression
        'is_click',     # Boolean form
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
    
    # Memory settings (optimized for 64GB system)
    MAX_MEMORY_GB = 60
    CHUNK_SIZE = 100000
    BATCH_SIZE_GPU = 12288
    BATCH_SIZE_CPU = 4096
    PREFETCH_FACTOR = 6
    NUM_WORKERS = 12
    
    # Memory thresholds (adjusted for performance)
    MEMORY_WARNING_THRESHOLD = 50
    MEMORY_CRITICAL_THRESHOLD = 55
    MEMORY_ABORT_THRESHOLD = 60
    
    # Data size limits
    MAX_TRAIN_SIZE = 20000000
    MAX_TEST_SIZE = 3000000
    MAX_INTERACTION_FEATURES = 250
    
    # Model training settings (tuned parameters)
    MODEL_TRAINING_CONFIG = {
        'lightgbm': {
            'max_depth': 8,
            'num_leaves': 63,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.88,
            'bagging_fraction': 0.88,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.2,
            'min_gain_to_split': 0.01,
            'max_cat_threshold': 32,
            'cat_smooth': 8.0,
            'cat_l2': 8.0
        },
        'xgboost': {
            'max_depth': 7,
            'learning_rate': 0.06,
            'n_estimators': 800,
            'subsample': 0.88,
            'colsample_bytree': 0.88,
            'min_child_weight': 5,
            'gamma': 0.05,
            'alpha': 0.08,
            'lambda': 0.2,
            'scale_pos_weight': 52.3
        },
        'logistic': {
            'C': 1.2,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 3000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    }
    
    # Feature engineering settings (expanded)
    FEATURE_ENGINEERING_CONFIG = {
        'enable_interaction_features': True,
        'enable_polynomial_features': True,
        'enable_binning': True,
        'enable_target_encoding': True,
        'enable_frequency_encoding': True,
        'enable_statistical_features': True,
        'max_interaction_degree': 3,
        'binning_strategy': 'quantile',
        'n_bins': 20,
        'min_frequency': 2,
        'target_encoding_smoothing': 4.0,
        'enable_cross_validation_encoding': True
    }
    
    # Cross-validation settings
    CV_FOLDS = 5
    CV_SHUFFLE = True
    RANDOM_STATE = 42
    
    # Early stopping settings (optimized)
    EARLY_STOPPING_ROUNDS = 250
    EARLY_STOPPING_TOLERANCE = 1e-6
    
    # Hyperparameter tuning settings  
    OPTUNA_N_TRIALS = 200
    OPTUNA_TIMEOUT = 5400
    OPTUNA_N_JOBS = 1
    OPTUNA_VERBOSITY = 1
    
    # Ensemble settings (adjusted weights)
    ENSEMBLE_CONFIG = {
        'voting_weights': {'lightgbm': 0.42, 'xgboost': 0.38, 'logistic': 0.20},
        'stacking_cv_folds': 5,
        'blending_ratio': 0.8,
        'diversity_threshold': 0.06,
        'performance_threshold': 0.28,
        'enable_meta_features': True
    }
    
    # Calibration settings
    CALIBRATION_METHOD = 'isotonic'
    CALIBRATION_CV_FOLDS = 5
    
    # Evaluation configuration (corrected parameters)
    EVALUATION_CONFIG = {
        'ap_weight': 0.6,
        'wll_weight': 0.4,
        'target_combined_score': 0.34,
        'target_ctr': 0.0191,
        'ctr_tolerance': 0.0002,
        'bias_penalty_weight': 8.0,
        'calibration_weight': 0.5,
        'pos_weight': 52.3,
        'neg_weight': 1.0,
        'wll_normalization_factor': 2.6
    }
    
    # Evaluation metrics
    PRIMARY_METRIC = 'combined_score'
    SECONDARY_METRICS = ['ap', 'auc', 'log_loss', 'ctr_bias']
    TARGET_COMBINED_SCORE = 0.34
    TARGET_CTR = 0.0191
    
    # Logging settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5
    
    # Performance settings
    ENABLE_PARALLEL_PROCESSING = True
    ENABLE_MEMORY_MAPPING = True
    ENABLE_CACHING = True
    CACHE_SIZE_MB = 3072
    
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