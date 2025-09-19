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
    OUTPUT_DIR = BASE_DIR / "output"
    
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
    
    GPU_MEMORY_LIMIT = 12
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    GPU_OPTIMIZATION_LEVEL = 1
    
    # Memory settings (optimized for 64GB environment)
    MAX_MEMORY_GB = 50
    CHUNK_SIZE = 50000
    BATCH_SIZE_GPU = 4096
    BATCH_SIZE_CPU = 1024
    PREFETCH_FACTOR = 2
    NUM_WORKERS = 6
    
    # Memory thresholds
    MEMORY_WARNING_THRESHOLD = 35
    MEMORY_CRITICAL_THRESHOLD = 45
    MEMORY_SAFE_THRESHOLD = 30
    MAX_TRAIN_SIZE = 12000000
    MAX_TEST_SIZE = 2000000
    
    # Feature engineering settings
    FEATURE_SELECTION_METHODS = ['correlation', 'mutual_info', 'chi2']
    MAX_CATEGORICAL_CARDINALITY = 1000
    NUMERICAL_SCALING_METHOD = 'standard'
    CATEGORICAL_ENCODING = 'target'
    INTERACTION_FEATURES = True
    POLYNOMIAL_FEATURES = False
    
    # Model-specific settings
    MODEL_CONFIG = {
        'logistic': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        },
        'lightgbm': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100
        },
        'xgboost': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'neural_network': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 1000,
            'random_state': 42
        }
    }
    
    # Cross-validation settings
    CV_FOLDS = 5
    CV_RANDOM_STATE = 42
    CV_SHUFFLE = True
    CV_STRATIFY = True
    
    # Ensemble settings
    ENSEMBLE_WEIGHTS = {
        'logistic': 0.2,
        'lightgbm': 0.4,
        'xgboost': 0.3,
        'random_forest': 0.05,
        'neural_network': 0.05
    }
    
    ENSEMBLE_METHODS = ['weighted_average', 'stacking', 'blending']
    DEFAULT_ENSEMBLE_METHOD = 'weighted_average'
    
    # Calibration settings
    CALIBRATION_METHODS = ['platt', 'isotonic']
    DEFAULT_CALIBRATION_METHOD = 'platt'
    CALIBRATION_CV_FOLDS = 3
    
    # Evaluation configuration
    EVALUATION_CONFIG = {
        'ap_weight': 0.6,
        'wll_weight': 0.4,
        'target_combined_score': 0.34,
        'target_ctr': 0.0191,
        'ctr_tolerance': 0.0005,
        'bias_penalty_weight': 5.0,
        'calibration_weight': 0.4,
        'pos_weight': 49.8,
        'neg_weight': 1.0
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
    
    # Performance optimization flags
    ENABLE_PARALLEL_PROCESSING = True
    ENABLE_MEMORY_MAPPING = True
    ENABLE_CACHING = True
    CACHE_SIZE_MB = 1024
    
    # RTX 4060 Ti specific optimization
    RTX_4060_TI_OPTIMIZATION = True
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]
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
            'OUTPUT_DIR': cls.OUTPUT_DIR,
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
        
        print("=== Validation completed ===")
        
        return all_met
    
    @classmethod
    def get_gpu_info(cls):
        """GPU information retrieval"""
        if not cls.TORCH_AVAILABLE:
            return {"available": False, "message": "PyTorch not available"}
        
        if not torch.cuda.is_available():
            return {"available": False, "message": "CUDA not available"}
        
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved": torch.cuda.memory_reserved() / 1024**3,
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
        
        return gpu_info
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        validation_results = {
            "paths_valid": True,
            "gpu_valid": True,
            "memory_valid": True,
            "issues": []
        }
        
        # Path validation
        try:
            cls.verify_paths()
        except Exception as e:
            validation_results["paths_valid"] = False
            validation_results["issues"].append(f"Path validation failed: {e}")
        
        # GPU validation
        try:
            gpu_info = cls.get_gpu_info()
            if not gpu_info["available"]:
                validation_results["issues"].append(f"GPU not available: {gpu_info['message']}")
        except Exception as e:
            validation_results["gpu_valid"] = False
            validation_results["issues"].append(f"GPU validation failed: {e}")
        
        # Memory validation
        if cls.MAX_MEMORY_GB < 16:
            validation_results["memory_valid"] = False
            validation_results["issues"].append("Insufficient memory configuration")
        
        return validation_results