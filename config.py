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
    MEMORY_CRITICAL_THRESHOLD = 40
    MEMORY_ABORT_THRESHOLD = 45
    
    # Data processing strategy
    TARGET_DATA_USAGE_RATIO = 1.0
    MIN_TRAIN_SIZE = 100000
    MAX_TRAIN_SIZE = 12000000
    MIN_TEST_SIZE = 50000
    MAX_TEST_SIZE = 2000000
    FORCE_FULL_TEST_PROCESSING = True
    
    # Feature engineering settings
    MAX_FEATURES = 500
    MAX_INTERACTION_FEATURES = 100
    MAX_TARGET_ENCODING_FEATURES = 50
    FEATURE_SELECTION_K = 200
    ENABLE_FEATURE_INTERACTION = True
    ENABLE_TARGET_ENCODING = True
    FEATURE_ENGINEERING_THREADS = 4
    
    # Model training settings
    RANDOM_STATE = 42
    CV_FOLDS = 5
    VALIDATION_SIZE = 0.15
    TEST_SIZE = 0.15
    STRATIFY = True
    
    # LightGBM specific settings
    LIGHTGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 2047,
        'learning_rate': 0.01,
        'feature_fraction': 0.95,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'min_child_samples': 200,
        'min_child_weight': 10,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'max_depth': 18,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 5000,
        'early_stopping_rounds': 300,
        'scale_pos_weight': 50,
        'force_row_wise': True,
        'max_bin': 255,
        'num_threads': 12,
        'device_type': 'cpu'
    }
    
    # XGBoost specific settings
    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': 18,
        'learning_rate': 0.01,
        'subsample': 0.85,
        'colsample_bytree': 0.95,
        'min_child_weight': 10,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 5000,
        'early_stopping_rounds': 300,
        'scale_pos_weight': 50,
        'n_jobs': 12
    }
    
    # Ensemble settings
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.4,
        'xgboost': 0.3,
        'logistic': 0.2,
        'neural_network': 0.1
    }
    
    ENSEMBLE_METHODS = ['weighted_average', 'stacking', 'blending']
    DEFAULT_ENSEMBLE_METHOD = 'weighted_average'
    
    # Calibration settings
    CALIBRATION_METHODS = ['platt', 'isotonic']
    DEFAULT_CALIBRATION_METHOD = 'platt'
    CALIBRATION_CV_FOLDS = 3
    
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
        
        return requirements
    
    @classmethod
    def setup_gpu_environment(cls):
        """GPU environment setup"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, using CPU mode")
            return False
        
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU mode")
            return False
        
        try:
            # GPU device information
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            print(f"GPU environment setup completed: {device_name} ({device_memory:.1f}GB)")
            
            # Memory usage limit
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                memory_fraction = 0.8
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                print(f"GPU memory usage limit: {memory_fraction*100:.1f}%")
            
            # Mixed precision
            if cls.USE_MIXED_PRECISION:
                print("Mixed Precision: True")
            
            # RTX 4060 Ti specific optimization
            if "RTX 4060 Ti" in device_name:
                cls.RTX_4060_TI_OPTIMIZATION = True
                print("RTX 4060 Ti optimization: True")
            
            return True
            
        except Exception as e:
            print(f"GPU environment setup failed: {e}")
            return False

# Environment variables setup
try:
    # Memory optimization
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '200000'
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '262144'
    os.environ['MALLOC_MMAP_MAX_'] = '65536'
    
    # Rule compliance environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'
    
    # PyArrow optimization
    os.environ['ARROW_USER_SIMD_LEVEL'] = 'AVX2'
    os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'
    
    # LightGBM/XGBoost optimization
    os.environ['LIGHTGBM_EXEC_PREFER'] = 'disk'
    os.environ['XGBOOST_CACHE_PREFERENCE'] = 'memory'
    
except Exception as e:
    print(f"Environment variable setup failed: {e}")

# Initialization validation
try:
    print("=== CTR modeling system initialization ===")
    Config.setup_directories()
    Config.verify_paths()
    Config.verify_data_requirements()
    
    # GPU environment setup
    if Config.setup_gpu_environment():
        print("GPU environment setup successful")
    else:
        print("Running in CPU mode")
        
    print("=== Initialization completed ===")
except Exception as e:
    print(f"Initialization failed: {e}")