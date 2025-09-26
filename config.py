# config.py

import os
import sys
from pathlib import Path
import psutil
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ctr_modeling.log', encoding='utf-8')
    ]
)

class Config:
    """Configuration class for CTR modeling system"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.absolute()
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Data file paths
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_TEMPLATE_PATH = DATA_DIR / "sample_submission.csv"
    
    # Memory configuration (35GB limit - adjusted for feature engineering)
    MAX_MEMORY_GB = 35.0  # Reduced from 40GB for stability
    MEMORY_WARNING_THRESHOLD = 0.70  # 24.5GB
    MEMORY_CRITICAL_THRESHOLD = 0.80  # 28GB
    MEMORY_ABORT_THRESHOLD = 0.90  # 31.5GB
    
    # Processing configuration for memory efficiency
    CHUNK_SIZE = 40000  # Increased for better processing
    BATCH_SIZE_CPU = 10000  # Increased batch size
    BATCH_SIZE_GPU = 20000
    
    # Data size limits
    MAX_TRAIN_SIZE = 10704179  # Full dataset
    MAX_TEST_SIZE = 1527298   # Full dataset
    
    # Sampling configuration for memory management
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.75  # Start sampling at 75% memory
    MIN_SAMPLE_SIZE = 100000
    MAX_SAMPLE_SIZE = 8000000  # Increased max sample size
    
    # Feature engineering limits
    MAX_FEATURES = 600  # Increased for more features
    MAX_CATEGORICAL_UNIQUE = 5000  # Increased limit for categorical features
    SEQ_HASH_SIZE = 100000  # Hash size for seq column processing
    
    # Model training configuration
    CV_FOLDS = 3
    EARLY_STOPPING_ROUNDS = 50
    RANDOM_STATE = 42
    
    # Quick mode configuration
    QUICK_SAMPLE_SIZE = 50
    QUICK_TEST_SIZE = 25
    
    # Thread configuration
    N_JOBS = min(8, os.cpu_count())  # Increased thread count
    
    # Target encoding configuration
    TARGET_ENCODING_CHUNKS = 20  # Number of chunks for target encoding
    MIN_CATEGORY_SAMPLES = 5  # Minimum samples for target encoding
    SMOOTHING_FACTOR = 100  # Smoothing factor for target encoding
    
    # CTR correction parameters
    TARGET_CTR = 0.0191
    CTR_CORRECTION_FACTOR = 0.5  # Factor to reduce over-prediction
    CTR_ALIGNMENT_WEIGHT = 0.3  # Weight for CTR alignment in scoring
    
    # Model parameters for memory efficiency and performance
    MODEL_PARAMS = {
        'lightgbm': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,  # Increased for better performance
            'learning_rate': 0.01,  # Reduced for stability
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS,
            'max_depth': 8,  # Increased depth
            'min_data_in_leaf': 50,  # Reduced for more splits
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'force_col_wise': True,
            'n_estimators': 1000,  # Increased iterations
        },
        'xgboost': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 10,  # Increased depth
            'learning_rate': 0.01,  # Reduced for stability
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS,
            'tree_method': 'hist',
            'max_leaves': 127,  # Increased leaves
            'n_estimators': 1000,  # Increased iterations
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        },
        'logistic': {
            'max_iter': 2000,  # Increased iterations
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS,
            'solver': 'saga',  # Changed solver for large datasets
            'C': 0.001,  # Increased regularization
            'class_weight': 'balanced',
        }
    }
    
    # Ensemble configuration
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.45,
        'xgboost': 0.35,
        'logistic': 0.20
    }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]
        created = []
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(str(directory))
        
        if created:
            print(f"Created directories: {created}")
        
        return created
    
    @classmethod
    def get_available_memory(cls) -> float:
        """Get available system memory in GB"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            return min(available_gb, cls.MAX_MEMORY_GB)
        except:
            return cls.MAX_MEMORY_GB * 0.8
    
    @classmethod
    def get_memory_usage_percent(cls) -> float:
        """Get current memory usage percentage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except:
            return 50.0
    
    @classmethod
    def should_use_sampling(cls) -> bool:
        """Check if sampling should be used"""
        try:
            memory_usage = cls.get_memory_usage_percent() / 100.0
            return memory_usage > cls.AGGRESSIVE_SAMPLING_THRESHOLD
        except:
            return False
    
    @classmethod
    def get_optimal_chunk_size(cls) -> int:
        """Get optimal chunk size based on available memory"""
        try:
            available_memory = cls.get_available_memory()
            
            if available_memory < 20:
                return 20000
            elif available_memory < 25:
                return cls.CHUNK_SIZE
            else:
                return min(cls.CHUNK_SIZE * 2, 80000)
        except:
            return 20000
    
    @classmethod
    def validate_system_requirements(cls):
        """Validate system has minimum requirements"""
        print("=== System requirements validation ===")
        
        # Check memory
        try:
            total_memory = psutil.virtual_memory().total / (1024**3)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            print(f"Total memory: {total_memory:.1f}GB")
            print(f"Available memory: {available_memory:.1f}GB")
            print(f"Configured limit: {cls.MAX_MEMORY_GB}GB")
            
            if total_memory < 32:
                print("WARNING: Less than 32GB total memory detected")
                return False
            
            if available_memory < 20:
                print("WARNING: Less than 20GB available memory")
                return False
                
        except Exception as e:
            print(f"Memory check failed: {e}")
            return False
        
        # Check CPU cores
        cpu_cores = os.cpu_count()
        print(f"CPU cores: {cpu_cores}")
        
        if cpu_cores < 4:
            print("WARNING: Less than 4 CPU cores detected")
        
        # Check disk space
        try:
            disk_usage = psutil.disk_usage(str(cls.BASE_DIR))
            free_gb = disk_usage.free / (1024**3)
            print(f"Free disk space: {free_gb:.1f}GB")
            
            if free_gb < 50:
                print("WARNING: Less than 50GB free disk space")
                return False
                
        except Exception as e:
            print(f"Disk check failed: {e}")
            return False
        
        print("System requirements check: PASSED")
        return True
    
    @classmethod
    def validate_paths(cls):
        """Validate all required paths"""
        print("=== Path validation ===")
        
        paths_to_check = {
            'BASE_DIR': cls.BASE_DIR,
            'DATA_DIR': cls.DATA_DIR,
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
        requirements['memory_adequate'] = requirements['memory_available'] >= 20
        
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
            print("Data processing ready with memory limit!")
        else:
            print("Requirements not met. Check data files and system resources.")
        
        return requirements
    
    @classmethod
    def get_memory_efficient_config(cls):
        """Get memory efficient configuration"""
        return {
            'chunk_size': cls.get_optimal_chunk_size(),
            'batch_size': cls.BATCH_SIZE_CPU,
            'max_train_size': cls.MAX_TRAIN_SIZE,
            'max_test_size': cls.MAX_TEST_SIZE,
            'memory_limit_gb': cls.MAX_MEMORY_GB,
            'target_encoding_chunks': cls.TARGET_ENCODING_CHUNKS,
            'seq_hash_size': cls.SEQ_HASH_SIZE,
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
            'feature_limits': {
                'max_features': cls.MAX_FEATURES,
                'max_categorical_unique': cls.MAX_CATEGORICAL_UNIQUE
            },
            'ctr_correction': {
                'target_ctr': cls.TARGET_CTR,
                'correction_factor': cls.CTR_CORRECTION_FACTOR,
                'alignment_weight': cls.CTR_ALIGNMENT_WEIGHT
            }
        }
    
    @classmethod
    def get_processing_config(cls):
        """Get processing configuration for current system"""
        available_memory = cls.get_available_memory()
        memory_usage = cls.get_memory_usage_percent()
        
        config = {
            'chunk_size': cls.get_optimal_chunk_size(),
            'n_jobs': cls.N_JOBS,
            'memory_efficient': True,
            'use_sampling': cls.should_use_sampling(),
            'available_memory': available_memory,
            'memory_usage': memory_usage
        }
        
        if available_memory < 25:
            config.update({
                'chunk_size': 15000,
                'n_jobs': max(1, cls.N_JOBS // 2),
                'aggressive_gc': True
            })
        elif available_memory > 30:
            config.update({
                'chunk_size': min(80000, cls.CHUNK_SIZE * 2),
                'n_jobs': cls.N_JOBS,
                'aggressive_gc': False
            })
        
        return config

# Create default instance
config = Config()

if __name__ == "__main__":
    print("CTR Modeling System Configuration Validation")
    print("=" * 50)
    
    Config.setup_directories()
    Config.validate_system_requirements()
    Config.validate_paths()
    Config.verify_data_requirements()
    
    print("\n=== Memory Configuration ===")
    mem_config = Config.get_memory_efficient_config()
    for key, value in mem_config.items():
        print(f"{key}: {value}")
    
    print(f"\nMemory limit: {Config.MAX_MEMORY_GB}GB")
    print(f"Current memory usage: {Config.get_memory_usage_percent():.1f}%")
    print(f"Available for processing: {Config.get_available_memory():.1f}GB")