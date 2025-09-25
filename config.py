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
    
    # Memory configuration (40GB limit - 60% of 64GB)
    MAX_MEMORY_GB = 40.0  # Reduced from higher values
    MEMORY_WARNING_THRESHOLD = 0.75  # 30GB
    MEMORY_CRITICAL_THRESHOLD = 0.85  # 34GB
    MEMORY_ABORT_THRESHOLD = 0.95  # 38GB
    
    # Processing configuration optimized for 40GB limit
    CHUNK_SIZE = 50000  # Reduced from 100000 for memory efficiency
    BATCH_SIZE_CPU = 8000  # Reduced batch size for CPU processing
    BATCH_SIZE_GPU = 16000
    
    # Data size limits (adjusted for memory efficiency)
    MAX_TRAIN_SIZE = 8000000  # Reduced from 10M to manage memory better
    MAX_TEST_SIZE = 1200000   # Reduced accordingly
    
    # Sampling configuration for memory management
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.8  # Start sampling at 80% memory
    MIN_SAMPLE_SIZE = 100000
    MAX_SAMPLE_SIZE = 5000000  # Reduced max sample size
    
    # Feature engineering limits
    MAX_FEATURES = 500  # Reduced from higher values
    MAX_CATEGORICAL_UNIQUE = 1000  # Limit unique categories
    
    # Model training configuration
    CV_FOLDS = 3  # Reduced from 5 for memory efficiency
    EARLY_STOPPING_ROUNDS = 50
    RANDOM_STATE = 42
    
    # Quick mode configuration
    QUICK_SAMPLE_SIZE = 50
    QUICK_TEST_SIZE = 25
    
    # Thread configuration
    N_JOBS = min(6, os.cpu_count())  # Limit threads to reduce memory overhead
    
    # Model parameters optimized for memory efficiency
    MODEL_PARAMS = {
        'lightgbm': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # Reduced for memory efficiency
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS,
            'max_depth': 6,  # Reduced depth
            'min_data_in_leaf': 100,  # Increased for stability
            'force_col_wise': True,  # Better memory usage
        },
        'xgboost': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,  # Reduced depth
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS,
            'tree_method': 'hist',  # Memory efficient method
            'max_leaves': 31,  # Limit leaves
        },
        'logistic': {
            'max_iter': 1000,
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS,
            'solver': 'lbfgs',  # Memory efficient solver
        }
    }
    
    # Ensemble configuration
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.4,
        'xgboost': 0.35,
        'logistic': 0.25
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
            return min(available_gb, cls.MAX_MEMORY_GB)  # Never exceed our limit
        except:
            return cls.MAX_MEMORY_GB * 0.8  # Conservative fallback
    
    @classmethod
    def get_memory_usage_percent(cls) -> float:
        """Get current memory usage percentage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except:
            return 50.0  # Conservative fallback
    
    @classmethod
    def should_use_sampling(cls) -> bool:
        """Check if aggressive sampling should be used"""
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
                return 25000  # Very conservative for low memory
            elif available_memory < 30:
                return cls.CHUNK_SIZE  # Use default
            else:
                return min(cls.CHUNK_SIZE * 2, 75000)  # Can use larger chunks
        except:
            return 25000  # Conservative fallback
    
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
        requirements['memory_adequate'] = requirements['memory_available'] >= 20       # >= 20GB minimum
        
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
            print("Large data processing ready with memory limit!")
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
            }
        }
    
    @classmethod
    def get_processing_config(cls):
        """Get processing configuration optimized for current system"""
        available_memory = cls.get_available_memory()
        memory_usage = cls.get_memory_usage_percent()
        
        # Adjust configuration based on memory status
        config = {
            'chunk_size': cls.get_optimal_chunk_size(),
            'n_jobs': cls.N_JOBS,
            'memory_efficient': True,
            'use_sampling': cls.should_use_sampling(),
            'available_memory': available_memory,
            'memory_usage': memory_usage
        }
        
        # Memory-based adjustments
        if available_memory < 25:
            config.update({
                'chunk_size': 20000,
                'n_jobs': max(1, cls.N_JOBS // 2),
                'aggressive_gc': True
            })
        elif available_memory > 35:
            config.update({
                'chunk_size': min(75000, cls.CHUNK_SIZE * 2),
                'n_jobs': cls.N_JOBS,
                'aggressive_gc': False
            })
        
        return config

# Create default instance
config = Config()

if __name__ == "__main__":
    # Validation script
    print("CTR Modeling System Configuration Validation")
    print("=" * 50)
    
    # Setup directories
    Config.setup_directories()
    
    # Validate system
    Config.validate_system_requirements()
    
    # Validate paths
    Config.validate_paths()
    
    # Verify data requirements
    Config.verify_data_requirements()
    
    # Display memory efficient config
    print("\n=== Memory Efficient Configuration ===")
    mem_config = Config.get_memory_efficient_config()
    for key, value in mem_config.items():
        print(f"{key}: {value}")
    
    print(f"\nMemory limit enforced: {Config.MAX_MEMORY_GB}GB (60% of 64GB system)")
    print(f"Current memory usage: {Config.get_memory_usage_percent():.1f}%")
    print(f"Available for processing: {Config.get_available_memory():.1f}GB")