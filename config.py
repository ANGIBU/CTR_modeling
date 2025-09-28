# config.py

import os
import sys
import gc
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
    """Memory optimized configuration for CTR modeling system"""
    
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
    
    # Conservative memory configuration (45GB available, use max 30GB)
    MAX_MEMORY_GB = 25.0  # Very conservative limit
    MEMORY_WARNING_THRESHOLD = 0.50  # 12.5GB warning
    MEMORY_CRITICAL_THRESHOLD = 0.70  # 17.5GB critical  
    MEMORY_ABORT_THRESHOLD = 0.85  # 21.25GB abort
    
    # Aggressive chunking for memory efficiency
    CHUNK_SIZE = 15000  # Much smaller chunks
    SMALL_CHUNK_SIZE = 5000  # For memory-intensive operations
    TINY_CHUNK_SIZE = 1000  # For critical memory operations
    
    # Batch processing optimization
    BATCH_SIZE_CPU = 2000  # Smaller batches for CPU
    BATCH_SIZE_GPU = 5000  # Smaller batches for GPU
    MAX_WORKERS = 2  # Reduced parallel workers
    
    # Data processing limits
    MAX_TRAIN_SIZE = 10704179
    MAX_TEST_SIZE = 1527298
    
    # Memory-conservative sampling
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.60  # Start sampling earlier
    MIN_SAMPLE_SIZE = 50000  # Smaller minimum sample
    MAX_SAMPLE_SIZE = 2000000  # Much smaller max sample
    EMERGENCY_SAMPLE_SIZE = 100000  # Emergency fallback sample
    
    # Feature engineering constraints
    MAX_FEATURES = 200  # Reduced max features
    MAX_CATEGORICAL_UNIQUE = 1000  # Reduced categorical limit
    SEQ_HASH_SIZE = 10000  # Smaller hash size
    MAX_INTERACTIONS = 20  # Limited interaction features
    
    # Model training configuration
    CV_FOLDS = 3
    EARLY_STOPPING_ROUNDS = 30
    RANDOM_STATE = 42
    
    # Memory optimization settings
    DTYPE_OPTIMIZATION = True
    AGGRESSIVE_GC = True
    GC_FREQUENCY = 5  # Run GC every 5 operations
    
    # Quick mode for testing
    QUICK_SAMPLE_SIZE = 1000
    QUICK_TEST_SIZE = 500
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_memory_usage_gb(cls) -> float:
        """Get current memory usage in GB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except:
            return 2.0
    
    @classmethod
    def get_available_memory_gb(cls) -> float:
        """Get available memory in GB"""
        try:
            return psutil.virtual_memory().available / (1024**3)
        except:
            return 30.0
    
    @classmethod
    def get_memory_usage_percent(cls) -> float:
        """Get memory usage percentage"""
        current_gb = cls.get_memory_usage_gb()
        return (current_gb / cls.MAX_MEMORY_GB) * 100
    
    @classmethod
    def should_use_sampling(cls) -> bool:
        """Check if sampling should be used"""
        usage_percent = cls.get_memory_usage_percent()
        return usage_percent > (cls.AGGRESSIVE_SAMPLING_THRESHOLD * 100)
    
    @classmethod
    def get_emergency_config(cls) -> dict:
        """Get emergency memory configuration"""
        return {
            'chunk_size': cls.TINY_CHUNK_SIZE,
            'batch_size': 500,
            'max_features': 50,
            'sample_size': cls.EMERGENCY_SAMPLE_SIZE,
            'disable_interactions': True,
            'aggressive_sampling': True
        }
    
    @classmethod
    def get_safe_chunk_size(cls) -> int:
        """Get safe chunk size based on current memory"""
        usage_percent = cls.get_memory_usage_percent()
        
        if usage_percent > 80:
            return cls.TINY_CHUNK_SIZE
        elif usage_percent > 60:
            return cls.SMALL_CHUNK_SIZE
        else:
            return cls.CHUNK_SIZE
    
    @classmethod
    def force_memory_cleanup(cls):
        """Force aggressive memory cleanup"""
        if cls.AGGRESSIVE_GC:
            gc.collect()
            gc.collect()  # Second pass for better cleanup
            gc.collect()  # Third pass for maximum cleanup
    
    @classmethod
    def monitor_memory_critical(cls) -> dict:
        """Monitor memory and return critical status"""
        current_gb = cls.get_memory_usage_gb()
        available_gb = cls.get_available_memory_gb()
        usage_percent = cls.get_memory_usage_percent()
        
        status = {
            'current_gb': current_gb,
            'available_gb': available_gb,
            'usage_percent': usage_percent,
            'level': 'normal',
            'action_required': False,
            'emergency_mode': False
        }
        
        if usage_percent > 85 or available_gb < 5:
            status['level'] = 'emergency'
            status['emergency_mode'] = True
            status['action_required'] = True
        elif usage_percent > 70 or available_gb < 10:
            status['level'] = 'critical' 
            status['action_required'] = True
        elif usage_percent > 50:
            status['level'] = 'warning'
        
        return status
    
    @classmethod
    def validate_system_requirements(cls):
        """Validate system requirements"""
        # Check available memory
        available_gb = cls.get_available_memory_gb()
        if available_gb < 15:
            raise RuntimeError(f"Insufficient memory: {available_gb:.1f}GB available, need at least 15GB")
        
        # Create directories
        cls.setup_directories()
        
        logging.info(f"System validation passed - Available memory: {available_gb:.1f}GB")
    
    @classmethod
    def validate_paths(cls):
        """Validate required file paths"""
        required_files = [cls.TRAIN_PATH, cls.TEST_PATH, cls.SUBMISSION_TEMPLATE_PATH]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        logging.info("All required files validated")
    
    @classmethod
    def verify_data_requirements(cls):
        """Verify data file requirements"""
        try:
            import pandas as pd
            
            # Quick validation without loading full data
            train_info = pd.read_parquet(cls.TRAIN_PATH, columns=[]).shape
            test_info = pd.read_parquet(cls.TEST_PATH, columns=[]).shape
            
            logging.info(f"Data validation - Train: {train_info}, Test: {test_info}")
            
        except Exception as e:
            logging.warning(f"Data validation failed: {e}")

# Memory optimization utilities
def optimize_dataframe_memory(df):
    """Optimize dataframe memory usage"""
    import pandas as pd
    import numpy as np
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df

def emergency_memory_cleanup():
    """Emergency memory cleanup procedure"""
    import gc
    
    # Multiple garbage collection passes
    for _ in range(5):
        gc.collect()
    
    # Force cleanup of unreferenced objects
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(700, 10, 10)

# Create default instance
config = Config()

if __name__ == "__main__":
    print("CTR Modeling System - Memory Optimized Configuration")
    print("=" * 60)
    
    Config.setup_directories()
    Config.validate_system_requirements()
    Config.validate_paths()
    Config.verify_data_requirements()
    
    print("\n=== Memory Status ===")
    memory_status = Config.monitor_memory_critical()
    for key, value in memory_status.items():
        print(f"{key}: {value}")
    
    print(f"\nMemory limit: {Config.MAX_MEMORY_GB}GB")
    print(f"Safe chunk size: {Config.get_safe_chunk_size()}")
    print(f"Should use sampling: {Config.should_use_sampling()}")
    
    if memory_status['emergency_mode']:
        print("\n*** EMERGENCY MODE CONFIGURATION ***")
        emergency_config = Config.get_emergency_config()
        for key, value in emergency_config.items():
            print(f"{key}: {value}")