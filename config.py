# config.py

import os
from pathlib import Path
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed. GPU functions will be disabled.")

try:
    import cudf
    import cupy as cp
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    logging.warning("RAPIDS not installed. GPU acceleration disabled.")

try:
    import nvtabular as nvt
    NVTABULAR_AVAILABLE = True
except ImportError:
    NVTABULAR_AVAILABLE = False
    logging.warning("NVTabular not installed. Merlin features disabled.")

class Config:
    """Project-wide configuration management"""
    
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    RESULTS_DIR = BASE_DIR / "results"
    
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_TEMPLATE_PATH = DATA_DIR / "sample_submission.csv"
    
    NVT_PROCESSED_DIR = DATA_DIR / "nvt_processed"
    NVT_WORKFLOW_DIR = NVT_PROCESSED_DIR / "workflow"
    
    TARGET_COLUMN_CANDIDATES = [
        'clicked', 'click', 'is_click', 'target', 'label', 'y',
        'ctr', 'response', 'conversion', 'action'
    ]
    
    TARGET_DETECTION_CONFIG = {
        'binary_values': {0, 1},
        'min_ctr': 0.001,
        'max_ctr': 0.1,
        'prefer_low_ctr': True,
        'typical_ctr_range': (0.005, 0.05)
    }
    
    if TORCH_AVAILABLE:
        try:
            GPU_AVAILABLE = torch.cuda.is_available()
            if GPU_AVAILABLE:
                test_tensor = torch.zeros(1, device='cuda:0')
                DEVICE = torch.device('cuda:0')
                GPU_COUNT = torch.cuda.device_count()
                GPU_NAME = torch.cuda.get_device_name(0)
                GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                del test_tensor
                torch.cuda.empty_cache()
                
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                logging.info(f"GPU detected: {GPU_NAME} ({GPU_MEMORY_GB:.1f}GB)")
            else:
                DEVICE = torch.device('cpu')
                GPU_COUNT = 0
                GPU_NAME = "None"
                GPU_MEMORY_GB = 0
                logging.info("GPU not available, using CPU mode")
        except Exception as e:
            GPU_AVAILABLE = False
            DEVICE = torch.device('cpu')
            GPU_COUNT = 0
            GPU_NAME = "None"
            GPU_MEMORY_GB = 0
            logging.warning(f"GPU initialization failed: {e}")
    else:
        DEVICE = 'cpu'
        GPU_AVAILABLE = False
        GPU_COUNT = 0
        GPU_NAME = "None"
        GPU_MEMORY_GB = 0
    
    RAPIDS_ENABLED = RAPIDS_AVAILABLE and GPU_AVAILABLE
    NVTABULAR_ENABLED = NVTABULAR_AVAILABLE and RAPIDS_ENABLED
    
    GPU_MEMORY_LIMIT = 14
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    GPU_OPTIMIZATION_LEVEL = 3
    
    MAX_MEMORY_GB = 45
    CHUNK_SIZE = 150000
    BATCH_SIZE_GPU = 20480
    BATCH_SIZE_CPU = 6144
    PREFETCH_FACTOR = 6
    NUM_WORKERS = 10
    
    MEMORY_WARNING_THRESHOLD = 50
    MEMORY_CRITICAL_THRESHOLD = 55
    MEMORY_ABORT_THRESHOLD = 58
    
    MAX_TRAIN_SIZE = 15000000
    MAX_TEST_SIZE = 2500000
    MAX_INTERACTION_FEATURES = 60
    
    NVT_CONFIG = {
        'part_size': '32MB',
        'shuffle': 'PER_PARTITION',
        'out_files_per_proc': 8,
        'freq_threshold': 0,
        'max_size': 50000,
        'exclude_columns': ['seq'],
    }
    
    CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    
    CONTINUOUS_FEATURES = (
        [f'feat_a_{i}' for i in range(1, 19)] +
        [f'feat_b_{i}' for i in range(1, 7)] +
        [f'feat_c_{i}' for i in range(1, 9)] +
        [f'feat_d_{i}' for i in range(1, 7)] +
        [f'feat_e_{i}' for i in range(1, 11)] +
        [f'history_a_{i}' for i in range(1, 8)] +
        [f'history_b_{i}' for i in range(1, 31)] +
        [f'l_feat_{i}' for i in range(1, 28)]
    )
    
    MODEL_TRAINING_CONFIG = {
        'lightgbm': {
            'max_depth': 8,
            'num_leaves': 63,
            'min_data_in_leaf': 200,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.02,
            'max_cat_threshold': 32,
            'cat_smooth': 10.0,
            'cat_l2': 10.0,
            'learning_rate': 0.05,
            'num_iterations': 800,
            'scale_pos_weight': 51.43,
            'is_unbalance': True,
            'device': 'gpu' if GPU_AVAILABLE else 'cpu'
        },
        'xgboost': {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'alpha': 0,
            'lambda': 1,
            'scale_pos_weight': 51.43,
            'gpu_id': 0 if GPU_AVAILABLE else None,
            'verbosity': 0,
            'seed': 42,
            'n_jobs': -1
        },
        'xgboost_gpu': {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'scale_pos_weight': 25.0,
            'min_child_weight': 5,
            'gamma': 0.3,
            'reg_alpha': 0.1,
            'reg_lambda': 2.0,
            'max_bin': 256,
            'gpu_id': 0 if GPU_AVAILABLE else None,
            'predictor': 'gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
            'verbosity': 0,
            'seed': 42,
            'n_jobs': -1
        },
        'logistic': {
            'C': 0.5,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 100,
            'random_state': 42,
            'n_jobs': 4,
            'tol': 0.001,
            'warm_start': True
        }
    }
    
    FEATURE_ENGINEERING_CONFIG = {
        'enable_interaction_features': False,
        'enable_polynomial_features': False,
        'enable_binning': False,
        'enable_target_encoding': False,
        'enable_frequency_encoding': True,
        'enable_statistical_features': False,
        'enable_cross_validation_encoding': False,
        'max_interaction_degree': 2,
        'binning_strategy': 'quantile',
        'n_bins': 6,
        'min_frequency': 10,
        'target_encoding_smoothing': 20.0,
        'target_feature_count': 150,
        'use_feature_selection': False,
        'feature_selection_method': 'f_classif',
        'feature_importance_threshold': 0.01,
        'max_categorical_for_encoding': 5,
        'max_numeric_for_interaction': 20,
        'max_cross_combinations': 10,
        'max_statistical_features': 0,
        'cleanup_after_each_step': True,
        'intermediate_storage': False,
        'use_nvtabular': NVTABULAR_ENABLED,
        'normalize_continuous': False
    }
    
    CV_FOLDS = 5
    CV_SHUFFLE = True
    RANDOM_STATE = 42
    
    EARLY_STOPPING_ROUNDS = 30
    EARLY_STOPPING_TOLERANCE = 1e-5
    
    OPTUNA_N_TRIALS = 150
    OPTUNA_TIMEOUT = 3600
    OPTUNA_N_JOBS = 2
    OPTUNA_VERBOSITY = 1
    
    ENSEMBLE_CONFIG = {
        'voting_weights': {'lightgbm': 0.45, 'xgboost': 0.35, 'logistic': 0.2},
        'stacking_cv_folds': 5,
        'blending_ratio': 0.8,
        'diversity_threshold': 0.03,
        'performance_threshold': 0.20,
        'enable_meta_features': True,
        'use_simple_average': False
    }
    
    CALIBRATION_METHOD = 'isotonic'
    CALIBRATION_CV_FOLDS = 5
    CALIBRATION_MANDATORY = False
    
    EVALUATION_CONFIG = {
        'ap_weight': 0.5,
        'wll_weight': 0.5,
        'target_combined_score': 0.35,
        'target_ctr': 0.0191,
        'ctr_tolerance': 0.001,
        'bias_penalty_weight': 5.0,
        'calibration_weight': 0.4,
        'pos_weight': 51.43,
        'neg_weight': 1.0,
        'wll_normalization_factor': 1.8,
        'ctr_bias_multiplier': 6.0
    }
    
    CTR_BIAS_CORRECTION = {
        'enable': False,
        'target_ctr': 0.0191,
        'correction_factor': 0.5,
        'post_processing': False,
        'clip_range': (0.0005, 0.1),
        'bias_threshold': 0.0003,
        'calibration_strength': 2.0,
        'prediction_scaling': 0.3
    }
    
    PRIMARY_METRIC = 'combined_score'
    SECONDARY_METRICS = ['ap', 'auc', 'log_loss', 'ctr_bias', 'ctr_score']
    TARGET_COMBINED_SCORE = 0.35
    TARGET_CTR = 0.0191
    
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024
    LOG_FILE_BACKUP_COUNT = 5
    
    ENABLE_PARALLEL_PROCESSING = True
    ENABLE_MEMORY_MAPPING = False
    ENABLE_CACHING = True
    CACHE_SIZE_MB = 2048
    
    LARGE_DATASET_MODE = True
    MEMORY_EFFICIENT_SAMPLING = False
    AGGRESSIVE_SAMPLING_THRESHOLD = 0.75
    MIN_SAMPLE_SIZE = 100000
    MAX_SAMPLE_SIZE = 1000000
    
    LOGISTIC_SAMPLING_CONFIG = {
        'normal_size': 1000000,
        'warning_size': 750000,
        'critical_size': 500000,
        'abort_size': 100000,
        'enable_dynamic_sizing': True
    }
    
    RTX_4060_TI_OPTIMIZATION = GPU_AVAILABLE and GPU_NAME and "4060 Ti" in GPU_NAME
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR, 
            cls.MODEL_DIR, 
            cls.LOG_DIR, 
            cls.OUTPUT_DIR,
            cls.RESULTS_DIR,
            cls.NVT_PROCESSED_DIR,
            cls.NVT_WORKFLOW_DIR
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
    def verify_gpu_availability(cls):
        """GPU availability check"""
        print("=== GPU Availability Check ===")
        
        status = {
            'torch_available': TORCH_AVAILABLE,
            'torch_gpu': cls.GPU_AVAILABLE if TORCH_AVAILABLE else False,
            'gpu_count': cls.GPU_COUNT if cls.GPU_AVAILABLE else 0,
            'gpu_name': cls.GPU_NAME if cls.GPU_AVAILABLE else "None",
            'gpu_memory_gb': cls.GPU_MEMORY_GB if cls.GPU_AVAILABLE else 0,
            'rapids_available': RAPIDS_AVAILABLE,
            'nvtabular_available': NVTABULAR_AVAILABLE,
            'rapids_enabled': cls.RAPIDS_ENABLED,
            'nvtabular_enabled': cls.NVTABULAR_ENABLED,
            'rtx_4060_ti_optimization': cls.RTX_4060_TI_OPTIMIZATION
        }
        
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        if cls.GPU_AVAILABLE:
            print(f"\n  GPU Details:")
            print(f"    Device: {cls.DEVICE}")
            print(f"    Name: {cls.GPU_NAME}")
            print(f"    Memory: {cls.GPU_MEMORY_GB:.1f}GB")
            print(f"    Count: {cls.GPU_COUNT}")
            
            xgb_gpu_params = cls.MODEL_TRAINING_CONFIG.get('xgboost_gpu', {})
            print(f"\n  XGBoost GPU Settings:")
            print(f"    tree_method: {xgb_gpu_params.get('tree_method')}")
            print(f"    predictor: {xgb_gpu_params.get('predictor')}")
            print(f"    gpu_id: {xgb_gpu_params.get('gpu_id')}")
            print(f"    max_depth: {xgb_gpu_params.get('max_depth')}")
            print(f"    learning_rate: {xgb_gpu_params.get('learning_rate')}")
        
        print("=== GPU Check Completed ===")
        return status
    
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
            },
            'gpu_features': {
                'gpu_available': cls.GPU_AVAILABLE,
                'gpu_name': cls.GPU_NAME,
                'gpu_memory_gb': cls.GPU_MEMORY_GB,
                'rapids_enabled': cls.RAPIDS_ENABLED,
                'nvtabular_enabled': cls.NVTABULAR_ENABLED,
                'rtx_4060_ti_optimization': cls.RTX_4060_TI_OPTIMIZATION
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
    config.verify_gpu_availability()
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