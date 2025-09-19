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
    
    # Memory settings
    MAX_MEMORY_GB = 45
    CHUNK_SIZE = 25000
    BATCH_SIZE_GPU = 4096
    BATCH_SIZE_CPU = 1024
    PREFETCH_FACTOR = 2
    NUM_WORKERS = 6
    
    # Data processing strategy
    TARGET_DATA_USAGE_RATIO = 1.0
    MIN_TRAIN_SIZE = 100000
    MAX_TRAIN_SIZE = 12000000
    MIN_TEST_SIZE = 50000
    MAX_TEST_SIZE = 2000000
    FORCE_FULL_TEST_PREDICTION = True
    FORCE_FULL_DATA_PROCESSING = True
    
    # Large data validation settings
    EXPECTED_TRAIN_SIZE = 10000000
    EXPECTED_TEST_SIZE = 1500000
    DATA_SIZE_TOLERANCE = 0.3
    REQUIRE_REAL_DATA = True
    SAMPLE_DATA_FALLBACK = False
    
    # Model hyperparameters
    RANDOM_STATE = 42
    N_SPLITS = 3
    TEST_SIZE = 0.2
    
    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 200,
        'min_child_weight': 20,
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,
        'early_stopping_rounds': 150,
        'scale_pos_weight': 49.0,
        'force_row_wise': True,
        'max_bin': 255,
        'num_threads': NUM_WORKERS,
        'device_type': 'cpu',
        'min_data_in_leaf': 100,
        'max_depth': 12,
        'feature_fraction_bynode': 0.8
    }
    
    # XGBoost parameters
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 20,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,
        'early_stopping_rounds': 150,
        'max_bin': 255,
        'nthread': NUM_WORKERS,
        'grow_policy': 'depthwise',
        'max_leaves': 127,
        'gamma': 0.1
    }
    
    # CatBoost parameters
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 10,
        'learning_rate': 0.05,
        'l2_leaf_reg': 15,
        'iterations': 1500,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 3,
        'thread_count': NUM_WORKERS,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'od_type': 'IncToDec',
        'od_wait': 150,
        'leaf_estimation_iterations': 15,
        'grow_policy': 'SymmetricTree',
        'max_leaves': 127,
        'min_data_in_leaf': 100,
        'rsm': 0.8
    }
    
    # Deep learning model parameters
    NN_PARAMS = {
        'hidden_dims': [512, 256, 128, 64],
        'dropout_rate': 0.3,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 60,
        'patience': 15,
        'use_batch_norm': True,
        'activation': 'relu',
        'use_residual': False,
        'use_attention': False,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'label_smoothing': 0.01,
        'gradient_clip_val': 1.0,
        'scheduler_type': 'step',
        'min_lr': 1e-6
    }
    
    # Feature engineering settings
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 150,
        'frequency_threshold': 100,
        'interaction_features': True,
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,
        'id_hash_features': True,
        'polynomial_features': False,
        'binning_features': True,
        'quantile_features': True,
        'rank_features': True,
        'group_statistics': True,
        'lag_features': False,
        'rolling_features': False,
        'fourier_features': False,
        'pca_features': False,
        'clustering_features': False,
        'text_features': False,
        'max_features': 800,
        'feature_selection_method': 'mutual_info',
        'feature_selection_k': 300,
        'memory_efficient_mode_threshold': 5000000,
        'chunked_feature_engineering': True,
        'feature_importance_threshold': 0.001,
        # Target column related settings
        'target_column_candidates': TARGET_COLUMN_CANDIDATES,
        'target_detection': TARGET_DETECTION_CONFIG,
        'auto_detect_target': True,
        'strict_target_validation': True
    }
    
    # Evaluation settings
    EVALUATION_CONFIG = {
        'ap_weight': 0.5,
        'wll_weight': 0.5,
        'actual_ctr': 0.0201,
        'pos_weight': 0.0201,
        'neg_weight': 0.9799,
        'target_score': 0.36000,
        'bootstrap_samples': 1000,
        'confidence_interval': 0.95,
        'stability_threshold': 0.015,
        'performance_metrics': ['ap', 'wll', 'auc', 'f1'],
        'ctr_tolerance': 0.0005,
        'bias_penalty_weight': 2.5,
        'calibration_weight': 0.4,
        'large_data_evaluation': True,
        'evaluation_sample_size': 500000
    }
    
    # Ensemble settings
    ENSEMBLE_CONFIG = {
        'use_optimal_ensemble': True,
        'use_stabilized_ensemble': True,
        'use_meta_learning': True,
        'use_stacking': True,
        'meta_model': 'ridge',
        'calibration_ensemble': True,
        'optimization_method': 'bayesian',
        'diversification_method': 'weighted',
        'ensemble_types': ['optimal', 'calibrated', 'stacked'],
        'blend_weights': {
            'lgbm': 0.35,
            'xgb': 0.35,
            'cat': 0.30
        },
        'ensemble_optimization_trials': 100,
        'ensemble_cv_folds': 5,
        'ensemble_early_stopping': 50,
        'ensemble_regularization': 0.01,
        'dynamic_weighting': True,
        'adaptive_blending': True,
        'temporal_ensemble': False,
        'multi_level_ensemble': True,
        'large_data_ensemble': True,
        'ensemble_memory_limit': 8.0
    }
    
    # Logging settings
    LOGGING_CONFIG = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True,
        'detailed_logging': True,
        'performance_logging': True,
        'memory_logging': True,
        'model_logging': True,
        'data_processing_logging': True
    }
    
    # Real-time inference settings
    INFERENCE_CONFIG = {
        'batch_size': 2000,
        'single_prediction_batch_size': 1,
        'timeout': 100,
        'cache_size': 20000,
        'model_version': 'v1.0',
        'use_gpu': GPU_AVAILABLE,
        'parallel_inference': True,
        'inference_optimization': True,
        'model_compilation': False,
        'quantization': False,
        'tensorrt_optimization': False,
        'onnx_optimization': False,
        'async_inference': True,
        'memory_pool': True,
        'connection_pool_size': 20,
        'warm_up_requests': 10,
        'feature_caching': True,
        'prediction_caching': True,
        'real_time_optimization': True,
        'latency_target_ms': 100,
        'throughput_target_qps': 200,
        'model_preloading': True,
        'feature_preprocessing_cache': True,
        'thread_pool_size': NUM_WORKERS,
        'response_compression': False,
        'monitoring_enabled': True,
        'performance_logging': True,
        'rule_compliance_check': True,
        'independent_execution': True,
        'no_external_api': True,
        'local_only': True,
        'utf8_encoding': True,
        'relative_path_only': True
    }
    
    # Hyperparameter tuning settings
    TUNING_CONFIG = {
        'n_trials': 50,
        'timeout': 7200,
        'parallel_jobs': 2,
        'pruner': 'MedianPruner',
        'sampler': 'TPESampler',
        'optimization_direction': 'maximize',
        'study_storage': None,
        'load_if_exists': True,
        'enable_pruning': True,
        'n_startup_trials': 10,
        'n_warmup_steps': 5,
        'interval_steps': 1,
        'percentile': 50.0,
        'min_resource': 1,
        'max_resource': 162,
        'reduction_factor': 3,
        'bootstrap_count': 100,
        'multi_objective': False,
        'large_data_tuning': True
    }
    
    # Memory management settings
    MEMORY_CONFIG = {
        'max_memory_gb': MAX_MEMORY_GB,
        'auto_gc': True,
        'gc_threshold': 0.7,
        'force_gc_interval': 50,
        'memory_monitoring': True,
        'memory_limit_warning': 0.75,
        'memory_limit_error': 0.85,
        'memory_limit_abort': 0.95,
        'chunk_memory_limit': 8.0,
        'batch_memory_limit': 4.0,
        'model_memory_limit': 12.0,
        'ensemble_memory_limit': 16.0,
        'swap_usage_limit': 0.05,
        'memory_profiling': True,
        'memory_optimization': True,
        'lazy_loading': True,
        'memory_mapping': False,
        'compressed_storage': True,
        'aggressive_memory_management': True,
        'large_data_memory_strategy': True
    }
    
    # GPU settings
    GPU_CONFIG = {
        'gpu_memory_fraction': 0.8,
        'allow_growth': True,
        'mixed_precision': USE_MIXED_PRECISION,
        'tensor_core_optimization': True,
        'cuda_optimization_level': GPU_OPTIMIZATION_LEVEL,
        'cuda_cache_config': 'PreferShared',
        'cudnn_benchmark': True,
        'cudnn_deterministic': False,
        'cuda_launch_blocking': False,
        'gpu_memory_monitoring': True,
        'gpu_utilization_monitoring': True,
        'multi_gpu': False,
        'gpu_sync_interval': 200,
        'gpu_memory_pool': True,
        'gpu_kernel_sync': False,
        'gpu_profiling': False,
        'tensor_parallelism': False,
        'pipeline_parallelism': False,
        'gradient_checkpointing': True,
        'activation_checkpointing': True
    }
    
    # Parallel processing settings
    PARALLEL_CONFIG = {
        'num_workers': NUM_WORKERS,
        'max_workers': NUM_WORKERS,
        'thread_pool_size': NUM_WORKERS * 3,
        'process_pool_size': NUM_WORKERS,
        'multiprocessing_context': 'spawn',
        'shared_memory': True,
        'parallel_backend': 'threading',
        'parallel_feature_engineering': True,
        'parallel_model_training': True,
        'parallel_inference': True,
        'parallel_evaluation': True,
        'thread_local_storage': True,
        'numa_optimization': False,
        'cpu_affinity': False,
        'priority_scheduling': False,
        'load_balancing': True,
        'work_stealing': True,
        'dynamic_scheduling': True,
        'parallel_io': True,
        'async_processing': True
    }
    
    # Data processing settings
    DATA_CONFIG = {
        'use_pyarrow': True,
        'compression': 'snappy',
        'memory_map': False,
        'lazy_loading': True,
        'batch_processing': True,
        'streaming_processing': True,
        'data_validation': True,
        'schema_validation': True,
        'type_optimization': True,
        'categorical_optimization': True,
        'string_optimization': True,
        'datetime_optimization': True,
        'numeric_optimization': True,
        'memory_efficient_dtypes': True,
        'sparse_arrays': False,
        'columnar_storage': True,
        'indexed_access': True,
        'cached_operations': True,
        'vectorized_operations': True,
        'broadcast_operations': True,
        'parallel_reading': True,
        'async_io': True,
        'prefetch_batches': True,
        'chunk_size': CHUNK_SIZE,
        'max_memory_usage': MAX_MEMORY_GB,
        'large_data_optimization': True,
        # Target column processing related
        'target_column_detection': True,
        'validate_target_column': True,
        'target_column_fallback': 'clicked'
    }
    
    # Model save/load settings
    MODEL_IO_CONFIG = {
        'compression_level': 6,
        'pickle_protocol': 5,
        'joblib_compression': 'lz4',
        'model_versioning': True,
        'incremental_saving': True,
        'checkpoint_frequency': 200,
        'backup_models': True,
        'model_metadata': True,
        'model_signature': True,
        'model_validation': True,
        'lazy_model_loading': True,
        'model_caching': True,
        'model_pooling': True,
        'distributed_storage': False,
        'cloud_storage': False,
        'local_storage_optimization': True,
        'model_compression': True,
        'large_model_handling': True
    }
    
    @classmethod
    def get_target_column_config(cls):
        """Return target column configuration"""
        return {
            'candidates': cls.TARGET_COLUMN_CANDIDATES,
            'detection_config': cls.TARGET_DETECTION_CONFIG,
            'feature_config': {
                'auto_detect': cls.FEATURE_CONFIG['auto_detect_target'],
                'strict_validation': cls.FEATURE_CONFIG['strict_target_validation']
            },
            'data_config': {
                'detection_enabled': cls.DATA_CONFIG['target_column_detection'],
                'validation_enabled': cls.DATA_CONFIG['validate_target_column'],
                'fallback_column': cls.DATA_CONFIG['target_column_fallback']
            }
        }
    
    @classmethod
    def verify_data_requirements(cls):
        """Verify data requirements"""
        print("=== Large data requirements verification ===")
        
        requirements = {
            'train_file_exists': cls.TRAIN_PATH.exists(),
            'test_file_exists': cls.TEST_PATH.exists(),
            'train_file_size_mb': cls.TRAIN_PATH.stat().st_size / (1024**2) if cls.TRAIN_PATH.exists() else 0,
            'test_file_size_mb': cls.TEST_PATH.stat().st_size / (1024**2) if cls.TEST_PATH.exists() else 0,
            'memory_available': cls.MAX_MEMORY_GB,
            'chunk_size': cls.CHUNK_SIZE,
            'expected_train_size': cls.EXPECTED_TRAIN_SIZE,
            'expected_test_size': cls.EXPECTED_TEST_SIZE
        }
        
        # Verification criteria based on actual file sizes
        min_train_size_mb = 5000
        min_test_size_mb = 800
        
        requirements['train_size_adequate'] = requirements['train_file_size_mb'] >= min_train_size_mb
        requirements['test_size_adequate'] = requirements['test_file_size_mb'] >= min_test_size_mb
        requirements['memory_adequate'] = requirements['memory_available'] >= 40
        
        for key, value in requirements.items():
            status = "✓" if (isinstance(value, bool) and value) or (isinstance(value, (int, float)) and value > 0) else "✗"
            print(f"{status} {key}: {value}")
        
        # Check if all requirements are met
        critical_checks = [
            requirements['train_file_exists'],
            requirements['test_file_exists'],
            requirements['train_size_adequate'],
            requirements['test_size_adequate'],
            requirements['memory_adequate']
        ]
        
        all_requirements_met = all(critical_checks)
        print(f"\nAll requirements met: {'✓' if all_requirements_met else '✗'}")
        
        if all_requirements_met:
            print("Large data processing ready!")
        else:
            print("Some requirements not met.")
            
        print("=== Verification completed ===\n")
        
        return requirements
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]
        
        created_dirs = []
        failed_dirs = []
        
        for dir_path in directories:
            try:
                dir_path = Path(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                
                if dir_path.exists() and dir_path.is_dir():
                    created_dirs.append(str(dir_path))
                else:
                    failed_dirs.append(str(dir_path))
                    
            except Exception as e:
                print(f"Directory creation failed {dir_path}: {e}")
                failed_dirs.append(str(dir_path))
        
        if created_dirs:
            print(f"Created directories: {created_dirs}")
        
        if failed_dirs:
            print(f"Failed to create directories: {failed_dirs}")
            
        # Force create data directory
        if not cls.DATA_DIR.exists():
            try:
                os.makedirs(cls.DATA_DIR, exist_ok=True)
                print(f"Force created data directory: {cls.DATA_DIR}")
            except Exception as e:
                raise RuntimeError(f"Data directory creation failed: {cls.DATA_DIR}, error: {e}")
    
    @classmethod
    def verify_paths(cls):
        """Verify path validity"""
        print("=== Path validity verification ===")
        
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
            abs_path = Path(path).resolve()
            exists = abs_path.exists()
            
            if name in ['TRAIN_PATH', 'TEST_PATH'] and exists:
                file_size_mb = abs_path.stat().st_size / (1024**2)
                print(f"{name}: {abs_path} (exists: {exists}, size: {file_size_mb:.1f}MB)")
            else:
                print(f"{name}: {abs_path} (exists: {exists})")
        
        print("=== Verification completed ===")
    
    @classmethod
    def setup_logging(cls):
        """Initialize logging configuration"""
        cls.setup_directories()
        
        logger = logging.getLogger()
        logger.setLevel(cls.LOGGING_CONFIG['level'])
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        formatter = logging.Formatter(cls.LOGGING_CONFIG['format'])
        
        if cls.LOGGING_CONFIG['console_handler']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if cls.LOGGING_CONFIG['file_handler']:
            try:
                log_file_path = cls.LOG_DIR / 'ctr_model.log'
                file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                print(f"Log file created: {log_file_path}")
            except Exception as e:
                print(f"File handler setup failed: {e}")
        
        return logger
    
    @classmethod
    def get_memory_config(cls):
        """Return memory configuration"""
        try:
            import psutil
            
            total_memory = psutil.virtual_memory().total / (1024**3)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            max_memory = min(cls.MAX_MEMORY_GB, available_memory * 0.8)
            
            return {
                'max_memory_gb': max_memory,
                'chunk_size': cls.CHUNK_SIZE,
                'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'prefetch_factor': cls.PREFETCH_FACTOR,
                'num_workers': cls.NUM_WORKERS,
                'memory_monitoring': cls.MEMORY_CONFIG['memory_monitoring'],
                'auto_gc': cls.MEMORY_CONFIG['auto_gc'],
                'gc_threshold': cls.MEMORY_CONFIG['gc_threshold'],
                'aggressive_cleanup': cls.MEMORY_CONFIG['aggressive_memory_management']
            }
        except ImportError:
            return {
                'max_memory_gb': cls.MAX_MEMORY_GB,
                'chunk_size': cls.CHUNK_SIZE,
                'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'prefetch_factor': cls.PREFETCH_FACTOR,
                'num_workers': cls.NUM_WORKERS,
                'memory_monitoring': True,
                'auto_gc': True,
                'gc_threshold': 0.7,
                'aggressive_cleanup': True
            }
    
    @classmethod
    def get_data_config(cls):
        """Return large data processing configuration"""
        memory_config = cls.get_memory_config()
        
        return {
            'max_train_size': cls.MAX_TRAIN_SIZE,
            'max_test_size': cls.MAX_TEST_SIZE,
            'min_train_size': cls.MIN_TRAIN_SIZE,
            'min_test_size': cls.MIN_TEST_SIZE,
            'expected_train_size': cls.EXPECTED_TRAIN_SIZE,
            'expected_test_size': cls.EXPECTED_TEST_SIZE,
            'chunk_size': memory_config['chunk_size'],
            'target_usage_ratio': cls.TARGET_DATA_USAGE_RATIO,
            'force_full_test_data': cls.FORCE_FULL_TEST_PREDICTION,
            'force_full_data_processing': cls.FORCE_FULL_DATA_PROCESSING,
            'require_real_data': cls.REQUIRE_REAL_DATA,
            'sample_data_fallback': cls.SAMPLE_DATA_FALLBACK,
            'data_size_tolerance': cls.DATA_SIZE_TOLERANCE,
            'use_pyarrow': cls.DATA_CONFIG['use_pyarrow'],
            'lazy_loading': cls.DATA_CONFIG['lazy_loading'],
            'batch_processing': cls.DATA_CONFIG['batch_processing'],
            'streaming_processing': cls.DATA_CONFIG['streaming_processing'],
            'type_optimization': cls.DATA_CONFIG['type_optimization'],
            'large_data_optimization': cls.DATA_CONFIG['large_data_optimization'],
            'target_column_detection': cls.DATA_CONFIG['target_column_detection'],
            'validate_target_column': cls.DATA_CONFIG['validate_target_column'],
            'target_column_fallback': cls.DATA_CONFIG['target_column_fallback']
        }
    
    @classmethod
    def get_safe_memory_limits(cls):
        """Set safe memory limits"""
        try:
            import psutil
            
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            
            safe_limit = min(cls.MAX_MEMORY_GB, available_gb * 0.8)
            
            return {
                'total_memory_gb': total_gb,
                'available_memory_gb': available_gb,
                'safe_memory_limit_gb': safe_limit,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'memory_monitoring_enabled': cls.MEMORY_CONFIG['memory_monitoring'],
                'auto_gc_enabled': cls.MEMORY_CONFIG['auto_gc'],
                'aggressive_cleanup_enabled': cls.MEMORY_CONFIG['aggressive_memory_management'],
                'memory_limits': {
                    'chunk': cls.MEMORY_CONFIG['chunk_memory_limit'],
                    'batch': cls.MEMORY_CONFIG['batch_memory_limit'],
                    'model': cls.MEMORY_CONFIG['model_memory_limit'],
                    'ensemble': cls.MEMORY_CONFIG['ensemble_memory_limit']
                },
                'gpu_limits': {
                    'gpu_memory_fraction': cls.GPU_CONFIG['gpu_memory_fraction'],
                    'gradient_checkpointing': cls.GPU_CONFIG['gradient_checkpointing'],
                    'mixed_precision': cls.GPU_CONFIG['mixed_precision']
                }
            }
        except ImportError:
            return {
                'total_memory_gb': 64.0,
                'available_memory_gb': 45.0,
                'safe_memory_limit_gb': cls.MAX_MEMORY_GB,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'memory_monitoring_enabled': True,
                'auto_gc_enabled': True,
                'aggressive_cleanup_enabled': True,
                'memory_limits': {
                    'chunk': 8.0,
                    'batch': 4.0,
                    'model': 12.0,
                    'ensemble': 16.0
                },
                'gpu_limits': {
                    'gpu_memory_fraction': 0.8,
                    'gradient_checkpointing': True,
                    'mixed_precision': True
                }
            }

    @classmethod
    def setup_gpu_environment(cls):
        """Set up GPU environment"""
        if not cls.GPU_AVAILABLE:
            print("GPU unavailable. Running in CPU mode.")
            return False
            
        try:
            if TORCH_AVAILABLE:
                import torch
                
                torch.backends.cudnn.benchmark = cls.GPU_CONFIG['cudnn_benchmark']
                torch.backends.cudnn.deterministic = cls.GPU_CONFIG['cudnn_deterministic']
                
                # Mixed Precision settings
                if cls.GPU_CONFIG['mixed_precision']:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # Memory optimization
                torch.cuda.set_per_process_memory_fraction(cls.GPU_CONFIG['gpu_memory_fraction'])
                
                # Display GPU information
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                print(f"GPU environment setup completed: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"GPU memory usage limit: {cls.GPU_CONFIG['gpu_memory_fraction']*100}%")
                print(f"Mixed Precision: {cls.GPU_CONFIG['mixed_precision']}")
                
                return True
                
        except Exception as e:
            print(f"GPU environment setup failed: {e}")
            
        return False

# Environment variable settings
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['MKL_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMEXPR_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMBA_NUM_THREADS'] = str(Config.NUM_WORKERS)
    
    # Environment variables for large data processing
    os.environ['PANDAS_MAX_COLUMNS'] = '2000'
    os.environ['PANDAS_MAX_ROWS'] = '15000000'
    
    # CUDA environment variables
    if Config.GPU_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_VISIBLE_DEVICES
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
        os.environ['CUDA_MEMORY_FRACTION'] = '0.8'
    
    # Memory management environment variables
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