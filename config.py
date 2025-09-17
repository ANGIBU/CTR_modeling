# config.py

import os
from pathlib import Path
import logging

# PyTorch import 안전 처리
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch가 설치되지 않았습니다. GPU 기능이 비활성화됩니다.")

class Config:
    """프로젝트 전체 설정 관리 - 메모리 최적화"""
    
    # 기본 경로 설정
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # 데이터 파일 경로
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_TEMPLATE_PATH = DATA_DIR / "sample_submission.csv"
    
    # GPU 및 하드웨어 설정
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
    
    # 메모리 설정 (대폭 개선)
    MAX_MEMORY_GB = 45           # 30GB → 45GB로 증가
    CHUNK_SIZE = 25000           # 100,000 → 25,000으로 대폭 축소
    BATCH_SIZE_GPU = 4096        # 8192 → 4096으로 축소 (안정성 확보)
    BATCH_SIZE_CPU = 1024        # 2048 → 1024로 축소
    PREFETCH_FACTOR = 2
    NUM_WORKERS = 6              # 4 → 6으로 증가 (AMD Ryzen 5 5600X 활용)
    
    # 데이터 처리 전략 (더 유연하게)
    TARGET_DATA_USAGE_RATIO = 1.0        # 0.8 → 1.0으로 증가 (전체 데이터 사용)
    MIN_TRAIN_SIZE = 100000              # 최소 10만행
    MAX_TRAIN_SIZE = 12000000            # 500만행 → 1200만행으로 대폭 증가
    MIN_TEST_SIZE = 50000                # 최소 5만행
    MAX_TEST_SIZE = 2000000
    FORCE_FULL_TEST_PREDICTION = True
    FORCE_FULL_DATA_PROCESSING = True    # False → True로 변경 (전체 데이터 처리 강제)
    
    # 대용량 데이터 검증 설정 (현실적으로 조정)
    EXPECTED_TRAIN_SIZE = 10000000       # 1,000,000 → 10,000,000으로 증가
    EXPECTED_TEST_SIZE = 1500000         # 500,000 → 1,500,000으로 증가
    DATA_SIZE_TOLERANCE = 0.3            # 0.2 → 0.3으로 확대 (허용 오차 30%)
    REQUIRE_REAL_DATA = True
    SAMPLE_DATA_FALLBACK = False         # True → False로 변경 (실제 데이터 필수)
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 3
    TEST_SIZE = 0.2
    
    # LightGBM 파라미터 (메모리 최적화)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,           # 255 → 127로 축소
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 200,    # 100 → 200으로 증가 (안정성)
        'min_child_weight': 20,      # 10 → 20으로 증가
        'lambda_l1': 2.0,            # 1.0 → 2.0으로 증가
        'lambda_l2': 2.0,            # 1.0 → 2.0으로 증가
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,        # 1000 → 1500으로 증가
        'early_stopping_rounds': 150, # 100 → 150으로 증가
        'scale_pos_weight': 49.0,
        'force_row_wise': True,
        'max_bin': 255,
        'num_threads': NUM_WORKERS,
        'device_type': 'cpu',
        'min_data_in_leaf': 100,     # 50 → 100으로 증가
        'max_depth': 12,             # 10 → 12로 증가
        'feature_fraction_bynode': 0.8
    }
    
    # XGBoost 파라미터 (메모리 최적화)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 10,             # 8 → 10으로 증가
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 20,      # 10 → 20으로 증가
        'reg_alpha': 2.0,            # 1.0 → 2.0으로 증가
        'reg_lambda': 2.0,           # 1.0 → 2.0으로 증가
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,        # 1000 → 1500으로 증가
        'early_stopping_rounds': 150, # 100 → 150으로 증가
        'max_bin': 255,
        'nthread': NUM_WORKERS,
        'grow_policy': 'depthwise',
        'max_leaves': 127,           # 255 → 127로 축소
        'gamma': 0.1
    }
    
    # CatBoost 파라미터 (메모리 최적화)
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 10,                 # 8 → 10으로 증가
        'learning_rate': 0.05,
        'l2_leaf_reg': 15,           # 10 → 15로 증가
        'iterations': 1500,          # 1000 → 1500으로 증가
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 3,     # 2 → 3으로 증가
        'thread_count': NUM_WORKERS,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'od_type': 'IncToDec',
        'od_wait': 150,              # 100 → 150으로 증가
        'leaf_estimation_iterations': 15, # 10 → 15로 증가
        'grow_policy': 'SymmetricTree',
        'max_leaves': 127,           # 255 → 127로 축소
        'min_data_in_leaf': 100,     # 50 → 100으로 증가
        'rsm': 0.8
    }
    
    # 딥러닝 모델 파라미터 (메모리 최적화)
    NN_PARAMS = {
        'hidden_dims': [512, 256, 128, 64],
        'dropout_rate': 0.3,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 60,                # 50 → 60으로 증가
        'patience': 15,              # 10 → 15로 증가
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
    
    # 피처 엔지니어링 설정 (메모리 효율적)
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 150,   # 100 → 150으로 증가
        'frequency_threshold': 100,         # 50 → 100으로 증가
        'interaction_features': True,       # False → True로 변경
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,
        'id_hash_features': True,
        'polynomial_features': False,
        'binning_features': True,
        'quantile_features': True,          # False → True로 변경
        'rank_features': True,              # False → True로 변경
        'group_statistics': True,
        'lag_features': False,
        'rolling_features': False,
        'fourier_features': False,
        'pca_features': False,
        'clustering_features': False,
        'text_features': False,
        'max_features': 800,                # 500 → 800으로 증가
        'feature_selection_method': 'mutual_info',
        'feature_selection_k': 300,         # 200 → 300으로 증가
        'memory_efficient_mode_threshold': 5000000,  # 1,000,000 → 5,000,000으로 증가
        'chunked_feature_engineering': True,
        'feature_importance_threshold': 0.001
    }
    
    # 평가 설정
    EVALUATION_CONFIG = {
        'ap_weight': 0.5,
        'wll_weight': 0.5,
        'actual_ctr': 0.0201,
        'pos_weight': 0.0201,
        'neg_weight': 0.9799,
        'target_score': 0.36000,
        'bootstrap_samples': 1000,          # 500 → 1000으로 증가
        'confidence_interval': 0.95,
        'stability_threshold': 0.015,
        'performance_metrics': ['ap', 'wll', 'auc', 'f1'],
        'ctr_tolerance': 0.0005,
        'bias_penalty_weight': 2.5,
        'calibration_weight': 0.4,
        'large_data_evaluation': True,      # False → True로 변경
        'evaluation_sample_size': 500000    # 100,000 → 500,000으로 증가
    }
    
    # 앙상블 설정 (향상된 성능)
    ENSEMBLE_CONFIG = {
        'use_optimal_ensemble': True,
        'use_stabilized_ensemble': True,    # False → True로 변경
        'use_meta_learning': True,          # False → True로 변경
        'use_stacking': True,               # False → True로 변경
        'meta_model': 'ridge',
        'calibration_ensemble': True,
        'optimization_method': 'bayesian',  # 'simple' → 'bayesian'로 변경
        'diversification_method': 'weighted',  # 'equal' → 'weighted'로 변경
        'ensemble_types': ['optimal', 'calibrated', 'stacked'],
        'blend_weights': {
            'lgbm': 0.35,
            'xgb': 0.35,
            'cat': 0.30
        },
        'ensemble_optimization_trials': 100,  # 50 → 100으로 증가
        'ensemble_cv_folds': 5,               # 3 → 5로 증가
        'ensemble_early_stopping': 50,       # 25 → 50으로 증가
        'ensemble_regularization': 0.01,
        'dynamic_weighting': True,            # False → True로 변경
        'adaptive_blending': True,            # False → True로 변경
        'temporal_ensemble': False,
        'multi_level_ensemble': True,         # False → True로 변경
        'large_data_ensemble': True,          # False → True로 변경
        'ensemble_memory_limit': 8.0          # 4.0 → 8.0으로 증가
    }
    
    # 로깅 설정
    LOGGING_CONFIG = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True,
        'detailed_logging': True,            # False → True로 변경
        'performance_logging': True,         # False → True로 변경
        'memory_logging': True,
        'model_logging': True,
        'data_processing_logging': True
    }
    
    # 실시간 추론 설정
    INFERENCE_CONFIG = {
        'batch_size': 2000,                  # 1000 → 2000으로 증가
        'single_prediction_batch_size': 1,
        'timeout': 100,
        'cache_size': 20000,                 # 10,000 → 20,000으로 증가
        'model_version': 'v1.0',
        'use_gpu': GPU_AVAILABLE,
        'parallel_inference': True,          # False → True로 변경
        'inference_optimization': True,
        'model_compilation': False,
        'quantization': False,
        'tensorrt_optimization': False,
        'onnx_optimization': False,
        'async_inference': True,             # False → True로 변경
        'memory_pool': True,
        'connection_pool_size': 20,          # 10 → 20으로 증가
        'warm_up_requests': 10,              # 5 → 10으로 증가
        'feature_caching': True,
        'prediction_caching': True,          # False → True로 변경
        'real_time_optimization': True,      # False → True로 변경
        'latency_target_ms': 100,
        'throughput_target_qps': 200,        # 100 → 200으로 증가
        'model_preloading': True,
        'feature_preprocessing_cache': True, # False → True로 변경
        'thread_pool_size': NUM_WORKERS,
        'response_compression': False,
        'monitoring_enabled': True,          # False → True로 변경
        'performance_logging': True,         # False로 변경
        'rule_compliance_check': True,
        'independent_execution': True,
        'no_external_api': True,
        'local_only': True,
        'utf8_encoding': True,
        'relative_path_only': True
    }
    
    # 하이퍼파라미터 튜닝 설정 (향상)
    TUNING_CONFIG = {
        'n_trials': 50,                      # 20 → 50으로 증가
        'timeout': 7200,                     # 3600 → 7200 (2시간)로 증가
        'parallel_jobs': 2,                  # 1 → 2로 증가
        'pruner': 'MedianPruner',
        'sampler': 'TPESampler',
        'optimization_direction': 'maximize',
        'study_storage': None,
        'load_if_exists': True,
        'enable_pruning': True,
        'n_startup_trials': 10,              # 5 → 10으로 증가
        'n_warmup_steps': 5,                 # 3 → 5로 증가
        'interval_steps': 1,
        'percentile': 50.0,
        'min_resource': 1,
        'max_resource': 162,                 # 81 → 162로 증가
        'reduction_factor': 3,
        'bootstrap_count': 100,              # 50 → 100으로 증가
        'multi_objective': False,
        'large_data_tuning': True            # False → True로 변경
    }
    
    # 메모리 관리 설정 (대폭 개선)
    MEMORY_CONFIG = {
        'max_memory_gb': MAX_MEMORY_GB,
        'auto_gc': True,
        'gc_threshold': 0.7,                 # 0.6 → 0.7로 조정 (덜 적극적)
        'force_gc_interval': 50,             # 100 → 50으로 변경 (더 빈번)
        'memory_monitoring': True,
        'memory_limit_warning': 0.75,        # 0.7 → 0.75로 조정
        'memory_limit_error': 0.85,          # 동일
        'memory_limit_abort': 0.95,          # 새로 추가
        'chunk_memory_limit': 8.0,           # 4.0 → 8.0으로 증가
        'batch_memory_limit': 4.0,           # 2.0 → 4.0으로 증가
        'model_memory_limit': 12.0,          # 6.0 → 12.0으로 증가
        'ensemble_memory_limit': 16.0,       # 8.0 → 16.0으로 증가
        'swap_usage_limit': 0.05,
        'memory_profiling': True,            # False → True로 변경
        'memory_optimization': True,
        'lazy_loading': True,
        'memory_mapping': False,
        'compressed_storage': True,
        'aggressive_memory_management': True,
        'large_data_memory_strategy': True   # False → True로 변경
    }
    
    # GPU 설정 (RTX 4060 Ti 16GB 최적화)
    GPU_CONFIG = {
        'gpu_memory_fraction': 0.8,          # 0.7 → 0.8로 증가
        'allow_growth': True,
        'mixed_precision': USE_MIXED_PRECISION,
        'tensor_core_optimization': True,    # False → True로 변경
        'cuda_optimization_level': GPU_OPTIMIZATION_LEVEL,
        'cuda_cache_config': 'PreferShared',
        'cudnn_benchmark': True,
        'cudnn_deterministic': False,
        'cuda_launch_blocking': False,
        'gpu_memory_monitoring': True,
        'gpu_utilization_monitoring': True,  # False → True로 변경
        'multi_gpu': False,
        'gpu_sync_interval': 200,
        'gpu_memory_pool': True,             # False → True로 변경
        'gpu_kernel_sync': False,
        'gpu_profiling': False,
        'tensor_parallelism': False,
        'pipeline_parallelism': False,
        'gradient_checkpointing': True,      # False → True로 변경 (메모리 절약)
        'activation_checkpointing': True     # False → True로 변경
    }
    
    # 병렬 처리 설정 (향상)
    PARALLEL_CONFIG = {
        'num_workers': NUM_WORKERS,
        'max_workers': NUM_WORKERS,
        'thread_pool_size': NUM_WORKERS * 3, # 2 → 3으로 증가
        'process_pool_size': NUM_WORKERS,
        'multiprocessing_context': 'spawn',
        'shared_memory': True,               # False → True로 변경
        'parallel_backend': 'threading',
        'parallel_feature_engineering': True, # False → True로 변경
        'parallel_model_training': True,     # False → True로 변경
        'parallel_inference': True,          # False → True로 변경
        'parallel_evaluation': True,         # False → True로 변경
        'thread_local_storage': True,
        'numa_optimization': False,
        'cpu_affinity': False,
        'priority_scheduling': False,
        'load_balancing': True,              # False → True로 변경
        'work_stealing': True,               # False → True로 변경
        'dynamic_scheduling': True,          # False → True로 변경
        'parallel_io': True,                 # False → True로 변경
        'async_processing': True             # False → True로 변경
    }
    
    # 데이터 처리 설정 (최적화)
    DATA_CONFIG = {
        'use_pyarrow': True,
        'compression': 'snappy',
        'memory_map': False,
        'lazy_loading': True,
        'batch_processing': True,
        'streaming_processing': True,        # False → True로 변경
        'data_validation': True,
        'schema_validation': True,           # False → True로 변경
        'type_optimization': True,
        'categorical_optimization': True,
        'string_optimization': True,
        'datetime_optimization': True,       # False → True로 변경
        'numeric_optimization': True,
        'memory_efficient_dtypes': True,
        'sparse_arrays': False,
        'columnar_storage': True,
        'indexed_access': True,              # False → True로 변경
        'cached_operations': True,           # False → True로 변경
        'vectorized_operations': True,
        'broadcast_operations': True,        # False → True로 변경
        'parallel_reading': True,            # False → True로 변경
        'async_io': True,                    # False → True로 변경
        'prefetch_batches': True,            # False → True로 변경
        'chunk_size': CHUNK_SIZE,
        'max_memory_usage': MAX_MEMORY_GB,
        'large_data_optimization': True      # False → True로 변경
    }
    
    # 모델 저장/로딩 설정
    MODEL_IO_CONFIG = {
        'compression_level': 6,              # 3 → 6으로 증가
        'pickle_protocol': 5,                # 4 → 5로 증가
        'joblib_compression': 'lz4',         # 'zlib' → 'lz4'로 변경 (더 빠름)
        'model_versioning': True,
        'incremental_saving': True,          # False → True로 변경
        'checkpoint_frequency': 200,         # 100 → 200으로 증가
        'backup_models': True,               # False → True로 변경
        'model_metadata': True,
        'model_signature': True,             # False → True로 변경
        'model_validation': True,
        'lazy_model_loading': True,          # False → True로 변경
        'model_caching': True,               # False → True로 변경
        'model_pooling': True,               # False → True로 변경
        'distributed_storage': False,
        'cloud_storage': False,
        'local_storage_optimization': True,
        'model_compression': True,
        'large_model_handling': True         # False → True로 변경
    }
    
    @classmethod
    def verify_data_requirements(cls):
        """개선된 데이터 요구사항 검증"""
        print("=== 대용량 데이터 요구사항 검증 ===")
        
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
        
        # 실제 파일 크기에 맞춘 검증 기준
        min_train_size_mb = 5000    # 100MB → 5000MB (5GB)로 증가
        min_test_size_mb = 800      # 50MB → 800MB로 증가
        
        requirements['train_size_adequate'] = requirements['train_file_size_mb'] >= min_train_size_mb
        requirements['test_size_adequate'] = requirements['test_file_size_mb'] >= min_test_size_mb
        requirements['memory_adequate'] = requirements['memory_available'] >= 40  # 20GB → 40GB로 증가
        
        for key, value in requirements.items():
            status = "✓" if (isinstance(value, bool) and value) or (isinstance(value, (int, float)) and value > 0) else "✗"
            print(f"{status} {key}: {value}")
        
        # 전체 요구사항 충족 여부
        critical_checks = [
            requirements['train_file_exists'],
            requirements['test_file_exists'],
            requirements['train_size_adequate'],
            requirements['test_size_adequate'],
            requirements['memory_adequate']
        ]
        
        all_requirements_met = all(critical_checks)
        print(f"\n전체 요구사항 충족: {'✓' if all_requirements_met else '✗'}")
        
        if all_requirements_met:
            print("대용량 데이터 처리 준비 완료!")
        else:
            print("일부 요구사항이 충족되지 않았습니다.")
            
        print("=== 검증 완료 ===\n")
        
        return requirements
    
    @classmethod
    def setup_directories(cls):
        """필요한 디렉터리 생성"""
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
                print(f"디렉터리 생성 실패 {dir_path}: {e}")
                failed_dirs.append(str(dir_path))
        
        if created_dirs:
            print(f"생성된 디렉터리: {created_dirs}")
        
        if failed_dirs:
            print(f"생성 실패한 디렉터리: {failed_dirs}")
            
        # 데이터 디렉터리 강제 생성
        if not cls.DATA_DIR.exists():
            try:
                os.makedirs(cls.DATA_DIR, exist_ok=True)
                print(f"강제 생성된 데이터 디렉터리: {cls.DATA_DIR}")
            except Exception as e:
                raise RuntimeError(f"데이터 디렉터리 생성 실패: {cls.DATA_DIR}, 오류: {e}")
    
    @classmethod
    def verify_paths(cls):
        """경로 유효성 검증"""
        print("=== 경로 유효성 검증 ===")
        
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
                print(f"{name}: {abs_path} (존재: {exists}, 크기: {file_size_mb:.1f}MB)")
            else:
                print(f"{name}: {abs_path} (존재: {exists})")
        
        print("=== 검증 완료 ===")
    
    @classmethod
    def setup_logging(cls):
        """로깅 설정 초기화"""
        cls.setup_directories()
        
        logger = logging.getLogger()
        logger.setLevel(cls.LOGGING_CONFIG['level'])
        
        # 기존 핸들러 제거
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
                print(f"로그 파일 생성: {log_file_path}")
            except Exception as e:
                print(f"파일 핸들러 설정 실패: {e}")
        
        return logger
    
    @classmethod
    def get_memory_config(cls):
        """개선된 메모리 설정 반환"""
        try:
            import psutil
            
            total_memory = psutil.virtual_memory().total / (1024**3)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            # 사용 가능 메모리의 80%로 확대 (70% → 80%)
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
        """대용량 데이터 처리 설정 반환"""
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
            'large_data_optimization': cls.DATA_CONFIG['large_data_optimization']
        }
    
    @classmethod
    def get_safe_memory_limits(cls):
        """안전한 메모리 한계 설정 (RTX 4060 Ti 16GB + 64GB RAM 최적화)"""
        try:
            import psutil
            
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            
            # RTX 4060 Ti 16GB + 64GB RAM 환경에 최적화된 메모리 사용
            safe_limit = min(cls.MAX_MEMORY_GB, available_gb * 0.8)  # 80%까지 사용
            
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
        """RTX 4060 Ti 16GB 최적화 GPU 환경 설정"""
        if not cls.GPU_AVAILABLE:
            print("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            return False
            
        try:
            if TORCH_AVAILABLE:
                import torch
                
                # RTX 4060 Ti 최적화 설정
                torch.backends.cudnn.benchmark = cls.GPU_CONFIG['cudnn_benchmark']
                torch.backends.cudnn.deterministic = cls.GPU_CONFIG['cudnn_deterministic']
                
                # Mixed Precision 설정
                if cls.GPU_CONFIG['mixed_precision']:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                # 메모리 최적화
                torch.cuda.set_per_process_memory_fraction(cls.GPU_CONFIG['gpu_memory_fraction'])
                
                # GPU 정보 출력
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                print(f"GPU 환경 설정 완료: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"GPU 메모리 사용률 제한: {cls.GPU_CONFIG['gpu_memory_fraction']*100}%")
                print(f"Mixed Precision: {cls.GPU_CONFIG['mixed_precision']}")
                
                return True
                
        except Exception as e:
            print(f"GPU 환경 설정 실패: {e}")
            
        return False

# 환경변수 설정 (RTX 4060 Ti + AMD Ryzen 5 5600X 최적화)
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['MKL_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMEXPR_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMBA_NUM_THREADS'] = str(Config.NUM_WORKERS)
    
    # 대용량 데이터 처리용 환경변수
    os.environ['PANDAS_MAX_COLUMNS'] = '2000'        # 1000 → 2000으로 증가
    os.environ['PANDAS_MAX_ROWS'] = '15000000'       # 5,000,000 → 15,000,000으로 증가
    
    # CUDA 환경 변수 (RTX 4060 Ti 16GB 최적화)
    if Config.GPU_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_VISIBLE_DEVICES
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # 512 → 1024로 증가
        os.environ['CUDA_MEMORY_FRACTION'] = '0.8'   # 추가
    
    # 메모리 관리 환경 변수 (64GB RAM 최적화)
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '200000'   # 100000 → 200000으로 증가
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '262144'   # 131072 → 262144로 증가
    os.environ['MALLOC_MMAP_MAX_'] = '65536'          # 추가
    
    # 규칙 준수 환경 변수
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'
    
    # PyArrow 최적화
    os.environ['ARROW_USER_SIMD_LEVEL'] = 'AVX2'      # 추가
    os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system' # 추가
    
    # LightGBM/XGBoost 최적화
    os.environ['LIGHTGBM_EXEC_PREFER'] = 'disk'       # 추가
    os.environ['XGBOOST_CACHE_PREFERENCE'] = 'memory'  # 추가
    
except Exception as e:
    print(f"환경 변수 설정 실패: {e}")

# 시작 시 검증
try:
    print("=== CTR 모델링 시스템 초기화 ===")
    Config.setup_directories()
    Config.verify_paths()
    Config.verify_data_requirements()
    
    # GPU 환경 설정
    if Config.setup_gpu_environment():
        print("GPU 환경 설정 성공")
    else:
        print("CPU 모드로 실행됩니다")
        
    print("=== 초기화 완료 ===")
except Exception as e:
    print(f"초기화 실패: {e}")