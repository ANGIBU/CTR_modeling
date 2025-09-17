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
    """프로젝트 전체 설정 관리"""
    
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
    
    # 메모리 설정 (보수적으로 조정)
    MAX_MEMORY_GB = 30
    CHUNK_SIZE = 100000  # 10만행으로 축소
    BATCH_SIZE_GPU = 8192  # GPU 배치 크기 축소
    BATCH_SIZE_CPU = 2048  # CPU 배치 크기 축소
    PREFETCH_FACTOR = 2
    NUM_WORKERS = 4  # 워커 수 축소
    
    # 데이터 처리 전략
    TARGET_DATA_USAGE_RATIO = 0.8  # 80% 사용으로 보수적 설정
    MIN_TRAIN_SIZE = 100000  # 최소 10만행
    MAX_TRAIN_SIZE = 5000000  # 최대 500만행으로 축소
    MIN_TEST_SIZE = 50000  # 최소 5만행
    MAX_TEST_SIZE = 2000000
    FORCE_FULL_TEST_PREDICTION = True
    FORCE_FULL_DATA_PROCESSING = False  # 전체 데이터 처리 강제하지 않음
    
    # 대용량 데이터 검증 설정
    EXPECTED_TRAIN_SIZE = 1000000  # 예상 크기 축소
    EXPECTED_TEST_SIZE = 500000
    DATA_SIZE_TOLERANCE = 0.2  # 허용 오차 20%로 확대
    REQUIRE_REAL_DATA = True
    SAMPLE_DATA_FALLBACK = True  # 샘플 데이터 대체 활성화
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 3
    TEST_SIZE = 0.2
    
    # LightGBM 파라미터 (보수적 설정)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 255,  # 축소
        'learning_rate': 0.05,  # 증가
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 100,  # 축소
        'min_child_weight': 10,  # 축소
        'lambda_l1': 1.0,  # 축소
        'lambda_l2': 1.0,  # 축소
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1000,  # 축소
        'early_stopping_rounds': 100,
        'scale_pos_weight': 49.0,
        'force_row_wise': True,
        'max_bin': 255,  # 축소
        'num_threads': NUM_WORKERS,
        'device_type': 'cpu',
        'min_data_in_leaf': 50,  # 축소
        'max_depth': 10,  # 축소
        'feature_fraction_bynode': 0.8
    }
    
    # XGBoost 파라미터 (보수적 설정)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 8,  # 축소
        'learning_rate': 0.05,  # 증가
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 10,  # 축소
        'reg_alpha': 1.0,  # 축소
        'reg_lambda': 1.0,  # 축소
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 1000,  # 축소
        'early_stopping_rounds': 100,
        'max_bin': 255,  # 축소
        'nthread': NUM_WORKERS,
        'grow_policy': 'depthwise',
        'max_leaves': 255,  # 축소
        'gamma': 0.1
    }
    
    # CatBoost 파라미터 (보수적 설정)
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 8,  # 축소
        'learning_rate': 0.05,  # 증가
        'l2_leaf_reg': 10,  # 축소
        'iterations': 1000,  # 축소
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 2,  # 축소
        'thread_count': NUM_WORKERS,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'od_type': 'IncToDec',
        'od_wait': 100,
        'leaf_estimation_iterations': 10,  # 축소
        'grow_policy': 'SymmetricTree',
        'max_leaves': 255,  # 축소
        'min_data_in_leaf': 50,  # 축소
        'rsm': 0.8
    }
    
    # 딥러닝 모델 파라미터 (보수적 설정)
    NN_PARAMS = {
        'hidden_dims': [512, 256, 128, 64],  # 축소
        'dropout_rate': 0.3,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.001,  # 증가
        'weight_decay': 1e-4,  # 증가
        'epochs': 50,  # 축소
        'patience': 10,  # 축소
        'use_batch_norm': True,
        'activation': 'relu',
        'use_residual': False,  # 비활성화
        'use_attention': False,  # 비활성화
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'label_smoothing': 0.01,  # 축소
        'gradient_clip_val': 1.0,
        'scheduler_type': 'step',
        'min_lr': 1e-6
    }
    
    # 피처 엔지니어링 설정 (보수적 설정)
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 100,  # 축소
        'frequency_threshold': 50,  # 축소
        'interaction_features': False,  # 비활성화
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,
        'id_hash_features': True,
        'polynomial_features': False,  # 비활성화
        'binning_features': True,
        'quantile_features': False,  # 비활성화
        'rank_features': False,  # 비활성화
        'group_statistics': True,
        'lag_features': False,  # 비활성화
        'rolling_features': False,  # 비활성화
        'fourier_features': False,
        'pca_features': False,
        'clustering_features': False,  # 비활성화
        'text_features': False,
        'max_features': 500,  # 축소
        'feature_selection_method': 'mutual_info',
        'feature_selection_k': 200,  # 축소
        'memory_efficient_mode_threshold': 1000000,  # 100만행으로 축소
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
        'bootstrap_samples': 500,  # 축소
        'confidence_interval': 0.95,
        'stability_threshold': 0.015,
        'performance_metrics': ['ap', 'wll', 'auc', 'f1'],
        'ctr_tolerance': 0.0005,
        'bias_penalty_weight': 2.5,
        'calibration_weight': 0.4,
        'large_data_evaluation': False,  # 비활성화
        'evaluation_sample_size': 100000  # 축소
    }
    
    # 앙상블 설정 (단순화)
    ENSEMBLE_CONFIG = {
        'use_optimal_ensemble': True,
        'use_stabilized_ensemble': False,  # 비활성화
        'use_meta_learning': False,  # 비활성화
        'use_stacking': False,  # 비활성화
        'meta_model': 'ridge',
        'calibration_ensemble': True,
        'optimization_method': 'simple',
        'diversification_method': 'equal',
        'ensemble_types': ['optimal', 'calibrated'],
        'blend_weights': {
            'lgbm': 0.35,
            'xgb': 0.35,
            'cat': 0.30
        },
        'ensemble_optimization_trials': 50,  # 축소
        'ensemble_cv_folds': 3,  # 축소
        'ensemble_early_stopping': 25,  # 축소
        'ensemble_regularization': 0.01,
        'dynamic_weighting': False,  # 비활성화
        'adaptive_blending': False,  # 비활성화
        'temporal_ensemble': False,  # 비활성화
        'multi_level_ensemble': False,  # 비활성화
        'large_data_ensemble': False,  # 비활성화
        'ensemble_memory_limit': 4.0  # 축소
    }
    
    # 로깅 설정
    LOGGING_CONFIG = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True,
        'detailed_logging': False,  # 비활성화
        'performance_logging': False,  # 비활성화
        'memory_logging': True,
        'model_logging': True,
        'data_processing_logging': True
    }
    
    # 실시간 추론 설정
    INFERENCE_CONFIG = {
        'batch_size': 1000,  # 축소
        'single_prediction_batch_size': 1,
        'timeout': 100,
        'cache_size': 10000,  # 축소
        'model_version': 'v1.0',
        'use_gpu': GPU_AVAILABLE,
        'parallel_inference': False,  # 비활성화
        'inference_optimization': True,
        'model_compilation': False,  # 비활성화
        'quantization': False,
        'tensorrt_optimization': False,
        'onnx_optimization': False,
        'async_inference': False,  # 비활성화
        'memory_pool': True,
        'connection_pool_size': 10,  # 축소
        'warm_up_requests': 5,  # 축소
        'feature_caching': True,
        'prediction_caching': False,  # 비활성화
        'real_time_optimization': False,  # 비활성화
        'latency_target_ms': 100,
        'throughput_target_qps': 100,  # 축소
        'model_preloading': True,
        'feature_preprocessing_cache': False,  # 비활성화
        'thread_pool_size': NUM_WORKERS,
        'response_compression': False,
        'monitoring_enabled': False,  # 비활성화
        'performance_logging': False,
        'rule_compliance_check': True,
        'independent_execution': True,
        'no_external_api': True,
        'local_only': True,
        'utf8_encoding': True,
        'relative_path_only': True
    }
    
    # 하이퍼파라미터 튜닝 설정 (축소)
    TUNING_CONFIG = {
        'n_trials': 20,  # 축소
        'timeout': 3600,  # 1시간으로 축소
        'parallel_jobs': 1,
        'pruner': 'MedianPruner',
        'sampler': 'TPESampler',
        'optimization_direction': 'maximize',
        'study_storage': None,
        'load_if_exists': True,
        'enable_pruning': True,
        'n_startup_trials': 5,  # 축소
        'n_warmup_steps': 3,  # 축소
        'interval_steps': 1,
        'percentile': 50.0,
        'min_resource': 1,
        'max_resource': 81,  # 축소
        'reduction_factor': 3,
        'bootstrap_count': 50,  # 축소
        'multi_objective': False,
        'large_data_tuning': False  # 비활성화
    }
    
    # 메모리 관리 설정 (보수적 조정)
    MEMORY_CONFIG = {
        'max_memory_gb': MAX_MEMORY_GB,
        'auto_gc': True,
        'gc_threshold': 0.6,  # 더 적극적 가비지 컬렉션
        'force_gc_interval': 100,  # 더 빈번한 가비지 컬렉션
        'memory_monitoring': True,
        'memory_limit_warning': 0.7,  # 70% 경고
        'memory_limit_error': 0.85,  # 85% 에러
        'chunk_memory_limit': 4.0,  # 청킹당 메모리 한계 축소
        'batch_memory_limit': 2.0,  # 축소
        'model_memory_limit': 6.0,  # 축소
        'ensemble_memory_limit': 8.0,  # 축소
        'swap_usage_limit': 0.05,
        'memory_profiling': False,  # 비활성화
        'memory_optimization': True,
        'lazy_loading': True,
        'memory_mapping': False,  # 비활성화
        'compressed_storage': True,
        'aggressive_memory_management': True,
        'large_data_memory_strategy': False  # 비활성화
    }
    
    # GPU 설정
    GPU_CONFIG = {
        'gpu_memory_fraction': 0.7,  # GPU 메모리 사용률 축소
        'allow_growth': True,
        'mixed_precision': USE_MIXED_PRECISION,
        'tensor_core_optimization': False,  # 비활성화
        'cuda_optimization_level': GPU_OPTIMIZATION_LEVEL,
        'cuda_cache_config': 'PreferShared',
        'cudnn_benchmark': True,
        'cudnn_deterministic': False,
        'cuda_launch_blocking': False,
        'gpu_memory_monitoring': True,
        'gpu_utilization_monitoring': False,  # 비활성화
        'multi_gpu': False,
        'gpu_sync_interval': 200,
        'gpu_memory_pool': False,  # 비활성화
        'gpu_kernel_sync': False,
        'gpu_profiling': False,
        'tensor_parallelism': False,
        'pipeline_parallelism': False,
        'gradient_checkpointing': False,  # 비활성화
        'activation_checkpointing': False  # 비활성화
    }
    
    # 병렬 처리 설정 (축소)
    PARALLEL_CONFIG = {
        'num_workers': NUM_WORKERS,
        'max_workers': NUM_WORKERS,
        'thread_pool_size': NUM_WORKERS * 2,  # 축소
        'process_pool_size': NUM_WORKERS,
        'multiprocessing_context': 'spawn',
        'shared_memory': False,  # 비활성화
        'parallel_backend': 'threading',
        'parallel_feature_engineering': False,  # 비활성화
        'parallel_model_training': False,  # 비활성화
        'parallel_inference': False,  # 비활성화
        'parallel_evaluation': False,  # 비활성화
        'thread_local_storage': True,
        'numa_optimization': False,
        'cpu_affinity': False,  # 비활성화
        'priority_scheduling': False,  # 비활성화
        'load_balancing': False,  # 비활성화
        'work_stealing': False,  # 비활성화
        'dynamic_scheduling': False,  # 비활성화
        'parallel_io': False,  # 비활성화
        'async_processing': False  # 비활성화
    }
    
    # 데이터 처리 설정 (보수적 조정)
    DATA_CONFIG = {
        'use_pyarrow': True,
        'compression': 'snappy',
        'memory_map': False,  # 비활성화
        'lazy_loading': True,
        'batch_processing': True,
        'streaming_processing': False,  # 비활성화
        'data_validation': True,
        'schema_validation': False,  # 비활성화
        'type_optimization': True,
        'categorical_optimization': True,
        'string_optimization': True,
        'datetime_optimization': False,  # 비활성화
        'numeric_optimization': True,
        'memory_efficient_dtypes': True,
        'sparse_arrays': False,
        'columnar_storage': True,
        'indexed_access': False,  # 비활성화
        'cached_operations': False,  # 비활성화
        'vectorized_operations': True,
        'broadcast_operations': False,  # 비활성화
        'parallel_reading': False,  # 비활성화
        'async_io': False,  # 비활성화
        'prefetch_batches': False,  # 비활성화
        'chunk_size': CHUNK_SIZE,
        'max_memory_usage': MAX_MEMORY_GB,
        'large_data_optimization': False  # 비활성화
    }
    
    # 모델 저장/로딩 설정
    MODEL_IO_CONFIG = {
        'compression_level': 3,  # 압축 레벨 축소
        'pickle_protocol': 4,  # 프로토콜 버전 축소
        'joblib_compression': 'zlib',
        'model_versioning': True,
        'incremental_saving': False,  # 비활성화
        'checkpoint_frequency': 100,  # 체크포인트 빈도 축소
        'backup_models': False,  # 비활성화
        'model_metadata': True,
        'model_signature': False,  # 비활성화
        'model_validation': True,
        'lazy_model_loading': False,  # 비활성화
        'model_caching': False,  # 비활성화
        'model_pooling': False,  # 비활성화
        'distributed_storage': False,
        'cloud_storage': False,
        'local_storage_optimization': True,
        'model_compression': True,
        'large_model_handling': False  # 비활성화
    }
    
    @classmethod
    def verify_data_requirements(cls):
        """데이터 요구사항 검증"""
        print("=== 데이터 요구사항 검증 ===")
        
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
        
        # 최소 파일 크기 검증
        min_train_size_mb = 100  # 최소 100MB로 축소
        min_test_size_mb = 50   # 최소 50MB로 축소
        
        requirements['train_size_adequate'] = requirements['train_file_size_mb'] >= min_train_size_mb
        requirements['test_size_adequate'] = requirements['test_file_size_mb'] >= min_test_size_mb
        requirements['memory_adequate'] = requirements['memory_available'] >= 20  # 20GB로 축소
        
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
        """메모리 설정 반환"""
        try:
            import psutil
            
            total_memory = psutil.virtual_memory().total / (1024**3)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            # 사용 가능 메모리의 70%로 제한
            max_memory = min(cls.MAX_MEMORY_GB, available_memory * 0.7)
            
            return {
                'max_memory_gb': max_memory,
                'chunk_size': cls.CHUNK_SIZE,
                'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'prefetch_factor': cls.PREFETCH_FACTOR,
                'num_workers': cls.NUM_WORKERS,
                'memory_monitoring': cls.MEMORY_CONFIG['memory_monitoring'],
                'auto_gc': cls.MEMORY_CONFIG['auto_gc'],
                'gc_threshold': cls.MEMORY_CONFIG['gc_threshold']
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
                'gc_threshold': 0.6
            }
    
    @classmethod
    def get_data_config(cls):
        """데이터 처리 설정 반환"""
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
            'type_optimization': cls.DATA_CONFIG['type_optimization']
        }
    
    @classmethod
    def get_safe_memory_limits(cls):
        """안전한 메모리 한계 설정"""
        try:
            import psutil
            
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            
            # 보수적 메모리 사용
            safe_limit = min(cls.MAX_MEMORY_GB, available_gb * 0.6)
            
            return {
                'total_memory_gb': total_gb,
                'available_memory_gb': available_gb,
                'safe_memory_limit_gb': safe_limit,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'memory_monitoring_enabled': cls.MEMORY_CONFIG['memory_monitoring'],
                'auto_gc_enabled': cls.MEMORY_CONFIG['auto_gc'],
                'memory_limits': {
                    'chunk': cls.MEMORY_CONFIG['chunk_memory_limit'],
                    'batch': cls.MEMORY_CONFIG['batch_memory_limit'],
                    'model': cls.MEMORY_CONFIG['model_memory_limit'],
                    'ensemble': cls.MEMORY_CONFIG['ensemble_memory_limit']
                }
            }
        except ImportError:
            return {
                'total_memory_gb': 64.0,
                'available_memory_gb': 30.0,
                'safe_memory_limit_gb': cls.MAX_MEMORY_GB,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'memory_monitoring_enabled': True,
                'auto_gc_enabled': True,
                'memory_limits': {
                    'chunk': 4.0,
                    'batch': 2.0,
                    'model': 6.0,
                    'ensemble': 8.0
                }
            }

    @classmethod
    def setup_gpu_environment(cls):
        """GPU 환경 설정"""
        if not cls.GPU_AVAILABLE:
            return False
            
        try:
            if TORCH_AVAILABLE:
                import torch
                
                # CUDA 설정
                torch.backends.cudnn.benchmark = cls.GPU_CONFIG['cudnn_benchmark']
                torch.backends.cudnn.deterministic = cls.GPU_CONFIG['cudnn_deterministic']
                
                # Mixed Precision 설정
                if cls.GPU_CONFIG['mixed_precision']:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                print(f"GPU 환경 설정 완료: {torch.cuda.get_device_name(0)}")
                return True
                
        except Exception as e:
            print(f"GPU 환경 설정 실패: {e}")
            
        return False

# 환경변수 설정
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['MKL_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMEXPR_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMBA_NUM_THREADS'] = str(Config.NUM_WORKERS)
    
    # 데이터 처리용 환경변수
    os.environ['PANDAS_MAX_COLUMNS'] = '1000'  # 축소
    os.environ['PANDAS_MAX_ROWS'] = '5000000'  # 축소
    
    # CUDA 환경 변수
    if Config.GPU_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_VISIBLE_DEVICES
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 축소
    
    # 메모리 관리 환경 변수
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
    
    # 규칙 준수 환경 변수
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'
    
except Exception as e:
    print(f"환경 변수 설정 실패: {e}")

# 시작 시 검증
try:
    Config.setup_directories()
    Config.verify_paths()
    Config.verify_data_requirements()
except Exception as e:
    print(f"초기화 실패: {e}")