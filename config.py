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
    """프로젝트 전체 설정 관리 - 대용량 데이터 처리 최적화"""
    
    # 기본 경로 설정
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # 데이터 파일 경로 - 실제 대용량 파일 우선 처리
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
    SUBMISSION_TEMPLATE_PATH = DATA_DIR / "sample_submission.csv"
    
    # GPU 및 하드웨어 설정 (RTX 4060 Ti 16GB 최적화)
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        GPU_AVAILABLE = torch.cuda.is_available()
    else:
        DEVICE = 'cpu'
        GPU_AVAILABLE = False
    
    GPU_MEMORY_LIMIT = 14  # RTX 4060 Ti 16GB 중 14GB 사용
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    GPU_OPTIMIZATION_LEVEL = 2  # 0: 기본, 1: 보통, 2: 적극적
    
    # 64GB RAM 환경 대용량 데이터 최적화 설정
    MAX_MEMORY_GB = 48  # 64GB 중 48GB 사용 (20GB 여유분 확보)
    CHUNK_SIZE = 1000000  # 청킹 크기 100만행으로 대폭 확대
    BATCH_SIZE_GPU = 15000  # GPU 배치 크기 확대
    BATCH_SIZE_CPU = 5000  # CPU 배치 크기 확대
    PREFETCH_FACTOR = 4
    NUM_WORKERS = 6  # AMD Ryzen 5 5600X 6코어 활용
    
    # 실제 대용량 데이터 처리 전략 (1070만행 처리 보장)
    TARGET_DATA_USAGE_RATIO = 1.0  # 전체 데이터 100% 사용 필수
    MIN_TRAIN_SIZE = 10000000  # 최소 1000만행 학습 데이터
    MAX_TRAIN_SIZE = 15000000  # 최대 1500만행 학습 데이터
    MIN_TEST_SIZE = 1527298  # 전체 테스트 데이터 필수
    MAX_TEST_SIZE = 2000000  # 테스트 데이터 최대 크기
    FORCE_FULL_TEST_PREDICTION = True  # 전체 테스트 데이터 예측 강제
    FORCE_FULL_DATA_PROCESSING = True  # 전체 데이터 처리 강제
    
    # 대용량 데이터 검증 설정
    EXPECTED_TRAIN_SIZE = 10000000  # 예상 학습 데이터 크기
    EXPECTED_TEST_SIZE = 1527298  # 예상 테스트 데이터 크기
    DATA_SIZE_TOLERANCE = 0.1  # 데이터 크기 허용 오차 10%
    REQUIRE_REAL_DATA = True  # 실제 데이터 파일 필수
    SAMPLE_DATA_FALLBACK = False  # 샘플 데이터 대체 비활성화
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 3  # CV 폴드 수
    TEST_SIZE = 0.2
    
    # LightGBM CTR 파라미터 (대용량 데이터 최적화)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 1023,  # 대용량 데이터용 증가
        'learning_rate': 0.015,  # 학습률 감소로 안정성 확보
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 500,  # 대용량 데이터용 증가
        'min_child_weight': 40,
        'lambda_l1': 3.0,
        'lambda_l2': 3.0,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 5000,  # 대용량 데이터용 증가
        'early_stopping_rounds': 300,
        'scale_pos_weight': 49.0,
        'force_row_wise': True,
        'max_bin': 511,  # 대용량 데이터용 증가
        'num_threads': NUM_WORKERS,
        'device_type': 'cpu',
        'min_data_in_leaf': 200,
        'max_depth': 18,  # 대용량 데이터용 증가
        'feature_fraction_bynode': 0.8,
        'extra_trees': True,
        'path_smooth': 2.0,
        'grow_policy': 'lossguide',
        'max_leaves': 1023,
        'min_gain_to_split': 0.01,
        'min_sum_hessian_in_leaf': 20.0,
        'feature_pre_filter': False,
        'linear_tree': False,
        'cat_smooth': 15.0,
        'cat_l2': 15.0
    }
    
    # XGBoost CTR 파라미터 (대용량 데이터 최적화)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 15,  # 대용량 데이터용 증가
        'learning_rate': 0.015,  # 학습률 감소
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 40,  # 대용량 데이터용 증가
        'reg_alpha': 3.0,
        'reg_lambda': 3.0,
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 5000,  # 대용량 데이터용 증가
        'early_stopping_rounds': 300,
        'max_bin': 511,
        'nthread': NUM_WORKERS,
        'grow_policy': 'lossguide',
        'max_leaves': 1023,
        'gamma': 0.1,
        'max_delta_step': 1,
        'monotone_constraints': None,
        'interaction_constraints': None,
        'validate_parameters': True,
        'predictor': 'gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
        'single_precision_histogram': False
    }
    
    # CatBoost CTR 파라미터 (대용량 데이터 최적화)
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 12,  # 대용량 데이터용 증가
        'learning_rate': 0.015,  # 학습률 감소
        'l2_leaf_reg': 20,
        'iterations': 5000,  # 대용량 데이터용 증가
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 4,  # 대용량 데이터용 증가
        'thread_count': NUM_WORKERS,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'od_type': 'IncToDec',
        'od_wait': 300,
        'leaf_estimation_iterations': 20,
        'leaf_estimation_method': 'Newton',
        'grow_policy': 'Lossguide',
        'max_leaves': 1023,
        'min_data_in_leaf': 200,
        'rsm': 0.8,
        'sampling_frequency': 'PerTreeLevel',
        'leaf_estimation_backtracking': 'AnyImprovement',
        'has_time': False,
        'allow_const_label': False,
        'score_function': 'Cosine'
    }
    
    # 딥러닝 모델 파라미터 (대용량 데이터 최적화)
    NN_PARAMS = {
        'hidden_dims': [1536, 768, 384, 192, 96],  # 대용량 데이터용 확대
        'dropout_rate': 0.3,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.0005,  # 학습률 감소
        'weight_decay': 3e-5,
        'epochs': 150,  # 대용량 데이터용 증가
        'patience': 25,
        'use_batch_norm': True,
        'activation': 'gelu',
        'use_residual': True,
        'use_attention': True,
        'attention_heads': 12,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0,
        'label_smoothing': 0.05,
        'gradient_clip_val': 1.0,
        'scheduler_type': 'cosine',
        'warmup_epochs': 15,
        'min_lr': 5e-7
    }
    
    # 피처 엔지니어링 설정 (대용량 데이터 최적화)
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 500,  # 대용량 데이터용 증가
        'frequency_threshold': 300,
        'interaction_features': True,
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,
        'id_hash_features': True,
        'polynomial_features': True,
        'max_polynomial_degree': 2,
        'binning_features': True,
        'quantile_features': True,
        'rank_features': True,
        'group_statistics': True,
        'lag_features': True,
        'rolling_features': True,
        'fourier_features': False,
        'pca_features': False,
        'clustering_features': True,
        'text_features': False,
        'max_features': 3000,  # 대용량 데이터용 증가
        'feature_selection_method': 'mutual_info',
        'feature_selection_k': 2000,
        'memory_efficient_mode_threshold': 5000000,  # 500만행 이상에서 메모리 효율 모드
        'chunked_feature_engineering': True,
        'feature_importance_threshold': 0.001
    }
    
    # 평가 설정 (대용량 데이터 최적화)
    EVALUATION_CONFIG = {
        'ap_weight': 0.5,
        'wll_weight': 0.5,
        'actual_ctr': 0.0201,
        'pos_weight': 0.0201,
        'neg_weight': 0.9799,
        'target_score': 0.36000,
        'bootstrap_samples': 2000,  # 대용량 데이터용 증가
        'confidence_interval': 0.95,
        'stability_threshold': 0.015,
        'performance_metrics': ['ap', 'wll', 'auc', 'f1', 'precision', 'recall'],
        'ctr_tolerance': 0.0005,
        'bias_penalty_weight': 2.5,
        'calibration_weight': 0.4,
        'large_data_evaluation': True,
        'evaluation_sample_size': 1000000  # 평가용 샘플 크기 증가
    }
    
    # 앙상블 설정 (대용량 데이터 최적화)
    ENSEMBLE_CONFIG = {
        'use_optimal_ensemble': True,
        'use_stabilized_ensemble': True,
        'use_meta_learning': True,
        'use_stacking': True,
        'meta_model': 'ridge',
        'calibration_ensemble': True,
        'optimization_method': 'combined',
        'diversification_method': 'rank_weighted',
        'meta_model_type': 'ridge',
        'use_meta_features': True,
        'ensemble_types': ['optimal', 'stabilized', 'meta', 'calibrated', 'weighted'],
        'blend_weights': {
            'lgbm': 0.30,
            'xgb': 0.25,
            'cat': 0.25,
            'deepctr': 0.20
        },
        'ensemble_optimization_trials': 300,  # 대용량 데이터용 증가
        'ensemble_cv_folds': 5,
        'ensemble_early_stopping': 75,
        'ensemble_regularization': 0.005,
        'dynamic_weighting': True,
        'adaptive_blending': True,
        'temporal_ensemble': True,
        'multi_level_ensemble': True,
        'large_data_ensemble': True,
        'ensemble_memory_limit': 8.0  # GB
    }
    
    # 로깅 설정
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
    
    # 실시간 추론 설정 (대용량 데이터 최적화)
    INFERENCE_CONFIG = {
        'batch_size': 50000,  # 대용량 배치 처리
        'timeout': 900,
        'cache_size': 200000,
        'model_version': 'v4.0',
        'use_gpu': GPU_AVAILABLE,
        'parallel_inference': True,
        'inference_optimization': True,
        'model_compilation': True,
        'quantization': False,
        'tensorrt_optimization': False,
        'onnx_optimization': False,
        'async_inference': True,
        'memory_pool': True,
        'connection_pool_size': 200
    }
    
    # 하이퍼파라미터 튜닝 설정 (대용량 데이터 최적화)
    TUNING_CONFIG = {
        'n_trials': 75,  # 대용량 데이터용 증가
        'timeout': 14400,  # 4시간
        'parallel_jobs': 1,
        'pruner': 'MedianPruner',
        'sampler': 'TPESampler',
        'optimization_direction': 'maximize',
        'study_storage': None,
        'load_if_exists': True,
        'enable_pruning': True,
        'n_startup_trials': 15,
        'n_warmup_steps': 8,
        'interval_steps': 1,
        'percentile': 40.0,
        'min_resource': 1,
        'max_resource': 243,
        'reduction_factor': 3,
        'bootstrap_count': 200,
        'multi_objective': False,
        'large_data_tuning': True
    }
    
    # 메모리 관리 설정 (48GB 환경 최적화)
    MEMORY_CONFIG = {
        'max_memory_gb': MAX_MEMORY_GB,
        'auto_gc': True,
        'gc_threshold': 0.75,  # 더 적극적 가비지 컬렉션
        'force_gc_interval': 500,  # 더 빈번한 가비지 컬렉션
        'memory_monitoring': True,
        'memory_limit_warning': 0.85,
        'memory_limit_error': 0.92,
        'chunk_memory_limit': 8.0,  # 청킹당 메모리 한계 증가
        'batch_memory_limit': 4.0,
        'model_memory_limit': 12.0,
        'ensemble_memory_limit': 16.0,
        'swap_usage_limit': 0.05,
        'memory_profiling': True,
        'memory_optimization': True,
        'lazy_loading': True,
        'memory_mapping': True,
        'compressed_storage': True,  # 압축 저장 활성화
        'aggressive_memory_management': True,
        'large_data_memory_strategy': True
    }
    
    # GPU 설정 (RTX 4060 Ti 16GB 최적화)
    GPU_CONFIG = {
        'gpu_memory_fraction': 0.9,  # GPU 메모리 사용률 증가
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
        'activation_checkpointing': True  # 대용량 데이터용 활성화
    }
    
    # 병렬 처리 설정 (6코어 최적화)
    PARALLEL_CONFIG = {
        'num_workers': NUM_WORKERS,
        'max_workers': NUM_WORKERS,
        'thread_pool_size': NUM_WORKERS * 3,  # 쓰레드 풀 확대
        'process_pool_size': NUM_WORKERS,
        'multiprocessing_context': 'spawn',
        'shared_memory': True,
        'parallel_backend': 'threading',
        'parallel_feature_engineering': True,
        'parallel_model_training': True,  # 병렬 학습 활성화
        'parallel_inference': True,
        'parallel_evaluation': True,
        'thread_local_storage': True,
        'numa_optimization': False,
        'cpu_affinity': True,
        'priority_scheduling': True,
        'load_balancing': True,
        'work_stealing': True,
        'dynamic_scheduling': True,
        'parallel_io': True,  # 병렬 I/O 활성화
        'async_processing': True
    }
    
    # 데이터 처리 설정 (대용량 최적화)
    DATA_CONFIG = {
        'use_pyarrow': True,
        'compression': 'snappy',
        'memory_map': True,
        'lazy_loading': True,
        'batch_processing': True,
        'streaming_processing': True,  # 스트리밍 처리 활성화
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
        'parallel_reading': True,  # 병렬 읽기 활성화
        'async_io': True,
        'prefetch_batches': True,
        'chunk_size': CHUNK_SIZE,
        'max_memory_usage': MAX_MEMORY_GB,
        'large_data_optimization': True
    }
    
    # 모델 저장/로딩 설정
    MODEL_IO_CONFIG = {
        'compression_level': 6,  # 압축 레벨 조정
        'pickle_protocol': 5,
        'joblib_compression': 'lz4',
        'model_versioning': True,
        'incremental_saving': True,
        'checkpoint_frequency': 500,  # 체크포인트 빈도 증가
        'backup_models': True,
        'model_metadata': True,
        'model_signature': True,
        'model_validation': True,
        'lazy_model_loading': True,
        'model_caching': True,
        'model_pooling': True,  # 모델 풀링 활성화
        'distributed_storage': False,
        'cloud_storage': False,
        'local_storage_optimization': True,
        'model_compression': True,
        'large_model_handling': True
    }
    
    @classmethod
    def verify_data_requirements(cls):
        """대용량 데이터 요구사항 검증"""
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
        
        # 최소 파일 크기 검증 (1070만행 기준)
        min_train_size_mb = 800  # 최소 800MB
        min_test_size_mb = 120   # 최소 120MB
        
        requirements['train_size_adequate'] = requirements['train_file_size_mb'] >= min_train_size_mb
        requirements['test_size_adequate'] = requirements['test_file_size_mb'] >= min_test_size_mb
        requirements['memory_adequate'] = requirements['memory_available'] >= 40
        
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
        """필요한 디렉터리 생성 - 대용량 데이터 처리 강화"""
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
        """경로 유효성 검증 - 대용량 데이터 파일 포함"""
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
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(cls.LOGGING_CONFIG['level'])
        
        # 기존 핸들러 모두 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 로그 포맷 설정
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        # 콘솔 핸들러 설정
        if cls.LOGGING_CONFIG['console_handler']:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 파일 핸들러 설정
        if cls.LOGGING_CONFIG['file_handler']:
            try:
                log_file_path = cls.LOG_DIR / 'ctr_model.log'
                file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='w')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                print(f"로그 파일 생성 완료: {log_file_path}")
            except Exception as e:
                print(f"파일 핸들러 설정 실패: {e}")
        
        # 프로젝트 전용 로거 생성
        project_logger = logging.getLogger('ctr_modeling')
        project_logger.setLevel(logging.INFO)
        
        return project_logger
    
    @classmethod
    def get_memory_config(cls):
        """메모리 설정 반환 - 대용량 데이터 최적화"""
        try:
            import psutil
            
            total_memory = psutil.virtual_memory().total / (1024**3)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            # 48GB 사용 가능 환경 기준
            max_memory = min(cls.MAX_MEMORY_GB, available_memory - 12)
            
            return {
                'max_memory_gb': max_memory,
                'chunk_size': cls.CHUNK_SIZE,
                'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'prefetch_factor': cls.PREFETCH_FACTOR,
                'num_workers': cls.NUM_WORKERS,
                'memory_monitoring': cls.MEMORY_CONFIG['memory_monitoring'],
                'auto_gc': cls.MEMORY_CONFIG['auto_gc'],
                'gc_threshold': cls.MEMORY_CONFIG['gc_threshold'],
                'large_data_strategy': cls.MEMORY_CONFIG['large_data_memory_strategy'],
                'aggressive_management': cls.MEMORY_CONFIG['aggressive_memory_management']
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
                'gc_threshold': 0.75,
                'large_data_strategy': True,
                'aggressive_management': True
            }
    
    @classmethod
    def get_data_config(cls):
        """데이터 처리 설정 반환 - 대용량 데이터 처리 보장"""
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
            'memory_map': cls.DATA_CONFIG['memory_map'],
            'lazy_loading': cls.DATA_CONFIG['lazy_loading'],
            'batch_processing': cls.DATA_CONFIG['batch_processing'],
            'streaming_processing': cls.DATA_CONFIG['streaming_processing'],
            'parallel_reading': cls.DATA_CONFIG['parallel_reading'],
            'async_io': cls.DATA_CONFIG['async_io'],
            'type_optimization': cls.DATA_CONFIG['type_optimization'],
            'large_data_optimization': cls.DATA_CONFIG['large_data_optimization']
        }
    
    @classmethod
    def get_safe_memory_limits(cls):
        """안전한 메모리 한계 설정 - 48GB 환경 최적화"""
        try:
            import psutil
            
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            
            # 48GB 사용 기준 설정
            safe_limit = min(cls.MAX_MEMORY_GB, available_gb - 12)
            
            return {
                'total_memory_gb': total_gb,
                'available_memory_gb': available_gb,
                'safe_memory_limit_gb': safe_limit,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'memory_monitoring_enabled': cls.MEMORY_CONFIG['memory_monitoring'],
                'auto_gc_enabled': cls.MEMORY_CONFIG['auto_gc'],
                'aggressive_management_enabled': cls.MEMORY_CONFIG['aggressive_memory_management'],
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
                'available_memory_gb': 48.0,
                'safe_memory_limit_gb': cls.MAX_MEMORY_GB,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'memory_monitoring_enabled': True,
                'auto_gc_enabled': True,
                'aggressive_management_enabled': True,
                'memory_limits': {
                    'chunk': 8.0,
                    'batch': 4.0,
                    'model': 12.0,
                    'ensemble': 16.0
                }
            }

# 환경변수 설정 - 대용량 데이터 처리 최적화
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['MKL_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMEXPR_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMBA_NUM_THREADS'] = str(Config.NUM_WORKERS)
    
    # 대용량 데이터 처리용 환경변수
    os.environ['PANDAS_MAX_COLUMNS'] = '5000'
    os.environ['PANDAS_MAX_ROWS'] = '20000000'
    
    # CUDA 환경 변수
    if Config.GPU_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_VISIBLE_DEVICES
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
    
    # 메모리 관리 환경 변수
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
    
except Exception as e:
    print(f"환경 변수 설정 실패: {e}")

# 시작 시 검증
try:
    Config.setup_directories()
    Config.verify_paths()
    Config.verify_data_requirements()
except Exception as e:
    print(f"초기화 실패: {e}")