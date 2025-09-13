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
    TRAIN_PATH = "./train.parquet"
    TEST_PATH = "./test.parquet"
    SUBMISSION_PATH = "./sample_submission.csv"
    
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
    
    # 64GB RAM 환경 최적화 설정
    MAX_MEMORY_GB = 48  # 64GB 중 48GB 사용 가능 (여유분 확보)
    CHUNK_SIZE = 600000  # 청킹 크기 확대
    BATCH_SIZE_GPU = 10240  # GPU 배치 크기 확대
    BATCH_SIZE_CPU = 3072  # CPU 배치 크기 확대
    PREFETCH_FACTOR = 3
    NUM_WORKERS = 6  # AMD Ryzen 5 5600X 6코어 활용
    
    # 전체 데이터 처리 전략 (성능 문제 해결)
    TARGET_DATA_USAGE_RATIO = 1.0  # 전체 데이터 100% 사용
    MIN_TRAIN_SIZE = 1000000  # 최소 학습 데이터 크기 확대
    MAX_TRAIN_SIZE = 2500000  # 최대 학습 데이터 크기 확대
    MIN_TEST_SIZE = 1527298  # 전체 테스트 데이터 필수
    MAX_TEST_SIZE = 1527298  # 전체 테스트 데이터 필수
    FORCE_FULL_TEST_PREDICTION = True  # 전체 테스트 데이터 예측 강제
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 3  # CV 폴드 수
    TEST_SIZE = 0.2
    
    # LightGBM CTR 파라미터 (성능 최적화)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 511,
        'learning_rate': 0.02,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.65,
        'bagging_freq': 5,
        'min_child_samples': 250,
        'min_child_weight': 20,
        'lambda_l1': 2.5,
        'lambda_l2': 2.5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 4000,
        'early_stopping_rounds': 200,
        'scale_pos_weight': 49.0,
        'force_row_wise': True,
        'max_bin': 255,
        'num_threads': NUM_WORKERS,
        'device_type': 'cpu',
        'min_data_in_leaf': 120,
        'max_depth': 15,
        'feature_fraction_bynode': 0.75,
        'extra_trees': True,
        'path_smooth': 1.5,
        'grow_policy': 'lossguide',
        'max_leaves': 511,
        'min_gain_to_split': 0.02,
        'min_sum_hessian_in_leaf': 15.0,
        'feature_pre_filter': False,
        'linear_tree': False,
        'cat_smooth': 10.0,
        'cat_l2': 10.0
    }
    
    # XGBoost CTR 파라미터 (성능 최적화)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 12,
        'learning_rate': 0.02,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'colsample_bylevel': 0.75,
        'colsample_bynode': 0.75,
        'min_child_weight': 20,
        'reg_alpha': 2.5,
        'reg_lambda': 2.5,
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 4000,
        'early_stopping_rounds': 200,
        'max_bin': 255,
        'nthread': NUM_WORKERS,
        'grow_policy': 'lossguide',
        'max_leaves': 511,
        'gamma': 0.15,
        'max_delta_step': 1,
        'monotone_constraints': None,
        'interaction_constraints': None,
        'validate_parameters': True,
        'predictor': 'gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
        'single_precision_histogram': False
    }
    
    # CatBoost CTR 파라미터 (충돌 해결)
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 10,
        'learning_rate': 0.02,
        'l2_leaf_reg': 15,
        'iterations': 4000,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 3,
        'thread_count': NUM_WORKERS,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.2,
        'od_type': 'IncToDec',
        'od_wait': 200,
        'leaf_estimation_iterations': 15,
        'leaf_estimation_method': 'Newton',
        'grow_policy': 'Lossguide',
        'max_leaves': 511,
        'min_data_in_leaf': 120,
        'rsm': 0.75,
        'sampling_frequency': 'PerTreeLevel',
        'leaf_estimation_backtracking': 'AnyImprovement',
        'has_time': False,
        'allow_const_label': False,
        'score_function': 'Cosine'
    }
    
    # 딥러닝 모델 파라미터 (GPU 최적화)
    NN_PARAMS = {
        'hidden_dims': [1024, 512, 256, 128, 64],
        'dropout_rate': 0.35,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.0008,
        'weight_decay': 2e-5,
        'epochs': 100,
        'patience': 20,
        'use_batch_norm': True,
        'activation': 'gelu',
        'use_residual': True,
        'use_attention': True,
        'attention_heads': 8,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.5,
        'label_smoothing': 0.1,
        'gradient_clip_val': 1.0,
        'scheduler_type': 'cosine',
        'warmup_epochs': 10,
        'min_lr': 1e-6
    }
    
    # Calibration 설정 (강화)
    CALIBRATION_CONFIG = {
        'target_ctr': 0.0201,
        'platt_scaling': True,
        'isotonic_regression': True,
        'temperature_scaling': True,
        'cv_folds': 5,
        'calibration_sample_size': 100000,
        'bias_correction': True,
        'distribution_matching': True,
        'quantile_calibration': True,
        'ensemble_calibration': True,
        'calibration_methods': ['platt', 'isotonic', 'temperature', 'beta'],
        'cross_validation_calibration': True,
        'calibration_regularization': 0.01
    }
    
    # 피처 엔지니어링 설정 (강화)
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 300,
        'frequency_threshold': 150,
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
        'max_features': 2000,
        'feature_selection_method': 'mutual_info',
        'feature_selection_k': 1500
    }
    
    # 평가 설정 (정밀화)
    EVALUATION_CONFIG = {
        'ap_weight': 0.5,
        'wll_weight': 0.5,
        'actual_ctr': 0.0201,
        'pos_weight': 0.0201,
        'neg_weight': 0.9799,
        'target_score': 0.36000,
        'bootstrap_samples': 1000,
        'confidence_interval': 0.95,
        'stability_threshold': 0.02,
        'performance_metrics': ['ap', 'wll', 'auc', 'f1', 'precision', 'recall'],
        'ctr_tolerance': 0.001,
        'bias_penalty_weight': 2.0,
        'calibration_weight': 0.3
    }
    
    # 앙상블 설정 (대폭 강화)
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
        'ensemble_optimization_trials': 200,
        'ensemble_cv_folds': 5,
        'ensemble_early_stopping': 50,
        'ensemble_regularization': 0.01,
        'dynamic_weighting': True,
        'adaptive_blending': True,
        'temporal_ensemble': True,
        'multi_level_ensemble': True
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
        'model_logging': True
    }
    
    # 실시간 추론 설정 (최적화)
    INFERENCE_CONFIG = {
        'batch_size': 20000,
        'timeout': 600,
        'cache_size': 100000,
        'model_version': 'v3.0',
        'use_gpu': GPU_AVAILABLE,
        'parallel_inference': True,
        'inference_optimization': True,
        'model_compilation': True,
        'quantization': False,
        'tensorrt_optimization': False,
        'onnx_optimization': False,
        'async_inference': True,
        'memory_pool': True,
        'connection_pool_size': 100
    }
    
    # 하이퍼파라미터 튜닝 설정 (강화)
    TUNING_CONFIG = {
        'n_trials': 50,
        'timeout': 7200,
        'parallel_jobs': 1,
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
        'max_resource': 81,
        'reduction_factor': 3,
        'bootstrap_count': 100,
        'multi_objective': False,
        'custom_sampler_params': {
            'consider_prior': True,
            'prior_weight': 1.0,
            'consider_magic_clip': True,
            'consider_endpoints': False,
            'n_startup_trials': 10,
            'n_ei_candidates': 24,
            'gamma': 0.25,
            'weights': {}
        }
    }
    
    # 메모리 관리 설정 (정밀화)
    MEMORY_CONFIG = {
        'auto_gc': True,
        'gc_threshold': 0.8,
        'force_gc_interval': 1000,
        'memory_monitoring': True,
        'memory_limit_warning': 0.9,
        'memory_limit_error': 0.95,
        'chunk_memory_limit': 5.0,
        'batch_memory_limit': 2.0,
        'model_memory_limit': 8.0,
        'ensemble_memory_limit': 12.0,
        'swap_usage_limit': 0.1,
        'memory_profiling': True,
        'memory_optimization': True,
        'lazy_loading': True,
        'memory_mapping': True,
        'compressed_storage': False
    }
    
    # GPU 설정 (RTX 4060 Ti 최적화)
    GPU_CONFIG = {
        'gpu_memory_fraction': 0.85,
        'allow_growth': True,
        'mixed_precision': USE_MIXED_PRECISION,
        'tensor_core_optimization': True,
        'cuda_optimization_level': GPU_OPTIMIZATION_LEVEL,
        'cuda_cache_config': 'PreferL1',
        'cudnn_benchmark': True,
        'cudnn_deterministic': False,
        'cuda_launch_blocking': False,
        'gpu_memory_monitoring': True,
        'gpu_utilization_monitoring': True,
        'multi_gpu': False,
        'gpu_sync_interval': 100,
        'gpu_memory_pool': True,
        'gpu_kernel_sync': False,
        'gpu_profiling': False,
        'tensor_parallelism': False,
        'pipeline_parallelism': False,
        'gradient_checkpointing': True,
        'activation_checkpointing': False
    }
    
    # 병렬 처리 설정 (6코어 최적화)
    PARALLEL_CONFIG = {
        'num_workers': NUM_WORKERS,
        'max_workers': NUM_WORKERS,
        'thread_pool_size': NUM_WORKERS * 2,
        'process_pool_size': NUM_WORKERS,
        'multiprocessing_context': 'spawn',
        'shared_memory': True,
        'parallel_backend': 'threading',
        'parallel_feature_engineering': True,
        'parallel_model_training': False,
        'parallel_inference': True,
        'parallel_evaluation': True,
        'thread_local_storage': True,
        'numa_optimization': False,
        'cpu_affinity': True,
        'priority_scheduling': True,
        'load_balancing': True,
        'work_stealing': True,
        'dynamic_scheduling': True
    }
    
    # 데이터 처리 설정 (확장)
    DATA_CONFIG = {
        'use_pyarrow': True,
        'compression': 'snappy',
        'memory_map': True,
        'lazy_loading': True,
        'batch_processing': True,
        'streaming_processing': False,
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
        'broadcast_operations': True
    }
    
    # 모델 저장/로딩 설정
    MODEL_IO_CONFIG = {
        'compression_level': 9,
        'pickle_protocol': 5,
        'joblib_compression': 'lz4',
        'model_versioning': True,
        'incremental_saving': True,
        'checkpoint_frequency': 1000,
        'backup_models': True,
        'model_metadata': True,
        'model_signature': True,
        'model_validation': True,
        'lazy_model_loading': True,
        'model_caching': True,
        'model_pooling': False,
        'distributed_storage': False,
        'cloud_storage': False,
        'local_storage_optimization': True
    }
    
    @classmethod
    def setup_directories(cls):
        """필요한 디렉터리 생성"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"디렉터리 생성 실패 {dir_path}: {e}")
    
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
                file_handler = logging.FileHandler(cls.LOG_DIR / 'ctr_model.log')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"파일 핸들러 설정 실패: {e}")
        
        return logger
    
    @classmethod
    def setup_gpu_environment(cls):
        """GPU 환경 설정"""
        if not TORCH_AVAILABLE:
            logger = logging.getLogger(__name__)
            logger.warning("PyTorch를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            return
        
        if torch.cuda.is_available():
            try:
                # CUDA 환경 변수 설정
                os.environ['CUDA_VISIBLE_DEVICES'] = cls.CUDA_VISIBLE_DEVICES
                
                # GPU 메모리 설정
                torch.cuda.empty_cache()
                
                # GPU 메모리 분율 설정
                if cls.GPU_CONFIG['gpu_memory_fraction'] < 1.0:
                    torch.cuda.set_per_process_memory_fraction(cls.GPU_CONFIG['gpu_memory_fraction'])
                
                # CUDA 최적화 설정
                if cls.GPU_CONFIG['cudnn_benchmark']:
                    torch.backends.cudnn.benchmark = True
                
                if cls.GPU_CONFIG['cudnn_deterministic']:
                    torch.backends.cudnn.deterministic = True
                
                # CUDA 캐시 설정
                if hasattr(torch.cuda, 'set_device'):
                    torch.cuda.set_device(0)
                
                # Mixed Precision 설정
                if cls.USE_MIXED_PRECISION and hasattr(torch.cuda, 'amp'):
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                logger = logging.getLogger(__name__)
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"GPU 설정: {gpu_name} ({gpu_memory:.1f}GB)")
                logger.info(f"GPU 메모리 분율: {cls.GPU_CONFIG['gpu_memory_fraction']:.1%}")
                logger.info(f"Mixed Precision: {cls.USE_MIXED_PRECISION}")
                logger.info(f"CUDA 최적화 레벨: {cls.GPU_OPTIMIZATION_LEVEL}")
                
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"GPU 환경 설정 실패: {e}")
        else:
            logger = logging.getLogger(__name__)
            logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    @classmethod
    def optimize_cpu_environment(cls):
        """CPU 환경 최적화"""
        try:
            # OMP 설정
            os.environ['OMP_NUM_THREADS'] = str(cls.NUM_WORKERS)
            os.environ['MKL_NUM_THREADS'] = str(cls.NUM_WORKERS)
            os.environ['NUMEXPR_NUM_THREADS'] = str(cls.NUM_WORKERS)
            
            # 메모리 설정
            if hasattr(os, 'sched_setaffinity') and cls.PARALLEL_CONFIG['cpu_affinity']:
                try:
                    # CPU 코어 할당
                    available_cpus = list(range(cls.NUM_WORKERS))
                    os.sched_setaffinity(0, available_cpus)
                except:
                    pass
            
            # 우선순위 설정
            if cls.PARALLEL_CONFIG['priority_scheduling']:
                try:
                    os.nice(-5)
                except:
                    pass
            
            logger = logging.getLogger(__name__)
            logger.info(f"CPU 최적화 완료 - 코어: {cls.NUM_WORKERS}개")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"CPU 환경 최적화 실패: {e}")
    
    @classmethod
    def get_memory_config(cls):
        """메모리 설정 반환"""
        try:
            import psutil
            
            total_memory = psutil.virtual_memory().total / (1024**3)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            # 64GB RAM 환경 최적화
            max_memory = min(cls.MAX_MEMORY_GB, available_memory - 8)
            
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
                'gc_threshold': 0.8
            }
    
    @classmethod
    def get_data_config(cls):
        """데이터 처리 설정 반환"""
        memory_config = cls.get_memory_config()
        
        # 전체 데이터 처리 보장
        return {
            'max_train_size': cls.MAX_TRAIN_SIZE,
            'max_test_size': cls.MAX_TEST_SIZE,
            'min_test_size': cls.MIN_TEST_SIZE,
            'chunk_size': memory_config['chunk_size'],
            'target_usage_ratio': cls.TARGET_DATA_USAGE_RATIO,
            'force_full_test_data': cls.FORCE_FULL_TEST_PREDICTION,
            'use_pyarrow': cls.DATA_CONFIG['use_pyarrow'],
            'memory_map': cls.DATA_CONFIG['memory_map'],
            'lazy_loading': cls.DATA_CONFIG['lazy_loading'],
            'batch_processing': cls.DATA_CONFIG['batch_processing'],
            'type_optimization': cls.DATA_CONFIG['type_optimization']
        }
    
    @classmethod
    def get_safe_memory_limits(cls):
        """안전한 메모리 한계 설정"""
        try:
            import psutil
            
            # 시스템 메모리 정보
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            
            # 64GB 환경 기준 설정
            safe_limit = min(cls.MAX_MEMORY_GB, available_gb - 8)
            
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
                'available_memory_gb': 48.0,
                'safe_memory_limit_gb': cls.MAX_MEMORY_GB,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'memory_monitoring_enabled': True,
                'auto_gc_enabled': True,
                'memory_limits': {
                    'chunk': 5.0,
                    'batch': 2.0,
                    'model': 8.0,
                    'ensemble': 12.0
                }
            }
    
    @classmethod
    def get_ensemble_config(cls):
        """앙상블 설정 반환"""
        return {
            'ensemble_types': cls.ENSEMBLE_CONFIG['ensemble_types'],
            'optimization_method': cls.ENSEMBLE_CONFIG['optimization_method'],
            'diversification_method': cls.ENSEMBLE_CONFIG['diversification_method'],
            'meta_model_type': cls.ENSEMBLE_CONFIG['meta_model_type'],
            'use_meta_features': cls.ENSEMBLE_CONFIG['use_meta_features'],
            'optimization_trials': cls.ENSEMBLE_CONFIG['ensemble_optimization_trials'],
            'cv_folds': cls.ENSEMBLE_CONFIG['ensemble_cv_folds'],
            'regularization': cls.ENSEMBLE_CONFIG['ensemble_regularization'],
            'dynamic_weighting': cls.ENSEMBLE_CONFIG['dynamic_weighting'],
            'adaptive_blending': cls.ENSEMBLE_CONFIG['adaptive_blending'],
            'temporal_ensemble': cls.ENSEMBLE_CONFIG['temporal_ensemble'],
            'multi_level_ensemble': cls.ENSEMBLE_CONFIG['multi_level_ensemble']
        }
    
    @classmethod
    def get_tuning_config(cls):
        """하이퍼파라미터 튜닝 설정 반환"""
        return {
            'n_trials': cls.TUNING_CONFIG['n_trials'],
            'timeout': cls.TUNING_CONFIG['timeout'],
            'parallel_jobs': cls.TUNING_CONFIG['parallel_jobs'],
            'pruner': cls.TUNING_CONFIG['pruner'],
            'sampler': cls.TUNING_CONFIG['sampler'],
            'optimization_direction': cls.TUNING_CONFIG['optimization_direction'],
            'enable_pruning': cls.TUNING_CONFIG['enable_pruning'],
            'n_startup_trials': cls.TUNING_CONFIG['n_startup_trials'],
            'n_warmup_steps': cls.TUNING_CONFIG['n_warmup_steps'],
            'custom_sampler_params': cls.TUNING_CONFIG['custom_sampler_params']
        }

# 환경변수 설정
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['MKL_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMEXPR_NUM_THREADS'] = str(Config.NUM_WORKERS)
    os.environ['NUMBA_NUM_THREADS'] = str(Config.NUM_WORKERS)
    
    # CUDA 환경 변수
    if Config.GPU_AVAILABLE:
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.CUDA_VISIBLE_DEVICES
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        
    # 메모리 관리 환경 변수
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
except Exception as e:
    print(f"환경 변수 설정 실패: {e}")

# 초기 환경 설정
try:
    if TORCH_AVAILABLE and Config.GPU_AVAILABLE:
        Config.setup_gpu_environment()
    
    Config.optimize_cpu_environment()
    
except Exception as e:
    print(f"초기 환경 설정 실패: {e}")