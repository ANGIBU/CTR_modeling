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
    
    # GPU 및 하드웨어 설정
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        GPU_AVAILABLE = torch.cuda.is_available()
    else:
        DEVICE = 'cpu'
        GPU_AVAILABLE = False
    
    GPU_MEMORY_LIMIT = 12  # RTX 4060 Ti 16GB 중 12GB 사용
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    
    # 64GB RAM 환경 최적화 설정
    MAX_MEMORY_GB = 45  # 64GB 중 45GB 사용 가능
    CHUNK_SIZE = 500000  # 청킹 크기 확대
    BATCH_SIZE_GPU = 8192  # GPU 배치 크기 확대
    BATCH_SIZE_CPU = 2048  # CPU 배치 크기 확대
    PREFETCH_FACTOR = 2
    
    # 전체 데이터 처리 전략
    TARGET_DATA_USAGE_RATIO = 1.0  # 전체 데이터 100% 사용
    MIN_TRAIN_SIZE = 800000  # 최소 학습 데이터 크기 확대
    MAX_TRAIN_SIZE = 2000000  # 최대 학습 데이터 크기 확대
    MIN_TEST_SIZE = 1527298  # 전체 테스트 데이터 필수
    MAX_TEST_SIZE = 1527298  # 전체 테스트 데이터 필수
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 3  # CV 폴드 수
    TEST_SIZE = 0.2
    
    # LightGBM CTR 파라미터
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_child_samples': 300,
        'min_child_weight': 15,
        'lambda_l1': 1.5,
        'lambda_l2': 1.5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,
        'early_stopping_rounds': 150,
        'scale_pos_weight': 49.0,
        'force_row_wise': True,
        'max_bin': 255,
        'num_threads': 6
    }
    
    # XGBoost CTR 파라미터
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'min_child_weight': 25,
        'reg_alpha': 1.5,
        'reg_lambda': 1.5,
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,
        'early_stopping_rounds': 150,
        'max_bin': 255,
        'nthread': 6
    }
    
    # CatBoost CTR 파라미터
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 7,
        'learning_rate': 0.05,
        'iterations': 1500,
        'l2_leaf_reg': 8,
        'random_seed': RANDOM_STATE,
        'early_stopping_rounds': 150,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 1,
        'thread_count': 6
    }
    
    # 딥러닝 모델 파라미터
    NN_PARAMS = {
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.4,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 80,
        'patience': 15,
        'use_batch_norm': True,
        'activation': 'relu'
    }
    
    # Calibration 설정
    CALIBRATION_CONFIG = {
        'target_ctr': 0.0201,
        'platt_scaling': True,
        'isotonic_regression': True,
        'temperature_scaling': True,
        'cv_folds': 3,
        'calibration_sample_size': 50000
    }
    
    # 피처 엔지니어링 설정
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 200,
        'frequency_threshold': 100,
        'interaction_features': True,
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,
        'id_hash_features': True,
        'polynomial_features': False,
        'max_polynomial_degree': 2
    }
    
    # 평가 설정
    EVALUATION_CONFIG = {
        'ap_weight': 0.5,
        'wll_weight': 0.5,
        'actual_ctr': 0.0201,
        'pos_weight': 0.0201,
        'neg_weight': 0.9799,
        'target_score': 0.34624
    }
    
    # 앙상블 설정
    ENSEMBLE_CONFIG = {
        'use_stacking': True,
        'meta_model': 'logistic',
        'calibration_ensemble': True,
        'blend_weights': {
            'lgbm': 0.35,
            'xgb': 0.25,
            'cat': 0.25,
            'deepctr': 0.15
        }
    }
    
    # 로깅 설정
    LOGGING_CONFIG = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_handler': True,
        'console_handler': True
    }
    
    # 실시간 추론 설정
    INFERENCE_CONFIG = {
        'batch_size': 10000,
        'timeout': 300,
        'cache_size': 50000,
        'model_version': 'v2.0',
        'use_gpu': GPU_AVAILABLE
    }
    
    # 하이퍼파라미터 튜닝 설정
    TUNING_CONFIG = {
        'n_trials': 30,
        'timeout': 3600,
        'parallel_jobs': 1,
        'pruner': 'MedianPruner',
        'sampler': 'TPESampler'
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
                
                # Mixed Precision 설정
                if cls.USE_MIXED_PRECISION:
                    torch.backends.cudnn.benchmark = True
                
                # GPU 메모리 제한 설정
                if torch.cuda.is_available():
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.75)
                    except Exception:
                        pass
                
                logger = logging.getLogger(__name__)
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"GPU 설정: {gpu_name} ({gpu_memory:.1f}GB)")
                logger.info(f"Mixed Precision: {cls.USE_MIXED_PRECISION}")
                
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"GPU 환경 설정 실패: {e}")
        else:
            logger = logging.getLogger(__name__)
            logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    @classmethod
    def get_memory_config(cls):
        """메모리 설정 반환"""
        try:
            import psutil
            
            total_memory = psutil.virtual_memory().total / (1024**3)
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            # 64GB RAM 환경 최적화
            max_memory = min(cls.MAX_MEMORY_GB, available_memory - 10)
            
            return {
                'max_memory_gb': max_memory,
                'chunk_size': cls.CHUNK_SIZE,
                'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'prefetch_factor': cls.PREFETCH_FACTOR
            }
        except ImportError:
            return {
                'max_memory_gb': cls.MAX_MEMORY_GB,
                'chunk_size': cls.CHUNK_SIZE,
                'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'prefetch_factor': cls.PREFETCH_FACTOR
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
            'force_full_test_data': True
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
            safe_limit = min(45, available_gb - 10)
            
            return {
                'total_memory_gb': total_gb,
                'available_memory_gb': available_gb,
                'safe_memory_limit_gb': safe_limit,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU
            }
        except ImportError:
            return {
                'total_memory_gb': 64.0,
                'available_memory_gb': 50.0,
                'safe_memory_limit_gb': 45.0,
                'recommended_chunk_size': cls.CHUNK_SIZE,
                'recommended_batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU
            }

# 환경변수 설정
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '6'
    os.environ['MKL_NUM_THREADS'] = '6'
    os.environ['NUMEXPR_NUM_THREADS'] = '6'
except Exception as e:
    print(f"환경 변수 설정 실패: {e}")

# GPU 환경 설정
try:
    if TORCH_AVAILABLE:
        Config.setup_gpu_environment()
except Exception as e:
    print(f"GPU 환경 설정 실패: {e}")