# config.py

import os
from pathlib import Path
import logging

# PyTorch import 안전하게 처리
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
    
    GPU_MEMORY_LIMIT = 14  # RTX 4060 Ti 16GB 중 14GB 사용
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    
    # 메모리 최적화 설정 (64GB RAM 기준)
    MAX_MEMORY_GB = 55  # 64GB 중 55GB 사용
    CHUNK_SIZE = 500000  # 청킹 크기
    BATCH_SIZE_GPU = 8192  # GPU 배치 크기
    BATCH_SIZE_CPU = 2048  # CPU 배치 크기
    PREFETCH_FACTOR = 4
    
    # 데이터 처리 전략
    TARGET_DATA_USAGE_RATIO = 0.15  # 전체 데이터의 15% 목표
    MIN_TRAIN_SIZE = 1000000  # 최소 학습 데이터 크기
    MAX_TRAIN_SIZE = 1500000  # 최대 학습 데이터 크기
    MIN_TEST_SIZE = 200000  # 최소 테스트 데이터 크기
    MAX_TEST_SIZE = 300000  # 최대 테스트 데이터 크기
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 5
    TEST_SIZE = 0.2
    
    # LightGBM CTR 특화 파라미터 (충돌 해결)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 255,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 200,
        'min_child_weight': 10,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 2000,
        'early_stopping_rounds': 200,
        'scale_pos_weight': 49.0  # is_unbalance 제거하여 충돌 방지
    }
    
    # XGBoost CTR 특화 파라미터
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 20,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 2000,
        'early_stopping_rounds': 200
    }
    
    # CatBoost CTR 특화 파라미터
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 10,
        'learning_rate': 0.03,
        'iterations': 2000,
        'l2_leaf_reg': 5,
        'random_seed': RANDOM_STATE,
        'early_stopping_rounds': 200,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }
    
    # 딥러닝 모델 파라미터 (GPU 최적화)
    NN_PARAMS = {
        'hidden_dims': [1024, 512, 256, 128],
        'dropout_rate': 0.4,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 20,
        'use_batch_norm': True,
        'activation': 'relu'
    }
    
    # Calibration 설정
    CALIBRATION_CONFIG = {
        'target_ctr': 0.0201,  # 실제 CTR
        'platt_scaling': True,
        'isotonic_regression': True,
        'temperature_scaling': True,
        'cv_folds': 3,
        'calibration_sample_size': 50000
    }
    
    # 피처 엔지니어링 설정
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 200,
        'frequency_threshold': 50,
        'interaction_features': True,
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,
        'id_hash_features': True,
        'polynomial_features': True,
        'max_polynomial_degree': 2
    }
    
    # 평가 설정 (실제 CTR 반영)
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
        'timeout': 100,
        'cache_size': 50000,
        'model_version': 'v2.0',
        'use_gpu': GPU_AVAILABLE
    }
    
    # 하이퍼파라미터 튜닝 설정
    TUNING_CONFIG = {
        'n_trials': 50,  # 충돌 방지를 위해 감소
        'timeout': 3600,
        'parallel_jobs': 6,  # 6코어 CPU 활용
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
                
                logger = logging.getLogger(__name__)
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"GPU 활성화: {gpu_name} ({gpu_memory:.1f}GB)")
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
            
            # 동적 메모리 설정
            if available_memory > 50:
                max_memory = min(cls.MAX_MEMORY_GB, available_memory - 5)
            else:
                max_memory = available_memory * 0.8
            
            return {
                'max_memory_gb': max_memory,
                'chunk_size': cls.CHUNK_SIZE,
                'batch_size': cls.BATCH_SIZE_GPU if cls.GPU_AVAILABLE else cls.BATCH_SIZE_CPU,
                'prefetch_factor': cls.PREFETCH_FACTOR
            }
        except ImportError:
            # psutil이 없는 경우 기본값 사용
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
        
        # 메모리 기반 동적 크기 조정
        if memory_config['max_memory_gb'] > 40:
            train_size = cls.MAX_TRAIN_SIZE
            test_size = cls.MAX_TEST_SIZE
        else:
            train_size = cls.MIN_TRAIN_SIZE
            test_size = cls.MIN_TEST_SIZE
        
        return {
            'max_train_size': train_size,
            'max_test_size': test_size,
            'chunk_size': memory_config['chunk_size'],
            'target_usage_ratio': cls.TARGET_DATA_USAGE_RATIO
        }

# 환경변수 설정 - 안전하게 처리
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
except Exception as e:
    print(f"환경 변수 설정 실패: {e}")

# GPU 환경 설정 (임포트 시 자동 실행)
try:
    if TORCH_AVAILABLE:
        Config.setup_gpu_environment()
except Exception as e:
    print(f"GPU 환경 설정 실패: {e}")