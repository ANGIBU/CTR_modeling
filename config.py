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
    
    GPU_MEMORY_LIMIT = 12  # RTX 4060 Ti 16GB 중 12GB 사용 (더 보수적으로)
    CUDA_VISIBLE_DEVICES = "0"
    USE_MIXED_PRECISION = True
    
    # 메모리 최적화 설정 (64GB RAM 기준, 더 보수적으로)
    MAX_MEMORY_GB = 32  # 64GB 중 32GB 사용 (기존 55GB에서 대폭 감소)
    CHUNK_SIZE = 300000  # 청킹 크기 감소 (기존 500000에서)
    BATCH_SIZE_GPU = 4096  # GPU 배치 크기 감소 (기존 8192에서)
    BATCH_SIZE_CPU = 1024  # CPU 배치 크기 감소 (기존 2048에서)
    PREFETCH_FACTOR = 2  # 프리페치 팩터 감소 (기존 4에서)
    
    # 데이터 처리 전략 (더 보수적으로)
    TARGET_DATA_USAGE_RATIO = 0.08  # 전체 데이터의 8% 목표 (기존 15%에서 감소)
    MIN_TRAIN_SIZE = 500000  # 최소 학습 데이터 크기 감소 (기존 1000000에서)
    MAX_TRAIN_SIZE = 800000  # 최대 학습 데이터 크기 감소 (기존 1500000에서)
    MIN_TEST_SIZE = 100000  # 최소 테스트 데이터 크기 감소 (기존 200000에서)
    MAX_TEST_SIZE = 150000  # 최대 테스트 데이터 크기 감소 (기존 300000에서)
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 3  # CV 폴드 수 감소 (기존 5에서)
    TEST_SIZE = 0.2
    
    # LightGBM CTR 파라미터 (메모리 최적화)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,  # 메모리 절약을 위해 감소 (기존 255에서)
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
        'n_estimators': 1500,  # 메모리 절약을 위해 감소 (기존 2000에서)
        'early_stopping_rounds': 150,  # 조기 종료 라운드 감소
        'scale_pos_weight': 49.0,
        'force_row_wise': True,  # 메모리 최적화
        'max_bin': 255  # 메모리 절약
    }
    
    # XGBoost CTR 파라미터 (메모리 최적화)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
        'gpu_id': 0 if GPU_AVAILABLE else None,
        'max_depth': 8,  # 메모리 절약을 위해 감소 (기존 10에서)
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 20,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'scale_pos_weight': 49.0,
        'random_state': RANDOM_STATE,
        'n_estimators': 1500,  # 메모리 절약을 위해 감소
        'early_stopping_rounds': 150,
        'max_bin': 256  # 메모리 절약
    }
    
    # CatBoost CTR 파라미터 (메모리 최적화)
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
        'devices': '0' if GPU_AVAILABLE else None,
        'depth': 8,  # 메모리 절약을 위해 감소 (기존 10에서)
        'learning_rate': 0.03,
        'iterations': 1500,  # 메모리 절약을 위해 감소 (기존 2000에서)
        'l2_leaf_reg': 5,
        'random_seed': RANDOM_STATE,
        'early_stopping_rounds': 150,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'max_ctr_complexity': 1  # 메모리 절약
    }
    
    # 딥러닝 모델 파라미터 (메모리 최적화)
    NN_PARAMS = {
        'hidden_dims': [512, 256, 128],  # 레이어 크기 감소 (기존 [1024, 512, 256, 128]에서)
        'dropout_rate': 0.4,
        'batch_size': BATCH_SIZE_GPU if GPU_AVAILABLE else BATCH_SIZE_CPU,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'epochs': 80,  # 에포크 수 감소 (기존 100에서)
        'patience': 15,  # 패션스 감소 (기존 20에서)
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
        'calibration_sample_size': 30000  # 메모리 절약을 위해 감소 (기존 50000에서)
    }
    
    # 피처 엔지니어링 설정 (메모리 최적화)
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 200,
        'frequency_threshold': 50,
        'interaction_features': False,  # 메모리 절약을 위해 비활성화
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,
        'id_hash_features': True,
        'polynomial_features': False,  # 메모리 절약을 위해 비활성화
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
    
    # 앙상블 설정 (메모리 최적화)
    ENSEMBLE_CONFIG = {
        'use_stacking': False,  # 메모리 절약을 위해 비활성화
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
    
    # 실시간 추론 설정 (메모리 최적화)
    INFERENCE_CONFIG = {
        'batch_size': 5000,  # 배치 크기 감소 (기존 10000에서)
        'timeout': 100,
        'cache_size': 25000,  # 캐시 크기 감소 (기존 50000에서)
        'model_version': 'v2.0',
        'use_gpu': GPU_AVAILABLE
    }
    
    # 하이퍼파라미터 튜닝 설정 (메모리 최적화)
    TUNING_CONFIG = {
        'n_trials': 30,  # 트라이얼 수 감소 (기존 50에서)
        'timeout': 2400,  # 타임아웃 감소 (기존 3600에서)
        'parallel_jobs': 3,  # 병렬 작업 수 감소 (기존 6에서)
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
                        torch.cuda.set_per_process_memory_fraction(0.75)  # GPU 메모리의 75%만 사용
                    except Exception:
                        pass
                
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
            
            # 동적 메모리 설정 (더 보수적으로)
            if available_memory > 40:
                max_memory = min(cls.MAX_MEMORY_GB, available_memory - 10)  # 10GB 여유 확보
            elif available_memory > 25:
                max_memory = available_memory * 0.6  # 60%만 사용
            else:
                max_memory = available_memory * 0.5  # 50%만 사용
            
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
        
        # 메모리 기반 동적 크기 조정 (더 보수적으로)
        if memory_config['max_memory_gb'] > 30:
            train_size = cls.MAX_TRAIN_SIZE
            test_size = cls.MAX_TEST_SIZE
        elif memory_config['max_memory_gb'] > 20:
            train_size = int(cls.MAX_TRAIN_SIZE * 0.7)  # 70%
            test_size = int(cls.MAX_TEST_SIZE * 0.7)
        else:
            train_size = cls.MIN_TRAIN_SIZE
            test_size = cls.MIN_TEST_SIZE
        
        return {
            'max_train_size': train_size,
            'max_test_size': test_size,
            'chunk_size': memory_config['chunk_size'],
            'target_usage_ratio': cls.TARGET_DATA_USAGE_RATIO
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
            
            # 안전한 메모리 사용량 계산
            if available_gb > 50:  # 64GB 시스템
                safe_limit = 25  # 25GB로 제한
            elif available_gb > 30:  # 32GB 시스템
                safe_limit = 15  # 15GB로 제한
            elif available_gb > 15:  # 16GB 시스템
                safe_limit = 8   # 8GB로 제한
            else:
                safe_limit = available_gb * 0.4  # 40%만 사용
            
            return {
                'total_memory_gb': total_gb,
                'available_memory_gb': available_gb,
                'safe_memory_limit_gb': safe_limit,
                'recommended_chunk_size': min(cls.CHUNK_SIZE, int(safe_limit * 10000)),
                'recommended_batch_size': min(cls.BATCH_SIZE_GPU, int(safe_limit * 100))
            }
        except ImportError:
            # psutil이 없는 경우 기본 안전 설정
            return {
                'total_memory_gb': 64.0,
                'available_memory_gb': 32.0,
                'safe_memory_limit_gb': 20.0,
                'recommended_chunk_size': 200000,
                'recommended_batch_size': 2048
            }

# 환경변수 설정 - 안전하게 처리
try:
    os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '6'  # CPU 코어 수에 맞춤
    os.environ['MKL_NUM_THREADS'] = '6'
    os.environ['NUMEXPR_NUM_THREADS'] = '6'
except Exception as e:
    print(f"환경 변수 설정 실패: {e}")

# GPU 환경 설정 (임포트 시 자동 실행)
try:
    if TORCH_AVAILABLE:
        Config.setup_gpu_environment()
except Exception as e:
    print(f"GPU 환경 설정 실패: {e}")