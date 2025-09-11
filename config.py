# config.py

import os
from pathlib import Path
import logging

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
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    N_SPLITS = 5
    TEST_SIZE = 0.2
    
    # LightGBM 기본 파라미터 (CTR 특화)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'min_child_weight': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_estimators': 1000,
        'early_stopping_rounds': 100,
        'is_unbalance': True
    }
    
    # XGBoost 기본 파라미터 (CTR 특화)
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': 51.2,  # 1/0.0191 - 1
        'random_state': RANDOM_STATE,
        'n_estimators': 1000,
        'early_stopping_rounds': 100
    }
    
    # CatBoost 기본 파라미터 (CTR 특화)
    CAT_PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'depth': 8,
        'learning_rate': 0.05,
        'iterations': 1000,
        'l2_leaf_reg': 3,
        'random_seed': RANDOM_STATE,
        'early_stopping_rounds': 100,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }
    
    # 신경망 모델 파라미터
    NN_PARAMS = {
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'batch_size': 1024,
        'learning_rate': 0.001,
        'epochs': 50,
        'patience': 10
    }
    
    # 피처 엔지니어링 설정
    FEATURE_CONFIG = {
        'target_encoding_smoothing': 100,
        'frequency_threshold': 10,
        'interaction_features': True,
        'time_features': True,
        'statistical_features': True,
        'preserve_ids': True,  # ID 피처 보존
        'id_hash_features': True  # ID 해시 피처 생성
    }
    
    # 평가 설정 (실제 CTR 반영)
    EVALUATION_CONFIG = {
        'ap_weight': 0.5,
        'wll_weight': 0.5,
        'actual_ctr': 0.0191,  # 실제 관찰된 CTR
        'pos_weight': 0.0191,
        'neg_weight': 0.9809
    }
    
    # 앙상블 설정
    ENSEMBLE_CONFIG = {
        'use_stacking': True,
        'meta_model': 'logistic',
        'blend_weights': {
            'lgbm': 0.4,
            'xgb': 0.3,
            'cat': 0.2,
            'nn': 0.1
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
        'batch_size': 1000,
        'timeout': 100,
        'cache_size': 10000,
        'model_version': 'v1.0'
    }
    
    @classmethod
    def setup_directories(cls):
        """필요한 디렉터리 생성"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_logging(cls):
        """로깅 설정 초기화"""
        cls.setup_directories()
        
        logger = logging.getLogger()
        logger.setLevel(cls.LOGGING_CONFIG['level'])
        
        formatter = logging.Formatter(cls.LOGGING_CONFIG['format'])
        
        if cls.LOGGING_CONFIG['console_handler']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if cls.LOGGING_CONFIG['file_handler']:
            file_handler = logging.FileHandler(cls.LOG_DIR / 'ctr_model.log')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

# 환경변수 설정
os.environ['PYTHONHASHSEED'] = str(Config.RANDOM_STATE)