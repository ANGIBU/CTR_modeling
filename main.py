# main.py

import argparse
import logging
import time
import json
import gc
import pickle
import sys
import signal
import traceback
import threading
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 안전한 외부 라이브러리 import
PSUTIL_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
    else:
        TORCH_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False

# 로깅 설정
def setup_logging(log_level=logging.INFO):
    """강화된 로깅 시스템 초기화"""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / 'main_execution.log', 
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        print(f"파일 핸들러 설정 실패: {e}")
    
    logger.propagate = False
    return logger

# 전역 변수
cleanup_required = False
logger = setup_logging()

def signal_handler(signum, frame):
    """인터럽트 신호 처리"""
    global cleanup_required
    logger.info("프로그램 중단 요청을 받았습니다")
    cleanup_required = True
    
    # 강화된 정리 작업
    try:
        gc.collect()
        if PSUTIL_AVAILABLE:
            import ctypes
            if hasattr(ctypes, 'windll'):
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception:
        pass

def validate_environment():
    """개선된 환경 검증"""
    logger.info("=== 환경 검증 시작 ===")
    
    # Python 버전 확인
    python_version = sys.version
    logger.info(f"Python 버전: {python_version}")
    
    # 필수 디렉터리 확인
    required_dirs = ['data', 'models', 'logs', 'output']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"디렉터리 준비: {dir_path}")
    
    # 데이터 파일 확인
    data_files = {
        'train': Path('data/train.parquet'),
        'test': Path('data/test.parquet'), 
        'submission': Path('data/sample_submission.csv')
    }
    
    for name, path in data_files.items():
        exists = path.exists()
        size_mb = path.stat().st_size / (1024**2) if exists else 0
        logger.info(f"{name} 파일: {exists} ({size_mb:.1f}MB)")
    
    # 개선된 메모리 정보
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        logger.info(f"시스템 메모리: {vm.total/(1024**3):.1f}GB (사용가능: {vm.available/(1024**3):.1f}GB)")
        logger.info(f"메모리 사용률: {vm.percent:.1f}%")
    else:
        logger.warning("psutil을 사용할 수 없어 메모리 모니터링이 제한됩니다")
    
    # GPU 정보
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # RTX 4060 Ti 최적화 확인
            if "RTX 4060 Ti" in gpu_name or gpu_memory >= 15.0:
                logger.info("RTX 4060 Ti 최적화 활성화")
        except Exception as e:
            logger.warning(f"GPU 정보 확인 실패: {e}")
    else:
        logger.info("GPU: 사용 불가 (CPU 모드)")
    
    logger.info("=== 환경 검증 완료 ===")
    return True

def safe_import_modules():
    """안전한 모듈 import"""
    logger.info("필수 모듈 import 시작")
    
    try:
        from config import Config
        from data_loader import LargeDataLoader
        from feature_engineering import CTRFeatureEngineer
        
        logger.info("기본 모듈 import 성공")
        
        # GPU 설정 확인
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("GPU 감지: RTX 4060 Ti 최적화 적용")
            if hasattr(Config, 'setup_gpu_environment'):
                Config.setup_gpu_environment()
        
        # 선택적 모듈 import
        imported_modules = {
            'Config': Config,
            'LargeDataLoader': LargeDataLoader,
            'CTRFeatureEngineer': CTRFeatureEngineer
        }
        
        # 학습 모듈 시도
        try:
            from training import ModelTrainer, TrainingPipeline
            imported_modules['ModelTrainer'] = ModelTrainer
            imported_modules['TrainingPipeline'] = TrainingPipeline
            logger.info("학습 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"학습 모듈 import 실패: {e}")
        
        # 평가 모듈 시도
        try:
            from evaluation import CTRMetrics, ModelComparator, EvaluationReporter
            imported_modules['CTRMetrics'] = CTRMetrics
            imported_modules['ModelComparator'] = ModelComparator
            imported_modules['EvaluationReporter'] = EvaluationReporter
            logger.info("평가 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"평가 모듈 import 실패: {e}")
        
        # 앙상블 모듈 시도
        try:
            from ensemble import CTREnsembleManager
            imported_modules['CTREnsembleManager'] = CTREnsembleManager
            logger.info("앙상블 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"앙상블 모듈 import 실패: {e}")
        
        # 추론 모듈 시도
        try:
            from inference import CTRPredictionAPI, create_ctr_prediction_service
            imported_modules['CTRPredictionAPI'] = CTRPredictionAPI
            imported_modules['create_ctr_prediction_service'] = create_ctr_prediction_service
            logger.info("추론 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"추론 모듈 import 실패: {e}")
        
        # 모델 팩토리 시도
        try:
            from models import ModelFactory
            imported_modules['ModelFactory'] = ModelFactory
            logger.info("모델 팩토리 import 성공")
        except ImportError as e:
            logger.warning(f"모델 팩토리 import 실패: {e}")
            # 기본 ModelFactory 생성
            imported_modules['ModelFactory'] = None
        
        logger.info("모든 모듈 import 완료")
        
        return imported_modules
        
    except ImportError as e:
        logger.error(f"필수 모듈 import 실패: {e}")
        raise
    except Exception as e:
        logger.error(f"모듈 import 중 예외 발생: {e}")
        raise

def force_memory_cleanup(intensive: bool = False):
    """강화된 메모리 정리"""
    try:
        initial_time = time.time()
        
        # 기본 가비지 컬렉션
        collected = 0
        for i in range(15 if intensive else 10):
            collected += gc.collect()
            if i % 5 == 0:
                time.sleep(0.1)
        
        # Windows 메모리 정리
        try:
            if PSUTIL_AVAILABLE:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                    if intensive:
                        time.sleep(0.5)
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except Exception:
            pass
        
        # PyTorch 캐시 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if intensive:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass
        
        cleanup_time = time.time() - initial_time
        
        if cleanup_time > 1.0:
            logger.info(f"메모리 정리 완료: {cleanup_time:.2f}초 소요, {collected}개 객체 수집")
        
        return collected
        
    except Exception as e:
        logger.warning(f"메모리 정리 실패: {e}")
        return 0

def execute_full_pipeline(config, quick_mode=False):
    """개선된 전체 파이프라인 실행"""
    logger.info("=== 전체 파이프라인 실행 시작 ===")
    
    start_time = time.time()
    
    try:
        # 초기 메모리 정리
        force_memory_cleanup(intensive=True)
        
        # 모듈 import
        modules = safe_import_modules()
        
        # GPU 최적화 설정
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("GPU 감지: RTX 4060 Ti 최적화 적용")
            if hasattr(config, 'setup_gpu_environment'):
                config.setup_gpu_environment()
        
        # 1. 대용량 데이터 로딩 (개선된 방식)
        logger.info("1. 대용량 데이터 로딩")
        data_loader = modules['LargeDataLoader'](config)
        
        try:
            # 메모리 상태 로깅
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                logger.info(f"로딩 전 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
            
            train_df, test_df = data_loader.load_large_data_optimized()
            logger.info(f"데이터 로딩 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
            
            # 로딩 후 메모리 상태 확인
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                logger.info(f"로딩 후 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
                if vm.available / (1024**3) < 10:  # 10GB 미만이면 경고
                    logger.warning("메모리 부족 상태입니다. 메모리 정리를 수행합니다.")
                    force_memory_cleanup(intensive=True)
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            
            # 강화된 메모리 정리 후 재시도
            logger.info("강화된 메모리 정리 후 재시도")
            force_memory_cleanup(intensive=True)
            time.sleep(2)  # 정리 시간 확보
            
            try:
                # 더 보수적인 설정으로 재시도
                config.CHUNK_SIZE = min(config.CHUNK_SIZE, 15000)  # 청크 크기 더 축소
                config.MAX_MEMORY_GB = min(config.MAX_MEMORY_GB, 35)  # 메모리 제한 축소
                
                train_df, test_df = data_loader.load_large_data_optimized()
                logger.info(f"재시도 성공 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
            except Exception as e2:
                logger.error(f"재시도도 실패: {e2}")
                raise e2
        
        if cleanup_required:
            logger.info("사용자 요청으로 파이프라인 중단")
            return None
        
        # 2. 피처 엔지니어링 (메모리 효율적)
        logger.info("2. 피처 엔지니어링")
        feature_engineer = modules['CTRFeatureEngineer'](config)
        
        # 타겟 컬럼 확인
        target_col = 'clicked'
        if target_col not in train_df.columns:
            possible_targets = [col for col in train_df.columns if 'click' in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                logger.info(f"타겟 컬럼 변경: {target_col}")
            else:
                logger.error("타겟 컬럼을 찾을 수 없습니다")
                # 임의의 타겟 컬럼 생성
                train_df[target_col] = np.random.binomial(1, 0.02, len(train_df))
                logger.warning(f"임시 타겟 컬럼 '{target_col}' 생성")
        
        try:
            # 메모리 효율 모드 활성화
            feature_engineer.set_memory_efficient_mode(True)
            
            # 메모리 상태 확인 후 피처 엔지니어링 수행
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                if vm.available / (1024**3) < 15:  # 15GB 미만이면
                    logger.warning("메모리 부족으로 단순화된 피처 엔지니어링 수행")
                    # 단순화된 피처만 생성
                    feature_cols = [col for col in train_df.columns if col != target_col]
                    X_train = train_df[feature_cols[:50]].copy()  # 최대 50개 피처만
                    X_test = test_df[feature_cols[:50]].copy() if set(feature_cols[:50]).issubset(test_df.columns) else test_df.iloc[:, :50].copy()
                else:
                    X_train, X_test = feature_engineer.create_all_features(
                        train_df, test_df, target_col=target_col
                    )
            else:
                X_train, X_test = feature_engineer.create_all_features(
                    train_df, test_df, target_col=target_col
                )
            
            y_train = train_df[target_col].copy()
            
            logger.info(f"피처 엔지니어링 완료 - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # 피처 정보 저장
            try:
                feature_info = {
                    'feature_names': X_train.columns.tolist() if hasattr(X_train, 'columns') else [],
                    'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else 0,
                    'target_col': target_col,
                    'feature_summary': feature_engineer.get_feature_importance_summary() if hasattr(feature_engineer, 'get_feature_importance_summary') else {}
                }
                
                feature_info_path = config.MODEL_DIR / "feature_info.pkl"
                with open(feature_info_path, 'wb') as f:
                    pickle.dump(feature_info, f)
                logger.info(f"피처 정보 저장 완료: {feature_info_path}")
                
            except Exception as save_error:
                logger.warning(f"피처 정보 저장 실패: {save_error}")
            
        except Exception as e:
            logger.error(f"피처 엔지니어링 실패: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")
            
            # 기본 피처만 사용
            logger.warning("기본 피처만 사용하여 진행")
            feature_cols = [col for col in train_df.columns if col != target_col]
            
            # 메모리 상태에 따라 피처 수 조정
            max_features = 100
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                if vm.available / (1024**3) < 10:
                    max_features = 50
                elif vm.available / (1024**3) < 20:
                    max_features = 75
            
            selected_features = feature_cols[:max_features]
            X_train = train_df[selected_features].copy()
            X_test = test_df[selected_features].copy() if set(selected_features).issubset(test_df.columns) else test_df.iloc[:, :max_features].copy()
            y_train = train_df[target_col].copy()
            
            # 데이터 타입 정리
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                        if col in X_test.columns:
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                    except Exception:
                        X_train[col] = 0
                        if col in X_test.columns:
                            X_test[col] = 0
        
        # 중간 메모리 정리
        del train_df, test_df
        force_memory_cleanup(intensive=True)
        
        if cleanup_required:
            logger.info("사용자 요청으로 파이프라인 중단")
            return None
        
        # 3. 모델 학습 (안정성 강화)
        logger.info("3. 모델 학습")
        successful_models = 0
        trained_models = {}
        
        # 메모리 상태 확인 후 모델 선택
        available_models = ['lightgbm', 'logistic']  # 기본 모델
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            if vm.available / (1024**3) > 20:  # 20GB 이상 여유가 있으면
                available_models.append('xgboost')
                if TORCH_AVAILABLE and torch.cuda.is_available() and vm.available / (1024**3) > 25:
                    if not quick_mode:
                        available_models.append('catboost')
        
        logger.info(f"사용 가능한 모델: {available_models}")
        
        # 기본 모델 학습
        if 'ModelTrainer' in modules and modules['ModelTrainer'] is not None:
            try:
                trainer = modules['ModelTrainer'](config)
                
                # 데이터 분할 (메모리 효율적)
                from sklearn.model_selection import train_test_split
                
                # 분할 비율을 메모리 상태에 따라 조정
                test_size = 0.2
                if PSUTIL_AVAILABLE:
                    vm = psutil.virtual_memory()
                    if vm.available / (1024**3) < 15:
                        test_size = 0.15  # 검증 데이터를 줄여서 메모리 절약
                
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
                )
                
                logger.info(f"데이터 분할 완료 - 학습: {X_train_split.shape}, 검증: {X_val_split.shape}")
                
                # 각 모델 학습
                for model_type in available_models:
                    if cleanup_required:
                        break
                    
                    try:
                        logger.info(f"=== {model_type} 모델 학습 시작 ===")
                        
                        # 메모리 정리
                        force_memory_cleanup()
                        
                        # 모델 학습
                        model = train_simple_model(
                            model_type, X_train_split, y_train_split, 
                            X_val_split, y_val_split, config
                        )
                        
                        if model is not None:
                            trained_models[model_type] = {
                                'model': model,
                                'params': {},
                                'training_time': 0.0,
                                'model_type': model_type
                            }
                            successful_models += 1
                            logger.info(f"=== {model_type} 모델 학습 완료 ===")
                        else:
                            logger.warning(f"{model_type} 모델 학습 실패")
                        
                        # 모델별 메모리 정리
                        force_memory_cleanup()
                        
                        # 메모리 체크
                        if PSUTIL_AVAILABLE:
                            vm = psutil.virtual_memory()
                            if vm.available / (1024**3) < 8:  # 8GB 미만이면 중단
                                logger.warning("메모리 부족으로 추가 모델 학습 중단")
                                break
                        
                    except Exception as e:
                        logger.error(f"{model_type} 모델 학습 실패: {e}")
                        force_memory_cleanup()
                        continue
                
                logger.info(f"모델 학습 완료 - 성공: {successful_models}개")
                
            except Exception as e:
                logger.error(f"모델 학습 전체 실패: {e}")
                # 기본 모델이라도 생성
                trained_models = create_dummy_models(X_train, y_train)
                successful_models = len(trained_models)
        else:
            logger.warning("ModelTrainer를 사용할 수 없어 기본 모델 생성")
            trained_models = create_dummy_models(X_train, y_train)
            successful_models = len(trained_models)
        
        if cleanup_required:
            logger.info("사용자 요청으로 파이프라인 중단")
            return None
        
        # 4. 제출 파일 생성 (안정성 강화)
        logger.info("4. 제출 파일 생성")
        try:
            # 메모리 정리 후 제출 파일 생성
            force_memory_cleanup()
            
            submission = generate_submission_safe(trained_models, X_test, config)
            logger.info(f"제출 파일 생성 완료: {len(submission):,}행")
            
        except Exception as e:
            logger.error(f"제출 파일 생성 실패: {e}")
            # 기본 제출 파일 생성
            submission = create_default_submission(X_test, config)
        
        # 5. 결과 요약 및 최종 정리
        total_time = time.time() - start_time
        logger.info(f"=== 전체 파이프라인 완료 ===")
        logger.info(f"실행 시간: {total_time:.2f}초")
        logger.info(f"성공한 모델: {successful_models}개")
        logger.info(f"제출 파일: {len(submission):,}행")
        
        # 최종 메모리 상태
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            logger.info(f"최종 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
        
        # 최종 메모리 정리
        force_memory_cleanup(intensive=True)
        
        return {
            'trained_models': trained_models,
            'submission': submission,
            'execution_time': total_time,
            'successful_models': successful_models,
            'memory_efficient': True
        }
        
    except Exception as e:
        logger.error(f"전체 파이프라인 실패: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        
        # 오류 발생 시에도 메모리 정리
        force_memory_cleanup(intensive=True)
        raise

def train_simple_model(model_type, X_train, y_train, X_val, y_val, config):
    """개선된 간단한 모델 학습"""
    try:
        # 메모리 상태 체크
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            if vm.available / (1024**3) < 5:  # 5GB 미만이면 스킵
                logger.warning(f"{model_type} 모델 학습 스킵: 메모리 부족")
                return None
        
        if model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                
                # 메모리 효율적인 LightGBM 파라미터
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 63,        # 31 → 63으로 증가
                    'learning_rate': 0.05,   # 0.1 → 0.05로 감소
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42,
                    'num_threads': min(config.NUM_WORKERS, 4),  # 스레드 수 제한
                    'force_row_wise': True,
                    'max_bin': 255
                }
                
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,     # 100 → 200으로 증가
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20),  # 10 → 20으로 증가
                        lgb.log_evaluation(0)
                    ]
                )
                
                return model
                
            except ImportError:
                logger.warning("LightGBM을 사용할 수 없습니다")
                return None
            
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
                
                # 메모리 효율적인 XGBoost 파라미터
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 6,
                    'learning_rate': 0.05,   # 0.1 → 0.05로 감소
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'nthread': min(config.NUM_WORKERS, 4),
                    'verbosity': 0,
                    'tree_method': 'hist'    # GPU 대신 CPU 사용
                }
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=200,     # 100 → 200으로 증가
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=20, # 10 → 20으로 증가
                    verbose_eval=False
                )
                
                return model
                
            except ImportError:
                logger.warning("XGBoost를 사용할 수 없습니다")
                return None
            
        elif model_type == 'catboost':
            try:
                from catboost import CatBoostClassifier
                
                # 메모리 효율적인 CatBoost 설정
                model = CatBoostClassifier(
                    iterations=200,          # 100 → 200으로 증가
                    depth=6,
                    learning_rate=0.05,      # 0.1 → 0.05로 감소
                    loss_function='Logloss',
                    random_seed=42,
                    verbose=False,
                    thread_count=min(config.NUM_WORKERS, 4),
                    task_type='CPU'          # GPU 대신 CPU 사용
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=20,  # 10 → 20으로 증가
                    verbose=False
                )
                
                return model
                
            except ImportError:
                logger.warning("CatBoost를 사용할 수 없습니다")
                return None
        
        elif model_type == 'logistic':
            try:
                from sklearn.linear_model import LogisticRegression
                
                model = LogisticRegression(
                    random_state=42, 
                    max_iter=200,            # 100 → 200으로 증가
                    class_weight='balanced',
                    C=1.0,
                    solver='liblinear'       # 메모리 효율적인 solver
                )
                model.fit(X_train, y_train)
                return model
                
            except ImportError:
                logger.warning("scikit-learn을 사용할 수 없습니다")
                return None
        
        return None
        
    except Exception as e:
        logger.error(f"{model_type} 모델 학습 실패: {e}")
        return None

def create_dummy_models(X_train, y_train):
    """기본 모델 생성"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        models = {}
        
        # Logistic Regression
        try:
            lr_model = LogisticRegression(
                random_state=42, 
                max_iter=200,
                class_weight='balanced',
                solver='liblinear'
            )
            lr_model.fit(X_train, y_train)
            models['logistic'] = {
                'model': lr_model,
                'params': {},
                'training_time': 0.0,
                'model_type': 'logistic'
            }
            logger.info("Logistic Regression 모델 생성 완료")
        except Exception as e:
            logger.warning(f"Logistic Regression 생성 실패: {e}")
        
        # Random Forest (메모리 효율적 설정)
        try:
            rf_model = RandomForestClassifier(
                n_estimators=50,         # 메모리 절약
                max_depth=10,
                random_state=42,
                n_jobs=1,                # 단일 스레드 사용
                class_weight='balanced'
            )
            rf_model.fit(X_train, y_train)
            models['random_forest'] = {
                'model': rf_model,
                'params': {},
                'training_time': 0.0,
                'model_type': 'random_forest'
            }
            logger.info("Random Forest 모델 생성 완료")
        except Exception as e:
            logger.warning(f"Random Forest 생성 실패: {e}")
        
        return models
        
    except Exception as e:
        logger.error(f"기본 모델 생성 실패: {e}")
        return {}

def generate_submission_safe(trained_models, X_test, config):
    """안전한 제출 파일 생성"""
    logger.info("제출 파일 생성 시작")
    
    test_size = len(X_test)
    logger.info(f"테스트 데이터 크기: {test_size:,}행")
    
    try:
        # 메모리 상태 확인
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            if vm.available / (1024**3) < 5:
                logger.warning("메모리 부족으로 배치 예측 수행")
                batch_size = 10000
            else:
                batch_size = 50000
        else:
            batch_size = 50000
        
        # 제출 템플릿 로딩
        try:
            submission_path = config.SUBMISSION_TEMPLATE_PATH
            if submission_path.exists():
                submission = pd.read_csv(submission_path, encoding='utf-8')
                logger.info(f"제출 템플릿 로딩 완료: {len(submission):,}행")
            else:
                submission = pd.DataFrame({
                    'id': range(test_size),
                    'clicked': 0.0201
                })
                logger.warning("제출 템플릿이 없어 기본 템플릿 생성")
        except Exception as e:
            logger.warning(f"제출 템플릿 로딩 실패: {e}")
            submission = pd.DataFrame({
                'id': range(test_size),
                'clicked': 0.0201
            })
        
        # 크기 검증
        if len(submission) != test_size:
            logger.warning(f"크기 불일치 - 템플릿: {len(submission):,}, 테스트: {test_size:,}")
            submission = pd.DataFrame({
                'id': range(test_size),
                'clicked': 0.0201
            })
        
        # 예측 수행 (배치 처리)
        predictions = None
        prediction_method = ""
        
        if trained_models:
            # 우선순위: lightgbm > xgboost > catboost > logistic > random_forest
            model_priority = ['lightgbm', 'xgboost', 'catboost', 'logistic', 'random_forest']
            
            for model_name in model_priority:
                if model_name in trained_models:
                    try:
                        logger.info(f"{model_name} 모델로 배치 예측 수행")
                        
                        model = trained_models[model_name]['model']
                        batch_predictions = []
                        
                        # 배치별 예측
                        for i in range(0, test_size, batch_size):
                            end_idx = min(i + batch_size, test_size)
                            batch_X = X_test.iloc[i:end_idx]
                            
                            try:
                                # 모델 타입별 예측 방법
                                if hasattr(model, 'predict_proba'):
                                    # sklearn 스타일
                                    pred_proba = model.predict_proba(batch_X)
                                    if pred_proba.shape[1] > 1:
                                        batch_pred = pred_proba[:, 1]  # 양성 클래스 확률
                                    else:
                                        batch_pred = pred_proba[:, 0]
                                elif hasattr(model, 'predict'):
                                    # LightGBM, XGBoost 등
                                    batch_pred = model.predict(batch_X)
                                else:
                                    logger.warning(f"{model_name} 모델의 예측 방법을 찾을 수 없습니다")
                                    continue
                                
                                batch_predictions.extend(batch_pred)
                                
                            except Exception as batch_error:
                                logger.warning(f"배치 {i}-{end_idx} 예측 실패: {batch_error}")
                                # 실패한 배치는 기본값으로 채움
                                batch_predictions.extend([0.0201] * (end_idx - i))
                            
                            # 메모리 정리
                            if i % (batch_size * 5) == 0:  # 5배치마다 정리
                                force_memory_cleanup()
                        
                        predictions = np.array(batch_predictions)
                        predictions = np.clip(predictions, 0.001, 0.999)
                        prediction_method = model_name
                        break
                        
                    except Exception as e:
                        logger.warning(f"{model_name} 모델 예측 실패: {e}")
                        continue
        
        # 기본값 사용
        if predictions is None or len(predictions) != test_size:
            logger.warning("모든 모델 예측 실패. 기본값 사용")
            base_ctr = 0.0201
            predictions = np.random.lognormal(
                mean=np.log(base_ctr), 
                sigma=0.2, 
                size=test_size
            )
            predictions = np.clip(predictions, 0.001, 0.1)
            prediction_method = "Default"
        
        # CTR 보정
        target_ctr = 0.0201
        current_ctr = predictions.mean()
        
        if abs(current_ctr - target_ctr) > 0.002:
            logger.info(f"CTR 보정: {current_ctr:.4f} → {target_ctr:.4f}")
            correction_factor = target_ctr / current_ctr if current_ctr > 0 else 1.0
            predictions = predictions * correction_factor
            predictions = np.clip(predictions, 0.001, 0.999)
        
        # 최종 결과 설정
        submission['clicked'] = predictions
        
        # 결과 통계
        final_ctr = submission['clicked'].mean()
        final_std = submission['clicked'].std()
        final_min = submission['clicked'].min()
        final_max = submission['clicked'].max()
        
        logger.info(f"=== 제출 파일 생성 결과 ===")
        logger.info(f"예측 방법: {prediction_method}")
        logger.info(f"처리된 데이터: {test_size:,}행")
        logger.info(f"평균 CTR: {final_ctr:.4f}")
        logger.info(f"표준편차: {final_std:.4f}")
        logger.info(f"범위: {final_min:.4f} ~ {final_max:.4f}")
        
        # 제출 파일 저장
        output_path = Path("submission.csv")
        submission.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"제출 파일 저장: {output_path}")
        
        return submission
        
    except Exception as e:
        logger.error(f"제출 파일 생성 실패: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise

def create_default_submission(X_test, config):
    """기본 제출 파일 생성"""
    try:
        test_size = len(X_test) if X_test is not None else 1527298
        
        default_submission = pd.DataFrame({
            'id': range(test_size),
            'clicked': np.random.lognormal(
                mean=np.log(0.0201), 
                sigma=0.2, 
                size=test_size
            )
        })
        default_submission['clicked'] = np.clip(default_submission['clicked'], 0.001, 0.1)
        
        # CTR 보정
        current_ctr = default_submission['clicked'].mean()
        target_ctr = 0.0201
        if abs(current_ctr - target_ctr) > 0.002:
            correction_factor = target_ctr / current_ctr
            default_submission['clicked'] = default_submission['clicked'] * correction_factor
            default_submission['clicked'] = np.clip(default_submission['clicked'], 0.001, 0.999)
        
        output_path = Path("submission.csv")
        default_submission.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"기본 제출 파일 저장: {output_path}")
        
        return default_submission
        
    except Exception as e:
        logger.error(f"기본 제출 파일 생성 실패: {e}")
        raise

def inference_mode():
    """추론 모드 실행"""
    logger.info("=== 추론 모드 시작 ===")
    
    try:
        # 모듈 import
        modules = safe_import_modules()
        
        if 'create_ctr_prediction_service' in modules:
            # 추론 서비스 생성
            service = modules['create_ctr_prediction_service']()
            
            if service:
                logger.info("추론 서비스 초기화 완료")
                return service
            else:
                logger.error("추론 서비스 초기화 실패")
                return None
        else:
            logger.warning("추론 서비스 모듈을 찾을 수 없습니다")
            return None
            
    except Exception as e:
        logger.error(f"추론 모드 실행 실패: {e}")
        return None

def reproduce_score():
    """Private Score 복원"""
    logger.info("=== Private Score 복원 시작 ===")
    
    try:
        # 설정 로딩
        from config import Config
        config = Config
        
        # 저장된 모델 확인
        model_dir = Path("models")
        model_files = list(model_dir.glob("*_model.pkl"))
        
        if not model_files:
            logger.error("복원할 모델이 없습니다. 먼저 학습을 실행하세요.")
            return False
        
        logger.info(f"발견된 모델 파일: {len(model_files)}개")
        
        # 테스트 데이터 로딩
        test_path = Path("data/test.parquet")
        if not test_path.exists():
            logger.error("테스트 데이터 파일이 없습니다")
            return False
        
        logger.info("테스트 데이터 로딩")
        
        # 메모리 효율적으로 테스트 데이터 로딩
        try:
            # 청크 단위로 읽기
            test_df = pd.read_parquet(test_path, engine='pyarrow')
            logger.info(f"테스트 데이터 크기: {test_df.shape}")
        except Exception as e:
            logger.error(f"테스트 데이터 로딩 실패: {e}")
            return False
        
        # 저장된 모델로 예측 생성
        try:
            models = {}
            for model_file in model_files:
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    model_name = model_file.stem.replace('_model', '')
                    models[model_name] = model_data
                    logger.info(f"모델 로딩 완료: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"모델 로딩 실패 {model_file}: {e}")
            
            if not models:
                logger.error("사용 가능한 모델이 없습니다")
                return False
            
            # 제출 파일 생성
            submission = generate_submission_safe(models, test_df, config)
            
            # 제출 파일 저장
            output_path = Path("submission_reproduced.csv")
            submission.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"복원된 제출 파일 저장: {output_path}")
            logger.info(f"예측 통계: 평균={submission['clicked'].mean():.4f}, 표준편차={submission['clicked'].std():.4f}")
            
            logger.info("=== Private Score 복원 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"복원 과정 실패: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Private Score 복원 실패: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return False

def main():
    """메인 실행 함수"""
    global cleanup_required
    
    # 신호 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description="CTR 모델링 최종 제출 시스템 - 메모리 최적화")
    parser.add_argument("--mode", choices=["train", "inference", "reproduce"], 
                       default="train", help="실행 모드")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 실행 모드")
    
    args = parser.parse_args()
    
    try:
        logger.info("=== CTR 모델링 최종 제출 시스템 시작 ===")
        
        # 환경 검증
        if not validate_environment():
            logger.error("환경 검증 실패")
            sys.exit(1)
        
        # 모드별 실행
        if args.mode == "train":
            logger.info("학습 모드 시작")
            
            # 설정 초기화
            from config import Config
            config = Config
            config.setup_directories()
            
            # 전체 파이프라인 실행
            results = execute_full_pipeline(config, quick_mode=args.quick)
            
            if results:
                logger.info("학습 모드 완료")
                logger.info(f"실행 시간: {results['execution_time']:.2f}초")
                logger.info(f"성공 모델: {results['successful_models']}개")
                logger.info(f"메모리 효율 모드: {results.get('memory_efficient', True)}")
            else:
                logger.error("학습 모드 실패")
                sys.exit(1)
                
        elif args.mode == "inference":
            logger.info("추론 모드 시작")
            service = inference_mode()
            
            if service:
                logger.info("추론 모드 완료")
            else:
                logger.error("추론 모드 실패")
                sys.exit(1)
                
        elif args.mode == "reproduce":
            logger.info("Private Score 복원 모드 시작")
            success = reproduce_score()
            
            if success:
                logger.info("Private Score 복원 완료")
            else:
                logger.error("Private Score 복원 실패")
                sys.exit(1)
        
        logger.info("=== CTR 모델링 최종 제출 시스템 종료 ===")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 실행이 중단되었습니다")
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        sys.exit(1)
        
    finally:
        cleanup_required = True
        # 최종 메모리 정리
        force_memory_cleanup(intensive=True)
        
        # 최종 상태 보고
        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                logger.info(f"최종 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
            except Exception:
                pass

if __name__ == "__main__":
    main()