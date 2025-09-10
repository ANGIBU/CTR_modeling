# main.py

import argparse
import logging
import time
import json
import gc
import psutil
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from config import Config
from data_loader import DataLoader, DataValidator
from feature_engineering import FeatureEngineer
from training import ModelTrainer, TrainingPipeline
from evaluation import CTRMetrics, ModelComparator, EvaluationReporter
from ensemble import EnsembleManager
from inference import CTRPredictionAPI
from models import ModelFactory

def get_memory_usage() -> float:
    """현재 메모리 사용량 (GB)"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def setup_logging():
    """로깅 설정"""
    logger = Config.setup_logging()
    logger.info("=== 파이프라인 시작 ===")
    return logger

def memory_monitor_decorator(func):
    """메모리 모니터링 데코레이터"""
    def wrapper(*args, **kwargs):
        memory_before = get_memory_usage()
        logger = logging.getLogger(__name__)
        logger.info(f"{func.__name__} 시작 - 메모리: {memory_before:.2f} GB")
        
        result = func(*args, **kwargs)
        
        memory_after = get_memory_usage()
        logger.info(f"{func.__name__} 완료 - 메모리: {memory_after:.2f} GB (변화: {memory_after-memory_before:+.2f} GB)")
        
        return result
    return wrapper

@memory_monitor_decorator
def load_and_preprocess_data(config: Config, memory_limit: float = 50.0) -> tuple:
    """메모리 모니터링과 함께 데이터 로딩 및 전처리"""
    logger = logging.getLogger(__name__)
    logger.info("데이터 로딩 및 전처리 단계 시작")
    
    data_loader = DataLoader(config)
    
    try:
        # 데이터 로딩
        train_df, test_df = data_loader.load_data()
        
        # 메모리 체크
        memory_usage = get_memory_usage()
        logger.info(f"데이터 로딩 후 메모리: {memory_usage:.2f} GB")
        
        if memory_usage > memory_limit:
            logger.warning(f"메모리 사용량이 한계 초과: {memory_usage:.2f} GB > {memory_limit} GB")
            
            # 데이터 샘플링으로 메모리 절약
            original_train_size = len(train_df)
            original_test_size = len(test_df)
            
            sample_ratio = min(0.7, memory_limit / memory_usage)
            
            train_sample_size = int(len(train_df) * sample_ratio)
            test_sample_size = int(len(test_df) * sample_ratio)
            
            train_df = train_df.sample(n=train_sample_size, random_state=42)
            test_df = test_df.sample(n=test_sample_size, random_state=42)
            
            logger.info(f"메모리 절약을 위한 샘플링 수행")
            logger.info(f"학습 데이터: {original_train_size:,} → {len(train_df):,}")
            logger.info(f"테스트 데이터: {original_test_size:,} → {len(test_df):,}")
            
            # 메모리 정리
            gc.collect()
        
        # 데이터 요약 정보
        train_summary = data_loader.get_data_summary(train_df)
        test_summary = data_loader.get_data_summary(test_df)
        
        logger.info(f"학습 데이터 요약: {train_summary['shape']}")
        logger.info(f"테스트 데이터 요약: {test_summary['shape']}")
        
        if 'target_distribution' in train_summary:
            ctr = train_summary['target_distribution']['ctr']
            logger.info(f"전체 CTR: {ctr:.4f}")
        
        # 데이터 검증
        validator = DataValidator()
        validation_results = validator.validate_data_consistency(train_df, test_df)
        
        if validation_results['missing_in_test'] or validation_results['dtype_mismatches']:
            logger.warning(f"데이터 일관성 문제 발견: {validation_results}")
        
        # 기본 전처리
        train_processed = data_loader.basic_preprocessing(train_df)
        test_processed = data_loader.basic_preprocessing(test_df)
        
        # 메모리 정리
        del train_df, test_df
        gc.collect()
        
        logger.info("데이터 로딩 및 전처리 완료")
        return train_processed, test_processed, data_loader
        
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {str(e)}")
        raise

@memory_monitor_decorator
def feature_engineering_pipeline(train_df: pd.DataFrame, 
                                test_df: pd.DataFrame,
                                config: Config,
                                memory_limit: float = 55.0) -> tuple:
    """메모리 제한을 고려한 피처 엔지니어링 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("피처 엔지니어링 단계 시작")
    
    try:
        # 메모리 체크
        initial_memory = get_memory_usage()
        logger.info(f"피처 엔지니어링 시작 메모리: {initial_memory:.2f} GB")
        
        if initial_memory > memory_limit * 0.8:
            logger.warning(f"높은 메모리 사용량으로 인한 제한적 피처 엔지니어링 수행")
        
        # 피처 엔지니어 초기화
        feature_engineer = FeatureEngineer(config)
        
        # 피처 생성 (메모리 최적화됨)
        X_train, X_test = feature_engineer.create_all_features(
            train_df, test_df, target_col='clicked'
        )
        
        # 타겟 변수 분리
        y_train = train_df['clicked']
        
        # 피처 엔지니어링 결과 검증
        logger.info("피처 엔지니어링 결과 검증 시작")
        
        # 최종 데이터 검증 및 정리
        X_train, X_test = validate_and_clean_final_data(X_train, X_test)
        
        # 메모리 정리
        del train_df, test_df
        gc.collect()
        
        # 피처 정보 출력
        feature_summary = feature_engineer.get_feature_importance_summary()
        logger.info(f"생성된 피처 수: {feature_summary['total_generated_features']}")
        logger.info(f"최종 피처 차원: {X_train.shape}")
        
        # 최종 메모리 체크
        final_memory = get_memory_usage()
        logger.info(f"피처 엔지니어링 완료 메모리: {final_memory:.2f} GB")
        
        if final_memory > memory_limit:
            logger.error(f"메모리 한계 초과: {final_memory:.2f} GB > {memory_limit} GB")
            raise MemoryError("메모리 부족으로 인한 파이프라인 중단")
        
        logger.info("피처 엔지니어링 완료")
        return X_train, X_test, y_train, feature_engineer
        
    except Exception as e:
        logger.error(f"피처 엔지니어링 실패: {str(e)}")
        # 메모리 정리
        gc.collect()
        raise

def validate_and_clean_final_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """최종 데이터 검증 및 정리"""
    logger = logging.getLogger(__name__)
    logger.info("최종 데이터 검증 및 정리 시작")
    
    # 1. 컬럼명 검증 (None 값 제거)
    valid_columns = [col for col in X_train.columns if col is not None and str(col).strip() != '']
    if len(valid_columns) != len(X_train.columns):
        logger.warning(f"잘못된 컬럼명 발견: {len(X_train.columns) - len(valid_columns)}개")
        X_train = X_train[valid_columns]
        X_test = X_test[valid_columns]
    
    # 2. object 타입 컬럼 강제 제거
    object_cols_train = X_train.select_dtypes(include=['object']).columns.tolist()
    object_cols_test = X_test.select_dtypes(include=['object']).columns.tolist()
    all_object_cols = list(set(object_cols_train + object_cols_test))
    
    if all_object_cols:
        logger.warning(f"object 타입 컬럼 발견하여 제거: {all_object_cols}")
        X_train = X_train.drop(columns=all_object_cols, errors='ignore')
        X_test = X_test.drop(columns=all_object_cols, errors='ignore')
    
    # 3. 데이터 타입 강제 통일
    for col in X_train.columns:
        try:
            # 문자열이나 카테고리 타입을 수치형으로 변환
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                logger.warning(f"{col}: {X_train[col].dtype} → 수치형 변환")
                
                # 카테고리 타입인 경우 코드로 변환
                if X_train[col].dtype.name == 'category':
                    X_train[col] = X_train[col].cat.codes.astype('float32')
                    X_test[col] = X_test[col].cat.codes.astype('float32')
                else:
                    # 문자열을 수치형으로 변환 시도
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
            
            # 다른 수치형 타입을 float32로 통일
            elif X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
                X_train[col] = X_train[col].astype('float32')
                X_test[col] = X_test[col].astype('float32')
            elif X_train[col].dtype == 'float64':
                X_train[col] = X_train[col].astype('float32')
                X_test[col] = X_test[col].astype('float32')
                
        except Exception as e:
            logger.warning(f"{col} 데이터 타입 변환 실패: {str(e)}")
            # 변환 실패한 컬럼은 제거
            X_train = X_train.drop(columns=[col], errors='ignore')
            X_test = X_test.drop(columns=[col], errors='ignore')
    
    # 4. 결측치 및 무한값 처리
    if X_train.isnull().any().any():
        logger.warning("X_train에 결측치 발견, 0으로 대치")
        X_train = X_train.fillna(0)
    
    if X_test.isnull().any().any():
        logger.warning("X_test에 결측치 발견, 0으로 대치")
        X_test = X_test.fillna(0)
    
    # 무한값 처리
    try:
        # 수치형 컬럼만 선택
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # X_train 무한값 처리
            X_train[numeric_columns] = X_train[numeric_columns].replace([np.inf, -np.inf], [1e6, -1e6])
            X_test[numeric_columns] = X_test[numeric_columns].replace([np.inf, -np.inf], [1e6, -1e6])
        
    except Exception as e:
        logger.warning(f"무한값 처리 중 오류: {str(e)}")
        # 안전한 방식으로 무한값 처리
        X_train = X_train.replace([np.inf, -np.inf], [1e6, -1e6])
        X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
    
    # 5. 최종 데이터 타입 확인
    logger.info(f"X_train 최종 데이터 타입 분포: {X_train.dtypes.value_counts().to_dict()}")
    logger.info(f"X_test 최종 데이터 타입 분포: {X_test.dtypes.value_counts().to_dict()}")
    
    # 6. object 타입이 남아있는지 최종 확인
    final_object_cols_train = X_train.select_dtypes(include=['object']).columns.tolist()
    final_object_cols_test = X_test.select_dtypes(include=['object']).columns.tolist()
    
    if final_object_cols_train or final_object_cols_test:
        logger.error(f"최종 검증에서 object 타입 발견! train: {final_object_cols_train}, test: {final_object_cols_test}")
        all_remaining_object_cols = list(set(final_object_cols_train + final_object_cols_test))
        X_train = X_train.drop(columns=all_remaining_object_cols, errors='ignore')
        X_test = X_test.drop(columns=all_remaining_object_cols, errors='ignore')
        logger.info(f"남은 object 컬럼 강제 제거: {all_remaining_object_cols}")
    
    logger.info("최종 데이터 검증 및 정리 완료")
    return X_train, X_test

@memory_monitor_decorator
def model_training_pipeline(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           config: Config,
                           tune_hyperparameters: bool = True,
                           memory_safe_mode: bool = False) -> tuple:
    """메모리 안전 모드를 포함한 모델 학습 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("모델 학습 단계 시작")
    
    try:
        # 메모리 체크
        memory_usage = get_memory_usage()
        
        if memory_usage > 45 or memory_safe_mode:
            logger.info("메모리 안전 모드로 학습 수행")
            
            # 데이터 크기 줄이기
            if len(X_train) > 5000000:
                sample_size = 3000000
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train.iloc[sample_indices]
                y_train = y_train.iloc[sample_indices]
                logger.info(f"메모리 절약을 위한 학습 데이터 샘플링: {sample_size:,}개")
            
            # 하이퍼파라미터 튜닝 비활성화
            tune_hyperparameters = False
            logger.info("메모리 절약을 위해 하이퍼파라미터 튜닝 비활성화")
        
        # 학습 파이프라인 초기화
        training_pipeline = TrainingPipeline(config)
        
        # 데이터 분할
        data_loader = DataLoader(config)
        X_train_split, X_val_split, y_train_split, y_val_split = \
            data_loader.train_test_split_data(
                pd.concat([X_train, y_train], axis=1)
            )
        
        # 메모리 정리
        del X_train
        gc.collect()
        
        # 전체 학습 파이프라인 실행
        pipeline_results = training_pipeline.run_full_pipeline(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            tune_hyperparameters=tune_hyperparameters,
            n_trials=30 if tune_hyperparameters else 0
        )
        
        # 결과 출력
        logger.info(f"학습 완료된 모델 수: {pipeline_results['model_count']}")
        if 'best_model' in pipeline_results:
            best = pipeline_results['best_model']
            logger.info(f"최고 성능 모델: {best['name']} (점수: {best['score']:.4f})")
        
        logger.info("모델 학습 완료")
        return training_pipeline.trainer, X_val_split, y_val_split
        
    except Exception as e:
        logger.error(f"모델 학습 실패: {str(e)}")
        gc.collect()
        raise

@memory_monitor_decorator
def ensemble_pipeline(trainer: ModelTrainer,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     config: Config,
                     memory_safe_mode: bool = False) -> EnsembleManager:
    """메모리 안전 모드 앙상블 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("앙상블 단계 시작")
    
    try:
        # 메모리 체크
        memory_usage = get_memory_usage()
        
        # 앙상블 매니저 초기화
        ensemble_manager = EnsembleManager(config)
        
        # 학습된 모델들 추가
        for model_name, model_info in trainer.trained_models.items():
            ensemble_manager.add_base_model(model_name, model_info['model'])
        
        # 메모리 안전 모드에서는 제한적 앙상블만 생성
        if memory_usage > 50 or memory_safe_mode:
            ensemble_types = ['weighted']
            logger.info("메모리 절약을 위해 가중 블렌딩 앙상블만 생성")
        else:
            ensemble_types = ['weighted', 'rank']
        
        for ensemble_type in ensemble_types:
            try:
                ensemble_manager.create_ensemble(ensemble_type)
                logger.info(f"{ensemble_type} 앙상블 생성 완료")
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 생성 실패: {str(e)}")
        
        # 앙상블 학습
        ensemble_manager.train_all_ensembles(X_val, y_val)
        
        # 앙상블 평가
        ensemble_results = ensemble_manager.evaluate_ensembles(X_val, y_val)
        
        # 결과 출력
        for name, score in ensemble_results.items():
            logger.info(f"{name}: {score:.4f}")
        
        # 앙상블 저장
        ensemble_manager.save_ensembles()
        
        logger.info("앙상블 파이프라인 완료")
        return ensemble_manager
        
    except Exception as e:
        logger.error(f"앙상블 파이프라인 실패: {str(e)}")
        gc.collect()
        raise

def safe_pipeline_execution(args, config: Config, logger):
    """안전한 파이프라인 실행 (메모리 모니터링 포함)"""
    
    memory_limit = 55.0
    memory_safe_mode = False
    
    try:
        # 1. 데이터 로딩 및 전처리
        train_df, test_df, data_loader = load_and_preprocess_data(config, memory_limit)
        
        # 메모리 체크
        current_memory = get_memory_usage()
        if current_memory > memory_limit * 0.7:
            memory_safe_mode = True
            logger.warning(f"메모리 안전 모드 활성화: {current_memory:.2f} GB")
        
        # 빠른 실행 모드 (샘플링)
        if args.quick or memory_safe_mode:
            logger.info("빠른 실행 모드: 데이터 샘플링")
            train_sample_size = min(100000, len(train_df))
            test_sample_size = min(50000, len(test_df))
            
            train_df = train_df.sample(n=train_sample_size, random_state=42)
            test_df = test_df.sample(n=test_sample_size, random_state=42)
            
            logger.info(f"샘플링 크기 - 학습: {train_sample_size:,}, 테스트: {test_sample_size:,}")
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train, feature_engineer = feature_engineering_pipeline(
            train_df, test_df, config, memory_limit
        )
        
        # 메모리 정리
        del train_df, test_df
        gc.collect()
        
        # 3. 모델 학습
        trainer, X_val, y_val = model_training_pipeline(
            X_train, y_train, config, 
            tune_hyperparameters=not args.no_tune and not memory_safe_mode,
            memory_safe_mode=memory_safe_mode
        )
        
        # 4. 앙상블 (메모리 여유가 있을 때만)
        ensemble_manager = None
        if get_memory_usage() < memory_limit * 0.8:
            ensemble_manager = ensemble_pipeline(trainer, X_val, y_val, config, memory_safe_mode)
        else:
            logger.warning("메모리 부족으로 앙상블 단계 생략")
        
        # 5. 평가
        evaluation_pipeline(trainer, X_val, y_val, config)
        
        # 6. 최종 예측 (데이터 검증 강화)
        submission = generate_predictions(trainer, ensemble_manager, X_test, config)
        
        # 7. 추론 시스템 설정 (메모리 여유가 있을 때만)
        if get_memory_usage() < memory_limit * 0.9:
            prediction_api = inference_setup(config)
        else:
            logger.warning("메모리 부족으로 추론 시스템 설정 생략")
            prediction_api = None
        
        return {
            'trainer': trainer,
            'ensemble_manager': ensemble_manager,
            'prediction_api': prediction_api,
            'submission': submission
        }
        
    except MemoryError as e:
        logger.error(f"메모리 부족 오류: {str(e)}")
        logger.info("메모리 안전 모드로 재시도를 권장합니다.")
        raise
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류: {str(e)}")
        raise

def generate_predictions(trainer: ModelTrainer,
                        ensemble_manager: EnsembleManager,
                        X_test: pd.DataFrame,
                        config: Config) -> pd.DataFrame:
    """최종 예측 생성 (데이터 검증 강화)"""
    logger = logging.getLogger(__name__)
    logger.info("최종 예측 생성 시작")
    
    try:
        # 제출 템플릿 로딩
        data_loader = DataLoader(config)
        submission = data_loader.load_submission_template()
        
        # X_test 최종 검증
        logger.info("예측 전 X_test 데이터 검증 시작")
        
        # object 타입 컬럼 확인 및 제거
        object_columns = X_test.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            logger.error(f"X_test에 object 타입 컬럼 발견! 제거 진행: {object_columns}")
            X_test = X_test.drop(columns=object_columns)
        
        # 데이터 타입 재확인
        logger.info(f"예측 전 X_test 데이터 타입: {X_test.dtypes.value_counts().to_dict()}")
        
        # 모든 컬럼이 수치형인지 확인
        non_numeric_columns = []
        for col in X_test.columns:
            if not np.issubdtype(X_test[col].dtype, np.number):
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            logger.error(f"비수치형 컬럼 발견하여 제거: {non_numeric_columns}")
            X_test = X_test.drop(columns=non_numeric_columns)
        
        # 결측치 및 무한값 최종 처리
        if X_test.isnull().any().any():
            logger.warning("예측 전 X_test 결측치 처리")
            X_test = X_test.fillna(0)
        
        if np.isinf(X_test.values).any():
            logger.warning("예측 전 X_test 무한값 처리")
            X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 데이터 타입을 float32로 통일
        for col in X_test.columns:
            if X_test[col].dtype != 'float32':
                X_test[col] = X_test[col].astype('float32')
        
        logger.info(f"검증 완료 - X_test 형태: {X_test.shape}")
        logger.info(f"검증 완료 - X_test 데이터 타입: {X_test.dtypes.value_counts().to_dict()}")
        
        # 앙상블 예측 사용 (가능한 경우)
        if ensemble_manager and ensemble_manager.best_ensemble is not None:
            logger.info("앙상블 모델로 예측 수행")
            predictions = ensemble_manager.predict_with_best_ensemble(X_test)
        else:
            # 최고 성능 단일 모델 사용
            logger.info("단일 모델로 예측 수행")
            best_model_name = None
            best_score = 0
            
            for model_name, results in trainer.cv_results.items():
                if results['combined_mean'] > best_score:
                    best_score = results['combined_mean']
                    best_model_name = model_name
            
            if best_model_name:
                best_model = trainer.trained_models[best_model_name]['model']
                logger.info(f"사용할 모델: {best_model_name}")
                predictions = best_model.predict_proba(X_test)
            else:
                raise ValueError("사용 가능한 모델이 없습니다.")
        
        # 제출 파일 생성
        submission['clicked'] = predictions
        
        # 예측값 범위 확인
        logger.info(f"예측값 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
        logger.info(f"예측 평균: {predictions.mean():.4f}")
        
        # 제출 파일 저장
        output_path = config.OUTPUT_DIR / "submission.csv"
        submission.to_csv(output_path, index=False)
        logger.info(f"제출 파일 저장: {output_path}")
        
        logger.info("최종 예측 생성 완료")
        return submission
        
    except Exception as e:
        logger.error(f"예측 생성 실패: {str(e)}")
        logger.error(f"X_test 정보 - 형태: {X_test.shape if 'X_test' in locals() else 'N/A'}")
        if 'X_test' in locals():
            logger.error(f"X_test 데이터 타입: {X_test.dtypes.value_counts().to_dict()}")
            logger.error(f"X_test 컬럼 샘플: {list(X_test.columns[:10])}")
        raise

def evaluation_pipeline(trainer: ModelTrainer,
                       X_val: pd.DataFrame,
                       y_val: pd.Series,
                       config: Config):
    """평가 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("평가 단계 시작")
    
    try:
        # 모델별 예측 수집
        models_predictions = {}
        
        for model_name, model_info in trainer.trained_models.items():
            model = model_info['model']
            pred = model.predict_proba(X_val)
            models_predictions[model_name] = pred
        
        # 모델 비교
        comparator = ModelComparator()
        comparison_df = comparator.compare_models(models_predictions, y_val)
        
        logger.info("모델 성능 비교:")
        logger.info(f"\n{comparison_df[['combined_score', 'ap', 'wll', 'auc', 'f1']].round(4)}")
        
        # 종합 평가 보고서 생성 (메모리 제한으로 차트 생성은 스킵)
        reporter = EvaluationReporter()
        report = reporter.generate_comprehensive_report(
            models_predictions, y_val,
            output_dir=None
        )
        
        # 보고서 저장
        report_path = config.OUTPUT_DIR / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"평가 보고서 저장: {report_path}")
        logger.info("평가 파이프라인 완료")
        
    except Exception as e:
        logger.error(f"평가 파이프라인 실패: {str(e)}")
        raise

def inference_setup(config: Config) -> CTRPredictionAPI:
    """추론 시스템 설정"""
    logger = logging.getLogger(__name__)
    logger.info("추론 시스템 설정 시작")
    
    try:
        # CTR 예측 API 초기화
        prediction_api = CTRPredictionAPI(config)
        
        # 모델 로딩
        success = prediction_api.initialize()
        
        if success:
            # 상태 확인
            status = prediction_api.get_api_status()
            logger.info(f"추론 시스템 상태: {status['engine_status']['system_status']}")
            logger.info(f"로딩된 모델 수: {status['engine_status']['models_count']}")
            
            # 테스트 예측
            test_result = prediction_api.predict_ctr(
                user_id="test_user",
                ad_id="test_ad",
                context={"device": "mobile", "page": "home"}
            )
            
            logger.info(f"테스트 예측 결과: {test_result['ctr_prediction']:.4f}")
            logger.info("추론 시스템 설정 완료")
            
            return prediction_api
        else:
            logger.error("추론 시스템 초기화 실패")
            return None
            
    except Exception as e:
        logger.error(f"추론 시스템 설정 실패: {str(e)}")
        return None

def main():
    """메인 실행 함수"""
    
    # 인자 파싱
    parser = argparse.ArgumentParser(description="CTR 모델 파이프라인")
    parser.add_argument("--mode", choices=["train", "inference", "evaluate"], 
                       default="train", help="실행 모드")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--no-tune", action="store_true", 
                       help="하이퍼파라미터 튜닝 비활성화")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 실행 (샘플 데이터)")
    parser.add_argument("--memory-safe", action="store_true",
                       help="메모리 안전 모드")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    
    # 설정 초기화
    config = Config
    config.setup_directories()
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 초기 메모리 상태 확인
    initial_memory = get_memory_usage()
    logger.info(f"시작 메모리 사용량: {initial_memory:.2f} GB")
    
    try:
        if args.mode == "train":
            # 전체 학습 파이프라인 실행
            logger.info("=== 학습 모드 시작 ===")
            
            # 안전한 파이프라인 실행
            results = safe_pipeline_execution(args, config, logger)
            
        elif args.mode == "inference":
            # 추론 모드
            logger.info("=== 추론 모드 시작 ===")
            prediction_api = inference_setup(config)
            
            if prediction_api:
                logger.info("추론 API 준비 완료")
            
        elif args.mode == "evaluate":
            # 평가 모드
            logger.info("=== 평가 모드 시작 ===")
            
            # 저장된 모델 로딩 및 평가
            trainer = ModelTrainer(config)
            loaded_models = trainer.load_models()
            
            if loaded_models:
                logger.info(f"로딩된 모델: {list(loaded_models.keys())}")
            else:
                logger.error("로딩할 모델이 없습니다.")
        
        # 실행 시간 및 메모리 사용량 출력
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        logger.info(f"전체 파이프라인 실행 시간: {total_time:.2f}초")
        logger.info(f"최대 메모리 사용량: {final_memory:.2f} GB")
        logger.info(f"메모리 증가량: {memory_increase:+.2f} GB")
        logger.info("=== 파이프라인 완료 ===")
        
    except MemoryError as e:
        logger.error(f"메모리 부족으로 파이프라인 중단: {str(e)}")
        logger.info("해결 방안:")
        logger.info("1. --quick 옵션 사용")
        logger.info("2. --memory-safe 옵션 사용")
        logger.info("3. 시스템 메모리 증설")
        logger.info("4. 데이터 크기 축소")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}")
        raise
    
    finally:
        # 정리 작업
        gc.collect()
        final_memory = get_memory_usage()
        logger.info(f"정리 후 메모리: {final_memory:.2f} GB")
        logger.info("파이프라인 종료")

if __name__ == "__main__":
    main()