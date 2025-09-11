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
from ensemble import CTREnsembleManager
from inference import CTRPredictionAPI
from models import ModelFactory

def get_memory_usage() -> float:
    """현재 메모리 사용량 (GB)"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def get_available_memory() -> float:
    """사용 가능한 메모리 (GB)"""
    return psutil.virtual_memory().available / (1024**3)

def force_memory_cleanup():
    """강제 메모리 정리"""
    gc.collect()
    import ctypes
    try:
        ctypes.CDLL("kernel32.dll").SetProcessWorkingSetSize(-1, -1, -1)
    except:
        pass

def setup_logging():
    """로깅 설정"""
    logger = Config.setup_logging()
    logger.info("=== CTR 모델 파이프라인 시작 ===")
    return logger

def memory_monitor_decorator(func):
    """메모리 모니터링 데코레이터"""
    def wrapper(*args, **kwargs):
        memory_before = get_memory_usage()
        available_before = get_available_memory()
        logger = logging.getLogger(__name__)
        logger.info(f"{func.__name__} 시작 - 메모리: {memory_before:.2f} GB, 사용 가능: {available_before:.2f} GB")
        
        result = func(*args, **kwargs)
        
        memory_after = get_memory_usage()
        available_after = get_available_memory()
        logger.info(f"{func.__name__} 완료 - 메모리: {memory_after:.2f} GB, 사용 가능: {available_after:.2f} GB")
        
        # 메모리 정리
        if available_after < 10:
            force_memory_cleanup()
        
        return result
    return wrapper

@memory_monitor_decorator
def load_and_preprocess_data(config: Config, quick_mode: bool = False, memory_safe: bool = False) -> tuple:
    """메모리 효율적인 데이터 로딩 및 전처리"""
    logger = logging.getLogger(__name__)
    logger.info("데이터 로딩 및 전처리 단계 시작")
    
    data_loader = DataLoader(config)
    
    try:
        # 메모리 기반 샘플 크기 결정
        available_memory = get_available_memory()
        
        if quick_mode:
            # 빠른 모드: 매우 작은 샘플
            max_train_size = 50000
            max_test_size = 10000
            logger.info(f"빠른 모드 활성화: 학습 {max_train_size:,}, 테스트 {max_test_size:,}")
        elif memory_safe or available_memory < 20:
            # 메모리 안전 모드
            max_train_size = 200000
            max_test_size = 50000
            logger.info(f"메모리 안전 모드: 학습 {max_train_size:,}, 테스트 {max_test_size:,}")
        else:
            # 일반 모드
            max_train_size = 1000000
            max_test_size = 300000
            logger.info(f"일반 모드: 학습 {max_train_size:,}, 테스트 {max_test_size:,}")
        
        # 청킹 기반 데이터 로딩
        train_df, test_df = data_loader.load_data_chunked(
            max_train_size=max_train_size,
            max_test_size=max_test_size
        )
        
        # 메모리 체크
        current_memory = get_memory_usage()
        available_memory = get_available_memory()
        logger.info(f"데이터 로딩 후 - 사용 메모리: {current_memory:.2f} GB, 사용 가능: {available_memory:.2f} GB")
        
        # 메모리 부족 시 추가 샘플링
        if available_memory < 15:
            logger.warning(f"메모리 부족으로 추가 샘플링 수행")
            sample_ratio = min(0.5, available_memory / 20)
            
            original_train_size = len(train_df)
            original_test_size = len(test_df)
            
            train_sample_size = max(10000, int(len(train_df) * sample_ratio))
            test_sample_size = max(5000, int(len(test_df) * sample_ratio))
            
            train_df = train_df.sample(n=train_sample_size, random_state=42).reset_index(drop=True)
            test_df = test_df.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
            
            logger.info(f"추가 샘플링 - 학습: {original_train_size:,} → {len(train_df):,}")
            logger.info(f"추가 샘플링 - 테스트: {original_test_size:,} → {len(test_df):,}")
            
            force_memory_cleanup()
        
        # 데이터 요약 정보
        train_summary = data_loader.get_data_summary(train_df)
        test_summary = data_loader.get_data_summary(test_df)
        
        logger.info(f"최종 학습 데이터: {train_summary['shape']}")
        logger.info(f"최종 테스트 데이터: {test_summary['shape']}")
        
        if 'target_distribution' in train_summary:
            ctr = train_summary['target_distribution']['ctr']
            logger.info(f"실제 CTR: {ctr:.4f}")
        
        # 데이터 검증
        validator = DataValidator()
        validation_results = validator.validate_data_consistency(train_df, test_df)
        
        if validation_results['missing_in_test'] or validation_results['dtype_mismatches']:
            logger.warning(f"데이터 일관성 문제: {len(validation_results['missing_in_test'])}개 누락, {len(validation_results['dtype_mismatches'])}개 타입 불일치")
        
        # 기본 전처리
        train_processed = data_loader.basic_preprocessing(train_df)
        test_processed = data_loader.basic_preprocessing(test_df)
        
        # 메모리 정리
        del train_df, test_df
        force_memory_cleanup()
        
        logger.info("데이터 로딩 및 전처리 완료")
        return train_processed, test_processed, data_loader
        
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {str(e)}")
        force_memory_cleanup()
        raise

@memory_monitor_decorator
def feature_engineering_pipeline(train_df: pd.DataFrame, 
                                test_df: pd.DataFrame,
                                config: Config,
                                memory_limit: bool = False) -> tuple:
    """메모리 효율적인 피처 엔지니어링"""
    logger = logging.getLogger(__name__)
    logger.info("피처 엔지니어링 단계 시작")
    
    try:
        available_memory = get_available_memory()
        logger.info(f"피처 엔지니어링 시작 - 사용 가능 메모리: {available_memory:.2f} GB")
        
        # 메모리 기반 피처 생성 제한
        if available_memory < 15 or memory_limit:
            logger.info("제한적 피처 엔지니어링 모드")
            feature_engineer = FeatureEngineer(config)
            feature_engineer.set_memory_efficient_mode(True)
        else:
            feature_engineer = FeatureEngineer(config)
        
        # 피처 생성
        X_train, X_test = feature_engineer.create_all_features(
            train_df, test_df, target_col='clicked'
        )
        
        # 타겟 변수 분리
        y_train = train_df['clicked'].copy()
        
        # 메모리 정리
        del train_df, test_df
        force_memory_cleanup()
        
        # 피처 정보
        feature_summary = feature_engineer.get_feature_importance_summary()
        logger.info(f"생성된 피처 수: {feature_summary['total_generated_features']}")
        logger.info(f"최종 피처 차원: {X_train.shape}")
        
        available_memory = get_available_memory()
        logger.info(f"피처 엔지니어링 완료 - 사용 가능 메모리: {available_memory:.2f} GB")
        
        return X_train, X_test, y_train, feature_engineer
        
    except Exception as e:
        logger.error(f"피처 엔지니어링 실패: {str(e)}")
        force_memory_cleanup()
        raise

@memory_monitor_decorator
def model_training_pipeline(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           config: Config,
                           tune_hyperparameters: bool = True,
                           memory_safe_mode: bool = False) -> tuple:
    """메모리 효율적인 모델 학습"""
    logger = logging.getLogger(__name__)
    logger.info("모델 학습 단계 시작")
    
    try:
        available_memory = get_available_memory()
        
        # 메모리 기반 학습 모드 결정
        if available_memory < 15 or memory_safe_mode:
            logger.info("메모리 안전 모드로 학습 수행")
            
            # 데이터 크기 제한
            if len(X_train) > 100000:
                sample_size = 80000
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train.iloc[sample_indices].reset_index(drop=True)
                y_train = y_train.iloc[sample_indices].reset_index(drop=True)
                logger.info(f"메모리 절약을 위한 학습 데이터 샘플링: {sample_size:,}개")
            
            tune_hyperparameters = False
            logger.info("메모리 절약을 위해 하이퍼파라미터 튜닝 비활성화")
        
        # 학습 파이프라인 초기화
        training_pipeline = TrainingPipeline(config)
        
        # 메모리 효율적인 데이터 분할
        data_loader = DataLoader(config)
        split_result = data_loader.memory_efficient_train_test_split(X_train, y_train)
        X_train_split, X_val_split, y_train_split, y_val_split = split_result
        
        # 메모리 정리
        del X_train, y_train
        force_memory_cleanup()
        
        # 전체 학습 파이프라인 실행
        pipeline_results = training_pipeline.run_full_pipeline(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            tune_hyperparameters=tune_hyperparameters,
            n_trials=20 if tune_hyperparameters else 0
        )
        
        logger.info(f"학습 완료된 모델 수: {pipeline_results['model_count']}")
        if 'best_model' in pipeline_results:
            best = pipeline_results['best_model']
            logger.info(f"최고 성능 모델: {best['name']} (Combined Score: {best['score']:.4f})")
        
        return training_pipeline.trainer, X_val_split, y_val_split
        
    except Exception as e:
        logger.error(f"모델 학습 실패: {str(e)}")
        force_memory_cleanup()
        raise

@memory_monitor_decorator
def ensemble_pipeline(trainer: ModelTrainer,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     config: Config) -> CTREnsembleManager:
    """메모리 효율적인 앙상블"""
    logger = logging.getLogger(__name__)
    logger.info("앙상블 단계 시작")
    
    try:
        available_memory = get_available_memory()
        
        ensemble_manager = CTREnsembleManager(config)
        
        # 학습된 모델들 추가
        for model_name, model_info in trainer.trained_models.items():
            ensemble_manager.add_base_model(model_name, model_info['model'])
        
        # 메모리 상태에 따른 앙상블 타입 선택
        if available_memory > 10:
            ensemble_types = ['weighted', 'rank']
        else:
            ensemble_types = ['weighted']
            logger.info("메모리 절약을 위해 가중 블렌딩만 생성")
        
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
        
        for name, score in ensemble_results.items():
            logger.info(f"{name}: {score:.4f}")
        
        # 앙상블 저장
        ensemble_manager.save_ensembles()
        
        return ensemble_manager
        
    except Exception as e:
        logger.error(f"앙상블 파이프라인 실패: {str(e)}")
        force_memory_cleanup()
        raise

def safe_pipeline_execution(args, config: Config, logger):
    """안전한 파이프라인 실행"""
    
    try:
        # 초기 메모리 상태 확인
        available_memory = get_available_memory()
        logger.info(f"시작 시 사용 가능 메모리: {available_memory:.2f} GB")
        
        # 메모리 안전 모드 결정
        memory_safe_mode = args.memory_safe or available_memory < 25
        if memory_safe_mode:
            logger.info("메모리 안전 모드 활성화")
        
        # 1. 데이터 로딩 및 전처리
        train_df, test_df, data_loader = load_and_preprocess_data(
            config, 
            quick_mode=args.quick,
            memory_safe=memory_safe_mode
        )
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train, feature_engineer = feature_engineering_pipeline(
            train_df, test_df, config, memory_limit=memory_safe_mode
        )
        
        # 메모리 정리
        del train_df, test_df
        force_memory_cleanup()
        
        # 3. 모델 학습
        trainer, X_val, y_val = model_training_pipeline(
            X_train, y_train, config, 
            tune_hyperparameters=not args.no_tune and not memory_safe_mode,
            memory_safe_mode=memory_safe_mode
        )
        
        # 4. 앙상블 (메모리 여유가 있을 때만)
        available_memory = get_available_memory()
        ensemble_manager = None
        if available_memory > 8:
            ensemble_manager = ensemble_pipeline(trainer, X_val, y_val, config)
        else:
            logger.warning("메모리 부족으로 앙상블 단계 생략")
        
        # 5. 평가
        evaluation_pipeline(trainer, X_val, y_val, config)
        
        # 6. 최종 예측
        submission = generate_predictions(trainer, ensemble_manager, X_test, config)
        
        # 7. 추론 시스템 설정 (메모리 여유가 있을 때만)
        available_memory = get_available_memory()
        prediction_api = None
        if available_memory > 5:
            prediction_api = inference_setup(config)
        else:
            logger.warning("메모리 부족으로 추론 시스템 설정 생략")
        
        return {
            'trainer': trainer,
            'ensemble_manager': ensemble_manager,
            'prediction_api': prediction_api,
            'submission': submission
        }
        
    except MemoryError as e:
        logger.error(f"메모리 부족 오류: {str(e)}")
        force_memory_cleanup()
        raise
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류: {str(e)}")
        force_memory_cleanup()
        raise

def generate_predictions(trainer: ModelTrainer,
                        ensemble_manager: CTREnsembleManager,
                        X_test: pd.DataFrame,
                        config: Config) -> pd.DataFrame:
    """최종 예측 생성"""
    logger = logging.getLogger(__name__)
    logger.info("최종 예측 생성 시작")
    
    try:
        # 제출 템플릿 로딩
        data_loader = DataLoader(config)
        submission = data_loader.load_submission_template()
        
        # 샘플링 모드 확인
        is_sampled = len(X_test) != len(submission)
        if is_sampled:
            logger.info(f"샘플링 모드: X_test={len(X_test)}, submission={len(submission)}")
        
        # X_test 데이터 검증
        logger.info("예측 전 X_test 데이터 검증")
        
        # object 타입 컬럼 제거
        object_columns = X_test.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            logger.warning(f"object 타입 컬럼 제거: {object_columns}")
            X_test = X_test.drop(columns=object_columns)
        
        # 비수치형 컬럼 제거
        non_numeric_columns = []
        for col in X_test.columns:
            if not np.issubdtype(X_test[col].dtype, np.number):
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            logger.warning(f"비수치형 컬럼 제거: {non_numeric_columns}")
            X_test = X_test.drop(columns=non_numeric_columns)
        
        # 결측치 및 무한값 처리
        X_test = X_test.fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 데이터 타입 통일
        for col in X_test.columns:
            if X_test[col].dtype != 'float32':
                X_test[col] = X_test[col].astype('float32')
        
        logger.info(f"검증 완료 - X_test 형태: {X_test.shape}")
        
        # 모델 예측 수행
        if ensemble_manager and ensemble_manager.best_ensemble is not None:
            logger.info("앙상블 모델로 예측 수행")
            predictions = ensemble_manager.predict_with_best_ensemble(X_test)
        else:
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
        if is_sampled:
            logger.info("샘플링 모드: 기본값으로 초기화 후 샘플 예측값 할당")
            default_ctr = 0.0191
            submission['clicked'] = default_ctr
            submission.iloc[:len(predictions), submission.columns.get_loc('clicked')] = predictions
        else:
            submission['clicked'] = predictions
        
        # 예측값 정보
        logger.info(f"예측값 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
        logger.info(f"예측 평균: {predictions.mean():.4f}")
        
        # 제출 파일 저장
        output_path = config.BASE_DIR / "submission.csv"
        submission.to_csv(output_path, index=False)
        logger.info(f"제출 파일 저장: {output_path}")
        
        return submission
        
    except Exception as e:
        logger.error(f"예측 생성 실패: {str(e)}")
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
        
        # 보고서 생성
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
        
    except Exception as e:
        logger.error(f"평가 파이프라인 실패: {str(e)}")
        raise

def inference_setup(config: Config) -> CTRPredictionAPI:
    """추론 시스템 설정"""
    logger = logging.getLogger(__name__)
    logger.info("추론 시스템 설정 시작")
    
    try:
        prediction_api = CTRPredictionAPI(config)
        success = prediction_api.initialize()
        
        if success:
            status = prediction_api.get_api_status()
            logger.info(f"추론 시스템 상태: {status['engine_status']['system_status']}")
            logger.info(f"로딩된 모델 수: {status['engine_status']['models_count']}")
            
            test_result = prediction_api.predict_ctr(
                user_id="test_user",
                ad_id="test_ad",
                context={"device": "mobile", "page": "home"}
            )
            
            logger.info(f"테스트 예측 결과: {test_result['ctr_prediction']:.4f}")
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
    
    # 초기 메모리 상태
    initial_memory = get_memory_usage()
    available_memory = get_available_memory()
    logger.info(f"시작 메모리: 사용 {initial_memory:.2f} GB, 사용 가능 {available_memory:.2f} GB")
    
    try:
        if args.mode == "train":
            logger.info("=== 학습 모드 시작 ===")
            results = safe_pipeline_execution(args, config, logger)
            
        elif args.mode == "inference":
            logger.info("=== 추론 모드 시작 ===")
            prediction_api = inference_setup(config)
            
            if prediction_api:
                logger.info("추론 API 준비 완료")
            
        elif args.mode == "evaluate":
            logger.info("=== 평가 모드 시작 ===")
            trainer = ModelTrainer(config)
            loaded_models = trainer.load_models()
            
            if loaded_models:
                logger.info(f"로딩된 모델: {list(loaded_models.keys())}")
            else:
                logger.error("로딩할 모델이 없습니다.")
        
        # 실행 시간 및 메모리 사용량
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        final_available = get_available_memory()
        memory_increase = final_memory - initial_memory
        
        logger.info(f"전체 파이프라인 실행 시간: {total_time:.2f}초")
        logger.info(f"최종 메모리: 사용 {final_memory:.2f} GB, 사용 가능 {final_available:.2f} GB")
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
        force_memory_cleanup()
        final_memory = get_memory_usage()
        logger.info(f"정리 후 메모리: {final_memory:.2f} GB")
        logger.info("파이프라인 종료")

if __name__ == "__main__":
    main()