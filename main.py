# main.py

import argparse
import logging
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# 프로젝트 모듈들
from config import Config
from data_loader import DataLoader, DataValidator
from feature_engineering import FeatureEngineer
from training import ModelTrainer, TrainingPipeline
from evaluation import CTRMetrics, ModelComparator, EvaluationReporter
from ensemble import EnsembleManager
from inference import CTRPredictionAPI
from models import ModelFactory

def setup_logging():
    """로깅 설정"""
    logger = Config.setup_logging()
    logger.info("=== CTR 모델 파이프라인 시작 ===")
    return logger

def load_and_preprocess_data(config: Config) -> tuple:
    """데이터 로딩 및 전처리"""
    logger = logging.getLogger(__name__)
    logger.info("데이터 로딩 및 전처리 단계 시작")
    
    # 데이터 로더 초기화
    data_loader = DataLoader(config)
    
    try:
        # 데이터 로딩
        train_df, test_df = data_loader.load_data()
        
        # 데이터 요약 정보 출력
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
        
        logger.info("데이터 로딩 및 전처리 완료")
        return train_processed, test_processed, data_loader
        
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {str(e)}")
        raise

def feature_engineering_pipeline(train_df: pd.DataFrame, 
                                test_df: pd.DataFrame,
                                config: Config) -> tuple:
    """피처 엔지니어링 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("피처 엔지니어링 단계 시작")
    
    try:
        # 피처 엔지니어 초기화
        feature_engineer = FeatureEngineer(config)
        
        # 피처 생성
        X_train, X_test = feature_engineer.create_all_features(
            train_df, test_df, target_col='clicked'
        )
        
        # 타겟 변수 분리
        y_train = train_df['clicked']
        
        # 피처 정보 출력
        feature_summary = feature_engineer.get_feature_importance_summary()
        logger.info(f"생성된 피처 수: {feature_summary['total_generated_features']}")
        logger.info(f"최종 피처 차원: {X_train.shape}")
        
        logger.info("피처 엔지니어링 완료")
        return X_train, X_test, y_train, feature_engineer
        
    except Exception as e:
        logger.error(f"피처 엔지니어링 실패: {str(e)}")
        raise

def model_training_pipeline(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           config: Config,
                           tune_hyperparameters: bool = True) -> tuple:
    """모델 학습 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("모델 학습 단계 시작")
    
    try:
        # 학습 파이프라인 초기화
        training_pipeline = TrainingPipeline(config)
        
        # 데이터 분할
        data_loader = DataLoader(config)
        X_train_split, X_val_split, y_train_split, y_val_split = \
            data_loader.train_test_split_data(
                pd.concat([X_train, y_train], axis=1)
            )
        
        # 전체 학습 파이프라인 실행
        pipeline_results = training_pipeline.run_full_pipeline(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            tune_hyperparameters=tune_hyperparameters,
            n_trials=50 if tune_hyperparameters else 0
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
        raise

def ensemble_pipeline(trainer: ModelTrainer,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     config: Config) -> EnsembleManager:
    """앙상블 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("앙상블 단계 시작")
    
    try:
        # 앙상블 매니저 초기화
        ensemble_manager = EnsembleManager(config)
        
        # 학습된 모델들 추가
        for model_name, model_info in trainer.trained_models.items():
            ensemble_manager.add_base_model(model_name, model_info['model'])
        
        # 다양한 앙상블 생성
        ensemble_types = ['weighted', 'stacking', 'rank']
        
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
        raise

def generate_predictions(trainer: ModelTrainer,
                        ensemble_manager: EnsembleManager,
                        X_test: pd.DataFrame,
                        config: Config) -> pd.DataFrame:
    """최종 예측 생성"""
    logger = logging.getLogger(__name__)
    logger.info("최종 예측 생성 시작")
    
    try:
        # 제출 템플릿 로딩
        data_loader = DataLoader(config)
        submission = data_loader.load_submission_template()
        
        # 앙상블 예측 사용 (가능한 경우)
        if ensemble_manager.best_ensemble is not None:
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
        
        # 종합 평가 보고서 생성
        reporter = EvaluationReporter()
        report = reporter.generate_comprehensive_report(
            models_predictions, y_val,
            output_dir=str(config.OUTPUT_DIR / "evaluation_charts")
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
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    
    # 설정 초기화
    config = Config
    config.setup_directories()
    
    # 시작 시간 기록
    start_time = time.time()
    
    try:
        if args.mode == "train":
            # 전체 학습 파이프라인 실행
            logger.info("=== 학습 모드 시작 ===")
            
            # 1. 데이터 로딩 및 전처리
            train_df, test_df, data_loader = load_and_preprocess_data(config)
            
            # 빠른 실행 모드 (샘플링)
            if args.quick:
                logger.info("빠른 실행 모드: 데이터 샘플링")
                train_df = train_df.sample(n=min(10000, len(train_df)), random_state=42)
                test_df = test_df.sample(n=min(5000, len(test_df)), random_state=42)
            
            # 2. 피처 엔지니어링
            X_train, X_test, y_train, feature_engineer = feature_engineering_pipeline(
                train_df, test_df, config
            )
            
            # 3. 모델 학습
            trainer, X_val, y_val = model_training_pipeline(
                X_train, y_train, config, 
                tune_hyperparameters=not args.no_tune
            )
            
            # 4. 앙상블
            ensemble_manager = ensemble_pipeline(trainer, X_val, y_val, config)
            
            # 5. 평가
            evaluation_pipeline(trainer, X_val, y_val, config)
            
            # 6. 최종 예측
            submission = generate_predictions(trainer, ensemble_manager, X_test, config)
            
            # 7. 추론 시스템 설정
            prediction_api = inference_setup(config)
            
        elif args.mode == "inference":
            # 추론 모드
            logger.info("=== 추론 모드 시작 ===")
            prediction_api = inference_setup(config)
            
            if prediction_api:
                logger.info("추론 API 준비 완료")
                # 여기에 웹 서버나 API 서비스 코드 추가 가능
            
        elif args.mode == "evaluate":
            # 평가 모드
            logger.info("=== 평가 모드 시작 ===")
            
            # 저장된 모델 로딩 및 평가
            trainer = ModelTrainer(config)
            loaded_models = trainer.load_models()
            
            if loaded_models:
                logger.info(f"로딩된 모델: {list(loaded_models.keys())}")
                # 추가 평가 로직 구현 가능
            else:
                logger.error("로딩할 모델이 없습니다.")
        
        # 실행 시간 출력
        total_time = time.time() - start_time
        logger.info(f"전체 파이프라인 실행 시간: {total_time:.2f}초")
        logger.info("=== 파이프라인 완료 ===")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}")
        raise
    
    finally:
        # 정리 작업
        logger.info("파이프라인 종료")

if __name__ == "__main__":
    main()