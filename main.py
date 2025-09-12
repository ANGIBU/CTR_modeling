# main.py

import argparse
import logging
import time
import json
import gc
import psutil
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import torch

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ctypes
        if hasattr(ctypes, 'windll'):
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception as e:
        pass

def setup_logging():
    """로깅 설정"""
    logger = Config.setup_logging()
    logger.info("=== CTR 모델링 파이프라인 시작 ===")
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
        
        if available_after < 8:
            force_memory_cleanup()
        
        return result
    return wrapper

@memory_monitor_decorator
def load_and_preprocess_large_data(config: Config, quick_mode: bool = False) -> tuple:
    """대용량 데이터 로딩 및 전처리"""
    logger = logging.getLogger(__name__)
    logger.info("대용량 데이터 로딩 및 전처리 시작")
    
    data_loader = DataLoader(config)
    
    try:
        available_memory = get_available_memory()
        logger.info(f"사용 가능 메모리: {available_memory:.2f} GB")
        
        # 메모리 기반 동적 크기 결정
        if quick_mode:
            max_train_size = 100000
            max_test_size = 20000
            logger.info(f"빠른 모드: 학습 {max_train_size:,}, 테스트 {max_test_size:,}")
        elif available_memory > 45:
            # 64GB 환경에서 최대 활용
            max_train_size = 1500000
            max_test_size = 300000
            logger.info(f"대용량 모드: 학습 {max_train_size:,}, 테스트 {max_test_size:,}")
        elif available_memory > 30:
            max_train_size = 1000000
            max_test_size = 200000
            logger.info(f"중간 모드: 학습 {max_train_size:,}, 테스트 {max_test_size:,}")
        else:
            max_train_size = 500000
            max_test_size = 100000
            logger.info(f"절약 모드: 학습 {max_train_size:,}, 테스트 {max_test_size:,}")
        
        # 대용량 데이터 로딩
        train_df, test_df = data_loader.load_large_data_optimized()
        
        # 메모리 상태 확인
        current_memory = get_memory_usage()
        available_memory = get_available_memory()
        logger.info(f"데이터 로딩 후 - 사용: {current_memory:.2f} GB, 사용 가능: {available_memory:.2f} GB")
        
        # 메모리 부족 시 점진적 감소
        if available_memory < 12:
            logger.warning("메모리 부족으로 데이터 크기 조정")
            target_ratio = max(0.3, available_memory / 20)
            
            new_train_size = int(len(train_df) * target_ratio)
            new_test_size = int(len(test_df) * target_ratio)
            
            logger.info(f"데이터 크기 조정: 학습 {len(train_df):,} → {new_train_size:,}")
            logger.info(f"데이터 크기 조정: 테스트 {len(test_df):,} → {new_test_size:,}")
            
            train_df = train_df.sample(n=new_train_size, random_state=42).reset_index(drop=True)
            test_df = test_df.sample(n=new_test_size, random_state=42).reset_index(drop=True)
            
            force_memory_cleanup()
        
        # 데이터 요약
        train_summary = data_loader.get_data_summary(train_df)
        test_summary = data_loader.get_data_summary(test_df)
        
        logger.info(f"최종 학습 데이터: {train_summary['shape']}")
        logger.info(f"최종 테스트 데이터: {test_summary['shape']}")
        
        if 'target_distribution' in train_summary:
            actual_ctr = train_summary['target_distribution']['ctr']
            logger.info(f"실제 CTR: {actual_ctr:.4f}")
        
        # 데이터 검증
        validator = DataValidator()
        validation_results = validator.validate_data_consistency(train_df, test_df)
        
        if validation_results['missing_in_test'] or validation_results['dtype_mismatches']:
            logger.warning(f"데이터 일관성 문제: {len(validation_results['missing_in_test'])}개 누락")
        
        # 기본 전처리
        train_processed = data_loader.basic_preprocessing(train_df)
        test_processed = data_loader.basic_preprocessing(test_df)
        
        del train_df, test_df
        force_memory_cleanup()
        
        logger.info("대용량 데이터 로딩 및 전처리 완료")
        return train_processed, test_processed, data_loader
        
    except Exception as e:
        logger.error(f"대용량 데이터 로딩 실패: {str(e)}")
        force_memory_cleanup()
        raise

@memory_monitor_decorator
def advanced_feature_engineering(train_df: pd.DataFrame, 
                                test_df: pd.DataFrame,
                                config: Config) -> tuple:
    """피처 엔지니어링"""
    logger = logging.getLogger(__name__)
    logger.info("피처 엔지니어링 시작")
    
    try:
        available_memory = get_available_memory()
        logger.info(f"피처 엔지니어링 시작 - 사용 가능 메모리: {available_memory:.2f} GB")
        
        # 메모리 기반 모드 설정
        memory_efficient_mode = available_memory < 20
        
        feature_engineer = FeatureEngineer(config)
        feature_engineer.set_memory_efficient_mode(memory_efficient_mode)
        
        if memory_efficient_mode:
            logger.info("메모리 효율 모드로 피처 엔지니어링")
        
        # 피처 생성
        X_train, X_test = feature_engineer.create_all_features(
            train_df, test_df, target_col='clicked'
        )
        
        # 타겟 변수 분리
        y_train = train_df['clicked'].copy()
        
        # 피처 엔지니어 저장
        feature_engineer_path = config.MODEL_DIR / "feature_engineer.pkl"
        try:
            with open(feature_engineer_path, 'wb') as f:
                pickle.dump(feature_engineer, f)
            logger.info(f"피처 엔지니어 저장: {feature_engineer_path}")
        except Exception as e:
            logger.warning(f"피처 엔지니어 저장 실패: {str(e)}")
        
        del train_df, test_df
        force_memory_cleanup()
        
        # 피처 정보
        feature_summary = feature_engineer.get_feature_importance_summary()
        logger.info(f"생성된 피처 수: {feature_summary['total_generated_features']}")
        logger.info(f"최종 피처 차원: {X_train.shape}")
        
        return X_train, X_test, y_train, feature_engineer
        
    except Exception as e:
        logger.error(f"피처 엔지니어링 실패: {str(e)}")
        force_memory_cleanup()
        raise

@memory_monitor_decorator
def comprehensive_model_training(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               config: Config,
                               tune_hyperparameters: bool = True) -> tuple:
    """종합 모델 학습 (GPU 활용)"""
    logger = logging.getLogger(__name__)
    logger.info("종합 모델 학습 시작")
    
    try:
        available_memory = get_available_memory()
        gpu_available = torch.cuda.is_available()
        
        logger.info(f"GPU 사용 가능: {gpu_available}")
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 메모리 기반 학습 전략
        if available_memory < 15:
            logger.info("메모리 절약 모드")
            if len(X_train) > 200000:
                sample_size = 150000
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train.iloc[sample_indices].reset_index(drop=True)
                y_train = y_train.iloc[sample_indices].reset_index(drop=True)
                logger.info(f"학습 데이터 샘플링: {sample_size:,}개")
            
            tune_hyperparameters = False
        
        # 학습 파이프라인 초기화
        training_pipeline = TrainingPipeline(config)
        
        # 데이터 분할
        data_loader = DataLoader(config)
        split_result = data_loader.memory_efficient_train_test_split(X_train, y_train)
        X_train_split, X_val_split, y_train_split, y_val_split = split_result
        
        del X_train, y_train
        force_memory_cleanup()
        
        # 모델 타입 결정
        model_types = ['lightgbm', 'xgboost', 'catboost']
        
        # GPU 사용 가능시 DeepCTR 추가
        if gpu_available and available_memory > 20:
            model_types.append('deepctr')
            logger.info("GPU 환경: DeepCTR 모델 추가")
        
        # 전체 학습 파이프라인 실행
        pipeline_results = {}
        
        # 개별 모델 학습
        for model_type in model_types:
            try:
                logger.info(f"{model_type} 모델 학습 시작")
                
                # 하이퍼파라미터 튜닝
                if tune_hyperparameters:
                    n_trials = 50 if model_type == 'deepctr' else 100
                    training_pipeline.trainer.hyperparameter_tuning_optuna(
                        model_type, X_train_split, y_train_split, n_trials=n_trials, cv_folds=3
                    )
                
                # 교차검증 평가
                cv_result = training_pipeline.trainer.cross_validate_model(
                    model_type, X_train_split, y_train_split, cv_folds=3
                )
                
                # 최종 모델 학습
                params = training_pipeline.trainer.best_params.get(model_type, None)
                model_kwargs = {'params': params}
                if model_type == 'deepctr':
                    model_kwargs['input_dim'] = X_train_split.shape[1]
                
                model = ModelFactory.create_model(model_type, **model_kwargs)
                model.fit(X_train_split, y_train_split, X_val_split, y_val_split)
                
                # Calibration 적용
                model.apply_calibration(X_val_split, y_val_split, method='platt', cv_folds=3)
                
                training_pipeline.trainer.trained_models[model_type] = {
                    'model': model,
                    'params': params or {},
                    'cv_result': cv_result
                }
                
                logger.info(f"{model_type} 모델 학습 완료")
                
                # GPU 메모리 정리
                if model_type == 'deepctr' and gpu_available:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"{model_type} 모델 학습 실패: {str(e)}")
                continue
        
        # 모델 저장
        training_pipeline.trainer.save_models()
        
        pipeline_results = {
            'model_count': len(training_pipeline.trainer.trained_models),
            'trained_models': list(training_pipeline.trainer.trained_models.keys()),
            'gpu_used': gpu_available and 'deepctr' in training_pipeline.trainer.trained_models
        }
        
        # 최고 성능 모델 찾기
        if training_pipeline.trainer.cv_results:
            best_model_name = max(
                training_pipeline.trainer.cv_results.keys(),
                key=lambda x: training_pipeline.trainer.cv_results[x]['combined_mean']
            )
            best_score = training_pipeline.trainer.cv_results[best_model_name]['combined_mean']
            
            pipeline_results['best_model'] = {
                'name': best_model_name,
                'score': best_score
            }
        
        logger.info(f"종합 학습 완료 - 모델 수: {pipeline_results['model_count']}")
        
        return training_pipeline.trainer, X_val_split, y_val_split, pipeline_results
        
    except Exception as e:
        logger.error(f"종합 모델 학습 실패: {str(e)}")
        force_memory_cleanup()
        raise

@memory_monitor_decorator
def advanced_ensemble_pipeline(trainer: ModelTrainer,
                             X_val: pd.DataFrame,
                             y_val: pd.Series,
                             config: Config) -> CTREnsembleManager:
    """Calibration 앙상블 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("Calibration 앙상블 파이프라인 시작")
    
    try:
        available_memory = get_available_memory()
        
        ensemble_manager = CTREnsembleManager(config)
        
        # 학습된 모델들 추가
        for model_name, model_info in trainer.trained_models.items():
            ensemble_manager.add_base_model(model_name, model_info['model'])
        
        # 실제 CTR 계산
        actual_ctr = y_val.mean()
        logger.info(f"실제 CTR: {actual_ctr:.4f}")
        
        # 앙상블 타입별 생성
        ensemble_types = []
        
        # 메모리 여유가 있을 때 더 많은 앙상블 생성
        if available_memory > 15:
            ensemble_types = ['weighted', 'calibrated', 'rank']
            logger.info("전체 앙상블 모드")
        elif available_memory > 10:
            ensemble_types = ['weighted', 'calibrated']
            logger.info("제한 앙상블 모드")
        else:
            ensemble_types = ['calibrated']
            logger.info("Calibration 앙상블만 생성")
        
        for ensemble_type in ensemble_types:
            try:
                if ensemble_type == 'calibrated':
                    # CTR 보정 앙상블 (핵심)
                    ensemble_manager.create_ensemble(
                        'calibrated', 
                        target_ctr=actual_ctr, 
                        calibration_method='platt'
                    )
                else:
                    ensemble_manager.create_ensemble(ensemble_type)
                
                logger.info(f"{ensemble_type} 앙상블 생성 완료")
                
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 생성 실패: {str(e)}")
        
        # 앙상블 학습
        ensemble_manager.train_all_ensembles(X_val, y_val)
        
        # 앙상블 평가
        ensemble_results = ensemble_manager.evaluate_ensembles(X_val, y_val)
        
        for name, score in ensemble_results.items():
            logger.info(f"{name}: Combined Score {score:.4f}")
        
        # 앙상블 저장
        ensemble_manager.save_ensembles()
        
        logger.info("Calibration 앙상블 파이프라인 완료")
        return ensemble_manager
        
    except Exception as e:
        logger.error(f"앙상블 파이프라인 실패: {str(e)}")
        force_memory_cleanup()
        raise

def comprehensive_evaluation(trainer: ModelTrainer,
                           ensemble_manager: CTREnsembleManager,
                           X_val: pd.DataFrame,
                           y_val: pd.Series,
                           config: Config):
    """종합 평가"""
    logger = logging.getLogger(__name__)
    logger.info("종합 평가 시작")
    
    try:
        # 모델별 예측 수집
        models_predictions = {}
        
        for model_name, model_info in trainer.trained_models.items():
            model = model_info['model']
            pred = model.predict_proba(X_val)
            models_predictions[model_name] = pred
        
        # 앙상블 예측 추가
        for ensemble_name, ensemble_score in ensemble_manager.ensemble_results.items():
            if ensemble_name.startswith('ensemble_'):
                ensemble_type = ensemble_name.replace('ensemble_', '')
                if ensemble_type in ensemble_manager.ensembles:
                    ensemble = ensemble_manager.ensembles[ensemble_type]
                    if ensemble.is_fitted:
                        base_predictions = {}
                        for model_name, model_info in trainer.trained_models.items():
                            base_predictions[model_name] = model_info['model'].predict_proba(X_val)
                        
                        ensemble_pred = ensemble.predict_proba(base_predictions)
                        models_predictions[f'ensemble_{ensemble_type}'] = ensemble_pred
        
        # 모델 비교
        comparator = ModelComparator()
        comparison_df = comparator.compare_models(models_predictions, y_val)
        
        logger.info("모델 성능 비교:")
        key_metrics = ['combined_score', 'ap', 'wll', 'auc', 'f1']
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        
        if available_metrics:
            logger.info(f"\n{comparison_df[available_metrics].round(4)}")
        
        # CTR 분석
        actual_ctr = y_val.mean()
        logger.info(f"\n=== CTR 분석 ===")
        logger.info(f"실제 CTR: {actual_ctr:.4f}")
        
        for model_name, pred in models_predictions.items():
            predicted_ctr = pred.mean()
            bias = predicted_ctr - actual_ctr
            bias_ratio = predicted_ctr / actual_ctr if actual_ctr > 0 else float('inf')
            logger.info(f"{model_name}: 예측 {predicted_ctr:.4f} (편향 {bias:+.4f}, 비율 {bias_ratio:.2f}x)")
        
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
        
        logger.info(f"종합 평가 보고서 저장: {report_path}")
        
    except Exception as e:
        logger.error(f"종합 평가 실패: {str(e)}")
        raise

def generate_optimized_predictions(trainer: ModelTrainer,
                                 ensemble_manager: CTREnsembleManager,
                                 X_test: pd.DataFrame,
                                 config: Config) -> pd.DataFrame:
    """최적화된 예측 생성 (CTR 보정 적용)"""
    logger = logging.getLogger(__name__)
    logger.info("최적화된 예측 생성 시작")
    
    try:
        # 제출 템플릿 로딩
        data_loader = DataLoader(config)
        submission = data_loader.load_submission_template()
        
        # 샘플링 모드 확인
        is_sampled = len(X_test) != len(submission)
        if is_sampled:
            logger.info(f"샘플링 모드: X_test={len(X_test)}, submission={len(submission)}")
        
        # X_test 데이터 검증 및 정리
        logger.info("예측 전 X_test 데이터 검증")
        
        # object 타입 및 비수치형 컬럼 제거
        object_columns = X_test.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            logger.warning(f"object 타입 컬럼 제거: {len(object_columns)}개")
            X_test = X_test.drop(columns=object_columns)
        
        non_numeric_columns = []
        for col in X_test.columns:
            if not np.issubdtype(X_test[col].dtype, np.number):
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            logger.warning(f"비수치형 컬럼 제거: {len(non_numeric_columns)}개")
            X_test = X_test.drop(columns=non_numeric_columns)
        
        # 결측치 및 무한값 처리
        X_test = X_test.fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 데이터 타입 통일
        for col in X_test.columns:
            if X_test[col].dtype != 'float32':
                X_test[col] = X_test[col].astype('float32')
        
        logger.info(f"검증 완료 - X_test 형태: {X_test.shape}")
        
        # 예측 수행 (우선순위: Calibrated Ensemble → Best Ensemble → Best Model)
        predictions = None
        prediction_method = ""
        
        # 1순위: Calibrated Ensemble 사용
        if (ensemble_manager and 
            ensemble_manager.calibrated_ensemble and 
            ensemble_manager.calibrated_ensemble.is_fitted):
            
            logger.info("Calibrated Ensemble로 예측 수행")
            base_predictions = {}
            for model_name, model_info in trainer.trained_models.items():
                try:
                    pred = model_info['model'].predict_proba(X_test)
                    base_predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"{model_name} 예측 실패: {str(e)}")
                    base_predictions[model_name] = np.full(len(X_test), 0.0201)
            
            predictions = ensemble_manager.calibrated_ensemble.predict_proba(base_predictions)
            prediction_method = "Calibrated Ensemble"
            
        # 2순위: Best Ensemble 사용
        elif ensemble_manager and ensemble_manager.best_ensemble:
            logger.info("Best Ensemble로 예측 수행")
            predictions = ensemble_manager.predict_with_best_ensemble(X_test)
            prediction_method = f"Best Ensemble ({ensemble_manager.best_ensemble.name})"
            
        # 3순위: Best Single Model 사용
        else:
            logger.info("Best Single Model로 예측 수행")
            best_model_name = None
            best_score = 0
            
            for model_name, model_info in trainer.trained_models.items():
                if 'cv_result' in model_info:
                    score = model_info['cv_result']['combined_mean']
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
            
            if best_model_name:
                best_model = trainer.trained_models[best_model_name]['model']
                logger.info(f"사용할 모델: {best_model_name}")
                predictions = best_model.predict_proba(X_test)
                prediction_method = f"Best Model ({best_model_name})"
            else:
                raise ValueError("사용 가능한 모델이 없습니다")
        
        # 추가 CTR 보정 (실제 CTR에 더 가깝게)
        target_ctr = 0.0201
        current_ctr = predictions.mean()
        
        if abs(current_ctr - target_ctr) > 0.005:  # 0.5% 이상 차이 시 보정
            logger.info(f"추가 CTR 보정: {current_ctr:.4f} → {target_ctr:.4f}")
            
            # 선형 보정
            bias_correction = target_ctr - current_ctr
            predictions = predictions + bias_correction
            
            # 범위 클리핑
            predictions = np.clip(predictions, 0.001, 0.999)
            
            corrected_ctr = predictions.mean()
            logger.info(f"보정 후 CTR: {corrected_ctr:.4f}")
        
        # 제출 파일 생성
        if is_sampled:
            logger.info("샘플링 모드: 기본값으로 초기화 후 샘플 예측값 할당")
            submission['clicked'] = target_ctr
            submission.iloc[:len(predictions), submission.columns.get_loc('clicked')] = predictions
        else:
            submission['clicked'] = predictions
        
        # 예측 통계
        final_ctr = submission['clicked'].mean()
        logger.info(f"=== 예측 결과 ===")
        logger.info(f"예측 방법: {prediction_method}")
        logger.info(f"예측값 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
        logger.info(f"예측 평균 CTR: {predictions.mean():.4f}")
        logger.info(f"최종 제출 CTR: {final_ctr:.4f}")
        logger.info(f"목표 CTR: {target_ctr:.4f}")
        logger.info(f"CTR 편향: {final_ctr - target_ctr:+.4f}")
        
        # 제출 파일 저장
        output_path = config.BASE_DIR / "submission.csv"
        submission.to_csv(output_path, index=False)
        logger.info(f"제출 파일 저장: {output_path}")
        
        return submission
        
    except Exception as e:
        logger.error(f"최적화된 예측 생성 실패: {str(e)}")
        raise

def setup_inference_system(config: Config) -> Optional[CTRPredictionAPI]:
    """추론 시스템 설정"""
    logger = logging.getLogger(__name__)
    logger.info("추론 시스템 설정 시작")
    
    try:
        available_memory = get_available_memory()
        
        if available_memory < 5:
            logger.warning("메모리 부족으로 추론 시스템 설정 생략")
            return None
        
        prediction_api = CTRPredictionAPI(config)
        success = prediction_api.initialize()
        
        if success:
            status = prediction_api.get_api_status()
            logger.info(f"추론 시스템 상태: {status['engine_status']['system_status']}")
            logger.info(f"로딩된 모델 수: {status['engine_status']['models_count']}")
            
            # 테스트 예측
            test_result = prediction_api.predict_ctr(
                user_id="test_user",
                ad_id="test_ad",
                context={
                    "device": "mobile", 
                    "page": "home",
                    "hour": 14,
                    "day_of_week": "tuesday"
                }
            )
            
            logger.info(f"테스트 예측 결과: {test_result['ctr_prediction']:.4f}")
            return prediction_api
        else:
            logger.error("추론 시스템 초기화 실패")
            return None
            
    except Exception as e:
        logger.error(f"추론 시스템 설정 실패: {str(e)}")
        return None

def execute_comprehensive_pipeline(args, config: Config, logger):
    """종합 파이프라인 실행"""
    
    try:
        # 초기 상태
        initial_memory = get_memory_usage()
        available_memory = get_available_memory()
        logger.info(f"시작 상태 - 사용: {initial_memory:.2f} GB, 사용 가능: {available_memory:.2f} GB")
        
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_info.total_memory / (1024**3):.1f}GB)")
        
        # 1. 대용량 데이터 로딩 및 전처리
        train_df, test_df, data_loader = load_and_preprocess_large_data(
            config, quick_mode=args.quick
        )
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train, feature_engineer = advanced_feature_engineering(
            train_df, test_df, config
        )
        
        del train_df, test_df
        force_memory_cleanup()
        
        # 3. 종합 모델 학습 (GPU 포함)
        trainer, X_val, y_val, training_results = comprehensive_model_training(
            X_train, y_train, config, 
            tune_hyperparameters=not args.no_tune
        )
        
        logger.info(f"학습 결과: {training_results}")
        
        # 4. Calibration 앙상블
        available_memory = get_available_memory()
        ensemble_manager = None
        
        if available_memory > 5:
            ensemble_manager = advanced_ensemble_pipeline(trainer, X_val, y_val, config)
        else:
            logger.warning("메모리 부족으로 앙상블 단계 생략")
        
        # 5. 종합 평가
        comprehensive_evaluation(trainer, ensemble_manager, X_val, y_val, config)
        
        # 6. 최적화된 예측 생성 (CTR 보정)
        submission = generate_optimized_predictions(trainer, ensemble_manager, X_test, config)
        
        # 7. 추론 시스템 설정
        available_memory = get_available_memory()
        prediction_api = None
        
        if available_memory > 3:
            prediction_api = setup_inference_system(config)
        else:
            logger.warning("메모리 부족으로 추론 시스템 설정 생략")
        
        return {
            'trainer': trainer,
            'ensemble_manager': ensemble_manager,
            'prediction_api': prediction_api,
            'submission': submission,
            'training_results': training_results
        }
        
    except MemoryError as e:
        logger.error(f"메모리 부족 오류: {str(e)}")
        logger.info("해결 방안:")
        logger.info("1. --quick 옵션 사용")
        logger.info("2. 시스템 메모리 증설")
        logger.info("3. 다른 프로그램 종료")
        force_memory_cleanup()
        raise
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}")
        force_memory_cleanup()
        raise

def main():
    """메인 실행 함수"""
    
    # 인자 파싱
    parser = argparse.ArgumentParser(description="CTR 모델링 종합 파이프라인")
    parser.add_argument("--mode", choices=["train", "inference", "evaluate"], 
                       default="train", help="실행 모드")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--no-tune", action="store_true", 
                       help="하이퍼파라미터 튜닝 비활성화")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 실행 (소규모 데이터)")
    parser.add_argument("--gpu", action="store_true",
                       help="GPU 강제 사용")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    
    # 설정 초기화
    config = Config
    config.setup_directories()
    
    # GPU 환경 설정
    if args.gpu and torch.cuda.is_available():
        config.setup_gpu_environment()
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 초기 상태
    initial_memory = get_memory_usage()
    available_memory = get_available_memory()
    logger.info(f"시작 메모리: 사용 {initial_memory:.2f} GB, 사용 가능 {available_memory:.2f} GB")
    
    try:
        if args.mode == "train":
            logger.info("=== 종합 학습 모드 시작 ===")
            results = execute_comprehensive_pipeline(args, config, logger)
            
            # 학습 결과 요약
            logger.info("=== 학습 결과 요약 ===")
            logger.info(f"학습된 모델 수: {results['training_results']['model_count']}")
            logger.info(f"GPU 사용 여부: {results['training_results'].get('gpu_used', False)}")
            
            if 'best_model' in results['training_results']:
                best = results['training_results']['best_model']
                logger.info(f"최고 성능 모델: {best['name']} (Score: {best['score']:.4f})")
            
            if results['ensemble_manager']:
                ensemble_summary = results['ensemble_manager'].get_ensemble_summary()
                logger.info(f"앙상블 수: {ensemble_summary['fitted_ensembles']}/{ensemble_summary['total_ensembles']}")
                logger.info(f"Calibrated Ensemble: {ensemble_summary['calibrated_ensemble_available']}")
            
        elif args.mode == "inference":
            logger.info("=== 추론 모드 시작 ===")
            prediction_api = setup_inference_system(config)
            
            if prediction_api:
                logger.info("추론 API 준비 완료")
                status = prediction_api.get_api_status()
                logger.info(f"API 상태: {status}")
            
        elif args.mode == "evaluate":
            logger.info("=== 평가 모드 시작 ===")
            trainer = ModelTrainer(config)
            loaded_models = trainer.load_models()
            
            if loaded_models:
                logger.info(f"로딩된 모델: {list(loaded_models.keys())}")
            else:
                logger.error("로딩할 모델이 없습니다")
        
        # 실행 시간 및 메모리 사용량
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        final_available = get_available_memory()
        memory_increase = final_memory - initial_memory
        
        logger.info("=== 파이프라인 완료 ===")
        logger.info(f"전체 실행 시간: {total_time:.2f}초")
        logger.info(f"최종 메모리: 사용 {final_memory:.2f} GB, 사용 가능 {final_available:.2f} GB")
        logger.info(f"메모리 증가량: {memory_increase:+.2f} GB")
        
        # GPU 메모리 정보
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
            logger.info(f"GPU 메모리 사용량: {gpu_memory:.2f} GB")
        
        logger.info("=== CTR 모델링 파이프라인 종료 ===")
        
    except MemoryError as e:
        logger.error(f"메모리 부족으로 파이프라인 중단: {str(e)}")
        logger.info("메모리 최적화 방안:")
        logger.info("1. --quick 옵션으로 소규모 실행")
        logger.info("2. 다른 프로그램 종료 후 재시도")
        logger.info("3. 가상 메모리 설정 증가")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise
    
    finally:
        # 정리 작업
        force_memory_cleanup()
        final_memory = get_memory_usage()
        logger.info(f"정리 후 메모리: {final_memory:.2f} GB")

if __name__ == "__main__":
    main()