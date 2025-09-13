# main.py

import argparse
import logging 
import time
import json
import gc
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import sys
import signal

# 필수 라이브러리 import 안전 처리
PSUTIL_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil이 설치되지 않았습니다. 메모리 모니터링 기능이 제한됩니다.")

try:
    import torch
    if torch.cuda.is_available():
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            test_tensor.cpu()
            del test_tensor
            torch.cuda.empty_cache()
            TORCH_AVAILABLE = True
        except Exception as e:
            print(f"GPU 테스트 실패: {e}. CPU 모드만 사용")
            TORCH_AVAILABLE = True
    else:
        TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch가 설치되지 않았습니다. GPU 기능이 비활성화됩니다.")

# 프로젝트 모듈 import
try:
    from config import Config
    from data_loader import DataLoader, DataValidator
    from feature_engineering import FeatureEngineer
    from training import ModelTrainer, TrainingPipeline
    from evaluation import CTRMetrics, ModelComparator, EvaluationReporter
    from ensemble import CTREnsembleManager
    from inference import CTRPredictionAPI
    from models import ModelFactory
except ImportError as e:
    print(f"필수 모듈 import 실패: {e}")
    print("필요한 모든 파일이 프로젝트 디렉터리에 있는지 확인하세요.")
    sys.exit(1)

# 전역 변수로 정리 작업을 위한 플래그
cleanup_required = False
training_pipeline = None

def signal_handler(signum, frame):
    """인터럽트 신호 처리"""
    global cleanup_required
    print("\n프로그램 중단 요청을 받았습니다. 정리 작업을 진행합니다...")
    cleanup_required = True
    force_memory_cleanup()

def get_memory_usage() -> float:
    """현재 메모리 사용량 (GB)"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except:
            return 0.0
    return 0.0

def get_available_memory() -> float:
    """사용 가능한 메모리 (GB)"""
    if PSUTIL_AVAILABLE:
        try:
            return psutil.virtual_memory().available / (1024**3)
        except:
            return 45.0
    return 45.0

def force_memory_cleanup():
    """강제 메모리 정리"""
    try:
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"메모리 정리 중 오류: {e}")
    
    try:
        import ctypes
        if hasattr(ctypes, 'windll'):
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except:
        pass

def setup_logging():
    """로깅 설정"""
    try:
        logger = Config.setup_logging()
        logger.info("=== CTR 모델링 파이프라인 시작 ===")
        return logger
    except Exception as e:
        print(f"로깅 설정 실패: {e}")
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger

def memory_monitor_decorator(func):
    """메모리 모니터링 데코레이터"""
    def wrapper(*args, **kwargs):
        if not PSUTIL_AVAILABLE:
            return func(*args, **kwargs)
            
        memory_before = get_memory_usage()
        available_before = get_available_memory()
        logger = logging.getLogger(__name__)
        logger.info(f"{func.__name__} 시작 - 메모리: {memory_before:.2f} GB, 사용 가능: {available_before:.2f} GB")
        
        try:
            result = func(*args, **kwargs)
            
            memory_after = get_memory_usage()
            available_after = get_available_memory()
            logger.info(f"{func.__name__} 완료 - 메모리: {memory_after:.2f} GB, 사용 가능: {available_after:.2f} GB")
            
            if available_after < 15:
                logger.warning("메모리 부족 상황 - 정리 작업 수행")
                force_memory_cleanup()
            
            return result
        except Exception as e:
            logger.error(f"{func.__name__} 실행 중 오류: {e}")
            force_memory_cleanup()
            raise
    
    return wrapper

@memory_monitor_decorator
def load_and_preprocess_large_data(config: Config, quick_mode: bool = False) -> tuple:
    """대용량 데이터 로딩 및 전처리"""
    logger = logging.getLogger(__name__)
    logger.info("대용량 데이터 로딩 및 전처리 시작")
    
    # 파일 존재 여부 확인
    train_path = Path(config.TRAIN_PATH)
    test_path = Path(config.TEST_PATH)
    
    if not train_path.exists():
        logger.error(f"학습 데이터 파일이 없습니다: {train_path}")
        raise FileNotFoundError(f"학습 데이터 파일이 없습니다: {train_path}")
    
    if not test_path.exists():
        logger.error(f"테스트 데이터 파일이 없습니다: {test_path}")
        raise FileNotFoundError(f"테스트 데이터 파일이 없습니다: {test_path}")
    
    try:
        data_loader = DataLoader(config)
        
        available_memory = get_available_memory()
        logger.info(f"사용 가능 메모리: {available_memory:.2f} GB")
        
        # 메모리 기반 동적 크기 결정
        if quick_mode:
            max_train_size = 100000
            logger.info(f"빠른 모드: 학습 {max_train_size:,}")
        else:
            # 전체 데이터 처리 모드
            if available_memory > 30:
                max_train_size = 1500000
            elif available_memory > 20:
                max_train_size = 1000000
            else:
                max_train_size = 800000
            logger.info(f"전체 데이터 모드: 학습 {max_train_size:,}")
        
        # 전체 데이터 로딩 (테스트 데이터는 항상 전체)
        try:
            train_df, test_df = data_loader.load_large_data_optimized()
        except Exception as e:
            logger.warning(f"최적화 로딩 실패: {e}. 기본 로딩 시도")
            try:
                train_df, test_df = data_loader.load_data()
            except Exception as e2:
                logger.error(f"기본 로딩도 실패: {e2}")
                raise
        
        # 테스트 데이터 크기 검증
        if len(test_df) < 1500000:
            logger.error(f"테스트 데이터 크기가 부족합니다: {len(test_df):,} < 1,500,000")
            raise ValueError("전체 테스트 데이터를 로딩하지 못했습니다")
        
        # 메모리 상태 확인
        current_memory = get_memory_usage()
        available_memory = get_available_memory()
        logger.info(f"데이터 로딩 후 - 사용: {current_memory:.2f} GB, 사용 가능: {available_memory:.2f} GB")
        
        # 메모리 부족 시 학습 데이터만 조정 (테스트 데이터는 유지)
        if available_memory < 20:
            logger.warning("메모리 부족으로 학습 데이터 크기 조정")
            target_ratio = max(0.5, available_memory / 40)
            
            new_train_size = int(len(train_df) * target_ratio)
            logger.info(f"학습 데이터 크기 조정: {len(train_df):,} → {new_train_size:,}")
            
            if new_train_size > 50000:
                train_df = train_df.sample(n=new_train_size, random_state=42).reset_index(drop=True)
            
            force_memory_cleanup()
        
        # 데이터 요약
        try:
            train_summary = data_loader.get_data_summary(train_df)
            test_summary = data_loader.get_data_summary(test_df)
            
            logger.info(f"최종 학습 데이터: {train_summary['shape']}")
            logger.info(f"최종 테스트 데이터: {test_summary['shape']}")
            
            if 'target_distribution' in train_summary:
                actual_ctr = train_summary['target_distribution']['ctr']
                logger.info(f"실제 CTR: {actual_ctr:.4f}")
        except Exception as e:
            logger.warning(f"데이터 요약 생성 실패: {e}")
        
        # 데이터 검증
        try:
            validator = DataValidator()
            validation_results = validator.validate_data_consistency(train_df, test_df)
            
            if validation_results['missing_in_test'] or validation_results['dtype_mismatches']:
                logger.warning(f"데이터 일관성 문제: {len(validation_results['missing_in_test'])}개 누락")
        except Exception as e:
            logger.warning(f"데이터 검증 실패: {e}")
        
        # 기본 전처리
        try:
            train_processed = data_loader.basic_preprocessing(train_df)
            test_processed = data_loader.basic_preprocessing(test_df)
        except Exception as e:
            logger.error(f"기본 전처리 실패: {e}")
            train_processed = train_df
            test_processed = test_df
        
        del train_df, test_df
        force_memory_cleanup()
        
        logger.info("대용량 데이터 로딩 및 전처리 완료")
        logger.info(f"테스트 데이터 크기 확인: {len(test_processed):,}행")
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
        
        memory_efficient_mode = available_memory < 25
        
        feature_engineer = FeatureEngineer(config)
        feature_engineer.set_memory_efficient_mode(memory_efficient_mode)
        
        if memory_efficient_mode:
            logger.info("메모리 효율 모드로 피처 엔지니어링")
        
        # 타겟 컬럼 확인
        target_col = 'clicked'
        if target_col not in train_df.columns:
            logger.error(f"타겟 컬럼 '{target_col}'이 없습니다.")
            possible_targets = [col for col in train_df.columns if 'click' in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                logger.info(f"대체 타겟 컬럼 사용: {target_col}")
            else:
                raise ValueError("타겟 컬럼을 찾을 수 없습니다.")
        
        # 피처 생성
        X_train, X_test = feature_engineer.create_all_features(
            train_df, test_df, target_col=target_col
        )
        
        # 타겟 변수 분리
        y_train = train_df[target_col].copy()
        
        # 피처 엔지니어 저장
        try:
            feature_engineer_path = config.MODEL_DIR / "feature_engineer.pkl"
            config.setup_directories()
            with open(feature_engineer_path, 'wb') as f:
                pickle.dump(feature_engineer, f)
            logger.info(f"피처 엔지니어 저장: {feature_engineer_path}")
        except Exception as e:
            logger.warning(f"피처 엔지니어 저장 실패: {str(e)}")
        
        del train_df, test_df
        force_memory_cleanup()
        
        # 피처 정보
        try:
            feature_summary = feature_engineer.get_feature_importance_summary()
            logger.info(f"생성된 피처 수: {feature_summary['total_generated_features']}")
            logger.info(f"최종 피처 차원: {X_train.shape}")
            logger.info(f"테스트 데이터 피처 차원: {X_test.shape}")
        except Exception as e:
            logger.warning(f"피처 요약 정보 생성 실패: {e}")
        
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
    """종합 모델 학습"""
    logger = logging.getLogger(__name__)
    logger.info("종합 모델 학습 시작")
    
    global training_pipeline
    
    try:
        available_memory = get_available_memory()
        gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        logger.info(f"GPU 사용 가능: {gpu_available}")
        if gpu_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            except Exception as e:
                logger.warning(f"GPU 정보 조회 실패: {e}")
        
        # 메모리 기반 학습 전략
        if available_memory < 20:
            logger.info("메모리 절약 모드")
            if len(X_train) > 800000:
                sample_size = 700000
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train.iloc[sample_indices].reset_index(drop=True)
                y_train = y_train.iloc[sample_indices].reset_index(drop=True)
                logger.info(f"학습 데이터 샘플링: {sample_size:,}개")
            
            tune_hyperparameters = False
        
        # 학습 파이프라인 초기화
        training_pipeline = TrainingPipeline(config)
        
        # 데이터 분할
        data_loader = DataLoader(config)
        try:
            split_result = data_loader.memory_efficient_train_test_split(X_train, y_train)
            X_train_split, X_val_split, y_train_split, y_val_split = split_result
        except Exception as e:
            logger.error(f"데이터 분할 실패: {e}")
            from sklearn.model_selection import train_test_split
            try:
                split_result = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
                X_train_split, X_val_split, y_train_split, y_val_split = split_result
                logger.info("기본 데이터 분할 사용")
            except Exception as e2:
                logger.error(f"기본 데이터 분할도 실패: {e2}")
                raise
        
        del X_train, y_train
        force_memory_cleanup()
        
        # 사용 가능한 모델 타입 확인
        available_models = ModelFactory.get_available_models()
        logger.info(f"사용 가능한 모델: {available_models}")
        
        model_types = []
        if 'lightgbm' in available_models:
            model_types.append('lightgbm')
        if 'xgboost' in available_models:
            model_types.append('xgboost')
        if 'catboost' in available_models:
            model_types.append('catboost')
        
        # GPU 사용 가능시 DeepCTR 추가
        if gpu_available and available_memory > 20 and 'deepctr' in available_models:
            model_types.append('deepctr')
            logger.info("GPU 환경: DeepCTR 모델 추가")
        
        if not model_types:
            model_types = ['logistic']
            logger.warning("기본 모델만 사용 가능합니다.")
        
        # 전체 학습 파이프라인 실행
        pipeline_results = {}
        
        # 개별 모델 학습
        successful_models = 0
        for model_type in model_types:
            if cleanup_required:
                logger.info("사용자 중단 요청으로 학습 중단")
                break
                
            try:
                logger.info(f"{model_type} 모델 학습 시작")
                
                # 하이퍼파라미터 튜닝
                if tune_hyperparameters and available_memory > 15:
                    try:
                        n_trials = 20 if model_type == 'deepctr' else 30
                        training_pipeline.trainer.hyperparameter_tuning_ctr_optuna(
                            model_type, X_train_split, y_train_split, n_trials=n_trials, cv_folds=3
                        )
                    except Exception as e:
                        logger.warning(f"{model_type} 하이퍼파라미터 튜닝 실패: {e}")
                
                # 교차검증 평가
                try:
                    cv_result = training_pipeline.trainer.cross_validate_ctr_model(
                        model_type, X_train_split, y_train_split, cv_folds=3
                    )
                    
                    if cv_result['combined_mean'] > 0:
                        successful_models += 1
                    
                except Exception as e:
                    logger.warning(f"{model_type} 교차검증 실패: {e}")
                    cv_result = None
                
                # 최종 모델 학습
                params = training_pipeline.trainer.best_params.get(model_type, None)
                if params is None:
                    params = training_pipeline.trainer._get_ctr_optimized_params(model_type)
                
                model_kwargs = {'params': params}
                if model_type == 'deepctr':
                    model_kwargs['input_dim'] = X_train_split.shape[1]
                
                model = ModelFactory.create_model(model_type, **model_kwargs)
                model.fit(X_train_split, y_train_split, X_val_split, y_val_split)
                
                # Calibration 적용
                try:
                    model.apply_calibration(X_val_split, y_val_split, method='platt', cv_folds=3)
                except Exception as e:
                    logger.warning(f"{model_type} Calibration 실패: {e}")
                
                training_pipeline.trainer.trained_models[model_type] = {
                    'model': model,
                    'params': params or {},
                    'cv_result': cv_result
                }
                
                logger.info(f"{model_type} 모델 학습 완료")
                
                # GPU 메모리 정리
                if model_type == 'deepctr' and gpu_available:
                    force_memory_cleanup()
                
            except Exception as e:
                logger.error(f"{model_type} 모델 학습 실패: {str(e)}")
                force_memory_cleanup()
                continue
        
        # 모델 저장
        try:
            training_pipeline.trainer.save_models()
        except Exception as e:
            logger.warning(f"모델 저장 실패: {e}")
        
        pipeline_results = {
            'model_count': len(training_pipeline.trainer.trained_models),
            'trained_models': list(training_pipeline.trainer.trained_models.keys()),
            'gpu_used': gpu_available and 'deepctr' in training_pipeline.trainer.trained_models,
            'successful_models': successful_models
        }
        
        # 최고 성능 모델 찾기
        if training_pipeline.trainer.cv_results:
            try:
                valid_results = {k: v for k, v in training_pipeline.trainer.cv_results.items() if v['combined_mean'] > 0}
                if valid_results:
                    best_model_name = max(
                        valid_results.keys(),
                        key=lambda x: valid_results[x]['combined_mean']
                    )
                    best_score = valid_results[best_model_name]['combined_mean']
                    
                    pipeline_results['best_model'] = {
                        'name': best_model_name,
                        'score': best_score
                    }
            except Exception as e:
                logger.warning(f"최고 모델 찾기 실패: {e}")
        
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
                             config: Config) -> Optional[CTREnsembleManager]:
    """앙상블 파이프라인"""
    logger = logging.getLogger(__name__)
    logger.info("앙상블 파이프라인 시작")
    
    try:
        available_memory = get_available_memory()
        
        if len(trainer.trained_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            return None
        
        ensemble_manager = CTREnsembleManager(config)
        
        # 학습된 모델들 추가
        for model_name, model_info in trainer.trained_models.items():
            ensemble_manager.add_base_model(model_name, model_info['model'])
        
        # 실제 CTR 계산
        actual_ctr = y_val.mean()
        logger.info(f"실제 CTR: {actual_ctr:.4f}")
        
        # 앙상블 타입별 생성
        ensemble_types = []
        
        if available_memory > 20:
            ensemble_types = ['weighted', 'calibrated', 'rank']
            logger.info("전체 앙상블 모드")
        elif available_memory > 15:
            ensemble_types = ['weighted', 'calibrated']
            logger.info("제한 앙상블 모드")
        else:
            ensemble_types = ['calibrated']
            logger.info("Calibration 앙상블만 생성")
        
        for ensemble_type in ensemble_types:
            try:
                if ensemble_type == 'calibrated':
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
        try:
            ensemble_manager.save_ensembles()
        except Exception as e:
            logger.warning(f"앙상블 저장 실패: {e}")
        
        logger.info("앙상블 파이프라인 완료")
        return ensemble_manager
        
    except Exception as e:
        logger.error(f"앙상블 파이프라인 실패: {str(e)}")
        force_memory_cleanup()
        return None

def comprehensive_evaluation(trainer: ModelTrainer,
                           ensemble_manager: Optional[CTREnsembleManager],
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
            try:
                model = model_info['model']
                pred = model.predict_proba(X_val)
                models_predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"{model_name} 예측 실패: {e}")
        
        # 앙상블 예측 추가
        if ensemble_manager:
            for ensemble_name, ensemble_score in ensemble_manager.ensemble_results.items():
                if ensemble_name.startswith('ensemble_'):
                    ensemble_type = ensemble_name.replace('ensemble_', '')
                    if ensemble_type in ensemble_manager.ensembles:
                        try:
                            ensemble = ensemble_manager.ensembles[ensemble_type]
                            if ensemble.is_fitted:
                                base_predictions = {}
                                for model_name, model_info in trainer.trained_models.items():
                                    base_predictions[model_name] = model_info['model'].predict_proba(X_val)
                                
                                ensemble_pred = ensemble.predict_proba(base_predictions)
                                models_predictions[f'ensemble_{ensemble_type}'] = ensemble_pred
                        except Exception as e:
                            logger.warning(f"{ensemble_name} 앙상블 예측 실패: {e}")
        
        if not models_predictions:
            logger.warning("평가할 예측이 없습니다")
            return
        
        # 모델 비교
        try:
            comparator = ModelComparator()
            comparison_df = comparator.compare_models(models_predictions, y_val)
            
            logger.info("모델 성능 비교:")
            key_metrics = ['combined_score', 'ap', 'wll', 'auc', 'f1']
            available_metrics = [m for m in key_metrics if m in comparison_df.columns]
            
            if available_metrics:
                logger.info(f"\n{comparison_df[available_metrics].round(4)}")
        except Exception as e:
            logger.warning(f"모델 비교 실패: {e}")
        
        # CTR 분석
        actual_ctr = y_val.mean()
        logger.info(f"\n=== CTR 분석 ===")
        logger.info(f"실제 CTR: {actual_ctr:.4f}")
        
        for model_name, pred in models_predictions.items():
            try:
                predicted_ctr = pred.mean()
                bias = predicted_ctr - actual_ctr
                bias_ratio = predicted_ctr / actual_ctr if actual_ctr > 0 else float('inf')
                logger.info(f"{model_name}: 예측 {predicted_ctr:.4f} (편향 {bias:+.4f}, 비율 {bias_ratio:.2f}x)")
            except Exception as e:
                logger.warning(f"{model_name} CTR 분석 실패: {e}")
        
        # 보고서 생성
        try:
            reporter = EvaluationReporter()
            report = reporter.generate_comprehensive_report(
                models_predictions, y_val,
                output_dir=None
            )
            
            # 보고서 저장
            report_path = config.OUTPUT_DIR / "evaluation_report.json"
            config.setup_directories()
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"종합 평가 보고서 저장: {report_path}")
        except Exception as e:
            logger.warning(f"보고서 생성 실패: {e}")
        
    except Exception as e:
        logger.error(f"종합 평가 실패: {str(e)}")

def generate_full_test_predictions(trainer: ModelTrainer,
                                 ensemble_manager: Optional[CTREnsembleManager],
                                 X_test: pd.DataFrame,
                                 config: Config) -> pd.DataFrame:
    """전체 테스트 데이터 예측 생성"""
    logger = logging.getLogger(__name__)
    logger.info("전체 테스트 데이터 예측 생성 시작")
    
    # 테스트 데이터 크기 확인
    test_size = len(X_test)
    logger.info(f"테스트 데이터 크기: {test_size:,}행")
    
    if test_size < 1500000:
        logger.error(f"테스트 데이터 크기가 부족합니다: {test_size:,} < 1,500,000")
        raise ValueError("전체 테스트 데이터가 아닙니다")
    
    try:
        # 제출 템플릿 로딩
        try:
            data_loader = DataLoader(config)
            submission = data_loader.load_submission_template()
        except Exception as e:
            logger.warning(f"제출 템플릿 로딩 실패: {e}. 기본 템플릿 생성")
            submission = pd.DataFrame({
                'id': range(test_size),
                'clicked': 0.0201
            })
        
        logger.info(f"제출 템플릿 크기: {len(submission):,}행")
        logger.info(f"테스트 데이터 크기: {test_size:,}행")
        
        # 크기 일치 확인
        if len(submission) != test_size:
            logger.error(f"크기 불일치: 제출 템플릿 {len(submission):,} vs 테스트 데이터 {test_size:,}")
            raise ValueError("제출 템플릿과 테스트 데이터 크기가 일치하지 않습니다")
        
        # X_test 데이터 검증 및 정리
        logger.info("테스트 데이터 검증 및 정리")
        
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
                try:
                    X_test[col] = X_test[col].astype('float32')
                except:
                    X_test[col] = 0.0
        
        logger.info(f"검증 완료 - 테스트 데이터 형태: {X_test.shape}")
        
        # 배치 단위 예측
        batch_size = 50000  # 5만개씩 처리
        total_batches = (test_size + batch_size - 1) // batch_size
        predictions = np.zeros(test_size)
        
        # 예측 방법 선택
        prediction_method = ""
        
        # 1순위: Calibrated Ensemble
        if (ensemble_manager and 
            hasattr(ensemble_manager, 'calibrated_ensemble') and
            ensemble_manager.calibrated_ensemble and 
            ensemble_manager.calibrated_ensemble.is_fitted):
            
            logger.info("Calibrated Ensemble로 전체 데이터 예측 수행")
            prediction_method = "Calibrated Ensemble"
            
            try:
                for i in range(0, test_size, batch_size):
                    end_idx = min(i + batch_size, test_size)
                    X_batch = X_test.iloc[i:end_idx]
                    
                    logger.info(f"배치 {i//batch_size + 1}/{total_batches} 처리 중 ({i:,}~{end_idx:,})")
                    
                    # 기본 모델들의 예측
                    base_predictions = {}
                    for model_name, model_info in trainer.trained_models.items():
                        try:
                            pred = model_info['model'].predict_proba(X_batch)
                            base_predictions[model_name] = pred
                        except Exception as e:
                            logger.warning(f"{model_name} 배치 예측 실패: {str(e)}")
                            base_predictions[model_name] = np.full(len(X_batch), 0.0201)
                    
                    # 앙상블 예측
                    batch_pred = ensemble_manager.calibrated_ensemble.predict_proba(base_predictions)
                    predictions[i:end_idx] = batch_pred
                    
                    # 메모리 정리
                    if (i // batch_size) % 5 == 0:
                        force_memory_cleanup()
                        
            except Exception as e:
                logger.error(f"Calibrated Ensemble 배치 예측 실패: {e}")
                predictions = None
        
        # 2순위: Best Ensemble 사용
        elif ensemble_manager and hasattr(ensemble_manager, 'best_ensemble') and ensemble_manager.best_ensemble:
            logger.info("Best Ensemble로 전체 데이터 예측 수행")
            prediction_method = f"Best Ensemble ({ensemble_manager.best_ensemble.name})"
            
            try:
                for i in range(0, test_size, batch_size):
                    end_idx = min(i + batch_size, test_size)
                    X_batch = X_test.iloc[i:end_idx]
                    
                    logger.info(f"배치 {i//batch_size + 1}/{total_batches} 처리 중 ({i:,}~{end_idx:,})")
                    
                    base_predictions = {}
                    for model_name, model_info in trainer.trained_models.items():
                        try:
                            pred = model_info['model'].predict_proba(X_batch)
                            base_predictions[model_name] = pred
                        except Exception as e:
                            logger.warning(f"{model_name} 배치 예측 실패: {str(e)}")
                            base_predictions[model_name] = np.full(len(X_batch), 0.0201)
                    
                    batch_pred = ensemble_manager.best_ensemble.predict_proba(base_predictions)
                    predictions[i:end_idx] = batch_pred
                    
                    if (i // batch_size) % 5 == 0:
                        force_memory_cleanup()
                        
            except Exception as e:
                logger.error(f"Best Ensemble 배치 예측 실패: {e}")
                predictions = None
        
        # 3순위: Best Single Model
        if predictions is None:
            logger.info("Best Single Model로 전체 데이터 예측 수행")
            best_model_name = None
            best_score = 0
            
            for model_name, model_info in trainer.trained_models.items():
                if 'cv_result' in model_info and model_info['cv_result']:
                    try:
                        score = model_info['cv_result']['combined_mean']
                        if score > best_score:
                            best_score = score
                            best_model_name = model_name
                    except:
                        continue
            
            if best_model_name:
                logger.info(f"사용할 모델: {best_model_name}")
                prediction_method = f"Best Model ({best_model_name})"
                
                try:
                    best_model = trainer.trained_models[best_model_name]['model']
                    
                    for i in range(0, test_size, batch_size):
                        end_idx = min(i + batch_size, test_size)
                        X_batch = X_test.iloc[i:end_idx]
                        
                        logger.info(f"배치 {i//batch_size + 1}/{total_batches} 처리 중 ({i:,}~{end_idx:,})")
                        
                        batch_pred = best_model.predict_proba(X_batch)
                        predictions[i:end_idx] = batch_pred
                        
                        if (i // batch_size) % 5 == 0:
                            force_memory_cleanup()
                            
                except Exception as e:
                    logger.error(f"Best Model 배치 예측 실패: {e}")
                    predictions = None
        
        # 기본값 사용
        if predictions is None:
            logger.warning("모든 예측 실패. 기본값 사용")
            predictions = np.full(test_size, 0.0201)
            prediction_method = "Default"
        
        # CTR 보정 
        target_ctr = 0.0201
        current_ctr = predictions.mean()
        
        if abs(current_ctr - target_ctr) > 0.002:
            logger.info(f"CTR 보정: {current_ctr:.4f} → {target_ctr:.4f}")
            
            # 선형 보정
            bias_correction = target_ctr - current_ctr
            predictions = predictions + bias_correction
            
            # 범위 클리핑
            predictions = np.clip(predictions, 0.001, 0.999)
            
            corrected_ctr = predictions.mean()
            logger.info(f"보정 후 CTR: {corrected_ctr:.4f}")
        
        # 제출 파일 생성
        submission['clicked'] = predictions
        
        # 예측 통계
        final_ctr = submission['clicked'].mean()
        logger.info(f"=== 전체 데이터 예측 결과 ===")
        logger.info(f"예측 방법: {prediction_method}")
        logger.info(f"처리된 데이터 수: {test_size:,}행")
        logger.info(f"예측값 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
        logger.info(f"최종 제출 CTR: {final_ctr:.4f}")
        logger.info(f"목표 CTR: {target_ctr:.4f}")
        logger.info(f"CTR 편향: {final_ctr - target_ctr:+.4f}")
        
        # 제출 파일 저장
        output_path = config.BASE_DIR / "submission.csv"
        submission.to_csv(output_path, index=False)
        logger.info(f"제출 파일 저장: {output_path}")
        
        return submission
        
    except Exception as e:
        logger.error(f"전체 테스트 데이터 예측 생성 실패: {str(e)}")
        try:
            default_submission = pd.DataFrame({
                'id': range(test_size if 'test_size' in locals() else 1527298),
                'clicked': 0.0201
            })
            output_path = config.BASE_DIR / "submission.csv"
            default_submission.to_csv(output_path, index=False)
            logger.info(f"기본 제출 파일 저장: {output_path}")
            return default_submission
        except Exception as e2:
            logger.error(f"기본 제출 파일 생성도 실패: {e2}")
            raise e

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
            try:
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
            except Exception as e:
                logger.warning(f"추론 시스템 테스트 실패: {e}")
            
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
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_info = torch.cuda.get_device_properties(0)
                logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_info.total_memory / (1024**3):.1f}GB)")
            except Exception as e:
                logger.warning(f"GPU 정보 조회 실패: {e}")
        
        # 1. 대용량 데이터 로딩 및 전처리
        train_df, test_df, data_loader = load_and_preprocess_large_data(
            config, quick_mode=args.quick
        )
        
        if cleanup_required:
            logger.info("사용자 중단 요청")
            return None
        
        # 2. 피처 엔지니어링
        X_train, X_test, y_train, feature_engineer = advanced_feature_engineering(
            train_df, test_df, config
        )
        
        del train_df, test_df
        force_memory_cleanup()
        
        if cleanup_required:
            logger.info("사용자 중단 요청")
            return None
        
        # 3. 종합 모델 학습
        trainer, X_val, y_val, training_results = comprehensive_model_training(
            X_train, y_train, config, 
            tune_hyperparameters=not args.no_tune
        )
        
        logger.info(f"학습 결과: {training_results}")
        
        if cleanup_required:
            logger.info("사용자 중단 요청")
            return {'trainer': trainer, 'submission': None}
        
        # 4. 앙상블
        available_memory = get_available_memory()
        ensemble_manager = None
        
        if available_memory > 10 and training_results.get('successful_models', 0) >= 2:
            ensemble_manager = advanced_ensemble_pipeline(trainer, X_val, y_val, config)
        else:
            logger.warning("메모리 부족 또는 모델 부족으로 앙상블 단계 생략")
        
        # 5. 종합 평가
        try:
            comprehensive_evaluation(trainer, ensemble_manager, X_val, y_val, config)
        except Exception as e:
            logger.warning(f"종합 평가 실패: {e}")
        
        # 6. 전체 테스트 데이터 예측 생성
        submission = generate_full_test_predictions(trainer, ensemble_manager, X_test, config)
        
        # 7. 추론 시스템 설정
        available_memory = get_available_memory()
        prediction_api = None
        
        if available_memory > 5 and not cleanup_required:
            prediction_api = setup_inference_system(config)
        else:
            logger.warning("메모리 부족 또는 중단 요청으로 추론 시스템 설정 생략")
        
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
    except KeyboardInterrupt:
        logger.info("사용자에 의해 실행이 중단되었습니다.")
        return None
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}")
        force_memory_cleanup()
        raise

def main():
    """메인 실행 함수"""
    
    global cleanup_required
    
    # 신호 처리기 설정
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
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
    try:
        config.setup_directories()
    except Exception as e:
        logger.warning(f"디렉터리 설정 실패: {e}")
    
    # GPU 환경 설정
    if args.gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            config.setup_gpu_environment()
        except Exception as e:
            logger.warning(f"GPU 환경 설정 실패: {e}")
    
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
            
            if results is None:
                logger.info("파이프라인이 중단되었습니다.")
                return
            
            # 학습 결과 요약
            logger.info("=== 학습 결과 요약 ===")
            logger.info(f"학습된 모델 수: {results['training_results']['model_count']}")
            logger.info(f"GPU 사용 여부: {results['training_results'].get('gpu_used', False)}")
            
            if 'best_model' in results['training_results']:
                best = results['training_results']['best_model']
                logger.info(f"최고 성능 모델: {best['name']} (Score: {best['score']:.4f})")
            
            if results['ensemble_manager']:
                try:
                    ensemble_summary = results['ensemble_manager'].get_ensemble_summary()
                    logger.info(f"앙상블 수: {ensemble_summary['fitted_ensembles']}/{ensemble_summary['total_ensembles']}")
                    logger.info(f"Calibrated Ensemble: {ensemble_summary['calibrated_ensemble_available']}")
                except Exception as e:
                    logger.warning(f"앙상블 요약 실패: {e}")
            
            # 제출 파일 정보
            if results['submission'] is not None:
                logger.info(f"제출 파일 생성: {len(results['submission']):,}행")
            
        elif args.mode == "inference":
            logger.info("=== 추론 모드 시작 ===")
            prediction_api = setup_inference_system(config)
            
            if prediction_api:
                logger.info("추론 API 준비 완료")
                try:
                    status = prediction_api.get_api_status()
                    logger.info(f"API 상태: {status}")
                except Exception as e:
                    logger.warning(f"API 상태 조회 실패: {e}")
            
        elif args.mode == "evaluate":
            logger.info("=== 평가 모드 시작 ===")
            try:
                trainer = ModelTrainer(config)
                loaded_models = trainer.load_models()
                
                if loaded_models:
                    logger.info(f"로딩된 모델: {list(loaded_models.keys())}")
                else:
                    logger.error("로딩할 모델이 없습니다")
            except Exception as e:
                logger.error(f"평가 모드 실행 실패: {e}")
        
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
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"GPU 메모리 사용량: {gpu_memory:.2f} GB")
            except Exception as e:
                logger.warning(f"GPU 메모리 조회 실패: {e}")
        
        logger.info("=== CTR 모델링 파이프라인 종료 ===")
        
    except MemoryError as e:
        logger.error(f"메모리 부족으로 파이프라인 중단: {str(e)}")
        logger.info("메모리 최적화 방안:")
        logger.info("1. --quick 옵션으로 소규모 실행")
        logger.info("2. 다른 프로그램 종료 후 재시도")
        logger.info("3. 가상 메모리 설정 증가")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 실행이 중단되었습니다.")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {str(e)}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        
    finally:
        # 정리 작업
        cleanup_required = True
        force_memory_cleanup()
        if PSUTIL_AVAILABLE:
            final_memory = get_memory_usage()
            logger.info(f"정리 후 메모리: {final_memory:.2f} GB")

if __name__ == "__main__":
    main()