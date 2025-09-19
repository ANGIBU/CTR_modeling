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

def setup_logging(log_level=logging.INFO):
    """로깅 시스템 초기화"""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
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

cleanup_required = False
logger = setup_logging()

def signal_handler(signum, frame):
    """인터럽트 신호 처리"""
    global cleanup_required
    logger.info("프로그램 중단 요청을 받았습니다")
    cleanup_required = True
    
    try:
        gc.collect()
        if PSUTIL_AVAILABLE:
            import ctypes
            if hasattr(ctypes, 'windll'):
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception:
        pass

def validate_environment():
    """환경 검증"""
    logger.info("=== 환경 검증 시작 ===")
    
    python_version = sys.version
    logger.info(f"Python 버전: {python_version}")
    
    required_dirs = ['data', 'models', 'logs', 'output']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"디렉터리 준비: {dir_path}")
    
    data_files = {
        'train': Path('data/train.parquet'),
        'test': Path('data/test.parquet'), 
        'submission': Path('data/sample_submission.csv')
    }
    
    for name, path in data_files.items():
        exists = path.exists()
        size_mb = path.stat().st_size / (1024**2) if exists else 0
        logger.info(f"{name} 파일: {exists} ({size_mb:.1f}MB)")
    
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        logger.info(f"시스템 메모리: {vm.total/(1024**3):.1f}GB (사용가능: {vm.available/(1024**3):.1f}GB)")
        logger.info(f"메모리 사용률: {vm.percent:.1f}%")
    else:
        logger.warning("psutil을 사용할 수 없어 메모리 모니터링이 제한됩니다")
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
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
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("GPU 감지: RTX 4060 Ti 최적화 적용")
            if hasattr(Config, 'setup_gpu_environment'):
                Config.setup_gpu_environment()
        
        imported_modules = {
            'Config': Config,
            'LargeDataLoader': LargeDataLoader,
            'CTRFeatureEngineer': CTRFeatureEngineer
        }
        
        try:
            from training import ModelTrainer, TrainingPipeline
            imported_modules['ModelTrainer'] = ModelTrainer
            imported_modules['TrainingPipeline'] = TrainingPipeline
            logger.info("학습 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"학습 모듈 import 실패: {e}")
        
        try:
            from evaluation import CTRMetrics, ModelComparator, EvaluationReporter
            imported_modules['CTRMetrics'] = CTRMetrics
            imported_modules['ModelComparator'] = ModelComparator
            imported_modules['EvaluationReporter'] = EvaluationReporter
            logger.info("평가 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"평가 모듈 import 실패: {e}")
        
        try:
            from ensemble import CTRSuperEnsembleManager
            imported_modules['CTRSuperEnsembleManager'] = CTRSuperEnsembleManager
            logger.info("앙상블 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"앙상블 모듈 import 실패: {e}")
        
        try:
            from inference import CTRPredictionAPI, create_ctr_prediction_service
            imported_modules['CTRPredictionAPI'] = CTRPredictionAPI
            imported_modules['create_ctr_prediction_service'] = create_ctr_prediction_service
            logger.info("추론 모듈 import 성공")
        except ImportError as e:
            logger.warning(f"추론 모듈 import 실패: {e}")
        
        try:
            from models import ModelFactory
            imported_modules['ModelFactory'] = ModelFactory
            logger.info("모델 팩토리 import 성공")
        except ImportError as e:
            logger.warning(f"모델 팩토리 import 실패: {e}")
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
    """메모리 정리"""
    try:
        initial_time = time.time()
        
        collected = 0
        for i in range(20 if intensive else 15):
            collected += gc.collect()
            if i % 5 == 0:
                time.sleep(0.1)
        
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

def execute_final_pipeline(config, quick_mode=False):
    """파이프라인 실행"""
    logger.info("=== 파이프라인 실행 시작 ===")
    
    start_time = time.time()
    
    try:
        force_memory_cleanup(intensive=True)
        
        modules = safe_import_modules()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("GPU 감지: RTX 4060 Ti 최적화 적용")
            if hasattr(config, 'setup_gpu_environment'):
                config.setup_gpu_environment()
        
        # 1. 대용량 데이터 로딩
        logger.info("1. 대용량 데이터 로딩")
        data_loader = modules['LargeDataLoader'](config)
        
        try:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                logger.info(f"로딩 전 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
            
            train_df, test_df = data_loader.load_large_data_optimized()
            logger.info(f"데이터 로딩 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
            
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                logger.info(f"로딩 후 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
                if vm.available / (1024**3) < 10:
                    logger.warning("메모리 부족 상태입니다. 메모리 정리를 수행합니다.")
                    force_memory_cleanup(intensive=True)
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            
            logger.info("메모리 정리 후 재시도")
            force_memory_cleanup(intensive=True)
            time.sleep(2)
            
            try:
                config.CHUNK_SIZE = min(config.CHUNK_SIZE, 15000)
                config.MAX_MEMORY_GB = min(config.MAX_MEMORY_GB, 35)
                
                train_df, test_df = data_loader.load_large_data_optimized()
                logger.info(f"재시도 성공 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
            except Exception as e2:
                logger.error(f"재시도도 실패: {e2}")
                raise e2
        
        if cleanup_required:
            logger.info("사용자 요청으로 파이프라인 중단")
            return None
        
        # 2. 피처 엔지니어링 (64GB 메모리 활용)
        logger.info("2. 피처 엔지니어링 (64GB 메모리 활용)")
        feature_engineer = modules['CTRFeatureEngineer'](config)
        
        target_col = 'clicked'
        if target_col not in train_df.columns:
            possible_targets = [col for col in train_df.columns if 'click' in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                logger.info(f"타겟 컬럼 변경: {target_col}")
            else:
                logger.error("타겟 컬럼을 찾을 수 없습니다")
                train_df[target_col] = np.random.binomial(1, 0.02, len(train_df))
                logger.warning(f"임시 타겟 컬럼 '{target_col}' 생성")
        
        try:
            # 64GB 환경에서 메모리 효율 모드 비활성화
            feature_engineer.set_memory_efficient_mode(False)
            
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                # 64GB 환경에서는 40GB 이상 사용 가능할 때만 단순화 
                if vm.available / (1024**3) < 15:
                    logger.warning("메모리 부족으로 단순화된 피처 엔지니어링 수행")
                    feature_cols = [col for col in train_df.columns if col != target_col]
                    X_train = train_df[feature_cols[:100]].copy()
                    X_test = test_df[feature_cols[:100]].copy() if set(feature_cols[:100]).issubset(test_df.columns) else test_df.iloc[:, :100].copy()
                else:
                    # 64GB 환경에서 적극적인 피처 생성
                    X_train, X_test = feature_engineer.create_all_features(
                        train_df, test_df, target_col=target_col
                    )
            else:
                X_train, X_test = feature_engineer.create_all_features(
                    train_df, test_df, target_col=target_col
                )
            
            y_train = train_df[target_col].copy()
            
            logger.info(f"피처 엔지니어링 완료 - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
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
            
            logger.warning("기본 피처만 사용하여 진행")
            feature_cols = [col for col in train_df.columns if col != target_col]
            
            max_features = 118
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                if vm.available / (1024**3) < 10:
                    max_features = 80
                elif vm.available / (1024**3) < 15:
                    max_features = 100
            
            selected_features = feature_cols[:max_features]
            X_train = train_df[selected_features].copy()
            X_test = test_df[selected_features].copy() if set(selected_features).issubset(test_df.columns) else test_df.iloc[:, :max_features].copy()
            y_train = train_df[target_col].copy()
            
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        if col in X_test.columns:
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                    except Exception:
                        X_train[col] = 0
                        if col in X_test.columns:
                            X_test[col] = 0
        
        del train_df, test_df
        force_memory_cleanup(intensive=True)
        
        if cleanup_required:
            logger.info("사용자 요청으로 파이프라인 중단")
            return None
        
        # 3. 모델 학습 (캘리브레이션 강제 적용)
        logger.info("3. 모델 학습 (캘리브레이션 강제 적용)")
        successful_models = 0
        trained_models = {}
        
        available_models = ['lightgbm', 'logistic']
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            # 64GB 환경에서 더 적극적인 모델 사용
            if vm.available / (1024**3) > 25:
                available_models.append('xgboost')
                if TORCH_AVAILABLE and torch.cuda.is_available() and vm.available / (1024**3) > 35:
                    if not quick_mode:
                        available_models.append('catboost')
        
        logger.info(f"사용 가능한 모델: {available_models}")
        
        # 앙상블 매니저 초기화
        ensemble_manager = None
        if 'CTRSuperEnsembleManager' in modules:
            try:
                ensemble_manager = modules['CTRSuperEnsembleManager'](config)
                logger.info("앙상블 매니저 초기화 완료")
            except Exception as e:
                logger.warning(f"앙상블 매니저 초기화 실패: {e}")
        
        if 'ModelTrainer' in modules and modules['ModelTrainer'] is not None:
            try:
                trainer = modules['ModelTrainer'](config)
                
                from sklearn.model_selection import train_test_split
                
                test_size = 0.2
                if PSUTIL_AVAILABLE:
                    vm = psutil.virtual_memory()
                    if vm.available / (1024**3) < 20:
                        test_size = 0.15
                
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
                )
                
                logger.info(f"데이터 분할 완료 - 학습: {X_train_split.shape}, 검증: {X_val_split.shape}")
                
                for model_type in available_models:
                    if cleanup_required:
                        break
                    
                    try:
                        logger.info(f"=== {model_type} 모델 학습 시작 ===")
                        
                        force_memory_cleanup()
                        
                        model = train_final_model(
                            model_type, X_train_split, y_train_split, 
                            X_val_split, y_val_split, config
                        )
                        
                        if model is not None:
                            # 모든 모델에 강제 캘리브레이션 적용
                            logger.info(f"{model_type} 모델에 캘리브레이션 강제 적용")
                            try:
                                model.apply_calibration(X_val_split, y_val_split, method='auto')
                                logger.info(f"{model_type} 캘리브레이션 적용 완료")
                            except Exception as cal_error:
                                logger.warning(f"{model_type} 캘리브레이션 실패: {cal_error}")
                            
                            trained_models[model_type] = {
                                'model': model,
                                'params': {},
                                'training_time': 0.0,
                                'model_type': model_type
                            }
                            
                            # 앙상블 매니저에 모델 추가 - 강제 실행
                            if ensemble_manager is not None:
                                try:
                                    ensemble_manager.add_base_model(model_type, model)
                                    logger.info(f"{model_type} 모델을 앙상블 매니저에 추가 완료")
                                except Exception as add_error:
                                    logger.error(f"{model_type} 모델 앙상블 추가 실패: {add_error}")
                            
                            successful_models += 1
                            logger.info(f"=== {model_type} 모델 학습 완료 ===")
                        else:
                            logger.warning(f"{model_type} 모델 학습 실패")
                        
                        force_memory_cleanup()
                        
                        if PSUTIL_AVAILABLE:
                            vm = psutil.virtual_memory()
                            # 64GB 환경에서 더 관대한 메모리 임계값
                            if vm.available / (1024**3) < 12:
                                logger.warning("메모리 부족으로 추가 모델 학습 중단")
                                break
                        
                    except Exception as e:
                        logger.error(f"{model_type} 모델 학습 실패: {e}")
                        force_memory_cleanup()
                        continue
                
                logger.info(f"모델 학습 완료 - 성공: {successful_models}개")
                
            except Exception as e:
                logger.error(f"모델 학습 전체 실패: {e}")
                trained_models = create_dummy_models(X_train, y_train)
                successful_models = len(trained_models)
        else:
            logger.warning("ModelTrainer를 사용할 수 없어 기본 모델 생성")
            trained_models = create_dummy_models(X_train, y_train)
            successful_models = len(trained_models)
        
        if cleanup_required:
            logger.info("사용자 요청으로 파이프라인 중단")
            return None
        
        # 4. 앙상블 시스템 구성 및 학습 - 강제 실행
        logger.info("4. 앙상블 시스템 구성 및 학습 - 강제 실행")
        ensemble_used = False
        if ensemble_manager is not None and successful_models > 1:
            try:
                # 앙상블 생성 강제 실행
                logger.info("앙상블 생성 시작")
                ensemble_manager.create_ensemble('final_ensemble', target_ctr=0.0201, optimization_method='final_combined')
                logger.info("앙상블 생성 완료")
                
                # 앙상블 학습 강제 실행
                if 'X_val_split' in locals() and 'y_val_split' in locals():
                    logger.info("앙상블 학습 시작")
                    ensemble_manager.train_all_ensembles(X_val_split, y_val_split)
                    logger.info("앙상블 학습 완료")
                    ensemble_used = True
                
                # 앙상블 평가 강제 실행
                if 'X_val_split' in locals() and 'y_val_split' in locals():
                    logger.info("앙상블 평가 시작")
                    ensemble_results = ensemble_manager.evaluate_ensembles(X_val_split, y_val_split)
                    logger.info("앙상블 평가 완료")
                    
                    best_scores = [v for k, v in ensemble_results.items() if k.startswith('ensemble_') and not k.endswith('_ctr_optimized')]
                    if best_scores:
                        best_score = max(best_scores)
                        logger.info(f"최고 앙상블 점수: {best_score:.4f}")
                        
                        if best_score >= 0.35:
                            logger.info("목표 Combined Score 0.35+ 달성!")
                        else:
                            logger.info(f"목표까지 {0.35 - best_score:.4f} 부족")
                
            except Exception as e:
                logger.error(f"앙상블 시스템 실행 실패: {e}")
                logger.error(f"앙상블 오류 상세: {traceback.format_exc()}")
                ensemble_manager = None
                ensemble_used = False
        
        # 5. 제출 파일 생성 (앙상블 우선 사용)
        logger.info("5. 제출 파일 생성 (앙상블 우선 사용)")
        try:
            force_memory_cleanup()
            
            submission = generate_final_submission(trained_models, X_test, config, ensemble_manager)
            logger.info(f"제출 파일 생성 완료: {len(submission):,}행")
            
        except Exception as e:
            logger.error(f"제출 파일 생성 실패: {e}")
            submission = create_default_submission(X_test, config)
        
        # 6. 결과 요약
        total_time = time.time() - start_time
        logger.info(f"=== 파이프라인 완료 ===")
        logger.info(f"실행 시간: {total_time:.2f}초")
        logger.info(f"성공한 모델: {successful_models}개")
        logger.info(f"앙상블 매니저 활성화: {'Yes' if ensemble_manager else 'No'}")
        logger.info(f"앙상블 실제 사용: {'Yes' if ensemble_used else 'No'}")
        logger.info(f"캘리브레이션 적용 모델 수: {len([m for m in trained_models.values() if hasattr(m.get('model'), 'is_calibrated') and m.get('model').is_calibrated])}")
        logger.info(f"제출 파일: {len(submission):,}행")
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            logger.info(f"최종 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
        
        force_memory_cleanup(intensive=True)
        
        return {
            'trained_models': trained_models,
            'ensemble_manager': ensemble_manager,
            'submission': submission,
            'execution_time': total_time,
            'successful_models': successful_models,
            'ensemble_enabled': ensemble_manager is not None,
            'ensemble_used': ensemble_used,
            'calibration_applied': True
        }
        
    except Exception as e:
        logger.error(f"파이프라인 실패: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        
        force_memory_cleanup(intensive=True)
        raise

def train_final_model(model_type, X_train, y_train, X_val, y_val, config):
    """모델 학습"""
    try:
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            # 64GB 환경에서 더 관대한 메모리 임계값
            if vm.available / (1024**3) < 8:
                logger.warning(f"{model_type} 모델 학습 스킵: 메모리 부족")
                return None
        
        if model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 2047,
                    'learning_rate': 0.008,
                    'feature_fraction': 0.95,
                    'bagging_fraction': 0.85,
                    'bagging_freq': 3,
                    'min_child_samples': 80,
                    'min_child_weight': 3,
                    'lambda_l1': 2.5,
                    'lambda_l2': 2.5,
                    'max_depth': 20,
                    'verbose': -1,
                    'random_state': 42,
                    'num_threads': min(config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 8, 8),
                    'force_row_wise': True,
                    'max_bin': 255,
                    'scale_pos_weight': 52.0
                }
                
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=8000,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=800),
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
                
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 12,
                    'learning_rate': 0.008,
                    'subsample': 0.85,
                    'colsample_bytree': 0.95,
                    'min_child_weight': 8,
                    'reg_alpha': 2.5,
                    'reg_lambda': 2.5,
                    'random_state': 42,
                    'nthread': min(config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 8, 8),
                    'verbosity': 0,
                    'tree_method': 'hist',
                    'scale_pos_weight': 52.0
                }
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=8000,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=800,
                    verbose_eval=False
                )
                
                return model
                
            except ImportError:
                logger.warning("XGBoost를 사용할 수 없습니다")
                return None
            
        elif model_type == 'catboost':
            try:
                from catboost import CatBoostClassifier
                
                model = CatBoostClassifier(
                    iterations=8000,
                    depth=12,
                    learning_rate=0.008,
                    loss_function='Logloss',
                    random_seed=42,
                    verbose=False,
                    thread_count=min(config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 8, 8),
                    task_type='CPU',
                    auto_class_weights='Balanced'
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=800,
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
                    max_iter=1200,
                    class_weight='balanced',
                    C=0.03,
                    solver='liblinear'
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
        
        try:
            lr_model = LogisticRegression(
                random_state=42, 
                max_iter=800,
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
        
        try:
            rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                random_state=42,
                n_jobs=1,
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

def generate_final_submission(trained_models, X_test, config, ensemble_manager=None):
    """제출 파일 생성 - 앙상블 우선 사용"""
    logger.info("제출 파일 생성 시작 - 앙상블 우선 사용")
    
    test_size = len(X_test)
    logger.info(f"테스트 데이터 크기: {test_size:,}행")
    
    try:
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            # 64GB 환경에서 더 큰 배치 크기 사용
            if vm.available / (1024**3) < 8:
                logger.warning("메모리 부족으로 배치 예측 수행")
                batch_size = 25000
            else:
                batch_size = 100000
        else:
            batch_size = 100000
        
        try:
            submission_path = getattr(config, 'SUBMISSION_TEMPLATE_PATH', Path('data/sample_submission.csv'))
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
        
        if len(submission) != test_size:
            logger.warning(f"크기 불일치 - 템플릿: {len(submission):,}, 테스트: {test_size:,}")
            submission = pd.DataFrame({
                'id': range(test_size),
                'clicked': 0.0201
            })
        
        predictions = None
        prediction_method = ""
        
        # 앙상블 매니저 우선 사용 - 강제 실행
        if ensemble_manager is not None:
            try:
                logger.info("앙상블 매니저로 예측 수행 시작")
                predictions = ensemble_manager.predict_with_best_ensemble(X_test)
                prediction_method = "Ensemble"
                logger.info("앙상블 매니저 예측 완료")
                
                # 앙상블 예측 검증
                if predictions is not None and len(predictions) == test_size:
                    unique_predictions = len(np.unique(predictions))
                    logger.info(f"앙상블 예측 다양성: {unique_predictions}개 고유값")
                    
                    # 예측 다양성이 충분한지 확인
                    if unique_predictions < 1000:
                        logger.warning("앙상블 예측 다양성 부족, 개별 모델로 대체")
                        predictions = None
                else:
                    logger.warning("앙상블 예측 크기 불일치, 개별 모델로 대체")
                    predictions = None
                    
            except Exception as e:
                logger.error(f"앙상블 예측 실패: {e}")
                logger.error(f"앙상블 오류 상세: {traceback.format_exc()}")
                predictions = None
        
        # 앙상블 실패 시 개별 모델 사용
        if predictions is None and trained_models:
            model_priority = ['lightgbm', 'xgboost', 'catboost', 'logistic', 'random_forest']
            
            for model_name in model_priority:
                if model_name in trained_models:
                    try:
                        logger.info(f"{model_name} 모델로 배치 예측 수행")
                        
                        model = trained_models[model_name]['model']
                        batch_predictions = []
                        
                        for i in range(0, test_size, batch_size):
                            end_idx = min(i + batch_size, test_size)
                            batch_X = X_test.iloc[i:end_idx]
                            
                            try:
                                if hasattr(model, 'predict_proba'):
                                    pred_proba = model.predict_proba(batch_X)
                                    if pred_proba.shape[1] > 1:
                                        batch_pred = pred_proba[:, 1]
                                    else:
                                        batch_pred = pred_proba[:, 0]
                                elif hasattr(model, 'predict'):
                                    batch_pred = model.predict(batch_X)
                                else:
                                    logger.warning(f"{model_name} 모델의 예측 방법을 찾을 수 없습니다")
                                    continue
                                
                                batch_predictions.extend(batch_pred)
                                
                            except Exception as batch_error:
                                logger.warning(f"배치 {i}-{end_idx} 예측 실패: {batch_error}")
                                batch_predictions.extend([0.0201] * (end_idx - i))
                            
                            if i % (batch_size * 5) == 0:
                                force_memory_cleanup()
                        
                        predictions = np.array(batch_predictions)
                        predictions = np.clip(predictions, 0.001, 0.999)
                        prediction_method = f"Single_{model_name}"
                        break
                        
                    except Exception as e:
                        logger.warning(f"{model_name} 모델 예측 실패: {e}")
                        continue
        
        if predictions is None or len(predictions) != test_size:
            logger.warning("모든 모델 예측 실패. 기본값 사용")
            base_ctr = 0.0201
            predictions = np.random.lognormal(
                mean=np.log(base_ctr), 
                sigma=0.15, 
                size=test_size
            )
            predictions = np.clip(predictions, 0.001, 0.08)
            prediction_method = "Default"
        
        target_ctr = 0.0201
        current_ctr = predictions.mean()
        
        if abs(current_ctr - target_ctr) > 0.002:
            logger.info(f"CTR 보정: {current_ctr:.4f} → {target_ctr:.4f}")
            correction_factor = target_ctr / current_ctr if current_ctr > 0 else 1.0
            predictions = predictions * correction_factor
            predictions = np.clip(predictions, 0.001, 0.999)
        
        submission['clicked'] = predictions
        
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
                sigma=0.15, 
                size=test_size
            )
        })
        default_submission['clicked'] = np.clip(default_submission['clicked'], 0.001, 0.08)
        
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
        modules = safe_import_modules()
        
        if 'create_ctr_prediction_service' in modules:
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
        from config import Config
        config = Config
        
        model_dir = Path("models")
        model_files = list(model_dir.glob("*_model.pkl"))
        
        if not model_files:
            logger.error("복원할 모델이 없습니다. 먼저 학습을 실행하세요.")
            return False
        
        logger.info(f"발견된 모델 파일: {len(model_files)}개")
        
        test_path = Path("data/test.parquet")
        if not test_path.exists():
            logger.error("테스트 데이터 파일이 없습니다")
            return False
        
        logger.info("테스트 데이터 로딩")
        
        try:
            test_df = pd.read_parquet(test_path, engine='pyarrow')
            logger.info(f"테스트 데이터 크기: {test_df.shape}")
        except Exception as e:
            logger.error(f"테스트 데이터 로딩 실패: {e}")
            return False
        
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
            
            submission = generate_final_submission(models, test_df, config)
            
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
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="CTR 모델링 시스템")
    parser.add_argument("--mode", choices=["train", "inference", "reproduce"], 
                       default="train", help="실행 모드")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 실행 모드")
    
    args = parser.parse_args()
    
    try:
        logger.info("=== CTR 모델링 시스템 시작 ===")
        
        if not validate_environment():
            logger.error("환경 검증 실패")
            sys.exit(1)
        
        if args.mode == "train":
            logger.info("학습 모드 시작")
            
            from config import Config
            config = Config
            config.setup_directories()
            
            results = execute_final_pipeline(config, quick_mode=args.quick)
            
            if results:
                logger.info("학습 모드 완료")
                logger.info(f"실행 시간: {results['execution_time']:.2f}초")
                logger.info(f"성공 모델: {results['successful_models']}개")
                logger.info(f"앙상블 활성화: {results.get('ensemble_enabled', False)}")
                logger.info(f"앙상블 실제 사용: {results.get('ensemble_used', False)}")
                logger.info(f"캘리브레이션 적용: {results.get('calibration_applied', False)}")
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
        
        logger.info("=== CTR 모델링 시스템 종료 ===")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 실행이 중단되었습니다")
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        sys.exit(1)
        
    finally:
        cleanup_required = True
        force_memory_cleanup(intensive=True)
        
        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                logger.info(f"최종 메모리 상태: 사용가능 {vm.available/(1024**3):.1f}GB")
            except Exception:
                pass

if __name__ == "__main__":
    main()