# training.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import json
from pathlib import Path
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import warnings
warnings.filterwarnings('ignore')

# GPU 최적화 import
TORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        try:
            # RTX 4060 Ti 16GB 최적화 테스트
            test_tensor = torch.zeros(2000, 2000).cuda()
            test_result = test_tensor.sum().item()
            del test_tensor
            torch.cuda.empty_cache()
            TORCH_AVAILABLE = True
            
            # RTX 4060 Ti 메모리 최적화 설정
            torch.cuda.set_per_process_memory_fraction(0.85)  # 16GB의 85% 사용
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        except Exception as e:
            logging.warning(f"GPU 최적화 테스트 실패: {e}")
            TORCH_AVAILABLE = True
    else:
        TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch가 설치되지 않았습니다.")

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna가 설치되지 않았습니다.")

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import make_scorer

from config import Config
from models import ModelFactory, BaseModel, CTRCalibrator
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class AdvancedMemoryManager:
    """대용량 데이터 특화 고급 메모리 관리자"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_threshold = 32.0  # 32GB 임계값
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.cleanup_count = 0
        
    def get_memory_status(self) -> Dict[str, float]:
        """상세 메모리 상태 반환"""
        vm = psutil.virtual_memory()
        process_memory = self.process.memory_info().rss / (1024**3)
        
        status = {
            'process_gb': process_memory,
            'available_gb': vm.available / (1024**3),
            'total_gb': vm.total / (1024**3),
            'usage_percent': vm.percent,
            'swap_used_gb': psutil.swap_memory().used / (1024**3)
        }
        
        if self.gpu_available:
            try:
                gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                status.update({
                    'gpu_used_gb': gpu_memory,
                    'gpu_reserved_gb': gpu_reserved,
                    'gpu_total_gb': gpu_total,
                    'gpu_free_gb': gpu_total - gpu_reserved
                })
            except Exception:
                pass
        
        return status
    
    def aggressive_cleanup(self):
        """공격적 메모리 정리"""
        self.cleanup_count += 1
        
        # Python 가비지 컬렉션
        for _ in range(3):
            collected = gc.collect()
            if collected == 0:
                break
        
        # GPU 메모리 정리
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        
        # 시스템 메모리 최적화
        try:
            import ctypes
            if hasattr(ctypes, 'windll'):
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except Exception:
            pass
        
        logger.debug(f"메모리 정리 완료 #{self.cleanup_count}")
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        status = self.get_memory_status()
        
        # CPU 메모리 압박
        cpu_pressure = (status['available_gb'] < 8 or 
                       status['usage_percent'] > 85 or 
                       status['process_gb'] > self.memory_threshold)
        
        # GPU 메모리 압박
        gpu_pressure = False
        if self.gpu_available and 'gpu_free_gb' in status:
            gpu_pressure = status['gpu_free_gb'] < 2
        
        return cpu_pressure or gpu_pressure

class HighPerformanceCTRTrainer:
    """고성능 CTR 모델 학습기 - 1070만행 특화"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.memory_manager = AdvancedMemoryManager()
        
        # 학습 결과 저장
        self.trained_models = {}
        self.cv_results = {}
        self.best_params = {}
        self.calibrators = {}
        self.performance_history = []
        
        # RTX 4060 Ti 최적화 설정
        self.gpu_optimized = False
        if self.device == 'cuda':
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = gpu_props.total_memory / (1024**3)
                
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # RTX 4060 Ti 특화 최적화
                if 'RTX 4060 Ti' in gpu_name or gpu_memory > 14:
                    self.gpu_optimized = True
                    torch.cuda.set_per_process_memory_fraction(0.9)
                    logger.info("RTX 4060 Ti 최적화 활성화")
                
            except Exception as e:
                logger.warning(f"GPU 최적화 실패: {e}")
    
    def get_optimized_batch_size(self, model_type: str, data_size: int) -> int:
        """모델별 최적 배치 크기 계산"""
        memory_status = self.memory_manager.get_memory_status()
        available_memory = memory_status['available_gb']
        
        # 기본 배치 크기
        base_batch_sizes = {
            'lightgbm': min(100000, max(10000, data_size // 100)),
            'xgboost': min(80000, max(8000, data_size // 120)),
            'catboost': min(60000, max(6000, data_size // 150)),
            'deepctr': 4096 if self.gpu_optimized else 2048
        }
        
        batch_size = base_batch_sizes.get(model_type, 10000)
        
        # 메모리 기반 조정
        if available_memory < 20:
            batch_size = int(batch_size * 0.6)
        elif available_memory > 35:
            batch_size = int(batch_size * 1.4)
        
        # GPU 메모리 기반 조정 (DeepCTR)
        if model_type == 'deepctr' and self.gpu_optimized:
            gpu_memory = memory_status.get('gpu_free_gb', 8)
            if gpu_memory > 12:
                batch_size = 8192
            elif gpu_memory > 8:
                batch_size = 6144
            elif gpu_memory > 4:
                batch_size = 4096
            else:
                batch_size = 2048
        
        return max(1000, batch_size)
    
    def get_high_performance_params(self, model_type: str, data_size: int) -> Dict[str, Any]:
        """고성능 파라미터 설정 - Combined Score 0.30+ 목표"""
        
        if model_type.lower() == 'lightgbm':
            # 1070만행 특화 LightGBM 파라미터
            return {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 511,  # 최대값으로 설정
                'learning_rate': 0.02,  # 낮은 학습률로 안정성 확보
                'feature_fraction': 0.85,
                'bagging_fraction': 0.8,
                'bagging_freq': 3,
                'min_child_samples': 300,  # 대용량 데이터용 증가
                'min_child_weight': 8,
                'lambda_l1': 1.5,
                'lambda_l2': 1.5,
                'max_depth': 15,
                'max_bin': 255,
                'path_smooth': 2.0,
                'verbose': -1,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 5000,  # 대용량 데이터용 증가
                'early_stopping_rounds': 100,
                'scale_pos_weight': 49.0,
                'force_row_wise': True,
                'num_threads': 12,  # Ryzen 5600X 최적화
                'device_type': 'cpu',
                'min_data_in_leaf': 150,
                'feature_fraction_bynode': 0.85,
                'extra_trees': True,
                'grow_policy': 'lossguide',
                'max_cat_threshold': 64,
                'cat_l2': 1.0,
                'cat_smooth': 10.0
            }
        
        elif model_type.lower() == 'xgboost':
            # 1070만행 특화 XGBoost 파라미터
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'max_depth': 10,
                'learning_rate': 0.02,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'colsample_bylevel': 0.85,
                'colsample_bynode': 0.85,
                'min_child_weight': 12,
                'reg_alpha': 1.5,
                'reg_lambda': 1.5,
                'scale_pos_weight': 49.0,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 5000,
                'early_stopping_rounds': 100,
                'max_bin': 255,
                'nthread': 12,
                'grow_policy': 'lossguide',
                'max_leaves': 511,
                'gamma': 0.05,
                'validate_parameters': True,
                'predictor': 'cpu_predictor'
            }
            
            # GPU 최적화
            if self.gpu_optimized:
                params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor'
                })
            
            return params
        
        elif model_type.lower() == 'catboost':
            # 1070만행 특화 CatBoost 파라미터
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'task_type': 'CPU',
                'depth': 10,
                'learning_rate': 0.02,
                'l2_leaf_reg': 8,
                'iterations': 5000,
                'random_seed': self.config.RANDOM_STATE,
                'od_wait': 100,
                'od_type': 'IncToDec',
                'verbose': False,
                'auto_class_weights': 'Balanced',
                'max_ctr_complexity': 4,
                'thread_count': 12,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.2,
                'leaf_estimation_iterations': 12,
                'leaf_estimation_method': 'Newton',
                'grow_policy': 'Lossguide',
                'max_leaves': 511,
                'min_data_in_leaf': 80,
                'model_size_reg': 0.1,
                'feature_border_type': 'GreedyLogSum',
                'ctr_leaf_count_limit': 64
            }
            
            # GPU 최적화
            if self.gpu_optimized:
                params.update({
                    'task_type': 'GPU',
                    'devices': '0',
                    'gpu_ram_part': 0.8
                })
            
            return params
        
        elif model_type.lower() == 'deepctr':
            # RTX 4060 Ti 특화 DeepCTR 파라미터
            return {
                'hidden_dims': [1024, 768, 512, 256, 128, 64],  # 더 깊은 네트워크
                'dropout_rate': 0.3,
                'learning_rate': 0.0008,
                'weight_decay': 5e-6,
                'batch_size': self.get_optimized_batch_size('deepctr', data_size),
                'epochs': 80,
                'patience': 20,
                'use_batch_norm': True,
                'activation': 'swish',
                'use_residual': True,
                'use_attention': True,
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0,
                'label_smoothing': 0.02,
                'gradient_accumulation_steps': 2,
                'warmup_steps': 1000,
                'use_mixed_precision': self.gpu_optimized,
                'use_gradient_checkpointing': True
            }
        
        else:
            return {}
    
    def advanced_hyperparameter_tuning(self, 
                                      model_type: str,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      n_trials: int = None,
                                      cv_folds: int = 3) -> Dict[str, Any]:
        """고급 하이퍼파라미터 튜닝 - Combined Score 0.30+ 목표"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna가 없어 기본 고성능 파라미터 사용")
            return self.get_high_performance_params(model_type, len(X))
        
        memory_status = self.memory_manager.get_memory_status()
        
        # 메모리 기반 trials 수 조정
        if n_trials is None:
            if memory_status['available_gb'] > 30:
                n_trials = 50 if model_type != 'deepctr' else 30
            elif memory_status['available_gb'] > 20:
                n_trials = 35 if model_type != 'deepctr' else 20
            else:
                n_trials = 25 if model_type != 'deepctr' else 15
        
        # 대용량 데이터용 샘플링
        if len(X) > 5000000:
            sample_size = min(2000000, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices].copy()
            y_sample = y.iloc[sample_indices].copy()
            logger.info(f"튜닝용 데이터 샘플링: {len(X):,} → {sample_size:,}")
        else:
            X_sample = X
            y_sample = y
        
        logger.info(f"{model_type} 고급 하이퍼파라미터 튜닝 시작 (trials: {n_trials})")
        
        def high_performance_objective(trial):
            """고성능 목표 objective 함수"""
            try:
                if self.memory_manager.check_memory_pressure():
                    self.memory_manager.aggressive_cleanup()
                
                if model_type.lower() == 'lightgbm':
                    params = {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 255, 511),
                        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.035, log=True),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.75, 0.9),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.85),
                        'bagging_freq': trial.suggest_int('bagging_freq', 2, 5),
                        'min_child_samples': trial.suggest_int('min_child_samples', 200, 500),
                        'min_child_weight': trial.suggest_float('min_child_weight', 5, 15),
                        'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 3.0),
                        'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 3.0),
                        'max_depth': trial.suggest_int('max_depth', 12, 18),
                        'path_smooth': trial.suggest_float('path_smooth', 1.0, 3.0),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 45, 55),
                        'verbose': -1,
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 3000,
                        'early_stopping_rounds': 80,
                        'force_row_wise': True,
                        'num_threads': 12,
                        'max_bin': 255,
                        'grow_policy': 'lossguide',
                        'extra_trees': True
                    }
                
                elif model_type.lower() == 'xgboost':
                    base_params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'tree_method': 'gpu_hist' if self.gpu_optimized else 'hist',
                        'max_depth': trial.suggest_int('max_depth', 8, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.035, log=True),
                        'subsample': trial.suggest_float('subsample', 0.75, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.9),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.75, 0.9),
                        'min_child_weight': trial.suggest_float('min_child_weight', 8, 16),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 3.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
                        'gamma': trial.suggest_float('gamma', 0.0, 0.2),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 45, 55),
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 3000,
                        'early_stopping_rounds': 80,
                        'nthread': 12,
                        'grow_policy': 'lossguide',
                        'max_leaves': 511
                    }
                    
                    if self.gpu_optimized:
                        base_params['gpu_id'] = 0
                    
                    params = base_params
                
                elif model_type.lower() == 'catboost':
                    base_params = {
                        'loss_function': 'Logloss',
                        'eval_metric': 'Logloss',
                        'task_type': 'GPU' if self.gpu_optimized else 'CPU',
                        'depth': trial.suggest_int('depth', 8, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.035, log=True),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 15),
                        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.8, 1.5),
                        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 8, 15),
                        'max_leaves': trial.suggest_int('max_leaves', 255, 511),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 120),
                        'iterations': 3000,
                        'random_seed': self.config.RANDOM_STATE,
                        'od_wait': 80,
                        'od_type': 'IncToDec',
                        'verbose': False,
                        'auto_class_weights': 'Balanced',
                        'thread_count': 12,
                        'grow_policy': 'Lossguide'
                    }
                    
                    if self.gpu_optimized:
                        base_params['devices'] = '0'
                        base_params['gpu_ram_part'] = 0.8
                    
                    params = base_params
                
                elif model_type.lower() == 'deepctr':
                    if not self.gpu_optimized:
                        return 0.0
                    
                    hidden_options = [
                        [512, 256, 128, 64],
                        [1024, 512, 256, 128],
                        [1024, 768, 512, 256, 128],
                        [1024, 768, 512, 256, 128, 64],
                        [2048, 1024, 512, 256, 128]
                    ]
                    params = {
                        'hidden_dims': trial.suggest_categorical('hidden_dims', hidden_options),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.4),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.002, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [2048, 4096, 6144, 8192]),
                        'epochs': 60,
                        'patience': 15,
                        'use_batch_norm': True,
                        'activation': trial.suggest_categorical('activation', ['relu', 'swish', 'gelu']),
                        'use_residual': trial.suggest_categorical('use_residual', [True, False]),
                        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
                        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.05),
                        'use_mixed_precision': self.gpu_optimized
                    }
                
                else:
                    return 0.0
                
                # 교차검증 실행
                cv_result = self.advanced_cross_validation(
                    model_type, X_sample, y_sample, cv_folds, params
                )
                
                combined_score = cv_result.get('combined_mean', 0.0)
                ctr_bias_penalty = cv_result.get('ctr_bias_penalty', 1.0)
                stability_bonus = cv_result.get('stability_bonus', 1.0)
                
                # 최종 점수 (Combined Score 0.30+ 목표)
                final_score = combined_score * ctr_bias_penalty * stability_bonus
                
                return final_score if final_score > 0 else 0.0
            
            except Exception as e:
                logger.error(f"Hyperparameter trial 실패: {e}")
                return 0.0
        
        try:
            # Optuna 스터디 생성 (고성능 최적화)
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(
                    seed=self.config.RANDOM_STATE,
                    n_startup_trials=15,
                    n_ei_candidates=48,
                    multivariate=True,
                    warn_independent_sampling=False
                ),
                pruner=HyperbandPruner(
                    min_resource=3,
                    max_resource=cv_folds,
                    reduction_factor=2
                )
            )
            
            study.optimize(
                high_performance_objective,
                n_trials=n_trials,
                timeout=3600,  # 1시간 제한
                n_jobs=1,
                show_progress_bar=False,
                gc_after_trial=True
            )
            
            if study.best_value and study.best_value > 0:
                logger.info(f"{model_type} 튜닝 완료 - 최고 점수: {study.best_value:.4f}")
                self.best_params[model_type] = study.best_params
                return study.best_params
            else:
                logger.warning(f"{model_type} 튜닝에서 유효한 결과를 얻지 못했습니다.")
                
        except Exception as e:
            logger.error(f"{model_type} 하이퍼파라미터 튜닝 실패: {e}")
        finally:
            self.memory_manager.aggressive_cleanup()
        
        # 기본 고성능 파라미터 반환
        return self.get_high_performance_params(model_type, len(X))
    
    def advanced_cross_validation(self, 
                                 model_type: str,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 cv_folds: int = 5,
                                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """고급 교차검증 - TimeSeriesSplit + 안정성 평가"""
        
        logger.info(f"{model_type} 고급 교차검증 시작 (folds: {cv_folds})")
        
        # 파라미터 검증
        if params is None:
            params = self.get_high_performance_params(model_type, len(X))
        
        # 메모리 기반 데이터 크기 조정
        if self.memory_manager.check_memory_pressure() and len(X) > 2000000:
            sample_size = min(1500000, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[sample_indices].copy()
            y = y.iloc[sample_indices].copy()
            logger.info(f"메모리 절약을 위한 샘플링: {sample_size:,}행")
        
        # TimeSeriesSplit for temporal consistency
        tscv = TimeSeriesSplit(n_splits=cv_folds, test_size=None)
        
        cv_scores = {
            'ap_scores': [],
            'wll_scores': [],
            'combined_scores': [],
            'ctr_biases': [],
            'training_times': [],
            'prediction_diversities': []
        }
        
        metrics_calculator = CTRMetrics()
        actual_ctr = y.mean()
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"폴드 {fold + 1}/{cv_folds} 시작")
            fold_start_time = time.time()
            
            try:
                # 폴드 데이터 준비
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # 모델 생성 및 학습
                model_kwargs = {'params': params}
                if model_type == 'deepctr':
                    model_kwargs['input_dim'] = X_train_fold.shape[1]
                
                model = ModelFactory.create_model(model_type, **model_kwargs)
                model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                
                # 예측 및 평가
                y_pred_proba = model.predict_proba(X_val_fold)
                
                # 다양성 검증
                prediction_diversity = len(np.unique(np.round(y_pred_proba, 6))) / len(y_pred_proba)
                
                # 메트릭 계산
                ap_score = metrics_calculator.average_precision(y_val_fold, y_pred_proba)
                wll_score = metrics_calculator.weighted_log_loss(y_val_fold, y_pred_proba)
                combined_score = metrics_calculator.combined_score(y_val_fold, y_pred_proba)
                
                # CTR 편향 계산
                predicted_ctr = y_pred_proba.mean()
                fold_actual_ctr = y_val_fold.mean()
                ctr_bias = abs(predicted_ctr - fold_actual_ctr)
                
                fold_time = time.time() - fold_start_time
                
                # 결과 저장
                cv_scores['ap_scores'].append(ap_score)
                cv_scores['wll_scores'].append(wll_score)
                cv_scores['combined_scores'].append(combined_score)
                cv_scores['ctr_biases'].append(ctr_bias)
                cv_scores['training_times'].append(fold_time)
                cv_scores['prediction_diversities'].append(prediction_diversity)
                
                logger.info(f"폴드 {fold + 1} 완료 - Combined: {combined_score:.4f}, CTR편향: {ctr_bias:.4f}")
                
                # 메모리 정리
                del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model
                self.memory_manager.aggressive_cleanup()
                
            except Exception as e:
                logger.error(f"폴드 {fold + 1} 실패: {e}")
                cv_scores['ap_scores'].append(0.0)
                cv_scores['wll_scores'].append(float('inf'))
                cv_scores['combined_scores'].append(0.0)
                cv_scores['ctr_biases'].append(1.0)
                cv_scores['training_times'].append(0.0)
                cv_scores['prediction_diversities'].append(0.0)
        
        # 결과 분석
        valid_scores = [s for s in cv_scores['combined_scores'] if s > 0]
        valid_biases = [b for b in cv_scores['ctr_biases'] if b < 1.0]
        
        if not valid_scores:
            logger.warning(f"{model_type} 모든 폴드가 실패했습니다.")
            return {'combined_mean': 0.0, 'ctr_bias_penalty': 0.0, 'stability_bonus': 0.0}
        
        # 통계 계산
        combined_mean = np.mean(valid_scores)
        combined_std = np.std(valid_scores) if len(valid_scores) > 1 else 0.0
        ctr_bias_mean = np.mean(valid_biases) if valid_biases else 0.1
        
        # CTR 편향 패널티 (목표: 0.001 이하)
        ctr_bias_penalty = np.exp(-ctr_bias_mean * 1000) if ctr_bias_mean < 0.01 else 0.5
        
        # 안정성 보너스
        cv_coefficient = combined_std / combined_mean if combined_mean > 0 else 1.0
        stability_bonus = np.exp(-cv_coefficient * 2) if cv_coefficient < 0.2 else 0.8
        
        # 다양성 보너스
        avg_diversity = np.mean([d for d in cv_scores['prediction_diversities'] if d > 0])
        diversity_bonus = min(1.2, 1.0 + avg_diversity * 0.5)
        
        cv_results = {
            'model_type': model_type,
            'combined_mean': combined_mean,
            'combined_std': combined_std,
            'ctr_bias_mean': ctr_bias_mean,
            'ctr_bias_penalty': ctr_bias_penalty,
            'stability_bonus': stability_bonus,
            'diversity_bonus': diversity_bonus,
            'final_score': combined_mean * ctr_bias_penalty * stability_bonus * diversity_bonus,
            'successful_folds': len(valid_scores),
            'avg_training_time': np.mean([t for t in cv_scores['training_times'] if t > 0]),
            'scores_detail': cv_scores,
            'params': params
        }
        
        self.cv_results[model_type] = cv_results
        
        logger.info(f"{model_type} 교차검증 완료")
        logger.info(f"Combined Score: {combined_mean:.4f} (±{combined_std:.4f})")
        logger.info(f"CTR 편향: {ctr_bias_mean:.4f} (패널티: {ctr_bias_penalty:.3f})")
        logger.info(f"안정성: {stability_bonus:.3f}, 다양성: {diversity_bonus:.3f}")
        logger.info(f"최종 점수: {cv_results['final_score']:.4f}")
        
        return cv_results
    
    def train_high_performance_model(self,
                                   model_type: str,
                                   X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_val: Optional[pd.DataFrame] = None,
                                   y_val: Optional[pd.Series] = None,
                                   params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """고성능 모델 학습"""
        
        logger.info(f"{model_type} 고성능 모델 학습 시작")
        start_time = time.time()
        memory_before = self.memory_manager.get_memory_status()
        
        try:
            # 파라미터 준비
            if params is None:
                params = self.best_params.get(model_type) or self.get_high_performance_params(model_type, len(X_train))
            
            # 메모리 기반 데이터 조정
            if self.memory_manager.check_memory_pressure():
                if len(X_train) > 3000000:
                    sample_size = 2500000
                    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                    X_train = X_train.iloc[sample_indices].copy()
                    y_train = y_train.iloc[sample_indices].copy()
                    logger.info(f"메모리 절약을 위한 학습 데이터 샘플링: {sample_size:,}")
                
                if X_val is not None and len(X_val) > 500000:
                    val_indices = np.random.choice(len(X_val), 400000, replace=False)
                    X_val = X_val.iloc[val_indices].copy()
                    y_val = y_val.iloc[val_indices].copy()
            
            # 모델 생성
            model_kwargs = {'params': params}
            if model_type == 'deepctr':
                model_kwargs['input_dim'] = X_train.shape[1]
            
            model = ModelFactory.create_model(model_type, **model_kwargs)
            
            # 학습 실행
            model.fit(X_train, y_train, X_val, y_val)
            
            # 고성능 Calibration 적용
            if X_val is not None and y_val is not None:
                try:
                    self.apply_advanced_calibration(model, X_val, y_val)
                except Exception as e:
                    logger.warning(f"고급 Calibration 실패: {e}")
            
            training_time = time.time() - start_time
            memory_after = self.memory_manager.get_memory_status()
            
            # 모델 정보 저장
            self.trained_models[model_type] = {
                'model': model,
                'params': params,
                'training_time': training_time,
                'memory_used': memory_after['process_gb'] - memory_before['process_gb'],
                'calibrated': True,
                'performance_target': 'Combined Score 0.30+'
            }
            
            logger.info(f"{model_type} 고성능 학습 완료 (시간: {training_time:.2f}초)")
            logger.info(f"메모리 사용: {memory_after['process_gb'] - memory_before['process_gb']:+.2f}GB")
            
            return model
            
        except Exception as e:
            logger.error(f"{model_type} 고성능 학습 실패: {e}")
            self.memory_manager.aggressive_cleanup()
            raise
    
    def apply_advanced_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """고급 Calibration - CTR 편향 0.001 이하 목표"""
        
        logger.info(f"{model.name} 고급 Calibration 시작")
        
        try:
            # 메모리 효율을 위한 샘플링
            if len(X_val) > 100000:
                sample_indices = np.random.choice(len(X_val), 80000, replace=False)
                X_val_sample = X_val.iloc[sample_indices]
                y_val_sample = y_val.iloc[sample_indices]
            else:
                X_val_sample = X_val
                y_val_sample = y_val
            
            # 원본 예측
            raw_predictions = model.predict_proba_raw(X_val_sample)
            
            # 고급 CTR Calibrator
            calibrator = CTRCalibrator(target_ctr=self.config.CALIBRATION_CONFIG['target_ctr'])
            
            # 다중 Calibration 방법 적용
            calibrator.fit_platt_scaling(y_val_sample.values, raw_predictions)
            calibrator.fit_isotonic_regression(y_val_sample.values, raw_predictions)
            calibrator.fit_temperature_scaling(y_val_sample.values, raw_predictions)
            calibrator.fit_bias_correction(y_val_sample.values, raw_predictions)
            
            # 최고 성능 Calibration 선택
            model.apply_calibration(X_val_sample, y_val_sample, method='platt', cv_folds=5)
            
            # Calibrator 저장
            self.calibrators[model.name] = calibrator
            
            # 결과 검증
            calibrated_predictions = model.predict_proba(X_val_sample)
            original_ctr = raw_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val_sample.mean()
            
            ctr_improvement = abs(calibrated_ctr - actual_ctr) - abs(original_ctr - actual_ctr)
            
            logger.info(f"고급 Calibration 완료:")
            logger.info(f"  원본 CTR: {original_ctr:.4f}")
            logger.info(f"  보정 CTR: {calibrated_ctr:.4f}")
            logger.info(f"  실제 CTR: {actual_ctr:.4f}")
            logger.info(f"  편향 개선: {ctr_improvement:+.4f}")
            
        except Exception as e:
            logger.error(f"고급 Calibration 실패: {e}")
    
    def train_all_high_performance_models(self,
                                         X_train: pd.DataFrame,
                                         y_train: pd.Series,
                                         X_val: Optional[pd.DataFrame] = None,
                                         y_val: Optional[pd.Series] = None,
                                         enable_hyperparameter_tuning: bool = True) -> Dict[str, BaseModel]:
        """모든 고성능 모델 학습 - Combined Score 0.30+ 목표"""
        
        logger.info("=== 고성능 모델 학습 파이프라인 시작 ===")
        logger.info(f"목표: Combined Score 0.30+, CTR 편향 0.001 이하")
        
        memory_status = self.memory_manager.get_memory_status()
        logger.info(f"시스템 상태: CPU {memory_status['available_gb']:.1f}GB")
        if self.gpu_optimized:
            logger.info(f"GPU: {memory_status.get('gpu_free_gb', 0):.1f}GB 사용가능")
        
        # 모델 우선순위 (성능 기반)
        priority_models = ['lightgbm', 'xgboost', 'catboost']
        if self.gpu_optimized and memory_status['available_gb'] > 15:
            priority_models.append('deepctr')
            logger.info("GPU 최적화 환경: DeepCTR 모델 추가")
        
        trained_models = {}
        
        # 하이퍼파라미터 튜닝 단계
        if enable_hyperparameter_tuning:
            logger.info("=== 하이퍼파라미터 튜닝 단계 ===")
            for model_type in priority_models:
                try:
                    if self.memory_manager.check_memory_pressure():
                        logger.warning(f"메모리 압박으로 {model_type} 튜닝 생략")
                        continue
                    
                    logger.info(f"{model_type} 튜닝 시작")
                    
                    tuned_params = self.advanced_hyperparameter_tuning(
                        model_type, X_train, y_train,
                        n_trials=30 if model_type != 'deepctr' else 20,
                        cv_folds=5
                    )
                    
                    if tuned_params:
                        self.best_params[model_type] = tuned_params
                        logger.info(f"{model_type} 튜닝 완료")
                    
                    self.memory_manager.aggressive_cleanup()
                    
                except Exception as e:
                    logger.error(f"{model_type} 튜닝 실패: {e}")
                    continue
        
        # 모델 학습 단계
        logger.info("=== 고성능 모델 학습 단계 ===")
        for model_type in priority_models:
            try:
                if self.memory_manager.check_memory_pressure():
                    logger.warning(f"메모리 압박으로 {model_type} 학습 생략")
                    continue
                
                logger.info(f"{model_type} 고성능 학습 시작")
                
                model = self.train_high_performance_model(
                    model_type, X_train, y_train, X_val, y_val
                )
                
                if model:
                    trained_models[model_type] = model
                    logger.info(f"{model_type} 학습 성공")
                
                self.memory_manager.aggressive_cleanup()
                
            except Exception as e:
                logger.error(f"{model_type} 학습 실패: {e}")
                continue
        
        # 성능 요약
        logger.info("=== 고성능 학습 결과 요약 ===")
        logger.info(f"성공한 모델: {list(trained_models.keys())}")
        logger.info(f"총 모델 수: {len(trained_models)}")
        
        if self.cv_results:
            best_cv = max(self.cv_results.items(), key=lambda x: x[1].get('final_score', 0))
            logger.info(f"최고 CV 성능: {best_cv[0]} (점수: {best_cv[1]['final_score']:.4f})")
        
        final_memory = self.memory_manager.get_memory_status()
        logger.info(f"최종 메모리: {final_memory['process_gb']:.1f}GB")
        
        return trained_models
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        return {
            'trained_models': list(self.trained_models.keys()),
            'model_count': len(self.trained_models),
            'gpu_optimized': self.gpu_optimized,
            'cv_results': self.cv_results,
            'best_params': self.best_params,
            'target_achieved': any(
                cv.get('combined_mean', 0) >= 0.30 
                for cv in self.cv_results.values()
            ),
            'ctr_bias_target_achieved': any(
                cv.get('ctr_bias_mean', 1) <= 0.001
                for cv in self.cv_results.values()
            ),
            'memory_optimized': True,
            'performance_history': self.performance_history
        }
    
    def save_high_performance_models(self, output_dir: Path = None):
        """고성능 모델 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"고성능 모델 저장: {output_dir}")
        
        for model_name, model_info in self.trained_models.items():
            try:
                # 모델 저장
                model_path = output_dir / f"{model_name}_high_performance.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['model'], f)
                
                # 메타데이터 저장
                metadata = {
                    'model_type': model_name,
                    'performance_target': 'Combined Score 0.30+',
                    'ctr_bias_target': 0.001,
                    'training_time': model_info.get('training_time', 0.0),
                    'params': model_info.get('params', {}),
                    'memory_used': model_info.get('memory_used', 0.0),
                    'gpu_optimized': self.gpu_optimized,
                    'calibrated': model_info.get('calibrated', False),
                    'cv_result': self.cv_results.get(model_name, {}),
                    'timestamp': time.time()
                }
                
                metadata_path = output_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info(f"{model_name} 고성능 모델 저장 완료")
                
            except Exception as e:
                logger.error(f"{model_name} 저장 실패: {e}")
        
        # Calibrator 저장
        if self.calibrators:
            calibrator_path = output_dir / "high_performance_calibrators.pkl"
            with open(calibrator_path, 'wb') as f:
                pickle.dump(self.calibrators, f)
            logger.info("고성능 Calibrator 저장 완료")
        
        # 성능 요약 저장
        summary = self.get_performance_summary()
        summary_path = output_dir / "high_performance_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("고성능 요약 정보 저장 완료")

# 메인 클래스 alias
CTRModelTrainer = HighPerformanceCTRTrainer
ModelTrainer = HighPerformanceCTRTrainer

class HighPerformanceTrainingPipeline:
    """고성능 학습 파이프라인"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = HighPerformanceCTRTrainer(config)
        
    def run_high_performance_pipeline(self,
                                    X_train: pd.DataFrame,
                                    y_train: pd.Series,
                                    X_val: Optional[pd.DataFrame] = None,
                                    y_val: Optional[pd.Series] = None,
                                    enable_hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """고성능 파이프라인 실행"""
        
        logger.info("=== 고성능 CTR 학습 파이프라인 시작 ===")
        logger.info("목표: Combined Score 0.30+, CTR 편향 0.001 이하")
        
        start_time = time.time()
        
        try:
            # 고성능 모델 학습
            trained_models = self.trainer.train_all_high_performance_models(
                X_train, y_train, X_val, y_val,
                enable_hyperparameter_tuning=enable_hyperparameter_tuning
            )
            
            # 모델 저장
            self.trainer.save_high_performance_models()
            
            # 결과 요약
            total_time = time.time() - start_time
            summary = self.trainer.get_performance_summary()
            summary.update({
                'total_pipeline_time': total_time,
                'data_size_train': len(X_train),
                'data_size_val': len(X_val) if X_val is not None else 0,
                'pipeline_version': 'High Performance v2.0'
            })
            
            logger.info(f"=== 고성능 파이프라인 완료 (시간: {total_time:.2f}초) ===")
            logger.info(f"성공 모델: {len(trained_models)}개")
            logger.info(f"목표 달성: Combined Score {summary['target_achieved']}, CTR 편향 {summary['ctr_bias_target_achieved']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"고성능 파이프라인 실패: {e}")
            raise

# 호환성을 위한 alias
TrainingPipeline = HighPerformanceTrainingPipeline