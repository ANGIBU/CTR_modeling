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

TORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(500, 500).cuda()
            test_result = test_tensor.sum().item()
            del test_tensor
            torch.cuda.empty_cache()
            TORCH_AVAILABLE = True
        except Exception as e:
            logging.warning(f"GPU 테스트 실패: {e}. CPU 모드만 사용")
            TORCH_AVAILABLE = True
    else:
        TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch가 설치되지 않았습니다. GPU 기능이 비활성화됩니다.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil이 설치되지 않았습니다. 메모리 모니터링 기능이 제한됩니다.")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna가 설치되지 않았습니다. 하이퍼파라미터 튜닝 기능이 비활성화됩니다.")

from config import Config
from models import ModelFactory, BaseModel, CTRCalibrator
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class MemoryTracker:
    """메모리 추적 클래스"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """현재 메모리 사용량 (GB)"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024**3)
            except:
                return 0.0
        return 0.0
    
    @staticmethod
    def get_available_memory() -> float:
        """사용 가능한 메모리 (GB)"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.virtual_memory().available / (1024**3)
            except:
                return 20.0
        return 20.0
    
    @staticmethod
    def force_cleanup():
        """강제 메모리 정리"""
        try:
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")

class CTRModelTrainer:
    """CTR 특화 모델 학습 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.device = getattr(config, 'DEVICE', 'cpu')
        self.trained_models = {}
        self.cv_results = {}
        self.best_params = {}
        self.calibrators = {}
        self.memory_tracker = MemoryTracker()
        
        self.gpu_available = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1000, 1000).cuda()
                test_result = test_tensor.sum().item()
                del test_tensor
                torch.cuda.empty_cache()
                
                torch.cuda.set_per_process_memory_fraction(0.5)
                
                self.gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU 학습 환경: {gpu_name}")
                logger.info(f"GPU 메모리: {gpu_memory:.1f}GB (50% 사용)")
            except Exception as e:
                logger.warning(f"GPU 초기화 실패: {e}. CPU 모드 사용")
                self.gpu_available = False
        else:
            logger.info("CPU 학습 환경")
    
    def train_single_model(self, 
                          model_type: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          params: Optional[Dict[str, Any]] = None,
                          apply_calibration: bool = True) -> BaseModel:
        """CTR 특화 단일 모델 학습"""
        
        logger.info(f"{model_type} CTR 모델 학습 시작")
        start_time = time.time()
        memory_before = self.memory_tracker.get_memory_usage()
        
        try:
            available_memory = self.memory_tracker.get_available_memory()
            if available_memory < 5:
                logger.warning(f"메모리 부족: {available_memory:.2f}GB. 데이터 크기 조정")
                if len(X_train) > 200000:
                    sample_indices = np.random.choice(len(X_train), 200000, replace=False)
                    X_train = X_train.iloc[sample_indices].copy()
                    y_train = y_train.iloc[sample_indices].copy()
                    if X_val is not None and len(X_val) > 50000:
                        val_indices = np.random.choice(len(X_val), 50000, replace=False)
                        X_val = X_val.iloc[val_indices].copy()
                        y_val = y_val.iloc[val_indices].copy()
            
            if params:
                params = self._validate_and_fix_ctr_params(model_type, params)
            else:
                params = self._get_ctr_optimized_params(model_type)
            
            model_kwargs = {'params': params}
            if model_type.lower() == 'deepctr':
                model_kwargs['input_dim'] = X_train.shape[1]
            
            model = ModelFactory.create_model(model_type, **model_kwargs)
            
            model.fit(X_train, y_train, X_val, y_val)
            
            if apply_calibration and X_val is not None and y_val is not None:
                current_memory = self.memory_tracker.get_available_memory()
                if current_memory > 3:
                    self._apply_ctr_calibration(model, X_val, y_val)
                else:
                    logger.warning("메모리 부족으로 Calibration 생략")
            
            training_time = time.time() - start_time
            memory_after = self.memory_tracker.get_memory_usage()
            
            logger.info(f"{model_type} CTR 모델 학습 완료 (소요시간: {training_time:.2f}초)")
            logger.info(f"메모리 사용량: {memory_before:.2f}GB → {memory_after:.2f}GB")
            
            self.trained_models[model_type] = {
                'model': model,
                'training_time': training_time,
                'params': params or {},
                'calibrated': apply_calibration,
                'memory_used': memory_after - memory_before
            }
            
            self._cleanup_memory_after_training(model_type)
            
            return model
            
        except Exception as e:
            logger.error(f"{model_type} CTR 모델 학습 실패: {str(e)}")
            self._cleanup_memory_after_training(model_type)
            raise
    
    def _validate_and_fix_ctr_params(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """CTR 특화 파라미터 검증 및 수정"""
        fixed_params = params.copy()
        
        try:
            if model_type.lower() == 'lightgbm':
                if 'is_unbalance' in fixed_params and 'scale_pos_weight' in fixed_params:
                    fixed_params.pop('is_unbalance', None)
                    logger.info("LightGBM: is_unbalance 제거하여 충돌 방지")
                
                fixed_params.setdefault('objective', 'binary')
                fixed_params.setdefault('metric', 'binary_logloss')
                fixed_params.setdefault('verbose', -1)
                
                # CTR 특화 파라미터 조정
                fixed_params['num_leaves'] = min(fixed_params.get('num_leaves', 255), 511)
                fixed_params['max_bin'] = min(fixed_params.get('max_bin', 255), 255)
                fixed_params['num_threads'] = min(fixed_params.get('num_threads', 6), 6)
                fixed_params['force_row_wise'] = True
                fixed_params['scale_pos_weight'] = fixed_params.get('scale_pos_weight', 49.0)
                
                # CTR 특화 정규화
                fixed_params['lambda_l1'] = max(fixed_params.get('lambda_l1', 2.0), 1.0)
                fixed_params['lambda_l2'] = max(fixed_params.get('lambda_l2', 2.0), 1.0)
                fixed_params['min_child_samples'] = max(fixed_params.get('min_child_samples', 200), 100)
                
            elif model_type.lower() == 'xgboost':
                fixed_params.setdefault('objective', 'binary:logistic')
                fixed_params.setdefault('eval_metric', 'logloss')
                
                if self.gpu_available:
                    fixed_params['tree_method'] = 'gpu_hist'
                    fixed_params['gpu_id'] = 0
                else:
                    fixed_params['tree_method'] = 'hist'
                    fixed_params.pop('gpu_id', None)
                
                # CTR 특화 파라미터 조정
                fixed_params['max_depth'] = min(fixed_params.get('max_depth', 8), 12)
                fixed_params['max_bin'] = min(fixed_params.get('max_bin', 255), 255)
                fixed_params['nthread'] = min(fixed_params.get('nthread', 6), 6)
                fixed_params['scale_pos_weight'] = fixed_params.get('scale_pos_weight', 49.0)
                
                # CTR 특화 정규화
                fixed_params['reg_alpha'] = max(fixed_params.get('reg_alpha', 2.0), 1.0)
                fixed_params['reg_lambda'] = max(fixed_params.get('reg_lambda', 2.0), 1.0)
                fixed_params['min_child_weight'] = max(fixed_params.get('min_child_weight', 15), 10)
                
                # CTR 특화 구조 파라미터
                fixed_params['grow_policy'] = 'lossguide'
                fixed_params['max_leaves'] = min(fixed_params.get('max_leaves', 255), 511)
                
            elif model_type.lower() == 'catboost':
                fixed_params.setdefault('loss_function', 'Logloss')
                fixed_params.setdefault('verbose', False)
                
                if self.gpu_available:
                    fixed_params['task_type'] = 'GPU'
                    fixed_params['devices'] = '0'
                else:
                    fixed_params['task_type'] = 'CPU'
                    fixed_params.pop('devices', None)
                
                # CTR 특화 파라미터 조정
                fixed_params['depth'] = min(fixed_params.get('depth', 8), 10)
                fixed_params['thread_count'] = min(fixed_params.get('thread_count', 6), 6)
                fixed_params['auto_class_weights'] = 'Balanced'
                
                # CTR 특화 정규화
                fixed_params['l2_leaf_reg'] = max(fixed_params.get('l2_leaf_reg', 10), 5)
                fixed_params['min_data_in_leaf'] = max(fixed_params.get('min_data_in_leaf', 100), 50)
                
                # CTR 특화 구조 파라미터
                fixed_params['grow_policy'] = 'Lossguide'
                fixed_params['max_leaves'] = min(fixed_params.get('max_leaves', 255), 511)
                
            elif model_type.lower() == 'deepctr':
                # CTR 특화 신경망 파라미터
                fixed_params['hidden_dims'] = fixed_params.get('hidden_dims', [512, 256, 128, 64])
                fixed_params['batch_size'] = min(fixed_params.get('batch_size', 1024), 2048)
                fixed_params['epochs'] = min(fixed_params.get('epochs', 50), 80)
                fixed_params['dropout_rate'] = min(max(fixed_params.get('dropout_rate', 0.3), 0.1), 0.5)
                fixed_params['learning_rate'] = min(max(fixed_params.get('learning_rate', 0.001), 0.0001), 0.01)
                
                # CTR 특화 정규화
                fixed_params['weight_decay'] = max(fixed_params.get('weight_decay', 1e-5), 1e-6)
                fixed_params['use_batch_norm'] = fixed_params.get('use_batch_norm', True)
                
        except Exception as e:
            logger.warning(f"CTR 파라미터 검증 실패: {e}")
        
        return fixed_params
    
    def _apply_ctr_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """CTR 특화 Calibration 적용"""
        try:
            logger.info(f"{model.name} CTR Calibration 적용 시작")
            
            if self.memory_tracker.get_available_memory() < 2:
                logger.warning("메모리 부족으로 Calibration 생략")
                return
            
            if len(X_val) > 30000:
                sample_indices = np.random.choice(len(X_val), 30000, replace=False)
                X_val_sample = X_val.iloc[sample_indices]
                y_val_sample = y_val.iloc[sample_indices]
            else:
                X_val_sample = X_val
                y_val_sample = y_val
            
            raw_predictions = model.predict_proba_raw(X_val_sample)
            
            calibrator = CTRCalibrator(target_ctr=self.config.CALIBRATION_CONFIG['target_ctr'])
            
            if self.config.CALIBRATION_CONFIG.get('platt_scaling', True):
                calibrator.fit_platt_scaling(y_val_sample.values, raw_predictions)
            
            if self.config.CALIBRATION_CONFIG.get('isotonic_regression', True):
                calibrator.fit_isotonic_regression(y_val_sample.values, raw_predictions)
            
            if self.config.CALIBRATION_CONFIG.get('temperature_scaling', True):
                calibrator.fit_temperature_scaling(y_val_sample.values, raw_predictions)
            
            calibrator.fit_bias_correction(y_val_sample.values, raw_predictions)
            
            model.apply_calibration(X_val_sample, y_val_sample, method='platt', cv_folds=3)
            
            self.calibrators[model.name] = calibrator
            
            calibrated_predictions = model.predict_proba(X_val_sample)
            
            original_ctr = raw_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val_sample.mean()
            
            logger.info(f"CTR Calibration 결과 - 원본: {original_ctr:.4f}, 보정: {calibrated_ctr:.4f}, 실제: {actual_ctr:.4f}")
            
            del raw_predictions, calibrated_predictions
            MemoryTracker.force_cleanup()
            
        except Exception as e:
            logger.error(f"CTR Calibration 적용 실패 ({model.name}): {str(e)}")
    
    def cross_validate_ctr_model(self,
                                model_type: str,
                                X: pd.DataFrame,
                                y: pd.Series,
                                cv_folds: int = None,
                                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """CTR 특화 시간적 순서 고려 교차검증"""
        
        if cv_folds is None:
            cv_folds = min(3, self.config.N_SPLITS)
        
        logger.info(f"{model_type} CTR 모델 {cv_folds}폴드 교차검증 시작")
        
        available_memory = self.memory_tracker.get_available_memory()
        original_size = len(X)
        
        if available_memory < 8:
            if len(X) > 300000:
                sample_size = 300000
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X = X.iloc[sample_indices].copy()
                y = y.iloc[sample_indices].copy()
                logger.warning(f"메모리 절약을 위해 데이터 축소: {original_size:,} → {sample_size:,}")
        
        try:
            if params:
                params = self._validate_and_fix_ctr_params(model_type, params)
            else:
                params = self._get_ctr_optimized_params(model_type)
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            cv_scores = {
                'ap_scores': [],
                'wll_scores': [],
                'combined_scores': [],
                'ctr_bias_scores': [],
                'training_times': [],
                'memory_usage': []
            }
            
            metrics_calculator = CTRMetrics()
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"CTR 폴드 {fold + 1}/{cv_folds} 시작")
                
                try:
                    memory_before = self.memory_tracker.get_memory_usage()
                    
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    start_time = time.time()
                    model = self.train_single_model(
                        model_type, X_train_fold, y_train_fold,
                        X_val_fold, y_val_fold, params,
                        apply_calibration=False
                    )
                    training_time = time.time() - start_time
                    
                    y_pred_proba = model.predict_proba(X_val_fold)
                    
                    ap_score = metrics_calculator.average_precision(y_val_fold, y_pred_proba)
                    wll_score = metrics_calculator.weighted_log_loss(y_val_fold, y_pred_proba)
                    combined_score = metrics_calculator.combined_score(y_val_fold, y_pred_proba)
                    
                    # CTR 편향 점수 추가
                    actual_ctr = y_val_fold.mean()
                    predicted_ctr = y_pred_proba.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    ctr_bias_score = 1.0 / (1.0 + ctr_bias * 100)
                    
                    memory_after = self.memory_tracker.get_memory_usage()
                    
                    cv_scores['ap_scores'].append(ap_score)
                    cv_scores['wll_scores'].append(wll_score)
                    cv_scores['combined_scores'].append(combined_score)
                    cv_scores['ctr_bias_scores'].append(ctr_bias_score)
                    cv_scores['training_times'].append(training_time)
                    cv_scores['memory_usage'].append(memory_after - memory_before)
                    
                    logger.info(f"CTR 폴드 {fold + 1} 완료 - AP: {ap_score:.4f}, WLL: {wll_score:.4f}, Combined: {combined_score:.4f}, CTR편향: {ctr_bias:.4f}")
                    
                    del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model, y_pred_proba
                    MemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"CTR 폴드 {fold + 1} 실행 실패: {str(e)}")
                    cv_scores['ap_scores'].append(0.0)
                    cv_scores['wll_scores'].append(float('inf'))
                    cv_scores['combined_scores'].append(0.0)
                    cv_scores['ctr_bias_scores'].append(0.0)
                    cv_scores['training_times'].append(0.0)
                    cv_scores['memory_usage'].append(0.0)
                    MemoryTracker.force_cleanup()
            
            valid_scores = [s for s in cv_scores['combined_scores'] if s > 0]
            valid_ctr_scores = [s for s in cv_scores['ctr_bias_scores'] if s > 0]
            
            if not valid_scores:
                logger.warning(f"{model_type} CTR 모든 폴드가 실패했습니다")
                return {
                    'model_type': model_type,
                    'combined_mean': 0.0,
                    'combined_std': 0.0,
                    'ctr_bias_mean': 0.0,
                    'scores_detail': cv_scores,
                    'params': params or {}
                }
            
            cv_results = {
                'model_type': model_type,
                'combined_mean': np.mean(valid_scores),
                'combined_std': np.std(valid_scores) if len(valid_scores) > 1 else 0.0,
                'ctr_bias_mean': np.mean(valid_ctr_scores) if valid_ctr_scores else 0.0,
                'ctr_bias_std': np.std(valid_ctr_scores) if len(valid_ctr_scores) > 1 else 0.0,
                'scores_detail': cv_scores,
                'params': params or {},
                'successful_folds': len(valid_scores),
                'avg_training_time': np.mean([t for t in cv_scores['training_times'] if t > 0]),
                'avg_memory_usage': np.mean([m for m in cv_scores['memory_usage'] if m > 0])
            }
            
            self.cv_results[model_type] = cv_results
            
            logger.info(f"{model_type} CTR 교차검증 완료")
            logger.info(f"평균 Combined Score: {cv_results['combined_mean']:.4f} (±{cv_results['combined_std']:.4f})")
            logger.info(f"평균 CTR 편향 점수: {cv_results['ctr_bias_mean']:.4f}")
            logger.info(f"성공한 폴드: {cv_results['successful_folds']}/{cv_folds}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"{model_type} CTR 교차검증 실패: {str(e)}")
            MemoryTracker.force_cleanup()
            raise
    
    def hyperparameter_tuning_ctr_optuna(self,
                                        model_type: str,
                                        X: pd.DataFrame,
                                        y: pd.Series,
                                        n_trials: int = None,
                                        cv_folds: int = 3) -> Dict[str, Any]:
        """CTR 특화 하이퍼파라미터 튜닝"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna가 설치되지 않았습니다. CTR 기본 파라미터 사용")
            best_params = self._get_ctr_optimized_params(model_type)
            self.best_params[model_type] = best_params
            return {
                'model_type': model_type,
                'best_params': best_params,
                'best_score': 0.0,
                'n_trials': 0,
                'study': None
            }
        
        available_memory = self.memory_tracker.get_available_memory()
        if n_trials is None:
            if available_memory > 15:
                n_trials = min(30, self.config.TUNING_CONFIG['n_trials'])
            elif available_memory > 10:
                n_trials = min(20, self.config.TUNING_CONFIG['n_trials'])
            else:
                n_trials = min(15, self.config.TUNING_CONFIG['n_trials'])
        
        logger.info(f"{model_type} CTR 하이퍼파라미터 튜닝 시작 (trials: {n_trials})")
        
        original_size = len(X)
        if available_memory < 10 and len(X) > 200000:
            sample_size = 200000
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[sample_indices].copy()
            y = y.iloc[sample_indices].copy()
            logger.warning(f"메모리 절약을 위해 튜닝 데이터 축소: {original_size:,} → {sample_size:,}")
        
        def ctr_objective(trial):
            try:
                if self.memory_tracker.get_available_memory() < 3:
                    logger.warning("메모리 부족으로 trial 중단")
                    return 0.0
                
                if model_type.lower() == 'lightgbm':
                    params = {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 127, 511),
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.8),
                        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
                        'min_child_samples': trial.suggest_int('min_child_samples', 100, 500),
                        'min_child_weight': trial.suggest_float('min_child_weight', 5, 30),
                        'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 5.0),
                        'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 5.0),
                        'max_depth': trial.suggest_int('max_depth', 8, 15),
                        'path_smooth': trial.suggest_float('path_smooth', 0.5, 2.0),
                        'verbose': -1,
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 1000,
                        'early_stopping_rounds': 100,
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 40, 60),
                        'force_row_wise': True,
                        'max_bin': 255,
                        'num_threads': 6
                    }
                
                elif model_type.lower() == 'xgboost':
                    params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'tree_method': 'gpu_hist' if self.gpu_available else 'hist',
                        'gpu_id': 0 if self.gpu_available else None,
                        'max_depth': trial.suggest_int('max_depth', 6, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.9),
                        'min_child_weight': trial.suggest_float('min_child_weight', 10, 30),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 5.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
                        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                        'max_leaves': trial.suggest_int('max_leaves', 127, 511),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 40, 60),
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 1000,
                        'early_stopping_rounds': 100,
                        'max_bin': 255,
                        'nthread': 6,
                        'grow_policy': 'lossguide'
                    }
                
                elif model_type.lower() == 'catboost':
                    params = {
                        'loss_function': 'Logloss',
                        'eval_metric': 'Logloss',
                        'task_type': 'GPU' if self.gpu_available else 'CPU',
                        'devices': '0' if self.gpu_available else None,
                        'depth': trial.suggest_int('depth', 6, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 20),
                        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 2.0),
                        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 5, 15),
                        'max_leaves': trial.suggest_int('max_leaves', 127, 511),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
                        'iterations': 1000,
                        'random_seed': self.config.RANDOM_STATE,
                        'early_stopping_rounds': 100,
                        'verbose': False,
                        'auto_class_weights': 'Balanced',
                        'thread_count': 6,
                        'grow_policy': 'Lossguide'
                    }
                
                elif model_type.lower() == 'deepctr':
                    if not self.gpu_available:
                        return 0.0
                        
                    hidden_dims_options = [
                        [256, 128, 64],
                        [512, 256, 128],
                        [512, 256, 128, 64],
                        [256, 128],
                        [1024, 512, 256, 128]
                    ]
                    params = {
                        'hidden_dims': trial.suggest_categorical('hidden_dims', hidden_dims_options),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.003, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
                        'epochs': 30,
                        'patience': 10,
                        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
                        'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
                        'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True, False])
                    }
                
                else:
                    params = {}
                
                cv_result = self.cross_validate_ctr_model(model_type, X, y, cv_folds, params)
                
                # CTR 특화 점수 계산 (Combined Score + CTR 편향 고려)
                combined_score = cv_result['combined_mean']
                ctr_bias_score = cv_result.get('ctr_bias_mean', 0.0)
                
                # 가중 점수 (Combined Score 70% + CTR 편향 30%)
                final_score = 0.7 * combined_score + 0.3 * ctr_bias_score
                
                MemoryTracker.force_cleanup()
                
                return final_score if final_score > 0 else 0.0
            
            except Exception as e:
                logger.error(f"CTR Trial 실행 실패: {str(e)}")
                MemoryTracker.force_cleanup()
                return 0.0
        
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.RANDOM_STATE),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            )
            
            study.optimize(
                ctr_objective, 
                n_trials=n_trials,
                timeout=1200,
                n_jobs=1,
                show_progress_bar=False
            )
            
        except KeyboardInterrupt:
            logger.info("CTR 하이퍼파라미터 튜닝이 중단되었습니다.")
        except Exception as e:
            logger.error(f"CTR 하이퍼파라미터 튜닝 중 오류 발생: {str(e)}")
        finally:
            MemoryTracker.force_cleanup()
        
        if not hasattr(study, 'best_value') or study.best_value is None or study.best_value <= 0:
            logger.warning(f"{model_type} CTR 하이퍼파라미터 튜닝에서 유효한 결과를 얻지 못했습니다.")
            best_params = self._get_ctr_optimized_params(model_type)
        else:
            best_params = study.best_params
            
        tuning_results = {
            'model_type': model_type,
            'best_params': best_params,
            'best_score': getattr(study, 'best_value', 0.0) if hasattr(study, 'best_value') else 0.0,
            'n_trials': len(getattr(study, 'trials', [])),
            'study': study if hasattr(study, 'best_value') else None
        }
        
        self.best_params[model_type] = best_params
        
        logger.info(f"{model_type} CTR 하이퍼파라미터 튜닝 완료")
        logger.info(f"최적 점수: {tuning_results['best_score']:.4f}")
        logger.info(f"수행된 trials: {tuning_results['n_trials']}/{n_trials}")
        
        return tuning_results
    
    def train_all_ctr_models(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None,
                            model_types: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """모든 CTR 모델 학습"""
        
        if model_types is None:
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'catboost'] if m in available_models]
            
            available_memory = self.memory_tracker.get_available_memory()
            if self.gpu_available and 'deepctr' in available_models and available_memory > 10:
                model_types.append('deepctr')
        
        logger.info(f"모든 CTR 모델 학습 시작: {model_types}")
        logger.info(f"사용 가능 메모리: {self.memory_tracker.get_available_memory():.2f}GB")
        
        trained_models = {}
        
        for model_type in model_types:
            try:
                available_memory = self.memory_tracker.get_available_memory()
                if available_memory < 3:
                    logger.warning(f"메모리 부족으로 {model_type} CTR 모델 학습 생략")
                    continue
                
                logger.info(f"{model_type} CTR 모델 학습 시작 (메모리: {available_memory:.2f}GB)")
                
                if model_type in self.best_params:
                    params = self.best_params[model_type]
                else:
                    params = self._get_ctr_optimized_params(model_type)
                
                model = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val, params
                )
                
                trained_models[model_type] = model
                
                logger.info(f"{model_type} CTR 모델 학습 완료")
                
                self._cleanup_memory_after_training(model_type)
                
                current_memory = self.memory_tracker.get_available_memory()
                logger.info(f"{model_type} 학습 후 사용 가능 메모리: {current_memory:.2f}GB")
                
            except Exception as e:
                logger.error(f"{model_type} CTR 모델 학습 실패: {str(e)}")
                self._cleanup_memory_after_training(model_type)
                continue
        
        logger.info(f"모든 CTR 모델 학습 완료. 성공한 모델: {list(trained_models.keys())}")
        
        return trained_models
    
    def _get_ctr_optimized_params(self, model_type: str) -> Dict[str, Any]:
        """CTR 예측에 특화된 기본 파라미터"""
        
        if model_type.lower() == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 255,
                'learning_rate': 0.03,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'min_child_samples': 200,
                'min_child_weight': 10,
                'lambda_l1': 2.0,
                'lambda_l2': 2.0,
                'max_depth': 12,
                'verbose': -1,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 3000,
                'early_stopping_rounds': 200,
                'scale_pos_weight': 49.0,
                'force_row_wise': True,
                'max_bin': 255,
                'num_threads': 6,
                'device_type': 'cpu',
                'min_data_in_leaf': 100,
                'feature_fraction_bynode': 0.8,
                'extra_trees': True,
                'path_smooth': 1.0,
                'grow_policy': 'lossguide'
            }
            return params
        
        elif model_type.lower() == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'gpu_hist' if self.gpu_available else 'hist',
                'gpu_id': 0 if self.gpu_available else None,
                'max_depth': 8,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'colsample_bynode': 0.8,
                'min_child_weight': 15,
                'reg_alpha': 2.0,
                'reg_lambda': 2.0,
                'scale_pos_weight': 49.0,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 3000,
                'early_stopping_rounds': 200,
                'max_bin': 255,
                'nthread': 6,
                'grow_policy': 'lossguide',
                'max_leaves': 255,
                'gamma': 0.1
            }
            if not self.gpu_available:
                params.pop('gpu_id', None)
            return params
        
        elif model_type.lower() == 'catboost':
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'task_type': 'GPU' if self.gpu_available else 'CPU',
                'devices': '0' if self.gpu_available else None,
                'depth': 8,
                'learning_rate': 0.03,
                'l2_leaf_reg': 10,
                'iterations': 3000,
                'random_seed': self.config.RANDOM_STATE,
                'od_wait': 200,
                'verbose': False,
                'auto_class_weights': 'Balanced',
                'max_ctr_complexity': 2,
                'thread_count': 6,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.0,
                'od_type': 'IncToDec',
                'leaf_estimation_iterations': 10,
                'leaf_estimation_method': 'Newton',
                'grow_policy': 'Lossguide',
                'max_leaves': 255,
                'min_data_in_leaf': 100
            }
            if not self.gpu_available:
                params.pop('devices', None)
            return params
        
        elif model_type.lower() == 'deepctr':
            params = {
                'hidden_dims': [512, 256, 128, 64],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'batch_size': 1024 if self.gpu_available else 512,
                'epochs': 50,
                'patience': 15,
                'use_batch_norm': True,
                'activation': 'relu',
                'use_residual': True,
                'use_attention': False,
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0
            }
            return params
        
        else:
            return {}
    
    def _cleanup_memory_after_training(self, model_type: str):
        """모델별 메모리 정리"""
        try:
            gc.collect()
            
            if model_type.lower() in ['deepctr', 'xgboost', 'catboost'] and self.gpu_available:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except:
                pass
                
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")
    
    def save_models(self, output_dir: Path = None):
        """모델 및 Calibrator 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CTR 모델 저장 시작: {output_dir}")
        
        for model_name, model_info in self.trained_models.items():
            try:
                model_path = output_dir / f"{model_name}_model.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['model'], f)
                
                metadata = {
                    'model_type': model_name,
                    'training_time': model_info.get('training_time', 0.0),
                    'params': model_info.get('params', {}),
                    'calibrated': model_info.get('calibrated', False),
                    'memory_used': model_info.get('memory_used', 0.0),
                    'device': str(self.device),
                    'ctr_optimized': True
                }
                
                metadata_path = output_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info(f"{model_name} CTR 모델 저장 완료: {model_path}")
                
            except Exception as e:
                logger.error(f"{model_name} CTR 모델 저장 실패: {str(e)}")
        
        if self.calibrators:
            calibrator_path = output_dir / "ctr_calibrators.pkl"
            try:
                with open(calibrator_path, 'wb') as f:
                    pickle.dump(self.calibrators, f)
                logger.info(f"CTR Calibrator 저장 완료: {calibrator_path}")
            except Exception as e:
                logger.error(f"CTR Calibrator 저장 실패: {str(e)}")
    
    def load_models(self, input_dir: Path = None) -> Dict[str, BaseModel]:
        """저장된 CTR 모델 로딩"""
        if input_dir is None:
            input_dir = self.config.MODEL_DIR
        
        input_dir = Path(input_dir)
        logger.info(f"CTR 모델 로딩 시작: {input_dir}")
        
        loaded_models = {}
        
        model_files = list(input_dir.glob("*_model.pkl"))
        
        for model_file in model_files:
            try:
                model_name = model_file.stem.replace('_model', '')
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                loaded_models[model_name] = model
                logger.info(f"{model_name} CTR 모델 로딩 완료")
                
            except Exception as e:
                logger.error(f"{model_file} CTR 모델 로딩 실패: {str(e)}")
        
        calibrator_path = input_dir / "ctr_calibrators.pkl"
        if calibrator_path.exists():
            try:
                with open(calibrator_path, 'rb') as f:
                    self.calibrators = pickle.load(f)
                logger.info("CTR Calibrator 로딩 완료")
            except Exception as e:
                logger.error(f"CTR Calibrator 로딩 실패: {str(e)}")
        
        self.trained_models = {name: {'model': model} for name, model in loaded_models.items()}
        
        return loaded_models
    
    def get_training_summary(self) -> Dict[str, Any]:
        """CTR 학습 결과 요약"""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'cv_results': self.cv_results,
            'best_params': self.best_params,
            'model_count': len(self.trained_models),
            'device_used': str(self.device),
            'gpu_available': self.gpu_available,
            'calibration_applied': len(self.calibrators) > 0,
            'ctr_optimized': True,
            'total_memory_used': sum(
                info.get('memory_used', 0.0) for info in self.trained_models.values()
            ),
            'avg_training_time': np.mean([
                info.get('training_time', 0.0) for info in self.trained_models.values()
            ]) if self.trained_models else 0.0
        }
        
        if self.cv_results:
            valid_results = {k: v for k, v in self.cv_results.items() if v['combined_mean'] > 0}
            if valid_results:
                best_model = max(
                    valid_results.items(),
                    key=lambda x: x[1]['combined_mean']
                )
                summary['best_model'] = {
                    'name': best_model[0],
                    'score': best_model[1]['combined_mean'],
                    'std': best_model[1]['combined_std'],
                    'ctr_bias_score': best_model[1].get('ctr_bias_mean', 0.0)
                }
        
        return summary

# 기존 클래스명 유지
ModelTrainer = CTRModelTrainer

class TrainingPipeline:
    """CTR 특화 전체 학습 파이프라인"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = CTRModelTrainer(config)
        self.memory_tracker = MemoryTracker()
        
    def run_full_ctr_pipeline(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame] = None,
                             y_val: Optional[pd.Series] = None,
                             tune_hyperparameters: bool = True,
                             n_trials: int = None) -> Dict[str, Any]:
        """CTR 특화 전체 학습 파이프라인 실행"""
        
        logger.info("CTR 특화 전체 학습 파이프라인 시작")
        logger.info(f"초기 메모리 상태: {self.memory_tracker.get_available_memory():.2f}GB")
        
        pipeline_start_time = time.time()
        
        try:
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'catboost'] if m in available_models]
            
            available_memory = self.memory_tracker.get_available_memory()
            if self.trainer.gpu_available and 'deepctr' in available_models and available_memory > 12:
                model_types.append('deepctr')
                logger.info("GPU 환경: DeepCTR CTR 모델 추가")
            
            if n_trials is None:
                if available_memory > 15:
                    n_trials = 20
                elif available_memory > 10:
                    n_trials = 15
                else:
                    n_trials = 10
            
            # 1. CTR 특화 하이퍼파라미터 튜닝
            if tune_hyperparameters and OPTUNA_AVAILABLE:
                logger.info("CTR 하이퍼파라미터 튜닝 단계")
                for model_type in model_types:
                    try:
                        if self.memory_tracker.get_available_memory() < 5:
                            logger.warning(f"메모리 부족으로 {model_type} CTR 튜닝 생략")
                            continue
                        
                        current_trials = max(5, n_trials // 3) if model_type == 'deepctr' else n_trials
                        
                        self.trainer.hyperparameter_tuning_ctr_optuna(
                            model_type, X_train, y_train, n_trials=current_trials, cv_folds=3
                        )
                        
                        MemoryTracker.force_cleanup()
                        
                    except Exception as e:
                        logger.error(f"{model_type} CTR 하이퍼파라미터 튜닝 실패: {str(e)}")
                        MemoryTracker.force_cleanup()
            else:
                logger.info("CTR 하이퍼파라미터 튜닝 생략")
            
            # 2. CTR 교차검증 평가
            logger.info("CTR 교차검증 평가 단계")
            for model_type in model_types:
                try:
                    if self.memory_tracker.get_available_memory() < 4:
                        logger.warning(f"메모리 부족으로 {model_type} CTR 교차검증 생략")
                        continue
                    
                    params = self.trainer.best_params.get(model_type, None)
                    if params is None:
                        params = self.trainer._get_ctr_optimized_params(model_type)
                    
                    self.trainer.cross_validate_ctr_model(model_type, X_train, y_train, params=params)
                    
                    MemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"{model_type} CTR 교차검증 실패: {str(e)}")
                    MemoryTracker.force_cleanup()
            
            # 3. CTR 최종 모델 학습
            logger.info("CTR 최종 모델 학습 단계")
            logger.info(f"학습 전 메모리 상태: {self.memory_tracker.get_available_memory():.2f}GB")
            
            trained_models = self.trainer.train_all_ctr_models(X_train, y_train, X_val, y_val, model_types)
            
            logger.info(f"학습 후 메모리 상태: {self.memory_tracker.get_available_memory():.2f}GB")
            
            # 4. CTR 모델 저장
            try:
                self.trainer.save_models()
            except Exception as e:
                logger.warning(f"CTR 모델 저장 실패: {str(e)}")
            
            # 5. CTR 결과 요약
            pipeline_time = time.time() - pipeline_start_time
            summary = self.trainer.get_training_summary()
            summary['total_pipeline_time'] = pipeline_time
            summary['memory_peak'] = self.memory_tracker.get_memory_usage()
            summary['memory_available_end'] = self.memory_tracker.get_available_memory()
            summary['ctr_pipeline'] = True
            
            logger.info(f"CTR 전체 학습 파이프라인 완료 (소요시간: {pipeline_time:.2f}초)")
            logger.info(f"최종 메모리 상태: {summary['memory_available_end']:.2f}GB")
            
            return summary
            
        except Exception as e:
            logger.error(f"CTR 파이프라인 실행 실패: {str(e)}")
            MemoryTracker.force_cleanup()
            raise
        finally:
            MemoryTracker.force_cleanup()

class GPUMemoryManager:
    """GPU 메모리 관리 클래스"""
    
    @staticmethod
    def clear_gpu_memory():
        """GPU 메모리 정리"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"GPU 메모리 정리 실패: {e}")
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """GPU 메모리 정보 반환"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'total_gb': 0, 'allocated_gb': 0, 'cached_gb': 0, 'free_gb': 0}
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - cached
            
            return {
                'total_gb': total,
                'allocated_gb': allocated,
                'cached_gb': cached,
                'free_gb': free
            }
        except Exception as e:
            logger.warning(f"GPU 메모리 정보 조회 실패: {e}")
            return {'total_gb': 0, 'allocated_gb': 0, 'cached_gb': 0, 'free_gb': 0}
    
    @staticmethod
    def monitor_gpu_usage(func):
        """GPU 메모리 사용량 모니터링 데코레이터"""
        def wrapper(*args, **kwargs):
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    before = GPUMemoryManager.get_gpu_memory_info()
                    logger.info(f"GPU 메모리 (실행 전): {before['allocated_gb']:.2f}GB 사용")
                    
                    result = func(*args, **kwargs)
                    
                    after = GPUMemoryManager.get_gpu_memory_info()
                    logger.info(f"GPU 메모리 (실행 후): {after['allocated_gb']:.2f}GB 사용")
                    
                    return result
                except Exception as e:
                    logger.warning(f"GPU 메모리 모니터링 실패: {e}")
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper