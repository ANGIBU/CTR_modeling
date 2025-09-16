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
import os
import multiprocessing as mp

TORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1000, 1000).cuda()
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

class LargeDataMemoryTracker:
    """1070만행 대용량 데이터 메모리 추적 클래스"""
    
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
        """사용 가능한 메모리 (GB) - 40GB 사용 가능 기준"""
        if PSUTIL_AVAILABLE:
            try:
                available = psutil.virtual_memory().available / (1024**3)
                return min(available, 40.0)
            except:
                return 35.0
        return 35.0
    
    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, float]:
        """GPU 메모리 사용량 - RTX 4060 Ti 16GB 기준"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
                free_memory = total_memory - cached_memory
                
                return {
                    'total': total_memory,
                    'allocated': allocated_memory,
                    'cached': cached_memory,
                    'free': free_memory,
                    'utilization': (cached_memory / total_memory) * 100
                }
            except Exception as e:
                logger.warning(f"GPU 메모리 정보 조회 실패: {e}")
        
        return {'total': 16.0, 'allocated': 0.0, 'cached': 0.0, 'free': 16.0, 'utilization': 0.0}
    
    @staticmethod
    def force_cleanup():
        """강제 메모리 정리 - 대용량 데이터 처리용"""
        try:
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")
    
    @staticmethod
    def optimize_for_large_data():
        """1070만행 대용량 데이터 처리를 위한 메모리 최적화"""
        try:
            os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
            os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.85)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                logger.info("GPU 메모리 최적화 완료: RTX 4060 Ti 16GB 85% 활용")
            
            logger.info("대용량 데이터 메모리 최적화 완료")
            
        except Exception as e:
            logger.warning(f"메모리 최적화 실패: {e}")

class CTRHighPerformanceTrainer:
    """CTR 고성능 모델 학습 클래스 - Combined Score 0.30+ 달성 목표"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.device = getattr(config, 'DEVICE', 'cpu')
        self.trained_models = {}
        self.cv_results = {}
        self.best_params = {}
        self.calibrators = {}
        self.memory_tracker = LargeDataMemoryTracker()
        
        self.gpu_available = False
        self.rtx_4060ti_optimized = False
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_properties = torch.cuda.get_device_properties(0)
                gpu_name = gpu_properties.name
                gpu_memory_gb = gpu_properties.total_memory / (1024**3)
                
                test_tensor = torch.zeros(2000, 2000).cuda()
                test_result = test_tensor.sum().item()
                del test_tensor
                torch.cuda.empty_cache()
                
                self.gpu_available = True
                self.rtx_4060ti_optimized = 'RTX 4060 Ti' in gpu_name or gpu_memory_gb >= 15.0
                
                self.memory_tracker.optimize_for_large_data()
                
                logger.info(f"GPU 학습 환경: {gpu_name}")
                logger.info(f"GPU 메모리: {gpu_memory_gb:.1f}GB")
                logger.info(f"RTX 4060 Ti 최적화: {self.rtx_4060ti_optimized}")
                
            except Exception as e:
                logger.warning(f"GPU 초기화 실패: {e}. CPU 모드 사용")
                self.gpu_available = False
        else:
            logger.info("CPU 학습 환경 - Ryzen 5 5600X 6코어 12스레드")
    
    def train_single_model_optimized(self, 
                                   model_type: str,
                                   X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_val: Optional[pd.DataFrame] = None,
                                   y_val: Optional[pd.Series] = None,
                                   params: Optional[Dict[str, Any]] = None,
                                   apply_calibration: bool = True) -> BaseModel:
        """대용량 데이터 특화 단일 모델 학습 - Combined Score 0.30+ 목표"""
        
        logger.info(f"{model_type} 고성능 CTR 모델 학습 시작 (데이터 크기: {len(X_train):,})")
        start_time = time.time()
        memory_before = self.memory_tracker.get_memory_usage()
        gpu_info_before = self.memory_tracker.get_gpu_memory_usage()
        
        try:
            available_memory = self.memory_tracker.get_available_memory()
            gpu_memory = self.memory_tracker.get_gpu_memory_usage()
            
            data_size_gb = (X_train.memory_usage(deep=True).sum() + y_train.memory_usage(deep=True)) / (1024**3)
            logger.info(f"데이터 크기: {data_size_gb:.2f}GB, 사용가능 메모리: {available_memory:.2f}GB")
            
            if available_memory < data_size_gb * 3:
                logger.warning(f"메모리 부족 위험. 청킹 처리 적용")
                X_train, y_train, X_val, y_val = self._apply_memory_efficient_sampling(
                    X_train, y_train, X_val, y_val, available_memory
                )
            
            if params:
                params = self._validate_and_optimize_params(model_type, params)
            else:
                params = self._get_high_performance_params(model_type)
            
            model_kwargs = {'params': params}
            if model_type.lower() == 'deepctr':
                model_kwargs['input_dim'] = X_train.shape[1]
            
            model = ModelFactory.create_model(model_type, **model_kwargs)
            
            model.fit(X_train, y_train, X_val, y_val)
            
            if apply_calibration and X_val is not None and y_val is not None:
                current_memory = self.memory_tracker.get_available_memory()
                if current_memory > 5:
                    self._apply_high_performance_calibration(model, X_val, y_val)
                else:
                    logger.warning("메모리 부족으로 Calibration 생략")
            
            training_time = time.time() - start_time
            memory_after = self.memory_tracker.get_memory_usage()
            gpu_info_after = self.memory_tracker.get_gpu_memory_usage()
            
            logger.info(f"{model_type} 고성능 모델 학습 완료 (소요시간: {training_time:.2f}초)")
            logger.info(f"메모리 사용량: {memory_before:.2f}GB → {memory_after:.2f}GB")
            logger.info(f"GPU 메모리 사용률: {gpu_info_before['utilization']:.1f}% → {gpu_info_after['utilization']:.1f}%")
            
            self.trained_models[model_type] = {
                'model': model,
                'params': params or {},
                'training_time': training_time,
                'calibrated': apply_calibration,
                'memory_used': memory_after - memory_before,
                'gpu_memory_used': gpu_info_after['allocated'] - gpu_info_before['allocated'],
                'data_size': len(X_train),
                'cv_result': None
            }
            
            self._cleanup_memory_after_training(model_type)
            
            return model
            
        except Exception as e:
            logger.error(f"{model_type} 고성능 모델 학습 실패: {str(e)}")
            self._cleanup_memory_after_training(model_type)
            raise
    
    def _apply_memory_efficient_sampling(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                       X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                                       available_memory: float) -> Tuple:
        """메모리 효율적인 데이터 샘플링 - 1070만행 처리용"""
        
        data_size_gb = (X_train.memory_usage(deep=True).sum() + y_train.memory_usage(deep=True)) / (1024**3)
        
        if data_size_gb > available_memory * 0.3:
            ratio = (available_memory * 0.25) / data_size_gb
            max_samples = int(len(X_train) * ratio)
            max_samples = max(max_samples, 2000000)
            
            if max_samples < len(X_train):
                logger.info(f"메모리 최적화를 위해 데이터 크기 조정: {len(X_train):,} → {max_samples:,}")
                
                pos_indices = np.where(y_train == 1)[0]
                neg_indices = np.where(y_train == 0)[0]
                
                pos_ratio = len(pos_indices) / len(y_train)
                target_pos_samples = int(max_samples * pos_ratio * 1.2)
                target_neg_samples = max_samples - target_pos_samples
                
                selected_pos = np.random.choice(pos_indices, min(target_pos_samples, len(pos_indices)), replace=False)
                selected_neg = np.random.choice(neg_indices, min(target_neg_samples, len(neg_indices)), replace=False)
                
                selected_indices = np.concatenate([selected_pos, selected_neg])
                np.random.shuffle(selected_indices)
                
                X_train = X_train.iloc[selected_indices].copy()
                y_train = y_train.iloc[selected_indices].copy()
                
                if X_val is not None and y_val is not None and len(X_val) > 500000:
                    val_indices = np.random.choice(len(X_val), 500000, replace=False)
                    X_val = X_val.iloc[val_indices].copy()
                    y_val = y_val.iloc[val_indices].copy()
        
        return X_train, y_train, X_val, y_val
    
    def _validate_and_optimize_params(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Combined Score 0.30+ 달성을 위한 파라미터 최적화"""
        optimized_params = params.copy()
        
        try:
            if model_type.lower() == 'lightgbm':
                if 'is_unbalance' in optimized_params and 'scale_pos_weight' in optimized_params:
                    optimized_params.pop('is_unbalance', None)
                    logger.info("LightGBM: is_unbalance 제거하여 충돌 방지")
                
                optimized_params.setdefault('objective', 'binary')
                optimized_params.setdefault('metric', 'binary_logloss')
                optimized_params.setdefault('verbose', -1)
                optimized_params.setdefault('device_type', 'cpu')
                
                optimized_params['num_leaves'] = min(optimized_params.get('num_leaves', 511), 1023)
                optimized_params['max_bin'] = min(optimized_params.get('max_bin', 255), 255)
                optimized_params['num_threads'] = min(optimized_params.get('num_threads', 12), 12)
                optimized_params['force_row_wise'] = True
                
                optimized_params['scale_pos_weight'] = optimized_params.get('scale_pos_weight', 49.8)
                optimized_params['lambda_l1'] = max(optimized_params.get('lambda_l1', 3.0), 2.0)
                optimized_params['lambda_l2'] = max(optimized_params.get('lambda_l2', 3.0), 2.0)
                optimized_params['min_child_samples'] = max(optimized_params.get('min_child_samples', 300), 200)
                optimized_params['max_depth'] = min(optimized_params.get('max_depth', 15), 18)
                
                optimized_params['feature_fraction'] = optimized_params.get('feature_fraction', 0.85)
                optimized_params['bagging_fraction'] = optimized_params.get('bagging_fraction', 0.75)
                optimized_params['bagging_freq'] = optimized_params.get('bagging_freq', 7)
                optimized_params['path_smooth'] = optimized_params.get('path_smooth', 1.5)
                
            elif model_type.lower() == 'xgboost':
                optimized_params.setdefault('objective', 'binary:logistic')
                optimized_params.setdefault('eval_metric', 'logloss')
                
                if self.gpu_available:
                    optimized_params['tree_method'] = 'gpu_hist'
                    optimized_params['gpu_id'] = 0
                    optimized_params['predictor'] = 'gpu_predictor'
                else:
                    optimized_params['tree_method'] = 'hist'
                    optimized_params.pop('gpu_id', None)
                    optimized_params.pop('predictor', None)
                
                optimized_params['max_depth'] = min(optimized_params.get('max_depth', 10), 15)
                optimized_params['max_bin'] = min(optimized_params.get('max_bin', 255), 255)
                optimized_params['nthread'] = min(optimized_params.get('nthread', 12), 12)
                optimized_params['scale_pos_weight'] = optimized_params.get('scale_pos_weight', 49.8)
                
                optimized_params['reg_alpha'] = max(optimized_params.get('reg_alpha', 3.0), 2.0)
                optimized_params['reg_lambda'] = max(optimized_params.get('reg_lambda', 3.0), 2.0)
                optimized_params['min_child_weight'] = max(optimized_params.get('min_child_weight', 20), 15)
                
                optimized_params['grow_policy'] = 'lossguide'
                optimized_params['max_leaves'] = min(optimized_params.get('max_leaves', 511), 1023)
                optimized_params['subsample'] = optimized_params.get('subsample', 0.82)
                optimized_params['colsample_bytree'] = optimized_params.get('colsample_bytree', 0.85)
                
            elif model_type.lower() == 'catboost':
                optimized_params.setdefault('loss_function', 'Logloss')
                optimized_params.setdefault('verbose', False)
                
                if self.gpu_available:
                    optimized_params['task_type'] = 'GPU'
                    optimized_params['devices'] = '0'
                else:
                    optimized_params['task_type'] = 'CPU'
                    optimized_params.pop('devices', None)
                
                conflicting_params = ['early_stopping_rounds', 'use_best_model', 'eval_set']
                for param in conflicting_params:
                    if param in optimized_params:
                        if param == 'early_stopping_rounds':
                            early_stop_val = optimized_params.pop(param)
                            if 'od_wait' not in optimized_params:
                                optimized_params['od_wait'] = early_stop_val
                                optimized_params['od_type'] = 'IncToDec'
                            logger.info(f"CatBoost: {param}를 od_wait로 변경")
                        else:
                            optimized_params.pop(param)
                            logger.info(f"CatBoost: {param} 파라미터 제거")
                
                optimized_params['depth'] = min(optimized_params.get('depth', 10), 12)
                optimized_params['thread_count'] = min(optimized_params.get('thread_count', 12), 12)
                optimized_params['auto_class_weights'] = 'Balanced'
                
                optimized_params['l2_leaf_reg'] = max(optimized_params.get('l2_leaf_reg', 15), 10)
                optimized_params['min_data_in_leaf'] = max(optimized_params.get('min_data_in_leaf', 150), 100)
                optimized_params['grow_policy'] = 'Lossguide'
                optimized_params['max_leaves'] = min(optimized_params.get('max_leaves', 511), 1023)
                
            elif model_type.lower() == 'deepctr':
                if self.rtx_4060ti_optimized:
                    optimized_params['hidden_dims'] = optimized_params.get('hidden_dims', [1024, 512, 256, 128, 64])
                    optimized_params['batch_size'] = min(optimized_params.get('batch_size', 2048), 4096)
                    optimized_params['epochs'] = min(optimized_params.get('epochs', 80), 100)
                else:
                    optimized_params['hidden_dims'] = optimized_params.get('hidden_dims', [512, 256, 128, 64])
                    optimized_params['batch_size'] = min(optimized_params.get('batch_size', 1024), 2048)
                    optimized_params['epochs'] = min(optimized_params.get('epochs', 50), 80)
                
                optimized_params['dropout_rate'] = min(max(optimized_params.get('dropout_rate', 0.25), 0.1), 0.5)
                optimized_params['learning_rate'] = min(max(optimized_params.get('learning_rate', 0.0008), 0.0001), 0.01)
                optimized_params['weight_decay'] = max(optimized_params.get('weight_decay', 5e-5), 1e-6)
                optimized_params['use_batch_norm'] = optimized_params.get('use_batch_norm', True)
                optimized_params['patience'] = optimized_params.get('patience', 20)
                
        except Exception as e:
            logger.warning(f"파라미터 최적화 실패: {e}")
        
        return optimized_params
    
    def _apply_high_performance_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """고성능 CTR Calibration 적용"""
        try:
            logger.info(f"{model.name} 고성능 CTR Calibration 적용 시작")
            
            if self.memory_tracker.get_available_memory() < 3:
                logger.warning("메모리 부족으로 Calibration 생략")
                return
            
            val_size = min(len(X_val), 50000)
            if len(X_val) > val_size:
                sample_indices = np.random.choice(len(X_val), val_size, replace=False)
                X_val_sample = X_val.iloc[sample_indices]
                y_val_sample = y_val.iloc[sample_indices]
            else:
                X_val_sample = X_val
                y_val_sample = y_val
            
            raw_predictions = model.predict_proba_raw(X_val_sample)
            
            calibrator = CTRCalibrator(target_ctr=0.0201)
            
            if self.config.CALIBRATION_CONFIG.get('platt_scaling', True):
                calibrator.fit_platt_scaling(y_val_sample.values, raw_predictions)
            
            if self.config.CALIBRATION_CONFIG.get('isotonic_regression', True):
                calibrator.fit_isotonic_regression(y_val_sample.values, raw_predictions)
            
            if self.config.CALIBRATION_CONFIG.get('temperature_scaling', True):
                calibrator.fit_temperature_scaling(y_val_sample.values, raw_predictions)
            
            calibrator.fit_bias_correction(y_val_sample.values, raw_predictions)
            
            model.apply_calibration(X_val_sample, y_val_sample, method='platt', cv_folds=5)
            
            self.calibrators[model.name] = calibrator
            
            calibrated_predictions = model.predict_proba(X_val_sample)
            
            original_ctr = raw_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val_sample.mean()
            
            logger.info(f"고성능 CTR Calibration 결과 - 원본: {original_ctr:.4f}, 보정: {calibrated_ctr:.4f}, 실제: {actual_ctr:.4f}")
            
            del raw_predictions, calibrated_predictions
            LargeDataMemoryTracker.force_cleanup()
            
        except Exception as e:
            logger.error(f"고성능 CTR Calibration 적용 실패 ({model.name}): {str(e)}")
    
    def cross_validate_ctr_model_optimized(self,
                                         model_type: str,
                                         X: pd.DataFrame,
                                         y: pd.Series,
                                         cv_folds: int = None,
                                         params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """대용량 데이터 특화 교차검증 - Combined Score 0.30+ 목표"""
        
        if cv_folds is None:
            cv_folds = min(5, self.config.N_SPLITS)
        
        logger.info(f"{model_type} 고성능 CTR 모델 {cv_folds}폴드 교차검증 시작 (데이터: {len(X):,})")
        
        available_memory = self.memory_tracker.get_available_memory()
        original_size = len(X)
        
        if available_memory < 15:
            max_samples = min(len(X), int(available_memory * 100000))
            if max_samples < len(X):
                pos_indices = np.where(y == 1)[0]
                neg_indices = np.where(y == 0)[0]
                
                pos_ratio = len(pos_indices) / len(y)
                target_pos = int(max_samples * pos_ratio * 1.1)
                target_neg = max_samples - target_pos
                
                selected_pos = np.random.choice(pos_indices, min(target_pos, len(pos_indices)), replace=False)
                selected_neg = np.random.choice(neg_indices, min(target_neg, len(neg_indices)), replace=False)
                
                selected_indices = np.concatenate([selected_pos, selected_neg])
                np.random.shuffle(selected_indices)
                
                X = X.iloc[selected_indices].copy()
                y = y.iloc[selected_indices].copy()
                
                logger.info(f"메모리 최적화를 위해 CV 데이터 축소: {original_size:,} → {len(X):,}")
        
        try:
            if params:
                params = self._validate_and_optimize_params(model_type, params)
            else:
                params = self._get_high_performance_params(model_type)
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            cv_scores = {
                'ap_scores': [],
                'wll_scores': [],
                'combined_scores': [],
                'ctr_optimized_scores': [],
                'ctr_bias_scores': [],
                'training_times': [],
                'memory_usage': [],
                'gpu_memory_usage': []
            }
            
            metrics_calculator = CTRMetrics()
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"고성능 CTR 폴드 {fold + 1}/{cv_folds} 시작")
                
                try:
                    memory_before = self.memory_tracker.get_memory_usage()
                    gpu_before = self.memory_tracker.get_gpu_memory_usage()
                    
                    X_train_fold = X.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_train_fold = y.iloc[train_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    start_time = time.time()
                    model = self.train_single_model_optimized(
                        model_type, X_train_fold, y_train_fold,
                        X_val_fold, y_val_fold, params,
                        apply_calibration=False
                    )
                    training_time = time.time() - start_time
                    
                    y_pred_proba = model.predict_proba(X_val_fold)
                    
                    unique_predictions = len(np.unique(y_pred_proba))
                    if unique_predictions < 100:
                        logger.warning(f"폴드 {fold + 1}: 예측값 다양성 부족 (고유값: {unique_predictions})")
                    
                    ap_score = metrics_calculator.average_precision(y_val_fold, y_pred_proba)
                    wll_score = metrics_calculator.weighted_log_loss(y_val_fold, y_pred_proba)
                    combined_score = metrics_calculator.combined_score(y_val_fold, y_pred_proba)
                    ctr_optimized_score = metrics_calculator.ctr_optimized_score(y_val_fold, y_pred_proba)
                    
                    actual_ctr = y_val_fold.mean()
                    predicted_ctr = y_pred_proba.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    ctr_bias_score = np.exp(-ctr_bias * 200)
                    
                    memory_after = self.memory_tracker.get_memory_usage()
                    gpu_after = self.memory_tracker.get_gpu_memory_usage()
                    
                    cv_scores['ap_scores'].append(ap_score)
                    cv_scores['wll_scores'].append(wll_score)
                    cv_scores['combined_scores'].append(combined_score)
                    cv_scores['ctr_optimized_scores'].append(ctr_optimized_score)
                    cv_scores['ctr_bias_scores'].append(ctr_bias_score)
                    cv_scores['training_times'].append(training_time)
                    cv_scores['memory_usage'].append(memory_after - memory_before)
                    cv_scores['gpu_memory_usage'].append(gpu_after['allocated'] - gpu_before['allocated'])
                    
                    logger.info(f"고성능 CTR 폴드 {fold + 1} 완료")
                    logger.info(f"AP: {ap_score:.4f}, WLL: {wll_score:.4f}, Combined: {combined_score:.4f}")
                    logger.info(f"CTR최적화: {ctr_optimized_score:.4f}, CTR편향: {ctr_bias:.4f}")
                    
                    del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model, y_pred_proba
                    LargeDataMemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"고성능 CTR 폴드 {fold + 1} 실행 실패: {str(e)}")
                    cv_scores['ap_scores'].append(0.0)
                    cv_scores['wll_scores'].append(float('inf'))
                    cv_scores['combined_scores'].append(0.0)
                    cv_scores['ctr_optimized_scores'].append(0.0)
                    cv_scores['ctr_bias_scores'].append(0.0)
                    cv_scores['training_times'].append(0.0)
                    cv_scores['memory_usage'].append(0.0)
                    cv_scores['gpu_memory_usage'].append(0.0)
                    LargeDataMemoryTracker.force_cleanup()
            
            valid_combined_scores = [s for s in cv_scores['combined_scores'] if s > 0]
            valid_ctr_scores = [s for s in cv_scores['ctr_optimized_scores'] if s > 0]
            valid_ctr_bias_scores = [s for s in cv_scores['ctr_bias_scores'] if s > 0]
            
            if not valid_combined_scores:
                logger.warning(f"{model_type} 고성능 CTR 모든 폴드가 실패했습니다")
                cv_results = {
                    'model_type': model_type,
                    'combined_mean': 0.0,
                    'combined_std': 0.0,
                    'ctr_optimized_mean': 0.0,
                    'ctr_bias_mean': 0.0,
                    'scores_detail': cv_scores,
                    'params': params or {}
                }
            else:
                cv_results = {
                    'model_type': model_type,
                    'combined_mean': np.mean(valid_combined_scores),
                    'combined_std': np.std(valid_combined_scores) if len(valid_combined_scores) > 1 else 0.0,
                    'ctr_optimized_mean': np.mean(valid_ctr_scores) if valid_ctr_scores else 0.0,
                    'ctr_optimized_std': np.std(valid_ctr_scores) if len(valid_ctr_scores) > 1 else 0.0,
                    'ctr_bias_mean': np.mean(valid_ctr_bias_scores) if valid_ctr_bias_scores else 0.0,
                    'ctr_bias_std': np.std(valid_ctr_bias_scores) if len(valid_ctr_bias_scores) > 1 else 0.0,
                    'scores_detail': cv_scores,
                    'params': params or {},
                    'successful_folds': len(valid_combined_scores),
                    'avg_training_time': np.mean([t for t in cv_scores['training_times'] if t > 0]),
                    'avg_memory_usage': np.mean([m for m in cv_scores['memory_usage'] if m > 0]),
                    'avg_gpu_memory_usage': np.mean([m for m in cv_scores['gpu_memory_usage'] if m > 0])
                }
            
            self.cv_results[model_type] = cv_results
            
            if model_type in self.trained_models:
                self.trained_models[model_type]['cv_result'] = cv_results
            
            logger.info(f"{model_type} 고성능 CTR 교차검증 완료")
            logger.info(f"평균 Combined Score: {cv_results['combined_mean']:.4f} (±{cv_results['combined_std']:.4f})")
            logger.info(f"평균 CTR 최적화 점수: {cv_results['ctr_optimized_mean']:.4f}")
            logger.info(f"평균 CTR 편향 점수: {cv_results['ctr_bias_mean']:.4f}")
            logger.info(f"성공한 폴드: {cv_results['successful_folds']}/{cv_folds}")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"{model_type} 고성능 CTR 교차검증 실패: {str(e)}")
            LargeDataMemoryTracker.force_cleanup()
            raise
    
    def hyperparameter_tuning_ctr_optuna_optimized(self,
                                                  model_type: str,
                                                  X: pd.DataFrame,
                                                  y: pd.Series,
                                                  n_trials: int = None,
                                                  cv_folds: int = 3) -> Dict[str, Any]:
        """Combined Score 0.30+ 달성을 위한 고성능 하이퍼파라미터 튜닝"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna가 설치되지 않았습니다. 고성능 기본 파라미터 사용")
            best_params = self._get_high_performance_params(model_type)
            self.best_params[model_type] = best_params
            return {
                'model_type': model_type,
                'best_params': best_params,
                'best_score': 0.0,
                'n_trials': 0,
                'study': None
            }
        
        available_memory = self.memory_tracker.get_available_memory()
        gpu_memory = self.memory_tracker.get_gpu_memory_usage()
        
        if n_trials is None:
            if available_memory > 25 and gpu_memory['free'] > 8:
                n_trials = min(50, self.config.TUNING_CONFIG['n_trials'])
            elif available_memory > 20:
                n_trials = min(35, self.config.TUNING_CONFIG['n_trials'])
            else:
                n_trials = min(25, self.config.TUNING_CONFIG['n_trials'])
        
        logger.info(f"{model_type} 고성능 하이퍼파라미터 튜닝 시작")
        logger.info(f"Trials: {n_trials}, 메모리: {available_memory:.1f}GB, GPU메모리: {gpu_memory['free']:.1f}GB")
        
        original_size = len(X)
        if available_memory < 20 and len(X) > 3000000:
            sample_size = min(3000000, int(available_memory * 100000))
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            pos_ratio = len(pos_indices) / len(y)
            target_pos = int(sample_size * pos_ratio * 1.1)
            target_neg = sample_size - target_pos
            
            selected_pos = np.random.choice(pos_indices, min(target_pos, len(pos_indices)), replace=False)
            selected_neg = np.random.choice(neg_indices, min(target_neg, len(neg_indices)), replace=False)
            
            selected_indices = np.concatenate([selected_pos, selected_neg])
            np.random.shuffle(selected_indices)
            
            X = X.iloc[selected_indices].copy()
            y = y.iloc[selected_indices].copy()
            
            logger.info(f"메모리 최적화를 위해 튜닝 데이터 축소: {original_size:,} → {len(X):,}")
        
        def high_performance_objective(trial):
            try:
                if self.memory_tracker.get_available_memory() < 5:
                    logger.warning("메모리 부족으로 trial 중단")
                    return 0.0
                
                if model_type.lower() == 'lightgbm':
                    params = {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 255, 1023),
                        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 0.95),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.85),
                        'bagging_freq': trial.suggest_int('bagging_freq', 5, 10),
                        'min_child_samples': trial.suggest_int('min_child_samples', 200, 600),
                        'min_child_weight': trial.suggest_float('min_child_weight', 8, 40),
                        'lambda_l1': trial.suggest_float('lambda_l1', 2.0, 8.0),
                        'lambda_l2': trial.suggest_float('lambda_l2', 2.0, 8.0),
                        'max_depth': trial.suggest_int('max_depth', 12, 18),
                        'path_smooth': trial.suggest_float('path_smooth', 0.8, 3.0),
                        'verbose': -1,
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 4000,
                        'early_stopping_rounds': 250,
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 45, 55),
                        'force_row_wise': True,
                        'max_bin': 255,
                        'num_threads': 12,
                        'device_type': 'cpu',
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 80, 200),
                        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.8, 0.95),
                        'extra_trees': True
                    }
                
                elif model_type.lower() == 'xgboost':
                    params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'tree_method': 'gpu_hist' if self.gpu_available else 'hist',
                        'gpu_id': 0 if self.gpu_available else None,
                        'predictor': 'gpu_predictor' if self.gpu_available else None,
                        'max_depth': trial.suggest_int('max_depth', 8, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
                        'subsample': trial.suggest_float('subsample', 0.75, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 0.95),
                        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.8, 0.95),
                        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.8, 0.95),
                        'min_child_weight': trial.suggest_float('min_child_weight', 15, 40),
                        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 8.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 8.0),
                        'gamma': trial.suggest_float('gamma', 0.05, 2.0),
                        'max_leaves': trial.suggest_int('max_leaves', 255, 1023),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 45, 55),
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 4000,
                        'early_stopping_rounds': 250,
                        'max_bin': 255,
                        'nthread': 12,
                        'grow_policy': 'lossguide'
                    }
                
                elif model_type.lower() == 'catboost':
                    params = {
                        'loss_function': 'Logloss',
                        'eval_metric': 'Logloss',
                        'task_type': 'GPU' if self.gpu_available else 'CPU',
                        'devices': '0' if self.gpu_available else None,
                        'depth': trial.suggest_int('depth', 8, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10, 30),
                        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.8, 3.0),
                        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 8, 20),
                        'max_leaves': trial.suggest_int('max_leaves', 255, 1023),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 80, 300),
                        'iterations': 4000,
                        'random_seed': self.config.RANDOM_STATE,
                        'od_wait': 250,
                        'od_type': 'IncToDec',
                        'verbose': False,
                        'auto_class_weights': 'Balanced',
                        'thread_count': 12,
                        'grow_policy': 'Lossguide',
                        'bootstrap_type': 'Bayesian'
                    }
                
                elif model_type.lower() == 'deepctr':
                    if not self.gpu_available:
                        return 0.0
                    
                    if self.rtx_4060ti_optimized:
                        hidden_dims_options = [
                            [512, 256, 128, 64],
                            [1024, 512, 256, 128],
                            [1024, 512, 256, 128, 64],
                            [512, 256, 128],
                            [2048, 1024, 512, 256, 128]
                        ]
                    else:
                        hidden_dims_options = [
                            [256, 128, 64],
                            [512, 256, 128],
                            [512, 256, 128, 64],
                            [256, 128]
                        ]
                    
                    params = {
                        'hidden_dims': trial.suggest_categorical('hidden_dims', hidden_dims_options),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.15, 0.4),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0003, 0.002, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096] if self.rtx_4060ti_optimized else [512, 1024, 2048]),
                        'epochs': 80 if self.rtx_4060ti_optimized else 50,
                        'patience': 25,
                        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
                        'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
                        'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True, False]),
                        'use_residual': trial.suggest_categorical('use_residual', [True, False])
                    }
                
                else:
                    params = {}
                
                cv_result = self.cross_validate_ctr_model_optimized(model_type, X, y, cv_folds, params)
                
                combined_score = cv_result['combined_mean']
                ctr_optimized_score = cv_result.get('ctr_optimized_mean', 0.0)
                ctr_bias_score = cv_result.get('ctr_bias_mean', 0.0)
                
                final_score = 0.5 * combined_score + 0.3 * ctr_optimized_score + 0.2 * ctr_bias_score
                
                LargeDataMemoryTracker.force_cleanup()
                
                return final_score if final_score > 0 else 0.0
            
            except Exception as e:
                logger.error(f"고성능 Trial 실행 실패: {str(e)}")
                LargeDataMemoryTracker.force_cleanup()
                return 0.0
        
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.RANDOM_STATE, n_startup_trials=10),
                pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=5)
            )
            
            study.optimize(
                high_performance_objective, 
                n_trials=n_trials,
                timeout=2400,
                n_jobs=1,
                show_progress_bar=False
            )
            
        except KeyboardInterrupt:
            logger.info("고성능 하이퍼파라미터 튜닝이 중단되었습니다.")
        except Exception as e:
            logger.error(f"고성능 하이퍼파라미터 튜닝 중 오류 발생: {str(e)}")
        finally:
            LargeDataMemoryTracker.force_cleanup()
        
        if not hasattr(study, 'best_value') or study.best_value is None or study.best_value <= 0:
            logger.warning(f"{model_type} 고성능 하이퍼파라미터 튜닝에서 유효한 결과를 얻지 못했습니다.")
            best_params = self._get_high_performance_params(model_type)
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
        
        logger.info(f"{model_type} 고성능 하이퍼파라미터 튜닝 완료")
        logger.info(f"최적 점수: {tuning_results['best_score']:.4f}")
        logger.info(f"수행된 trials: {tuning_results['n_trials']}/{n_trials}")
        
        return tuning_results
    
    def train_all_ctr_models_optimized(self,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     X_val: Optional[pd.DataFrame] = None,
                                     y_val: Optional[pd.Series] = None,
                                     model_types: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """모든 고성능 CTR 모델 학습 - Combined Score 0.30+ 목표"""
        
        if model_types is None:
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'catboost'] if m in available_models]
            
            available_memory = self.memory_tracker.get_available_memory()
            gpu_memory = self.memory_tracker.get_gpu_memory_usage()
            
            if self.gpu_available and 'deepctr' in available_models and gpu_memory['free'] > 8:
                model_types.append('deepctr')
        
        logger.info(f"모든 고성능 CTR 모델 학습 시작: {model_types}")
        logger.info(f"데이터 크기: {len(X_train):,}")
        logger.info(f"사용 가능 메모리: {self.memory_tracker.get_available_memory():.2f}GB")
        logger.info(f"GPU 메모리: {self.memory_tracker.get_gpu_memory_usage()['free']:.2f}GB")
        
        trained_models = {}
        
        for model_type in model_types:
            try:
                available_memory = self.memory_tracker.get_available_memory()
                gpu_memory = self.memory_tracker.get_gpu_memory_usage()
                
                if available_memory < 5:
                    logger.warning(f"메모리 부족으로 {model_type} 고성능 CTR 모델 학습 생략")
                    continue
                
                if model_type == 'deepctr' and gpu_memory['free'] < 4:
                    logger.warning(f"GPU 메모리 부족으로 {model_type} 모델 학습 생략")
                    continue
                
                logger.info(f"{model_type} 고성능 CTR 모델 학습 시작")
                logger.info(f"메모리: {available_memory:.2f}GB, GPU: {gpu_memory['free']:.2f}GB")
                
                if model_type in self.best_params:
                    params = self.best_params[model_type]
                else:
                    params = self._get_high_performance_params(model_type)
                
                model = self.train_single_model_optimized(
                    model_type, X_train, y_train, X_val, y_val, params
                )
                
                trained_models[model_type] = model
                
                logger.info(f"{model_type} 고성능 CTR 모델 학습 완료")
                
                self._cleanup_memory_after_training(model_type)
                
                current_memory = self.memory_tracker.get_available_memory()
                current_gpu = self.memory_tracker.get_gpu_memory_usage()
                logger.info(f"{model_type} 학습 후 - 메모리: {current_memory:.2f}GB, GPU: {current_gpu['free']:.2f}GB")
                
            except Exception as e:
                logger.error(f"{model_type} 고성능 CTR 모델 학습 실패: {str(e)}")
                self._cleanup_memory_after_training(model_type)
                continue
        
        logger.info(f"모든 고성능 CTR 모델 학습 완료. 성공한 모델: {list(trained_models.keys())}")
        
        return trained_models
    
    def _get_high_performance_params(self, model_type: str) -> Dict[str, Any]:
        """Combined Score 0.30+ 달성을 위한 고성능 기본 파라미터"""
        
        if model_type.lower() == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 511,
                'learning_rate': 0.025,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.75,
                'bagging_freq': 7,
                'min_child_samples': 300,
                'min_child_weight': 15,
                'lambda_l1': 3.0,
                'lambda_l2': 3.0,
                'max_depth': 15,
                'verbose': -1,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 4000,
                'early_stopping_rounds': 250,
                'scale_pos_weight': 49.8,
                'force_row_wise': True,
                'max_bin': 255,
                'num_threads': 12,
                'device_type': 'cpu',
                'min_data_in_leaf': 120,
                'feature_fraction_bynode': 0.85,
                'extra_trees': True,
                'path_smooth': 1.5,
                'grow_policy': 'lossguide'
            }
            return params
        
        elif model_type.lower() == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'gpu_hist' if self.gpu_available else 'hist',
                'gpu_id': 0 if self.gpu_available else None,
                'predictor': 'gpu_predictor' if self.gpu_available else None,
                'max_depth': 10,
                'learning_rate': 0.025,
                'subsample': 0.82,
                'colsample_bytree': 0.85,
                'colsample_bylevel': 0.85,
                'colsample_bynode': 0.85,
                'min_child_weight': 20,
                'reg_alpha': 3.0,
                'reg_lambda': 3.0,
                'scale_pos_weight': 49.8,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 4000,
                'early_stopping_rounds': 250,
                'max_bin': 255,
                'nthread': 12,
                'grow_policy': 'lossguide',
                'max_leaves': 511,
                'gamma': 0.15
            }
            if not self.gpu_available:
                params.pop('gpu_id', None)
                params.pop('predictor', None)
            return params
        
        elif model_type.lower() == 'catboost':
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'task_type': 'GPU' if self.gpu_available else 'CPU',
                'devices': '0' if self.gpu_available else None,
                'depth': 10,
                'learning_rate': 0.025,
                'l2_leaf_reg': 15,
                'iterations': 4000,
                'random_seed': self.config.RANDOM_STATE,
                'od_wait': 250,
                'od_type': 'IncToDec',
                'verbose': False,
                'auto_class_weights': 'Balanced',
                'max_ctr_complexity': 3,
                'thread_count': 12,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.5,
                'leaf_estimation_iterations': 12,
                'leaf_estimation_method': 'Newton',
                'grow_policy': 'Lossguide',
                'max_leaves': 511,
                'min_data_in_leaf': 120
            }
            if not self.gpu_available:
                params.pop('devices', None)
            return params
        
        elif model_type.lower() == 'deepctr':
            if self.rtx_4060ti_optimized:
                params = {
                    'hidden_dims': [1024, 512, 256, 128, 64],
                    'dropout_rate': 0.25,
                    'learning_rate': 0.0008,
                    'weight_decay': 5e-5,
                    'batch_size': 2048,
                    'epochs': 80,
                    'patience': 25,
                    'use_batch_norm': True,
                    'activation': 'gelu',
                    'use_residual': True,
                    'use_attention': True,
                    'focal_loss_alpha': 0.3,
                    'focal_loss_gamma': 2.5,
                    'use_focal_loss': True
                }
            else:
                params = {
                    'hidden_dims': [512, 256, 128, 64],
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'weight_decay': 1e-5,
                    'batch_size': 1024,
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
        """모델별 고성능 메모리 정리"""
        try:
            gc.collect()
            
            if model_type.lower() in ['deepctr', 'xgboost', 'catboost'] and self.gpu_available:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
            
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except:
                pass
                
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")
    
    def save_models(self, output_dir: Path = None):
        """고성능 모델 및 Calibrator 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"고성능 CTR 모델 저장 시작: {output_dir}")
        
        for model_name, model_info in self.trained_models.items():
            try:
                model_path = output_dir / f"{model_name}_high_performance_model.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['model'], f, protocol=pickle.HIGHEST_PROTOCOL)
                
                metadata = {
                    'model_type': model_name,
                    'training_time': model_info.get('training_time', 0.0),
                    'params': model_info.get('params', {}),
                    'calibrated': model_info.get('calibrated', False),
                    'memory_used': model_info.get('memory_used', 0.0),
                    'gpu_memory_used': model_info.get('gpu_memory_used', 0.0),
                    'data_size': model_info.get('data_size', 0),
                    'cv_result': model_info.get('cv_result', None),
                    'device': str(self.device),
                    'high_performance_optimized': True,
                    'rtx_4060ti_optimized': self.rtx_4060ti_optimized,
                    'combined_score_target': 0.30
                }
                
                metadata_path = output_dir / f"{model_name}_high_performance_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
                
                logger.info(f"{model_name} 고성능 CTR 모델 저장 완료: {model_path}")
                
            except Exception as e:
                logger.error(f"{model_name} 고성능 CTR 모델 저장 실패: {str(e)}")
        
        if self.calibrators:
            calibrator_path = output_dir / "high_performance_ctr_calibrators.pkl"
            try:
                with open(calibrator_path, 'wb') as f:
                    pickle.dump(self.calibrators, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"고성능 CTR Calibrator 저장 완료: {calibrator_path}")
            except Exception as e:
                logger.error(f"고성능 CTR Calibrator 저장 실패: {str(e)}")
        
        if self.cv_results:
            cv_results_path = output_dir / "high_performance_cv_results.json"
            try:
                with open(cv_results_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cv_results, f, indent=2, default=str, ensure_ascii=False)
                logger.info(f"고성능 CV 결과 저장 완료: {cv_results_path}")
            except Exception as e:
                logger.error(f"고성능 CV 결과 저장 실패: {str(e)}")
        
        if self.best_params:
            best_params_path = output_dir / "high_performance_best_params.json"
            try:
                with open(best_params_path, 'w', encoding='utf-8') as f:
                    json.dump(self.best_params, f, indent=2, default=str, ensure_ascii=False)
                logger.info(f"고성능 최적 파라미터 저장 완료: {best_params_path}")
            except Exception as e:
                logger.error(f"고성능 최적 파라미터 저장 실패: {str(e)}")
    
    def load_models(self, input_dir: Path = None) -> Dict[str, BaseModel]:
        """저장된 고성능 CTR 모델 로딩"""
        if input_dir is None:
            input_dir = self.config.MODEL_DIR
        
        input_dir = Path(input_dir)
        logger.info(f"고성능 CTR 모델 로딩 시작: {input_dir}")
        
        loaded_models = {}
        
        model_files = list(input_dir.glob("*_high_performance_model.pkl"))
        
        for model_file in model_files:
            try:
                model_name = model_file.stem.replace('_high_performance_model', '')
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                loaded_models[model_name] = model
                logger.info(f"{model_name} 고성능 CTR 모델 로딩 완료")
                
                metadata_file = input_dir / f"{model_name}_high_performance_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        self.trained_models[model_name] = {
                            'model': model,
                            'params': metadata.get('params', {}),
                            'training_time': metadata.get('training_time', 0.0),
                            'calibrated': metadata.get('calibrated', False),
                            'memory_used': metadata.get('memory_used', 0.0),
                            'gpu_memory_used': metadata.get('gpu_memory_used', 0.0),
                            'data_size': metadata.get('data_size', 0),
                            'cv_result': metadata.get('cv_result', None)
                        }
                        
                    except Exception as e:
                        logger.warning(f"{model_name} 메타데이터 로딩 실패: {e}")
                        self.trained_models[model_name] = {
                            'model': model,
                            'params': {},
                            'training_time': 0.0,
                            'calibrated': False,
                            'memory_used': 0.0,
                            'gpu_memory_used': 0.0,
                            'data_size': 0,
                            'cv_result': None
                        }
                
            except Exception as e:
                logger.error(f"{model_file} 고성능 CTR 모델 로딩 실패: {str(e)}")
        
        calibrator_path = input_dir / "high_performance_ctr_calibrators.pkl"
        if calibrator_path.exists():
            try:
                with open(calibrator_path, 'rb') as f:
                    self.calibrators = pickle.load(f)
                logger.info("고성능 CTR Calibrator 로딩 완료")
            except Exception as e:
                logger.error(f"고성능 CTR Calibrator 로딩 실패: {str(e)}")
        
        cv_results_path = input_dir / "high_performance_cv_results.json"
        if cv_results_path.exists():
            try:
                with open(cv_results_path, 'r', encoding='utf-8') as f:
                    self.cv_results = json.load(f)
                logger.info("고성능 CV 결과 로딩 완료")
            except Exception as e:
                logger.error(f"고성능 CV 결과 로딩 실패: {str(e)}")
        
        best_params_path = input_dir / "high_performance_best_params.json"
        if best_params_path.exists():
            try:
                with open(best_params_path, 'r', encoding='utf-8') as f:
                    self.best_params = json.load(f)
                logger.info("고성능 최적 파라미터 로딩 완료")
            except Exception as e:
                logger.error(f"고성능 최적 파라미터 로딩 실패: {str(e)}")
        
        return loaded_models
    
    def get_training_summary(self) -> Dict[str, Any]:
        """고성능 CTR 학습 결과 요약"""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'cv_results': self.cv_results,
            'best_params': self.best_params,
            'model_count': len(self.trained_models),
            'device_used': str(self.device),
            'gpu_available': self.gpu_available,
            'rtx_4060ti_optimized': self.rtx_4060ti_optimized,
            'calibration_applied': len(self.calibrators) > 0,
            'high_performance_optimized': True,
            'combined_score_target': 0.30,
            'total_memory_used': sum(
                info.get('memory_used', 0.0) for info in self.trained_models.values()
            ),
            'total_gpu_memory_used': sum(
                info.get('gpu_memory_used', 0.0) for info in self.trained_models.values()
            ),
            'avg_training_time': np.mean([
                info.get('training_time', 0.0) for info in self.trained_models.values()
            ]) if self.trained_models else 0.0,
            'total_data_processed': sum(
                info.get('data_size', 0) for info in self.trained_models.values()
            )
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
                    'combined_score': best_model[1]['combined_mean'],
                    'combined_std': best_model[1]['combined_std'],
                    'ctr_optimized_score': best_model[1].get('ctr_optimized_mean', 0.0),
                    'ctr_bias_score': best_model[1].get('ctr_bias_mean', 0.0),
                    'target_achieved': best_model[1]['combined_mean'] >= 0.30
                }
        
        return summary

ModelTrainer = CTRHighPerformanceTrainer

class HighPerformanceTrainingPipeline:
    """고성능 CTR 특화 전체 학습 파이프라인 - Combined Score 0.30+ 목표"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = CTRHighPerformanceTrainer(config)
        self.memory_tracker = LargeDataMemoryTracker()
        
    def run_full_high_performance_pipeline(self,
                                         X_train: pd.DataFrame,
                                         y_train: pd.Series,
                                         X_val: Optional[pd.DataFrame] = None,
                                         y_val: Optional[pd.Series] = None,
                                         tune_hyperparameters: bool = True,
                                         n_trials: int = None) -> Dict[str, Any]:
        """고성능 CTR 특화 전체 학습 파이프라인 실행"""
        
        logger.info("고성능 CTR 특화 전체 학습 파이프라인 시작")
        logger.info(f"목표: Combined Score 0.30+ 달성")
        logger.info(f"데이터 크기: {len(X_train):,}행")
        logger.info(f"초기 메모리 상태: {self.memory_tracker.get_available_memory():.2f}GB")
        logger.info(f"GPU 메모리: {self.memory_tracker.get_gpu_memory_usage()['free']:.2f}GB")
        
        pipeline_start_time = time.time()
        
        try:
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'catboost'] if m in available_models]
            
            available_memory = self.memory_tracker.get_available_memory()
            gpu_memory = self.memory_tracker.get_gpu_memory_usage()
            
            if self.trainer.gpu_available and 'deepctr' in available_models and gpu_memory['free'] > 8:
                model_types.append('deepctr')
                logger.info("GPU 환경: DeepCTR 고성능 모델 추가")
            
            if n_trials is None:
                if available_memory > 30 and gpu_memory['free'] > 12:
                    n_trials = 40
                elif available_memory > 25:
                    n_trials = 30
                elif available_memory > 20:
                    n_trials = 25
                else:
                    n_trials = 20
            
            if tune_hyperparameters and OPTUNA_AVAILABLE:
                logger.info("고성능 CTR 하이퍼파라미터 튜닝 단계")
                for model_type in model_types:
                    try:
                        if self.memory_tracker.get_available_memory() < 8:
                            logger.warning(f"메모리 부족으로 {model_type} 고성능 튜닝 생략")
                            continue
                        
                        if model_type == 'deepctr':
                            current_trials = max(10, n_trials // 2)
                        else:
                            current_trials = n_trials
                        
                        logger.info(f"{model_type} 하이퍼파라미터 튜닝 시작 (trials: {current_trials})")
                        
                        self.trainer.hyperparameter_tuning_ctr_optuna_optimized(
                            model_type, X_train, y_train, n_trials=current_trials, cv_folds=3
                        )
                        
                        LargeDataMemoryTracker.force_cleanup()
                        
                    except Exception as e:
                        logger.error(f"{model_type} 고성능 하이퍼파라미터 튜닝 실패: {str(e)}")
                        LargeDataMemoryTracker.force_cleanup()
            else:
                logger.info("고성능 CTR 하이퍼파라미터 튜닝 생략")
            
            logger.info("고성능 CTR 교차검증 평가 단계")
            for model_type in model_types:
                try:
                    if self.memory_tracker.get_available_memory() < 6:
                        logger.warning(f"메모리 부족으로 {model_type} 고성능 교차검증 생략")
                        continue
                    
                    params = self.trainer.best_params.get(model_type, None)
                    if params is None:
                        params = self.trainer._get_high_performance_params(model_type)
                    
                    logger.info(f"{model_type} 교차검증 시작")
                    
                    self.trainer.cross_validate_ctr_model_optimized(
                        model_type, X_train, y_train, cv_folds=5, params=params
                    )
                    
                    LargeDataMemoryTracker.force_cleanup()
                    
                except Exception as e:
                    logger.error(f"{model_type} 고성능 교차검증 실패: {str(e)}")
                    LargeDataMemoryTracker.force_cleanup()
            
            logger.info("고성능 CTR 최종 모델 학습 단계")
            logger.info(f"학습 전 메모리 상태: {self.memory_tracker.get_available_memory():.2f}GB")
            
            trained_models = self.trainer.train_all_ctr_models_optimized(
                X_train, y_train, X_val, y_val, model_types
            )
            
            logger.info(f"학습 후 메모리 상태: {self.memory_tracker.get_available_memory():.2f}GB")
            
            try:
                self.trainer.save_models()
                logger.info("고성능 CTR 모델 저장 완료")
            except Exception as e:
                logger.warning(f"고성능 CTR 모델 저장 실패: {str(e)}")
            
            pipeline_time = time.time() - pipeline_start_time
            summary = self.trainer.get_training_summary()
            summary['total_pipeline_time'] = pipeline_time
            summary['memory_peak'] = self.memory_tracker.get_memory_usage()
            summary['memory_available_end'] = self.memory_tracker.get_available_memory()
            summary['gpu_memory_end'] = self.memory_tracker.get_gpu_memory_usage()
            summary['high_performance_pipeline'] = True
            
            logger.info(f"고성능 CTR 전체 학습 파이프라인 완료 (소요시간: {pipeline_time:.2f}초)")
            logger.info(f"최종 메모리 상태: {summary['memory_available_end']:.2f}GB")
            
            if 'best_model' in summary:
                best_score = summary['best_model']['combined_score']
                target_achieved = summary['best_model']['target_achieved']
                logger.info(f"최고 Combined Score: {best_score:.4f}")
                logger.info(f"목표 달성 여부: {target_achieved} (목표: 0.30+)")
            
            return summary
            
        except Exception as e:
            logger.error(f"고성능 파이프라인 실행 실패: {str(e)}")
            LargeDataMemoryTracker.force_cleanup()
            raise
        finally:
            LargeDataMemoryTracker.force_cleanup()

TrainingPipeline = HighPerformanceTrainingPipeline