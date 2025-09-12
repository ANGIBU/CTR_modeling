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

# PyTorch import 안전 처리
TORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        try:
            # GPU 테스트
            test_tensor = torch.tensor([1.0]).cuda()
            test_tensor.cpu()
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

# Optuna import 안전 처리
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

class ModelTrainer:
    """GPU 및 Calibration을 지원하는 모델 학습 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.device = getattr(config, 'DEVICE', 'cpu')
        self.trained_models = {}
        self.cv_results = {}
        self.best_params = {}
        self.calibrators = {}
        
        # GPU 환경 확인 및 안전 설정
        self.gpu_available = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                test_tensor.cpu()
                del test_tensor
                torch.cuda.empty_cache()
                self.gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU 학습 환경: {gpu_name}")
                logger.info(f"GPU 메모리: {gpu_memory:.1f}GB")
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
        """단일 모델 학습 (Calibration 포함)"""
        
        logger.info(f"{model_type} 모델 학습 시작")
        start_time = time.time()
        
        try:
            # 파라미터 검증 및 수정
            if params:
                params = self._validate_and_fix_params(model_type, params)
            
            # 모델 생성
            model_kwargs = {'params': params}
            if model_type.lower() == 'deepctr':
                model_kwargs['input_dim'] = X_train.shape[1]
            
            model = ModelFactory.create_model(model_type, **model_kwargs)
            
            # 모델 학습
            model.fit(X_train, y_train, X_val, y_val)
            
            # Calibration 적용
            if apply_calibration and X_val is not None and y_val is not None:
                self._apply_model_calibration(model, X_val, y_val)
            
            # 학습 시간 기록
            training_time = time.time() - start_time
            logger.info(f"{model_type} 모델 학습 완료 (소요시간: {training_time:.2f}초)")
            
            # 모델 저장
            self.trained_models[model_type] = {
                'model': model,
                'training_time': training_time,
                'params': params or {},
                'calibrated': apply_calibration
            }
            
            # 메모리 정리
            if model_type.lower() == 'deepctr':
                self._cleanup_gpu_memory()
            
            return model
            
        except Exception as e:
            logger.error(f"{model_type} 모델 학습 실패: {str(e)}")
            self._cleanup_gpu_memory()
            raise
    
    def _validate_and_fix_params(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """파라미터 검증 및 수정"""
        fixed_params = params.copy()
        
        try:
            if model_type.lower() == 'lightgbm':
                # LightGBM 충돌 방지
                if 'is_unbalance' in fixed_params and 'scale_pos_weight' in fixed_params:
                    fixed_params.pop('is_unbalance', None)
                    logger.info("LightGBM: is_unbalance 제거하여 충돌 방지")
                
                # 필수 파라미터 확인
                if 'objective' not in fixed_params:
                    fixed_params['objective'] = 'binary'
                if 'metric' not in fixed_params:
                    fixed_params['metric'] = 'binary_logloss'
                if 'verbose' not in fixed_params:
                    fixed_params['verbose'] = -1
                    
            elif model_type.lower() == 'xgboost':
                # XGBoost 파라미터 검증
                if 'objective' not in fixed_params:
                    fixed_params['objective'] = 'binary:logistic'
                if 'eval_metric' not in fixed_params:
                    fixed_params['eval_metric'] = 'logloss'
                
                # GPU 설정 안전 처리
                if self.gpu_available:
                    fixed_params['tree_method'] = 'gpu_hist'
                    fixed_params['gpu_id'] = 0
                else:
                    fixed_params['tree_method'] = 'hist'
                    fixed_params.pop('gpu_id', None)
                    
            elif model_type.lower() == 'catboost':
                # CatBoost 파라미터 검증
                if 'loss_function' not in fixed_params:
                    fixed_params['loss_function'] = 'Logloss'
                if 'verbose' not in fixed_params:
                    fixed_params['verbose'] = False
                
                # GPU 설정 안전 처리
                if self.gpu_available:
                    fixed_params['task_type'] = 'GPU'
                    fixed_params['devices'] = '0'
                else:
                    fixed_params['task_type'] = 'CPU'
                    fixed_params.pop('devices', None)
        except Exception as e:
            logger.warning(f"파라미터 검증 실패: {e}")
        
        return fixed_params
    
    def _apply_model_calibration(self, model: BaseModel, X_val: pd.DataFrame, y_val: pd.Series):
        """모델에 Calibration 적용"""
        try:
            logger.info(f"{model.name} Calibration 적용 시작")
            
            # 원본 예측 생성
            raw_predictions = model.predict_proba_raw(X_val)
            
            # CTR 특화 Calibration
            calibrator = CTRCalibrator(target_ctr=self.config.CALIBRATION_CONFIG['target_ctr'])
            
            # Platt Scaling 적용
            if self.config.CALIBRATION_CONFIG.get('platt_scaling', True):
                calibrator.fit_platt_scaling(y_val.values, raw_predictions)
            
            # Isotonic Regression 적용  
            if self.config.CALIBRATION_CONFIG.get('isotonic_regression', True):
                calibrator.fit_isotonic_regression(y_val.values, raw_predictions)
            
            # 편향 보정 적용
            calibrator.fit_bias_correction(y_val.values, raw_predictions)
            
            # 모델에 Calibration 적용
            model.apply_calibration(X_val, y_val, method='platt', cv_folds=3)
            
            # Calibrator 저장
            self.calibrators[model.name] = calibrator
            
            # 보정 전후 비교
            calibrated_predictions = model.predict_proba(X_val)
            
            original_ctr = raw_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val.mean()
            
            logger.info(f"Calibration 결과 - 원본: {original_ctr:.4f}, 보정: {calibrated_ctr:.4f}, 실제: {actual_ctr:.4f}")
            
        except Exception as e:
            logger.error(f"Calibration 적용 실패 ({model.name}): {str(e)}")
    
    def cross_validate_model(self,
                           model_type: str,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv_folds: int = None,
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """시간적 순서를 고려한 교차검증 모델 평가"""
        
        if cv_folds is None:
            cv_folds = min(3, self.config.N_SPLITS)  # 메모리 절약을 위해 3으로 제한
        
        logger.info(f"{model_type} 모델 {cv_folds}폴드 교차검증 시작")
        
        try:
            # 파라미터 검증
            if params:
                params = self._validate_and_fix_params(model_type, params)
            
            # 시간적 순서를 고려한 교차검증
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            cv_scores = {
                'ap_scores': [],
                'wll_scores': [],
                'combined_scores': [],
                'training_times': [],
                'fold_models': []
            }
            
            metrics_calculator = CTRMetrics()
            
            # 각 폴드별 학습 및 평가
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"폴드 {fold + 1}/{cv_folds} 시작")
                
                try:
                    # 데이터 분할
                    X_train_fold = X.iloc[train_idx].copy()
                    X_val_fold = X.iloc[val_idx].copy()
                    y_train_fold = y.iloc[train_idx].copy()
                    y_val_fold = y.iloc[val_idx].copy()
                    
                    # 모델 학습
                    start_time = time.time()
                    model = self.train_single_model(
                        model_type, X_train_fold, y_train_fold,
                        X_val_fold, y_val_fold, params,
                        apply_calibration=True
                    )
                    training_time = time.time() - start_time
                    
                    # 예측 및 평가
                    y_pred_proba = model.predict_proba(X_val_fold)
                    
                    # 평가 지표 계산
                    ap_score = metrics_calculator.average_precision(y_val_fold, y_pred_proba)
                    wll_score = metrics_calculator.weighted_log_loss(y_val_fold, y_pred_proba)
                    combined_score = metrics_calculator.combined_score(y_val_fold, y_pred_proba)
                    
                    # 결과 저장
                    cv_scores['ap_scores'].append(ap_score)
                    cv_scores['wll_scores'].append(wll_score)
                    cv_scores['combined_scores'].append(combined_score)
                    cv_scores['training_times'].append(training_time)
                    cv_scores['fold_models'].append(model)
                    
                    logger.info(f"폴드 {fold + 1} 완료 - AP: {ap_score:.4f}, WLL: {wll_score:.4f}, Combined: {combined_score:.4f}")
                    
                    # 메모리 정리
                    del X_train_fold, X_val_fold, y_train_fold, y_val_fold
                    self._cleanup_gpu_memory()
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"폴드 {fold + 1} 실행 실패: {str(e)}")
                    # 실패한 폴드는 기본값으로 채움
                    cv_scores['ap_scores'].append(0.0)
                    cv_scores['wll_scores'].append(float('inf'))
                    cv_scores['combined_scores'].append(0.0)
                    cv_scores['training_times'].append(0.0)
                    cv_scores['fold_models'].append(None)
                    self._cleanup_gpu_memory()
            
            # 유효한 점수만 사용하여 평균 계산
            valid_ap_scores = [s for s in cv_scores['ap_scores'] if s > 0]
            valid_wll_scores = [s for s in cv_scores['wll_scores'] if s != float('inf')]
            valid_combined_scores = [s for s in cv_scores['combined_scores'] if s > 0]
            
            if not valid_combined_scores:
                logger.warning(f"{model_type} 모든 폴드가 실패했습니다")
                return {
                    'model_type': model_type,
                    'ap_mean': 0.0,
                    'ap_std': 0.0,
                    'wll_mean': float('inf'),
                    'wll_std': 0.0,
                    'combined_mean': 0.0,
                    'combined_std': 0.0,
                    'training_time_mean': 0.0,
                    'scores_detail': cv_scores,
                    'params': params or {}
                }
            
            # 평균 및 표준편차 계산
            cv_results = {
                'model_type': model_type,
                'ap_mean': np.mean(valid_ap_scores) if valid_ap_scores else 0.0,
                'ap_std': np.std(valid_ap_scores) if len(valid_ap_scores) > 1 else 0.0,
                'wll_mean': np.mean(valid_wll_scores) if valid_wll_scores else float('inf'),
                'wll_std': np.std(valid_wll_scores) if len(valid_wll_scores) > 1 else 0.0,
                'combined_mean': np.mean(valid_combined_scores),
                'combined_std': np.std(valid_combined_scores) if len(valid_combined_scores) > 1 else 0.0,
                'training_time_mean': np.mean([t for t in cv_scores['training_times'] if t > 0]),
                'scores_detail': cv_scores,
                'params': params or {}
            }
            
            self.cv_results[model_type] = cv_results
            
            logger.info(f"{model_type} 교차검증 완료")
            logger.info(f"평균 Combined Score: {cv_results['combined_mean']:.4f} (±{cv_results['combined_std']:.4f})")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"{model_type} 교차검증 실패: {str(e)}")
            self._cleanup_gpu_memory()
            raise
    
    def hyperparameter_tuning_optuna(self,
                                   model_type: str,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   n_trials: int = None,
                                   cv_folds: int = 3) -> Dict[str, Any]:
        """CTR 특화 하이퍼파라미터 튜닝"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna가 설치되지 않았습니다. 기본 파라미터 사용")
            best_params = self._get_ctr_optimized_params(model_type)
            self.best_params[model_type] = best_params
            return {
                'model_type': model_type,
                'best_params': best_params,
                'best_score': 0.0,
                'n_trials': 0,
                'study': None
            }
        
        if n_trials is None:
            n_trials = min(30, self.config.TUNING_CONFIG['n_trials'])  # 메모리 절약
        
        logger.info(f"{model_type} 모델 하이퍼파라미터 튜닝 시작 (trials: {n_trials})")
        
        def objective(trial):
            try:
                # 모델 타입별 CTR 특화 하이퍼파라미터 공간
                if model_type.lower() == 'lightgbm':
                    params = {
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': trial.suggest_int('num_leaves', 127, 255),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                        'min_child_samples': trial.suggest_int('min_child_samples', 100, 300),
                        'min_child_weight': trial.suggest_float('min_child_weight', 5, 30),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 3.0),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 3.0),
                        'verbose': -1,
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 500,  # 튜닝 속도를 위해 감소
                        'early_stopping_rounds': 50,
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 30, 60)
                    }
                
                elif model_type.lower() == 'xgboost':
                    params = {
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'tree_method': 'gpu_hist' if self.gpu_available else 'hist',
                        'gpu_id': 0 if self.gpu_available else None,
                        'max_depth': trial.suggest_int('max_depth', 6, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                        'min_child_weight': trial.suggest_float('min_child_weight', 10, 40),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 3.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
                        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 30, 60),
                        'random_state': self.config.RANDOM_STATE,
                        'n_estimators': 500,
                        'early_stopping_rounds': 50
                    }
                
                elif model_type.lower() == 'catboost':
                    params = {
                        'loss_function': 'Logloss',
                        'eval_metric': 'Logloss',
                        'task_type': 'GPU' if self.gpu_available else 'CPU',
                        'devices': '0' if self.gpu_available else None,
                        'depth': trial.suggest_int('depth', 6, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 8),
                        'iterations': 500,
                        'random_seed': self.config.RANDOM_STATE,
                        'early_stopping_rounds': 50,
                        'verbose': False,
                        'auto_class_weights': 'Balanced'
                    }
                
                elif model_type.lower() == 'deepctr':
                    if not self.gpu_available:
                        return 0.0  # GPU가 없으면 DeepCTR 스킵
                        
                    hidden_dims_options = [
                        [512, 256, 128],
                        [1024, 512, 256],
                        [256, 128, 64]
                    ]
                    params = {
                        'hidden_dims': trial.suggest_categorical('hidden_dims', hidden_dims_options),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.001, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [2048, 4096]),
                        'epochs': 30,  # 튜닝 속도를 위해 감소
                        'patience': 8,
                        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
                        'activation': trial.suggest_categorical('activation', ['relu', 'gelu'])
                    }
                
                else:
                    params = {}
                
                # 교차검증 수행 (간소화)
                cv_result = self.cross_validate_model(model_type, X, y, cv_folds, params)
                
                score = cv_result['combined_mean']
                
                if score <= 0:
                    return 0.0
                    
                return score
            
            except Exception as e:
                logger.error(f"Trial 실행 실패: {str(e)}")
                self._cleanup_gpu_memory()
                return 0.0
        
        # Optuna 스터디 생성
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.RANDOM_STATE),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            )
            
            # 최적화 실행
            study.optimize(
                objective, 
                n_trials=n_trials,
                timeout=1200,  # 20분으로 제한
                n_jobs=1,  # GPU 사용 시 병렬 처리 제한
                show_progress_bar=False
            )
            
        except KeyboardInterrupt:
            logger.info("하이퍼파라미터 튜닝이 중단되었습니다.")
        except Exception as e:
            logger.error(f"하이퍼파라미터 튜닝 중 오류 발생: {str(e)}")
        finally:
            self._cleanup_gpu_memory()
        
        # 결과 정리
        if not hasattr(study, 'best_value') or study.best_value is None or study.best_value <= 0:
            logger.warning(f"{model_type} 하이퍼파라미터 튜닝에서 유효한 결과를 얻지 못했습니다.")
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
        
        logger.info(f"{model_type} 하이퍼파라미터 튜닝 완료")
        logger.info(f"최적 점수: {tuning_results['best_score']:.4f}")
        logger.info(f"최적 파라미터: {best_params}")
        
        return tuning_results
    
    def parallel_model_training(self,
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_val: Optional[pd.DataFrame] = None,
                               y_val: Optional[pd.Series] = None,
                               model_types: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """병렬 모델 학습 (GPU 모델 제외)"""
        
        if model_types is None:
            model_types = ['lightgbm', 'xgboost', 'catboost']
        
        # GPU 모델과 CPU 모델 분리
        gpu_models = ['deepctr']
        cpu_models = [m for m in model_types if m not in gpu_models]
        
        trained_models = {}
        
        # GPU 모델 순차 학습
        for model_type in gpu_models:
            if model_type in model_types and self.gpu_available:
                try:
                    params = self.best_params.get(model_type) or self._get_ctr_optimized_params(model_type)
                    model = self.train_single_model(
                        model_type, X_train, y_train, X_val, y_val, params
                    )
                    trained_models[model_type] = model
                    
                    # GPU 메모리 정리
                    self._cleanup_gpu_memory()
                    
                except Exception as e:
                    logger.error(f"{model_type} GPU 모델 학습 실패: {str(e)}")
                    self._cleanup_gpu_memory()
        
        # CPU 모델 학습 (순차적으로 - 메모리 안정성을 위해)
        for model_type in cpu_models:
            try:
                params = self.best_params.get(model_type) or self._get_ctr_optimized_params(model_type)
                model = self.train_single_model(
                    model_type, X_train.copy(), y_train.copy(), 
                    X_val.copy() if X_val is not None else None,
                    y_val.copy() if y_val is not None else None,
                    params
                )
                trained_models[model_type] = model
                logger.info(f"{model_type} 모델 학습 완료")
                
                # 메모리 정리
                gc.collect()
                
            except Exception as e:
                logger.error(f"{model_type} 모델 학습 실패: {str(e)}")
                continue
        
        logger.info(f"모델 학습 완료. 성공한 모델: {list(trained_models.keys())}")
        return trained_models
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        model_types: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """모든 모델 학습"""
        
        if model_types is None:
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'catboost'] if m in available_models]
            
            # GPU 사용 가능시에만 DeepCTR 추가
            if self.gpu_available and 'deepctr' in available_models:
                model_types.append('deepctr')
        
        logger.info(f"모든 모델 학습 시작: {model_types}")
        
        trained_models = {}
        
        for model_type in model_types:
            try:
                # 최적 파라미터 사용
                if model_type in self.best_params:
                    params = self.best_params[model_type]
                else:
                    params = self._get_ctr_optimized_params(model_type)
                
                # 모델 학습
                model = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val, params
                )
                
                trained_models[model_type] = model
                
                # DeepCTR 후 메모리 정리
                if model_type.lower() == 'deepctr':
                    self._cleanup_gpu_memory()
                
            except Exception as e:
                logger.error(f"{model_type} 모델 학습 실패: {str(e)}")
                self._cleanup_gpu_memory()
                continue
        
        logger.info(f"모든 모델 학습 완료. 성공한 모델: {list(trained_models.keys())}")
        
        return trained_models
    
    def _get_ctr_optimized_params(self, model_type: str) -> Dict[str, Any]:
        """CTR 예측에 특화된 기본 파라미터"""
        
        if model_type.lower() == 'lightgbm':
            params = self.config.LGBM_PARAMS.copy()
            params.pop('is_unbalance', None)  # 충돌 방지
            return params
        
        elif model_type.lower() == 'xgboost':
            params = self.config.XGB_PARAMS.copy()
            if not self.gpu_available:
                params['tree_method'] = 'hist'
                params.pop('gpu_id', None)
            return params
        
        elif model_type.lower() == 'catboost':
            params = self.config.CAT_PARAMS.copy()
            if not self.gpu_available:
                params['task_type'] = 'CPU'
                params.pop('devices', None)
            return params
        
        elif model_type.lower() == 'deepctr':
            return self.config.NN_PARAMS.copy()
        
        else:
            return {}
    
    def _cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"GPU 메모리 정리 실패: {e}")
        
        # CPU 메모리도 정리
        gc.collect()
    
    def save_models(self, output_dir: Path = None):
        """모델 및 Calibrator 저장"""
        if output_dir is None:
            output_dir = self.config.MODEL_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"모델 저장 시작: {output_dir}")
        
        for model_name, model_info in self.trained_models.items():
            try:
                model_path = output_dir / f"{model_name}_model.pkl"
                
                # 모델 저장
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['model'], f)
                
                # 메타데이터 저장
                metadata = {
                    'model_type': model_name,
                    'training_time': model_info['training_time'],
                    'params': model_info['params'],
                    'calibrated': model_info.get('calibrated', False),
                    'device': str(self.device)
                }
                
                metadata_path = output_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                logger.info(f"{model_name} 모델 저장 완료: {model_path}")
                
            except Exception as e:
                logger.error(f"{model_name} 모델 저장 실패: {str(e)}")
        
        # Calibrator 저장
        if self.calibrators:
            calibrator_path = output_dir / "calibrators.pkl"
            try:
                with open(calibrator_path, 'wb') as f:
                    pickle.dump(self.calibrators, f)
                logger.info(f"Calibrator 저장 완료: {calibrator_path}")
            except Exception as e:
                logger.error(f"Calibrator 저장 실패: {str(e)}")
    
    def load_models(self, input_dir: Path = None) -> Dict[str, BaseModel]:
        """저장된 모델 로딩"""
        if input_dir is None:
            input_dir = self.config.MODEL_DIR
        
        input_dir = Path(input_dir)
        logger.info(f"모델 로딩 시작: {input_dir}")
        
        loaded_models = {}
        
        # 모델 파일 찾기
        model_files = list(input_dir.glob("*_model.pkl"))
        
        for model_file in model_files:
            try:
                model_name = model_file.stem.replace('_model', '')
                
                # 모델 로딩
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                loaded_models[model_name] = model
                logger.info(f"{model_name} 모델 로딩 완료")
                
            except Exception as e:
                logger.error(f"{model_file} 모델 로딩 실패: {str(e)}")
        
        # Calibrator 로딩
        calibrator_path = input_dir / "calibrators.pkl"
        if calibrator_path.exists():
            try:
                with open(calibrator_path, 'rb') as f:
                    self.calibrators = pickle.load(f)
                logger.info("Calibrator 로딩 완료")
            except Exception as e:
                logger.error(f"Calibrator 로딩 실패: {str(e)}")
        
        self.trained_models = {name: {'model': model} for name, model in loaded_models.items()}
        
        return loaded_models
    
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 결과 요약"""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'cv_results': self.cv_results,
            'best_params': self.best_params,
            'model_count': len(self.trained_models),
            'device_used': str(self.device),
            'gpu_available': self.gpu_available,
            'calibration_applied': len(self.calibrators) > 0
        }
        
        if self.cv_results:
            # 최고 성능 모델 찾기
            valid_results = {k: v for k, v in self.cv_results.items() if v['combined_mean'] > 0}
            if valid_results:
                best_model = max(
                    valid_results.items(),
                    key=lambda x: x[1]['combined_mean']
                )
                summary['best_model'] = {
                    'name': best_model[0],
                    'score': best_model[1]['combined_mean'],
                    'std': best_model[1]['combined_std']
                }
        
        return summary

class TrainingPipeline:
    """전체 학습 파이프라인"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trainer = ModelTrainer(config)
        
    def run_full_pipeline(self,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: Optional[pd.DataFrame] = None,
                         y_val: Optional[pd.Series] = None,
                         tune_hyperparameters: bool = True,
                         n_trials: int = None) -> Dict[str, Any]:
        """전체 학습 파이프라인 실행"""
        
        logger.info("전체 학습 파이프라인 시작")
        pipeline_start_time = time.time()
        
        try:
            # CTR 예측에 효과적인 모델들
            available_models = ModelFactory.get_available_models()
            model_types = [m for m in ['lightgbm', 'xgboost', 'catboost'] if m in available_models]
            
            # GPU 사용 가능 시 DeepCTR 추가
            if self.trainer.gpu_available and 'deepctr' in available_models:
                model_types.append('deepctr')
                logger.info("GPU 환경: DeepCTR 모델 추가")
            
            if n_trials is None:
                n_trials = min(20, self.config.TUNING_CONFIG['n_trials'])  # 메모리 절약
            
            # 1. 하이퍼파라미터 튜닝 (옵션)
            if tune_hyperparameters and OPTUNA_AVAILABLE:
                logger.info("하이퍼파라미터 튜닝 단계")
                for model_type in model_types:
                    try:
                        # GPU 모델은 적은 trial로 제한
                        current_trials = max(5, n_trials // 4) if model_type == 'deepctr' else n_trials
                        
                        self.trainer.hyperparameter_tuning_optuna(
                            model_type, X_train, y_train, n_trials=current_trials, cv_folds=3
                        )
                    except Exception as e:
                        logger.error(f"{model_type} 하이퍼파라미터 튜닝 실패: {str(e)}")
            else:
                logger.info("하이퍼파라미터 튜닝 생략")
            
            # 2. 교차검증 평가
            logger.info("교차검증 평가 단계")
            for model_type in model_types:
                try:
                    params = self.trainer.best_params.get(model_type, None)
                    if params is None:
                        params = self.trainer._get_ctr_optimized_params(model_type)
                    self.trainer.cross_validate_model(model_type, X_train, y_train, params=params)
                except Exception as e:
                    logger.error(f"{model_type} 교차검증 실패: {str(e)}")
            
            # 3. 최종 모델 학습
            logger.info("최종 모델 학습 단계")
            trained_models = self.trainer.train_all_models(X_train, y_train, X_val, y_val, model_types)
            
            # 4. 모델 저장
            try:
                self.trainer.save_models()
            except Exception as e:
                logger.warning(f"모델 저장 실패: {str(e)}")
            
            # 5. 결과 요약
            pipeline_time = time.time() - pipeline_start_time
            summary = self.trainer.get_training_summary()
            summary['total_pipeline_time'] = pipeline_time
            
            logger.info(f"전체 학습 파이프라인 완료 (소요시간: {pipeline_time:.2f}초)")
            
            return summary
            
        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {str(e)}")
            self.trainer._cleanup_gpu_memory()
            raise
        finally:
            # 최종 정리
            self.trainer._cleanup_gpu_memory()

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
                    logger.info(f"GPU 메모리 (실행 전): {before['allocated_gb']:.2f}GB 사용 / {before['total_gb']:.2f}GB")
                    
                    result = func(*args, **kwargs)
                    
                    after = GPUMemoryManager.get_gpu_memory_info()
                    logger.info(f"GPU 메모리 (실행 후): {after['allocated_gb']:.2f}GB 사용 / {after['total_gb']:.2f}GB")
                    
                    return result
                except Exception as e:
                    logger.warning(f"GPU 메모리 모니터링 실패: {e}")
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper