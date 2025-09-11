# training.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import json
from pathlib import Path
import time

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
import optuna
from optuna.samplers import TPESampler

from config import Config
from models import ModelFactory, BaseModel
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class ModelTrainer:
    """모델 학습 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.trained_models = {}
        self.cv_results = {}
        self.best_params = {}
        
    def train_single_model(self, 
                          model_type: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          params: Optional[Dict[str, Any]] = None) -> BaseModel:
        """단일 모델 학습"""
        
        logger.info(f"{model_type} 모델 학습 시작")
        start_time = time.time()
        
        # 모델 생성
        model_kwargs = {'params': params}
        if model_type.lower() == 'deepctr':
            model_kwargs['input_dim'] = X_train.shape[1]
        
        model = ModelFactory.create_model(model_type, **model_kwargs)
        
        # 모델 학습
        model.fit(X_train, y_train, X_val, y_val)
        
        # 학습 시간 기록
        training_time = time.time() - start_time
        logger.info(f"{model_type} 모델 학습 완료 (소요시간: {training_time:.2f}초)")
        
        # 모델 저장
        self.trained_models[model_type] = {
            'model': model,
            'training_time': training_time,
            'params': params or {}
        }
        
        return model
    
    def cross_validate_model(self,
                           model_type: str,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv_folds: int = None,
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """시간적 순서를 고려한 교차검증 모델 평가"""
        
        if cv_folds is None:
            cv_folds = self.config.N_SPLITS
        
        logger.info(f"{model_type} 모델 {cv_folds}폴드 교차검증 시작")
        
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
            
            # 데이터 분할
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # 모델 학습
            start_time = time.time()
            model = self.train_single_model(
                model_type, X_train_fold, y_train_fold,
                X_val_fold, y_val_fold, params
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
        
        # 평균 및 표준편차 계산
        cv_results = {
            'model_type': model_type,
            'ap_mean': np.mean(cv_scores['ap_scores']),
            'ap_std': np.std(cv_scores['ap_scores']),
            'wll_mean': np.mean(cv_scores['wll_scores']),
            'wll_std': np.std(cv_scores['wll_scores']),
            'combined_mean': np.mean(cv_scores['combined_scores']),
            'combined_std': np.std(cv_scores['combined_scores']),
            'training_time_mean': np.mean(cv_scores['training_times']),
            'scores_detail': cv_scores,
            'params': params or {}
        }
        
        self.cv_results[model_type] = cv_results
        
        logger.info(f"{model_type} 교차검증 완료")
        logger.info(f"평균 Combined Score: {cv_results['combined_mean']:.4f} (±{cv_results['combined_std']:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning_optuna(self,
                                   model_type: str,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   n_trials: int = 100,
                                   cv_folds: int = 3) -> Dict[str, Any]:
        """CTR 특화 하이퍼파라미터 튜닝"""
        
        logger.info(f"{model_type} 모델 하이퍼파라미터 튜닝 시작 (trials: {n_trials})")
        
        def objective(trial):
            # 모델 타입별 CTR 특화 하이퍼파라미터 공간
            if model_type.lower() == 'lightgbm':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                    'verbose': -1,
                    'random_state': self.config.RANDOM_STATE,
                    'n_estimators': 1000,
                    'early_stopping_rounds': 100,
                    'is_unbalance': True
                }
            
            elif model_type.lower() == 'xgboost':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1, 20),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 30, 70),
                    'random_state': self.config.RANDOM_STATE,
                    'n_estimators': 1000,
                    'early_stopping_rounds': 100
                }
            
            elif model_type.lower() == 'catboost':
                params = {
                    'loss_function': 'Logloss',
                    'eval_metric': 'Logloss',
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'iterations': 1000,
                    'random_seed': self.config.RANDOM_STATE,
                    'early_stopping_rounds': 100,
                    'verbose': False,
                    'auto_class_weights': 'Balanced'
                }
            
            elif model_type.lower() == 'deepctr':
                hidden_dims_options = [
                    [256, 128, 64],
                    [512, 256, 128],
                    [1024, 512, 256],
                    [256, 128],
                    [512, 256]
                ]
                params = {
                    'hidden_dims': trial.suggest_categorical('hidden_dims', hidden_dims_options),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01),
                    'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),
                    'epochs': 50,
                    'patience': 10
                }
            
            else:
                # 기본 파라미터 사용
                params = {}
            
            # 교차검증 수행
            cv_result = self.cross_validate_model(model_type, X, y, cv_folds, params)
            
            # Combined Score 최적화
            return cv_result['combined_mean']
        
        # Optuna 스터디 생성
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.RANDOM_STATE)
        )
        
        # 최적화 실행
        study.optimize(objective, n_trials=n_trials)
        
        # 결과 정리
        tuning_results = {
            'model_type': model_type,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'study': study
        }
        
        self.best_params[model_type] = study.best_params
        
        logger.info(f"{model_type} 하이퍼파라미터 튜닝 완료")
        logger.info(f"최적 점수: {study.best_value:.4f}")
        logger.info(f"최적 파라미터: {study.best_params}")
        
        return tuning_results
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        model_types: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """모든 모델 학습"""
        
        if model_types is None:
            model_types = ['lightgbm', 'xgboost', 'catboost']
        
        logger.info(f"모든 모델 학습 시작: {model_types}")
        
        trained_models = {}
        
        for model_type in model_types:
            try:
                # 최적 파라미터가 있으면 사용, 없으면 CTR 특화 기본 파라미터 사용
                if model_type in self.best_params:
                    params = self.best_params[model_type]
                else:
                    params = self._get_ctr_optimized_params(model_type)
                
                # 모델 학습
                model = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val, params
                )
                
                trained_models[model_type] = model
                
            except Exception as e:
                logger.error(f"{model_type} 모델 학습 실패: {str(e)}")
                continue
        
        logger.info(f"모든 모델 학습 완료. 성공한 모델: {list(trained_models.keys())}")
        
        return trained_models
    
    def _get_ctr_optimized_params(self, model_type: str) -> Dict[str, Any]:
        """CTR 예측에 특화된 기본 파라미터"""
        
        if model_type.lower() == 'lightgbm':
            return {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 127,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 100,
                'min_child_weight': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 1000,
                'early_stopping_rounds': 100,
                'is_unbalance': True
            }
        
        elif model_type.lower() == 'xgboost':
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 10,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'scale_pos_weight': 51.2,  # 1/0.0191 - 1
                'random_state': self.config.RANDOM_STATE,
                'n_estimators': 1000,
                'early_stopping_rounds': 100
            }
        
        elif model_type.lower() == 'catboost':
            return {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'depth': 8,
                'learning_rate': 0.05,
                'iterations': 1000,
                'l2_leaf_reg': 3,
                'random_seed': self.config.RANDOM_STATE,
                'early_stopping_rounds': 100,
                'verbose': False,
                'auto_class_weights': 'Balanced'
            }
        
        else:
            return {}
    
    def save_models(self, output_dir: Path = None):
        """모델 저장"""
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
                    'params': model_info['params']
                }
                
                metadata_path = output_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"{model_name} 모델 저장 완료: {model_path}")
                
            except Exception as e:
                logger.error(f"{model_name} 모델 저장 실패: {str(e)}")
    
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
        
        self.trained_models = {name: {'model': model} for name, model in loaded_models.items()}
        
        return loaded_models
    
    def get_training_summary(self) -> Dict[str, Any]:
        """학습 결과 요약"""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'cv_results': self.cv_results,
            'best_params': self.best_params,
            'model_count': len(self.trained_models)
        }
        
        if self.cv_results:
            # 최고 성능 모델 찾기
            best_model = max(
                self.cv_results.items(),
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
                         n_trials: int = 50) -> Dict[str, Any]:
        """전체 학습 파이프라인 실행"""
        
        logger.info("전체 학습 파이프라인 시작")
        pipeline_start_time = time.time()
        
        # CTR 예측에 효과적인 모델들
        model_types = ['lightgbm', 'xgboost', 'catboost']
        
        # 1. 하이퍼파라미터 튜닝 (옵션)
        if tune_hyperparameters:
            logger.info("하이퍼파라미터 튜닝 단계")
            for model_type in model_types:
                try:
                    self.trainer.hyperparameter_tuning_optuna(
                        model_type, X_train, y_train, n_trials=n_trials, cv_folds=3
                    )
                except Exception as e:
                    logger.error(f"{model_type} 하이퍼파라미터 튜닝 실패: {str(e)}")
        
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
        self.trainer.save_models()
        
        # 5. 결과 요약
        pipeline_time = time.time() - pipeline_start_time
        summary = self.trainer.get_training_summary()
        summary['total_pipeline_time'] = pipeline_time
        
        logger.info(f"전체 학습 파이프라인 완료 (소요시간: {pipeline_time:.2f}초)")
        
        return summary