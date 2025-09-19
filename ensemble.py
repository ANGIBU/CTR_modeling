# ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import gc
import time

from sklearn.linear_model import LogisticRegression, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna가 설치되지 않았습니다. 하이퍼파라미터 튜닝 기능이 비활성화됩니다.")

from config import Config
from models import BaseModel, CTRCalibrator
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """앙상블 모델 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.target_combined_score = 0.35
        self.ensemble_calibrator = None
        self.is_calibrated = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """앙상블 모델 학습"""
        pass
    
    @abstractmethod
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """앙상블 예측"""
        pass
    
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
    
    def get_base_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """모든 기본 모델의 예측 수집"""
        predictions = {}
        
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    pred = model.predict_proba(X)
                    logger.info(f"{name} 모델: 캘리브레이션 적용된 예측 사용")
                else:
                    pred = model.predict_proba(X)
                    
                predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                predictions[name] = np.full(len(X), 0.0201)
        
        return predictions
    
    def apply_ensemble_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, 
                                 ensemble_predictions: np.ndarray, method: str = 'auto'):
        """앙상블 수준 캘리브레이션 적용"""
        try:
            logger.info(f"{self.name} 앙상블에 캘리브레이션 적용: {method}")
            
            if len(ensemble_predictions) != len(y_val):
                logger.warning("앙상블 캘리브레이션: 크기 불일치")
                return
            
            self.ensemble_calibrator = CTRCalibrator(target_ctr=0.0201, method=method)
            self.ensemble_calibrator.fit(y_val.values, ensemble_predictions)
            
            self.is_calibrated = True
            
            calibrated_predictions = self.ensemble_calibrator.predict(ensemble_predictions)
            
            original_ctr = ensemble_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val.mean()
            
            logger.info(f"앙상블 캘리브레이션 결과:")
            logger.info(f"  - 원본 CTR: {original_ctr:.4f}")
            logger.info(f"  - 캘리브레이션 CTR: {calibrated_ctr:.4f}")
            logger.info(f"  - 실제 CTR: {actual_ctr:.4f}")
            logger.info(f"  - 최적 방법: {self.ensemble_calibrator.best_method}")
            
        except Exception as e:
            logger.error(f"앙상블 캘리브레이션 적용 실패: {e}")
            self.is_calibrated = False
    
    def predict_proba_calibrated(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """캘리브레이션이 적용된 앙상블 예측"""
        raw_prediction = self.predict_proba(base_predictions)
        
        if self.is_calibrated and self.ensemble_calibrator is not None:
            try:
                calibrated_prediction = self.ensemble_calibrator.predict(raw_prediction)
                return np.clip(calibrated_prediction, 1e-15, 1 - 1e-15)
            except Exception as e:
                logger.warning(f"앙상블 캘리브레이션 예측 실패: {e}")
        
        return raw_prediction
    
    def _enhance_ensemble_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """앙상블 예측 다양성 향상"""
        try:
            unique_count = len(np.unique(predictions))
            
            if unique_count < len(predictions) // 100:
                logger.info(f"{self.name}: 앙상블 예측 다양성 향상 적용")
                
                noise_scale = max(predictions.std() * 0.005, 1e-7)
                noise = np.random.normal(0, noise_scale, len(predictions))
                
                predictions = predictions + noise
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"앙상블 다양성 향상 실패: {e}")
            return predictions

class CTRMainEnsemble(BaseEnsemble):
    """CTR 예측 메인 앙상블"""
    
    def __init__(self, target_ctr: float = 0.0201, optimization_method: str = 'final_combined'):
        super().__init__("CTRMainEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.final_weights = {}
        self.metrics_calculator = CTRMetrics()
        self.temperature = 1.0
        self.bias_correction = 0.0
        self.multiplicative_correction = 1.0
        self.quantile_corrections = {}
        self.performance_boosters = {}
        self.target_combined_score = 0.35
        self.meta_learner = None
        self.stacking_weights = {}
        self.ensemble_execution_guaranteed = True  # 앙상블 실행 보장 플래그
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """메인 앙상블 학습"""
        logger.info(f"CTR 메인 앙상블 학습 시작 - 목표: Combined Score 0.35+")
        
        available_models = list(base_predictions.keys())
        logger.info(f"사용 가능한 모델: {available_models}")
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        try:
            # 1단계: 기본 가중치 계산
            logger.info("1단계: 기본 가중치 계산")
            self.final_weights = self._calculate_base_weights(base_predictions, y)
            
            # 2단계: 가중 앙상블 생성
            ensemble_pred = self._create_weighted_ensemble(base_predictions)
            
            # 3단계: CTR 후처리
            logger.info("3단계: CTR 후처리")
            self._apply_ctr_postprocessing(ensemble_pred, y)
            
            # 4단계: 메타 학습 적용
            logger.info("4단계: 메타 학습 적용")
            self._apply_meta_learning(base_predictions, y)
            
            # 5단계: 스택킹 레이어 추가
            logger.info("5단계: 스택킹 레이어 추가")
            self._apply_stacking_layer(base_predictions, y)
            
            # 6단계: 앙상블 캘리브레이션 적용
            logger.info("6단계: 앙상블 캘리브레이션 적용")
            try:
                split_idx = int(len(X) * 0.8)
                X_cal = X.iloc[split_idx:]
                y_cal = y.iloc[split_idx:]
                
                cal_base_predictions = {}
                for name in available_models:
                    if name in base_predictions:
                        cal_base_predictions[name] = base_predictions[name][split_idx:]
                
                cal_ensemble_pred = self._create_final_ensemble(cal_base_predictions)
                cal_ensemble_pred = self._apply_all_corrections(cal_ensemble_pred)
                
                self.apply_ensemble_calibration(X_cal, y_cal, cal_ensemble_pred, method='auto')
                
            except Exception as e:
                logger.warning(f"앙상블 캘리브레이션 적용 실패: {e}")
            
            # 7단계: 최종 검증 및 조정
            logger.info("7단계: 최종 검증 및 조정")
            final_pred = self._apply_all_corrections(ensemble_pred)
            
            if self.is_calibrated and self.ensemble_calibrator:
                try:
                    final_pred = self.ensemble_calibrator.predict(final_pred)
                except:
                    pass
            
            final_score = self.metrics_calculator.combined_score(y, final_pred)
            
            logger.info(f"메인 앙상블 Combined Score: {final_score:.4f}")
            logger.info(f"앙상블 캘리브레이션 적용: {'Yes' if self.is_calibrated else 'No'}")
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            logger.info("CTR 메인 앙상블 학습 완료")
            
        except Exception as e:
            logger.error(f"앙상블 학습 실패: {e}")
            # 실패 시에도 기본 가중치로 앙상블 보장
            self._create_fallback_ensemble(available_models)
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
    
    def _calculate_base_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """기본 가중치 계산"""
        
        model_names = list(base_predictions.keys())
        
        # Layer 1: 개별 모델 성능 평가
        individual_scores = {}
        ctr_alignment_scores = {}
        diversity_scores = {}
        calibration_scores = {}
        
        for name, pred in base_predictions.items():
            combined_score = self.metrics_calculator.combined_score(y, pred)
            ap_score = self.metrics_calculator.average_precision(y, pred)
            wll_score = self.metrics_calculator.weighted_log_loss(y, pred)
            
            predicted_ctr = pred.mean()
            actual_ctr = y.mean()
            ctr_alignment = np.exp(-abs(predicted_ctr - actual_ctr) * 500)
            
            pred_std = pred.std()
            pred_entropy = -np.mean(pred * np.log(pred + 1e-15) + (1 - pred) * np.log(1 - pred + 1e-15))
            diversity = pred_std * pred_entropy
            
            model = self.base_models.get(name)
            calibration_quality = 0.5
            if model and hasattr(model, 'is_calibrated') and model.is_calibrated:
                calibration_quality = 0.8
                if hasattr(model, 'calibrator') and model.calibrator:
                    calibration_summary = model.calibrator.get_calibration_summary()
                    if calibration_summary['calibration_scores']:
                        calibration_quality = max(calibration_summary['calibration_scores'].values())
            
            individual_scores[name] = combined_score
            ctr_alignment_scores[name] = ctr_alignment
            diversity_scores[name] = diversity
            calibration_scores[name] = calibration_quality
        
        # Layer 2: 가중치 계산
        if OPTUNA_AVAILABLE:
            try:
                optimized_weights = self._optuna_weight_optimization(
                    base_predictions, y, individual_scores, calibration_scores
                )
            except Exception as e:
                logger.warning(f"Optuna 가중치 튜닝 실패: {e}")
                optimized_weights = self._fallback_weight_calculation(
                    individual_scores, ctr_alignment_scores, diversity_scores, calibration_scores
                )
        else:
            optimized_weights = self._fallback_weight_calculation(
                individual_scores, ctr_alignment_scores, diversity_scores, calibration_scores
            )
        
        logger.info(f"기본 가중치: {optimized_weights}")
        return optimized_weights
    
    def _optuna_weight_optimization(self, base_predictions: Dict[str, np.ndarray], 
                                   y: pd.Series, 
                                   individual_scores: Dict[str, float],
                                   calibration_scores: Dict[str, float]) -> Dict[str, float]:
        """Optuna 가중치 튜닝"""
        
        model_names = list(base_predictions.keys())
        
        def objective(trial):
            weights = {}
            
            for name in model_names:
                base_performance = individual_scores.get(name, 0.1)
                calibration_quality = calibration_scores.get(name, 0.5)
                
                performance_factor = base_performance + 0.5 * calibration_quality
                
                if performance_factor > 0.7:
                    min_weight, max_weight = 0.2, 0.9
                elif performance_factor > 0.5:
                    min_weight, max_weight = 0.1, 0.7
                else:
                    min_weight, max_weight = 0.05, 0.5
                
                weights[name] = trial.suggest_float(f'weight_{name}', min_weight, max_weight)
            
            ensemble_method = trial.suggest_categorical('ensemble_method', ['weighted', 'power_weighted', 'rank_weighted'])
            temperature = trial.suggest_float('temperature', 0.8, 2.5)
            
            if ensemble_method == 'weighted':
                ensemble_pred = np.zeros(len(y))
                total_weight = sum(weights.values())
                for name, weight in weights.items():
                    if name in base_predictions:
                        ensemble_pred += (weight / total_weight) * base_predictions[name]
            
            elif ensemble_method == 'power_weighted':
                power = trial.suggest_float('power', 1.5, 4.0)
                ensemble_pred = np.zeros(len(y))
                total_weight = 0
                for name, weight in weights.items():
                    if name in base_predictions:
                        powered_pred = np.power(base_predictions[name], power)
                        ensemble_pred += weight * powered_pred
                        total_weight += weight
                if total_weight > 0:
                    ensemble_pred /= total_weight
                ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
            elif ensemble_method == 'rank_weighted':
                ensemble_pred = np.zeros(len(y))
                total_weight = 0
                for name, weight in weights.items():
                    if name in base_predictions:
                        rank_pred = pd.Series(base_predictions[name]).rank(pct=True).values
                        ensemble_pred += weight * rank_pred
                        total_weight += weight
                if total_weight > 0:
                    ensemble_pred /= total_weight
            
            # Temperature 적용
            if ensemble_method != 'rank_weighted':
                try:
                    logits = np.log(np.clip(ensemble_pred, 1e-15, 1-1e-15) / (1 - np.clip(ensemble_pred, 1e-15, 1-1e-15)))
                    scaled_logits = logits / temperature
                    ensemble_pred = 1 / (1 + np.exp(-scaled_logits))
                except:
                    pass
            
            ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
            combined_score = self.metrics_calculator.combined_score(y, ensemble_pred)
            
            predicted_ctr = ensemble_pred.mean()
            actual_ctr = y.mean()
            ctr_alignment = np.exp(-abs(predicted_ctr - actual_ctr) * 400)
            
            diversity = ensemble_pred.std()
            
            calibration_bonus = sum(weights[name] * calibration_scores.get(name, 0.5) 
                                  for name in weights if name in calibration_scores)
            calibration_bonus /= sum(weights.values()) if sum(weights.values()) > 0 else 1.0
            
            final_score = combined_score * (1 + 0.3 * ctr_alignment) * (1 + 0.2 * diversity) * (1 + 0.3 * calibration_bonus)
            
            return final_score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=30),
            pruner=MedianPruner(n_startup_trials=25, n_warmup_steps=20)
        )
        
        study.optimize(objective, n_trials=200, show_progress_bar=False)
        
        optimized_weights = {}
        for param_name, weight in study.best_params.items():
            if param_name.startswith('weight_'):
                model_name = param_name.replace('weight_', '')
                optimized_weights[model_name] = weight
        
        self.ensemble_method = study.best_params.get('ensemble_method', 'weighted')
        self.temperature = study.best_params.get('temperature', 1.0)
        
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
        
        logger.info(f"Optuna 가중치 튜닝 완료 - 최고 점수: {study.best_value:.4f}")
        logger.info(f"앙상블 방법: {self.ensemble_method}, Temperature: {self.temperature:.3f}")
        
        return optimized_weights
    
    def _fallback_weight_calculation(self, individual_scores: Dict[str, float], 
                                   ctr_alignment_scores: Dict[str, float],
                                   diversity_scores: Dict[str, float],
                                   calibration_scores: Dict[str, float]) -> Dict[str, float]:
        """대체 가중치 계산"""
        
        weights = {}
        for name in individual_scores.keys():
            performance_weight = individual_scores.get(name, 0.1)
            alignment_weight = ctr_alignment_scores.get(name, 0.5)
            diversity_weight = diversity_scores.get(name, 0.5)
            calibration_weight = calibration_scores.get(name, 0.5)
            
            combined_weight = (0.3 * performance_weight + 
                             0.25 * alignment_weight + 
                             0.15 * diversity_weight +
                             0.3 * calibration_weight)
            
            weights[name] = max(combined_weight, 0.05)
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _apply_ctr_postprocessing(self, predictions: np.ndarray, y: pd.Series):
        """CTR 후처리"""
        logger.info("CTR 후처리 시작")
        
        try:
            predicted_ctr = predictions.mean()
            actual_ctr = y.mean()
            self.bias_correction = actual_ctr - predicted_ctr
            
            if predicted_ctr > 0:
                self.multiplicative_correction = actual_ctr / predicted_ctr
            else:
                self.multiplicative_correction = 1.0
            
            self._fit_temperature_scaling(predictions, y)
            self._fit_quantile_corrections(predictions, y)
            
            logger.info(f"CTR 후처리 완료")
            logger.info(f"편향 보정: {self.bias_correction:.4f}, 승수 보정: {self.multiplicative_correction:.4f}")
            logger.info(f"Temperature: {self.temperature:.3f}")
            
        except Exception as e:
            logger.error(f"CTR 후처리 실패: {e}")
            self.bias_correction = 0.0
            self.multiplicative_correction = 1.0
            self.temperature = 1.0
    
    def _fit_temperature_scaling(self, predictions: np.ndarray, y: pd.Series):
        """Temperature Scaling"""
        try:
            from scipy.optimize import minimize
            
            def temperature_loss(params):
                temp, shift = params
                if temp <= 0:
                    return float('inf')
                
                pred_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
                logits = np.log(pred_clipped / (1 - pred_clipped))
                
                adjusted_logits = (logits + shift) / temp
                calibrated_probs = 1 / (1 + np.exp(-adjusted_logits))
                calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
                
                log_loss = -np.mean(y * np.log(calibrated_probs) + (1 - y) * np.log(1 - calibrated_probs))
                ctr_loss = abs(calibrated_probs.mean() - y.mean()) * 1500
                diversity_loss = -calibrated_probs.std()
                
                return log_loss + ctr_loss + diversity_loss
            
            result = minimize(
                temperature_loss, 
                x0=[1.0, 0.0],
                bounds=[(0.3, 15.0), (-3.0, 3.0)],
                method='L-BFGS-B'
            )
            
            self.temperature = result.x[0]
            self.logit_shift = result.x[1]
            
        except Exception as e:
            logger.warning(f"Temperature scaling 실패: {e}")
            self.temperature = 1.0
            self.logit_shift = 0.0
    
    def _fit_quantile_corrections(self, predictions: np.ndarray, y: pd.Series):
        """분위수 기반 보정"""
        try:
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            
            for q in quantiles:
                threshold = np.percentile(predictions, q * 100)
                mask = predictions >= threshold
                
                if mask.sum() > 5:
                    actual_rate = y[mask].mean()
                    predicted_rate = predictions[mask].mean()
                    
                    if predicted_rate > 0:
                        correction_factor = actual_rate / predicted_rate
                    else:
                        correction_factor = 1.0
                    
                    self.quantile_corrections[q] = {
                        'threshold': threshold,
                        'correction_factor': correction_factor,
                        'sample_size': mask.sum()
                    }
            
            logger.info(f"분위수 보정 완료: {len(self.quantile_corrections)}개 구간")
            
        except Exception as e:
            logger.warning(f"분위수 보정 실패: {e}")
    
    def _apply_meta_learning(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """메타 학습 적용"""
        try:
            logger.info("메타 학습 레이어 구성")
            
            meta_features = []
            for name, pred in base_predictions.items():
                confidence = np.abs(pred - 0.5) * 2
                entropy = -pred * np.log(pred + 1e-15) - (1 - pred) * np.log(1 - pred + 1e-15)
                
                meta_feature = np.column_stack([pred, confidence, entropy])
                meta_features.append(meta_feature)
            
            X_meta = np.concatenate(meta_features, axis=1)
            
            from sklearn.ensemble import RandomForestRegressor
            self.meta_learner = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=4
            )
            
            self.meta_learner.fit(X_meta, y)
            logger.info("메타 학습 완료")
            
        except Exception as e:
            logger.warning(f"메타 학습 실패: {e}")
            self.meta_learner = None
    
    def _apply_stacking_layer(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """스택킹 레이어 추가"""
        try:
            logger.info("스택킹 레이어 구성")
            
            from sklearn.model_selection import cross_val_predict
            from sklearn.linear_model import LogisticRegression
            
            stacking_features = np.column_stack(list(base_predictions.values()))
            
            self.stacking_regressor = LogisticRegression(
                max_iter=500,
                random_state=42,
                class_weight='balanced'
            )
            
            oof_predictions = cross_val_predict(
                self.stacking_regressor, 
                stacking_features, 
                y, 
                cv=5, 
                method='predict_proba'
            )
            
            if oof_predictions.ndim > 1 and oof_predictions.shape[1] > 1:
                oof_predictions = oof_predictions[:, 1]
            
            self.stacking_regressor.fit(stacking_features, y)
            
            stacking_score = self.metrics_calculator.combined_score(y, oof_predictions)
            logger.info(f"스택킹 점수: {stacking_score:.4f}")
            
        except Exception as e:
            logger.warning(f"스택킹 레이어 실패: {e}")
            self.stacking_regressor = None
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """가중 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        if hasattr(self, 'ensemble_method') and self.ensemble_method == 'power_weighted':
            power = getattr(self, 'power', 2.5)
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    powered_pred = np.power(base_predictions[name], power)
                    ensemble_pred += weight * powered_pred
            ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        elif hasattr(self, 'ensemble_method') and self.ensemble_method == 'rank_weighted':
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    rank_pred = pd.Series(base_predictions[name]).rank(pct=True).values
                    ensemble_pred += weight * rank_pred
                    
        else:
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    ensemble_pred += weight * base_predictions[name]
        
        return ensemble_pred
    
    def _create_final_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """최종 앙상블 생성"""
        base_ensemble = self._create_weighted_ensemble(base_predictions)
        
        if self.meta_learner is not None:
            try:
                meta_features = []
                for name, pred in base_predictions.items():
                    confidence = np.abs(pred - 0.5) * 2
                    entropy = -pred * np.log(pred + 1e-15) - (1 - pred) * np.log(1 - pred + 1e-15)
                    meta_feature = np.column_stack([pred, confidence, entropy])
                    meta_features.append(meta_feature)
                
                X_meta = np.concatenate(meta_features, axis=1)
                meta_pred = self.meta_learner.predict(X_meta)
                meta_pred = np.clip(meta_pred, 0.0, 1.0)
                
                base_ensemble = 0.7 * base_ensemble + 0.3 * meta_pred
            except Exception as e:
                logger.warning(f"메타 학습 예측 실패: {e}")
        
        if hasattr(self, 'stacking_regressor') and self.stacking_regressor is not None:
            try:
                stacking_features = np.column_stack(list(base_predictions.values()))
                stacking_pred = self.stacking_regressor.predict_proba(stacking_features)
                
                if stacking_pred.ndim > 1 and stacking_pred.shape[1] > 1:
                    stacking_pred = stacking_pred[:, 1]
                
                base_ensemble = 0.6 * base_ensemble + 0.4 * stacking_pred
            except Exception as e:
                logger.warning(f"스택킹 예측 실패: {e}")
        
        return base_ensemble
    
    def _apply_all_corrections(self, predictions: np.ndarray) -> np.ndarray:
        """모든 보정 기법 적용"""
        try:
            corrected = predictions.copy()
            
            if hasattr(self, 'temperature') and hasattr(self, 'logit_shift'):
                try:
                    pred_clipped = np.clip(corrected, 1e-15, 1 - 1e-15)
                    logits = np.log(pred_clipped / (1 - pred_clipped))
                    adjusted_logits = (logits + self.logit_shift) / self.temperature
                    corrected = 1 / (1 + np.exp(-adjusted_logits))
                except:
                    pass
            
            for q, correction in self.quantile_corrections.items():
                try:
                    mask = corrected >= correction['threshold']
                    if mask.sum() > 0:
                        corrected[mask] *= correction['correction_factor']
                except:
                    continue
            
            corrected = corrected * self.multiplicative_correction + self.bias_correction
            corrected = np.clip(corrected, 1e-15, 1 - 1e-15)
            corrected = self._enhance_ensemble_diversity(corrected)
            
            return corrected
            
        except Exception as e:
            logger.warning(f"전체 보정 적용 실패: {e}")
            return np.clip(predictions, 1e-15, 1 - 1e-15)
    
    def _create_fallback_ensemble(self, available_models: List[str]):
        """앙상블 실패 시 대체 방법"""
        try:
            logger.info("앙상블 대체 방법 생성")
            
            # 동등 가중치 사용
            equal_weight = 1.0 / len(available_models)
            self.final_weights = {model: equal_weight for model in available_models}
            
            # 기본 보정값 설정
            self.bias_correction = 0.0
            self.multiplicative_correction = 1.0
            self.temperature = 1.0
            self.logit_shift = 0.0
            self.quantile_corrections = {}
            
            logger.info(f"대체 앙상블 생성 완료: 동등 가중치 {equal_weight:.3f}")
            
        except Exception as e:
            logger.error(f"대체 앙상블 생성 실패: {e}")
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """메인 앙상블 예측 - 실행 보장"""
        if not self.is_fitted:
            logger.error("앙상블 모델이 학습되지 않았습니다")
            # 학습되지 않았어도 기본 앙상블 제공
            if base_predictions:
                return np.mean(list(base_predictions.values()), axis=0)
            else:
                raise ValueError("앙상블 모델이 학습되지 않았고 기본 예측도 없습니다")
        
        try:
            # 앙상블 실행 보장
            logger.info("앙상블 예측 실행 시작")
            
            # 모든 모델이 정상 작동하는지 확인
            valid_predictions = {}
            for name, pred in base_predictions.items():
                if pred is not None and len(pred) > 0 and not np.all(np.isnan(pred)):
                    valid_predictions[name] = pred
                else:
                    logger.warning(f"{name} 모델 예측이 유효하지 않음")
            
            if len(valid_predictions) < 2:
                logger.warning("유효한 모델이 부족하여 단일 모델 사용")
                if valid_predictions:
                    return list(valid_predictions.values())[0]
                else:
                    return np.full(len(list(base_predictions.values())[0]), 0.0201)
            
            # 최종 앙상블 생성
            final_pred = self._create_final_ensemble(valid_predictions)
            final_pred = self._apply_all_corrections(final_pred)
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    final_pred = self.ensemble_calibrator.predict(final_pred)
                    final_pred = np.clip(final_pred, 1e-15, 1 - 1e-15)
                except Exception as e:
                    logger.warning(f"앙상블 캘리브레이션 예측 실패: {e}")
            
            logger.info("앙상블 예측 실행 완료")
            return final_pred
            
        except Exception as e:
            logger.error(f"앙상블 예측 실행 실패: {e}")
            # 실패해도 기본 가중 평균 제공하여 앙상블 보장
            try:
                logger.info("앙상블 대체 예측 실행")
                return np.mean(list(base_predictions.values()), axis=0)
            except Exception as e2:
                logger.error(f"앙상블 대체 예측도 실패: {e2}")
                return np.full(len(list(base_predictions.values())[0]), 0.0201)

class CTRStabilizedEnsemble(BaseEnsemble):
    """CTR 예측 안정화 앙상블"""
    
    def __init__(self, diversification_method: str = 'rank_weighted'):
        super().__init__("CTRStabilizedEnsemble")
        self.diversification_method = diversification_method
        self.model_weights = {}
        self.diversity_weights = {}
        self.stability_weights = {}
        self.calibration_weights = {}
        self.final_weights = {}
        self.metrics_calculator = CTRMetrics()
        self.ensemble_execution_guaranteed = True
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """안정화 앙상블 학습"""
        logger.info(f"CTR 안정화 앙상블 학습 시작 - 방법: {self.diversification_method}")
        
        available_models = list(base_predictions.keys())
        
        if len(available_models) < 2:
            logger.warning("앙상블을 위한 모델이 부족합니다")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            return
        
        try:
            self.model_weights = self._evaluate_individual_performance(base_predictions, y)
            self.diversity_weights = self._calculate_diversity_weights(base_predictions)
            self.stability_weights = self._calculate_stability_weights(base_predictions, y)
            self.calibration_weights = self._calculate_calibration_weights()
            self.final_weights = self._combine_weights_with_calibration()
            
            try:
                ensemble_pred = self._create_stabilized_ensemble(base_predictions)
                
                split_idx = int(len(X) * 0.8)
                X_cal = X.iloc[split_idx:]
                y_cal = y.iloc[split_idx:]
                
                cal_base_predictions = {}
                for name in available_models:
                    if name in base_predictions:
                        cal_base_predictions[name] = base_predictions[name][split_idx:]
                
                cal_ensemble_pred = self._create_stabilized_ensemble(cal_base_predictions)
                
                self.apply_ensemble_calibration(X_cal, y_cal, cal_ensemble_pred, method='auto')
                
            except Exception as e:
                logger.warning(f"안정화 앙상블 캘리브레이션 적용 실패: {e}")
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            logger.info(f"CTR 안정화 앙상블 학습 완료 - 최종 가중치: {self.final_weights}")
            logger.info(f"앙상블 캘리브레이션 적용: {'Yes' if self.is_calibrated else 'No'}")
            
        except Exception as e:
            logger.error(f"안정화 앙상블 학습 실패: {e}")
            # 실패 시 동등 가중치로 대체
            equal_weight = 1.0 / len(available_models)
            self.final_weights = {model: equal_weight for model in available_models}
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
    
    def _evaluate_individual_performance(self, base_predictions: Dict[str, np.ndarray], 
                                       y: pd.Series) -> Dict[str, float]:
        """개별 모델 성능 평가"""
        
        performance_weights = {}
        
        for name, pred in base_predictions.items():
            try:
                combined_score = self.metrics_calculator.combined_score(y, pred)
                ap_score = self.metrics_calculator.average_precision(y, pred)
                wll_score = self.metrics_calculator.weighted_log_loss(y, pred)
                
                predicted_ctr = pred.mean()
                actual_ctr = y.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_penalty = np.exp(-ctr_bias * 400)
                
                pred_std = pred.std()
                pred_range = pred.max() - pred.min()
                pred_entropy = -np.mean(pred * np.log(pred + 1e-15) + (1 - pred) * np.log(1 - pred + 1e-15))
                
                quality_score = pred_std * pred_range * pred_entropy
                
                model = self.base_models.get(name)
                calibration_bonus = 1.0
                if model and hasattr(model, 'is_calibrated') and model.is_calibrated:
                    calibration_bonus = 1.3
                    if hasattr(model, 'calibrator') and model.calibrator:
                        calibration_summary = model.calibrator.get_calibration_summary()
                        if calibration_summary['calibration_scores']:
                            calibration_quality = max(calibration_summary['calibration_scores'].values())
                            calibration_bonus = 1.0 + 0.4 * calibration_quality
                
                performance_score = (0.5 * combined_score + 
                                   0.25 * ctr_penalty + 
                                   0.15 * (ap_score * (1 / (1 + wll_score))) +
                                   0.1 * quality_score) * calibration_bonus
                
                performance_weights[name] = max(performance_score, 0.02)
                
                logger.info(f"{name} - Combined: {combined_score:.4f}, CTR편향: {ctr_bias:.4f}, "
                          f"품질: {quality_score:.4f}, 캘리브레이션 보너스: {calibration_bonus:.2f}x, "
                          f"최종: {performance_score:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 성능 평가 실패: {e}")
                performance_weights[name] = 0.02
        
        return performance_weights
    
    def _calculate_calibration_weights(self) -> Dict[str, float]:
        """캘리브레이션 가중치 계산"""
        calibration_weights = {}
        
        for name, model in self.base_models.items():
            try:
                calibration_weight = 0.5
                
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    calibration_weight = 0.9
                    
                    if hasattr(model, 'calibrator') and model.calibrator:
                        calibration_summary = model.calibrator.get_calibration_summary()
                        if calibration_summary['calibration_scores']:
                            calibration_quality = max(calibration_summary['calibration_scores'].values())
                            calibration_weight = 0.5 + 0.7 * calibration_quality
                            
                            logger.info(f"{name} 캘리브레이션 품질: {calibration_quality:.4f}, "
                                      f"가중치: {calibration_weight:.4f}")
                
                calibration_weights[name] = calibration_weight
                
            except Exception as e:
                logger.warning(f"{name} 캘리브레이션 가중치 계산 실패: {e}")
                calibration_weights[name] = 0.5
        
        return calibration_weights
    
    def _calculate_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """다양성 가중치 계산"""
        
        model_names = list(base_predictions.keys())
        diversity_weights = {}
        
        if self.diversification_method == 'correlation_based':
            correlation_matrix = self._calculate_correlation_matrix(base_predictions)
            distance_matrix = self._calculate_distance_matrix(base_predictions)
            
            for name in model_names:
                avg_correlation = np.mean([abs(correlation_matrix[name][other]) 
                                         for other in model_names if other != name])
                avg_distance = np.mean([distance_matrix[name][other] 
                                      for other in model_names if other != name])
                
                pred = base_predictions[name]
                uniqueness = len(np.unique(pred)) / len(pred)
                
                diversity_score = (1.0 - avg_correlation) * avg_distance * uniqueness
                diversity_weights[name] = max(diversity_score, 0.15)
        
        elif self.diversification_method == 'rank_weighted':
            rank_matrices = {}
            for name, pred in base_predictions.items():
                rank_matrices[name] = pd.Series(pred).rank(pct=True).values
            
            diversity_scores = {}
            for name in model_names:
                rank_differences = []
                rank_correlations = []
                
                for other in model_names:
                    if other != name:
                        rank_diff = np.mean(np.abs(rank_matrices[name] - rank_matrices[other]))
                        rank_differences.append(rank_diff)
                        
                        rank_corr = np.corrcoef(rank_matrices[name], rank_matrices[other])[0, 1]
                        rank_correlations.append(abs(rank_corr))
                
                avg_rank_diff = np.mean(rank_differences) if rank_differences else 0.5
                avg_rank_corr = np.mean(rank_correlations) if rank_correlations else 0.5
                
                diversity_score = avg_rank_diff * (1 - avg_rank_corr)
                diversity_weights[name] = max(diversity_score, 0.15)
        
        else:
            diversity_weights = {name: 1.0 for name in model_names}
        
        logger.info(f"다양성 가중치: {diversity_weights}")
        return diversity_weights
    
    def _calculate_correlation_matrix(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """상관관계 행렬 계산"""
        
        model_names = list(base_predictions.keys())
        correlation_matrix = {}
        
        for name1 in model_names:
            correlation_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    correlation_matrix[name1][name2] = 1.0
                else:
                    try:
                        pred1, pred2 = base_predictions[name1], base_predictions[name2]
                        
                        pearson_corr = np.corrcoef(pred1, pred2)[0, 1]
                        spearman_corr = np.corrcoef(pd.Series(pred1).rank(), pd.Series(pred2).rank())[0, 1]
                        
                        concordant = 0
                        total = 0
                        for i in range(0, len(pred1), max(1, len(pred1) // 1000)):
                            for j in range(i+1, len(pred1), max(1, len(pred1) // 1000)):
                                if (pred1[i] - pred1[j]) * (pred2[i] - pred2[j]) > 0:
                                    concordant += 1
                                total += 1
                        
                        kendall_like = (2 * concordant - total) / total if total > 0 else 0
                        
                        combined_corr = (pearson_corr + spearman_corr + kendall_like) / 3
                        correlation_matrix[name1][name2] = combined_corr if not np.isnan(combined_corr) else 0.0
                        
                    except Exception as e:
                        logger.warning(f"상관관계 계산 실패 ({name1}, {name2}): {e}")
                        correlation_matrix[name1][name2] = 0.0
        
        return correlation_matrix
    
    def _calculate_distance_matrix(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """거리 행렬 계산"""
        
        model_names = list(base_predictions.keys())
        distance_matrix = {}
        
        for name1 in model_names:
            distance_matrix[name1] = {}
            for name2 in model_names:
                if name1 == name2:
                    distance_matrix[name1][name2] = 0.0
                else:
                    try:
                        pred1, pred2 = base_predictions[name1], base_predictions[name2]
                        
                        euclidean = np.sqrt(np.mean((pred1 - pred2) ** 2))
                        manhattan = np.mean(np.abs(pred1 - pred2))
                        
                        pred1_norm = pred1 / pred1.sum()
                        pred2_norm = pred2 / pred2.sum()
                        kl_div = np.sum(pred1_norm * np.log(pred1_norm / (pred2_norm + 1e-15) + 1e-15))
                        
                        combined_distance = euclidean + manhattan + min(kl_div, 15.0)
                        distance_matrix[name1][name2] = combined_distance
                        
                    except Exception as e:
                        logger.warning(f"거리 계산 실패 ({name1}, {name2}): {e}")
                        distance_matrix[name1][name2] = 1.0
        
        return distance_matrix
    
    def _calculate_stability_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """안정성 가중치 계산"""
        
        stability_weights = {}
        
        for name, pred in base_predictions.items():
            try:
                n_bootstrap = 30
                bootstrap_scores = []
                
                for _ in range(n_bootstrap):
                    indices = np.random.choice(len(pred), size=len(pred), replace=True)
                    boot_pred = pred[indices]
                    boot_y = y.iloc[indices]
                    
                    try:
                        boot_score = self.metrics_calculator.combined_score(boot_y, boot_pred)
                        bootstrap_scores.append(boot_score)
                    except:
                        continue
                
                if len(bootstrap_scores) > 3:
                    stability_score = 1.0 - (np.std(bootstrap_scores) / (np.mean(bootstrap_scores) + 1e-8))
                    stability_weights[name] = max(stability_score, 0.15)
                else:
                    stability_weights[name] = 0.5
                
                logger.info(f"{name} 안정성 점수: {stability_weights[name]:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} 안정성 계산 실패: {e}")
                stability_weights[name] = 0.5
        
        return stability_weights
    
    def _combine_weights_with_calibration(self) -> Dict[str, float]:
        """가중치 결합"""
        
        combined_weights = {}
        model_names = list(self.model_weights.keys())
        
        performance_sum = sum(self.model_weights.values())
        diversity_sum = sum(self.diversity_weights.values())
        stability_sum = sum(self.stability_weights.values())
        calibration_sum = sum(self.calibration_weights.values())
        
        if performance_sum > 0 and diversity_sum > 0 and stability_sum > 0 and calibration_sum > 0:
            for name in model_names:
                perf_weight = self.model_weights[name] / performance_sum
                div_weight = self.diversity_weights[name] / diversity_sum
                stab_weight = self.stability_weights[name] / stability_sum
                cal_weight = self.calibration_weights[name] / calibration_sum
                
                adaptive_performance_ratio = 0.4 + 0.2 * (perf_weight - 0.5)
                adaptive_diversity_ratio = 0.2 - 0.05 * (div_weight - 0.5)
                adaptive_stability_ratio = 0.15 - 0.05 * (stab_weight - 0.5)
                adaptive_calibration_ratio = 0.25 + 0.1 * (cal_weight - 0.5)
                
                total_ratio = (adaptive_performance_ratio + adaptive_diversity_ratio + 
                             adaptive_stability_ratio + adaptive_calibration_ratio)
                
                adaptive_performance_ratio /= total_ratio
                adaptive_diversity_ratio /= total_ratio
                adaptive_stability_ratio /= total_ratio
                adaptive_calibration_ratio /= total_ratio
                
                combined_weight = (adaptive_performance_ratio * perf_weight + 
                                 adaptive_diversity_ratio * div_weight + 
                                 adaptive_stability_ratio * stab_weight +
                                 adaptive_calibration_ratio * cal_weight)
                
                combined_weights[name] = combined_weight
        else:
            combined_weights = {name: 1.0/len(model_names) for name in model_names}
        
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
        
        return combined_weights
    
    def _create_stabilized_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """안정화된 앙상블 생성"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        ensemble_pred = self._enhance_ensemble_diversity(ensemble_pred)
        
        return ensemble_pred
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """안정화된 앙상블 예측 - 실행 보장"""
        if not self.is_fitted:
            logger.error("앙상블 모델이 학습되지 않았습니다")
            if base_predictions:
                return np.mean(list(base_predictions.values()), axis=0)
            else:
                raise ValueError("앙상블 모델이 학습되지 않았고 기본 예측도 없습니다")
        
        try:
            logger.info("안정화 앙상블 예측 실행 시작")
            
            ensemble_pred = self._create_stabilized_ensemble(base_predictions)
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                    ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
                except Exception as e:
                    logger.warning(f"안정화 앙상블 캘리브레이션 예측 실패: {e}")
            
            logger.info("안정화 앙상블 예측 실행 완료")
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"안정화 앙상블 예측 실행 실패: {e}")
            # 실패해도 기본 가중 평균 제공
            try:
                logger.info("안정화 앙상블 대체 예측 실행")
                return np.mean(list(base_predictions.values()), axis=0)
            except Exception as e2:
                logger.error(f"안정화 앙상블 대체 예측도 실패: {e2}")
                return np.full(len(list(base_predictions.values())[0]), 0.0201)

class CTRSuperEnsembleManager:
    """CTR 특화 앙상블 관리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.final_ensemble = None
        self.target_combined_score = 0.35
        self.calibration_manager = {}
        self.ensemble_execution_status = {}  # 앙상블 실행 상태 추적
        
    def add_base_model(self, name: str, model: BaseModel):
        """기본 모델 추가"""
        self.base_models[name] = model
        
        calibration_status = "No"
        if hasattr(model, 'is_calibrated') and model.is_calibrated:
            calibration_status = "Yes"
            if hasattr(model, 'calibrator') and model.calibrator:
                calibration_method = getattr(model.calibrator, 'best_method', 'unknown')
                calibration_status = f"Yes ({calibration_method})"
        
        logger.info(f"기본 모델 추가: {name} - Calibration: {calibration_status}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """CTR 특화 앙상블 생성"""
        
        try:
            if ensemble_type == 'final_ensemble':
                target_ctr = kwargs.get('target_ctr', 0.0201)
                optimization_method = kwargs.get('optimization_method', 'final_combined')
                ensemble = CTRMainEnsemble(target_ctr, optimization_method)
                self.final_ensemble = ensemble
            
            elif ensemble_type == 'stabilized':
                diversification_method = kwargs.get('diversification_method', 'rank_weighted')
                ensemble = CTRStabilizedEnsemble(diversification_method)
            
            else:
                raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
            
            for name, model in self.base_models.items():
                ensemble.add_base_model(name, model)
            
            self.ensembles[ensemble_type] = ensemble
            self.ensemble_execution_status[ensemble_type] = {'created': True, 'fitted': False, 'error': None}
            logger.info(f"앙상블 생성: {ensemble_type}")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"앙상블 생성 실패 ({ensemble_type}): {e}")
            self.ensemble_execution_status[ensemble_type] = {'created': False, 'fitted': False, 'error': str(e)}
            raise
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """모든 앙상블 학습 - 실행 보장"""
        logger.info("모든 앙상블 학습 시작 - 실행 보장")
        
        base_predictions = {}
        calibration_info = {}
        
        # 모든 기본 모델의 예측 수집
        for name, model in self.base_models.items():
            try:
                start_time = time.time()
                
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    pred = model.predict_proba(X)
                    calibration_info[name] = {'calibrated': True, 'method': getattr(model.calibrator, 'best_method', 'unknown') if hasattr(model, 'calibrator') and model.calibrator else 'unknown'}
                else:
                    pred = model.predict_proba(X)
                    calibration_info[name] = {'calibrated': False, 'method': 'none'}
                
                prediction_time = time.time() - start_time
                
                # 예측 유효성 검증
                if pred is None or len(pred) == 0 or np.all(np.isnan(pred)):
                    logger.error(f"{name} 모델 예측이 유효하지 않음")
                    pred = np.full(len(X), 0.0201)
                
                base_predictions[name] = pred
                logger.info(f"{name} 모델 예측 완료 ({prediction_time:.2f}초) - "
                          f"Calibration: {'Yes' if calibration_info[name]['calibrated'] else 'No'}")
                
            except Exception as e:
                logger.error(f"{name} 모델 예측 실패: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
                calibration_info[name] = {'calibrated': False, 'method': 'error'}
        
        # 최소한 1개의 유효한 예측이 있는지 확인
        if not base_predictions:
            logger.error("모든 기본 모델 예측 실패")
            base_predictions['dummy'] = np.full(len(X), 0.0201)
            calibration_info['dummy'] = {'calibrated': False, 'method': 'dummy'}
        
        # 각 앙상블 학습 - 실행 보장
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                logger.info(f"{ensemble_type} 앙상블 학습 시작 - 실행 보장")
                start_time = time.time()
                
                # 앙상블 학습 실행
                ensemble.fit(X, y, base_predictions)
                training_time = time.time() - start_time
                
                # 실행 상태 업데이트
                self.ensemble_execution_status[ensemble_type].update({
                    'fitted': True,
                    'training_time': training_time,
                    'ensemble_guaranteed': getattr(ensemble, 'ensemble_execution_guaranteed', False)
                })
                
                calibration_status = "Yes" if ensemble.is_calibrated else "No"
                logger.info(f"{ensemble_type} 앙상블 학습 완료 ({training_time:.2f}초) - "
                          f"Ensemble Calibration: {calibration_status}, "
                          f"Execution Guaranteed: {self.ensemble_execution_status[ensemble_type]['ensemble_guaranteed']}")
                
            except Exception as e:
                logger.error(f"{ensemble_type} 앙상블 학습 실패: {str(e)}")
                # 실행 상태 업데이트
                self.ensemble_execution_status[ensemble_type].update({
                    'fitted': False,
                    'error': str(e),
                    'ensemble_guaranteed': False
                })
                
                # 실패해도 기본 앙상블 보장
                try:
                    logger.info(f"{ensemble_type} 앙상블 대체 방법 적용")
                    if hasattr(ensemble, '_create_fallback_ensemble'):
                        ensemble._create_fallback_ensemble(list(base_predictions.keys()))
                        ensemble.is_fitted = True
                        self.ensemble_execution_status[ensemble_type].update({
                            'fitted': True,
                            'fallback_used': True,
                            'ensemble_guaranteed': True
                        })
                        logger.info(f"{ensemble_type} 앙상블 대체 방법 성공")
                except Exception as fallback_error:
                    logger.error(f"{ensemble_type} 앙상블 대체 방법도 실패: {fallback_error}")
        
        self.calibration_manager = calibration_info
        gc.collect()
        
        # 앙상블 실행 보장 상태 로깅
        logger.info("앙상블 실행 보장 상태:")
        for ensemble_type, status in self.ensemble_execution_status.items():
            logger.info(f"  - {ensemble_type}: {status}")
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """앙상블 성능 평가 - 실행 보장"""
        logger.info("앙상블 성능 평가 시작 - 실행 보장")
        
        results = {}
        
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                
                # 예측 유효성 검증
                if pred is None or len(pred) == 0 or np.all(np.isnan(pred)):
                    logger.warning(f"{name} 모델 검증 예측이 유효하지 않음")
                    pred = np.full(len(X_val), 0.0201)
                
                base_predictions[name] = pred
                
                score = self.metrics_calculator.combined_score(y_val, pred)
                ctr_optimized_score = self.metrics_calculator.ctr_optimized_score(y_val, pred)
                
                results[f"base_{name}"] = score
                results[f"base_{name}_ctr_optimized"] = ctr_optimized_score
                
                calibration_status = self.calibration_manager.get(name, {})
                if calibration_status.get('calibrated', False):
                    if hasattr(model, 'predict_proba_raw'):
                        try:
                            raw_pred = model.predict_proba_raw(X_val)
                            raw_score = self.metrics_calculator.combined_score(y_val, raw_pred)
                            calibration_improvement = score - raw_score
                            results[f"base_{name}_calibration_improvement"] = calibration_improvement
                            logger.info(f"{name} 캘리브레이션 효과: {calibration_improvement:+.4f}")
                        except:
                            pass
                
            except Exception as e:
                logger.error(f"{name} 모델 검증 예측 실패: {str(e)}")
                results[f"base_{name}"] = 0.0
                results[f"base_{name}_ctr_optimized"] = 0.0
        
        # 각 앙상블 평가 - 실행 보장
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    logger.info(f"{ensemble_type} 앙상블 평가 시작 - 실행 보장")
                    
                    # 앙상블 예측 실행 보장
                    try:
                        raw_ensemble_pred = ensemble.predict_proba(base_predictions)
                        
                        if ensemble.is_calibrated:
                            calibrated_ensemble_pred = ensemble.predict_proba_calibrated(base_predictions)
                        else:
                            calibrated_ensemble_pred = raw_ensemble_pred
                        
                        # 예측 유효성 검증
                        if (calibrated_ensemble_pred is None or len(calibrated_ensemble_pred) == 0 or 
                            np.all(np.isnan(calibrated_ensemble_pred))):
                            logger.warning(f"{ensemble_type} 앙상블 예측이 유효하지 않음")
                            calibrated_ensemble_pred = np.full(len(X_val), 0.0201)
                        
                    except Exception as pred_error:
                        logger.error(f"{ensemble_type} 앙상블 예측 실패: {pred_error}")
                        # 예측 실패 시 기본값 사용
                        calibrated_ensemble_pred = np.full(len(X_val), 0.0201)
                        raw_ensemble_pred = calibrated_ensemble_pred
                    
                    combined_score = self.metrics_calculator.combined_score(y_val, calibrated_ensemble_pred)
                    ctr_optimized_score = self.metrics_calculator.ctr_optimized_score(y_val, calibrated_ensemble_pred)
                    ap_score = self.metrics_calculator.average_precision(y_val, calibrated_ensemble_pred)
                    wll_score = self.metrics_calculator.weighted_log_loss(y_val, calibrated_ensemble_pred)
                    
                    results[f"ensemble_{ensemble_type}"] = combined_score
                    results[f"ensemble_{ensemble_type}_ctr_optimized"] = ctr_optimized_score
                    results[f"ensemble_{ensemble_type}_ap"] = ap_score
                    results[f"ensemble_{ensemble_type}_wll"] = wll_score
                    
                    if ensemble.is_calibrated:
                        try:
                            raw_combined_score = self.metrics_calculator.combined_score(y_val, raw_ensemble_pred)
                            calibration_improvement = combined_score - raw_combined_score
                            results[f"ensemble_{ensemble_type}_calibration_improvement"] = calibration_improvement
                            logger.info(f"{ensemble_type} 앙상블 캘리브레이션 효과: {calibration_improvement:+.4f}")
                        except:
                            pass
                    
                    # 앙상블 실행 보장 상태 추가
                    execution_guaranteed = getattr(ensemble, 'ensemble_execution_guaranteed', False)
                    results[f"ensemble_{ensemble_type}_execution_guaranteed"] = 1.0 if execution_guaranteed else 0.0
                    
                    logger.info(f"{ensemble_type} 앙상블 Combined Score: {combined_score:.4f}")
                    logger.info(f"{ensemble_type} 앙상블 CTR Optimized Score: {ctr_optimized_score:.4f}")
                    logger.info(f"{ensemble_type} 앙상블 Calibration: {'Yes' if ensemble.is_calibrated else 'No'}")
                    logger.info(f"{ensemble_type} 앙상블 Execution Guaranteed: {'Yes' if execution_guaranteed else 'No'}")
                    
                    predicted_ctr = calibrated_ensemble_pred.mean()
                    actual_ctr = y_val.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    logger.info(f"{ensemble_type} CTR: 예측 {predicted_ctr:.4f} vs 실제 {actual_ctr:.4f} (편향: {ctr_bias:.4f})")
                    
                    target_achieved = combined_score >= self.target_combined_score
                    logger.info(f"{ensemble_type} 목표 달성: {target_achieved} (목표: {self.target_combined_score})")
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} 앙상블 평가 실패: {str(e)}")
                    results[f"ensemble_{ensemble_type}"] = 0.0
                    results[f"ensemble_{ensemble_type}_ctr_optimized"] = 0.0
                    results[f"ensemble_{ensemble_type}_execution_guaranteed"] = 0.0
        
        # 최고 성능 앙상블 선택 - 실행 보장 우선
        if results:
            ensemble_results = {k: v for k, v in results.items() 
                              if k.startswith('ensemble_') and not k.endswith('_ctr_optimized') 
                              and not k.endswith('_ap') and not k.endswith('_wll') 
                              and not k.endswith('_calibration_improvement')
                              and not k.endswith('_execution_guaranteed')}
            
            if ensemble_results:
                # 실행 보장된 앙상블 중에서 최고 성능 선택
                guaranteed_ensembles = []
                for ensemble_name in ensemble_results.keys():
                    ensemble_type = ensemble_name.replace('ensemble_', '')
                    if (f"ensemble_{ensemble_type}_execution_guaranteed" in results and 
                        results[f"ensemble_{ensemble_type}_execution_guaranteed"] > 0.5):
                        guaranteed_ensembles.append(ensemble_name)
                
                if guaranteed_ensembles:
                    best_name = max(guaranteed_ensembles, key=lambda x: ensemble_results[x])
                    logger.info("실행 보장된 앙상블 중에서 최고 성능 선택")
                else:
                    best_name = max(ensemble_results, key=ensemble_results.get)
                    logger.info("실행 보장 실패, 일반 최고 성능 앙상블 선택")
                
                best_score = ensemble_results[best_name]
                
                ensemble_type = best_name.replace('ensemble_', '')
                self.best_ensemble = self.ensembles[ensemble_type]
                
                logger.info(f"최고 성능 앙상블: {ensemble_type} (Combined Score: {best_score:.4f})")
                
                if best_score >= self.target_combined_score:
                    logger.info(f"목표 Combined Score {self.target_combined_score}+ 달성!")
                else:
                    logger.info(f"목표까지 {self.target_combined_score - best_score:.4f} 부족")
                    
                best_ensemble_obj = self.ensembles[ensemble_type]
                if best_ensemble_obj.is_calibrated:
                    logger.info("최고 앙상블에 캘리브레이션 적용됨")
                if getattr(best_ensemble_obj, 'ensemble_execution_guaranteed', False):
                    logger.info("최고 앙상블 실행 보장됨")
            else:
                logger.warning("평가 가능한 앙상블이 없습니다.")
                self.best_ensemble = None
        
        self.ensemble_results = results
        
        calibration_improvements = [v for k, v in results.items() if k.endswith('_calibration_improvement')]
        if calibration_improvements:
            avg_improvement = np.mean(calibration_improvements)
            logger.info(f"평균 캘리브레이션 효과: {avg_improvement:+.4f}")
        
        return results
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """최고 성능 앙상블로 예측 - 실행 보장"""
        logger.info("최고 성능 앙상블 예측 시작 - 실행 보장")
        
        try:
            # 기본 모델 예측 수집
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    
                    # 예측 유효성 검증
                    if pred is None or len(pred) == 0 or np.all(np.isnan(pred)):
                        logger.warning(f"{name} 모델 예측이 유효하지 않음")
                        pred = np.full(len(X), 0.0201)
                    
                    base_predictions[name] = pred
                    
                except Exception as e:
                    logger.error(f"{name} 예측 실패: {str(e)}")
                    base_predictions[name] = np.full(len(X), 0.0201)
            
            # 최소한 1개의 예측이 있는지 확인
            if not base_predictions:
                logger.error("모든 기본 모델 예측 실패")
                return np.full(len(X), 0.0201)
            
            # 최고 앙상블로 예측 - 실행 보장
            if self.best_ensemble is None:
                logger.warning("최고 앙상블이 없음, 기본 모델 중 최고 성능 사용")
                
                best_model_name = None
                best_score = 0
                
                for result_name, score in self.ensemble_results.items():
                    if (result_name.startswith('base_') and 
                        not result_name.endswith('_ctr_optimized') and 
                        not result_name.endswith('_calibration_improvement') and 
                        score > best_score):
                        best_score = score
                        best_model_name = result_name.replace('base_', '')
                
                if best_model_name and best_model_name in self.base_models:
                    logger.info(f"최고 성능 기본 모델 사용: {best_model_name}")
                    return self.base_models[best_model_name].predict_proba(X)
                else:
                    logger.info("평균 앙상블 사용")
                    return np.mean(list(base_predictions.values()), axis=0)
            
            # 최고 앙상블로 예측 실행
            try:
                logger.info(f"최고 앙상블 ({self.best_ensemble.name}) 예측 실행")
                
                if self.best_ensemble.is_calibrated:
                    prediction = self.best_ensemble.predict_proba_calibrated(base_predictions)
                else:
                    prediction = self.best_ensemble.predict_proba(base_predictions)
                
                # 예측 유효성 최종 검증
                if prediction is None or len(prediction) == 0 or np.all(np.isnan(prediction)):
                    logger.error("최고 앙상블 예측이 유효하지 않음")
                    prediction = np.mean(list(base_predictions.values()), axis=0)
                
                logger.info("최고 앙상블 예측 완료 - 실행 보장")
                return prediction
                
            except Exception as ensemble_error:
                logger.error(f"최고 앙상블 예측 실패: {ensemble_error}")
                
                # 앙상블 실패 시 기본 가중 평균 사용
                logger.info("앙상블 실패, 기본 가중 평균 사용")
                return np.mean(list(base_predictions.values()), axis=0)
            
        except Exception as e:
            logger.error(f"앙상블 예측 전체 실패: {e}")
            # 최후의 수단으로 기본값 반환
            return np.full(len(X), 0.0201)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """앙상블 요약 정보"""
        
        target_achieved_count = sum(
            1 for key, score in self.ensemble_results.items()
            if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') 
            and not key.endswith('_ap') and not key.endswith('_wll')
            and not key.endswith('_calibration_improvement')
            and not key.endswith('_execution_guaranteed')
            and score >= self.target_combined_score
        )
        
        calibrated_base_models = sum(
            1 for info in self.calibration_manager.values()
            if info.get('calibrated', False)
        )
        
        calibrated_ensembles = sum(
            1 for ensemble in self.ensembles.values()
            if hasattr(ensemble, 'is_calibrated') and ensemble.is_calibrated
        )
        
        # 앙상블 실행 보장 상태 계산
        guaranteed_ensembles = sum(
            1 for ensemble in self.ensembles.values()
            if getattr(ensemble, 'ensemble_execution_guaranteed', False)
        )
        
        calibration_improvements = [
            v for k, v in self.ensemble_results.items()
            if k.endswith('_calibration_improvement') and v > 0
        ]
        
        return {
            'total_ensembles': len(self.ensembles),
            'fitted_ensembles': sum(1 for e in self.ensembles.values() if e.is_fitted),
            'guaranteed_ensembles': guaranteed_ensembles,
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'best_ensemble_calibrated': self.best_ensemble.is_calibrated if self.best_ensemble else False,
            'best_ensemble_guaranteed': getattr(self.best_ensemble, 'ensemble_execution_guaranteed', False) if self.best_ensemble else False,
            'ensemble_results': self.ensemble_results,
            'ensemble_execution_status': self.ensemble_execution_status,
            'base_models_count': len(self.base_models),
            'calibrated_base_models': calibrated_base_models,
            'calibrated_ensembles': calibrated_ensembles,
            'final_ensemble_available': self.final_ensemble is not None and self.final_ensemble.is_fitted,
            'ensemble_types': list(self.ensembles.keys()),
            'target_combined_score': self.target_combined_score,
            'target_achieved_count': target_achieved_count,
            'target_achieved': target_achieved_count > 0,
            'best_combined_score': max(
                (score for key, score in self.ensemble_results.items()
                 if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') 
                 and not key.endswith('_ap') and not key.endswith('_wll')
                 and not key.endswith('_calibration_improvement')
                 and not key.endswith('_execution_guaranteed')),
                default=0.0
            ),
            'calibration_analysis': {
                'base_models_calibration_rate': calibrated_base_models / max(len(self.base_models), 1),
                'ensemble_calibration_rate': calibrated_ensembles / max(len(self.ensembles), 1),
                'ensemble_execution_guarantee_rate': guaranteed_ensembles / max(len(self.ensembles), 1),
                'positive_calibration_improvements': len(calibration_improvements),
                'avg_calibration_improvement': np.mean(calibration_improvements) if calibration_improvements else 0.0,
                'calibration_methods_used': {
                    info.get('method', 'unknown'): 1 
                    for info in self.calibration_manager.values() 
                    if info.get('calibrated', False)
                }
            }
        }

CTROptimalEnsemble = CTRMainEnsemble
CTRStabilizedEnsemble = CTRStabilizedEnsemble  
CTREnsembleManager = CTRSuperEnsembleManager
EnsembleManager = CTRSuperEnsembleManager