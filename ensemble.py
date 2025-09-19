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
    logging.warning("Optuna not installed. Hyperparameter tuning functionality will be disabled.")

from config import Config
from models import BaseModel, CTRCalibrator
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

class BaseEnsemble(ABC):
    """Base ensemble model class"""
    
    def __init__(self, name: str):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.target_combined_score = 0.35
        self.ensemble_calibrator = None
        self.is_calibrated = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Train ensemble model"""
        pass
    
    @abstractmethod
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Ensemble prediction"""
        pass
    
    def add_base_model(self, name: str, model: BaseModel):
        """Add base model"""
        self.base_models[name] = model
    
    def get_base_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Collect predictions from all base models"""
        predictions = {}
        
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    pred = model.predict_proba(X)
                    logger.info(f"{name} model: Using calibrated predictions")
                else:
                    pred = model.predict_proba(X)
                    
                predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} model prediction failed: {str(e)}")
                predictions[name] = np.full(len(X), 0.0201)
        
        return predictions
    
    def apply_ensemble_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, 
                                 ensemble_predictions: np.ndarray, method: str = 'auto'):
        """Apply ensemble-level calibration"""
        try:
            logger.info(f"Applying calibration to {self.name} ensemble: {method}")
            
            if len(ensemble_predictions) != len(y_val):
                logger.warning("Ensemble calibration: Size mismatch")
                return
            
            self.ensemble_calibrator = CTRCalibrator(target_ctr=0.0201, method=method)
            self.ensemble_calibrator.fit(y_val.values, ensemble_predictions)
            
            self.is_calibrated = True
            
            calibrated_predictions = self.ensemble_calibrator.predict(ensemble_predictions)
            
            original_ctr = ensemble_predictions.mean()
            calibrated_ctr = calibrated_predictions.mean()
            actual_ctr = y_val.mean()
            
            logger.info(f"Ensemble calibration results:")
            logger.info(f"  - Original CTR: {original_ctr:.4f}")
            logger.info(f"  - Calibrated CTR: {calibrated_ctr:.4f}")
            logger.info(f"  - Actual CTR: {actual_ctr:.4f}")
            logger.info(f"  - Best method: {self.ensemble_calibrator.best_method}")
            
        except Exception as e:
            logger.error(f"Ensemble calibration application failed: {e}")
            self.is_calibrated = False
    
    def predict_proba_calibrated(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calibrated ensemble prediction"""
        raw_prediction = self.predict_proba(base_predictions)
        
        if self.is_calibrated and self.ensemble_calibrator is not None:
            try:
                calibrated_prediction = self.ensemble_calibrator.predict(raw_prediction)
                return np.clip(calibrated_prediction, 1e-15, 1 - 1e-15)
            except Exception as e:
                logger.warning(f"Ensemble calibration prediction failed: {e}")
        
        return raw_prediction
    
    def _enhance_ensemble_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """Enhance ensemble prediction diversity"""
        try:
            unique_count = len(np.unique(predictions))
            
            if unique_count < len(predictions) // 100:
                logger.info(f"{self.name}: Applying ensemble prediction diversity enhancement")
                
                noise_scale = max(predictions.std() * 0.005, 1e-7)
                noise = np.random.normal(0, noise_scale, len(predictions))
                
                predictions = predictions + noise
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Ensemble diversity enhancement failed: {e}")
            return predictions

class CTRMainEnsemble(BaseEnsemble):
    """CTR prediction main ensemble"""
    
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
        self.ensemble_execution_guaranteed = True  # Ensemble execution guarantee flag
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Main ensemble training"""
        logger.info(f"CTR main ensemble training started - Target: Combined Score 0.35+")
        
        available_models = list(base_predictions.keys())
        logger.info(f"Available models: {available_models}")
        
        if len(available_models) < 2:
            logger.warning("Insufficient models for ensemble")
            if available_models:
                self.final_weights = {available_models[0]: 1.0}
            self.is_fitted = True
            return
        
        try:
            # Stage 1: Calculate base weights
            logger.info("Stage 1: Calculating base weights")
            self.final_weights = self._calculate_base_weights(base_predictions, y)
            
            # Stage 2: Create weighted ensemble
            ensemble_pred = self._create_weighted_ensemble(base_predictions)
            
            # Stage 3: CTR post-processing
            logger.info("Stage 3: CTR post-processing")
            self._apply_ctr_postprocessing(ensemble_pred, y)
            
            # Stage 4: Apply meta learning
            logger.info("Stage 4: Applying meta learning")
            self._apply_meta_learning(base_predictions, y)
            
            # Stage 5: Add stacking layer
            logger.info("Stage 5: Adding stacking layer")
            self._apply_stacking_layer(base_predictions, y)
            
            # Stage 6: Apply ensemble calibration
            logger.info("Stage 6: Applying ensemble calibration")
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
                logger.warning(f"Ensemble calibration application failed: {e}")
            
            # Stage 7: Final validation and adjustment
            logger.info("Stage 7: Final validation and adjustment")
            final_pred = self._apply_all_corrections(ensemble_pred)
            
            if self.is_calibrated and self.ensemble_calibrator:
                try:
                    final_pred = self.ensemble_calibrator.predict(final_pred)
                except:
                    pass
            
            final_score = self.metrics_calculator.combined_score(y, final_pred)
            
            logger.info(f"Main ensemble Combined Score: {final_score:.4f}")
            logger.info(f"Ensemble calibration applied: {'Yes' if self.is_calibrated else 'No'}")
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            logger.info("CTR main ensemble training completed")
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            # Guarantee ensemble even on failure with default weights
            self._create_fallback_ensemble(available_models)
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
    
    def _calculate_base_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Calculate base weights"""
        
        model_names = list(base_predictions.keys())
        
        # Layer 1: Individual model performance evaluation
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
        
        # Layer 2: Weight calculation
        if OPTUNA_AVAILABLE:
            try:
                optimized_weights = self._optuna_weight_optimization(
                    base_predictions, y, individual_scores, calibration_scores
                )
            except Exception as e:
                logger.warning(f"Optuna weight tuning failed: {e}")
                optimized_weights = self._fallback_weight_calculation(
                    individual_scores, ctr_alignment_scores, diversity_scores, calibration_scores
                )
        else:
            optimized_weights = self._fallback_weight_calculation(
                individual_scores, ctr_alignment_scores, diversity_scores, calibration_scores
            )
        
        logger.info(f"Base weights: {optimized_weights}")
        return optimized_weights
    
    def _optuna_weight_optimization(self, base_predictions: Dict[str, np.ndarray], 
                                   y: pd.Series, 
                                   individual_scores: Dict[str, float],
                                   calibration_scores: Dict[str, float]) -> Dict[str, float]:
        """Optuna weight tuning"""
        
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
            
            # Apply temperature
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
        
        logger.info(f"Optuna weight tuning completed - Best score: {study.best_value:.4f}")
        logger.info(f"Ensemble method: {self.ensemble_method}, Temperature: {self.temperature:.3f}")
        
        return optimized_weights
    
    def _fallback_weight_calculation(self, individual_scores: Dict[str, float], 
                                   ctr_alignment_scores: Dict[str, float],
                                   diversity_scores: Dict[str, float],
                                   calibration_scores: Dict[str, float]) -> Dict[str, float]:
        """Fallback weight calculation"""
        
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
        """CTR post-processing"""
        logger.info("CTR post-processing started")
        
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
            
            logger.info(f"CTR post-processing completed")
            logger.info(f"Bias correction: {self.bias_correction:.4f}, Multiplicative correction: {self.multiplicative_correction:.4f}")
            logger.info(f"Temperature: {self.temperature:.3f}")
            
        except Exception as e:
            logger.error(f"CTR post-processing failed: {e}")
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
            logger.warning(f"Temperature scaling failed: {e}")
            self.temperature = 1.0
            self.logit_shift = 0.0
    
    def _fit_quantile_corrections(self, predictions: np.ndarray, y: pd.Series):
        """Quantile-based corrections"""
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
            
            logger.info(f"Quantile correction completed: {len(self.quantile_corrections)} segments")
            
        except Exception as e:
            logger.warning(f"Quantile correction failed: {e}")
    
    def _apply_meta_learning(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """Apply meta learning"""
        try:
            logger.info("Configuring meta learning layer")
            
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
            logger.info("Meta learning completed")
            
        except Exception as e:
            logger.warning(f"Meta learning failed: {e}")
            self.meta_learner = None
    
    def _apply_stacking_layer(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """Add stacking layer"""
        try:
            logger.info("Configuring stacking layer")
            
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
            logger.info(f"Stacking score: {stacking_score:.4f}")
            
        except Exception as e:
            logger.warning(f"Stacking layer failed: {e}")
            self.stacking_regressor = None
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create weighted ensemble"""
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
        """Create final ensemble"""
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
                logger.warning(f"Meta learning prediction failed: {e}")
        
        if hasattr(self, 'stacking_regressor') and self.stacking_regressor is not None:
            try:
                stacking_features = np.column_stack(list(base_predictions.values()))
                stacking_pred = self.stacking_regressor.predict_proba(stacking_features)
                
                if stacking_pred.ndim > 1 and stacking_pred.shape[1] > 1:
                    stacking_pred = stacking_pred[:, 1]
                
                base_ensemble = 0.6 * base_ensemble + 0.4 * stacking_pred
            except Exception as e:
                logger.warning(f"Stacking prediction failed: {e}")
        
        return base_ensemble
    
    def _apply_all_corrections(self, predictions: np.ndarray) -> np.ndarray:
        """Apply all correction techniques"""
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
            logger.warning(f"All corrections application failed: {e}")
            return np.clip(predictions, 1e-15, 1 - 1e-15)
    
    def _create_fallback_ensemble(self, available_models: List[str]):
        """Create fallback ensemble on failure"""
        try:
            logger.info("Creating fallback ensemble")
            
            # Use equal weights
            equal_weight = 1.0 / len(available_models)
            self.final_weights = {model: equal_weight for model in available_models}
            
            # Set default correction values
            self.bias_correction = 0.0
            self.multiplicative_correction = 1.0
            self.temperature = 1.0
            self.logit_shift = 0.0
            self.quantile_corrections = {}
            
            logger.info(f"Fallback ensemble created successfully: equal weight {equal_weight:.3f}")
            
        except Exception as e:
            logger.error(f"Fallback ensemble creation failed: {e}")
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Main ensemble prediction - execution guaranteed"""
        if not self.is_fitted:
            logger.error("Ensemble model not trained")
            # Provide default ensemble even when not trained
            if base_predictions:
                return np.mean(list(base_predictions.values()), axis=0)
            else:
                raise ValueError("Ensemble model not trained and no default predictions available")
        
        try:
            # Guarantee ensemble execution
            logger.info("Ensemble prediction execution started")
            
            # Check if all models are working properly
            valid_predictions = {}
            for name, pred in base_predictions.items():
                if pred is not None and len(pred) > 0 and not np.all(np.isnan(pred)):
                    valid_predictions[name] = pred
                else:
                    logger.warning(f"{name} model prediction is invalid")
            
            if len(valid_predictions) < 2:
                logger.warning("Insufficient valid models, using single model")
                if valid_predictions:
                    return list(valid_predictions.values())[0]
                else:
                    return np.full(len(list(base_predictions.values())[0]), 0.0201)
            
            # Create final ensemble
            final_pred = self._create_final_ensemble(valid_predictions)
            final_pred = self._apply_all_corrections(final_pred)
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    final_pred = self.ensemble_calibrator.predict(final_pred)
                    final_pred = np.clip(final_pred, 1e-15, 1 - 1e-15)
                except Exception as e:
                    logger.warning(f"Ensemble calibration prediction failed: {e}")
            
            logger.info("Ensemble prediction execution completed")
            return final_pred
            
        except Exception as e:
            logger.error(f"Ensemble prediction execution failed: {e}")
            # Provide basic weighted average even on failure to guarantee ensemble
            try:
                logger.info("Executing fallback ensemble prediction")
                return np.mean(list(base_predictions.values()), axis=0)
            except Exception as e2:
                logger.error(f"Fallback ensemble prediction also failed: {e2}")
                return np.full(len(list(base_predictions.values())[0]), 0.0201)

class CTRStabilizedEnsemble(BaseEnsemble):
    """CTR prediction stabilized ensemble"""
    
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
        """Stabilized ensemble training"""
        logger.info(f"CTR stabilized ensemble training started - Method: {self.diversification_method}")
        
        available_models = list(base_predictions.keys())
        
        if len(available_models) < 2:
            logger.warning("Insufficient models for ensemble")
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
                logger.warning(f"Stabilized ensemble calibration application failed: {e}")
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            logger.info(f"CTR stabilized ensemble training completed - Final weights: {self.final_weights}")
            logger.info(f"Ensemble calibration applied: {'Yes' if self.is_calibrated else 'No'}")
            
        except Exception as e:
            logger.error(f"Stabilized ensemble training failed: {e}")
            # Use equal weights as fallback
            equal_weight = 1.0 / len(available_models)
            self.final_weights = {model: equal_weight for model in available_models}
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
    
    def _evaluate_individual_performance(self, base_predictions: Dict[str, np.ndarray], 
                                       y: pd.Series) -> Dict[str, float]:
        """Evaluate individual model performance"""
        
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
                
                logger.info(f"{name} - Combined: {combined_score:.4f}, CTR bias: {ctr_bias:.4f}, "
                          f"Quality: {quality_score:.4f}, Calibration bonus: {calibration_bonus:.2f}x, "
                          f"Final: {performance_score:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} performance evaluation failed: {e}")
                performance_weights[name] = 0.02
        
        return performance_weights
    
    def _calculate_calibration_weights(self) -> Dict[str, float]:
        """Calculate calibration weights"""
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
                            
                            logger.info(f"{name} calibration quality: {calibration_quality:.4f}, "
                                      f"Weight: {calibration_weight:.4f}")
                
                calibration_weights[name] = calibration_weight
                
            except Exception as e:
                logger.warning(f"{name} calibration weight calculation failed: {e}")
                calibration_weights[name] = 0.5
        
        return calibration_weights
    
    def _calculate_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate diversity weights"""
        
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
        
        logger.info(f"Diversity weights: {diversity_weights}")
        return diversity_weights
    
    def _calculate_correlation_matrix(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix"""
        
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
                        logger.warning(f"Correlation calculation failed ({name1}, {name2}): {e}")
                        correlation_matrix[name1][name2] = 0.0
        
        return correlation_matrix
    
    def _calculate_distance_matrix(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate distance matrix"""
        
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
                        logger.warning(f"Distance calculation failed ({name1}, {name2}): {e}")
                        distance_matrix[name1][name2] = 1.0
        
        return distance_matrix
    
    def _calculate_stability_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Calculate stability weights"""
        
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
                
                logger.info(f"{name} stability score: {stability_weights[name]:.4f}")
                
            except Exception as e:
                logger.warning(f"{name} stability calculation failed: {e}")
                stability_weights[name] = 0.5
        
        return stability_weights
    
    def _combine_weights_with_calibration(self) -> Dict[str, float]:
        """Combine weights with calibration consideration"""
        
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
        """Create stabilized ensemble"""
        ensemble_pred = np.zeros(len(list(base_predictions.values())[0]))
        
        for name, weight in self.final_weights.items():
            if name in base_predictions:
                ensemble_pred += weight * base_predictions[name]
        
        ensemble_pred = self._enhance_ensemble_diversity(ensemble_pred)
        
        return ensemble_pred
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stabilized ensemble prediction - execution guaranteed"""
        if not self.is_fitted:
            logger.error("Ensemble model not trained")
            if base_predictions:
                return np.mean(list(base_predictions.values()), axis=0)
            else:
                raise ValueError("Ensemble model not trained and no default predictions available")
        
        try:
            logger.info("Stabilized ensemble prediction execution started")
            
            ensemble_pred = self._create_stabilized_ensemble(base_predictions)
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                    ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
                except Exception as e:
                    logger.warning(f"Stabilized ensemble calibration prediction failed: {e}")
            
            logger.info("Stabilized ensemble prediction execution completed")
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Stabilized ensemble prediction execution failed: {e}")
            # Provide basic weighted average even on failure
            try:
                logger.info("Executing stabilized ensemble fallback prediction")
                return np.mean(list(base_predictions.values()), axis=0)
            except Exception as e2:
                logger.error(f"Stabilized ensemble fallback prediction also failed: {e2}")
                return np.full(len(list(base_predictions.values())[0]), 0.0201)

class CTRSuperEnsembleManager:
    """CTR specialized ensemble management class"""
    
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
        self.ensemble_execution_status = {}  # Track ensemble execution status
        
    def add_base_model(self, name: str, model: BaseModel):
        """Add base model"""
        self.base_models[name] = model
        
        calibration_status = "No"
        if hasattr(model, 'is_calibrated') and model.is_calibrated:
            calibration_status = "Yes"
            if hasattr(model, 'calibrator') and model.calibrator:
                calibration_method = getattr(model.calibrator, 'best_method', 'unknown')
                calibration_status = f"Yes ({calibration_method})"
        
        logger.info(f"Base model added: {name} - Calibration: {calibration_status}")
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """Create CTR specialized ensemble"""
        
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
                raise ValueError(f"Unsupported ensemble type: {ensemble_type}")
            
            for name, model in self.base_models.items():
                ensemble.add_base_model(name, model)
            
            self.ensembles[ensemble_type] = ensemble
            self.ensemble_execution_status[ensemble_type] = {'created': True, 'fitted': False, 'error': None}
            logger.info(f"Ensemble created: {ensemble_type}")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Ensemble creation failed ({ensemble_type}): {e}")
            self.ensemble_execution_status[ensemble_type] = {'created': False, 'fitted': False, 'error': str(e)}
            raise
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """Train all ensembles - execution guaranteed"""
        logger.info("All ensemble training started - execution guaranteed")
        
        base_predictions = {}
        calibration_info = {}
        
        # Collect predictions from all base models
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
                
                # Validate prediction
                if pred is None or len(pred) == 0 or np.all(np.isnan(pred)):
                    logger.error(f"{name} model prediction is invalid")
                    pred = np.full(len(X), 0.0201)
                
                base_predictions[name] = pred
                logger.info(f"{name} model prediction completed ({prediction_time:.2f}s) - "
                          f"Calibration: {'Yes' if calibration_info[name]['calibrated'] else 'No'}")
                
            except Exception as e:
                logger.error(f"{name} model prediction failed: {str(e)}")
                base_predictions[name] = np.full(len(X), 0.0201)
                calibration_info[name] = {'calibrated': False, 'method': 'error'}
        
        # Ensure at least one valid prediction exists
        if not base_predictions:
            logger.error("All base model predictions failed")
            base_predictions['dummy'] = np.full(len(X), 0.0201)
            calibration_info['dummy'] = {'calibrated': False, 'method': 'dummy'}
        
        # Train each ensemble - execution guaranteed
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                logger.info(f"{ensemble_type} ensemble training started - execution guaranteed")
                start_time = time.time()
                
                # Execute ensemble training
                ensemble.fit(X, y, base_predictions)
                training_time = time.time() - start_time
                
                # Update execution status
                self.ensemble_execution_status[ensemble_type].update({
                    'fitted': True,
                    'training_time': training_time,
                    'ensemble_guaranteed': getattr(ensemble, 'ensemble_execution_guaranteed', False)
                })
                
                calibration_status = "Yes" if ensemble.is_calibrated else "No"
                logger.info(f"{ensemble_type} ensemble training completed ({training_time:.2f}s) - "
                          f"Ensemble Calibration: {calibration_status}, "
                          f"Execution Guaranteed: {self.ensemble_execution_status[ensemble_type]['ensemble_guaranteed']}")
                
            except Exception as e:
                logger.error(f"{ensemble_type} ensemble training failed: {str(e)}")
                # Update execution status
                self.ensemble_execution_status[ensemble_type].update({
                    'fitted': False,
                    'error': str(e),
                    'ensemble_guaranteed': False
                })
                
                # Guarantee basic ensemble even on failure
                try:
                    logger.info(f"{ensemble_type} ensemble applying fallback method")
                    if hasattr(ensemble, '_create_fallback_ensemble'):
                        ensemble._create_fallback_ensemble(list(base_predictions.keys()))
                        ensemble.is_fitted = True
                        self.ensemble_execution_status[ensemble_type].update({
                            'fitted': True,
                            'fallback_used': True,
                            'ensemble_guaranteed': True
                        })
                        logger.info(f"{ensemble_type} ensemble fallback method successful")
                except Exception as fallback_error:
                    logger.error(f"{ensemble_type} ensemble fallback method also failed: {fallback_error}")
        
        self.calibration_manager = calibration_info
        gc.collect()
        
        # Log ensemble execution guarantee status
        logger.info("Ensemble execution guarantee status:")
        for ensemble_type, status in self.ensemble_execution_status.items():
            logger.info(f"  - {ensemble_type}: {status}")
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance - execution guaranteed"""
        logger.info("Ensemble performance evaluation started - execution guaranteed")
        
        results = {}
        
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                
                # Validate prediction
                if pred is None or len(pred) == 0 or np.all(np.isnan(pred)):
                    logger.warning(f"{name} model validation prediction is invalid")
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
                            logger.info(f"{name} calibration effect: {calibration_improvement:+.4f}")
                        except:
                            pass
                
            except Exception as e:
                logger.error(f"{name} model validation prediction failed: {str(e)}")
                results[f"base_{name}"] = 0.0
                results[f"base_{name}_ctr_optimized"] = 0.0
        
        # Evaluate each ensemble - execution guaranteed
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    logger.info(f"{ensemble_type} ensemble evaluation started - execution guaranteed")
                    
                    # Guarantee ensemble prediction execution
                    try:
                        raw_ensemble_pred = ensemble.predict_proba(base_predictions)
                        
                        if ensemble.is_calibrated:
                            calibrated_ensemble_pred = ensemble.predict_proba_calibrated(base_predictions)
                        else:
                            calibrated_ensemble_pred = raw_ensemble_pred
                        
                        # Validate prediction
                        if (calibrated_ensemble_pred is None or len(calibrated_ensemble_pred) == 0 or 
                            np.all(np.isnan(calibrated_ensemble_pred))):
                            logger.warning(f"{ensemble_type} ensemble prediction is invalid")
                            calibrated_ensemble_pred = np.full(len(X_val), 0.0201)
                        
                    except Exception as pred_error:
                        logger.error(f"{ensemble_type} ensemble prediction failed: {pred_error}")
                        # Use default value on prediction failure
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
                            logger.info(f"{ensemble_type} ensemble calibration effect: {calibration_improvement:+.4f}")
                        except:
                            pass
                    
                    # Add ensemble execution guarantee status
                    execution_guaranteed = getattr(ensemble, 'ensemble_execution_guaranteed', False)
                    results[f"ensemble_{ensemble_type}_execution_guaranteed"] = 1.0 if execution_guaranteed else 0.0
                    
                    logger.info(f"{ensemble_type} ensemble Combined Score: {combined_score:.4f}")
                    logger.info(f"{ensemble_type} ensemble CTR Optimized Score: {ctr_optimized_score:.4f}")
                    logger.info(f"{ensemble_type} ensemble Calibration: {'Yes' if ensemble.is_calibrated else 'No'}")
                    logger.info(f"{ensemble_type} ensemble Execution Guaranteed: {'Yes' if execution_guaranteed else 'No'}")
                    
                    predicted_ctr = calibrated_ensemble_pred.mean()
                    actual_ctr = y_val.mean()
                    ctr_bias = abs(predicted_ctr - actual_ctr)
                    logger.info(f"{ensemble_type} CTR: Predicted {predicted_ctr:.4f} vs Actual {actual_ctr:.4f} (Bias: {ctr_bias:.4f})")
                    
                    target_achieved = combined_score >= self.target_combined_score
                    logger.info(f"{ensemble_type} target achieved: {target_achieved} (Target: {self.target_combined_score})")
                    
                except Exception as e:
                    logger.error(f"{ensemble_type} ensemble evaluation failed: {str(e)}")
                    results[f"ensemble_{ensemble_type}"] = 0.0
                    results[f"ensemble_{ensemble_type}_ctr_optimized"] = 0.0
                    results[f"ensemble_{ensemble_type}_execution_guaranteed"] = 0.0
        
        # Select best performance ensemble - execution guarantee priority
        if results:
            ensemble_results = {k: v for k, v in results.items() 
                              if k.startswith('ensemble_') and not k.endswith('_ctr_optimized') 
                              and not k.endswith('_ap') and not k.endswith('_wll') 
                              and not k.endswith('_calibration_improvement')
                              and not k.endswith('_execution_guaranteed')}
            
            if ensemble_results:
                # Select best performance among execution-guaranteed ensembles
                guaranteed_ensembles = []
                for ensemble_name in ensemble_results.keys():
                    ensemble_type = ensemble_name.replace('ensemble_', '')
                    if (f"ensemble_{ensemble_type}_execution_guaranteed" in results and 
                        results[f"ensemble_{ensemble_type}_execution_guaranteed"] > 0.5):
                        guaranteed_ensembles.append(ensemble_name)
                
                if guaranteed_ensembles:
                    best_name = max(guaranteed_ensembles, key=lambda x: ensemble_results[x])
                    logger.info("Best performance selected among execution-guaranteed ensembles")
                else:
                    best_name = max(ensemble_results, key=ensemble_results.get)
                    logger.info("Execution guarantee failed, general best performance ensemble selected")
                
                best_score = ensemble_results[best_name]
                
                ensemble_type = best_name.replace('ensemble_', '')
                self.best_ensemble = self.ensembles[ensemble_type]
                
                logger.info(f"Best performance ensemble: {ensemble_type} (Combined Score: {best_score:.4f})")
                
                if best_score >= self.target_combined_score:
                    logger.info(f"Target Combined Score {self.target_combined_score}+ achieved!")
                else:
                    logger.info(f"Still {self.target_combined_score - best_score:.4f} short of target")
                    
                best_ensemble_obj = self.ensembles[ensemble_type]
                if best_ensemble_obj.is_calibrated:
                    logger.info("Best ensemble has calibration applied")
                if getattr(best_ensemble_obj, 'ensemble_execution_guaranteed', False):
                    logger.info("Best ensemble execution guaranteed")
            else:
                logger.warning("No evaluable ensembles available.")
                self.best_ensemble = None
        
        self.ensemble_results = results
        
        calibration_improvements = [v for k, v in results.items() if k.endswith('_calibration_improvement')]
        if calibration_improvements:
            avg_improvement = np.mean(calibration_improvements)
            logger.info(f"Average calibration effect: {avg_improvement:+.4f}")
        
        return results
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with best performance ensemble - execution guaranteed"""
        logger.info("Best performance ensemble prediction started - execution guaranteed")
        
        try:
            # Collect base model predictions
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    
                    # Validate prediction
                    if pred is None or len(pred) == 0 or np.all(np.isnan(pred)):
                        logger.warning(f"{name} model prediction is invalid")
                        pred = np.full(len(X), 0.0201)
                    
                    base_predictions[name] = pred
                    
                except Exception as e:
                    logger.error(f"{name} prediction failed: {str(e)}")
                    base_predictions[name] = np.full(len(X), 0.0201)
            
            # Ensure at least one prediction exists
            if not base_predictions:
                logger.error("All base model predictions failed")
                return np.full(len(X), 0.0201)
            
            # Predict with best ensemble - execution guaranteed
            if self.best_ensemble is None:
                logger.warning("No best ensemble available, using best performance base model")
                
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
                    logger.info(f"Using best performance base model: {best_model_name}")
                    return self.base_models[best_model_name].predict_proba(X)
                else:
                    logger.info("Using average ensemble")
                    return np.mean(list(base_predictions.values()), axis=0)
            
            # Execute prediction with best ensemble
            try:
                logger.info(f"Best ensemble ({self.best_ensemble.name}) prediction execution")
                
                if self.best_ensemble.is_calibrated:
                    prediction = self.best_ensemble.predict_proba_calibrated(base_predictions)
                else:
                    prediction = self.best_ensemble.predict_proba(base_predictions)
                
                # Final prediction validation
                if prediction is None or len(prediction) == 0 or np.all(np.isnan(prediction)):
                    logger.error("Best ensemble prediction is invalid")
                    prediction = np.mean(list(base_predictions.values()), axis=0)
                
                logger.info("Best ensemble prediction completed - execution guaranteed")
                return prediction
                
            except Exception as ensemble_error:
                logger.error(f"Best ensemble prediction failed: {ensemble_error}")
                
                # Use basic weighted average on ensemble failure
                logger.info("Ensemble failed, using basic weighted average")
                return np.mean(list(base_predictions.values()), axis=0)
            
        except Exception as e:
            logger.error(f"Ensemble prediction total failure: {e}")
            # Last resort: return default value
            return np.full(len(X), 0.0201)
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble summary information"""
        
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
        
        # Calculate ensemble execution guarantee status
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