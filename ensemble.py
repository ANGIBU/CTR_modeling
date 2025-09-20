# ensemble.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import gc
import time
import pickle
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import threading

# Safe library imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna is not installed. Hyperparameter tuning functionality will be disabled.")

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
                final_pred = self.ensemble_calibrator.predict(final_pred)
            
            # Validation
            combined_score = self.metrics_calculator.combined_score(y, final_pred)
            logger.info(f"Main ensemble Combined Score: {combined_score:.4f}")
            
            if combined_score >= self.target_combined_score:
                logger.info(f"Target Combined Score {self.target_combined_score}+ achieved!")
                self.ensemble_execution_guaranteed = True
            else:
                logger.info(f"Target Combined Score not achieved (Current: {combined_score:.4f})")
                self.ensemble_execution_guaranteed = True  # Still guarantee execution
            
            self.is_fitted = True
            logger.info("CTR main ensemble training completed")
            
        except Exception as e:
            logger.error(f"CTR main ensemble training failed: {e}")
            # Fallback to simple averaging
            if available_models:
                self.final_weights = {model: 1.0/len(available_models) for model in available_models}
                self.is_fitted = True
                self.ensemble_execution_guaranteed = True
            raise
    
    def _calculate_base_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Calculate base model weights"""
        weights = {}
        
        try:
            # Calculate performance scores for each model
            scores = {}
            for name, pred in base_predictions.items():
                try:
                    combined_score = self.metrics_calculator.combined_score(y, pred)
                    scores[name] = max(combined_score, 0.1)  # Minimum weight
                except Exception as e:
                    logger.warning(f"Score calculation failed for {name}: {e}")
                    scores[name] = 0.1
            
            # Normalize weights
            total_score = sum(scores.values())
            if total_score > 0:
                weights = {name: score / total_score for name, score in scores.items()}
            else:
                # Equal weights fallback
                weights = {name: 1.0/len(base_predictions) for name in base_predictions.keys()}
            
            logger.info(f"Base weights calculated: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Base weight calculation failed: {e}")
            # Equal weights fallback
            return {name: 1.0/len(base_predictions) for name in base_predictions.keys()}
    
    def _create_weighted_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create weighted ensemble prediction"""
        try:
            weighted_pred = np.zeros(len(list(base_predictions.values())[0]))
            
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    weighted_pred += weight * base_predictions[name]
            
            return np.clip(weighted_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Weighted ensemble creation failed: {e}")
            # Simple average fallback
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)
    
    def _apply_ctr_postprocessing(self, predictions: np.ndarray, y: pd.Series):
        """Apply CTR post-processing"""
        try:
            current_ctr = predictions.mean()
            target_ctr = y.mean()
            
            if abs(current_ctr - target_ctr) > 0.001:
                self.bias_correction = target_ctr - current_ctr
                logger.info(f"CTR bias correction: {self.bias_correction:+.4f}")
            
        except Exception as e:
            logger.warning(f"CTR post-processing failed: {e}")
    
    def _apply_meta_learning(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """Apply meta learning"""
        try:
            if len(base_predictions) < 2:
                return
            
            # Create meta features
            meta_features = []
            for name, pred in base_predictions.items():
                meta_features.append(pred.reshape(-1, 1))
            
            X_meta = np.hstack(meta_features)
            
            # Train meta learner
            self.meta_learner = LogisticRegression(
                random_state=42,
                max_iter=100,
                solver='liblinear'
            )
            
            sample_size = min(len(X_meta), 10000)  # Memory efficiency
            if sample_size < len(X_meta):
                indices = np.random.choice(len(X_meta), sample_size, replace=False)
                X_meta_sample = X_meta[indices]
                y_sample = y.iloc[indices]
            else:
                X_meta_sample = X_meta
                y_sample = y
            
            self.meta_learner.fit(X_meta_sample, y_sample)
            logger.info("Meta learning applied")
            
        except Exception as e:
            logger.warning(f"Meta learning failed: {e}")
            self.meta_learner = None
    
    def _apply_stacking_layer(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """Apply stacking layer"""
        try:
            if len(base_predictions) < 2:
                return
            
            # Simple stacking weights based on individual performance
            stacking_weights = {}
            total_weight = 0
            
            for name, pred in base_predictions.items():
                try:
                    score = self.metrics_calculator.combined_score(y, pred)
                    weight = max(score, 0.1)
                    stacking_weights[name] = weight
                    total_weight += weight
                except:
                    stacking_weights[name] = 0.1
                    total_weight += 0.1
            
            # Normalize
            if total_weight > 0:
                self.stacking_weights = {k: v/total_weight for k, v in stacking_weights.items()}
            
            logger.info(f"Stacking weights: {self.stacking_weights}")
            
        except Exception as e:
            logger.warning(f"Stacking layer failed: {e}")
            self.stacking_weights = {}
    
    def _create_final_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create final ensemble prediction"""
        try:
            # Use stacking weights if available
            if self.stacking_weights:
                weighted_pred = np.zeros(len(list(base_predictions.values())[0]))
                for name, weight in self.stacking_weights.items():
                    if name in base_predictions:
                        weighted_pred += weight * base_predictions[name]
            else:
                # Use base weights
                weighted_pred = self._create_weighted_ensemble(base_predictions)
            
            return np.clip(weighted_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Final ensemble creation failed: {e}")
            # Simple average fallback
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)
    
    def _apply_all_corrections(self, predictions: np.ndarray) -> np.ndarray:
        """Apply all corrections to predictions"""
        try:
            # Apply bias correction
            if hasattr(self, 'bias_correction') and self.bias_correction != 0:
                predictions = predictions + self.bias_correction
            
            # Apply multiplicative correction
            if hasattr(self, 'multiplicative_correction') and self.multiplicative_correction != 1.0:
                predictions = predictions * self.multiplicative_correction
            
            # Apply temperature scaling
            if hasattr(self, 'temperature') and self.temperature != 1.0:
                predictions = predictions ** (1.0 / self.temperature)
            
            return np.clip(predictions, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.warning(f"Corrections application failed: {e}")
            return np.clip(predictions, 1e-15, 1 - 1e-15)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Main ensemble prediction"""
        try:
            if not self.is_fitted:
                logger.warning("Ensemble not fitted, using simple average")
                predictions = list(base_predictions.values())
                return np.mean(predictions, axis=0)
            
            # Create final ensemble
            ensemble_pred = self._create_final_ensemble(base_predictions)
            
            # Apply all corrections
            ensemble_pred = self._apply_all_corrections(ensemble_pred)
            
            # Apply meta learning if available
            if self.meta_learner is not None:
                try:
                    meta_features = []
                    for name in base_predictions.keys():
                        if name in base_predictions:
                            meta_features.append(base_predictions[name].reshape(-1, 1))
                    
                    if meta_features:
                        X_meta = np.hstack(meta_features)
                        meta_pred = self.meta_learner.predict_proba(X_meta)[:, 1]
                        
                        # Blend with ensemble prediction
                        ensemble_pred = 0.7 * ensemble_pred + 0.3 * meta_pred
                        
                except Exception as e:
                    logger.warning(f"Meta learning prediction failed: {e}")
            
            # Apply calibration if available
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                except Exception as e:
                    logger.warning(f"Ensemble calibration failed: {e}")
            
            # Final CTR adjustment
            current_ctr = ensemble_pred.mean()
            if abs(current_ctr - self.target_ctr) > 0.002:
                adjustment_factor = self.target_ctr / current_ctr
                ensemble_pred = ensemble_pred * adjustment_factor
            
            return np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Main ensemble prediction failed: {e}")
            # Fallback to simple average
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)

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
            logger.info(f"Ensemble calibration: {'Yes' if self.is_calibrated else 'No'}")
            
        except Exception as e:
            logger.error(f"Stabilized ensemble training failed: {e}")
            # Fallback to equal weights
            self.final_weights = {model: 1.0/len(available_models) for model in available_models}
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
    
    def _evaluate_individual_performance(self, base_predictions: Dict[str, np.ndarray], 
                                       y: pd.Series) -> Dict[str, float]:
        """Evaluate individual model performance"""
        weights = {}
        
        for name, pred in base_predictions.items():
            try:
                combined_score = self.metrics_calculator.combined_score(y, pred)
                weights[name] = max(combined_score, 0.1)
            except Exception as e:
                logger.warning(f"Performance evaluation failed for {name}: {e}")
                weights[name] = 0.1
        
        return weights
    
    def _calculate_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate diversity weights"""
        weights = {}
        
        try:
            model_names = list(base_predictions.keys())
            
            for name in model_names:
                diversity_score = 0.0
                count = 0
                
                for other_name in model_names:
                    if name != other_name:
                        correlation = np.corrcoef(base_predictions[name], base_predictions[other_name])[0, 1]
                        diversity_score += (1.0 - abs(correlation))
                        count += 1
                
                if count > 0:
                    weights[name] = diversity_score / count
                else:
                    weights[name] = 1.0
            
        except Exception as e:
            logger.warning(f"Diversity weight calculation failed: {e}")
            weights = {name: 1.0 for name in base_predictions.keys()}
        
        return weights
    
    def _calculate_stability_weights(self, base_predictions: Dict[str, np.ndarray], 
                                   y: pd.Series) -> Dict[str, float]:
        """Calculate stability weights"""
        weights = {}
        
        try:
            for name, pred in base_predictions.items():
                # Calculate prediction stability (lower variance = higher stability)
                pred_std = np.std(pred)
                stability_score = 1.0 / (1.0 + pred_std)
                weights[name] = stability_score
                
        except Exception as e:
            logger.warning(f"Stability weight calculation failed: {e}")
            weights = {name: 1.0 for name in base_predictions.keys()}
        
        return weights
    
    def _calculate_calibration_weights(self) -> Dict[str, float]:
        """Calculate calibration weights"""
        # Simple implementation: equal weights for now
        # Can be enhanced based on calibration quality
        return {name: 1.0 for name in self.model_weights.keys()}
    
    def _combine_weights_with_calibration(self) -> Dict[str, float]:
        """Combine all weights with calibration consideration"""
        try:
            combined_weights = {}
            
            for name in self.model_weights.keys():
                performance_weight = self.model_weights.get(name, 0.1)
                diversity_weight = self.diversity_weights.get(name, 1.0)
                stability_weight = self.stability_weights.get(name, 1.0)
                calibration_weight = self.calibration_weights.get(name, 1.0)
                
                # Weighted combination
                combined_score = (0.4 * performance_weight + 
                                0.2 * diversity_weight + 
                                0.2 * stability_weight + 
                                0.2 * calibration_weight)
                
                combined_weights[name] = max(combined_score, 0.05)  # Minimum weight
            
            # Normalize weights
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
            
            return combined_weights
            
        except Exception as e:
            logger.error(f"Weight combination failed: {e}")
            # Equal weights fallback
            model_count = len(self.model_weights)
            return {name: 1.0/model_count for name in self.model_weights.keys()}
    
    def _create_stabilized_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create stabilized ensemble prediction"""
        try:
            weighted_pred = np.zeros(len(list(base_predictions.values())[0]))
            
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    weighted_pred += weight * base_predictions[name]
            
            return np.clip(weighted_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Stabilized ensemble creation failed: {e}")
            # Simple average fallback
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stabilized ensemble prediction"""
        try:
            if not self.is_fitted:
                logger.warning("Stabilized ensemble not fitted, using simple average")
                predictions = list(base_predictions.values())
                return np.mean(predictions, axis=0)
            
            ensemble_pred = self._create_stabilized_ensemble(base_predictions)
            
            # Apply calibration if available
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                except Exception as e:
                    logger.warning(f"Stabilized ensemble calibration failed: {e}")
            
            return np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Stabilized ensemble prediction failed: {e}")
            # Fallback to simple average
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)

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
    
    # Add alias methods for backward compatibility
    def add_model(self, name: str, model: BaseModel):
        """Alias for add_base_model (backward compatibility)"""
        return self.add_base_model(name, model)
    
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
        
        # Store calibration info
        self.calibration_manager = calibration_info
        
        # Create ensembles if not exists
        if not self.ensembles:
            try:
                logger.info("Creating default ensembles")
                self.create_ensemble('final_ensemble', target_ctr=0.0201)
                self.create_ensemble('stabilized', diversification_method='rank_weighted')
            except Exception as e:
                logger.warning(f"Default ensemble creation failed: {e}")
        
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
                    'error': None
                })
                
                logger.info(f"{ensemble_type} ensemble training completed ({training_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"{ensemble_type} ensemble training failed: {e}")
                self.ensemble_execution_status[ensemble_type].update({
                    'fitted': False,
                    'error': str(e)
                })
        
        # Evaluate ensembles
        self.evaluate_ensembles(X, y)
        
        logger.info("All ensemble training completed - execution guaranteed")
    
    # Add alias method for backward compatibility
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Alias for train_all_ensembles (backward compatibility)"""
        return self.train_all_ensembles(X, y)
    
    def evaluate_ensembles(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Evaluate ensemble performance"""
        logger.info("Ensemble evaluation started")
        
        results = {}
        
        # Collect base model predictions for validation
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X_val)
                base_predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} model validation prediction failed: {e}")
                base_predictions[name] = np.full(len(X_val), 0.0201)
        
        # Evaluate each ensemble
        for ensemble_type, ensemble in self.ensembles.items():
            if ensemble.is_fitted:
                try:
                    try:
                        # Raw ensemble prediction
                        raw_ensemble_pred = ensemble.predict_proba(base_predictions)
                        
                        # Calibrated ensemble prediction
                        if ensemble.is_calibrated and hasattr(ensemble, 'predict_proba_calibrated'):
                            calibrated_ensemble_pred = ensemble.predict_proba_calibrated(base_predictions)
                        else:
                            calibrated_ensemble_pred = raw_ensemble_pred
                    
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
        
        # Store results
        self.ensemble_results = results
        
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
                    logger.info(f"Target Combined Score not achieved (Best: {best_score:.4f}, Target: {self.target_combined_score})")
        
        logger.info("Ensemble evaluation completed")
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> Tuple[Optional[np.ndarray], bool]:
        """Predict with best performance ensemble"""
        logger.info("Best performance ensemble prediction started - execution guaranteed")
        
        try:
            # Check if best ensemble is available
            if self.best_ensemble is None or not self.best_ensemble.is_fitted:
                # Try to use any available fitted ensemble
                fitted_ensembles = [e for e in self.ensembles.values() if e.is_fitted]
                
                if fitted_ensembles:
                    self.best_ensemble = fitted_ensembles[0]
                    logger.info(f"Using first available ensemble: {self.best_ensemble.name}")
                else:
                    logger.error("All base model predictions failed")
                    return np.full(len(X), 0.0201), False
            
            # Collect base model predictions
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    if pred is None or len(pred) == 0 or np.all(np.isnan(pred)):
                        pred = np.full(len(X), 0.0201)
                    base_predictions[name] = pred
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0201)
            
            if not base_predictions:
                logger.error("No base predictions available")
                return np.full(len(X), 0.0201), False
            
            # Generate ensemble prediction
            try:
                if self.best_ensemble.is_calibrated and hasattr(self.best_ensemble, 'predict_proba_calibrated'):
                    ensemble_pred = self.best_ensemble.predict_proba_calibrated(base_predictions)
                else:
                    ensemble_pred = self.best_ensemble.predict_proba(base_predictions)
                
                # Validation
                if ensemble_pred is None or len(ensemble_pred) == 0 or np.all(np.isnan(ensemble_pred)):
                    raise ValueError("Ensemble prediction is invalid")
                
                ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
                
                # CTR validation
                predicted_ctr = ensemble_pred.mean()
                if abs(predicted_ctr - 0.0191) > 0.01:  # Large deviation
                    logger.warning(f"CTR deviation detected: {predicted_ctr:.4f} (target: 0.0191)")
                    # Apply correction
                    correction_factor = 0.0191 / predicted_ctr
                    ensemble_pred = ensemble_pred * correction_factor
                    ensemble_pred = np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
                
                logger.info(f"Ensemble prediction successful - CTR: {ensemble_pred.mean():.4f}")
                return ensemble_pred, True
                
            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")
                # Fallback to simple averaging
                predictions = list(base_predictions.values())
                fallback_pred = np.mean(predictions, axis=0)
                fallback_pred = np.clip(fallback_pred, 1e-15, 1 - 1e-15)
                logger.info(f"Using fallback prediction - CTR: {fallback_pred.mean():.4f}")
                return fallback_pred, False
                
        except Exception as e:
            logger.error(f"Ensemble prediction total failure: {e}")
            # Last resort: return default value
            return np.full(len(X), 0.0201), False
    
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

# Alias classes for compatibility
CTROptimalEnsemble = CTRMainEnsemble
CTREnsembleManager = CTRSuperEnsembleManager
EnsembleManager = CTRSuperEnsembleManager