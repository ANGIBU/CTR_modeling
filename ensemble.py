# ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
import pickle
import gc
import warnings
import time
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not installed. Hyperparameter tuning functionality will be disabled.")

from config import Config
from models import BaseModel, CTRCalibrator
from evaluation import CTRAdvancedMetrics

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
            
            # Fixed: Remove target_ctr parameter from CTRCalibrator initialization
            self.ensemble_calibrator = CTRCalibrator()
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
                
                logger.info(f"{self.name}: Diversity enhancement applied")
            
            return predictions
        except Exception:
            return predictions

class CTRMainEnsemble(BaseEnsemble):
    """CTR main ensemble model"""
    
    def __init__(self, target_ctr: float = 0.0201, optimization_method: str = 'final_combined'):
        super().__init__("CTRMainEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.model_weights = {}
        self.diversity_weights = {}
        self.stability_weights = {}
        self.calibration_weights = {}
        self.final_weights = {}
        self.meta_learner = None
        self.stacking_weights = {}
        self.ensemble_execution_guaranteed = True
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """CTR main ensemble training"""
        logger.info("CTR main ensemble training started - Target: Combined Score 0.35+")
        
        try:
            logger.info(f"Available models: {list(base_predictions.keys())}")
            
            if len(base_predictions) == 0:
                raise ValueError("No base predictions available")
            
            logger.info("Stage 1: Calculating base weights")
            self.model_weights = self._calculate_base_weights(base_predictions, y)
            logger.info(f"Base weights calculated: {self.model_weights}")
            
            logger.info("Stage 3: CTR post-processing")
            self._post_process_ctr_alignment(base_predictions, y)
            
            logger.info("Stage 4: Applying meta learning")
            self._apply_meta_learning(base_predictions, y)
            logger.info("Meta learning applied")
            
            logger.info("Stage 5: Adding stacking layer")
            self.stacking_weights = self._create_stacking_layer(base_predictions, y)
            logger.info(f"Stacking weights: {self.stacking_weights}")
            
            logger.info("Stage 6: Applying ensemble calibration")
            ensemble_pred = self._create_main_ensemble_prediction(base_predictions)
            
            try:
                self.apply_ensemble_calibration(X, y, ensemble_pred, method='auto')
            except Exception as e:
                logger.error(f"Ensemble calibration application failed: {e}")
                self.is_calibrated = False
            
            logger.info("Stage 7: Final validation and adjustment")
            
            # Calculate performance
            final_pred = self.predict_proba(base_predictions)
            
            metrics_calculator = CTRAdvancedMetrics()
            combined_score = metrics_calculator.combined_score(y.values, final_pred)
            
            logger.info(f"Main ensemble Combined Score: {combined_score:.4f}")
            
            if combined_score >= self.target_combined_score:
                logger.info(f"Target Combined Score achieved! (Current: {combined_score:.4f})")
            else:
                logger.info(f"Target Combined Score not achieved (Current: {combined_score:.4f})")
            
            self.is_fitted = True
            logger.info("CTR main ensemble training completed")
            
        except Exception as e:
            logger.error(f"CTR main ensemble training failed: {e}")
            self.is_fitted = False
            raise
    
    def _calculate_base_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Calculate base model weights"""
        try:
            weights = {}
            metrics_calculator = CTRAdvancedMetrics()
            
            for name, pred in base_predictions.items():
                try:
                    combined_score = metrics_calculator.combined_score(y.values, pred)
                    weights[name] = max(combined_score, 0.1)
                except Exception as e:
                    logger.warning(f"Score calculation failed for {name}: {e}")
                    weights[name] = 0.5
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                model_count = len(base_predictions)
                weights = {name: 1.0/model_count for name in base_predictions.keys()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Base weight calculation failed: {e}")
            model_count = len(base_predictions)
            return {name: 1.0/model_count for name in base_predictions.keys()}
    
    def _post_process_ctr_alignment(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """CTR alignment post-processing"""
        try:
            actual_ctr = y.mean()
            
            for name, pred in base_predictions.items():
                predicted_ctr = pred.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                
                alignment_score = max(0, 1 - ctr_bias * 1000)
                
                if name in self.model_weights:
                    self.model_weights[name] *= (0.8 + 0.2 * alignment_score)
                    
        except Exception as e:
            logger.warning(f"CTR alignment post-processing failed: {e}")
    
    def _apply_meta_learning(self, base_predictions: Dict[str, np.ndarray], y: pd.Series):
        """Apply meta learning"""
        try:
            if SKLEARN_AVAILABLE and len(base_predictions) > 1:
                prediction_matrix = np.column_stack(list(base_predictions.values()))
                
                self.meta_learner = LogisticRegression(random_state=42, max_iter=500)
                self.meta_learner.fit(prediction_matrix, y)
                
                meta_weights = self.meta_learner.coef_[0]
                meta_weights = np.abs(meta_weights)
                meta_weights = meta_weights / np.sum(meta_weights)
                
                for i, name in enumerate(base_predictions.keys()):
                    if name in self.model_weights:
                        self.model_weights[name] = 0.7 * self.model_weights[name] + 0.3 * meta_weights[i]
            
        except Exception as e:
            logger.warning(f"Meta learning failed: {e}")
    
    def _create_stacking_layer(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Create stacking layer"""
        try:
            normalized_weights = {}
            total_weight = sum(self.model_weights.values())
            
            if total_weight > 0:
                for name, weight in self.model_weights.items():
                    normalized_weights[name] = weight / total_weight
            else:
                model_count = len(self.model_weights)
                for name in self.model_weights.keys():
                    normalized_weights[name] = 1.0 / model_count
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Stacking layer creation failed: {e}")
            model_count = len(base_predictions)
            return {name: 1.0/model_count for name in base_predictions.keys()}
    
    def _create_main_ensemble_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create main ensemble prediction"""
        try:
            if len(self.stacking_weights) == 0:
                predictions = list(base_predictions.values())
                return np.mean(predictions, axis=0)
            
            weighted_pred = np.zeros(len(list(base_predictions.values())[0]))
            
            for name, weight in self.stacking_weights.items():
                if name in base_predictions:
                    weighted_pred += weight * base_predictions[name]
            
            return self._enhance_ensemble_diversity(weighted_pred)
            
        except Exception as e:
            logger.error(f"Main ensemble prediction creation failed: {e}")
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Main ensemble prediction"""
        try:
            if not self.is_fitted:
                logger.warning("Main ensemble not fitted, using weighted average")
                return self._create_main_ensemble_prediction(base_predictions)
            
            ensemble_pred = self._create_main_ensemble_prediction(base_predictions)
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                except Exception as e:
                    logger.warning(f"Main ensemble calibration failed: {e}")
            
            return np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Main ensemble prediction failed: {e}")
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)

class CTRStabilizedEnsemble(BaseEnsemble):
    """CTR stabilized ensemble model"""
    
    def __init__(self, diversification_method: str = 'rank_weighted'):
        super().__init__("CTRStabilizedEnsemble")
        self.diversification_method = diversification_method
        self.model_weights = {}
        self.diversity_weights = {}
        self.stability_weights = {}
        self.calibration_weights = {}
        self.final_weights = {}
        self.ensemble_execution_guaranteed = True
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """CTR stabilized ensemble training"""
        logger.info(f"CTR stabilized ensemble training started - Method: {self.diversification_method}")
        
        try:
            if len(base_predictions) == 0:
                raise ValueError("No base predictions available")
            
            metrics_calculator = CTRAdvancedMetrics()
            
            for name, pred in base_predictions.items():
                try:
                    combined_score = metrics_calculator.combined_score(y.values, pred)
                    self.model_weights[name] = max(combined_score, 0.1)
                except Exception as e:
                    logger.warning(f"Performance evaluation failed for {name}: {e}")
                    self.model_weights[name] = 0.5
            
            self.diversity_weights = self._calculate_diversity_weights(base_predictions)
            self.stability_weights = self._calculate_stability_weights(base_predictions)
            self.calibration_weights = self._calculate_calibration_weights()
            
            self.final_weights = self._combine_weights_with_calibration()
            
            ensemble_pred = self._create_stabilized_ensemble(base_predictions)
            
            try:
                self.apply_ensemble_calibration(X, y, ensemble_pred, method='auto')
            except Exception as e:
                logger.error(f"Ensemble calibration application failed: {e}")
                self.is_calibrated = False
            
            self.is_fitted = True
            
            final_weights_str = {k: f"{v:.4f}" for k, v in self.final_weights.items()}
            logger.info(f"CTR stabilized ensemble training completed - Final weights: {final_weights_str}")
            logger.info(f"Ensemble calibration: {'Yes' if self.is_calibrated else 'No'}")
            
        except Exception as e:
            logger.error(f"CTR stabilized ensemble training failed: {e}")
            self.is_fitted = False
            raise
    
    def _calculate_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate diversity weights"""
        try:
            weights = {}
            
            prediction_matrix = np.column_stack(list(base_predictions.values()))
            model_names = list(base_predictions.keys())
            
            for i, name in enumerate(model_names):
                correlations = []
                for j, other_name in enumerate(model_names):
                    if i != j:
                        corr = np.corrcoef(prediction_matrix[:, i], prediction_matrix[:, j])[0, 1]
                        correlations.append(abs(corr))
                
                avg_correlation = np.mean(correlations) if correlations else 0.5
                diversity_score = 1.0 - avg_correlation
                weights[name] = max(diversity_score, 0.1)
            
            return weights
            
        except Exception as e:
            logger.warning(f"Diversity weight calculation failed: {e}")
            return {name: 1.0 for name in base_predictions.keys()}
    
    def _calculate_stability_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate stability weights"""
        try:
            weights = {}
            
            for name, pred in base_predictions.items():
                pred_std = np.std(pred)
                stability_score = 1.0 / (1.0 + pred_std)
                weights[name] = stability_score
                
        except Exception as e:
            logger.warning(f"Stability weight calculation failed: {e}")
            weights = {name: 1.0 for name in base_predictions.keys()}
        
        return weights
    
    def _calculate_calibration_weights(self) -> Dict[str, float]:
        """Calculate calibration weights"""
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
                
                combined_score = (0.4 * performance_weight + 
                                0.2 * diversity_weight + 
                                0.2 * stability_weight + 
                                0.2 * calibration_weight)
                
                combined_weights[name] = max(combined_score, 0.05)
            
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
            
            return combined_weights
            
        except Exception as e:
            logger.error(f"Weight combination failed: {e}")
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
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                except Exception as e:
                    logger.warning(f"Stabilized ensemble calibration failed: {e}")
            
            return np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Stabilized ensemble prediction failed: {e}")
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)

class CTRSuperEnsembleManager:
    """CTR ensemble management class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRAdvancedMetrics()
        self.final_ensemble = None
        self.target_combined_score = 0.35
        self.calibration_manager = {}
        self.ensemble_execution_status = {}
        
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
    
    def add_model(self, name: str, model: BaseModel):
        """Alias for add_base_model"""
        return self.add_base_model(name, model)
    
    def create_ensemble(self, ensemble_type: str, **kwargs) -> BaseEnsemble:
        """Create ensemble"""
        
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
        """Train all ensembles"""
        logger.info("All ensemble training started - execution guaranteed")
        
        base_predictions = {}
        calibration_info = {}
        
        for name, model in self.base_models.items():
            try:
                start_time = time.time()
                
                if hasattr(model, 'is_calibrated') and model.is_calibrated:
                    pred = model.predict_proba(X)
                    calibration_info[name] = {'calibrated': True, 'method': getattr(model.calibrator, 'best_method', 'unknown') if hasattr(model, 'calibrator') and model.calibrator else 'unknown'}
                else:
                    pred = model.predict_proba(X)
                    calibration_info[name] = {'calibrated': False, 'method': 'none'}
                
                base_predictions[name] = pred
                
                elapsed_time = time.time() - start_time
                calibration_status = "Yes" if calibration_info[name]['calibrated'] else "No"
                logger.info(f"{name} model prediction completed ({elapsed_time:.2f}s) - Calibration: {calibration_status}")
                
            except Exception as e:
                logger.error(f"{name} model prediction failed: {e}")
                base_predictions[name] = np.full(len(X), 0.0201)
                calibration_info[name] = {'calibrated': False, 'method': 'none'}
        
        logger.info("Creating default ensembles")
        if 'final_ensemble' not in self.ensembles:
            self.create_ensemble('final_ensemble')
        if 'stabilized' not in self.ensembles:
            self.create_ensemble('stabilized')
        
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                logger.info(f"{ensemble_type} ensemble training started - execution guaranteed")
                
                start_time = time.time()
                ensemble.fit(X, y, base_predictions)
                elapsed_time = time.time() - start_time
                
                self.ensemble_execution_status[ensemble_type]['fitted'] = True
                logger.info(f"{ensemble_type} ensemble training completed ({elapsed_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"{ensemble_type} ensemble training failed: {e}")
                self.ensemble_execution_status[ensemble_type]['error'] = str(e)
                self.ensemble_execution_status[ensemble_type]['fitted'] = False
        
        logger.info("All ensemble training completed - execution guaranteed")
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, bool]:
        """Predict using best ensemble"""
        logger.info("Best ensemble prediction started")
        
        try:
            # Check if any ensemble is available and fitted
            available_ensemble = None
            for ensemble_type, ensemble in self.ensembles.items():
                if ensemble.is_fitted:
                    available_ensemble = ensemble
                    self.best_ensemble = ensemble
                    logger.info(f"Using {ensemble_type} ensemble for prediction")
                    break
            
            if available_ensemble is None:
                logger.warning("No fitted ensemble available, using individual model")
                if len(self.base_models) > 0:
                    # Use first available model
                    model_name = list(self.base_models.keys())[0]
                    model = self.base_models[model_name]
                    predictions = model.predict_proba(X)
                    logger.info(f"Using individual model: {model_name}")
                    return predictions, False
                else:
                    # Default fallback
                    logger.warning("No models available, using default predictions")
                    return np.full(len(X), 0.0201), False
            
            # Get base predictions
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    base_predictions[name] = pred
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0201)
            
            # Get ensemble prediction
            ensemble_pred = available_ensemble.predict_proba(base_predictions)
            predicted_ctr = ensemble_pred.mean()
            
            logger.info(f"Ensemble prediction completed - CTR: {predicted_ctr:.4f}")
            
            return ensemble_pred, True
            
        except Exception as e:
            logger.error(f"Best ensemble prediction failed: {e}")
            
            # Fallback to individual model
            if len(self.base_models) > 0:
                logger.info("Falling back to individual model")
                model_name = list(self.base_models.keys())[0]
                model = self.base_models[model_name]
                try:
                    predictions = model.predict_proba(X)
                    return predictions, False
                except Exception as e2:
                    logger.error(f"Individual model fallback failed: {e2}")
                    return np.full(len(X), 0.0201), False
            else:
                logger.error("No models available for prediction")
                return np.full(len(X), 0.0201), False
    
    def evaluate_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """Evaluate ensembles"""
        logger.info("Ensemble evaluation started")
        
        base_predictions = {}
        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                base_predictions[name] = pred
            except Exception as e:
                logger.error(f"{name} model prediction failed during evaluation: {e}")
                base_predictions[name] = np.full(len(X), 0.0201)
        
        best_score = -1
        best_ensemble_name = None
        
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                if not ensemble.is_fitted:
                    continue
                
                ensemble_pred = ensemble.predict_proba(base_predictions)
                combined_score = self.metrics_calculator.combined_score(y.values, ensemble_pred)
                ctr_optimized_score = self.metrics_calculator.ctr_optimized_score(y.values, ensemble_pred)
                
                self.ensemble_results[f"ensemble_{ensemble_type}"] = combined_score
                self.ensemble_results[f"ensemble_{ensemble_type}_ctr_optimized"] = ctr_optimized_score
                
                execution_guaranteed = getattr(ensemble, 'ensemble_execution_guaranteed', False)
                self.ensemble_results[f"ensemble_{ensemble_type}_execution_guaranteed"] = 1.0 if execution_guaranteed else 0.0
                
                logger.info(f"{ensemble_type} ensemble Combined Score: {combined_score:.4f}")
                logger.info(f"{ensemble_type} ensemble CTR Optimized Score: {ctr_optimized_score:.4f}")
                logger.info(f"{ensemble_type} ensemble Calibration: {'Yes' if ensemble.is_calibrated else 'No'}")
                logger.info(f"{ensemble_type} ensemble Execution Guaranteed: {'Yes' if execution_guaranteed else 'No'}")
                
                predicted_ctr = ensemble_pred.mean()
                actual_ctr = y.mean()
                ctr_bias = abs(predicted_ctr - actual_ctr)
                logger.info(f"{ensemble_type} CTR: Predicted {predicted_ctr:.4f} vs Actual {actual_ctr:.4f} (Bias: {ctr_bias:.4f})")
                
                target_achieved = combined_score >= self.target_combined_score
                logger.info(f"{ensemble_type} target achieved: {target_achieved} (Target: {self.target_combined_score})")
                
                if execution_guaranteed and combined_score > best_score:
                    best_score = combined_score
                    best_ensemble_name = ensemble_type
                
            except Exception as e:
                logger.error(f"{ensemble_type} ensemble evaluation failed: {e}")
        
        if best_ensemble_name:
            self.best_ensemble = self.ensembles[best_ensemble_name]
            logger.info(f"Best performance selected among execution-guaranteed ensembles")
            logger.info(f"Best performance ensemble: {best_ensemble_name} (Combined Score: {best_score:.4f})")
        else:
            logger.warning("No execution-guaranteed ensemble found, selecting first available")
            if self.ensembles:
                self.best_ensemble = list(self.ensembles.values())[0]
                best_ensemble_name = list(self.ensembles.keys())[0]
        
        if best_score < self.target_combined_score:
            logger.info(f"Target Combined Score not achieved (Best: {best_score:.4f}, Target: {self.target_combined_score})")
        
        logger.info("Ensemble evaluation completed")
    
    def predict_best_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using best ensemble"""
        logger.info("Best performance ensemble prediction started - execution guaranteed")
        
        try:
            if self.best_ensemble is None:
                raise ValueError("No best ensemble available")
            
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    base_predictions[name] = pred
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0201)
            
            ensemble_pred = self.best_ensemble.predict_proba(base_predictions)
            predicted_ctr = ensemble_pred.mean()
            
            logger.info(f"Ensemble prediction successful - CTR: {predicted_ctr:.4f}")
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Best ensemble prediction failed: {e}")
            
            if len(self.base_models) > 0:
                logger.info("Falling back to simple model average")
                predictions = []
                for model in self.base_models.values():
                    try:
                        pred = model.predict_proba(X)
                        predictions.append(pred)
                    except Exception:
                        predictions.append(np.full(len(X), 0.0201))
                
                return np.mean(predictions, axis=0)
            else:
                logger.error("No models available for prediction")
                return np.full(len(X), 0.0201)
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get ensemble status"""
        calibrated_base_models = sum(1 for model in self.base_models.values() 
                                   if hasattr(model, 'is_calibrated') and model.is_calibrated)
        
        calibrated_ensembles = sum(1 for ensemble in self.ensembles.values() 
                                 if hasattr(ensemble, 'is_calibrated') and ensemble.is_calibrated)
        
        guaranteed_ensembles = sum(1 for ensemble in self.ensembles.values() 
                                 if getattr(ensemble, 'ensemble_execution_guaranteed', False))
        
        target_achieved_count = sum(1 for key, score in self.ensemble_results.items()
                                  if key.startswith('ensemble_') and not key.endswith('_ctr_optimized') 
                                  and not key.endswith('_ap') and not key.endswith('_wll')
                                  and not key.endswith('_calibration_improvement')
                                  and not key.endswith('_execution_guaranteed')
                                  and score >= self.target_combined_score)
        
        calibration_improvements = [score for key, score in self.ensemble_results.items()
                                  if key.endswith('_calibration_improvement') and score > 0]
        
        return {
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

# Compatibility aliases
CTROptimalEnsemble = CTRMainEnsemble
CTREnsembleManager = CTRSuperEnsembleManager
EnsembleManager = CTRSuperEnsembleManager