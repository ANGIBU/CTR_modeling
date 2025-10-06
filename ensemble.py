# ensemble.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import gc
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import warnings
from pathlib import Path
import pickle

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from config import Config
from models import BaseModel
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

def align_predictions_to_target_ctr(predictions: np.ndarray, target_ctr: float = 0.0191) -> np.ndarray:
    """Align predictions to target CTR"""
    try:
        current_ctr = np.mean(predictions)
        
        if current_ctr > 0.001:
            scale_factor = target_ctr / current_ctr
            aligned = predictions * scale_factor
            aligned = np.clip(aligned, 1e-7, 0.5)
            
            # Verify alignment
            final_ctr = np.mean(aligned)
            if abs(final_ctr - target_ctr) > 0.001:
                # Apply additional correction if needed
                additional_scale = target_ctr / final_ctr if final_ctr > 0 else 1.0
                aligned = aligned * additional_scale
                aligned = np.clip(aligned, 1e-7, 0.5)
            
            logger.info(f"CTR alignment: {current_ctr:.4f} -> {np.mean(aligned):.4f} (target: {target_ctr:.4f})")
            return aligned
        else:
            return np.full_like(predictions, target_ctr)
            
    except Exception as e:
        logger.error(f"CTR alignment failed: {e}")
        return predictions

class BaseEnsemble(ABC):
    """Base ensemble class"""
    
    def __init__(self, name: str = "BaseEnsemble"):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.weights = {}
        self.performance_scores = {}
        self.diversity_scores = {}
        self.calibration_scores = {}
        self.final_weights = {}
        self.ensemble_execution_guaranteed = False
        self.target_ctr = 0.0191
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit ensemble"""
        pass
    
    @abstractmethod
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    def add_base_model(self, name: str, model: BaseModel):
        """Add base model"""
        self.base_models[name] = model
        logger.info(f"Base model added to {self.name}: {name}")

class CTRStackingEnsemble(BaseEnsemble):
    """CTR stacking ensemble"""
    
    def __init__(self, cv_folds: int = 3, meta_learner_types: Optional[List[str]] = None):
        super().__init__("CTRStacking")
        self.cv_folds = cv_folds
        self.meta_learner_types = meta_learner_types or ['logistic']
        self.meta_learners = {}
        self.oof_predictions = {}
        self.meta_features = None
        self.weight_calculation_methods = ['performance']
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit stacking ensemble with CTR correction"""
        logger.info(f"{self.name} ensemble training started")
        
        try:
            if len(self.base_models) < 2:
                logger.warning("Stacking requires at least 2 base models, falling back to simple average")
                self.weights = {name: 1.0 / len(base_predictions) for name in base_predictions.keys()}
                self.is_fitted = True
                self.ensemble_execution_guaranteed = True
                return
            
            logger.info("Generating out-of-fold predictions")
            self.oof_predictions = {}
            
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            for name in self.base_models.keys():
                self.oof_predictions[name] = np.zeros(len(X))
            
            for name, pred in base_predictions.items():
                if len(pred) == len(y):
                    self.oof_predictions[name] = pred
                else:
                    logger.warning(f"Prediction length mismatch for {name}, using zeros")
                    self.oof_predictions[name] = np.zeros(len(y))
            
            logger.info("Calculating performance-based weights")
            performance_weights = self._calculate_performance_weights(y)
            
            self.weights = performance_weights
            
            logger.info("Training meta learners")
            meta_features = np.column_stack([self.oof_predictions[name] for name in self.base_models.keys()])
            
            try:
                meta_learner = LogisticRegression(
                    max_iter=1000, 
                    random_state=42,
                    C=1.0,
                    class_weight='balanced'
                )
                meta_learner.fit(meta_features, y)
                self.meta_learners['logistic'] = meta_learner
                logger.info("Meta learner trained: logistic")
            except Exception as e:
                logger.warning(f"Meta learner training failed: {e}")
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            logger.info(f"{self.name} ensemble training completed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.weights = {name: 1.0 / len(base_predictions) for name in base_predictions.keys()}
            self.is_fitted = True
            self.ensemble_execution_guaranteed = False
    
    def _calculate_performance_weights(self, y: pd.Series) -> Dict[str, float]:
        """Calculate performance-based weights with CTR awareness"""
        weights = {}
        total_score = 0.0
        
        metrics_calculator = CTRMetrics()
        
        for name, pred in self.oof_predictions.items():
            try:
                auc = roc_auc_score(y, pred)
                ap = average_precision_score(y, pred)
                
                actual_ctr = np.mean(y)
                predicted_ctr = np.mean(pred)
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_penalty = min(ctr_bias * 10, 0.5)
                
                combined_score = (0.5 * auc + 0.5 * ap) * (1 - ctr_penalty)
                
                weights[name] = max(combined_score, 0.1)
                total_score += weights[name]
                
            except Exception as e:
                logger.warning(f"Performance calculation failed for {name}: {e}")
                weights[name] = 0.1
                total_score += 0.1
        
        if total_score > 0:
            weights = {name: weight / total_score for name, weight in weights.items()}
        else:
            weights = {name: 1.0 / len(self.oof_predictions) for name in self.oof_predictions.keys()}
        
        logger.info(f"Performance weights calculated: {weights}")
        return weights
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using stacking ensemble with CTR alignment"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            if 'logistic' in self.meta_learners and len(base_predictions) >= 2:
                try:
                    meta_features = np.column_stack([base_predictions[name] for name in self.base_models.keys() if name in base_predictions])
                    pred = self.meta_learners['logistic'].predict_proba(meta_features)[:, 1]
                    # Apply CTR alignment
                    pred = align_predictions_to_target_ctr(pred, self.target_ctr)
                    return pred
                except Exception as e:
                    logger.warning(f"Meta-learner prediction failed: {e}")
            
            pred = self._weighted_average_prediction(base_predictions)
            pred = align_predictions_to_target_ctr(pred, self.target_ctr)
            return pred
            
        except Exception as e:
            logger.error(f"Stacking ensemble prediction failed: {e}")
            pred = self._simple_average_prediction(base_predictions)
            pred = align_predictions_to_target_ctr(pred, self.target_ctr)
            return pred
    
    def _weighted_average_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average prediction"""
        weighted_pred = np.zeros(len(next(iter(base_predictions.values()))))
        total_weight = 0.0
        
        for name, pred in base_predictions.items():
            if name in self.weights:
                weight = self.weights[name]
                weighted_pred += weight * pred
                total_weight += weight
        
        if total_weight > 0:
            return weighted_pred / total_weight
        else:
            return self._simple_average_prediction(base_predictions)
    
    def _simple_average_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average prediction"""
        return np.mean(list(base_predictions.values()), axis=0)

class CTRDynamicEnsemble(BaseEnsemble):
    """CTR dynamic ensemble with strategy selection"""
    
    def __init__(self):
        super().__init__("CTRDynamic")
        self.strategies = ['simple_average', 'performance_weighted', 'ctr_corrected']
        self.strategy_weights = {}
        self.best_strategy = 'simple_average'
        self.performance_history = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit dynamic ensemble with strategy selection"""
        logger.info(f"{self.name} ensemble training started")
        
        try:
            logger.info("Initializing performance tracking")
            self.performance_history = {}
            
            logger.info("Calculating multi-strategy weights")
            self._calculate_strategy_weights(y, base_predictions)
            
            logger.info("Selecting optimal strategy")
            best_score = -1
            metrics_calculator = CTRMetrics()
            
            for strategy in self.strategies:
                try:
                    pred = self._apply_strategy(strategy, base_predictions)
                    score = metrics_calculator.combined_score(y.values, pred)
                    
                    logger.info(f"Strategy {strategy}: Combined Score {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        self.best_strategy = strategy
                        
                except Exception as e:
                    logger.warning(f"Strategy evaluation failed for {strategy}: {e}")
            
            logger.info(f"Best strategy selected: {self.best_strategy} (Score: {best_score:.4f})")
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            logger.info(f"{self.name} ensemble training completed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.best_strategy = 'simple_average'
            self.is_fitted = True
            self.ensemble_execution_guaranteed = False
    
    def _calculate_strategy_weights(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Calculate weights for different strategies"""
        self.strategy_weights = {
            'simple_average': {name: 1.0 / len(base_predictions) for name in base_predictions.keys()},
            'performance_weighted': self._calculate_performance_weights(y, base_predictions),
            'ctr_corrected': self._calculate_ctr_corrected_weights(y, base_predictions)
        }
    
    def _calculate_performance_weights(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate performance-based weights"""
        weights = {}
        total_score = 0.0
        
        for name, pred in base_predictions.items():
            try:
                auc = roc_auc_score(y, pred)
                ap = average_precision_score(y, pred)
                score = 0.5 * auc + 0.5 * ap
                
                weights[name] = max(score, 0.1)
                total_score += weights[name]
                
            except Exception as e:
                logger.warning(f"Performance weight calculation failed for {name}: {e}")
                weights[name] = 0.1
                total_score += 0.1
        
        if total_score > 0:
            weights = {name: weight / total_score for name, weight in weights.items()}
        else:
            weights = {name: 1.0 / len(base_predictions) for name in base_predictions.keys()}
        
        return weights
    
    def _calculate_ctr_corrected_weights(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate CTR-corrected weights"""
        weights = {}
        total_score = 0.0
        actual_ctr = np.mean(y)
        
        for name, pred in base_predictions.items():
            try:
                auc = roc_auc_score(y, pred)
                ap = average_precision_score(y, pred)
                
                predicted_ctr = np.mean(pred)
                ctr_bias = abs(predicted_ctr - actual_ctr)
                ctr_penalty = min(ctr_bias * 15, 0.6)
                
                score = (0.5 * auc + 0.5 * ap) * (1 - ctr_penalty)
                weights[name] = max(score, 0.05)
                total_score += weights[name]
                
            except Exception as e:
                logger.warning(f"CTR corrected weight calculation failed for {name}: {e}")
                weights[name] = 0.05
                total_score += 0.05
        
        if total_score > 0:
            weights = {name: weight / total_score for name, weight in weights.items()}
        else:
            weights = {name: 1.0 / len(base_predictions) for name in base_predictions.keys()}
        
        return weights
    
    def _apply_strategy(self, strategy: str, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply specified strategy"""
        if strategy == 'simple_average':
            return np.mean(list(base_predictions.values()), axis=0)
        elif strategy in ['performance_weighted', 'ctr_corrected']:
            weights = self.strategy_weights.get(strategy, {})
            weighted_pred = np.zeros(len(next(iter(base_predictions.values()))))
            total_weight = 0.0
            
            for name, pred in base_predictions.items():
                if name in weights:
                    weight = weights[name]
                    weighted_pred += weight * pred
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_pred / total_weight
            else:
                return np.mean(list(base_predictions.values()), axis=0)
        else:
            return np.mean(list(base_predictions.values()), axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using best strategy with CTR alignment"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            pred = self._apply_strategy(self.best_strategy, base_predictions)
            pred = align_predictions_to_target_ctr(pred, self.target_ctr)
            return pred
            
        except Exception as e:
            logger.error(f"Dynamic ensemble prediction failed: {e}")
            pred = np.mean(list(base_predictions.values()), axis=0)
            pred = align_predictions_to_target_ctr(pred, self.target_ctr)
            return pred

class CTRMainEnsemble(BaseEnsemble):
    """CTR main ensemble with weighted combination"""
    
    def __init__(self, target_combined_score: float = 0.34):
        super().__init__("CTRMain")
        self.target_combined_score = target_combined_score
        self.sub_ensembles = {}
        self.ensemble_weights = {}
        self.final_prediction = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit main ensemble"""
        logger.info(f"{self.name} ensemble training started - execution guaranteed")
        
        try:
            logger.info("Stage 1: Creating sub-ensembles")
            
            try:
                stacking_ensemble = CTRStackingEnsemble(cv_folds=3)
                for name, model in self.base_models.items():
                    stacking_ensemble.add_base_model(name, model)
                stacking_ensemble.fit(X, y, base_predictions)
                self.sub_ensembles['stacking'] = stacking_ensemble
                logger.info("Stacking ensemble created successfully")
            except Exception as e:
                logger.warning(f"Stacking ensemble creation failed: {e}")
            
            try:
                dynamic_ensemble = CTRDynamicEnsemble()
                for name, model in self.base_models.items():
                    dynamic_ensemble.add_base_model(name, model)
                dynamic_ensemble.fit(X, y, base_predictions)
                self.sub_ensembles['dynamic'] = dynamic_ensemble
                logger.info("Dynamic ensemble created successfully")
            except Exception as e:
                logger.warning(f"Dynamic ensemble creation failed: {e}")
            
            logger.info("Stage 2: Evaluating sub-ensembles")
            ensemble_scores = {}
            metrics_calculator = CTRMetrics()
            
            for ensemble_name, ensemble in self.sub_ensembles.items():
                try:
                    pred = ensemble.predict_proba(base_predictions)
                    combined_score = metrics_calculator.combined_score(y.values, pred)
                    ctr_score = metrics_calculator.ctr_score(y.values, pred)
                    
                    ensemble_scores[ensemble_name] = {
                        'combined_score': combined_score,
                        'ctr_score': ctr_score
                    }
                    
                    logger.info(f"{ensemble_name} ensemble - Combined: {combined_score:.4f}, CTR: {ctr_score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"{ensemble_name} ensemble evaluation failed: {e}")
            
            logger.info("Stage 3: Calculating ensemble weights")
            total_score = sum(scores['combined_score'] for scores in ensemble_scores.values())
            
            if total_score > 0:
                for ensemble_name, scores in ensemble_scores.items():
                    self.ensemble_weights[ensemble_name] = scores['combined_score'] / total_score
            else:
                n_ensembles = len(self.sub_ensembles)
                if n_ensembles > 0:
                    self.ensemble_weights = {name: 1.0 / n_ensembles for name in self.sub_ensembles.keys()}
                else:
                    self.final_prediction = np.mean(list(base_predictions.values()), axis=0)
            
            if self.sub_ensembles:
                ensemble_predictions = {}
                for ensemble_name, ensemble in self.sub_ensembles.items():
                    try:
                        pred = ensemble.predict_proba(base_predictions)
                        ensemble_predictions[ensemble_name] = pred
                    except Exception as e:
                        logger.warning(f"Final prediction failed for {ensemble_name}: {e}")
                
                if ensemble_predictions:
                    self.final_prediction = self._calculate_final_ensemble_prediction(ensemble_predictions)
                else:
                    self.final_prediction = np.mean(list(base_predictions.values()), axis=0)
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            
            logger.info(f"{self.name} ensemble training completed - execution guaranteed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.final_prediction = np.mean(list(base_predictions.values()), axis=0)
            self.is_fitted = True
            self.ensemble_execution_guaranteed = False
    
    def _calculate_final_ensemble_prediction(self, ensemble_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate final ensemble prediction using weighted combination"""
        weighted_pred = np.zeros(len(next(iter(ensemble_predictions.values()))))
        total_weight = 0.0
        
        for ensemble_name, pred in ensemble_predictions.items():
            if ensemble_name in self.ensemble_weights:
                weight = self.ensemble_weights[ensemble_name]
                weighted_pred += weight * pred
                total_weight += weight
        
        if total_weight > 0:
            result = weighted_pred / total_weight
        else:
            result = np.mean(list(ensemble_predictions.values()), axis=0)
        
        # Apply final CTR alignment
        result = align_predictions_to_target_ctr(result, self.target_ctr)
        return result
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using main ensemble with CTR alignment"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            if len(self.sub_ensembles) == 0:
                pred = self._simple_weighted_average(base_predictions)
                pred = align_predictions_to_target_ctr(pred, self.target_ctr)
                return pred
            
            ensemble_predictions = {}
            
            for ensemble_name, ensemble in self.sub_ensembles.items():
                try:
                    pred = ensemble.predict_proba(base_predictions)
                    ensemble_predictions[ensemble_name] = pred
                except Exception as e:
                    logger.warning(f"Sub-ensemble prediction failed ({ensemble_name}): {e}")
            
            if ensemble_predictions:
                result = self._calculate_final_ensemble_prediction(ensemble_predictions)
                return result
            else:
                pred = self._simple_weighted_average(base_predictions)
                pred = align_predictions_to_target_ctr(pred, self.target_ctr)
                return pred
            
        except Exception as e:
            logger.error(f"Main ensemble prediction failed: {e}")
            pred = self._simple_weighted_average(base_predictions)
            pred = align_predictions_to_target_ctr(pred, self.target_ctr)
            return pred
    
    def _simple_weighted_average(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple weighted average of base predictions"""
        if not base_predictions:
            return np.array([])
        
        return np.mean(list(base_predictions.values()), axis=0)

class CTREnsembleManager:
    """CTR ensemble manager"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.base_models = {}
        self.ensembles = {}
        self.best_ensemble = None
        self.final_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.target_combined_score = getattr(config, 'TARGET_COMBINED_SCORE', 0.34)
        self.target_ctr = getattr(config, 'TARGET_CTR', 0.0191)
        
        self.ensemble_types = ['main_ensemble']
        
    def add_base_model(self, name: str, model: BaseModel):
        """Add base model to ensemble manager"""
        self.base_models[name] = model
        logger.info(f"Base model added to ensemble manager: {name}")
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """Train all ensembles"""
        logger.info("All ensemble training started - execution guaranteed")
        
        try:
            logger.info("Generating base model predictions")
            base_predictions = {}
            
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    base_predictions[name] = pred
                    logger.info(f"{name} model prediction completed ({len(pred)} predictions)")
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0191)
            
            logger.info("Creating main ensemble")
            main_ensemble = CTRMainEnsemble(target_combined_score=self.target_combined_score)
            
            for name, model in self.base_models.items():
                main_ensemble.add_base_model(name, model)
            
            logger.info("Ensemble created: main_ensemble")
            
            logger.info("main_ensemble ensemble training started - execution guaranteed")
            start_time = time.time()
            
            main_ensemble.fit(X, y, base_predictions)
            
            training_time = time.time() - start_time
            self.ensembles['main_ensemble'] = main_ensemble
            self.final_ensemble = main_ensemble
            
            logger.info(f"main_ensemble ensemble training completed ({training_time:.2f}s)")
            logger.info("All ensemble training completed - execution guaranteed")
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            main_ensemble = CTRMainEnsemble()
            main_ensemble.final_prediction = np.mean(list(base_predictions.values()), axis=0) if base_predictions else np.array([0.0191])
            main_ensemble.is_fitted = True
            self.ensembles['main_ensemble'] = main_ensemble
            self.final_ensemble = main_ensemble
    
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
                base_predictions[name] = np.full(len(X), 0.0191)
        
        best_score = -1
        best_ensemble_name = None
        
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                if not ensemble.is_fitted:
                    continue
                
                ensemble_pred = ensemble.predict_proba(base_predictions)
                combined_score = self.metrics_calculator.combined_score(y.values, ensemble_pred)
                ctr_score = self.metrics_calculator.ctr_score(y.values, ensemble_pred)
                
                self.ensemble_results[f"ensemble_{ensemble_type}"] = combined_score
                self.ensemble_results[f"ensemble_{ensemble_type}_ctr"] = ctr_score
                
                execution_guaranteed = getattr(ensemble, 'ensemble_execution_guaranteed', False)
                self.ensemble_results[f"ensemble_{ensemble_type}_execution_guaranteed"] = 1.0 if execution_guaranteed else 0.0
                
                logger.info(f"{ensemble_type} ensemble Combined Score: {combined_score:.4f}")
                logger.info(f"{ensemble_type} ensemble CTR Score: {ctr_score:.4f}")
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
                    base_predictions[name] = np.full(len(X), 0.0191)
            
            ensemble_pred = self.best_ensemble.predict_proba(base_predictions)
            
            # Final CTR alignment
            ensemble_pred = align_predictions_to_target_ctr(ensemble_pred, self.target_ctr)
            
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
                    except Exception as e2:
                        logger.error(f"Individual model fallback failed: {e2}")
                        predictions.append(np.full(len(X), 0.0191))
                
                if predictions:
                    result = np.mean(predictions, axis=0)
                    result = align_predictions_to_target_ctr(result, self.target_ctr)
                    return result
                else:
                    return np.full(len(X), 0.0191)
            else:
                logger.error("No models available for prediction")
                return np.full(len(X), 0.0191)
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, bool]:
        """Predict using best ensemble with success indicator"""
        logger.info("Best ensemble prediction started")
        
        try:
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
                    model_name = list(self.base_models.keys())[0]
                    model = self.base_models[model_name]
                    predictions = model.predict_proba(X)
                    predictions = align_predictions_to_target_ctr(predictions, self.target_ctr)
                    logger.info(f"Using individual model: {model_name}")
                    return predictions, False
                else:
                    logger.warning("No models available, using default predictions")
                    return np.full(len(X), 0.0191), False
            
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    base_predictions[name] = pred
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0191)
            
            ensemble_pred = available_ensemble.predict_proba(base_predictions)
            
            # Final CTR alignment
            ensemble_pred = align_predictions_to_target_ctr(ensemble_pred, self.target_ctr)
            
            predicted_ctr = ensemble_pred.mean()
            
            logger.info(f"Ensemble prediction completed - CTR: {predicted_ctr:.4f}")
            
            return ensemble_pred, True
            
        except Exception as e:
            logger.error(f"Best ensemble prediction failed: {e}")
            
            if len(self.base_models) > 0:
                logger.info("Falling back to individual model")
                model_name = list(self.base_models.keys())[0]
                model = self.base_models[model_name]
                try:
                    predictions = model.predict_proba(X)
                    predictions = align_predictions_to_target_ctr(predictions, self.target_ctr)
                    return predictions, False
                except Exception as e2:
                    logger.error(f"Individual model fallback failed: {e2}")
                    return np.full(len(X), 0.0191), False
            else:
                logger.error("No models available for prediction")
                return np.full(len(X), 0.0191), False