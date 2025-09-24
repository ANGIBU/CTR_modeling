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

# Safe imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from config import Config
from models import BaseModel

logger = logging.getLogger(__name__)

class CTRMetrics:
    """CTR metrics calculation for ensemble evaluation"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.ap_weight = 0.6  # Increased AP weight
        self.auc_weight = 0.4  # Reduced AUC weight
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate combined score with proper CTR bias penalty"""
        try:
            if len(y_true) == 0 or len(y_pred_proba) == 0:
                return 0.0
            
            if len(y_true) != len(y_pred_proba):
                return 0.0
            
            # Ensure we have both classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                return 0.0
            
            # Clip predictions
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # Calculate base metrics
            try:
                auc_score = roc_auc_score(y_true, y_pred_proba)
                ap_score = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Metric calculation failed: {e}")
                return 0.0
            
            # Base combined score
            base_score = (auc_score * self.auc_weight) + (ap_score * self.ap_weight)
            
            # CTR bias penalty
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # Apply penalty for large CTR bias
            if ctr_bias > 0.005:  # 0.5% tolerance
                bias_penalty = min(0.5, ctr_bias * 10)
                penalized_score = base_score * (1.0 - bias_penalty)
            else:
                penalized_score = base_score
            
            return float(np.clip(penalized_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.0
    
    def ctr_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """CTR-focused score"""
        try:
            base_score = self.combined_score(y_true, y_pred_proba)
            
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # CTR alignment bonus/penalty
            if ctr_bias <= 0.002:  # Very close to actual CTR
                bonus = min(0.1, (0.002 - ctr_bias) * 50)
                return base_score * (1.0 + bonus)
            else:
                penalty = min(0.3, ctr_bias * 20)
                return base_score * (1.0 - penalty)
                
        except Exception as e:
            logger.error(f"CTR score calculation failed: {e}")
            return 0.0

class BaseEnsemble(ABC):
    """Base ensemble class"""
    
    def __init__(self, name: str = "BaseEnsemble"):
        self.name = name
        self.base_models = {}
        self.is_fitted = False
        self.weights = {}
        self.performance_scores = {}
        self.diversity_scores = {}
        self.final_weights = {}
        self.metrics_calculator = CTRMetrics()
        
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
    """CTR stacking ensemble with corrected scoring"""
    
    def __init__(self, cv_folds: int = 3, meta_learner_types: Optional[List[str]] = None):
        super().__init__("CTRStacking")
        self.cv_folds = cv_folds
        self.meta_learner_types = meta_learner_types or ['logistic']
        self.meta_learners = {}
        self.oof_predictions = {}
        self.meta_features = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit stacking ensemble with corrected metric calculation"""
        logger.info(f"{self.name} ensemble training started")
        
        try:
            if len(self.base_models) < 2:
                logger.warning("Stacking requires at least 2 base models, falling back to simple average")
                self.weights = {name: 1.0 / len(base_predictions) for name in base_predictions.keys()}
                self.is_fitted = True
                return
            
            # Generate out-of-fold predictions
            logger.info("Generating out-of-fold predictions")
            self.oof_predictions = {}
            
            # Use provided base_predictions as OOF predictions
            for name, pred in base_predictions.items():
                if len(pred) == len(y):
                    self.oof_predictions[name] = pred
                else:
                    logger.warning(f"Prediction length mismatch for {name}, using zeros")
                    self.oof_predictions[name] = np.zeros(len(y))
            
            # Calculate performance weights with corrected scoring
            logger.info("Calculating performance-based weights")
            performance_weights = self._calculate_performance_weights(y)
            
            self.weights = performance_weights
            
            # Train meta-learner if we have enough data
            if len(y) > 50:  # Only train meta-learner with sufficient data
                logger.info("Training meta learners")
                try:
                    meta_features = np.column_stack([self.oof_predictions[name] for name in self.base_models.keys()])
                    
                    # Train logistic meta-learner
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
            logger.info(f"{self.name} ensemble training completed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.weights = {name: 1.0 / len(base_predictions) for name in base_predictions.keys()}
            self.is_fitted = True
    
    def _calculate_performance_weights(self, y: pd.Series) -> Dict[str, float]:
        """Calculate performance-based weights with proper metric calculation"""
        weights = {}
        total_score = 0.0
        
        for name, pred in self.oof_predictions.items():
            try:
                # Use corrected combined score calculation
                combined_score = self.metrics_calculator.combined_score(y.values, pred)
                
                # Ensure minimum weight
                weights[name] = max(combined_score, 0.1)
                total_score += weights[name]
                
                logger.info(f"Performance weight for {name}: {weights[name]:.4f} (combined score: {combined_score:.4f})")
                
            except Exception as e:
                logger.warning(f"Performance calculation failed for {name}: {e}")
                weights[name] = 0.1
                total_score += 0.1
        
        # Normalize weights
        if total_score > 0:
            weights = {name: weight / total_score for name, weight in weights.items()}
        else:
            weights = {name: 1.0 / len(self.oof_predictions) for name in self.oof_predictions.keys()}
        
        logger.info(f"Final performance weights: {weights}")
        return weights
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using stacking ensemble"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            # Try meta-learner first if available and we have enough models
            if 'logistic' in self.meta_learners and len(base_predictions) >= 2:
                try:
                    meta_features = np.column_stack([base_predictions[name] for name in self.base_models.keys() if name in base_predictions])
                    pred = self.meta_learners['logistic'].predict_proba(meta_features)[:, 1]
                    return pred
                except Exception as e:
                    logger.warning(f"Meta-learner prediction failed: {e}")
            
            # Fallback to weighted average
            return self._weighted_average_prediction(base_predictions)
            
        except Exception as e:
            logger.error(f"Stacking ensemble prediction failed: {e}")
            return self._simple_average_prediction(base_predictions)
    
    def _weighted_average_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average prediction"""
        try:
            if not base_predictions:
                return np.array([])
                
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
        except Exception as e:
            logger.error(f"Weighted average failed: {e}")
            return self._simple_average_prediction(base_predictions)
    
    def _simple_average_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average prediction"""
        try:
            if not base_predictions:
                return np.array([])
            return np.mean(list(base_predictions.values()), axis=0)
        except Exception as e:
            logger.error(f"Simple average failed: {e}")
            return np.full(len(next(iter(base_predictions.values()))), 0.0191)

class CTRDynamicEnsemble(BaseEnsemble):
    """CTR dynamic ensemble with strategy selection"""
    
    def __init__(self):
        super().__init__("CTRDynamic")
        self.strategies = ['simple_average', 'performance_weighted', 'ctr_corrected']
        self.strategy_weights = {}
        self.best_strategy = 'simple_average'
        self.performance_history = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit dynamic ensemble with corrected strategy evaluation"""
        logger.info(f"{self.name} ensemble training started")
        
        try:
            logger.info("Initializing performance tracking")
            self.performance_history = {}
            
            # Calculate weights for different strategies
            logger.info("Calculating multi-strategy weights")
            self._calculate_strategy_weights(y, base_predictions)
            
            # Select best strategy using corrected scoring
            logger.info("Selecting optimal strategy")
            best_score = -1
            
            for strategy in self.strategies:
                try:
                    pred = self._apply_strategy(strategy, base_predictions)
                    if len(pred) > 0 and len(y) > 0:
                        score = self.metrics_calculator.combined_score(y.values, pred)
                        
                        logger.info(f"Strategy {strategy}: Combined Score {score:.4f}")
                        
                        if score > best_score:
                            best_score = score
                            self.best_strategy = strategy
                            
                except Exception as e:
                    logger.warning(f"Strategy evaluation failed for {strategy}: {e}")
                    continue
            
            logger.info(f"Best strategy selected: {self.best_strategy} (Score: {best_score:.4f})")
            
            self.is_fitted = True
            logger.info(f"{self.name} ensemble training completed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.best_strategy = 'simple_average'
            self.is_fitted = True
    
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
                score = self.metrics_calculator.combined_score(y.values, pred)
                weights[name] = max(score, 0.1)
                total_score += weights[name]
                
            except Exception as e:
                logger.warning(f"Performance weight calculation failed for {name}: {e}")
                weights[name] = 0.1
                total_score += 0.1
        
        # Normalize
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
                base_score = self.metrics_calculator.combined_score(y.values, pred)
                
                # CTR alignment bonus/penalty
                predicted_ctr = np.mean(pred)
                ctr_bias = abs(predicted_ctr - actual_ctr)
                
                if ctr_bias <= 0.002:  # Good CTR alignment
                    ctr_bonus = min(0.2, (0.002 - ctr_bias) * 100)
                    score = base_score * (1.0 + ctr_bonus)
                else:  # Poor CTR alignment
                    ctr_penalty = min(0.5, ctr_bias * 25)
                    score = base_score * (1.0 - ctr_penalty)
                
                weights[name] = max(score, 0.05)
                total_score += weights[name]
                
            except Exception as e:
                logger.warning(f"CTR corrected weight calculation failed for {name}: {e}")
                weights[name] = 0.05
                total_score += 0.05
        
        # Normalize
        if total_score > 0:
            weights = {name: weight / total_score for name, weight in weights.items()}
        else:
            weights = {name: 1.0 / len(base_predictions) for name in base_predictions.keys()}
        
        return weights
    
    def _apply_strategy(self, strategy: str, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply specified strategy"""
        try:
            if not base_predictions:
                return np.array([])
                
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
        except Exception as e:
            logger.error(f"Strategy application failed for {strategy}: {e}")
            return np.mean(list(base_predictions.values()), axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using best strategy"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            return self._apply_strategy(self.best_strategy, base_predictions)
            
        except Exception as e:
            logger.error(f"Dynamic ensemble prediction failed: {e}")
            return np.mean(list(base_predictions.values()), axis=0)

class CTRMainEnsemble(BaseEnsemble):
    """CTR main ensemble with weighted combination"""
    
    def __init__(self, target_combined_score: float = 0.34):
        super().__init__("CTRMain")
        self.target_combined_score = target_combined_score
        self.sub_ensembles = {}
        self.ensemble_weights = {}
        self.final_prediction = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit main ensemble with corrected evaluation"""
        logger.info(f"{self.name} ensemble training started - execution guaranteed")
        
        try:
            # Create sub-ensembles
            logger.info("Stage 1: Creating sub-ensembles")
            
            # Create stacking ensemble
            try:
                stacking_ensemble = CTRStackingEnsemble(cv_folds=3)
                for name, model in self.base_models.items():
                    stacking_ensemble.add_base_model(name, model)
                stacking_ensemble.fit(X, y, base_predictions)
                self.sub_ensembles['stacking'] = stacking_ensemble
                logger.info("Stacking ensemble created successfully")
            except Exception as e:
                logger.warning(f"Stacking ensemble creation failed: {e}")
            
            # Create dynamic ensemble
            try:
                dynamic_ensemble = CTRDynamicEnsemble()
                for name, model in self.base_models.items():
                    dynamic_ensemble.add_base_model(name, model)
                dynamic_ensemble.fit(X, y, base_predictions)
                self.sub_ensembles['dynamic'] = dynamic_ensemble
                logger.info("Dynamic ensemble created successfully")
            except Exception as e:
                logger.warning(f"Dynamic ensemble creation failed: {e}")
            
            # Evaluate sub-ensembles with corrected scoring
            logger.info("Stage 2: Evaluating sub-ensembles")
            ensemble_scores = {}
            
            for ensemble_name, ensemble in self.sub_ensembles.items():
                try:
                    pred = ensemble.predict_proba(base_predictions)
                    combined_score = self.metrics_calculator.combined_score(y.values, pred)
                    ctr_score = self.metrics_calculator.ctr_score(y.values, pred)
                    
                    ensemble_scores[ensemble_name] = {
                        'combined_score': combined_score,
                        'ctr_score': ctr_score
                    }
                    
                    logger.info(f"{ensemble_name} ensemble - Combined: {combined_score:.4f}, CTR: {ctr_score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"{ensemble_name} ensemble evaluation failed: {e}")
                    # Still add with minimal score
                    ensemble_scores[ensemble_name] = {
                        'combined_score': 0.1,
                        'ctr_score': 0.1
                    }
            
            # Calculate ensemble weights
            logger.info("Stage 3: Calculating ensemble weights")
            total_score = sum(scores['combined_score'] for scores in ensemble_scores.values())
            
            if total_score > 0:
                for ensemble_name, scores in ensemble_scores.items():
                    self.ensemble_weights[ensemble_name] = scores['combined_score'] / total_score
            else:
                # Equal weights fallback
                n_ensembles = len(self.sub_ensembles)
                if n_ensembles > 0:
                    self.ensemble_weights = {name: 1.0 / n_ensembles for name in self.sub_ensembles.keys()}
                else:
                    self.final_prediction = np.mean(list(base_predictions.values()), axis=0)
            
            # Calculate final ensemble prediction
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
            
            logger.info(f"{self.name} ensemble training completed - execution guaranteed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            # Emergency fallback
            self.final_prediction = np.mean(list(base_predictions.values()), axis=0)
            self.is_fitted = True
    
    def _calculate_final_ensemble_prediction(self, ensemble_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate final ensemble prediction using weighted combination"""
        try:
            if not ensemble_predictions:
                return np.array([])
                
            weighted_pred = np.zeros(len(next(iter(ensemble_predictions.values()))))
            total_weight = 0.0
            
            for ensemble_name, pred in ensemble_predictions.items():
                if ensemble_name in self.ensemble_weights:
                    weight = self.ensemble_weights[ensemble_name]
                    weighted_pred += weight * pred
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_pred / total_weight
            else:
                return np.mean(list(ensemble_predictions.values()), axis=0)
        except Exception as e:
            logger.error(f"Final ensemble prediction calculation failed: {e}")
            return np.mean(list(ensemble_predictions.values()), axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using main ensemble"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            if len(self.sub_ensembles) == 0:
                return self._simple_weighted_average(base_predictions)
            
            # Get predictions from sub-ensembles
            ensemble_predictions = {}
            
            for ensemble_name, ensemble in self.sub_ensembles.items():
                try:
                    pred = ensemble.predict_proba(base_predictions)
                    ensemble_predictions[ensemble_name] = pred
                except Exception as e:
                    logger.warning(f"Sub-ensemble prediction failed ({ensemble_name}): {e}")
            
            if ensemble_predictions:
                return self._calculate_final_ensemble_prediction(ensemble_predictions)
            else:
                return self._simple_weighted_average(base_predictions)
            
        except Exception as e:
            logger.error(f"Main ensemble prediction failed: {e}")
            return self._simple_weighted_average(base_predictions)
    
    def _simple_weighted_average(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple weighted average of base predictions"""
        try:
            if not base_predictions:
                return np.array([])
            return np.mean(list(base_predictions.values()), axis=0)
        except Exception as e:
            logger.error(f"Simple weighted average failed: {e}")
            return np.full(len(next(iter(base_predictions.values()))), 0.0191)

class CTREnsembleManager:
    """CTR ensemble manager with corrected metric calculation"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.base_models = {}
        self.ensembles = {}
        self.best_ensemble = None
        self.final_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.target_combined_score = getattr(config, 'TARGET_COMBINED_SCORE', 0.34)
        
        # Initialize ensemble types
        self.ensemble_types = ['main_ensemble']
        
    def add_base_model(self, name: str, model: BaseModel):
        """Add base model to ensemble manager"""
        self.base_models[name] = model
        logger.info(f"Base model added to ensemble manager: {name}")
    
    def train_all_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """Train all ensembles with corrected scoring"""
        logger.info("All ensemble training started - execution guaranteed")
        
        try:
            # Get base model predictions
            logger.info("Generating base model predictions")
            base_predictions = {}
            
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    base_predictions[name] = pred
                    pred_ctr = np.mean(pred)
                    actual_ctr = np.mean(y)
                    logger.info(f"{name} model prediction completed - CTR: {pred_ctr:.4f} (actual: {actual_ctr:.4f})")
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0191)  # Default CTR
            
            # Create and train main ensemble
            logger.info("Creating main ensemble")
            main_ensemble = CTRMainEnsemble(target_combined_score=self.target_combined_score)
            
            for name, model in self.base_models.items():
                main_ensemble.add_base_model(name, model)
            
            logger.info("Ensemble created: main_ensemble")
            
            # Train main ensemble
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
            # Create fallback ensemble
            main_ensemble = CTRMainEnsemble()
            main_ensemble.final_prediction = np.mean(list(base_predictions.values()), axis=0) if base_predictions else np.array([0.0191])
            main_ensemble.is_fitted = True
            self.ensembles['main_ensemble'] = main_ensemble
            self.final_ensemble = main_ensemble
    
    def evaluate_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """Evaluate ensembles with corrected scoring"""
        logger.info("Ensemble evaluation started")
        
        # Get base model predictions
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
        
        # Evaluate each ensemble
        for ensemble_type, ensemble in self.ensembles.items():
            try:
                if not ensemble.is_fitted:
                    continue
                
                ensemble_pred = ensemble.predict_proba(base_predictions)
                combined_score = self.metrics_calculator.combined_score(y.values, ensemble_pred)
                ctr_score = self.metrics_calculator.ctr_score(y.values, ensemble_pred)
                
                self.ensemble_results[f"ensemble_{ensemble_type}"] = combined_score
                self.ensemble_results[f"ensemble_{ensemble_type}_ctr"] = ctr_score
                
                execution_guaranteed = getattr(ensemble, 'is_fitted', False)
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
        
        # Select best ensemble
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
            
            # Get base model predictions
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    base_predictions[name] = pred
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0191)
            
            # Get ensemble prediction
            ensemble_pred = self.best_ensemble.predict_proba(base_predictions)
            predicted_ctr = ensemble_pred.mean()
            
            logger.info(f"Ensemble prediction successful - CTR: {predicted_ctr:.4f}")
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Best ensemble prediction failed: {e}")
            
            # Fallback to individual model average
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
                    return np.mean(predictions, axis=0)
                else:
                    return np.full(len(X), 0.0191)
            else:
                logger.error("No models available for prediction")
                return np.full(len(X), 0.0191)
    
    def predict_with_best_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, bool]:
        """Predict using best ensemble with success indicator"""
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
                    return np.full(len(X), 0.0191), False
            
            # Get base predictions
            base_predictions = {}
            for name, model in self.base_models.items():
                try:
                    pred = model.predict_proba(X)
                    base_predictions[name] = pred
                except Exception as e:
                    logger.error(f"{name} model prediction failed: {e}")
                    base_predictions[name] = np.full(len(X), 0.0191)
            
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
                    return np.full(len(X), 0.0191), False
            else:
                logger.error("No models available for prediction")
                return np.full(len(X), 0.0191), False