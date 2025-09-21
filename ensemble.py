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
from evaluation import CTRMetrics

logger = logging.getLogger(__name__)

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
    
    def __init__(self, cv_folds: int = 4, meta_learner_types: Optional[List[str]] = None):
        super().__init__("CTRStacking")
        self.cv_folds = cv_folds
        self.meta_learner_types = meta_learner_types or ['logistic', 'random_forest']
        self.meta_learners = {}
        self.oof_predictions = {}
        self.meta_features = None
        self.weight_calculation_methods = ['performance', 'diversity', 'calibration', 'combined']
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit stacking ensemble with multiple weight calculation methods"""
        logger.info(f"{self.name} ensemble training started")
        
        try:
            if len(self.base_models) < 2:
                raise ValueError("Stacking requires at least 2 base models")
            
            # Generate out-of-fold predictions
            self._generate_oof_predictions(X, y)
            
            # Calculate multiple types of weights
            self._calculate_performance_weights(y)
            self._calculate_diversity_weights()
            self._calculate_calibration_weights(y)
            self._calculate_combined_weights()
            
            # Train meta learners with different weight strategies
            self._train_meta_learners(y)
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            
            logger.info(f"{self.name} ensemble training completed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.ensemble_execution_guaranteed = False
            raise
    
    def _generate_oof_predictions(self, X: pd.DataFrame, y: pd.Series):
        """Generate out-of-fold predictions"""
        logger.info("Generating out-of-fold predictions")
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.base_models.items():
            oof_pred = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                try:
                    X_fold_train = X.iloc[train_idx]
                    y_fold_train = y.iloc[train_idx]
                    X_fold_val = X.iloc[val_idx]
                    
                    # Clone and train model on fold
                    fold_model = self._clone_model(model)
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Predict on validation fold
                    fold_pred = fold_model.predict_proba(X_fold_val)
                    oof_pred[val_idx] = fold_pred
                    
                except Exception as e:
                    logger.warning(f"Fold {fold} failed for {name}: {e}")
                    oof_pred[val_idx] = np.full(len(val_idx), y.mean())
            
            self.oof_predictions[name] = oof_pred
        
        # Create meta features matrix
        self.meta_features = np.column_stack([
            self.oof_predictions[name] for name in self.base_models.keys()
        ])
    
    def _clone_model(self, model: BaseModel) -> BaseModel:
        """Clone model for cross-validation"""
        try:
            model_type = type(model).__name__
            
            if hasattr(model, 'params'):
                params = model.params.copy()
            else:
                params = {}
            
            # Create new model instance
            from models import ModelFactory
            cloned_model = ModelFactory.create_model(model_type.replace('Model', '').lower(), params=params)
            
            return cloned_model
            
        except Exception as e:
            logger.warning(f"Model cloning failed: {e}")
            return model
    
    def _calculate_performance_weights(self, y: pd.Series):
        """Calculate performance-based weights"""
        logger.info("Calculating performance-based weights")
        
        performance_weights = {}
        
        for name, pred in self.oof_predictions.items():
            try:
                # Multiple performance metrics
                auc_score = roc_auc_score(y, pred)
                ap_score = average_precision_score(y, pred)
                logloss_score = log_loss(y, pred)
                
                # Combined performance score (lower logloss is better)
                performance_score = 0.4 * auc_score + 0.4 * ap_score + 0.2 * (1 - min(logloss_score, 1.0))
                
                performance_weights[name] = performance_score
                self.performance_scores[name] = {
                    'auc': auc_score,
                    'ap': ap_score,
                    'logloss': logloss_score,
                    'combined': performance_score
                }
                
            except Exception as e:
                logger.warning(f"Performance calculation failed for {name}: {e}")
                performance_weights[name] = 0.1
                self.performance_scores[name] = {'combined': 0.1}
        
        # Normalize weights
        total_weight = sum(performance_weights.values())
        if total_weight > 0:
            self.weights['performance'] = {
                name: weight / total_weight for name, weight in performance_weights.items()
            }
        else:
            # Equal weights fallback
            n_models = len(self.base_models)
            self.weights['performance'] = {name: 1.0 / n_models for name in self.base_models.keys()}
    
    def _calculate_diversity_weights(self):
        """Calculate diversity-based weights"""
        logger.info("Calculating diversity-based weights")
        
        diversity_weights = {}
        model_names = list(self.oof_predictions.keys())
        
        for name in model_names:
            diversity_score = 0.0
            count = 0
            
            # Calculate pairwise diversity with other models
            for other_name in model_names:
                if name != other_name:
                    try:
                        pred1 = self.oof_predictions[name]
                        pred2 = self.oof_predictions[other_name]
                        
                        # Correlation-based diversity (lower correlation = higher diversity)
                        correlation = np.corrcoef(pred1, pred2)[0, 1]
                        diversity_score += (1 - abs(correlation))
                        count += 1
                        
                    except Exception as e:
                        logger.warning(f"Diversity calculation failed between {name} and {other_name}: {e}")
            
            if count > 0:
                diversity_weights[name] = diversity_score / count
                self.diversity_scores[name] = diversity_score / count
            else:
                diversity_weights[name] = 0.5
                self.diversity_scores[name] = 0.5
        
        # Normalize diversity weights
        total_weight = sum(diversity_weights.values())
        if total_weight > 0:
            self.weights['diversity'] = {
                name: weight / total_weight for name, weight in diversity_weights.items()
            }
        else:
            n_models = len(self.base_models)
            self.weights['diversity'] = {name: 1.0 / n_models for name in self.base_models.keys()}
    
    def _calculate_calibration_weights(self, y: pd.Series):
        """Calculate calibration-based weights"""
        logger.info("Calculating calibration-based weights")
        
        calibration_weights = {}
        
        for name, pred in self.oof_predictions.items():
            try:
                # Bin predictions and calculate calibration error
                n_bins = 10
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                calibration_error = 0.0
                total_samples = 0
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (pred > bin_lower) & (pred <= bin_upper)
                    prop_in_bin = in_bin.sum()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y[in_bin].mean()
                        avg_confidence_in_bin = pred[in_bin].mean()
                        
                        calibration_error += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
                        total_samples += prop_in_bin
                
                if total_samples > 0:
                    calibration_error /= total_samples
                    calibration_score = 1.0 - calibration_error  # Higher score for better calibration
                else:
                    calibration_score = 0.5
                
                calibration_weights[name] = max(calibration_score, 0.1)
                self.calibration_scores[name] = calibration_score
                
            except Exception as e:
                logger.warning(f"Calibration calculation failed for {name}: {e}")
                calibration_weights[name] = 0.5
                self.calibration_scores[name] = 0.5
        
        # Normalize calibration weights
        total_weight = sum(calibration_weights.values())
        if total_weight > 0:
            self.weights['calibration'] = {
                name: weight / total_weight for name, weight in calibration_weights.items()
            }
        else:
            n_models = len(self.base_models)
            self.weights['calibration'] = {name: 1.0 / n_models for name in self.base_models.keys()}
    
    def _calculate_combined_weights(self):
        """Calculate combined weights using multiple strategies"""
        logger.info("Calculating combined weights")
        
        combined_weights = {}
        model_names = list(self.base_models.keys())
        
        # Weight combination coefficients
        performance_coeff = 0.5
        diversity_coeff = 0.3
        calibration_coeff = 0.2
        
        for name in model_names:
            performance_weight = self.weights.get('performance', {}).get(name, 0.33)
            diversity_weight = self.weights.get('diversity', {}).get(name, 0.33)
            calibration_weight = self.weights.get('calibration', {}).get(name, 0.33)
            
            combined_score = (
                performance_coeff * performance_weight +
                diversity_coeff * diversity_weight +
                calibration_coeff * calibration_weight
            )
            
            combined_weights[name] = combined_score
        
        # Normalize combined weights
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            self.weights['combined'] = {
                name: weight / total_weight for name, weight in combined_weights.items()
            }
        else:
            n_models = len(self.base_models)
            self.weights['combined'] = {name: 1.0 / n_models for name in self.base_models.keys()}
        
        # Set final weights to combined weights
        self.final_weights = self.weights['combined'].copy()
    
    def _train_meta_learners(self, y: pd.Series):
        """Train meta learners"""
        logger.info("Training meta learners")
        
        for learner_type in self.meta_learner_types:
            try:
                if learner_type == 'logistic':
                    meta_learner = LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                        solver='liblinear'
                    )
                elif learner_type == 'random_forest':
                    meta_learner = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        max_depth=5,
                        min_samples_split=10
                    )
                else:
                    continue
                
                meta_learner.fit(self.meta_features, y)
                self.meta_learners[learner_type] = meta_learner
                
                logger.info(f"Meta learner trained: {learner_type}")
                
            except Exception as e:
                logger.warning(f"Meta learner training failed ({learner_type}): {e}")
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using stacking ensemble"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            # Create meta features from base predictions
            meta_features = np.column_stack([
                base_predictions[name] for name in self.base_models.keys()
                if name in base_predictions
            ])
            
            # Use primary meta learner (logistic by default)
            primary_learner = 'logistic'
            if primary_learner in self.meta_learners:
                predictions = self.meta_learners[primary_learner].predict_proba(meta_features)[:, 1]
            else:
                # Fallback to weighted average
                predictions = self._weighted_average_prediction(base_predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Stacking prediction failed: {e}")
            return self._weighted_average_prediction(base_predictions)
    
    def _weighted_average_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average prediction fallback"""
        weighted_pred = np.zeros(len(next(iter(base_predictions.values()))))
        total_weight = 0.0
        
        for name, pred in base_predictions.items():
            if name in self.final_weights:
                weight = self.final_weights[name]
                weighted_pred += weight * pred
                total_weight += weight
        
        if total_weight > 0:
            return weighted_pred / total_weight
        else:
            return np.mean(list(base_predictions.values()), axis=0)

class CTRDynamicEnsemble(BaseEnsemble):
    """CTR dynamic ensemble with multiple weighting strategies"""
    
    def __init__(self, weighting_method: str = 'multi_strategy'):
        super().__init__("CTRDynamic")
        self.weighting_method = weighting_method
        self.dynamic_weights = {}
        self.performance_history = {}
        self.weight_strategies = ['performance', 'diversity', 'calibration', 'temporal', 'adaptive']
        self.strategy_weights = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit dynamic ensemble with multiple weighting strategies"""
        logger.info(f"{self.name} ensemble training started")
        
        try:
            if len(self.base_models) < 2:
                raise ValueError("Dynamic ensemble requires at least 2 base models")
            
            # Initialize performance tracking
            self._initialize_performance_tracking(y, base_predictions)
            
            # Calculate weights using multiple strategies
            self._calculate_multi_strategy_weights(y, base_predictions)
            
            # Determine optimal strategy combination
            self._optimize_strategy_combination(y, base_predictions)
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            
            logger.info(f"{self.name} ensemble training completed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.ensemble_execution_guaranteed = False
            raise
    
    def _initialize_performance_tracking(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Initialize performance tracking for dynamic weighting"""
        logger.info("Initializing performance tracking")
        
        for name, pred in base_predictions.items():
            try:
                performance_metrics = {
                    'auc': roc_auc_score(y, pred),
                    'ap': average_precision_score(y, pred),
                    'logloss': log_loss(y, pred),
                    'prediction_variance': np.var(pred),
                    'prediction_mean': np.mean(pred),
                    'calibration_score': self._calculate_calibration_score(y, pred)
                }
                
                self.performance_history[name] = performance_metrics
                
            except Exception as e:
                logger.warning(f"Performance tracking initialization failed for {name}: {e}")
                self.performance_history[name] = {
                    'auc': 0.5, 'ap': 0.1, 'logloss': 1.0,
                    'prediction_variance': 0.1, 'prediction_mean': 0.02,
                    'calibration_score': 0.5
                }
    
    def _calculate_calibration_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate calibration score"""
        try:
            n_bins = 5
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            calibration_error = 0.0
            total_samples = 0
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
                prop_in_bin = in_bin.sum()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred[in_bin].mean()
                    
                    calibration_error += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
                    total_samples += prop_in_bin
            
            if total_samples > 0:
                calibration_error /= total_samples
                return 1.0 - calibration_error
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_multi_strategy_weights(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Calculate weights using multiple strategies"""
        logger.info("Calculating multi-strategy weights")
        
        self.dynamic_weights = {}
        
        # Strategy 1: Performance-based weights
        self._calculate_performance_weights_dynamic(y, base_predictions)
        
        # Strategy 2: Diversity-based weights
        self._calculate_diversity_weights_dynamic(base_predictions)
        
        # Strategy 3: Calibration-based weights
        self._calculate_calibration_weights_dynamic(y, base_predictions)
        
        # Strategy 4: Temporal weights (based on prediction stability)
        self._calculate_temporal_weights_dynamic(base_predictions)
        
        # Strategy 5: Adaptive weights (combination of all factors)
        self._calculate_adaptive_weights_dynamic()
    
    def _calculate_performance_weights_dynamic(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Calculate performance-based dynamic weights"""
        performance_weights = {}
        
        for name, pred in base_predictions.items():
            metrics = self.performance_history[name]
            
            # Multi-metric performance score
            performance_score = (
                0.3 * metrics['auc'] +
                0.3 * metrics['ap'] +
                0.2 * (1 - min(metrics['logloss'], 2.0) / 2.0) +
                0.2 * metrics['calibration_score']
            )
            
            performance_weights[name] = performance_score
        
        # Normalize weights
        total_weight = sum(performance_weights.values())
        if total_weight > 0:
            self.dynamic_weights['performance'] = {
                name: weight / total_weight for name, weight in performance_weights.items()
            }
        else:
            n_models = len(base_predictions)
            self.dynamic_weights['performance'] = {name: 1.0 / n_models for name in base_predictions.keys()}
    
    def _calculate_diversity_weights_dynamic(self, base_predictions: Dict[str, np.ndarray]):
        """Calculate diversity-based dynamic weights"""
        diversity_weights = {}
        model_names = list(base_predictions.keys())
        
        for name in model_names:
            diversity_score = 0.0
            count = 0
            
            for other_name in model_names:
                if name != other_name:
                    try:
                        pred1 = base_predictions[name]
                        pred2 = base_predictions[other_name]
                        
                        # Multiple diversity measures
                        correlation_diversity = 1 - abs(np.corrcoef(pred1, pred2)[0, 1])
                        variance_diversity = abs(np.var(pred1) - np.var(pred2)) / (np.var(pred1) + np.var(pred2) + 1e-8)
                        
                        diversity_score += 0.7 * correlation_diversity + 0.3 * variance_diversity
                        count += 1
                        
                    except Exception:
                        diversity_score += 0.5
                        count += 1
            
            if count > 0:
                diversity_weights[name] = diversity_score / count
            else:
                diversity_weights[name] = 0.5
        
        # Normalize weights
        total_weight = sum(diversity_weights.values())
        if total_weight > 0:
            self.dynamic_weights['diversity'] = {
                name: weight / total_weight for name, weight in diversity_weights.items()
            }
        else:
            n_models = len(base_predictions)
            self.dynamic_weights['diversity'] = {name: 1.0 / n_models for name in base_predictions.keys()}
    
    def _calculate_calibration_weights_dynamic(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Calculate calibration-based dynamic weights"""
        calibration_weights = {}
        
        for name, pred in base_predictions.items():
            calibration_score = self.performance_history[name]['calibration_score']
            calibration_weights[name] = calibration_score
        
        # Normalize weights
        total_weight = sum(calibration_weights.values())
        if total_weight > 0:
            self.dynamic_weights['calibration'] = {
                name: weight / total_weight for name, weight in calibration_weights.items()
            }
        else:
            n_models = len(base_predictions)
            self.dynamic_weights['calibration'] = {name: 1.0 / n_models for name in base_predictions.keys()}
    
    def _calculate_temporal_weights_dynamic(self, base_predictions: Dict[str, np.ndarray]):
        """Calculate temporal stability-based weights"""
        temporal_weights = {}
        
        for name, pred in base_predictions.items():
            # Stability measure based on prediction variance and distribution
            variance = self.performance_history[name]['prediction_variance']
            mean_pred = self.performance_history[name]['prediction_mean']
            
            # Lower variance and reasonable mean indicates stability
            stability_score = 1.0 / (1.0 + variance) * min(mean_pred / 0.02, 1.0)
            temporal_weights[name] = stability_score
        
        # Normalize weights
        total_weight = sum(temporal_weights.values())
        if total_weight > 0:
            self.dynamic_weights['temporal'] = {
                name: weight / total_weight for name, weight in temporal_weights.items()
            }
        else:
            n_models = len(base_predictions)
            self.dynamic_weights['temporal'] = {name: 1.0 / n_models for name in base_predictions.keys()}
    
    def _calculate_adaptive_weights_dynamic(self):
        """Calculate adaptive weights combining all strategies"""
        adaptive_weights = {}
        model_names = list(self.base_models.keys())
        
        # Strategy combination coefficients
        strategy_coeffs = {
            'performance': 0.35,
            'diversity': 0.25,
            'calibration': 0.25,
            'temporal': 0.15
        }
        
        for name in model_names:
            adaptive_score = 0.0
            
            for strategy, coeff in strategy_coeffs.items():
                if strategy in self.dynamic_weights:
                    weight = self.dynamic_weights[strategy].get(name, 1.0 / len(model_names))
                    adaptive_score += coeff * weight
            
            adaptive_weights[name] = adaptive_score
        
        # Normalize weights
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            self.dynamic_weights['adaptive'] = {
                name: weight / total_weight for name, weight in adaptive_weights.items()
            }
        else:
            n_models = len(model_names)
            self.dynamic_weights['adaptive'] = {name: 1.0 / n_models for name in model_names}
    
    def _optimize_strategy_combination(self, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Optimize strategy combination using validation performance"""
        logger.info("Optimizing strategy combination")
        
        best_score = -1
        best_strategy = 'adaptive'
        
        for strategy_name, weights in self.dynamic_weights.items():
            try:
                # Calculate ensemble prediction using this strategy
                ensemble_pred = self._calculate_weighted_prediction(base_predictions, weights)
                
                # Evaluate performance
                auc_score = roc_auc_score(y, ensemble_pred)
                ap_score = average_precision_score(y, ensemble_pred)
                logloss_score = log_loss(y, ensemble_pred)
                
                # Combined score
                combined_score = 0.4 * auc_score + 0.4 * ap_score + 0.2 * (1 - min(logloss_score, 1.0))
                
                logger.info(f"Strategy {strategy_name}: Combined Score {combined_score:.4f}")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_strategy = strategy_name
                
                self.strategy_weights[strategy_name] = combined_score
                
            except Exception as e:
                logger.warning(f"Strategy evaluation failed ({strategy_name}): {e}")
                self.strategy_weights[strategy_name] = 0.1
        
        # Set final weights to best strategy
        self.final_weights = self.dynamic_weights[best_strategy].copy()
        
        logger.info(f"Best strategy selected: {best_strategy} (Score: {best_score:.4f})")
    
    def _calculate_weighted_prediction(self, base_predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """Calculate weighted prediction"""
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
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using dynamic ensemble"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            return self._calculate_weighted_prediction(base_predictions, self.final_weights)
            
        except Exception as e:
            logger.error(f"Dynamic ensemble prediction failed: {e}")
            return np.mean(list(base_predictions.values()), axis=0)

class CTRMainEnsemble(BaseEnsemble):
    """CTR main ensemble with multiple optimization strategies"""
    
    def __init__(self, target_ctr: float = 0.0191, optimization_method: str = 'multi_strategy'):
        super().__init__("CTRMain")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.sub_ensembles = {}
        self.ensemble_weights = {}
        self.is_calibrated = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Fit main ensemble with comprehensive weighting strategies"""
        logger.info(f"{self.name} ensemble training started - execution guaranteed")
        
        try:
            if len(self.base_models) < 1:
                raise ValueError("Main ensemble requires at least 1 base model")
            
            # Create sub-ensembles with different strategies
            logger.info("Stage 1: Creating sub-ensembles")
            
            # Stacking ensemble
            try:
                stacking_ensemble = CTRStackingEnsemble(cv_folds=4)
                for name, model in self.base_models.items():
                    stacking_ensemble.add_base_model(name, model)
                stacking_ensemble.fit(X, y, base_predictions)
                self.sub_ensembles['stacking'] = stacking_ensemble
                logger.info("Stacking ensemble created successfully")
            except Exception as e:
                logger.warning(f"Stacking ensemble creation failed: {e}")
            
            # Dynamic ensemble
            try:
                dynamic_ensemble = CTRDynamicEnsemble(weighting_method='multi_strategy')
                for name, model in self.base_models.items():
                    dynamic_ensemble.add_base_model(name, model)
                dynamic_ensemble.fit(X, y, base_predictions)
                self.sub_ensembles['dynamic'] = dynamic_ensemble
                logger.info("Dynamic ensemble created successfully")
            except Exception as e:
                logger.warning(f"Dynamic ensemble creation failed: {e}")
            
            # Evaluate sub-ensembles
            logger.info("Stage 2: Evaluating sub-ensembles")
            ensemble_predictions = {}
            ensemble_scores = {}
            
            metrics_calculator = CTRMetrics()
            
            for ensemble_name, ensemble in self.sub_ensembles.items():
                try:
                    pred = ensemble.predict_proba(base_predictions)
                    ensemble_predictions[ensemble_name] = pred
                    
                    combined_score = metrics_calculator.combined_score(y.values, pred)
                    ctr_score = metrics_calculator.ctr_score(y.values, pred)
                    
                    ensemble_scores[ensemble_name] = {
                        'combined_score': combined_score,
                        'ctr_score': ctr_score,
                        'weight_score': 0.7 * combined_score + 0.3 * ctr_score
                    }
                    
                    logger.info(f"{ensemble_name} ensemble - Combined: {combined_score:.4f}, CTR: {ctr_score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"{ensemble_name} ensemble evaluation failed: {e}")
            
            # Calculate ensemble weights
            logger.info("Stage 3: Calculating ensemble weights")
            total_weight_score = sum(scores['weight_score'] for scores in ensemble_scores.values())
            
            if total_weight_score > 0:
                for ensemble_name, scores in ensemble_scores.items():
                    self.ensemble_weights[ensemble_name] = scores['weight_score'] / total_weight_score
            else:
                # Equal weights fallback
                n_ensembles = len(self.sub_ensembles)
                self.ensemble_weights = {name: 1.0 / n_ensembles for name in self.sub_ensembles.keys()}
            
            # Final ensemble prediction
            if ensemble_predictions:
                self.final_prediction = self._calculate_final_ensemble_prediction(ensemble_predictions)
            else:
                # Fallback to simple average
                self.final_prediction = np.mean(list(base_predictions.values()), axis=0)
            
            self.is_fitted = True
            self.ensemble_execution_guaranteed = True
            
            logger.info(f"{self.name} ensemble training completed - execution guaranteed")
            
        except Exception as e:
            logger.error(f"{self.name} ensemble training failed: {e}")
            self.ensemble_execution_guaranteed = False
            raise
    
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
            return weighted_pred / total_weight
        else:
            return np.mean(list(ensemble_predictions.values()), axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities using main ensemble"""
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble not fitted")
            
            if len(self.sub_ensembles) == 0:
                # Fallback to simple weighted average of base predictions
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
        if not base_predictions:
            return np.array([])
        
        return np.mean(list(base_predictions.values()), axis=0)

class CTREnsembleManager:
    """CTR ensemble manager with comprehensive weight calculation"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ensembles = {}
        self.base_models = {}
        self.best_ensemble = None
        self.ensemble_results = {}
        self.metrics_calculator = CTRMetrics()
        self.final_ensemble = None
        self.target_combined_score = 0.34
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
            if ensemble_type == 'main_ensemble':
                target_ctr = kwargs.get('target_ctr', 0.0191)
                optimization_method = kwargs.get('optimization_method', 'multi_strategy')
                ensemble = CTRMainEnsemble(target_ctr, optimization_method)
                self.final_ensemble = ensemble
            
            elif ensemble_type == 'stacking':
                cv_folds = kwargs.get('cv_folds', 4)
                meta_learner_types = kwargs.get('meta_learner_types', None)
                ensemble = CTRStackingEnsemble(cv_folds, meta_learner_types)
            
            elif ensemble_type == 'dynamic':
                weighting_method = kwargs.get('weighting_method', 'multi_strategy')
                ensemble = CTRDynamicEnsemble(weighting_method)
            
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
                base_predictions[name] = np.full(len(X), 0.0191)
                calibration_info[name] = {'calibrated': False, 'method': 'none'}
        
        logger.info("Creating default ensembles")
        if 'main_ensemble' not in self.ensembles:
            self.create_ensemble('main_ensemble')
        
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
                    base_predictions[name] = np.full(len(X), 0.0191)
            
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
                        predictions.append(np.full(len(X), 0.0191))
                
                return np.mean(predictions, axis=0)
            else:
                logger.error("No models available for prediction")
                return np.full(len(X), 0.0191)
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get ensemble status"""
        calibrated_base_models = sum(1 for model in self.base_models.values() 
                                   if hasattr(model, 'is_calibrated') and model.is_calibrated)
        
        calibrated_ensembles = sum(1 for ensemble in self.ensembles.values() 
                                 if hasattr(ensemble, 'is_calibrated') and ensemble.is_calibrated)
        
        guaranteed_ensembles = sum(1 for ensemble in self.ensembles.values() 
                                 if getattr(ensemble, 'ensemble_execution_guaranteed', False))
        
        target_achieved_count = sum(1 for key, score in self.ensemble_results.items()
                                  if key.startswith('ensemble_') and not key.endswith('_ctr') 
                                  and not key.endswith('_execution_guaranteed')
                                  and score >= self.target_combined_score)
        
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
                 if key.startswith('ensemble_') and not key.endswith('_ctr')
                 and not key.endswith('_execution_guaranteed')),
                default=0.0
            ),
            'calibration_analysis': {
                'base_models_calibration_rate': calibrated_base_models / max(len(self.base_models), 1),
                'ensemble_calibration_rate': calibrated_ensembles / max(len(self.ensembles), 1),
                'ensemble_execution_guarantee_rate': guaranteed_ensembles / max(len(self.ensembles), 1)
            }
        }

# Compatibility aliases
CTRSuperEnsembleManager = CTREnsembleManager
EnsembleManager = CTREnsembleManager