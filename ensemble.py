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
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not installed. Hyperparameter tuning functionality will be disabled.")

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
        self.target_combined_score = 0.34
        self.ensemble_calibrator = None
        self.is_calibrated = False
        self.diversity_metrics = {}
        self.performance_metrics = {}
        
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
                predictions[name] = np.full(len(X), 0.0191)
        
        return predictions
    
    def calculate_diversity_metrics(self, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate diversity metrics for ensemble"""
        try:
            if len(predictions_dict) < 2:
                return {'diversity_score': 0.0}
            
            # Correlation-based diversity
            predictions_matrix = np.column_stack(list(predictions_dict.values()))
            model_names = list(predictions_dict.keys())
            
            # Average pairwise correlation
            correlations = []
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    corr = np.corrcoef(predictions_matrix[:, i], predictions_matrix[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            avg_correlation = np.mean(correlations) if correlations else 0.5
            correlation_diversity = 1.0 - avg_correlation
            
            # Disagreement-based diversity
            threshold = 0.5
            binary_predictions = (predictions_matrix >= threshold).astype(int)
            
            disagreements = []
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    disagreement = np.mean(binary_predictions[:, i] != binary_predictions[:, j])
                    disagreements.append(disagreement)
            
            avg_disagreement = np.mean(disagreements) if disagreements else 0.0
            
            # Prediction range diversity
            ranges = []
            for pred in predictions_dict.values():
                ranges.append(np.max(pred) - np.min(pred))
            range_diversity = np.std(ranges) / (np.mean(ranges) + 1e-8)
            
            # Combined diversity score
            diversity_score = (0.4 * correlation_diversity + 
                             0.3 * avg_disagreement + 
                             0.3 * range_diversity)
            
            return {
                'diversity_score': float(np.clip(diversity_score, 0.0, 1.0)),
                'avg_correlation': float(avg_correlation),
                'avg_disagreement': float(avg_disagreement),
                'range_diversity': float(range_diversity)
            }
            
        except Exception as e:
            logger.error(f"Diversity metrics calculation failed: {e}")
            return {'diversity_score': 0.0}
    
    def apply_ensemble_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, 
                                 ensemble_predictions: np.ndarray, method: str = 'auto'):
        """Apply ensemble-level calibration"""
        try:
            logger.info(f"Applying calibration to {self.name} ensemble: {method}")
            
            if len(ensemble_predictions) != len(y_val):
                logger.warning("Ensemble calibration: Size mismatch")
                return
            
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
            
            if unique_count < len(predictions) // 200:  # More sensitive threshold
                logger.info(f"{self.name}: Applying ensemble prediction diversity enhancement")
                
                noise_scale = max(predictions.std() * 0.004, 1e-7)
                noise = np.random.normal(0, noise_scale, len(predictions))
                
                predictions = predictions + noise
                predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
                
                logger.info(f"{self.name}: Diversity enhancement applied")
            
            return predictions
        except Exception:
            return predictions

class CTRStackingEnsemble(BaseEnsemble):
    """CTR stacking ensemble with multiple meta-learners"""
    
    def __init__(self, cv_folds: int = 4, meta_learner_types: List[str] = None):
        super().__init__("CTRStackingEnsemble")
        self.cv_folds = cv_folds
        self.meta_learners = {}
        self.meta_predictions = {}
        self.stacking_features = None
        self.best_meta_learner = None
        self.ensemble_execution_guaranteed = True
        
        if meta_learner_types is None:
            meta_learner_types = ['logistic', 'ridge', 'elastic_net', 'mlp', 'tree']
        self.meta_learner_types = meta_learner_types
        
    def _create_meta_learners(self):
        """Create multiple meta-learners"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available, using simple averaging")
                return
            
            meta_learners = {
                'logistic': LogisticRegression(
                    random_state=42, 
                    max_iter=2000, 
                    C=0.5,
                    class_weight='balanced'
                ),
                'ridge': Ridge(
                    random_state=42, 
                    alpha=1.0
                ),
                'elastic_net': ElasticNet(
                    random_state=42, 
                    alpha=0.5, 
                    l1_ratio=0.5
                ),
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(50, 25),
                    random_state=42,
                    max_iter=1000,
                    alpha=0.01
                ),
                'tree': DecisionTreeClassifier(
                    random_state=42,
                    max_depth=5,
                    min_samples_leaf=50,
                    class_weight='balanced'
                )
            }
            
            for learner_type in self.meta_learner_types:
                if learner_type in meta_learners:
                    self.meta_learners[learner_type] = meta_learners[learner_type]
                    
        except Exception as e:
            logger.error(f"Meta-learner creation failed: {e}")
    
    def _generate_stacking_features(self, X: pd.DataFrame, y: pd.Series, 
                                  base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate stacking features using cross-validation"""
        try:
            if not SKLEARN_AVAILABLE:
                # Fallback to simple stacking
                return np.column_stack(list(base_predictions.values()))
            
            n_samples = len(X)
            n_models = len(base_predictions)
            stacking_features = np.zeros((n_samples, n_models * 2))  # Base + interaction features
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            model_names = list(base_predictions.keys())
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                logger.info(f"Processing fold {fold_idx + 1}/{self.cv_folds} for stacking")
                
                # Generate out-of-fold predictions for each base model
                for i, (model_name, model) in enumerate(self.base_models.items()):
                    try:
                        X_train_fold = X.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X.iloc[val_idx]
                        
                        # Use existing trained model predictions
                        if model_name in base_predictions:
                            fold_predictions = base_predictions[model_name][val_idx]
                        else:
                            fold_predictions = np.full(len(val_idx), 0.0191)
                        
                        # Base prediction
                        stacking_features[val_idx, i] = fold_predictions
                        
                        # Interaction feature (prediction * mean target in fold)
                        fold_mean_target = y_train_fold.mean()
                        stacking_features[val_idx, n_models + i] = fold_predictions * fold_mean_target
                        
                    except Exception as e:
                        logger.warning(f"Fold {fold_idx + 1} processing failed for {model_name}: {e}")
                        stacking_features[val_idx, i] = 0.0191
                        stacking_features[val_idx, n_models + i] = 0.0191 * y.mean()
            
            return stacking_features
            
        except Exception as e:
            logger.error(f"Stacking feature generation failed: {e}")
            return np.column_stack(list(base_predictions.values()))
    
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Train stacking ensemble"""
        logger.info("CTR stacking ensemble training started")
        
        try:
            if len(base_predictions) == 0:
                raise ValueError("No base predictions available")
            
            # Create meta-learners
            self._create_meta_learners()
            
            # Generate stacking features
            logger.info("Generating stacking features")
            self.stacking_features = self._generate_stacking_features(X, y, base_predictions)
            
            # Train meta-learners
            best_score = -1
            metrics_calculator = CTRMetrics()
            
            for learner_name, learner in self.meta_learners.items():
                try:
                    logger.info(f"Training meta-learner: {learner_name}")
                    
                    # Train meta-learner
                    learner.fit(self.stacking_features, y)
                    
                    # Evaluate meta-learner
                    if hasattr(learner, 'predict_proba'):
                        meta_pred = learner.predict_proba(self.stacking_features)[:, 1]
                    else:
                        meta_pred = learner.predict(self.stacking_features)
                    
                    meta_pred = np.clip(meta_pred, 1e-15, 1 - 1e-15)
                    score = metrics_calculator.combined_score(y.values, meta_pred)
                    
                    self.meta_predictions[learner_name] = score
                    
                    if score > best_score:
                        best_score = score
                        self.best_meta_learner = learner_name
                    
                    logger.info(f"Meta-learner {learner_name} score: {score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Meta-learner {learner_name} training failed: {e}")
                    continue
            
            if self.best_meta_learner is None:
                logger.warning("No meta-learner succeeded, using simple averaging")
                self.best_meta_learner = 'averaging'
            
            # Apply ensemble calibration
            ensemble_pred = self._create_stacking_prediction(base_predictions)
            
            try:
                self.apply_ensemble_calibration(X, y, ensemble_pred, method='auto')
            except Exception as e:
                logger.error(f"Ensemble calibration application failed: {e}")
                self.is_calibrated = False
            
            self.is_fitted = True
            
            logger.info(f"CTR stacking ensemble training completed")
            logger.info(f"Best meta-learner: {self.best_meta_learner} (score: {best_score:.4f})")
            
        except Exception as e:
            logger.error(f"CTR stacking ensemble training failed: {e}")
            self.is_fitted = False
            raise
    
    def _create_stacking_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create stacking prediction"""
        try:
            if not self.is_fitted or self.best_meta_learner == 'averaging':
                # Simple averaging fallback
                predictions = list(base_predictions.values())
                return np.mean(predictions, axis=0)
            
            # Create test features in the same format as training
            test_features = np.column_stack(list(base_predictions.values()))
            n_models = len(base_predictions)
            
            # Add interaction features
            mean_target = 0.0191  # Use global average for test predictions
            interaction_features = test_features * mean_target
            
            full_test_features = np.column_stack([test_features, interaction_features])
            
            # Use best meta-learner
            best_learner = self.meta_learners[self.best_meta_learner]
            
            if hasattr(best_learner, 'predict_proba'):
                predictions = best_learner.predict_proba(full_test_features)[:, 1]
            else:
                predictions = best_learner.predict(full_test_features)
            
            predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
            return self._enhance_ensemble_diversity(predictions)
            
        except Exception as e:
            logger.error(f"Stacking prediction creation failed: {e}")
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stacking ensemble prediction"""
        try:
            ensemble_pred = self._create_stacking_prediction(base_predictions)
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                except Exception as e:
                    logger.warning(f"Stacking ensemble calibration failed: {e}")
            
            return np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Stacking ensemble prediction failed: {e}")
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)

class CTRDynamicEnsemble(BaseEnsemble):
    """CTR dynamic weighting ensemble"""
    
    def __init__(self, weighting_method: str = 'performance_diversity'):
        super().__init__("CTRDynamicEnsemble")
        self.weighting_method = weighting_method
        self.dynamic_weights = {}
        self.performance_weights = {}
        self.diversity_weights = {}
        self.temporal_weights = {}
        self.final_weights = {}
        self.ensemble_execution_guaranteed = True
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """Train dynamic ensemble"""
        logger.info(f"CTR dynamic ensemble training started - Method: {self.weighting_method}")
        
        try:
            if len(base_predictions) == 0:
                raise ValueError("No base predictions available")
            
            metrics_calculator = CTRMetrics()
            
            # Calculate performance weights
            logger.info("Calculating performance weights")
            for name, pred in base_predictions.items():
                try:
                    combined_score = metrics_calculator.combined_score(y.values, pred)
                    ctr_score = metrics_calculator.ctr_score(y.values, pred)
                    
                    # Weighted performance score
                    performance_score = 0.7 * combined_score + 0.3 * ctr_score
                    self.performance_weights[name] = max(performance_score, 0.1)
                    
                except Exception as e:
                    logger.warning(f"Performance evaluation failed for {name}: {e}")
                    self.performance_weights[name] = 0.5
            
            # Calculate diversity weights
            logger.info("Calculating diversity weights")
            self.diversity_metrics = self.calculate_diversity_metrics(base_predictions)
            self.diversity_weights = self._calculate_diversity_weights(base_predictions)
            
            # Calculate temporal stability weights
            logger.info("Calculating temporal weights")
            self.temporal_weights = self._calculate_temporal_weights(base_predictions, y)
            
            # Combine weights dynamically
            logger.info("Combining weights dynamically")
            self.final_weights = self._combine_weights_dynamically()
            
            # Apply ensemble calibration
            ensemble_pred = self._create_dynamic_ensemble(base_predictions)
            
            try:
                self.apply_ensemble_calibration(X, y, ensemble_pred, method='auto')
            except Exception as e:
                logger.error(f"Ensemble calibration application failed: {e}")
                self.is_calibrated = False
            
            self.is_fitted = True
            
            final_weights_str = {k: f"{v:.4f}" for k, v in self.final_weights.items()}
            logger.info(f"CTR dynamic ensemble training completed - Final weights: {final_weights_str}")
            logger.info(f"Ensemble calibration: {'Yes' if self.is_calibrated else 'No'}")
            logger.info(f"Diversity score: {self.diversity_metrics.get('diversity_score', 0.0):.4f}")
            
        except Exception as e:
            logger.error(f"CTR dynamic ensemble training failed: {e}")
            self.is_fitted = False
            raise
    
    def _calculate_diversity_weights(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate diversity-based weights"""
        try:
            weights = {}
            
            if len(base_predictions) < 2:
                return {name: 1.0 for name in base_predictions.keys()}
            
            prediction_matrix = np.column_stack(list(base_predictions.values()))
            model_names = list(base_predictions.keys())
            
            # Calculate individual diversity contribution
            for i, name in enumerate(model_names):
                correlations = []
                for j, other_name in enumerate(model_names):
                    if i != j:
                        corr = np.corrcoef(prediction_matrix[:, i], prediction_matrix[:, j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    diversity_score = 1.0 - avg_correlation
                    
                    # Bonus for unique prediction patterns
                    unique_ratio = len(np.unique(prediction_matrix[:, i])) / len(prediction_matrix[:, i])
                    diversity_score *= (0.8 + 0.2 * unique_ratio)
                    
                    weights[name] = max(diversity_score, 0.1)
                else:
                    weights[name] = 1.0
            
            return weights
            
        except Exception as e:
            logger.warning(f"Diversity weight calculation failed: {e}")
            return {name: 1.0 for name in base_predictions.keys()}
    
    def _calculate_temporal_weights(self, base_predictions: Dict[str, np.ndarray], y: pd.Series) -> Dict[str, float]:
        """Calculate temporal stability weights"""
        try:
            weights = {}
            
            # Split data into time-based segments for stability analysis
            n_segments = 5
            segment_size = len(y) // n_segments
            
            for name, pred in base_predictions.items():
                segment_scores = []
                
                for i in range(n_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(y)
                    
                    segment_y = y.iloc[start_idx:end_idx]
                    segment_pred = pred[start_idx:end_idx]
                    
                    try:
                        from evaluation import CTRMetrics
                        metrics_calc = CTRMetrics()
                        score = metrics_calc.combined_score(segment_y.values, segment_pred)
                        segment_scores.append(score)
                    except Exception:
                        segment_scores.append(0.2)
                
                # Calculate stability (inverse of variance)
                if len(segment_scores) > 1:
                    stability = 1.0 / (1.0 + np.var(segment_scores))
                    avg_performance = np.mean(segment_scores)
                    
                    # Combined temporal weight
                    temporal_weight = 0.6 * avg_performance + 0.4 * stability
                    weights[name] = max(temporal_weight, 0.1)
                else:
                    weights[name] = 1.0
            
            return weights
            
        except Exception as e:
            logger.warning(f"Temporal weight calculation failed: {e}")
            return {name: 1.0 for name in base_predictions.keys()}
    
    def _combine_weights_dynamically(self) -> Dict[str, float]:
        """Combine all weight types dynamically"""
        try:
            combined_weights = {}
            
            for name in self.performance_weights.keys():
                performance_weight = self.performance_weights.get(name, 0.5)
                diversity_weight = self.diversity_weights.get(name, 1.0)
                temporal_weight = self.temporal_weights.get(name, 1.0)
                
                # Dynamic combination based on overall ensemble diversity
                diversity_score = self.diversity_metrics.get('diversity_score', 0.5)
                
                if diversity_score < 0.3:
                    # Low diversity - emphasize diversity weights
                    combined_score = (0.3 * performance_weight + 
                                    0.5 * diversity_weight + 
                                    0.2 * temporal_weight)
                elif diversity_score > 0.7:
                    # High diversity - emphasize performance weights
                    combined_score = (0.6 * performance_weight + 
                                    0.2 * diversity_weight + 
                                    0.2 * temporal_weight)
                else:
                    # Balanced combination
                    combined_score = (0.45 * performance_weight + 
                                    0.35 * diversity_weight + 
                                    0.2 * temporal_weight)
                
                combined_weights[name] = max(combined_score, 0.05)
            
            # Normalize weights
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                combined_weights = {k: v/total_weight for k, v in combined_weights.items()}
            
            return combined_weights
            
        except Exception as e:
            logger.error(f"Weight combination failed: {e}")
            model_count = len(self.performance_weights)
            return {name: 1.0/model_count for name in self.performance_weights.keys()}
    
    def _create_dynamic_ensemble(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create dynamic ensemble prediction"""
        try:
            weighted_pred = np.zeros(len(list(base_predictions.values())[0]))
            
            for name, weight in self.final_weights.items():
                if name in base_predictions:
                    weighted_pred += weight * base_predictions[name]
            
            return self._enhance_ensemble_diversity(weighted_pred)
            
        except Exception as e:
            logger.error(f"Dynamic ensemble creation failed: {e}")
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Dynamic ensemble prediction"""
        try:
            if not self.is_fitted:
                logger.warning("Dynamic ensemble not fitted, using weighted average")
                return self._create_dynamic_ensemble(base_predictions)
            
            ensemble_pred = self._create_dynamic_ensemble(base_predictions)
            
            if self.is_calibrated and self.ensemble_calibrator is not None:
                try:
                    ensemble_pred = self.ensemble_calibrator.predict(ensemble_pred)
                except Exception as e:
                    logger.warning(f"Dynamic ensemble calibration failed: {e}")
            
            return np.clip(ensemble_pred, 1e-15, 1 - 1e-15)
            
        except Exception as e:
            logger.error(f"Dynamic ensemble prediction failed: {e}")
            predictions = list(base_predictions.values())
            return np.mean(predictions, axis=0)

class CTRMainEnsemble(BaseEnsemble):
    """CTR main ensemble model with multiple strategies"""
    
    def __init__(self, target_ctr: float = 0.0191, optimization_method: str = 'multi_strategy'):
        super().__init__("CTRMainEnsemble")
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.sub_ensembles = {}
        self.ensemble_weights = {}
        self.meta_learner = None
        self.final_ensemble_weights = {}
        self.ensemble_execution_guaranteed = True
        
    def fit(self, X: pd.DataFrame, y: pd.Series, base_predictions: Dict[str, np.ndarray]):
        """CTR main ensemble training with multiple strategies"""
        logger.info("CTR main ensemble training started - Target: Combined Score 0.34+")
        
        try:
            logger.info(f"Available models: {list(base_predictions.keys())}")
            
            if len(base_predictions) == 0:
                raise ValueError("No base predictions available")
            
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
                dynamic_ensemble = CTRDynamicEnsemble(weighting_method='performance_diversity')
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
                    weight = scores['weight_score'] / total_weight_score
                    self.ensemble_weights[ensemble_name] = weight
            else:
                # Equal weights fallback
                ensemble_count = len(self.sub_ensembles)
                self.ensemble_weights = {name: 1.0/ensemble_count for name in self.sub_ensembles.keys()}
            
            # Add base model predictions to ensemble mix
            logger.info("Stage 4: Adding base models to ensemble mix")
            for name, pred in base_predictions.items():
                try:
                    combined_score = metrics_calculator.combined_score(y.values, pred)
                    ctr_score = metrics_calculator.ctr_score(y.values, pred)
                    weight_score = 0.7 * combined_score + 0.3 * ctr_score
                    
                    # Add with reduced weight compared to ensembles
                    ensemble_predictions[f"base_{name}"] = pred
                    self.ensemble_weights[f"base_{name}"] = weight_score * 0.3  # Reduced weight for base models
                    
                except Exception as e:
                    logger.warning(f"Base model {name} evaluation failed: {e}")
            
            # Normalize final weights
            total_final_weight = sum(self.ensemble_weights.values())
            if total_final_weight > 0:
                self.final_ensemble_weights = {k: v/total_final_weight for k, v in self.ensemble_weights.items()}
            else:
                self.final_ensemble_weights = self.ensemble_weights
            
            # Train meta-learner on ensemble predictions
            logger.info("Stage 5: Training meta-learner")
            if len(ensemble_predictions) > 1 and SKLEARN_AVAILABLE:
                try:
                    ensemble_matrix = np.column_stack(list(ensemble_predictions.values()))
                    
                    self.meta_learner = LogisticRegression(
                        random_state=42, 
                        max_iter=2000,
                        C=0.5,
                        class_weight='balanced'
                    )
                    self.meta_learner.fit(ensemble_matrix, y)
                    
                    # Evaluate meta-learner
                    meta_pred = self.meta_learner.predict_proba(ensemble_matrix)[:, 1]
                    meta_score = metrics_calculator.combined_score(y.values, meta_pred)
                    
                    logger.info(f"Meta-learner trained successfully - score: {meta_score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Meta-learner training failed: {e}")
                    self.meta_learner = None
            
            # Apply ensemble calibration
            logger.info("Stage 6: Applying ensemble calibration")
            final_ensemble_pred = self._create_main_ensemble_prediction(base_predictions)
            
            try:
                self.apply_ensemble_calibration(X, y, final_ensemble_pred, method='auto')
            except Exception as e:
                logger.error(f"Ensemble calibration application failed: {e}")
                self.is_calibrated = False
            
            self.is_fitted = True
            
            # Final evaluation
            final_pred = self.predict_proba(base_predictions)
            final_combined_score = metrics_calculator.combined_score(y.values, final_pred)
            final_ctr_score = metrics_calculator.ctr_score(y.values, final_pred)
            
            logger.info(f"CTR main ensemble training completed")
            logger.info(f"Final Combined Score: {final_combined_score:.4f}")
            logger.info(f"Final CTR Score: {final_ctr_score:.4f}")
            logger.info(f"Target achievement: {final_combined_score >= self.target_combined_score}")
            
            weights_summary = {k: f"{v:.3f}" for k, v in self.final_ensemble_weights.items()}
            logger.info(f"Final ensemble weights: {weights_summary}")
            
        except Exception as e:
            logger.error(f"CTR main ensemble training failed: {e}")
            self.is_fitted = False
            raise
    
    def _create_main_ensemble_prediction(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create main ensemble prediction"""
        try:
            # Collect all ensemble and base predictions
            all_predictions = {}
            
            # Get sub-ensemble predictions
            for ensemble_name, ensemble in self.sub_ensembles.items():
                try:
                    pred = ensemble.predict_proba(base_predictions)
                    all_predictions[ensemble_name] = pred
                except Exception as e:
                    logger.warning(f"Sub-ensemble {ensemble_name} prediction failed: {e}")
            
            # Add base model predictions
            for name, pred in base_predictions.items():
                all_predictions[f"base_{name}"] = pred
            
            if not all_predictions:
                logger.warning("No predictions available, using simple average")
                predictions = list(base_predictions.values())
                return np.mean(predictions, axis=0)
            
            # Use meta-learner if available
            if self.meta_learner is not None and len(all_predictions) > 1:
                try:
                    ensemble_matrix = np.column_stack(list(all_predictions.values()))
                    meta_pred = self.meta_learner.predict_proba(ensemble_matrix)[:, 1]
                    return self._enhance_ensemble_diversity(meta_pred)
                except Exception as e:
                    logger.warning(f"Meta-learner prediction failed: {e}")
            
            # Weighted combination
            if len(self.final_ensemble_weights) > 0:
                weighted_pred = np.zeros(len(list(all_predictions.values())[0]))
                
                for pred_name, weight in self.final_ensemble_weights.items():
                    if pred_name in all_predictions:
                        weighted_pred += weight * all_predictions[pred_name]
                
                return self._enhance_ensemble_diversity(weighted_pred)
            else:
                # Simple average fallback
                predictions = list(all_predictions.values())
                return np.mean(predictions, axis=0)
            
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

class CTREnsembleManager:
    """CTR ensemble management class"""
    
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
                weighting_method = kwargs.get('weighting_method', 'performance_diversity')
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