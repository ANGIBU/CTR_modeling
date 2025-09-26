# ensemble.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import gc
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr
import warnings
from pathlib import Path
import pickle

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class CTRMetrics:
    """CTR-focused metrics calculation with bias correction"""
    
    def __init__(self, target_ctr: float = 0.0191):
        self.target_ctr = target_ctr
        self.ctr_tolerance = 0.002
        
        # Weights optimized for CTR prediction quality
        self.auc_weight = 0.3
        self.ap_weight = 0.4
        self.ctr_quality_weight = 0.3
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray, penalize_ctr_bias: bool = True) -> float:
        """Calculate combined score with CTR bias penalty"""
        try:
            if len(y_true) == 0 or len(y_pred_proba) == 0:
                return 0.0
            
            if len(y_true) != len(y_pred_proba):
                return 0.0
            
            # Ensure we have both classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                return 0.0
            
            # Ensure 1D arrays
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
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
            
            if not penalize_ctr_bias:
                return float(np.clip(base_score, 0.0, 1.0))
            
            # CTR quality assessment
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            
            # CTR quality score
            if ctr_bias <= self.ctr_tolerance:
                ctr_quality_score = 1.0 - (ctr_bias / self.ctr_tolerance) * 0.2
            else:
                # Heavy penalty for large CTR bias
                excess_bias = ctr_bias - self.ctr_tolerance
                ctr_quality_score = max(0.1, 0.8 - excess_bias * 20)
            
            # Combined score with CTR quality
            final_score = base_score * (1 - self.ctr_quality_weight) + ctr_quality_score * self.ctr_quality_weight
            
            return float(np.clip(final_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.0
    
    def ctr_quality_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Detailed CTR quality assessment"""
        try:
            y_true = np.asarray(y_true).flatten()
            y_pred_proba = np.asarray(y_pred_proba).flatten()
            
            predicted_ctr = np.mean(y_pred_proba)
            actual_ctr = np.mean(y_true)
            ctr_bias = abs(predicted_ctr - actual_ctr)
            ctr_relative_error = ctr_bias / max(actual_ctr, 1e-8)
            
            # CTR alignment with target
            target_alignment = abs(actual_ctr - self.target_ctr)
            target_prediction_alignment = abs(predicted_ctr - self.target_ctr)
            
            # Quality metrics
            metrics = {
                'predicted_ctr': predicted_ctr,
                'actual_ctr': actual_ctr,
                'ctr_bias': ctr_bias,
                'ctr_relative_error': ctr_relative_error,
                'target_alignment': target_alignment,
                'target_prediction_alignment': target_prediction_alignment,
                'within_tolerance': ctr_bias <= self.ctr_tolerance
            }
            
            # Overall quality score
            if ctr_bias <= self.ctr_tolerance:
                quality_score = 1.0 - (ctr_bias / self.ctr_tolerance) * 0.3
            else:
                excess_bias = ctr_bias - self.ctr_tolerance
                quality_score = max(0.0, 0.7 - excess_bias * 15)
            
            return quality_score, metrics
            
        except Exception as e:
            logger.error(f"CTR quality calculation failed: {e}")
            return 0.0, {}

class CTRCalibrationEnsemble:
    """Specialized ensemble for CTR calibration"""
    
    def __init__(self, target_ctr: float = 0.0191, calibration_method: str = "isotonic"):
        self.target_ctr = target_ctr
        self.calibration_method = calibration_method
        self.calibrators = {}
        self.is_fitted = False
        self.base_predictions = {}
        
    def fit(self, base_predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """Fit calibration for each base model"""
        logger.info("CTR calibration ensemble fitting started")
        
        try:
            y_true = np.asarray(y_true).flatten()
            
            for model_name, pred in base_predictions.items():
                pred_1d = np.asarray(pred).flatten()
                
                if len(pred_1d) != len(y_true):
                    logger.warning(f"Length mismatch for {model_name}, skipping")
                    continue
                
                # Fit calibrator
                calibrator = self._create_calibrator(pred_1d, y_true)
                self.calibrators[model_name] = calibrator
                
                logger.info(f"Calibrator fitted for {model_name}")
            
            self.is_fitted = True
            logger.info("CTR calibration ensemble fitting completed")
            
        except Exception as e:
            logger.error(f"Calibration ensemble fitting failed: {e}")
            self.is_fitted = False
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply calibration to base predictions"""
        if not self.is_fitted:
            return base_predictions
        
        calibrated_predictions = {}
        
        for model_name, pred in base_predictions.items():
            if model_name in self.calibrators:
                try:
                    pred_1d = np.asarray(pred).flatten()
                    calibrated = self._apply_calibration(model_name, pred_1d)
                    calibrated_predictions[model_name] = calibrated
                except Exception as e:
                    logger.warning(f"Calibration failed for {model_name}: {e}")
                    calibrated_predictions[model_name] = pred
            else:
                calibrated_predictions[model_name] = pred
        
        return calibrated_predictions
    
    def _create_calibrator(self, pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Create calibrator for specific predictions"""
        try:
            from sklearn.isotonic import IsotonicRegression
            
            # Calculate current bias
            predicted_ctr = np.mean(pred)
            actual_ctr = np.mean(y_true)
            bias = predicted_ctr - actual_ctr
            
            calibrator = {
                'method': self.calibration_method,
                'bias': bias,
                'predicted_ctr': predicted_ctr,
                'actual_ctr': actual_ctr
            }
            
            if self.calibration_method == 'isotonic':
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(pred, y_true)
                calibrator['model'] = iso_reg
            elif self.calibration_method == 'platt':
                # Platt scaling
                platt_model = LogisticRegression()
                platt_model.fit(pred.reshape(-1, 1), y_true)
                calibrator['model'] = platt_model
            elif self.calibration_method == 'linear_bias':
                # Simple linear bias correction
                if predicted_ctr > 0:
                    calibrator['correction_factor'] = actual_ctr / predicted_ctr
                else:
                    calibrator['correction_factor'] = 1.0
            else:
                # Temperature scaling
                calibrator['temperature'] = self._fit_temperature(pred, y_true)
            
            return calibrator
            
        except Exception as e:
            logger.error(f"Calibrator creation failed: {e}")
            return {'method': 'none'}
    
    def _apply_calibration(self, model_name: str, pred: np.ndarray) -> np.ndarray:
        """Apply calibration to predictions"""
        try:
            calibrator = self.calibrators[model_name]
            method = calibrator.get('method', 'none')
            
            if method == 'isotonic':
                return np.clip(calibrator['model'].predict(pred), 0.001, 0.999)
            elif method == 'platt':
                return np.clip(calibrator['model'].predict_proba(pred.reshape(-1, 1))[:, 1], 0.001, 0.999)
            elif method == 'linear_bias':
                factor = calibrator['correction_factor']
                return np.clip(pred * factor, 0.001, 0.999)
            elif method == 'temperature':
                temp = calibrator['temperature']
                logits = np.log(pred / (1 - pred))
                calibrated_logits = logits / temp
                return np.clip(1 / (1 + np.exp(-calibrated_logits)), 0.001, 0.999)
            else:
                return pred
                
        except Exception as e:
            logger.error(f"Calibration application failed for {model_name}: {e}")
            return pred
    
    def _fit_temperature(self, pred: np.ndarray, y_true: np.ndarray) -> float:
        """Fit temperature scaling parameter"""
        try:
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(T):
                eps = 1e-15
                logits = np.log(pred / (1 - pred))
                calibrated_logits = logits / T
                calibrated_pred = 1 / (1 + np.exp(-calibrated_logits))
                calibrated_pred = np.clip(calibrated_pred, eps, 1 - eps)
                return -np.sum(y_true * np.log(calibrated_pred) + (1 - y_true) * np.log(1 - calibrated_pred))
            
            result = minimize_scalar(temperature_loss, bounds=(0.1, 5.0), method='bounded')
            return result.x
            
        except Exception:
            return 1.0

class CTROptimizedEnsemble:
    """CTR-optimized ensemble with bias-aware weighting"""
    
    def __init__(self, target_ctr: float = 0.0191, optimization_method: str = "ctr_focused"):
        self.target_ctr = target_ctr
        self.optimization_method = optimization_method
        self.weights = {}
        self.performance_scores = {}
        self.ctr_metrics = CTRMetrics(target_ctr)
        self.calibration_ensemble = CTRCalibrationEnsemble(target_ctr)
        self.is_fitted = False
    
    def fit(self, base_predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """Fit ensemble with CTR-focused optimization"""
        logger.info(f"CTR-optimized ensemble fitting - method: {self.optimization_method}")
        
        try:
            y_true = np.asarray(y_true).flatten()
            
            # Step 1: Fit calibration ensemble
            self.calibration_ensemble.fit(base_predictions, y_true)
            
            # Step 2: Get calibrated predictions
            calibrated_predictions = self.calibration_ensemble.predict_proba(base_predictions)
            
            # Step 3: Calculate individual model performance
            for model_name, pred in calibrated_predictions.items():
                pred_1d = np.asarray(pred).flatten()
                if len(pred_1d) == len(y_true):
                    combined_score = self.ctr_metrics.combined_score(y_true, pred_1d)
                    ctr_quality, ctr_details = self.ctr_metrics.ctr_quality_score(y_true, pred_1d)
                    
                    self.performance_scores[model_name] = {
                        'combined_score': combined_score,
                        'ctr_quality': ctr_quality,
                        'ctr_details': ctr_details
                    }
            
            # Step 4: Optimize ensemble weights
            self.weights = self._optimize_weights(calibrated_predictions, y_true)
            
            self.is_fitted = True
            
            # Log results
            for model_name, perf in self.performance_scores.items():
                logger.info(f"{model_name}: Combined={perf['combined_score']:.4f}, CTR Quality={perf['ctr_quality']:.4f}")
            
            logger.info(f"Final weights: {self.weights}")
            logger.info("CTR-optimized ensemble fitting completed")
            
        except Exception as e:
            logger.error(f"CTR ensemble fitting failed: {e}")
            self.is_fitted = False
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict with optimized weights"""
        if not self.is_fitted:
            logger.warning("Ensemble not fitted, using equal weights")
            clean_preds = [np.asarray(pred).flatten() for pred in base_predictions.values()]
            return np.mean(clean_preds, axis=0) if clean_preds else np.array([self.target_ctr])
        
        try:
            # Apply calibration
            calibrated_predictions = self.calibration_ensemble.predict_proba(base_predictions)
            
            # Apply weights
            weighted_sum = np.zeros(len(list(calibrated_predictions.values())[0]))
            total_weight = 0
            
            for model_name, pred in calibrated_predictions.items():
                if model_name in self.weights:
                    weight = self.weights[model_name]
                    pred_1d = np.asarray(pred).flatten()
                    weighted_sum += weight * pred_1d
                    total_weight += weight
            
            if total_weight > 0:
                final_prediction = weighted_sum / total_weight
            else:
                # Fallback to equal weights
                clean_preds = [np.asarray(pred).flatten() for pred in calibrated_predictions.values()]
                final_prediction = np.mean(clean_preds, axis=0) if clean_preds else np.array([self.target_ctr])
            
            # Final CTR adjustment
            predicted_ctr = np.mean(final_prediction)
            if abs(predicted_ctr - self.target_ctr) > 0.005:
                # Apply final adjustment to match target CTR better
                adjustment_factor = self.target_ctr / predicted_ctr if predicted_ctr > 0 else 1.0
                # Apply gentle adjustment
                adjustment_factor = 1.0 + 0.3 * (adjustment_factor - 1.0)
                final_prediction = np.clip(final_prediction * adjustment_factor, 0.001, 0.999)
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return np.full(len(list(base_predictions.values())[0]), self.target_ctr)
    
    def _optimize_weights(self, calibrated_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
        """Optimize ensemble weights for CTR quality"""
        
        model_names = list(calibrated_predictions.keys())
        predictions_matrix = np.column_stack([
            np.asarray(calibrated_predictions[name]).flatten() for name in model_names
        ])
        
        if self.optimization_method == "ctr_focused":
            return self._ctr_focused_optimization(predictions_matrix, y_true, model_names)
        elif self.optimization_method == "differential_evolution":
            return self._differential_evolution_optimization(predictions_matrix, y_true, model_names)
        else:
            return self._grid_search_optimization(predictions_matrix, y_true, model_names)
    
    def _ctr_focused_optimization(self, predictions_matrix: np.ndarray, y_true: np.ndarray, model_names: List[str]) -> Dict[str, float]:
        """CTR-focused weight optimization"""
        
        def objective(weights):
            weights = np.array(weights)
            weights = np.abs(weights)  # Ensure non-negative
            if np.sum(weights) == 0:
                return 1.0  # Bad score
            
            weights = weights / np.sum(weights)  # Normalize
            
            # Calculate ensemble prediction
            ensemble_pred = np.dot(predictions_matrix, weights)
            
            # CTR quality is primary objective
            ctr_quality, _ = self.ctr_metrics.ctr_quality_score(y_true, ensemble_pred)
            combined_score = self.ctr_metrics.combined_score(y_true, ensemble_pred)
            
            # Weighted objective (CTR quality is more important)
            objective_score = 0.7 * ctr_quality + 0.3 * combined_score
            
            return -objective_score  # Minimize negative
        
        # Initial weights based on individual performance
        initial_weights = []
        for name in model_names:
            perf = self.performance_scores.get(name, {'ctr_quality': 0.1})
            initial_weights.append(max(0.1, perf['ctr_quality']))
        
        initial_weights = np.array(initial_weights)
        initial_weights = initial_weights / np.sum(initial_weights)
        
        # Optimization
        try:
            from scipy.optimize import minimize
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in model_names]
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
            
            if result.success:
                weights = np.abs(result.x)
                weights = weights / np.sum(weights)
            else:
                weights = initial_weights
                
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using performance-based weights")
            weights = initial_weights
        
        return dict(zip(model_names, weights))
    
    def _differential_evolution_optimization(self, predictions_matrix: np.ndarray, y_true: np.ndarray, model_names: List[str]) -> Dict[str, float]:
        """Differential evolution optimization"""
        
        def objective(weights):
            weights = np.abs(weights)
            if np.sum(weights) == 0:
                return 1.0
            
            weights = weights / np.sum(weights)
            ensemble_pred = np.dot(predictions_matrix, weights)
            
            ctr_quality, _ = self.ctr_metrics.ctr_quality_score(y_true, ensemble_pred)
            combined_score = self.ctr_metrics.combined_score(y_true, ensemble_pred)
            
            return -(0.8 * ctr_quality + 0.2 * combined_score)
        
        try:
            bounds = [(0, 1) for _ in model_names]
            result = differential_evolution(objective, bounds, seed=42, maxiter=50)
            
            if result.success:
                weights = np.abs(result.x)
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(model_names)) / len(model_names)
                
        except Exception as e:
            logger.warning(f"Differential evolution failed: {e}")
            weights = np.ones(len(model_names)) / len(model_names)
        
        return dict(zip(model_names, weights))
    
    def _grid_search_optimization(self, predictions_matrix: np.ndarray, y_true: np.ndarray, model_names: List[str]) -> Dict[str, float]:
        """Grid search optimization"""
        
        best_weights = None
        best_score = -1
        
        # Simple grid search for 2-3 models
        if len(model_names) <= 3:
            weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            if len(model_names) == 2:
                for w1 in weight_options:
                    w2 = 1 - w1
                    weights = np.array([w1, w2])
                    
                    ensemble_pred = np.dot(predictions_matrix, weights)
                    ctr_quality, _ = self.ctr_metrics.ctr_quality_score(y_true, ensemble_pred)
                    combined_score = self.ctr_metrics.combined_score(y_true, ensemble_pred)
                    
                    score = 0.7 * ctr_quality + 0.3 * combined_score
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weights
            
            elif len(model_names) == 3:
                for w1 in weight_options:
                    for w2 in weight_options:
                        if w1 + w2 >= 1:
                            continue
                        w3 = 1 - w1 - w2
                        weights = np.array([w1, w2, w3])
                        
                        ensemble_pred = np.dot(predictions_matrix, weights)
                        ctr_quality, _ = self.ctr_metrics.ctr_quality_score(y_true, ensemble_pred)
                        combined_score = self.ctr_metrics.combined_score(y_true, ensemble_pred)
                        
                        score = 0.7 * ctr_quality + 0.3 * combined_score
                        
                        if score > best_score:
                            best_score = score
                            best_weights = weights
        
        if best_weights is None:
            # Fallback to equal weights
            best_weights = np.ones(len(model_names)) / len(model_names)
        
        return dict(zip(model_names, best_weights))

class CTREnsembleManager:
    """Main CTR ensemble management system"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.target_ctr = 0.0191
        
        # Ensemble components
        self.ensembles = {}
        self.base_models = {}
        self.final_ensemble = None
        
        # Performance tracking
        self.performance_history = {}
        self.best_ensemble = None
        self.best_score = 0.0
        
        logger.info("CTR Ensemble Manager initialized")
    
    def add_base_model(self, name: str, model, predictions: np.ndarray = None):
        """Add base model to ensemble"""
        self.base_models[name] = {
            'model': model,
            'predictions': predictions
        }
        logger.info(f"Base model added: {name}")
    
    def create_ensembles(self, base_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Any]:
        """Create multiple ensemble strategies"""
        logger.info("Creating ensemble strategies")
        
        ensemble_results = {}
        
        try:
            # 1. CTR-Optimized Ensemble (Primary)
            ctr_ensemble = CTROptimizedEnsemble(
                target_ctr=self.target_ctr,
                optimization_method="ctr_focused"
            )
            ctr_ensemble.fit(base_predictions, y_true)
            self.ensembles['ctr_optimized'] = ctr_ensemble
            
            # Evaluate
            ctr_pred = ctr_ensemble.predict_proba(base_predictions)
            ctr_metrics = CTRMetrics(self.target_ctr)
            ctr_score = ctr_metrics.combined_score(y_true, ctr_pred)
            ctr_quality, ctr_details = ctr_metrics.ctr_quality_score(y_true, ctr_pred)
            
            ensemble_results['ctr_optimized'] = {
                'combined_score': ctr_score,
                'ctr_quality': ctr_quality,
                'ctr_details': ctr_details,
                'predicted_ctr': np.mean(ctr_pred),
                'actual_ctr': np.mean(y_true)
            }
            
            # 2. Differential Evolution Ensemble
            try:
                de_ensemble = CTROptimizedEnsemble(
                    target_ctr=self.target_ctr,
                    optimization_method="differential_evolution"
                )
                de_ensemble.fit(base_predictions, y_true)
                self.ensembles['differential_evolution'] = de_ensemble
                
                de_pred = de_ensemble.predict_proba(base_predictions)
                de_score = ctr_metrics.combined_score(y_true, de_pred)
                de_quality, de_details = ctr_metrics.ctr_quality_score(y_true, de_pred)
                
                ensemble_results['differential_evolution'] = {
                    'combined_score': de_score,
                    'ctr_quality': de_quality,
                    'ctr_details': de_details,
                    'predicted_ctr': np.mean(de_pred),
                    'actual_ctr': np.mean(y_true)
                }
                
            except Exception as e:
                logger.warning(f"Differential evolution ensemble failed: {e}")
            
            # 3. Grid Search Ensemble
            try:
                grid_ensemble = CTROptimizedEnsemble(
                    target_ctr=self.target_ctr,
                    optimization_method="grid_search"
                )
                grid_ensemble.fit(base_predictions, y_true)
                self.ensembles['grid_search'] = grid_ensemble
                
                grid_pred = grid_ensemble.predict_proba(base_predictions)
                grid_score = ctr_metrics.combined_score(y_true, grid_pred)
                grid_quality, grid_details = ctr_metrics.ctr_quality_score(y_true, grid_pred)
                
                ensemble_results['grid_search'] = {
                    'combined_score': grid_score,
                    'ctr_quality': grid_quality,
                    'ctr_details': grid_details,
                    'predicted_ctr': np.mean(grid_pred),
                    'actual_ctr': np.mean(y_true)
                }
                
            except Exception as e:
                logger.warning(f"Grid search ensemble failed: {e}")
            
            # Select best ensemble based on CTR quality
            best_ensemble_name = max(ensemble_results.keys(), 
                                   key=lambda x: 0.7 * ensemble_results[x]['ctr_quality'] + 0.3 * ensemble_results[x]['combined_score'])
            
            self.final_ensemble = self.ensembles[best_ensemble_name]
            self.best_ensemble = best_ensemble_name
            self.best_score = ensemble_results[best_ensemble_name]['combined_score']
            
            logger.info(f"Best ensemble: {best_ensemble_name}")
            logger.info(f"Best combined score: {self.best_score:.4f}")
            logger.info(f"Best CTR quality: {ensemble_results[best_ensemble_name]['ctr_quality']:.4f}")
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return {}
    
    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate final ensemble prediction"""
        if self.final_ensemble is None:
            logger.warning("No ensemble fitted, using equal weights")
            clean_preds = [np.asarray(pred).flatten() for pred in base_predictions.values()]
            return np.mean(clean_preds, axis=0) if clean_preds else np.array([self.target_ctr])
        
        try:
            return self.final_ensemble.predict_proba(base_predictions)
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            clean_preds = [np.asarray(pred).flatten() for pred in base_predictions.values()]
            return np.mean(clean_preds, axis=0) if clean_preds else np.array([self.target_ctr])
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble performance summary"""
        return {
            'best_ensemble': self.best_ensemble,
            'best_score': self.best_score,
            'available_ensembles': list(self.ensembles.keys()),
            'base_models': list(self.base_models.keys()),
            'target_ctr': self.target_ctr
        }
    
    def save_ensemble(self, filepath: str):
        """Save ensemble system"""
        try:
            ensemble_data = {
                'ensembles': self.ensembles,
                'final_ensemble': self.final_ensemble,
                'best_ensemble': self.best_ensemble,
                'best_score': self.best_score,
                'target_ctr': self.target_ctr,
                'config': self.config
            }
            
            if JOBLIB_AVAILABLE:
                joblib.dump(ensemble_data, filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(ensemble_data, f)
            
            logger.info(f"Ensemble saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble system"""
        try:
            if JOBLIB_AVAILABLE:
                ensemble_data = joblib.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    ensemble_data = pickle.load(f)
            
            self.ensembles = ensemble_data.get('ensembles', {})
            self.final_ensemble = ensemble_data.get('final_ensemble')
            self.best_ensemble = ensemble_data.get('best_ensemble')
            self.best_score = ensemble_data.get('best_score', 0.0)
            self.target_ctr = ensemble_data.get('target_ctr', 0.0191)
            
            logger.info(f"Ensemble loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("CTR Ensemble Test")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate target with CTR-like distribution
    y_true = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    
    # Generate mock base model predictions
    base_predictions = {
        'model_1': np.random.beta(1, 50, n_samples),  # Low CTR predictions
        'model_2': np.random.beta(2, 80, n_samples),  # Very low CTR predictions  
        'model_3': np.random.beta(1.5, 60, n_samples)  # Low CTR predictions
    }
    
    # Add some correlation with target
    for name in base_predictions:
        # Make positive samples slightly higher
        positive_mask = y_true == 1
        base_predictions[name][positive_mask] *= 2
        base_predictions[name] = np.clip(base_predictions[name], 0.001, 0.999)
    
    print(f"Sample data: {n_samples} samples, CTR: {np.mean(y_true):.4f}")
    for name, pred in base_predictions.items():
        print(f"{name}: predicted CTR = {np.mean(pred):.4f}")
    
    # Test ensemble manager
    ensemble_manager = CTREnsembleManager()
    
    # Create ensembles
    results = ensemble_manager.create_ensembles(base_predictions, y_true)
    
    print("\nEnsemble Results:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Combined Score: {result['combined_score']:.4f}")
        print(f"  CTR Quality: {result['ctr_quality']:.4f}")
        print(f"  Predicted CTR: {result['predicted_ctr']:.4f}")
        print(f"  CTR Bias: {abs(result['predicted_ctr'] - result['actual_ctr']):.4f}")
    
    # Test final prediction
    final_pred = ensemble_manager.predict(base_predictions)
    final_ctr = np.mean(final_pred)
    
    print(f"\nFinal Ensemble:")
    print(f"Predicted CTR: {final_ctr:.4f}")
    print(f"CTR Bias: {abs(final_ctr - np.mean(y_true)):.4f}")
    
    # Get summary
    summary = ensemble_manager.get_ensemble_summary()
    print(f"\nSummary: {summary}")
    
    print("\nCTR Ensemble test completed!")