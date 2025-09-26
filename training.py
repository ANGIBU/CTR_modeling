# training.py

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import gc

# CTR-specific imports
from config import Config

# Setup logging
logger = logging.getLogger(__name__)

# Conditional imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

class CTRClassBalancer:
    """Class balancer for CTR prediction with multiple strategies"""
    
    def __init__(self, target_ctr: float = 0.0191, strategy: str = "hybrid"):
        self.target_ctr = target_ctr
        self.strategy = strategy
        self.applied_method = None
        self.balance_ratio = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply class balancing strategy"""
        try:
            current_ctr = np.mean(y)
            positive_count = np.sum(y == 1)
            negative_count = np.sum(y == 0)
            
            logger.info(f"Original CTR: {current_ctr:.4f}, Positive: {positive_count}, Negative: {negative_count}")
            
            if self.strategy == "none" or current_ctr >= 0.8 * self.target_ctr:
                self.applied_method = "none"
                return X.copy(), y.copy()
            
            elif self.strategy == "undersample":
                return self._undersample(X, y, current_ctr)
            
            elif self.strategy == "oversample":
                return self._oversample(X, y, current_ctr)
            
            elif self.strategy == "hybrid":
                return self._hybrid_balance(X, y, current_ctr)
            
            else:
                self.applied_method = "none"
                return X.copy(), y.copy()
                
        except Exception as e:
            logger.error(f"Class balancing failed: {e}")
            self.applied_method = "failed"
            return X.copy(), y.copy()
    
    def _undersample(self, X: pd.DataFrame, y: pd.Series, current_ctr: float) -> Tuple[pd.DataFrame, pd.Series]:
        """Undersample negative class"""
        try:
            positive_indices = y[y == 1].index
            negative_indices = y[y == 0].index
            
            target_negative_count = int(len(positive_indices) / self.target_ctr - len(positive_indices))
            target_negative_count = min(target_negative_count, len(negative_indices))
            
            sampled_negative_indices = np.random.choice(
                negative_indices, size=target_negative_count, replace=False
            )
            
            balanced_indices = np.concatenate([positive_indices, sampled_negative_indices])
            X_balanced = X.loc[balanced_indices].copy()
            y_balanced = y.loc[balanced_indices].copy()
            
            self.applied_method = "undersample"
            self.balance_ratio = len(positive_indices) / len(balanced_indices)
            
            logger.info(f"Undersampling: {len(X)} -> {len(X_balanced)} samples")
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Undersampling failed: {e}")
            self.applied_method = "failed"
            return X.copy(), y.copy()
    
    def _oversample(self, X: pd.DataFrame, y: pd.Series, current_ctr: float) -> Tuple[pd.DataFrame, pd.Series]:
        """Oversample positive class"""
        try:
            positive_indices = y[y == 1].index
            negative_indices = y[y == 0].index
            
            target_positive_count = int(len(negative_indices) * self.target_ctr / (1 - self.target_ctr))
            oversample_count = target_positive_count - len(positive_indices)
            
            if oversample_count > 0:
                oversampled_indices = np.random.choice(
                    positive_indices, size=oversample_count, replace=True
                )
                
                balanced_indices = np.concatenate([
                    positive_indices, negative_indices, oversampled_indices
                ])
                
                X_balanced = X.loc[balanced_indices].copy()
                y_balanced = y.loc[balanced_indices].copy()
            else:
                X_balanced = X.copy()
                y_balanced = y.copy()
            
            self.applied_method = "oversample"
            self.balance_ratio = len(positive_indices) / len(X_balanced)
            
            logger.info(f"Oversampling: {len(X)} -> {len(X_balanced)} samples")
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Oversampling failed: {e}")
            self.applied_method = "failed"
            return X.copy(), y.copy()
    
    def _hybrid_balance(self, X: pd.DataFrame, y: pd.Series, current_ctr: float) -> Tuple[pd.DataFrame, pd.Series]:
        """Hybrid balancing strategy"""
        try:
            positive_indices = y[y == 1].index
            negative_indices = y[y == 0].index
            
            # Moderate undersampling + light oversampling
            target_negative_reduction = 0.3
            target_positive_increase = 1.5
            
            # Undersample negatives
            new_negative_count = int(len(negative_indices) * (1 - target_negative_reduction))
            sampled_negative_indices = np.random.choice(
                negative_indices, size=new_negative_count, replace=False
            )
            
            # Oversample positives
            oversample_count = int(len(positive_indices) * target_positive_increase) - len(positive_indices)
            oversampled_positive_indices = np.random.choice(
                positive_indices, size=oversample_count, replace=True
            )
            
            balanced_indices = np.concatenate([
                positive_indices, sampled_negative_indices, oversampled_positive_indices
            ])
            
            X_balanced = X.loc[balanced_indices].copy()
            y_balanced = y.loc[balanced_indices].copy()
            
            self.applied_method = "hybrid"
            self.balance_ratio = (len(positive_indices) + oversample_count) / len(X_balanced)
            
            logger.info(f"Hybrid balancing: {len(X)} -> {len(X_balanced)} samples")
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Hybrid balancing failed: {e}")
            self.applied_method = "failed"
            return X.copy(), y.copy()

class CTRCalibrator:
    """CTR-specific calibration for probability outputs"""
    
    def __init__(self, target_ctr: float = 0.0191, method: str = "isotonic"):
        self.target_ctr = target_ctr
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self.beta_params = [1.0, 1.0]
        self.temperature = 1.0
        self.linear_params = [1.0, 0.0]  # slope, intercept
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit calibration"""
        try:
            if self.method == "isotonic":
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_pred, y_true)
                
            elif self.method == "platt":
                self.calibrator = LogisticRegression()
                self.calibrator.fit(y_pred.reshape(-1, 1), y_true)
                
            elif self.method == "beta":
                self._fit_beta_calibration(y_true, y_pred)
                
            elif self.method == "temperature":
                self._fit_temperature_scaling(y_true, y_pred)
                
            elif self.method == "linear":
                self._fit_linear_calibration(y_true, y_pred)
            
            self.is_fitted = True
            logger.info(f"CTR calibration fitted: {self.method}")
            
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            self.is_fitted = False
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply calibration"""
        if not self.is_fitted:
            return np.clip(y_pred, 0.001, 0.999)
        
        try:
            if self.method == "isotonic":
                return np.clip(self.calibrator.transform(y_pred), 0.001, 0.999)
                
            elif self.method == "platt":
                return np.clip(self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1], 0.001, 0.999)
                
            elif self.method == "beta":
                return self._transform_beta_calibration(y_pred)
                
            elif self.method == "temperature":
                return self._transform_temperature_scaling(y_pred)
                
            elif self.method == "linear":
                return self._transform_linear_calibration(y_pred)
            
            else:
                return np.clip(y_pred, 0.001, 0.999)
                
        except Exception as e:
            logger.error(f"Calibration transform failed: {e}")
            return np.clip(y_pred, 0.001, 0.999)
    
    def _fit_linear_calibration(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit linear calibration"""
        try:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(y_pred.reshape(-1, 1), y_true)
            self.linear_params = [model.coef_[0], model.intercept_]
        except Exception as e:
            logger.warning(f"Linear calibration fitting failed: {e}")
            self.linear_params = [1.0, 0.0]
    
    def _transform_linear_calibration(self, y_pred: np.ndarray) -> np.ndarray:
        """Transform using linear calibration"""
        try:
            slope, intercept = self.linear_params
            return np.clip(slope * y_pred + intercept, 0.001, 0.999)
        except:
            return np.clip(y_pred, 0.001, 0.999)
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit beta calibration"""
        try:
            from scipy.optimize import minimize
            
            def beta_loss(params, y_true, y_pred):
                a, b = params
                if a <= 0 or b <= 0:
                    return 1e6
                
                eps = 1e-15
                y_pred_cal = y_pred ** (1/a) * (1-y_pred) ** (1/b)
                y_pred_cal = np.clip(y_pred_cal, eps, 1-eps)
                
                return -np.sum(y_true * np.log(y_pred_cal) + (1-y_true) * np.log(1-y_pred_cal))
            
            result = minimize(beta_loss, [1.0, 1.0], args=(y_true, y_pred), 
                            bounds=[(0.1, 10), (0.1, 10)])
            self.beta_params = result.x
            
        except Exception as e:
            logger.warning(f"Beta calibration fitting failed: {e}")
            self.beta_params = [1.0, 1.0]
    
    def _transform_beta_calibration(self, y_pred: np.ndarray) -> np.ndarray:
        """Transform using beta calibration"""
        try:
            a, b = self.beta_params
            return np.clip(y_pred ** (1/a) * (1-y_pred) ** (1/b), 0.001, 0.999)
        except:
            return np.clip(y_pred, 0.001, 0.999)
    
    def _fit_temperature_scaling(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit temperature scaling"""
        try:
            from scipy.optimize import minimize_scalar
            
            def temperature_nll(T, y_true, y_pred):
                eps = 1e-15
                y_pred_cal = 1 / (1 + np.exp(-(np.log(y_pred / (1-y_pred)) / T)))
                y_pred_cal = np.clip(y_pred_cal, eps, 1-eps)
                return -np.sum(y_true * np.log(y_pred_cal) + (1-y_true) * np.log(1-y_pred_cal))
            
            result = minimize_scalar(temperature_nll, args=(y_true, y_pred), 
                                   bounds=(0.1, 5.0), method='bounded')
            self.temperature = result.x
            
        except Exception as e:
            logger.warning(f"Temperature scaling fitting failed: {e}")
            self.temperature = 1.0
    
    def _transform_temperature_scaling(self, y_pred: np.ndarray) -> np.ndarray:
        """Transform using temperature scaling"""
        try:
            logits = np.log(y_pred / (1-y_pred))
            calibrated_logits = logits / self.temperature
            return np.clip(1 / (1 + np.exp(-calibrated_logits)), 0.001, 0.999)
        except:
            return np.clip(y_pred, 0.001, 0.999)

class CTRTrainer:
    """CTR model trainer with balanced learning and calibration"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.class_balancers = {}
        self.feature_importances = {}
        self.training_history = {}
        
        # CTR specific settings
        self.target_ctr = 0.0191
        self.ctr_tolerance = 0.002
        
        # Available models based on imports
        self.available_models = self._get_available_models()
        logger.info(f"CTR Trainer initialized - available models: {self.available_models}")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        models = ['logistic']  # Always available
        
        if LIGHTGBM_AVAILABLE:
            models.append('lightgbm')
        if XGBOOST_AVAILABLE:
            models.append('xgboost')
            
        return models
    
    def get_available_models(self) -> List[str]:
        """Public method to get available models"""
        return self.available_models.copy()
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                   balance_strategy: str = "hybrid", calibration_method: str = "isotonic",
                   quick_mode: Optional[bool] = None) -> Dict[str, Any]:
        """Train CTR model with balancing and calibration"""
        
        logger.info(f"Training {model_name} with CTR optimization")
        start_time = time.time()
        
        try:
            # Handle quick mode
            if quick_mode is not None:
                logger.info(f"Quick mode: {'ON' if quick_mode else 'OFF'}")
            
            # Step 1: Class balancing
            logger.info(f"Step 1: Class balancing - strategy: {balance_strategy}")
            balancer = CTRClassBalancer(self.target_ctr, balance_strategy)
            X_train_balanced, y_train_balanced = balancer.fit_transform(X_train, y_train)
            self.class_balancers[model_name] = balancer
            
            # Step 2: Feature scaling
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_balanced),
                columns=X_train_balanced.columns,
                index=X_train_balanced.index
            )
            self.scalers[model_name] = scaler
            
            # Scale validation set if provided
            X_val_scaled = None
            if X_val is not None:
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
            
            # Step 3: Model training with CTR-optimized parameters
            logger.info(f"Step 2: Model training")
            model = self._create_model(model_name, y_train_balanced, quick_mode=quick_mode)
            
            # Train the model
            if model_name in ['lightgbm', 'xgboost'] and X_val_scaled is not None and y_val is not None:
                # Use validation set for early stopping
                model = self._train_boosting_model(model_name, model, X_train_scaled, y_train_balanced,
                                                 X_val_scaled, y_val, quick_mode=quick_mode)
            else:
                # Standard training
                model.fit(X_train_scaled, y_train_balanced)
     
            self.models[model_name] = model
            
            # Step 4: Model calibration
            logger.info(f"Step 3: Model calibration - method: {calibration_method}")
            
            # Get predictions for calibration
            if X_val_scaled is not None and y_val is not None:
                val_pred = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val_scaled)
                calibrator = CTRCalibrator(self.target_ctr, calibration_method)
                calibrator.fit(y_val.values, val_pred)
                self.calibrators[model_name] = calibrator
            else:
                # Use training set for calibration (not ideal but necessary)
                train_pred = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train_scaled)
                calibrator = CTRCalibrator(self.target_ctr, calibration_method)
                calibrator.fit(y_train_balanced.values, train_pred)
                self.calibrators[model_name] = calibrator
            
            # Step 5: Feature importance extraction
            self._extract_feature_importance(model_name, model, X_train_scaled.columns)
            
            training_time = time.time() - start_time
            
            # Training summary
            training_summary = {
                'model_name': model_name,
                'training_time': training_time,
                'balance_method': balancer.applied_method,
                'calibration_method': calibration_method,
                'training_samples': len(X_train_balanced),
                'final_ctr': np.mean(y_train_balanced),
                'feature_count': X_train_scaled.shape[1],
                'quick_mode': quick_mode or False
            }
            
            self.training_history[model_name] = training_summary
            
            logger.info(f"{model_name} training completed in {training_time:.2f}s")
            logger.info(f"Balance method: {balancer.applied_method}, Training samples: {len(X_train_balanced)}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"{model_name} training failed: {e}")
            return {'model_name': model_name, 'error': str(e), 'training_time': 0}
    
    def _create_model(self, model_name: str, y_train: pd.Series, quick_mode: Optional[bool] = None):
        """Create model with CTR-optimized parameters"""
        
        # Calculate class weights
        classes = np.unique(y_train)
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        
        # Adjust parameters based on quick mode
        is_quick = quick_mode or False
        
        if model_name == 'logistic':
            class_weight = {0: 1.0, 1: min(neg_count/pos_count, 10.0)} if pos_count > 0 else 'balanced'
            
            params = {
                'C': 0.1 if is_quick else 1.0,
                'class_weight': class_weight,
                'random_state': 42,
                'max_iter': 100 if is_quick else 1000,
                'solver': 'liblinear'
            }
            return LogisticRegression(**params)
        
        elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 15 if is_quick else 31,
                'learning_rate': 0.1 if is_quick else 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'scale_pos_weight': min(scale_pos_weight, 10.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
            return lgb.LGBMClassifier(**params)
        
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 3 if is_quick else 6,
                'learning_rate': 0.1 if is_quick else 0.05,
                'n_estimators': 50 if is_quick else 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': min(scale_pos_weight, 10.0),
                'random_state': 42,
                'n_jobs': -1
            }
            return xgb.XGBClassifier(**params)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _train_boosting_model(self, model_name: str, model, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series, quick_mode: Optional[bool] = None):
        """Train boosting models with early stopping"""
        
        is_quick = quick_mode or False
        early_stopping_rounds = 10 if is_quick else 20
        
        if model_name == 'lightgbm':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
            )
        
        elif model_name == 'xgboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        return model
    
    def _extract_feature_importance(self, model_name: str, model, feature_names: List[str]):
        """Extract feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance_dict = dict(zip(feature_names, np.abs(model.coef_[0])))
            else:
                importance_dict = {}
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            self.feature_importances[model_name] = sorted_importance
            
            logger.info(f"Feature importance extracted for {model_name}: {len(sorted_importance)} features")
            
        except Exception as e:
            logger.warning(f"Feature importance extraction failed for {model_name}: {e}")
            self.feature_importances[model_name] = {}
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with calibration"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scalers[model_name].transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Get predictions
            model = self.models[model_name]
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_scaled)[:, 1]
            else:
                y_pred = model.predict(X_scaled)
            
            # Apply calibration
            if model_name in self.calibrators:
                y_pred = self.calibrators[model_name].transform(y_pred)
            
            return np.clip(y_pred, 0.001, 0.999)
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            return np.full(len(X), self.target_ctr)
    
    def predict_batch(self, model_name: str, X: pd.DataFrame, batch_size: int = 10000) -> np.ndarray:
        """Make predictions in batches"""
        try:
            predictions = []
            
            for i in range(0, len(X), batch_size):
                batch = X.iloc[i:i + batch_size]
                pred_batch = self.predict(model_name, batch)
                predictions.extend(pred_batch)
                
                # Memory cleanup
                if i % (batch_size * 10) == 0:
                    gc.collect()
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Batch prediction failed for {model_name}: {e}")
            return np.full(len(X), self.target_ctr)
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        try:
            import pickle
            
            model_data = {
                'model': self.models.get(model_name),
                'scaler': self.scalers.get(model_name),
                'calibrator': self.calibrators.get(model_name),
                'class_balancer': self.class_balancers.get(model_name),
                'feature_importance': self.feature_importances.get(model_name, {}),
                'training_history': self.training_history.get(model_name, {})
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model {model_name} saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[model_name] = model_data.get('model')
            self.scalers[model_name] = model_data.get('scaler')
            self.calibrators[model_name] = model_data.get('calibrator')
            self.class_balancers[model_name] = model_data.get('class_balancer')
            self.feature_importances[model_name] = model_data.get('feature_importance', {})
            self.training_history[model_name] = model_data.get('training_history', {})
            
            logger.info(f"Model {model_name} loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary for all models"""
        return {
            'models_trained': list(self.models.keys()),
            'training_history': self.training_history,
            'feature_importances': self.feature_importances,
            'target_ctr': self.target_ctr
        }

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("CTR Trainer Test")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    
    # Generate features
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.choice([0, 1, 2], n_samples),
        'feature_4': np.random.uniform(0, 1, n_samples),
        'feature_5': np.random.randn(n_samples)
    })
    
    # Generate imbalanced target (CTR-like)
    y = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    
    print(f"Sample data: {X.shape}, CTR: {np.mean(y):.4f}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Test trainer
    trainer = CTRTrainer()
    
    # Train models
    for model_name in trainer.get_available_models():
        print(f"\nTraining {model_name}...")
        summary = trainer.train_model(
            model_name, X_train, y_train, X_val, y_val,
            balance_strategy="hybrid", calibration_method="isotonic"
        )
        print(f"Training summary: {summary}")
        676
        # Test prediction
        pred = trainer.predict(model_name, X_test)
        predicted_ctr = np.mean(pred)
        actual_ctr = np.mean(y_test)6
        
        print(f"Predicted CTR: {predicted_ctr:.4f}, Actual CTR: {actual_ctr:.4f}")
    
    print("\nCTR Trainer test completed!")66