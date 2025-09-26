# training.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import gc
import warnings
from pathlib import Path
import pickle

# Core ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

# Gradient boosting libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class CTRClassBalancer:
    """CTR specialized class balancing with multiple strategies"""
    
    def __init__(self, target_ctr: float = 0.0191, strategy: str = "hybrid"):
        self.target_ctr = target_ctr
        self.strategy = strategy
        self.balancer = None
        self.applied_method = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply class balancing strategy"""
        logger.info(f"Class balancing started - strategy: {self.strategy}")
        
        try:
            # Calculate current class distribution
            class_counts = y.value_counts()
            current_ctr = class_counts.get(1, 0) / len(y)
            logger.info(f"Current CTR: {current_ctr:.4f}, Target CTR: {self.target_ctr:.4f}")
            
            X_balanced, y_balanced = X.copy(), y.copy()
            
            if self.strategy == "hybrid":
                # Hybrid approach: SMOTE + undersampling
                try:
                    # Calculate target ratio for SMOTE
                    minority_count = class_counts.get(1, 0)
                    majority_count = class_counts.get(0, 0)
                    
                    # Target: achieve 5% positive samples (reasonable for CTR)
                    target_ratio = 0.05 / 0.95
                    
                    # Step 1: SMOTE with limited oversampling
                    if minority_count > 10:
                        smote = SMOTE(
                            sampling_strategy=min(target_ratio, minority_count / majority_count * 3),
                            random_state=42,
                            k_neighbors=min(5, minority_count - 1)
                        )
                        X_resampled, y_resampled = smote.fit_resample(X_balanced, y_balanced)
                        logger.info(f"SMOTE applied - samples: {len(X_resampled)}")
                        
                        # Step 2: Moderate undersampling
                        undersampler = RandomUnderSampler(
                            sampling_strategy=0.08,  # 8% positive samples
                            random_state=42
                        )
                        X_balanced, y_balanced = undersampler.fit_resample(X_resampled, y_resampled)
                        self.applied_method = "SMOTE + UnderSampling"
                    else:
                        # Too few positive samples, only use class weights
                        self.applied_method = "ClassWeight_Only"
                        
                except Exception as e:
                    logger.warning(f"Hybrid balancing failed: {e}, using class weights only")
                    self.applied_method = "ClassWeight_Only"
                    
            elif self.strategy == "smote_tomek":
                # SMOTE + Tomek cleaning
                try:
                    minority_count = class_counts.get(1, 0)
                    if minority_count > 10:
                        smote_tomek = SMOTETomek(
                            sampling_strategy=0.1,
                            random_state=42,
                            smote=SMOTE(k_neighbors=min(5, minority_count - 1))
                        )
                        X_balanced, y_balanced = smote_tomek.fit_resample(X_balanced, y_balanced)
                        self.applied_method = "SMOTE_Tomek"
                    else:
                        self.applied_method = "ClassWeight_Only"
                except Exception as e:
                    logger.warning(f"SMOTE-Tomek failed: {e}")
                    self.applied_method = "ClassWeight_Only"
                    
            elif self.strategy == "adasyn":
                # Adaptive synthetic sampling
                try:
                    minority_count = class_counts.get(1, 0)
                    if minority_count > 10:
                        adasyn = ADASYN(
                            sampling_strategy=0.08,
                            random_state=42,
                            n_neighbors=min(5, minority_count - 1)
                        )
                        X_balanced, y_balanced = adasyn.fit_resample(X_balanced, y_balanced)
                        self.applied_method = "ADASYN"
                    else:
                        self.applied_method = "ClassWeight_Only"
                except Exception as e:
                    logger.warning(f"ADASYN failed: {e}")
                    self.applied_method = "ClassWeight_Only"
                    
            else:
                self.applied_method = "ClassWeight_Only"
            
            # Log final distribution
            final_counts = y_balanced.value_counts()
            final_ctr = final_counts.get(1, 0) / len(y_balanced)
            logger.info(f"Balancing completed - method: {self.applied_method}")
            logger.info(f"Final CTR: {final_ctr:.4f}, samples: {len(X_balanced)}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Class balancing failed: {e}")
            return X, y

class CTRCalibrator:
    """CTR probability calibration specialist"""
    
    def __init__(self, target_ctr: float = 0.0191, method: str = "isotonic"):
        self.target_ctr = target_ctr
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit calibration model"""
        try:
            from sklearn.calibration import CalibratedClassifierCV, calibration_curve
            from sklearn.isotonic import IsotonicRegression
            
            logger.info(f"CTR calibration fitting - method: {self.method}")
            
            if self.method == "isotonic":
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_pred, y_true)
            elif self.method == "platt":
                # Platt scaling (sigmoid)
                from sklearn.linear_model import LogisticRegression
                self.calibrator = LogisticRegression()
                self.calibrator.fit(y_pred.reshape(-1, 1), y_true)
            elif self.method == "beta":
                # Beta calibration for imbalanced data
                self._fit_beta_calibration(y_true, y_pred)
            else:
                # Temperature scaling
                self._fit_temperature_scaling(y_true, y_pred)
                
            self.is_fitted = True
            
            # Validate calibration quality
            calibrated_pred = self.transform(y_pred)
            predicted_ctr = np.mean(calibrated_pred)
            actual_ctr = np.mean(y_true)
            
            logger.info(f"Calibration completed - Predicted CTR: {predicted_ctr:.4f}, Actual CTR: {actual_ctr:.4f}")
            
        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            self.is_fitted = False
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Transform predictions using calibration"""
        try:
            if not self.is_fitted:
                return np.clip(y_pred, 0.001, 0.999)
                
            if self.method == "isotonic":
                return np.clip(self.calibrator.predict(y_pred), 0.001, 0.999)
            elif self.method == "platt":
                return np.clip(self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1], 0.001, 0.999)
            elif self.method == "beta":
                return self._transform_beta_calibration(y_pred)
            else:
                return self._transform_temperature_scaling(y_pred)
                
        except Exception as e:
            logger.error(f"Calibration transform failed: {e}")
            return np.clip(y_pred, 0.001, 0.999)
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit beta calibration for imbalanced data"""
        try:
            from scipy.optimize import minimize
            
            def beta_nll(params, y_true, y_pred):
                a, b = params
                eps = 1e-15
                y_pred_cal = np.clip(y_pred ** (1/a) * (1-y_pred) ** (1/b), eps, 1-eps)
                return -np.sum(y_true * np.log(y_pred_cal) + (1-y_true) * np.log(1-y_pred_cal))
            
            result = minimize(beta_nll, [1.0, 1.0], args=(y_true, y_pred), 
                            bounds=[(0.1, 10.0), (0.1, 10.0)], method='L-BFGS-B')
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
                   balance_strategy: str = "hybrid", calibration_method: str = "isotonic") -> Dict[str, Any]:
        """Train CTR model with balancing and calibration"""
        
        logger.info(f"Training {model_name} with CTR optimization")
        start_time = time.time()
        
        try:
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
            model = self._create_model(model_name, y_train_balanced)
            
            # Train the model
            if model_name in ['lightgbm', 'xgboost'] and X_val_scaled is not None and y_val is not None:
                # Use validation set for early stopping
                model = self._train_boosting_model(model_name, model, X_train_scaled, y_train_balanced,
                                                 X_val_scaled, y_val)
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
                'feature_count': X_train_scaled.shape[1]
            }
            
            self.training_history[model_name] = training_summary
            
            logger.info(f"{model_name} training completed in {training_time:.2f}s")
            logger.info(f"Balance method: {balancer.applied_method}, Training samples: {len(X_train_balanced)}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"{model_name} training failed: {e}")
            return {'model_name': model_name, 'error': str(e), 'training_time': 0}
    
    def _create_model(self, model_name: str, y_train: pd.Series):
        """Create model with CTR-optimized parameters"""
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        if model_name == 'logistic':
            return LogisticRegression(
                class_weight=class_weight_dict,
                max_iter=1000,
                solver='liblinear',
                random_state=42,
                penalty='l2',
                C=0.1  # Stronger regularization for stability
            )
            
        elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # LightGBM parameters optimized for CTR prediction
            pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
            
            return lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=20,
                min_split_gain=0.1,
                scale_pos_weight=pos_weight,
                n_estimators=500,
                random_state=42,
                verbose=-1,
                is_unbalance=True
            )
            
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost parameters optimized for CTR prediction
            pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0
            
            return xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_weight=5,
                scale_pos_weight=pos_weight,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _train_boosting_model(self, model_name: str, model, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series):
        """Train boosting models with early stopping"""
        
        if model_name == 'lightgbm':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_names=['validation'],
                eval_metric='logloss',
                early_stopping_rounds=50,
                verbose=0
            )
        elif model_name == 'xgboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        
        return model
    
    def _extract_feature_importance(self, model_name: str, model, feature_names: List[str]):
        """Extract and store feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = np.zeros(len(feature_names))
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            self.feature_importances[model_name] = importance_dict
            
            # Log top 10 features
            top_features = list(importance_dict.keys())[:10]
            logger.info(f"{model_name} top features: {', '.join(top_features[:5])}")
            
        except Exception as e:
            logger.warning(f"Feature importance extraction failed for {model_name}: {e}")
            self.feature_importances[model_name] = {}
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Predict with calibration"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            model = self.models[model_name]
            scaler = self.scalers.get(model_name)
            calibrator = self.calibrators.get(model_name)
            
            # Scale features
            if scaler:
                X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
            else:
                X_scaled = X
            
            # Get raw predictions
            if hasattr(model, 'predict_proba'):
                raw_pred = model.predict_proba(X_scaled)[:, 1]
            else:
                raw_pred = model.predict(X_scaled)
            
            # Apply calibration
            if calibrator and calibrator.is_fitted:
                calibrated_pred = calibrator.transform(raw_pred)
            else:
                calibrated_pred = np.clip(raw_pred, 0.001, 0.999)
            
            # Final CTR adjustment to target
            predicted_ctr = np.mean(calibrated_pred)
            if abs(predicted_ctr - self.target_ctr) > self.ctr_tolerance:
                # Apply linear adjustment
                adjustment_factor = self.target_ctr / predicted_ctr
                calibrated_pred = np.clip(calibrated_pred * adjustment_factor, 0.001, 0.999)
            
            return calibrated_pred
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            return np.full(len(X), self.target_ctr)
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            model_data = {
                'model': self.models[model_name],
                'scaler': self.scalers.get(model_name),
                'calibrator': self.calibrators.get(model_name),
                'class_balancer': self.class_balancers.get(model_name),
                'feature_importance': self.feature_importances.get(model_name, {}),
                'training_history': self.training_history.get(model_name, {}),
                'config': self.config
            }
            
            if JOBLIB_AVAILABLE:
                joblib.dump(model_data, filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info(f"Model {model_name} saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        try:
            if JOBLIB_AVAILABLE:
                model_data = joblib.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            
            self.models[model_name] = model_data['model']
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
        
        # Test prediction
        pred = trainer.predict(model_name, X_test)
        predicted_ctr = np.mean(pred)
        actual_ctr = np.mean(y_test)
        
        print(f"Predicted CTR: {predicted_ctr:.4f}, Actual CTR: {actual_ctr:.4f}")
    
    print("\nCTR Trainer test completed!")