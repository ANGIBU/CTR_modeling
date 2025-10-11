# models.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import gc
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import pickle

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
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
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        GPU_NAME = "None"
        GPU_MEMORY_GB = 0
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_NAME = "None"
    GPU_MEMORY_GB = 0

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 45.0,
            'critical': 50.0,
            'abort': 55.0
        }
        
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        try:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                return {
                    'available_gb': vm.available / (1024**3),
                    'used_gb': vm.used / (1024**3),
                    'total_gb': vm.total / (1024**3),
                    'percent': vm.percent
                }
            else:
                return {
                    'available_gb': 34.0,
                    'used_gb': 30.0,
                    'total_gb': 64.0,
                    'percent': 47.0
                }
        except Exception:
            return {
                'available_gb': 34.0,
                'used_gb': 30.0,
                'total_gb': 64.0,
                'percent': 47.0
            }
    
    def force_memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        
        if GPU_AVAILABLE and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass

class BaseModel(ABC):
    """Base class for all models"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.memory_monitor = MemoryMonitor()
        self.training_time = 0.0
        self.validation_score = 0.0
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability predictions"""
        pass

class XGBoostModel(BaseModel):
    """XGBoost model with GPU"""
    
    def __init__(self, name: str = "XGBoost_GPU", params: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 51.43,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if GPU_AVAILABLE:
            default_params.update({
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
                'max_bin': 256
            })
            logger.info(f"XGBoost GPU mode enabled: {GPU_NAME} ({GPU_MEMORY_GB:.1f}GB)")
        else:
            logger.warning("XGBoost GPU not available, using CPU mode")
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training with GPU"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0).astype('float32')
            
            logger.info(f"{self.name}: Starting XGBoost training")
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                X_val_clean = X_val.fillna(0).astype('float32')
                
                dtrain = xgb.DMatrix(X_train_clean, label=y_train)
                dval = xgb.DMatrix(X_val_clean, label=y_val)
                
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=200,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )
            else:
                dtrain = xgb.DMatrix(X_train_clean, label=y_train)
                
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=200,
                    verbose_eval=False
                )
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            self.training_time = time.time() - start_time
            
            del X_train_clean
            gc.collect()
            
            return self
            
        except Exception as e:
            logger.error(f"{self.name} model training failed: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        try:
            X_processed = X.fillna(0).astype('float32')
            
            if self.feature_names is not None:
                missing_features = set(self.feature_names) - set(X_processed.columns)
                extra_features = set(X_processed.columns) - set(self.feature_names)
                
                if missing_features:
                    for feature in missing_features:
                        X_processed[feature] = 0
                
                if extra_features:
                    X_processed = X_processed.drop(columns=list(extra_features))
                
                X_processed = X_processed[self.feature_names]
            
            dmatrix = xgb.DMatrix(X_processed)
            proba = self.model.predict(dmatrix)
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
            
        except Exception as e:
            logger.error(f"{self.name} prediction failed: {e}")
            return np.full(len(X), 0.0191)

class LightGBMModel(BaseModel):
    """LightGBM model"""
    
    def __init__(self, name: str = "LightGBM", params: Dict[str, Any] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'cpu',
            'is_unbalance': True,
            'scale_pos_weight': 51.43
        }
        
        logger.info(f"LightGBM CPU mode enabled")
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0).astype('float32')
            
            logger.info(f"{self.name}: Starting LightGBM training")
            
            self.model = lgb.LGBMClassifier(**self.params)
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                X_val_clean = X_val.fillna(0).astype('float32')
                self.model.fit(
                    X_train_clean, y_train,
                    eval_set=[(X_val_clean, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                self.model.fit(X_train_clean, y_train)
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            self.training_time = time.time() - start_time
            
            del X_train_clean
            gc.collect()
            
            return self
            
        except Exception as e:
            logger.error(f"{self.name} model training failed: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        try:
            X_processed = X.fillna(0).astype('float32')
            
            if self.feature_names is not None:
                missing_features = set(self.feature_names) - set(X_processed.columns)
                extra_features = set(X_processed.columns) - set(self.feature_names)
                
                if missing_features:
                    for feature in missing_features:
                        X_processed[feature] = 0
                
                if extra_features:
                    X_processed = X_processed.drop(columns=list(extra_features))
                
                X_processed = X_processed[self.feature_names]
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
            
        except Exception as e:
            logger.error(f"{self.name} prediction failed: {e}")
            return np.full(len(X), 0.0191)

class LogisticModel(BaseModel):
    """Logistic Regression model"""
    
    def __init__(self, name: str = "LogisticRegression", params: Dict[str, Any] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not installed.")
        
        default_params = {
            'C': 0.5,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'tol': 0.0001
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        self.model = LogisticRegression(**self.params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Training"""
        logger.info(f"{self.name} model training started (data: {len(X_train):,})")
        start_time = time.time()
        
        try:
            self.feature_names = list(X_train.columns)
            
            X_train_clean = X_train.fillna(0).astype('float32')
            
            logger.info(f"{self.name}: Starting training")
            self.model.fit(X_train_clean, y_train)
            
            logger.info(f"{self.name}: Training completed successfully")
            self.is_fitted = True
            
            if X_val is not None and y_val is not None and len(X_val) > 0:
                try:
                    val_pred = self.predict_proba(X_val)
                    self.validation_score = roc_auc_score(y_val, val_pred)
                except:
                    self.validation_score = 0.5
            
            self.training_time = time.time() - start_time
            
            del X_train_clean
            gc.collect()
            
            return self
            
        except Exception as e:
            logger.error(f"{self.name} model training failed: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        try:
            X_processed = X.fillna(0).astype('float32')
            
            if self.feature_names is not None:
                missing_features = set(self.feature_names) - set(X_processed.columns)
                extra_features = set(X_processed.columns) - set(self.feature_names)
                
                if missing_features:
                    for feature in missing_features:
                        X_processed[feature] = 0
                
                if extra_features:
                    X_processed = X_processed.drop(columns=list(extra_features))
                
                X_processed = X_processed[self.feature_names]
            
            proba = self.model.predict_proba(X_processed)[:, 1]
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            
            return proba
            
        except Exception as e:
            logger.error(f"{self.name} prediction failed: {e}")
            return np.full(len(X), 0.0191)

class ModelFactory:
    """Model factory"""
    
    _factory_logged = False
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """Create model by type"""
        try:
            if not ModelFactory._factory_logged:
                memory_monitor = MemoryMonitor()
                logger.info(f"Creating model: {model_type}")
                
                if GPU_AVAILABLE:
                    logger.info(f"GPU available: {GPU_NAME} ({GPU_MEMORY_GB:.1f}GB)")
                else:
                    logger.info("GPU not available, using CPU mode")
                
                ModelFactory._factory_logged = True
            
            if model_type.lower() == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost is not installed.")
                model = XGBoostModel(params=kwargs.get('params'))
                
            elif model_type.lower() == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM is not installed.")
                model = LightGBMModel(params=kwargs.get('params'))
                
            elif model_type.lower() == 'logistic':
                model = LogisticModel(params=kwargs.get('params'))
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"{model_type} model creation complete")
            return model
                
        except Exception as e:
            logger.error(f"Model creation failed ({model_type}): {e}")
            raise
    
    @staticmethod
    def get_available_models() -> List[str]:
        """List of available model types"""
        available = []
        
        available.append('logistic')
        
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        
        logger.info(f"Available models: {available}")
        return available
    
    @staticmethod
    def get_model_priority() -> List[str]:
        """Model priority list"""
        priority_order = []
        
        if XGBOOST_AVAILABLE:
            priority_order.append('xgboost')
        
        if LIGHTGBM_AVAILABLE:
            priority_order.append('lightgbm')
            
        if SKLEARN_AVAILABLE:
            priority_order.append('logistic')
        
        return priority_order

FinalLightGBMModel = LightGBMModel
FinalXGBoostModel = XGBoostModel  
FinalLogisticModel = LogisticModel