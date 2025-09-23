# models.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import joblib
import logging
import gc
import warnings
warnings.filterwarnings('ignore')

class BaseModel:
    """Base model class for CTR prediction"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        self.training_history = {}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model"""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError
        
    def predict_proba(self, X):
        """Predict probabilities"""
        raise NotImplementedError
        
    def save_model(self, filepath):
        """Save trained model"""
        raise NotImplementedError
        
    def load_model(self, filepath):
        """Load trained model"""
        raise NotImplementedError

class LogisticModel(BaseModel):
    """Logistic Regression model for CTR prediction"""
    
    def __init__(self, name="logistic", **params):
        super().__init__(name)
        # Default parameters
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'liblinear',
            'class_weight': 'balanced'
        }
        # Update with provided parameters
        default_params.update(params)
        self.params = default_params
        self.model = LogisticRegression(**self.params)
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train logistic regression model"""
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Calculate feature importance
            if hasattr(self.model, 'coef_'):
                self.feature_importance = np.abs(self.model.coef_[0])
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_pred)
                val_logloss = log_loss(y_val, val_pred)
                self.training_history = {
                    'val_auc': val_auc,
                    'val_logloss': val_logloss
                }
                
            return self
            
        except Exception as e:
            logging.error(f"Logistic regression training failed: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath):
        """Save model"""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

class LightGBMModel(BaseModel):
    """LightGBM model for CTR prediction"""
    
    def __init__(self, name="lightgbm", **params):
        super().__init__(name)
        # Default parameters
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100
        }
        default_params.update(params)
        self.params = default_params
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train LightGBM model"""
        try:
            # Prepare validation data
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(50, verbose=False)] if eval_set else None
            )
            
            self.is_trained = True
            self.feature_importance = self.model.feature_importances_
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_pred)
                val_logloss = log_loss(y_val, val_pred)
                self.training_history = {
                    'val_auc': val_auc,
                    'val_logloss': val_logloss
                }
            
            return self
            
        except Exception as e:
            logging.error(f"LightGBM training failed: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath):
        """Save model"""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

class XGBoostModel(BaseModel):
    """XGBoost model for CTR prediction"""
    
    def __init__(self, name="xgboost", **params):
        super().__init__(name)
        # Default parameters
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        default_params.update(params)
        self.params = default_params
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train XGBoost model"""
        try:
            # Prepare validation data
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
            
            self.is_trained = True
            self.feature_importance = self.model.feature_importances_
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_pred)
                val_logloss = log_loss(y_val, val_pred)
                self.training_history = {
                    'val_auc': val_auc,
                    'val_logloss': val_logloss
                }
            
            return self
            
        except Exception as e:
            logging.error(f"XGBoost training failed: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath):
        """Save model"""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

class CatBoostModel(BaseModel):
    """CatBoost model for CTR prediction"""
    
    def __init__(self, name="catboost", **params):
        super().__init__(name)
        # Default parameters
        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False
        }
        default_params.update(params)
        self.params = default_params
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train CatBoost model"""
        try:
            self.model = cb.CatBoostClassifier(**self.params)
            
            # Prepare validation data
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = cb.Pool(X_val, y_val)
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
            
            self.is_trained = True
            self.feature_importance = self.model.feature_importances_
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_pred)
                val_logloss = log_loss(y_val, val_pred)
                self.training_history = {
                    'val_auc': val_auc,
                    'val_logloss': val_logloss
                }
            
            return self
            
        except Exception as e:
            logging.error(f"CatBoost training failed: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filepath):
        """Save model"""
        self.model.save_model(filepath)
        
    def load_model(self, filepath):
        """Load model"""
        self.model = cb.CatBoostClassifier()
        self.model.load_model(filepath)
        self.is_trained = True

class DeepCTRModel(BaseModel):
    """Deep Neural Network for CTR prediction"""
    
    def __init__(self, name="deep_ctr", input_dim=None, **params):
        super().__init__(name)
        self.input_dim = input_dim
        # Default parameters
        default_params = {
            'hidden_dims': [512, 256, 128],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 1024,
            'epochs': 50,
            'early_stopping_patience': 10
        }
        default_params.update(params)
        self.params = default_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_model(self):
        """Build neural network architecture"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.params['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.params['dropout_rate'])
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train deep neural network"""
        try:
            if self.input_dim is None:
                self.input_dim = X_train.shape[1]
            
            self.model = self._build_model().to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
            
            # Prepare data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train),
                torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
            )
            train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
            
            # Validation data
            val_loader = None
            if X_val is not None and y_val is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val),
                    torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
                )
                val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=False)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.params['epochs']):
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                if val_loader is not None:
                    self.model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            outputs = self.model(batch_X).squeeze()
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.params['early_stopping_patience']:
                            break
            
            self.is_trained = True
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_pred)
                val_logloss = log_loss(y_val, val_pred)
                self.training_history = {
                    'val_auc': val_auc,
                    'val_logloss': val_logloss
                }
            
            return self
            
        except Exception as e:
            logging.error(f"Deep CTR model training failed: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
            outputs = self.model(X_tensor).squeeze().cpu().numpy()
        
        return (outputs > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
            outputs = self.model(X_tensor).squeeze().cpu().numpy()
        
        return outputs
    
    def save_model(self, filepath):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'params': self.params
        }, filepath)
        
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.params = checkpoint['params']
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True

class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_model(model_name, **params):
        """Create model instance based on name"""
        model_map = {
            'logistic': LogisticModel,
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel,
            'catboost': CatBoostModel,
            'deep_ctr': DeepCTRModel
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_map[model_name](name=model_name, **params)

    @staticmethod
    def get_available_models():
        """Get list of available models"""
        return ['logistic', 'lightgbm', 'xgboost', 'catboost', 'deep_ctr']

# Memory management
def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()