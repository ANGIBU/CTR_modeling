# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union
import logging
import time
import gc
import warnings
import os
import pickle
import tempfile
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    """CTR feature engineering class - simplified to 117 features"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        self.quick_mode = False
        self.memory_efficient_mode = True
        
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        self.generated_features = []
        self.numerical_features = []
        self.categorical_features = []
        self.removed_columns = []
        self.original_feature_order = []
        self.final_feature_columns = []
        self.target_column = None
        
        self.target_feature_count = config.FEATURE_ENGINEERING_CONFIG.get('target_feature_count', 117)
        self.use_feature_selection = config.FEATURE_ENGINEERING_CONFIG.get('use_feature_selection', False)
        self.disable_normalization_for_trees = config.FEATURE_ENGINEERING_CONFIG.get('disable_normalization_for_trees', True)
        self.use_original_features_only = config.FEATURE_ENGINEERING_CONFIG.get('use_original_features_only', True)
        self.feature_importance_scores = {}
        
        self.processing_stats = {
            'start_time': None,
            'processing_time': 0,
            'memory_usage': 0,
            'feature_types_count': {},
            'total_features_generated': 0
        }
    
    def set_memory_efficient_mode(self, enabled: bool):
        """Enable or disable memory efficient mode"""
        self.memory_efficient_mode = enabled
        if enabled:
            logger.info("Memory efficient mode enabled")
        else:
            logger.info("Memory efficient mode disabled")
    
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode for rapid testing"""
        self.quick_mode = enabled
        if enabled:
            logger.info("Quick mode enabled")
            self.memory_efficient_mode = True
        else:
            logger.info("Quick mode disabled")
    
    def engineer_features(self, 
                         train_df: pd.DataFrame, 
                         test_df: pd.DataFrame, 
                         target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main feature engineering pipeline entry point"""
        if self.quick_mode:
            logger.info("=== Quick Mode Feature Engineering Started ===")
            return self.create_quick_features(train_df, test_df, target_col)
        else:
            logger.info("=== Simplified Feature Engineering Started (117 features) ===")
            return self.create_simplified_features(train_df, test_df, target_col)
    
    def create_quick_features(self,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features for quick testing"""
        logger.info("Creating basic features for quick mode")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            self._classify_columns_basic(X_train)
            
            X_train, X_test = self._convert_to_numeric_safe(X_train, X_test)
            
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            X_train, X_test = self._encode_categorical_safe(X_train, X_test)
            
            X_train, X_test = self._clean_final_features(X_train, X_test)
            
            self.final_feature_columns = list(X_train.columns)
            
            logger.info(f"Quick feature engineering completed: {X_train.shape[1]} features")
            logger.info(f"Final features - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Quick feature engineering failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def create_simplified_features(self, 
                                   train_df: pd.DataFrame, 
                                   test_df: pd.DataFrame, 
                                   target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create simplified features - 117 features matching top performer"""
        logger.info("Creating simplified features (117 target)")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            self._force_memory_cleanup()
            
            self._classify_original_features(X_train)
            
            X_train, X_test = self._prepare_original_features(X_train, X_test)
            self._force_memory_cleanup()
            
            X_train, X_test = self._encode_categorical_minimal(X_train, X_test)
            self._force_memory_cleanup()
            
            X_train, X_test = self._final_data_cleanup(X_train, X_test)
            self._force_memory_cleanup()
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== Simplified feature engineering completed: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Simplified feature engineering failed: {e}")
            self._force_memory_cleanup()
            
            logger.warning("Falling back to basic features")
            return self._create_basic_features_only(train_df, test_df, target_col)
    
    def _force_memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        self.memory_monitor.force_memory_cleanup()
    
    def _classify_original_features(self, X: pd.DataFrame):
        """Classify original features based on top performer strategy"""
        try:
            # TRUE CATEGORICAL (5 types expected)
            true_categorical = []
            for col in X.columns:
                col_lower = col.lower()
                if any(cat in col_lower for cat in ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']):
                    if X[col].nunique() < 50:
                        true_categorical.append(col)
            
            self.categorical_features = true_categorical
            
            # CONTINUOUS (all remaining numeric features)
            continuous_features = []
            for col in X.columns:
                if col not in true_categorical:
                    if X[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                        continuous_features.append(col)
            
            self.numerical_features = continuous_features
            
            logger.info(f"Feature classification - Categorical: {len(self.categorical_features)}, Continuous: {len(self.numerical_features)}")
            logger.info(f"Categorical features: {self.categorical_features}")
            
        except Exception as e:
            logger.error(f"Feature classification failed: {e}")
            self.numerical_features = list(X.select_dtypes(include=[np.number]).columns)
            self.categorical_features = []
    
    def _prepare_original_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare original features with minimal processing"""
        try:
            logger.info("Preparing original features with minimal processing")
            
            # Convert all to numeric
            for col in X_train.columns:
                try:
                    if X_train[col].dtype == 'object' or X_train[col].dtype == 'category':
                        # Label encode string columns
                        train_unique = set(X_train[col].fillna('missing').astype(str).unique())
                        test_unique = set(X_test[col].fillna('missing').astype(str).unique())
                        all_unique = sorted(train_unique | test_unique)
                        
                        mapping = {val: idx for idx, val in enumerate(all_unique)}
                        
                        X_train[col] = X_train[col].fillna('missing').astype(str).map(mapping).fillna(0).astype('float32')
                        X_test[col] = X_test[col].fillna('missing').astype(str).map(mapping).fillna(0).astype('float32')
                    else:
                        # Numeric columns - just convert to float32
                        X_train[col] = X_train[col].fillna(0).astype('float32')
                        X_test[col] = X_test[col].fillna(0).astype('float32')
                        
                except Exception as e:
                    logger.warning(f"Feature preparation failed for {col}: {e}")
                    X_train[col] = 0.0
                    X_test[col] = 0.0
            
            logger.info(f"Feature preparation completed - Features: {X_train.shape[1]}")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return X_train, X_test
    
    def _encode_categorical_minimal(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Minimal categorical encoding - use Categorify approach"""
        try:
            logger.info("Applying minimal categorical encoding")
            
            for col in self.categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        # Already encoded in _prepare_original_features
                        # Just ensure float32 type
                        if X_train[col].dtype != 'float32':
                            X_train[col] = X_train[col].astype('float32')
                        if X_test[col].dtype != 'float32':
                            X_test[col] = X_test[col].astype('float32')
                            
                    except Exception as e:
                        logger.warning(f"Encoding failed for {col}: {e}")
                        continue
            
            logger.info(f"Categorical encoding completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Categorical encoding failed: {e}")
            return X_train, X_test
    
    def _convert_to_numeric_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert categorical columns to numeric using label encoding"""
        try:
            logger.info("Converting categorical columns to numeric")
            
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        if X_train[col].dtype == 'category':
                            X_train[col] = X_train[col].astype('object')
                        if X_test[col].dtype == 'category':
                            X_test[col] = X_test[col].astype('object')
                        
                        if X_train[col].dtype == 'object' or not np.issubdtype(X_train[col].dtype, np.number):
                            train_str = X_train[col].fillna('missing').astype(str)
                            test_str = X_test[col].fillna('missing').astype(str)
                            
                            all_categories = sorted(set(train_str.unique()) | set(test_str.unique()))
                            category_map = {cat: idx for idx, cat in enumerate(all_categories)}
                            
                            X_train[col] = train_str.map(category_map).fillna(0).astype('float32')
                            X_test[col] = test_str.map(category_map).fillna(0).astype('float32')
                            
                            if col not in self.categorical_features:
                                self.categorical_features.append(col)
                        else:
                            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                            
                            if col not in self.numerical_features:
                                self.numerical_features.append(col)
                                
                    except Exception as e:
                        logger.warning(f"Conversion failed for {col}: {e}")
                        X_train[col] = 0.0
                        X_test[col] = 0.0
            
            logger.info(f"Conversion completed - Numeric: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Numeric conversion failed: {e}")
            return X_train, X_test
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """Initialize feature engineering processing"""
        try:
            self.processing_stats['start_time'] = time.time()
            
            self.target_column = self._detect_target_column(train_df, target_col)
            
            self.original_feature_order = sorted([col for col in train_df.columns if col != self.target_column])
            
            mode_info = "QUICK MODE" if self.quick_mode else "SIMPLIFIED MODE"
            logger.info(f"Feature engineering initialization ({mode_info})")
            logger.info(f"Initial data: Training {train_df.shape}, Test {test_df.shape}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Original feature count: {len(self.original_feature_order)}")
            
            self.memory_monitor.log_memory_status("Initialization")
            
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            self.target_column = target_col
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
        """Detect CTR target column"""
        try:
            if provided_target_col and provided_target_col in train_df.columns:
                unique_values = train_df[provided_target_col].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    positive_ratio = train_df[provided_target_col].mean()
                    logger.info(f"Target column confirmed: {provided_target_col} (CTR: {positive_ratio:.4f})")
                    return provided_target_col
            
            ctr_target_names = ['clicked', 'click', 'is_click', 'target', 'label', 'y']
            
            for col_name in ctr_target_names:
                if col_name in train_df.columns:
                    unique_values = train_df[col_name].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        positive_ratio = train_df[col_name].mean()
                        logger.info(f"Target column detected: {col_name} (CTR: {positive_ratio:.4f})")
                        return col_name
            
            logger.warning("Target column not detected, using default: clicked")
            return 'clicked'
            
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return provided_target_col if provided_target_col else 'clicked'
    
    def _prepare_basic_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepare basic data"""
        try:
            if target_col in train_df.columns:
                y_train = train_df[target_col].copy()
                X_train = train_df.drop(columns=[target_col])
            else:
                y_train = pd.Series([0] * len(train_df))
                X_train = train_df.copy()
            
            X_test = test_df.copy()
            
            id_cols = [col for col in X_train.columns if 'id' in col.lower() or 'ID' in col]
            if id_cols:
                X_train = X_train.drop(columns=id_cols, errors='ignore')
                X_test = X_test.drop(columns=id_cols, errors='ignore')
            
            logger.info(f"Data preparation completed - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def _classify_columns_basic(self, X: pd.DataFrame):
        """Basic column classification"""
        try:
            self.numerical_features = []
            self.categorical_features = []
            
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                    unique_count = X[col].nunique()
                    if unique_count > 20:
                        self.numerical_features.append(col)
                    else:
                        self.categorical_features.append(col)
                else:
                    self.categorical_features.append(col)
            
            logger.info(f"Column classification completed - Numeric: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
            
        except Exception as e:
            logger.error(f"Column classification failed: {e}")
            self.numerical_features = list(X.select_dtypes(include=[np.number]).columns)
            self.categorical_features = list(X.select_dtypes(exclude=[np.number]).columns)
    
    def _safe_fillna(self, X: pd.DataFrame) -> pd.DataFrame:
        """Safe fillna"""
        try:
            return X.fillna(0)
        except:
            return X
    
    def _encode_categorical_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical encoding"""
        try:
            for col in self.categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        if X_train[col].dtype == 'category':
                            X_train[col] = X_train[col].astype('object')
                        if X_test[col].dtype == 'category':
                            X_test[col] = X_test[col].astype('object')
                        
                        train_str = X_train[col].fillna('missing').astype(str)
                        test_str = X_test[col].fillna('missing').astype(str)
                        
                        all_values = sorted(set(train_str.unique()) | set(test_str.unique()))
                        value_map = {val: idx for idx, val in enumerate(all_values)}
                        
                        X_train[col] = train_str.map(value_map).fillna(0).astype('float32')
                        X_test[col] = test_str.map(value_map).fillna(0).astype('float32')
                        
                    except Exception as e:
                        logger.warning(f"Encoding failed for {col}: {e}")
                        continue
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Categorical encoding failed: {e}")
            return X_train, X_test
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final data cleanup"""
        try:
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        if X_train[col].dtype != 'float32':
                            X_train[col] = X_train[col].astype('float32')
                        if X_test[col].dtype != 'float32':
                            X_test[col] = X_test[col].astype('float32')
                    except:
                        pass
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final data cleanup failed: {e}")
            return X_train, X_test
    
    def _clean_final_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean final features"""
        return self._final_data_cleanup(X_train, X_test)
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Finalize processing"""
        try:
            self.final_feature_columns = list(X_train.columns)
            
            self.processing_stats['processing_time'] = time.time() - self.processing_stats['start_time']
            self.processing_stats['feature_types_count'] = {
                'original': len(self.original_feature_order),
                'final': len(self.final_feature_columns)
            }
            
            self.processing_stats['total_features_generated'] = 0
            
            logger.info(f"Feature engineering finalized - {len(self.final_feature_columns)} features in {self.processing_stats['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing finalization failed: {e}")
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                    target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features only"""
        try:
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            self._classify_columns_basic(X_train)
            X_train, X_test = self._convert_to_numeric_safe(X_train, X_test)
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            X_train, X_test = self._encode_categorical_safe(X_train, X_test)
            X_train, X_test = self._clean_final_features(X_train, X_test)
            
            self.final_feature_columns = list(X_train.columns)
            
            logger.info(f"Basic features only: {X_train.shape[1]} features")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic features creation failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def _create_minimal_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal features"""
        try:
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
            X_test = X_test.select_dtypes(include=[np.number]).fillna(0)
            
            common_cols = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            logger.info(f"Minimal features created: {X_train.shape[1]} features")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Minimal features creation failed: {e}")
            raise