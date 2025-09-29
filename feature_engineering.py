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
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    """CTR feature engineering class with enhanced capabilities"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        # Processing modes
        self.quick_mode = False
        self.memory_efficient_mode = True
        
        # Feature engineering state variables
        self.target_encoders = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        self.generated_features = []
        self.numerical_features = []
        self.categorical_features = []
        self.interaction_features = []
        self.target_encoding_features = []
        self.temporal_features = []
        self.statistical_features = []
        self.frequency_features = []
        self.polynomial_features = []
        self.cross_features = []
        self.binning_features = []
        self.rank_features = []
        self.ratio_features = []
        self.advanced_features = []
        self.clustering_features = []
        self.removed_columns = []
        self.original_feature_order = []
        self.final_feature_columns = []
        self.target_column = None
        
        # Processing statistics for monitoring
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
            logger.info("Memory efficient mode enabled - simplified features only")
        else:
            logger.info("Memory efficient mode disabled - full feature engineering")
    
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode for rapid testing"""
        self.quick_mode = enabled
        if enabled:
            logger.info("Quick mode enabled - basic features only for rapid testing")
            self.memory_efficient_mode = True
        else:
            logger.info("Quick mode disabled - full feature engineering available")
    
    def engineer_features(self, 
                         train_df: pd.DataFrame, 
                         test_df: pd.DataFrame, 
                         target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main feature engineering pipeline entry point"""
        if self.quick_mode:
            logger.info("=== Quick Mode Feature Engineering Started ===")
            return self.create_quick_features(train_df, test_df, target_col)
        else:
            logger.info("=== Enhanced Feature Engineering Started ===")
            return self.create_enhanced_features(train_df, test_df, target_col)
    
    def create_quick_features(self,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features for quick testing (50 samples)"""
        logger.info("Creating basic features for quick mode testing")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            # 1. Basic data preparation
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            # 2. Basic column classification
            self._classify_columns_basic(X_train)
            
            # 3. Safe data type fixes with categorical handling
            X_train, X_test = self._fix_basic_data_types_safe(X_train, X_test)
            
            # 4. Fill missing values with safe categorical handling
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            # 5. Safe categorical encoding
            X_train, X_test = self._encode_categorical_safe(X_train, X_test)
            
            # 6. Simple numeric normalization
            X_train, X_test = self._normalize_numeric_basic(X_train, X_test)
            
            # 7. Remove any problematic columns
            X_train, X_test = self._clean_final_features(X_train, X_test)
            
            # Store final feature order
            self.final_feature_columns = list(X_train.columns)
            
            logger.info(f"Quick feature engineering completed: {X_train.shape[1]} features")
            logger.info(f"Final features - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Quick feature engineering failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def create_enhanced_features(self, 
                               train_df: pd.DataFrame, 
                               test_df: pd.DataFrame, 
                               target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Enhanced feature engineering pipeline for full dataset processing"""
        logger.info("Creating enhanced features for full dataset")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            # 1. Basic data preparation
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            # 2. Column classification
            self._classify_columns(X_train)
            
            # 3. Safe data type unification
            X_train, X_test = self._unify_data_types_safe(X_train, X_test)
            
            # 4. Basic feature cleanup
            X_train, X_test = self._clean_basic_features(X_train, X_test)
            
            # Memory check for full processing
            memory_status = self.memory_monitor.get_memory_status()
            if not memory_status['should_cleanup'] and memory_status['available_gb'] > 10:
                logger.info("Enhanced feature engineering enabled - sufficient memory available")
                
                # 5. Enhanced target encoding features
                X_train, X_test = self._create_enhanced_target_encoding_features(X_train, X_test, y_train)
                
                # 6. Expanded interaction feature creation
                X_train, X_test = self._create_expanded_interaction_features(X_train, X_test, y_train)
                
                # 7. Enhanced statistical features
                X_train, X_test = self._create_enhanced_statistical_features(X_train, X_test)
                
                # 8. Frequency-based features with CV
                X_train, X_test = self._create_cv_frequency_features(X_train, X_test, y_train)
                
                # 9. Enhanced cross features
                X_train, X_test = self._create_enhanced_cross_features(X_train, X_test, y_train)
                
                # 10. Time-based features
                X_train, X_test = self._create_temporal_features(X_train, X_test)
                
                # 11. Enhanced binning features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_binning', True):
                    X_train, X_test = self._create_enhanced_binning_features(X_train, X_test)
                
                # 12. Rank features with multiple methods
                X_train, X_test = self._create_enhanced_rank_features(X_train, X_test)
                
                # 13. Enhanced ratio features
                X_train, X_test = self._create_enhanced_ratio_features(X_train, X_test)
                
                # 14. Polynomial features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_polynomial_features', True):
                    X_train, X_test = self._create_enhanced_polynomial_features(X_train, X_test)
                
                # 15. Clustering-based features
                X_train, X_test = self._create_clustering_features(X_train, X_test)
                
                # 16. Advanced mathematical features
                X_train, X_test = self._create_advanced_math_features(X_train, X_test)
                
            else:
                logger.warning("Simplified feature engineering due to memory constraints")
                # Fallback to basic enhanced features
                X_train, X_test = self._create_target_encoding_features(X_train, X_test, y_train)
                X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
            
            # 17. Safe categorical feature encoding
            X_train, X_test = self._encode_categorical_features_safe(X_train, X_test, y_train)
            
            # 18. Numeric feature transformation
            X_train, X_test = self._create_numeric_features(X_train, X_test)
            
            # 19. Final data cleanup and optimization
            X_train, X_test = self._final_data_cleanup_enhanced(X_train, X_test)
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== Enhanced feature engineering completed: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced feature engineering failed: {e}")
            self.memory_monitor.force_memory_cleanup()
            
            # Use basic features only on failure
            logger.warning("Using basic features only due to error")
            return self._create_basic_features_only(train_df, test_df, target_col)
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """Initialize feature engineering processing"""
        try:
            self.processing_stats['start_time'] = time.time()
            
            # Target column detection
            self.target_column = self._detect_target_column(train_df, target_col)
            
            self.original_feature_order = sorted([col for col in train_df.columns if col != self.target_column])
            
            mode_info = "QUICK MODE" if self.quick_mode else "ENHANCED MODE"
            logger.info(f"Feature engineering initialization ({mode_info})")
            logger.info(f"Initial data: Training {train_df.shape}, Test {test_df.shape}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Original feature count: {len(self.original_feature_order)}")
            
            self.memory_monitor.log_memory_status("Initialization")
            
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            self.target_column = target_col
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
        """Detect CTR target column with validation"""
        try:
            # First, try provided target column
            if provided_target_col and provided_target_col in train_df.columns:
                unique_values = train_df[provided_target_col].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    positive_ratio = train_df[provided_target_col].mean()
                    logger.info(f"Target column confirmed: {provided_target_col} (CTR: {positive_ratio:.4f})")
                    return provided_target_col
            
            # Search for common CTR target column names
            ctr_target_names = ['clicked', 'click', 'is_click', 'target', 'label', 'y']
            
            for col_name in ctr_target_names:
                if col_name in train_df.columns:
                    unique_values = train_df[col_name].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        positive_ratio = train_df[col_name].mean()
                        logger.info(f"Target column detected: {col_name} (CTR: {positive_ratio:.4f})")
                        return col_name
            
            # Fallback
            return provided_target_col or 'clicked'
            
        except Exception as e:
            logger.warning(f"Target column detection failed: {e}")
            return provided_target_col or 'clicked'
    
    def _prepare_basic_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepare basic data for feature engineering"""
        try:
            # Extract target
            if target_col in train_df.columns:
                y_train = train_df[target_col].copy()
                X_train = train_df.drop(columns=[target_col]).copy()
            else:
                logger.warning(f"Target column '{target_col}' not found, using zeros")
                y_train = pd.Series([0] * len(train_df))
                X_train = train_df.copy()
            
            X_test = test_df.copy()
            
            # Ensure consistent columns
            common_cols = sorted(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            logger.info(f"Data preparation completed - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def _classify_columns(self, X: pd.DataFrame):
        """Classify columns into numerical and categorical"""
        try:
            self.numerical_features = []
            self.categorical_features = []
            
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
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
    
    def _classify_columns_basic(self, X: pd.DataFrame):
        """Basic column classification for quick mode"""
        try:
            self.numerical_features = list(X.select_dtypes(include=[np.number]).columns)
            self.categorical_features = list(X.select_dtypes(exclude=[np.number]).columns)
            
            logger.info(f"Basic column classification - Numeric: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
            
        except Exception as e:
            logger.error(f"Basic column classification failed: {e}")
            self.numerical_features = []
            self.categorical_features = []
    
    def _convert_categorical_to_object(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to object type for safe processing"""
        try:
            df_converted = df.copy()
            for col in df_converted.columns:
                if df_converted[col].dtype.name == 'category':
                    df_converted[col] = df_converted[col].astype('object')
                    logger.debug(f"Converted {col} from category to object")
            return df_converted
        except Exception as e:
            logger.warning(f"Categorical conversion failed: {e}")
            return df
    
    def _safe_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe fillna that handles categorical columns properly"""
        try:
            df_filled = self._convert_categorical_to_object(df)
            
            for col in df_filled.columns:
                if df_filled[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    df_filled[col] = df_filled[col].fillna(0)
                else:
                    df_filled[col] = df_filled[col].fillna('missing').astype('object')
            
            return df_filled
            
        except Exception as e:
            logger.warning(f"Safe fillna failed: {e}")
            df_converted = self._convert_categorical_to_object(df)
            return df_converted.fillna(0)
    
    def _unify_data_types_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe data type unification"""
        try:
            X_train = self._convert_categorical_to_object(X_train)
            X_test = self._convert_categorical_to_object(X_test)
            
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        if X_train[col].dtype != X_test[col].dtype:
                            train_str = X_train[col].astype(str)
                            test_str = X_test[col].astype(str)
                            
                            try:
                                X_train[col] = pd.to_numeric(train_str, errors='coerce').fillna(0)
                                X_test[col] = pd.to_numeric(test_str, errors='coerce').fillna(0)
                            except:
                                X_train[col] = train_str.astype('object')
                                X_test[col] = test_str.astype('object')
                    
                    except Exception as e:
                        logger.warning(f"Type unification failed for {col}: {e}")
                        X_train[col] = X_train[col].fillna(0)
                        X_test[col] = X_test[col].fillna(0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Data type unification failed: {e}")
            return X_train, X_test
    
    def _fix_basic_data_types_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe basic data type fixes"""
        try:
            X_train = self._convert_categorical_to_object(X_train)
            X_test = self._convert_categorical_to_object(X_test)
            
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        if X_train[col].dtype == 'object':
                            X_train[col] = X_train[col].astype(str)
                            X_test[col] = X_test[col].astype(str)
                    except:
                        X_train[col] = X_train[col].fillna('unknown').astype('object')
                        X_test[col] = X_test[col].fillna('unknown').astype('object')
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic data type fix failed: {e}")
            return X_train, X_test
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean basic features"""
        try:
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic feature cleaning failed: {e}")
            return X_train, X_test
    
    def _create_enhanced_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced target encoding features with cross-validation"""
        try:
            if len(self.categorical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating enhanced target encoding features")
            
            # Select top categorical features by cardinality
            cat_features_to_encode = []
            for col in self.categorical_features:
                if col in X_train.columns:
                    cardinality = X_train[col].nunique()
                    if 2 < cardinality < 2000:
                        cat_features_to_encode.append(col)
            
            if len(cat_features_to_encode) > 20:
                cat_features_to_encode = cat_features_to_encode[:20]
            
            for col in cat_features_to_encode:
                try:
                    X_train[col] = X_train[col].astype('object')
                    X_test[col] = X_test[col].astype('object')
                    
                    if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_cross_validation_encoding', True):
                        # Cross-validation target encoding (FIXED: use fit and transform separately)
                        encoded_train = np.zeros(len(X_train))
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        
                        for train_idx, val_idx in kf.split(X_train):
                            encoder = TargetEncoder(target_type='binary', smooth='auto')
                            # Fix: Use fit and transform separately
                            encoder.fit(
                                X_train[col].iloc[train_idx].values.reshape(-1, 1), 
                                y_train.iloc[train_idx]
                            )
                            encoded_train[val_idx] = encoder.transform(
                                X_train[col].iloc[val_idx].values.reshape(-1, 1)
                            ).flatten()
                        
                        # Train encoder on full data for test set
                        encoder = TargetEncoder(target_type='binary', smooth='auto')
                        encoder.fit(X_train[[col]], y_train)
                        encoded_test = encoder.transform(X_test[[col]]).flatten()
                        
                        feature_name = f"{col}_target_encoded_cv"
                        X_train[feature_name] = encoded_train
                        X_test[feature_name] = encoded_test
                    else:
                        # Simple target encoding
                        encoder = TargetEncoder(target_type='binary', smooth='auto')
                        
                        X_train_encoded = encoder.fit_transform(X_train[[col]], y_train).flatten()
                        X_test_encoded = encoder.transform(X_test[[col]]).flatten()
                        
                        feature_name = f"{col}_target_encoded"
                        X_train[feature_name] = X_train_encoded
                        X_test[feature_name] = X_test_encoded
                    
                    self.target_encoders[col] = encoder
                    self.target_encoding_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Enhanced target encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Enhanced target encoding completed: {len(self.target_encoding_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced target encoding features failed: {e}")
            return X_train, X_test
    
    def _create_expanded_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create expanded interaction features with higher complexity"""
        try:
            if len(self.numerical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating expanded interaction features")
            
            numeric_features_selected = []
            for col in self.numerical_features:
                if col in X_train.columns:
                    if X_train[col].var() > 0:
                        numeric_features_selected.append(col)
            
            if len(numeric_features_selected) > 15:
                numeric_features_selected = numeric_features_selected[:15]
            
            interaction_count = 0
            max_interactions = 50
            
            for i in range(len(numeric_features_selected)):
                for j in range(i + 1, len(numeric_features_selected)):
                    if interaction_count >= max_interactions:
                        break
                    
                    col1, col2 = numeric_features_selected[i], numeric_features_selected[j]
                    
                    try:
                        # Multiplication interaction
                        feature_name = f"{col1}_x_{col2}"
                        X_train[feature_name] = X_train[col1] * X_train[col2]
                        X_test[feature_name] = X_test[col1] * X_test[col2]
                        self.interaction_features.append(feature_name)
                        interaction_count += 1
                        
                        # Addition interaction
                        if interaction_count < max_interactions:
                            feature_name = f"{col1}_plus_{col2}"
                            X_train[feature_name] = X_train[col1] + X_train[col2]
                            X_test[feature_name] = X_test[col1] + X_test[col2]
                            self.interaction_features.append(feature_name)
                            interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Expanded interaction feature creation failed for {col1} x {col2}: {e}")
                        continue
            
            # Three-way interactions
            if len(numeric_features_selected) >= 3:
                max_three_way = 10
                three_way_count = 0
                
                for i in range(min(5, len(numeric_features_selected))):
                    for j in range(i + 1, min(5, len(numeric_features_selected))):
                        for k in range(j + 1, min(5, len(numeric_features_selected))):
                            if three_way_count >= max_three_way:
                                break
                            
                            col1, col2, col3 = numeric_features_selected[i], numeric_features_selected[j], numeric_features_selected[k]
                            
                            try:
                                feature_name = f"{col1}_x_{col2}_x_{col3}"
                                X_train[feature_name] = X_train[col1] * X_train[col2] * X_train[col3]
                                X_test[feature_name] = X_test[col1] * X_test[col2] * X_test[col3]
                                self.interaction_features.append(feature_name)
                                three_way_count += 1
                                
                            except Exception as e:
                                logger.warning(f"Three-way interaction failed for {col1}x{col2}x{col3}: {e}")
                                continue
            
            logger.info(f"Expanded interaction features completed: {len(self.interaction_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Expanded interaction features failed: {e}")
            return X_train, X_test
    
    def _create_enhanced_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced statistical features with more complexity"""
        try:
            if len(self.numerical_features) < 3:
                return X_train, X_test
            
            logger.info("Creating enhanced statistical features")
            
            numeric_cols = [col for col in self.numerical_features if col in X_train.columns][:12]
            
            if len(numeric_cols) < 3:
                return X_train, X_test
            
            numeric_data_train = X_train[numeric_cols].values
            numeric_data_test = X_test[numeric_cols].values
            
            X_train['row_mean'] = np.mean(numeric_data_train, axis=1)
            X_test['row_mean'] = np.mean(numeric_data_test, axis=1)
            
            X_train['row_std'] = np.std(numeric_data_train, axis=1)
            X_test['row_std'] = np.std(numeric_data_test, axis=1)
            
            X_train['row_min'] = np.min(numeric_data_train, axis=1)
            X_test['row_min'] = np.min(numeric_data_test, axis=1)
            
            X_train['row_max'] = np.max(numeric_data_train, axis=1)
            X_test['row_max'] = np.max(numeric_data_test, axis=1)
            
            X_train['row_median'] = np.median(numeric_data_train, axis=1)
            X_test['row_median'] = np.median(numeric_data_test, axis=1)
            
            X_train['row_q25'] = np.percentile(numeric_data_train, 25, axis=1)
            X_test['row_q25'] = np.percentile(numeric_data_test, 25, axis=1)
            
            X_train['row_q75'] = np.percentile(numeric_data_train, 75, axis=1)
            X_test['row_q75'] = np.percentile(numeric_data_test, 75, axis=1)
            
            X_train['row_range'] = X_train['row_max'] - X_train['row_min']
            X_test['row_range'] = X_test['row_max'] - X_test['row_min']
            
            X_train['row_skew'] = np.apply_along_axis(lambda x: pd.Series(x).skew(), 1, numeric_data_train)
            X_test['row_skew'] = np.apply_along_axis(lambda x: pd.Series(x).skew(), 1, numeric_data_test)
            
            for feature in ['row_mean', 'row_std', 'row_min', 'row_max', 'row_median', 
                           'row_q25', 'row_q75', 'row_range', 'row_skew']:
                X_train[feature] = X_train[feature].fillna(0)
                X_test[feature] = X_test[feature].fillna(0)
            
            enhanced_stats = ['row_mean', 'row_std', 'row_min', 'row_max', 'row_median', 
                             'row_q25', 'row_q75', 'row_range', 'row_skew']
            self.statistical_features.extend(enhanced_stats)
            
            logger.info(f"Enhanced statistical features completed: {len(self.statistical_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced statistical features failed: {e}")
            return X_train, X_test
    
    def _create_cv_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create frequency features with cross-validation to prevent overfitting"""
        try:
            if len(self.categorical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating CV frequency features")
            
            cat_features_selected = [col for col in self.categorical_features if col in X_train.columns][:8]
            
            for col in cat_features_selected:
                try:
                    X_train[col] = X_train[col].astype('object')
                    X_test[col] = X_test[col].astype('object')
                    
                    encoded_train = np.zeros(len(X_train))
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    
                    for train_idx, val_idx in kf.split(X_train):
                        freq_map = X_train[col].iloc[train_idx].value_counts().to_dict()
                        encoded_train[val_idx] = X_train[col].iloc[val_idx].map(freq_map).fillna(0)
                    
                    freq_map_full = X_train[col].value_counts().to_dict()
                    encoded_test = X_test[col].map(freq_map_full).fillna(0)
                    
                    feature_name = f"{col}_frequency_cv"
                    X_train[feature_name] = encoded_train
                    X_test[feature_name] = encoded_test
                    
                    feature_name_norm = f"{col}_frequency_norm"
                    max_freq = max(encoded_train.max(), encoded_test.max())
                    if max_freq > 0:
                        X_train[feature_name_norm] = encoded_train / max_freq
                        X_test[feature_name_norm] = encoded_test / max_freq
                        self.frequency_features.append(feature_name_norm)
                    
                    self.frequency_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"CV frequency encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"CV frequency features completed: {len(self.frequency_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"CV frequency features failed: {e}")
            return X_train, X_test
    
    def _create_enhanced_cross_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced cross features with better combinations"""
        try:
            if len(self.categorical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating enhanced cross features")
            
            cat_features_selected = [col for col in self.categorical_features if col in X_train.columns][:6]
            
            cross_count = 0
            max_crosses = 15
            
            for i in range(len(cat_features_selected)):
                for j in range(i + 1, len(cat_features_selected)):
                    if cross_count >= max_crosses:
                        break
                    
                    col1, col2 = cat_features_selected[i], cat_features_selected[j]
                    
                    try:
                        X_train[col1] = X_train[col1].astype('object')
                        X_train[col2] = X_train[col2].astype('object')
                        X_test[col1] = X_test[col1].astype('object')
                        X_test[col2] = X_test[col2].astype('object')
                        
                        feature_name = f"{col1}_cross_{col2}"
                        X_train[feature_name] = X_train[col1].astype(str) + "_X_" + X_train[col2].astype(str)
                        X_test[feature_name] = X_test[col1].astype(str) + "_X_" + X_test[col2].astype(str)
                        
                        self.cross_features.append(feature_name)
                        cross_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Enhanced cross feature creation failed for {col1} x {col2}: {e}")
                        continue
            
            logger.info(f"Enhanced cross features completed: {len(self.cross_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced cross features failed: {e}")
            return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create temporal features if time-related columns exist"""
        try:
            time_columns = []
            for col in X_train.columns:
                if any(time_keyword in col.lower() for time_keyword in ['time', 'date', 'hour', 'day', 'minute']):
                    time_columns.append(col)
            
            if not time_columns:
                return X_train, X_test
            
            logger.info(f"Creating temporal features from: {time_columns}")
            
            for col in time_columns:
                try:
                    if col in X_train.columns and X_train[col].dtype in ['int64', 'float64']:
                        if 'hour' in col.lower():
                            X_train[f'{col}_sin'] = np.sin(2 * np.pi * X_train[col] / 24)
                            X_train[f'{col}_cos'] = np.cos(2 * np.pi * X_train[col] / 24)
                            X_test[f'{col}_sin'] = np.sin(2 * np.pi * X_test[col] / 24)
                            X_test[f'{col}_cos'] = np.cos(2 * np.pi * X_test[col] / 24)
                            self.temporal_features.extend([f'{col}_sin', f'{col}_cos'])
                        
                        elif 'day' in col.lower():
                            X_train[f'{col}_sin'] = np.sin(2 * np.pi * X_train[col] / 7)
                            X_train[f'{col}_cos'] = np.cos(2 * np.pi * X_train[col] / 7)
                            X_test[f'{col}_sin'] = np.sin(2 * np.pi * X_test[col] / 7)
                            X_test[f'{col}_cos'] = np.cos(2 * np.pi * X_test[col] / 7)
                            self.temporal_features.extend([f'{col}_sin', f'{col}_cos'])
                
                except Exception as e:
                    logger.warning(f"Temporal feature creation failed for {col}: {e}")
                    continue
            
            if self.temporal_features:
                logger.info(f"Temporal features completed: {len(self.temporal_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Temporal features failed: {e}")
            return X_train, X_test
    
    def _create_enhanced_binning_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced binning features with multiple strategies"""
        try:
            if len(self.numerical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating enhanced binning features")
            
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:8]
            
            for col in numeric_features_selected:
                try:
                    if X_train[col].nunique() > 10:
                        # Quantile binning
                        try:
                            bins = np.percentile(X_train[col].dropna(), [0, 25, 50, 75, 100])
                            bins = np.unique(bins)
                            
                            if len(bins) > 2:
                                train_binned = pd.cut(X_train[col], bins=bins, labels=False, duplicates='drop')
                                test_binned = pd.cut(X_test[col], bins=bins, labels=False, duplicates='drop')
                                
                                feature_name = f"{col}_binned"
                                X_train[feature_name] = train_binned.fillna(0).astype('int32')
                                X_test[feature_name] = test_binned.fillna(0).astype('int32')
                                self.binning_features.append(feature_name)
                        
                        except Exception as e:
                            logger.warning(f"Quantile binning failed for {col}: {e}")
                        
                        # Standard deviation binning
                        try:
                            mean = X_train[col].mean()
                            std = X_train[col].std()
                            
                            if std > 0:
                                bins = [float('-inf'), mean - std, mean, mean + std, float('inf')]
                                train_std_binned = pd.cut(X_train[col], bins=bins, labels=False)
                                test_std_binned = pd.cut(X_test[col], bins=bins, labels=False)
                                
                                feature_name = f"{col}_stdbinned"
                                X_train[feature_name] = train_std_binned.fillna(2).astype('int32')
                                X_test[feature_name] = test_std_binned.fillna(2).astype('int32')
                                self.binning_features.append(feature_name)
                        
                        except Exception as e:
                            logger.warning(f"Std binning failed for {col}: {e}")
                    
                except Exception as e:
                    logger.warning(f"Enhanced binning failed for {col}: {e}")
                    continue
            
            logger.info(f"Enhanced binning features completed: {len(self.binning_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced binning features failed: {e}")
            return X_train, X_test
    
    def _create_enhanced_rank_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced rank features with multiple ranking methods"""
        try:
            if len(self.numerical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating enhanced rank features")
            
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:6]
            
            for col in numeric_features_selected:
                try:
                    # Standard percentile rank
                    feature_name = f"{col}_pctrank"
                    X_train[feature_name] = X_train[col].rank(pct=True)
                    X_test[feature_name] = X_test[col].rank(pct=True)
                    self.rank_features.append(feature_name)
                    
                    # Dense rank
                    feature_name = f"{col}_denserank"
                    X_train[feature_name] = X_train[col].rank(method='dense')
                    X_test[feature_name] = X_test[col].rank(method='dense')
                    self.rank_features.append(feature_name)
                    
                    # Quantile-based rank
                    try:
                        feature_name = f"{col}_quantilerank"
                        quantiles = X_train[col].quantile([0.25, 0.5, 0.75]).values
                        
                        train_qrank = np.searchsorted(quantiles, X_train[col])
                        test_qrank = np.searchsorted(quantiles, X_test[col])
                        
                        X_train[feature_name] = train_qrank
                        X_test[feature_name] = test_qrank
                        self.rank_features.append(feature_name)
                        
                    except Exception as e:
                        logger.warning(f"Quantile rank failed for {col}: {e}")
                    
                except Exception as e:
                    logger.warning(f"Enhanced rank feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Enhanced rank features completed: {len(self.rank_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced rank features failed: {e}")
            return X_train, X_test
    
    def _create_enhanced_ratio_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced ratio features with safe operations"""
        try:
            if len(self.numerical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating enhanced ratio features")
            
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:8]
            
            ratio_count = 0
            max_ratios = 20
            
            for i in range(len(numeric_features_selected)):
                for j in range(i + 1, len(numeric_features_selected)):
                    if ratio_count >= max_ratios:
                        break
                    
                    col1, col2 = numeric_features_selected[i], numeric_features_selected[j]
                    
                    try:
                        epsilon = 1e-8
                        
                        # Ratio feature col1/col2
                        feature_name = f"{col1}_div_{col2}"
                        denominator_train = X_train[col2].replace(0, epsilon)
                        denominator_test = X_test[col2].replace(0, epsilon)
                        
                        X_train[feature_name] = X_train[col1] / denominator_train
                        X_test[feature_name] = X_test[col1] / denominator_test
                        
                        X_train[feature_name] = np.clip(X_train[feature_name], -1e6, 1e6)
                        X_test[feature_name] = np.clip(X_test[feature_name], -1e6, 1e6)
                        
                        self.ratio_features.append(feature_name)
                        ratio_count += 1
                        
                        # Reverse ratio col2/col1
                        if ratio_count < max_ratios:
                            feature_name = f"{col2}_div_{col1}"
                            denominator_train = X_train[col1].replace(0, epsilon)
                            denominator_test = X_test[col1].replace(0, epsilon)
                            
                            X_train[feature_name] = X_train[col2] / denominator_train
                            X_test[feature_name] = X_test[col2] / denominator_test
                            
                            X_train[feature_name] = np.clip(X_train[feature_name], -1e6, 1e6)
                            X_test[feature_name] = np.clip(X_test[feature_name], -1e6, 1e6)
                            
                            self.ratio_features.append(feature_name)
                            ratio_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Enhanced ratio feature creation failed for {col1}/{col2}: {e}")
                        continue
            
            logger.info(f"Enhanced ratio features completed: {len(self.ratio_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced ratio features failed: {e}")
            return X_train, X_test
    
    def _create_enhanced_polynomial_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced polynomial features with multiple degrees"""
        try:
            if len(self.numerical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating enhanced polynomial features")
            
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:5]
            
            for col in numeric_features_selected:
                try:
                    # Squared feature
                    feature_name = f"{col}_squared"
                    X_train[feature_name] = X_train[col] ** 2
                    X_test[feature_name] = X_test[col] ** 2
                    self.polynomial_features.append(feature_name)
                    
                    # Cubic feature
                    feature_name = f"{col}_cubed"
                    X_train[feature_name] = X_train[col] ** 3
                    X_test[feature_name] = X_test[col] ** 3
                    self.polynomial_features.append(feature_name)
                    
                    # Square root feature
                    try:
                        if X_train[col].min() >= 0:
                            feature_name = f"{col}_sqrt"
                            X_train[feature_name] = np.sqrt(np.abs(X_train[col]))
                            X_test[feature_name] = np.sqrt(np.abs(X_test[col]))
                            self.polynomial_features.append(feature_name)
                    except Exception:
                        pass
                    
                    # Log feature
                    try:
                        if X_train[col].min() > 0:
                            feature_name = f"{col}_log"
                            X_train[feature_name] = np.log1p(X_train[col])
                            X_test[feature_name] = np.log1p(X_test[col])
                            self.polynomial_features.append(feature_name)
                    except Exception:
                        pass
                    
                except Exception as e:
                    logger.warning(f"Enhanced polynomial feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Enhanced polynomial features completed: {len(self.polynomial_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced polynomial features failed: {e}")
            return X_train, X_test
    
    def _create_clustering_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create clustering-based features"""
        try:
            if len(self.numerical_features) < 3:
                return X_train, X_test
            
            logger.info("Creating clustering features")
            
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            cluster_features = [col for col in self.numerical_features if col in X_train.columns][:10]
            
            if len(cluster_features) < 3:
                return X_train, X_test
            
            cluster_data_train = X_train[cluster_features].fillna(0)
            cluster_data_test = X_test[cluster_features].fillna(0)
            
            scaler = StandardScaler()
            cluster_data_train_scaled = scaler.fit_transform(cluster_data_train)
            cluster_data_test_scaled = scaler.transform(cluster_data_test)
            
            for n_clusters in [5, 10, 15]:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    
                    train_clusters = kmeans.fit_predict(cluster_data_train_scaled)
                    test_clusters = kmeans.predict(cluster_data_test_scaled)
                    
                    feature_name = f"cluster_{n_clusters}"
                    X_train[feature_name] = train_clusters
                    X_test[feature_name] = test_clusters
                    
                    train_distances = np.min(kmeans.transform(cluster_data_train_scaled), axis=1)
                    test_distances = np.min(kmeans.transform(cluster_data_test_scaled), axis=1)
                    
                    feature_name = f"cluster_distance_{n_clusters}"
                    X_train[feature_name] = train_distances
                    X_test[feature_name] = test_distances
                    
                    self.clustering_features.extend([f"cluster_{n_clusters}", f"cluster_distance_{n_clusters}"])
                    
                except Exception as e:
                    logger.warning(f"Clustering with {n_clusters} clusters failed: {e}")
                    continue
            
            logger.info(f"Clustering features completed: {len(self.clustering_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Clustering features failed: {e}")
            return X_train, X_test
    
    def _create_advanced_math_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create advanced mathematical features"""
        try:
            if len(self.numerical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating advanced mathematical features")
            
            math_features = [col for col in self.numerical_features if col in X_train.columns][:8]
            
            for i, col in enumerate(math_features):
                try:
                    # Trigonometric features
                    if X_train[col].std() > 0:
                        col_normalized_train = (X_train[col] - X_train[col].min()) / (X_train[col].max() - X_train[col].min()) * 2 * np.pi
                        col_normalized_test = (X_test[col] - X_train[col].min()) / (X_train[col].max() - X_train[col].min()) * 2 * np.pi
                        
                        X_train[f"{col}_sin"] = np.sin(col_normalized_train)
                        X_test[f"{col}_sin"] = np.sin(col_normalized_test)
                        
                        X_train[f"{col}_cos"] = np.cos(col_normalized_train)
                        X_test[f"{col}_cos"] = np.cos(col_normalized_test)
                        
                        self.advanced_features.extend([f"{col}_sin", f"{col}_cos"])
                    
                    # Exponential features
                    if X_train[col].max() < 10:
                        X_train[f"{col}_exp"] = np.exp(np.clip(X_train[col], -10, 10))
                        X_test[f"{col}_exp"] = np.exp(np.clip(X_test[col], -10, 10))
                        self.advanced_features.append(f"{col}_exp")
                    
                except Exception as e:
                    logger.warning(f"Advanced math features failed for {col}: {e}")
                    continue
            
            # Pairwise advanced operations
            for i in range(min(4, len(math_features))):
                for j in range(i + 1, min(4, len(math_features))):
                    col1, col2 = math_features[i], math_features[j]
                    
                    try:
                        # Euclidean distance
                        feature_name = f"{col1}_{col2}_euclidean"
                        X_train[feature_name] = np.sqrt(X_train[col1]**2 + X_train[col2]**2)
                        X_test[feature_name] = np.sqrt(X_test[col1]**2 + X_test[col2]**2)
                        self.advanced_features.append(feature_name)
                        
                    except Exception as e:
                        logger.warning(f"Pairwise advanced feature failed for {col1}, {col2}: {e}")
                        continue
            
            logger.info(f"Advanced mathematical features completed: {len(self.advanced_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Advanced mathematical features failed: {e}")
            return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original target encoding method for fallback"""
        return self._create_enhanced_target_encoding_features(X_train, X_test, y_train)
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original interaction method for fallback"""
        return self._create_expanded_interaction_features(X_train, X_test, y_train)
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original statistical method for fallback"""
        return self._create_enhanced_statistical_features(X_train, X_test)
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original frequency method for fallback"""
        return self._create_cv_frequency_features(X_train, X_test, pd.Series([0] * len(X_train)))
    
    def _create_cross_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original cross method for fallback"""
        return self._create_enhanced_cross_features(X_train, X_test, y_train)
    
    def _create_binning_features_fixed(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original binning method for fallback"""
        return self._create_enhanced_binning_features(X_train, X_test)
    
    def _create_rank_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original rank method for fallback"""
        return self._create_enhanced_rank_features(X_train, X_test)
    
    def _create_ratio_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original ratio method for fallback"""
        return self._create_enhanced_ratio_features(X_train, X_test)
    
    def _create_polynomial_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Original polynomial method for fallback"""
        return self._create_enhanced_polynomial_features(X_train, X_test)
    
    def _encode_categorical_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical feature encoding"""
        try:
            all_categorical_features = self.categorical_features + self.cross_features
            
            for col in all_categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        train_str = X_train[col].astype('object').fillna('missing')
                        test_str = X_test[col].astype('object').fillna('missing')
                        
                        all_values = sorted(set(train_str.unique()) | set(test_str.unique()))
                        value_map = {val: idx for idx, val in enumerate(all_values)}
                        
                        X_train[col] = train_str.map(value_map).fillna(0).astype('int32')
                        X_test[col] = test_str.map(value_map).fillna(0).astype('int32')
                        
                    except Exception as e:
                        logger.warning(f"Safe categorical encoding failed for {col}: {e}")
                        X_train[col] = 0
                        X_test[col] = 0
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Safe categorical feature encoding failed: {e}")
            return X_train, X_test
    
    def _encode_categorical_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical encoding for quick mode"""
        try:
            for col in self.categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    train_str = X_train[col].astype('object')
                    test_str = X_test[col].astype('object')
                    
                    all_values = sorted(set(train_str.unique()) | set(test_str.unique()))
                    value_map = {val: idx for idx, val in enumerate(all_values)}
                    
                    X_train[col] = train_str.map(value_map).fillna(0)
                    X_test[col] = test_str.map(value_map).fillna(0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Safe categorical encoding failed: {e}")
            return X_train, X_test
    
    def _create_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create numeric features"""
        try:
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                    except:
                        pass
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Numeric feature creation failed: {e}")
            return X_train, X_test
    
    def _normalize_numeric_basic(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic numeric normalization"""
        try:
            for col in self.numerical_features:
                if col in X_train.columns and col in X_test.columns:
                    if X_train[col].std() > 0:
                        mean_val = X_train[col].mean()
                        std_val = X_train[col].std()
                        
                        X_train[col] = (X_train[col] - mean_val) / std_val
                        X_test[col] = (X_test[col] - mean_val) / std_val
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic numeric normalization failed: {e}")
            return X_train, X_test
    
    def _final_data_cleanup_enhanced(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Enhanced final data cleanup"""
        try:
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Remove constant features
            constant_features = []
            for col in X_train.columns:
                if X_train[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                X_train = X_train.drop(columns=constant_features)
                X_test = X_test.drop(columns=constant_features)
                logger.info(f"Removed {len(constant_features)} constant features")
            
            # Remove highly correlated features
            if len(X_train.columns) > 100:
                try:
                    corr_matrix = X_train.corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
                    
                    if to_drop:
                        X_train = X_train.drop(columns=to_drop)
                        X_test = X_test.drop(columns=to_drop)
                        logger.info(f"Removed {len(to_drop)} highly correlated features")
                
                except Exception as e:
                    logger.warning(f"Correlation-based feature removal failed: {e}")
            
            # Ensure consistent data types
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        X_train[col] = X_train[col].astype('float32')
                        X_test[col] = X_test[col].astype('float32')
                    except:
                        pass
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Enhanced final data cleanup failed: {e}")
            return X_train, X_test
    
    def _clean_final_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean final features for quick mode"""
        try:
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final feature cleaning failed: {e}")
            return X_train, X_test
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Finalize processing and update statistics"""
        try:
            self.final_feature_columns = list(X_train.columns)
            
            self.processing_stats['processing_time'] = time.time() - self.processing_stats['start_time']
            self.processing_stats['feature_types_count'] = {
                'original': len(self.original_feature_order),
                'target_encoding': len(self.target_encoding_features),
                'interaction': len(self.interaction_features),
                'statistical': len(self.statistical_features),
                'frequency': len(self.frequency_features),
                'cross': len(self.cross_features),
                'binning': len(self.binning_features),
                'rank': len(self.rank_features),
                'ratio': len(self.ratio_features),
                'polynomial': len(self.polynomial_features),
                'temporal': len(self.temporal_features),
                'clustering': len(self.clustering_features),
                'advanced': len(self.advanced_features),
                'final': len(self.final_feature_columns)
            }
            
            total_generated = sum([
                len(self.target_encoding_features),
                len(self.interaction_features),
                len(self.statistical_features),
                len(self.frequency_features),
                len(self.cross_features),
                len(self.binning_features),
                len(self.rank_features),
                len(self.ratio_features),
                len(self.polynomial_features),
                len(self.temporal_features),
                len(self.clustering_features),
                len(self.advanced_features)
            ])
            
            self.processing_stats['total_features_generated'] = total_generated
            
            logger.info(f"Enhanced feature engineering finalized - {len(self.final_feature_columns)} features in {self.processing_stats['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing finalization failed: {e}")
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features only as fallback"""
        try:
            logger.info("Creating basic features only (fallback)")
            
            if target_col in train_df.columns:
                X_train = train_df.drop(columns=[target_col]).copy()
            else:
                X_train = train_df.copy()
            
            X_test = test_df.copy()
            
            common_cols = sorted(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            X_train = self._convert_categorical_to_object(X_train)
            X_test = self._convert_categorical_to_object(X_test)
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                except:
                    pass
            
            logger.info(f"Basic features created - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic features creation failed: {e}")
            raise
    
    def _create_minimal_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal features as emergency fallback"""
        try:
            logger.warning("Creating minimal features (emergency fallback)")
            
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            if target_col in numeric_cols:
                numeric_cols = [col for col in numeric_cols if col != target_col]
            
            if len(numeric_cols) > 10:
                numeric_cols = numeric_cols[:10]
            elif len(numeric_cols) == 0:
                X_train = pd.DataFrame({'feature_0': [0] * len(train_df)})
                X_test = pd.DataFrame({'feature_0': [0] * len(test_df)})
                return X_train, X_test
            
            X_train = train_df[numeric_cols].fillna(0)
            X_test = test_df[numeric_cols].fillna(0)
            
            logger.info(f"Minimal features created - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Minimal features creation failed: {e}")
            X_train = pd.DataFrame({'feature_0': [0] * len(train_df)})
            X_test = pd.DataFrame({'feature_0': [0] * len(test_df)})
            return X_train, X_test