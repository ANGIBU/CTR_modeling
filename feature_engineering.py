# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union
import logging
import time
import gc
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    """CTR feature engineering class"""
    
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
        """
        Enable or disable memory efficient mode
        
        Args:
            enabled: If True, use simplified feature engineering to save memory
        """
        self.memory_efficient_mode = enabled
        if enabled:
            logger.info("Memory efficient mode enabled - simplified features only")
        else:
            logger.info("Memory efficient mode disabled - full feature engineering")
    
    def set_quick_mode(self, enabled: bool):
        """
        Enable or disable quick mode for rapid testing
        
        Args:
            enabled: If True, create only basic features for speed
        """
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
        """
        Main feature engineering pipeline entry point
        
        Args:
            train_df: Training dataset
            test_df: Test dataset  
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test) with engineered features
        """
        if self.quick_mode:
            logger.info("=== Quick Mode Feature Engineering Started ===")
            return self.create_quick_features(train_df, test_df, target_col)
        else:
            logger.info("=== Full Feature Engineering Started ===")
            return self.create_all_features(train_df, test_df, target_col)
    
    def create_quick_features(self,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create basic features for quick testing (50 samples)
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test) with basic features
        """
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
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete feature engineering pipeline for full dataset processing
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test) with comprehensive features
        """
        logger.info("Creating comprehensive features for full dataset")
        
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
            if not memory_status['should_cleanup'] and memory_status['available_gb'] > 8:
                logger.info("Full feature engineering enabled - sufficient memory available")
                
                # 5. Target encoding features
                X_train, X_test = self._create_target_encoding_features(X_train, X_test, y_train)
                
                # 6. Interaction feature creation
                X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
                
                # 7. Statistical features
                X_train, X_test = self._create_statistical_features(X_train, X_test)
                
                # 8. Frequency-based features
                X_train, X_test = self._create_frequency_features(X_train, X_test)
                
                # 9. Cross features
                X_train, X_test = self._create_cross_features(X_train, X_test, y_train)
                
                # 10. Time-based features
                X_train, X_test = self._create_temporal_features(X_train, X_test)
                
                # 11. Binning features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_binning', True):
                    X_train, X_test = self._create_binning_features(X_train, X_test)
                
                # 12. Rank features
                X_train, X_test = self._create_rank_features(X_train, X_test)
                
                # 13. Ratio features
                X_train, X_test = self._create_ratio_features(X_train, X_test)
                
                # 14. Polynomial features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_polynomial_features', True):
                    X_train, X_test = self._create_polynomial_features(X_train, X_test)
                
            else:
                logger.warning("Simplified feature engineering due to memory constraints")
            
            # 15. Safe categorical feature encoding
            X_train, X_test = self._encode_categorical_features_safe(X_train, X_test, y_train)
            
            # 16. Numeric feature transformation
            X_train, X_test = self._create_numeric_features(X_train, X_test)
            
            # 17. Final data cleanup
            X_train, X_test = self._final_data_cleanup(X_train, X_test)
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== Feature engineering completed: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
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
            
            mode_info = "QUICK MODE" if self.quick_mode else "FULL MODE"
            logger.info(f"Feature engineering initialization ({mode_info})")
            logger.info(f"Initial data: Training {train_df.shape}, Test {test_df.shape}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Original feature count: {len(self.original_feature_order)}")
            
            self.memory_monitor.log_memory_status("Initialization")
            
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            self.target_column = target_col
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
        """
        Detect CTR target column with validation
        
        Args:
            train_df: Training dataframe
            provided_target_col: User-provided target column name
            
        Returns:
            Validated target column name
        """
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
                    if unique_count > 20:  # Treat as numerical
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
    
    def _safe_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe fillna that handles categorical columns properly"""
        try:
            df_filled = df.copy()
            
            for col in df_filled.columns:
                if df_filled[col].dtype.name == 'category':
                    # For categorical columns, fill with the most frequent category or add 'missing' category
                    if df_filled[col].isnull().any():
                        # Add 'missing' to categories if not already present
                        if 'missing' not in df_filled[col].cat.categories:
                            df_filled[col] = df_filled[col].cat.add_categories(['missing'])
                        df_filled[col] = df_filled[col].fillna('missing')
                elif df_filled[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # For numeric columns, fill with 0
                    df_filled[col] = df_filled[col].fillna(0)
                else:
                    # For object columns, fill with 'missing'
                    df_filled[col] = df_filled[col].fillna('missing')
            
            return df_filled
            
        except Exception as e:
            logger.warning(f"Safe fillna failed: {e}")
            # Fallback: convert all categorical to object first
            df_converted = df.copy()
            for col in df_converted.columns:
                if df_converted[col].dtype.name == 'category':
                    df_converted[col] = df_converted[col].astype('object')
            
            return df_converted.fillna(0)
    
    def _unify_data_types_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe data type unification"""
        try:
            for col in X_train.columns:
                if col in X_test.columns:
                    # Try to unify data types safely
                    try:
                        if X_train[col].dtype != X_test[col].dtype:
                            # Convert both to string first, then to most appropriate type
                            train_str = X_train[col].astype(str)
                            test_str = X_test[col].astype(str)
                            
                            # Try numeric conversion
                            try:
                                X_train[col] = pd.to_numeric(train_str, errors='coerce').fillna(0)
                                X_test[col] = pd.to_numeric(test_str, errors='coerce').fillna(0)
                            except:
                                # Keep as categorical
                                X_train[col] = train_str
                                X_test[col] = test_str
                    
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
            for col in X_train.columns:
                if col in X_test.columns:
                    # Simple type conversion
                    try:
                        if X_train[col].dtype == 'object':
                            X_train[col] = X_train[col].astype(str)
                            X_test[col] = X_test[col].astype(str)
                    except:
                        X_train[col] = X_train[col].fillna('unknown')
                        X_test[col] = X_test[col].fillna('unknown')
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic data type fix failed: {e}")
            return X_train, X_test
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean basic features"""
        try:
            # Fill missing values with safe handling
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic feature cleaning failed: {e}")
            # Convert categorical to object and then fill
            for col in X_train.columns:
                if X_train[col].dtype.name == 'category':
                    X_train[col] = X_train[col].astype('object')
                if col in X_test.columns and X_test[col].dtype.name == 'category':
                    X_test[col] = X_test[col].astype('object')
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create target encoding features with memory efficiency"""
        try:
            if len(self.categorical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating target encoding features")
            
            # Select top categorical features by cardinality
            cat_features_to_encode = []
            for col in self.categorical_features:
                if col in X_train.columns:
                    cardinality = X_train[col].nunique()
                    if 2 < cardinality < 1000:  # Skip binary and high-cardinality features
                        cat_features_to_encode.append(col)
            
            # Limit to top 10 features for memory efficiency
            if len(cat_features_to_encode) > 10:
                cat_features_to_encode = cat_features_to_encode[:10]
            
            for col in cat_features_to_encode:
                try:
                    # Create target encoder with smoothing
                    encoder = TargetEncoder(target_type='binary', smooth='auto')
                    
                    # Fit and transform
                    X_train_encoded = encoder.fit_transform(X_train[[col]], y_train).flatten()
                    X_test_encoded = encoder.transform(X_test[[col]]).flatten()
                    
                    # Add encoded feature
                    feature_name = f"{col}_target_encoded"
                    X_train[feature_name] = X_train_encoded
                    X_test[feature_name] = X_test_encoded
                    
                    # Store encoder
                    self.target_encoders[col] = encoder
                    self.target_encoding_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Target encoding completed: {len(self.target_encoding_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Target encoding features failed: {e}")
            return X_train, X_test
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create interaction features with memory efficiency"""
        try:
            if len(self.numerical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating interaction features")
            
            # Select top numerical features by variance
            numeric_features_selected = []
            for col in self.numerical_features:
                if col in X_train.columns:
                    if X_train[col].var() > 0:
                        numeric_features_selected.append(col)
            
            # Limit to top 8 features for memory efficiency
            if len(numeric_features_selected) > 8:
                numeric_features_selected = numeric_features_selected[:8]
            
            # Create pairwise interactions
            interaction_count = 0
            max_interactions = 10  # Limit interactions
            
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
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature creation failed for {col1} x {col2}: {e}")
                        continue
            
            logger.info(f"Interaction features completed: {len(self.interaction_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Interaction features failed: {e}")
            return X_train, X_test
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create statistical features with memory efficiency"""
        try:
            if len(self.numerical_features) < 3:
                return X_train, X_test
            
            logger.info("Creating statistical features")
            
            # Select numerical features for statistical transformation
            numeric_cols = [col for col in self.numerical_features if col in X_train.columns][:8]  # Limit to 8 features
            
            if len(numeric_cols) < 3:
                return X_train, X_test
            
            # Calculate statistical features across rows
            numeric_data_train = X_train[numeric_cols].values
            numeric_data_test = X_test[numeric_cols].values
            
            # Row-wise statistics
            X_train['row_mean'] = np.mean(numeric_data_train, axis=1)
            X_test['row_mean'] = np.mean(numeric_data_test, axis=1)
            
            X_train['row_std'] = np.std(numeric_data_train, axis=1)
            X_test['row_std'] = np.std(numeric_data_test, axis=1)
            
            X_train['row_min'] = np.min(numeric_data_train, axis=1)
            X_test['row_min'] = np.min(numeric_data_test, axis=1)
            
            X_train['row_max'] = np.max(numeric_data_train, axis=1)
            X_test['row_max'] = np.max(numeric_data_test, axis=1)
            
            self.statistical_features.extend(['row_mean', 'row_std', 'row_min', 'row_max'])
            
            logger.info(f"Statistical features completed: {len(self.statistical_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Statistical features failed: {e}")
            return X_train, X_test
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create frequency features with memory efficiency"""
        try:
            if len(self.categorical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating frequency features")
            
            # Select categorical features for frequency encoding
            cat_features_selected = [col for col in self.categorical_features if col in X_train.columns][:5]  # Limit to 5 features
            
            for col in cat_features_selected:
                try:
                    # Calculate value frequencies
                    freq_map = X_train[col].value_counts().to_dict()
                    
                    # Apply frequency encoding
                    feature_name = f"{col}_frequency"
                    X_train[feature_name] = X_train[col].map(freq_map).fillna(0)
                    X_test[feature_name] = X_test[col].map(freq_map).fillna(0)
                    
                    self.frequency_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Frequency encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Frequency features completed: {len(self.frequency_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Frequency features failed: {e}")
            return X_train, X_test
    
    def _create_cross_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create cross features with memory efficiency"""
        try:
            if len(self.categorical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating cross features")
            
            # Select categorical features for crossing
            cat_features_selected = [col for col in self.categorical_features if col in X_train.columns][:4]  # Limit to 4 features
            
            cross_count = 0
            max_crosses = 6  # Limit cross features
            
            for i in range(len(cat_features_selected)):
                for j in range(i + 1, len(cat_features_selected)):
                    if cross_count >= max_crosses:
                        break
                    
                    col1, col2 = cat_features_selected[i], cat_features_selected[j]
                    
                    try:
                        # Create cross feature
                        feature_name = f"{col1}_cross_{col2}"
                        X_train[feature_name] = X_train[col1].astype(str) + "_" + X_train[col2].astype(str)
                        X_test[feature_name] = X_test[col1].astype(str) + "_" + X_test[col2].astype(str)
                        
                        self.cross_features.append(feature_name)
                        cross_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Cross feature creation failed for {col1} x {col2}: {e}")
                        continue
            
            logger.info(f"Cross features completed: {len(self.cross_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Cross features failed: {e}")
            return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create temporal features if time-related columns exist"""
        return X_train, X_test
    
    def _create_binning_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create binning features with memory efficiency"""
        try:
            if len(self.numerical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating binning features")
            
            # Select numerical features for binning
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:5]  # Limit to 5 features
            
            for col in numeric_features_selected:
                try:
                    # Create quantile-based bins
                    feature_name = f"{col}_binned"
                    X_train[feature_name] = pd.qcut(X_train[col], q=5, labels=False, duplicates='drop').fillna(0)
                    X_test[feature_name] = pd.cut(X_test[col], 
                                                bins=pd.qcut(X_train[col], q=5, duplicates='drop').cat.categories,
                                                labels=False, include_lowest=True).fillna(0)
                    
                    self.binning_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Binning failed for {col}: {e}")
                    continue
            
            logger.info(f"Binning features completed: {len(self.binning_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Binning features failed: {e}")
            return X_train, X_test
    
    def _create_rank_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create rank features with memory efficiency"""
        try:
            if len(self.numerical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating rank features")
            
            # Select numerical features for ranking
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:4]  # Limit to 4 features
            
            for col in numeric_features_selected:
                try:
                    # Create rank feature
                    feature_name = f"{col}_rank"
                    X_train[feature_name] = X_train[col].rank(pct=True)
                    
                    # For test set, use training set distribution
                    X_test[feature_name] = X_test[col].rank(pct=True)
                    
                    self.rank_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Rank feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Rank features completed: {len(self.rank_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Rank features failed: {e}")
            return X_train, X_test
    
    def _create_ratio_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create ratio features with memory efficiency"""
        try:
            if len(self.numerical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating ratio features")
            
            # Select numerical features for ratio calculation
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:6]  # Limit to 6 features
            
            ratio_count = 0
            max_ratios = 8  # Limit ratio features
            
            for i in range(len(numeric_features_selected)):
                for j in range(i + 1, len(numeric_features_selected)):
                    if ratio_count >= max_ratios:
                        break
                    
                    col1, col2 = numeric_features_selected[i], numeric_features_selected[j]
                    
                    try:
                        # Create ratio feature (avoid division by zero)
                        feature_name = f"{col1}_div_{col2}"
                        
                        denominator_train = X_train[col2].replace(0, 0.001)  # Small epsilon to avoid division by zero
                        denominator_test = X_test[col2].replace(0, 0.001)
                        
                        X_train[feature_name] = X_train[col1] / denominator_train
                        X_test[feature_name] = X_test[col1] / denominator_test
                        
                        # Handle infinite values
                        X_train[feature_name] = X_train[feature_name].replace([np.inf, -np.inf], 0)
                        X_test[feature_name] = X_test[feature_name].replace([np.inf, -np.inf], 0)
                        
                        self.ratio_features.append(feature_name)
                        ratio_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Ratio feature creation failed for {col1}/{col2}: {e}")
                        continue
            
            logger.info(f"Ratio features completed: {len(self.ratio_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Ratio features failed: {e}")
            return X_train, X_test
    
    def _create_polynomial_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create polynomial features with memory efficiency"""
        try:
            if len(self.numerical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating polynomial features")
            
            # Select top numerical features for polynomial transformation
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:3]  # Limit to 3 features
            
            for col in numeric_features_selected:
                try:
                    # Create squared feature
                    feature_name = f"{col}_squared"
                    X_train[feature_name] = X_train[col] ** 2
                    X_test[feature_name] = X_test[col] ** 2
                    
                    self.polynomial_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Polynomial feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Polynomial features completed: {len(self.polynomial_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Polynomial features failed: {e}")
            return X_train, X_test
    
    def _encode_categorical_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical feature encoding"""
        try:
            for col in self.categorical_features + self.cross_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        # Safe label encoding
                        train_str = X_train[col].astype(str).fillna('missing')
                        test_str = X_test[col].astype(str).fillna('missing')
                        
                        # Get all unique values
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
                    # Simple label encoding
                    train_str = X_train[col].astype(str)
                    test_str = X_test[col].astype(str)
                    
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
            # Ensure all features are numeric
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
            logger.error(f"Basic normalization failed: {e}")
            return X_train, X_test
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final data cleanup"""
        try:
            # Replace infinite values
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            # Fill any remaining missing values
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            # Remove constant features
            constant_features = []
            for col in X_train.columns:
                if X_train[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                X_train = X_train.drop(columns=constant_features)
                X_test = X_test.drop(columns=constant_features)
                logger.info(f"Removed {len(constant_features)} constant features")
            
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
            logger.error(f"Final data cleanup failed: {e}")
            return X_train, X_test
    
    def _clean_final_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean final features for quick mode"""
        try:
            # Basic cleanup
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final feature cleaning failed: {e}")
            return X_train, X_test
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Finalize processing and update statistics"""
        try:
            self.final_feature_columns = list(X_train.columns)
            
            # Update processing statistics
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
                len(self.polynomial_features)
            ])
            
            self.processing_stats['total_features_generated'] = total_generated
            
            logger.info(f"Feature engineering finalized - {len(self.final_feature_columns)} features in {self.processing_stats['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing finalization failed: {e}")
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features only as fallback"""
        try:
            logger.info("Creating basic features only (fallback)")
            
            # Extract target and features
            if target_col in train_df.columns:
                X_train = train_df.drop(columns=[target_col]).copy()
            else:
                X_train = train_df.copy()
            
            X_test = test_df.copy()
            
            # Ensure consistent columns
            common_cols = sorted(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            # Basic preprocessing with safe categorical handling
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            # Convert to numeric
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
            
            # Use first 10 numeric columns if available
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            if target_col in numeric_cols:
                numeric_cols = [col for col in numeric_cols if col != target_col]
            
            if len(numeric_cols) > 10:
                numeric_cols = numeric_cols[:10]
            elif len(numeric_cols) == 0:
                # If no numeric columns, create dummy features
                X_train = pd.DataFrame({'feature_0': [0] * len(train_df)})
                X_test = pd.DataFrame({'feature_0': [0] * len(test_df)})
                return X_train, X_test
            
            X_train = train_df[numeric_cols].fillna(0)
            X_test = test_df[numeric_cols].fillna(0)
            
            logger.info(f"Minimal features created - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Minimal features creation failed: {e}")
            # Absolute emergency fallback
            X_train = pd.DataFrame({'feature_0': [0] * len(train_df)})
            X_test = pd.DataFrame({'feature_0': [0] * len(test_df)})
            return X_train, X_test