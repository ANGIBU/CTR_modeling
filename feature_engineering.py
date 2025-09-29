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
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import KFold
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    """CTR feature engineering class with memory optimization"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        self.quick_mode = False
        self.memory_efficient_mode = True
        
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
        
        self.target_feature_count = 200
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
            logger.info("=== Memory-Optimized Feature Engineering Started ===")
            return self.create_optimized_features(train_df, test_df, target_col)
    
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
            
            X_train, X_test = self._normalize_numeric_basic(X_train, X_test)
            
            X_train, X_test = self._clean_final_features(X_train, X_test)
            
            self.final_feature_columns = list(X_train.columns)
            
            logger.info(f"Quick feature engineering completed: {X_train.shape[1]} features")
            logger.info(f"Final features - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Quick feature engineering failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def create_optimized_features(self, 
                                 train_df: pd.DataFrame, 
                                 test_df: pd.DataFrame, 
                                 target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized feature engineering pipeline"""
        logger.info("Creating features with memory optimization")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            self._force_memory_cleanup()
            
            self._classify_columns(X_train)
            
            X_train, X_test = self._convert_to_numeric_safe(X_train, X_test)
            self._force_memory_cleanup()
            
            X_train, X_test = self._clean_basic_features(X_train, X_test)
            self._force_memory_cleanup()
            
            memory_status = self.memory_monitor.get_memory_status()
            logger.info(f"Memory available before feature generation: {memory_status['available_gb']:.1f}GB")
            
            if memory_status['available_gb'] > 8:
                X_train, X_test = self._create_target_encoding_features_optimized(X_train, X_test, y_train)
                self._force_memory_cleanup()
                
                X_train, X_test = self._create_interaction_features_optimized(X_train, X_test, y_train)
                self._force_memory_cleanup()
                
                X_train, X_test = self._create_statistical_features_optimized(X_train, X_test)
                self._force_memory_cleanup()
                
                X_train, X_test = self._create_frequency_features_optimized(X_train, X_test, y_train)
                self._force_memory_cleanup()
                
                X_train, X_test = self._create_cross_features_optimized(X_train, X_test, y_train)
                self._force_memory_cleanup()
                
                X_train, X_test = self._create_temporal_features(X_train, X_test)
                self._force_memory_cleanup()
                
                X_train, X_test = self._create_rank_features_optimized(X_train, X_test)
                self._force_memory_cleanup()
                
                X_train, X_test = self._create_ratio_features_optimized(X_train, X_test)
                self._force_memory_cleanup()
                
            else:
                logger.warning("Limited memory - using minimal feature set")
                X_train, X_test = self._create_target_encoding_features_optimized(X_train, X_test, y_train)
                self._force_memory_cleanup()
            
            X_train, X_test = self._encode_remaining_categorical(X_train, X_test, y_train)
            self._force_memory_cleanup()
            
            X_train, X_test = self._create_numeric_features(X_train, X_test)
            self._force_memory_cleanup()
            
            logger.info(f"Starting feature selection: {X_train.shape[1]} -> {self.target_feature_count} features")
            X_train, X_test = self._select_important_features(X_train, X_test, y_train)
            self._force_memory_cleanup()
            
            X_train, X_test = self._final_data_cleanup_optimized(X_train, X_test)
            self._force_memory_cleanup()
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== Memory-optimized feature engineering completed: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Optimized feature engineering failed: {e}")
            self._force_memory_cleanup()
            
            logger.warning("Falling back to basic features")
            return self._create_basic_features_only(train_df, test_df, target_col)
    
    def _force_memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        self.memory_monitor.force_memory_cleanup()
    
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
    
    def _select_important_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select top important features"""
        try:
            current_feature_count = X_train.shape[1]
            
            if current_feature_count <= self.target_feature_count:
                logger.info(f"Current feature count ({current_feature_count}) already within target")
                return X_train, X_test
            
            logger.info(f"Feature selection: {current_feature_count} -> {self.target_feature_count}")
            
            selector = SelectKBest(score_func=f_classif, k=self.target_feature_count)
            
            sample_size = min(100000, len(X_train))
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]
            
            selector.fit(X_sample, y_sample)
            
            selected_mask = selector.get_support()
            selected_features = X_train.columns[selected_mask].tolist()
            
            scores = selector.scores_
            for i, feature in enumerate(X_train.columns):
                if not np.isnan(scores[i]):
                    self.feature_importance_scores[feature] = float(scores[i])
            
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
            
            logger.info(f"Feature selection completed: {len(selected_features)} features selected")
            
            del selector, X_sample, y_sample
            gc.collect()
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return X_train, X_test
    
    def _create_target_encoding_features_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized target encoding"""
        try:
            logger.info("Creating target encoding features (optimized)")
            
            categorical_features = [col for col in self.categorical_features if col in X_train.columns][:15]
            
            for col in categorical_features:
                try:
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    encoded_col = np.zeros(len(X_train), dtype='float32')
                    
                    for train_idx, val_idx in kf.split(X_train):
                        X_fold_train = X_train.iloc[train_idx]
                        y_fold_train = y_train.iloc[train_idx]
                        
                        means = X_fold_train[[col]].assign(target=y_fold_train).groupby(col)['target'].mean()
                        
                        encoded_col[val_idx] = X_train.iloc[val_idx][col].map(means).fillna(y_train.mean()).astype('float32')
                    
                    train_means = X_train[[col]].assign(target=y_train).groupby(col)['target'].mean()
                    test_encoded = X_test[col].map(train_means).fillna(y_train.mean()).astype('float32')
                    
                    feature_name = f"{col}_target_encoded_cv"
                    X_train[feature_name] = encoded_col
                    X_test[feature_name] = test_encoded
                    self.target_encoding_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Target encoding completed: {len(self.target_encoding_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Target encoding features failed: {e}")
            return X_train, X_test
    
    def _create_interaction_features_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                               y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized interaction features"""
        try:
            logger.info("Creating interaction features (optimized)")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns][:10]
            
            for i in range(min(5, len(numeric_features))):
                for j in range(i + 1, min(5, len(numeric_features))):
                    col1, col2 = numeric_features[i], numeric_features[j]
                    
                    try:
                        feature_name = f"{col1}_mul_{col2}"
                        X_train[feature_name] = (X_train[col1] * X_train[col2]).astype('float32')
                        X_test[feature_name] = (X_test[col1] * X_test[col2]).astype('float32')
                        self.interaction_features.append(feature_name)
                        
                        feature_name = f"{col1}_add_{col2}"
                        X_train[feature_name] = (X_train[col1] + X_train[col2]).astype('float32')
                        X_test[feature_name] = (X_test[col1] + X_test[col2]).astype('float32')
                        self.interaction_features.append(feature_name)
                        
                    except Exception as e:
                        logger.warning(f"Interaction failed for {col1}, {col2}: {e}")
                        continue
            
            logger.info(f"Interaction features completed: {len(self.interaction_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Interaction features failed: {e}")
            return X_train, X_test
    
    def _create_statistical_features_optimized(self, X_train: pd.DataFrame, 
                                               X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized statistical features"""
        try:
            logger.info("Creating statistical features (optimized)")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns][:8]
            
            if len(numeric_features) >= 3:
                feature_subset_train = X_train[numeric_features].astype('float32')
                feature_subset_test = X_test[numeric_features].astype('float32')
                
                X_train['row_mean'] = feature_subset_train.mean(axis=1).astype('float32')
                X_test['row_mean'] = feature_subset_test.mean(axis=1).astype('float32')
                self.statistical_features.append('row_mean')
                
                X_train['row_std'] = feature_subset_train.std(axis=1).astype('float32')
                X_test['row_std'] = feature_subset_test.std(axis=1).astype('float32')
                self.statistical_features.append('row_std')
                
                X_train['row_max'] = feature_subset_train.max(axis=1).astype('float32')
                X_test['row_max'] = feature_subset_test.max(axis=1).astype('float32')
                self.statistical_features.append('row_max')
                
                X_train['row_min'] = feature_subset_train.min(axis=1).astype('float32')
                X_test['row_min'] = feature_subset_test.min(axis=1).astype('float32')
                self.statistical_features.append('row_min')
                
                del feature_subset_train, feature_subset_test
                gc.collect()
            
            logger.info(f"Statistical features completed: {len(self.statistical_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Statistical features failed: {e}")
            return X_train, X_test
    
    def _create_frequency_features_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                             y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized frequency features"""
        try:
            logger.info("Creating frequency features (optimized)")
            
            categorical_features = [col for col in self.categorical_features if col in X_train.columns][:10]
            
            for col in categorical_features:
                try:
                    freq_map = X_train[col].value_counts(normalize=True).to_dict()
                    
                    feature_name = f"{col}_frequency"
                    X_train[feature_name] = X_train[col].map(freq_map).fillna(0).astype('float32')
                    X_test[feature_name] = X_test[col].map(freq_map).fillna(0).astype('float32')
                    self.frequency_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Frequency encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Frequency features completed: {len(self.frequency_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Frequency features failed: {e}")
            return X_train, X_test
    
    def _create_cross_features_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                         y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized cross features"""
        try:
            logger.info("Creating cross features (optimized)")
            
            categorical_features = [col for col in self.categorical_features if col in X_train.columns][:8]
            
            for i in range(min(3, len(categorical_features))):
                for j in range(i + 1, min(3, len(categorical_features))):
                    col1, col2 = categorical_features[i], categorical_features[j]
                    
                    try:
                        feature_name = f"{col1}_cross_{col2}"
                        train_cross = X_train[col1].astype(str) + '_' + X_train[col2].astype(str)
                        test_cross = X_test[col1].astype(str) + '_' + X_test[col2].astype(str)
                        
                        all_categories = sorted(set(train_cross.unique()) | set(test_cross.unique()))
                        category_map = {cat: idx for idx, cat in enumerate(all_categories)}
                        
                        X_train[feature_name] = train_cross.map(category_map).fillna(0).astype('float32')
                        X_test[feature_name] = test_cross.map(category_map).fillna(0).astype('float32')
                        self.cross_features.append(feature_name)
                        
                    except Exception as e:
                        logger.warning(f"Cross feature failed for {col1}, {col2}: {e}")
                        continue
            
            logger.info(f"Cross features completed: {len(self.cross_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Cross features failed: {e}")
            return X_train, X_test
    
    def _create_rank_features_optimized(self, X_train: pd.DataFrame, 
                                        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized rank features"""
        try:
            logger.info("Creating rank features (optimized)")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns][:8]
            
            for col in numeric_features:
                try:
                    feature_name = f"{col}_rank"
                    X_train[feature_name] = X_train[col].rank(method='dense').astype('float32')
                    X_test[feature_name] = X_test[col].rank(method='dense').astype('float32')
                    self.rank_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Rank feature failed for {col}: {e}")
                    continue
            
            logger.info(f"Rank features completed: {len(self.rank_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Rank features failed: {e}")
            return X_train, X_test
    
    def _create_ratio_features_optimized(self, X_train: pd.DataFrame, 
                                         X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-optimized ratio features"""
        try:
            logger.info("Creating ratio features (optimized)")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns][:6]
            
            for i in range(min(3, len(numeric_features))):
                for j in range(i + 1, min(3, len(numeric_features))):
                    col1, col2 = numeric_features[i], numeric_features[j]
                    
                    try:
                        feature_name = f"{col1}_div_{col2}"
                        X_train[feature_name] = (X_train[col1] / (X_train[col2] + 1e-5)).astype('float32')
                        X_test[feature_name] = (X_test[col1] / (X_test[col2] + 1e-5)).astype('float32')
                        self.ratio_features.append(feature_name)
                        
                    except Exception as e:
                        logger.warning(f"Ratio feature failed for {col1}, {col2}: {e}")
                        continue
            
            logger.info(f"Ratio features completed: {len(self.ratio_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Ratio features failed: {e}")
            return X_train, X_test
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """Initialize feature engineering processing"""
        try:
            self.processing_stats['start_time'] = time.time()
            
            self.target_column = self._detect_target_column(train_df, target_col)
            
            self.original_feature_order = sorted([col for col in train_df.columns if col != self.target_column])
            
            mode_info = "QUICK MODE" if self.quick_mode else "OPTIMIZED MODE"
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
    
    def _classify_columns(self, X: pd.DataFrame):
        """Classify columns into numerical and categorical"""
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
    
    def _classify_columns_basic(self, X: pd.DataFrame):
        """Basic column classification"""
        self._classify_columns(X)
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean basic features"""
        try:
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic feature cleaning failed: {e}")
            return X_train, X_test
    
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
    
    def _encode_remaining_categorical(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode remaining categorical features"""
        return X_train, X_test
    
    def _normalize_numeric_basic(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic numeric normalization"""
        try:
            for col in self.numerical_features:
                if col in X_train.columns and col in X_test.columns:
                    mean_val = X_train[col].mean()
                    std_val = X_train[col].std()
                    
                    if std_val > 0:
                        X_train[col] = ((X_train[col] - mean_val) / std_val).astype('float32')
                        X_test[col] = ((X_test[col] - mean_val) / std_val).astype('float32')
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Numeric normalization failed: {e}")
            return X_train, X_test
    
    def _create_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create numeric features"""
        return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create temporal features"""
        try:
            temporal_cols = ['hour', 'day_of_week', 'day', 'month']
            
            for col in temporal_cols:
                if col in X_train.columns:
                    try:
                        if X_train[col].dtype == 'category':
                            X_train[col] = X_train[col].astype('float32')
                        if X_test[col].dtype == 'category':
                            X_test[col] = X_test[col].astype('float32')
                        
                        max_val = max(X_train[col].max(), X_test[col].max())
                        if max_val > 0:
                            feature_name = f"{col}_sin"
                            X_train[feature_name] = np.sin(2 * np.pi * X_train[col] / max_val).astype('float32')
                            X_test[feature_name] = np.sin(2 * np.pi * X_test[col] / max_val).astype('float32')
                            self.temporal_features.append(feature_name)
                            
                            feature_name = f"{col}_cos"
                            X_train[feature_name] = np.cos(2 * np.pi * X_train[col] / max_val).astype('float32')
                            X_test[feature_name] = np.cos(2 * np.pi * X_test[col] / max_val).astype('float32')
                            self.temporal_features.append(feature_name)
                        
                    except Exception as e:
                        logger.warning(f"Temporal feature failed for {col}: {e}")
                        continue
            
            logger.info(f"Temporal features completed: {len(self.temporal_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Temporal features failed: {e}")
            return X_train, X_test
    
    def _final_data_cleanup_optimized(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Optimized final data cleanup"""
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
        return self._final_data_cleanup_optimized(X_train, X_test)
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Finalize processing"""
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
                'rank': len(self.rank_features),
                'ratio': len(self.ratio_features),
                'temporal': len(self.temporal_features),
                'final': len(self.final_feature_columns)
            }
            
            total_generated = sum([
                len(self.target_encoding_features),
                len(self.interaction_features),
                len(self.statistical_features),
                len(self.frequency_features),
                len(self.cross_features),
                len(self.rank_features),
                len(self.ratio_features),
                len(self.temporal_features)
            ])
            
            self.processing_stats['total_features_generated'] = total_generated
            
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
            X_train, X_test = self._normalize_numeric_basic(X_train, X_test)
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