# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging
import time
import gc
import warnings
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        self.memory_efficient_mode = True
        
        self.label_encoders = {}
        self.feature_stats = {}
        self.generated_features = []
        self.numerical_features = []
        self.categorical_features = []
        self.removed_columns = []
        self.original_feature_order = []
        self.final_feature_columns = []
        self.target_column = None
        
        self.target_encoding_maps = {}
        
        self.target_feature_count = 117
        self.expected_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
        self.expected_continuous_patterns = [
            'feat_a_', 'feat_b_', 'feat_c_', 'feat_d_', 'feat_e_',
            'history_a_', 'history_b_', 'l_feat_'
        ]
        
        self.processing_stats = {
            'start_time': None,
            'processing_time': 0,
            'memory_usage': 0,
            'feature_types_count': {},
            'total_features_generated': 0
        }
    
    def set_memory_efficient_mode(self, enabled: bool):
        self.memory_efficient_mode = enabled
        if enabled:
            logger.info("Memory efficient mode enabled")
        else:
            logger.info("Memory efficient mode disabled")
    
    def engineer_features(self, 
                         train_df: pd.DataFrame, 
                         test_df: pd.DataFrame, 
                         target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=== Feature Engineering with Target Encoding and Interactions ===")
        return self.create_enhanced_features(train_df, test_df, target_col)
    
    def create_enhanced_features(self,
                                train_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Creating features with target encoding and interactions")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            X_train, X_test, y_train = self._prepare_reference_data(train_df, test_df, target_col)
            self._force_memory_cleanup()
            
            self._identify_feature_types_reference(X_train)
            
            X_train, X_test = self._prepare_features_reference(X_train, X_test)
            self._force_memory_cleanup()
            
            X_train, X_test = self._encode_categorical_reference(X_train, X_test)
            self._force_memory_cleanup()
            
            logger.info("Creating target encoding features")
            X_train, X_test = self._create_target_encoding_features(X_train, X_test, y_train)
            self._force_memory_cleanup()
            
            logger.info("Creating interaction features")
            X_train = self._create_interaction_features(X_train)
            X_test = self._create_interaction_features(X_test)
            self._force_memory_cleanup()
            
            logger.info("Creating statistical features")
            X_train = self._create_statistical_features(X_train)
            X_test = self._create_statistical_features(X_test)
            self._force_memory_cleanup()
            
            X_train, X_test = self._final_data_cleanup(X_train, X_test)
            self._force_memory_cleanup()
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== Feature engineering completed: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            self._force_memory_cleanup()
            
            logger.warning("Falling back to basic features")
            return self._create_basic_features_only(train_df, test_df, target_col)
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                        y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create target encoding features with smoothing"""
        try:
            logger.info("Applying target encoding with smoothing")
            
            cat_cols = ['inventory_id', 'gender', 'age_group', 'day_of_week', 'hour']
            global_mean = y_train.mean()
            smoothing = 100
            
            for col in cat_cols:
                if col not in X_train.columns:
                    continue
                
                try:
                    temp_df = pd.DataFrame({col: X_train[col], 'target': y_train})
                    target_mean = temp_df.groupby(col)['target'].agg(['mean', 'count'])
                    
                    smooth_target = (target_mean['count'] * target_mean['mean'] + smoothing * global_mean) / (target_mean['count'] + smoothing)
                    
                    self.target_encoding_maps[col] = smooth_target.to_dict()
                    
                    X_train[f'{col}_target_enc'] = X_train[col].map(smooth_target).fillna(global_mean).astype('float32')
                    X_test[f'{col}_target_enc'] = X_test[col].map(smooth_target).fillna(global_mean).astype('float32')
                    
                    logger.info(f"Target encoded {col}: {len(smooth_target)} unique values")
                    
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
                    continue
            
            logger.info("Target encoding completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Target encoding failed: {e}")
            return X_train, X_test
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        try:
            if 'hour' in df.columns and 'inventory_id' in df.columns:
                df['hour_inv_inter'] = (df['hour'].astype(str) + '_' + df['inventory_id'].astype(str)).astype('category').cat.codes.astype('float32')
            
            if 'day_of_week' in df.columns and 'inventory_id' in df.columns:
                df['dow_inv_inter'] = (df['day_of_week'].astype(str) + '_' + df['inventory_id'].astype(str)).astype('category').cat.codes.astype('float32')
            
            if 'gender' in df.columns and 'age_group' in df.columns:
                df['gender_age_inter'] = (df['gender'].astype(str) + '_' + df['age_group'].astype(str)).astype('category').cat.codes.astype('float32')
            
            if 'hour' in df.columns:
                df['time_period'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], 
                                          labels=[0, 1, 2, 3], include_lowest=True).astype('float32')
            
            logger.info("Interaction features created")
            return df
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {e}")
            return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from history columns"""
        try:
            history_b_cols = [col for col in df.columns if col.startswith('history_b_')]
            if len(history_b_cols) > 0:
                df['history_b_mean'] = df[history_b_cols].mean(axis=1).astype('float32')
                df['history_b_std'] = df[history_b_cols].std(axis=1).astype('float32')
                df['history_b_max'] = df[history_b_cols].max(axis=1).astype('float32')
                df['history_b_min'] = df[history_b_cols].min(axis=1).astype('float32')
                logger.info(f"Created statistics for {len(history_b_cols)} history_b features")
            
            feat_a_cols = [col for col in df.columns if col.startswith('feat_a_')]
            if len(feat_a_cols) > 0:
                df['feat_a_mean'] = df[feat_a_cols].mean(axis=1).astype('float32')
                df['feat_a_std'] = df[feat_a_cols].std(axis=1).astype('float32')
                logger.info(f"Created statistics for {len(feat_a_cols)} feat_a features")
            
            logger.info("Statistical features created")
            return df
            
        except Exception as e:
            logger.error(f"Statistical feature creation failed: {e}")
            return df
    
    def _prepare_reference_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                               target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        try:
            logger.info("Preparing data")
            
            if target_col in train_df.columns:
                y_train = train_df[target_col].copy()
                X_train = train_df.drop(columns=[target_col])
            else:
                y_train = pd.Series([0] * len(train_df))
                X_train = train_df.copy()
            
            X_test = test_df.copy()
            
            if 'seq' in X_train.columns:
                logger.info("Removing 'seq' column from training data")
                X_train = X_train.drop(columns=['seq'])
            
            if 'seq' in X_test.columns:
                logger.info("Removing 'seq' column from test data")
                X_test = X_test.drop(columns=['seq'])
            
            id_cols = [col for col in X_train.columns 
                      if ('id' in col.lower() or 'ID' in col) 
                      and col.lower() not in ['inventory_id']]
            
            if id_cols:
                logger.info(f"Removing ID columns (preserving inventory_id): {id_cols}")
                X_train = X_train.drop(columns=id_cols, errors='ignore')
                X_test = X_test.drop(columns=id_cols, errors='ignore')
            
            logger.info(f"Data preparation completed - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Columns preserved: {X_train.columns.tolist()}")
            
            return X_train, X_test, y_train
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def _identify_feature_types_reference(self, X: pd.DataFrame):
        try:
            logger.info("Identifying feature types")
            
            categorical_features = []
            for expected_cat in self.expected_categorical:
                for col in X.columns:
                    if expected_cat in col.lower():
                        if X[col].nunique() < 10000:
                            categorical_features.append(col)
                            break
            
            self.categorical_features = categorical_features
            
            continuous_features = []
            for col in X.columns:
                if col not in categorical_features:
                    for pattern in self.expected_continuous_patterns:
                        if pattern in col:
                            if X[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                                continuous_features.append(col)
                                break
            
            self.numerical_features = continuous_features
            
            logger.info(f"Feature identification - Categorical: {len(self.categorical_features)}, "
                       f"Continuous: {len(self.numerical_features)}")
            logger.info(f"Categorical features: {self.categorical_features}")
            logger.info(f"Total features: {len(self.categorical_features) + len(self.numerical_features)}")
            
        except Exception as e:
            logger.error(f"Feature identification failed: {e}")
            self.numerical_features = list(X.select_dtypes(include=[np.number]).columns)
            self.categorical_features = []
    
    def _prepare_features_reference(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logger.info("Preparing features (no normalization for tree models)")
            
            for col in X_train.columns:
                try:
                    if col in self.categorical_features:
                        train_values = X_train[col].fillna('missing')
                        test_values = X_test[col].fillna('missing')
                        
                        train_values = train_values.astype(str).str.replace('None', 'missing', regex=False)
                        test_values = test_values.astype(str).str.replace('None', 'missing', regex=False)
                        
                        train_values = train_values.str.replace('.0', '', regex=False)
                        test_values = test_values.str.replace('.0', '', regex=False)
                        
                        try:
                            X_train[col] = pd.to_numeric(train_values, errors='coerce').fillna(-1).astype('int32')
                            X_test[col] = pd.to_numeric(test_values, errors='coerce').fillna(-1).astype('int32')
                        except (ValueError, TypeError):
                            X_train[col] = train_values.astype('object')
                            X_test[col] = test_values.astype('object')
                    else:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                        
                except Exception as e:
                    logger.warning(f"Feature preparation failed for {col}: {e}")
                    X_train[col] = 0.0
                    X_test[col] = 0.0
            
            logger.info(f"Feature preparation completed - Features: {X_train.shape[1]}")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return X_train, X_test
    
    def _encode_categorical_reference(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logger.info("Applying categorical encoding")
            
            for col in self.categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        train_str = X_train[col].fillna('missing').astype(str).str.replace('None', 'missing', regex=False)
                        test_str = X_test[col].fillna('missing').astype(str).str.replace('None', 'missing', regex=False)
                        
                        train_str = train_str.str.replace('.0', '', regex=False)
                        test_str = test_str.str.replace('.0', '', regex=False)
                        
                        train_unique = set(train_str.unique())
                        test_unique = set(test_str.unique())
                        all_unique = train_unique | test_unique
                        
                        value_counts = train_str.value_counts()
                        sorted_by_freq = [val for val in value_counts.index if val in all_unique]
                        
                        remaining = sorted([val for val in all_unique if val not in sorted_by_freq])
                        sorted_values = sorted_by_freq + remaining
                        
                        mapping = {val: idx for idx, val in enumerate(sorted_values)}
                        
                        X_train[col] = train_str.map(mapping).fillna(0).astype('float32')
                        X_test[col] = test_str.map(mapping).fillna(0).astype('float32')
                        
                        logger.info(f"Encoded {col}: {len(all_unique)} unique values")
                        
                    except Exception as e:
                        logger.warning(f"Encoding failed for {col}: {e}")
                        continue
            
            logger.info(f"Categorical encoding completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Categorical encoding failed: {e}")
            return X_train, X_test
    
    def _force_memory_cleanup(self):
        gc.collect()
        self.memory_monitor.force_memory_cleanup()
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        try:
            self.processing_stats['start_time'] = time.time()
            
            self.target_column = self._detect_target_column(train_df, target_col)
            
            self.original_feature_order = sorted([col for col in train_df.columns if col != self.target_column])
            
            logger.info(f"Feature engineering initialization")
            logger.info(f"Initial data: Training {train_df.shape}, Test {test_df.shape}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Original feature count: {len(self.original_feature_order)}")
            
            self.memory_monitor.log_memory_status("Initialization")
            
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            self.target_column = target_col
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
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
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logger.info("Starting final data cleanup")
            
            non_numeric_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            non_numeric_test = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
            
            if non_numeric_train:
                logger.warning(f"Removing remaining non-numeric columns from train: {non_numeric_train}")
                X_train = X_train.drop(columns=non_numeric_train)
            
            if non_numeric_test:
                logger.warning(f"Removing remaining non-numeric columns from test: {non_numeric_test}")
                X_test = X_test.drop(columns=non_numeric_test)
            
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
                    except Exception as e:
                        logger.warning(f"Type conversion failed for {col}: {e}")
                        pass
            
            logger.info(f"Final data types - Train: {X_train.dtypes.value_counts().to_dict()}")
            logger.info(f"Final data types - Test: {X_test.dtypes.value_counts().to_dict()}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final data cleanup failed: {e}")
            return X_train, X_test
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        try:
            self.final_feature_columns = list(X_train.columns)
            
            self.processing_stats['processing_time'] = time.time() - self.processing_stats['start_time']
            self.processing_stats['feature_types_count'] = {
                'original': len(self.original_feature_order),
                'final': len(self.final_feature_columns)
            }
            
            self.processing_stats['total_features_generated'] = len(self.final_feature_columns) - len(self.original_feature_order)
            
            logger.info(f"Feature engineering finalized - {len(self.final_feature_columns)} features in {self.processing_stats['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing finalization failed: {e}")
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                    target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            X_train, X_test, y_train = self._prepare_reference_data(train_df, test_df, target_col)
            self._identify_feature_types_reference(X_train)
            X_train, X_test = self._prepare_features_reference(X_train, X_test)
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            X_train, X_test = self._encode_categorical_reference(X_train, X_test)
            X_train, X_test = self._final_data_cleanup(X_train, X_test)
            
            self.final_feature_columns = list(X_train.columns)
            
            logger.info(f"Basic features only: {X_train.shape[1]} features")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic features creation failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def _create_minimal_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            X_train, X_test, y_train = self._prepare_reference_data(train_df, test_df, target_col)
            
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