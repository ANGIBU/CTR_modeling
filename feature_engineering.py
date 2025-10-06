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
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import multiprocessing as mp
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    """CTR feature engineering optimized for tree models"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        self.quick_mode = False
        self.memory_efficient_mode = True
        
        self.label_encoders = {}
        self.feature_stats = {}
        self.numerical_features = []
        self.categorical_features = []
        self.target_encoding_features = []
        self.interaction_features = []
        self.removed_columns = []
        self.original_feature_order = []
        self.final_feature_columns = []
        self.target_column = None
        
        self.true_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
        
        self.all_continuous = (
            [f'feat_a_{i}' for i in range(1, 19)] +
            [f'feat_b_{i}' for i in range(1, 7)] +
            [f'feat_c_{i}' for i in range(1, 9)] +
            [f'feat_d_{i}' for i in range(1, 7)] +
            [f'feat_e_{i}' for i in range(1, 11)] +
            [f'history_a_{i}' for i in range(1, 8)] +
            [f'history_b_{i}' for i in range(1, 31)] +
            [f'l_feat_{i}' for i in range(1, 28)]
        )
        
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
        """Enable or disable quick mode"""
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
        """Main feature engineering pipeline for tree models"""
        if self.quick_mode:
            logger.info("=== Quick Mode Feature Engineering Started ===")
            return self.create_quick_features(train_df, test_df, target_col)
        else:
            logger.info("=== Tree Model Feature Engineering Started ===")
            return self.create_tree_model_features(train_df, test_df, target_col)
    
    def create_quick_features(self,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features for quick testing"""
        logger.info("Creating basic features for quick mode")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            available_features = list(set(self.true_categorical + self.all_continuous) & set(X_train.columns))
            X_train = X_train[available_features]
            X_test = X_test[available_features]
            
            X_train, X_test = self._encode_categorical_minimal(X_train, X_test)
            X_train, X_test = self._fill_missing_values(X_train, X_test)
            X_train, X_test = self._ensure_float32(X_train, X_test)
            
            self.final_feature_columns = list(X_train.columns)
            
            logger.info(f"Quick feature engineering completed: {X_train.shape[1]} features")
            logger.info(f"Final features - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Quick feature engineering failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def create_tree_model_features(self, 
                                  train_df: pd.DataFrame, 
                                  test_df: pd.DataFrame, 
                                  target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features optimized for tree models"""
        logger.info("Creating features for tree models")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            self._force_memory_cleanup()
            
            available_features = list(set(self.true_categorical + self.all_continuous) & set(X_train.columns))
            logger.info(f"Available features: {len(available_features)} (5 categorical + {len(available_features)-5} continuous)")
            
            X_train = X_train[available_features]
            X_test = X_test[available_features]
            
            X_train, X_test = self._encode_categorical_minimal(X_train, X_test)
            self._force_memory_cleanup()
            
            X_train, X_test = self._fill_missing_values_efficient(X_train, X_test)
            self._force_memory_cleanup()
            
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['available_gb'] > 8 and not self.quick_mode:
                X_train, X_test = self._create_target_encoding_minimal(X_train, X_test, y_train)
                self._force_memory_cleanup()
            
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['available_gb'] > 10 and not self.quick_mode:
                X_train, X_test = self._create_interaction_features(X_train, X_test)
                self._force_memory_cleanup()
            
            X_train, X_test = self._ensure_float32(X_train, X_test)
            self._force_memory_cleanup()
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== Tree model feature engineering completed: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Tree model feature engineering failed: {e}")
            self._force_memory_cleanup()
            
            logger.warning("Falling back to basic features")
            return self._create_basic_features_only(train_df, test_df, target_col)
    
    def _force_memory_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        self.memory_monitor.force_memory_cleanup()
    
    def _encode_categorical_minimal(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Minimal categorical encoding for tree models"""
        try:
            logger.info("Encoding categorical features")
            
            for col in self.true_categorical:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        X_train_copy = X_train[col].copy()
                        X_test_copy = X_test[col].copy()
                        
                        if pd.api.types.is_categorical_dtype(X_train_copy):
                            train_str = X_train_copy.astype(str)
                        else:
                            train_str = X_train_copy.fillna('missing').astype(str)
                        
                        if pd.api.types.is_categorical_dtype(X_test_copy):
                            test_str = X_test_copy.astype(str)
                        else:
                            test_str = X_test_copy.fillna('missing').astype(str)
                        
                        all_categories = sorted(set(train_str.unique()) | set(test_str.unique()))
                        category_map = {cat: idx for idx, cat in enumerate(all_categories)}
                        
                        X_train[col] = train_str.map(category_map).fillna(0).astype('float32')
                        X_test[col] = test_str.map(category_map).fillna(0).astype('float32')
                        
                        self.categorical_features.append(col)
                        
                    except Exception as e:
                        logger.warning(f"Encoding failed for {col}: {e}")
                        X_train[col] = 0.0
                        X_test[col] = 0.0
            
            logger.info(f"Categorical encoding completed: {len(self.categorical_features)} features")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Categorical encoding failed: {e}")
            return X_train, X_test
    
    def _fill_missing_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fill missing values"""
        try:
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Missing value filling failed: {e}")
            return X_train, X_test
    
    def _fill_missing_values_efficient(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fill missing values with memory efficiency"""
        try:
            logger.info("Filling missing values efficiently")
            
            chunk_size = 10
            train_cols = list(X_train.columns)
            
            for i in range(0, len(train_cols), chunk_size):
                chunk_cols = train_cols[i:i+chunk_size]
                
                for col in chunk_cols:
                    if X_train[col].isna().any():
                        X_train[col] = X_train[col].fillna(0)
                    if np.isinf(X_train[col]).any():
                        X_train[col] = X_train[col].replace([np.inf, -np.inf], 0)
                
                for col in chunk_cols:
                    if col in X_test.columns:
                        if X_test[col].isna().any():
                            X_test[col] = X_test[col].fillna(0)
                        if np.isinf(X_test[col]).any():
                            X_test[col] = X_test[col].replace([np.inf, -np.inf], 0)
                
                if i % (chunk_size * 3) == 0:
                    gc.collect()
            
            logger.info("Missing value filling completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Efficient missing value filling failed: {e}")
            try:
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)
                X_train = X_train.replace([np.inf, -np.inf], 0)
                X_test = X_test.replace([np.inf, -np.inf], 0)
            except:
                pass
            return X_train, X_test
    
    def _ensure_float32(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ensure all features are float32"""
        try:
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
            logger.error(f"Float32 conversion failed: {e}")
            return X_train, X_test
    
    def _create_target_encoding_minimal(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                       y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Minimal target encoding for categorical features"""
        try:
            logger.info("Creating target encoding features")
            
            for col in self.categorical_features[:3]:
                if col in X_train.columns:
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
                        
                        feature_name = f"{col}_target_encoded"
                        X_train[feature_name] = encoded_col
                        X_test[feature_name] = test_encoded
                        self.target_encoding_features.append(feature_name)
                        
                    except Exception as e:
                        logger.warning(f"Target encoding failed for {col}: {e}")
                        continue
            
            logger.info(f"Target encoding completed: {len(self.target_encoding_features)} features")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Target encoding features failed: {e}")
            return X_train, X_test
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create interaction features between continuous variables"""
        try:
            logger.info("Creating interaction features")
            
            continuous_cols = [col for col in X_train.columns if col in self.all_continuous]
            
            if len(continuous_cols) < 2:
                logger.warning("Not enough continuous features for interactions")
                return X_train, X_test
            
            important_features = []
            for prefix in ['feat_e', 'feat_d', 'history_a']:
                important_features.extend([col for col in continuous_cols if col.startswith(prefix)])
            
            important_features = important_features[:15]
            
            interaction_count = 0
            max_interactions = self.config.FEATURE_ENGINEERING_CONFIG['max_numeric_for_interaction']
            
            for i in range(len(important_features)):
                for j in range(i + 1, len(important_features)):
                    if interaction_count >= max_interactions:
                        break
                    
                    col1 = important_features[i]
                    col2 = important_features[j]
                    
                    try:
                        mult_name = f"{col1}_x_{col2}"
                        X_train[mult_name] = (X_train[col1] * X_train[col2]).astype('float32')
                        X_test[mult_name] = (X_test[col1] * X_test[col2]).astype('float32')
                        self.interaction_features.append(mult_name)
                        interaction_count += 1
                        
                        if interaction_count >= max_interactions:
                            break
                        
                        ratio_name = f"{col1}_div_{col2}"
                        denominator_train = X_train[col2].replace(0, 1e-10)
                        denominator_test = X_test[col2].replace(0, 1e-10)
                        X_train[ratio_name] = (X_train[col1] / denominator_train).astype('float32')
                        X_test[ratio_name] = (X_test[col1] / denominator_test).astype('float32')
                        
                        X_train[ratio_name] = X_train[ratio_name].replace([np.inf, -np.inf], 0).fillna(0)
                        X_test[ratio_name] = X_test[ratio_name].replace([np.inf, -np.inf], 0).fillna(0)
                        
                        self.interaction_features.append(ratio_name)
                        interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Interaction creation failed for {col1} and {col2}: {e}")
                        continue
                
                if interaction_count >= max_interactions:
                    break
            
            logger.info(f"Interaction features created: {len(self.interaction_features)} features")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {e}")
            return X_train, X_test
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """Initialize feature engineering processing"""
        try:
            self.processing_stats['start_time'] = time.time()
            
            self.target_column = self._detect_target_column(train_df, target_col)
            
            self.original_feature_order = sorted([col for col in train_df.columns if col != self.target_column])
            
            mode_info = "QUICK MODE" if self.quick_mode else "TREE MODEL MODE"
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
            
            id_cols = [col for col in X_train.columns if col == 'seq']
            if id_cols:
                X_train = X_train.drop(columns=id_cols, errors='ignore')
                X_test = X_test.drop(columns=id_cols, errors='ignore')
                logger.info(f"Removed ID columns: {id_cols}")
            
            logger.info(f"Data preparation completed - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Finalize processing"""
        try:
            self.final_feature_columns = list(X_train.columns)
            
            self.processing_stats['processing_time'] = time.time() - self.processing_stats['start_time']
            self.processing_stats['feature_types_count'] = {
                'original': len(self.original_feature_order),
                'categorical': len(self.categorical_features),
                'target_encoding': len(self.target_encoding_features),
                'interaction': len(self.interaction_features),
                'final': len(self.final_feature_columns)
            }
            
            total_generated = len(self.target_encoding_features) + len(self.interaction_features)
            self.processing_stats['total_features_generated'] = total_generated
            
            logger.info(f"Feature engineering finalized - {len(self.final_feature_columns)} features in {self.processing_stats['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Processing finalization failed: {e}")
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                    target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features only"""
        try:
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            available_features = list(set(self.true_categorical + self.all_continuous) & set(X_train.columns))
            X_train = X_train[available_features]
            X_test = X_test[available_features]
            
            X_train, X_test = self._encode_categorical_minimal(X_train, X_test)
            X_train, X_test = self._fill_missing_values_efficient(X_train, X_test)
            X_train, X_test = self._ensure_float32(X_train, X_test)
            
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