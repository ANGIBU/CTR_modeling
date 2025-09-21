# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import gc
import time
import pickle
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
import threading
import hashlib

# Safe library imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from sklearn.preprocessing import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False
    warnings.warn("TargetEncoder not available. Using manual target encoding.")

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring class"""
    
    def __init__(self):
        self.psutil_available = PSUTIL_AVAILABLE
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            if self.psutil_available:
                process = psutil.Process()
                return process.memory_info().rss / (1024**3)
            else:
                return 2.0  # Default value
        except Exception:
            return 2.0
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get memory status"""
        try:
            if self.psutil_available:
                vm = psutil.virtual_memory()
                usage_gb = (vm.total - vm.available) / (1024**3)
                available_gb = vm.available / (1024**3)
                
                if available_gb < 2:
                    level = 'abort'
                elif available_gb < 4:
                    level = 'critical'
                elif available_gb < 8:
                    level = 'warning'
                else:
                    level = 'normal'
                
                return {
                    'usage_gb': usage_gb,
                    'available_gb': available_gb,
                    'level': level,
                    'should_cleanup': level in ['warning', 'critical', 'abort'],
                    'should_simplify': level in ['critical', 'abort']
                }
            else:
                return {
                    'usage_gb': 2.0,
                    'available_gb': 45.0,
                    'level': 'normal',
                    'should_cleanup': False,
                    'should_simplify': False
                }
        except Exception:
            return {
                'usage_gb': 2.0,
                'available_gb': 45.0,
                'level': 'normal',
                'should_cleanup': False,
                'should_simplify': False
            }
    
    def force_memory_cleanup(self):
        """Memory cleanup"""
        try:
            for _ in range(15):
                gc.collect()
                time.sleep(0.1)
            
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass
        except Exception:
            pass
    
    def log_memory_status(self, context: str = ""):
        """Memory status logging"""
        try:
            status = self.get_memory_status()
            
            if status['level'] != 'normal' or context:
                logger.info(f"Memory status [{context}]: Usage {status['usage_gb']:.1f}GB, "
                           f"Available {status['available_gb']:.1f}GB - {status['level'].upper()}")
                
        except Exception as e:
            logger.warning(f"Memory status logging failed: {e}")

class CTRFeatureEngineer:
    """CTR feature engineering class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.memory_efficient_mode = False  # Default disabled for 64GB environment
        
        # Feature engineering state
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
        
        # Processing statistics
        self.processing_stats = {
            'start_time': None,
            'processing_time': 0,
            'memory_usage': 0,
            'feature_types_count': {},
            'total_features_generated': 0
        }
    
    def set_memory_efficient_mode(self, enabled: bool):
        """Set memory efficient mode"""
        self.memory_efficient_mode = enabled
        if enabled:
            logger.info("Memory efficient mode enabled")
        else:
            logger.info("Memory efficient mode disabled")
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
        """Detect CTR target column"""
        try:
            # First, try provided target column
            if provided_target_col and provided_target_col in train_df.columns:
                unique_values = train_df[provided_target_col].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    positive_ratio = train_df[provided_target_col].mean()
                    if self.config.TARGET_DETECTION_CONFIG['min_ctr'] <= positive_ratio <= self.config.TARGET_DETECTION_CONFIG['max_ctr']:
                        logger.info(f"Target column confirmed: {provided_target_col} (CTR: {positive_ratio:.4f})")
                        return provided_target_col
            
            # Search for CTR pattern in candidate columns
            for candidate in self.config.TARGET_COLUMN_CANDIDATES:
                if candidate in train_df.columns:
                    try:
                        unique_values = train_df[candidate].dropna().unique()
                        if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                            positive_ratio = train_df[candidate].mean()
                            if self.config.TARGET_DETECTION_CONFIG['min_ctr'] <= positive_ratio <= self.config.TARGET_DETECTION_CONFIG['max_ctr']:
                                logger.info(f"CTR target column detected: {candidate} (CTR: {positive_ratio:.4f})")
                                return candidate
                    except Exception:
                        continue
            
            # Search for binary pattern
            for col in train_df.columns:
                if train_df[col].dtype in ['int64', 'int32', 'int8', 'uint8']:
                    unique_values = train_df[col].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        positive_ratio = train_df[col].mean()
                        if 0.001 <= positive_ratio <= 0.1:  # 0.1% ~ 10% range
                            logger.info(f"CTR pattern target column detected: {col} (CTR: {positive_ratio:.4f})")
                            return col
            
            # Use first candidate as default
            if provided_target_col:
                logger.warning(f"Target column '{provided_target_col}' not found, using default")
                return provided_target_col
            else:
                raise ValueError("Target column not found")
                
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return provided_target_col or 'clicked'
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete feature engineering pipeline"""
        logger.info("=== Feature engineering started ===")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            # 1. Basic data preparation
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            # 2. Column classification
            self._classify_columns(X_train)
            
            # 3. Data type unification
            X_train, X_test = self._unify_data_types(X_train, X_test)
            
            # 4. Basic feature cleanup
            X_train, X_test = self._clean_basic_features(X_train, X_test)
            
            # Memory check for full processing
            memory_status = self.memory_monitor.get_memory_status()
            if not memory_status['should_simplify'] and memory_status['available_gb'] > 8:
                logger.info("Full feature engineering enabled - sufficient memory available")
                
                # 5. Interaction feature creation (expanded)
                X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
                
                # 6. Target encoding (tuned)
                X_train, X_test = self._create_target_encoding_features(X_train, X_test, y_train)
                
                # 7. Time-based features
                X_train, X_test = self._create_temporal_features(X_train, X_test)
                
                # 8. Statistical features (expanded)
                X_train, X_test = self._create_statistical_features(X_train, X_test)
                
                # 9. Frequency-based features
                X_train, X_test = self._create_frequency_features(X_train, X_test)
                
                # 10. Polynomial features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_polynomial_features', True):
                    X_train, X_test = self._create_polynomial_features(X_train, X_test)
                
                # 11. Cross features
                X_train, X_test = self._create_cross_features(X_train, X_test, y_train)
                
                # 12. Binning features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_binning', True):
                    X_train, X_test = self._create_binning_features(X_train, X_test)
                
                # 13. Rank features
                X_train, X_test = self._create_rank_features(X_train, X_test)
                
                # 14. Ratio features
                X_train, X_test = self._create_ratio_features(X_train, X_test)
                
            else:
                logger.warning("Simplified feature engineering due to memory constraints")
            
            # 15. Categorical feature encoding
            X_train, X_test = self._encode_categorical_features(X_train, X_test, y_train)
            
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
        """Initialization"""
        try:
            self.processing_stats['start_time'] = time.time()
            
            # Target column detection
            self.target_column = self._detect_target_column(train_df, target_col)
            
            self.original_feature_order = sorted([col for col in train_df.columns if col != self.target_column])
            
            logger.info(f"Initial data: Training {train_df.shape}, Test {test_df.shape}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Original feature count: {len(self.original_feature_order)}")
            
            self.memory_monitor.log_memory_status("Initialization")
            
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            self.target_column = target_col
    
    def _prepare_basic_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Basic data preparation"""
        try:
            # Use detected target column
            actual_target_col = self.target_column
            
            if actual_target_col not in train_df.columns:
                raise ValueError(f"Target column '{actual_target_col}' not found")
            
            X_train = train_df.drop(columns=[actual_target_col]).copy()
            y_train = train_df[actual_target_col].copy()
            X_test = test_df.copy()
            
            # Check target distribution
            target_dist = y_train.value_counts()
            actual_ctr = y_train.mean()
            
            logger.info(f"Target distribution: {target_dist.to_dict()}")
            logger.info(f"Actual CTR: {actual_ctr:.4f}")
            
            return X_train, X_test, y_train
            
        except Exception as e:
            logger.error(f"Basic data preparation failed: {e}")
            raise
    
    def _classify_columns(self, X_train: pd.DataFrame):
        """Column classification"""
        logger.info("Column type classification started")
        
        try:
            self.numerical_features = []
            self.categorical_features = []
            
            for col in X_train.columns:
                try:
                    # Check if numeric
                    if X_train[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                        unique_count = X_train[col].nunique()
                        
                        # High cardinality numeric = numeric
                        if unique_count > 50:
                            self.numerical_features.append(col)
                        # Low cardinality numeric = categorical
                        else:
                            self.categorical_features.append(col)
                    
                    # Object type = categorical
                    elif X_train[col].dtype in ['object', 'category']:
                        self.categorical_features.append(col)
                    
                    # Boolean = categorical
                    elif X_train[col].dtype == 'bool':
                        self.categorical_features.append(col)
                    
                    # Default to numeric
                    else:
                        self.numerical_features.append(col)
                
                except Exception as e:
                    logger.warning(f"Column {col} classification failed: {e}")
                    self.numerical_features.append(col)
            
            logger.info(f"Column classification completed - Numeric: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
            
        except Exception as e:
            logger.error(f"Column classification failed: {e}")
            # Fallback classification
            self.numerical_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
            self.categorical_features = [col for col in X_train.columns if col not in self.numerical_features]
    
    def _unify_data_types(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Data type unification"""
        logger.info("Data type unification started")
        
        try:
            processed_count = 0
            batch_size = 20  # Process in batches for memory efficiency
            
            for i in range(0, len(X_train.columns), batch_size):
                batch_cols = X_train.columns[i:i + batch_size]
                
                for col in batch_cols:
                    try:
                        train_dtype = str(X_train[col].dtype)
                        test_dtype = str(X_test[col].dtype)
                        
                        # Special column handling
                        if col == 'seq' or 'seq' in str(col).lower():
                            X_train[col] = self._safe_hash_column(X_train[col])
                            X_test[col] = self._safe_hash_column(X_test[col])
                            processed_count += 1
                            continue
                        
                        # Resolve type mismatch
                        if train_dtype != test_dtype or train_dtype in ['object', 'category']:
                            try:
                                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                            except Exception:
                                X_train[col] = self._safe_hash_column(X_train[col])
                                X_test[col] = self._safe_hash_column(X_test[col])
                        
                        # Memory optimization
                        elif train_dtype in ['int64', 'float64']:
                            try:
                                if train_dtype == 'int64':
                                    X_train[col], X_test[col] = self._optimize_int_columns(X_train[col], X_test[col])
                                else:
                                    X_train[col] = X_train[col].astype('float32')
                                    X_test[col] = X_test[col].astype('float32')
                            except Exception:
                                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Column {col} type unification failed: {e}")
                        try:
                            X_train[col] = 0.0
                            X_test[col] = 0.0
                        except Exception:
                            pass
                
                if i % (batch_size * 3) == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"Data type unification completed: {processed_count} columns processed")
            
        except Exception as e:
            logger.error(f"Data type unification failed: {e}")
        
        return X_train, X_test
    
    def _safe_hash_column(self, column: pd.Series) -> pd.Series:
        """Safe hash column conversion"""
        try:
            return column.astype(str).apply(lambda x: int(hashlib.md5(str(x).encode()).hexdigest()[:8], 16) % 100000).astype('float32')
        except Exception:
            return pd.Series([0.0] * len(column), dtype='float32', index=column.index)
    
    def _optimize_int_columns(self, train_col: pd.Series, test_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Optimize integer columns"""
        try:
            min_val = min(train_col.min(), test_col.min())
            max_val = max(train_col.max(), test_col.max())
            
            if min_val >= 0 and max_val <= 255:
                return train_col.astype('uint8'), test_col.astype('uint8')
            elif min_val >= -128 and max_val <= 127:
                return train_col.astype('int8'), test_col.astype('int8')
            elif min_val >= 0 and max_val <= 65535:
                return train_col.astype('uint16'), test_col.astype('uint16')
            elif min_val >= -32768 and max_val <= 32767:
                return train_col.astype('int16'), test_col.astype('int16')
            else:
                return train_col.astype('int32'), test_col.astype('int32')
                
        except Exception:
            return train_col.astype('float32'), test_col.astype('float32')
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic feature cleanup"""
        logger.info("Basic feature cleanup started")
        
        try:
            cols_to_remove = []
            batch_size = 10
            
            for i in range(0, len(X_train.columns), batch_size):
                batch_cols = X_train.columns[i:i + batch_size]
                
                for col in batch_cols:
                    try:
                        if X_train[col].nunique() <= 1:
                            cols_to_remove.append(col)
                    except Exception:
                        continue
                
                memory_status = self.memory_monitor.get_memory_status()
                if memory_status['should_simplify']:
                    break
            
            if cols_to_remove:
                X_train = X_train.drop(columns=cols_to_remove)
                X_test = X_test.drop(columns=[col for col in cols_to_remove if col in X_test.columns])
                self.removed_columns.extend(cols_to_remove)
                logger.info(f"Removed {len(cols_to_remove)} constant columns")
            
        except Exception as e:
            logger.warning(f"Basic feature cleanup failed: {e}")
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Interaction feature creation (expanded)"""
        logger.info("Interaction feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            max_interactions = min(self.config.MAX_INTERACTION_FEATURES, 250)
            
            if memory_status['should_simplify']:
                max_interactions = min(max_interactions, 80)
            
            # Identify important features for interaction
            important_features = []
            
            # Prioritize feat_ prefixed features
            feat_cols = [col for col in X_train.columns if str(col).startswith('feat_')][:20]
            important_features.extend(feat_cols)
            
            # Add high-variance numerical features
            for col in self.numerical_features:
                if col not in important_features:
                    try:
                        if X_train[col].std() > 0.1:
                            important_features.append(col)
                            if len(important_features) >= 35:
                                break
                    except Exception:
                        continue
            
            # Add categorical features with good cardinality
            for col in self.categorical_features:
                if col not in important_features:
                    try:
                        unique_count = X_train[col].nunique()
                        if 2 <= unique_count <= 40:
                            important_features.append(col)
                            if len(important_features) >= 40:
                                break
                    except Exception:
                        continue
            
            important_features = important_features[:40]
            logger.info(f"Selected {len(important_features)} features for interaction")
            
            interaction_count = 0
            
            # Second-order interactions
            for i, feat1 in enumerate(important_features):
                if interaction_count >= max_interactions * 0.75:
                    break
                
                for j, feat2 in enumerate(important_features[i+1:], i+1):
                    if interaction_count >= max_interactions * 0.75:
                        break
                    
                    try:
                        # Memory check
                        if interaction_count % 15 == 0:
                            memory_status = self.memory_monitor.get_memory_status()
                            if memory_status['should_simplify']:
                                logger.warning("Stopping interaction creation due to memory pressure")
                                break
                        
                        interaction_name = f"interact_{feat1}_{feat2}"
                        
                        # Create interaction based on feature types
                        if feat1 in self.numerical_features and feat2 in self.numerical_features:
                            # Numerical * Numerical
                            X_train[interaction_name] = (X_train[feat1] * X_train[feat2]).astype('float32')
                            X_test[interaction_name] = (X_test[feat1] * X_test[feat2]).astype('float32')
                        
                        elif feat1 in self.categorical_features or feat2 in self.categorical_features:
                            # Categorical interaction (combination)
                            X_train[interaction_name] = (X_train[feat1].astype(str) + "_" + X_train[feat2].astype(str)).astype('category').cat.codes.astype('float32')
                            X_test[interaction_name] = (X_test[feat1].astype(str) + "_" + X_test[feat2].astype(str)).astype('category').cat.codes.astype('float32')
                        
                        else:
                            # Mixed interaction
                            X_train[interaction_name] = (X_train[feat1] + X_train[feat2]).astype('float32')
                            X_test[interaction_name] = (X_test[feat1] + X_test[feat2]).astype('float32')
                        
                        self.interaction_features.append(interaction_name)
                        self.generated_features.append(interaction_name)
                        interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Interaction {feat1} x {feat2} failed: {e}")
                        continue
                
                if memory_status['should_simplify']:
                    break
            
            # Third-order interactions (limited)
            if not memory_status['should_simplify'] and interaction_count < max_interactions:
                remaining_slots = max_interactions - interaction_count
                third_order_limit = min(remaining_slots, 50)
                
                top_features = important_features[:12]
                third_order_count = 0
                
                for i, feat1 in enumerate(top_features):
                    if third_order_count >= third_order_limit:
                        break
                    for j, feat2 in enumerate(top_features[i+1:], i+1):
                        if third_order_count >= third_order_limit:
                            break
                        for k, feat3 in enumerate(top_features[j+1:], j+1):
                            if third_order_count >= third_order_limit:
                                break
                            
                            try:
                                interaction_name = f"interact3_{feat1}_{feat2}_{feat3}"
                                
                                if all(f in self.numerical_features for f in [feat1, feat2, feat3]):
                                    X_train[interaction_name] = (X_train[feat1] * X_train[feat2] * X_train[feat3]).astype('float32')
                                    X_test[interaction_name] = (X_test[feat1] * X_test[feat2] * X_test[feat3]).astype('float32')
                                    
                                    self.interaction_features.append(interaction_name)
                                    self.generated_features.append(interaction_name)
                                    interaction_count += 1
                                    third_order_count += 1
                                    
                            except Exception as e:
                                logger.warning(f"Third-order interaction failed: {e}")
                                continue
            
            logger.info(f"Interaction feature creation completed: {len(self.interaction_features)} features created")
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                       y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Target encoding feature creation (tuned)"""
        logger.info("Target encoding started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping target encoding due to memory constraints")
                return X_train, X_test
            
            # Select categorical features for target encoding
            categorical_for_encoding = []
            for col in self.categorical_features:
                try:
                    unique_count = X_train[col].nunique()
                    if 2 <= unique_count <= 200:  # Expanded range
                        categorical_for_encoding.append(col)
                        if len(categorical_for_encoding) >= 20:  # Increased limit
                            break
                except Exception:
                    continue
            
            logger.info(f"Target encoding {len(categorical_for_encoding)} categorical features")
            
            # Get tuned smoothing factor from config
            smoothing_factor = self.config.FEATURE_ENGINEERING_CONFIG.get('target_encoding_smoothing', 4.0)
            
            # CV-based target encoding for better generalization
            if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_cross_validation_encoding', True):
                cv_folds = 5
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                for col in categorical_for_encoding:
                    try:
                        target_col_name = f"target_enc_cv_{col}"
                        
                        # Initialize with global mean
                        global_mean = y_train.mean()
                        cv_predictions = np.full(len(X_train), global_mean, dtype='float32')
                        
                        # Cross-validation encoding
                        for train_idx, val_idx in skf.split(X_train, y_train):
                            X_fold_train = X_train.iloc[train_idx]
                            y_fold_train = y_train.iloc[train_idx]
                            X_fold_val = X_train.iloc[val_idx]
                            
                            # Calculate category means for this fold
                            category_stats = y_fold_train.groupby(X_fold_train[col]).agg(['mean', 'count']).reset_index()
                            category_stats.columns = [col, 'target_mean', 'count']
                            
                            # Apply tuned smoothing
                            category_stats['smoothed_mean'] = (
                                (category_stats['target_mean'] * category_stats['count'] + global_mean * smoothing_factor) /
                                (category_stats['count'] + smoothing_factor)
                            )
                            
                            # Create encoding mapping for this fold
                            fold_encoding_map = dict(zip(category_stats[col], category_stats['smoothed_mean']))
                            
                            # Apply to validation set
                            cv_predictions[val_idx] = X_fold_val[col].map(fold_encoding_map).fillna(global_mean).astype('float32')
                        
                        X_train[target_col_name] = cv_predictions
                        
                        # For test set, use full training data
                        category_stats_full = y_train.groupby(X_train[col]).agg(['mean', 'count']).reset_index()
                        category_stats_full.columns = [col, 'target_mean', 'count']
                        category_stats_full['smoothed_mean'] = (
                            (category_stats_full['target_mean'] * category_stats_full['count'] + global_mean * smoothing_factor) /
                            (category_stats_full['count'] + smoothing_factor)
                        )
                        
                        full_encoding_map = dict(zip(category_stats_full[col], category_stats_full['smoothed_mean']))
                        X_test[target_col_name] = X_test[col].map(full_encoding_map).fillna(global_mean).astype('float32')
                        
                        self.target_encoding_features.append(target_col_name)
                        self.generated_features.append(target_col_name)
                        self.target_encoders[col] = full_encoding_map
                        
                    except Exception as e:
                        logger.warning(f"CV target encoding for {col} failed: {e}")
                        continue
            else:
                # Simple target encoding with tuned smoothing
                for col in categorical_for_encoding:
                    try:
                        target_col_name = f"target_enc_{col}"
                        
                        # Calculate global mean
                        global_mean = y_train.mean()
                        
                        # Calculate category means
                        category_means = y_train.groupby(X_train[col]).agg(['mean', 'count']).reset_index()
                        category_means.columns = [col, 'target_mean', 'count']
                        
                        # Apply tuned smoothing
                        category_means['smoothed_mean'] = (
                            (category_means['target_mean'] * category_means['count'] + global_mean * smoothing_factor) /
                            (category_means['count'] + smoothing_factor)
                        )
                        
                        # Create encoding mapping
                        encoding_map = dict(zip(category_means[col], category_means['smoothed_mean']))
                        
                        # Apply encoding
                        X_train[target_col_name] = X_train[col].map(encoding_map).fillna(global_mean).astype('float32')
                        X_test[target_col_name] = X_test[col].map(encoding_map).fillna(global_mean).astype('float32')
                        
                        self.target_encoding_features.append(target_col_name)
                        self.generated_features.append(target_col_name)
                        self.target_encoders[col] = encoding_map
                        
                    except Exception as e:
                        logger.warning(f"Target encoding for {col} failed: {e}")
                        continue
            
            logger.info(f"Target encoding completed: {len(self.target_encoding_features)} features created")
            
        except Exception as e:
            logger.error(f"Target encoding failed: {e}")
        
        return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based feature creation"""
        logger.info("Temporal feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping temporal features due to memory constraints")
                return X_train, X_test
            
            # Look for time-related columns
            time_related_cols = []
            for col in X_train.columns:
                col_name_lower = str(col).lower()
                if any(keyword in col_name_lower for keyword in ['time', 'date', 'hour', 'day', 'week', 'month']):
                    time_related_cols.append(col)
            
            if not time_related_cols:
                # Create synthetic time features based on data index patterns
                logger.info("No explicit time columns found, creating index-based temporal features")
                
                n_samples_train = len(X_train)
                n_samples_test = len(X_test)
                
                # Multiple temporal cycles
                for cycle_length, cycle_name in [(24, 'hour'), (7, 'day'), (30, 'month'), (365, 'year')]:
                    # Training data temporal features
                    cycle_feature = np.arange(n_samples_train) % cycle_length
                    X_train[f'temporal_{cycle_name}'] = cycle_feature.astype('float32')
                    
                    # Cyclic encoding
                    X_train[f'{cycle_name}_sin'] = np.sin(2 * np.pi * cycle_feature / cycle_length).astype('float32')
                    X_train[f'{cycle_name}_cos'] = np.cos(2 * np.pi * cycle_feature / cycle_length).astype('float32')
                    
                    # Test data temporal features
                    test_cycle_feature = np.arange(n_samples_test) % cycle_length
                    X_test[f'temporal_{cycle_name}'] = test_cycle_feature.astype('float32')
                    X_test[f'{cycle_name}_sin'] = np.sin(2 * np.pi * test_cycle_feature / cycle_length).astype('float32')
                    X_test[f'{cycle_name}_cos'] = np.cos(2 * np.pi * test_cycle_feature / cycle_length).astype('float32')
                    
                    created_features = [f'temporal_{cycle_name}', f'{cycle_name}_sin', f'{cycle_name}_cos']
                    self.temporal_features.extend(created_features)
                    self.generated_features.extend(created_features)
                
                # Time trend feature
                time_trend_train = np.arange(n_samples_train) / n_samples_train
                time_trend_test = np.arange(n_samples_test) / n_samples_test
                
                X_train['time_trend'] = time_trend_train.astype('float32')
                X_test['time_trend'] = time_trend_test.astype('float32')
                
                # Time acceleration feature
                time_accel_train = (np.arange(n_samples_train) / n_samples_train) ** 2
                time_accel_test = (np.arange(n_samples_test) / n_samples_test) ** 2
                
                X_train['time_accel'] = time_accel_train.astype('float32')
                X_test['time_accel'] = time_accel_test.astype('float32')
                
                self.temporal_features.extend(['time_trend', 'time_accel'])
                self.generated_features.extend(['time_trend', 'time_accel'])
                
            else:
                # Process existing time-related columns
                for col in time_related_cols[:6]:  # Limit to 6 columns for memory
                    try:
                        # Extract different temporal components
                        if 'hour' in str(col).lower():
                            for func_name, func in [('sin', np.sin), ('cos', np.cos)]:
                                feature_name = f"{col}_hour_{func_name}"
                                X_train[feature_name] = func(2 * np.pi * X_train[col] / 24).astype('float32')
                                X_test[feature_name] = func(2 * np.pi * X_test[col] / 24).astype('float32')
                                
                                self.temporal_features.append(feature_name)
                                self.generated_features.append(feature_name)
                        
                        elif 'day' in str(col).lower():
                            for func_name, func in [('sin', np.sin), ('cos', np.cos)]:
                                feature_name = f"{col}_day_{func_name}"
                                X_train[feature_name] = func(2 * np.pi * X_train[col] / 7).astype('float32')
                                X_test[feature_name] = func(2 * np.pi * X_test[col] / 7).astype('float32')
                                
                                self.temporal_features.append(feature_name)
                                self.generated_features.append(feature_name)
                        
                        # Time-based binning with tuned parameters
                        try:
                            n_bins = self.config.FEATURE_ENGINEERING_CONFIG.get('n_bins', 20)
                            bins = np.percentile(X_train[col], np.linspace(0, 100, n_bins + 1))
                            bins = np.unique(bins)  # Remove duplicates
                            
                            if len(bins) > 1:
                                binned_name = f"{col}_binned"
                                X_train[binned_name] = pd.cut(X_train[col], bins=bins, labels=False, duplicates='drop').astype('float32')
                                X_test[binned_name] = pd.cut(X_test[col], bins=bins, labels=False, duplicates='drop').astype('float32')
                                
                                X_train[binned_name] = X_train[binned_name].fillna(0).astype('float32')
                                X_test[binned_name] = X_test[binned_name].fillna(0).astype('float32')
                                
                                self.temporal_features.append(binned_name)
                                self.generated_features.append(binned_name)
                        except Exception:
                            pass
                            
                    except Exception as e:
                        logger.warning(f"Temporal feature creation for {col} failed: {e}")
                        continue
            
            logger.info(f"Temporal feature creation completed: {len(self.temporal_features)} features created")
            
        except Exception as e:
            logger.error(f"Temporal feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Statistical feature creation (expanded)"""
        logger.info("Statistical feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping statistical features due to memory constraints")
                return X_train, X_test
            
            # Select numerical features for statistical operations
            numerical_for_stats = [col for col in self.numerical_features if col in X_train.columns][:20]
            
            if len(numerical_for_stats) >= 2:
                # Row-wise statistics
                numerical_data_train = X_train[numerical_for_stats].values
                numerical_data_test = X_test[numerical_for_stats].values
                
                # Basic statistics
                stat_features = {
                    'row_sum': np.sum,
                    'row_mean': np.mean,
                    'row_std': np.std,
                    'row_min': np.min,
                    'row_max': np.max,
                    'row_median': np.median
                }
                
                for stat_name, stat_func in stat_features.items():
                    try:
                        X_train[stat_name] = stat_func(numerical_data_train, axis=1).astype('float32')
                        X_test[stat_name] = stat_func(numerical_data_test, axis=1).astype('float32')
                        
                        self.statistical_features.append(stat_name)
                        self.generated_features.append(stat_name)
                    except Exception as e:
                        logger.warning(f"Statistical feature {stat_name} failed: {e}")
                
                # Advanced statistics
                try:
                    # Row range and IQR
                    X_train['row_range'] = (np.max(numerical_data_train, axis=1) - np.min(numerical_data_train, axis=1)).astype('float32')
                    X_test['row_range'] = (np.max(numerical_data_test, axis=1) - np.min(numerical_data_test, axis=1)).astype('float32')
                    
                    # Row percentiles
                    X_train['row_q25'] = np.percentile(numerical_data_train, 25, axis=1).astype('float32')
                    X_test['row_q25'] = np.percentile(numerical_data_test, 25, axis=1).astype('float32')
                    
                    X_train['row_q75'] = np.percentile(numerical_data_train, 75, axis=1).astype('float32')
                    X_test['row_q75'] = np.percentile(numerical_data_test, 75, axis=1).astype('float32')
                    
                    X_train['row_iqr'] = (X_train['row_q75'] - X_train['row_q25']).astype('float32')
                    X_test['row_iqr'] = (X_test['row_q75'] - X_test['row_q25']).astype('float32')
                    
                    # Row kurtosis (simplified)
                    row_mean_train = X_train['row_mean'].values.reshape(-1, 1)
                    row_mean_test = X_test['row_mean'].values.reshape(-1, 1)
                    row_std_train = X_train['row_std'].values.reshape(-1, 1)
                    row_std_test = X_test['row_std'].values.reshape(-1, 1)
                    
                    # Avoid division by zero
                    row_std_train = np.where(row_std_train == 0, 1e-8, row_std_train)
                    row_std_test = np.where(row_std_test == 0, 1e-8, row_std_test)
                    
                    kurtosis_train = np.mean(((numerical_data_train - row_mean_train) / row_std_train) ** 4, axis=1)
                    kurtosis_test = np.mean(((numerical_data_test - row_mean_test) / row_std_test) ** 4, axis=1)
                    
                    X_train['row_kurtosis'] = kurtosis_train.astype('float32')
                    X_test['row_kurtosis'] = kurtosis_test.astype('float32')
                    
                    # Count of zeros and non-zeros
                    X_train['row_zero_count'] = np.sum(numerical_data_train == 0, axis=1).astype('float32')
                    X_test['row_zero_count'] = np.sum(numerical_data_test == 0, axis=1).astype('float32')
                    
                    X_train['row_nonzero_count'] = np.sum(numerical_data_train != 0, axis=1).astype('float32')
                    X_test['row_nonzero_count'] = np.sum(numerical_data_test != 0, axis=1).astype('float32')
                    
                    # Row variance and coefficient of variation
                    X_train['row_var'] = np.var(numerical_data_train, axis=1).astype('float32')
                    X_test['row_var'] = np.var(numerical_data_test, axis=1).astype('float32')
                    
                    # Coefficient of variation (avoid division by zero)
                    mean_nonzero_train = np.where(X_train['row_mean'] == 0, 1e-8, X_train['row_mean'])
                    mean_nonzero_test = np.where(X_test['row_mean'] == 0, 1e-8, X_test['row_mean'])
                    
                    X_train['row_cv'] = (X_train['row_std'] / mean_nonzero_train).astype('float32')
                    X_test['row_cv'] = (X_test['row_std'] / mean_nonzero_test).astype('float32')
                    
                    advanced_features = [
                        'row_range', 'row_q25', 'row_q75', 'row_iqr', 'row_kurtosis', 
                        'row_zero_count', 'row_nonzero_count', 'row_var', 'row_cv'
                    ]
                    self.statistical_features.extend(advanced_features)
                    self.generated_features.extend(advanced_features)
                    
                except Exception as e:
                    logger.warning(f"Advanced statistical features failed: {e}")
                
                logger.info(f"Statistical feature creation completed: {len([f for f in self.generated_features if f.startswith('row_')])} features")
            else:
                logger.info("Insufficient numerical features for statistical operations")
            
        except Exception as e:
            logger.error(f"Statistical feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Frequency-based feature creation"""
        logger.info("Frequency feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping frequency features due to memory constraints")
                return X_train, X_test
            
            # Select categorical features for frequency encoding
            categorical_for_freq = [col for col in self.categorical_features if col in X_train.columns][:10]
            min_frequency = self.config.FEATURE_ENGINEERING_CONFIG.get('min_frequency', 2)
            
            for col in categorical_for_freq:
                try:
                    # Basic frequency encoding
                    freq_col_name = f"freq_{col}"
                    freq_map = X_train[col].value_counts().to_dict()
                    
                    X_train[freq_col_name] = X_train[col].map(freq_map).fillna(0).astype('float32')
                    X_test[freq_col_name] = X_test[col].map(freq_map).fillna(0).astype('float32')
                    
                    self.frequency_features.append(freq_col_name)
                    self.generated_features.append(freq_col_name)
                    
                    # Frequency rank encoding
                    freq_rank_col_name = f"freq_rank_{col}"
                    freq_ranks = X_train[col].value_counts().rank(ascending=False).to_dict()
                    
                    X_train[freq_rank_col_name] = X_train[col].map(freq_ranks).fillna(len(freq_ranks) + 1).astype('float32')
                    X_test[freq_rank_col_name] = X_test[col].map(freq_ranks).fillna(len(freq_ranks) + 1).astype('float32')
                    
                    self.frequency_features.append(freq_rank_col_name)
                    self.generated_features.append(freq_rank_col_name)
                    
                    # Relative frequency
                    total_count = len(X_train)
                    rel_freq_col_name = f"rel_freq_{col}"
                    rel_freq_map = {k: v / total_count for k, v in freq_map.items()}
                    
                    X_train[rel_freq_col_name] = X_train[col].map(rel_freq_map).fillna(0).astype('float32')
                    X_test[rel_freq_col_name] = X_test[col].map(rel_freq_map).fillna(0).astype('float32')
                    
                    self.frequency_features.append(rel_freq_col_name)
                    self.generated_features.append(rel_freq_col_name)
                    
                    # Rare category indicator
                    rare_col_name = f"rare_{col}"
                    X_train[rare_col_name] = (X_train[freq_col_name] <= min_frequency).astype('float32')
                    X_test[rare_col_name] = (X_test[freq_col_name] <= min_frequency).astype('float32')
                    
                    self.frequency_features.append(rare_col_name)
                    self.generated_features.append(rare_col_name)
                    
                except Exception as e:
                    logger.warning(f"Frequency encoding for {col} failed: {e}")
                    continue
            
            logger.info(f"Frequency feature creation completed: {len(self.frequency_features)} features created")
            
        except Exception as e:
            logger.error(f"Frequency feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_polynomial_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Polynomial feature creation"""
        logger.info("Polynomial feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping polynomial features due to memory constraints")
                return X_train, X_test
            
            # Select top numerical features for polynomial transformation
            numerical_for_poly = [col for col in self.numerical_features if col in X_train.columns][:12]
            
            for col in numerical_for_poly:
                try:
                    # Skip if feature has very low variance
                    if X_train[col].std() <= 1e-8:
                        continue
                    
                    # Quadratic features
                    quad_col_name = f"poly2_{col}"
                    X_train[quad_col_name] = (X_train[col] ** 2).astype('float32')
                    X_test[quad_col_name] = (X_test[col] ** 2).astype('float32')
                    
                    self.polynomial_features.append(quad_col_name)
                    self.generated_features.append(quad_col_name)
                    
                    # Cubic features (for selected features)
                    if col in numerical_for_poly[:6]:  # Only for top 6 features
                        cubic_col_name = f"poly3_{col}"
                        X_train[cubic_col_name] = (X_train[col] ** 3).astype('float32')
                        X_test[cubic_col_name] = (X_test[col] ** 3).astype('float32')
                        
                        self.polynomial_features.append(cubic_col_name)
                        self.generated_features.append(cubic_col_name)
                    
                    # Log transformation (for positive values)
                    if X_train[col].min() > 0:
                        log_col_name = f"log_{col}"
                        X_train[log_col_name] = np.log1p(X_train[col]).astype('float32')
                        X_test[log_col_name] = np.log1p(X_test[col]).astype('float32')
                        
                        self.polynomial_features.append(log_col_name)
                        self.generated_features.append(log_col_name)
                    
                    # Square root transformation (for non-negative values)
                    if X_train[col].min() >= 0:
                        sqrt_col_name = f"sqrt_{col}"
                        X_train[sqrt_col_name] = np.sqrt(X_train[col]).astype('float32')
                        X_test[sqrt_col_name] = np.sqrt(X_test[col]).astype('float32')
                        
                        self.polynomial_features.append(sqrt_col_name)
                        self.generated_features.append(sqrt_col_name)
                        
                except Exception as e:
                    logger.warning(f"Polynomial transformation for {col} failed: {e}")
                    continue
            
            logger.info(f"Polynomial feature creation completed: {len(self.polynomial_features)} features created")
            
        except Exception as e:
            logger.error(f"Polynomial feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_cross_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cross feature creation"""
        logger.info("Cross feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping cross features due to memory constraints")
                return X_train, X_test
            
            # Identify important features for cross-features
            important_numerical = [col for col in self.numerical_features if col in X_train.columns][:10]
            important_categorical = [col for col in self.categorical_features if col in X_train.columns][:10]
            
            cross_count = 0
            max_cross_features = 60
            
            # Numerical cross features (ratios and differences)
            for i, feat1 in enumerate(important_numerical):
                if cross_count >= max_cross_features * 0.6:
                    break
                    
                for feat2 in important_numerical[i+1:]:
                    if cross_count >= max_cross_features * 0.6:
                        break
                    
                    try:
                        # Ratio features
                        ratio_name = f"ratio_{feat1}_{feat2}"
                        denominator_train = X_train[feat2].replace(0, 1e-8)
                        denominator_test = X_test[feat2].replace(0, 1e-8)
                        
                        X_train[ratio_name] = (X_train[feat1] / denominator_train).astype('float32')
                        X_test[ratio_name] = (X_test[feat1] / denominator_test).astype('float32')
                        
                        # Difference features
                        diff_name = f"diff_{feat1}_{feat2}"
                        X_train[diff_name] = (X_train[feat1] - X_train[feat2]).astype('float32')
                        X_test[diff_name] = (X_test[feat1] - X_test[feat2]).astype('float32')
                        
                        self.cross_features.extend([ratio_name, diff_name])
                        self.generated_features.extend([ratio_name, diff_name])
                        cross_count += 2
                        
                    except Exception as e:
                        logger.warning(f"Cross feature {feat1} x {feat2} failed: {e}")
                        continue
            
            # Categorical group statistics
            for cat_col in important_categorical[:6]:
                if cross_count >= max_cross_features:
                    break
                    
                for num_col in important_numerical[:5]:
                    if cross_count >= max_cross_features:
                        break
                    
                    try:
                        # Group mean
                        group_mean_name = f"group_mean_{cat_col}_{num_col}"
                        group_means = X_train.groupby(cat_col)[num_col].mean().to_dict()
                        
                        X_train[group_mean_name] = X_train[cat_col].map(group_means).fillna(X_train[num_col].mean()).astype('float32')
                        X_test[group_mean_name] = X_test[cat_col].map(group_means).fillna(X_train[num_col].mean()).astype('float32')
                        
                        # Group std
                        group_std_name = f"group_std_{cat_col}_{num_col}"
                        group_stds = X_train.groupby(cat_col)[num_col].std().to_dict()
                        
                        X_train[group_std_name] = X_train[cat_col].map(group_stds).fillna(X_train[num_col].std()).astype('float32')
                        X_test[group_std_name] = X_test[cat_col].map(group_stds).fillna(X_train[num_col].std()).astype('float32')
                        
                        # Group median
                        group_median_name = f"group_median_{cat_col}_{num_col}"
                        group_medians = X_train.groupby(cat_col)[num_col].median().to_dict()
                        
                        X_train[group_median_name] = X_train[cat_col].map(group_medians).fillna(X_train[num_col].median()).astype('float32')
                        X_test[group_median_name] = X_test[cat_col].map(group_medians).fillna(X_train[num_col].median()).astype('float32')
                        
                        self.cross_features.extend([group_mean_name, group_std_name, group_median_name])
                        self.generated_features.extend([group_mean_name, group_std_name, group_median_name])
                        cross_count += 3
                        
                    except Exception as e:
                        logger.warning(f"Group statistics {cat_col} x {num_col} failed: {e}")
                        continue
            
            logger.info(f"Cross feature creation completed: {len(self.cross_features)} features created")
            
        except Exception as e:
            logger.error(f"Cross feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_binning_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Binning feature creation"""
        logger.info("Binning feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping binning features due to memory constraints")
                return X_train, X_test
            
            # Select numerical features for binning
            numerical_for_binning = [col for col in self.numerical_features if col in X_train.columns][:15]
            n_bins = self.config.FEATURE_ENGINEERING_CONFIG.get('n_bins', 20)
            binning_strategy = self.config.FEATURE_ENGINEERING_CONFIG.get('binning_strategy', 'quantile')
            
            for col in numerical_for_binning:
                try:
                    if X_train[col].nunique() <= 5:
                        continue
                    
                    if binning_strategy == 'quantile':
                        # Quantile-based binning
                        bins = np.percentile(X_train[col], np.linspace(0, 100, n_bins + 1))
                    else:
                        # Equal-width binning
                        min_val, max_val = X_train[col].min(), X_train[col].max()
                        bins = np.linspace(min_val, max_val, n_bins + 1)
                    
                    bins = np.unique(bins)
                    
                    if len(bins) > 1:
                        binned_name = f"binned_{col}"
                        X_train[binned_name] = pd.cut(X_train[col], bins=bins, labels=False, duplicates='drop').astype('float32')
                        X_test[binned_name] = pd.cut(X_test[col], bins=bins, labels=False, duplicates='drop').astype('float32')
                        
                        X_train[binned_name] = X_train[binned_name].fillna(-1).astype('float32')
                        X_test[binned_name] = X_test[binned_name].fillna(-1).astype('float32')
                        
                        self.binning_features.append(binned_name)
                        self.generated_features.append(binned_name)
                    
                except Exception as e:
                    logger.warning(f"Binning for {col} failed: {e}")
                    continue
            
            logger.info(f"Binning feature creation completed: {len(self.binning_features)} features created")
            
        except Exception as e:
            logger.error(f"Binning feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_rank_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Rank feature creation"""
        logger.info("Rank feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping rank features due to memory constraints")
                return X_train, X_test
            
            # Select numerical features for ranking
            numerical_for_rank = [col for col in self.numerical_features if col in X_train.columns][:10]
            
            for col in numerical_for_rank:
                try:
                    # Skip if feature has very low variance
                    if X_train[col].std() <= 1e-8:
                        continue
                    
                    # Rank transformation
                    rank_col_name = f"rank_{col}"
                    rank_values = X_train[col].rank(pct=True).astype('float32')
                    
                    # For test data, approximate rank based on training data percentiles
                    train_percentiles = np.percentile(X_train[col], np.arange(0, 101))
                    test_ranks = np.searchsorted(train_percentiles, X_test[col]) / 100.0
                    
                    X_train[rank_col_name] = rank_values
                    X_test[rank_col_name] = test_ranks.astype('float32')
                    
                    self.rank_features.append(rank_col_name)
                    self.generated_features.append(rank_col_name)
                    
                except Exception as e:
                    logger.warning(f"Rank transformation for {col} failed: {e}")
                    continue
            
            logger.info(f"Rank feature creation completed: {len(self.rank_features)} features created")
            
        except Exception as e:
            logger.error(f"Rank feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_ratio_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ratio feature creation"""
        logger.info("Ratio feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Skipping ratio features due to memory constraints")
                return X_train, X_test
            
            # Select numerical features for ratio calculation
            numerical_for_ratio = [col for col in self.numerical_features if col in X_train.columns][:8]
            
            ratio_count = 0
            max_ratio_features = 30
            
            for i, feat1 in enumerate(numerical_for_ratio):
                if ratio_count >= max_ratio_features:
                    break
                    
                for feat2 in numerical_for_ratio[i+1:]:
                    if ratio_count >= max_ratio_features:
                        break
                    
                    try:
                        # Create ratio with safe division
                        ratio_name = f"ratio_{feat1}_{feat2}"
                        
                        # Safe division (avoid division by zero)
                        denominator_train = X_train[feat2].replace(0, 1e-8)
                        denominator_test = X_test[feat2].replace(0, 1e-8)
                        
                        ratio_train = X_train[feat1] / denominator_train
                        ratio_test = X_test[feat1] / denominator_test
                        
                        # Clip extreme values
                        ratio_train = np.clip(ratio_train, -1e6, 1e6)
                        ratio_test = np.clip(ratio_test, -1e6, 1e6)
                        
                        X_train[ratio_name] = ratio_train.astype('float32')
                        X_test[ratio_name] = ratio_test.astype('float32')
                        
                        self.ratio_features.append(ratio_name)
                        self.generated_features.append(ratio_name)
                        ratio_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Ratio feature {feat1}/{feat2} failed: {e}")
                        continue
            
            logger.info(f"Ratio feature creation completed: {len(self.ratio_features)} features created")
            
        except Exception as e:
            logger.error(f"Ratio feature creation failed: {e}")
        
        return X_train, X_test
    
    def _encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Categorical feature encoding"""
        logger.info("Categorical feature encoding started")
        
        try:
            for col in self.categorical_features:
                if col not in X_train.columns:
                    continue
                
                try:
                    # Label encoding for categorical features
                    encoder = LabelEncoder()
                    
                    # Combine train and test for consistent encoding
                    combined_values = pd.concat([X_train[col], X_test[col]], ignore_index=True).astype(str)
                    encoder.fit(combined_values)
                    
                    # Apply encoding
                    X_train[col] = encoder.transform(X_train[col].astype(str)).astype('float32')
                    X_test[col] = encoder.transform(X_test[col].astype(str)).astype('float32')
                    
                    # Store encoder
                    self.label_encoders[col] = encoder
                    
                except Exception as e:
                    logger.warning(f"Categorical encoding for {col} failed: {e}")
                    # Fallback to simple numeric conversion
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                    except Exception:
                        X_train[col] = 0.0
                        X_test[col] = 0.0
            
            logger.info("Categorical feature encoding completed")
            
        except Exception as e:
            logger.error(f"Categorical feature encoding failed: {e}")
        
        return X_train, X_test
    
    def _create_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Numeric feature transformation"""
        logger.info("Numeric feature transformation started")
        
        try:
            # Select numerical features for transformation
            numerical_for_transform = [col for col in self.numerical_features if col in X_train.columns][:10]
            
            for col in numerical_for_transform:
                try:
                    # Skip if already transformed or problematic
                    if X_train[col].std() <= 1e-8:
                        continue
                    
                    # Skip if feature name suggests it's already a transformation
                    if any(prefix in col for prefix in ['log_', 'sqrt_', 'poly', 'interact', 'target_enc', 'rank_', 'ratio_', 'binned_']):
                        continue
                    
                    # Robust scaling for outlier handling
                    try:
                        robust_col_name = f"robust_{col}"
                        
                        # Calculate robust statistics
                        median_val = X_train[col].median()
                        mad_val = np.median(np.abs(X_train[col] - median_val))
                        mad_val = mad_val if mad_val > 0 else 1.0
                        
                        X_train[robust_col_name] = ((X_train[col] - median_val) / mad_val).astype('float32')
                        X_test[robust_col_name] = ((X_test[col] - median_val) / mad_val).astype('float32')
                        
                        self.generated_features.append(robust_col_name)
                        
                    except Exception:
                        pass
                    
                except Exception as e:
                    logger.warning(f"Numeric transformation for {col} failed: {e}")
                    continue
            
            logger.info(f"Numeric feature transformation completed")
            
        except Exception as e:
            logger.error(f"Numeric feature transformation failed: {e}")
        
        return X_train, X_test
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final data cleanup"""
        logger.info("Final data cleanup started")
        
        try:
            # Handle any remaining NaN values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Handle infinite values
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            # Ensure consistent data types
            for col in X_train.columns:
                if col in X_test.columns:
                    try:
                        X_train[col] = X_train[col].astype('float32')
                        X_test[col] = X_test[col].astype('float32')
                    except Exception:
                        X_train[col] = 0.0
                        X_test[col] = 0.0
            
            # Remove any columns that are still problematic
            problematic_cols = []
            for col in X_train.columns:
                try:
                    if X_train[col].isna().all() or X_train[col].nunique() <= 1:
                        problematic_cols.append(col)
                except Exception:
                    problematic_cols.append(col)
            
            if problematic_cols:
                X_train = X_train.drop(columns=problematic_cols)
                X_test = X_test.drop(columns=[col for col in problematic_cols if col in X_test.columns])
                self.removed_columns.extend(problematic_cols)
                logger.info(f"Removed {len(problematic_cols)} problematic columns")
            
            logger.info("Final data cleanup completed")
            
        except Exception as e:
            logger.error(f"Final data cleanup failed: {e}")
        
        return X_train, X_test
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                  target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features only (fallback)"""
        logger.warning("Creating basic features only")
        
        try:
            # Basic data preparation
            actual_target_col = self._detect_target_column(train_df, target_col)
            
            X_train = train_df.drop(columns=[actual_target_col]).copy()
            X_test = test_df.copy()
            
            # Simple numeric conversion
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                except Exception:
                    try:
                        # Hash encoding for non-numeric
                        X_train[col] = self._safe_hash_column(X_train[col])
                        X_test[col] = self._safe_hash_column(X_test[col])
                    except Exception:
                        X_train[col] = 0.0
                        X_test[col] = 0.0
            
            # Remove constant columns
            constant_cols = []
            for col in X_train.columns:
                try:
                    if X_train[col].nunique() <= 1:
                        constant_cols.append(col)
                except Exception:
                    pass
            
            if constant_cols:
                X_train = X_train.drop(columns=constant_cols)
                X_test = X_test.drop(columns=[col for col in constant_cols if col in X_test.columns])
            
            logger.info(f"Basic features created: {X_train.shape}")
            
        except Exception as e:
            logger.error(f"Basic feature creation failed: {e}")
            
            # Ultimate fallback
            try:
                X_train = train_df.drop(columns=[target_col]).iloc[:, :100].copy() if len(train_df.columns) >= 100 else train_df.drop(columns=[target_col]).copy()
                X_test = test_df.iloc[:, :100].copy() if len(test_df.columns) >= 100 else test_df.copy()
                
                X_train = X_train.fillna(0).astype('float32')
                X_test = X_test.fillna(0).astype('float32')
                
            except Exception as e2:
                logger.error(f"Fallback feature creation failed: {e2}")
                raise e2
        
        try:
            self.memory_monitor.force_memory_cleanup()
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
        
        return X_train, X_test
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Processing completion"""
        try:
            processing_time = time.time() - self.processing_stats['start_time']
            memory_usage = self.memory_monitor.get_memory_usage()
            
            self.final_feature_columns = list(X_train.columns)
            
            # Feature type statistics
            self.processing_stats['feature_types_count'] = {
                'interaction': len(self.interaction_features),
                'target_encoding': len(self.target_encoding_features),
                'temporal': len(self.temporal_features),
                'statistical': len(self.statistical_features),
                'frequency': len(self.frequency_features),
                'polynomial': len(self.polynomial_features),
                'cross': len(self.cross_features),
                'binning': len(self.binning_features),
                'rank': len(self.rank_features),
                'ratio': len(self.ratio_features)
            }
            
            self.processing_stats.update({
                'total_features_generated': len(self.generated_features),
                'processing_time': processing_time,
                'memory_usage': memory_usage
            })
            
            logger.info(f"Feature engineering statistics:")
            logger.info(f"  - Processing time: {processing_time:.2f}s")
            logger.info(f"  - Generated features: {len(self.generated_features)}")
            logger.info(f"  - Removed features: {len(self.removed_columns)}")
            logger.info(f"  - Final feature count: {X_train.shape[1]}")
            logger.info(f"  - Memory usage: {memory_usage:.2f}GB")
            logger.info(f"  - Interaction features: {len(self.interaction_features)}")
            logger.info(f"  - Target encoding: {len(self.target_encoding_features)}")
            logger.info(f"  - Time features: {len(self.temporal_features)}")
            logger.info(f"  - Statistical features: {len(self.statistical_features)}")
            logger.info(f"  - Frequency features: {len(self.frequency_features)}")
            logger.info(f"  - Polynomial features: {len(self.polynomial_features)}")
            logger.info(f"  - Cross features: {len(self.cross_features)}")
            logger.info(f"  - Binning features: {len(self.binning_features)}")
            logger.info(f"  - Rank features: {len(self.rank_features)}")
            logger.info(f"  - Ratio features: {len(self.ratio_features)}")
            
            self.memory_monitor.force_memory_cleanup()
            
        except Exception as e:
            logger.warning(f"Processing completion failed: {e}")