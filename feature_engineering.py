# feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
import gc
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import re
warnings.filterwarnings('ignore')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config import Config

def get_safe_logger(name: str):
    """Logger creation"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_safe_logger(__name__)

class MemoryMonitor:
    """Memory monitoring"""
    
    def __init__(self, max_memory_gb: float = 50.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 3.0
        self.max_memory_gb = max_memory_gb
        
        self.warning_threshold = max_memory_gb * 0.70
        self.critical_threshold = max_memory_gb * 0.80
        self.abort_threshold = max_memory_gb * 0.90
    
    def get_memory_usage(self) -> float:
        """Memory usage (GB)"""
        if not self.monitoring_enabled:
            return 2.0
        
        try:
            with self.lock:
                current_time = time.time()
                if current_time - self._last_check_time < self._check_interval:
                    return getattr(self, '_cached_memory', 2.0)
                
                self._last_check_time = current_time
                memory_gb = psutil.virtual_memory().used / (1024**3)
                self._cached_memory = memory_gb
                return memory_gb
        except Exception:
            return 2.0
    
    def get_available_memory(self) -> float:
        """Available memory (GB)"""
        if not self.monitoring_enabled:
            return 45.0
        
        try:
            return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 45.0
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Memory status"""
        try:
            if not self.monitoring_enabled:
                return {
                    'usage_gb': 2.0,
                    'available_gb': 45.0,
                    'level': 'normal',
                    'should_cleanup': False,
                    'should_simplify': False
                }
            
            vm = psutil.virtual_memory()
            usage_gb = vm.used / (1024**3)
            available_gb = vm.available / (1024**3)
            
            if usage_gb > self.abort_threshold:
                level = 'abort'
            elif usage_gb > self.critical_threshold:
                level = 'critical'
            elif usage_gb > self.warning_threshold:
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
        self.numeric_columns = []
        self.categorical_columns = []
        self.id_columns = []
        self.removed_columns = []
        self.interaction_features = []
        self.target_encoding_features = []
        self.temporal_features = []
        self.statistical_features = []
        self.frequency_features = []
        
        # Processing state
        self.target_column = None
        self.original_feature_order = []
        self.final_feature_columns = []
        self.processing_stats = {
            'start_time': 0,
            'feature_types_count': {},
            'total_features_generated': 0,
            'processing_time': 0,
            'memory_usage': 0
        }
        
        logger.info("CTR feature engineer initialization completed")
    
    def set_memory_efficient_mode(self, enabled: bool):
        """Set memory efficient mode"""
        self.memory_efficient_mode = enabled
        mode_text = "enabled" if enabled else "disabled"
        logger.info(f"Memory efficient mode {mode_text}")
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
        """Target column detection with CTR pattern recognition"""
        try:
            if provided_target_col and provided_target_col in train_df.columns:
                return provided_target_col
            
            # Check candidates in order of priority
            for candidate in self.config.TARGET_COLUMN_CANDIDATES:
                if candidate in train_df.columns:
                    unique_values = train_df[candidate].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        positive_ratio = train_df[candidate].mean()
                        if self.config.TARGET_DETECTION_CONFIG['min_ctr'] <= positive_ratio <= self.config.TARGET_DETECTION_CONFIG['max_ctr']:
                            logger.info(f"CTR target column detected: {candidate} (CTR: {positive_ratio:.4f})")
                            return candidate
            
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
            
            # For 64GB environment - enable all feature engineering
            memory_status = self.memory_monitor.get_memory_status()
            if not memory_status['should_simplify']:
                # 5. Interaction feature creation
                X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
                
                # 6. Target encoding
                X_train, X_test = self._create_target_encoding_features(X_train, X_test, y_train)
                
                # 7. Time-based features
                X_train, X_test = self._create_temporal_features(X_train, X_test)
                
                # 8. Statistical features
                X_train, X_test = self._create_statistical_features(X_train, X_test)
                
                # 9. Frequency-based features
                X_train, X_test = self._create_frequency_features(X_train, X_test)
            else:
                logger.warning("Simplified feature engineering due to memory constraints")
            
            # 10. Categorical feature encoding
            X_train, X_test = self._encode_categorical_features(X_train, X_test, y_train)
            
            # 11. Numeric feature transformation
            X_train, X_test = self._create_numeric_features(X_train, X_test)
            
            # 12. Final data cleanup
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
    
    def _classify_columns(self, df: pd.DataFrame):
        """Column type classification"""
        logger.info("Column type classification started")
        
        self.numeric_columns = []
        self.categorical_columns = []
        self.id_columns = []
        
        try:
            for col in df.columns:
                try:
                    dtype_str = str(df[col].dtype)
                    col_lower = str(col).lower()
                    
                    # ID column identification
                    if any(pattern in col_lower for pattern in ['id', 'uuid', 'key', 'hash']):
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio > 0.9:
                            self.id_columns.append(col)
                            continue
                    
                    # Numeric columns
                    if dtype_str in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                   'float16', 'float32', 'float64']:
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio > 0.95:
                            self.id_columns.append(col)
                        else:
                            self.numeric_columns.append(col)
                    else:
                        self.categorical_columns.append(col)
                        
                except Exception as e:
                    logger.warning(f"Column {col} classification failed: {e}")
                    self.numeric_columns.append(col)
            
            logger.info(f"Column classification completed - Numeric: {len(self.numeric_columns)}, "
                       f"Categorical: {len(self.categorical_columns)}, ID: {len(self.id_columns)}")
            
        except Exception as e:
            logger.error(f"Column classification failed: {e}")
            self.numeric_columns = list(df.columns)
            self.categorical_columns = []
            self.id_columns = []
    
    def _unify_data_types(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Data type unification"""
        logger.info("Data type unification started")
        
        try:
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            processed_count = 0
            
            batch_size = 20
            
            for i in range(0, len(common_columns), batch_size):
                batch_cols = common_columns[i:i + batch_size]
                
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
    
    def _safe_hash_column(self, series: pd.Series) -> pd.Series:
        """Safe column hashing"""
        try:
            return series.astype(str).apply(lambda x: hash(x) % 1000000).astype('int32')
        except Exception:
            return pd.Series([0] * len(series), dtype='int32', index=series.index)
    
    def _optimize_int_columns(self, train_col: pd.Series, test_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Integer column optimization"""
        try:
            combined_min = min(train_col.min(), test_col.min())
            combined_max = max(train_col.max(), test_col.max())
            
            if combined_min >= 0:
                if combined_max <= 255:
                    return train_col.astype('uint8'), test_col.astype('uint8')
                elif combined_max <= 65535:
                    return train_col.astype('uint16'), test_col.astype('uint16')
                else:
                    return train_col.astype('int32'), test_col.astype('int32')
            else:
                if combined_min >= -128 and combined_max <= 127:
                    return train_col.astype('int8'), test_col.astype('int8')
                elif combined_min >= -32768 and combined_max <= 32767:
                    return train_col.astype('int16'), test_col.astype('int16')
                else:
                    return train_col.astype('int32'), test_col.astype('int32')
        except Exception:
            return train_col.astype('float32'), test_col.astype('float32')
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic feature cleanup"""
        logger.info("Basic feature cleanup started")
        
        try:
            # Remove ID columns
            if self.id_columns:
                existing_id_cols = [col for col in self.id_columns if col in X_train.columns]
                if existing_id_cols:
                    X_train = X_train.drop(columns=existing_id_cols)
                    X_test = X_test.drop(columns=[col for col in existing_id_cols if col in X_test.columns])
                    self.removed_columns.extend(existing_id_cols)
                    logger.info(f"Removed {len(existing_id_cols)} ID columns")
            
            # Remove constant columns
            cols_to_remove = []
            
            batch_size = 30
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
        """Interaction feature creation"""
        logger.info("Interaction feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            max_interactions = self.config.MAX_INTERACTION_FEATURES if not memory_status['should_simplify'] else 20
            
            # Identify important features
            important_features = []
            
            # Prioritize feat_ prefixed features
            for prefix in ['feat_', 'l_feat_', 'gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']:
                prefix_features = [col for col in X_train.columns if str(col).startswith(str(prefix))][:8]
                important_features.extend(prefix_features)
            
            # Add categorical features
            categorical_features = []
            for col in X_train.columns:
                if col not in important_features and X_train[col].nunique() < 100:
                    categorical_features.append(col)
                    if len(categorical_features) >= 8:
                        break
            
            important_features.extend(categorical_features)
            important_features = list(set(important_features))[:20]
            
            interaction_count = 0
            for i, feat1 in enumerate(important_features):
                if interaction_count >= max_interactions:
                    break
                
                for j, feat2 in enumerate(important_features[i+1:], i+1):
                    if interaction_count >= max_interactions:
                        break
                    
                    try:
                        # Multiplication interaction
                        interaction_name = f"interact_{feat1}_{feat2}"
                        X_train[interaction_name] = X_train[feat1] * X_train[feat2]
                        X_test[interaction_name] = X_test[feat1] * X_test[feat2]
                        
                        self.interaction_features.append(interaction_name)
                        self.generated_features.append(interaction_name)
                        interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature {feat1}_{feat2} creation failed: {e}")
                        continue
                
                if interaction_count % 10 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"Interaction feature creation completed: {interaction_count} features")
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                       y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Target encoding feature creation"""
        logger.info("Target encoding feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Target encoding skipped due to memory constraints")
                return X_train, X_test
            
            # Select categorical columns for target encoding
            categorical_candidates = []
            for col in X_train.columns:
                try:
                    if X_train[col].nunique() > 2 and X_train[col].nunique() < 1000:
                        categorical_candidates.append(col)
                except Exception:
                    continue
            
            max_target_encoding = min(self.config.MAX_TARGET_ENCODING_FEATURES, len(categorical_candidates))
            selected_categorical = categorical_candidates[:max_target_encoding]
            
            for col in selected_categorical:
                try:
                    # Simple target encoding (mean of target by category)
                    target_mean = y_train.groupby(X_train[col]).mean()
                    global_mean = y_train.mean()
                    
                    # Apply to train
                    X_train[f"{col}_target_encoded"] = X_train[col].map(target_mean).fillna(global_mean).astype('float32')
                    
                    # Apply to test
                    X_test[f"{col}_target_encoded"] = X_test[col].map(target_mean).fillna(global_mean).astype('float32')
                    
                    self.target_encoding_features.append(f"{col}_target_encoded")
                    self.generated_features.append(f"{col}_target_encoded")
                    
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Target encoding completed: {len(self.target_encoding_features)} features")
            
        except Exception as e:
            logger.error(f"Target encoding feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based feature creation"""
        logger.info("Time-based feature creation started")
        
        try:
            # Find time-related columns
            time_columns = [col for col in X_train.columns if any(time_pattern in str(col).lower() 
                          for time_pattern in ['hour', 'day', 'time', 'date', 'week'])]
            
            for col in time_columns:
                try:
                    if 'hour' in str(col).lower():
                        # Hour-based features
                        X_train[f"{col}_sin"] = np.sin(2 * np.pi * X_train[col] / 24)
                        X_train[f"{col}_cos"] = np.cos(2 * np.pi * X_train[col] / 24)
                        X_test[f"{col}_sin"] = np.sin(2 * np.pi * X_test[col] / 24)
                        X_test[f"{col}_cos"] = np.cos(2 * np.pi * X_test[col] / 24)
                        
                        self.temporal_features.extend([f"{col}_sin", f"{col}_cos"])
                        self.generated_features.extend([f"{col}_sin", f"{col}_cos"])
                    
                    elif 'day' in str(col).lower():
                        # Day-based features
                        X_train[f"{col}_sin"] = np.sin(2 * np.pi * X_train[col] / 7)
                        X_train[f"{col}_cos"] = np.cos(2 * np.pi * X_train[col] / 7)
                        X_test[f"{col}_sin"] = np.sin(2 * np.pi * X_test[col] / 7)
                        X_test[f"{col}_cos"] = np.cos(2 * np.pi * X_test[col] / 7)
                        
                        self.temporal_features.extend([f"{col}_sin", f"{col}_cos"])
                        self.generated_features.extend([f"{col}_sin", f"{col}_cos"])
                        
                except Exception as e:
                    logger.warning(f"Temporal feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Temporal feature creation completed: {len(self.temporal_features)} features")
            
        except Exception as e:
            logger.error(f"Temporal feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Statistical feature creation"""
        logger.info("Statistical feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['should_simplify']:
                logger.warning("Statistical features skipped due to memory constraints")
                return X_train, X_test
            
            # Select numeric columns for statistical features
            numeric_cols = [col for col in X_train.columns 
                          if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                                   'float16', 'float32', 'float64']][:15]
            
            # Row-wise statistics
            if len(numeric_cols) >= 3:
                try:
                    X_train_numeric = X_train[numeric_cols]
                    X_test_numeric = X_test[numeric_cols]
                    
                    # Mean
                    X_train['row_mean'] = X_train_numeric.mean(axis=1).astype('float32')
                    X_test['row_mean'] = X_test_numeric.mean(axis=1).astype('float32')
                    
                    # Standard deviation
                    X_train['row_std'] = X_train_numeric.std(axis=1).fillna(0).astype('float32')
                    X_test['row_std'] = X_test_numeric.std(axis=1).fillna(0).astype('float32')
                    
                    # Max
                    X_train['row_max'] = X_train_numeric.max(axis=1).astype('float32')
                    X_test['row_max'] = X_test_numeric.max(axis=1).astype('float32')
                    
                    # Min
                    X_train['row_min'] = X_train_numeric.min(axis=1).astype('float32')
                    X_test['row_min'] = X_test_numeric.min(axis=1).astype('float32')
                    
                    statistical_features = ['row_mean', 'row_std', 'row_max', 'row_min']
                    self.statistical_features.extend(statistical_features)
                    self.generated_features.extend(statistical_features)
                    
                except Exception as e:
                    logger.warning(f"Statistical feature creation failed: {e}")
            
            logger.info(f"Statistical feature creation completed: {len(self.statistical_features)} features")
            
        except Exception as e:
            logger.error(f"Statistical feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Frequency-based feature creation"""
        logger.info("Frequency-based feature creation started")
        
        try:
            # Select categorical columns for frequency encoding
            categorical_cols = [col for col in X_train.columns if X_train[col].nunique() < 1000][:10]
            
            for col in categorical_cols:
                try:
                    # Value frequency in training data
                    freq_map = X_train[col].value_counts().to_dict()
                    
                    X_train[f"{col}_freq"] = X_train[col].map(freq_map).fillna(0).astype('int32')
                    X_test[f"{col}_freq"] = X_test[col].map(freq_map).fillna(0).astype('int32')
                    
                    self.frequency_features.append(f"{col}_freq")
                    self.generated_features.append(f"{col}_freq")
                    
                except Exception as e:
                    logger.warning(f"Frequency feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Frequency feature creation completed: {len(self.frequency_features)} features")
            
        except Exception as e:
            logger.error(f"Frequency feature creation failed: {e}")
        
        return X_train, X_test
    
    def _encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Categorical feature encoding"""
        logger.info("Categorical feature encoding started")
        
        try:
            current_categorical_cols = [col for col in X_train.columns 
                                      if X_train[col].dtype in ['object', 'category'] or 
                                      (X_train[col].dtype in ['int64', 'int32'] and X_train[col].nunique() < 100)]
            
            for col in current_categorical_cols:
                try:
                    # Label encoding
                    try:
                        le = LabelEncoder()
                        
                        # Combine train and test for consistent encoding
                        combined_values = pd.concat([X_train[col].astype(str), X_test[col].astype(str)]).unique()
                        le.fit(combined_values)
                        
                        train_encoded = []
                        for val in X_train[col].astype(str):
                            try:
                                train_encoded.append(le.transform([val])[0])
                            except:
                                train_encoded.append(-1)
                        
                        test_encoded = []
                        for val in X_test[col].astype(str):
                            try:
                                test_encoded.append(le.transform([val])[0])
                            except:
                                test_encoded.append(-1)
                        
                        X_train[f'{col}_encoded'] = train_encoded
                        X_test[f'{col}_encoded'] = np.array(test_encoded, dtype='int16')
                        
                        self.label_encoders[col] = le
                        self.generated_features.append(f'{col}_encoded')
                        
                    except Exception as e:
                        logger.warning(f"{col} Label Encoding failed: {e}")
                    
                except Exception as e:
                    logger.warning(f"Categorical feature {col} processing failed: {e}")
                
                if len(self.generated_features) % 5 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            # Remove original categorical columns
            try:
                existing_categorical = [col for col in current_categorical_cols if col in X_train.columns]
                if existing_categorical:
                    X_train = X_train.drop(columns=existing_categorical)
                    X_test = X_test.drop(columns=[col for col in existing_categorical if col in X_test.columns])
                    self.removed_columns.extend(existing_categorical)
            except Exception as e:
                logger.warning(f"Categorical column removal failed: {e}")
            
            logger.info(f"Categorical feature encoding completed")
            
        except Exception as e:
            logger.error(f"Categorical feature encoding failed: {e}")
        
        return X_train, X_test
    
    def _create_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Numeric feature transformation"""
        logger.info("Numeric feature transformation started")
        
        try:
            current_numeric_cols = [col for col in X_train.columns 
                                  if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                                           'float16', 'float32', 'float64']]
            
            if not current_numeric_cols:
                return X_train, X_test
            
            memory_status = self.memory_monitor.get_memory_status()
            feature_count = 0
            max_features = 25 if not memory_status['should_simplify'] else 10
            
            for col in current_numeric_cols[:20]:
                try:
                    if feature_count >= max_features:
                        break
                    
                    col_data = X_train[col]
                    
                    if col_data.std() > 0:
                        # Log transformation (for positive values)
                        if col_data.min() > 0:
                            X_train[f"{col}_log"] = np.log1p(col_data).astype('float32')
                            X_test[f"{col}_log"] = np.log1p(X_test[col]).astype('float32')
                            self.generated_features.append(f"{col}_log")
                            feature_count += 1
                        
                        # Square root transformation
                        if col_data.min() >= 0:
                            X_train[f"{col}_sqrt"] = np.sqrt(col_data + 1e-8).astype('float32')
                            X_test[f"{col}_sqrt"] = np.sqrt(X_test[col] + 1e-8).astype('float32')
                            self.generated_features.append(f"{col}_sqrt")
                            feature_count += 1
                        
                except Exception as e:
                    logger.warning(f"Numeric transformation failed for {col}: {e}")
                    continue
                
                if feature_count % 5 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"Numeric feature transformation completed: {feature_count} features")
            
        except Exception as e:
            logger.error(f"Numeric feature transformation failed: {e}")
        
        return X_train, X_test
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final data cleanup"""
        logger.info("Final data cleanup started")
        
        try:
            # Keep common columns only
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            
            # Limit column count for memory management
            memory_status = self.memory_monitor.get_memory_status()
            max_cols = self.config.MAX_FEATURES if not memory_status['should_simplify'] else 200
            
            if len(common_columns) > max_cols:
                logger.warning(f"Too many columns, limiting to {max_cols}")
                selected_columns = common_columns[:max_cols]
                X_train = X_train[selected_columns]
                X_test = X_test[selected_columns]
            
            # Force data type unification
            for col in X_train.columns:
                try:
                    train_dtype = X_train[col].dtype
                    test_dtype = X_test[col].dtype
                    
                    if train_dtype == 'object' or test_dtype == 'object':
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                    elif train_dtype != test_dtype or str(train_dtype) not in ['float32', 'int32', 'int16', 'int8', 'uint8', 'uint16']:
                        X_train[col] = X_train[col].astype('float32')
                        X_test[col] = X_test[col].astype('float32')
                        
                except Exception as e:
                    logger.warning(f"Column {col} type cleanup failed: {e}")
                    try:
                        X_train[col] = 0.0
                        X_test[col] = 0.0
                    except Exception:
                        pass
            
            # Handle missing values and infinite values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            X_train = X_train.replace([np.inf, -np.inf], [1e6, -1e6])
            X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # Final validation
            if list(X_train.columns) != list(X_test.columns):
                logger.warning("Column mismatch detected, performing realignment")
                common_columns = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_columns]
                X_test = X_test[common_columns]
            
            logger.info("Final data cleanup completed")
            
        except Exception as e:
            logger.error(f"Final data cleanup failed: {e}")
        
        return X_train, X_test
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features only"""
        logger.warning("Creating basic features only")
        
        try:
            # Use detected target column
            actual_target_col = self.target_column
            
            # Select numeric columns only
            numeric_cols = []
            for col in train_df.columns:
                if col != actual_target_col:
                    try:
                        if train_df[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                                 'float16', 'float32', 'float64']:
                            numeric_cols.append(col)
                    except Exception:
                        continue
            
            # Limit to first 150 columns
            selected_cols = numeric_cols[:150]
            
            X_train = train_df[selected_cols].copy()
            X_test = test_df[selected_cols].copy() if set(selected_cols).issubset(test_df.columns) else test_df.iloc[:, :150].copy()
            
            # Basic cleanup
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            X_train = X_train.replace([np.inf, -np.inf], [1e6, -1e6])
            X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # Type conversion
            for col in X_train.columns:
                try:
                    X_train[col] = X_train[col].astype('float32')
                    if col in X_test.columns:
                        X_test[col] = X_test[col].astype('float32')
                except Exception:
                    X_train[col] = 0.0
                    if col in X_test.columns:
                        X_test[col] = 0.0
            
            logger.info(f"Basic features created: {X_train.shape}")
            
        except Exception as e:
            logger.error(f"Basic feature creation failed: {e}")
            
            # Ultimate fallback
            try:
                feature_cols = [col for col in train_df.columns if col != target_col][:100]
                X_train = train_df[feature_cols].copy()
                X_test = test_df[feature_cols[:100]].copy() if len(test_df.columns) >= 100 else test_df.iloc[:, :100].copy()
                
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
                'frequency': len(self.frequency_features)
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
            
            self.memory_monitor.force_memory_cleanup()
            
        except Exception as e:
            logger.warning(f"Processing completion failed: {e}")