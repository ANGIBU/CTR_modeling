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
    
    def __init__(self, max_memory_gb: float = 60.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 3.0
        self.max_memory_gb = max_memory_gb
        
        self.warning_threshold = max_memory_gb * 0.80
        self.critical_threshold = max_memory_gb * 0.90
        self.abort_threshold = max_memory_gb * 0.95
    
    def get_memory_usage(self) -> float:
        """Memory usage (GB)"""
        if not self.monitoring_enabled:
            return 2.0
        
        try:
            with self.lock:
                current_time = time.time()
                if current_time - self._last_check_time < self._check_interval:
                    return getattr(self, '_cached_memory', 2.0)
                
                process = psutil.Process()
                memory_gb = process.memory_info().rss / (1024**3)
                self._cached_memory = memory_gb
                self._last_check_time = current_time
                return memory_gb
        except Exception:
            return 2.0
    
    def get_available_memory(self) -> float:
        """Available memory (GB)"""
        if not self.monitoring_enabled:
            return 45.0
        
        try:
            with self.lock:
                return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 45.0
    
    def check_memory_pressure(self) -> bool:
        """Check memory pressure"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            return usage > self.critical_threshold or available < 15.0
        except Exception:
            return False
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Memory status"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            if usage > self.abort_threshold or available < 8:
                level = "abort"
            elif usage > self.critical_threshold or available < 15:
                level = "critical"
            elif usage > self.warning_threshold or available < 25:
                level = "warning"
            else:
                level = "normal"
            
            return {
                'usage_gb': usage,
                'available_gb': available,
                'level': level,
                'should_cleanup': level in ['warning', 'critical', 'abort'],
                'should_simplify': level in ['abort']
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
        self.memory_efficient_mode = False
        
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
        self.final_feature_columns = []
        self.original_feature_order = []
        
        # Target column management
        self.target_column = None
        self.target_candidates = ['clicked', 'click', 'target', 'label', 'y', 'is_click', 'ctr']
        
        # Feature generation counters
        self.interaction_features = []
        self.target_encoding_features = []
        self.statistical_features = []
        self.temporal_features = []
        self.frequency_features = []
        self.ngram_features = []
        
        # Performance statistics
        self.processing_stats = {
            'start_time': time.time(),
            'total_features_generated': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'feature_types_count': {}
        }
        
        logger.info("CTR feature engineer initialization completed")
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
        """Auto-detect target column"""
        try:
            # Use provided target column if exists
            if provided_target_col and provided_target_col in train_df.columns:
                logger.info(f"Using provided target column: {provided_target_col}")
                return provided_target_col
            
            # Check target column candidates
            for candidate in self.target_candidates:
                if candidate in train_df.columns:
                    # Check if binary classification column
                    unique_values = train_df[candidate].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        logger.info(f"Target column auto-detected: {candidate}")
                        return candidate
            
            # Find columns with binary classification pattern
            for col in train_df.columns:
                if train_df[col].dtype in ['int64', 'int32', 'int8', 'uint8']:
                    unique_values = train_df[col].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        # Check CTR characteristics (very low click rate)
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
    
    def set_memory_efficient_mode(self, enabled: bool):
        """Set memory efficient mode"""
        self.memory_efficient_mode = enabled
        mode_text = "enabled" if enabled else "disabled"
        logger.info(f"Memory efficient mode {mode_text}")
    
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
            
            logger.info(f"Data type unification completed: {processed_count}/{len(common_columns)} columns")
            
        except Exception as e:
            logger.error(f"Data type unification failed: {e}")
        
        return X_train, X_test
    
    def _safe_hash_column(self, series: pd.Series) -> pd.Series:
        """Safe column hash transformation"""
        try:
            def safe_hash(x):
                try:
                    if pd.isna(x):
                        return 0
                    str_val = str(x)[:25]
                    return hash(str_val) % 30000
                except Exception:
                    return 0
            
            return series.apply(safe_hash).astype('int32')
            
        except Exception:
            return pd.Series([0] * len(series), dtype='int32')
    
    def _optimize_int_columns(self, train_col: pd.Series, test_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Integer column optimization"""
        try:
            train_min, train_max = train_col.min(), train_col.max()
            test_min, test_max = test_col.min(), test_col.max()
            
            overall_min = min(train_min, test_min)
            overall_max = max(train_max, test_max)
            
            if pd.isna(overall_min) or pd.isna(overall_max):
                return train_col.astype('float32'), test_col.astype('float32')
            
            if overall_min >= 0:
                if overall_max < 255:
                    return train_col.astype('uint8'), test_col.astype('uint8')
                elif overall_max < 65535:
                    return train_col.astype('uint16'), test_col.astype('uint16')
                else:
                    return train_col.astype('uint32'), test_col.astype('uint32')
            else:
                if overall_min > -128 and overall_max < 127:
                    return train_col.astype('int8'), test_col.astype('int8')
                elif overall_min > -32768 and overall_max < 32767:
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
            max_interactions = 20 if not memory_status['should_simplify'] else 12
            
            # Identify important features
            important_features = []
            
            # Prioritize feat_ prefixed features
            for prefix in ['feat_e', 'feat_d', 'feat_c', 'feat_b']:
                prefix_features = [col for col in X_train.columns if col.startswith(prefix)][:4]
                important_features.extend(prefix_features)
            
            # Categorical features
            categorical_features = []
            for col in X_train.columns:
                if col not in important_features and X_train[col].nunique() < 100:
                    categorical_features.append(col)
                    if len(categorical_features) >= 4:
                        break
            
            important_features.extend(categorical_features[:4])
            important_features = important_features[:12]
            
            interaction_count = 0
            for i, feat1 in enumerate(important_features):
                if interaction_count >= max_interactions:
                    break
                    
                for j, feat2 in enumerate(important_features[i+1:], i+1):
                    if interaction_count >= max_interactions:
                        break
                    
                    try:
                        interaction_name = f'{feat1}_x_{feat2}'
                        
                        # Multiplication interaction
                        X_train[interaction_name] = (X_train[feat1] * X_train[feat2]).astype('float32')
                        X_test[interaction_name] = (X_test[feat1] * X_test[feat2]).astype('float32')
                        
                        self.interaction_features.append(interaction_name)
                        self.generated_features.append(interaction_name)
                        interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature {feat1} x {feat2} creation failed: {e}")
                    
                    if interaction_count % 5 == 0:
                        self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"Created {len(self.interaction_features)} interaction features")
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                       y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Target encoding feature creation"""
        logger.info("Target encoding feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            max_target_features = 15 if not memory_status['should_simplify'] else 8
            
            # Select categorical features
            categorical_candidates = []
            for col in X_train.columns:
                unique_count = X_train[col].nunique()
                if 2 <= unique_count <= 500:
                    categorical_candidates.append(col)
            
            categorical_candidates = categorical_candidates[:max_target_features]
            
            # Target mean value
            target_mean = y_train.mean()
            
            for col in categorical_candidates:
                try:
                    # Calculate group statistics
                    group_stats = X_train.groupby(col).agg({
                        col: 'count'
                    }).rename(columns={col: 'count'})
                    
                    # Group by col and calculate mean with y_train index
                    target_by_group = X_train.groupby(col).apply(
                        lambda x: y_train.iloc[x.index].mean()
                    )
                    
                    group_stats['mean'] = target_by_group
                    
                    # Bayesian smoothing
                    alpha = 100
                    group_stats['target_encoded'] = (
                        group_stats['mean'] * group_stats['count'] + target_mean * alpha
                    ) / (group_stats['count'] + alpha)
                    
                    # Apply mapping
                    target_map = group_stats['target_encoded'].to_dict()
                    
                    feature_name = f'{col}_target_encoded'
                    X_train[feature_name] = X_train[col].map(target_map).fillna(target_mean).astype('float32')
                    X_test[feature_name] = X_test[col].map(target_map).fillna(target_mean).astype('float32')
                    
                    self.target_encoding_features.append(feature_name)
                    self.generated_features.append(feature_name)
                    self.target_encoders[col] = target_map
                    
                    logger.info(f"Target encoding created: {feature_name}")
                    
                except Exception as e:
                    logger.warning(f"Target encoding {col} failed: {e}")
                
                if len(self.target_encoding_features) % 5 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"Created {len(self.target_encoding_features)} target encoding features")
            
        except Exception as e:
            logger.error(f"Target encoding feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based feature creation"""
        logger.info("Time-based feature creation started")
        
        try:
            # Basic time index feature
            X_train['time_index'] = (X_train.index / len(X_train)).astype('float32')
            X_test['time_index'] = (X_test.index / len(X_test)).astype('float32')
            
            # Quantile-based time segments
            X_train['time_quartile'] = pd.qcut(X_train.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            X_test['time_quartile'] = pd.qcut(X_test.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            
            # Decile-based time segments
            X_train['time_decile'] = pd.qcut(X_train.index, q=10, labels=list(range(10))).astype('int8')
            X_test['time_decile'] = pd.qcut(X_test.index, q=10, labels=list(range(10))).astype('int8')
            
            # Time-based relative position
            X_train['time_sin'] = np.sin(2 * np.pi * X_train['time_index']).astype('float32')
            X_test['time_sin'] = np.sin(2 * np.pi * X_test['time_index']).astype('float32')
            
            X_train['time_cos'] = np.cos(2 * np.pi * X_train['time_index']).astype('float32')
            X_test['time_cos'] = np.cos(2 * np.pi * X_test['time_index']).astype('float32')
            
            # Additional time features
            memory_status = self.memory_monitor.get_memory_status()
            if not memory_status['should_simplify']:
                # Relative position within time segments
                X_train['time_relative_pos'] = ((X_train.index % 1000) / 1000).astype('float32')
                X_test['time_relative_pos'] = ((X_test.index % 1000) / 1000).astype('float32')
                
                # Time-based cyclic feature
                X_train['time_cycle'] = (X_train.index % 5000 / 5000).astype('float32')
                X_test['time_cycle'] = (X_test.index % 5000 / 5000).astype('float32')
                
                temporal_features = ['time_index', 'time_quartile', 'time_decile', 'time_sin', 'time_cos', 
                                   'time_relative_pos', 'time_cycle']
            else:
                temporal_features = ['time_index', 'time_quartile', 'time_decile', 'time_sin', 'time_cos']
            
            self.temporal_features.extend(temporal_features)
            self.generated_features.extend(temporal_features)
            
            logger.info(f"Created {len(temporal_features)} time-based features")
            
        except Exception as e:
            logger.error(f"Time-based feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Statistical feature creation"""
        logger.info("Statistical feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            # Statistics by feature group
            feature_groups = {
                'feat_e': [col for col in X_train.columns if col.startswith('feat_e')][:6],
                'feat_d': [col for col in X_train.columns if col.startswith('feat_d')][:6],
                'feat_c': [col for col in X_train.columns if col.startswith('feat_c')][:6],
                'feat_b': [col for col in X_train.columns if col.startswith('feat_b')][:6]
            }
            
            for group_name, group_cols in feature_groups.items():
                if len(group_cols) >= 2:
                    try:
                        # Group sum
                        sum_feature = f'{group_name}_sum'
                        X_train[sum_feature] = X_train[group_cols].sum(axis=1).astype('float32')
                        X_test[sum_feature] = X_test[group_cols].sum(axis=1).astype('float32')
                        
                        # Group mean
                        mean_feature = f'{group_name}_mean'
                        X_train[mean_feature] = X_train[group_cols].mean(axis=1).astype('float32')
                        X_test[mean_feature] = X_test[group_cols].mean(axis=1).astype('float32')
                        
                        if not memory_status['should_simplify']:
                            # Group standard deviation
                            std_feature = f'{group_name}_std'
                            X_train[std_feature] = X_train[group_cols].std(axis=1).fillna(0).astype('float32')
                            X_test[std_feature] = X_test[group_cols].std(axis=1).fillna(0).astype('float32')
                            
                            # Group maximum
                            max_feature = f'{group_name}_max'
                            X_train[max_feature] = X_train[group_cols].max(axis=1).astype('float32')
                            X_test[max_feature] = X_test[group_cols].max(axis=1).astype('float32')
                            
                            # Group minimum
                            min_feature = f'{group_name}_min'
                            X_train[min_feature] = X_train[group_cols].min(axis=1).astype('float32')
                            X_test[min_feature] = X_test[group_cols].min(axis=1).astype('float32')
                            
                            group_features = [sum_feature, mean_feature, std_feature, max_feature, min_feature]
                        else:
                            group_features = [sum_feature, mean_feature]
                        
                        self.statistical_features.extend(group_features)
                        self.generated_features.extend(group_features)
                        
                    except Exception as e:
                        logger.warning(f"{group_name} group statistics creation failed: {e}")
                
                if len(self.statistical_features) % 5 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"Created {len(self.statistical_features)} statistical features")
            
        except Exception as e:
            logger.error(f"Statistical feature creation failed: {e}")
        
        return X_train, X_test
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Frequency-based feature creation"""
        logger.info("Frequency-based feature creation started")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            max_freq_features = 12 if not memory_status['should_simplify'] else 6
            
            # Select features for frequency encoding
            freq_candidates = []
            for col in X_train.columns:
                unique_count = X_train[col].nunique()
                if 5 <= unique_count <= 1000:
                    freq_candidates.append(col)
            
            freq_candidates = freq_candidates[:max_freq_features]
            
            for col in freq_candidates:
                try:
                    # Calculate frequency
                    value_counts = X_train[col].value_counts()
                    freq_map = value_counts.to_dict()
                    
                    # Frequency feature
                    freq_feature = f'{col}_freq'
                    X_train[freq_feature] = X_train[col].map(freq_map).fillna(0).astype('int32')
                    X_test[freq_feature] = X_test[col].map(freq_map).fillna(0).astype('int32')
                    
                    # Rarity feature
                    rarity_feature = f'{col}_rarity'
                    max_freq = max(freq_map.values())
                    rarity_map = {k: max_freq / v for k, v in freq_map.items()}
                    
                    X_train[rarity_feature] = X_train[col].map(rarity_map).fillna(max_freq).astype('float32')
                    X_test[rarity_feature] = X_test[col].map(rarity_map).fillna(max_freq).astype('float32')
                    
                    freq_features = [freq_feature, rarity_feature]
                    self.frequency_features.extend(freq_features)
                    self.generated_features.extend(freq_features)
                    
                except Exception as e:
                    logger.warning(f"Frequency feature {col} creation failed: {e}")
                
                if len(self.frequency_features) % 6 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"Created {len(self.frequency_features)} frequency-based features")
            
        except Exception as e:
            logger.error(f"Frequency-based feature creation failed: {e}")
        
        return X_train, X_test
    
    def _encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Categorical feature encoding"""
        current_categorical_cols = []
        for col in X_train.columns:
            try:
                dtype_str = str(X_train[col].dtype)
                if dtype_str in ['object', 'category', 'string']:
                    current_categorical_cols.append(col)
            except Exception:
                continue
        
        if not current_categorical_cols:
            return X_train, X_test
        
        logger.info(f"Categorical feature encoding started: {len(current_categorical_cols)} columns")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            max_categorical = 15 if not memory_status['should_simplify'] else 8
            
            for col in current_categorical_cols[:max_categorical]:
                try:
                    train_values = X_train[col].astype(str).fillna('missing')
                    test_values = X_test[col].astype(str).fillna('missing')
                    
                    # High cardinality handling
                    unique_count = len(train_values.unique())
                    max_categories = 50 if not memory_status['should_simplify'] else 30
                    
                    if unique_count > max_categories:
                        top_categories = train_values.value_counts().head(max_categories).index
                        train_values = train_values.where(train_values.isin(top_categories), 'other')
                        test_values = test_values.where(test_values.isin(top_categories), 'other')
                    
                    # Label Encoding
                    try:
                        le = LabelEncoder()
                        le.fit(train_values)
                        
                        train_encoded = le.transform(train_values).astype('int16')
                        
                        test_encoded = []
                        for val in test_values:
                            if val in le.classes_:
                                test_encoded.append(le.transform([val])[0])
                            else:
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
            max_features = 18 if not memory_status['should_simplify'] else 8
            
            for col in current_numeric_cols[:15]:
                try:
                    if feature_count >= max_features:
                        break
                    
                    # Log transformation
                    train_positive = (X_train[col] > 0) & X_train[col].notna()
                    test_positive = (X_test[col] > 0) & X_test[col].notna()
                    
                    if train_positive.sum() > len(X_train) * 0.7:
                        try:
                            X_train[f'{col}_log'] = np.where(train_positive, np.log1p(X_train[col]), 0).astype('float32')
                            X_test[f'{col}_log'] = np.where(test_positive, np.log1p(X_test[col]), 0).astype('float32')
                            
                            self.generated_features.append(f'{col}_log')
                            feature_count += 1
                        except Exception as e:
                            logger.warning(f"{col} log transformation failed: {e}")
                    
                    # Square root transformation
                    if feature_count < max_features:
                        train_non_negative = (X_train[col] >= 0) & X_train[col].notna()
                        test_non_negative = (X_test[col] >= 0) & X_test[col].notna()
                        
                        if train_non_negative.sum() > len(X_train) * 0.8:
                            try:
                                X_train[f'{col}_sqrt'] = np.where(train_non_negative, np.sqrt(X_train[col]), 0).astype('float32')
                                X_test[f'{col}_sqrt'] = np.where(test_non_negative, np.sqrt(X_test[col]), 0).astype('float32')
                                
                                self.generated_features.append(f'{col}_sqrt')
                                feature_count += 1
                            except Exception as e:
                                logger.warning(f"{col} square root transformation failed: {e}")
                    
                    if feature_count >= max_features:
                        break
                        
                except Exception as e:
                    logger.warning(f"Numeric feature {col} processing failed: {e}")
            
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
            
            # Limit column count
            memory_status = self.memory_monitor.get_memory_status()
            max_cols = 250 if not memory_status['should_simplify'] else 150
            
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
                if col != actual_target_col and train_df[col].dtype in ['int8', 'int16', 'int32', 'int64', 
                                                                        'uint8', 'uint16', 'uint32', 'uint64',
                                                                        'float16', 'float32', 'float64']:
                    numeric_cols.append(col)
                    if len(numeric_cols) >= 100:
                        break
            
            if not numeric_cols:
                numeric_cols = [col for col in train_df.columns if col != actual_target_col][:100]
            
            X_train = train_df[numeric_cols].copy()
            X_test = test_df[numeric_cols].copy() if set(numeric_cols).issubset(test_df.columns) else test_df.iloc[:, :len(numeric_cols)].copy()
            
            # Convert to float32
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                    if col in X_test.columns:
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                except Exception:
                    X_train[col] = 0.0
                    if col in X_test.columns:
                        X_test[col] = 0.0
            
            logger.info(f"Basic features only creation completed: {X_train.shape}")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic feature creation failed: {e}")
            n_train = len(train_df)
            n_test = len(test_df)
            
            X_train = pd.DataFrame({
                'dummy_feature_1': np.random.normal(0, 1, n_train).astype('float32'),
                'dummy_feature_2': np.random.uniform(0, 1, n_train).astype('float32'),
                'dummy_feature_3': np.ones(n_train, dtype='float32')
            })
            
            X_test = pd.DataFrame({
                'dummy_feature_1': np.random.normal(0, 1, n_test).astype('float32'),
                'dummy_feature_2': np.random.uniform(0, 1, n_test).astype('float32'),
                'dummy_feature_3': np.ones(n_test, dtype='float32')
            })
            
            return X_train, X_test
    
    def get_feature_columns_for_inference(self) -> List[str]:
        """Return feature columns for inference"""
        return self.final_feature_columns.copy() if self.final_feature_columns else []
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Feature importance summary information"""
        return {
            'total_generated_features': len(self.generated_features),
            'generated_features': self.generated_features,
            'removed_columns': self.removed_columns,
            'final_feature_columns': self.final_feature_columns,
            'original_feature_order': self.original_feature_order,
            'interaction_features': self.interaction_features,
            'target_encoding_features': self.target_encoding_features,
            'temporal_features': self.temporal_features,
            'statistical_features': self.statistical_features,
            'frequency_features': self.frequency_features,
            'processing_stats': self.processing_stats,
            'encoders_count': {
                'target_encoders': len(self.target_encoders),
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            },
            'target_column': self.target_column
        }

FeatureEngineer = CTRFeatureEngineer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test code
    np.random.seed(42)
    n_samples = 50000
    
    train_data = {
        'clicked': np.random.binomial(1, 0.0191, n_samples),
        'feat_e_1': np.random.normal(0, 100, n_samples),
        'feat_e_2': np.random.normal(50, 25, n_samples),
        'feat_d_1': np.random.poisson(1, n_samples),
        'feat_d_2': np.random.poisson(2, n_samples),
        'feat_c_1': np.random.uniform(0, 10, n_samples),
        'feat_c_2': np.random.uniform(5, 15, n_samples),
        'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'user_id': [f'user_{i % 10000}' for i in range(n_samples)]
    }
    
    test_data = {col: val for col, val in train_data.items() if col != 'clicked'}
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Feature engineering test
    try:
        engineer = CTRFeatureEngineer()
        X_train, X_test = engineer.create_all_features(train_df, test_df)
        
        print(f"Original data: {train_df.shape}, {test_df.shape}")
        print(f"After processing: {X_train.shape}, {X_test.shape}")
        print(f"Generated features: {len(engineer.generated_features)}")
        
        summary = engineer.get_feature_importance_summary()
        print(f"Feature generation count by type: {summary['processing_stats']['feature_types_count']}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")