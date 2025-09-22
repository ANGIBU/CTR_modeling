# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import gc
import warnings
from pathlib import Path
import pickle
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
    from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available.")

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Memory monitoring for feature engineering operations
    Tracks system resources and provides memory management guidance
    """
    
    def __init__(self):
        # Memory thresholds for different operations (in GB)
        self.memory_thresholds = {
            'warning': 15.0,    # Issue warning below 15GB
            'critical': 10.0,   # Critical state below 10GB
            'abort': 5.0        # Abort operations below 5GB
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().used / (1024**3)
        return 0.0
    
    def get_available_memory(self) -> float:
        """Get available memory in GB"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available / (1024**3)
        return 64.0  # Assume 64GB if can't detect
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Check current memory status and provide recommendations
        
        Returns:
            Dict containing memory status and action recommendations
        """
        if not PSUTIL_AVAILABLE:
            return {
                'usage_gb': 0.0,
                'available_gb': 64.0,
                'level': 'unknown',
                'should_simplify': False
            }
        
        vm = psutil.virtual_memory()
        usage_gb = vm.used / (1024**3)
        available_gb = vm.available / (1024**3)
        
        level = 'normal'
        should_simplify = False
        
        if available_gb < self.memory_thresholds['abort']:
            level = 'abort'
            should_simplify = True
        elif available_gb < self.memory_thresholds['critical']:
            level = 'critical'
            should_simplify = True
        elif available_gb < self.memory_thresholds['warning']:
            level = 'warning'
            should_simplify = True
        
        return {
            'usage_gb': usage_gb,
            'available_gb': available_gb,
            'level': level,
            'should_simplify': should_simplify
        }
    
    def force_memory_cleanup(self):
        """Force garbage collection and memory cleanup"""
        try:
            # Multiple rounds of garbage collection
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            
            # Windows-specific memory optimization
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass
        except Exception:
            pass
    
    def log_memory_status(self, context: str = ""):
        """Log current memory status with context"""
        try:
            status = self.get_memory_status()
            
            if status['level'] != 'normal' or context:
                logger.info(f"Memory status [{context}]: Usage {status['usage_gb']:.1f}GB, "
                           f"Available {status['available_gb']:.1f}GB - {status['level'].upper()}")
                
        except Exception as e:
            logger.warning(f"Memory status logging failed: {e}")

class CTRFeatureEngineer:
    """
    CTR Feature Engineering Class with Quick Mode Support
    Handles feature creation for Click-Through Rate prediction with memory management
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.memory_efficient_mode = False  # Default disabled for 64GB environment
        self.quick_mode = False  # Quick mode for testing with minimal features
        
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
            # Quick mode automatically enables memory efficient mode
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
            
            # 4. Fill missing values with simple strategy
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
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
            # Create absolute minimal features
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
            if not memory_status['should_simplify'] and memory_status['available_gb'] > 8:
                logger.info("Full feature engineering enabled - sufficient memory available")
                
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
            
            # Search for binary pattern in all columns
            for col in train_df.columns:
                if train_df[col].dtype in ['int64', 'int32', 'int8', 'uint8']:
                    unique_values = train_df[col].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        positive_ratio = train_df[col].mean()
                        if 0.001 <= positive_ratio <= 0.1:  # 0.1% ~ 10% range
                            logger.info(f"CTR pattern target column detected: {col} (CTR: {positive_ratio:.4f})")
                            return col
            
            # Use provided column as default
            if provided_target_col:
                logger.warning(f"Target column '{provided_target_col}' validation failed, using anyway")
                return provided_target_col
            else:
                raise ValueError("No suitable target column found")
                
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return provided_target_col or 'clicked'
    
    def _prepare_basic_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepare basic training and test data"""
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
    
    def _classify_columns_basic(self, X_train: pd.DataFrame):
        """Basic column classification for quick mode"""
        logger.info("Basic column classification started")
        
        try:
            self.numerical_features = []
            self.categorical_features = []
            
            for col in X_train.columns:
                try:
                    # Simple classification based on data type
                    if X_train[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
                        self.numerical_features.append(col)
                    else:
                        self.categorical_features.append(col)
                
                except Exception as e:
                    logger.warning(f"Column {col} classification failed: {e}")
                    self.numerical_features.append(col)  # Default to numeric
            
            logger.info(f"Basic classification - Numeric: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
            
        except Exception as e:
            logger.error(f"Basic column classification failed: {e}")
            # Fallback: treat all as numeric
            self.numerical_features = list(X_train.columns)
            self.categorical_features = []
    
    def _classify_columns(self, X_train: pd.DataFrame):
        """Comprehensive column classification for full mode"""
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
    
    def _fix_basic_data_types_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe data type fixes with proper categorical handling"""
        try:
            for col in X_train.columns:
                if col in X_test.columns:
                    # Convert both to numeric if possible
                    try:
                        # Check if column is already categorical
                        if X_train[col].dtype.name == 'category':
                            # Convert categorical to numeric safely
                            X_train[col] = pd.Categorical(X_train[col]).codes
                            X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].unique()).codes
                            X_train[col] = X_train[col].fillna(-1).astype('int32')
                            X_test[col] = X_test[col].fillna(-1).astype('int32')
                        else:
                            # Try numeric conversion
                            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                    except Exception:
                        # Convert to string and then to codes
                        try:
                            combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
                            unique_values = sorted(combined_values.unique())
                            value_map = {val: idx for idx, val in enumerate(unique_values)}
                            
                            X_train[col] = X_train[col].astype(str).map(value_map).fillna(0).astype('int32')
                            X_test[col] = X_test[col].astype(str).map(value_map).fillna(0).astype('int32')
                        except Exception:
                            # Fill with zeros as last resort
                            X_train[col] = 0
                            X_test[col] = 0
            
            logger.info("Safe data type fixes completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Safe data type fixing failed: {e}")
            return X_train.fillna(0), X_test.fillna(0)
    
    def _encode_categorical_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical encoding"""
        try:
            for col in self.categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        # Convert to string first
                        train_str = X_train[col].astype(str).fillna('missing')
                        test_str = X_test[col].astype(str).fillna('missing')
                        
                        # Get all unique values from both sets
                        combined_values = pd.concat([train_str, test_str])
                        unique_values = sorted(combined_values.unique())
                        
                        # Create mapping
                        value_map = {val: idx for idx, val in enumerate(unique_values)}
                        
                        # Apply mapping
                        X_train[col] = train_str.map(value_map).fillna(0).astype('int32')
                        X_test[col] = test_str.map(value_map).fillna(0).astype('int32')
                        
                    except Exception as e:
                        logger.warning(f"Safe categorical encoding failed for {col}: {e}")
                        X_train[col] = 0
                        X_test[col] = 0
            
            logger.info("Safe categorical encoding completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Safe categorical encoding failed: {e}")
            return X_train, X_test
    
    def _normalize_numeric_basic(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic numeric normalization for quick mode"""
        try:
            for col in self.numerical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        # Simple min-max normalization
                        if X_train[col].std() > 0:
                            min_val = X_train[col].min()
                            max_val = X_train[col].max()
                            
                            if max_val > min_val:
                                X_train[col] = (X_train[col] - min_val) / (max_val - min_val)
                                X_test[col] = (X_test[col] - min_val) / (max_val - min_val)
                                
                                # Clip to [0, 1] range
                                X_train[col] = np.clip(X_train[col], 0, 1)
                                X_test[col] = np.clip(X_test[col], 0, 1)
                        
                    except Exception as e:
                        logger.warning(f"Normalization failed for {col}: {e}")
            
            logger.info("Basic numeric normalization completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic numeric normalization failed: {e}")
            return X_train, X_test
    
    def _clean_final_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean final features for output"""
        try:
            # Remove columns with all NaN or infinite values
            cols_to_remove = []
            
            for col in X_train.columns:
                try:
                    if X_train[col].isna().all() or np.isinf(X_train[col]).all():
                        cols_to_remove.append(col)
                    elif col in X_test.columns and (X_test[col].isna().all() or np.isinf(X_test[col]).all()):
                        cols_to_remove.append(col)
                except Exception:
                    cols_to_remove.append(col)
            
            if cols_to_remove:
                X_train = X_train.drop(columns=cols_to_remove)
                X_test = X_test.drop(columns=[col for col in cols_to_remove if col in X_test.columns])
                logger.info(f"Removed {len(cols_to_remove)} problematic columns")
            
            # Final NaN and infinity cleanup
            X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
            X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Ensure matching columns
            common_cols = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            # Convert all to numeric types
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                except Exception:
                    X_train[col] = 0
                    X_test[col] = 0
            
            logger.info(f"Final cleanup completed - {len(common_cols)} features")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final feature cleaning failed: {e}")
            # Return simplified versions
            try:
                return X_train.fillna(0), X_test.fillna(0)
            except Exception:
                # Last resort: create minimal features
                return pd.DataFrame({'feature_1': [0] * len(X_train)}), pd.DataFrame({'feature_1': [0] * len(X_test)})
    
    def _create_minimal_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create absolute minimal features as fallback"""
        try:
            logger.warning("Creating minimal features as fallback")
            
            # Create basic numeric features from first few columns
            feature_cols = []
            for col in train_df.columns:
                if col != target_col and len(feature_cols) < 5:
                    try:
                        # Try to convert to numeric
                        train_numeric = pd.to_numeric(train_df[col], errors='coerce')
                        test_numeric = pd.to_numeric(test_df[col], errors='coerce')
                        
                        if not train_numeric.isna().all():
                            feature_cols.append(col)
                    except Exception:
                        continue
            
            # If no numeric columns found, create dummy features
            if not feature_cols:
                X_train = pd.DataFrame({
                    'feature_1': range(len(train_df)),
                    'feature_2': np.random.rand(len(train_df))
                })
                X_test = pd.DataFrame({
                    'feature_1': range(len(test_df)),
                    'feature_2': np.random.rand(len(test_df))
                })
            else:
                X_train = train_df[feature_cols].copy()
                X_test = test_df[feature_cols].copy()
                
                # Convert to numeric and fill NaN
                for col in feature_cols:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
            
            self.final_feature_columns = list(X_train.columns)
            logger.info(f"Minimal features created: {X_train.shape[1]} features")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Minimal feature creation failed: {e}")
            # Absolute last resort
            X_train = pd.DataFrame({'dummy_feature': [1.0] * len(train_df)})
            X_test = pd.DataFrame({'dummy_feature': [1.0] * len(test_df)})
            self.final_feature_columns = ['dummy_feature']
            return X_train, X_test
    
    # Safe implementations for full feature engineering methods
    def _unify_data_types_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe data type unification"""
        try:
            for col in X_train.columns:
                if col in X_test.columns:
                    # Handle categorical columns safely
                    if X_train[col].dtype.name == 'category' or X_test[col].dtype.name == 'category':
                        # Convert categorical to codes
                        if X_train[col].dtype.name == 'category':
                            X_train[col] = X_train[col].cat.codes
                        if X_test[col].dtype.name == 'category':
                            X_test[col] = X_test[col].cat.codes
                        
                        # Fill missing values
                        X_train[col] = X_train[col].fillna(-1).astype('int32')
                        X_test[col] = X_test[col].fillna(-1).astype('int32')
                    
                    # Handle object columns
                    elif X_train[col].dtype == 'object' or X_test[col].dtype == 'object':
                        # Label encode object columns
                        combined_data = pd.concat([X_train[col], X_test[col]]).astype(str)
                        unique_values = sorted(combined_data.unique())
                        value_map = {val: idx for idx, val in enumerate(unique_values)}
                        
                        X_train[col] = X_train[col].astype(str).map(value_map).fillna(0).astype('int32')
                        X_test[col] = X_test[col].astype(str).map(value_map).fillna(0).astype('int32')
                    
                    # Handle numeric columns
                    else:
                        try:
                            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                        except Exception:
                            X_train[col] = 0
                            X_test[col] = 0
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Safe data type unification failed: {e}")
            return X_train, X_test
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic feature cleaning"""
        try:
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
                logger.info(f"Removed {len(constant_cols)} constant columns")
            
            # Fill missing values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic feature cleaning failed: {e}")
            return X_train, X_test
    
    def _encode_categorical_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical feature encoding"""
        try:
            for col in self.categorical_features:
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
    
    # Placeholder methods for additional feature engineering
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create interaction features (placeholder)"""
        return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create target encoding features (placeholder)"""
        return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create temporal features (placeholder)"""
        return X_train, X_test
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create statistical features (placeholder)"""
        return X_train, X_test
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create frequency features (placeholder)"""
        return X_train, X_test
    
    def _create_polynomial_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create polynomial features (placeholder)"""
        return X_train, X_test
    
    def _create_cross_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create cross features (placeholder)"""
        return X_train, X_test
    
    def _create_binning_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create binning features (placeholder)"""
        return X_train, X_test
    
    def _create_rank_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create rank features (placeholder)"""
        return X_train, X_test
    
    def _create_ratio_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create ratio features (placeholder)"""
        return X_train, X_test
    
    def _create_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create numeric features (placeholder)"""
        return X_train, X_test
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final data cleanup"""
        try:
            # Ensure all data is numeric
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                except Exception:
                    X_train[col] = 0
                    X_test[col] = 0
            
            # Remove infinite values
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            # Final NaN check
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final data cleanup failed: {e}")
            return X_train, X_test
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Finalize processing statistics"""
        try:
            self.final_feature_columns = list(X_train.columns)
            self.processing_stats['processing_time'] = time.time() - self.processing_stats['start_time']
            self.processing_stats['total_features_generated'] = len(self.final_feature_columns)
            
            logger.info(f"Feature engineering finalized - {len(self.final_feature_columns)} features in {self.processing_stats['processing_time']:.2f}s")
            
        except Exception as e:
            logger.warning(f"Processing finalization failed: {e}")
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create basic features only (fallback method)"""
        logger.warning("Using basic features only due to error in main pipeline")
        
        try:
            # Remove target column from features
            feature_cols = [col for col in train_df.columns if col != target_col]
            
            X_train = train_df[feature_cols].copy()
            X_test = test_df[feature_cols].copy()
            
            # Basic cleanup
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Convert object columns to numeric where possible
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                    except Exception:
                        # Use label encoding for categorical
                        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
                        unique_vals = sorted(combined.unique())
                        mapping = {val: idx for idx, val in enumerate(unique_vals)}
                        
                        X_train[col] = X_train[col].astype(str).map(mapping).fillna(0)
                        X_test[col] = X_test[col].astype(str).map(mapping).fillna(0)
                
                # Handle categorical columns
                elif X_train[col].dtype.name == 'category':
                    X_train[col] = X_train[col].cat.codes.fillna(-1)
                    X_test[col] = X_test[col].cat.codes.fillna(-1) if X_test[col].dtype.name == 'category' else 0
            
            # Ensure all columns are numeric
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                except Exception:
                    X_train[col] = 0
                    X_test[col] = 0
            
            self.final_feature_columns = list(X_train.columns)
            logger.info(f"Basic features created: {X_train.shape[1]} features")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Basic feature creation failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)

if __name__ == "__main__":
    # Test code for development
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create dummy data for testing
        train_data = {
            'feature_1': [1, 2, 3, 4, 5] * 10,
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5] * 10,
            'feature_3': ['A', 'B', 'C', 'A', 'B'] * 10,
            'clicked': [0, 1, 0, 1, 0] * 10
        }
        
        test_data = {
            'feature_1': [6, 7, 8, 9, 10] * 5,
            'feature_2': [0.6, 0.7, 0.8, 0.9, 1.0] * 5,
            'feature_3': ['A', 'B', 'C', 'A', 'B'] * 5
        }
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        config = Config()
        engineer = CTRFeatureEngineer(config)
        
        # Test quick mode
        print("Testing quick mode...")
        engineer.set_quick_mode(True)
        X_train_quick, X_test_quick = engineer.engineer_features(train_df, test_df)
        
        print(f"Quick mode results:")
        print(f"X_train shape: {X_train_quick.shape}")
        print(f"X_test shape: {X_test_quick.shape}")
        print(f"Features: {list(X_train_quick.columns)}")
        
        # Test full mode
        print("\nTesting full mode...")
        engineer.set_quick_mode(False)
        X_train_full, X_test_full = engineer.engineer_features(train_df, test_df)
        
        print(f"Full mode results:")
        print(f"X_train shape: {X_train_full.shape}")
        print(f"X_test shape: {X_test_full.shape}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")