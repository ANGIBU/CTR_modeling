# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import gc
import warnings
from collections import defaultdict
import hashlib
from pathlib import Path

# Scientific libraries
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory usage monitoring"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in GB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        except ImportError:
            return 0.0
    
    def log_memory_status(self, stage: str = ""):
        """Log current memory status"""
        memory_gb = self.get_memory_usage()
        logger.info(f"Memory usage {stage}: {memory_gb:.2f}GB")

class SafeDataTypeConverter:
    """Safe data type conversion with memory optimization"""
    
    def __init__(self):
        self.conversion_map = {}
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        try:
            logger.info("Data type optimization started")
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            for col in df.columns:
                col_type = df[col].dtype
                
                if col_type != object:
                    # Numeric optimization
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float32)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
                else:
                    # String/categorical optimization
                    if df[col].nunique() / len(df) < 0.5:
                        df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            reduction = (1 - optimized_memory / original_memory) * 100
            logger.info(f"Memory optimization: {reduction:.1f}% reduction ({original_memory:.1f}MB -> {optimized_memory:.1f}MB)")
            
            return df
            
        except Exception as e:
            logger.warning(f"Data type optimization failed: {e}")
            return df

class SeqColumnProcessor:
    """Sequence column processing specialized for CTR"""
    
    def __init__(self, hash_size: int = 10000):
        self.hash_size = hash_size
        self.seq_stats = {}
        self.item_encoders = {}
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Process sequence columns with CTR-aware features"""
        logger.info("Sequence column processing started")
        
        try:
            X_processed = X.copy()
            seq_columns = [col for col in X.columns if 'seq' in col.lower()]
            
            if not seq_columns:
                logger.info("No sequence columns found")
                return X_processed
            
            for col in seq_columns:
                logger.info(f"Processing sequence column: {col}")
                X_processed = self._process_single_seq_column(X_processed, col, y)
            
            # Remove original sequence columns to save memory
            X_processed = X_processed.drop(columns=seq_columns)
            
            self.is_fitted = True
            logger.info(f"Sequence processing completed - {len(seq_columns)} columns processed")
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Sequence processing failed: {e}")
            return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform sequence columns using fitted encoders"""
        if not self.is_fitted:
            logger.warning("SeqColumnProcessor not fitted, returning original data")
            return X
        
        try:
            X_processed = X.copy()
            seq_columns = [col for col in X.columns if 'seq' in col.lower()]
            
            for col in seq_columns:
                if col in self.seq_stats:
                    X_processed = self._transform_single_seq_column(X_processed, col)
            
            # Remove original sequence columns
            X_processed = X_processed.drop(columns=seq_columns, errors='ignore')
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Sequence transform failed: {e}")
            return X
    
    def _process_single_seq_column(self, X: pd.DataFrame, col: str, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Process single sequence column with CTR features"""
        try:
            # Parse sequences
            sequences = X[col].fillna('').astype(str)
            
            # Basic sequence features
            X[f"{col}_length"] = sequences.apply(lambda x: len(x.split(',')) if x else 0)
            X[f"{col}_is_empty"] = (sequences == '').astype(int)
            
            # Item frequency and CTR features
            if y is not None:
                item_ctr_stats = self._calculate_item_ctr_stats(sequences, y)
                self.seq_stats[col] = item_ctr_stats
                
                # CTR-based sequence features
                X[f"{col}_avg_item_ctr"] = sequences.apply(
                    lambda x: self._get_sequence_avg_ctr(x, item_ctr_stats)
                )
                X[f"{col}_max_item_ctr"] = sequences.apply(
                    lambda x: self._get_sequence_max_ctr(x, item_ctr_stats)
                )
                X[f"{col}_ctr_variance"] = sequences.apply(
                    lambda x: self._get_sequence_ctr_variance(x, item_ctr_stats)
                )
            
            # Hash-based item encoding (memory efficient)
            unique_items = set()
            for seq in sequences:
                if seq:
                    unique_items.update(seq.split(','))
            
            # Create hash encoding for top items
            item_counts = defaultdict(int)
            for seq in sequences:
                if seq:
                    for item in seq.split(','):
                        item_counts[item] += 1
            
            # Keep only top items by frequency
            top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:100]
            
            for item, _ in top_items:
                item_hash = hash(item) % self.hash_size
                X[f"{col}_has_item_{item_hash}"] = sequences.apply(
                    lambda x: 1 if item in x.split(',') else 0
                )
            
            # Sequence diversity features
            X[f"{col}_unique_ratio"] = sequences.apply(
                lambda x: len(set(x.split(','))) / max(1, len(x.split(','))) if x else 0
            )
            
            return X
            
        except Exception as e:
            logger.error(f"Single sequence processing failed for {col}: {e}")
            return X
    
    def _calculate_item_ctr_stats(self, sequences: pd.Series, y: pd.Series) -> Dict[str, float]:
        """Calculate CTR statistics for each item"""
        item_clicks = defaultdict(int)
        item_impressions = defaultdict(int)
        
        for seq, clicked in zip(sequences, y):
            if seq:
                items = seq.split(',')
                for item in items:
                    item_impressions[item] += 1
                    if clicked:
                        item_clicks[item] += 1
        
        # Calculate CTR with smoothing
        item_ctr = {}
        global_ctr = np.mean(y)
        smoothing_factor = 10  # Laplace smoothing
        
        for item in item_impressions:
            clicks = item_clicks.get(item, 0)
            impressions = item_impressions[item]
            
            # Smoothed CTR
            smoothed_ctr = (clicks + smoothing_factor * global_ctr) / (impressions + smoothing_factor)
            item_ctr[item] = smoothed_ctr
        
        return item_ctr
    
    def _get_sequence_avg_ctr(self, seq: str, item_ctr_stats: Dict[str, float]) -> float:
        """Get average CTR of items in sequence"""
        if not seq:
            return 0.0
        
        items = seq.split(',')
        ctrs = [item_ctr_stats.get(item, 0.0191) for item in items]  # Default to target CTR
        return np.mean(ctrs)
    
    def _get_sequence_max_ctr(self, seq: str, item_ctr_stats: Dict[str, float]) -> float:
        """Get maximum CTR of items in sequence"""
        if not seq:
            return 0.0
        
        items = seq.split(',')
        ctrs = [item_ctr_stats.get(item, 0.0191) for item in items]
        return np.max(ctrs)
    
    def _get_sequence_ctr_variance(self, seq: str, item_ctr_stats: Dict[str, float]) -> float:
        """Get CTR variance of items in sequence"""
        if not seq:
            return 0.0
        
        items = seq.split(',')
        if len(items) < 2:
            return 0.0
        
        ctrs = [item_ctr_stats.get(item, 0.0191) for item in items]
        return np.var(ctrs)
    
    def _transform_single_seq_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """Transform single sequence column using fitted stats"""
        try:
            sequences = X[col].fillna('').astype(str)
            
            # Basic features
            X[f"{col}_length"] = sequences.apply(lambda x: len(x.split(',')) if x else 0)
            X[f"{col}_is_empty"] = (sequences == '').astype(int)
            
            # CTR features if stats available
            if col in self.seq_stats:
                item_ctr_stats = self.seq_stats[col]
                
                X[f"{col}_avg_item_ctr"] = sequences.apply(
                    lambda x: self._get_sequence_avg_ctr(x, item_ctr_stats)
                )
                X[f"{col}_max_item_ctr"] = sequences.apply(
                    lambda x: self._get_sequence_max_ctr(x, item_ctr_stats)
                )
                X[f"{col}_ctr_variance"] = sequences.apply(
                    lambda x: self._get_sequence_ctr_variance(x, item_ctr_stats)
                )
            
            return X
            
        except Exception as e:
            logger.error(f"Single sequence transform failed for {col}: {e}")
            return X

class CTRTargetEncoder:
    """CTR-specialized target encoder with robust smoothing"""
    
    def __init__(self, smoothing: float = 10.0, min_samples: int = 5, target_ctr: float = 0.0191):
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.target_ctr = target_ctr
        self.global_mean = target_ctr
        self.category_stats = {}
        self.is_fitted = False
    
    def fit_transform(self, X: pd.Series, y: pd.Series, column_name: str) -> pd.Series:
        """Fit and transform with CTR-aware smoothing"""
        try:
            logger.info(f"CTR target encoding: {column_name}")
            
            # Convert to string to handle mixed types
            X_str = X.astype(str)
            self.global_mean = np.mean(y)
            
            # Calculate category statistics
            category_stats = {}
            for cat in X_str.unique():
                if pd.isna(cat) or cat == 'nan':
                    continue
                    
                mask = (X_str == cat)
                cat_y = y[mask]
                
                if len(cat_y) >= self.min_samples:
                    cat_mean = np.mean(cat_y)
                    count = len(cat_y)
                    
                    # Enhanced smoothing for CTR
                    # Higher smoothing for categories with extreme CTR values
                    ctr_deviation = abs(cat_mean - self.global_mean)
                    adaptive_smoothing = self.smoothing * (1 + ctr_deviation * 100)
                    
                    smoothed_mean = (count * cat_mean + adaptive_smoothing * self.global_mean) / (count + adaptive_smoothing)
                    
                    category_stats[cat] = {
                        'mean': smoothed_mean,
                        'count': count,
                        'raw_mean': cat_mean
                    }
            
            self.category_stats[column_name] = category_stats
            self.is_fitted = True
            
            # Transform
            result = self._transform_series(X_str, column_name)
            
            logger.info(f"Target encoding completed: {len(category_stats)} categories encoded")
            return result
            
        except Exception as e:
            logger.error(f"Target encoding failed for {column_name}: {e}")
            return pd.Series([self.global_mean] * len(X), index=X.index)
    
    def transform(self, X: pd.Series, column_name: str) -> pd.Series:
        """Transform using fitted statistics"""
        if not self.is_fitted or column_name not in self.category_stats:
            return pd.Series([self.global_mean] * len(X), index=X.index)
        
        X_str = X.astype(str)
        return self._transform_series(X_str, column_name)
    
    def _transform_series(self, X_str: pd.Series, column_name: str) -> pd.Series:
        """Transform series using category statistics"""
        try:
            stats = self.category_stats[column_name]
            result = []
            
            for cat in X_str:
                if cat in stats:
                    result.append(stats[cat]['mean'])
                else:
                    result.append(self.global_mean)
            
            return pd.Series(result, index=X_str.index)
            
        except Exception as e:
            logger.error(f"Transform series failed: {e}")
            return pd.Series([self.global_mean] * len(X_str), index=X_str.index)

class CTRFeatureEngineer:
    """CTR feature engineering with focus on prediction accuracy"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.data_converter = SafeDataTypeConverter()
        self.seq_processor = SeqColumnProcessor(hash_size=config.SEQ_HASH_SIZE)
        
        # CTR specific settings
        self.target_ctr = 0.0191
        self.target_column = 'clicked'
        
        # Processing modes
        self.quick_mode = False
        self.memory_efficient_mode = True
        
        # Feature engineering components
        self.target_encoders = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # Feature tracking
        self.original_features = []
        self.numerical_features = []
        self.categorical_features = []
        self.generated_features = []
        self.selected_features = []
        
        # Processing stats
        self.processing_stats = {}
    
    def set_quick_mode(self, quick_mode: bool = True):
        """Set quick processing mode"""
        self.quick_mode = quick_mode
        logger.info(f"Quick mode: {'ON' if quick_mode else 'OFF'}")
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Main feature engineering pipeline for CTR prediction"""
        
        logger.info("=== CTR Feature Engineering Started ===")
        start_time = time.time()
        
        try:
            # Detect target column
            if self.target_column not in train_df.columns:
                possible_targets = ['clicked', 'click', 'target', 'label', 'y']
                for col in possible_targets:
                    if col in train_df.columns:
                        self.target_column = col
                        break
                else:
                    logger.error("No target column found")
                    return None, None
            
            logger.info(f"Target column detected: {self.target_column}")
            
            # Prepare data
            feature_columns = [col for col in train_df.columns if col != self.target_column]
            self.original_features = feature_columns.copy()
            
            X_train = train_df[feature_columns].copy()
            X_test = test_df[feature_columns].copy()
            y_train = train_df[self.target_column].copy()
            
            # Log initial statistics
            initial_ctr = np.mean(y_train)
            logger.info(f"Initial data - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Target CTR: {initial_ctr:.4f}, Features: {len(feature_columns)}")
            
            # Memory optimization
            self.memory_monitor.log_memory_status("Initial")
            X_train = self.data_converter.optimize_dtypes(X_train)
            X_test = self.data_converter.optimize_dtypes(X_test)
            
            # Phase 1: Column classification
            logger.info("Phase 1: Column classification")
            self._classify_columns(X_train)
            
            # Phase 2: Sequence processing
            logger.info("Phase 2: Sequence column processing")
            X_train = self.seq_processor.fit_transform(X_train, y_train)
            X_test = self.seq_processor.transform(X_test)
            
            # Update column lists after sequence processing
            self._classify_columns(X_train)
            
            # Phase 3: Feature engineering based on mode
            if self.quick_mode:
                logger.info("Phase 3: Quick feature engineering")
                X_train, X_test = self._quick_feature_engineering(X_train, X_test, y_train)
            else:
                logger.info("Phase 3: Full feature engineering")
                X_train, X_test = self._full_feature_engineering(X_train, X_test, y_train)
            
            # Phase 4: Feature selection
            logger.info("Phase 4: Feature selection")
            X_train, X_test = self._ctr_feature_selection(X_train, X_test, y_train)
            
            # Phase 5: Final preprocessing
            logger.info("Phase 5: Final preprocessing")
            X_train, X_test = self._final_preprocessing(X_train, X_test)
            
            # Statistics
            processing_time = time.time() - start_time
            self.processing_stats = {
                'processing_time': processing_time,
                'original_features': len(self.original_features),
                'final_features': X_train.shape[1],
                'generated_features': X_train.shape[1] - len(self.original_features),
                'target_ctr': initial_ctr,
                'feature_types': {
                    'numerical': len(self.numerical_features),
                    'categorical': len(self.categorical_features),
                    'generated': len(self.generated_features)
                }
            }
            
            logger.info("=== CTR Feature Engineering Completed ===")
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Features: {len(self.original_features)} → {X_train.shape[1]} (generated: {X_train.shape[1] - len(self.original_features)})")
            logger.info(f"Final data - Train: {X_train.shape}, Test: {X_test.shape}")
            
            self.memory_monitor.log_memory_status("Completed")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return None, None
    
    def _classify_columns(self, X: pd.DataFrame):
        """Classify columns by type"""
        self.numerical_features = []
        self.categorical_features = []
        
        for col in X.columns:
            if X[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                if X[col].nunique() > 20:  # Threshold for numerical vs categorical
                    self.numerical_features.append(col)
                else:
                    self.categorical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        logger.info(f"Columns classified - Numerical: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
    
    def _quick_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Quick feature engineering for fast processing"""
        
        # Only essential features for quick mode
        
        # 1. Basic target encoding for top categorical features
        top_categorical = self.categorical_features[:5]  # Limit to top 5
        for col in top_categorical:
            if col in X_train.columns:
                try:
                    encoder = CTRTargetEncoder(target_ctr=self.target_ctr)
                    X_train[f"{col}_target_enc"] = encoder.fit_transform(X_train[col], y_train, col)
                    X_test[f"{col}_target_enc"] = encoder.transform(X_test[col], col)
                    self.target_encoders[col] = encoder
                    self.generated_features.append(f"{col}_target_enc")
                except Exception as e:
                    logger.warning(f"Quick target encoding failed for {col}: {e}")
        
        # 2. Basic numerical features
        top_numerical = self.numerical_features[:5]  # Limit to top 5
        for col in top_numerical:
            if col in X_train.columns:
                try:
                    # Log transformation for skewed features
                    if X_train[col].min() > 0:
                        X_train[f"{col}_log"] = np.log1p(X_train[col])
                        X_test[f"{col}_log"] = np.log1p(X_test[col])
                        self.generated_features.append(f"{col}_log")
                except Exception as e:
                    logger.warning(f"Quick numerical feature failed for {col}: {e}")
        
        logger.info(f"Quick feature engineering - generated: {len(self.generated_features)} features")
        return X_train, X_test
    
    def _full_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Full feature engineering pipeline"""
        
        # 1. Target encoding for categorical features
        for col in self.categorical_features:
            if col in X_train.columns:
                try:
                    encoder = CTRTargetEncoder(target_ctr=self.target_ctr)
                    X_train[f"{col}_target_enc"] = encoder.fit_transform(X_train[col], y_train, col)
                    X_test[f"{col}_target_enc"] = encoder.transform(X_test[col], col)
                    self.target_encoders[col] = encoder
                    self.generated_features.append(f"{col}_target_enc")
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
        
        # 2. Frequency encoding
        for col in self.categorical_features[:10]:  # Limit to prevent memory issues
            if col in X_train.columns:
                try:
                    freq_map = X_train[col].value_counts().to_dict()
                    X_train[f"{col}_freq"] = X_train[col].map(freq_map).fillna(0)
                    X_test[f"{col}_freq"] = X_test[col].map(freq_map).fillna(0)
                    self.generated_features.append(f"{col}_freq")
                except Exception as e:
                    logger.warning(f"Frequency encoding failed for {col}: {e}")
        
        # 3. Numerical transformations
        for col in self.numerical_features:
            if col in X_train.columns:
                try:
                    # Log transformation
                    if X_train[col].min() > 0:
                        X_train[f"{col}_log"] = np.log1p(X_train[col])
                        X_test[f"{col}_log"] = np.log1p(X_test[col])
                        self.generated_features.append(f"{col}_log")
                    
                    # Square root transformation
                    if X_train[col].min() >= 0:
                        X_train[f"{col}_sqrt"] = np.sqrt(X_train[col])
                        X_test[f"{col}_sqrt"] = np.sqrt(X_test[col])
                        self.generated_features.append(f"{col}_sqrt")
                    
                    # Binning
                    try:
                        X_train[f"{col}_binned"] = pd.qcut(X_train[col], q=5, labels=False, duplicates='drop')
                        bin_edges = pd.qcut(X_train[col], q=5, duplicates='drop').cat.categories
                        X_test[f"{col}_binned"] = pd.cut(X_test[col], bins=bin_edges, labels=False, include_lowest=True)
                        self.generated_features.append(f"{col}_binned")
                    except Exception:
                        pass  # Skip if binning fails
                        
                except Exception as e:
                    logger.warning(f"Numerical transformation failed for {col}: {e}")
        
        # 4. Interaction features (limited to prevent explosion)
        top_features = (self.numerical_features[:3] + [f"{col}_target_enc" for col in self.categorical_features[:2] if f"{col}_target_enc" in X_train.columns])
        
        for i, col1 in enumerate(top_features):
            for col2 in top_features[i+1:]:
                if col1 in X_train.columns and col2 in X_train.columns:
                    try:
                        # Multiplication interaction
                        X_train[f"{col1}_x_{col2}"] = X_train[col1] * X_train[col2]
                        X_test[f"{col1}_x_{col2}"] = X_test[col1] * X_test[col2]
                        self.generated_features.append(f"{col1}_x_{col2}")
                        
                        # Ratio interaction (if col2 != 0)
                        mask_train = X_train[col2] != 0
                        mask_test = X_test[col2] != 0
                        X_train[f"{col1}_div_{col2}"] = 0.0
                        X_test[f"{col1}_div_{col2}"] = 0.0
                        X_train.loc[mask_train, f"{col1}_div_{col2}"] = X_train.loc[mask_train, col1] / X_train.loc[mask_train, col2]
                        X_test.loc[mask_test, f"{col1}_div_{col2}"] = X_test.loc[mask_test, col1] / X_test.loc[mask_test, col2]
                        self.generated_features.append(f"{col1}_div_{col2}")
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature failed for {col1} x {col2}: {e}")
                        
                    # Limit interactions to prevent memory issues
                    if len(self.generated_features) > 200:
                        break
            if len(self.generated_features) > 200:
                break
        
        logger.info(f"Full feature engineering - generated: {len(self.generated_features)} features")
        return X_train, X_test
    
    def _ctr_feature_selection(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CTR-specific feature selection"""
        
        if self.quick_mode or X_train.shape[1] <= 50:
            logger.info("Skipping feature selection (quick mode or few features)")
            return X_train, X_test
        
        try:
            logger.info(f"Feature selection started - input features: {X_train.shape[1]}")
            
            # Remove highly correlated features
            corr_threshold = 0.95
            corr_matrix = X_train.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
            
            if high_corr_features:
                X_train = X_train.drop(columns=high_corr_features)
                X_test = X_test.drop(columns=high_corr_features)
                logger.info(f"Removed {len(high_corr_features)} highly correlated features")
            
            # Mutual information feature selection
            if X_train.shape[1] > 100:
                n_features = min(100, X_train.shape[1] // 2)
                
                # Fill NaN values for feature selection
                X_train_filled = X_train.fillna(0)
                
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
                X_train_selected = selector.fit_transform(X_train_filled, y_train)
                
                selected_features = X_train.columns[selector.get_support()].tolist()
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
                
                self.feature_selectors['mutual_info'] = selector
                logger.info(f"Mutual information selection: {len(selected_features)} features selected")
            
            logger.info(f"Feature selection completed - final features: {X_train.shape[1]}")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return X_train, X_test
    
    def _final_preprocessing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final preprocessing steps"""
        
        try:
            # Handle missing values
            for col in X_train.columns:
                if X_train[col].isnull().any():
                    if X_train[col].dtype in ['float16', 'float32', 'float64']:
                        fill_value = X_train[col].median()
                    else:
                        fill_value = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else 0
                    
                    X_train[col] = X_train[col].fillna(fill_value)
                    X_test[col] = X_test[col].fillna(fill_value)
            
            # Ensure no infinite values
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            # Final data type optimization
            X_train = self.data_converter.optimize_dtypes(X_train)
            X_test = self.data_converter.optimize_dtypes(X_test)
            
            logger.info("Final preprocessing completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final preprocessing failed: {e}")
            return X_train, X_test
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering process"""
        return {
            'processing_stats': self.processing_stats,
            'original_features': len(self.original_features),
            'generated_features': len(self.generated_features),
            'feature_types': {
                'numerical': len(self.numerical_features),
                'categorical': len(self.categorical_features)
            },
            'target_ctr': self.target_ctr
        }

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("CTR Feature Engineering Test")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'feat_a_1': np.random.choice(['A', 'B', 'C', None], n_samples),
        'feat_a_2': np.random.choice([1, 2, 3], n_samples),
        'feat_b_1': np.random.randn(n_samples),
        'feat_b_2': np.random.randn(n_samples),
        'seq': [f"item_{i%10},item_{(i+1)%10},item_{(i+2)%10}" for i in range(n_samples)],
        'clicked': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    # Split into train and test
    train_df = sample_data.iloc[:800].copy()
    test_df = sample_data.iloc[800:].copy()
    test_df = test_df.drop(columns=['clicked'])  # Remove target from test
    
    print(f"Sample data: Train {train_df.shape}, Test {test_df.shape}")
    print(f"Target CTR: {np.mean(train_df['clicked']):.4f}")
    
    # Test feature engineering
    feature_engineer = CTRFeatureEngineer()
    feature_engineer.set_quick_mode(True)  # Quick test
    
    X_train, X_test = feature_engineer.engineer_features(train_df, test_df)
    
    if X_train is not None and X_test is not None:
        print(f"Feature engineering successful!")
        print(f"Output: Train {X_train.shape}, Test {X_test.shape}")
        print(f"Generated features: {X_train.shape[1] - 4}")
        
        # Show feature summary
        summary = feature_engineer.get_feature_importance_summary()
        print(f"Processing stats: {summary['processing_stats']}")
    else:
        print("Feature engineering failed!")