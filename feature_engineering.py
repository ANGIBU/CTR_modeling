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
import hashlib
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class SafeDataTypeConverter:
    """Safe data type conversion for mixed categorical data"""
    
    @staticmethod
    def safe_categorical_conversion(series: pd.Series, handle_mixed: bool = True) -> pd.Series:
        """Safely convert mixed data types to categorical"""
        try:
            if series.dtype == 'category':
                return series
            
            # Handle mixed float/string data
            if handle_mixed and series.dtype == 'object':
                # Convert all to string first to ensure consistency
                series_str = series.astype(str)
                # Replace 'nan' and 'None' with actual NaN
                series_str = series_str.replace(['nan', 'None', 'NaT', 'null'], np.nan)
                return series_str.astype('category')
            
            # For numeric data with NaN, convert to string then categorical
            if series.dtype in ['float64', 'float32'] and series.isna().any():
                series_str = series.fillna('missing').astype(str)
                return series_str.astype('category')
            
            return series.astype('category')
            
        except Exception as e:
            logger.warning(f"Categorical conversion failed: {e}")
            # Fallback: convert everything to string then categorical
            try:
                return series.astype(str).astype('category')
            except:
                return series  # Return original if all conversions fail
    
    @staticmethod
    def safe_fillna_categorical(series: pd.Series, fill_value: str = 'missing') -> pd.Series:
        """Safely fill NaN in categorical series"""
        try:
            if series.dtype.name == 'category':
                # Add the fill_value to categories if not present
                if fill_value not in series.cat.categories:
                    series = series.cat.add_categories([fill_value])
                return series.fillna(fill_value)
            else:
                return series.fillna(fill_value)
                
        except Exception as e:
            logger.warning(f"Safe fillna failed: {e}")
            # Fallback: convert to string and fill
            try:
                return series.astype(str).fillna(fill_value).astype('category')
            except:
                return series.fillna(fill_value)
    
    @staticmethod
    def ensure_string_categorical(series: pd.Series) -> pd.Series:
        """Ensure categorical series contains only strings"""
        try:
            # Convert to string first, then to category
            str_series = series.astype(str)
            # Replace string representations of NaN
            str_series = str_series.replace(['nan', 'None', 'NaT', 'null'], 'missing')
            return str_series.astype('category')
        except Exception as e:
            logger.warning(f"String categorical conversion failed: {e}")
            return series

class SeqColumnProcessor:
    """Sequence column processor with hash-based feature extraction"""
    
    def __init__(self, hash_size: int = 100000):
        self.hash_size = hash_size
        self.seq_features = {}
        self.is_fitted = False
        
    def process_seq_column(self, seq_series: pd.Series) -> Dict[str, pd.Series]:
        """Process sequence column into hash-based features"""
        try:
            logger.info(f"Processing seq column: {len(seq_series)} rows")
            
            features = {}
            
            # Convert to string and handle NaN
            seq_str = seq_series.astype(str).fillna('')
            
            # Extract sequence length
            features['seq_length'] = seq_str.str.len()
            
            # Count commas (sequence item count)
            features['seq_item_count'] = seq_str.str.count(',') + 1
            features['seq_item_count'] = features['seq_item_count'].where(seq_str != '', 0)
            
            # Hash the full sequence
            def hash_seq(x):
                if x == '':
                    return 0
                return int(hashlib.md5(x.encode()).hexdigest(), 16) % self.hash_size
            
            features['seq_hash_full'] = seq_str.apply(hash_seq)
            
            # Extract first few items if comma-separated
            for i in range(3):  # First 3 items
                item_series = seq_str.str.split(',').str.get(i).fillna('missing')
                features[f'seq_item_{i}_hash'] = item_series.apply(
                    lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % self.hash_size
                )
            
            # Statistical features
            features['seq_unique_chars'] = seq_str.apply(lambda x: len(set(x)) if x else 0)
            features['seq_digit_count'] = seq_str.str.count(r'\d')
            features['seq_alpha_count'] = seq_str.str.count(r'[a-zA-Z]')
            
            logger.info(f"Seq column processed: {len(features)} features created")
            return features
            
        except Exception as e:
            logger.error(f"Seq column processing failed: {e}")
            return {'seq_length': pd.Series([0] * len(seq_series))}

class ChunkTargetEncoder:
    """Memory-efficient chunk-based target encoder"""
    
    def __init__(self, chunk_size: int = 50000, smoothing: int = 100):
        self.chunk_size = chunk_size
        self.smoothing = smoothing
        self.category_stats = {}
        self.global_mean = 0.0
        self.is_fitted = False
    
    def fit_chunk_based(self, X: pd.Series, y: pd.Series, column_name: str) -> bool:
        """Fit target encoder using chunk-based processing"""
        try:
            logger.info(f"Chunk-based target encoding for {column_name}: {len(X)} samples")
            
            if len(X) != len(y):
                logger.error(f"Length mismatch: X={len(X)}, y={len(y)}")
                return False
            
            self.global_mean = y.mean()
            category_counts = {}
            category_sums = {}
            
            # Process in chunks to manage memory
            for start_idx in range(0, len(X), self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, len(X))
                
                X_chunk = X.iloc[start_idx:end_idx].astype(str)
                y_chunk = y.iloc[start_idx:end_idx]
                
                # Aggregate statistics
                for cat, target in zip(X_chunk, y_chunk):
                    if cat not in category_counts:
                        category_counts[cat] = 0
                        category_sums[cat] = 0.0
                    
                    category_counts[cat] += 1
                    category_sums[cat] += target
                
                # Memory cleanup every 10 chunks
                if (start_idx // self.chunk_size) % 10 == 0:
                    gc.collect()
            
            # Calculate smoothed means
            self.category_stats[column_name] = {}
            for cat in category_counts:
                count = category_counts[cat]
                cat_mean = category_sums[cat] / count if count > 0 else self.global_mean
                
                # Apply smoothing
                smoothed_mean = (count * cat_mean + self.smoothing * self.global_mean) / (count + self.smoothing)
                
                self.category_stats[column_name][cat] = {
                    'mean': smoothed_mean,
                    'count': count
                }
            
            self.is_fitted = True
            logger.info(f"Target encoding completed: {len(self.category_stats[column_name])} categories")
            return True
            
        except Exception as e:
            logger.error(f"Chunk-based target encoding failed: {e}")
            return False
    
    def transform(self, X: pd.Series, column_name: str) -> pd.Series:
        """Transform using fitted statistics"""
        try:
            if not self.is_fitted or column_name not in self.category_stats:
                return pd.Series([self.global_mean] * len(X))
            
            X_str = X.astype(str)
            result = []
            
            stats = self.category_stats[column_name]
            
            for cat in X_str:
                if cat in stats:
                    result.append(stats[cat]['mean'])
                else:
                    result.append(self.global_mean)
            
            return pd.Series(result, index=X.index)
            
        except Exception as e:
            logger.error(f"Target encoding transform failed: {e}")
            return pd.Series([self.global_mean] * len(X))

class CTRFeatureEngineer:
    """CTR feature engineering class with memory efficiency and safe data handling"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.data_converter = SafeDataTypeConverter()
        self.seq_processor = SeqColumnProcessor(hash_size=config.SEQ_HASH_SIZE)
        
        # Processing modes
        self.quick_mode = False
        self.memory_efficient_mode = True
        
        # Feature engineering state variables
        self.target_encoders = {}
        self.chunk_encoders = {}
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
        self.seq_features = []
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
        else:
            logger.info("Quick mode disabled - comprehensive feature engineering")
    
    def _classify_columns(self, X_train: pd.DataFrame) -> None:
        """Classify columns into numerical and categorical with safe handling"""
        try:
            self.numerical_features = []
            self.categorical_features = []
            
            for col in X_train.columns:
                if col == self.target_column:
                    continue
                
                # Check data type and unique values
                dtype = X_train[col].dtype
                nunique = X_train[col].nunique()
                total_rows = len(X_train)
                
                # Determine if column should be treated as categorical
                is_categorical = False
                
                if dtype == 'object' or dtype.name == 'category':
                    is_categorical = True
                elif dtype in ['int64', 'int32', 'int16', 'int8']:
                    # Integer columns with limited unique values might be categorical
                    if nunique < min(50, total_rows * 0.05):
                        is_categorical = True
                elif dtype in ['float64', 'float32']:
                    # Float columns with very few unique values might be categorical IDs
                    if nunique < min(20, total_rows * 0.01):
                        is_categorical = True
                
                # Enforce limits for memory efficiency
                if is_categorical and nunique > self.config.MAX_CATEGORICAL_UNIQUE:
                    logger.warning(f"Column {col} has {nunique} unique values, treating as numerical")
                    is_categorical = False
                
                if is_categorical:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
            
            logger.info(f"Column classification completed - Numeric: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
            
        except Exception as e:
            logger.error(f"Column classification failed: {e}")
            # Fallback: treat object columns as categorical, others as numerical
            for col in X_train.columns:
                if col != self.target_column:
                    if X_train[col].dtype == 'object':
                        self.categorical_features.append(col)
                    else:
                        self.numerical_features.append(col)
    
    def _process_seq_columns(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process sequence columns with hash-based features"""
        try:
            seq_columns = [col for col in X_train.columns if 'seq' in col.lower()]
            
            if not seq_columns:
                return X_train, X_test
                
            logger.info(f"Processing {len(seq_columns)} sequence columns")
            
            for seq_col in seq_columns:
                if not self.memory_monitor.check_memory_safety(2.0):
                    logger.warning(f"Skipping seq column {seq_col} due to memory constraints")
                    continue
                
                try:
                    # Process training data
                    train_seq_features = self.seq_processor.process_seq_column(X_train[seq_col])
                    
                    # Process test data
                    test_seq_features = self.seq_processor.process_seq_column(X_test[seq_col])
                    
                    # Add features to dataframes
                    for feat_name, feat_series in train_seq_features.items():
                        new_col_name = f"{seq_col}_{feat_name}"
                        X_train[new_col_name] = feat_series
                        self.seq_features.append(new_col_name)
                    
                    for feat_name, feat_series in test_seq_features.items():
                        new_col_name = f"{seq_col}_{feat_name}"
                        X_test[new_col_name] = feat_series
                    
                    # Remove original seq column
                    X_train = X_train.drop(columns=[seq_col])
                    X_test = X_test.drop(columns=[seq_col])
                    self.removed_columns.append(seq_col)
                    
                    logger.info(f"Seq column {seq_col} processed: {len(train_seq_features)} features created")
                    
                except Exception as e:
                    logger.warning(f"Seq column processing failed for {seq_col}: {e}")
                    continue
            
            logger.info(f"Seq column processing completed: {len(self.seq_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Seq column processing failed: {e}")
            return X_train, X_test
    
    def _safe_preprocessing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe data preprocessing with memory efficiency and complex data handling"""
        try:
            logger.info("Safe data preprocessing started")
            
            # Check memory before preprocessing
            if not self.memory_monitor.check_memory_safety(5.0):
                self.memory_monitor.force_cleanup()
            
            # Special handling for complex sequence columns
            problematic_columns = []
            for col in X_train.columns:
                if col in self.numerical_features:
                    # Check for complex string sequences in "numerical" columns
                    sample_values = X_train[col].dropna().head(100)
                    if sample_values.dtype == 'object':
                        # Check if contains comma-separated sequences
                        has_sequences = sample_values.astype(str).str.contains(',', na=False).any()
                        if has_sequences:
                            logger.warning(f"Complex sequence data detected in {col}, removing from numerical features")
                            problematic_columns.append(col)
                            # Remove from numerical features and mark for removal
                            if col in self.numerical_features:
                                self.numerical_features.remove(col)
                            self.removed_columns.append(col)
            
            # Handle missing values safely for categorical columns
            for col in self.categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        # Skip problematic columns
                        if col in problematic_columns:
                            continue
                            
                        # Safe categorical conversion
                        X_train[col] = self.data_converter.safe_categorical_conversion(X_train[col])
                        X_test[col] = self.data_converter.safe_categorical_conversion(X_test[col])
                        
                        # Safe fillna
                        X_train[col] = self.data_converter.safe_fillna_categorical(X_train[col], 'missing')
                        X_test[col] = self.data_converter.safe_fillna_categorical(X_test[col], 'missing')
                        
                        # Ensure both train and test have consistent categories
                        all_categories = set(X_train[col].cat.categories) | set(X_test[col].cat.categories)
                        X_train[col] = X_train[col].cat.set_categories(all_categories)
                        X_test[col] = X_test[col].cat.set_categories(all_categories)
                        
                    except Exception as e:
                        logger.warning(f"Categorical preprocessing failed for {col}: {e}")
                        # Fallback to string conversion
                        try:
                            X_train[col] = X_train[col].astype(str).fillna('missing')
                            X_test[col] = X_test[col].astype(str).fillna('missing')
                        except:
                            # If even string conversion fails, mark for removal
                            problematic_columns.append(col)
                            logger.warning(f"Marking {col} for removal due to conversion issues")
            
            # Handle missing values for numerical columns
            for col in self.numerical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        # Skip problematic columns
                        if col in problematic_columns:
                            continue
                            
                        # Ensure numeric data type
                        if X_train[col].dtype == 'object':
                            # Try to convert to numeric
                            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                        
                        # Use median for numerical columns
                        median_val = X_train[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        
                        X_train[col] = X_train[col].fillna(median_val)
                        X_test[col] = X_test[col].fillna(median_val)
                        
                    except Exception as e:
                        logger.warning(f"Numerical preprocessing failed for {col}: {e}")
                        try:
                            X_train[col] = X_train[col].fillna(0)
                            X_test[col] = X_test[col].fillna(0)
                        except:
                            problematic_columns.append(col)
                            logger.warning(f"Marking {col} for removal due to preprocessing issues")
            
            # Remove problematic columns
            if problematic_columns:
                columns_to_remove = [col for col in problematic_columns if col in X_train.columns]
                if columns_to_remove:
                    logger.warning(f"Removing {len(columns_to_remove)} problematic columns: {columns_to_remove[:5]}...")  # Show first 5
                    X_train = X_train.drop(columns=columns_to_remove)
                    X_test = X_test.drop(columns=columns_to_remove)
                    self.removed_columns.extend(columns_to_remove)
            
            logger.info("Safe data preprocessing completed")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Safe preprocessing failed: {e}")
            return X_train, X_test
    
    def _chunk_target_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-efficient chunk-based target encoding"""
        try:
            if self.memory_efficient_mode:
                # Limit number of columns for memory efficiency
                categorical_subset = self.categorical_features[:min(10, len(self.categorical_features))]
            else:
                categorical_subset = self.categorical_features
            
            logger.info(f"Chunk-based target encoding started for {len(categorical_subset)} categorical features")
            
            for col in categorical_subset:
                if col not in X_train.columns or col not in X_test.columns:
                    continue
                
                try:
                    # Check for memory before encoding
                    if not self.memory_monitor.check_memory_safety(3.0):
                        logger.warning(f"Skipping target encoding for {col} due to memory constraints")
                        continue
                    
                    # Skip if too many unique values
                    if X_train[col].nunique() > self.config.MAX_CATEGORICAL_UNIQUE:
                        logger.warning(f"Skipping {col}: too many unique values ({X_train[col].nunique()})")
                        continue
                    
                    # Initialize chunk encoder
                    chunk_encoder = ChunkTargetEncoder(
                        chunk_size=self.config.get_optimal_chunk_size() // 2,
                        smoothing=self.config.SMOOTHING_FACTOR
                    )
                    
                    # Fit encoder
                    if chunk_encoder.fit_chunk_based(X_train[col], y_train, col):
                        
                        # Transform training and test data
                        feature_name = f"{col}_target_enc"
                        X_train[feature_name] = chunk_encoder.transform(X_train[col], col)
                        X_test[feature_name] = chunk_encoder.transform(X_test[col], col)
                        
                        # Additional count-based feature
                        value_counts = X_train[col].astype(str).value_counts()
                        count_feature = f"{col}_count"
                        X_train[count_feature] = X_train[col].astype(str).map(value_counts).fillna(0)
                        X_test[count_feature] = X_test[col].astype(str).map(value_counts).fillna(0)
                        
                        self.target_encoding_features.extend([feature_name, count_feature])
                        self.chunk_encoders[col] = chunk_encoder
                        
                        logger.info(f"Target encoding successful for {col}: 2 features created")
                        
                        # Memory cleanup after each encoding
                        if len(self.target_encoding_features) % 10 == 0:
                            self.memory_monitor.force_cleanup()
                    
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Chunk-based target encoding completed: {len(self.target_encoding_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Chunk-based target encoding failed: {e}")
            return X_train, X_test
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create CTR interaction features with memory management"""
        try:
            if self.quick_mode:
                logger.info("Quick mode: Skipping interaction features")
                return X_train, X_test
            
            logger.info("Creating CTR interaction features")
            
            # Select most important features for interactions (limit for memory)
            important_features = self._select_important_features(X_train, y_train, max_features=8)
            
            interaction_count = 0
            max_interactions = 8 if self.memory_efficient_mode else 12
            
            for i in range(len(important_features)):
                for j in range(i + 1, len(important_features)):
                    if interaction_count >= max_interactions:
                        break
                    
                    # Memory check
                    if not self.memory_monitor.check_memory_safety(2.0):
                        logger.warning("Stopping interaction feature creation due to memory constraints")
                        break
                    
                    feat1, feat2 = important_features[i], important_features[j]
                    
                    try:
                        # Multiplication interaction
                        mult_feature = f"{feat1}_x_{feat2}"
                        X_train[mult_feature] = X_train[feat1] * X_train[feat2]
                        X_test[mult_feature] = X_test[feat1] * X_test[feat2]
                        
                        # Addition interaction  
                        add_feature = f"{feat1}_plus_{feat2}"
                        X_train[add_feature] = X_train[feat1] + X_train[feat2]
                        X_test[add_feature] = X_test[feat1] + X_test[feat2]
                        
                        # Ratio interaction (safe division)
                        ratio_feature = f"{feat1}_div_{feat2}"
                        denominator_train = X_train[feat2].replace(0, 1e-8)
                        denominator_test = X_test[feat2].replace(0, 1e-8)
                        X_train[ratio_feature] = X_train[feat1] / denominator_train
                        X_test[ratio_feature] = X_test[feat1] / denominator_test
                        
                        self.interaction_features.extend([mult_feature, add_feature, ratio_feature])
                        interaction_count += 3
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature creation failed for {feat1} x {feat2}: {e}")
                        continue
                
                if not self.memory_monitor.check_memory_safety(2.0):
                    break
            
            logger.info(f"Interaction features completed: {len(self.interaction_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {e}")
            return X_train, X_test
    
    def _select_important_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 10) -> List[str]:
        """Select most important numerical features for interactions"""
        try:
            # Use only numerical features for feature selection
            numerical_cols = [col for col in self.numerical_features if col in X.columns]
            
            if len(numerical_cols) == 0:
                return []
            
            # Limit to prevent memory issues
            if len(numerical_cols) > 50:
                numerical_cols = numerical_cols[:50]
            
            X_num = X[numerical_cols].fillna(0)
            
            # Use SelectKBest with f_classif for binary classification
            selector = SelectKBest(score_func=f_classif, k=min(max_features, len(numerical_cols)))
            selector.fit(X_num, y)
            
            selected_features = [numerical_cols[i] for i in selector.get_support(indices=True)]
            
            return selected_features
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            # Fallback: return first few numerical features
            return self.numerical_features[:max_features]
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create statistical features with memory efficiency"""
        try:
            if self.quick_mode or self.memory_efficient_mode:
                logger.info("Skipping statistical features for memory efficiency")
                return X_train, X_test
            
            logger.info("Creating statistical features")
            
            numerical_cols = [col for col in self.numerical_features if col in X_train.columns][:10]  # Limit columns
            
            if len(numerical_cols) < 2:
                return X_train, X_test
            
            # Row-wise statistics
            X_train_num = X_train[numerical_cols]
            X_test_num = X_test[numerical_cols]
            
            # Mean, std, min, max
            X_train['row_mean'] = X_train_num.mean(axis=1)
            X_test['row_mean'] = X_test_num.mean(axis=1)
            
            X_train['row_std'] = X_train_num.std(axis=1).fillna(0)
            X_test['row_std'] = X_test_num.std(axis=1).fillna(0)
            
            X_train['row_skew'] = X_train_num.skew(axis=1).fillna(0)
            X_test['row_skew'] = X_test_num.skew(axis=1).fillna(0)
            
            self.statistical_features.extend(['row_mean', 'row_std', 'row_skew'])
            
            logger.info(f"Statistical features completed: {len(self.statistical_features)} features created")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Statistical feature creation failed: {e}")
            return X_train, X_test
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Main feature engineering pipeline with memory management and safe data handling"""
        try:
            logger.info("=== CTR Feature engineering started ===")
            self.processing_stats['start_time'] = time.time()
            
            # Check initial memory
            if not self.memory_monitor.check_memory_safety(8.0):
                logger.warning("Low memory detected, enabling memory efficient mode")
                self.memory_efficient_mode = True
            
            # Detect target column and CTR
            target_candidates = ['clicked', 'click', 'target', 'label', 'y']
            self.target_column = None
            
            for col in target_candidates:
                if col in train_df.columns:
                    self.target_column = col
                    ctr = train_df[col].mean()
                    logger.info(f"Target column confirmed: {col} (CTR: {ctr:.4f})")
                    break
            
            if self.target_column is None:
                logger.error("No target column found")
                return None, None
            
            # Create feature datasets (exclude target)
            feature_columns = [col for col in train_df.columns if col != self.target_column]
            X_train = train_df[feature_columns].copy()
            X_test = test_df[feature_columns].copy()
            y_train = train_df[self.target_column].copy()
            
            mode_info = "QUICK MODE" if self.quick_mode else "FULL MODE"
            efficiency_info = "MEMORY EFFICIENT" if self.memory_efficient_mode else "FULL FEATURES"
            logger.info(f"Feature engineering initialization ({mode_info}, {efficiency_info})")
            logger.info(f"Initial data: Training {X_train.shape}, Test {X_test.shape}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Original feature count: {len(feature_columns)}")
            
            # Memory status before processing
            self.memory_monitor.log_memory_status("Initialization")
            
            # Phase 1: Sequence column processing
            logger.info("Phase 1: Sequence column processing")
            X_train, X_test = self._process_seq_columns(X_train, X_test)
            
            # Phase 2: Data preparation and classification
            logger.info("Phase 2: Data preparation")
            
            # Classify columns
            self._classify_columns(X_train)
            
            # Log feature distribution
            logger.info(f"Data preparation completed - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Phase 3: Safe preprocessing
            logger.info("Phase 3: Safe preprocessing")
            X_train, X_test = self._safe_preprocessing(X_train, X_test)
            
            # Phase 4: Feature engineering based on mode
            if self.quick_mode:
                logger.info("Quick mode: Basic feature engineering only")
                # Only do essential target encoding for a few features
                categorical_subset = self.categorical_features[:3]
                for col in categorical_subset:
                    if col in X_train.columns:
                        try:
                            # Simple frequency encoding
                            value_counts = X_train[col].value_counts()
                            freq_feature = f"{col}_freq"
                            X_train[freq_feature] = X_train[col].map(value_counts).fillna(0)
                            X_test[freq_feature] = X_test[col].map(value_counts).fillna(0)
                            self.frequency_features.append(freq_feature)
                        except Exception as e:
                            logger.warning(f"Quick frequency encoding failed for {col}: {e}")
            else:
                # Full feature engineering
                logger.info("Phase 4: Chunk-based target encoding")
                X_train, X_test = self._chunk_target_encoding(X_train, X_test, y_train)
                
                logger.info("Phase 5: Interaction features")
                X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
                
                logger.info("Phase 6: Statistical features")
                X_train, X_test = self._create_statistical_features(X_train, X_test)
            
            # Final cleanup and validation
            logger.info("Phase 7: Final processing")
            
            # Remove any remaining object columns that couldn't be processed
            for col in list(X_train.columns):
                if X_train[col].dtype == 'object':
                    logger.warning(f"Removing unprocessed object column: {col}")
                    X_train = X_train.drop(columns=[col])
                    X_test = X_test.drop(columns=[col])
            
            # Feature count validation
            max_features = self.config.MAX_FEATURES
            if X_train.shape[1] > max_features:
                logger.warning(f"Too many features ({X_train.shape[1]}), selecting top {max_features}")
                # Select features based on variance
                try:
                    feature_variances = X_train.var().fillna(0)
                    top_features = feature_variances.nlargest(max_features).index.tolist()
                    X_train = X_train[top_features]
                    X_test = X_test[top_features]
                except Exception as e:
                    logger.warning(f"Feature selection failed: {e}")
                    # Fallback: take first N features
                    X_train = X_train.iloc[:, :max_features]
                    X_test = X_test.iloc[:, :max_features]
            
            # Final data validation
            X_train = X_train.select_dtypes(include=[np.number])
            X_test = X_test.select_dtypes(include=[np.number])
            
            # Align columns
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            
            # Fill any remaining NaN values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Final memory cleanup
            self.memory_monitor.force_cleanup()
            
            # Update statistics
            self.processing_stats.update({
                'processing_time': time.time() - self.processing_stats['start_time'],
                'total_features_generated': X_train.shape[1] - len(feature_columns),
                'feature_types_count': {
                    'target_encoding': len(self.target_encoding_features),
                    'interaction': len(self.interaction_features),
                    'statistical': len(self.statistical_features),
                    'frequency': len(self.frequency_features),
                    'seq_features': len(self.seq_features)
                }
            })
            
            logger.info(f"=== CTR Feature engineering completed ===")
            logger.info(f"Final features: {X_train.shape[1]} (created: {self.processing_stats['total_features_generated']})")
            logger.info(f"Feature breakdown: Target enc: {len(self.target_encoding_features)}, "
                       f"Interaction: {len(self.interaction_features)}, "
                       f"Seq: {len(self.seq_features)}, "
                       f"Statistical: {len(self.statistical_features)}")
            logger.info(f"Processing time: {self.processing_stats['processing_time']:.2f}s")
            self.memory_monitor.log_memory_status("Feature engineering completed")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return None, None

# Test function
if __name__ == "__main__":
    # Test feature engineering
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
    
    # Test feature engineering
    feature_engineer = CTRFeatureEngineer()
    feature_engineer.set_quick_mode(True)  # Quick test
    
    X_train, X_test = feature_engineer.engineer_features(train_df, test_df)
    
    if X_train is not None and X_test is not None:
        print(f"Feature engineering successful!")
        print(f"Output: Train {X_train.shape}, Test {X_test.shape}")
        print(f"Generated features: {X_train.shape[1] - 4}")
    else:
        print("Feature engineering failed!")