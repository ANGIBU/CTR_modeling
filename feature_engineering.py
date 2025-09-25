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

class CTRFeatureEngineer:
    """CTR feature engineering class with memory efficiency and safe data handling"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.data_converter = SafeDataTypeConverter()
        
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
    
    def _safe_target_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe target encoding for categorical features with proper data handling"""
        try:
            if self.memory_efficient_mode:
                # Limit number of columns for memory efficiency
                categorical_subset = self.categorical_features[:min(15, len(self.categorical_features))]
            else:
                categorical_subset = self.categorical_features
            
            logger.info(f"Target encoding started for {len(categorical_subset)} categorical features")
            
            for col in categorical_subset:
                if col not in X_train.columns or col not in X_test.columns:
                    continue
                
                try:
                    # Check for memory before encoding
                    if not self.memory_monitor.check_memory_safety(2.0):
                        logger.warning(f"Skipping target encoding for {col} due to memory constraints")
                        continue
                    
                    # Convert to consistent string format for encoding
                    train_col_str = X_train[col].astype(str).fillna('missing')
                    test_col_str = X_test[col].astype(str).fillna('missing')
                    
                    # Skip if too many unique values (memory constraint)
                    if train_col_str.nunique() > self.config.MAX_CATEGORICAL_UNIQUE:
                        logger.warning(f"Skipping {col}: too many unique values ({train_col_str.nunique()})")
                        continue
                    
                    # Create temporary DataFrames for encoding
                    train_temp_df = pd.DataFrame({col: train_col_str})
                    test_temp_df = pd.DataFrame({col: test_col_str})
                    
                    # Target encoding with smoothing
                    encoder = TargetEncoder(smooth='auto', target_type='binary')
                    
                    # Fit and transform
                    train_encoded = encoder.fit_transform(train_temp_df, y_train)
                    test_encoded = encoder.transform(test_temp_df)
                    
                    # Create new feature names and add as new columns
                    feature_name = f"{col}_target_enc"
                    X_train[feature_name] = train_encoded[col].values
                    X_test[feature_name] = test_encoded[col].values
                    
                    # Additional frequency-based features
                    value_counts = train_col_str.value_counts()
                    freq_feature = f"{col}_freq"
                    X_train[freq_feature] = train_col_str.map(value_counts).fillna(0)
                    X_test[freq_feature] = test_col_str.map(value_counts).fillna(0)
                    
                    # CTR variance feature (safe calculation)
                    try:
                        ctr_var_feature = f"{col}_ctr_var"
                        # Calculate CTR variance by category
                        category_stats = train_temp_df.assign(target=y_train).groupby(col)['target'].agg(['var', 'count']).fillna(0)
                        # Only use categories with sufficient samples
                        category_stats = category_stats[category_stats['count'] >= 5]
                        ctr_variance_map = category_stats['var'].to_dict()
                        
                        X_train[ctr_var_feature] = train_col_str.map(ctr_variance_map).fillna(0)
                        X_test[ctr_var_feature] = test_col_str.map(ctr_variance_map).fillna(0)
                        
                        self.target_encoding_features.extend([feature_name, freq_feature, ctr_var_feature])
                    except Exception as var_e:
                        logger.warning(f"CTR variance calculation failed for {col}: {var_e}")
                        self.target_encoding_features.extend([feature_name, freq_feature])
                    
                    self.target_encoders[col] = encoder
                    
                    # Memory cleanup after each encoding
                    if len(self.target_encoding_features) % 5 == 0:
                        self.memory_monitor.force_cleanup()
                    
                    logger.info(f"Target encoding successful for {col}: {len([f for f in [feature_name, freq_feature, ctr_var_feature] if f in self.target_encoding_features])} features")
                    
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Target encoding completed: {len(self.target_encoding_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Target encoding features failed: {e}")
            return X_train, X_test
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create CTR-optimized interaction features with memory management"""
        try:
            if self.quick_mode:
                logger.info("Quick mode: Skipping interaction features")
                return X_train, X_test
            
            logger.info("Creating CTR interaction features")
            
            # Select most important features for interactions (limit for memory)
            important_features = self._select_important_features(X_train, y_train, max_features=8)
            
            interaction_count = 0
            max_interactions = 10 if self.memory_efficient_mode else 15
            
            for i in range(len(important_features)):
                for j in range(i + 1, len(important_features)):
                    if interaction_count >= max_interactions:
                        break
                    
                    # Memory check
                    if not self.memory_monitor.check_memory_safety(1.5):
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
                        
                        self.interaction_features.extend([mult_feature, add_feature])
                        interaction_count += 2
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature creation failed for {feat1} x {feat2}: {e}")
                        continue
                
                if not self.memory_monitor.check_memory_safety(1.5):
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
            
            self.statistical_features.extend(['row_mean', 'row_std'])
            
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
            
            # Phase 1: Data preparation and classification
            logger.info("Phase 1: Data preparation")
            
            # Classify columns
            self._classify_columns(X_train)
            
            # Log feature distribution
            logger.info(f"Data preparation completed - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Phase 2: Safe preprocessing
            logger.info("Phase 2: Safe preprocessing")
            X_train, X_test = self._safe_preprocessing(X_train, X_test)
            
            # Phase 3: Feature engineering based on mode
            if self.quick_mode:
                logger.info("Quick mode: Basic feature engineering only")
                # Only do essential target encoding for a few features
                categorical_subset = self.categorical_features[:5]
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
                logger.info("Phase 3: Target encoding")
                X_train, X_test = self._safe_target_encoding(X_train, X_test, y_train)
                
                logger.info("Phase 4: Interaction features")
                X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
                
                logger.info("Phase 5: Statistical features")
                X_train, X_test = self._create_statistical_features(X_train, X_test)
            
            # Final cleanup and validation
            logger.info("Phase 6: Final processing")
            
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
                    'frequency': len(self.frequency_features)
                }
            })
            
            logger.info(f"=== CTR Feature engineering completed ===")
            logger.info(f"Final features: {X_train.shape[1]} (created: {self.processing_stats['total_features_generated']})")
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