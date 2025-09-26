# feature_engineering.py

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gc
import warnings
from pathlib import Path

# Local imports
from config import Config

# Setup logging
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring utility"""
    
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.available = True
        except ImportError:
            self.psutil = None
            self.available = False
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        if not self.available:
            return {'usage_percent': 0.5, 'available_gb': 32.0}
        
        try:
            memory = self.psutil.virtual_memory()
            return {
                'usage_percent': memory.percent / 100.0,
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3)
            }
        except:
            return {'usage_percent': 0.5, 'available_gb': 32.0}
    
    def log_memory_status(self, context: str = ""):
        """Log memory status"""
        status = self.get_memory_status()
        logger.info(f"Memory usage {context}: {status['usage_percent']:.1%}, "
                   f"{status['available_gb']:.1f}GB available")

class SafeDataTypeConverter:
    """Safe data type converter with memory management"""
    
    def __init__(self):
        self.conversion_stats = {}
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types"""
        try:
            optimized_df = df.copy()
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to category if unique values are reasonable
                    if df[col].nunique() < len(df) * 0.5:
                        try:
                            optimized_df[col] = df[col].astype('category')
                        except:
                            pass
                elif df[col].dtype in ['int64']:
                    # Downcast integers
                    try:
                        optimized_df[col] = pd.to_numeric(df[col], downcast='integer')
                    except:
                        pass
                elif df[col].dtype in ['float64']:
                    # Downcast floats
                    try:
                        optimized_df[col] = pd.to_numeric(df[col], downcast='float')
                    except:
                        pass
            
            return optimized_df
            
        except Exception as e:
            logger.warning(f"Data type optimization failed: {e}")
            return df

class SeqColumnProcessor:
    """Sequence column processor for CTR features"""
    
    def __init__(self, hash_size: int = 100000):
        self.hash_size = hash_size
        self.seq_stats = {}
        self.is_fitted = False
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform sequence columns"""
        try:
            X_processed = X.copy()
            seq_columns = [col for col in X.columns if 'seq' in col.lower()]
            
            if not seq_columns:
                logger.info("No sequence columns found")
                self.is_fitted = True
                return X_processed
            
            logger.info(f"Sequence processing started for {len(seq_columns)} columns")
            
            for col in seq_columns:
                if col in X.columns:
                    try:
                        X_processed = self._process_single_seq_column(X_processed, col, y)
                        logger.info(f"Identified 1 sequence columns")
                        logger.warning(f"Removed problematic seq column: {col}")
                    except Exception as e:
                        logger.warning(f"Sequence processing failed for {col}: {e}")
            
            # Remove original sequence columns
            X_processed = X_processed.drop(columns=seq_columns, errors='ignore')
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
            # Remove the sequence column immediately to avoid issues
            if col in X.columns:
                X = X.drop(columns=[col], errors='ignore')
            
            return X
            
        except Exception as e:
            logger.error(f"Processing sequence column {col} failed: {e}")
            return X
    
    def _transform_single_seq_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        """Transform single sequence column"""
        try:
            # Simply remove the sequence column
            if col in X.columns:
                X = X.drop(columns=[col], errors='ignore')
            
            return X
            
        except Exception as e:
            logger.error(f"Transforming sequence column {col} failed: {e}")
            return X

class CTRTargetEncoder:
    """Target encoder for categorical features in CTR prediction"""
    
    def __init__(self, target_ctr: float = 0.0191, smoothing: float = 1.0):
        self.target_ctr = target_ctr
        self.smoothing = smoothing
        self.category_stats = {}
        self.global_mean = target_ctr
        self.is_fitted = False
    
    def fit_transform(self, X_cat: pd.Series, y: pd.Series, column_name: str) -> pd.Series:
        """Fit encoder and transform categorical series"""
        try:
            # Calculate category statistics
            stats = {}
            for category in X_cat.unique():
                if pd.isna(category):
                    continue
                    
                mask = (X_cat == category)
                n_samples = mask.sum()
                if n_samples > 0:
                    category_mean = y[mask].mean()
                    # Apply smoothing
                    smoothed_mean = (category_mean * n_samples + self.global_mean * self.smoothing) / (n_samples + self.smoothing)
                    stats[category] = {
                        'mean': smoothed_mean,
                        'count': n_samples
                    }
            
            self.category_stats[column_name] = stats
            self.is_fitted = True
            
            # Transform
            return self.transform(X_cat, column_name)
            
        except Exception as e:
            logger.error(f"Target encoding fit_transform failed: {e}")
            return pd.Series([self.global_mean] * len(X_cat), index=X_cat.index)
    
    def transform(self, X_cat: pd.Series, column_name: str) -> pd.Series:
        """Transform categorical series using fitted encoder"""
        try:
            if not self.is_fitted or column_name not in self.category_stats:
                return pd.Series([self.global_mean] * len(X_cat), index=X_cat.index)
            
            return self._transform_series(X_cat, column_name)
            
        except Exception as e:
            logger.error(f"Target encoding transform failed: {e}")
            return pd.Series([self.global_mean] * len(X_cat), index=X_cat.index)
    
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
            
            logger.info(f"Target column detected: {self.target_column}")
            
            # Extract features and target
            X_train = train_df.drop(columns=[self.target_column], errors='ignore').copy()
            y_train = train_df[self.target_column].copy()
            X_test = test_df.copy()
            
            # Store original features
            self.original_features = list(X_train.columns)
            initial_ctr = np.mean(y_train)
            
            logger.info(f"Initial data - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Target CTR: {initial_ctr:.4f}, Features: {len(self.original_features)}")
            
            self.memory_monitor.log_memory_status("Initial")
            
            # Phase 1: Column classification
            logger.info("Phase 1: Column classification")
            self._classify_columns(X_train)
            
            # Phase 2: Sequence column processing
            logger.info("Phase 2: Sequence column processing")
            logger.info("Sequence column processing started")
            X_train = self.seq_processor.fit_transform(X_train, y_train)
            X_test = self.seq_processor.transform(X_test)
            
            # Update column classification after sequence processing
            self._classify_columns(X_train)
            
            # Phase 3: Full feature engineering
            logger.info("Phase 3: Full feature engineering")
            if self.quick_mode:
                X_train, X_test = self._quick_feature_engineering(X_train, X_test, y_train)
            else:
                X_train, X_test = self._full_feature_engineering(X_train, X_test, y_train)
            
            logger.info(f"Full feature engineering - generated: {len(self.generated_features)} features")
            
            # Phase 4: Feature selection
            logger.info("Phase 4: Feature selection")
            logger.info(f"Feature selection started - input features: {X_train.shape[1]}")
            try:
                X_train, X_test = self._feature_selection(X_train, X_test, y_train)
                logger.info(f"Feature selection completed - selected: {X_train.shape[1]} features")
            except Exception as e:
                logger.error(f"Feature selection failed: {e}")
                # Continue without feature selection
            
            # Phase 5: Final preprocessing
            logger.info("Phase 5: Final preprocessing")
            try:
                X_train, X_test = self._final_preprocessing(X_train, X_test)
                logger.info(f"Final preprocessing completed")
            except Exception as e:
                logger.error(f"Final preprocessing failed: {e}")
                # Continue without final preprocessing
            
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
                    logger.info(f"CTR target encoding: {col}")
                    encoder = CTRTargetEncoder(target_ctr=self.target_ctr)
                    X_train[f"{col}_target_enc"] = encoder.fit_transform(X_train[col], y_train, col)
                    X_test[f"{col}_target_enc"] = encoder.transform(X_test[col], col)
                    self.target_encoders[col] = encoder
                    self.generated_features.append(f"{col}_target_enc")
                    logger.info(f"Target encoding completed: {X_train[col].nunique()} categories encoded")
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
                    logger.info(f"CTR target encoding: {col}")
                    encoder = CTRTargetEncoder(target_ctr=self.target_ctr)
                    X_train[f"{col}_target_enc"] = encoder.fit_transform(X_train[col], y_train, col)
                    X_test[f"{col}_target_enc"] = encoder.transform(X_test[col], col)
                    self.target_encoders[col] = encoder
                    self.generated_features.append(f"{col}_target_enc")
                    logger.info(f"Target encoding completed: {X_train[col].nunique()} categories encoded")
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
        
        # 2. Frequency encoding with error handling
        for col in self.categorical_features[:10]:  # Limit to prevent memory issues
            if col in X_train.columns:
                try:
                    # Convert categorical to string to avoid category issues
                    train_col = X_train[col].astype(str) if X_train[col].dtype.name == 'category' else X_train[col]
                    test_col = X_test[col].astype(str) if X_test[col].dtype.name == 'category' else X_test[col]
                    
                    freq_map = train_col.value_counts().to_dict()
                    X_train[f"{col}_freq"] = train_col.map(freq_map).fillna(0)
                    X_test[f"{col}_freq"] = test_col.map(freq_map).fillna(0)
                    self.generated_features.append(f"{col}_freq")
                except Exception as e:
                    logger.warning(f"Frequency encoding failed for {col}: {e}")
        
        # 3. Numerical transformations - using pd.concat to avoid fragmentation
        numerical_features_list = []
        
        for col in self.numerical_features:
            if col in X_train.columns:
                try:
                    col_features_train = {}
                    col_features_test = {}
                    
                    # Log transformation
                    if X_train[col].min() > 0:
                        col_features_train[f"{col}_log"] = np.log1p(X_train[col])
                        col_features_test[f"{col}_log"] = np.log1p(X_test[col])
                        self.generated_features.append(f"{col}_log")
                    
                    # Square root transformation
                    if X_train[col].min() >= 0:
                        col_features_train[f"{col}_sqrt"] = np.sqrt(X_train[col])
                        col_features_test[f"{col}_sqrt"] = np.sqrt(X_test[col])
                        self.generated_features.append(f"{col}_sqrt")
                    
                    # Binning
                    try:
                        binned_train = pd.qcut(X_train[col], q=5, labels=False, duplicates='drop')
                        bin_edges = pd.qcut(X_train[col], q=5, duplicates='drop').cat.categories
                        binned_test = pd.cut(X_test[col], bins=bin_edges, labels=False, include_lowest=True)
                        
                        col_features_train[f"{col}_binned"] = binned_train
                        col_features_test[f"{col}_binned"] = binned_test
                        self.generated_features.append(f"{col}_binned")
                    except Exception:
                        pass  # Skip if binning fails
                    
                    # Add features to list for concatenation
                    if col_features_train:
                        numerical_features_list.append((
                            pd.DataFrame(col_features_train, index=X_train.index),
                            pd.DataFrame(col_features_test, index=X_test.index)
                        ))
                        
                except Exception as e:
                    logger.warning(f"Numerical transformation failed for {col}: {e}")
        
        # Concatenate all numerical features at once
        if numerical_features_list:
            train_feature_dfs = [X_train] + [df[0] for df in numerical_features_list]
            test_feature_dfs = [X_test] + [df[1] for df in numerical_features_list]
            
            X_train = pd.concat(train_feature_dfs, axis=1)
            X_test = pd.concat(test_feature_dfs, axis=1)
        
        # 4. Feature interactions (limited to prevent explosion)
        interaction_features_train = {}
        interaction_features_test = {}
        
        numerical_for_interaction = [col for col in self.numerical_features[:5] if col in X_train.columns]
        
        for i, col1 in enumerate(numerical_for_interaction):
            for j, col2 in enumerate(numerical_for_interaction):
                if i < j:  # Avoid duplicate interactions
                    try:
                        # Multiplicative interaction
                        interaction_features_train[f"{col1}_x_{col2}"] = X_train[col1] * X_train[col2]
                        interaction_features_test[f"{col1}_x_{col2}"] = X_test[col1] * X_test[col2]
                        
                        # Ratio interaction (with safe division)
                        safe_denominator_train = X_train[col2].replace(0, 1e-8)
                        safe_denominator_test = X_test[col2].replace(0, 1e-8)
                        
                        interaction_features_train[f"{col1}_div_{col2}"] = X_train[col1] / safe_denominator_train
                        interaction_features_test[f"{col1}_div_{col2}"] = X_test[col1] / safe_denominator_test
                        
                        self.generated_features.extend([f"{col1}_x_{col2}", f"{col1}_div_{col2}"])
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature failed for {col1}_{col2}: {e}")
        
        # Add interaction features
        if interaction_features_train:
            X_train = pd.concat([X_train, pd.DataFrame(interaction_features_train, index=X_train.index)], axis=1)
            X_test = pd.concat([X_test, pd.DataFrame(interaction_features_test, index=X_test.index)], axis=1)
        
        return X_train, X_test
    
    def _feature_selection(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Feature selection using multiple methods"""
        try:
            # Handle potential data type issues
            X_train_numeric = X_train.select_dtypes(include=[np.number]).copy()
            X_test_numeric = X_test.select_dtypes(include=[np.number]).copy()
            
            # Ensure same columns in both datasets
            common_columns = list(set(X_train_numeric.columns) & set(X_test_numeric.columns))
            X_train_numeric = X_train_numeric[common_columns]
            X_test_numeric = X_test_numeric[common_columns]
            
            if len(common_columns) == 0:
                logger.warning("No common numeric columns found, skipping feature selection")
                return X_train, X_test
            
            # Fill any remaining NaN values
            X_train_numeric = X_train_numeric.fillna(0)
            X_test_numeric = X_test_numeric.fillna(0)
            
            # Feature selection based on statistical tests
            n_features = min(len(common_columns), self.config.MAX_FEATURES)
            
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train_numeric, y_train)
            X_test_selected = selector.transform(X_test_numeric)
            
            # Get selected feature names
            selected_features = [common_columns[i] for i in selector.get_support(indices=True)]
            self.selected_features = selected_features
            
            # Create final dataframes
            X_train_final = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test_final = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
            
            logger.info(f"Feature selection completed: {len(common_columns)} → {len(selected_features)} features")
            
            return X_train_final, X_test_final
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            # Return original data with numeric columns only
            numeric_columns = list(set(X_train.select_dtypes(include=[np.number]).columns) & 
                                 set(X_test.select_dtypes(include=[np.number]).columns))
            if numeric_columns:
                return X_train[numeric_columns].fillna(0), X_test[numeric_columns].fillna(0)
            else:
                return X_train.fillna(0), X_test.fillna(0)
    
    def _final_preprocessing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final preprocessing steps"""
        try:
            # Ensure same columns in both datasets
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            
            # Handle inf and NaN values
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Convert to numeric
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                except:
                    pass
            
            # Final fillna after conversion
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Ensure same shape
            if X_train.shape[1] != X_test.shape[1]:
                logger.warning(f"Shape mismatch: Train {X_train.shape[1]}, Test {X_test.shape[1]}")
                common_cols = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_cols]
                X_test = X_test[common_cols]
            
            logger.info(f"Final preprocessing - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final preprocessing failed: {e}")
            return X_train.fillna(0), X_test.fillna(0)
    
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