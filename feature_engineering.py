# feature_engineering.py

import pandas as pd
import numpy as np
import gc
import time
import logging
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import utilities with memory optimization
try:
    from config import Config, optimize_dataframe_memory, emergency_memory_cleanup
    PSUTIL_AVAILABLE = True
    import psutil
except ImportError:
    PSUTIL_AVAILABLE = False
    class Config:
        CHUNK_SIZE = 5000
        MAX_MEMORY_GB = 20.0
        AGGRESSIVE_GC = True
        SEQ_HASH_SIZE = 10000

class MemoryMonitor:
    """Memory monitoring for feature engineering"""
    
    def __init__(self):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.operation_count = 0
    
    def check_memory_status(self) -> Dict[str, Any]:
        """Check current memory status"""
        self.operation_count += 1
        
        if not self.monitoring_enabled:
            return {'usage_gb': 5.0, 'available_gb': 30.0, 'level': 'normal'}
        
        try:
            process = psutil.Process()
            usage_gb = process.memory_info().rss / (1024**3)
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            if usage_gb > 20 or available_gb < 8:
                level = "critical"
            elif usage_gb > 15 or available_gb < 15:
                level = "warning"
            else:
                level = "normal"
            
            return {
                'usage_gb': usage_gb,
                'available_gb': available_gb,
                'level': level
            }
        except:
            return {'usage_gb': 5.0, 'available_gb': 30.0, 'level': 'normal'}
    
    def cleanup_memory(self):
        """Perform memory cleanup when needed"""
        status = self.check_memory_status()
        if status['level'] in ['warning', 'critical']:
            gc.collect()
            if status['level'] == 'critical':
                emergency_memory_cleanup()
    
    def log_memory_status(self, operation: str = ""):
        """Log current memory status"""
        status = self.check_memory_status()
        logger.info(f"Memory {operation}: {status['usage_gb']:.1f}GB used, {status['available_gb']:.1f}GB available ({status['level']})")

class SafeDataTypeConverter:
    """Safe data type converter for memory optimization"""
    
    @staticmethod
    def convert_to_optimal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Convert dataframe to optimal data types"""
        for col in df.columns:
            try:
                if df[col].dtype == 'int64':
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
                
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                
                elif df[col].dtype == 'object':
                    if df[col].nunique() < len(df) * 0.5:
                        df[col] = df[col].astype('category')
                        
            except:
                continue
        
        return df

class SeqColumnProcessor:
    """Memory-efficient sequence column processor"""
    
    def __init__(self, hash_size: int = 10000):
        self.hash_size = hash_size
        self.is_fitted = False
        self.seq_stats = {}
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Process sequence columns with memory efficiency"""
        X = X.copy()
        
        # Find sequence-like columns
        seq_columns = []
        for col in X.columns:
            if X[col].dtype == 'object':
                sample_vals = X[col].dropna().head(100)
                if any(',' in str(val) or '_' in str(val) for val in sample_vals):
                    seq_columns.append(col)
        
        logger.info(f"Processing {len(seq_columns)} sequence columns")
        
        for col in seq_columns:
            try:
                # Simple length and count features
                X[f'{col}_len'] = X[col].fillna('').astype(str).str.len()
                X[f'{col}_count'] = X[col].fillna('').astype(str).str.count(',') + 1
                
                # Remove original sequence column
                X = X.drop(columns=[col])
                
                # Memory cleanup
                gc.collect()
                
            except Exception as e:
                logger.error(f"Sequence processing failed for {col}: {e}")
                continue
        
        self.is_fitted = True
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform sequence columns"""
        return self.fit_transform(X) if not self.is_fitted else X

class CTRFeatureEngineer:
    """CTR feature engineering with memory optimization"""
    
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
        """Main feature engineering pipeline with memory optimization"""
        
        logger.info("=== CTR Feature Engineering Started (Memory Optimized) ===")
        start_time = time.time()
        
        try:
            self.memory_monitor.log_memory_status("Start")
            
            # Quick mode sampling
            if self.quick_mode:
                train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)
                test_df = test_df.sample(n=min(2000, len(test_df)), random_state=42)
                logger.info(f"Quick mode sampling - Train: {len(train_df)}, Test: {len(test_df)}")
            
            # Emergency sampling if memory critical
            memory_status = self.memory_monitor.check_memory_status()
            if memory_status['level'] == 'critical':
                emergency_sample_size = min(100000, len(train_df))
                train_df = train_df.sample(n=emergency_sample_size, random_state=42)
                test_df = test_df.sample(n=min(50000, len(test_df)), random_state=42)
                logger.warning(f"Emergency sampling - Train: {len(train_df)}, Test: {len(test_df)}")
            
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
            
            # Free original dataframes
            del train_df, test_df
            gc.collect()
            
            # Store original features
            self.original_features = list(X_train.columns)
            initial_ctr = np.mean(y_train)
            
            logger.info(f"Initial data - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Target CTR: {initial_ctr:.4f}, Features: {len(self.original_features)}")
            
            # Phase 1: Data type optimization
            logger.info("Phase 1: Data type optimization")
            X_train = self.data_converter.convert_to_optimal_dtypes(X_train)
            X_test = self.data_converter.convert_to_optimal_dtypes(X_test)
            
            self.memory_monitor.log_memory_status("After optimization")
            
            # Phase 2: Column classification
            logger.info("Phase 2: Column classification")
            self._classify_columns(X_train)
            
            # Phase 3: Sequence column processing
            logger.info("Phase 3: Sequence column processing")
            X_train = self.seq_processor.fit_transform(X_train, y_train)
            X_test = self.seq_processor.transform(X_test)
            
            # Update column classification after sequence processing
            self._classify_columns(X_train)
            
            self.memory_monitor.log_memory_status("After sequence processing")
            
            # Phase 4: Feature engineering
            logger.info("Phase 4: Feature engineering")
            if self.quick_mode:
                X_train, X_test = self._quick_feature_engineering(X_train, X_test, y_train)
            else:
                X_train, X_test = self._memory_safe_feature_engineering(X_train, X_test, y_train)
            
            # Phase 5: Feature selection
            logger.info("Phase 5: Feature selection")
            try:
                X_train, X_test = self._feature_selection(X_train, X_test, y_train)
            except Exception as e:
                logger.error(f"Feature selection failed: {e}")
            
            # Phase 6: Final preprocessing
            logger.info("Phase 6: Final preprocessing")
            X_train, X_test = self._final_preprocessing(X_train, X_test)
            
            # Final optimization
            X_train = self.data_converter.convert_to_optimal_dtypes(X_train)
            X_test = self.data_converter.convert_to_optimal_dtypes(X_test)
            
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
                if X[col].nunique() > 20:
                    self.numerical_features.append(col)
                else:
                    self.categorical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        logger.info(f"Columns classified - Numerical: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
    
    def _quick_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Quick feature engineering for fast processing"""
        
        # Only essential categorical encoding
        for col in X_train.select_dtypes(include=['object', 'category']).columns[:5]:
            try:
                le = LabelEncoder()
                combined_values = pd.concat([X_train[col], X_test[col]], ignore_index=True)
                le.fit(combined_values.fillna('missing'))
                
                X_train[col] = le.transform(X_train[col].fillna('missing'))
                X_test[col] = le.transform(X_test[col].fillna('missing'))
                
                self.label_encoders[col] = le
                
            except:
                continue
        
        return X_train, X_test
    
    def _memory_safe_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-safe feature engineering"""
        
        # Limit feature generation to prevent memory issues
        max_numerical = min(len(self.numerical_features), 20)
        max_categorical = min(len(self.categorical_features), 10)
        
        # Process numerical features (limited)
        numerical_cols = self.numerical_features[:max_numerical]
        
        for col in numerical_cols:
            try:
                self.memory_monitor.cleanup_memory()
                
                # Simple log transformation
                X_train[f'{col}_log'] = np.log1p(X_train[col].clip(lower=0))
                X_test[f'{col}_log'] = np.log1p(X_test[col].clip(lower=0))
                
                self.generated_features.append(f'{col}_log')
                
                # Memory check
                if self.memory_monitor.check_memory_status()['level'] == 'critical':
                    logger.warning("Critical memory reached, stopping feature generation")
                    break
                
            except Exception as e:
                logger.error(f"Feature generation failed for {col}: {e}")
                continue
        
        # Process categorical features (limited)
        categorical_cols = [col for col in self.categorical_features if col in X_train.columns][:max_categorical]
        
        for col in categorical_cols:
            try:
                self.memory_monitor.cleanup_memory()
                
                # Skip high cardinality columns
                if X_train[col].nunique() > 100:
                    continue
                
                le = LabelEncoder()
                combined_values = pd.concat([X_train[col], X_test[col]], ignore_index=True)
                le.fit(combined_values.fillna('missing'))
                
                X_train[f'{col}_encoded'] = le.transform(X_train[col].fillna('missing'))
                X_test[f'{col}_encoded'] = le.transform(X_test[col].fillna('missing'))
                
                self.label_encoders[col] = le
                self.generated_features.append(f'{col}_encoded')
                
                # Remove original categorical column
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
                
                del combined_values
                gc.collect()
                
            except Exception as e:
                logger.error(f"Categorical encoding failed for {col}: {e}")
                continue
        
        logger.info(f"Generated {len(self.generated_features)} features")
        return X_train, X_test
    
    def _feature_selection(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-efficient feature selection"""
        
        try:
            # Limit features for memory efficiency
            max_features = min(X_train.shape[1], 200)
            
            if X_train.shape[1] > max_features:
                # Simple variance-based selection
                variances = X_train.var()
                selected_features = variances.nlargest(max_features).index.tolist()
                
                X_train = X_train[selected_features]
                X_test = X_test[selected_features]
                
                self.selected_features = selected_features
                logger.info(f"Feature selection: {len(selected_features)} features selected")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return X_train, X_test
    
    def _final_preprocessing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final preprocessing with memory safety"""
        
        try:
            # Fill NaN values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Ensure consistent columns
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            
            logger.info(f"Final preprocessing completed - Features: {len(common_columns)}")
            
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
    test_df = test_df.drop(columns=['clicked'])
    
    print(f"Sample data: Train {train_df.shape}, Test {test_df.shape}")
    print(f"Target CTR: {np.mean(train_df['clicked']):.4f}")
    
    # Test feature engineering
    feature_engineer = CTRFeatureEngineer()
    feature_engineer.set_quick_mode(True)
    
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