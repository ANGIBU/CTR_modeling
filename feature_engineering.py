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

# Import utilities
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

class MemoryMonitor:
    """Lightweight memory monitoring for feature engineering"""
    
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
            
            # Determine memory level
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
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        if Config.AGGRESSIVE_GC and self.operation_count % Config.GC_FREQUENCY == 0:
            return True
        
        status = self.check_memory_status()
        return status['level'] in ['warning', 'critical']
    
    def cleanup_memory(self):
        """Perform memory cleanup"""
        if self.should_cleanup():
            gc.collect()
            if self.check_memory_status()['level'] == 'critical':
                emergency_memory_cleanup()
    
    def log_memory_status(self, operation: str = ""):
        """Log current memory status"""
        status = self.check_memory_status()
        logger.info(f"Memory {operation}: {status['usage_gb']:.1f}GB used, {status['available_gb']:.1f}GB available ({status['level']})")

class SafeDataTypeConverter:
    """Safe data type converter for memory optimization"""
    
    @staticmethod
    def optimize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize numeric column data types"""
        for col in df.select_dtypes(include=[np.number]).columns:
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
            except:
                continue
        
        return df
    
    @staticmethod
    def optimize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize categorical column data types"""
        for col in df.select_dtypes(include=['object']).columns:
            try:
                if df[col].nunique() < len(df) * 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
            except:
                continue
        
        return df

class ChunkedFeatureProcessor:
    """Memory-efficient chunked feature processing"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
    
    def process_chunks(self, df: pd.DataFrame, operation_func, **kwargs) -> pd.DataFrame:
        """Process dataframe in memory-safe chunks"""
        chunk_size = self.config.get_safe_chunk_size()
        results = []
        
        logger.info(f"Processing {len(df)} rows in chunks of {chunk_size}")
        
        for i in range(0, len(df), chunk_size):
            self.memory_monitor.cleanup_memory()
            
            chunk = df.iloc[i:i+chunk_size].copy()
            
            try:
                processed_chunk = operation_func(chunk, **kwargs)
                results.append(processed_chunk)
                
                # Clean up chunk from memory
                del chunk
                
                if self.memory_monitor.check_memory_status()['level'] == 'critical':
                    logger.warning(f"Critical memory at chunk {i//chunk_size + 1}")
                    emergency_memory_cleanup()
                
            except Exception as e:
                logger.error(f"Chunk processing failed at {i}: {e}")
                continue
        
        if results:
            final_result = pd.concat(results, ignore_index=True)
            del results
            gc.collect()
            return final_result
        
        return df

class MinimalSequenceProcessor:
    """Minimal memory-efficient sequence processing"""
    
    def __init__(self, hash_size: int = 5000):
        self.hash_size = hash_size
        self.seq_encoder = {}
        self.fitted = False
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform sequence columns with minimal memory usage"""
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
                # Simple hash-based encoding for sequences
                X[f'{col}_len'] = X[col].fillna('').astype(str).str.len()
                X[f'{col}_count'] = X[col].fillna('').astype(str).str.count(',') + 1
                
                # Remove original sequence column to save memory
                X = X.drop(columns=[col])
                
                # Memory cleanup
                gc.collect()
                
            except Exception as e:
                logger.error(f"Sequence processing failed for {col}: {e}")
                continue
        
        self.fitted = True
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform sequence columns"""
        return self.fit_transform(X) if not self.fitted else X

class MemoryEfficientCTRFeatureEngineer:
    """Memory-efficient CTR feature engineering"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.data_converter = SafeDataTypeConverter()
        self.chunked_processor = ChunkedFeatureProcessor(config)
        self.seq_processor = MinimalSequenceProcessor(hash_size=config.SEQ_HASH_SIZE)
        
        # CTR modeling settings
        self.target_column = 'clicked'
        self.quick_mode = False
        
        # Feature tracking
        self.numerical_features = []
        self.categorical_features = []
        self.generated_features = []
        
        # Preprocessing components
        self.label_encoders = {}
        self.scalers = {}
        
    def set_quick_mode(self, quick_mode: bool = True):
        """Enable quick mode for testing"""
        self.quick_mode = quick_mode
        logger.info(f"Quick mode: {'ENABLED' if quick_mode else 'DISABLED'}")
    
    def engineer_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Memory-efficient feature engineering pipeline"""
        
        logger.info("=== Memory-Efficient CTR Feature Engineering ===")
        start_time = time.time()
        
        try:
            self.memory_monitor.log_memory_status("Start")
            
            # Quick mode sampling for testing
            if self.quick_mode:
                train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)
                test_df = test_df.sample(n=min(2000, len(test_df)), random_state=42)
                logger.info(f"Quick mode sampling - Train: {len(train_df)}, Test: {len(test_df)}")
            
            # Memory check and emergency sampling if needed
            memory_status = self.memory_monitor.check_memory_status()
            if memory_status['level'] == 'critical':
                emergency_sample_size = min(100000, len(train_df))
                train_df = train_df.sample(n=emergency_sample_size, random_state=42)
                test_df = test_df.sample(n=min(50000, len(test_df)), random_state=42)
                logger.warning(f"Emergency sampling applied - Train: {len(train_df)}, Test: {len(test_df)}")
            
            # Extract target and features
            if self.target_column not in train_df.columns:
                possible_targets = ['clicked', 'click', 'target', 'label']
                for col in possible_targets:
                    if col in train_df.columns:
                        self.target_column = col
                        break
            
            logger.info(f"Target column: {self.target_column}")
            
            # Create feature sets
            X_train = train_df.drop(columns=[self.target_column], errors='ignore').copy()
            y_train = train_df[self.target_column].copy()
            X_test = test_df.copy()
            
            # Free original dataframes
            del train_df, test_df
            gc.collect()
            
            logger.info(f"Initial shapes - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Target CTR: {np.mean(y_train):.4f}")
            
            # Phase 1: Data type optimization
            logger.info("Phase 1: Data type optimization")
            X_train = self.data_converter.optimize_numeric_columns(X_train)
            X_train = self.data_converter.optimize_categorical_columns(X_train)
            X_test = self.data_converter.optimize_numeric_columns(X_test)
            X_test = self.data_converter.optimize_categorical_columns(X_test)
            
            self.memory_monitor.log_memory_status("After optimization")
            
            # Phase 2: Basic feature classification
            logger.info("Phase 2: Feature classification")
            self._classify_columns_minimal(X_train)
            
            # Phase 3: Sequence processing (minimal)
            logger.info("Phase 3: Minimal sequence processing")
            X_train = self.seq_processor.fit_transform(X_train, y_train)
            X_test = self.seq_processor.transform(X_test)
            
            self.memory_monitor.log_memory_status("After sequence processing")
            
            # Phase 4: Memory-safe feature engineering
            logger.info("Phase 4: Memory-safe feature engineering")
            X_train, X_test = self._memory_safe_feature_engineering(X_train, X_test, y_train)
            
            # Phase 5: Final preprocessing
            logger.info("Phase 5: Final preprocessing")
            X_train, X_test = self._final_memory_safe_preprocessing(X_train, X_test)
            
            # Final memory optimization
            X_train = self.data_converter.optimize_numeric_columns(X_train)
            X_test = self.data_converter.optimize_numeric_columns(X_test)
            
            # Final cleanup
            gc.collect()
            
            processing_time = time.time() - start_time
            logger.info("=== Feature Engineering Completed ===")
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
            
            self.memory_monitor.log_memory_status("Final")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            emergency_memory_cleanup()
            return None, None
    
    def _classify_columns_minimal(self, X: pd.DataFrame):
        """Minimal column classification"""
        self.numerical_features = []
        self.categorical_features = []
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        logger.info(f"Classified - Numerical: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
    
    def _memory_safe_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Memory-safe feature engineering with chunked processing"""
        
        if self.quick_mode:
            return self._minimal_feature_engineering(X_train, X_test, y_train)
        
        # Limit features to prevent memory issues
        max_features = min(len(X_train.columns), 100)  # Hard limit
        
        # Basic statistical features for numerical columns
        numerical_cols = [col for col in self.numerical_features if col in X_train.columns][:20]  # Limit to 20
        
        for col in numerical_cols:
            try:
                self.memory_monitor.cleanup_memory()
                
                # Simple statistical features
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
        
        # Basic categorical encoding (limit to top categories)
        categorical_cols = [col for col in self.categorical_features if col in X_train.columns][:10]  # Limit to 10
        
        for col in categorical_cols:
            try:
                self.memory_monitor.cleanup_memory()
                
                # Simple label encoding with frequency limit
                unique_values = X_train[col].nunique()
                if unique_values > 100:  # Skip high cardinality columns
                    continue
                
                le = LabelEncoder()
                
                # Combine train and test for consistent encoding
                combined_values = pd.concat([X_train[col], X_test[col]], ignore_index=True)
                le.fit(combined_values.fillna('missing'))
                
                X_train[f'{col}_encoded'] = le.transform(X_train[col].fillna('missing'))
                X_test[f'{col}_encoded'] = le.transform(X_test[col].fillna('missing'))
                
                self.label_encoders[col] = le
                self.generated_features.append(f'{col}_encoded')
                
                # Remove original categorical column to save memory
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
                
                del combined_values
                gc.collect()
                
            except Exception as e:
                logger.error(f"Categorical encoding failed for {col}: {e}")
                continue
        
        logger.info(f"Generated {len(self.generated_features)} features")
        return X_train, X_test
    
    def _minimal_feature_engineering(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Minimal feature engineering for quick mode"""
        
        # Only essential transformations
        for col in X_train.select_dtypes(include=['object']).columns[:5]:  # Limit to 5 columns
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
    
    def _final_memory_safe_preprocessing(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final memory-safe preprocessing"""
        
        try:
            # Fill NaN values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Ensure consistent columns
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            
            # Final data type optimization
            X_train = self.data_converter.optimize_numeric_columns(X_train)
            X_test = self.data_converter.optimize_numeric_columns(X_test)
            
            logger.info(f"Final preprocessing completed - Features: {len(common_columns)}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final preprocessing failed: {e}")
            return X_train.fillna(0), X_test.fillna(0)

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Memory-Efficient CTR Feature Engineering Test")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    
    sample_data = pd.DataFrame({
        'feature_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_2': np.random.randint(1, 100, n_samples),
        'feature_3': np.random.randn(n_samples),
        'sequence_col': [f"item_{i%5},item_{(i+1)%5}" for i in range(n_samples)],
        'clicked': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    # Split data
    train_df = sample_data.iloc[:1600].copy()
    test_df = sample_data.iloc[1600:].copy()
    test_df = test_df.drop(columns=['clicked'])
    
    print(f"Sample data - Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Target CTR: {np.mean(train_df['clicked']):.4f}")
    
    # Test feature engineering
    feature_engineer = MemoryEfficientCTRFeatureEngineer()
    feature_engineer.set_quick_mode(True)
    
    X_train, X_test = feature_engineer.engineer_features(train_df, test_df)
    
    if X_train is not None and X_test is not None:
        print(f"Success! Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Generated features: {len(feature_engineer.generated_features)}")
        print(f"Memory usage: {feature_engineer.memory_monitor.check_memory_status()['usage_gb']:.1f}GB")
    else:
        print("Feature engineering failed!")