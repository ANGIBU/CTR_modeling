# data_loader.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
import gc
import warnings
import os
import pickle
import tempfile
from pathlib import Path
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("PyArrow not available.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Memory monitoring for data loading operations
    Tracks system memory usage and provides optimization recommendations
    """
    
    def __init__(self):
        # Memory thresholds for different alert levels (in GB)
        self.memory_thresholds = {
            'warning': 10.0,    # Issue warning below 10GB
            'critical': 5.0,    # Critical state below 5GB
            'abort': 2.0        # Abort operations below 2GB
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
        """Get detailed memory status information"""
        if not PSUTIL_AVAILABLE:
            return {
                'available_gb': 64.0,
                'used_gb': 16.0,
                'total_gb': 80.0,
                'percent': 20.0,
                'should_cleanup': False,
                'should_abort': False,
                'level': 'unknown'
            }
        
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024**3)
        used_gb = vm.used / (1024**3)
        total_gb = vm.total / (1024**3)
        
        return {
            'available_gb': available_gb,
            'used_gb': used_gb,
            'total_gb': total_gb,
            'percent': vm.percent,
            'should_cleanup': available_gb < self.memory_thresholds['warning'],
            'should_abort': available_gb < self.memory_thresholds['abort'],
            'level': self._get_memory_level(available_gb)
        }
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """
        Check current memory pressure and recommend actions
        
        Returns:
            Dict containing memory status and recommended actions
        """
        if not PSUTIL_AVAILABLE:
            return {
                'available_gb': 64.0,
                'should_cleanup': False,
                'should_abort': False,
                'level': 'unknown'
            }
        
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024**3)
        
        return {
            'available_gb': available_gb,
            'should_cleanup': available_gb < self.memory_thresholds['warning'],
            'should_abort': available_gb < self.memory_thresholds['abort'],
            'level': self._get_memory_level(available_gb)
        }
    
    def _get_memory_level(self, available_gb: float) -> str:
        """Determine memory pressure level"""
        if available_gb < self.memory_thresholds['abort']:
            return 'abort'
        elif available_gb < self.memory_thresholds['critical']:
            return 'critical'
        elif available_gb < self.memory_thresholds['warning']:
            return 'warning'
        else:
            return 'normal'
    
    def log_memory_status(self, context: str, force: bool = False):
        """Log current memory status with context"""
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            if force or vm.percent > 80:  # Only log if memory usage > 80% or forced
                logger.info(f"Memory status ({context}): {vm.percent:.1f}% used, "
                           f"{vm.available/(1024**3):.1f}GB available")
    
    def force_memory_cleanup(self, intensive: bool = False):
        """Force garbage collection and memory cleanup"""
        try:
            collected = gc.collect()
            
            if intensive:
                # Multiple rounds of cleanup
                for _ in range(3):
                    collected += gc.collect()
                    time.sleep(0.1)
            
            return collected
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0

class DataColumnAnalyzer:
    """
    Analyzes data columns to detect target columns and data characteristics
    Optimized for CTR prediction tasks
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_config = config.TARGET_DETECTION_CONFIG
        self.target_candidates = config.TARGET_COLUMN_CANDIDATES
    
    def detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect the target column for CTR prediction
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Name of detected target column or None
        """
        try:
            logger.info("Target column detection started")
            
            # Check for exact matches first
            for candidate in self.target_candidates:
                if candidate in df.columns:
                    if self._validate_target_column(df[candidate]):
                        logger.info(f"Target column detected: {candidate}")
                        return candidate
            
            # Check for partial matches
            for col in df.columns:
                col_lower = col.lower()
                for candidate in self.target_candidates:
                    if candidate.lower() in col_lower:
                        if self._validate_target_column(df[col]):
                            logger.info(f"Target column detected (partial match): {col}")
                            return col
            
            # Analyze all binary columns
            binary_columns = []
            for col in df.columns:
                if self._is_binary_column(df[col]):
                    binary_columns.append(col)
            
            # Select best binary column based on CTR characteristics
            if binary_columns:
                best_col = self._select_best_ctr_column(df, binary_columns)
                if best_col:
                    logger.info(f"Target column detected (binary analysis): {best_col}")
                    return best_col
            
            logger.warning("No suitable target column found")
            return None
            
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return None
    
    def _is_binary_column(self, series: pd.Series) -> bool:
        """Check if column contains binary values"""
        try:
            unique_values = set(series.dropna().unique())
            return unique_values.issubset(self.target_config['binary_values'])
        except Exception:
            return False
    
    def _validate_target_column(self, series: pd.Series) -> bool:
        """Validate if column is suitable as CTR target"""
        try:
            if not self._is_binary_column(series):
                return False
            
            # Calculate CTR (positive rate)
            ctr = series.mean()
            
            # Check if CTR is within expected range
            min_ctr = self.target_config['min_ctr']
            max_ctr = self.target_config['max_ctr']
            
            if not (min_ctr <= ctr <= max_ctr):
                return False
            
            # Prefer CTR values in typical range
            typical_range = self.target_config['typical_ctr_range']
            if typical_range[0] <= ctr <= typical_range[1]:
                return True
            
            # Accept if within general CTR bounds
            return True
            
        except Exception:
            return False
    
    def _select_best_ctr_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Select the best CTR column from candidates"""
        try:
            scores = {}
            
            for col in candidates:
                series = df[col]
                ctr = series.mean()
                
                # Score based on how close to typical CTR range
                typical_range = self.target_config['typical_ctr_range']
                
                if typical_range[0] <= ctr <= typical_range[1]:
                    # Perfect score for typical range
                    scores[col] = 100
                else:
                    # Score based on distance from typical range
                    if ctr < typical_range[0]:
                        distance = typical_range[0] - ctr
                    else:
                        distance = ctr - typical_range[1]
                    
                    # Lower distance = higher score
                    scores[col] = max(0, 100 - (distance * 1000))
            
            if scores:
                best_col = max(scores.keys(), key=lambda k: scores[k])
                return best_col
            
            return None
            
        except Exception:
            return None

class StreamingDataLoader:
    """
    Streaming data loader with memory management and quick mode support
    Handles large parquet files with memory-efficient processing
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        self.temp_dir = tempfile.mkdtemp()  # Temporary directory for intermediate storage
        self.quick_mode = False  # Quick mode flag for 50-sample testing
        
        logger.info("Streaming data loader initialization completed")
    
    def __del__(self):
        """Cleanup temporary directory on destruction"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def set_quick_mode(self, quick_mode: bool):
        """
        Enable or disable quick mode
        
        Args:
            quick_mode: If True, load only 50 samples for testing
        """
        self.quick_mode = quick_mode
        if quick_mode:
            logger.info("Quick mode enabled: Will load 50 samples only")
        else:
            logger.info("Full mode enabled: Will load complete dataset")
    
    def load_quick_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load small sample data for quick testing (50 samples)
        
        Returns:
            Tuple of (train_df, test_df) with 50 samples each
        """
        logger.info("=== Quick sample data loading started ===")
        
        try:
            # Load small samples from both files
            train_sample = self._load_sample_from_file(self.config.TRAIN_PATH, 35, is_train=True)
            test_sample = self._load_sample_from_file(self.config.TEST_PATH, 15, is_train=False)
            
            # Ensure we have the target column
            if self.target_column and self.target_column in train_sample.columns:
                logger.info(f"Target column confirmed: {self.target_column}")
            else:
                # Create dummy target if needed
                self.target_column = 'clicked'
                if self.target_column not in train_sample.columns:
                    train_sample[self.target_column] = np.random.binomial(1, 0.02, len(train_sample))
                    logger.info(f"Created dummy target column: {self.target_column}")
            
            logger.info(f"Quick sample loading completed - train: {train_sample.shape}, test: {test_sample.shape}")
            
            return train_sample, test_sample
            
        except Exception as e:
            logger.error(f"Quick sample loading failed: {e}")
            # Return minimal dummy data as fallback
            return self._create_dummy_data()
    
    def _load_sample_from_file(self, file_path: Path, sample_size: int, is_train: bool = True) -> pd.DataFrame:
        """
        Load a small sample from a parquet file
        
        Args:
            file_path: Path to parquet file
            sample_size: Number of samples to load
            is_train: Whether this is training data
            
        Returns:
            DataFrame with sampled data
        """
        try:
            if not PYARROW_AVAILABLE:
                raise ValueError("PyArrow is required for parquet loading")
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return self._create_dummy_sample(sample_size, is_train)
            
            # Read first row group and sample from it
            parquet_file = pq.ParquetFile(file_path)
            
            # Read first row group
            table = parquet_file.read_row_group(0)
            df = table.to_pandas()
            
            # Detect target column if this is training data
            if is_train and self.target_column is None:
                self.target_column = self.column_analyzer.detect_target_column(df)
            
            # Sample the requested number of rows
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # Basic data optimization
            df = self._optimize_dataframe_memory(df)
            
            logger.info(f"Sample loaded from {file_path.name}: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"Sample loading failed for {file_path}: {e}")
            return self._create_dummy_sample(sample_size, is_train)
    
    def _create_dummy_sample(self, sample_size: int, is_train: bool) -> pd.DataFrame:
        """Create dummy data when file loading fails"""
        try:
            # Create basic dummy features
            data = {
                'feature_1': np.random.randint(0, 100, sample_size),
                'feature_2': np.random.normal(0, 1, sample_size),
                'feature_3': np.random.choice(['A', 'B', 'C'], sample_size),
                'feature_4': np.random.uniform(0, 1, sample_size)
            }
            
            # Add target column for training data
            if is_train:
                data['clicked'] = np.random.binomial(1, 0.02, sample_size)
                self.target_column = 'clicked'
            
            df = pd.DataFrame(data)
            logger.info(f"Created dummy sample: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Dummy sample creation failed: {e}")
            # Return absolute minimal data
            return pd.DataFrame({'dummy': [0] * sample_size})
    
    def _create_dummy_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal dummy data as absolute fallback"""
        train_data = {
            'feature_1': [1, 2, 3] * 17,  # 51 samples
            'feature_2': [0.1, 0.2, 0.3] * 17,
            'clicked': [0, 1, 0] * 17
        }
        
        test_data = {
            'feature_1': [4, 5, 6] * 5,  # 15 samples
            'feature_2': [0.4, 0.5, 0.6] * 5
        }
        
        train_df = pd.DataFrame(train_data).iloc[:35]  # Exactly 35 samples
        test_df = pd.DataFrame(test_data).iloc[:15]    # Exactly 15 samples
        
        self.target_column = 'clicked'
        logger.warning("Using minimal dummy data as fallback")
        
        return train_df, test_df
    
    def load_full_data_streaming(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete processing via streaming with memory management
        
        Returns:
            Tuple of (train_df, test_df) with full dataset
        """
        logger.info("=== Full data streaming loading started ===")
        
        try:
            # Memory status check
            self.memory_monitor.log_memory_status("Streaming start", force=True)
            
            # File validation
            if not self._validate_files():
                raise ValueError("Data files do not exist")
            
            # 1. Training data streaming processing
            logger.info("Training data streaming processing started")
            train_df = self._stream_process_file(str(self.config.TRAIN_PATH), is_train=True)
            
            if train_df is None or train_df.empty:
                raise ValueError("Training data streaming processing failed")
            
            # Save training data to temporary file to free memory
            temp_train_file = os.path.join(self.temp_dir, 'temp_train.pkl')
            with open(temp_train_file, 'wb') as f:
                pickle.dump(train_df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get basic info before clearing memory
            train_shape = train_df.shape
            target_column = self.target_column
            
            # Clear training data from memory
            del train_df
            
            # Intensive memory cleanup
            self.memory_monitor.force_memory_cleanup(intensive=True)
            time.sleep(2)  # Additional wait for system cleanup
            self.memory_monitor.log_memory_status("Training data saved, memory cleared", force=True)
            
            # 2. Test data streaming processing
            logger.info("Test data streaming processing started") 
            test_df = self._stream_process_file(str(self.config.TEST_PATH), is_train=False)
            
            if test_df is None or test_df.empty:
                raise ValueError("Test data streaming processing failed")
            
            # Reload training data
            logger.info("Reloading training data from temporary storage")
            with open(temp_train_file, 'rb') as f:
                train_df = pickle.load(f)
            
            # Clean up temporary file
            os.remove(temp_train_file)
            
            # Restore target column info
            self.target_column = target_column
            
            # Final memory status
            self.memory_monitor.log_memory_status("Streaming completed", force=True)
            
            logger.info(f"=== Full data streaming completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Streaming data loading failed: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            
            # Clean up temporary files
            try:
                temp_train_file = os.path.join(self.temp_dir, 'temp_train.pkl')
                if os.path.exists(temp_train_file):
                    os.remove(temp_train_file)
            except Exception:
                pass
            
            raise
    
    def _stream_process_file(self, file_path: str, is_train: bool = True) -> pd.DataFrame:
        """
        File streaming processing with memory optimization
        
        Args:
            file_path: Path to parquet file
            is_train: Whether this is training data
            
        Returns:
            Processed DataFrame
        """
        try:
            # Check metadata with PyArrow
            if not PYARROW_AVAILABLE:
                raise ValueError("PyArrow is required")
            
            # Initial memory check
            pressure = self.memory_monitor.check_memory_pressure()
            if pressure['should_abort']:
                logger.error(f"Insufficient memory for processing: {pressure['available_gb']:.1f}GB available")
                raise MemoryError(f"Insufficient memory: {pressure['available_gb']:.1f}GB available")
            
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            num_row_groups = parquet_file.num_row_groups
            
            logger.info(f"File analysis - Total {total_rows:,} rows, {num_row_groups} row groups")
            
            # Column analysis with first row group (for training data)
            if is_train:
                sample_table = parquet_file.read_row_group(0)
                sample_df = sample_table.to_pandas()
                
                # Limit sample size for analysis
                if len(sample_df) > 1000:
                    sample_df = sample_df.sample(n=1000, random_state=42)
                
                # Detect target column
                self.target_column = self.column_analyzer.detect_target_column(sample_df)
                logger.info(f"Detected target column: {self.target_column}")
                
                # Clean up sample
                del sample_df, sample_table
                gc.collect()
            
            # Process row groups in smaller batches for memory efficiency
            all_chunks = []
            processed_rows = 0
            batch_size = 1  # Process one row group at a time for large files
            
            for rg_start in range(0, num_row_groups, batch_size):
                rg_end = min(rg_start + batch_size, num_row_groups)
                
                try:
                    # Memory check before processing each batch
                    pressure = self.memory_monitor.check_memory_pressure()
                    if pressure['should_abort']:
                        logger.warning(f"Memory limit approaching - Processed rows: {processed_rows}, continuing attempt")
                        
                        # Try emergency memory cleanup
                        self.memory_monitor.force_memory_cleanup(intensive=True)
                        time.sleep(1)
                        
                        # Check again after cleanup
                        pressure_after = self.memory_monitor.check_memory_pressure()
                        if pressure_after['should_abort']:
                            logger.error(f"Memory limit reached even after cleanup - Stopping: {processed_rows} rows")
                            break
                    
                    # Read row group batch
                    row_group_tables = []
                    for rg_idx in range(rg_start, rg_end):
                        table = parquet_file.read_row_group(rg_idx)
                        row_group_tables.append(table)
                   
                    # Combine row groups
                    if len(row_group_tables) == 1:
                        combined_table = row_group_tables[0]
                    else:
                        combined_table = pa.concat_tables(row_group_tables)
                    
                    # Convert to pandas
                    chunk_df = combined_table.to_pandas()
                    
                    # Clean up Arrow objects immediately
                    del row_group_tables, combined_table
                    
                    # Optimize memory usage
                    chunk_df = self._optimize_dataframe_memory(chunk_df)
                    
                    all_chunks.append(chunk_df)
                    processed_rows += len(chunk_df)
                    
                    logger.info(f"Row groups {rg_start+1}-{rg_end} processed: {len(chunk_df):,} rows")
                    
                    # Memory cleanup between chunks
                    if len(all_chunks) % 2 == 0:  # Every 2 chunks
                        self.memory_monitor.force_memory_cleanup()
                    
                    # Progress output
                    progress_pct = (rg_end / num_row_groups) * 100
                    logger.info(f"Progress: {progress_pct:.1f}% ({processed_rows:,} rows)")
                    
                except Exception as e:
                    logger.error(f"Row group batch {rg_start+1}-{rg_end} processing failed: {e}")
                    
                    # Emergency memory cleanup
                    self.memory_monitor.force_memory_cleanup(intensive=True)
                    
                    # Skip this batch and continue
                    continue
            
            # Final combination
            if not all_chunks:
                logger.error("No data processed successfully")
                return pd.DataFrame()
            
            logger.info(f"Final combination: {len(all_chunks)} chunks")
            final_df = self._combine_chunks_memory_efficient(all_chunks)
            
            logger.info(f"File streaming completed: {len(final_df):,} rows")
            
            # Final memory cleanup
            self.memory_monitor.force_memory_cleanup(intensive=True)
            
            return final_df
            
        except Exception as e:
            logger.error(f"File streaming processing failed: {e}")
            return pd.DataFrame()
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        try:
            for col in df.columns:
                col_type = df[col].dtype
                
                # Optimize integer columns
                if col_type == 'int64':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if min_val >= 0:  # Unsigned integers
                        if max_val < 255:
                            df[col] = df[col].astype('uint8')
                        elif max_val < 65535:
                            df[col] = df[col].astype('uint16')
                        elif max_val < 4294967295:
                            df[col] = df[col].astype('uint32')
                    else:  # Signed integers
                        if min_val > -128 and max_val < 127:
                            df[col] = df[col].astype('int8')
                        elif min_val > -32768 and max_val < 32767:
                            df[col] = df[col].astype('int16')
                        elif min_val > -2147483648 and max_val < 2147483647:
                            df[col] = df[col].astype('int32')
                
                # Optimize float columns
                elif col_type == 'float64':
                    df[col] = df[col].astype('float32')
                
                # Optimize object columns
                elif col_type == 'object':
                    # Try to convert to category if low cardinality
                    if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                        df[col] = df[col].astype('category')
            
            return df
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return df
    
    def _combine_chunks_memory_efficient(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine chunks with memory management
        
        Args:
            chunks: List of DataFrame chunks to combine
            
        Returns:
            Combined DataFrame
        """
        try:
            if not chunks:
                return pd.DataFrame()
            
            if len(chunks) == 1:
                return chunks[0]
            
            # Combine chunks progressively to manage memory
            logger.info(f"Combining {len(chunks)} chunks progressively")
            
            result = chunks[0]
            for i, chunk in enumerate(chunks[1:], 1):
                try:
                    result = pd.concat([result, chunk], ignore_index=True)
                    
                    # Memory cleanup after every few combinations
                    if i % 3 == 0:
                        self.memory_monitor.force_memory_cleanup()
                        
                        # Check memory pressure
                        pressure = self.memory_monitor.check_memory_pressure()
                        if pressure['should_cleanup']:
                            logger.info(f"Memory cleanup during combination: chunk {i}/{len(chunks)-1}")
                    
                except Exception as e:
                    logger.error(f"Chunk combination failed at chunk {i}: {e}")
                    continue
            
            logger.info(f"Chunk combination successful: {len(result):,} rows")
            return result
            
        except Exception as e:
            logger.error(f"Chunk combination failed: {e}")
            # Return first valid chunk on failure
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    return chunk
            return pd.DataFrame()
    
    def _validate_files(self) -> bool:
        """
        Validate input files exist and have reasonable sizes
        
        Returns:
            True if files are valid
        """
        try:
            train_path = Path(self.config.TRAIN_PATH)
            test_path = Path(self.config.TEST_PATH)
            
            if not train_path.exists():
                logger.error(f"Training file not found: {train_path}")
                return False
            
            if not test_path.exists():
                logger.error(f"Test file not found: {test_path}")
                return False
            
            # Check file sizes
            train_size_mb = train_path.stat().st_size / (1024**2)
            test_size_mb = test_path.stat().st_size / (1024**2)
            
            if train_size_mb < 100:  # Less than 100MB seems too small
                logger.warning(f"Training file seems small: {train_size_mb:.1f}MB")
            
            if test_size_mb < 10:  # Less than 10MB seems too small
                logger.warning(f"Test file seems small: {test_size_mb:.1f}MB")
            
            logger.info("File validation successful")
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return False
    
    def get_detected_target_column(self) -> Optional[str]:
        """Return detected target column name"""
        return self.target_column

class LargeDataLoader:
    """
    Large data loader with quick mode support
    Main interface for data loading operations
    """
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.target_column = None
        self.quick_mode = False
        
        # Performance statistics
        self.loading_stats = {
            'start_time': time.time(),
            'data_loaded': False,
            'train_rows': 0,
            'test_rows': 0,
            'loading_time': 0.0,
            'memory_usage': 0.0
        }
        
        logger.info("Large data loader initialization completed")
    
    def set_quick_mode(self, quick_mode: bool):
        """
        Enable or disable quick mode for testing
        
        Args:
            quick_mode: If True, load only 50 samples
        """
        self.quick_mode = quick_mode
        if quick_mode:
            logger.info("Large data loader set to quick mode (50 samples)")
        else:
            logger.info("Large data loader set to full mode (complete dataset)")
    
    def load_quick_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load small sample data for quick testing
        
        Returns:
            Tuple of (train_df, test_df) with ~50 samples total
        """
        logger.info("=== Quick sample data loading via LargeDataLoader ===")
        
        # Use streaming loader in quick mode
        streaming_loader = StreamingDataLoader(self.config)
        streaming_loader.set_quick_mode(True)
        result = streaming_loader.load_quick_sample_data()
        
        # Store detected target column
        self.target_column = streaming_loader.get_detected_target_column()
        
        # Update statistics
        self.loading_stats.update({
            'data_loaded': True,
            'train_rows': result[0].shape[0] if result[0] is not None else 0,
            'test_rows': result[1].shape[0] if result[1] is not None else 0,
            'loading_time': time.time() - self.loading_stats['start_time']
        })
        
        return result
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data processing (full dataset)
        
        Returns:
            Tuple of (train_df, test_df) with complete dataset
        """
        logger.info("=== Complete data processing started ===")
        
        # Process complete data via streaming
        streaming_loader = StreamingDataLoader(self.config)
        streaming_loader.set_quick_mode(False)
        result = streaming_loader.load_full_data_streaming()
        
        # Store detected target column
        self.target_column = streaming_loader.get_detected_target_column()
        
        # Update statistics
        self.loading_stats.update({
            'data_loaded': True,
            'train_rows': result[0].shape[0] if result[0] is not None else 0,
            'test_rows': result[1].shape[0] if result[1] is not None else 0,
            'loading_time': time.time() - self.loading_stats['start_time']
        })
        
        return result
    
    def get_detected_target_column(self) -> Optional[str]:
        """Return detected target column name"""
        return self.target_column
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get loading performance statistics"""
        return self.loading_stats.copy()

# Aliases for backward compatibility
DataLoader = StreamingDataLoader
SimpleDataLoader = StreamingDataLoader

if __name__ == "__main__":
    # Test code for development
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    
    try:
        loader = LargeDataLoader(config)
        
        # Test quick mode
        print("Testing quick mode...")
        loader.set_quick_mode(True)
        train_df, test_df = loader.load_quick_sample_data()
        
        print(f"Quick mode results:")
        print(f"Training data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        print(f"Detected target column: {loader.get_detected_target_column()}")
        print(f"Total samples: {len(train_df) + len(test_df)}")
        
        # Test full mode
        print("\nTesting full mode...")
        loader.set_quick_mode(False)
        train_df_full, test_df_full = loader.load_large_data_optimized()
        
        print(f"Full mode results:")
        print(f"Training data: {train_df_full.shape}")
        print(f"Test data: {test_df_full.shape}")
        print(f"Complete processing finished: {len(train_df_full) + len(test_df_full):,} rows")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")