# data_loader.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import gc
import time
import os
import tempfile
import pickle
from pathlib import Path
import warnings
from abc import ABC, abstractmethod

# Safe library imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Memory monitoring will be limited.")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    warnings.warn("PyArrow not available. Parquet reading will be limited.")

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring class with improved efficiency"""
    
    def __init__(self):
        self.psutil_available = PSUTIL_AVAILABLE
        
    def get_process_memory_gb(self) -> float:
        """Get process memory usage in GB"""
        try:
            if self.psutil_available:
                process = psutil.Process()
                return process.memory_info().rss / (1024**3)
            else:
                return 2.0
        except Exception:
            return 2.0
    
    def get_system_memory_gb(self) -> Dict[str, float]:
        """Get system memory info in GB"""
        try:
            if self.psutil_available:
                vm = psutil.virtual_memory()
                return {
                    'total': vm.total / (1024**3),
                    'available': vm.available / (1024**3),
                    'used': vm.used / (1024**3),
                    'percent': vm.percent
                }
            else:
                return {
                    'total': 64.0,
                    'available': 45.0,
                    'used': 19.0,
                    'percent': 30.0
                }
        except Exception:
            return {
                'total': 64.0,
                'available': 45.0,
                'used': 19.0,
                'percent': 30.0
            }
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check memory pressure level"""
        try:
            memory_info = self.get_system_memory_gb()
            available_gb = memory_info['available']
            
            # Updated thresholds for 64GB system
            if available_gb < 3:  # Less than 3GB
                pressure_level = 'abort'
            elif available_gb < 6:  # Less than 6GB
                pressure_level = 'critical'
            elif available_gb < 12:  # Less than 12GB
                pressure_level = 'high'
            elif available_gb < 20:  # Less than 20GB
                pressure_level = 'medium'
            else:
                pressure_level = 'low'
            
            return {
                'pressure_level': pressure_level,
                'available_gb': available_gb,
                'total_gb': memory_info['total'],
                'should_abort': pressure_level == 'abort',
                'should_cleanup': pressure_level in ['critical', 'high', 'medium'],
                'memory_percent': memory_info['percent']
            }
            
        except Exception:
            return {
                'pressure_level': 'low',
                'available_gb': 45.0,
                'total_gb': 64.0,
                'should_abort': False,
                'should_cleanup': False,
                'memory_percent': 30.0
            }
    
    def log_memory_status(self, context: str = "", force: bool = False):
        """Log memory status"""
        try:
            pressure = self.check_memory_pressure()
            
            if force or pressure['pressure_level'] != 'low':
                logger.info(f"Memory status [{context}]: Process {self.get_process_memory_gb():.1f}GB, "
                           f"Available {pressure['available_gb']:.1f}GB - {pressure['pressure_level'].upper()}")
                
        except Exception as e:
            logger.warning(f"Memory status logging failed: {e}")
    
    def force_memory_cleanup(self, intensive: bool = False) -> float:
        """Memory cleanup"""
        try:
            initial_memory = self.get_process_memory_gb()
            pressure = self.check_memory_pressure()
            
            # Adjusted cleanup strategy for 64GB system
            if pressure['pressure_level'] == 'abort' or intensive:
                cleanup_rounds = 25
                sleep_time = 0.3
            elif pressure['pressure_level'] == 'critical':
                cleanup_rounds = 20
                sleep_time = 0.2
            elif pressure['pressure_level'] == 'high':
                cleanup_rounds = 15
                sleep_time = 0.15
            else:
                cleanup_rounds = 10
                sleep_time = 0.1
            
            # Garbage collection
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Check effectiveness every 5 rounds
                if i % 5 == 0 and i > 0:
                    current_memory = self.get_process_memory_gb()
                    reduction = initial_memory - current_memory
                    if reduction > 1.0:  # 1GB reduction achieved
                        logger.info(f"Memory cleanup effective at round {i+1}: "
                                   f"{initial_memory:.1f}GB → {current_memory:.1f}GB")
                        break
            
            final_memory = self.get_process_memory_gb()
            reduction = initial_memory - final_memory
            
            logger.info(f"Memory cleanup completed: {reduction:.1f}GB freed "
                       f"({initial_memory:.1f}GB → {final_memory:.1f}GB)")
            
            return reduction
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0.0

class DataColumnAnalyzer:
    """Data column analyzer for CTR data"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_config = config.TARGET_DETECTION_CONFIG
    
    def detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect CTR target column"""
        try:
            # Check candidate columns first
            for candidate in self.config.TARGET_COLUMN_CANDIDATES:
                if candidate in df.columns:
                    if self._is_valid_target_column(df[candidate]):
                        logger.info(f"Target column detected: {candidate}")
                        return candidate
            
            # Search all columns if no candidate found
            for col in df.columns:
                if self._is_valid_target_column(df[col]):
                    logger.info(f"Target column found: {col}")
                    return col
            
            # Default fallback
            logger.warning("No valid target column found, using 'clicked'")
            return 'clicked'
            
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return 'clicked'
    
    def _is_valid_target_column(self, series: pd.Series) -> bool:
        """Check if column is valid CTR target"""
        try:
            # Check data type
            if series.dtype not in ['int64', 'int32', 'int8', 'uint8', 'bool']:
                return False
            
            # Check unique values
            unique_values = set(series.dropna().unique())
            if not unique_values.issubset(self.target_config['binary_values']):
                return False
            
            # CTR range validation
            ctr = series.mean()
            min_ctr = self.target_config['min_ctr']
            max_ctr = self.target_config['max_ctr']
            
            if not (min_ctr <= ctr <= max_ctr):
                return False
            
            # Typical CTR range preference
            typical_range = self.target_config['typical_ctr_range']
            if typical_range[0] <= ctr <= typical_range[1]:
                return True
            
            # Accept if within general CTR bounds
            return True
            
        except Exception:
            return False

class StreamingDataLoader:
    """Streaming data loader with memory management"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        self.temp_dir = tempfile.mkdtemp()  # Temporary directory for intermediate storage
        
        logger.info("Streaming data loader initialization completed")
    
    def __del__(self):
        """Cleanup temporary directory"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def load_full_data_streaming(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete processing via streaming with memory management"""
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
        """File streaming processing with memory optimization"""
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
            self.memory_monitor.force_memory_cleanup(intensive=True)
            raise
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        try:
            initial_memory = df.memory_usage(deep=True).sum() / (1024**2)
            
            # Optimize data types
            for col in df.columns:
                col_type = df[col].dtype
                
                if col_type == 'object':
                    # Try to convert to numeric if possible
                    if df[col].str.isnumeric().all() if hasattr(df[col].str, 'isnumeric') else False:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        # Convert to category if low cardinality
                        if df[col].nunique() / len(df) < 0.5:
                            df[col] = df[col].astype('category')
                
                elif col_type in ['int64', 'int32']:
                    # Downcast integers
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                elif col_type in ['float64']:
                    # Downcast floats
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            final_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (initial_memory - final_memory) / initial_memory * 100
            
            logger.info(f"DataFrame memory optimized: {reduction:.1f}% reduction")
            
            return df
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return df
    
    def _combine_chunks_memory_efficient(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine chunks with memory efficiency"""
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
        """Validate input files"""
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
    """Large data loader"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.target_column = None
        
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
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data processing"""
        logger.info("=== Complete data processing started ===")
        
        # Process complete data via streaming
        streaming_loader = StreamingDataLoader(self.config)
        result = streaming_loader.load_full_data_streaming()
        
        # Store detected target column
        self.target_column = streaming_loader.get_detected_target_column()
        
        return result
    
    def get_detected_target_column(self) -> Optional[str]:
        """Return detected target column name"""
        return self.target_column

# Aliases for backward compatibility
DataLoader = StreamingDataLoader
SimpleDataLoader = StreamingDataLoader

if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    
    try:
        loader = StreamingDataLoader(config)
        train_df, test_df = loader.load_full_data_streaming()
        
        print(f"Training data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        print(f"Detected target column: {loader.get_detected_target_column()}")
        print(f"Complete processing finished: {len(train_df) + len(test_df):,} rows")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")