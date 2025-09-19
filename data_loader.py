# data_loader.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import gc
import os
import pickle
import tempfile
from pathlib import Path
import warnings

# Safe imports with error handling
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    warnings.warn("PyArrow is not installed. Using default pandas loading.")

# Safe Psutil import handling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil is not installed. Memory monitoring features will be limited.")

from config import Config

logger = logging.getLogger(__name__)

class ProgressReporter:
    """Progress reporting"""
    
    def __init__(self, total_rows: int, operation_name: str = "Data processing"):
        self.total_rows = total_rows
        self.processed_rows = 0
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.report_interval = 30  # 30 seconds interval
        
    def update(self, processed_count: int):
        """Update progress"""
        self.processed_rows += processed_count
        current_time = time.time()
        
        if current_time - self.last_report_time >= self.report_interval:
            self.report_progress()
            self.last_report_time = current_time
    
    def report_progress(self):
        """Report current progress"""
        if self.total_rows <= 0:
            return
        
        elapsed_time = time.time() - self.start_time
        progress_rate = self.processed_rows / self.total_rows
        
        if progress_rate > 0:
            estimated_total_time = elapsed_time / progress_rate
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        logger.info(f"{self.operation_name}: {self.processed_rows:,}/{self.total_rows:,} "
                   f"({progress_rate:.1%}) - {elapsed_time:.1f}s elapsed, "
                   f"~{remaining_time:.1f}s remaining")

class MemoryMonitor:
    """Memory monitoring class"""
    
    def __init__(self):
        # More aggressive memory thresholds for 64GB environment
        self.process_memory_warning = 40.0  # 40GB process memory warning
        self.process_memory_critical = 45.0  # 45GB process memory critical  
        self.available_memory_warning = 10.0  # 10GB available memory warning
        self.available_memory_critical = 5.0  # 5GB available memory critical
        
    def get_process_memory_gb(self) -> float:
        """Get current process memory in GB"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024**2)
                return memory_mb / 1024
            return 2.0
        except Exception:
            return 2.0
    
    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        try:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                return vm.available / (1024**3)
            return 40.0
        except Exception:
            return 40.0
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check memory pressure status"""
        try:
            process_memory = self.get_process_memory_gb()
            available_memory = self.get_available_memory_gb()
            
            # More aggressive pressure levels for large data
            if (process_memory >= self.process_memory_critical or 
                available_memory <= self.available_memory_critical):
                pressure_level = 'abort'
                should_abort = True
                should_cleanup = True
                should_simplify = True
                should_reduce_batch = True
            elif (process_memory >= self.process_memory_warning or 
                  available_memory <= self.available_memory_warning):
                pressure_level = 'critical'
                should_abort = False
                should_cleanup = True
                should_simplify = True
                should_reduce_batch = True
            elif (process_memory >= 30 or available_memory <= 15):
                pressure_level = 'high'
                should_abort = False
                should_cleanup = True
                should_simplify = False
                should_reduce_batch = True
            elif (process_memory >= 20 or available_memory <= 20):
                pressure_level = 'warning'
                should_abort = False
                should_cleanup = False
                should_simplify = False
                should_reduce_batch = False
            else:
                pressure_level = 'normal'
                should_abort = False
                should_cleanup = False
                should_simplify = False
                should_reduce_batch = False
            
            return {
                'pressure_level': pressure_level,
                'process_memory_gb': process_memory,
                'available_memory_gb': available_memory,
                'should_abort': should_abort,
                'should_cleanup': should_cleanup,
                'should_simplify': should_simplify,
                'should_reduce_batch': should_reduce_batch
            }
            
        except Exception:
            return {
                'pressure_level': 'normal',
                'process_memory_gb': 2.0,
                'available_memory_gb': 40.0,
                'should_abort': False,
                'should_cleanup': False,
                'should_simplify': False,
                'should_reduce_batch': False
            }
    
    def log_memory_status(self, operation: str = "", force: bool = False):
        """Log current memory status"""
        try:
            pressure = self.check_memory_pressure()
            
            if force or pressure['pressure_level'] in ['warning', 'high', 'critical', 'abort']:
                if pressure['pressure_level'] in ['critical', 'abort']:
                    logger.warning(f"Memory pressure: Process {pressure['process_memory_gb']:.1f}GB, "
                                 f"Available {pressure['available_memory_gb']:.1f}GB")
                else:
                    logger.info(f"{operation} - Memory status: "
                               f"Process {pressure['process_memory_gb']:.1f}GB, "
                               f"Available {pressure['available_memory_gb']:.1f}GB")
                
        except Exception as e:
            logger.warning(f"Memory status logging failed: {e}")
    
    def force_memory_cleanup(self, intensive: bool = False) -> float:
        """Memory cleanup"""
        try:
            initial_memory = self.get_process_memory_gb()
            pressure = self.check_memory_pressure()
            
            # More intensive cleanup for large data
            if pressure['pressure_level'] == 'abort' or intensive:
                cleanup_rounds = 30  # Increased from 20 to 30
                sleep_time = 0.5
            elif pressure['pressure_level'] == 'critical':
                cleanup_rounds = 20
                sleep_time = 0.3
            elif pressure['pressure_level'] == 'high':
                cleanup_rounds = 15
                sleep_time = 0.2
            else:
                cleanup_rounds = 10
                sleep_time = 0.1
            
            # Garbage collection
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Intermediate check
                if i % 10 == 0:  # More frequent checks
                    current_memory = self.get_process_memory_gb()
                    if current_memory < initial_memory * 0.85:  # 15% reduction target
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
    """Data column analyzer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.target_candidates = config.TARGET_COLUMN_CANDIDATES
        self.target_config = config.TARGET_DETECTION_CONFIG
        
    def detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect target column"""
        try:
            # Direct candidate search
            for candidate in self.target_candidates:
                if candidate in df.columns:
                    if self._validate_target_column(df[candidate]):
                        logger.info(f"Target column detected: {candidate}")
                        return candidate
            
            # Binary column search
            binary_columns = []
            for col in df.columns:
                unique_values = set(df[col].dropna().unique())
                if unique_values.issubset({0, 1}) or unique_values.issubset({0.0, 1.0}):
                    binary_columns.append(col)
            
            if binary_columns:
                # Select column with CTR-like distribution
                for col in binary_columns:
                    if self._validate_target_column(df[col]):
                        logger.info(f"Target column detected via binary search: {col}")
                        return col
                
                # Fallback to first binary column
                logger.info(f"Target column detected (fallback): {binary_columns[0]}")
                return binary_columns[0]
            
            logger.warning("Target column not detected")
            return None
            
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return None
    
    def _validate_target_column(self, series: pd.Series) -> bool:
        """Validate target column characteristics"""
        try:
            unique_values = set(series.dropna().unique())
            
            # Binary validation
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
                    # Memory check before processing
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
                    if df[col].str.replace('.', '', regex=False).str.replace('-', '', regex=False).str.isdigit().all():
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass
                    else:
                        # Convert to category if low cardinality
                        if df[col].nunique() / len(df) < 0.5:
                            df[col] = df[col].astype('category')
                
                elif col_type == 'int64':
                    # Downcast integers
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
                
                elif col_type == 'float64':
                    # Downcast floats
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            final_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (initial_memory - final_memory) / initial_memory * 100
            
            logger.info(f"DataFrame memory optimized: {reduction:.1f}% reduction")
            
            return df
            
        except Exception as e:
            logger.warning(f"DataFrame memory optimization failed: {e}")
            return df
    
    def _combine_chunks_memory_efficient(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine chunks with memory efficiency"""
        try:
            if not chunks:
                return pd.DataFrame()
            
            if len(chunks) == 1:
                return chunks[0]
            
            logger.info(f"Combining {len(chunks)} chunks with memory efficiency")
            
            # Combine in pairs to manage memory
            while len(chunks) > 1:
                new_chunks = []
                
                for i in range(0, len(chunks), 2):
                    if i + 1 < len(chunks):
                        # Combine two chunks
                        try:
                            combined = pd.concat([chunks[i], chunks[i+1]], 
                                               ignore_index=True, copy=False)
                            new_chunks.append(combined)
                            
                            # Clean up original chunks
                            del chunks[i], chunks[i+1]
                            
                        except Exception as e:
                            logger.warning(f"Chunk combination failed: {e}")
                            new_chunks.extend([chunks[i], chunks[i+1]])
                    else:
                        # Odd chunk
                        new_chunks.append(chunks[i])
                
                chunks = new_chunks
                
                # Memory cleanup between iterations
                gc.collect()
                
                logger.info(f"Combination iteration: {len(chunks)} chunks remaining")
            
            logger.info(f"Chunk combination successful: {len(chunks[0]):,} rows")
            return chunks[0]
            
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