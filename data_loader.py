# data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, Iterator, List
import logging
import gc
import mmap
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
import time
import threading
import sys

# Safe PyArrow import handling
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pyarrow.compute as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("PyArrow is not installed. Using default pandas loading.")

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
        # Reduced memory thresholds to be more conservative
        self.process_memory_warning = 25.0  # 25GB process memory warning
        self.process_memory_critical = 30.0  # 30GB process memory critical  
        self.available_memory_warning = 15.0  # 15GB available memory warning
        self.available_memory_critical = 10.0  # 10GB available memory critical
        
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
            
            # Determine pressure level
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
            elif (process_memory >= 20 or available_memory <= 20):
                pressure_level = 'high'
                should_abort = False
                should_cleanup = True
                should_simplify = False
                should_reduce_batch = True
            elif (process_memory >= 15 or available_memory <= 25):
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
            
            # Cleanup intensity based on pressure level
            if pressure['pressure_level'] == 'abort' or intensive:
                cleanup_rounds = 20
                sleep_time = 0.3
            elif pressure['pressure_level'] == 'critical':
                cleanup_rounds = 15
                sleep_time = 0.2
            elif pressure['pressure_level'] == 'high':
                cleanup_rounds = 10
                sleep_time = 0.1
            else:
                cleanup_rounds = 5
                sleep_time = 0.05
            
            # Garbage collection
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Intermediate check
                if i % 5 == 0:
                    current_memory = self.get_process_memory_gb()
                    if initial_memory - current_memory > 3.0:
                        break
            
            # Windows memory cleanup
            if cleanup_rounds >= 10:
                try:
                    import ctypes
                    if hasattr(ctypes, 'windll'):
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                        time.sleep(0.5)
                        gc.collect()
                        if intensive:
                            time.sleep(1.0)
                            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                except Exception:
                    pass
            
            final_memory = self.get_process_memory_gb()
            memory_freed = initial_memory - final_memory
            
            if memory_freed > 0.2:
                logger.info(f"Memory cleanup: {memory_freed:.2f}GB freed ({cleanup_rounds} rounds)")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0.0

class DataColumnAnalyzer:
    """Data column analyzer"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_candidates = config.TARGET_COLUMN_CANDIDATES if hasattr(config, 'TARGET_COLUMN_CANDIDATES') else ['clicked', 'target', 'label', 'y']
        
    def detect_target_column(self, sample_df: pd.DataFrame) -> Optional[str]:
        """Detect target column from sample data"""
        try:
            # Check exact matches first
            for candidate in self.target_candidates:
                if candidate in sample_df.columns:
                    logger.info(f"Target column detected: {candidate}")
                    return candidate
            
            # Check partial matches
            for col in sample_df.columns:
                col_lower = col.lower()
                for candidate in self.target_candidates:
                    if candidate.lower() in col_lower:
                        logger.info(f"Target column detected (partial match): {col}")
                        return col
            
            # Check binary columns (0/1 values only)
            for col in sample_df.columns:
                if sample_df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    unique_values = sample_df[col].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
                        logger.info(f"Target column detected (binary): {col}")
                        return col
            
            logger.warning("Target column not detected")
            return None
            
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return None
    
    def validate_data_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Validate data consistency between train and test"""
        try:
            train_cols = set(train_df.columns)
            test_cols = set(test_df.columns)
            
            # Check column alignment
            common_cols = train_cols & test_cols
            train_only = train_cols - test_cols
            test_only = test_cols - train_cols
            
            if train_only:
                logger.info(f"Train-only columns: {list(train_only)}")
            if test_only:
                logger.info(f"Test-only columns: {list(test_only)}")
            
            if len(common_cols) < len(train_cols) * 0.8:
                logger.warning("Significant column mismatch between train and test")
                return False
            
            logger.info(f"Data consistency validation passed: {len(common_cols)} common columns")
            return True
            
        except Exception as e:
            logger.error(f"Data consistency validation failed: {e}")
            return False
    
    def get_detected_target_column(self) -> Optional[str]:
        """Return detected target column name"""
        return self.target_column

class DataChunkProcessor:
    """Data chunk processor"""
    
    def __init__(self, file_path: str, chunk_size: int = 50000, memory_monitor: MemoryMonitor = None):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.min_chunk_size = 5000
        self.max_chunk_size = 100000
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.total_rows = 0
        self.progress_reporter = None
        
    def __enter__(self):
        """Context manager entry"""
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File does not exist: {self.file_path}")
            
            file_size_mb = os.path.getsize(self.file_path) / (1024**2)
            logger.info(f"File size: {file_size_mb:.1f}MB")
            
            # Check total row count
            self.total_rows = self._estimate_total_rows()
            
            # Initialize progress reporter
            self.progress_reporter = ProgressReporter(
                self.total_rows, 
                f"Data loading ({file_size_mb:.0f}MB)"
            )
            
            logger.info(f"Processing ready: {self.total_rows:,} rows")
            return self
            
        except Exception as e:
            logger.error(f"Chunk processor initialization failed: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        try:
            if self.progress_reporter:
                self.progress_reporter.report_progress()
            
            # Memory cleanup
            self.memory_monitor.force_memory_cleanup()
            
            logger.info("Chunk processor cleanup completed")
            
        except Exception as e:
            logger.warning(f"Chunk processor cleanup failed: {e}")
    
    def _estimate_total_rows(self) -> int:
        """Estimate total row count"""
        try:
            if PYARROW_AVAILABLE:
                try:
                    parquet_file = pq.ParquetFile(self.file_path)
                    total_rows = parquet_file.metadata.num_rows
                    logger.info(f"PyArrow metadata: {total_rows:,} rows")
                    return total_rows
                except Exception as e:
                    logger.warning(f"PyArrow metadata failed: {e}")
            
            # Estimation using pandas
            logger.info("Row count estimation started")
            sample_df = pd.read_parquet(self.file_path, nrows=5000)
            file_size = os.path.getsize(self.file_path)
            sample_memory = sample_df.memory_usage(deep=True).sum()
            
            estimated_rows = int((file_size / sample_memory) * 5000 * 0.7)
            
            del sample_df
            gc.collect()
            
            logger.info(f"Row count estimation completed: {estimated_rows:,} rows")
            return estimated_rows
            
        except Exception as e:
            logger.warning(f"Row count check failed: {e}. Using default value")
            return 1000000
    
    def _adjust_chunk_size(self, pressure: Dict[str, Any]):
        """Adjust chunk size based on memory pressure"""
        try:
            if pressure['should_reduce_batch']:
                if pressure['pressure_level'] == 'abort':
                    self.chunk_size = max(self.min_chunk_size, self.chunk_size // 4)
                elif pressure['pressure_level'] == 'critical':
                    self.chunk_size = max(self.min_chunk_size, self.chunk_size // 3)
                elif pressure['pressure_level'] == 'high':
                    self.chunk_size = max(self.min_chunk_size, self.chunk_size // 2)
                
                logger.info(f"Chunk size adjusted to {self.chunk_size:,} due to memory pressure")
            
        except Exception as e:
            logger.warning(f"Chunk size adjustment failed: {e}")
    
    def _optimize_chunk_memory(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Memory optimization for chunk"""
        try:
            initial_memory = chunk_df.memory_usage(deep=True).sum() / (1024**2)
            
            # Data type optimization
            for col in chunk_df.columns:
                if chunk_df[col].dtype == 'object':
                    try:
                        # Try categorical conversion for string columns
                        if chunk_df[col].nunique() / len(chunk_df) < 0.5:
                            chunk_df[col] = chunk_df[col].astype('category')
                    except Exception:
                        pass
                elif chunk_df[col].dtype == 'int64':
                    # Downcast integers
                    col_min, col_max = chunk_df[col].min(), chunk_df[col].max()
                    if col_min >= -128 and col_max <= 127:
                        chunk_df[col] = chunk_df[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        chunk_df[col] = chunk_df[col].astype('int16')
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        chunk_df[col] = chunk_df[col].astype('int32')
                elif chunk_df[col].dtype == 'float64':
                    # Downcast floats
                    chunk_df[col] = pd.to_numeric(chunk_df[col], downcast='float')
            
            final_memory = chunk_df.memory_usage(deep=True).sum() / (1024**2)
            memory_reduction = (initial_memory - final_memory) / initial_memory * 100
            
            if memory_reduction > 5:
                logger.info(f"Chunk memory optimized: {memory_reduction:.1f}% reduction")
            
            return chunk_df
            
        except Exception as e:
            logger.warning(f"Chunk memory optimization failed: {e}")
            return chunk_df
    
    def _read_chunk_safe(self, start_row: int, chunk_size: int) -> Optional[pd.DataFrame]:
        """Safe chunk reading"""
        try:
            if PYARROW_AVAILABLE:
                parquet_file = pq.ParquetFile(self.file_path)
                
                # Calculate row group ranges
                row_groups = []
                current_row = 0
                
                for rg_idx in range(parquet_file.num_row_groups):
                    rg_rows = parquet_file.metadata.row_group(rg_idx).num_rows
                    if current_row + rg_rows > start_row:
                        if current_row < start_row + chunk_size:
                            row_groups.append(rg_idx)
                    current_row += rg_rows
                    
                    if current_row >= start_row + chunk_size:
                        break
                
                if not row_groups:
                    return None
                
                # Read relevant row groups
                tables = []
                for rg_idx in row_groups:
                    table = parquet_file.read_row_group(rg_idx)
                    tables.append(table)
                
                if len(tables) == 1:
                    chunk_df = tables[0].to_pandas()
                else:
                    combined_table = pa.concat_tables(tables)
                    chunk_df = combined_table.to_pandas()
                
                # Apply row range filtering
                rows_before_start = max(0, start_row - sum(parquet_file.metadata.row_group(i).num_rows for i in range(row_groups[0])))
                if rows_before_start > 0 or len(chunk_df) > chunk_size:
                    end_idx = min(len(chunk_df), rows_before_start + chunk_size)
                    chunk_df = chunk_df.iloc[rows_before_start:end_idx].copy()
                
                return chunk_df
                
            else:
                # Fallback to pandas
                chunk_df = pd.read_parquet(self.file_path, 
                                         engine='pyarrow' if PYARROW_AVAILABLE else 'fastparquet')
                
                end_row = min(start_row + chunk_size, len(chunk_df))
                return chunk_df.iloc[start_row:end_row].copy()
                
        except Exception as e:
            logger.error(f"Chunk reading failed: {e}")
            return None
    
    def process_in_chunks(self) -> pd.DataFrame:
        """Process data in chunks"""
        logger.info(f"Chunk-wise data processing started: {self.total_rows:,} rows")
        
        all_chunks = []
        processed_rows = 0
        chunk_number = 0
        
        try:
            # Initial memory status check
            self.memory_monitor.log_memory_status("Processing start", force=True)
            
            while processed_rows < self.total_rows:
                # Memory pressure check
                pressure = self.memory_monitor.check_memory_pressure()
                
                if pressure['should_abort']:
                    logger.warning(f"Memory limit approaching. Processing stopped: {processed_rows:,} rows")
                    # Continue with intermediate data even in ABORT state
                    if len(all_chunks) > 0:
                        logger.info("Intermediate data available, attempting final combination")
                        break
                    else:
                        logger.error(f"Processing stopped due to memory limit. Processed data: {processed_rows:,} rows")
                        break
                
                # Chunk size adjustment
                self._adjust_chunk_size(pressure)
                
                # Read chunk
                remaining_rows = self.total_rows - processed_rows
                current_chunk_size = min(self.chunk_size, remaining_rows)
                
                try:
                    # Process single chunk
                    chunk_df = self._read_chunk_safe(processed_rows, current_chunk_size)
                    
                    if chunk_df is None or len(chunk_df) == 0:
                        logger.warning(f"Empty chunk: position {processed_rows:,}")
                        processed_rows += current_chunk_size
                        continue
                    
                    # Data type optimization
                    chunk_df = self._optimize_chunk_memory(chunk_df)
                    
                    # Store chunk
                    all_chunks.append(chunk_df)
                    chunk_number += 1
                    processed_rows += len(chunk_df)
                    
                    # Update progress
                    if self.progress_reporter:
                        self.progress_reporter.update(len(chunk_df))
                    
                    logger.info(f"Chunk {chunk_number} processed: {len(chunk_df):,} rows "
                               f"(Memory: {self.memory_monitor.get_process_memory_gb():.1f}GB)")
                    
                    # More frequent memory cleanup
                    if chunk_number % 2 == 0:
                        self.memory_monitor.force_memory_cleanup()
                        self.memory_monitor.log_memory_status(f"Chunk{chunk_number}")
                    
                    # Intermediate combination (more frequent under memory pressure)
                    combine_threshold = 2 if pressure['pressure_level'] in ['high', 'critical', 'abort'] else 4
                    if len(all_chunks) >= combine_threshold:
                        logger.info(f"Performing intermediate combination: {len(all_chunks)} chunks")
                        combined_df = self._combine_chunks_safe(all_chunks)
                        all_chunks = [combined_df] if not combined_df.empty else []
                        self.memory_monitor.force_memory_cleanup(intensive=True)
                    
                except Exception as e:
                    logger.error(f"Chunk {chunk_number + 1} processing failed: {e}")
                    
                    # Error recovery attempt
                    if self._try_recover_chunk(processed_rows, current_chunk_size):
                        continue
                    else:
                        logger.error("Recovery failed. Skipping problematic data")
                        processed_rows += current_chunk_size
                        continue
            
            # Final combination
            if len(all_chunks) == 0:
                logger.error("No data processed successfully")
                return pd.DataFrame()
            
            logger.info(f"Final combination: {len(all_chunks)} chunks")
            final_df = self._combine_chunks_safe(all_chunks)
            
            logger.info(f"Chunk processing completed: {len(final_df):,} rows")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            
            return final_df
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            raise
    
    def _combine_chunks_safe(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Safe chunk combination"""
        if not chunks:
            return pd.DataFrame()
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # Memory efficient combination
            combined = pd.concat(chunks, ignore_index=True, copy=False)
            
            # Clear chunk references
            for chunk in chunks:
                del chunk
            chunks.clear()
            gc.collect()
            
            logger.info(f"Chunks combined successfully: {len(combined):,} rows")
            return combined
            
        except Exception as e:
            logger.error(f"Chunk combination failed: {e}")
            # Return first valid chunk on failure
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    return chunk
            return pd.DataFrame()
    
    def _try_recover_chunk(self, failed_position: int, failed_chunk_size: int) -> bool:
        """Attempt chunk recovery"""
        try:
            logger.info(f"Chunk recovery attempt: position {failed_position:,}")
            
            # Reduce chunk size more aggressively for retry
            recovery_chunk_size = max(self.min_chunk_size, failed_chunk_size // 4)
            
            recovery_df = self._read_chunk_safe(failed_position, recovery_chunk_size)
            
            if recovery_df is not None and not recovery_df.empty:
                logger.info(f"Chunk recovery successful: {len(recovery_df):,} rows")
                self.chunk_size = recovery_chunk_size
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Chunk recovery failed: {e}")
            return False

class StreamingDataLoader:
    """Streaming data loader"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        
        logger.info("Streaming data loader initialization completed")
    
    def load_full_data_streaming(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete processing via streaming"""
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
            
            # Intermediate memory cleanup
            self.memory_monitor.force_memory_cleanup(intensive=True)
            self.memory_monitor.log_memory_status("Training data completed", force=True)
            
            # 2. Test data streaming processing
            logger.info("Test data streaming processing started") 
            test_df = self._stream_process_file(str(self.config.TEST_PATH), is_train=False)
            
            if test_df is None or test_df.empty:
                raise ValueError("Test data streaming processing failed")
            
            # Final memory status
            self.memory_monitor.log_memory_status("Streaming completed", force=True)
            
            logger.info(f"=== Full data streaming completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Streaming data loading failed: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            raise
    
    def _stream_process_file(self, file_path: str, is_train: bool = True) -> pd.DataFrame:
        """File streaming processing"""
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
                
                # Limit sample size (to 3000 rows for memory efficiency)
                if len(sample_df) > 3000:
                    sample_df = sample_df.head(3000)
                
                # Target column detection
                self.target_column = self.column_analyzer.detect_target_column(sample_df)
                logger.info(f"Detected target column: {self.target_column}")
                
                del sample_df, sample_table
                gc.collect()
            
            # Result collection
            all_chunks = []
            processed_rows = 0
            
            # More conservative row group processing
            max_groups_per_batch = 2  # Process fewer row groups at once
            
            for rg_start in range(0, num_row_groups, max_groups_per_batch):
                try:
                    # Memory pressure check
                    pressure = self.memory_monitor.check_memory_pressure()
                    if pressure['should_abort']:
                        logger.warning(f"Memory limit approaching - Processed rows: {processed_rows:,}, continuing attempt")
                        # Force memory cleanup and continue even in ABORT state
                        self.memory_monitor.force_memory_cleanup(intensive=True)
                        
                        # Recheck after cleanup
                        pressure_after = self.memory_monitor.check_memory_pressure()
                        if pressure_after['should_abort']:
                            logger.error(f"Memory limit reached even after cleanup - Stopping: {processed_rows:,} rows")
                            break
                    
                    # Process batch of row groups
                    rg_end = min(rg_start + max_groups_per_batch, num_row_groups)
                    batch_tables = []
                    
                    for rg_idx in range(rg_start, rg_end):
                        table = parquet_file.read_row_group(rg_idx)
                        batch_tables.append(table)
                    
                    # Combine tables in batch
                    if len(batch_tables) == 1:
                        batch_table = batch_tables[0]
                    else:
                        batch_table = pa.concat_tables(batch_tables)
                    
                    chunk_df = batch_table.to_pandas()
                    
                    logger.info(f"Row groups {rg_start+1}-{rg_end} processed: {len(chunk_df):,} rows")
                    
                    # Immediate memory optimization
                    chunk_df = self._optimize_dataframe_memory(chunk_df)
                    
                    # Store chunk
                    all_chunks.append(chunk_df)
                    processed_rows += len(chunk_df)
                    
                    # Memory release
                    del batch_tables, batch_table
                    gc.collect()
                    
                    # More frequent intermediate combination (every 2 chunks for memory efficiency)
                    if len(all_chunks) >= 2:
                        logger.info(f"Performing intermediate combination: {len(all_chunks)} chunks")
                        combined = self._combine_chunks_immediate(all_chunks)
                        all_chunks = [combined]
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
            final_df = self._combine_chunks_immediate(all_chunks)
            
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
                    if df[col].str.replace('.', '').str.isdigit().all():
                        try:
                            df[col] = pd.to_numeric(df[col], downcast='integer')
                            continue
                        except:
                            pass
                    
                    # Convert to category if cardinality is low
                    if df[col].nunique() / len(df) < 0.3:
                        df[col] = df[col].astype('category')
                
                elif col_type in ['int64', 'int32']:
                    # Downcast integers
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                
                elif col_type in ['float64', 'float32']:
                    # Downcast floats
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            final_memory = df.memory_usage(deep=True).sum() / (1024**2)
            memory_reduction = (initial_memory - final_memory) / initial_memory * 100
            
            if memory_reduction > 10:
                logger.info(f"DataFrame memory optimized: {memory_reduction:.1f}% reduction")
            
            return df
            
        except Exception as e:
            logger.warning(f"DataFrame memory optimization failed: {e}")
            return df
    
    def _combine_chunks_immediate(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Immediate chunk combination"""
        if not chunks:
            return pd.DataFrame()
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            logger.info(f"Combining {len(chunks)} chunks...")
            
            # Use concat with minimal memory overhead
            combined = pd.concat(chunks, ignore_index=True, copy=False)
            
            # Clear chunk references immediately
            for chunk in chunks:
                del chunk
            chunks.clear()
            gc.collect()
            
            logger.info(f"Chunk combination successful: {len(combined):,} rows")
            return combined
            
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
MemoryOptimizedChunkReader = DataChunkProcessor
AggressiveMemoryMonitor = MemoryMonitor

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