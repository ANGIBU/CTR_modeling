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
    """Progress reporting class"""
    
    def __init__(self, total_items: int, description: str = "Progress"):
        self.total_items = total_items
        self.current_items = 0
        self.description = description
        self.start_time = time.time()
        self.last_report_time = 0
        self.report_interval = 5.0
        
    def update(self, increment: int = 1):
        """Update progress"""
        self.current_items += increment
        current_time = time.time()
        
        if current_time - self.last_report_time >= self.report_interval or self.current_items >= self.total_items:
            self.report_progress()
            self.last_report_time = current_time
    
    def report_progress(self):
        """Report progress"""
        if self.total_items <= 0:
            return
            
        progress_percent = min(100.0, (self.current_items / self.total_items) * 100)
        elapsed_time = time.time() - self.start_time
        
        if progress_percent > 0 and elapsed_time > 0:
            estimated_total_time = elapsed_time * (100 / progress_percent)
            remaining_time = max(0, estimated_total_time - elapsed_time)
            
            items_per_sec = self.current_items / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"{self.description}: {progress_percent:.1f}% "
                       f"({self.current_items:,}/{self.total_items:,}) - "
                       f"Speed: {items_per_sec:,.0f} rows/sec - "
                       f"Remaining time: {remaining_time:.0f}s")
        else:
            logger.info(f"{self.description}: {progress_percent:.1f}% ({self.current_items:,}/{self.total_items:,})")

class MemoryMonitor:
    """Memory monitoring and management"""
    
    def __init__(self, max_memory_gb: float = 45.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.max_memory_gb = max_memory_gb
        self.memory_history = []
        self.peak_memory = 0.0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Set memory thresholds
        self.warning_threshold = max_memory_gb * 0.75
        self.critical_threshold = max_memory_gb * 0.85
        self.abort_threshold = max_memory_gb * 0.95
        
        logger.info(f"Memory thresholds set - Warning: {self.warning_threshold:.1f}GB, "
                   f"Critical: {self.critical_threshold:.1f}GB, Abort: {self.abort_threshold:.1f}GB")
        
    def get_process_memory_gb(self) -> float:
        """Current process memory usage (GB)"""
        if not self.monitoring_enabled:
            return 2.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception:
            return 2.0
    
    def get_available_memory_gb(self) -> float:
        """System available memory (GB)"""
        if not self.monitoring_enabled:
            return 30.0
        
        try:
            return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 30.0
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check memory pressure status"""
        process_memory = self.get_process_memory_gb()
        available_memory = self.get_available_memory_gb()
        
        # Determine pressure level
        if process_memory > self.abort_threshold or available_memory < 2:
            pressure_level = "abort"
        elif process_memory > self.critical_threshold or available_memory < 5:
            pressure_level = "critical" 
        elif process_memory > self.warning_threshold or available_memory < 8:
            pressure_level = "high"
        elif available_memory < 15:
            pressure_level = "moderate"
        else:
            pressure_level = "low"
        
        return {
            'memory_pressure': pressure_level in ['high', 'critical', 'abort'],
            'pressure_level': pressure_level,
            'process_memory_gb': process_memory,
            'available_memory_gb': available_memory,
            'should_reduce_chunk_size': pressure_level in ['moderate', 'high', 'critical', 'abort'],
            'should_force_gc': pressure_level in ['high', 'critical', 'abort'],
            'should_abort': pressure_level == 'abort'
        }
    
    def log_memory_status(self, context: str = "", force: bool = False):
        """Log memory status"""
        try:
            pressure = self.check_memory_pressure()
            
            if force or pressure['pressure_level'] != 'low':
                logger.info(f"Memory [{context}]: Process {pressure['process_memory_gb']:.1f}GB, "
                           f"Available {pressure['available_memory_gb']:.1f}GB - "
                           f"{pressure['pressure_level'].upper()}")
            
            if pressure['memory_pressure']:
                logger.warning(f"Memory pressure: Process {pressure['process_memory_gb']:.1f}GB, "
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
                cleanup_rounds = 15
                sleep_time = 0.5
            elif pressure['pressure_level'] == 'critical':
                cleanup_rounds = 12
                sleep_time = 0.3
            elif pressure['pressure_level'] == 'high':
                cleanup_rounds = 8
                sleep_time = 0.2
            else:
                cleanup_rounds = 5
                sleep_time = 0.1
            
            # Garbage collection
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Intermediate check
                if i % 5 == 0:
                    current_memory = self.get_process_memory_gb()
                    if initial_memory - current_memory > 2.0:
                        break
            
            # Windows memory cleanup
            if cleanup_rounds >= 8:
                try:
                    import ctypes
                    if hasattr(ctypes, 'windll'):
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                        time.sleep(0.5)
                        gc.collect()
                except Exception:
                    pass
            
            final_memory = self.get_process_memory_gb()
            memory_freed = initial_memory - final_memory
            
            if memory_freed > 0.1:
                logger.info(f"Memory cleanup: {memory_freed:.2f}GB freed ({cleanup_rounds} rounds)")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0.0

class DataColumnAnalyzer:
    """Data column analyzer"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_candidates = config.TARGET_COLUMN_CANDIDATES
        self.detection_config = config.TARGET_DETECTION_CONFIG
        
    def analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataframe columns"""
        try:
            analysis = {
                'total_columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': {},
                'target_candidates': [],
                'binary_columns': [],
                'categorical_columns': [],
                'numeric_columns': [],
                'id_columns': []
            }
            
            logger.info(f"Column analysis started: {len(df.columns)} columns")
            
            # Detailed analysis for each column
            for col in df.columns:
                try:
                    unique_count = df[col].nunique()
                    analysis['unique_counts'][col] = unique_count
                    
                    # Column type classification
                    dtype_str = str(df[col].dtype)
                    col_lower = col.lower()
                    
                    # ID column identification
                    if any(pattern in col_lower for pattern in ['id', 'uuid', 'key', 'hash']):
                        unique_ratio = unique_count / len(df)
                        if unique_ratio > 0.9:
                            analysis['id_columns'].append(col)
                            continue
                    
                    # Binary column identification (target candidates)
                    if unique_count == 2:
                        unique_values = df[col].dropna().unique()
                        if set(unique_values).issubset(self.detection_config['binary_values']):
                            analysis['binary_columns'].append(col)
                            
                            # CTR characteristics check
                            positive_ratio = df[col].mean()
                            if (self.detection_config['min_ctr'] <= positive_ratio <= 
                                self.detection_config['max_ctr']):
                                analysis['target_candidates'].append({
                                    'column': col,
                                    'ctr': positive_ratio,
                                    'distribution': df[col].value_counts().to_dict()
                                })
                    
                    # Numeric/categorical classification
                    if dtype_str in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                   'float16', 'float32', 'float64']:
                        analysis['numeric_columns'].append(col)
                    else:
                        analysis['categorical_columns'].append(col)
                        
                except Exception as e:
                    logger.warning(f"Column {col} analysis failed: {e}")
            
            # Sort target candidates by CTR
            analysis['target_candidates'].sort(key=lambda x: x['ctr'])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Column analysis failed: {e}")
            return {'error': str(e)}
    
    def detect_target_column(self, df: pd.DataFrame, provided_target: str = None) -> str:
        """Detect target column"""
        try:
            analysis = self.analyze_columns(df)
            
            # Logging
            logger.info(f"Data column structure:")
            logger.info(f"  - Total columns: {analysis['total_columns']}")
            logger.info(f"  - Numeric: {len(analysis['numeric_columns'])} columns")
            logger.info(f"  - Categorical: {len(analysis['categorical_columns'])} columns")
            logger.info(f"  - Binary: {len(analysis['binary_columns'])} columns")
            logger.info(f"  - ID: {len(analysis['id_columns'])} columns")
            logger.info(f"  - Target candidates: {len(analysis['target_candidates'])} columns")
            
            # Column name logging (first 20 only)
            sample_columns = analysis['column_names'][:20]
            if len(analysis['column_names']) > 20:
                sample_columns.append(f"... (+{len(analysis['column_names']) - 20} more)")
            logger.info(f"  - Column name samples: {sample_columns}")
            
            # Target candidate detailed logging
            if analysis['target_candidates']:
                logger.info("Target candidate analysis:")
                for candidate in analysis['target_candidates']:
                    logger.info(f"  - {candidate['column']}: CTR {candidate['ctr']:.4f}, "
                               f"Distribution {candidate['distribution']}")
            
            # Check provided target column
            if provided_target and provided_target in df.columns:
                if provided_target in [c['column'] for c in analysis['target_candidates']]:
                    logger.info(f"Provided target column confirmed: {provided_target}")
                    return provided_target
                else:
                    logger.warning(f"Provided target column '{provided_target}' does not meet binary classification criteria")
            
            # Automatic detection
            for candidate in self.target_candidates:
                if candidate in df.columns:
                    for target_info in analysis['target_candidates']:
                        if target_info['column'] == candidate:
                            logger.info(f"Target column auto-detected: {candidate}")
                            return candidate
            
            # Select columns within CTR range
            for candidate in analysis['target_candidates']:
                ctr = candidate['ctr']
                if (self.detection_config['typical_ctr_range'][0] <= ctr <= 
                    self.detection_config['typical_ctr_range'][1]):
                    logger.info(f"CTR pattern based target detection: {candidate['column']} (CTR: {ctr:.4f})")
                    return candidate['column']
            
            # Select lowest CTR column
            if analysis['target_candidates']:
                best_candidate = analysis['target_candidates'][0]
                logger.info(f"Lowest CTR target selected: {best_candidate['column']} (CTR: {best_candidate['ctr']:.4f})")
                return best_candidate['column']
            
            # Default
            default_target = provided_target or 'clicked'
            logger.warning(f"Target column detection failed, using default: {default_target}")
            return default_target
            
        except Exception as e:
            logger.error(f"Target column detection failed: {e}")
            return provided_target or 'clicked'

class DataChunkProcessor:
    """Data chunk processor"""
    
    def __init__(self, file_path: str, chunk_size: int = 25000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.total_rows = 0
        self.memory_monitor = MemoryMonitor()
        self.progress_reporter = None
        
        # Chunk size range adjustment
        self.min_chunk_size = 5000
        self.max_chunk_size = 100000
        
    def __enter__(self):
        """Initialization"""
        logger.info(f"Data chunk processor initialization: {self.file_path}")
        
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
                    combine_threshold = 3 if pressure['pressure_level'] in ['high', 'critical'] else 5
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
                        logger.error("Recovery failed. Processing stopped")
                        break
            
            # Final data combination
            if all_chunks:
                logger.info(f"Final data combination: {len(all_chunks)} chunks")
                final_df = self._combine_chunks_safe(all_chunks)
                
                if final_df is not None and not final_df.empty:
                    logger.info(f"Chunk processing completed: {final_df.shape}")
                    return final_df
                else:
                    raise ValueError("Final data combination result is empty")
            else:
                raise ValueError("No processed chunks available")
                
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            raise
    
    def _adjust_chunk_size(self, pressure: Dict[str, Any]):
        """Adjust chunk size"""
        old_size = self.chunk_size
        
        if pressure['pressure_level'] == 'abort':
            self.chunk_size = self.min_chunk_size
        elif pressure['pressure_level'] == 'critical':
            self.chunk_size = max(self.min_chunk_size, self.chunk_size // 5)
        elif pressure['pressure_level'] == 'high':
            self.chunk_size = max(self.min_chunk_size, self.chunk_size // 3)
        elif pressure['pressure_level'] == 'moderate':
            self.chunk_size = max(self.min_chunk_size, int(self.chunk_size * 0.7))
        elif pressure['pressure_level'] == 'low':
            self.chunk_size = min(self.max_chunk_size, int(self.chunk_size * 1.2))
        
        if old_size != self.chunk_size:
            logger.info(f"Chunk size adjusted: {old_size:,} → {self.chunk_size:,}")
    
    def _read_chunk_safe(self, start_row: int, num_rows: int) -> Optional[pd.DataFrame]:
        """Safe chunk reading"""
        try:
            # PyArrow chunk reading
            if PYARROW_AVAILABLE:
                try:
                    # Read in smaller batches
                    batch_size = min(num_rows, 10000)
                    batches = []
                    
                    for i in range(0, num_rows, batch_size):
                        current_batch_size = min(batch_size, num_rows - i)
                        current_start = start_row + i
                        
                        try:
                            # Use pandas skiprows/nrows approach
                            batch_df = pd.read_parquet(
                                self.file_path,
                                engine='pyarrow'
                            )
                            
                            # Extract requested range
                            end_row = min(current_start + current_batch_size, len(batch_df))
                            if current_start >= len(batch_df):
                                break
                            
                            batch_chunk = batch_df.iloc[current_start:end_row].copy()
                            batches.append(batch_chunk)
                            
                            # Memory release
                            del batch_df
                            gc.collect()
                            
                            # Early termination under memory pressure
                            if self.memory_monitor.check_memory_pressure()['should_abort']:
                                logger.warning("Memory pressure detected, stopping batch reading")
                                break
                                
                        except Exception as e:
                            logger.warning(f"Batch {i} reading failed: {e}")
                            break
                    
                    if batches:
                        result_df = pd.concat(batches, ignore_index=True)
                        for batch in batches:
                            del batch
                        del batches
                        gc.collect()
                        return result_df
                    else:
                        return None
                        
                except Exception as e:
                    logger.warning(f"PyArrow chunk reading failed: {e}")
            
            # Default pandas approach
            try:
                df = pd.read_parquet(self.file_path, engine='pyarrow')
                
                end_row = min(start_row + num_rows, len(df))
                if start_row >= len(df):
                    return None
                
                result_df = df.iloc[start_row:end_row].copy()
                
                # Memory release
                del df
                gc.collect()
                
                return result_df
                
            except Exception as e:
                logger.error(f"Pandas chunk reading failed: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Chunk reading failed (position: {start_row:,}, size: {num_rows:,}): {e}")
            return None
    
    def _optimize_chunk_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chunk memory optimization"""
        if df is None or df.empty:
            return df
        
        try:
            # Integer optimization
            for col in df.select_dtypes(include=['int64', 'int32']).columns:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= 0:
                        if col_max <= 255:
                            df[col] = df[col].astype('uint8')
                        elif col_max <= 65535:
                            df[col] = df[col].astype('uint16')
                        else:
                            df[col] = df[col].astype('uint32')
                    else:
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype('int8')
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype('int16')
                        else:
                            df[col] = df[col].astype('int32')
                            
                except Exception:
                    try:
                        df[col] = df[col].astype('int32')
                    except Exception:
                        pass
            
            # Float optimization
            for col in df.select_dtypes(include=['float64']).columns:
                try:
                    df[col] = df[col].astype('float32')
                except Exception:
                    pass
            
            # Categorical optimization
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    # Stricter criteria
                    if unique_count < total_count * 0.3 and unique_count < 50:
                        df[col] = df[col].astype('category')
                except Exception:
                    pass
            
            return df
            
        except Exception as e:
            logger.warning(f"Chunk optimization failed: {e}")
            return df
    
    def _combine_chunks_safe(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Safe chunk combination"""
        if not chunks:
            return pd.DataFrame()
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # Filter valid DataFrames only
            valid_chunks = []
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                logger.warning("No valid chunks available")
                return pd.DataFrame()
            
            # Batch-wise combination
            batch_size = 2
            combined_chunks = []
            
            for i in range(0, len(valid_chunks), batch_size):
                try:
                    batch = valid_chunks[i:i + batch_size]
                    
                    if len(batch) == 1:
                        combined_chunks.append(batch[0])
                    else:
                        # Batch combination
                        batch_combined = pd.concat(batch, ignore_index=True)
                        batch_combined = self._optimize_chunk_memory(batch_combined)
                        combined_chunks.append(batch_combined)
                    
                    # Release original chunks
                    for chunk in batch:
                        del chunk
                    
                    gc.collect()
                    
                    # Intermediate cleanup under memory pressure
                    if self.memory_monitor.check_memory_pressure()['should_force_gc']:
                        self.memory_monitor.force_memory_cleanup()
                    
                except Exception as e:
                    logger.warning(f"Batch combination failed: {e}")
                    # Keep failed batch as individual chunks
                    for chunk in batch:
                        if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                            combined_chunks.append(chunk)
            
            # Final combination
            if len(combined_chunks) == 1:
                final_df = combined_chunks[0]
            else:
                final_df = pd.concat(combined_chunks, ignore_index=True)
                final_df = self._optimize_chunk_memory(final_df)
            
            # Memory cleanup
            for chunk in combined_chunks:
                del chunk
            del combined_chunks
            gc.collect()
            
            logger.info(f"Chunk combination completed: {final_df.shape}")
            return final_df
            
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
            recovery_chunk_size = max(self.min_chunk_size, failed_chunk_size // 3)
            
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
                # Remove batch_size parameter
                sample_table = parquet_file.read_row_group(0)
                sample_df = sample_table.to_pandas()
                
                # Limit sample size (to 5000 rows)
                if len(sample_df) > 5000:
                    sample_df = sample_df.head(5000)
                
                # Target column detection
                self.target_column = self.column_analyzer.detect_target_column(sample_df)
                logger.info(f"Detected target column: {self.target_column}")
                
                del sample_df, sample_table
                gc.collect()
            
            # Result collection
            all_chunks = []
            processed_rows = 0
            
            # Streaming processing by row group
            for rg_idx in range(num_row_groups):
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
                    
                    # Read row group (remove batch_size parameter)
                    table = parquet_file.read_row_group(rg_idx)
                    chunk_df = table.to_pandas()
                    
                    logger.info(f"Row group {rg_idx+1}/{num_row_groups} processed: {len(chunk_df):,} rows")
                    
                    # Immediate memory optimization
                    chunk_df = self._optimize_dataframe_aggressive(chunk_df)
                    
                    # Store chunk
                    all_chunks.append(chunk_df)
                    processed_rows += len(chunk_df)
                    
                    # Memory release
                    del table
                    gc.collect()
                    
                    # More frequent intermediate combination
                    if len(all_chunks) >= 3:
                        logger.info(f"Performing intermediate combination: {len(all_chunks)} chunks")
                        combined = self._combine_chunks_immediate(all_chunks)
                        all_chunks = [combined]
                        self.memory_monitor.force_memory_cleanup()
                    
                    # Progress output
                    progress = (rg_idx + 1) / num_row_groups * 100
                    if rg_idx % 5 == 0 or rg_idx == num_row_groups - 1:
                        logger.info(f"Progress: {progress:.1f}% ({processed_rows:,}/{total_rows:,} rows)")
                    
                except Exception as e:
                    logger.error(f"Row group {rg_idx} processing failed: {e}")
                    # Continue despite failure
                    continue
            
            # Final combination
            if not all_chunks:
                raise ValueError("No processed data available")
            
            logger.info(f"Final combination: {len(all_chunks)} chunks")
            final_df = self._combine_chunks_immediate(all_chunks)
            
            # Target column check (for training data)
            if is_train and self.target_column:
                if self.target_column in final_df.columns:
                    # CTR check
                    target_ctr = final_df[self.target_column].mean()
                    logger.info(f"Actual CTR: {target_ctr:.4f}")
                else:
                    logger.warning(f"Detected target column '{self.target_column}' not found in final data")
            
            logger.info(f"Streaming processing completed: {final_df.shape} ({processed_rows:,} rows processed)")
            
            return final_df
            
        except Exception as e:
            logger.error(f"File streaming processing failed: {e}")
            raise
    
    def _combine_chunks_immediate(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Immediate chunk combination"""
        if not chunks:
            return pd.DataFrame()
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # Select valid chunks only
            valid_chunks = [chunk for chunk in chunks if isinstance(chunk, pd.DataFrame) and not chunk.empty]
            
            if not valid_chunks:
                return pd.DataFrame()
            
            # Direct combination
            combined_df = pd.concat(valid_chunks, ignore_index=True)
            
            # Immediate optimization
            combined_df = self._optimize_dataframe_aggressive(combined_df)
            
            # Release original chunks
            for chunk in chunks:
                del chunk
            del chunks, valid_chunks
            gc.collect()
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Chunk combination failed: {e}")
            # Return first valid chunk on failure
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    return chunk
            return pd.DataFrame()
    
    def _optimize_dataframe_aggressive(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame optimization"""
        if df is None or df.empty:
            return df
        
        try:
            # Integer optimization
            for col in df.select_dtypes(include=['int64', 'int32']).columns:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= 0:
                        if col_max <= 255:
                            df[col] = df[col].astype('uint8')
                        elif col_max <= 65535:
                            df[col] = df[col].astype('uint16')
                        else:
                            df[col] = df[col].astype('uint32')
                    else:
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype('int8')
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype('int16')
                        else:
                            df[col] = df[col].astype('int32')
                            
                except Exception:
                    try:
                        df[col] = df[col].astype('int32')
                    except Exception:
                        pass
            
            # Float optimization
            for col in df.select_dtypes(include=['float64']).columns:
                try:
                    df[col] = df[col].astype('float32')
                except Exception:
                    pass
            
            # Categorical optimization
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    # Very low cardinality only
                    if unique_count < 50 and unique_count < total_count * 0.05:
                        df[col] = df[col].astype('category')
                    else:
                        # Attempt numeric conversion
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_series.isna().all():
                            df[col] = numeric_series.fillna(0).astype('float32')
                        else:
                            # Hash conversion
                            df[col] = df[col].astype(str).apply(lambda x: hash(x) % 50000).astype('int32')
                except Exception:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
                    except Exception:
                        pass
            
            # Handle missing values and infinite values
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], [0, 0])
            
            return df
            
        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return df
    
    def _validate_files(self) -> bool:
        """File validation"""
        try:
            train_exists = self.config.TRAIN_PATH.exists()
            test_exists = self.config.TEST_PATH.exists()
            
            if not train_exists or not test_exists:
                return False
            
            train_size_mb = self.config.TRAIN_PATH.stat().st_size / (1024**2)
            test_size_mb = self.config.TEST_PATH.stat().st_size / (1024**2)
            
            logger.info(f"File sizes - Training: {train_size_mb:.1f}MB, Test: {test_size_mb:.1f}MB")
            
            return train_size_mb > 10 and test_size_mb > 10
            
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