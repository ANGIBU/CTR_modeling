# data_loader.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import logging
import time
import gc
import psutil
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any, Union
from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring and management utility"""
    
    def __init__(self):
        self.memory_limit_gb = Config.MAX_MEMORY_GB
        self.warning_threshold = Config.MEMORY_WARNING_THRESHOLD
        self.critical_threshold = Config.MEMORY_CRITICAL_THRESHOLD
        self.abort_threshold = Config.MEMORY_ABORT_THRESHOLD
        self.quick_mode = False
    
    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory status"""
        try:
            memory = psutil.virtual_memory()
            used_gb = (memory.total - memory.available) / (1024**3)
            available_gb = memory.available / (1024**3)
            usage_percent = memory.percent / 100.0
            
            # Apply our limit
            effective_available = min(available_gb, self.memory_limit_gb - used_gb)
            if effective_available < 0:
                effective_available = min(5.0, available_gb)  # Emergency minimum
            
            return {
                'used_gb': used_gb,
                'available_gb': effective_available,
                'total_system_gb': memory.total / (1024**3),
                'usage_percent': usage_percent,
                'within_limit': used_gb <= self.memory_limit_gb
            }
        except Exception as e:
            logger.warning(f"Memory status check failed: {e}")
            return {
                'used_gb': 20.0,
                'available_gb': 20.0,
                'total_system_gb': 64.0,
                'usage_percent': 0.5,
                'within_limit': True
            }
    
    def check_memory_safety(self, required_gb: float = 5.0) -> bool:
        """Check if we have enough memory for operation"""
        status = self.get_memory_status()
        
        if not status['within_limit']:
            logger.warning(f"Memory usage exceeds {self.memory_limit_gb}GB limit")
            return False
        
        if status['available_gb'] < required_gb:
            logger.warning(f"Insufficient memory: {status['available_gb']:.1f}GB < {required_gb}GB required")
            return False
        
        return True
    
    def force_cleanup(self):
        """Force memory cleanup"""
        try:
            collected = gc.collect()
            logger.info(f"Memory cleanup: {collected} objects collected")
            return collected
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0
    
    def log_memory_status(self, context: str = ""):
        """Log current memory status with context"""
        status = self.get_memory_status()
        limit_status = "within limit" if status['within_limit'] else "EXCEEDS LIMIT"
        logger.info(f"Memory status ({context}): {status['usage_percent']:.1f}% used, {status['available_gb']:.1f}GB available ({limit_status})")
    
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for testing"""
        self.quick_mode = enabled

class StreamingDataLoader:
    """Memory-efficient streaming data loader with 40GB limit enforcement"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        # Configuration
        self.chunk_size = config.get_optimal_chunk_size()
        self.quick_mode = False
        
        # State tracking
        self.detected_target_column = None
        self.column_info = {}
        self.temp_files = []
        
        # Processing statistics
        self.processing_stats = {
            'start_time': None,
            'total_rows_processed': 0,
            'chunks_processed': 0,
            'memory_cleanups': 0,
            'temp_files_created': 0
        }
        
    def set_quick_mode(self, enabled: bool):
        """Enable quick mode for rapid testing"""
        self.quick_mode = enabled
        if enabled:
            self.chunk_size = min(1000, self.chunk_size)
            logger.info("Quick mode enabled - small chunks for testing")
    
    def _detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect target column in the dataset"""
        try:
            # Known CTR target columns
            target_candidates = ['clicked', 'click', 'target', 'label', 'y']
            
            for col in target_candidates:
                if col in df.columns:
                    unique_values = df[col].nunique()
                    if unique_values == 2:  # Binary classification
                        logger.info(f"Target column detected: {col}")
                        self.detected_target_column = col
                        return col
            
            # Look for binary columns
            binary_columns = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'int32', 'bool']:
                    unique_vals = df[col].unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                        binary_columns.append(col)
            
            if binary_columns:
                # Select the most likely target
                target_col = binary_columns[0]  # Take first binary column
                logger.info(f"Binary target column detected: {target_col}")
                self.detected_target_column = target_col
                return target_col
                
        except Exception as e:
            logger.error(f"Target detection failed: {e}")
        
        return None
    
    def _analyze_parquet_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze parquet file structure efficiently"""
        try:
            # Read metadata only
            parquet_file = pq.ParquetFile(file_path)
            
            info = {
                'total_rows': parquet_file.metadata.num_rows,
                'num_row_groups': parquet_file.num_row_groups,
                'columns': parquet_file.schema_arrow.names,
                'file_size_mb': file_path.stat().st_size / (1024**2)
            }
            
            logger.info(f"File analysis - Total {info['total_rows']:,} rows, {info['num_row_groups']} row groups")
            return info
            
        except Exception as e:
            logger.error(f"File analysis failed: {e}")
            return {'total_rows': 0, 'num_row_groups': 0, 'columns': [], 'file_size_mb': 0}
    
    def _create_temp_file(self, df: pd.DataFrame, prefix: str) -> Path:
        """Create temporary file to store processed data"""
        try:
            temp_dir = Path(tempfile.gettempdir()) / "ctr_modeling"
            temp_dir.mkdir(exist_ok=True)
            
            temp_file = temp_dir / f"{prefix}_{int(time.time())}_{os.getpid()}.parquet"
            df.to_parquet(temp_file, engine='pyarrow', compression='snappy')
            
            self.temp_files.append(temp_file)
            self.processing_stats['temp_files_created'] += 1
            
            logger.info(f"Temp file created: {temp_file.name} ({len(df):,} rows)")
            return temp_file
            
        except Exception as e:
            logger.error(f"Temp file creation failed: {e}")
            raise
    
    def _load_from_temp_file(self, temp_file: Path) -> pd.DataFrame:
        """Load data from temporary file"""
        try:
            df = pd.read_parquet(temp_file, engine='pyarrow')
            logger.info(f"Loaded from temp file: {temp_file.name} ({len(df):,} rows)")
            return df
        except Exception as e:
            logger.error(f"Temp file loading failed: {e}")
            raise
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        cleaned = 0
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        self.temp_files = []
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} temporary files")
    
    def _process_chunk_streaming(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process data chunk with memory efficiency"""
        try:
            # Memory check before processing
            if not self.memory_monitor.check_memory_safety(2.0):
                self.memory_monitor.force_cleanup()
                if not self.memory_monitor.check_memory_safety(2.0):
                    raise MemoryError("Insufficient memory for chunk processing")
            
            # Basic data type optimization
            for col in chunk.columns:
                if chunk[col].dtype == 'object':
                    # Try to convert to category if not too many unique values
                    if chunk[col].nunique() < len(chunk) * 0.5:
                        try:
                            chunk[col] = chunk[col].astype('category')
                        except:
                            pass
                elif chunk[col].dtype == 'float64':
                    # Downcast float64 to float32 where possible
                    try:
                        chunk[col] = pd.to_numeric(chunk[col], downcast='float')
                    except:
                        pass
                elif chunk[col].dtype == 'int64':
                    # Downcast int64 where possible
                    try:
                        chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
                    except:
                        pass
            
            return chunk
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return chunk
    
    def load_parquet_streaming(self, file_path: Path, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load parquet file with memory-efficient streaming"""
        try:
            self.processing_stats['start_time'] = time.time()
            self.memory_monitor.log_memory_status("Streaming start")
            
            # Analyze file first
            file_info = self._analyze_parquet_file(file_path)
            if file_info['total_rows'] == 0:
                return None
            
            # Determine processing strategy
            total_rows = file_info['total_rows']
            if max_rows:
                total_rows = min(total_rows, max_rows)
            
            # Memory-based chunk size adjustment
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['available_gb'] < 15:
                self.chunk_size = min(self.chunk_size, 25000)
                logger.info(f"Reduced chunk size to {self.chunk_size} due to low memory")
            
            processed_chunks = []
            rows_processed = 0
            temp_file_path = None
            
            # Process in chunks
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                if max_rows and rows_processed >= max_rows:
                    break
                
                # Convert to pandas DataFrame
                chunk_df = batch.to_pandas()
                
                # Limit rows if needed
                if max_rows and rows_processed + len(chunk_df) > max_rows:
                    chunk_df = chunk_df.head(max_rows - rows_processed)
                
                # Detect target column in first chunk
                if self.detected_target_column is None and len(chunk_df) > 0:
                    self._detect_target_column(chunk_df)
                
                # Process chunk
                processed_chunk = self._process_chunk_streaming(chunk_df)
                
                # Memory management
                if memory_status['available_gb'] < 10 or len(processed_chunks) > 5:
                    # Save to temp file and clear memory
                    if processed_chunks:
                        combined_chunk = pd.concat(processed_chunks, ignore_index=True)
                        temp_file_path = self._create_temp_file(combined_chunk, "streaming_data")
                        processed_chunks = []
                        del combined_chunk
                        self.memory_monitor.force_cleanup()
                
                processed_chunks.append(processed_chunk)
                rows_processed += len(processed_chunk)
                self.processing_stats['chunks_processed'] += 1
                
                # Progress logging
                if self.processing_stats['chunks_processed'] % 10 == 0:
                    progress = (rows_processed / total_rows) * 100
                    logger.info(f"Progress: {progress:.1f}% ({rows_processed:,} rows)")
                    self.memory_monitor.log_memory_status("Processing")
                
                # Memory safety check
                if not self.memory_monitor.check_memory_safety(5.0):
                    logger.warning("Memory safety threshold reached")
                    break
            
            # Combine final result
            final_df = None
            if processed_chunks:
                final_df = pd.concat(processed_chunks, ignore_index=True)
            
            # If we have a temp file, combine with it
            if temp_file_path and temp_file_path.exists():
                temp_df = self._load_from_temp_file(temp_file_path)
                if final_df is not None:
                    final_df = pd.concat([temp_df, final_df], ignore_index=True)
                else:
                    final_df = temp_df
                del temp_df
            
            # Final cleanup
            self.memory_monitor.force_cleanup()
            self.processing_stats['total_rows_processed'] = rows_processed
            
            logger.info(f"File streaming completed: {rows_processed:,} rows")
            self.memory_monitor.log_memory_status("Streaming completed")
            
            return final_df
            
        except Exception as e:
            logger.error(f"Streaming load failed: {e}")
            return None
        finally:
            self._cleanup_temp_files()
    
    def load_full_data_streaming(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load full training and test data with streaming"""
        logger.info("=== Full data streaming loading started ===")
        
        try:
            # Check memory before starting
            if not self.memory_monitor.check_memory_safety(10.0):
                logger.error("Insufficient memory for full data loading")
                return None, None
            
            train_df = None
            test_df = None
            
            # Load training data
            logger.info("Training data streaming processing started")
            if self.config.TRAIN_PATH.exists():
                max_train_rows = None if not self.quick_mode else Config.QUICK_SAMPLE_SIZE
                train_df = self.load_parquet_streaming(self.config.TRAIN_PATH, max_train_rows)
                
                if train_df is not None:
                    # Save to temp file and clear from memory for large datasets
                    if len(train_df) > 1000000:  # More than 1M rows
                        train_temp_file = self._create_temp_file(train_df, "train_processed")
                        del train_df
                        self.memory_monitor.force_cleanup()
                        self.memory_monitor.log_memory_status("Training data saved, memory cleared")
                    
            # Load test data
            logger.info("Test data streaming processing started")
            if self.config.TEST_PATH.exists():
                max_test_rows = None if not self.quick_mode else Config.QUICK_TEST_SIZE
                test_df = self.load_parquet_streaming(self.config.TEST_PATH, max_test_rows)
            
            # Reload training data if it was saved to temp
            if train_df is None and hasattr(self, 'train_temp_file'):
                logger.info("Reloading training data from temporary storage")
                train_df = self._load_from_temp_file(self.train_temp_file)
            
            self.memory_monitor.log_memory_status("Streaming completed")
            
            if train_df is not None and test_df is not None:
                logger.info(f"=== Full data streaming completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
                return train_df, test_df
            else:
                logger.error("Data loading failed")
                return None, None
                
        except Exception as e:
            logger.error(f"Full data streaming failed: {e}")
            return None, None
        finally:
            self._cleanup_temp_files()
    
    def get_detected_target_column(self) -> Optional[str]:
        """Get detected target column name"""
        return self.detected_target_column

class LargeDataLoader:
    """Main data loader interface with memory management"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.quick_mode = False
        self.target_column = None
        
        # Loading statistics
        self.loading_stats = {
            'start_time': time.time(),
            'data_loaded': False,
            'quick_mode': False,
            'train_rows': 0,
            'test_rows': 0,
            'loading_time': 0,
            'memory_usage': 0
        }
        
        logger.info("Large data loader initialization completed")
    
    def set_quick_mode(self, enabled: bool):
        """Enable quick mode for testing"""
        self.quick_mode = enabled
        self.loading_stats['quick_mode'] = enabled
        if enabled:
            logger.info("Quick mode enabled for data loader")
    
    def load_quick_sample_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load small sample data for quick testing"""
        try:
            logger.info(f"Quick mode: Loading {Config.QUICK_SAMPLE_SIZE} training and {Config.QUICK_TEST_SIZE} test samples")
            
            # Use streaming loader in quick mode
            streaming_loader = StreamingDataLoader(self.config)
            streaming_loader.set_quick_mode(True)
            
            train_df = streaming_loader.load_parquet_streaming(
                self.config.TRAIN_PATH, 
                max_rows=Config.QUICK_SAMPLE_SIZE
            )
            
            test_df = streaming_loader.load_parquet_streaming(
                self.config.TEST_PATH, 
                max_rows=Config.QUICK_TEST_SIZE
            )
            
            # Store detected target column
            self.target_column = streaming_loader.get_detected_target_column()
            
            # Update statistics
            self.loading_stats.update({
                'data_loaded': True,
                'train_rows': len(train_df) if train_df is not None else 0,
                'test_rows': len(test_df) if test_df is not None else 0,
                'loading_time': time.time() - self.loading_stats['start_time']
            })
            
            logger.info(f"Quick sample loading completed - Train: {len(train_df) if train_df is not None else 0}, Test: {len(test_df) if test_df is not None else 0}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Quick sample loading failed: {e}")
            return None, None
    
    def load_large_data_optimized(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load complete dataset with memory optimization"""
        logger.info("=== Complete data processing started ===")
        
        # Use streaming loader
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
        print(f"Training data: {train_df.shape if train_df is not None else 'None'}")
        print(f"Test data: {test_df.shape if test_df is not None else 'None'}")
        print(f"Detected target column: {loader.get_detected_target_column()}")
        print(f"Total samples: {(len(train_df) if train_df is not None else 0) + (len(test_df) if test_df is not None else 0)}")
        
        # Test memory status
        memory_monitor = MemoryMonitor()
        memory_monitor.log_memory_status("After quick test")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")