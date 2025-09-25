# data_loader.py

import pandas as pd
import numpy as np
import logging
import time
import gc
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pyarrow.parquet as pq
from config import Config

# Safe imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring limited")

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Enhanced memory monitoring class with accurate tracking"""
    
    def __init__(self, max_memory_gb: float = 40.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.max_memory_gb = max_memory_gb
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.initial_memory = None
        self.peak_memory = 0.0
        
        if self.monitoring_enabled:
            self.initial_memory = self.process.memory_info().rss / (1024**3)
            logger.info(f"Memory monitor initialized - Max limit: {max_memory_gb}GB")
        else:
            logger.warning("Memory monitoring disabled - psutil not available")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Return detailed memory status"""
        if not self.monitoring_enabled:
            return {
                'usage_gb': 2.0,
                'available_gb': 38.0,
                'process_gb': 1.0,
                'system_percent': 5.0,
                'level': 'normal',
                'should_cleanup': False,
                'within_limit': True
            }
        
        try:
            # System memory
            vm = psutil.virtual_memory()
            system_total_gb = vm.total / (1024**3)
            system_used_gb = (vm.total - vm.available) / (1024**3)
            system_available_gb = vm.available / (1024**3)
            system_percent = vm.percent
            
            # Process memory
            process_memory = self.process.memory_info()
            process_gb = process_memory.rss / (1024**3)
            
            # Track peak memory
            if process_gb > self.peak_memory:
                self.peak_memory = process_gb
            
            # Determine status level based on our 40GB limit
            within_limit = process_gb <= self.max_memory_gb
            
            if process_gb > self.max_memory_gb * 0.95 or system_available_gb < 2:
                level = "critical"
            elif process_gb > self.max_memory_gb * 0.85 or system_available_gb < 5:
                level = "warning" 
            elif process_gb > self.max_memory_gb * 0.70:
                level = "moderate"
            else:
                level = "normal"
            
            should_cleanup = level in ['warning', 'critical'] or not within_limit
            
            return {
                'usage_gb': system_used_gb,
                'available_gb': system_available_gb,
                'process_gb': process_gb,
                'system_percent': system_percent,
                'system_total_gb': system_total_gb,
                'peak_memory_gb': self.peak_memory,
                'level': level,
                'should_cleanup': should_cleanup,
                'within_limit': within_limit,
                'limit_gb': self.max_memory_gb
            }
            
        except Exception as e:
            logger.error(f"Memory status check failed: {e}")
            return {
                'usage_gb': 5.0,
                'available_gb': 35.0,
                'process_gb': 2.0,
                'system_percent': 10.0,
                'level': 'unknown',
                'should_cleanup': True,
                'within_limit': False
            }
    
    def log_memory_status(self, context: str = ""):
        """Log current memory status"""
        status = self.get_memory_status()
        
        log_msg = f"Memory status ({context}): " \
                 f"{status['system_percent']:.1f}% used, " \
                 f"{status['available_gb']:.1f}GB available"
        
        if self.monitoring_enabled:
            log_msg += f", Process: {status['process_gb']:.1f}GB"
            if status['peak_memory_gb'] > 0:
                log_msg += f", Peak: {status['peak_memory_gb']:.1f}GB"
            log_msg += f", Limit: {status['limit_gb']:.1f}GB"
        
        if status['level'] == 'critical':
            logger.error(log_msg)
        elif status['level'] == 'warning':
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        return status
    
    def check_memory_limit(self, action: str = "") -> bool:
        """Check if within memory limits"""
        status = self.get_memory_status()
        
        if not status['within_limit']:
            logger.error(f"Memory limit exceeded during {action}: "
                        f"{status['process_gb']:.1f}GB > {self.max_memory_gb}GB")
            return False
        
        if status['level'] == 'critical':
            logger.warning(f"Critical memory usage during {action}: "
                          f"{status['process_gb']:.1f}GB")
            return False
        
        return True
    
    def force_cleanup(self):
        """Force memory cleanup"""
        initial_status = self.get_memory_status()
        
        # Multiple cleanup attempts
        for i in range(3):
            gc.collect()
            time.sleep(0.1)
        
        final_status = self.get_memory_status()
        
        if self.monitoring_enabled:
            freed = initial_status['process_gb'] - final_status['process_gb']
            if freed > 0.1:  # Only log if significant cleanup
                logger.info(f"Memory cleanup completed: {freed:.2f}GB freed")
        
        return final_status

class CTRTargetDetector:
    """CTR target column detection with enhanced validation"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_config = config.TARGET_DETECTION_CONFIG
        self.candidates = config.TARGET_COLUMN_CANDIDATES
    
    def detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect CTR target column from dataframe"""
        logger.info("Target column detection started")
        
        # Check for obvious candidates first
        for candidate in self.candidates:
            if candidate in df.columns:
                if self._validate_ctr_target(df[candidate]):
                    logger.info(f"Target column detected: {candidate}")
                    return candidate
        
        # Check all binary columns
        binary_columns = []
        for col in df.columns:
            if self._is_binary_column(df[col]):
                binary_columns.append(col)
        
        if binary_columns:
            # Select best CTR column from binary candidates
            best_column = self._select_best_ctr_column(df, binary_columns)
            if best_column:
                logger.info(f"Detected target column: {best_column}")
                return best_column
        
        logger.warning("No suitable CTR target column found")
        return None
    
    def _is_binary_column(self, series: pd.Series) -> bool:
        """Check if column is binary"""
        unique_values = set(series.dropna().unique())
        return unique_values.issubset(self.target_config['binary_values'])
    
    def _validate_ctr_target(self, series: pd.Series) -> bool:
        """Validate if column is suitable as CTR target"""
        try:
            if not self._is_binary_column(series):
                return False
            
            # Calculate CTR
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
                
                # Score based on typical CTR range
                typical_range = self.target_config['typical_ctr_range']
                
                if typical_range[0] <= ctr <= typical_range[1]:
                    scores[col] = 100
                else:
                    if ctr < typical_range[0]:
                        distance = typical_range[0] - ctr
                    else:
                        distance = ctr - typical_range[1]
                    
                    scores[col] = max(0, 100 - (distance * 1000))
            
            if scores:
                best_col = max(scores.keys(), key=lambda k: scores[k])
                return best_col
            
            return None
            
        except Exception:
            return None

class StreamingDataLoader:
    """Streaming data loader with enhanced memory management"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor(max_memory_gb=config.MAX_MEMORY_GB)
        self.chunk_size = config.CHUNK_SIZE
        self.target_detector = CTRTargetDetector(config)
        
        # Processing state
        self.detected_target_column = None
        self.processing_stats = {
            'total_rows': 0,
            'total_chunks': 0,
            'processing_time': 0,
            'memory_peak': 0
        }
    
    def _validate_files(self, train_path: Path, test_path: Path) -> bool:
        """Validate data files exist and are readable"""
        try:
            if not train_path.exists():
                logger.error(f"Training file not found: {train_path}")
                return False
            
            if not test_path.exists():
                logger.error(f"Test file not found: {test_path}")
                return False
            
            # Test file readability
            try:
                pq.read_schema(train_path)
                pq.read_schema(test_path)
                logger.info("File validation successful")
                return True
            except Exception as e:
                logger.error(f"File validation failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    def _process_data_chunk(self, chunk: pd.DataFrame, chunk_idx: int) -> pd.DataFrame:
        """Process individual data chunk with memory monitoring"""
        try:
            # Memory check before processing
            if not self.memory_monitor.check_memory_limit(f"chunk {chunk_idx}"):
                logger.warning(f"Memory limit reached at chunk {chunk_idx}")
                self.memory_monitor.force_cleanup()
            
            # Basic data cleaning
            chunk = chunk.dropna(thresh=len(chunk.columns) * 0.5)  # Drop rows with >50% missing
            
            # Memory optimization
            for col in chunk.columns:
                if chunk[col].dtype == 'object':
                    # Convert string columns to category if beneficial
                    if chunk[col].nunique() / len(chunk) < 0.5:
                        chunk[col] = chunk[col].astype('category')
                elif chunk[col].dtype == 'float64':
                    # Downcast float64 to float32 if no precision loss
                    chunk[col] = pd.to_numeric(chunk[col], downcast='float')
                elif chunk[col].dtype == 'int64':
                    # Downcast int64 if possible
                    chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
            
            return chunk
            
        except Exception as e:
            logger.error(f"Chunk processing failed at chunk {chunk_idx}: {e}")
            return chunk
    
    def load_streaming_data(self, file_path: Path, is_training: bool = True) -> Optional[pd.DataFrame]:
        """Load data with streaming and memory management"""
        logger.info(f"{'Training' if is_training else 'Test'} data streaming processing started")
        
        try:
            start_time = time.time()
            
            # File analysis
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            num_row_groups = parquet_file.num_row_groups
            
            logger.info(f"File analysis - Total {total_rows:,} rows, {num_row_groups} row groups")
            
            # Memory status check
            self.memory_monitor.log_memory_status("Streaming start")
            
            # Process data in chunks
            chunks = []
            processed_rows = 0
            
            for i in range(num_row_groups):
                # Memory check before each row group
                memory_status = self.memory_monitor.get_memory_status()
                if memory_status['level'] == 'critical':
                    logger.error("Critical memory usage - stopping data loading")
                    break
                
                # Read row group
                row_group = parquet_file.read_row_group(i).to_pandas()
                processed_rows += len(row_group)
                
                # Process chunk
                processed_chunk = self._process_data_chunk(row_group, i)
                
                # Target detection on first training chunk
                if is_training and i == 0 and not self.detected_target_column:
                    logger.info("Target column detection started")
                    self.detected_target_column = self.target_detector.detect_target_column(processed_chunk)
                    if self.detected_target_column:
                        logger.info(f"Detected target column: {self.detected_target_column}")
                
                chunks.append(processed_chunk)
                
                # Log progress
                progress = (i + 1) / num_row_groups * 100
                logger.info(f"Row groups {i+1}-{i+1} processed: {len(processed_chunk):,} rows")
                logger.info(f"Progress: {progress:.1f}% ({processed_rows:,} rows)")
                
                # Memory cleanup after each chunk
                del row_group, processed_chunk
                gc.collect()
            
            # Combine chunks
            logger.info(f"Final combination: {len(chunks)} chunks")
            if chunks:
                combined_df = pd.concat(chunks, ignore_index=True)
                del chunks  # Free memory
                gc.collect()
                
                logger.info(f"File streaming completed: {len(combined_df):,} rows")
                
                # Final memory status
                final_memory = self.memory_monitor.log_memory_status("Streaming completed")
                
                # Update processing stats
                self.processing_stats.update({
                    'total_rows': len(combined_df),
                    'total_chunks': len(chunks),
                    'processing_time': time.time() - start_time,
                    'memory_peak': final_memory['peak_memory_gb']
                })
                
                return combined_df
            else:
                logger.error("No data chunks processed successfully")
                return None
                
        except Exception as e:
            logger.error(f"Streaming data loading failed: {e}")
            return None
    
    def get_detected_target_column(self) -> Optional[str]:
        """Return detected target column"""
        return self.detected_target_column

class LargeDataLoader:
    """Main large data loader with comprehensive memory management"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor(max_memory_gb=config.MAX_MEMORY_GB)
        self.streaming_loader = StreamingDataLoader(config)
        self.quick_mode = False
        
        # File paths
        self.train_path = config.TRAIN_PATH
        self.test_path = config.TEST_PATH
        
        # Data storage
        self.train_data = None
        self.test_data = None
        
    def set_quick_mode(self, enabled: bool):
        """Set quick mode for testing"""
        self.quick_mode = enabled
        if enabled:
            logger.info("Quick mode enabled - will sample 50 rows")
        else:
            logger.info("Quick mode disabled - will load full dataset")
    
    def load_quick_sample_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load quick sample data for testing"""
        logger.info("Quick sample data loading started")
        
        try:
            # Sample small amount for testing
            train_sample = pd.read_parquet(self.train_path, nrows=50)
            test_sample = pd.read_parquet(self.test_path, nrows=50)
            
            # Target detection
            target_column = self.streaming_loader.target_detector.detect_target_column(train_sample)
            self.streaming_loader.detected_target_column = target_column
            
            logger.info(f"Quick sample loaded - Train: {train_sample.shape}, Test: {test_sample.shape}")
            return train_sample, test_sample
            
        except Exception as e:
            logger.error(f"Quick sample loading failed: {e}")
            return None, None
    
    def load_large_data_optimized(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load large data with optimization and memory management"""
        logger.info("=== Complete data processing started ===")
        
        try:
            # Initialize streaming loader
            logger.info("Streaming data loader initialization completed")
            logger.info("Full mode enabled: Will load complete dataset")
            
            # Validate files
            if not self.streaming_loader._validate_files(self.train_path, self.test_path):
                logger.error("File validation failed")
                return None, None
            
            # Load training data
            logger.info("=== Full data streaming loading started ===")
            self.memory_monitor.log_memory_status("Streaming start")
            
            train_data = self.streaming_loader.load_streaming_data(self.train_path, is_training=True)
            if train_data is None:
                logger.error("Training data loading failed")
                return None, None
            
            self.memory_monitor.log_memory_status("Training data loaded")
            
            # Load test data
            test_data = self.streaming_loader.load_streaming_data(self.test_path, is_training=False)
            if test_data is None:
                logger.error("Test data loading failed")
                return None, None
            
            # Final memory status
            final_memory = self.memory_monitor.log_memory_status("All data loaded")
            
            logger.info(f"=== Full data streaming completed - Training: {train_data.shape}, Test: {test_data.shape} ===")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Large data loading failed: {e}")
            return None, None
    
    def get_detected_target_column(self) -> Optional[str]:
        """Return detected target column"""
        return self.streaming_loader.get_detected_target_column()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory statistics"""
        return {
            'current_status': self.memory_monitor.get_memory_status(),
            'processing_stats': self.streaming_loader.processing_stats,
            'quick_mode': self.quick_mode
        }