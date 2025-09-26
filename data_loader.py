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
            
            effective_available = min(available_gb, self.memory_limit_gb - used_gb)
            if effective_available < 0:
                effective_available = min(5.0, available_gb)
            
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

class SeqColumnPreprocessor:
    """Sequence column preprocessing for complex data"""
    
    def __init__(self):
        self.seq_patterns = {}
        self.problematic_indicators = [',', ';', '|', '[', ']', '{', '}']
        
    def analyze_seq_column(self, column: pd.Series) -> Dict[str, Any]:
        """Analyze sequence column characteristics"""
        try:
            sample_size = min(10000, len(column))
            sample = column.dropna().head(sample_size)
            
            analysis = {
                'total_values': len(column),
                'unique_values': column.nunique(),
                'null_count': column.isnull().sum(),
                'avg_length': 0,
                'has_sequences': False,
                'complexity_score': 0,
                'data_type': str(column.dtype)
            }
            
            if len(sample) > 0:
                # Convert to string for analysis
                sample_str = sample.astype(str)
                
                # Calculate average length
                analysis['avg_length'] = sample_str.str.len().mean()
                
                # Check for sequence patterns
                has_comma = sample_str.str.contains(',', na=False).any()
                has_semicolon = sample_str.str.contains(';', na=False).any()
                has_brackets = sample_str.str.contains(r'[\[\]{}]', na=False).any()
                
                analysis['has_sequences'] = has_comma or has_semicolon or has_brackets
                
                # Calculate complexity score
                complexity_factors = [
                    analysis['unique_values'] / analysis['total_values'],  # Uniqueness ratio
                    min(1.0, analysis['avg_length'] / 100),  # Length complexity
                    int(analysis['has_sequences']) * 0.5  # Sequence complexity
                ]
                analysis['complexity_score'] = sum(complexity_factors)
                
            return analysis
            
        except Exception as e:
            logger.error(f"Seq column analysis failed: {e}")
            return {'complexity_score': 1.0, 'has_sequences': True}
    
    def should_process_seq_column(self, column: pd.Series) -> bool:
        """Determine if sequence column should be processed or removed"""
        try:
            analysis = self.analyze_seq_column(column)
            
            # Remove if too complex or problematic
            if analysis['complexity_score'] > 0.8:
                return False
            
            # Remove if too many unique values relative to total
            uniqueness_ratio = analysis['unique_values'] / max(analysis['total_values'], 1)
            if uniqueness_ratio > 0.9:
                return False
                
            # Keep if moderate complexity and has patterns
            return analysis['has_sequences'] and analysis['complexity_score'] <= 0.8
            
        except Exception as e:
            logger.warning(f"Seq column processing decision failed: {e}")
            return False
    
    def preprocess_seq_column(self, column: pd.Series) -> pd.Series:
        """Preprocess sequence column for feature engineering"""
        try:
            # Convert to string and handle missing values
            processed = column.astype(str).fillna('')
            
            # Clean common problematic characters
            processed = processed.str.replace(r'[\[\]{}]', '', regex=True)
            processed = processed.str.replace(r'["\']', '', regex=True)
            
            # Normalize separators to comma
            processed = processed.str.replace(r'[;|]', ',', regex=True)
            
            # Remove multiple consecutive separators
            processed = processed.str.replace(r',+', ',', regex=True)
            
            # Remove leading/trailing separators
            processed = processed.str.strip(',')
            
            return processed
            
        except Exception as e:
            logger.error(f"Seq column preprocessing failed: {e}")
            return column

class StreamingDataLoader:
    """Memory-efficient streaming data loader with complete data processing"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.seq_preprocessor = SeqColumnPreprocessor()
        
        self.chunk_size = config.get_optimal_chunk_size()
        self.quick_mode = False
        
        self.detected_target_column = None
        self.column_info = {}
        self.temp_files = []
        self.seq_column_analysis = {}
        
        self.processing_stats = {
            'start_time': None,
            'total_rows_processed': 0,
            'chunks_processed': 0,
            'memory_cleanups': 0,
            'temp_files_created': 0,
            'seq_columns_processed': 0,
            'problematic_columns_removed': 0
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
            target_candidates = ['clicked', 'click', 'target', 'label', 'y']
            
            for col in target_candidates:
                if col in df.columns:
                    unique_values = df[col].nunique()
                    if unique_values == 2:
                        logger.info(f"Target column detected: {col}")
                        self.detected_target_column = col
                        return col
            
            binary_columns = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'int32', 'bool']:
                    unique_vals = df[col].unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                        binary_columns.append(col)
            
            if binary_columns:
                target_col = binary_columns[0]
                logger.info(f"Binary target column detected: {target_col}")
                self.detected_target_column = target_col
                return target_col
                
        except Exception as e:
            logger.error(f"Target detection failed: {e}")
        
        return None
    
    def _analyze_parquet_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze parquet file structure efficiently"""
        try:
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
    
    def _identify_seq_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify sequence columns that need special handling"""
        try:
            seq_columns = []
            
            for col in df.columns:
                # Skip target column
                if col == self.detected_target_column:
                    continue
                    
                # Check column name patterns
                if any(pattern in col.lower() for pattern in ['seq', 'sequence', 'list', 'array']):
                    seq_columns.append(col)
                    continue
                
                # Analyze column content for complex sequences
                if col not in self.seq_column_analysis:
                    analysis = self.seq_preprocessor.analyze_seq_column(df[col])
                    self.seq_column_analysis[col] = analysis
                    
                    # Add to seq_columns if has sequence patterns and moderate complexity
                    if analysis['has_sequences'] and analysis['complexity_score'] > 0.3:
                        seq_columns.append(col)
            
            logger.info(f"Identified {len(seq_columns)} sequence columns")
            return seq_columns
            
        except Exception as e:
            logger.error(f"Seq column identification failed: {e}")
            return []
    
    def _process_seq_columns_in_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Process sequence columns in data chunk"""
        try:
            seq_columns = self._identify_seq_columns(chunk_df)
            
            for col in seq_columns:
                try:
                    if self.seq_preprocessor.should_process_seq_column(chunk_df[col]):
                        # Preprocess the column
                        chunk_df[col] = self.seq_preprocessor.preprocess_seq_column(chunk_df[col])
                        logger.debug(f"Preprocessed seq column: {col}")
                        self.processing_stats['seq_columns_processed'] += 1
                    else:
                        # Remove problematic sequence columns
                        chunk_df = chunk_df.drop(columns=[col])
                        logger.warning(f"Removed problematic seq column: {col}")
                        self.processing_stats['problematic_columns_removed'] += 1
                        
                except Exception as e:
                    logger.warning(f"Seq column processing failed for {col}: {e}")
                    # Remove failed columns
                    if col in chunk_df.columns:
                        chunk_df = chunk_df.drop(columns=[col])
                        self.processing_stats['problematic_columns_removed'] += 1
            
            return chunk_df
            
        except Exception as e:
            logger.error(f"Chunk seq processing failed: {e}")
            return chunk_df
    
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
        """Process data chunk with memory efficiency and seq column handling"""
        try:
            if not self.memory_monitor.check_memory_safety(2.0):
                self.memory_monitor.force_cleanup()
                if not self.memory_monitor.check_memory_safety(2.0):
                    raise MemoryError("Insufficient memory for chunk processing")
            
            # Process sequence columns first
            chunk = self._process_seq_columns_in_chunk(chunk)
            
            # Basic data type optimization
            for col in chunk.columns:
                try:
                    if chunk[col].dtype == 'object':
                        # Check if it's a reasonable categorical
                        unique_count = chunk[col].nunique()
                        total_count = len(chunk[col])
                        
                        if unique_count < total_count * 0.5 and unique_count < 1000:
                            try:
                                chunk[col] = chunk[col].astype('category')
                            except:
                                pass
                        else:
                            # Convert to string for consistency
                            chunk[col] = chunk[col].astype(str)
                    elif chunk[col].dtype == 'float64':
                        try:
                            chunk[col] = pd.to_numeric(chunk[col], downcast='float')
                        except:
                            pass
                    elif chunk[col].dtype == 'int64':
                        try:
                            chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"Data type optimization failed for {col}: {e}")
                    continue
            
            return chunk
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return chunk
    
    def load_parquet_streaming(self, file_path: Path, max_rows: Optional[int] = None, 
                             ensure_complete: bool = True) -> Optional[pd.DataFrame]:
        """Load parquet file with memory-efficient streaming and complete data processing"""
        try:
            self.processing_stats['start_time'] = time.time()
            self.memory_monitor.log_memory_status("Streaming start")
            
            # Analyze file first
            file_info = self._analyze_parquet_file(file_path)
            if file_info['total_rows'] == 0:
                return None
            
            total_rows = file_info['total_rows']
            target_rows = total_rows
            
            # Apply max_rows limit only if specified and not ensuring complete data
            if max_rows and not ensure_complete:
                target_rows = min(total_rows, max_rows)
                logger.info(f"Processing limited to {target_rows:,} rows (max_rows applied)")
            elif ensure_complete:
                logger.info(f"Complete data processing mode: {total_rows:,} rows")
            
            # Memory-based chunk size adjustment
            memory_status = self.memory_monitor.get_memory_status()
            if memory_status['available_gb'] < 15:
                self.chunk_size = min(self.chunk_size, 25000)
                logger.info(f"Reduced chunk size to {self.chunk_size} due to low memory")
            
            processed_data_chunks = []
            temp_files_created = []
            rows_processed = 0
            
            # Process in chunks
            parquet_file = pq.ParquetFile(file_path)
            
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=self.chunk_size)):
                if not ensure_complete and max_rows and rows_processed >= max_rows:
                    break
                
                # Convert to pandas DataFrame
                chunk_df = batch.to_pandas()
                
                # Apply row limit if specified and not ensuring complete data
                if not ensure_complete and max_rows and rows_processed + len(chunk_df) > max_rows:
                    chunk_df = chunk_df.head(max_rows - rows_processed)
                
                # Detect target column in first chunk
                if self.detected_target_column is None and len(chunk_df) > 0:
                    self._detect_target_column(chunk_df)
                
                # Process chunk with seq column handling
                processed_chunk = self._process_chunk_streaming(chunk_df)
                
                rows_processed += len(processed_chunk)
                self.processing_stats['chunks_processed'] += 1
                
                # Memory management - save to temp files for large datasets
                if len(processed_data_chunks) >= 8 or memory_status['available_gb'] < 10:
                    if processed_data_chunks:
                        # Combine current chunks and save to temp file
                        combined_chunk = pd.concat(processed_data_chunks, ignore_index=True)
                        temp_file_path = self._create_temp_file(combined_chunk, "streaming_data")
                        temp_files_created.append(temp_file_path)
                        processed_data_chunks = []
                        del combined_chunk
                        self.memory_monitor.force_cleanup()
                        
                        logger.info(f"Saved batch {batch_idx + 1} to temp file, memory cleared")
                
                processed_data_chunks.append(processed_chunk)
                
                # Progress logging
                if self.processing_stats['chunks_processed'] % 20 == 0:
                    progress = (rows_processed / target_rows) * 100
                    logger.info(f"Progress: {progress:.1f}% ({rows_processed:,} rows)")
                    logger.info(f"Seq processing stats: Processed {self.processing_stats['seq_columns_processed']}, "
                              f"Removed {self.processing_stats['problematic_columns_removed']}")
                    self.memory_monitor.log_memory_status("Processing")
                
                # Memory safety check
                if not self.memory_monitor.check_memory_safety(3.0):
                    logger.warning("Memory safety threshold reached")
                    if ensure_complete:
                        # Force save current data to temp file and continue
                        if processed_data_chunks:
                            combined_chunk = pd.concat(processed_data_chunks, ignore_index=True)
                            temp_file_path = self._create_temp_file(combined_chunk, "streaming_data")
                            temp_files_created.append(temp_file_path)
                            processed_data_chunks = []
                            del combined_chunk
                            self.memory_monitor.force_cleanup()
                    else:
                        break
            
            # Combine final result
            final_df = None
            
            # Handle remaining chunks in memory
            if processed_data_chunks:
                final_df = pd.concat(processed_data_chunks, ignore_index=True)
                logger.info(f"Combined {len(processed_data_chunks)} in-memory chunks")
            
            # Load and combine temp files
            if temp_files_created:
                temp_dataframes = []
                
                # Add in-memory data first
                if final_df is not None:
                    temp_dataframes.append(final_df)
                
                # Load all temp files
                for temp_file in temp_files_created:
                    try:
                        temp_df = self._load_from_temp_file(temp_file)
                        temp_dataframes.append(temp_df)
                    except Exception as e:
                        logger.error(f"Failed to load temp file {temp_file}: {e}")
                
                # Combine all data
                if temp_dataframes:
                    final_df = pd.concat(temp_dataframes, ignore_index=True)
                    logger.info(f"Combined {len(temp_dataframes)} data chunks")
                
                # Cleanup temp dataframes from memory
                for temp_df in temp_dataframes:
                    del temp_df
                self.memory_monitor.force_cleanup()
            
            # Final cleanup
            self.memory_monitor.force_cleanup()
            self.processing_stats['total_rows_processed'] = rows_processed
            
            if final_df is not None:
                logger.info(f"File streaming completed: {len(final_df):,} rows (target: {target_rows:,})")
                logger.info(f"Final processing stats: Seq processed {self.processing_stats['seq_columns_processed']}, "
                          f"Problematic removed {self.processing_stats['problematic_columns_removed']}")
                
                # Verify we got complete data if ensure_complete is True
                if ensure_complete and len(final_df) < total_rows * 0.95:  # Allow 5% tolerance
                    logger.warning(f"Incomplete data loaded: {len(final_df):,} / {total_rows:,} rows")
                
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
            if not self.memory_monitor.check_memory_safety(10.0):
                logger.error("Insufficient memory for full data loading")
                return None, None
            
            train_df = None
            test_df = None
            
            # Load training data with complete processing
            logger.info("Training data streaming processing started")
            if self.config.TRAIN_PATH.exists():
                max_train_rows = None if not self.quick_mode else Config.QUICK_SAMPLE_SIZE
                train_df = self.load_parquet_streaming(
                    self.config.TRAIN_PATH, 
                    max_rows=max_train_rows,
                    ensure_complete=not self.quick_mode
                )
            
            # Memory cleanup between files
            self.memory_monitor.force_cleanup()
            
            # Load test data with complete processing
            logger.info("Test data streaming processing started")
            if self.config.TEST_PATH.exists():
                max_test_rows = None if not self.quick_mode else Config.QUICK_TEST_SIZE
                test_df = self.load_parquet_streaming(
                    self.config.TEST_PATH, 
                    max_rows=max_test_rows,
                    ensure_complete=not self.quick_mode
                )
            
            self.memory_monitor.log_memory_status("Streaming completed")
            
            if train_df is not None and test_df is not None:
                logger.info(f"=== Full data streaming completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
                
                # Verify complete data loading
                if not self.quick_mode:
                    expected_train_rows = 10704179
                    expected_test_rows = 1527298
                    
                    train_completeness = len(train_df) / expected_train_rows
                    test_completeness = len(test_df) / expected_test_rows
                    
                    logger.info(f"Data completeness - Train: {train_completeness:.2%}, Test: {test_completeness:.2%}")
                    logger.info(f"Seq processing summary - Train processed: {self.processing_stats['seq_columns_processed']}")
                    
                    if train_completeness < 0.95 or test_completeness < 0.95:
                        logger.warning("Incomplete data loading detected")
                
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
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()

class LargeDataLoader:
    """Main data loader interface with memory management"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.quick_mode = False
        self.target_column = None
        
        self.loading_stats = {
            'start_time': time.time(),
            'data_loaded': False,
            'quick_mode': False,
            'train_rows': 0,
            'test_rows': 0,
            'loading_time': 0,
            'memory_usage': 0,
            'seq_processing_stats': {}
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
            
            streaming_loader = StreamingDataLoader(self.config)
            streaming_loader.set_quick_mode(True)
            
            train_df = streaming_loader.load_parquet_streaming(
                self.config.TRAIN_PATH, 
                max_rows=Config.QUICK_SAMPLE_SIZE,
                ensure_complete=False
            )
            
            test_df = streaming_loader.load_parquet_streaming(
                self.config.TEST_PATH, 
                max_rows=Config.QUICK_TEST_SIZE,
                ensure_complete=False
            )
            
            self.target_column = streaming_loader.get_detected_target_column()
            
            self.loading_stats.update({
                'data_loaded': True,
                'train_rows': len(train_df) if train_df is not None else 0,
                'test_rows': len(test_df) if test_df is not None else 0,
                'loading_time': time.time() - self.loading_stats['start_time'],
                'seq_processing_stats': streaming_loader.get_processing_stats()
            })
            
            logger.info(f"Quick sample loading completed - Train: {len(train_df) if train_df is not None else 0}, Test: {len(test_df) if test_df is not None else 0}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Quick sample loading failed: {e}")
            return None, None
    
    def load_large_data_optimized(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load complete dataset with memory optimization"""
        logger.info("=== Complete data processing started ===")
        
        streaming_loader = StreamingDataLoader(self.config)
        streaming_loader.set_quick_mode(False)
        result = streaming_loader.load_full_data_streaming()
        
        self.target_column = streaming_loader.get_detected_target_column()
        
        self.loading_stats.update({
            'data_loaded': True,
            'train_rows': result[0].shape[0] if result[0] is not None else 0,
            'test_rows': result[1].shape[0] if result[1] is not None else 0,
            'loading_time': time.time() - self.loading_stats['start_time'],
            'seq_processing_stats': streaming_loader.get_processing_stats()
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
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    
    try:
        loader = LargeDataLoader(config)
        
        print("Testing quick mode...")
        loader.set_quick_mode(True)
        train_df, test_df = loader.load_quick_sample_data()
        
        print(f"Quick mode results:")
        print(f"Training data: {train_df.shape if train_df is not None else 'None'}")
        print(f"Test data: {test_df.shape if test_df is not None else 'None'}")
        print(f"Detected target column: {loader.get_detected_target_column()}")
        
        stats = loader.get_loading_stats()
        print(f"Seq processing: {stats['seq_processing_stats']}")
        
        memory_monitor = MemoryMonitor()
        memory_monitor.log_memory_status("After quick test")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")