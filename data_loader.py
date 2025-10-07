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
import shutil
from pathlib import Path
warnings.filterwarnings('ignore')

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
    """Memory monitoring for data loading operations"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 10.0,
            'critical': 8.0,
            'abort': 5.0
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
        return 64.0
    
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
        """Check current memory pressure and recommend actions"""
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
            if force or vm.percent > 70:
                logger.info(f"Memory status ({context}): {vm.percent:.1f}% used, "
                           f"{vm.available/(1024**3):.1f}GB available")
    
    def force_memory_cleanup(self, intensive: bool = False):
        """Force garbage collection and memory cleanup"""
        try:
            collected = gc.collect()
            
            if intensive:
                for _ in range(3):
                    collected += gc.collect()
                    time.sleep(0.1)
            
            return collected
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
            return 0

class DataColumnAnalyzer:
    """Analyzes data columns to detect target columns and data characteristics"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_config = config.TARGET_DETECTION_CONFIG
        self.target_candidates = config.TARGET_COLUMN_CANDIDATES
    
    def detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the target column for CTR prediction"""
        try:
            logger.info("Target column detection started")
            
            for candidate in self.target_candidates:
                if candidate in df.columns:
                    if self._validate_target_column(df[candidate]):
                        logger.info(f"Target column detected: {candidate}")
                        return candidate
            
            for col in df.columns:
                col_lower = col.lower()
                for candidate in self.target_candidates:
                    if candidate.lower() in col_lower:
                        if self._validate_target_column(df[col]):
                            logger.info(f"Target column detected (partial match): {col}")
                            return col
            
            binary_columns = []
            for col in df.columns:
                if self._is_binary_column(df[col]):
                    binary_columns.append(col)
            
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
            
            ctr = series.mean()
            
            min_ctr = self.target_config['min_ctr']
            max_ctr = self.target_config['max_ctr']
            
            if not (min_ctr <= ctr <= max_ctr):
                return False
            
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
    """Streaming data loader with memory management"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        self.temp_dir = tempfile.mkdtemp()
        
        logger.info("Streaming data loader initialization completed")
    
    def __del__(self):
        """Cleanup temporary directory on destruction"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def load_full_data_streaming(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete processing via streaming with memory management"""
        logger.info("=== Full data streaming loading started ===")
        
        try:
            self.memory_monitor.log_memory_status("Streaming start", force=True)
            
            if not self._validate_files():
                raise ValueError("Data files do not exist")
            
            logger.info("Training data streaming processing started")
            train_df = self._stream_process_file_optimized(str(self.config.TRAIN_PATH), is_train=True)
            
            if train_df is None or train_df.empty:
                raise ValueError("Training data streaming processing failed")
            
            logger.info(f"Training data loaded: {train_df.shape}")
            
            self.memory_monitor.force_memory_cleanup(intensive=True)
            self.memory_monitor.log_memory_status("After train load", force=True)
            
            logger.info("Test data streaming processing started") 
            test_df = self._stream_process_file_optimized(str(self.config.TEST_PATH), is_train=False)
            
            if test_df is None or test_df.empty:
                raise ValueError("Test data streaming processing failed")
            
            self.memory_monitor.log_memory_status("Streaming completed", force=True)
            
            logger.info(f"=== Full data streaming completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Streaming data loading failed: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            raise
    
    def _stream_process_file_optimized(self, file_path: str, is_train: bool = True) -> pd.DataFrame:
        """File streaming processing with memory optimization"""
        try:
            if not PYARROW_AVAILABLE:
                raise ValueError("PyArrow is required")
            
            pressure = self.memory_monitor.check_memory_pressure()
            if pressure['should_abort']:
                logger.error(f"Insufficient memory for processing: {pressure['available_gb']:.1f}GB available")
                raise MemoryError(f"Insufficient memory: {pressure['available_gb']:.1f}GB available")
            
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            num_row_groups = parquet_file.num_row_groups
            
            logger.info(f"File analysis - Total {total_rows:,} rows, {num_row_groups} row groups")
            
            essential_cols = self._get_essential_columns(parquet_file.schema.names, is_train)
            logger.info(f"Loading {len(essential_cols)} essential columns out of {len(parquet_file.schema.names)}")
            
            if is_train:
                sample_table = parquet_file.read_row_group(0, columns=essential_cols)
                sample_df = sample_table.to_pandas()
                
                if len(sample_df) > 1000:
                    sample_df = sample_df.sample(n=1000, random_state=42)
                
                self.target_column = self.column_analyzer.detect_target_column(sample_df)
                logger.info(f"Detected target column: {self.target_column}")
                
                del sample_df, sample_table
                gc.collect()
            
            table = parquet_file.read(columns=essential_cols)
            df = table.to_pandas()
            del table
            gc.collect()
            
            df = self._optimize_dataframe_memory(df, for_feature_engineering=True)
            
            logger.info(f"File streaming completed: {len(df):,} rows, {len(df.columns)} columns")
            
            self.memory_monitor.force_memory_cleanup(intensive=True)
            
            return df
            
        except Exception as e:
            logger.error(f"File streaming processing failed: {e}")
            return pd.DataFrame()
    
    def _get_essential_columns(self, all_columns: List[str], is_train: bool) -> List[str]:
        """Get essential columns to load"""
        try:
            exclude = ['seq']
            
            essential = [col for col in all_columns if col not in exclude]
            
            return essential
            
        except Exception as e:
            logger.warning(f"Essential column selection failed: {e}")
            return all_columns
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame, for_feature_engineering: bool = False) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        try:
            for col in df.columns:
                col_type = df[col].dtype
                
                if col_type == 'int64':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if min_val >= 0:
                        if max_val < 255:
                            df[col] = df[col].astype('uint8')
                        elif max_val < 65535:
                            df[col] = df[col].astype('uint16')
                        elif max_val < 4294967295:
                            df[col] = df[col].astype('uint32')
                    else:
                        if min_val > -128 and max_val < 127:
                            df[col] = df[col].astype('int8')
                        elif min_val > -32768 and max_val < 32767:
                            df[col] = df[col].astype('int16')
                        elif min_val > -2147483648 and max_val < 2147483647:
                            df[col] = df[col].astype('int32')
                
                elif col_type == 'float64':
                    df[col] = df[col].astype('float32')
                
                elif col_type == 'object':
                    if for_feature_engineering:
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio < 0.3:
                            try:
                                df[col] = df[col].astype('category')
                            except:
                                pass
                    else:
                        if df[col].nunique() / len(df) < 0.5:
                            try:
                                df[col] = df[col].astype('category')
                            except:
                                pass
            
            return df
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return df
    
    def _validate_files(self) -> bool:
        """Validate input files exist and have reasonable sizes"""
        try:
            train_path = Path(self.config.TRAIN_PATH)
            test_path = Path(self.config.TEST_PATH)
            
            if not train_path.exists():
                logger.error(f"Training file not found: {train_path}")
                return False
            
            if not test_path.exists():
                logger.error(f"Test file not found: {test_path}")
                return False
            
            train_size_mb = train_path.stat().st_size / (1024**2)
            test_size_mb = test_path.stat().st_size / (1024**2)
            
            if train_size_mb < 100:
                logger.warning(f"Training file seems small: {train_size_mb:.1f}MB")
            
            if test_size_mb < 10:
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
        
        self.streaming_loader = StreamingDataLoader(config)
        
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
        """Complete data processing (full dataset)"""
        logger.info("=== Complete data processing started ===")
        
        result = self.streaming_loader.load_full_data_streaming()
        
        self.target_column = self.streaming_loader.get_detected_target_column()
        
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

DataLoader = LargeDataLoader
SimpleDataLoader = StreamingDataLoader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    
    try:
        loader = LargeDataLoader(config)
        
        print("Testing full mode...")
        train_df, test_df = loader.load_large_data_optimized()
        
        print(f"Full mode results:")
        print(f"Training data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        print(f"Detected target column: {loader.get_detected_target_column()}")
        
        stats = loader.get_loading_stats()
        print(f"\nLoading stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")