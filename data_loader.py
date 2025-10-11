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

try:
    import nvtabular as nvt
    from nvtabular import ops
    from merlin.io import Dataset
    NVTABULAR_AVAILABLE = True
except ImportError:
    NVTABULAR_AVAILABLE = False
    logging.warning("NVTabular not available. Using pandas fallback. Install with: conda install -c nvidia -c rapidsai -c conda-forge nvtabular python=3.10 cudatoolkit=11.8")

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    logging.warning("cuDF not available. Using pandas fallback.")

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring for data loading operations"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 10.0,
            'critical': 5.0,
            'abort': 2.0
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
            if force or vm.percent > 80:
                logger.info(f"Memory status ({context}): {vm.percent:.1f}% used, "
                           f"{vm.available/(1024**3):.1f}GB available")
    
    def force_memory_cleanup(self):
        """Force garbage collection and memory cleanup"""
        try:
            collected = gc.collect()
            
            if collected > 0:
                logger.debug(f"Memory cleanup: {collected} objects collected")
            
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

class PandasDataLoader:
    """Pandas-based data loader for Windows compatibility"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        self.temp_dir = tempfile.mkdtemp()
        self.quick_mode = False
        
        logger.info("Pandas data loader initialized (Windows compatible)")
    
    def __del__(self):
        """Cleanup temporary directory on destruction"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def set_quick_mode(self, quick_mode: bool):
        """Enable or disable quick mode"""
        self.quick_mode = quick_mode
        if quick_mode:
            logger.info("Quick mode enabled: Will load 50 samples only")
        else:
            logger.info("Full mode enabled: Will load complete dataset")
    
    def load_quick_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load small sample data for quick testing"""
        logger.info("=== Quick sample data loading started ===")
        
        try:
            train_sample = self._load_sample_from_file(self.config.TRAIN_PATH, 35, is_train=True)
            test_sample = self._load_sample_from_file(self.config.TEST_PATH, 15, is_train=False)
            
            if self.target_column and self.target_column in train_sample.columns:
                logger.info(f"Target column confirmed: {self.target_column}")
            else:
                self.target_column = 'clicked'
                if self.target_column not in train_sample.columns:
                    train_sample[self.target_column] = np.random.binomial(1, 0.02, len(train_sample))
                    logger.info(f"Created dummy target column: {self.target_column}")
            
            logger.info(f"Quick sample loading completed - train: {train_sample.shape}, test: {test_sample.shape}")
            
            return train_sample, test_sample
            
        except Exception as e:
            logger.error(f"Quick sample loading failed: {e}")
            return self._create_dummy_data()
    
    def _load_sample_from_file(self, file_path: Path, sample_size: int, is_train: bool = True) -> pd.DataFrame:
        """Load a small sample from a parquet file"""
        try:
            if not PYARROW_AVAILABLE:
                raise ValueError("PyArrow is required for parquet loading")
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return self._create_dummy_sample(sample_size, is_train)
            
            parquet_file = pq.ParquetFile(file_path)
            
            table = parquet_file.read_row_group(0)
            df = table.to_pandas()
            
            if is_train and self.target_column is None:
                self.target_column = self.column_analyzer.detect_target_column(df)
            
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            logger.info(f"Sample loaded from {file_path.name}: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"Sample loading failed for {file_path}: {e}")
            return self._create_dummy_sample(sample_size, is_train)
    
    def _create_dummy_sample(self, sample_size: int, is_train: bool) -> pd.DataFrame:
        """Create dummy data when file loading fails"""
        try:
            data = {
                'feature_1': np.random.randint(0, 100, sample_size),
                'feature_2': np.random.normal(0, 1, sample_size),
                'feature_3': np.random.choice(['A', 'B', 'C'], sample_size),
                'feature_4': np.random.uniform(0, 1, sample_size)
            }
            
            if is_train:
                data['clicked'] = np.random.binomial(1, 0.02, sample_size)
                self.target_column = 'clicked'
            
            df = pd.DataFrame(data)
            logger.info(f"Created dummy sample: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Dummy sample creation failed: {e}")
            return pd.DataFrame({'dummy': [0] * sample_size})
    
    def _create_dummy_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal dummy data as absolute fallback"""
        train_data = {
            'feature_1': [1, 2, 3] * 17,
            'feature_2': [0.1, 0.2, 0.3] * 17,
            'clicked': [0, 1, 0] * 17
        }
        
        test_data = {
            'feature_1': [4, 5, 6] * 5,
            'feature_2': [0.4, 0.5, 0.6] * 5
        }
        
        train_df = pd.DataFrame(train_data).iloc[:35]
        test_df = pd.DataFrame(test_data).iloc[:15]
        
        self.target_column = 'clicked'
        logger.warning("Using minimal dummy data as fallback")
        
        return train_df, test_df
    
    def load_full_data_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load full data using pandas with chunking"""
        logger.info("=== Pandas full data loading started ===")
        
        try:
            self.memory_monitor.log_memory_status("Pandas loading start", force=True)
            
            if not self._validate_files():
                raise ValueError("Data files do not exist")
            
            logger.info("Loading training data with pandas")
            train_df = self._load_with_pandas(str(self.config.TRAIN_PATH), is_train=True)
            
            if train_df is None or train_df.empty:
                raise ValueError("Training data loading failed")
            
            logger.info("Loading test data with pandas")
            test_df = self._load_with_pandas(str(self.config.TEST_PATH), is_train=False)
            
            if test_df is None or test_df.empty:
                raise ValueError("Test data loading failed")
            
            self.memory_monitor.log_memory_status("Pandas loading completed", force=True)
            
            logger.info(f"=== Pandas loading completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Pandas data loading failed: {e}")
            self.memory_monitor.force_memory_cleanup()
            raise
    
    def _load_with_pandas(self, file_path: str, is_train: bool = True) -> pd.DataFrame:
        """Load data using pandas with memory optimization"""
        try:
            logger.info(f"Loading with pandas: {file_path}")
            
            if not PYARROW_AVAILABLE:
                logger.warning("PyArrow not available, using slower engine")
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_parquet(file_path, engine='pyarrow')
            
            if is_train and self.target_column is None:
                self.target_column = self.column_analyzer.detect_target_column(df)
                logger.info(f"Target column detected: {self.target_column}")
            
            logger.info(f"Pandas loading completed: {df.shape}")
            
            self.memory_monitor.force_memory_cleanup()
            
            return df
            
        except Exception as e:
            logger.error(f"Pandas loading failed: {e}")
            return pd.DataFrame()
    
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

class NVTabularDataLoader:
    """NVTabular-based data loader with GPU acceleration (Linux only)"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        self.temp_dir = tempfile.mkdtemp()
        self.quick_mode = False
        
        if not NVTABULAR_AVAILABLE:
            logger.error("NVTabular is not available. This loader requires NVTabular installation.")
            raise ImportError("NVTabular is required for this data loader")
        
        logger.info("NVTabular data loader initialization completed")
    
    def __del__(self):
        """Cleanup temporary directory on destruction"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def set_quick_mode(self, quick_mode: bool):
        """Enable or disable quick mode"""
        self.quick_mode = quick_mode
        if quick_mode:
            logger.info("Quick mode enabled: Will load 50 samples only")
        else:
            logger.info("Full mode enabled: Will load complete dataset")
    
    def load_quick_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load small sample data for quick testing"""
        logger.info("=== Quick sample data loading started ===")
        
        try:
            train_sample = self._load_sample_from_file(self.config.TRAIN_PATH, 35, is_train=True)
            test_sample = self._load_sample_from_file(self.config.TEST_PATH, 15, is_train=False)
            
            if self.target_column and self.target_column in train_sample.columns:
                logger.info(f"Target column confirmed: {self.target_column}")
            else:
                self.target_column = 'clicked'
                if self.target_column not in train_sample.columns:
                    train_sample[self.target_column] = np.random.binomial(1, 0.02, len(train_sample))
                    logger.info(f"Created dummy target column: {self.target_column}")
            
            logger.info(f"Quick sample loading completed - train: {train_sample.shape}, test: {test_sample.shape}")
            
            return train_sample, test_sample
            
        except Exception as e:
            logger.error(f"Quick sample loading failed: {e}")
            return self._create_dummy_data()
    
    def _load_sample_from_file(self, file_path: Path, sample_size: int, is_train: bool = True) -> pd.DataFrame:
        """Load a small sample from a parquet file"""
        try:
            if not PYARROW_AVAILABLE:
                raise ValueError("PyArrow is required for parquet loading")
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return self._create_dummy_sample(sample_size, is_train)
            
            parquet_file = pq.ParquetFile(file_path)
            
            table = parquet_file.read_row_group(0)
            df = table.to_pandas()
            
            if is_train and self.target_column is None:
                self.target_column = self.column_analyzer.detect_target_column(df)
            
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            logger.info(f"Sample loaded from {file_path.name}: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"Sample loading failed for {file_path}: {e}")
            return self._create_dummy_sample(sample_size, is_train)
    
    def _create_dummy_sample(self, sample_size: int, is_train: bool) -> pd.DataFrame:
        """Create dummy data when file loading fails"""
        try:
            data = {
                'feature_1': np.random.randint(0, 100, sample_size),
                'feature_2': np.random.normal(0, 1, sample_size),
                'feature_3': np.random.choice(['A', 'B', 'C'], sample_size),
                'feature_4': np.random.uniform(0, 1, sample_size)
            }
            
            if is_train:
                data['clicked'] = np.random.binomial(1, 0.02, sample_size)
                self.target_column = 'clicked'
            
            df = pd.DataFrame(data)
            logger.info(f"Created dummy sample: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Dummy sample creation failed: {e}")
            return pd.DataFrame({'dummy': [0] * sample_size})
    
    def _create_dummy_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal dummy data as absolute fallback"""
        train_data = {
            'feature_1': [1, 2, 3] * 17,
            'feature_2': [0.1, 0.2, 0.3] * 17,
            'clicked': [0, 1, 0] * 17
        }
        
        test_data = {
            'feature_1': [4, 5, 6] * 5,
            'feature_2': [0.4, 0.5, 0.6] * 5
        }
        
        train_df = pd.DataFrame(train_data).iloc[:35]
        test_df = pd.DataFrame(test_data).iloc[:15]
        
        self.target_column = 'clicked'
        logger.warning("Using minimal dummy data as fallback")
        
        return train_df, test_df
    
    def load_full_data_nvtabular(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load full data using NVTabular with GPU acceleration"""
        logger.info("=== NVTabular full data loading started ===")
        
        try:
            self.memory_monitor.log_memory_status("NVTabular loading start", force=True)
            
            if not self._validate_files():
                raise ValueError("Data files do not exist")
            
            logger.info("Processing training data with NVTabular")
            train_df = self._load_with_nvtabular(str(self.config.TRAIN_PATH), is_train=True)
            
            if train_df is None or train_df.empty:
                raise ValueError("Training data loading failed")
            
            logger.info("Processing test data with NVTabular")
            test_df = self._load_with_nvtabular(str(self.config.TEST_PATH), is_train=False)
            
            if test_df is None or test_df.empty:
                raise ValueError("Test data loading failed")
            
            self.memory_monitor.log_memory_status("NVTabular loading completed", force=True)
            
            logger.info(f"=== NVTabular loading completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"NVTabular data loading failed: {e}")
            self.memory_monitor.force_memory_cleanup()
            raise
    
    def _load_with_nvtabular(self, file_path: str, is_train: bool = True) -> pd.DataFrame:
        """Load data using NVTabular Dataset"""
        try:
            logger.info(f"Loading with NVTabular: {file_path}")
            
            dataset = Dataset(
                file_path,
                engine='parquet',
                part_size=self.config.NVTABULAR_PARTITION_SIZE
            )
            
            logger.info(f"NVTabular Dataset created with partition size: {self.config.NVTABULAR_PARTITION_SIZE}")
            
            if is_train and self.target_column is None:
                sample_batch = dataset.to_ddf().head(1000)
                sample_pd = sample_batch.to_pandas() if hasattr(sample_batch, 'to_pandas') else sample_batch
                self.target_column = self.column_analyzer.detect_target_column(sample_pd)
                logger.info(f"Target column detected: {self.target_column}")
            
            logger.info("Converting NVTabular Dataset to DataFrame")
            ddf = dataset.to_ddf()
            df = ddf.compute()
            
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            
            logger.info(f"NVTabular loading completed: {df.shape}")
            
            del dataset, ddf
            self.memory_monitor.force_memory_cleanup()
            
            return df
            
        except Exception as e:
            logger.error(f"NVTabular loading failed: {e}")
            return pd.DataFrame()
    
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
    """Large data loader with automatic backend selection"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.target_column = None
        self.quick_mode = False
        
        self.backend = 'pandas'
        if NVTABULAR_AVAILABLE:
            self.backend = 'nvtabular'
            logger.info("NVTabular available - using GPU accelerated backend")
        else:
            logger.info("NVTabular not available - using pandas backend (Windows compatible)")
        
        self.loading_stats = {
            'start_time': time.time(),
            'data_loaded': False,
            'train_rows': 0,
            'test_rows': 0,
            'loading_time': 0.0,
            'memory_usage': 0.0,
            'backend': self.backend
        }
        
        logger.info(f"Large data loader initialized with {self.backend} backend")
    
    def set_quick_mode(self, quick_mode: bool):
        """Enable or disable quick mode for testing"""
        self.quick_mode = quick_mode
        if quick_mode:
            logger.info("Large data loader set to quick mode (50 samples)")
        else:
            logger.info("Large data loader set to full mode (complete dataset)")
    
    def load_quick_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load small sample data for quick testing"""
        logger.info(f"=== Quick sample data loading via LargeDataLoader ({self.backend}) ===")
        
        if self.backend == 'nvtabular' and NVTABULAR_AVAILABLE:
            loader = NVTabularDataLoader(self.config)
            loader.set_quick_mode(True)
            result = loader.load_quick_sample_data()
            self.target_column = loader.get_detected_target_column()
        else:
            loader = PandasDataLoader(self.config)
            loader.set_quick_mode(True)
            result = loader.load_quick_sample_data()
            self.target_column = loader.get_detected_target_column()
        
        self.loading_stats.update({
            'data_loaded': True,
            'train_rows': result[0].shape[0] if result[0] is not None else 0,
            'test_rows': result[1].shape[0] if result[1] is not None else 0,
            'loading_time': time.time() - self.loading_stats['start_time']
        })
        
        return result
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load complete data with automatic backend selection"""
        logger.info(f"=== Complete data processing started ({self.backend} backend) ===")
        
        if self.backend == 'nvtabular' and NVTABULAR_AVAILABLE:
            loader = NVTabularDataLoader(self.config)
            loader.set_quick_mode(False)
            result = loader.load_full_data_nvtabular()
            self.target_column = loader.get_detected_target_column()
        else:
            loader = PandasDataLoader(self.config)
            loader.set_quick_mode(False)
            result = loader.load_full_data_pandas()
            self.target_column = loader.get_detected_target_column()
        
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
SimpleDataLoader = LargeDataLoader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    
    try:
        loader = LargeDataLoader(config)
        
        print(f"Backend: {loader.backend}")
        print(f"NVTabular available: {NVTABULAR_AVAILABLE}")
        
        print("\nTesting quick mode...")
        loader.set_quick_mode(True)
        train_df, test_df = loader.load_quick_sample_data()
        
        print(f"Quick mode results:")
        print(f"Training data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        print(f"Detected target column: {loader.get_detected_target_column()}")
        print(f"Total samples: {len(train_df) + len(test_df)}")
        
        if NVTABULAR_AVAILABLE:
            print("\nTesting full mode with NVTabular...")
        else:
            print("\nTesting full mode with pandas...")
        
        loader.set_quick_mode(False)
        train_df_full, test_df_full = loader.load_large_data_optimized()
        
        print(f"Full mode results:")
        print(f"Training data: {train_df_full.shape}")
        print(f"Test data: {test_df_full.shape}")
        print(f"Complete processing finished: {len(train_df_full) + len(test_df_full):,} rows")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")