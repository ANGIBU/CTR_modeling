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

try:
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    logging.warning("cuDF not available. GPU acceleration disabled.")

try:
    import nvtabular as nvt
    from nvtabular import ops
    from merlin.io import Dataset
    NVTABULAR_AVAILABLE = True
except ImportError:
    NVTABULAR_AVAILABLE = False
    logging.warning("NVTabular not available. Merlin features disabled.")

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring for data loading operations"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 8.0,
            'critical': 5.0,
            'abort': 3.0
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
            
            if CUDF_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            
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

class NVTabularDataLoader:
    """NVTabular-based GPU accelerated data loader"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        self.nvt_enabled = NVTABULAR_AVAILABLE and CUDF_AVAILABLE
        
        if not self.nvt_enabled:
            logger.warning("NVTabular not available. Falling back to standard loader.")
        else:
            logger.info("NVTabular data loader initialized")
    
    def process_with_nvtabular(self, force_reprocess: bool = False) -> Optional[str]:
        """Process data with NVTabular"""
        if not self.nvt_enabled:
            logger.warning("NVTabular not available")
            return None
        
        try:
            output_dir = str(self.config.NVT_PROCESSED_DIR)
            
            if os.path.exists(output_dir) and not force_reprocess:
                try:
                    test_dataset = Dataset(output_dir, engine='parquet')
                    logger.info(f"Using existing NVTabular processed data from {output_dir}")
                    return output_dir
                except:
                    logger.warning("Existing data corrupted, reprocessing")
                    shutil.rmtree(output_dir)
            
            if os.path.exists(output_dir):
                logger.info(f"Removing existing directory {output_dir}")
                shutil.rmtree(output_dir)
            
            logger.info("NVTabular data processing started")
            start_time = time.time()
            
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'train_no_seq.parquet')
            
            logger.info("Creating temp file without seq column")
            pf = pq.ParquetFile(str(self.config.TRAIN_PATH))
            exclude_cols = self.config.NVT_CONFIG['exclude_columns']
            cols = [c for c in pf.schema.names if c not in exclude_cols]
            
            logger.info(f"Total columns: {len(pf.schema.names)}")
            logger.info(f"Using columns: {len(cols)} (excluded: {exclude_cols})")
            
            df = pd.read_parquet(str(self.config.TRAIN_PATH), columns=cols)
            logger.info(f"Loaded {len(df):,} rows")
            
            if self.target_column is None:
                self.target_column = self.column_analyzer.detect_target_column(df)
            
            df.to_parquet(temp_path, index=False)
            del df
            gc.collect()
            logger.info("Temp file created")
            
            logger.info("Creating NVTabular Dataset")
            self.memory_monitor.force_memory_cleanup()
            
            dataset = Dataset(
                temp_path,
                engine='parquet',
                part_size=self.config.NVT_CONFIG['part_size']
            )
            logger.info("Dataset created")
            
            logger.info("Creating NVTabular workflow")
            workflow = self._create_nvt_workflow()
            
            logger.info("Fitting workflow")
            workflow.fit(dataset)
            logger.info("Workflow fitted")
            
            logger.info(f"Transforming and saving to {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            self.memory_monitor.force_memory_cleanup()
            
            workflow.transform(dataset).to_parquet(
                output_path=output_dir,
                shuffle=nvt.io.Shuffle.PER_PARTITION,
                out_files_per_proc=self.config.NVT_CONFIG['out_files_per_proc']
            )
            
            workflow_path = os.path.join(output_dir, 'workflow')
            workflow.save(workflow_path)
            logger.info(f"Workflow saved to {workflow_path}")
            
            shutil.rmtree(temp_dir)
            
            elapsed = time.time() - start_time
            logger.info(f"NVTabular processing completed in {elapsed:.1f}s")
            
            self.memory_monitor.force_memory_cleanup()
            return output_dir
            
        except Exception as e:
            logger.error(f"NVTabular processing failed: {e}")
            return None
    
    def _create_nvt_workflow(self):
        """Create NVTabular workflow"""
        logger.info("Creating XGBoost workflow")
        
        cat_features = self.config.CATEGORICAL_FEATURES
        cont_features = self.config.CONTINUOUS_FEATURES
        
        logger.info(f"Categorical: {len(cat_features)} columns")
        logger.info(f"Continuous: {len(cont_features)} columns")
        logger.info(f"Total features: {len(cat_features) + len(cont_features)}")
        
        cat_workflow = cat_features >> ops.Categorify(
            freq_threshold=self.config.NVT_CONFIG['freq_threshold'],
            max_size=self.config.NVT_CONFIG['max_size']
        )
        
        cont_workflow = cont_features >> ops.FillMissing(fill_val=0)
        
        target_col = self.target_column if self.target_column else 'clicked'
        workflow = nvt.Workflow(cat_workflow + cont_workflow + [target_col])
        
        logger.info("Workflow created (no normalization for tree models)")
        return workflow
    
    def load_nvt_processed_data(self, processed_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load NVTabular processed data"""
        try:
            logger.info("Loading NVTabular processed data")
            start_time = time.time()
            
            dataset = Dataset(processed_dir, engine='parquet', part_size='256MB')
            logger.info("Converting to GPU DataFrame")
            gdf = dataset.to_ddf().compute()
            
            logger.info(f"Loaded {len(gdf):,} rows x {len(gdf.columns)} columns")
            logger.info(f"Time: {time.time() - start_time:.1f}s")
            
            self.memory_monitor.log_memory_status("After loading", force=True)
            
            target_col = self.target_column if self.target_column else 'clicked'
            y = gdf[target_col].to_numpy()
            X = gdf.drop(target_col, axis=1)
            
            for col in X.columns:
                if X[col].dtype != 'float32':
                    X[col] = X[col].astype('float32')
            
            X_np = X.to_numpy()
            
            logger.info(f"Data prepared: X{X_np.shape}, y{y.shape}")
            
            del X, gdf
            self.memory_monitor.force_memory_cleanup()
            
            return X_np, y
            
        except Exception as e:
            logger.error(f"Loading NVTabular processed data failed: {e}")
            return None, None

class StreamingDataLoader:
    """Streaming data loader with memory management and quick mode support"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.column_analyzer = DataColumnAnalyzer(config)
        self.target_column = None
        self.temp_dir = tempfile.mkdtemp()
        self.quick_mode = False
        
        logger.info("Streaming data loader initialization completed")
    
    def __del__(self):
        """Cleanup temporary directory on destruction"""
        try:
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
        """Load small sample data for quick testing (50 samples)"""
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
            
            df = self._optimize_dataframe_memory(df, for_feature_engineering=True)
            
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
    
    def load_full_data_streaming(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete processing via streaming with memory management"""
        logger.info("=== Full data streaming loading started ===")
        
        try:
            self.memory_monitor.log_memory_status("Streaming start", force=True)
            
            if not self._validate_files():
                raise ValueError("Data files do not exist")
            
            logger.info("Training data streaming processing started")
            train_df = self._stream_process_file(str(self.config.TRAIN_PATH), is_train=True)
            
            if train_df is None or train_df.empty:
                raise ValueError("Training data streaming processing failed")
            
            temp_train_file = os.path.join(self.temp_dir, 'temp_train.pkl')
            with open(temp_train_file, 'wb') as f:
                pickle.dump(train_df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            train_shape = train_df.shape
            target_column = self.target_column
            
            del train_df
            
            self.memory_monitor.force_memory_cleanup(intensive=True)
            time.sleep(2)
            self.memory_monitor.log_memory_status("Training data saved, memory cleared", force=True)
            
            logger.info("Test data streaming processing started") 
            test_df = self._stream_process_file(str(self.config.TEST_PATH), is_train=False)
            
            if test_df is None or test_df.empty:
                raise ValueError("Test data streaming processing failed")
            
            logger.info("Reloading training data from temporary storage")
            with open(temp_train_file, 'rb') as f:
                train_df = pickle.load(f)
            
            os.remove(temp_train_file)
            
            self.target_column = target_column
            
            self.memory_monitor.log_memory_status("Streaming completed", force=True)
            
            logger.info(f"=== Full data streaming completed - Training: {train_df.shape}, Test: {test_df.shape} ===")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Streaming data loading failed: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            
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
            
            if is_train:
                sample_table = parquet_file.read_row_group(0)
                sample_df = sample_table.to_pandas()
                
                if len(sample_df) > 1000:
                    sample_df = sample_df.sample(n=1000, random_state=42)
                
                self.target_column = self.column_analyzer.detect_target_column(sample_df)
                logger.info(f"Detected target column: {self.target_column}")
                
                del sample_df, sample_table
                gc.collect()
            
            all_chunks = []
            processed_rows = 0
            batch_size = 1
            
            for rg_start in range(0, num_row_groups, batch_size):
                rg_end = min(rg_start + batch_size, num_row_groups)
                
                try:
                    pressure = self.memory_monitor.check_memory_pressure()
                    if pressure['should_abort']:
                        logger.warning(f"Memory limit approaching - Processed rows: {processed_rows}, continuing attempt")
                        
                        self.memory_monitor.force_memory_cleanup(intensive=True)
                        time.sleep(1)
                        
                        pressure_after = self.memory_monitor.check_memory_pressure()
                        if pressure_after['should_abort']:
                            logger.error(f"Memory limit reached even after cleanup - Stopping: {processed_rows} rows")
                            break
                    
                    row_group_tables = []
                    for rg_idx in range(rg_start, rg_end):
                        table = parquet_file.read_row_group(rg_idx)
                        row_group_tables.append(table)
                   
                    if len(row_group_tables) == 1:
                        combined_table = row_group_tables[0]
                    else:
                        combined_table = pa.concat_tables(row_group_tables)
                    
                    chunk_df = combined_table.to_pandas()
                    
                    del row_group_tables, combined_table
                    
                    chunk_df = self._optimize_dataframe_memory(chunk_df, for_feature_engineering=True)
                    
                    all_chunks.append(chunk_df)
                    processed_rows += len(chunk_df)
                    
                    logger.info(f"Row groups {rg_start+1}-{rg_end} processed: {len(chunk_df):,} rows")
                    
                    if len(all_chunks) % 2 == 0:
                        self.memory_monitor.force_memory_cleanup()
                    
                    progress_pct = (rg_end / num_row_groups) * 100
                    logger.info(f"Progress: {progress_pct:.1f}% ({processed_rows:,} rows)")
                    
                except Exception as e:
                    logger.error(f"Row group batch {rg_start+1}-{rg_end} processing failed: {e}")
                    
                    self.memory_monitor.force_memory_cleanup(intensive=True)
                    
                    continue
            
            if not all_chunks:
                logger.error("No data processed successfully")
                return pd.DataFrame()
            
            logger.info(f"Final combination: {len(all_chunks)} chunks")
            final_df = self._combine_chunks_memory_efficient(all_chunks)
            
            logger.info(f"File streaming completed: {len(final_df):,} rows")
            
            self.memory_monitor.force_memory_cleanup(intensive=True)
            
            return final_df
            
        except Exception as e:
            logger.error(f"File streaming processing failed: {e}")
            return pd.DataFrame()
    
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
    
    def _combine_chunks_memory_efficient(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine chunks with memory management"""
        try:
            if not chunks:
                return pd.DataFrame()
            
            if len(chunks) == 1:
                return chunks[0]
            
            logger.info(f"Combining {len(chunks)} chunks progressively")
            
            result = chunks[0]
            for i, chunk in enumerate(chunks[1:], 1):
                try:
                    result = pd.concat([result, chunk], ignore_index=True)
                    
                    if i % 3 == 0:
                        self.memory_monitor.force_memory_cleanup()
                        
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
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    return chunk
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
    """Large data loader with quick mode and NVTabular support"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.target_column = None
        self.quick_mode = False
        self.use_nvtabular = config.NVTABULAR_ENABLED
        
        self.nvt_loader = NVTabularDataLoader(config) if self.use_nvtabular else None
        self.streaming_loader = StreamingDataLoader(config)
        
        self.loading_stats = {
            'start_time': time.time(),
            'data_loaded': False,
            'train_rows': 0,
            'test_rows': 0,
            'loading_time': 0.0,
            'memory_usage': 0.0,
            'nvtabular_used': False
        }
        
        logger.info("Large data loader initialization completed")
    
    def set_quick_mode(self, quick_mode: bool):
        """Enable or disable quick mode for testing"""
        self.quick_mode = quick_mode
        self.streaming_loader.set_quick_mode(quick_mode)
        
        if quick_mode:
            logger.info("Large data loader set to quick mode (50 samples)")
        else:
            logger.info("Large data loader set to full mode (complete dataset)")
    
    def load_quick_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load small sample data for quick testing"""
        logger.info("=== Quick sample data loading via LargeDataLoader ===")
        
        result = self.streaming_loader.load_quick_sample_data()
        
        self.target_column = self.streaming_loader.get_detected_target_column()
        
        self.loading_stats.update({
            'data_loaded': True,
            'train_rows': result[0].shape[0] if result[0] is not None else 0,
            'test_rows': result[1].shape[0] if result[1] is not None else 0,
            'loading_time': time.time() - self.loading_stats['start_time'],
            'nvtabular_used': False
        })
        
        return result
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data processing (full dataset)"""
        logger.info("=== Complete data processing started ===")
        
        if self.use_nvtabular and self.nvt_loader:
            logger.info("Attempting NVTabular processing")
            processed_dir = self.nvt_loader.process_with_nvtabular()
            
            if processed_dir:
                self.loading_stats['nvtabular_used'] = True
                logger.info("NVTabular processing successful")
                return self._load_nvt_as_dataframes(processed_dir)
        
        logger.info("Using standard streaming loader")
        result = self.streaming_loader.load_full_data_streaming()
        
        self.target_column = self.streaming_loader.get_detected_target_column()
        
        self.loading_stats.update({
            'data_loaded': True,
            'train_rows': result[0].shape[0] if result[0] is not None else 0,
            'test_rows': result[1].shape[0] if result[1] is not None else 0,
            'loading_time': time.time() - self.loading_stats['start_time'],
            'nvtabular_used': False
        })
        
        return result
    
    def _load_nvt_as_dataframes(self, processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load NVTabular processed data as DataFrames for compatibility"""
        try:
            X_np, y_np = self.nvt_loader.load_nvt_processed_data(processed_dir)
            
            if X_np is None or y_np is None:
                logger.warning("NVTabular data loading failed, falling back to streaming")
                return self.streaming_loader.load_full_data_streaming()
            
            train_df = pd.DataFrame(X_np)
            target_col = self.nvt_loader.target_column if self.nvt_loader.target_column else 'clicked'
            train_df[target_col] = y_np
            
            self.target_column = target_col
            
            test_df = self.streaming_loader._stream_process_file(str(self.config.TEST_PATH), is_train=False)
            
            self.loading_stats.update({
                'data_loaded': True,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'loading_time': time.time() - self.loading_stats['start_time'],
                'nvtabular_used': True
            })
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Loading NVTabular as DataFrames failed: {e}")
            return self.streaming_loader.load_full_data_streaming()
    
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
        
        print("Testing quick mode...")
        loader.set_quick_mode(True)
        train_df, test_df = loader.load_quick_sample_data()
        
        print(f"Quick mode results:")
        print(f"Training data: {train_df.shape}")
        print(f"Test data: {test_df.shape}")
        print(f"Detected target column: {loader.get_detected_target_column()}")
        print(f"Total samples: {len(train_df) + len(test_df)}")
        
        stats = loader.get_loading_stats()
        print(f"\nLoading stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")