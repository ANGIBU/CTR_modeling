# data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, Iterator
import logging
import gc
import mmap
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit

# PyArrow import 안전 처리
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("PyArrow가 설치되지 않았습니다. 기본 pandas 로딩을 사용합니다.")

# Psutil import 안전 처리
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil이 설치되지 않았습니다. 메모리 모니터링 기능이 제한됩니다.")

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """현재 프로세스 메모리 사용량 (GB)"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024**3)
            except:
                return 0.0
        return 0.0
    
    @staticmethod
    def get_available_memory() -> float:
        """사용 가능한 메모리 (GB)"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.virtual_memory().available / (1024**3)
            except:
                return 45.0
        return 45.0
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """메모리 상태 정보"""
        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                process = psutil.Process()
                
                return {
                    'total_gb': vm.total / (1024**3),
                    'available_gb': vm.available / (1024**3),
                    'used_gb': vm.used / (1024**3),
                    'process_gb': process.memory_info().rss / (1024**3),
                    'usage_percent': vm.percent
                }
            except:
                pass
        
        return {
            'total_gb': 64.0,
            'available_gb': 45.0,
            'used_gb': 19.0,
            'process_gb': 0.0,
            'usage_percent': 30.0
        }
    
    @staticmethod
    def force_memory_cleanup():
        """강제 메모리 정리"""
        try:
            gc.collect()
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except:
                pass
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")

class ChunkedParquetReader:
    """청킹 기반 Parquet 리더"""
    
    def __init__(self, file_path: str, chunk_size: int = 500000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.parquet_file = None
        self.total_rows = 0
        self.current_position = 0
        
    def __enter__(self):
        if PYARROW_AVAILABLE:
            try:
                self.parquet_file = pq.ParquetFile(self.file_path)
                self.total_rows = self.parquet_file.metadata.num_rows
                return self
            except Exception as e:
                logger.warning(f"PyArrow로 파일 열기 실패: {e}. pandas 사용")
        
        # PyArrow 실패 시 pandas 사용
        try:
            sample_df = pd.read_parquet(self.file_path, engine='auto', nrows=10000)
            file_size = os.path.getsize(self.file_path)
            sample_size = sample_df.memory_usage(deep=True).sum()
            estimated_rows = int((file_size / sample_size) * 10000 * 0.9)
            self.total_rows = estimated_rows
            del sample_df
            gc.collect()
        except Exception as e:
            logger.warning(f"행 수 추정 실패: {e}. 기본값 사용")
            self.total_rows = 1527298
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parquet_file:
            self.parquet_file = None
        gc.collect()
    
    def get_total_rows(self) -> int:
        """전체 행 수 반환"""
        return self.total_rows
    
    def read_all_data(self) -> pd.DataFrame:
        """전체 데이터 로딩"""
        try:
            if PYARROW_AVAILABLE and self.parquet_file:
                table = self.parquet_file.read(use_threads=True)
                df = table.to_pandas()
            else:
                df = pd.read_parquet(self.file_path, engine='auto')
            
            df = self._optimize_chunk_memory(df)
            return df
            
        except Exception as e:
            logger.error(f"전체 데이터 로딩 실패: {e}")
            return pd.DataFrame()
    
    def read_chunk(self, start_row: int = None, num_rows: int = None) -> pd.DataFrame:
        """특정 청크 읽기"""
        if start_row is None:
            start_row = self.current_position
        
        if num_rows is None:
            num_rows = min(self.chunk_size, self.total_rows - start_row)
        
        if start_row >= self.total_rows:
            return pd.DataFrame()
        
        try:
            if PYARROW_AVAILABLE and self.parquet_file:
                try:
                    table = self.parquet_file.read(columns=None, use_threads=True)
                    df = table.to_pandas()
                    
                    end_row = min(start_row + num_rows, len(df))
                    df = df.iloc[start_row:end_row].copy()
                    df = self._optimize_chunk_memory(df)
                    
                except Exception as e:
                    logger.warning(f"PyArrow 청크 읽기 실패: {e}. pandas 사용")
                    df = pd.read_parquet(self.file_path, engine='auto')
                    end_row = min(start_row + num_rows, len(df))
                    df = df.iloc[start_row:end_row].copy()
                    df = self._optimize_chunk_memory(df)
            else:
                df = pd.read_parquet(self.file_path, engine='auto')
                end_row = min(start_row + num_rows, len(df))
                df = df.iloc[start_row:end_row].copy()
                df = self._optimize_chunk_memory(df)
            
            self.current_position = start_row + len(df)
            gc.collect()
            
            return df
            
        except Exception as e:
            logger.error(f"청크 읽기 실패: {e}")
            return pd.DataFrame()
    
    def _optimize_chunk_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """청크 단위 메모리 최적화"""
        try:
            # 정수형 최적화
            for col in df.select_dtypes(include=['int64', 'int32']).columns:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= 0:
                        if col_max < 255:
                            df[col] = df[col].astype('uint8')
                        elif col_max < 65535:
                            df[col] = df[col].astype('uint16')
                    else:
                        if col_min > -128 and col_max < 127:
                            df[col] = df[col].astype('int8')
                        elif col_min > -32768 and col_max < 32767:
                            df[col] = df[col].astype('int16')
                except:
                    continue
            
            # 실수형 최적화
            for col in df.select_dtypes(include=['float64']).columns:
                try:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                except:
                    continue
            
            return df
        except Exception as e:
            logger.warning(f"청크 메모리 최적화 실패: {e}")
            return df
    
    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """청크 단위로 순회"""
        self.current_position = 0
        
        while self.current_position < self.total_rows:
            chunk = self.read_chunk()
            if chunk.empty:
                break
            yield chunk
            
            if self.current_position % (self.chunk_size * 5) == 0:
                MemoryMonitor.force_memory_cleanup()

class DataLoader:
    """전체 데이터 처리 특화 데이터 로딩 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.train_data = None
        self.test_data = None
        self.feature_columns = None
        self.target_column = 'clicked'
        
        # 메모리 설정 (64GB RAM 환경)
        self.memory_config = config.get_safe_memory_limits()
        self.data_config = config.get_data_config()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 데이터 로딩"""
        try:
            logger.info("기본 데이터 로딩 시작")
            
            available_memory = self.memory_monitor.get_available_memory()
            logger.info(f"사용 가능 메모리: {available_memory:.2f}GB")
            
            # 학습 데이터 로딩
            max_train_size = self.data_config['max_train_size']
            
            with ChunkedParquetReader(self.config.TRAIN_PATH, 200000) as train_reader:
                train_total_rows = train_reader.get_total_rows()
                logger.info(f"전체 학습 데이터: {train_total_rows:,}행")
                
                if train_total_rows <= max_train_size:
                    self.train_data = train_reader.read_all_data()
                    logger.info(f"전체 학습 데이터 로딩: {self.train_data.shape}")
                else:
                    sample_indices = np.random.choice(train_total_rows, max_train_size, replace=False)
                    sample_indices.sort()
                    
                    chunks = []
                    for start_idx in range(0, len(sample_indices), 100000):
                        end_idx = min(start_idx + 100000, len(sample_indices))
                        chunk_indices = sample_indices[start_idx:end_idx]
                        
                        # 연속된 인덱스 범위로 변환
                        min_idx, max_idx = chunk_indices.min(), chunk_indices.max()
                        chunk_df = train_reader.read_chunk(min_idx, max_idx - min_idx + 1)
                        
                        if not chunk_df.empty:
                            # 실제 필요한 행만 선택
                            relative_indices = chunk_indices - min_idx
                            valid_indices = relative_indices[relative_indices < len(chunk_df)]
                            if len(valid_indices) > 0:
                                sampled_chunk = chunk_df.iloc[valid_indices].copy()
                                chunks.append(sampled_chunk)
                        
                        if start_idx % 500000 == 0:
                            MemoryMonitor.force_memory_cleanup()
                    
                    if chunks:
                        self.train_data = pd.concat(chunks, ignore_index=True)
                        logger.info(f"샘플링된 학습 데이터: {self.train_data.shape}")
                    else:
                        logger.error("학습 데이터 로딩 실패")
                        raise ValueError("학습 데이터를 로딩할 수 없습니다")
            
            MemoryMonitor.force_memory_cleanup()
            
            # 테스트 데이터 전체 로딩 (필수)
            with ChunkedParquetReader(self.config.TEST_PATH, 200000) as test_reader:
                test_total_rows = test_reader.get_total_rows()
                logger.info(f"전체 테스트 데이터: {test_total_rows:,}행")
                
                # 반드시 전체 테스트 데이터 로딩
                self.test_data = test_reader.read_all_data()
                logger.info(f"전체 테스트 데이터 로딩 완료: {self.test_data.shape}")
                
                if self.test_data.empty or len(self.test_data) < 1000000:
                    logger.error("테스트 데이터 로딩이 불완전합니다")
                    raise ValueError("테스트 데이터를 완전히 로딩하지 못했습니다")
            
            MemoryMonitor.force_memory_cleanup()
            
            self.feature_columns = [col for col in self.train_data.columns 
                                  if col != self.target_column]
            logger.info(f"피처 컬럼 수: {len(self.feature_columns)}")
            
            if self.target_column in self.train_data.columns:
                actual_ctr = self.train_data[self.target_column].mean()
                logger.info(f"실제 CTR: {actual_ctr:.4f}")
            
            return self.train_data, self.test_data
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            MemoryMonitor.force_memory_cleanup()
            raise
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """전체 데이터 처리 최적화 로딩"""
        logger.info("전체 데이터 처리 최적화 로딩 시작")
        
        # 초기 메모리 상태 확인
        memory_info = self.memory_monitor.get_memory_info()
        logger.info(f"초기 메모리: 사용가능 {memory_info['available_gb']:.1f}GB / 전체 {memory_info['total_gb']:.1f}GB")
        
        try:
            # 학습 데이터 로딩
            train_df = self._load_train_data_optimized()
            
            MemoryMonitor.force_memory_cleanup()
            memory_after_train = self.memory_monitor.get_memory_info()
            logger.info(f"학습 데이터 후 메모리: {memory_after_train['available_gb']:.1f}GB")
            
            # 테스트 데이터 전체 로딩 (필수)
            test_df = self._load_full_test_data()
            
            MemoryMonitor.force_memory_cleanup()
            memory_after_test = self.memory_monitor.get_memory_info()
            logger.info(f"테스트 데이터 후 메모리: {memory_after_test['available_gb']:.1f}GB")
            
            # 피처 컬럼 정의
            self.feature_columns = [col for col in train_df.columns 
                                  if col != self.target_column]
            
            # CTR 정보
            if self.target_column in train_df.columns:
                actual_ctr = train_df[self.target_column].mean()
                logger.info(f"실제 CTR: {actual_ctr:.4f}")
            
            logger.info("전체 데이터 처리 최적화 로딩 완료")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"전체 데이터 로딩 실패: {str(e)}")
            MemoryMonitor.force_memory_cleanup()
            raise
    
    def _load_train_data_optimized(self) -> pd.DataFrame:
        """최적화된 학습 데이터 로딩"""
        logger.info("최적화된 학습 데이터 로딩 시작")
        
        try:
            with ChunkedParquetReader(self.config.TRAIN_PATH, 200000) as reader:
                total_rows = reader.get_total_rows()
                logger.info(f"전체 학습 데이터: {total_rows:,}행")
                
                max_size = self.data_config['max_train_size']
                
                if total_rows <= max_size:
                    logger.info("전체 학습 데이터 로딩")
                    df = reader.read_all_data()
                else:
                    logger.info(f"학습 데이터 샘플링: {max_size:,}/{total_rows:,}")
                    
                    # 균등 샘플링
                    step = total_rows // max_size
                    sample_indices = np.arange(0, total_rows, step)[:max_size]
                    
                    chunks = []
                    current_start = 0
                    chunk_size = 200000
                    
                    while current_start < total_rows:
                        chunk_end = min(current_start + chunk_size, total_rows)
                        chunk = reader.read_chunk(current_start, chunk_size)
                        
                        if not chunk.empty:
                            # 청크 내에서 필요한 인덱스만 선택
                            chunk_indices = sample_indices[
                                (sample_indices >= current_start) & 
                                (sample_indices < chunk_end)
                            ] - current_start
                            
                            if len(chunk_indices) > 0:
                                valid_indices = chunk_indices[chunk_indices < len(chunk)]
                                if len(valid_indices) > 0:
                                    sampled_chunk = chunk.iloc[valid_indices].copy()
                                    chunks.append(sampled_chunk)
                        
                        current_start = chunk_end
                        
                        if len(chunks) % 10 == 0:
                            MemoryMonitor.force_memory_cleanup()
                    
                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        logger.info(f"샘플링 완료: {df.shape}")
                    else:
                        raise ValueError("학습 데이터 샘플링 실패")
                
                df = self._optimize_memory_usage(df)
                logger.info(f"학습 데이터 로딩 완료: {df.shape}")
                return df
                
        except Exception as e:
            logger.error(f"학습 데이터 로딩 실패: {e}")
            raise
    
    def _load_full_test_data(self) -> pd.DataFrame:
        """전체 테스트 데이터 로딩 (필수)"""
        logger.info("전체 테스트 데이터 로딩 시작")
        
        try:
            with ChunkedParquetReader(self.config.TEST_PATH, 200000) as reader:
                total_rows = reader.get_total_rows()
                logger.info(f"전체 테스트 데이터: {total_rows:,}행")
                
                # 반드시 전체 데이터 로딩
                df = reader.read_all_data()
                
                if df.empty:
                    raise ValueError("테스트 데이터 로딩 실패")
                
                if len(df) < 1500000:
                    logger.warning(f"테스트 데이터 크기가 예상보다 작습니다: {len(df):,}")
                
                df = self._optimize_memory_usage(df)
                logger.info(f"전체 테스트 데이터 로딩 완료: {df.shape}")
                
                return df
                
        except Exception as e:
            logger.error(f"테스트 데이터 로딩 실패: {e}")
            raise
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """메모리 사용량 최적화"""
        logger.info("메모리 사용량 최적화 시작")
        
        original_memory = df.memory_usage(deep=True).sum() / (1024**2)
        
        try:
            # 정수형 최적화
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= 0:
                        if col_max < 255:
                            df[col] = df[col].astype('uint8')
                        elif col_max < 65535:
                            df[col] = df[col].astype('uint16')
                        elif col_max < 4294967295:
                            df[col] = df[col].astype('uint32')
                    else:
                        if col_min > -128 and col_max < 127:
                            df[col] = df[col].astype('int8')
                        elif col_min > -32768 and col_max < 32767:
                            df[col] = df[col].astype('int16')
                        elif col_min > -2147483648 and col_max < 2147483647:
                            df[col] = df[col].astype('int32')
                except Exception as e:
                    logger.warning(f"{col} 정수형 최적화 실패: {e}")
                    continue
            
            # 실수형 최적화
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                try:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                except Exception as e:
                    logger.warning(f"{col} 실수형 최적화 실패: {e}")
                    continue
            
            optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            logger.info(f"메모리 최적화 완료: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({reduction:.1f}% 감소)")
            
        except Exception as e:
            logger.warning(f"메모리 최적화 실패: {str(e)}")
        
        gc.collect()
        
        return df
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리 수행"""
        logger.info("기본 전처리 시작")
        
        df_processed = df.copy()
        
        # 결측치 확인 및 처리
        missing_info = df_processed.isnull().sum()
        total_missing = missing_info.sum()
        
        if total_missing > 0:
            logger.warning(f"결측치 발견: {total_missing:,}개")
            
            for col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    try:
                        col_dtype = str(df_processed[col].dtype)
                        
                        if col_dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
                            df_processed[col].fillna(0, inplace=True)
                        elif col_dtype in ['float16', 'float32', 'float64']:
                            median_val = df_processed[col].median()
                            fill_value = median_val if not pd.isna(median_val) else 0.0
                            df_processed[col].fillna(fill_value, inplace=True)
                        elif col_dtype in ['object', 'category', 'string', 'bool']:
                            mode_val = df_processed[col].mode()
                            fill_value = mode_val[0] if len(mode_val) > 0 else 'unknown'
                            df_processed[col].fillna(fill_value, inplace=True)
                        else:
                            df_processed[col].fillna(0, inplace=True)
                    except Exception as e:
                        logger.warning(f"{col} 결측치 처리 실패: {e}")
                        df_processed[col].fillna(0, inplace=True)
        
        # 무한값 처리
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns[:30]
        for col in numeric_cols:
            try:
                inf_count = np.isinf(df_processed[col]).sum()
                if inf_count > 0:
                    finite_values = df_processed[col][np.isfinite(df_processed[col])]
                    if len(finite_values) > 0:
                        upper_bound = finite_values.quantile(0.99)
                        lower_bound = finite_values.quantile(0.01)
                        
                        df_processed.loc[df_processed[col] == np.inf, col] = upper_bound
                        df_processed.loc[df_processed[col] == -np.inf, col] = lower_bound
                    else:
                        df_processed[col].replace([np.inf, -np.inf], 0, inplace=True)
            except Exception as e:
                logger.warning(f"{col} 무한값 처리 실패: {e}")
                df_processed[col].replace([np.inf, -np.inf], 0, inplace=True)
        
        df_processed = self._optimize_memory_usage(df_processed)
        
        logger.info("기본 전처리 완료")
        return df_processed
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 요약 정보 생성"""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = {
                'count': len(numeric_cols)
            }
        
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts()
            summary['target_distribution'] = {
                'counts': target_dist.to_dict(),
                'ctr': df[self.target_column].mean()
            }
        
        return summary
    
    def memory_efficient_train_test_split(self,
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """메모리 효율적인 데이터 분할"""
        logger.info("메모리 효율적인 데이터 분할 시작")
        
        try:
            if len(X_train) != len(y_train):
                raise ValueError(f"X와 y의 길이가 다릅니다. X: {len(X_train)}, y: {len(y_train)}")
            
            split_idx = int(len(X_train) * (1 - test_size))
            logger.info(f"분할 인덱스: {split_idx}, 전체 크기: {len(X_train)}")
            
            memory_before = self.memory_monitor.get_memory_usage()
            
            X_train_split = X_train.iloc[:split_idx].copy()
            X_val_split = X_train.iloc[split_idx:].copy()
            y_train_split = y_train.iloc[:split_idx].copy()
            y_val_split = y_train.iloc[split_idx:].copy()
            
            del X_train, y_train
            MemoryMonitor.force_memory_cleanup()
            
            memory_after = self.memory_monitor.get_memory_usage()
            logger.info(f"메모리 사용량: {memory_before:.2f}GB → {memory_after:.2f}GB")
            
            self._validate_split_results(X_train_split, X_val_split, y_train_split, y_val_split)
            
            logger.info(f"메모리 효율적 분할 완료 - 학습: {X_train_split.shape}, 검증: {X_val_split.shape}")
            
            return X_train_split, X_val_split, y_train_split, y_val_split
            
        except Exception as e:
            logger.error(f"메모리 효율적 데이터 분할 실패: {str(e)}")
            MemoryMonitor.force_memory_cleanup()
            raise
    
    def _validate_split_results(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                               y_train: pd.Series, y_val: pd.Series):
        """분할 결과 검증"""
        logger.info("분할 결과 검증 시작")
        
        assert len(X_train) == len(y_train), f"학습 데이터 길이 불일치: {len(X_train)} != {len(y_train)}"
        assert len(X_val) == len(y_val), f"검증 데이터 길이 불일치: {len(X_val)} != {len(y_val)}"
        assert len(X_train.columns) == len(X_val.columns), f"피처 수 불일치: {len(X_train.columns)} != {len(X_val.columns)}"
        
        train_dist = y_train.value_counts()
        val_dist = y_val.value_counts()
        
        logger.info(f"학습 데이터 타겟 분포: {train_dist.to_dict()}")
        logger.info(f"검증 데이터 타겟 분포: {val_dist.to_dict()}")
        
        train_ctr = y_train.mean()
        val_ctr = y_val.mean()
        logger.info(f"학습 CTR: {train_ctr:.4f}, 검증 CTR: {val_ctr:.4f}")
        
        logger.info("분할 결과 검증 완료")
    
    def load_submission_template(self) -> pd.DataFrame:
        """제출 템플릿 로딩"""
        try:
            submission = pd.read_csv(self.config.SUBMISSION_PATH)
            logger.info(f"제출 템플릿 로딩 완료: {submission.shape}")
            return submission
        except Exception as e:
            logger.error(f"제출 템플릿 로딩 실패: {str(e)}")
            raise

class DataValidator:
    """데이터 품질 검증 클래스"""
    
    @staticmethod
    def validate_data_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """학습/테스트 데이터 일관성 검증"""
        validation_results = {}
        
        train_cols = set(train_df.columns) - {'clicked'}
        test_cols = set(test_df.columns)
        
        validation_results['missing_in_test'] = list(train_cols - test_cols)
        validation_results['extra_in_test'] = list(test_cols - train_cols)
        
        common_cols = train_cols & test_cols
        dtype_mismatches = []
        
        for col in list(common_cols)[:30]:
            try:
                train_dtype = str(train_df[col].dtype)
                test_dtype = str(test_df[col].dtype)
                
                if train_dtype == 'category' and test_dtype == 'object':
                    continue
                if train_dtype == 'object' and test_dtype == 'category':
                    continue
                    
                if train_dtype != test_dtype:
                    dtype_mismatches.append({
                        'column': col,
                        'train_dtype': train_dtype,
                        'test_dtype': test_dtype
                    })
            except:
                continue
        
        validation_results['dtype_mismatches'] = dtype_mismatches
        
        return validation_results