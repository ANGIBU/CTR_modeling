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
                return 32.0
        return 32.0
    
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
        
        # 기본값 반환
        return {
            'total_gb': 64.0,
            'available_gb': 32.0,
            'used_gb': 32.0,
            'process_gb': 0.0,
            'usage_percent': 50.0
        }
    
    @staticmethod
    def force_memory_cleanup():
        """강제 메모리 정리"""
        try:
            gc.collect()
            # Windows 메모리 정리 시도
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
    
    def __init__(self, file_path: str, chunk_size: int = 200000):  # 기본 청크 크기 감소
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
            # 메모리 절약을 위해 샘플만 읽어서 행 수 추정
            sample_df = pd.read_parquet(self.file_path, engine='auto', nrows=10000)
            file_size = os.path.getsize(self.file_path)
            sample_size = sample_df.memory_usage(deep=True).sum()
            estimated_rows = int((file_size / sample_size) * 10000 * 0.8)  # 보수적 추정
            self.total_rows = estimated_rows
            del sample_df
            gc.collect()
        except Exception as e:
            logger.warning(f"행 수 추정 실패: {e}. 기본값 사용")
            self.total_rows = 1000000
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parquet_file:
            self.parquet_file = None
        gc.collect()
    
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
                # PyArrow 방식
                try:
                    table = self.parquet_file.read(columns=None, use_threads=True)
                    df = table.to_pandas()
                    
                    end_row = min(start_row + num_rows, len(df))
                    df = df.iloc[start_row:end_row].copy()
                    
                    # 메모리 최적화
                    df = self._optimize_chunk_memory(df)
                    
                except Exception as e:
                    logger.warning(f"PyArrow 청크 읽기 실패: {e}. pandas 사용")
                    # pandas로 fallback
                    df = pd.read_parquet(self.file_path, engine='auto')
                    end_row = min(start_row + num_rows, len(df))
                    df = df.iloc[start_row:end_row].copy()
                    df = self._optimize_chunk_memory(df)
            else:
                # pandas 기본 방식
                try:
                    df = pd.read_parquet(self.file_path, engine='auto')
                    end_row = min(start_row + num_rows, len(df))
                    df = df.iloc[start_row:end_row].copy()
                    df = self._optimize_chunk_memory(df)
                except Exception as e:
                    logger.error(f"pandas 청크 읽기 실패: {e}")
                    return pd.DataFrame()
            
            self.current_position = start_row + len(df)
            
            # 메모리 정리
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
            
            # 메모리 정리
            if self.current_position % (self.chunk_size * 3) == 0:
                MemoryMonitor.force_memory_cleanup()
    
    def get_sample(self, sample_size: int, random_state: int = 42) -> pd.DataFrame:
        """메모리 효율적인 샘플링"""
        if sample_size >= self.total_rows:
            # 전체 데이터 반환
            try:
                if PYARROW_AVAILABLE and self.parquet_file:
                    table = self.parquet_file.read()
                    df = table.to_pandas()
                    return self._optimize_chunk_memory(df)
                else:
                    df = pd.read_parquet(self.file_path, engine='auto')
                    return self._optimize_chunk_memory(df)
            except Exception as e:
                logger.error(f"전체 데이터 로딩 실패: {e}")
                return pd.DataFrame()
        
        # 단계적 샘플링 (메모리 절약)
        np.random.seed(random_state)
        
        # 청크 단위로 샘플링
        target_chunks = max(1, sample_size // self.chunk_size + 1)
        chunk_sample_size = sample_size // target_chunks
        
        sample_chunks = []
        processed_chunks = 0
        current_pos = 0
        
        while current_pos < self.total_rows and processed_chunks < target_chunks:
            try:
                chunk = self.read_chunk(current_pos, self.chunk_size)
                if chunk.empty:
                    break
                
                # 청크에서 샘플링
                if len(chunk) > chunk_sample_size:
                    chunk_sample = chunk.sample(n=chunk_sample_size, random_state=random_state).copy()
                else:
                    chunk_sample = chunk.copy()
                
                sample_chunks.append(chunk_sample)
                processed_chunks += 1
                current_pos += self.chunk_size
                
                # 메모리 모니터링
                if MemoryMonitor.get_available_memory() < 5:
                    logger.warning("메모리 부족으로 샘플링 중단")
                    break
                
                # 청크별 메모리 정리
                del chunk
                if processed_chunks % 3 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.warning(f"샘플링 청크 처리 실패: {e}")
                current_pos += self.chunk_size
                continue
        
        if sample_chunks:
            try:
                sample_df = pd.concat(sample_chunks, ignore_index=True)
                
                # 최종 샘플 크기 조정
                if len(sample_df) > sample_size:
                    sample_df = sample_df.sample(n=sample_size, random_state=random_state).copy()
                
                # 메모리 최적화
                sample_df = self._optimize_chunk_memory(sample_df)
                
                # 정리
                del sample_chunks
                gc.collect()
                
                return sample_df
                
            except Exception as e:
                logger.error(f"샘플 결합 실패: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

class DataLoader:
    """메모리 효율적 데이터 로딩 및 기본 전처리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.train_data = None
        self.test_data = None
        self.feature_columns = None
        self.target_column = 'clicked'
        
        # 메모리 설정 (더 보수적으로)
        self.memory_config = config.get_safe_memory_limits()
        self.data_config = config.get_data_config()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 데이터 로딩"""
        try:
            logger.info("기본 데이터 로딩 시작")
            
            # 메모리 상태 확인
            available_memory = self.memory_monitor.get_available_memory()
            if available_memory < 10:
                logger.warning(f"메모리 부족 상태: {available_memory:.2f}GB")
                MemoryMonitor.force_memory_cleanup()
            
            # 엔진 자동 선택
            engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
            
            # 청킹 방식으로 로딩
            max_train_size = min(500000, self.data_config['max_train_size'])
            max_test_size = min(100000, self.data_config['max_test_size'])
            
            with ChunkedParquetReader(self.config.TRAIN_PATH, 100000) as train_reader:
                self.train_data = train_reader.get_sample(max_train_size)
                logger.info(f"학습 데이터 형태: {self.train_data.shape}")
            
            # 메모리 정리
            MemoryMonitor.force_memory_cleanup()
            
            with ChunkedParquetReader(self.config.TEST_PATH, 100000) as test_reader:
                self.test_data = test_reader.get_sample(max_test_size)
                logger.info(f"테스트 데이터 형태: {self.test_data.shape}")
            
            # 메모리 정리
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
        """대용량 데이터 최적화 로딩"""
        logger.info("대용량 데이터 최적화 로딩 시작")
        
        # 초기 메모리 상태 확인
        memory_info = self.memory_monitor.get_memory_info()
        logger.info(f"초기 메모리: 사용가능 {memory_info['available_gb']:.1f}GB / 전체 {memory_info['total_gb']:.1f}GB")
        
        # 동적 크기 계산 (매우 보수적으로)
        available_memory = memory_info['available_gb']
        
        if available_memory > 40:
            max_train_size = 800000  # 더 감소
            max_test_size = 150000
            chunk_size = 150000  # 청크 크기 감소
            logger.info("대용량 메모리 모드")
        elif available_memory > 25:
            max_train_size = 500000  # 더 감소
            max_test_size = 100000
            chunk_size = 100000
            logger.info("중간 메모리 모드")
        else:
            max_train_size = 300000
            max_test_size = 60000
            chunk_size = 60000
            logger.info("제한 메모리 모드")
        
        try:
            # 학습 데이터 로딩
            train_df = self._load_train_data_chunked(max_train_size, chunk_size)
            
            # 메모리 상태 확인 및 정리
            MemoryMonitor.force_memory_cleanup()
            memory_after_train = self.memory_monitor.get_memory_info()
            logger.info(f"학습 데이터 후 메모리: {memory_after_train['available_gb']:.1f}GB")
            
            # 메모리 부족시 크기 조정
            if memory_after_train['available_gb'] < 8:
                logger.warning("메모리 부족으로 데이터 크기 조정")
                if len(train_df) > 300000:
                    train_df = train_df.sample(n=300000, random_state=42).reset_index(drop=True)
                max_test_size = min(50000, max_test_size)
                chunk_size = min(50000, chunk_size)
            
            # 테스트 데이터 로딩
            test_df = self._load_test_data_chunked(max_test_size, chunk_size)
            
            # 메모리 상태 확인
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
            
            logger.info("대용량 데이터 최적화 로딩 완료")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"대용량 데이터 로딩 실패: {str(e)}")
            MemoryMonitor.force_memory_cleanup()
            raise
    
    def _load_train_data_chunked(self, max_size: int, chunk_size: int) -> pd.DataFrame:
        """청킹 기반 학습 데이터 로딩"""
        logger.info("청킹 기반 학습 데이터 로딩 시작")
        
        try:
            with ChunkedParquetReader(self.config.TRAIN_PATH, chunk_size) as reader:
                total_rows = reader.total_rows
                logger.info(f"전체 학습 데이터: {total_rows:,}행")
                
                if max_size >= total_rows:
                    logger.info("전체 학습 데이터 로딩 시도")
                    # 전체 데이터가 작으면 직접 로딩
                    try:
                        engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
                        df = pd.read_parquet(self.config.TRAIN_PATH, engine=engine)
                        df = self._optimize_memory_usage(df)
                    except Exception as e:
                        logger.warning(f"전체 로딩 실패: {e}. 샘플링 사용")
                        df = reader.get_sample(max_size)
                else:
                    # 샘플링 로딩
                    usage_ratio = max_size / total_rows
                    logger.info(f"학습 데이터 샘플링: {usage_ratio*100:.2f}% ({max_size:,}/{total_rows:,})")
                    df = reader.get_sample(max_size)
                
                # 메모리 최적화
                if not df.empty:
                    df = self._optimize_memory_usage(df)
                else:
                    logger.error("학습 데이터가 비어있습니다")
                    raise ValueError("학습 데이터 로딩 실패")
                
                logger.info(f"학습 데이터 로딩 완료: {df.shape}")
                return df
                
        except Exception as e:
            logger.error(f"학습 데이터 청킹 로딩 실패: {e}")
            # 대안: 축소된 기본 pandas 로딩
            try:
                logger.info("축소 모드로 기본 pandas 로딩 시도")
                engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
                df = pd.read_parquet(self.config.TRAIN_PATH, engine=engine)
                if len(df) > max_size:
                    df = df.sample(n=max_size, random_state=42).reset_index(drop=True)
                return self._optimize_memory_usage(df)
            except Exception as e2:
                logger.error(f"기본 로딩도 실패: {e2}")
                raise
    
    def _load_test_data_chunked(self, max_size: int, chunk_size: int) -> pd.DataFrame:
        """청킹 기반 테스트 데이터 로딩"""
        logger.info("청킹 기반 테스트 데이터 로딩 시작")
        
        try:
            with ChunkedParquetReader(self.config.TEST_PATH, chunk_size) as reader:
                total_rows = reader.total_rows
                logger.info(f"전체 테스트 데이터: {total_rows:,}행")
                
                if max_size >= total_rows:
                    logger.info("전체 테스트 데이터 로딩 시도")
                    try:
                        engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
                        df = pd.read_parquet(self.config.TEST_PATH, engine=engine)
                        df = self._optimize_memory_usage(df)
                    except Exception as e:
                        logger.warning(f"전체 로딩 실패: {e}. 순차 샘플링 사용")
                        df = reader.read_chunk(0, max_size)
                else:
                    # 순차적 샘플링 (시간순 보장)
                    logger.info(f"테스트 데이터 순차 샘플링: {max_size:,}/{total_rows:,}")
                    df = reader.read_chunk(0, max_size)
                
                # 메모리 최적화
                if not df.empty:
                    df = self._optimize_memory_usage(df)
                else:
                    logger.error("테스트 데이터가 비어있습니다")
                    raise ValueError("테스트 데이터 로딩 실패")
                
                logger.info(f"테스트 데이터 로딩 완료: {df.shape}")
                return df
                
        except Exception as e:
            logger.error(f"테스트 데이터 청킹 로딩 실패: {e}")
            # 대안: 축소된 기본 pandas 로딩
            try:
                logger.info("축소 모드로 기본 pandas 로딩 시도")
                engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
                df = pd.read_parquet(self.config.TEST_PATH, engine=engine)
                if len(df) > max_size:
                    df = df.head(max_size).copy()
                return self._optimize_memory_usage(df)
            except Exception as e2:
                logger.error(f"기본 로딩도 실패: {e2}")
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
            
            # 범주형 최적화 (메모리 절약을 위해 간소화)
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols[:5]:  # 최대 5개만 처리
                try:
                    num_unique = df[col].nunique()
                    total_count = len(df)
                    
                    if num_unique < total_count * 0.3:  # 30% 미만일 때만
                        df[col] = df[col].astype('category')
                except Exception as e:
                    logger.warning(f"{col} 범주형 최적화 실패: {e}")
                    continue
            
            optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            logger.info(f"메모리 최적화 완료: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({reduction:.1f}% 감소)")
            
        except Exception as e:
            logger.warning(f"메모리 최적화 실패: {str(e)}")
        
        # 메모리 정리
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
            
            # 타입별 결측치 처리 (간소화)
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
        
        # 무한값 처리 (주요 수치형 컬럼만)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns[:20]  # 최대 20개
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
        
        # 이상치 탐지 및 처리 (최대 10개 컬럼만)
        outlier_cols_processed = 0
        max_outlier_cols = 10
        
        for col in numeric_cols[:max_outlier_cols]:
            try:
                if df_processed[col].var() == 0:
                    continue
                
                q1 = df_processed[col].quantile(0.25)
                q3 = df_processed[col].quantile(0.75)
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    
                    outliers_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                    outlier_count = outliers_mask.sum()
                    
                    if outlier_count > 0 and outlier_count < len(df_processed) * 0.05:  # 5% 미만만
                        df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                        df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                        outlier_cols_processed += 1
            
            except Exception as e:
                logger.debug(f"{col} 이상치 처리 실패: {str(e)}")
                continue
        
        if outlier_cols_processed > 0:
            logger.info(f"이상치 처리 완료: {outlier_cols_processed}개 컬럼")
        
        # 최종 메모리 최적화
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
        
        # 수치형 변수 통계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = {
                'count': len(numeric_cols)
            }
        
        # 타겟 변수 분포
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
            # 입력 검증
            if len(X_train) != len(y_train):
                raise ValueError(f"X와 y의 길이가 다릅니다. X: {len(X_train)}, y: {len(y_train)}")
            
            # 시간적 순서를 고려한 분할 인덱스 계산
            split_idx = int(len(X_train) * (1 - test_size))
            
            logger.info(f"분할 인덱스: {split_idx}, 전체 크기: {len(X_train)}")
            
            # 메모리 효율적 분할
            memory_before = self.memory_monitor.get_memory_usage()
            
            # 인덱스 기반 분할
            X_train_split = X_train.iloc[:split_idx].copy()
            X_val_split = X_train.iloc[split_idx:].copy()
            y_train_split = y_train.iloc[:split_idx].copy()
            y_val_split = y_train.iloc[split_idx:].copy()
            
            # 원본 데이터 메모리 정리
            del X_train, y_train
            MemoryMonitor.force_memory_cleanup()
            
            memory_after = self.memory_monitor.get_memory_usage()
            logger.info(f"메모리 사용량: {memory_before:.2f}GB → {memory_after:.2f}GB")
            
            # 결과 검증
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
        
        # 기본 형태 검증
        assert len(X_train) == len(y_train), f"학습 데이터 길이 불일치: {len(X_train)} != {len(y_train)}"
        assert len(X_val) == len(y_val), f"검증 데이터 길이 불일치: {len(X_val)} != {len(y_val)}"
        assert len(X_train.columns) == len(X_val.columns), f"피처 수 불일치: {len(X_train.columns)} != {len(X_val.columns)}"
        
        # 타겟 분포 확인
        train_dist = y_train.value_counts()
        val_dist = y_val.value_counts()
        
        logger.info(f"학습 데이터 타겟 분포: {train_dist.to_dict()}")
        logger.info(f"검증 데이터 타겟 분포: {val_dist.to_dict()}")
        
        # CTR 확인
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
        
        # 컬럼 일관성 검증
        train_cols = set(train_df.columns) - {'clicked'}
        test_cols = set(test_df.columns)
        
        validation_results['missing_in_test'] = list(train_cols - test_cols)
        validation_results['extra_in_test'] = list(test_cols - train_cols)
        
        # 데이터 타입 일관성 검증 (간소화)
        common_cols = train_cols & test_cols
        dtype_mismatches = []
        
        for col in list(common_cols)[:20]:  # 최대 20개만 검사
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