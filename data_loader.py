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
            self.total_rows = len(pd.read_parquet(self.file_path, engine='auto'))
        except:
            self.total_rows = 1000000  # 기본값
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parquet_file:
            self.parquet_file = None
    
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
                row_groups = []
                current_row = 0
                
                for i in range(self.parquet_file.num_row_groups):
                    rg_rows = self.parquet_file.metadata.row_group(i).num_rows
                    
                    if current_row + rg_rows > start_row:
                        if current_row < start_row + num_rows:
                            row_groups.append(i)
                    
                    current_row += rg_rows
                    
                    if current_row >= start_row + num_rows:
                        break
                
                if row_groups:
                    table = self.parquet_file.read_row_groups(row_groups)
                    df = table.to_pandas()
                    
                    # 정확한 범위로 슬라이싱
                    relative_start = max(0, start_row - (current_row - len(df)))
                    relative_end = min(len(df), relative_start + num_rows)
                    
                    if relative_start < len(df):
                        df = df.iloc[relative_start:relative_end].copy()
                    else:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
            else:
                # pandas 기본 방식
                df = pd.read_parquet(self.file_path, engine='auto')
                df = df.iloc[start_row:start_row + num_rows].copy()
            
            self.current_position = start_row + len(df)
            return df
            
        except Exception as e:
            logger.error(f"청크 읽기 실패: {e}")
            return pd.DataFrame()
    
    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """청크 단위로 순회"""
        self.current_position = 0
        
        while self.current_position < self.total_rows:
            chunk = self.read_chunk()
            if chunk.empty:
                break
            yield chunk
    
    def get_sample(self, sample_size: int, random_state: int = 42) -> pd.DataFrame:
        """메모리 효율적인 샘플링"""
        if sample_size >= self.total_rows:
            # 전체 데이터 반환
            try:
                if PYARROW_AVAILABLE and self.parquet_file:
                    return self.parquet_file.read().to_pandas()
                else:
                    return pd.read_parquet(self.file_path, engine='auto')
            except Exception as e:
                logger.error(f"전체 데이터 로딩 실패: {e}")
                return pd.DataFrame()
        
        # 균등 샘플링
        np.random.seed(random_state)
        step = max(1, self.total_rows // sample_size)
        
        sample_chunks = []
        current_pos = 0
        
        while current_pos < self.total_rows and len(sample_chunks) * self.chunk_size < sample_size:
            chunk_size = min(self.chunk_size, sample_size - len(sample_chunks) * self.chunk_size)
            chunk = self.read_chunk(current_pos, chunk_size)
            
            if not chunk.empty:
                sample_chunks.append(chunk)
            
            current_pos += step
        
        if sample_chunks:
            sample_df = pd.concat(sample_chunks, ignore_index=True)
            return sample_df.iloc[:sample_size].copy()
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
        
        # 메모리 설정
        self.memory_config = config.get_memory_config()
        self.data_config = config.get_data_config()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 데이터 로딩"""
        try:
            logger.info("기본 데이터 로딩 시작")
            
            # 엔진 자동 선택
            engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
            
            self.train_data = pd.read_parquet(self.config.TRAIN_PATH, engine=engine)
            logger.info(f"학습 데이터 형태: {self.train_data.shape}")
            
            self.test_data = pd.read_parquet(self.config.TEST_PATH, engine=engine)
            logger.info(f"테스트 데이터 형태: {self.test_data.shape}")
            
            self.feature_columns = [col for col in self.train_data.columns 
                                  if col != self.target_column]
            logger.info(f"피처 컬럼 수: {len(self.feature_columns)}")
            
            if self.target_column in self.train_data.columns:
                actual_ctr = self.train_data[self.target_column].mean()
                logger.info(f"실제 CTR: {actual_ctr:.4f}")
            
            return self.train_data, self.test_data
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            raise
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """대용량 데이터 최적화 로딩"""
        logger.info("대용량 데이터 최적화 로딩 시작")
        
        # 초기 메모리 상태 확인
        memory_info = self.memory_monitor.get_memory_info()
        logger.info(f"초기 메모리: 사용가능 {memory_info['available_gb']:.1f}GB / 전체 {memory_info['total_gb']:.1f}GB")
        
        # 동적 크기 계산 (더 보수적으로)
        available_memory = memory_info['available_gb']
        
        if available_memory > 40:
            # 64GB 환경에서도 보수적으로
            max_train_size = 1200000
            max_test_size = 250000
            logger.info("대용량 메모리 모드 활성화")
        elif available_memory > 25:
            max_train_size = 800000
            max_test_size = 200000
            logger.info("중간 메모리 모드 활성화")
        else:
            max_train_size = 400000
            max_test_size = 100000
            logger.info("제한 메모리 모드 활성화")
        
        try:
            # 학습 데이터 로딩
            train_df = self._load_train_data_chunked(max_train_size)
            
            # 메모리 상태 확인
            memory_after_train = self.memory_monitor.get_memory_info()
            logger.info(f"학습 데이터 후 메모리: {memory_after_train['available_gb']:.1f}GB")
            
            # 테스트 데이터 로딩
            test_df = self._load_test_data_chunked(max_test_size)
            
            # 메모리 상태 확인
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
            gc.collect()
            raise
    
    def _load_train_data_chunked(self, max_size: int) -> pd.DataFrame:
        """청킹 기반 학습 데이터 로딩"""
        logger.info("청킹 기반 학습 데이터 로딩 시작")
        
        try:
            with ChunkedParquetReader(self.config.TRAIN_PATH, self.data_config['chunk_size']) as reader:
                total_rows = reader.total_rows
                logger.info(f"전체 학습 데이터: {total_rows:,}행")
                
                if max_size >= total_rows:
                    # 전체 데이터 로딩
                    logger.info("전체 학습 데이터 로딩")
                    engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
                    df = pd.read_parquet(self.config.TRAIN_PATH, engine=engine)
                else:
                    # 샘플링 로딩
                    usage_ratio = max_size / total_rows
                    logger.info(f"학습 데이터 샘플링: {usage_ratio*100:.2f}% ({max_size:,}/{total_rows:,})")
                    df = reader.get_sample(max_size)
                
                # 메모리 최적화
                df = self._optimize_memory_usage(df)
                
                logger.info(f"학습 데이터 로딩 완료: {df.shape}")
                return df
        except Exception as e:
            logger.error(f"학습 데이터 청킹 로딩 실패: {e}")
            # 대안: 기본 pandas 로딩
            try:
                logger.info("기본 pandas 로딩 시도")
                engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
                df = pd.read_parquet(self.config.TRAIN_PATH, engine=engine)
                if len(df) > max_size:
                    df = df.sample(n=max_size, random_state=42).reset_index(drop=True)
                return self._optimize_memory_usage(df)
            except Exception as e2:
                logger.error(f"기본 로딩도 실패: {e2}")
                raise
    
    def _load_test_data_chunked(self, max_size: int) -> pd.DataFrame:
        """청킹 기반 테스트 데이터 로딩"""
        logger.info("청킹 기반 테스트 데이터 로딩 시작")
        
        try:
            with ChunkedParquetReader(self.config.TEST_PATH, self.data_config['chunk_size']) as reader:
                total_rows = reader.total_rows
                logger.info(f"전체 테스트 데이터: {total_rows:,}행")
                
                if max_size >= total_rows:
                    # 전체 데이터 로딩
                    logger.info("전체 테스트 데이터 로딩")
                    engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
                    df = pd.read_parquet(self.config.TEST_PATH, engine=engine)
                else:
                    # 순차적 샘플링 (시간순 보장)
                    logger.info(f"테스트 데이터 순차 샘플링: {max_size:,}/{total_rows:,}")
                    df = reader.read_chunk(0, max_size)
                
                # 메모리 최적화
                df = self._optimize_memory_usage(df)
                
                logger.info(f"테스트 데이터 로딩 완료: {df.shape}")
                return df
        except Exception as e:
            logger.error(f"테스트 데이터 청킹 로딩 실패: {e}")
            # 대안: 기본 pandas 로딩
            try:
                logger.info("기본 pandas 로딩 시도")
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
            
            # 범주형 최적화
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                try:
                    num_unique = df[col].nunique()
                    total_count = len(df)
                    
                    if num_unique < total_count * 0.5:
                        df[col] = df[col].astype('category')
                except Exception as e:
                    logger.warning(f"{col} 범주형 최적화 실패: {e}")
                    continue
            
            optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            logger.info(f"메모리 최적화 완료: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({reduction:.1f}% 감소)")
            
        except Exception as e:
            logger.warning(f"메모리 최적화 실패: {str(e)}")
        
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
            
            # 컬럼별 결측치 정보
            missing_cols = missing_info[missing_info > 0].sort_values(ascending=False)
            if len(missing_cols) <= 10:
                logger.info("결측치가 많은 상위 컬럼:")
                for col, count in missing_cols.items():
                    ratio = count / len(df_processed) * 100
                    logger.info(f"  {col}: {count:,}개 ({ratio:.1f}%)")
            
            # 타입별 결측치 처리
            for col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    try:
                        col_dtype = str(df_processed[col].dtype)
                        
                        # 수치형 변수 처리
                        if col_dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
                            # 정수형은 0으로 대치
                            df_processed[col].fillna(0, inplace=True)
                        elif col_dtype in ['float16', 'float32', 'float64']:
                            # 실수형은 중앙값으로 대치
                            non_zero_median = df_processed[df_processed[col] != 0][col].median()
                            if pd.isna(non_zero_median):
                                fill_value = 0.0
                            else:
                                fill_value = non_zero_median
                            df_processed[col].fillna(fill_value, inplace=True)
                        # 범주형 변수 처리
                        elif col_dtype in ['object', 'category', 'string', 'bool']:
                            # 범주형은 최빈값으로 대치
                            if len(df_processed[col].mode()) > 0:
                                mode_val = df_processed[col].mode()[0]
                            else:
                                mode_val = 'unknown'
                            df_processed[col].fillna(mode_val, inplace=True)
                        else:
                            # 기타 타입은 0으로 대치
                            df_processed[col].fillna(0, inplace=True)
                    except Exception as e:
                        logger.warning(f"{col} 결측치 처리 실패: {e}")
                        df_processed[col].fillna(0, inplace=True)
        
        # 무한값 처리
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                inf_count = np.isinf(df_processed[col]).sum()
                if inf_count > 0:
                    logger.warning(f"{col}: 무한값 {inf_count:,}개 발견, 대체 처리")
                    
                    # 유한한 값들의 통계치로 대체
                    finite_values = df_processed[col][np.isfinite(df_processed[col])]
                    if len(finite_values) > 0:
                        if finite_values.std() == 0:
                            replacement = finite_values.mean()
                        else:
                            # 99.5퍼센타일과 0.5퍼센타일로 클리핑
                            upper_bound = finite_values.quantile(0.995)
                            lower_bound = finite_values.quantile(0.005)
                            
                            df_processed.loc[df_processed[col] == np.inf, col] = upper_bound
                            df_processed.loc[df_processed[col] == -np.inf, col] = lower_bound
                    else:
                        df_processed[col].replace([np.inf, -np.inf], 0, inplace=True)
            except Exception as e:
                logger.warning(f"{col} 무한값 처리 실패: {e}")
                df_processed[col].replace([np.inf, -np.inf], 0, inplace=True)
        
        # 이상치 탐지 및 처리 (수치형 변수에만 적용)
        outlier_cols_processed = 0
        max_outlier_cols = 20
        
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
                    
                    if outlier_count > 0 and outlier_count < len(df_processed) * 0.1:
                        # 클리핑 방식으로 이상치 대체
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
    
    def parallel_data_processing(self, df: pd.DataFrame, 
                                target_func, n_jobs: int = None) -> pd.DataFrame:
        """병렬 데이터 처리"""
        if n_jobs is None:
            n_jobs = min(4, os.cpu_count())  # 보수적으로 4개로 제한
        
        logger.info(f"병렬 데이터 처리 시작 ({n_jobs}개 스레드)")
        
        # 데이터를 청크로 분할
        chunk_size = len(df) // n_jobs
        chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
        
        processed_chunks = []
        
        try:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_chunk = {executor.submit(target_func, chunk): i for i, chunk in enumerate(chunks)}
                
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        result = future.result()
                        processed_chunks.append((chunk_idx, result))
                    except Exception as e:
                        logger.error(f"청크 {chunk_idx} 처리 실패: {str(e)}")
                        # 원본 청크 사용
                        processed_chunks.append((chunk_idx, chunks[chunk_idx]))
        except Exception as e:
            logger.error(f"병렬 처리 실패: {e}")
            return df
        
        # 순서대로 정렬 후 결합
        processed_chunks.sort(key=lambda x: x[0])
        result_df = pd.concat([chunk for _, chunk in processed_chunks], ignore_index=True)
        
        logger.info("병렬 데이터 처리 완료")
        return result_df
    
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
                'count': len(numeric_cols),
                'mean_values': df[numeric_cols].mean().describe().to_dict()
            }
        
        # 범주형 변수 통계
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical_summary'] = {
                'count': len(categorical_cols),
                'unique_counts': [df[col].nunique() for col in categorical_cols[:5]]
            }
        
        # 타겟 변수 분포
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts()
            summary['target_distribution'] = {
                'counts': target_dist.to_dict(),
                'ctr': df[self.target_column].mean()
            }
        
        return summary
    
    def create_time_series_folds(self, 
                                X: pd.DataFrame, 
                                y: pd.Series, 
                                n_splits: int = None) -> TimeSeriesSplit:
        """시간적 순서를 고려한 교차검증 폴드 생성"""
        if n_splits is None:
            n_splits = self.config.N_SPLITS
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"시간적 교차검증 {n_splits}개 폴드 설정 완료")
        return tscv
    
    def memory_efficient_train_test_split(self,
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """메모리 효율적인 시간적 순서 기반 데이터 분할"""
        logger.info("메모리 효율적인 데이터 분할 시작")
        
        try:
            # 입력 검증
            if len(X_train) != len(y_train):
                raise ValueError(f"X와 y의 길이가 다릅니다. X: {len(X_train)}, y: {len(y_train)}")
            
            if len(X_train) == 0:
                raise ValueError("분할할 데이터가 없습니다.")
            
            # 시간적 순서를 고려한 분할 인덱스 계산
            split_idx = int(len(X_train) * (1 - test_size))
            
            logger.info(f"분할 인덱스: {split_idx}, 전체 크기: {len(X_train)}")
            
            # 메모리 효율적 분할
            memory_before = self.memory_monitor.get_memory_usage()
            
            # 인덱스 기반 분할
            train_indices = np.arange(split_idx)
            val_indices = np.arange(split_idx, len(X_train))
            
            X_train_split = X_train.iloc[train_indices].copy()
            X_val_split = X_train.iloc[val_indices].copy()
            y_train_split = y_train.iloc[train_indices].copy()
            y_val_split = y_train.iloc[val_indices].copy()
            
            # 원본 데이터 메모리 정리
            del X_train, y_train
            gc.collect()
            
            memory_after = self.memory_monitor.get_memory_usage()
            logger.info(f"메모리 사용량: {memory_before:.2f}GB → {memory_after:.2f}GB")
            
            # 결과 검증
            self._validate_split_results(X_train_split, X_val_split, y_train_split, y_val_split)
            
            logger.info(f"메모리 효율적 분할 완료 - 학습: {X_train_split.shape}, 검증: {X_val_split.shape}")
            
            return X_train_split, X_val_split, y_train_split, y_val_split
            
        except Exception as e:
            logger.error(f"메모리 효율적 데이터 분할 실패: {str(e)}")
            gc.collect()
            raise
    
    def _validate_split_results(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                               y_train: pd.Series, y_val: pd.Series):
        """분할 결과 검증"""
        logger.info("분할 결과 검증 시작")
        
        # 기본 형태 검증
        assert len(X_train) == len(y_train), f"학습 데이터 길이 불일치: {len(X_train)} != {len(y_train)}"
        assert len(X_val) == len(y_val), f"검증 데이터 길이 불일치: {len(X_val)} != {len(y_val)}"
        assert len(X_train.columns) == len(X_val.columns), f"피처 수 불일치: {len(X_train.columns)} != {len(X_val.columns)}"
        
        # 컬럼 일치성 검증
        if list(X_train.columns) != list(X_val.columns):
            logger.error("학습/검증 데이터의 컬럼이 다릅니다!")
            raise ValueError("학습/검증 데이터의 컬럼 불일치")
        
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
    
    def save_processed_data(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          suffix: str = "processed"):
        """전처리된 데이터 저장"""
        try:
            train_path = self.config.DATA_DIR / f"train_{suffix}.parquet"
            test_path = self.config.DATA_DIR / f"test_{suffix}.parquet"
            
            # 메모리 효율적 저장
            engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
            train_df.to_parquet(train_path, index=False, compression='snappy', engine=engine)
            test_df.to_parquet(test_path, index=False, compression='snappy', engine=engine)
            
            logger.info(f"전처리 데이터 저장 완료: {train_path}, {test_path}")
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {str(e)}")
            raise
    
    def load_data_with_memory_management(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """메모리 관리를 고려한 데이터 로딩"""
        logger.info("메모리 관리 데이터 로딩 시작")
        
        # 초기 메모리 상태
        initial_memory = self.memory_monitor.get_memory_info()
        logger.info(f"초기 메모리 상태: {initial_memory}")
        
        # 사용 가능 메모리 기반 전략 결정
        available_gb = initial_memory['available_gb']
        
        if available_gb > 35:
            # 대용량 메모리 모드
            logger.info("대용량 메모리 전략 사용")
            return self.load_large_data_optimized()
        elif available_gb > 20:
            # 중간 메모리 모드  
            logger.info("중간 메모리 전략 사용")
            return self._load_medium_data()
        else:
            # 제한 메모리 모드
            logger.info("제한 메모리 전략 사용")
            return self._load_small_data()
    
    def _load_medium_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """중간 크기 데이터 로딩"""
        max_train_size = 600000
        max_test_size = 150000
        
        try:
            with ChunkedParquetReader(self.config.TRAIN_PATH, 200000) as train_reader:
                train_df = train_reader.get_sample(max_train_size)
                train_df = self._optimize_memory_usage(train_df)
        except Exception as e:
            logger.error(f"중간 크기 학습 데이터 로딩 실패: {e}")
            engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
            train_df = pd.read_parquet(self.config.TRAIN_PATH, engine=engine)
            train_df = train_df.sample(n=max_train_size, random_state=42).reset_index(drop=True)
            train_df = self._optimize_memory_usage(train_df)
        
        try:
            with ChunkedParquetReader(self.config.TEST_PATH, 100000) as test_reader:
                test_df = test_reader.read_chunk(0, max_test_size)
                test_df = self._optimize_memory_usage(test_df)
        except Exception as e:
            logger.error(f"중간 크기 테스트 데이터 로딩 실패: {e}")
            engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
            test_df = pd.read_parquet(self.config.TEST_PATH, engine=engine)
            test_df = test_df.head(max_test_size).copy()
            test_df = self._optimize_memory_usage(test_df)
        
        logger.info(f"중간 데이터 로딩 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
        return train_df, test_df
    
    def _load_small_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """소규모 데이터 로딩"""
        max_train_size = 300000
        max_test_size = 80000
        
        try:
            with ChunkedParquetReader(self.config.TRAIN_PATH, 150000) as train_reader:
                train_df = train_reader.get_sample(max_train_size)
                train_df = self._optimize_memory_usage(train_df)
        except Exception as e:
            logger.error(f"소규모 학습 데이터 로딩 실패: {e}")
            engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
            train_df = pd.read_parquet(self.config.TRAIN_PATH, engine=engine)
            train_df = train_df.sample(n=max_train_size, random_state=42).reset_index(drop=True)
            train_df = self._optimize_memory_usage(train_df)
        
        try:
            with ChunkedParquetReader(self.config.TEST_PATH, 80000) as test_reader:
                test_df = test_reader.read_chunk(0, max_test_size)
                test_df = self._optimize_memory_usage(test_df)
        except Exception as e:
            logger.error(f"소규모 테스트 데이터 로딩 실패: {e}")
            engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
            test_df = pd.read_parquet(self.config.TEST_PATH, engine=engine)
            test_df = test_df.head(max_test_size).copy()
            test_df = self._optimize_memory_usage(test_df)
        
        logger.info(f"소규모 데이터 로딩 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
        return train_df, test_df

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
        
        # 데이터 타입 일관성 검증
        common_cols = train_cols & test_cols
        dtype_mismatches = []
        
        for col in common_cols:
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
        
        validation_results['dtype_mismatches'] = dtype_mismatches
        
        # 범위 일관성 검증
        range_issues = []
        numeric_cols = [col for col in common_cols if train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        for col in numeric_cols[:10]:
            try:
                train_min, train_max = train_df[col].min(), train_df[col].max()
                test_min, test_max = test_df[col].min(), test_df[col].max()
                
                if test_min < train_min * 0.5 or test_max > train_max * 1.5:
                    range_issues.append({
                        'column': col,
                        'train_range': [float(train_min), float(train_max)],
                        'test_range': [float(test_min), float(test_max)]
                    })
            except:
                continue
        
        validation_results['range_issues'] = range_issues
        
        return validation_results
    
    @staticmethod
    def check_memory_efficiency(df: pd.DataFrame) -> Dict[str, Any]:
        """메모리 효율성 검증"""
        memory_info = {}
        
        # 전체 메모리 사용량
        total_memory = df.memory_usage(deep=True).sum() / (1024**2)
        memory_info['total_mb'] = total_memory
        
        # 컬럼별 메모리 사용량
        col_memory = df.memory_usage(deep=True) / (1024**2)
        memory_info['top_memory_columns'] = col_memory.nlargest(5).to_dict()
        
        # 최적화 가능한 컬럼
        optimization_candidates = []
        
        # float64 → float32 가능
        float64_cols = df.select_dtypes(include=['float64']).columns
        if len(float64_cols) > 0:
            optimization_candidates.append(f"float64→float32: {len(float64_cols)}개 컬럼")
        
        # int64 → 더 작은 타입 가능
        int64_cols = df.select_dtypes(include=['int64']).columns
        if len(int64_cols) > 0:
            optimization_candidates.append(f"int64→작은타입: {len(int64_cols)}개 컬럼")
        
        memory_info['optimization_candidates'] = optimization_candidates
        
        return memory_info