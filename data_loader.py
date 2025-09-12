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
import psutil
import pyarrow.parquet as pq

from config import Config

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """현재 프로세스 메모리 사용량 (GB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    @staticmethod
    def get_available_memory() -> float:
        """사용 가능한 메모리 (GB)"""
        return psutil.virtual_memory().available / (1024**3)
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """메모리 상태 정보"""
        vm = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total_gb': vm.total / (1024**3),
            'available_gb': vm.available / (1024**3),
            'used_gb': vm.used / (1024**3),
            'process_gb': process.memory_info().rss / (1024**3),
            'usage_percent': vm.percent
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
        self.parquet_file = pq.ParquetFile(self.file_path)
        self.total_rows = self.parquet_file.metadata.num_rows
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
        
        # 행 그룹 기반으로 읽기
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
        
        if not row_groups:
            return pd.DataFrame()
        
        table = self.parquet_file.read_row_groups(row_groups)
        df = table.to_pandas()
        
        # 정확한 범위로 슬라이싱
        relative_start = max(0, start_row - current_row + len(df))
        relative_end = min(len(df), relative_start + num_rows)
        
        if relative_start < len(df):
            df = df.iloc[relative_start:relative_end].copy()
        else:
            df = pd.DataFrame()
        
        self.current_position = start_row + len(df)
        return df
    
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
            return self.parquet_file.read().to_pandas()
        
        # 균등 샘플링
        np.random.seed(random_state)
        step = self.total_rows // sample_size
        
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
            
            self.train_data = pd.read_parquet(self.config.TRAIN_PATH)
            logger.info(f"학습 데이터 형태: {self.train_data.shape}")
            
            self.test_data = pd.read_parquet(self.config.TEST_PATH)
            logger.info(f"테스트 데이터 형태: {self.test_data.shape}")
            
            self.feature_columns = [col for col in self.train_data.columns 
                                  if col != self.target_column]
            logger.info(f"피처 컬럼 수: {len(self.feature_columns)}")
            
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
        
        # 동적 크기 계산
        available_memory = memory_info['available_gb']
        
        if available_memory > 50:
            # 64GB 환경 최적화
            max_train_size = 1500000
            max_test_size = 300000
            logger.info("대용량 메모리 모드 활성화")
        elif available_memory > 35:
            max_train_size = 1000000
            max_test_size = 250000
            logger.info("중간 메모리 모드 활성화")
        else:
            max_train_size = 500000
            max_test_size = 150000
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
        
        with ChunkedParquetReader(self.config.TRAIN_PATH, self.data_config['chunk_size']) as reader:
            total_rows = reader.total_rows
            logger.info(f"전체 학습 데이터: {total_rows:,}행")
            
            if max_size >= total_rows:
                # 전체 데이터 로딩
                logger.info("전체 학습 데이터 로딩")
                df = reader.parquet_file.read().to_pandas()
            else:
                # 샘플링 로딩
                usage_ratio = max_size / total_rows
                logger.info(f"학습 데이터 샘플링: {usage_ratio*100:.2f}% ({max_size:,}/{total_rows:,})")
                df = reader.get_sample(max_size)
            
            # 메모리 최적화
            df = self._optimize_memory_usage(df)
            
            logger.info(f"학습 데이터 로딩 완료: {df.shape}")
            return df
    
    def _load_test_data_chunked(self, max_size: int) -> pd.DataFrame:
        """청킹 기반 테스트 데이터 로딩"""
        logger.info("청킹 기반 테스트 데이터 로딩 시작")
        
        with ChunkedParquetReader(self.config.TEST_PATH, self.data_config['chunk_size']) as reader:
            total_rows = reader.total_rows
            logger.info(f"전체 테스트 데이터: {total_rows:,}행")
            
            if max_size >= total_rows:
                # 전체 데이터 로딩
                logger.info("전체 테스트 데이터 로딩")
                df = reader.parquet_file.read().to_pandas()
            else:
                # 순차적 샘플링 (시간순 보장)
                logger.info(f"테스트 데이터 순차 샘플링: {max_size:,}/{total_rows:,}")
                df = reader.read_chunk(0, max_size)
            
            # 메모리 최적화
            df = self._optimize_memory_usage(df)
            
            logger.info(f"테스트 데이터 로딩 완료: {df.shape}")
            return df
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """메모리 사용량 최적화"""
        logger.info("메모리 사용량 최적화 시작")
        
        original_memory = df.memory_usage(deep=True).sum() / (1024**2)
        
        try:
            # 정수형 최적화
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                col_min, col_max = df[col].min(), df[col].max()
                
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
            
            # 실수형 최적화
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # 범주형 최적화
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                num_unique = df[col].nunique()
                total_count = len(df)
                
                if num_unique < total_count * 0.5:
                    df[col] = df[col].astype('category')
            
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
        if missing_info.sum() > 0:
            logger.warning(f"결측치 발견: {missing_info.sum()}개")
            
            # 수치형 변수: 중앙값으로 대치
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
            
            # 범주형 변수: 최빈값으로 대치
            categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df_processed[col].isnull().sum() > 0:
                    mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
        
        # 최종 메모리 최적화
        df_processed = self._optimize_memory_usage(df_processed)
        
        logger.info("기본 전처리 완료")
        return df_processed
    
    def parallel_data_processing(self, df: pd.DataFrame, 
                                target_func, n_jobs: int = None) -> pd.DataFrame:
        """병렬 데이터 처리"""
        if n_jobs is None:
            n_jobs = min(6, os.cpu_count())  # 6코어 CPU 최대 활용
        
        logger.info(f"병렬 데이터 처리 시작 ({n_jobs}개 스레드)")
        
        # 데이터를 청크로 분할
        chunk_size = len(df) // n_jobs
        chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
        
        processed_chunks = []
        
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
            
            # 인덱스 기반 분할 (복사 최소화)
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
            train_df.to_parquet(train_path, index=False, compression='snappy')
            test_df.to_parquet(test_path, index=False, compression='snappy')
            
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
        
        if available_gb > 45:
            # 대용량 메모리 모드
            logger.info("대용량 메모리 전략 사용")
            return self.load_large_data_optimized()
        elif available_gb > 25:
            # 중간 메모리 모드  
            logger.info("중간 메모리 전략 사용")
            return self._load_medium_data()
        else:
            # 제한 메모리 모드
            logger.info("제한 메모리 전략 사용")
            return self._load_small_data()
    
    def _load_medium_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """중간 크기 데이터 로딩"""
        max_train_size = 800000
        max_test_size = 200000
        
        with ChunkedParquetReader(self.config.TRAIN_PATH, 300000) as train_reader:
            train_df = train_reader.get_sample(max_train_size)
            train_df = self._optimize_memory_usage(train_df)
        
        with ChunkedParquetReader(self.config.TEST_PATH, 150000) as test_reader:
            test_df = test_reader.read_chunk(0, max_test_size)
            test_df = self._optimize_memory_usage(test_df)
        
        logger.info(f"중간 데이터 로딩 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
        return train_df, test_df
    
    def _load_small_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """소규모 데이터 로딩"""
        max_train_size = 400000
        max_test_size = 100000
        
        with ChunkedParquetReader(self.config.TRAIN_PATH, 200000) as train_reader:
            train_df = train_reader.get_sample(max_train_size)
            train_df = self._optimize_memory_usage(train_df)
        
        with ChunkedParquetReader(self.config.TEST_PATH, 100000) as test_reader:
            test_df = test_reader.read_chunk(0, max_test_size)
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
        
        for col in numeric_cols[:10]:  # 처음 10개만 검증
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