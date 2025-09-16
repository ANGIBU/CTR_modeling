# data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, Iterator, List
import logging
import gc
import mmap
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
import time
import threading

# PyArrow import 안전 처리
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pyarrow.compute as pc
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

class AdvancedMemoryMonitor:
    """대용량 데이터 처리용 고급 메모리 모니터링 클래스"""
    
    def __init__(self):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.memory_history = []
        self.peak_memory = 0.0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def get_detailed_memory_info(self) -> Dict[str, float]:
        """상세 메모리 정보 반환"""
        if not self.monitoring_enabled:
            return self._get_fallback_memory_info()
        
        try:
            vm = psutil.virtual_memory()
            process = psutil.Process()
            memory_info = process.memory_info()
            
            info = {
                'total_gb': vm.total / (1024**3),
                'available_gb': vm.available / (1024**3),
                'used_gb': vm.used / (1024**3),
                'process_rss_gb': memory_info.rss / (1024**3),
                'process_vms_gb': memory_info.vms / (1024**3),
                'usage_percent': vm.percent,
                'available_percent': (vm.available / vm.total) * 100,
                'swap_total_gb': psutil.swap_memory().total / (1024**3),
                'swap_used_gb': psutil.swap_memory().used / (1024**3),
                'swap_percent': psutil.swap_memory().percent
            }
            
            # 피크 메모리 추적
            current_process_memory = info['process_rss_gb']
            if current_process_memory > self.peak_memory:
                self.peak_memory = current_process_memory
            
            info['peak_memory_gb'] = self.peak_memory
            
            return info
            
        except Exception as e:
            logger.warning(f"메모리 정보 조회 실패: {e}")
            return self._get_fallback_memory_info()
    
    def _get_fallback_memory_info(self) -> Dict[str, float]:
        """기본 메모리 정보"""
        return {
            'total_gb': 64.0,
            'available_gb': 48.0,
            'used_gb': 16.0,
            'process_rss_gb': 2.0,
            'process_vms_gb': 4.0,
            'usage_percent': 25.0,
            'available_percent': 75.0,
            'swap_total_gb': 8.0,
            'swap_used_gb': 0.5,
            'swap_percent': 6.25,
            'peak_memory_gb': 2.0
        }
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인"""
        info = self.get_detailed_memory_info()
        
        return {
            'memory_pressure': info['usage_percent'] > 85,
            'low_available_memory': info['available_gb'] < 8,
            'high_swap_usage': info['swap_percent'] > 20,
            'process_memory_high': info['process_rss_gb'] > 40,
            'available_gb': info['available_gb'],
            'usage_percent': info['usage_percent'],
            'recommendation': self._get_memory_recommendation(info)
        }
    
    def _get_memory_recommendation(self, info: Dict[str, float]) -> str:
        """메모리 상태에 따른 권장 사항"""
        if info['available_gb'] < 5:
            return "심각한 메모리 부족 - 즉시 가비지 컬렉션 필요"
        elif info['available_gb'] < 10:
            return "메모리 부족 - 청킹 크기 축소 필요"
        elif info['usage_percent'] > 90:
            return "높은 메모리 사용률 - 처리 속도 조절 필요"
        elif info['swap_percent'] > 10:
            return "스왑 사용 증가 - 메모리 최적화 필요"
        else:
            return "메모리 상태 양호"
    
    def log_memory_status(self, context: str = ""):
        """메모리 상태 로깅"""
        info = self.get_detailed_memory_info()
        pressure = self.check_memory_pressure()
        
        print(f"메모리 상태 {context}: "
              f"사용가능 {info['available_gb']:.1f}GB, "
              f"사용률 {info['usage_percent']:.1f}%, "
              f"프로세스 {info['process_rss_gb']:.1f}GB")
        
        if pressure['memory_pressure']:
            print(f"메모리 압박: {pressure['recommendation']}")
    
    def force_memory_cleanup(self, aggressive: bool = False):
        """강제 메모리 정리"""
        try:
            initial_info = self.get_detailed_memory_info()
            
            gc.collect()
            
            if aggressive:
                for _ in range(3):
                    gc.collect()
                    time.sleep(0.1)
                
                try:
                    import ctypes
                    if hasattr(ctypes, 'windll'):
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                except:
                    pass
            
            final_info = self.get_detailed_memory_info()
            memory_freed = initial_info['process_rss_gb'] - final_info['process_rss_gb']
            
            if memory_freed > 0.1:
                print(f"메모리 정리 완료: {memory_freed:.2f}GB 해제")
            
        except Exception as e:
            print(f"메모리 정리 실패: {e}")

class DataFileValidator:
    """대용량 데이터 파일 검증 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = AdvancedMemoryMonitor()
    
    def validate_data_files(self) -> Dict[str, Any]:
        """데이터 파일 존재 및 크기 검증"""
        logger.info("대용량 데이터 파일 검증 시작")
        
        validation_results = {
            'train_file_valid': False,
            'test_file_valid': False,
            'train_file_info': {},
            'test_file_info': {},
            'validation_passed': False,
            'error_messages': []
        }
        
        # 학습 데이터 파일 검증
        train_validation = self._validate_single_file(
            self.config.TRAIN_PATH, 
            "학습 데이터",
            expected_min_size_mb=500,  # 최소 500MB
            expected_min_rows=self.config.MIN_TRAIN_SIZE
        )
        validation_results['train_file_valid'] = train_validation['valid']
        validation_results['train_file_info'] = train_validation
        
        # 테스트 데이터 파일 검증
        test_validation = self._validate_single_file(
            self.config.TEST_PATH,
            "테스트 데이터", 
            expected_min_size_mb=100,  # 최소 100MB
            expected_min_rows=self.config.MIN_TEST_SIZE
        )
        validation_results['test_file_valid'] = test_validation['valid']
        validation_results['test_file_info'] = test_validation
        
        # 전체 검증 결과
        validation_results['validation_passed'] = (
            validation_results['train_file_valid'] and 
            validation_results['test_file_valid']
        )
        
        if not validation_results['validation_passed']:
            validation_results['error_messages'].extend([
                msg for result in [train_validation, test_validation] 
                for msg in result.get('error_messages', [])
            ])
        
        self._log_validation_results(validation_results)
        
        return validation_results
    
    def _validate_single_file(self, file_path: Path, file_type: str, 
                             expected_min_size_mb: float, 
                             expected_min_rows: int) -> Dict[str, Any]:
        """단일 파일 상세 검증"""
        result = {
            'valid': False,
            'exists': False,
            'file_size_mb': 0.0,
            'estimated_rows': 0,
            'file_type': file_type,
            'schema_info': {},
            'error_messages': []
        }
        
        try:
            if not file_path.exists():
                result['error_messages'].append(f"{file_type} 파일이 존재하지 않음: {file_path}")
                return result
            
            result['exists'] = True
            result['file_size_mb'] = file_path.stat().st_size / (1024**2)
            
            # 파일 크기 검증
            if result['file_size_mb'] < expected_min_size_mb:
                result['error_messages'].append(
                    f"{file_type} 파일 크기 부족: {result['file_size_mb']:.1f}MB < {expected_min_size_mb}MB"
                )
                return result
            
            # Parquet 파일 스키마 및 행 수 검증
            if file_path.suffix.lower() == '.parquet':
                parquet_info = self._validate_parquet_file(file_path, expected_min_rows)
                result.update(parquet_info)
                
                if not parquet_info.get('row_count_valid', False):
                    result['error_messages'].extend(parquet_info.get('error_messages', []))
                    return result
            
            result['valid'] = True
            logger.info(f"{file_type} 파일 검증 성공: {file_path} "
                       f"({result['file_size_mb']:.1f}MB, {result['estimated_rows']:,}행)")
            
        except Exception as e:
            error_msg = f"{file_type} 파일 검증 실패: {str(e)}"
            result['error_messages'].append(error_msg)
            logger.error(error_msg)
        
        return result
    
    def _validate_parquet_file(self, file_path: Path, expected_min_rows: int) -> Dict[str, Any]:
        """Parquet 파일 상세 검증"""
        result = {
            'estimated_rows': 0,
            'column_count': 0,
            'schema_info': {},
            'row_count_valid': False,
            'error_messages': []
        }
        
        try:
            if PYARROW_AVAILABLE:
                # PyArrow로 메타데이터 읽기
                parquet_file = pq.ParquetFile(file_path)
                metadata = parquet_file.metadata
                
                result['estimated_rows'] = metadata.num_rows
                result['column_count'] = metadata.num_columns
                result['schema_info'] = {
                    'column_names': [field.name for field in parquet_file.schema],
                    'num_row_groups': metadata.num_row_groups
                }
                
            else:
                # pandas로 샘플 읽기 후 추정
                sample_df = pd.read_parquet(file_path, nrows=10000)
                file_size = file_path.stat().st_size
                sample_memory = sample_df.memory_usage(deep=True).sum()
                
                estimated_rows = int((file_size / sample_memory) * 10000 * 0.8)
                result['estimated_rows'] = estimated_rows
                result['column_count'] = len(sample_df.columns)
                result['schema_info'] = {
                    'column_names': sample_df.columns.tolist(),
                    'dtypes': sample_df.dtypes.to_dict()
                }
                
                del sample_df
                gc.collect()
            
            # 행 수 검증
            if result['estimated_rows'] >= expected_min_rows:
                result['row_count_valid'] = True
            else:
                result['error_messages'].append(
                    f"데이터 행 수 부족: {result['estimated_rows']:,} < {expected_min_rows:,}"
                )
            
        except Exception as e:
            result['error_messages'].append(f"Parquet 파일 분석 실패: {str(e)}")
        
        return result
    
    def _log_validation_results(self, results: Dict[str, Any]):
        """검증 결과 로깅"""
        logger.info("=== 대용량 데이터 파일 검증 결과 ===")
        
        if results['validation_passed']:
            logger.info("✓ 모든 데이터 파일 검증 통과")
            logger.info(f"학습 데이터: {results['train_file_info']['estimated_rows']:,}행, "
                       f"{results['train_file_info']['file_size_mb']:.1f}MB")
            logger.info(f"테스트 데이터: {results['test_file_info']['estimated_rows']:,}행, "
                       f"{results['test_file_info']['file_size_mb']:.1f}MB")
        else:
            logger.error("✗ 데이터 파일 검증 실패")
            for error in results['error_messages']:
                logger.error(f"  - {error}")
        
        logger.info("=== 검증 완료 ===")

class OptimizedParquetReader:
    """대용량 데이터 최적화 Parquet 리더"""
    
    def __init__(self, file_path: str, chunk_size: int = 1000000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.parquet_file = None
        self.total_rows = 0
        self.current_position = 0
        self.memory_monitor = AdvancedMemoryMonitor()
        self.read_statistics = {
            'total_chunks_read': 0,
            'total_bytes_read': 0,
            'total_time': 0.0,
            'average_chunk_time': 0.0
        }
        
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.memory_monitor.log_memory_status("파일 열기 전")
        
        if PYARROW_AVAILABLE:
            try:
                self.parquet_file = pq.ParquetFile(self.file_path)
                self.total_rows = self.parquet_file.metadata.num_rows
                logger.info(f"PyArrow로 파일 열기 성공: {self.total_rows:,}행")
                return self
            except Exception as e:
                logger.warning(f"PyArrow로 파일 열기 실패: {e}. pandas 사용")
        
        # PyArrow 실패 시 pandas로 행 수 추정
        try:
            sample_df = pd.read_parquet(self.file_path, nrows=50000)
            file_size = os.path.getsize(self.file_path)
            sample_size = sample_df.memory_usage(deep=True).sum()
            
            # 더 정확한 추정을 위한 압축률 고려
            compression_ratio = 0.7  # Parquet 압축률 추정
            estimated_rows = int((file_size / sample_size) * 50000 * compression_ratio)
            
            self.total_rows = estimated_rows
            logger.info(f"pandas로 행 수 추정: {self.total_rows:,}행")
            
            del sample_df
            gc.collect()
            
        except Exception as e:
            logger.warning(f"행 수 추정 실패: {e}. 기본값 사용")
            self.total_rows = 10000000
        
        self.memory_monitor.log_memory_status("파일 열기 후")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        if self.parquet_file:
            self.parquet_file = None
        
        self.memory_monitor.force_memory_cleanup(aggressive=True)
        self._log_read_statistics()
    
    def get_total_rows(self) -> int:
        """전체 행 수 반환"""
        return self.total_rows
    
    def read_all_data_parallel(self, max_workers: int = 3) -> pd.DataFrame:
        """병렬 처리로 전체 데이터 로딩"""
        logger.info(f"병렬 데이터 로딩 시작: {self.total_rows:,}행, {max_workers}개 워커")
        start_time = time.time()
        
        try:
            if PYARROW_AVAILABLE and self.parquet_file:
                # PyArrow 병렬 로딩
                df = self._read_with_pyarrow_parallel(max_workers)
            else:
                # pandas 병렬 로딩
                df = self._read_with_pandas_parallel(max_workers)
            
            df = self._optimize_dataframe_memory(df)
            
            read_time = time.time() - start_time
            self.read_statistics['total_time'] = read_time
            
            logger.info(f"병렬 데이터 로딩 완료: {df.shape}, {read_time:.2f}초")
            
            return df
            
        except Exception as e:
            logger.error(f"병렬 데이터 로딩 실패: {e}")
            # 폴백: 일반 로딩
            return self.read_all_data()
    
    def _read_with_pyarrow_parallel(self, max_workers: int) -> pd.DataFrame:
        """PyArrow 병렬 로딩"""
        try:
            # Row group 기반 병렬 처리
            metadata = self.parquet_file.metadata
            num_row_groups = metadata.num_row_groups
            
            if num_row_groups <= 1:
                # Row group이 적으면 일반 로딩
                table = self.parquet_file.read(use_threads=True)
                return table.to_pandas()
            
            # Row group별 병렬 읽기
            row_group_chunks = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_rg = {
                    executor.submit(self._read_row_group, rg_idx): rg_idx 
                    for rg_idx in range(num_row_groups)
                }
                
                for future in as_completed(future_to_rg):
                    try:
                        chunk_df = future.result()
                        if not chunk_df.empty:
                            row_group_chunks.append(chunk_df)
                    except Exception as e:
                        logger.warning(f"Row group 읽기 실패: {e}")
            
            if row_group_chunks:
                df = pd.concat(row_group_chunks, ignore_index=True)
                logger.info(f"PyArrow 병렬 로딩 성공: {len(row_group_chunks)}개 청크")
                return df
            else:
                raise ValueError("모든 row group 읽기 실패")
                
        except Exception as e:
            logger.warning(f"PyArrow 병렬 로딩 실패: {e}")
            # 기본 PyArrow 로딩
            table = self.parquet_file.read(use_threads=True)
            return table.to_pandas()
    
    def _read_row_group(self, row_group_idx: int) -> pd.DataFrame:
        """개별 row group 읽기"""
        try:
            table = self.parquet_file.read_row_group(row_group_idx, use_threads=True)
            df = table.to_pandas()
            return self._optimize_chunk_memory(df)
        except Exception as e:
            logger.warning(f"Row group {row_group_idx} 읽기 실패: {e}")
            return pd.DataFrame()
    
    def _read_with_pandas_parallel(self, max_workers: int) -> pd.DataFrame:
        """pandas 병렬 로딩"""
        # 청크 크기 조정
        optimal_chunk_size = min(self.chunk_size, self.total_rows // max_workers)
        chunks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {}
            
            for i in range(0, self.total_rows, optimal_chunk_size):
                end_idx = min(i + optimal_chunk_size, self.total_rows)
                future = executor.submit(self._read_chunk_pandas, i, end_idx - i)
                future_to_chunk[future] = i
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_df = future.result()
                    if not chunk_df.empty:
                        chunks.append(chunk_df)
                        
                        # 메모리 압박 시 중간 정리
                        if len(chunks) % 5 == 0:
                            pressure = self.memory_monitor.check_memory_pressure()
                            if pressure['memory_pressure']:
                                self.memory_monitor.force_memory_cleanup()
                                
                except Exception as e:
                    chunk_idx = future_to_chunk[future]
                    logger.warning(f"청크 {chunk_idx} 읽기 실패: {e}")
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"pandas 병렬 로딩 성공: {len(chunks)}개 청크")
            return df
        else:
            raise ValueError("모든 청크 읽기 실패")
    
    def _read_chunk_pandas(self, start_row: int, num_rows: int) -> pd.DataFrame:
        """pandas로 청크 읽기"""
        try:
            df = pd.read_parquet(
                self.file_path,
                engine='pyarrow' if PYARROW_AVAILABLE else 'fastparquet'
            )
            
            end_row = min(start_row + num_rows, len(df))
            chunk_df = df.iloc[start_row:end_row].copy()
            
            del df
            gc.collect()
            
            return self._optimize_chunk_memory(chunk_df)
            
        except Exception as e:
            logger.warning(f"pandas 청크 읽기 실패: {e}")
            return pd.DataFrame()
    
    def read_all_data(self) -> pd.DataFrame:
        """일반 전체 데이터 로딩"""
        logger.info(f"일반 데이터 로딩 시작: {self.total_rows:,}행")
        start_time = time.time()
        
        try:
            if PYARROW_AVAILABLE and self.parquet_file:
                table = self.parquet_file.read(use_threads=True)
                df = table.to_pandas()
            else:
                df = pd.read_parquet(self.file_path, engine='auto')
            
            df = self._optimize_dataframe_memory(df)
            
            read_time = time.time() - start_time
            self.read_statistics['total_time'] = read_time
            
            logger.info(f"일반 데이터 로딩 완료: {df.shape}, {read_time:.2f}초")
            
            return df
            
        except Exception as e:
            logger.error(f"일반 데이터 로딩 실패: {e}")
            return pd.DataFrame()
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 메모리 최적화"""
        if df.empty:
            return df
        
        logger.info("DataFrame 메모리 최적화 시작")
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
                    # float32로 변환 시 정밀도 손실 확인
                    original_values = df[col].dropna()
                    if len(original_values) > 0:
                        converted_values = original_values.astype('float32')
                        
                        # 정밀도 손실이 적으면 변환
                        if np.allclose(original_values, converted_values, rtol=1e-6, equal_nan=True):
                            df[col] = df[col].astype('float32')
                        
                except Exception as e:
                    logger.warning(f"{col} 실수형 최적화 실패: {e}")
                    continue
            
            # 범주형 최적화
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:  # 카디널리티가 낮으면 category로 변환
                        df[col] = df[col].astype('category')
                except Exception as e:
                    logger.warning(f"{col} 범주형 최적화 실패: {e}")
                    continue
            
            optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            logger.info(f"메모리 최적화 완료: {original_memory:.1f}MB → {optimized_memory:.1f}MB "
                       f"({reduction:.1f}% 감소)")
            
        except Exception as e:
            logger.warning(f"DataFrame 메모리 최적화 실패: {e}")
        
        return df
    
    def _optimize_chunk_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """청크 단위 메모리 최적화"""
        try:
            # 기본 타입 최적화만 수행 (성능상 이유)
            for col in df.select_dtypes(include=['int64']).columns:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    if not pd.isna(col_min) and not pd.isna(col_max):
                        if col_min >= 0 and col_max < 65535:
                            df[col] = df[col].astype('uint16')
                        elif col_min > -32768 and col_max < 32767:
                            df[col] = df[col].astype('int16')
                except:
                    continue
            
            for col in df.select_dtypes(include=['float64']).columns:
                try:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                except:
                    continue
            
            return df
            
        except Exception as e:
            logger.warning(f"청크 메모리 최적화 실패: {e}")
            return df
    
    def _log_read_statistics(self):
        """읽기 통계 로깅"""
        stats = self.read_statistics
        if stats['total_time'] > 0:
            throughput = self.total_rows / stats['total_time']
            logger.info(f"읽기 성능: {throughput:,.0f}행/초, "
                       f"총 시간: {stats['total_time']:.2f}초")

class LargeDataLoader:
    """대용량 데이터 처리 특화 데이터 로더"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = AdvancedMemoryMonitor()
        self.file_validator = DataFileValidator(config)
        self.train_data = None
        self.test_data = None
        self.feature_columns = None
        self.target_column = 'clicked'
        
        # 대용량 데이터 처리 설정
        self.data_config = config.get_data_config()
        self.memory_config = config.get_memory_config()
        
        # 성능 통계
        self.loading_stats = {
            'start_time': time.time(),
            'data_loaded': False,
            'train_rows': 0,
            'test_rows': 0,
            'loading_time': 0.0,
            'memory_usage': 0.0,
            'performance_metrics': {}
        }
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """대용량 데이터 최적화 로딩 - 메인 메서드"""
        logger.info("=== 대용량 데이터 최적화 로딩 시작 ===")
        
        # 1. 데이터 파일 검증
        validation_results = self.file_validator.validate_data_files()
        if not validation_results['validation_passed']:
            if self.config.REQUIRE_REAL_DATA and not self.config.SAMPLE_DATA_FALLBACK:
                raise ValueError("실제 대용량 데이터 파일이 필요하지만 검증에 실패했습니다. "
                               "샘플 데이터 대체가 비활성화되어 있습니다.")
            else:
                logger.warning("실제 데이터 파일 검증 실패, 샘플 데이터로 대체")
                return self._create_enhanced_sample_data()
        
        # 2. 메모리 상태 확인
        self.memory_monitor.log_memory_status("로딩 시작")
        
        try:
            # 3. 병렬 데이터 로딩
            train_df = self._load_train_data_optimized()
            
            self.memory_monitor.log_memory_status("학습 데이터 로딩 후")
            
            test_df = self._load_full_test_data_guaranteed()
            
            self.memory_monitor.log_memory_status("테스트 데이터 로딩 후")
            
            # 4. 데이터 검증
            self._validate_loaded_data(train_df, test_df)
            
            # 5. 피처 컬럼 정의
            self.feature_columns = [col for col in train_df.columns 
                                  if col != self.target_column]
            
            # 6. 성능 통계 업데이트
            self.loading_stats.update({
                'data_loaded': True,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'loading_time': time.time() - self.loading_stats['start_time'],
                'memory_usage': self.memory_monitor.get_detailed_memory_info()['process_rss_gb']
            })
            
            self._log_loading_completion(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"대용량 데이터 로딩 실패: {e}")
            self.memory_monitor.force_memory_cleanup(aggressive=True)
            raise
    
    def _load_train_data_optimized(self) -> pd.DataFrame:
        """최적화된 학습 데이터 로딩"""
        logger.info("최적화된 학습 데이터 로딩 시작")
        
        try:
            with OptimizedParquetReader(
                self.config.TRAIN_PATH, 
                chunk_size=self.data_config['chunk_size']
            ) as reader:
                
                total_rows = reader.get_total_rows()
                logger.info(f"전체 학습 데이터: {total_rows:,}행")
                
                # 메모리 기반 처리 전략 결정
                available_memory = self.memory_monitor.get_detailed_memory_info()['available_gb']
                
                if available_memory > 35 and total_rows <= self.data_config['max_train_size']:
                    # 충분한 메모리: 전체 데이터 병렬 로딩
                    logger.info("전체 학습 데이터 병렬 로딩")
                    max_workers = min(self.config.NUM_WORKERS, 4)
                    df = reader.read_all_data_parallel(max_workers=max_workers)
                    
                elif available_memory > 25:
                    # 보통 메모리: 전체 데이터 일반 로딩
                    logger.info("전체 학습 데이터 일반 로딩")
                    df = reader.read_all_data()
                    
                else:
                    # 메모리 부족: 스마트 샘플링
                    logger.info("메모리 부족으로 스마트 샘플링 적용")
                    target_size = min(
                        int(available_memory * 200000),  # 메모리 기반 크기
                        self.data_config['max_train_size']
                    )
                    df = self._smart_sampling_load(reader, total_rows, target_size)
                
                if df.empty:
                    raise ValueError("학습 데이터 로딩 결과가 비어있습니다")
                
                logger.info(f"학습 데이터 로딩 완료: {df.shape}")
                return df
                
        except Exception as e:
            logger.error(f"학습 데이터 로딩 실패: {e}")
            raise
    
    def _load_full_test_data_guaranteed(self) -> pd.DataFrame:
        """전체 테스트 데이터 로딩 보장"""
        logger.info("전체 테스트 데이터 로딩 시작 (필수)")
        
        try:
            with OptimizedParquetReader(
                self.config.TEST_PATH,
                chunk_size=self.data_config['chunk_size']
            ) as reader:
                
                total_rows = reader.get_total_rows()
                logger.info(f"전체 테스트 데이터: {total_rows:,}행")
                
                # 테스트 데이터는 반드시 전체 로딩
                if total_rows < self.data_config['min_test_size'] * (1 - self.data_config['data_size_tolerance']):
                    logger.warning(f"테스트 데이터 크기가 예상보다 작습니다: {total_rows:,}")
                
                # 메모리 상태에 따른 로딩 전략
                available_memory = self.memory_monitor.get_detailed_memory_info()['available_gb']
                
                if available_memory > 20:
                    # 병렬 로딩
                    max_workers = min(self.config.NUM_WORKERS // 2, 3)
                    df = reader.read_all_data_parallel(max_workers=max_workers)
                else:
                    # 일반 로딩
                    df = reader.read_all_data()
                
                if df.empty or len(df) < 1000000:
                    raise ValueError(f"테스트 데이터 로딩이 불완전합니다: {len(df):,}행")
                
                logger.info(f"전체 테스트 데이터 로딩 완료: {df.shape}")
                return df
                
        except Exception as e:
            logger.error(f"테스트 데이터 로딩 실패: {e}")
            raise
    
    def _smart_sampling_load(self, reader: OptimizedParquetReader, 
                            total_rows: int, target_size: int) -> pd.DataFrame:
        """스마트 샘플링 로딩"""
        logger.info(f"스마트 샘플링: {total_rows:,} → {target_size:,}행")
        
        try:
            # 균등 분포 샘플링
            step = max(1, total_rows // target_size)
            sample_indices = np.arange(0, total_rows, step)[:target_size]
            
            # 청크 단위로 샘플링
            chunks = []
            chunk_size = reader.chunk_size
            
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                
                # 현재 청크에서 필요한 인덱스
                chunk_sample_indices = sample_indices[
                    (sample_indices >= start_idx) & (sample_indices < end_idx)
                ] - start_idx
                
                if len(chunk_sample_indices) > 0:
                    # 청크 전체 로딩 후 샘플링
                    if PYARROW_AVAILABLE and reader.parquet_file:
                        try:
                            table = reader.parquet_file.read(
                                columns=None,
                                use_threads=True
                            )
                            chunk_df = table.to_pandas()
                            
                            if start_idx + chunk_size < len(chunk_df):
                                chunk_df = chunk_df.iloc[start_idx:end_idx].copy()
                            
                        except Exception as e:
                            logger.warning(f"PyArrow 청크 로딩 실패: {e}")
                            chunk_df = pd.read_parquet(reader.file_path)
                            chunk_df = chunk_df.iloc[start_idx:end_idx].copy()
                    else:
                        chunk_df = pd.read_parquet(reader.file_path)
                        chunk_df = chunk_df.iloc[start_idx:end_idx].copy()
                    
                    # 샘플 선택
                    valid_indices = chunk_sample_indices[chunk_sample_indices < len(chunk_df)]
                    if len(valid_indices) > 0:
                        sampled_chunk = chunk_df.iloc[valid_indices].copy()
                        sampled_chunk = reader._optimize_chunk_memory(sampled_chunk)
                        chunks.append(sampled_chunk)
                    
                    del chunk_df
                
                # 메모리 정리
                if len(chunks) % 5 == 0:
                    gc.collect()
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"스마트 샘플링 완료: {df.shape}")
                return df
            else:
                raise ValueError("스마트 샘플링 실패: 유효한 데이터 없음")
                
        except Exception as e:
            logger.error(f"스마트 샘플링 실패: {e}")
            raise
    
    def _validate_loaded_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """로딩된 데이터 검증"""
        logger.info("로딩된 데이터 검증 시작")
        
        # 기본 형태 검증
        if train_df.empty or test_df.empty:
            raise ValueError("로딩된 데이터가 비어있습니다")
        
        # 최소 크기 검증
        min_train_size = self.config.MIN_TRAIN_SIZE * (1 - self.config.DATA_SIZE_TOLERANCE)
        min_test_size = self.config.MIN_TEST_SIZE * (1 - self.config.DATA_SIZE_TOLERANCE)
        
        if len(train_df) < min_train_size:
            logger.warning(f"학습 데이터 크기 부족: {len(train_df):,} < {min_train_size:,}")
        
        if len(test_df) < min_test_size:
            raise ValueError(f"테스트 데이터 크기 부족: {len(test_df):,} < {min_test_size:,}")
        
        # 타겟 컬럼 검증
        if self.target_column not in train_df.columns:
            raise ValueError(f"타겟 컬럼 '{self.target_column}'이 학습 데이터에 없습니다")
        
        # CTR 분포 검증
        target_dist = train_df[self.target_column].value_counts()
        actual_ctr = train_df[self.target_column].mean()
        
        logger.info(f"타겟 분포: {target_dist.to_dict()}")
        logger.info(f"실제 CTR: {actual_ctr:.4f}")
        
        # 공통 피처 검증
        train_features = set(train_df.columns) - {self.target_column}
        test_features = set(test_df.columns)
        
        missing_features = train_features - test_features
        extra_features = test_features - train_features
        
        if missing_features:
            logger.warning(f"테스트 데이터에 누락된 피처: {missing_features}")
        
        if extra_features:
            logger.warning(f"테스트 데이터의 추가 피처: {extra_features}")
        
        logger.info("로딩된 데이터 검증 완료")
    
    def _create_enhanced_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """향상된 샘플 데이터 생성 (실제 데이터 없을 때만)"""
        logger.warning("실제 데이터가 없어 향상된 샘플 데이터를 생성합니다")
        
        # 더 큰 샘플 데이터 생성
        np.random.seed(42)
        n_train = 2000000  # 200만행으로 확대
        n_test = 1527298   # 실제 테스트 크기 유지
        
        # 실제 CTR 패턴 반영
        train_data = {
            'clicked': np.random.binomial(1, 0.0201, n_train),
        }
        
        # 더 복잡한 피처 생성
        for i in range(1, 11):
            if i <= 5:
                train_data[f'feat_e_{i}'] = np.random.normal(65, 150, n_train)
            else:
                train_data[f'feat_e_{i}'] = np.random.normal(-100, 250, n_train)
        
        for i in range(1, 7):
            train_data[f'feat_d_{i}'] = np.random.exponential(2.5, n_train)
        
        for i in range(1, 9):
            train_data[f'feat_c_{i}'] = np.random.poisson(1.2, n_train)
        
        for i in range(1, 7):
            train_data[f'feat_b_{i}'] = np.random.uniform(0, 12, n_train)
        
        train_df = pd.DataFrame(train_data)
        
        # 테스트 데이터 생성
        test_data = {}
        for col in train_df.columns:
            if col != 'clicked':
                if col.startswith('feat_e'):
                    if int(col.split('_')[-1]) <= 5:
                        test_data[col] = np.random.normal(65, 150, n_test)
                    else:
                        test_data[col] = np.random.normal(-100, 250, n_test)
                elif col.startswith('feat_d'):
                    test_data[col] = np.random.exponential(2.5, n_test)
                elif col.startswith('feat_c'):
                    test_data[col] = np.random.poisson(1.2, n_test)
                elif col.startswith('feat_b'):
                    test_data[col] = np.random.uniform(0, 12, n_test)
        
        test_df = pd.DataFrame(test_data)
        
        # 메모리 최적화
        with OptimizedParquetReader('dummy', 1000000) as reader:
            train_df = reader._optimize_dataframe_memory(train_df)
            test_df = reader._optimize_dataframe_memory(test_df)
        
        logger.warning(f"향상된 샘플 데이터 생성 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
        
        return train_df, test_df
    
    def _log_loading_completion(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """로딩 완료 로깅"""
        stats = self.loading_stats
        memory_info = self.memory_monitor.get_detailed_memory_info()
        
        logger.info("=== 대용량 데이터 로딩 완료 ===")
        logger.info(f"학습 데이터: {train_df.shape}")
        logger.info(f"테스트 데이터: {test_df.shape}")
        logger.info(f"로딩 시간: {stats['loading_time']:.2f}초")
        logger.info(f"메모리 사용량: {memory_info['process_rss_gb']:.2f}GB")
        logger.info(f"피크 메모리: {memory_info['peak_memory_gb']:.2f}GB")
        
        if train_df.shape[0] >= self.config.MIN_TRAIN_SIZE:
            logger.info("✓ 학습 데이터 크기 요구사항 충족")
        else:
            logger.warning("△ 학습 데이터 크기 부족")
        
        if test_df.shape[0] >= self.config.MIN_TEST_SIZE:
            logger.info("✓ 테스트 데이터 크기 요구사항 충족")
        else:
            logger.warning("△ 테스트 데이터 크기 부족")
        
        logger.info("=== 로딩 완료 ===")
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """로딩 통계 반환"""
        return self.loading_stats.copy()

# 기존 클래스들과의 호환성을 위한 별칭
DataLoader = LargeDataLoader
MemoryMonitor = AdvancedMemoryMonitor
ChunkedParquetReader = OptimizedParquetReader

class DataValidator:
    """데이터 품질 검증 클래스 - 호환성 유지"""
    
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
        
        for col in list(common_cols)[:50]:
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

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 설정 확인
    config = Config()
    config.verify_data_requirements()
    
    # 데이터 로더 테스트
    try:
        loader = LargeDataLoader(config)
        train_df, test_df = loader.load_large_data_optimized()
        
        print(f"학습 데이터: {train_df.shape}")
        print(f"테스트 데이터: {test_df.shape}")
        print(f"로딩 통계: {loader.get_loading_stats()}")
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")