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
import sys

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

class ProgressReporter:
    """진행 상황 보고 클래스"""
    
    def __init__(self, total_items: int, description: str = "진행률"):
        self.total_items = total_items
        self.current_items = 0
        self.description = description
        self.start_time = time.time()
        self.last_report_time = 0
        self.report_interval = 3.0  # 3초마다 보고로 조정
        
    def update(self, increment: int = 1):
        """진행 상황 업데이트"""
        self.current_items += increment
        current_time = time.time()
        
        if current_time - self.last_report_time >= self.report_interval or self.current_items >= self.total_items:
            self.report_progress()
            self.last_report_time = current_time
    
    def report_progress(self):
        """진행 상황 보고"""
        if self.total_items <= 0:
            return
            
        progress_percent = min(100.0, (self.current_items / self.total_items) * 100)
        elapsed_time = time.time() - self.start_time
        
        if progress_percent > 0 and elapsed_time > 0:
            estimated_total_time = elapsed_time * (100 / progress_percent)
            remaining_time = max(0, estimated_total_time - elapsed_time)
            
            # 처리 속도 계산
            items_per_sec = self.current_items / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"{self.description}: {progress_percent:.1f}% "
                       f"({self.current_items:,}/{self.total_items:,}) - "
                       f"속도: {items_per_sec:,.0f}행/초 - "
                       f"남은 시간: {remaining_time:.0f}초")
        else:
            logger.info(f"{self.description}: {progress_percent:.1f}% ({self.current_items:,}/{self.total_items:,})")

class AggressiveMemoryMonitor:
    """적극적 메모리 모니터링 및 관리"""
    
    def __init__(self, max_memory_gb: float = 35.0):  # 35GB로 제한 축소
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.max_memory_gb = max_memory_gb
        self.memory_history = []
        self.peak_memory = 0.0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # 더 엄격한 임계값 설정
        self.warning_threshold = max_memory_gb * 0.6   # 60% 경고
        self.critical_threshold = max_memory_gb * 0.75  # 75% 위험
        self.abort_threshold = max_memory_gb * 0.85     # 85% 중단
        
    def get_process_memory_gb(self) -> float:
        """현재 프로세스 메모리 사용량 (GB)"""
        if not self.monitoring_enabled:
            return 2.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception:
            return 2.0
    
    def get_available_memory_gb(self) -> float:
        """시스템 사용 가능 메모리 (GB)"""
        if not self.monitoring_enabled:
            return 30.0
        
        try:
            return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 30.0
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인 - 엄격한 기준"""
        process_memory = self.get_process_memory_gb()
        available_memory = self.get_available_memory_gb()
        
        # 압박 수준 결정 - 더 엄격하게
        if process_memory > self.abort_threshold or available_memory < 3:
            pressure_level = "abort"
        elif process_memory > self.critical_threshold or available_memory < 8:
            pressure_level = "critical"
        elif process_memory > self.warning_threshold or available_memory < 15:
            pressure_level = "high"
        elif available_memory < 25:
            pressure_level = "moderate"
        else:
            pressure_level = "low"
        
        return {
            'memory_pressure': pressure_level in ['high', 'critical', 'abort'],
            'pressure_level': pressure_level,
            'process_memory_gb': process_memory,
            'available_memory_gb': available_memory,
            'should_reduce_chunk_size': pressure_level in ['moderate', 'high', 'critical', 'abort'],
            'should_force_gc': pressure_level in ['high', 'critical', 'abort'],
            'should_abort': pressure_level == 'abort',
            'recommendation': self._get_memory_recommendation(pressure_level, process_memory, available_memory)
        }
    
    def _get_memory_recommendation(self, pressure_level: str, process_gb: float, available_gb: float) -> str:
        """메모리 상태별 권장 사항"""
        if pressure_level == "abort":
            return f"중단: 프로세스 {process_gb:.1f}GB, 가용 {available_gb:.1f}GB - 즉시 중단 필요"
        elif pressure_level == "critical":
            return f"위험: 프로세스 {process_gb:.1f}GB, 가용 {available_gb:.1f}GB - 적극적 정리 필요"
        elif pressure_level == "high":
            return f"경고: 프로세스 {process_gb:.1f}GB, 가용 {available_gb:.1f}GB - 청크 크기 축소"
        elif pressure_level == "moderate":
            return f"주의: 프로세스 {process_gb:.1f}GB, 가용 {available_gb:.1f}GB - 모니터링 강화"
        else:
            return f"양호: 프로세스 {process_gb:.1f}GB, 가용 {available_gb:.1f}GB"
    
    def log_memory_status(self, context: str = "", force: bool = False):
        """메모리 상태 로깅"""
        try:
            pressure = self.check_memory_pressure()
            
            if force or pressure['pressure_level'] != 'low':
                logger.info(f"메모리 [{context}]: 프로세스 {pressure['process_memory_gb']:.1f}GB, "
                           f"가용 {pressure['available_memory_gb']:.1f}GB - "
                           f"{pressure['pressure_level'].upper()}")
            
            if pressure['memory_pressure']:
                logger.warning(f"메모리 압박: {pressure['recommendation']}")
                
        except Exception as e:
            logger.warning(f"메모리 상태 로깅 실패: {e}")
    
    def aggressive_memory_cleanup(self) -> float:
        """적극적 메모리 정리"""
        try:
            initial_memory = self.get_process_memory_gb()
            pressure = self.check_memory_pressure()
            
            # 압박 수준에 따른 정리 강도
            if pressure['pressure_level'] == 'abort':
                cleanup_rounds = 10
                sleep_time = 0.5
            elif pressure['pressure_level'] == 'critical':
                cleanup_rounds = 7
                sleep_time = 0.3
            elif pressure['pressure_level'] == 'high':
                cleanup_rounds = 5
                sleep_time = 0.2
            elif pressure['pressure_level'] == 'moderate':
                cleanup_rounds = 3
                sleep_time = 0.1
            else:
                cleanup_rounds = 2
                sleep_time = 0.05
            
            # 강력한 가비지 컬렉션
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                logger.debug(f"GC 라운드 {i+1}: {collected}개 객체 수집")
            
            # Windows 메모리 정리
            if cleanup_rounds >= 5:
                try:
                    import ctypes
                    if hasattr(ctypes, 'windll'):
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                except Exception:
                    pass
            
            final_memory = self.get_process_memory_gb()
            memory_freed = initial_memory - final_memory
            
            if memory_freed > 0.1:
                logger.info(f"적극적 메모리 정리: {memory_freed:.2f}GB 해제 ({cleanup_rounds}라운드)")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"적극적 메모리 정리 실패: {e}")
            return 0.0

class MemoryOptimizedChunkReader:
    """메모리 최적화 청크 리더"""
    
    def __init__(self, file_path: str, initial_chunk_size: int = 200000):  # 20만행으로 축소
        self.file_path = file_path
        self.initial_chunk_size = initial_chunk_size
        self.current_chunk_size = initial_chunk_size
        self.total_rows = 0
        self.memory_monitor = AggressiveMemoryMonitor()
        self.progress_reporter = None
        
        # 더 보수적인 청크 크기 설정
        self.min_chunk_size = 50000   # 5만행 최소
        self.max_chunk_size = 500000  # 50만행 최대
        
    def __enter__(self):
        """초기화"""
        logger.info(f"메모리 최적화 청크 리더 초기화: {self.file_path}")
        
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {self.file_path}")
            
            file_size_mb = os.path.getsize(self.file_path) / (1024**2)
            logger.info(f"파일 크기: {file_size_mb:.1f}MB")
            
            # 메타데이터 읽기
            self.total_rows = self._get_total_rows()
            
            # 진행 상황 리포터 초기화
            self.progress_reporter = ProgressReporter(
                self.total_rows, 
                f"데이터 로딩 ({file_size_mb:.0f}MB)"
            )
            
            logger.info(f"청크 리더 준비 완료: {self.total_rows:,}행")
            return self
            
        except Exception as e:
            logger.error(f"청크 리더 초기화 실패: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """정리"""
        try:
            if self.progress_reporter:
                self.progress_reporter.report_progress()
            
            # 강력한 메모리 정리
            self.memory_monitor.aggressive_memory_cleanup()
            
            logger.info("청크 리더 정리 완료")
            
        except Exception as e:
            logger.warning(f"청크 리더 정리 실패: {e}")
    
    def _get_total_rows(self) -> int:
        """총 행 수 확인"""
        try:
            if PYARROW_AVAILABLE:
                try:
                    parquet_file = pq.ParquetFile(self.file_path)
                    total_rows = parquet_file.metadata.num_rows
                    logger.info(f"PyArrow 메타데이터: {total_rows:,}행")
                    return total_rows
                except Exception as e:
                    logger.warning(f"PyArrow 메타데이터 실패: {e}")
            
            # 추정 방식
            logger.info("행 수 추정 시작")
            sample_df = pd.read_parquet(self.file_path, nrows=10000)
            file_size = os.path.getsize(self.file_path)
            sample_memory = sample_df.memory_usage(deep=True).sum()
            
            estimated_rows = int((file_size / sample_memory) * 10000 * 0.5)  # 더 보수적 추정
            
            del sample_df
            gc.collect()
            
            logger.info(f"행 수 추정 완료: {estimated_rows:,}행")
            return estimated_rows
            
        except Exception as e:
            logger.warning(f"행 수 확인 실패: {e}. 기본값 사용")
            return 5000000  # 기본값 축소
    
    def read_data_in_chunks(self) -> pd.DataFrame:
        """메모리 효율적 청크별 데이터 읽기"""
        logger.info(f"메모리 효율적 데이터 읽기 시작: {self.total_rows:,}행")
        
        final_chunks = []
        total_processed = 0
        chunk_count = 0
        start_time = time.time()
        
        try:
            # 초기 메모리 상태 확인
            self.memory_monitor.log_memory_status("읽기 시작", force=True)
            
            while total_processed < self.total_rows:
                # 메모리 압박 확인
                pressure = self.memory_monitor.check_memory_pressure()
                
                if pressure['should_abort']:
                    logger.error(f"메모리 한계 도달. 처리된 데이터: {total_processed:,}행")
                    break
                
                # 적응형 청크 크기 조정
                self._adjust_chunk_size_conservative(pressure)
                
                # 청크 읽기
                remaining_rows = self.total_rows - total_processed
                current_chunk_size = min(self.current_chunk_size, remaining_rows)
                
                try:
                    chunk_start_time = time.time()
                    
                    # 청크 데이터 읽기
                    chunk_df = self._read_single_chunk(total_processed, current_chunk_size)
                    
                    if chunk_df is None or chunk_df.empty:
                        logger.warning(f"빈 청크: {total_processed:,} 위치")
                        total_processed += current_chunk_size
                        continue
                    
                    # 즉시 메모리 최적화
                    chunk_df = self._aggressive_chunk_optimization(chunk_df)
                    
                    # 청크 저장 (메모리 효율적)
                    final_chunks.append(chunk_df)
                    
                    chunk_time = time.time() - chunk_start_time
                    chunk_count += 1
                    total_processed += len(chunk_df)
                    
                    # 진행 상황 업데이트
                    if self.progress_reporter:
                        self.progress_reporter.update(len(chunk_df))
                    
                    # 주기적 메모리 정리 및 상태 확인
                    if chunk_count % 3 == 0:  # 3청크마다 정리
                        self.memory_monitor.aggressive_memory_cleanup()
                        self.memory_monitor.log_memory_status(f"청크{chunk_count}")
                    
                    # 메모리 압박 시 중간 결합
                    if (len(final_chunks) >= 5 and 
                        pressure['pressure_level'] in ['high', 'critical']):
                        logger.info(f"메모리 압박으로 중간 결합 수행: {len(final_chunks)}개 청크")
                        final_chunks = self._combine_chunks_memory_efficient(final_chunks)
                        self.memory_monitor.aggressive_memory_cleanup()
                    
                except Exception as e:
                    logger.error(f"청크 {chunk_count + 1} 처리 실패 (위치: {total_processed:,}): {e}")
                    
                    # 에러 복구 시도
                    if self._attempt_chunk_recovery(total_processed, current_chunk_size):
                        continue
                    else:
                        logger.error("복구 실패. 처리 중단")
                        break
            
            # 최종 데이터 결합
            if final_chunks:
                logger.info(f"최종 데이터 결합: {len(final_chunks)}개 청크")
                combined_df = self._combine_chunks_memory_efficient(final_chunks)
                
                total_time = time.time() - start_time
                throughput = len(combined_df) / total_time if total_time > 0 else 0
                
                logger.info(f"청크 읽기 완료: {combined_df.shape}, "
                           f"{total_time:.2f}초, {throughput:,.0f}행/초")
                
                return combined_df
            else:
                logger.error("읽기 가능한 청크가 없습니다")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"청크 데이터 읽기 실패: {e}")
            return pd.DataFrame()
    
    def _adjust_chunk_size_conservative(self, pressure: Dict[str, Any]):
        """보수적 청크 크기 조정"""
        old_size = self.current_chunk_size
        
        if pressure['pressure_level'] == 'abort':
            self.current_chunk_size = self.min_chunk_size
        elif pressure['pressure_level'] == 'critical':
            self.current_chunk_size = max(self.min_chunk_size, self.current_chunk_size // 4)
        elif pressure['pressure_level'] == 'high':
            self.current_chunk_size = max(self.min_chunk_size, self.current_chunk_size // 2)
        elif pressure['pressure_level'] == 'moderate':
            self.current_chunk_size = max(self.min_chunk_size, int(self.current_chunk_size * 0.7))
        elif pressure['pressure_level'] == 'low':
            # 조심스럽게 증가
            self.current_chunk_size = min(self.max_chunk_size, int(self.current_chunk_size * 1.1))
        
        if old_size != self.current_chunk_size:
            logger.info(f"청크 크기 조정: {old_size:,} → {self.current_chunk_size:,} "
                       f"({pressure['pressure_level']})")
    
    def _read_single_chunk(self, start_row: int, num_rows: int) -> Optional[pd.DataFrame]:
        """단일 청크 읽기"""
        try:
            # PyArrow 시도
            if PYARROW_AVAILABLE:
                try:
                    df = pd.read_parquet(
                        self.file_path,
                        engine='pyarrow'
                    )
                    
                    end_row = min(start_row + num_rows, len(df))
                    if start_row >= len(df):
                        return None
                    
                    chunk_df = df.iloc[start_row:end_row].copy()
                    
                    # 즉시 원본 해제
                    del df
                    gc.collect()
                    
                    return chunk_df
                    
                except Exception as e:
                    logger.warning(f"PyArrow 청크 읽기 실패: {e}")
            
            # pandas 대안
            df = pd.read_parquet(self.file_path)
            end_row = min(start_row + num_rows, len(df))
            
            if start_row >= len(df):
                return None
            
            chunk_df = df.iloc[start_row:end_row].copy()
            del df
            gc.collect()
            
            return chunk_df
            
        except Exception as e:
            logger.warning(f"청크 읽기 실패: {e}")
            return None
    
    def _aggressive_chunk_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """적극적 청크 메모리 최적화"""
        if df.empty:
            return df
        
        try:
            original_memory = df.memory_usage(deep=True).sum() / (1024**2)
            
            # 정수형 대폭 최적화
            for col in df.select_dtypes(include=['int64', 'int32']).columns:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= 0:
                        if col_max <= 255:
                            df[col] = df[col].astype('uint8')
                        elif col_max <= 65535:
                            df[col] = df[col].astype('uint16')
                        else:
                            df[col] = df[col].astype('uint32')
                    else:
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype('int8')
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype('int16')
                        else:
                            df[col] = df[col].astype('int32')
                            
                except Exception:
                    try:
                        df[col] = df[col].astype('int32')
                    except Exception:
                        pass
            
            # 실수형 최적화
            for col in df.select_dtypes(include=['float64']).columns:
                try:
                    df[col] = df[col].astype('float32')
                except Exception:
                    pass
            
            # 범주형 최적화 (매우 선택적)
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    if unique_count < total_count * 0.3:  # 30% 미만만 범주형으로
                        df[col] = df[col].astype('category')
                except Exception:
                    pass
            
            optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            if reduction > 10:
                logger.debug(f"청크 최적화: {original_memory:.1f}MB → {optimized_memory:.1f}MB "
                           f"({reduction:.1f}% 감소)")
            
            return df
            
        except Exception as e:
            logger.warning(f"청크 최적화 실패: {e}")
            return df
    
    def _combine_chunks_memory_efficient(self, chunks: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """메모리 효율적 청크 결합"""
        if not chunks:
            return []
        
        if len(chunks) == 1:
            return chunks
        
        try:
            # 작은 배치로 결합
            combined_chunks = []
            batch_size = 3  # 3개씩 배치
            
            for i in range(0, len(chunks), batch_size):
                try:
                    batch = chunks[i:i + batch_size]
                    
                    if len(batch) == 1:
                        combined_chunks.append(batch[0])
                    else:
                        # 배치 결합
                        batch_combined = pd.concat(batch, ignore_index=True)
                        # 즉시 최적화
                        batch_combined = self._aggressive_chunk_optimization(batch_combined)
                        combined_chunks.append(batch_combined)
                    
                    # 원본 청크 해제
                    for chunk in batch:
                        del chunk
                    
                    # 메모리 정리
                    gc.collect()
                    
                    logger.debug(f"배치 {i//batch_size + 1} 결합 완료")
                    
                except Exception as e:
                    logger.warning(f"배치 {i//batch_size + 1} 결합 실패: {e}")
                    continue
            
            # 원본 청크 리스트 정리
            chunks.clear()
            gc.collect()
            
            logger.info(f"중간 결합 완료: {len(combined_chunks)}개 청크")
            return combined_chunks
            
        except Exception as e:
            logger.error(f"메모리 효율적 결합 실패: {e}")
            return chunks
    
    def _attempt_chunk_recovery(self, failed_position: int, failed_chunk_size: int) -> bool:
        """청크 에러 복구 시도"""
        try:
            logger.info(f"청크 복구 시도: 위치 {failed_position:,}")
            
            # 청크 크기를 1/4로 축소하여 재시도
            recovery_chunk_size = max(self.min_chunk_size, failed_chunk_size // 4)
            
            recovery_df = self._read_single_chunk(failed_position, recovery_chunk_size)
            
            if recovery_df is not None and not recovery_df.empty:
                logger.info(f"청크 복구 성공: {len(recovery_df):,}행")
                self.current_chunk_size = recovery_chunk_size
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"청크 복구 실패: {e}")
            return False

class LargeDataLoader:
    """메모리 효율성 극대화 데이터 로더"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = AggressiveMemoryMonitor()
        self.target_column = 'clicked'
        
        # 성능 통계
        self.loading_stats = {
            'start_time': time.time(),
            'data_loaded': False,
            'train_rows': 0,
            'test_rows': 0,
            'loading_time': 0.0,
            'memory_usage': 0.0
        }
        
        logger.info("메모리 효율성 극대화 데이터 로더 초기화 완료")
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """메모리 효율성 극대화 데이터 로딩"""
        logger.info("=== 메모리 효율성 극대화 데이터 로딩 시작 ===")
        
        try:
            # 1. 초기 메모리 상태
            self.memory_monitor.log_memory_status("로딩 시작", force=True)
            
            # 2. 데이터 파일 검증
            if not self._validate_data_files():
                logger.warning("실제 데이터 파일 검증 실패, 샘플 데이터로 대체")
                return self._create_optimized_sample_data()
            
            # 3. 학습 데이터 로딩 (보수적 접근)
            train_df = self._load_train_data_conservative()
            
            if train_df is None or train_df.empty:
                raise ValueError("학습 데이터 로딩 실패")
            
            # 중간 메모리 정리
            self.memory_monitor.aggressive_memory_cleanup()
            self.memory_monitor.log_memory_status("학습 데이터 로딩 후", force=True)
            
            # 4. 테스트 데이터 로딩 (메모리 상태 고려)
            test_df = self._load_test_data_conservative()
            
            if test_df is None or test_df.empty:
                raise ValueError("테스트 데이터 로딩 실패")
            
            # 5. 최종 검증
            self._validate_loaded_data(train_df, test_df)
            
            # 6. 통계 업데이트
            self.loading_stats.update({
                'data_loaded': True,
                'train_rows': len(train_df),
                'test_rows': len(test_df),
                'loading_time': time.time() - self.loading_stats['start_time'],
                'memory_usage': self.memory_monitor.get_process_memory_gb()
            })
            
            self._log_loading_completion(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"메모리 효율성 극대화 데이터 로딩 실패: {e}")
            self.memory_monitor.aggressive_memory_cleanup()
            raise
    
    def _validate_data_files(self) -> bool:
        """데이터 파일 검증"""
        try:
            if not self.config.TRAIN_PATH.exists() or not self.config.TEST_PATH.exists():
                return False
            
            train_size_mb = self.config.TRAIN_PATH.stat().st_size / (1024**2)
            test_size_mb = self.config.TEST_PATH.stat().st_size / (1024**2)
            
            logger.info(f"파일 크기 - 학습: {train_size_mb:.1f}MB, 테스트: {test_size_mb:.1f}MB")
            
            return train_size_mb > 100 and test_size_mb > 50
            
        except Exception as e:
            logger.error(f"파일 검증 실패: {e}")
            return False
    
    def _load_train_data_conservative(self) -> pd.DataFrame:
        """보수적 학습 데이터 로딩"""
        logger.info("보수적 학습 데이터 로딩 시작")
        
        try:
            # 메모리 상태에 따른 초기 청크 크기 결정 (매우 보수적)
            memory_info = self.memory_monitor.check_memory_pressure()
            
            if memory_info['available_memory_gb'] > 30:
                initial_chunk_size = 200000  # 20만행
            elif memory_info['available_memory_gb'] > 20:
                initial_chunk_size = 150000  # 15만행
            else:
                initial_chunk_size = 100000  # 10만행
            
            logger.info(f"초기 청크 크기: {initial_chunk_size:,}행 "
                       f"(가용메모리: {memory_info['available_memory_gb']:.1f}GB)")
            
            # 청크 리더로 데이터 로딩
            with MemoryOptimizedChunkReader(str(self.config.TRAIN_PATH), initial_chunk_size) as reader:
                df = reader.read_data_in_chunks()
                
                if df.empty:
                    raise ValueError("청크 읽기 결과가 비어있습니다")
                
                # 타겟 컬럼 확인
                if self.target_column not in df.columns:
                    possible_targets = [col for col in df.columns if 'click' in col.lower()]
                    if possible_targets:
                        self.target_column = possible_targets[0]
                        logger.info(f"타겟 컬럼 변경: {self.target_column}")
                    else:
                        raise ValueError(f"타겟 컬럼 '{self.target_column}'을 찾을 수 없습니다")
                
                # 최종 최적화
                df = self._final_dataframe_optimization(df)
                
                # 통계 확인
                target_ctr = df[self.target_column].mean()
                logger.info(f"보수적 학습 데이터 로딩 완료: {df.shape}, CTR: {target_ctr:.4f}")
                
                return df
                
        except Exception as e:
            logger.error(f"보수적 학습 데이터 로딩 실패: {e}")
            raise
    
    def _load_test_data_conservative(self) -> pd.DataFrame:
        """보수적 테스트 데이터 로딩"""
        logger.info("보수적 테스트 데이터 로딩 시작")
        
        try:
            # 현재 메모리 압박 상태 확인
            pressure = self.memory_monitor.check_memory_pressure()
            
            # 더 보수적인 청크 크기
            if pressure['pressure_level'] in ['abort', 'critical']:
                initial_chunk_size = 50000   # 5만행
            elif pressure['pressure_level'] == 'high':
                initial_chunk_size = 100000  # 10만행
            elif pressure['pressure_level'] == 'moderate':
                initial_chunk_size = 150000  # 15만행
            else:
                initial_chunk_size = 200000  # 20만행
            
            logger.info(f"테스트 데이터 청크 크기: {initial_chunk_size:,}행 "
                       f"(메모리 상태: {pressure['pressure_level']})")
            
            # 청크 리더로 데이터 로딩
            with MemoryOptimizedChunkReader(str(self.config.TEST_PATH), initial_chunk_size) as reader:
                df = reader.read_data_in_chunks()
                
                if df.empty:
                    raise ValueError("테스트 데이터 청크 읽기 결과가 비어있습니다")
                
                # 최종 최적화
                df = self._final_dataframe_optimization(df)
                
                logger.info(f"보수적 테스트 데이터 로딩 완료: {df.shape}")
                
                return df
                
        except Exception as e:
            logger.error(f"보수적 테스트 데이터 로딩 실패: {e}")
            raise
    
    def _final_dataframe_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """최종 DataFrame 최적화"""
        if df.empty:
            return df
        
        try:
            logger.info("최종 DataFrame 최적화 시작")
            original_memory = df.memory_usage(deep=True).sum() / (1024**2)
            
            # 매우 적극적인 타입 최적화
            for col in df.columns:
                try:
                    dtype = df[col].dtype
                    
                    if dtype == 'int64':
                        col_min, col_max = df[col].min(), df[col].max()
                        if not pd.isna(col_min) and not pd.isna(col_max):
                            if col_min >= 0:
                                if col_max <= 255:
                                    df[col] = df[col].astype('uint8')
                                elif col_max <= 65535:
                                    df[col] = df[col].astype('uint16')
                                else:
                                    df[col] = df[col].astype('uint32')
                            else:
                                if col_min >= -128 and col_max <= 127:
                                    df[col] = df[col].astype('int8')
                                elif col_min >= -32768 and col_max <= 32767:
                                    df[col] = df[col].astype('int16')
                                else:
                                    df[col] = df[col].astype('int32')
                    
                    elif dtype == 'float64':
                        df[col] = df[col].astype('float32')
                    
                    elif dtype == 'object':
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio < 0.5:
                            df[col] = df[col].astype('category')
                            
                except Exception as e:
                    logger.warning(f"컬럼 {col} 최적화 실패: {e}")
                    continue
            
            # 결측치 처리
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], [1e6, -1e6])
            
            optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            logger.info(f"최종 최적화 완료: {original_memory:.1f}MB → {optimized_memory:.1f}MB "
                       f"({reduction:.1f}% 감소)")
            
            return df
            
        except Exception as e:
            logger.warning(f"최종 최적화 실패: {e}")
            return df
    
    def _validate_loaded_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """로딩된 데이터 검증"""
        logger.info("로딩된 데이터 검증 시작")
        
        try:
            if train_df.empty or test_df.empty:
                raise ValueError("로딩된 데이터가 비어있습니다")
            
            # 관대한 최소 크기 검증
            min_train_threshold = 500000   # 50만행
            min_test_threshold = 100000    # 10만행
            
            if len(train_df) < min_train_threshold:
                logger.warning(f"학습 데이터 크기: {len(train_df):,} < {min_train_threshold:,}")
            
            if len(test_df) < min_test_threshold:
                logger.warning(f"테스트 데이터 크기: {len(test_df):,} < {min_test_threshold:,}")
            
            # 타겟 컬럼 검증
            if self.target_column not in train_df.columns:
                raise ValueError(f"타겟 컬럼 '{self.target_column}'이 학습 데이터에 없습니다")
            
            # CTR 확인
            target_dist = train_df[self.target_column].value_counts()
            actual_ctr = train_df[self.target_column].mean()
            
            logger.info(f"타겟 분포: {target_dist.to_dict()}")
            logger.info(f"실제 CTR: {actual_ctr:.4f}")
            
            logger.info("로딩된 데이터 검증 완료")
            
        except Exception as e:
            logger.error(f"데이터 검증 실패: {e}")
            raise
    
    def _create_optimized_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """메모리 최적화 샘플 데이터 생성"""
        logger.warning("최적화된 샘플 데이터를 생성합니다")
        
        try:
            np.random.seed(42)
            n_train = 1000000  # 100만행으로 축소
            n_test = 500000    # 50만행으로 축소
            
            # 학습 데이터
            train_data = {
                self.target_column: np.random.binomial(1, 0.0201, n_train).astype('uint8'),
            }
            
            # 간소화된 피처
            for i in range(1, 6):
                train_data[f'feat_e_{i}'] = np.random.normal(0, 100, n_train).astype('float32')
            
            for i in range(1, 4):
                train_data[f'feat_d_{i}'] = np.random.exponential(2, n_train).astype('float32')
            
            for i in range(1, 4):
                train_data[f'feat_c_{i}'] = np.random.poisson(1, n_train).astype('uint16')
            
            train_df = pd.DataFrame(train_data)
            
            # 테스트 데이터
            test_data = {}
            for col in train_df.columns:
                if col != self.target_column:
                    test_data[col] = train_data[col][:n_test].copy()
            
            test_df = pd.DataFrame(test_data)
            
            logger.warning(f"최적화 샘플 데이터 생성 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")
            raise
    
    def _log_loading_completion(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """로딩 완료 로깅"""
        try:
            stats = self.loading_stats
            
            logger.info("=== 메모리 효율성 극대화 데이터 로딩 완료 ===")
            logger.info(f"학습 데이터: {train_df.shape}")
            logger.info(f"테스트 데이터: {test_df.shape}")
            logger.info(f"로딩 시간: {stats['loading_time']:.2f}초")
            logger.info(f"최종 메모리 사용량: {stats['memory_usage']:.2f}GB")
            
            # 처리 속도
            total_rows = len(train_df) + len(test_df)
            if stats['loading_time'] > 0:
                throughput = total_rows / stats['loading_time']
                logger.info(f"전체 처리 속도: {throughput:,.0f}행/초")
            
            # 최종 메모리 상태
            self.memory_monitor.log_memory_status("로딩 완료", force=True)
            
            logger.info("=== 메모리 효율적 로딩 완료 ===")
            
        except Exception as e:
            logger.warning(f"완료 로깅 실패: {e}")
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """로딩 통계 반환"""
        return self.loading_stats.copy()

# 호환성 별칭
DataLoader = LargeDataLoader
MemoryMonitor = AggressiveMemoryMonitor

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
        
        for col in list(common_cols)[:50]:
            try:
                train_dtype = str(train_df[col].dtype)
                test_dtype = str(test_df[col].dtype)
                
                if train_dtype != test_dtype:
                    dtype_mismatches.append({
                        'column': col,
                        'train_dtype': train_dtype,
                        'test_dtype': test_dtype
                    })
            except Exception:
                continue
        
        validation_results['dtype_mismatches'] = dtype_mismatches
        
        return validation_results

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    
    try:
        loader = LargeDataLoader(config)
        train_df, test_df = loader.load_large_data_optimized()
        
        print(f"학습 데이터: {train_df.shape}")
        print(f"테스트 데이터: {test_df.shape}")
        print(f"로딩 통계: {loader.get_loading_stats()}")
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")