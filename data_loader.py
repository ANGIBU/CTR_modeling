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
        self.report_interval = 2.0  # 2초마다 보고
        
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

class StreamingMemoryMonitor:
    """스트리밍 처리용 고급 메모리 모니터링"""
    
    def __init__(self, max_memory_gb: float = 45.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.max_memory_gb = max_memory_gb
        self.memory_history = []
        self.peak_memory = 0.0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.warning_threshold = max_memory_gb * 0.8  # 80% 경고
        self.critical_threshold = max_memory_gb * 0.9  # 90% 위험
        
    def get_detailed_memory_info(self) -> Dict[str, float]:
        """상세 메모리 정보 - 스트리밍 최적화"""
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
            
            # 메모리 히스토리 관리 (최근 10개만 유지)
            with self.lock:
                self.memory_history.append({
                    'timestamp': time.time(),
                    'process_memory': current_process_memory,
                    'available_memory': info['available_gb']
                })
                if len(self.memory_history) > 10:
                    self.memory_history.pop(0)
            
            return info
            
        except Exception as e:
            logger.warning(f"메모리 정보 조회 실패: {e}")
            return self._get_fallback_memory_info()
    
    def _get_fallback_memory_info(self) -> Dict[str, float]:
        """기본 메모리 정보"""
        return {
            'total_gb': 64.0,
            'available_gb': 45.0,
            'used_gb': 19.0,
            'process_rss_gb': 2.0,
            'process_vms_gb': 4.0,
            'usage_percent': 30.0,
            'available_percent': 70.0,
            'swap_total_gb': 8.0,
            'swap_used_gb': 0.5,
            'swap_percent': 6.25,
            'peak_memory_gb': 2.0
        }
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인 - 임계값 기반"""
        info = self.get_detailed_memory_info()
        available_gb = info['available_gb']
        process_gb = info['process_rss_gb']
        
        # 메모리 상태 분류
        if available_gb < 5:
            pressure_level = "critical"
        elif available_gb < 10:
            pressure_level = "high"
        elif available_gb < 15:
            pressure_level = "moderate"
        else:
            pressure_level = "low"
        
        return {
            'memory_pressure': pressure_level in ['high', 'critical'],
            'pressure_level': pressure_level,
            'available_gb': available_gb,
            'process_gb': process_gb,
            'usage_percent': info['usage_percent'],
            'recommendation': self._get_memory_recommendation(pressure_level, available_gb),
            'should_reduce_chunk_size': available_gb < 15,
            'should_force_gc': available_gb < 12,
            'should_abort': available_gb < 3
        }
    
    def _get_memory_recommendation(self, pressure_level: str, available_gb: float) -> str:
        """메모리 상태별 권장 사항"""
        if pressure_level == "critical":
            return f"위험: 가용메모리 {available_gb:.1f}GB - 즉시 가비지컬렉션 및 청킹크기 축소 필요"
        elif pressure_level == "high":
            return f"경고: 가용메모리 {available_gb:.1f}GB - 청킹크기 축소 권장"
        elif pressure_level == "moderate":
            return f"주의: 가용메모리 {available_gb:.1f}GB - 처리속도 조절"
        else:
            return f"양호: 가용메모리 {available_gb:.1f}GB"
    
    def log_memory_status(self, context: str = "", force: bool = False):
        """메모리 상태 로깅"""
        try:
            info = self.get_detailed_memory_info()
            pressure = self.check_memory_pressure()
            
            if force or pressure['pressure_level'] != 'low':
                logger.info(f"메모리 [{context}]: 가용 {info['available_gb']:.1f}GB, "
                           f"프로세스 {info['process_rss_gb']:.1f}GB, "
                           f"피크 {info['peak_memory_gb']:.1f}GB - "
                           f"{pressure['pressure_level'].upper()}")
            
            if pressure['memory_pressure']:
                logger.warning(f"메모리 압박: {pressure['recommendation']}")
                
        except Exception as e:
            logger.warning(f"메모리 상태 로깅 실패: {e}")
    
    def adaptive_gc_cleanup(self) -> float:
        """적응형 가비지 컬렉션"""
        try:
            initial_info = self.get_detailed_memory_info()
            pressure = self.check_memory_pressure()
            
            # 압박 수준에 따른 정리 강도 결정
            if pressure['pressure_level'] == 'critical':
                cleanup_rounds = 5
                sleep_time = 0.2
            elif pressure['pressure_level'] == 'high':
                cleanup_rounds = 3
                sleep_time = 0.1
            elif pressure['pressure_level'] == 'moderate':
                cleanup_rounds = 2
                sleep_time = 0.05
            else:
                cleanup_rounds = 1
                sleep_time = 0
            
            # 가비지 컬렉션 수행
            for i in range(cleanup_rounds):
                gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Windows 전용 메모리 정리
            if cleanup_rounds >= 3:
                try:
                    import ctypes
                    if hasattr(ctypes, 'windll'):
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                except Exception:
                    pass
            
            final_info = self.get_detailed_memory_info()
            memory_freed = initial_info['process_rss_gb'] - final_info['process_rss_gb']
            
            if memory_freed > 0.1:
                logger.info(f"메모리 정리: {memory_freed:.2f}GB 해제 ({cleanup_rounds}라운드)")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"적응형 메모리 정리 실패: {e}")
            return 0.0

class StreamingParquetReader:
    """대용량 데이터용 스트리밍 Parquet 리더"""
    
    def __init__(self, file_path: str, initial_chunk_size: int = 500000):
        self.file_path = file_path
        self.initial_chunk_size = initial_chunk_size
        self.current_chunk_size = initial_chunk_size
        self.parquet_file = None
        self.total_rows = 0
        self.memory_monitor = StreamingMemoryMonitor()
        self.progress_reporter = None
        
        # 적응형 청크 크기 설정
        self.min_chunk_size = 100000
        self.max_chunk_size = 2000000
        self.chunk_size_history = []
        
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        logger.info(f"스트리밍 Parquet 리더 초기화: {self.file_path}")
        
        try:
            # 파일 존재 확인
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {self.file_path}")
            
            file_size_mb = os.path.getsize(self.file_path) / (1024**2)
            logger.info(f"파일 크기: {file_size_mb:.1f}MB")
            
            # PyArrow 사용 가능하면 메타데이터 읽기
            if PYARROW_AVAILABLE:
                try:
                    self.parquet_file = pq.ParquetFile(self.file_path)
                    self.total_rows = self.parquet_file.metadata.num_rows
                    logger.info(f"PyArrow 메타데이터 읽기 성공: {self.total_rows:,}행")
                    
                except Exception as e:
                    logger.warning(f"PyArrow 메타데이터 읽기 실패: {e}")
                    self.parquet_file = None
            
            # PyArrow 실패 시 추정
            if self.total_rows == 0:
                self.total_rows = self._estimate_row_count()
            
            # 진행 상황 리포터 초기화
            self.progress_reporter = ProgressReporter(
                self.total_rows, 
                f"데이터 로딩 ({file_size_mb:.0f}MB)"
            )
            
            logger.info(f"스트리밍 리더 준비 완료: {self.total_rows:,}행 예상")
            
            return self
            
        except Exception as e:
            logger.error(f"스트리밍 리더 초기화 실패: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        try:
            if self.parquet_file:
                self.parquet_file = None
            
            if self.progress_reporter:
                self.progress_reporter.report_progress()
            
            # 최종 메모리 정리
            self.memory_monitor.adaptive_gc_cleanup()
            
            logger.info("스트리밍 리더 정리 완료")
            
        except Exception as e:
            logger.warning(f"스트리밍 리더 정리 실패: {e}")
    
    def _estimate_row_count(self) -> int:
        """행 수 추정"""
        try:
            # 작은 샘플로 추정
            sample_size = 50000
            sample_df = pd.read_parquet(self.file_path, nrows=sample_size)
            
            file_size = os.path.getsize(self.file_path)
            sample_memory = sample_df.memory_usage(deep=True).sum()
            
            # 압축률 및 오버헤드 고려
            compression_ratio = 0.6  # Parquet 압축률 추정
            overhead_factor = 1.2    # 메타데이터 등 오버헤드
            
            estimated_rows = int((file_size / sample_memory) * sample_size * compression_ratio * overhead_factor)
            
            del sample_df
            gc.collect()
            
            logger.info(f"행 수 추정 완료: {estimated_rows:,}행")
            return estimated_rows
            
        except Exception as e:
            logger.warning(f"행 수 추정 실패: {e}. 기본값 사용")
            return 10000000
    
    def read_data_streaming(self) -> pd.DataFrame:
        """스트리밍 방식으로 데이터 읽기"""
        logger.info(f"스트리밍 데이터 읽기 시작: {self.total_rows:,}행")
        
        chunks = []
        total_processed = 0
        chunk_count = 0
        start_time = time.time()
        
        try:
            # 메모리 기반 청킹 전략
            self._adjust_initial_chunk_size()
            
            while total_processed < self.total_rows:
                # 메모리 상태 확인 및 청크 크기 조정
                pressure = self.memory_monitor.check_memory_pressure()
                
                if pressure['should_abort']:
                    logger.error("메모리 부족으로 데이터 로딩 중단")
                    break
                
                # 적응형 청크 크기 조정
                self._adaptive_chunk_size_adjustment(pressure)
                
                # 청크 읽기
                remaining_rows = self.total_rows - total_processed
                current_chunk_size = min(self.current_chunk_size, remaining_rows)
                
                try:
                    chunk_start_time = time.time()
                    
                    if PYARROW_AVAILABLE and self.parquet_file:
                        chunk_df = self._read_chunk_pyarrow(total_processed, current_chunk_size)
                    else:
                        chunk_df = self._read_chunk_pandas(total_processed, current_chunk_size)
                    
                    if chunk_df is None or chunk_df.empty:
                        logger.warning(f"빈 청크 감지: {total_processed:,} 위치")
                        total_processed += current_chunk_size
                        continue
                    
                    # 메모리 최적화
                    chunk_df = self._optimize_chunk_memory(chunk_df)
                    chunks.append(chunk_df)
                    
                    chunk_time = time.time() - chunk_start_time
                    chunk_count += 1
                    total_processed += len(chunk_df)
                    
                    # 진행 상황 업데이트
                    if self.progress_reporter:
                        self.progress_reporter.update(len(chunk_df))
                    
                    # 성능 모니터링
                    if chunk_count % 5 == 0:
                        self._log_performance_metrics(chunk_count, total_processed, start_time)
                    
                    # 주기적 메모리 정리
                    if pressure['should_force_gc'] or chunk_count % 10 == 0:
                        self.memory_monitor.adaptive_gc_cleanup()
                    
                    # 청크 크기 히스토리 업데이트
                    self.chunk_size_history.append({
                        'chunk_size': len(chunk_df),
                        'processing_time': chunk_time,
                        'memory_pressure': pressure['pressure_level']
                    })
                    
                    # 히스토리 크기 제한
                    if len(self.chunk_size_history) > 20:
                        self.chunk_size_history.pop(0)
                    
                except Exception as e:
                    logger.error(f"청크 {chunk_count + 1} 읽기 실패 (위치: {total_processed:,}): {e}")
                    
                    # 에러 복구 시도
                    if self._attempt_error_recovery(total_processed, current_chunk_size):
                        continue
                    else:
                        break
            
            # 모든 청크 결합
            if chunks:
                logger.info(f"청크 결합 시작: {len(chunks)}개 청크")
                combined_df = self._combine_chunks_efficiently(chunks)
                
                total_time = time.time() - start_time
                throughput = len(combined_df) / total_time if total_time > 0 else 0
                
                logger.info(f"스트리밍 로딩 완료: {combined_df.shape}, "
                           f"{total_time:.2f}초, {throughput:,.0f}행/초")
                
                return combined_df
            else:
                logger.error("읽기 가능한 청크가 없습니다")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"스트리밍 데이터 읽기 실패: {e}")
            # 부분 데이터라도 반환 시도
            if chunks:
                try:
                    partial_df = pd.concat(chunks, ignore_index=True)
                    logger.warning(f"부분 데이터 반환: {partial_df.shape}")
                    return partial_df
                except Exception:
                    pass
            
            return pd.DataFrame()
    
    def _adjust_initial_chunk_size(self):
        """초기 청크 크기 메모리 기반 조정"""
        try:
            memory_info = self.memory_monitor.get_detailed_memory_info()
            available_gb = memory_info['available_gb']
            
            # 가용 메모리에 따른 청크 크기 조정
            if available_gb > 30:
                self.current_chunk_size = min(1000000, self.max_chunk_size)
            elif available_gb > 20:
                self.current_chunk_size = 750000
            elif available_gb > 15:
                self.current_chunk_size = 500000
            elif available_gb > 10:
                self.current_chunk_size = 300000
            else:
                self.current_chunk_size = self.min_chunk_size
            
            logger.info(f"초기 청크 크기 조정: {self.current_chunk_size:,}행 (가용메모리: {available_gb:.1f}GB)")
            
        except Exception as e:
            logger.warning(f"초기 청크 크기 조정 실패: {e}")
    
    def _adaptive_chunk_size_adjustment(self, pressure: Dict[str, Any]):
        """적응형 청크 크기 조정"""
        try:
            old_chunk_size = self.current_chunk_size
            
            if pressure['pressure_level'] == 'critical':
                self.current_chunk_size = max(self.min_chunk_size, self.current_chunk_size // 4)
            elif pressure['pressure_level'] == 'high':
                self.current_chunk_size = max(self.min_chunk_size, self.current_chunk_size // 2)
            elif pressure['pressure_level'] == 'moderate':
                self.current_chunk_size = max(self.min_chunk_size, int(self.current_chunk_size * 0.8))
            elif pressure['pressure_level'] == 'low' and len(self.chunk_size_history) > 5:
                # 성능 기반 증가
                recent_performance = self.chunk_size_history[-5:]
                avg_time_per_row = sum(h['processing_time'] / h['chunk_size'] for h in recent_performance) / len(recent_performance)
                
                if avg_time_per_row < 1e-6:  # 매우 빠르면 증가
                    self.current_chunk_size = min(self.max_chunk_size, int(self.current_chunk_size * 1.2))
            
            if old_chunk_size != self.current_chunk_size:
                logger.debug(f"청크 크기 조정: {old_chunk_size:,} → {self.current_chunk_size:,} "
                            f"(메모리: {pressure['pressure_level']})")
                            
        except Exception as e:
            logger.warning(f"청크 크기 조정 실패: {e}")
    
    def _read_chunk_pyarrow(self, start_row: int, num_rows: int) -> Optional[pd.DataFrame]:
        """PyArrow로 청크 읽기"""
        try:
            if not self.parquet_file:
                return None
            
            # 행 범위 계산
            end_row = min(start_row + num_rows, self.total_rows)
            actual_num_rows = end_row - start_row
            
            if actual_num_rows <= 0:
                return None
            
            # Row group 기반 읽기
            metadata = self.parquet_file.metadata
            target_row_groups = self._get_target_row_groups(start_row, actual_num_rows, metadata)
            
            if not target_row_groups:
                return None
            
            # Row group들 읽기
            tables = []
            for rg_idx, rg_start, rg_end in target_row_groups:
                try:
                    table = self.parquet_file.read_row_group(rg_idx, use_threads=True)
                    
                    # 필요한 행만 슬라이싱
                    if rg_start > 0 or rg_end < table.num_rows:
                        table = table.slice(rg_start, rg_end - rg_start)
                    
                    tables.append(table)
                    
                except Exception as e:
                    logger.warning(f"Row group {rg_idx} 읽기 실패: {e}")
                    continue
            
            if not tables:
                return None
            
            # 테이블 결합 및 pandas 변환
            if len(tables) == 1:
                combined_table = tables[0]
            else:
                combined_table = pa.concat_tables(tables)
            
            df = combined_table.to_pandas()
            
            # 행 수 검증
            if len(df) != actual_num_rows:
                logger.warning(f"예상 행 수와 실제 행 수 불일치: {len(df)} != {actual_num_rows}")
            
            return df
            
        except Exception as e:
            logger.warning(f"PyArrow 청크 읽기 실패: {e}")
            return None
    
    def _get_target_row_groups(self, start_row: int, num_rows: int, metadata) -> List[Tuple[int, int, int]]:
        """대상 row group 계산"""
        try:
            target_row_groups = []
            current_row = 0
            end_row = start_row + num_rows
            
            for rg_idx in range(metadata.num_row_groups):
                rg_num_rows = metadata.row_group(rg_idx).num_rows
                rg_start = current_row
                rg_end = current_row + rg_num_rows
                
                # 겹치는 구간 확인
                if rg_start < end_row and rg_end > start_row:
                    # 실제 읽을 구간 계산
                    read_start = max(0, start_row - rg_start)
                    read_end = min(rg_num_rows, end_row - rg_start)
                    
                    if read_end > read_start:
                        target_row_groups.append((rg_idx, read_start, read_end))
                
                current_row = rg_end
                
                if current_row >= end_row:
                    break
            
            return target_row_groups
            
        except Exception as e:
            logger.warning(f"Row group 계산 실패: {e}")
            return []
    
    def _read_chunk_pandas(self, start_row: int, num_rows: int) -> Optional[pd.DataFrame]:
        """pandas로 청크 읽기 - 대안 방법"""
        try:
            # 전체 데이터를 읽고 슬라이싱 (비효율적이지만 안전)
            df = pd.read_parquet(self.file_path, engine='pyarrow' if PYARROW_AVAILABLE else 'fastparquet')
            
            end_row = min(start_row + num_rows, len(df))
            if start_row >= len(df):
                return None
            
            chunk_df = df.iloc[start_row:end_row].copy()
            del df
            gc.collect()
            
            return chunk_df
            
        except Exception as e:
            logger.warning(f"pandas 청크 읽기 실패: {e}")
            return None
    
    def _optimize_chunk_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """청크 메모리 최적화"""
        if df.empty:
            return df
        
        try:
            # 빠른 타입 최적화
            for col in df.select_dtypes(include=['int64']).columns:
                try:
                    col_min, col_max = df[col].min(), df[col].max()
                    if pd.isna(col_min) or pd.isna(col_max):
                        continue
                    
                    if col_min >= 0 and col_max < 65535:
                        df[col] = df[col].astype('uint16')
                    elif col_min > -32768 and col_max < 32767:
                        df[col] = df[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df[col] = df[col].astype('int32')
                except Exception:
                    continue
            
            for col in df.select_dtypes(include=['float64']).columns:
                try:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                except Exception:
                    continue
            
            # 범주형 최적화 (선택적)
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:
                        df[col] = df[col].astype('category')
                except Exception:
                    continue
            
            return df
            
        except Exception as e:
            logger.warning(f"청크 메모리 최적화 실패: {e}")
            return df
    
    def _log_performance_metrics(self, chunk_count: int, total_processed: int, start_time: float):
        """성능 메트릭 로깅"""
        try:
            elapsed_time = time.time() - start_time
            throughput = total_processed / elapsed_time if elapsed_time > 0 else 0
            progress_percent = (total_processed / self.total_rows) * 100 if self.total_rows > 0 else 0
            
            self.memory_monitor.log_memory_status(f"청크{chunk_count}")
            
            logger.info(f"처리 진행률: {progress_percent:.1f}% - "
                       f"{total_processed:,}/{self.total_rows:,}행 - "
                       f"속도: {throughput:,.0f}행/초")
                       
        except Exception as e:
            logger.warning(f"성능 메트릭 로깅 실패: {e}")
    
    def _attempt_error_recovery(self, failed_position: int, failed_chunk_size: int) -> bool:
        """에러 복구 시도"""
        try:
            logger.info(f"에러 복구 시도: 위치 {failed_position:,}, 크기 {failed_chunk_size:,}")
            
            # 청크 크기를 절반으로 줄여 재시도
            recovery_chunk_size = max(self.min_chunk_size, failed_chunk_size // 2)
            
            try:
                if PYARROW_AVAILABLE and self.parquet_file:
                    recovery_df = self._read_chunk_pyarrow(failed_position, recovery_chunk_size)
                else:
                    recovery_df = self._read_chunk_pandas(failed_position, recovery_chunk_size)
                
                if recovery_df is not None and not recovery_df.empty:
                    logger.info(f"에러 복구 성공: {len(recovery_df):,}행 읽기")
                    
                    # 청크 크기 감소
                    self.current_chunk_size = recovery_chunk_size
                    return True
                
            except Exception as e:
                logger.warning(f"에러 복구 실패: {e}")
            
            return False
            
        except Exception as e:
            logger.error(f"에러 복구 시도 중 예외: {e}")
            return False
    
    def _combine_chunks_efficiently(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """효율적인 청크 결합"""
        if not chunks:
            return pd.DataFrame()
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            logger.info(f"청크 결합 시작: {len(chunks)}개")
            
            # 메모리 효율적인 결합
            combined_chunks = []
            batch_size = 10  # 10개씩 배치로 결합
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_df = pd.concat(batch_chunks, ignore_index=True)
                combined_chunks.append(batch_df)
                
                # 중간 정리
                for chunk in batch_chunks:
                    del chunk
                gc.collect()
                
                logger.debug(f"배치 {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} 완료")
            
            # 최종 결합
            if len(combined_chunks) == 1:
                result = combined_chunks[0]
            else:
                result = pd.concat(combined_chunks, ignore_index=True)
                
                # 정리
                for chunk in combined_chunks:
                    del chunk
                gc.collect()
            
            logger.info(f"청크 결합 완료: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"청크 결합 실패: {e}")
            # 단순 결합 시도
            try:
                return pd.concat(chunks, ignore_index=True)
            except Exception as e2:
                logger.error(f"단순 결합도 실패: {e2}")
                return pd.DataFrame()

class LargeDataLoader:
    """대용량 데이터 처리 특화 데이터 로더 - 스트리밍 최적화"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = StreamingMemoryMonitor(config.MAX_MEMORY_GB)
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
        
        logger.info("대용량 데이터 로더 초기화 완료 (스트리밍 최적화)")
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """대용량 데이터 최적화 로딩 - 스트리밍 강화"""
        logger.info("=== 대용량 데이터 스트리밍 로딩 시작 ===")
        
        try:
            # 1. 초기 메모리 상태 확인
            self.memory_monitor.log_memory_status("로딩 시작", force=True)
            
            # 2. 데이터 파일 검증
            if not self._validate_data_files():
                if self.config.REQUIRE_REAL_DATA and not self.config.SAMPLE_DATA_FALLBACK:
                    raise ValueError("실제 대용량 데이터 파일이 필요하지만 검증에 실패했습니다")
                else:
                    logger.warning("실제 데이터 파일 검증 실패, 샘플 데이터로 대체")
                    return self._create_enhanced_sample_data()
            
            # 3. 스트리밍 방식으로 학습 데이터 로딩
            train_df = self._load_train_data_streaming()
            
            if train_df is None or train_df.empty:
                raise ValueError("학습 데이터 로딩 실패")
            
            self.memory_monitor.log_memory_status("학습 데이터 로딩 후", force=True)
            
            # 4. 스트리밍 방식으로 테스트 데이터 로딩
            test_df = self._load_test_data_streaming()
            
            if test_df is None or test_df.empty:
                raise ValueError("테스트 데이터 로딩 실패")
            
            self.memory_monitor.log_memory_status("테스트 데이터 로딩 후", force=True)
            
            # 5. 데이터 검증
            self._validate_loaded_data(train_df, test_df)
            
            # 6. 피처 컬럼 정의
            self.feature_columns = [col for col in train_df.columns 
                                  if col != self.target_column]
            
            # 7. 성능 통계 업데이트
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
            self.memory_monitor.adaptive_gc_cleanup()
            raise
    
    def _validate_data_files(self) -> bool:
        """데이터 파일 검증"""
        try:
            # 학습 데이터 파일 확인
            if not self.config.TRAIN_PATH.exists():
                logger.error(f"학습 데이터 파일이 없습니다: {self.config.TRAIN_PATH}")
                return False
            
            # 테스트 데이터 파일 확인
            if not self.config.TEST_PATH.exists():
                logger.error(f"테스트 데이터 파일이 없습니다: {self.config.TEST_PATH}")
                return False
            
            # 파일 크기 확인
            train_size_mb = self.config.TRAIN_PATH.stat().st_size / (1024**2)
            test_size_mb = self.config.TEST_PATH.stat().st_size / (1024**2)
            
            logger.info(f"파일 크기 확인 - 학습: {train_size_mb:.1f}MB, 테스트: {test_size_mb:.1f}MB")
            
            if train_size_mb < 100:  # 최소 100MB
                logger.warning(f"학습 데이터 파일 크기가 작습니다: {train_size_mb:.1f}MB")
                return False
            
            if test_size_mb < 50:  # 최소 50MB
                logger.warning(f"테스트 데이터 파일 크기가 작습니다: {test_size_mb:.1f}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"데이터 파일 검증 실패: {e}")
            return False
    
    def _load_train_data_streaming(self) -> pd.DataFrame:
        """스트리밍 방식으로 학습 데이터 로딩"""
        logger.info("스트리밍 학습 데이터 로딩 시작")
        
        try:
            # 메모리 상태에 따른 초기 청크 크기 결정
            memory_info = self.memory_monitor.get_detailed_memory_info()
            available_gb = memory_info['available_gb']
            
            if available_gb > 30:
                initial_chunk_size = 1000000
            elif available_gb > 20:
                initial_chunk_size = 750000
            elif available_gb > 15:
                initial_chunk_size = 500000
            else:
                initial_chunk_size = 300000
            
            logger.info(f"초기 청크 크기: {initial_chunk_size:,}행 (가용메모리: {available_gb:.1f}GB)")
            
            # 스트리밍 리더로 데이터 로딩
            with StreamingParquetReader(str(self.config.TRAIN_PATH), initial_chunk_size) as reader:
                df = reader.read_data_streaming()
                
                if df.empty:
                    raise ValueError("스트리밍 로딩 결과가 비어있습니다")
                
                # 타겟 컬럼 확인
                if self.target_column not in df.columns:
                    possible_targets = [col for col in df.columns if 'click' in col.lower()]
                    if possible_targets:
                        self.target_column = possible_targets[0]
                        logger.info(f"타겟 컬럼 변경: {self.target_column}")
                    else:
                        raise ValueError(f"타겟 컬럼 '{self.target_column}'을 찾을 수 없습니다")
                
                # 기본 통계 확인
                target_ctr = df[self.target_column].mean()
                logger.info(f"스트리밍 학습 데이터 로딩 완료: {df.shape}, CTR: {target_ctr:.4f}")
                
                return df
                
        except Exception as e:
            logger.error(f"스트리밍 학습 데이터 로딩 실패: {e}")
            raise
    
    def _load_test_data_streaming(self) -> pd.DataFrame:
        """스트리밍 방식으로 테스트 데이터 로딩"""
        logger.info("스트리밍 테스트 데이터 로딩 시작")
        
        try:
            # 메모리 압박을 고려한 청크 크기
            pressure = self.memory_monitor.check_memory_pressure()
            
            if pressure['pressure_level'] in ['critical', 'high']:
                initial_chunk_size = 300000
            elif pressure['pressure_level'] == 'moderate':
                initial_chunk_size = 500000
            else:
                initial_chunk_size = 750000
            
            logger.info(f"테스트 데이터 청크 크기: {initial_chunk_size:,}행 "
                       f"(메모리 상태: {pressure['pressure_level']})")
            
            # 스트리밍 리더로 데이터 로딩
            with StreamingParquetReader(str(self.config.TEST_PATH), initial_chunk_size) as reader:
                df = reader.read_data_streaming()
                
                if df.empty:
                    raise ValueError("테스트 데이터 스트리밍 로딩 결과가 비어있습니다")
                
                logger.info(f"스트리밍 테스트 데이터 로딩 완료: {df.shape}")
                
                # 최소 크기 검증
                if len(df) < self.config.MIN_TEST_SIZE * 0.8:
                    logger.warning(f"테스트 데이터 크기가 예상보다 작습니다: {len(df):,}")
                
                return df
                
        except Exception as e:
            logger.error(f"스트리밍 테스트 데이터 로딩 실패: {e}")
            raise
    
    def _validate_loaded_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """로딩된 데이터 검증"""
        logger.info("로딩된 데이터 검증 시작")
        
        try:
            # 기본 형태 검증
            if train_df.empty or test_df.empty:
                raise ValueError("로딩된 데이터가 비어있습니다")
            
            # 최소 크기 검증 (관대한 기준)
            min_train_threshold = max(100000, self.config.MIN_TRAIN_SIZE * 0.1)
            min_test_threshold = max(50000, self.config.MIN_TEST_SIZE * 0.8)
            
            if len(train_df) < min_train_threshold:
                raise ValueError(f"학습 데이터 크기 부족: {len(train_df):,} < {min_train_threshold:,}")
            
            if len(test_df) < min_test_threshold:
                raise ValueError(f"테스트 데이터 크기 부족: {len(test_df):,} < {min_test_threshold:,}")
            
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
            
            common_features = train_features & test_features
            missing_features = train_features - test_features
            extra_features = test_features - train_features
            
            logger.info(f"공통 피처: {len(common_features)}개")
            
            if missing_features:
                logger.warning(f"테스트 데이터에 누락된 피처: {len(missing_features)}개")
            
            if extra_features:
                logger.warning(f"테스트 데이터의 추가 피처: {len(extra_features)}개")
            
            logger.info("로딩된 데이터 검증 완료")
            
        except Exception as e:
            logger.error(f"데이터 검증 실패: {e}")
            raise
    
    def _create_enhanced_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """향상된 샘플 데이터 생성 (실제 데이터 없을 때만)"""
        logger.warning("실제 데이터가 없어 향상된 샘플 데이터를 생성합니다")
        
        try:
            # 더 큰 샘플 데이터 생성
            np.random.seed(42)
            n_train = 2000000  # 200만행
            n_test = 1527298   # 실제 테스트 크기
            
            # 실제 CTR 패턴 반영
            train_data = {
                self.target_column: np.random.binomial(1, 0.0201, n_train),
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
                if col != self.target_column:
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
            train_df = self._optimize_sample_data_memory(train_df)
            test_df = self._optimize_sample_data_memory(test_df)
            
            logger.warning(f"향상된 샘플 데이터 생성 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")
            raise
    
    def _optimize_sample_data_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """샘플 데이터 메모리 최적화"""
        try:
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    if df[col].min() >= 0 and df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                    elif df[col].min() > -32768 and df[col].max() < 32767:
                        df[col] = df[col].astype('int16')
                    else:
                        df[col] = df[col].astype('int32')
            
            return df
            
        except Exception as e:
            logger.warning(f"샘플 데이터 최적화 실패: {e}")
            return df
    
    def _log_loading_completion(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """로딩 완료 로깅"""
        try:
            stats = self.loading_stats
            memory_info = self.memory_monitor.get_detailed_memory_info()
            
            logger.info("=== 대용량 데이터 스트리밍 로딩 완료 ===")
            logger.info(f"학습 데이터: {train_df.shape}")
            logger.info(f"테스트 데이터: {test_df.shape}")
            logger.info(f"로딩 시간: {stats['loading_time']:.2f}초")
            logger.info(f"메모리 사용량: {memory_info['process_rss_gb']:.2f}GB")
            logger.info(f"피크 메모리: {memory_info['peak_memory_gb']:.2f}GB")
            
            # 처리 속도 계산
            total_rows = len(train_df) + len(test_df)
            if stats['loading_time'] > 0:
                throughput = total_rows / stats['loading_time']
                logger.info(f"전체 처리 속도: {throughput:,.0f}행/초")
            
            # 요구사항 충족 확인
            if train_df.shape[0] >= self.config.MIN_TRAIN_SIZE * 0.1:
                logger.info("✓ 학습 데이터 최소 요구사항 충족")
            else:
                logger.warning("△ 학습 데이터 크기 부족")
            
            if test_df.shape[0] >= self.config.MIN_TEST_SIZE * 0.8:
                logger.info("✓ 테스트 데이터 최소 요구사항 충족")
            else:
                logger.warning("△ 테스트 데이터 크기 부족")
            
            logger.info("=== 스트리밍 로딩 완료 ===")
            
        except Exception as e:
            logger.warning(f"완료 로깅 실패: {e}")
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """로딩 통계 반환"""
        return self.loading_stats.copy()

# 기존 클래스들과의 호환성을 위한 별칭
DataLoader = LargeDataLoader
MemoryMonitor = StreamingMemoryMonitor
ChunkedParquetReader = StreamingParquetReader

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
            except Exception:
                continue
        
        validation_results['dtype_mismatches'] = dtype_mismatches
        
        return validation_results

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 설정 확인
    config = Config()
    
    # 데이터 로더 테스트
    try:
        loader = LargeDataLoader(config)
        train_df, test_df = loader.load_large_data_optimized()
        
        print(f"학습 데이터: {train_df.shape}")
        print(f"테스트 데이터: {test_df.shape}")
        print(f"로딩 통계: {loader.get_loading_stats()}")
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")