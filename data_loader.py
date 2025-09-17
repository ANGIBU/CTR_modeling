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
        self.report_interval = 5.0
        
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
            
            items_per_sec = self.current_items / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"{self.description}: {progress_percent:.1f}% "
                       f"({self.current_items:,}/{self.total_items:,}) - "
                       f"속도: {items_per_sec:,.0f}행/초 - "
                       f"남은 시간: {remaining_time:.0f}초")
        else:
            logger.info(f"{self.description}: {progress_percent:.1f}% ({self.current_items:,}/{self.total_items:,})")

class MemoryMonitor:
    """개선된 메모리 모니터링 및 관리"""
    
    def __init__(self, max_memory_gb: float = 45.0):  # 30GB → 45GB로 증가
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.max_memory_gb = max_memory_gb
        self.memory_history = []
        self.peak_memory = 0.0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # 메모리 임계값 설정 (완화)
        self.warning_threshold = max_memory_gb * 0.75      # 30GB → 33.75GB
        self.critical_threshold = max_memory_gb * 0.85     # 38.25GB
        self.abort_threshold = max_memory_gb * 0.95        # 42.75GB (기존 28.5GB → 42.75GB)
        
        logger.info(f"메모리 임계값 설정 - 경고: {self.warning_threshold:.1f}GB, "
                   f"위험: {self.critical_threshold:.1f}GB, 중단: {self.abort_threshold:.1f}GB")
        
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
        """개선된 메모리 압박 상태 확인"""
        process_memory = self.get_process_memory_gb()
        available_memory = self.get_available_memory_gb()
        
        # 압박 수준 결정 (완화된 기준)
        if process_memory > self.abort_threshold or available_memory < 2:
            pressure_level = "abort"
        elif process_memory > self.critical_threshold or available_memory < 5:
            pressure_level = "critical" 
        elif process_memory > self.warning_threshold or available_memory < 8:
            pressure_level = "high"
        elif available_memory < 15:
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
            'should_abort': pressure_level == 'abort'
        }
    
    def log_memory_status(self, context: str = "", force: bool = False):
        """메모리 상태 로깅"""
        try:
            pressure = self.check_memory_pressure()
            
            if force or pressure['pressure_level'] != 'low':
                logger.info(f"메모리 [{context}]: 프로세스 {pressure['process_memory_gb']:.1f}GB, "
                           f"가용 {pressure['available_memory_gb']:.1f}GB - "
                           f"{pressure['pressure_level'].upper()}")
            
            if pressure['memory_pressure']:
                logger.warning(f"메모리 압박: 프로세스 {pressure['process_memory_gb']:.1f}GB, "
                             f"가용 {pressure['available_memory_gb']:.1f}GB")
                
        except Exception as e:
            logger.warning(f"메모리 상태 로깅 실패: {e}")
    
    def force_memory_cleanup(self, intensive: bool = False) -> float:
        """강화된 메모리 정리"""
        try:
            initial_memory = self.get_process_memory_gb()
            pressure = self.check_memory_pressure()
            
            # 압박 수준에 따른 정리 강도
            if pressure['pressure_level'] == 'abort' or intensive:
                cleanup_rounds = 15  # 10 → 15로 증가
                sleep_time = 0.5     # 0.3 → 0.5로 증가
            elif pressure['pressure_level'] == 'critical':
                cleanup_rounds = 12  # 7 → 12로 증가
                sleep_time = 0.3     # 0.2 → 0.3으로 증가
            elif pressure['pressure_level'] == 'high':
                cleanup_rounds = 8   # 5 → 8로 증가
                sleep_time = 0.2     # 0.1 → 0.2로 증가
            else:
                cleanup_rounds = 5   # 3 → 5로 증가
                sleep_time = 0.1     # 0.05 → 0.1로 증가
            
            # 강화된 가비지 컬렉션
            for i in range(cleanup_rounds):
                collected = gc.collect()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 중간 체크
                if i % 5 == 0:
                    current_memory = self.get_process_memory_gb()
                    if initial_memory - current_memory > 2.0:  # 2GB 이상 해제되면 조기 종료
                        break
            
            # Windows 메모리 정리 강화
            if cleanup_rounds >= 8:
                try:
                    import ctypes
                    if hasattr(ctypes, 'windll'):
                        # 더 강력한 메모리 정리
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                        time.sleep(0.5)
                        gc.collect()
                except Exception:
                    pass
            
            final_memory = self.get_process_memory_gb()
            memory_freed = initial_memory - final_memory
            
            if memory_freed > 0.1:
                logger.info(f"메모리 정리: {memory_freed:.2f}GB 해제 ({cleanup_rounds}라운드)")
            
            return memory_freed
            
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")
            return 0.0

class DataChunkProcessor:
    """개선된 데이터 청크 처리기"""
    
    def __init__(self, file_path: str, chunk_size: int = 25000):  # 100000 → 25000으로 축소
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.total_rows = 0
        self.memory_monitor = MemoryMonitor()
        self.progress_reporter = None
        
        # 청크 크기 범위 조정
        self.min_chunk_size = 5000   # 10000 → 5000으로 축소
        self.max_chunk_size = 100000 # 500000 → 100000으로 축소
        
    def __enter__(self):
        """초기화"""
        logger.info(f"데이터 청크 처리기 초기화: {self.file_path}")
        
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {self.file_path}")
            
            file_size_mb = os.path.getsize(self.file_path) / (1024**2)
            logger.info(f"파일 크기: {file_size_mb:.1f}MB")
            
            # 총 행 수 확인
            self.total_rows = self._estimate_total_rows()
            
            # 진행 상황 리포터 초기화
            self.progress_reporter = ProgressReporter(
                self.total_rows, 
                f"데이터 로딩 ({file_size_mb:.0f}MB)"
            )
            
            logger.info(f"처리 준비 완료: {self.total_rows:,}행")
            return self
            
        except Exception as e:
            logger.error(f"청크 처리기 초기화 실패: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """정리"""
        try:
            if self.progress_reporter:
                self.progress_reporter.report_progress()
            
            # 메모리 정리
            self.memory_monitor.force_memory_cleanup()
            
            logger.info("청크 처리기 정리 완료")
            
        except Exception as e:
            logger.warning(f"청크 처리기 정리 실패: {e}")
    
    def _estimate_total_rows(self) -> int:
        """총 행 수 추정"""
        try:
            if PYARROW_AVAILABLE:
                try:
                    parquet_file = pq.ParquetFile(self.file_path)
                    total_rows = parquet_file.metadata.num_rows
                    logger.info(f"PyArrow 메타데이터: {total_rows:,}행")
                    return total_rows
                except Exception as e:
                    logger.warning(f"PyArrow 메타데이터 실패: {e}")
            
            # pandas를 이용한 추정
            logger.info("행 수 추정 시작")
            sample_df = pd.read_parquet(self.file_path, nrows=5000)
            file_size = os.path.getsize(self.file_path)
            sample_memory = sample_df.memory_usage(deep=True).sum()
            
            estimated_rows = int((file_size / sample_memory) * 5000 * 0.7)
            
            del sample_df
            gc.collect()
            
            logger.info(f"행 수 추정 완료: {estimated_rows:,}행")
            return estimated_rows
            
        except Exception as e:
            logger.warning(f"행 수 확인 실패: {e}. 기본값 사용")
            return 1000000
    
    def process_in_chunks(self) -> pd.DataFrame:
        """개선된 청크별 데이터 처리"""
        logger.info(f"청크별 데이터 처리 시작: {self.total_rows:,}행")
        
        all_chunks = []
        processed_rows = 0
        chunk_number = 0
        
        try:
            # 초기 메모리 상태 확인
            self.memory_monitor.log_memory_status("처리 시작", force=True)
            
            while processed_rows < self.total_rows:
                # 메모리 압박 확인
                pressure = self.memory_monitor.check_memory_pressure()
                
                if pressure['should_abort']:
                    logger.warning(f"메모리 한계 접근. 처리 중단: {processed_rows:,}행")
                    # ABORT 상태에서도 중간 데이터가 있으면 계속 진행
                    if len(all_chunks) > 0:
                        logger.info("중간 데이터가 있어 최종 결합을 시도합니다")
                        break
                    else:
                        logger.error(f"메모리 한계로 중단. 처리된 데이터: {processed_rows:,}행")
                        break
                
                # 청크 크기 조정
                self._adjust_chunk_size(pressure)
                
                # 청크 읽기
                remaining_rows = self.total_rows - processed_rows
                current_chunk_size = min(self.chunk_size, remaining_rows)
                
                try:
                    # 단일 청크 처리
                    chunk_df = self._read_chunk_safe(processed_rows, current_chunk_size)
                    
                    if chunk_df is None or len(chunk_df) == 0:
                        logger.warning(f"빈 청크: {processed_rows:,} 위치")
                        processed_rows += current_chunk_size
                        continue
                    
                    # 데이터 타입 최적화
                    chunk_df = self._optimize_chunk_memory(chunk_df)
                    
                    # 청크 저장
                    all_chunks.append(chunk_df)
                    chunk_number += 1
                    processed_rows += len(chunk_df)
                    
                    # 진행 상황 업데이트
                    if self.progress_reporter:
                        self.progress_reporter.update(len(chunk_df))
                    
                    logger.info(f"청크 {chunk_number} 처리 완료: {len(chunk_df):,}행 "
                               f"(메모리: {self.memory_monitor.get_process_memory_gb():.1f}GB)")
                    
                    # 더 빈번한 메모리 정리
                    if chunk_number % 2 == 0:  # 3 → 2로 변경
                        self.memory_monitor.force_memory_cleanup()
                        self.memory_monitor.log_memory_status(f"청크{chunk_number}")
                    
                    # 중간 결합 (메모리 압박 시 더 자주)
                    combine_threshold = 3 if pressure['pressure_level'] in ['high', 'critical'] else 5
                    if len(all_chunks) >= combine_threshold:
                        logger.info(f"중간 결합 수행: {len(all_chunks)}개 청크")
                        combined_df = self._combine_chunks_safe(all_chunks)
                        all_chunks = [combined_df] if not combined_df.empty else []
                        self.memory_monitor.force_memory_cleanup(intensive=True)
                    
                except Exception as e:
                    logger.error(f"청크 {chunk_number + 1} 처리 실패: {e}")
                    
                    # 에러 복구 시도
                    if self._try_recover_chunk(processed_rows, current_chunk_size):
                        continue
                    else:
                        logger.error("복구 실패. 처리 중단")
                        break
            
            # 최종 데이터 결합
            if all_chunks:
                logger.info(f"최종 데이터 결합: {len(all_chunks)}개 청크")
                final_df = self._combine_chunks_safe(all_chunks)
                
                if final_df is not None and not final_df.empty:
                    logger.info(f"청크 처리 완료: {final_df.shape}")
                    return final_df
                else:
                    raise ValueError("최종 데이터 결합 결과가 비어있습니다")
            else:
                raise ValueError("처리된 청크가 없습니다")
                
        except Exception as e:
            logger.error(f"청크 처리 실패: {e}")
            raise
    
    def _adjust_chunk_size(self, pressure: Dict[str, Any]):
        """청크 크기 조정"""
        old_size = self.chunk_size
        
        if pressure['pressure_level'] == 'abort':
            self.chunk_size = self.min_chunk_size
        elif pressure['pressure_level'] == 'critical':
            self.chunk_size = max(self.min_chunk_size, self.chunk_size // 5)  # 4 → 5로 변경
        elif pressure['pressure_level'] == 'high':
            self.chunk_size = max(self.min_chunk_size, self.chunk_size // 3)  # 2 → 3으로 변경
        elif pressure['pressure_level'] == 'moderate':
            self.chunk_size = max(self.min_chunk_size, int(self.chunk_size * 0.7))  # 0.8 → 0.7로 변경
        elif pressure['pressure_level'] == 'low':
            self.chunk_size = min(self.max_chunk_size, int(self.chunk_size * 1.2))  # 1.1 → 1.2로 변경
        
        if old_size != self.chunk_size:
            logger.info(f"청크 크기 조정: {old_size:,} → {self.chunk_size:,}")
    
    def _read_chunk_safe(self, start_row: int, num_rows: int) -> Optional[pd.DataFrame]:
        """개선된 안전한 청크 읽기"""
        try:
            # PyArrow 청크 읽기 개선
            if PYARROW_AVAILABLE:
                try:
                    # 더 작은 배치로 읽기
                    batch_size = min(num_rows, 10000)  # 최대 1만행씩 읽기
                    batches = []
                    
                    for i in range(0, num_rows, batch_size):
                        current_batch_size = min(batch_size, num_rows - i)
                        current_start = start_row + i
                        
                        try:
                            # pandas skiprows/nrows 방식 사용
                            batch_df = pd.read_parquet(
                                self.file_path,
                                engine='pyarrow'
                            )
                            
                            # 요청된 범위 추출
                            end_row = min(current_start + current_batch_size, len(batch_df))
                            if current_start >= len(batch_df):
                                break
                            
                            batch_chunk = batch_df.iloc[current_start:end_row].copy()
                            batches.append(batch_chunk)
                            
                            # 메모리 해제
                            del batch_df
                            gc.collect()
                            
                            # 메모리 압박 시 조기 종료
                            if self.memory_monitor.check_memory_pressure()['should_abort']:
                                logger.warning("메모리 압박으로 배치 읽기 중단")
                                break
                                
                        except Exception as e:
                            logger.warning(f"배치 {i} 읽기 실패: {e}")
                            break
                    
                    if batches:
                        result_df = pd.concat(batches, ignore_index=True)
                        for batch in batches:
                            del batch
                        del batches
                        gc.collect()
                        return result_df
                    else:
                        return None
                        
                except Exception as e:
                    logger.warning(f"PyArrow 개선된 청크 읽기 실패: {e}")
            
            # 기본 pandas 방식
            try:
                df = pd.read_parquet(self.file_path, engine='pyarrow')
                
                end_row = min(start_row + num_rows, len(df))
                if start_row >= len(df):
                    return None
                
                result_df = df.iloc[start_row:end_row].copy()
                
                # 메모리 해제
                del df
                gc.collect()
                
                return result_df
                
            except Exception as e:
                logger.error(f"pandas 청크 읽기 실패: {e}")
                return None
            
        except Exception as e:
            logger.error(f"청크 읽기 실패 (위치: {start_row:,}, 크기: {num_rows:,}): {e}")
            return None
    
    def _optimize_chunk_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """청크 메모리 최적화"""
        if df is None or df.empty:
            return df
        
        try:
            # 정수형 최적화
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
            
            # 범주형 최적화 (더 보수적)
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    # 더 엄격한 기준
                    if unique_count < total_count * 0.3 and unique_count < 50:
                        df[col] = df[col].astype('category')
                except Exception:
                    pass
            
            return df
            
        except Exception as e:
            logger.warning(f"청크 최적화 실패: {e}")
            return df
    
    def _combine_chunks_safe(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """안전한 청크 결합 - 단일 DataFrame 반환"""
        if not chunks:
            return pd.DataFrame()
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # 유효한 DataFrame만 필터링
            valid_chunks = []
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                logger.warning("유효한 청크가 없습니다")
                return pd.DataFrame()
            
            # 더 작은 배치별 결합 (3 → 2로 변경)
            batch_size = 2
            combined_chunks = []
            
            for i in range(0, len(valid_chunks), batch_size):
                try:
                    batch = valid_chunks[i:i + batch_size]
                    
                    if len(batch) == 1:
                        combined_chunks.append(batch[0])
                    else:
                        # 배치 결합
                        batch_combined = pd.concat(batch, ignore_index=True)
                        batch_combined = self._optimize_chunk_memory(batch_combined)
                        combined_chunks.append(batch_combined)
                    
                    # 원본 청크 해제
                    for chunk in batch:
                        del chunk
                    
                    gc.collect()
                    
                    # 메모리 압박 시 중간 정리
                    if self.memory_monitor.check_memory_pressure()['should_force_gc']:
                        self.memory_monitor.force_memory_cleanup()
                    
                except Exception as e:
                    logger.warning(f"배치 결합 실패: {e}")
                    # 실패한 배치는 개별 청크로 유지
                    for chunk in batch:
                        if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                            combined_chunks.append(chunk)
            
            # 최종 결합
            if len(combined_chunks) == 1:
                final_df = combined_chunks[0]
            else:
                final_df = pd.concat(combined_chunks, ignore_index=True)
                final_df = self._optimize_chunk_memory(final_df)
            
            # 메모리 정리
            for chunk in combined_chunks:
                del chunk
            del combined_chunks
            gc.collect()
            
            logger.info(f"청크 결합 완료: {final_df.shape}")
            return final_df
            
        except Exception as e:
            logger.error(f"청크 결합 실패: {e}")
            # 실패 시 첫 번째 유효한 청크 반환
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    return chunk
            return pd.DataFrame()
    
    def _try_recover_chunk(self, failed_position: int, failed_chunk_size: int) -> bool:
        """청크 복구 시도"""
        try:
            logger.info(f"청크 복구 시도: 위치 {failed_position:,}")
            
            # 청크 크기를 더 크게 줄여서 재시도
            recovery_chunk_size = max(self.min_chunk_size, failed_chunk_size // 3)  # 2 → 3으로 변경
            
            recovery_df = self._read_chunk_safe(failed_position, recovery_chunk_size)
            
            if recovery_df is not None and not recovery_df.empty:
                logger.info(f"청크 복구 성공: {len(recovery_df):,}행")
                self.chunk_size = recovery_chunk_size
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"청크 복구 실패: {e}")
            return False

class StreamingDataLoader:
    """개선된 스트리밍 데이터 로더 - 1070만행 전체 처리"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.target_column = 'clicked'
        
        logger.info("개선된 스트리밍 데이터 로더 초기화 완료")
    
    def load_full_data_streaming(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """개선된 스트리밍 방식으로 1070만행 전체 처리"""
        logger.info("=== 1070만행 전체 데이터 스트리밍 로딩 시작 ===")
        
        try:
            # 메모리 상태 확인
            self.memory_monitor.log_memory_status("스트리밍 시작", force=True)
            
            # 파일 유효성 검증
            if not self._validate_files():
                raise ValueError("데이터 파일이 존재하지 않습니다")
            
            # 1. 학습 데이터 스트리밍 처리
            logger.info("학습 데이터 스트리밍 처리 시작")
            train_df = self._stream_process_file(str(self.config.TRAIN_PATH), is_train=True)
            
            if train_df is None or train_df.empty:
                raise ValueError("학습 데이터 스트리밍 처리 실패")
            
            # 중간 메모리 정리 강화
            self.memory_monitor.force_memory_cleanup(intensive=True)
            self.memory_monitor.log_memory_status("학습 데이터 완료", force=True)
            
            # 2. 테스트 데이터 스트리밍 처리
            logger.info("테스트 데이터 스트리밍 처리 시작") 
            test_df = self._stream_process_file(str(self.config.TEST_PATH), is_train=False)
            
            if test_df is None or test_df.empty:
                raise ValueError("테스트 데이터 스트리밍 처리 실패")
            
            # 최종 메모리 상태
            self.memory_monitor.log_memory_status("스트리밍 완료", force=True)
            
            logger.info(f"=== 전체 데이터 스트리밍 완료 - 학습: {train_df.shape}, 테스트: {test_df.shape} ===")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"개선된 스트리밍 데이터 로딩 실패: {e}")
            self.memory_monitor.force_memory_cleanup(intensive=True)
            raise
    
    def _stream_process_file(self, file_path: str, is_train: bool = True) -> pd.DataFrame:
        """개선된 파일 스트리밍 처리"""
        try:
            # PyArrow로 메타데이터 확인
            if not PYARROW_AVAILABLE:
                raise ValueError("PyArrow가 필요합니다")
            
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            num_row_groups = parquet_file.num_row_groups
            
            logger.info(f"파일 분석 - 총 {total_rows:,}행, {num_row_groups}개 row groups")
            
            # 결과 수집용
            all_chunks = []
            processed_rows = 0
            
            # Row group 단위로 개선된 스트리밍 처리
            for rg_idx in range(num_row_groups):
                try:
                    # 메모리 압박 확인
                    pressure = self.memory_monitor.check_memory_pressure()
                    if pressure['should_abort']:
                        logger.warning(f"메모리 한계 접근 - 처리된 행: {processed_rows:,}, 계속 시도")
                        # ABORT 상태에서도 강제 메모리 정리 후 계속 시도
                        self.memory_monitor.force_memory_cleanup(intensive=True)
                        
                        # 정리 후 재확인
                        pressure_after = self.memory_monitor.check_memory_pressure()
                        if pressure_after['should_abort']:
                            logger.error(f"메모리 정리 후에도 한계 - 중단: {processed_rows:,}행")
                            break
                    
                    # Row group 읽기
                    table = parquet_file.read_row_group(rg_idx)
                    chunk_df = table.to_pandas()
                    
                    logger.info(f"Row group {rg_idx+1}/{num_row_groups} 처리: {len(chunk_df):,}행")
                    
                    # 즉시 메모리 최적화
                    chunk_df = self._optimize_dataframe_aggressive(chunk_df)
                    
                    # 청크 저장
                    all_chunks.append(chunk_df)
                    processed_rows += len(chunk_df)
                    
                    # 메모리 해제
                    del table
                    gc.collect()
                    
                    # 더 빈번한 중간 결합 (5 → 3으로 변경)
                    if len(all_chunks) >= 3:
                        logger.info(f"중간 결합 수행: {len(all_chunks)}개 청크")
                        combined = self._combine_chunks_immediate(all_chunks)
                        all_chunks = [combined]
                        self.memory_monitor.force_memory_cleanup()
                    
                    # 진행률 출력
                    progress = (rg_idx + 1) / num_row_groups * 100
                    if rg_idx % 5 == 0 or rg_idx == num_row_groups - 1:  # 10 → 5로 변경
                        logger.info(f"진행률: {progress:.1f}% ({processed_rows:,}/{total_rows:,}행)")
                    
                except Exception as e:
                    logger.error(f"Row group {rg_idx} 처리 실패: {e}")
                    # 실패해도 계속 진행
                    continue
            
            # 최종 결합
            if not all_chunks:
                raise ValueError("처리된 데이터가 없습니다")
            
            logger.info(f"최종 결합: {len(all_chunks)}개 청크")
            final_df = self._combine_chunks_immediate(all_chunks)
            
            # 타겟 컬럼 확인 (학습 데이터인 경우)
            if is_train:
                if self.target_column not in final_df.columns:
                    possible_targets = [col for col in final_df.columns if 'click' in col.lower()]
                    if possible_targets:
                        self.target_column = possible_targets[0]
                        logger.info(f"타겟 컬럼 변경: {self.target_column}")
                    else:
                        raise ValueError(f"타겟 컬럼 '{self.target_column}'을 찾을 수 없습니다")
                
                # CTR 확인
                target_ctr = final_df[self.target_column].mean()
                logger.info(f"실제 CTR: {target_ctr:.4f}")
            
            logger.info(f"스트리밍 처리 완료: {final_df.shape} ({processed_rows:,}행 처리됨)")
            
            return final_df
            
        except Exception as e:
            logger.error(f"파일 스트리밍 처리 실패: {e}")
            raise
    
    def _combine_chunks_immediate(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """즉시 청크 결합 - 메모리 효율성 우선"""
        if not chunks:
            return pd.DataFrame()
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # 유효한 청크만 선별
            valid_chunks = [chunk for chunk in chunks if isinstance(chunk, pd.DataFrame) and not chunk.empty]
            
            if not valid_chunks:
                return pd.DataFrame()
            
            # 직접 결합
            combined_df = pd.concat(valid_chunks, ignore_index=True)
            
            # 즉시 최적화
            combined_df = self._optimize_dataframe_aggressive(combined_df)
            
            # 원본 청크들 해제
            for chunk in chunks:
                del chunk
            del chunks, valid_chunks
            gc.collect()
            
            return combined_df
            
        except Exception as e:
            logger.error(f"청크 결합 실패: {e}")
            # 실패 시 첫 번째 유효 청크 반환
            for chunk in chunks:
                if isinstance(chunk, pd.DataFrame) and not chunk.empty:
                    return chunk
            return pd.DataFrame()
    
    def _optimize_dataframe_aggressive(self, df: pd.DataFrame) -> pd.DataFrame:
        """공격적 DataFrame 최적화"""
        if df is None or df.empty:
            return df
        
        try:
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
            
            # 범주형 최적화 - 매우 선택적
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    # 매우 낮은 카디널리티만 범주형으로
                    if unique_count < 50 and unique_count < total_count * 0.05:  # 0.1 → 0.05로 변경
                        df[col] = df[col].astype('category')
                    else:
                        # 수치 변환 시도
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_series.isna().all():
                            df[col] = numeric_series.fillna(0).astype('float32')
                        else:
                            # 해시 변환
                            df[col] = df[col].astype(str).apply(lambda x: hash(x) % 50000).astype('int32')  # 100000 → 50000으로 변경
                except Exception:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
                    except Exception:
                        pass
            
            # 결측치 및 무한값 처리
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], [0, 0])
            
            return df
            
        except Exception as e:
            logger.warning(f"DataFrame 최적화 실패: {e}")
            return df
    
    def _validate_files(self) -> bool:
        """파일 유효성 검증"""
        try:
            train_exists = self.config.TRAIN_PATH.exists()
            test_exists = self.config.TEST_PATH.exists()
            
            if not train_exists or not test_exists:
                return False
            
            train_size_mb = self.config.TRAIN_PATH.stat().st_size / (1024**2)
            test_size_mb = self.config.TEST_PATH.stat().st_size / (1024**2)
            
            logger.info(f"파일 크기 - 학습: {train_size_mb:.1f}MB, 테스트: {test_size_mb:.1f}MB")
            
            return train_size_mb > 10 and test_size_mb > 10
            
        except Exception as e:
            logger.error(f"파일 검증 실패: {e}")
            return False

class LargeDataLoader:
    """대용량 데이터 로더"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
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
        
        logger.info("대용량 데이터 로더 초기화 완료")
    
    def load_large_data_optimized(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """1070만행 전체 데이터 처리"""
        logger.info("=== 1070만행 전체 데이터 처리 시작 ===")
        
        # 스트리밍 방식으로 전체 데이터 처리
        streaming_loader = StreamingDataLoader(self.config)
        return streaming_loader.load_full_data_streaming()

# 기존 코드와의 호환성을 위한 별칭
DataLoader = StreamingDataLoader
SimpleDataLoader = StreamingDataLoader
MemoryOptimizedChunkReader = DataChunkProcessor
AggressiveMemoryMonitor = MemoryMonitor

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    
    try:
        loader = StreamingDataLoader(config)
        train_df, test_df = loader.load_full_data_streaming()
        
        print(f"학습 데이터: {train_df.shape}")
        print(f"테스트 데이터: {test_df.shape}")
        print(f"전체 처리 완료: {len(train_df) + len(test_df):,}행")
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")