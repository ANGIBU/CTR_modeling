# feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
import gc
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
import hashlib
import pickle
warnings.filterwarnings('ignore')

# Psutil import 안전 처리
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil이 설치되지 않았습니다. 메모리 모니터링 기능이 제한됩니다.")

from config import Config

logger = logging.getLogger(__name__)

class LargeDataMemoryMonitor:
    """대용량 데이터 피처 엔지니어링용 메모리 모니터"""
    
    def __init__(self, memory_limit_gb: float = 40.0):
        self.memory_limit_gb = memory_limit_gb
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.memory_history = []
        self.lock = threading.Lock()
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
        if self.monitoring_enabled:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024**3)
            except:
                return 0.0
        return 0.0
    
    def get_available_memory(self) -> float:
        """사용 가능한 메모리 (GB)"""
        if self.monitoring_enabled:
            try:
                return psutil.virtual_memory().available / (1024**3)
            except:
                return 40.0
        return 40.0
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """메모리 압박 상태 확인"""
        current_usage = self.get_memory_usage()
        available = self.get_available_memory()
        
        pressure_level = 'normal'
        if available < 8:
            pressure_level = 'critical'
        elif available < 15:
            pressure_level = 'high'
        elif available < 25:
            pressure_level = 'moderate'
        
        return {
            'pressure_level': pressure_level,
            'current_usage_gb': current_usage,
            'available_gb': available,
            'memory_limit_gb': self.memory_limit_gb,
            'usage_ratio': current_usage / self.memory_limit_gb if self.memory_limit_gb > 0 else 0,
            'needs_cleanup': pressure_level in ['critical', 'high'] or current_usage > self.memory_limit_gb * 0.8
        }
    
    def force_memory_cleanup(self, aggressive: bool = False):
        """강제 메모리 정리"""
        try:
            initial_usage = self.get_memory_usage()
            
            # 기본 가비지 컬렉션
            for _ in range(3 if aggressive else 1):
                gc.collect()
                if aggressive:
                    time.sleep(0.1)
            
            # Windows 전용 메모리 압축
            if aggressive:
                try:
                    import ctypes
                    if hasattr(ctypes, 'windll'):
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                except:
                    pass
            
            final_usage = self.get_memory_usage()
            freed = initial_usage - final_usage
            
            if freed > 0.1:
                logger.info(f"메모리 정리: {freed:.2f}GB 해제")
                
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")

class ChunkedFeatureProcessor:
    """대용량 데이터 청킹 피처 처리 클래스"""
    
    def __init__(self, chunk_size: int = 500000, max_workers: int = 3):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.memory_monitor = LargeDataMemoryMonitor()
        
    def process_chunks_parallel(self, 
                               data: pd.DataFrame,
                               processing_func,
                               **kwargs) -> pd.DataFrame:
        """병렬 청킹 처리"""
        logger.info(f"병렬 청킹 처리 시작: {len(data):,}행 → {len(data)//self.chunk_size + 1}개 청크")
        
        # 메모리 상태 확인
        memory_status = self.memory_monitor.check_memory_pressure()
        if memory_status['needs_cleanup']:
            self.memory_monitor.force_memory_cleanup(aggressive=True)
        
        if memory_status['pressure_level'] == 'critical':
            # 메모리 부족 시 청크 크기 축소
            self.chunk_size = min(self.chunk_size, 200000)
            self.max_workers = 1
            logger.warning(f"메모리 부족으로 청킹 설정 조정: 크기={self.chunk_size}, 워커=1")
        
        chunks = []
        total_chunks = len(data) // self.chunk_size + (1 if len(data) % self.chunk_size > 0 else 0)
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 청크별 작업 제출
                future_to_chunk = {}
                
                for i in range(0, len(data), self.chunk_size):
                    end_idx = min(i + self.chunk_size, len(data))
                    chunk_data = data.iloc[i:end_idx].copy()
                    
                    future = executor.submit(processing_func, chunk_data, **kwargs)
                    future_to_chunk[future] = (i, end_idx)
                
                # 결과 수집
                processed_count = 0
                for future in as_completed(future_to_chunk):
                    try:
                        processed_chunk = future.result()
                        if processed_chunk is not None and not processed_chunk.empty:
                            chunks.append(processed_chunk)
                        
                        processed_count += 1
                        
                        # 진행 상황 로깅
                        if processed_count % 5 == 0:
                            logger.info(f"청킹 진행: {processed_count}/{total_chunks}")
                            
                            # 메모리 압박 시 중간 정리
                            memory_status = self.memory_monitor.check_memory_pressure()
                            if memory_status['needs_cleanup']:
                                self.memory_monitor.force_memory_cleanup()
                        
                    except Exception as e:
                        chunk_info = future_to_chunk[future]
                        logger.warning(f"청크 {chunk_info} 처리 실패: {e}")
                        continue
            
            # 결과 결합
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                logger.info(f"병렬 청킹 처리 완료: {result.shape}")
                return result
            else:
                logger.warning("유효한 청킹 결과가 없습니다")
                return data.copy()
                
        except Exception as e:
            logger.error(f"병렬 청킹 처리 실패: {e}")
            return data.copy()
        
        finally:
            self.memory_monitor.force_memory_cleanup()

class AdvancedCTRFeatureEngineer:
    """대용량 CTR 데이터 특화 고급 피처 엔지니어링"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = LargeDataMemoryMonitor(config.MAX_MEMORY_GB * 0.8)
        self.chunked_processor = ChunkedFeatureProcessor(
            chunk_size=config.CHUNK_SIZE // 2,
            max_workers=min(config.NUM_WORKERS, 4)
        )
        
        # 피처 엔지니어링 상태 관리
        self.target_encoders = {}
        self.label_encoders = {}
        self.scalers = {}
        self.quantile_transformers = {}
        self.feature_stats = {}
        self.generated_features = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.id_columns = []
        self.removed_columns = []
        self.final_feature_columns = []
        self.original_feature_order = []
        
        # 대용량 데이터 처리 설정
        self.memory_efficient_mode = False
        self.large_data_mode = False
        self.feature_selection_threshold = 0.001
        self.max_features_per_stage = 500
        
        # 성능 통계
        self.processing_stats = {
            'start_time': time.time(),
            'stages_completed': 0,
            'features_created': 0,
            'features_removed': 0,
            'memory_usage_peak': 0.0,
            'processing_time': 0.0
        }
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """대용량 데이터 CTR 피처 엔지니어링 메인 파이프라인"""
        logger.info("=== 대용량 CTR 피처 엔지니어링 시작 ===")
        
        # 초기 설정
        self._initialize_feature_engineering(train_df, test_df, target_col)
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.copy()
        
        try:
            # Stage 1: 기본 데이터 정리 및 타입 통일
            logger.info("Stage 1: 기본 데이터 정리")
            X_train, X_test = self._stage1_basic_preprocessing(X_train, X_test)
            self._update_stage_completion(1)
            
            # Stage 2: ID 피처 처리
            logger.info("Stage 2: ID 피처 고급 처리")
            X_train, X_test = self._stage2_advanced_id_processing(X_train, X_test)
            self._update_stage_completion(2)
            
            # Stage 3: 범주형 피처 고급 인코딩
            logger.info("Stage 3: 범주형 피처 고급 인코딩")
            X_train, X_test = self._stage3_advanced_categorical_encoding(X_train, X_test, y_train)
            self._update_stage_completion(3)
            
            # Stage 4: 수치형 피처 고급 변환
            logger.info("Stage 4: 수치형 피처 고급 변환")
            X_train, X_test = self._stage4_advanced_numeric_transformation(X_train, X_test)
            self._update_stage_completion(4)
            
            # Stage 5: CTR 특화 피처 생성
            logger.info("Stage 5: CTR 특화 피처 생성")
            X_train, X_test = self._stage5_ctr_specific_features(X_train, X_test, y_train)
            self._update_stage_completion(5)
            
            # Stage 6: 통계적 집계 피처
            logger.info("Stage 6: 통계적 집계 피처")
            X_train, X_test = self._stage6_statistical_aggregation(X_train, X_test)
            self._update_stage_completion(6)
            
            # Stage 7: 상호작용 피처 (메모리 허용 시)
            memory_status = self.memory_monitor.check_memory_pressure()
            if memory_status['pressure_level'] not in ['critical', 'high']:
                logger.info("Stage 7: 상호작용 피처")
                X_train, X_test = self._stage7_interaction_features(X_train, X_test)
                self._update_stage_completion(7)
            else:
                logger.info("Stage 7: 메모리 부족으로 상호작용 피처 생성 건너뛰기")
            
            # Stage 8: 피처 선택 및 최적화
            logger.info("Stage 8: 피처 선택 및 최적화")
            X_train, X_test = self._stage8_feature_selection_optimization(X_train, X_test, y_train)
            self._update_stage_completion(8)
            
            # Stage 9: 최종 데이터 정리
            logger.info("Stage 9: 최종 데이터 정리")
            X_train, X_test = self._stage9_final_cleanup(X_train, X_test)
            self._update_stage_completion(9)
            
            # 성능 통계 업데이트
            self.processing_stats.update({
                'processing_time': time.time() - self.processing_stats['start_time'],
                'final_train_shape': X_train.shape,
                'final_test_shape': X_test.shape,
                'memory_usage_peak': self.memory_monitor.get_memory_usage()
            })
            
            self._log_feature_engineering_completion(X_train, X_test)
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"피처 엔지니어링 실패: {e}")
            self.memory_monitor.force_memory_cleanup(aggressive=True)
            raise
    
    def _initialize_feature_engineering(self, train_df: pd.DataFrame, 
                                       test_df: pd.DataFrame, target_col: str):
        """피처 엔지니어링 초기화"""
        # 데이터 크기 기반 모드 설정
        total_rows = len(train_df) + len(test_df)
        
        if total_rows > 5000000:  # 500만행 이상
            self.large_data_mode = True
            self.memory_efficient_mode = True
            logger.info("대용량 데이터 모드 활성화")
        elif total_rows > 2000000:  # 200만행 이상
            self.memory_efficient_mode = True
            logger.info("메모리 효율 모드 활성화")
        
        # 원본 피처 순서 저장
        self.original_feature_order = sorted([col for col in train_df.columns if col != target_col])
        logger.info(f"원본 피처 수: {len(self.original_feature_order)}")
        
        # 초기 메모리 상태
        self.memory_monitor.force_memory_cleanup()
        memory_status = self.memory_monitor.check_memory_pressure()
        logger.info(f"초기 메모리 상태: {memory_status['pressure_level']}, "
                   f"사용가능 {memory_status['available_gb']:.1f}GB")
    
    def _stage1_basic_preprocessing(self, X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 1: 기본 데이터 정리 및 타입 통일"""
        
        def process_basic_chunk(chunk_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
            """청크 단위 기본 전처리"""
            try:
                # 결측치 처리
                for col in chunk_df.columns:
                    if chunk_df[col].isnull().sum() > 0:
                        if chunk_df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                            chunk_df[col].fillna(0, inplace=True)
                        else:
                            chunk_df[col].fillna('unknown', inplace=True)
                
                # 데이터 타입 최적화
                for col in chunk_df.columns:
                    if col.startswith('feat_'):
                        if chunk_df[col].dtype in ['int64', 'int32']:
                            chunk_df[col] = chunk_df[col].astype('int32')
                        elif chunk_df[col].dtype in ['float64']:
                            chunk_df[col] = chunk_df[col].astype('float32')
                
                return chunk_df
                
            except Exception as e:
                logger.warning(f"기본 전처리 청크 처리 실패: {e}")
                return chunk_df
        
        # 병렬 처리 적용
        if self.large_data_mode:
            X_train = self.chunked_processor.process_chunks_parallel(
                X_train, process_basic_chunk, is_train=True
            )
            X_test = self.chunked_processor.process_chunks_parallel(
                X_test, process_basic_chunk, is_train=False
            )
        else:
            X_train = process_basic_chunk(X_train, True)
            X_test = process_basic_chunk(X_test, False)
        
        logger.info("Stage 1 완료: 기본 데이터 정리")
        return X_train, X_test
    
    def _stage2_advanced_id_processing(self, X_train: pd.DataFrame, 
                                      X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 2: ID 피처 고급 처리"""
        
        # ID 컬럼 식별
        id_columns = []
        for col in X_train.columns:
            if any(pattern in col.lower() for pattern in ['id', 'uuid', 'key']):
                id_columns.append(col)
            elif X_train[col].nunique() / len(X_train) > 0.95:
                id_columns.append(col)
        
        logger.info(f"ID 컬럼 식별: {len(id_columns)}개")
        
        def process_id_chunk(chunk_df: pd.DataFrame, reference_stats: dict = None) -> pd.DataFrame:
            """청크 단위 ID 처리"""
            try:
                for col in id_columns:
                    if col in chunk_df.columns:
                        # 안전한 해시 인코딩
                        chunk_df[f'{col}_hash'] = chunk_df[col].astype(str).apply(
                            lambda x: hash(str(x)[:50]) % 1000000 if pd.notna(x) else 0
                        ).astype('int32')
                        
                        # 빈도 인코딩 (메모리 효율적)
                        if not self.memory_efficient_mode and reference_stats:
                            freq_map = reference_stats.get(f'{col}_freq', {})
                            chunk_df[f'{col}_freq'] = chunk_df[col].astype(str).map(freq_map).fillna(0).astype('int16')
                
                return chunk_df
                
            except Exception as e:
                logger.warning(f"ID 처리 청크 실패: {e}")
                return chunk_df
        
        # 빈도 통계 계산 (학습 데이터 기준)
        reference_stats = {}
        if not self.memory_efficient_mode and id_columns:
            for col in id_columns[:5]:  # 최대 5개만 처리
                if col in X_train.columns:
                    try:
                        value_counts = X_train[col].astype(str).value_counts()
                        reference_stats[f'{col}_freq'] = value_counts.to_dict()
                    except:
                        continue
        
        # 병렬 처리
        if self.large_data_mode:
            X_train = self.chunked_processor.process_chunks_parallel(
                X_train, process_id_chunk, reference_stats=reference_stats
            )
            X_test = self.chunked_processor.process_chunks_parallel(
                X_test, process_id_chunk, reference_stats=reference_stats
            )
        else:
            X_train = process_id_chunk(X_train, reference_stats)
            X_test = process_id_chunk(X_test, reference_stats)
        
        # 원본 ID 컬럼 제거
        X_train = X_train.drop(columns=[col for col in id_columns if col in X_train.columns])
        X_test = X_test.drop(columns=[col for col in id_columns if col in X_test.columns])
        
        self.removed_columns.extend(id_columns)
        logger.info(f"Stage 2 완료: {len(id_columns)}개 ID 컬럼 처리")
        return X_train, X_test
    
    def _stage3_advanced_categorical_encoding(self, X_train: pd.DataFrame, 
                                             X_test: pd.DataFrame,
                                             y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 3: 범주형 피처 고급 인코딩"""
        
        # 범주형 컬럼 식별
        categorical_columns = []
        for col in X_train.columns:
            if X_train[col].dtype in ['object', 'category'] or str(X_train[col].dtype) == 'string':
                categorical_columns.append(col)
        
        logger.info(f"범주형 컬럼: {len(categorical_columns)}개")
        
        if not categorical_columns:
            return X_train, X_test
        
        # 인코딩 통계 사전 계산
        encoding_stats = {}
        
        for col in categorical_columns:
            try:
                train_values = X_train[col].astype(str).fillna('missing')
                
                # Label 인코딩
                le = LabelEncoder()
                le.fit(train_values)
                self.label_encoders[col] = le
                
                # 빈도 인코딩
                freq_map = train_values.value_counts().to_dict()
                encoding_stats[f'{col}_freq'] = freq_map
                
                # 타겟 인코딩 (대용량 데이터용 간소화)
                if not self.memory_efficient_mode and len(X_train) < 3000000:
                    target_stats = pd.DataFrame({
                        'category': train_values,
                        'target': y_train
                    }).groupby('category')['target'].agg(['mean', 'count'])
                    
                    # 스무딩 적용
                    global_mean = y_train.mean()
                    smoothing = self.config.FEATURE_CONFIG['target_encoding_smoothing']
                    
                    target_stats['smoothed_mean'] = (
                        (target_stats['mean'] * target_stats['count'] + global_mean * smoothing) /
                        (target_stats['count'] + smoothing)
                    )
                    
                    encoding_stats[f'{col}_target'] = target_stats['smoothed_mean'].to_dict()
                
            except Exception as e:
                logger.warning(f"{col} 인코딩 통계 계산 실패: {e}")
                continue
        
        def process_categorical_chunk(chunk_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
            """청크 단위 범주형 인코딩"""
            try:
                for col in categorical_columns:
                    if col in chunk_df.columns:
                        values = chunk_df[col].astype(str).fillna('missing')
                        
                        # Label 인코딩
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            encoded_values = []
                            for val in values:
                                if val in le.classes_:
                                    encoded_values.append(le.transform([val])[0])
                                else:
                                    encoded_values.append(-1)
                            
                            chunk_df[f'{col}_encoded'] = np.array(encoded_values).astype('int16')
                        
                        # 빈도 인코딩
                        if f'{col}_freq' in encoding_stats:
                            freq_map = encoding_stats[f'{col}_freq']
                            chunk_df[f'{col}_freq'] = values.map(freq_map).fillna(0).astype('int16')
                        
                        # 타겟 인코딩
                        if f'{col}_target' in encoding_stats:
                            target_map = encoding_stats[f'{col}_target']
                            global_mean = 0.0201  # 기본 CTR
                            chunk_df[f'{col}_target'] = values.map(target_map).fillna(global_mean).astype('float32')
                
                return chunk_df
                
            except Exception as e:
                logger.warning(f"범주형 인코딩 청크 실패: {e}")
                return chunk_df
        
        # 병렬 처리
        if self.large_data_mode:
            X_train = self.chunked_processor.process_chunks_parallel(
                X_train, process_categorical_chunk, is_train=True
            )
            X_test = self.chunked_processor.process_chunks_parallel(
                X_test, process_categorical_chunk, is_train=False
            )
        else:
            X_train = process_categorical_chunk(X_train, True)
            X_test = process_categorical_chunk(X_test, False)
        
        # 원본 범주형 컬럼 제거
        X_train = X_train.drop(columns=categorical_columns)
        X_test = X_test.drop(columns=categorical_columns)
        
        self.removed_columns.extend(categorical_columns)
        logger.info(f"Stage 3 완료: {len(categorical_columns)}개 범주형 컬럼 인코딩")
        return X_train, X_test
    
    def _stage4_advanced_numeric_transformation(self, X_train: pd.DataFrame, 
                                               X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 4: 수치형 피처 고급 변환"""
        
        # 수치형 컬럼 식별
        numeric_columns = [col for col in X_train.columns 
                          if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 
                                                   'uint8', 'uint16', 'uint32', 'uint64',
                                                   'float16', 'float32', 'float64']]
        
        logger.info(f"수치형 컬럼: {len(numeric_columns)}개")
        
        # 변환 통계 사전 계산
        transform_stats = {}
        max_features = min(len(numeric_columns), self.max_features_per_stage)
        
        for col in numeric_columns[:max_features]:
            try:
                col_data = X_train[col].dropna()
                if len(col_data) == 0:
                    continue
                
                transform_stats[col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'q25': col_data.quantile(0.25),
                    'q50': col_data.quantile(0.50),
                    'q75': col_data.quantile(0.75),
                    'q90': col_data.quantile(0.90),
                    'positive_ratio': (col_data > 0).mean()
                }
                
            except Exception as e:
                logger.warning(f"{col} 변환 통계 계산 실패: {e}")
                continue
        
        def process_numeric_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
            """청크 단위 수치형 변환"""
            try:
                for col in numeric_columns[:max_features]:
                    if col not in chunk_df.columns or col not in transform_stats:
                        continue
                    
                    stats = transform_stats[col]
                    col_data = chunk_df[col]
                    
                    # 로그 변환 (양수 비율이 높을 때)
                    if stats['positive_ratio'] > 0.8:
                        chunk_df[f'{col}_log'] = np.log1p(np.maximum(col_data, 0)).astype('float32')
                    
                    # 제곱근 변환
                    if stats['min'] >= 0:
                        chunk_df[f'{col}_sqrt'] = np.sqrt(np.maximum(col_data, 0)).astype('float32')
                    
                    # 분위수 변환
                    quantile_bins = [-np.inf, stats['q25'], stats['q50'], stats['q75'], stats['q90'], np.inf]
                    chunk_df[f'{col}_quantile'] = pd.cut(
                        col_data, bins=quantile_bins, labels=range(5), include_lowest=True
                    ).astype('int8')
                    
                    # 이상치 플래그
                    iqr = stats['q75'] - stats['q25']
                    lower_bound = stats['q25'] - 1.5 * iqr
                    upper_bound = stats['q75'] + 1.5 * iqr
                    chunk_df[f'{col}_outlier'] = (
                        (col_data < lower_bound) | (col_data > upper_bound)
                    ).astype('int8')
                    
                    # Z-score (간소화)
                    if stats['std'] > 0:
                        chunk_df[f'{col}_zscore'] = (
                            (col_data - stats['mean']) / stats['std']
                        ).astype('float32')
                
                return chunk_df
                
            except Exception as e:
                logger.warning(f"수치형 변환 청크 실패: {e}")
                return chunk_df
        
        # 병렬 처리
        if self.large_data_mode:
            X_train = self.chunked_processor.process_chunks_parallel(X_train, process_numeric_chunk)
            X_test = self.chunked_processor.process_chunks_parallel(X_test, process_numeric_chunk)
        else:
            X_train = process_numeric_chunk(X_train)
            X_test = process_numeric_chunk(X_test)
        
        logger.info(f"Stage 4 완료: {max_features}개 수치형 컬럼 변환")
        return X_train, X_test
    
    def _stage5_ctr_specific_features(self, X_train: pd.DataFrame, 
                                     X_test: pd.DataFrame,
                                     y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 5: CTR 특화 피처 생성"""
        
        def create_ctr_features(chunk_df: pd.DataFrame, chunk_index: int = 0) -> pd.DataFrame:
            """CTR 특화 피처 생성"""
            try:
                # 시간적 특성 (인덱스 기반)
                chunk_df['time_index'] = (chunk_df.index / len(chunk_df)).astype('float32')
                chunk_df['position_quartile'] = pd.qcut(
                    chunk_df.index, q=4, labels=[0, 1, 2, 3]
                ).astype('int8')
                
                # 세션 특성 (가상)
                chunk_df['session_position'] = (chunk_df.index % 100).astype('int8')
                chunk_df['is_first_position'] = (chunk_df['session_position'] == 0).astype('int8')
                
                # 피처 그룹별 통계
                feature_groups = {
                    'feat_e': [col for col in chunk_df.columns if col.startswith('feat_e')],
                    'feat_d': [col for col in chunk_df.columns if col.startswith('feat_d')],
                    'feat_c': [col for col in chunk_df.columns if col.startswith('feat_c')],
                    'feat_b': [col for col in chunk_df.columns if col.startswith('feat_b')]
                }
                
                for group_name, group_cols in feature_groups.items():
                    if group_cols and len(group_cols) > 1:
                        group_data = chunk_df[group_cols]
                        
                        chunk_df[f'{group_name}_sum'] = group_data.sum(axis=1).astype('float32')
                        chunk_df[f'{group_name}_mean'] = group_data.mean(axis=1).astype('float32')
                        chunk_df[f'{group_name}_std'] = group_data.std(axis=1).astype('float32')
                        chunk_df[f'{group_name}_max'] = group_data.max(axis=1).astype('float32')
                        chunk_df[f'{group_name}_min'] = group_data.min(axis=1).astype('float32')
                        chunk_df[f'{group_name}_nonzero_count'] = (group_data != 0).sum(axis=1).astype('int8')
                
                # 전체 수치형 피처 통계
                numeric_cols = [col for col in chunk_df.columns 
                               if chunk_df[col].dtype in ['float32', 'int32', 'int16', 'int8']][:20]
                
                if len(numeric_cols) > 2:
                    numeric_data = chunk_df[numeric_cols]
                    chunk_df['numeric_sum'] = numeric_data.sum(axis=1).astype('float32')
                    chunk_df['numeric_mean'] = numeric_data.mean(axis=1).astype('float32')
                    chunk_df['numeric_std'] = numeric_data.std(axis=1).astype('float32')
                    chunk_df['nonzero_ratio'] = ((numeric_data != 0).sum(axis=1) / len(numeric_cols)).astype('float32')
                
                return chunk_df
                
            except Exception as e:
                logger.warning(f"CTR 피처 생성 실패: {e}")
                return chunk_df
        
        # 병렬 처리
        if self.large_data_mode:
            X_train = self.chunked_processor.process_chunks_parallel(X_train, create_ctr_features)
            X_test = self.chunked_processor.process_chunks_parallel(X_test, create_ctr_features)
        else:
            X_train = create_ctr_features(X_train)
            X_test = create_ctr_features(X_test)
        
        logger.info("Stage 5 완료: CTR 특화 피처 생성")
        return X_train, X_test
    
    def _stage6_statistical_aggregation(self, X_train: pd.DataFrame, 
                                       X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 6: 통계적 집계 피처"""
        
        def create_statistical_features(chunk_df: pd.DataFrame) -> pd.DataFrame:
            """통계적 집계 피처 생성"""
            try:
                # 수치형 컬럼만 선택
                numeric_cols = [col for col in chunk_df.columns 
                               if chunk_df[col].dtype in ['float32', 'int32', 'int16', 'int8']][:30]
                
                if len(numeric_cols) < 2:
                    return chunk_df
                
                numeric_data = chunk_df[numeric_cols]
                
                # 기본 통계
                chunk_df['all_features_sum'] = numeric_data.sum(axis=1).astype('float32')
                chunk_df['all_features_mean'] = numeric_data.mean(axis=1).astype('float32')
                chunk_df['all_features_std'] = numeric_data.std(axis=1).astype('float32')
                chunk_df['all_features_skew'] = numeric_data.skew(axis=1).astype('float32')
                
                # 극값 비율
                chunk_df['positive_features_ratio'] = (numeric_data > 0).mean(axis=1).astype('float32')
                chunk_df['negative_features_ratio'] = (numeric_data < 0).mean(axis=1).astype('float32')
                chunk_df['zero_features_ratio'] = (numeric_data == 0).mean(axis=1).astype('float32')
                
                # 분위수 기반 특성
                chunk_df['feature_range'] = (numeric_data.max(axis=1) - numeric_data.min(axis=1)).astype('float32')
                chunk_df['feature_iqr'] = (
                    numeric_data.quantile(0.75, axis=1) - numeric_data.quantile(0.25, axis=1)
                ).astype('float32')
                
                return chunk_df
                
            except Exception as e:
                logger.warning(f"통계적 피처 생성 실패: {e}")
                return chunk_df
        
        # 병렬 처리
        if self.large_data_mode:
            X_train = self.chunked_processor.process_chunks_parallel(X_train, create_statistical_features)
            X_test = self.chunked_processor.process_chunks_parallel(X_test, create_statistical_features)
        else:
            X_train = create_statistical_features(X_train)
            X_test = create_statistical_features(X_test)
        
        logger.info("Stage 6 완료: 통계적 집계 피처")
        return X_train, X_test
    
    def _stage7_interaction_features(self, X_train: pd.DataFrame, 
                                    X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 7: 상호작용 피처 (메모리 효율적)"""
        
        # 중요한 피처 선별 (분산 기준)
        numeric_cols = [col for col in X_train.columns 
                       if X_train[col].dtype in ['float32', 'int32']]
        
        if len(numeric_cols) < 2:
            return X_train, X_test
        
        # 상위 분산 피처 선택
        try:
            variances = X_train[numeric_cols].var()
            top_features = variances.nlargest(10).index.tolist()
        except:
            top_features = numeric_cols[:10]
        
        def create_interaction_features(chunk_df: pd.DataFrame) -> pd.DataFrame:
            """상호작용 피처 생성 (제한적)"""
            try:
                interaction_count = 0
                max_interactions = 15  # 메모리 절약
                
                for i, col1 in enumerate(top_features):
                    if interaction_count >= max_interactions or col1 not in chunk_df.columns:
                        break
                    
                    for j, col2 in enumerate(top_features[i+1:], i+1):
                        if interaction_count >= max_interactions or col2 not in chunk_df.columns:
                            break
                        
                        try:
                            # 곱셈 상호작용
                            chunk_df[f'{col1}_x_{col2}'] = (
                                chunk_df[col1] * chunk_df[col2]
                            ).astype('float32')
                            
                            # 비율 상호작용 (0으로 나누기 방지)
                            denominator = chunk_df[col2].replace(0, 1e-6)
                            chunk_df[f'{col1}_div_{col2}'] = (
                                chunk_df[col1] / denominator
                            ).astype('float32')
                            
                            interaction_count += 2
                            
                        except Exception as e:
                            logger.warning(f"상호작용 피처 {col1}_{col2} 생성 실패: {e}")
                            continue
                
                return chunk_df
                
            except Exception as e:
                logger.warning(f"상호작용 피처 생성 실패: {e}")
                return chunk_df
        
        # 병렬 처리
        if self.large_data_mode:
            X_train = self.chunked_processor.process_chunks_parallel(X_train, create_interaction_features)
            X_test = self.chunked_processor.process_chunks_parallel(X_test, create_interaction_features)
        else:
            X_train = create_interaction_features(X_train)
            X_test = create_interaction_features(X_test)
        
        logger.info(f"Stage 7 완료: 상호작용 피처 생성 ({len(top_features)}개 기준)")
        return X_train, X_test
    
    def _stage8_feature_selection_optimization(self, X_train: pd.DataFrame, 
                                              X_test: pd.DataFrame,
                                              y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 8: 피처 선택 및 최적화"""
        
        logger.info(f"피처 선택 전: {X_train.shape[1]}개 피처")
        
        # 1. 저분산 피처 제거
        low_variance_cols = []
        for col in X_train.columns:
            try:
                if X_train[col].var() < 1e-8:
                    low_variance_cols.append(col)
            except:
                continue
        
        if low_variance_cols:
            X_train = X_train.drop(columns=low_variance_cols)
            X_test = X_test.drop(columns=low_variance_cols)
            logger.info(f"저분산 피처 제거: {len(low_variance_cols)}개")
        
        # 2. 고상관 피처 제거 (메모리 효율적)
        if not self.memory_efficient_mode and X_train.shape[1] > 50:
            try:
                # 샘플링으로 상관관계 계산
                sample_size = min(100000, len(X_train))
                sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
                
                numeric_cols = X_train.select_dtypes(include=[np.number]).columns
                sample_corr = X_train.iloc[sample_idx][numeric_cols].corr().abs()
                
                # 상관계수 > 0.95인 피처 쌍 찾기
                upper_triangle = sample_corr.where(
                    np.triu(np.ones(sample_corr.shape), k=1).astype(bool)
                )
                
                high_corr_cols = [
                    column for column in upper_triangle.columns 
                    if any(upper_triangle[column] > 0.95)
                ]
                
                if high_corr_cols:
                    X_train = X_train.drop(columns=high_corr_cols)
                    X_test = X_test.drop(columns=high_corr_cols)
                    logger.info(f"고상관 피처 제거: {len(high_corr_cols)}개")
                
            except Exception as e:
                logger.warning(f"상관관계 기반 피처 제거 실패: {e}")
        
        # 3. 메모리 기반 피처 선택
        max_features = self.config.FEATURE_CONFIG['max_features']
        if X_train.shape[1] > max_features:
            try:
                # 분산 기반 선택
                feature_variances = X_train.var()
                top_features = feature_variances.nlargest(max_features).index.tolist()
                
                X_train = X_train[top_features]
                X_test = X_test[top_features]
                
                logger.info(f"분산 기반 피처 선택: {max_features}개 유지")
                
            except Exception as e:
                logger.warning(f"피처 선택 실패: {e}")
        
        logger.info(f"피처 선택 후: {X_train.shape[1]}개 피처")
        return X_train, X_test
    
    def _stage9_final_cleanup(self, X_train: pd.DataFrame, 
                             X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stage 9: 최종 데이터 정리"""
        
        # 1. 공통 컬럼만 유지
        common_columns = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]
        
        # 2. 데이터 타입 최종 통일
        for col in X_train.columns:
            try:
                if X_train[col].dtype != X_test[col].dtype:
                    # 더 넓은 타입으로 통일
                    if 'float' in str(X_train[col].dtype) or 'float' in str(X_test[col].dtype):
                        X_train[col] = X_train[col].astype('float32')
                        X_test[col] = X_test[col].astype('float32')
                    else:
                        X_train[col] = X_train[col].astype('int32')
                        X_test[col] = X_test[col].astype('int32')
            except:
                # 변환 실패 시 float32로 강제 변환
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
        
        # 3. 결측치 및 무한값 최종 정리
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        X_train = X_train.replace([np.inf, -np.inf], [1e6, -1e6])
        X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 4. 피처 순서 일관성 보장
        final_columns = sorted(common_columns)
        X_train = X_train[final_columns]
        X_test = X_test[final_columns]
        
        self.final_feature_columns = final_columns
        
        # 5. 최종 메모리 정리
        self.memory_monitor.force_memory_cleanup(aggressive=True)
        
        logger.info("Stage 9 완료: 최종 데이터 정리")
        return X_train, X_test
    
    def _update_stage_completion(self, stage_num: int):
        """스테이지 완료 업데이트"""
        self.processing_stats['stages_completed'] = stage_num
        
        # 메모리 상태 체크
        memory_status = self.memory_monitor.check_memory_pressure()
        if memory_status['needs_cleanup']:
            self.memory_monitor.force_memory_cleanup()
        
        logger.info(f"Stage {stage_num} 완료 - 메모리: {memory_status['available_gb']:.1f}GB 사용가능")
    
    def _log_feature_engineering_completion(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """피처 엔지니어링 완료 로깅"""
        stats = self.processing_stats
        memory_info = self.memory_monitor.check_memory_pressure()
        
        logger.info("=== 대용량 피처 엔지니어링 완료 ===")
        logger.info(f"최종 학습 데이터: {X_train.shape}")
        logger.info(f"최종 테스트 데이터: {X_test.shape}")
        logger.info(f"처리 시간: {stats['processing_time']:.2f}초")
        logger.info(f"완료된 스테이지: {stats['stages_completed']}/9")
        logger.info(f"메모리 사용 피크: {memory_info['current_usage_gb']:.2f}GB")
        logger.info(f"제거된 컬럼: {len(self.removed_columns)}개")
        logger.info(f"최종 피처 수: {len(self.final_feature_columns)}")
        
        if self.large_data_mode:
            logger.info("✓ 대용량 데이터 모드로 처리 완료")
        if self.memory_efficient_mode:
            logger.info("✓ 메모리 효율 모드로 처리 완료")
        
        logger.info("=== 피처 엔지니어링 완료 ===")
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약 정보"""
        return {
            'total_generated_features': len(self.final_feature_columns),
            'original_features': len(self.original_feature_order),
            'removed_features': len(self.removed_columns),
            'final_feature_columns': self.final_feature_columns,
            'removed_columns': self.removed_columns,
            'processing_stats': self.processing_stats,
            'memory_efficient_mode': self.memory_efficient_mode,
            'large_data_mode': self.large_data_mode,
            'encoders_info': {
                'label_encoders': len(self.label_encoders),
                'target_encoders': len(self.target_encoders),
                'scalers': len(self.scalers)
            }
        }

# 호환성을 위한 별칭
FeatureEngineer = AdvancedCTRFeatureEngineer
CTRFeatureEngineer = AdvancedCTRFeatureEngineer