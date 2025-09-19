# feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
import gc
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import re
warnings.filterwarnings('ignore')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config import Config

def get_safe_logger(name: str):
    """로거 생성"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_safe_logger(__name__)

class MemoryMonitor:
    """메모리 모니터링"""
    
    def __init__(self, max_memory_gb: float = 60.0):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 3.0
        self.max_memory_gb = max_memory_gb
        
        # 64GB 환경에 최적화된 임계값
        self.warning_threshold = max_memory_gb * 0.80
        self.critical_threshold = max_memory_gb * 0.90
        self.abort_threshold = max_memory_gb * 0.95
    
    def get_memory_usage(self) -> float:
        """메모리 사용량 (GB)"""
        if not self.monitoring_enabled:
            return 2.0
        
        try:
            with self.lock:
                current_time = time.time()
                if current_time - self._last_check_time < self._check_interval:
                    return getattr(self, '_cached_memory', 2.0)
                
                process = psutil.Process()
                memory_gb = process.memory_info().rss / (1024**3)
                self._cached_memory = memory_gb
                self._last_check_time = current_time
                return memory_gb
        except Exception:
            return 2.0
    
    def get_available_memory(self) -> float:
        """사용 가능 메모리 (GB)"""
        if not self.monitoring_enabled:
            return 45.0
        
        try:
            with self.lock:
                return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 45.0
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 확인"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            # 64GB 환경에서 더 관대한 기준
            return usage > self.critical_threshold or available < 15.0
        except Exception:
            return False
    
    def get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태"""
        try:
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            if usage > self.abort_threshold or available < 8:
                level = "abort"
            elif usage > self.critical_threshold or available < 15:
                level = "critical"
            elif usage > self.warning_threshold or available < 25:
                level = "warning"
            else:
                level = "normal"
            
            return {
                'usage_gb': usage,
                'available_gb': available,
                'level': level,
                'should_cleanup': level in ['warning', 'critical', 'abort'],
                'should_simplify': level in ['abort']  # 64GB 환경에서 완화
            }
        except Exception:
            return {
                'usage_gb': 2.0,
                'available_gb': 45.0,
                'level': 'normal',
                'should_cleanup': False,
                'should_simplify': False
            }
    
    def force_memory_cleanup(self):
        """메모리 정리"""
        try:
            for _ in range(15):
                gc.collect()
                time.sleep(0.1)
            
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass
        except Exception:
            pass
    
    def log_memory_status(self, context: str = ""):
        """메모리 상태 로깅"""
        try:
            status = self.get_memory_status()
            
            if status['level'] != 'normal' or context:
                logger.info(f"메모리 상태 [{context}]: 사용 {status['usage_gb']:.1f}GB, "
                           f"가용 {status['available_gb']:.1f}GB - {status['level'].upper()}")
                
        except Exception as e:
            logger.warning(f"메모리 상태 로깅 실패: {e}")

class CTRFeatureEngineer:
    """CTR 피처 엔지니어링 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.memory_efficient_mode = False  # 64GB 환경에서 기본 비활성화
        
        # 피처 엔지니어링 상태
        self.target_encoders = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        self.generated_features = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.id_columns = []
        self.removed_columns = []
        self.final_feature_columns = []
        self.original_feature_order = []
        
        # 피처 생성 카운터
        self.interaction_features = []
        self.target_encoding_features = []
        self.statistical_features = []
        self.temporal_features = []
        self.frequency_features = []
        self.ngram_features = []
        
        # 성능 통계
        self.processing_stats = {
            'start_time': time.time(),
            'total_features_generated': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'feature_types_count': {}
        }
        
        logger.info("CTR 피처 엔지니어 초기화 완료")
    
    def set_memory_efficient_mode(self, enabled: bool):
        """메모리 효율 모드 설정"""
        self.memory_efficient_mode = enabled
        mode_text = "활성화" if enabled else "비활성화"
        logger.info(f"메모리 효율 모드 {mode_text}")
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """전체 피처 엔지니어링 파이프라인"""
        logger.info("=== 피처 엔지니어링 시작 ===")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            # 1. 기본 데이터 준비
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            # 2. 컬럼 분류
            self._classify_columns(X_train)
            
            # 3. 데이터 타입 통일
            X_train, X_test = self._unify_data_types(X_train, X_test)
            
            # 4. 기본 피처 정리
            X_train, X_test = self._clean_basic_features(X_train, X_test)
            
            # 5. 교차 피처 생성 (64GB 환경에서 적극적 생성)
            X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
            
            # 6. 타겟 인코딩 (64GB 환경에서 더 많은 피처 생성)
            X_train, X_test = self._create_target_encoding_features(X_train, X_test, y_train)
            
            # 7. 시간 기반 피처
            X_train, X_test = self._create_temporal_features(X_train, X_test)
            
            # 8. 통계 피처 (64GB 환경에서 모든 피처 생성)
            X_train, X_test = self._create_statistical_features(X_train, X_test)
            
            # 9. 빈도 기반 피처
            X_train, X_test = self._create_frequency_features(X_train, X_test)
            
            # 10. 범주형 피처 인코딩
            X_train, X_test = self._encode_categorical_features(X_train, X_test, y_train)
            
            # 11. 수치형 피처 변환
            X_train, X_test = self._create_numeric_features(X_train, X_test)
            
            # 12. 최종 데이터 정리
            X_train, X_test = self._final_data_cleanup(X_train, X_test)
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== 피처 엔지니어링 완료: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"피처 엔지니어링 실패: {e}")
            self.memory_monitor.force_memory_cleanup()
            
            # 실패 시 기본 피처만 사용
            logger.warning("오류로 인해 기본 피처만 사용")
            return self._create_basic_features_only(train_df, test_df, target_col)
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """초기화"""
        try:
            self.processing_stats['start_time'] = time.time()
            self.original_feature_order = sorted([col for col in train_df.columns if col != target_col])
            
            logger.info(f"초기 데이터: 학습 {train_df.shape}, 테스트 {test_df.shape}")
            logger.info(f"원본 피처 수: {len(self.original_feature_order)}")
            
            self.memory_monitor.log_memory_status("초기화")
            
        except Exception as e:
            logger.warning(f"초기화 실패: {e}")
    
    def _prepare_basic_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """기본 데이터 준비"""
        try:
            if target_col not in train_df.columns:
                available_targets = [col for col in train_df.columns if 'click' in col.lower()]
                if available_targets:
                    target_col = available_targets[0]
                    logger.warning(f"타겟 컬럼 변경: {target_col}")
                else:
                    raise ValueError(f"타겟 컬럼 '{target_col}'을 찾을 수 없습니다")
            
            X_train = train_df.drop(columns=[target_col]).copy()
            y_train = train_df[target_col].copy()
            X_test = test_df.copy()
            
            # 타겟 분포 확인
            target_dist = y_train.value_counts()
            actual_ctr = y_train.mean()
            
            logger.info(f"타겟 분포: {target_dist.to_dict()}")
            logger.info(f"실제 CTR: {actual_ctr:.4f}")
            
            return X_train, X_test, y_train
            
        except Exception as e:
            logger.error(f"기본 데이터 준비 실패: {e}")
            raise
    
    def _classify_columns(self, df: pd.DataFrame):
        """컬럼 타입 분류"""
        logger.info("컬럼 타입 분류 시작")
        
        self.numeric_columns = []
        self.categorical_columns = []
        self.id_columns = []
        
        try:
            for col in df.columns:
                try:
                    dtype_str = str(df[col].dtype)
                    col_lower = str(col).lower()
                    
                    # ID 컬럼 식별
                    if any(pattern in col_lower for pattern in ['id', 'uuid', 'key', 'hash']):
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio > 0.9:
                            self.id_columns.append(col)
                            continue
                    
                    # 수치형 컬럼
                    if dtype_str in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                   'float16', 'float32', 'float64']:
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio > 0.95:
                            self.id_columns.append(col)
                        else:
                            self.numeric_columns.append(col)
                    else:
                        self.categorical_columns.append(col)
                        
                except Exception as e:
                    logger.warning(f"컬럼 {col} 분류 실패: {e}")
                    self.numeric_columns.append(col)
            
            logger.info(f"컬럼 분류 완료 - 수치형: {len(self.numeric_columns)}, "
                       f"범주형: {len(self.categorical_columns)}, ID: {len(self.id_columns)}")
            
        except Exception as e:
            logger.error(f"컬럼 분류 실패: {e}")
            self.numeric_columns = list(df.columns)
            self.categorical_columns = []
            self.id_columns = []
    
    def _unify_data_types(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 타입 통일"""
        logger.info("데이터 타입 통일 시작")
        
        try:
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            processed_count = 0
            
            batch_size = 20  # 64GB 환경에서 더 큰 배치
            
            for i in range(0, len(common_columns), batch_size):
                batch_cols = common_columns[i:i + batch_size]
                
                for col in batch_cols:
                    try:
                        train_dtype = str(X_train[col].dtype)
                        test_dtype = str(X_test[col].dtype)
                        
                        # 특수 컬럼 처리
                        if col == 'seq' or 'seq' in str(col).lower():
                            X_train[col] = self._safe_hash_column(X_train[col])
                            X_test[col] = self._safe_hash_column(X_test[col])
                            processed_count += 1
                            continue
                        
                        # 타입 불일치 해결
                        if train_dtype != test_dtype or train_dtype in ['object', 'category']:
                            try:
                                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                            except Exception:
                                X_train[col] = self._safe_hash_column(X_train[col])
                                X_test[col] = self._safe_hash_column(X_test[col])
                        
                        # 메모리 최적화
                        elif train_dtype in ['int64', 'float64']:
                            try:
                                if train_dtype == 'int64':
                                    X_train[col], X_test[col] = self._optimize_int_columns(X_train[col], X_test[col])
                                else:
                                    X_train[col] = X_train[col].astype('float32')
                                    X_test[col] = X_test[col].astype('float32')
                            except Exception:
                                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"컬럼 {col} 타입 통일 실패: {e}")
                        try:
                            X_train[col] = 0.0
                            X_test[col] = 0.0
                        except Exception:
                            pass
                
                if i % (batch_size * 3) == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"데이터 타입 통일 완료: {processed_count}/{len(common_columns)}개 컬럼")
            
        except Exception as e:
            logger.error(f"데이터 타입 통일 실패: {e}")
        
        return X_train, X_test
    
    def _safe_hash_column(self, series: pd.Series) -> pd.Series:
        """안전한 컬럼 해시 변환"""
        try:
            def safe_hash(x):
                try:
                    if pd.isna(x):
                        return 0
                    str_val = str(x)[:25]
                    return hash(str_val) % 30000
                except Exception:
                    return 0
            
            return series.apply(safe_hash).astype('int32')
            
        except Exception:
            return pd.Series([0] * len(series), dtype='int32')
    
    def _optimize_int_columns(self, train_col: pd.Series, test_col: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """정수 컬럼 최적화"""
        try:
            train_min, train_max = train_col.min(), train_col.max()
            test_min, test_max = test_col.min(), test_col.max()
            
            overall_min = min(train_min, test_min)
            overall_max = max(train_max, test_max)
            
            if pd.isna(overall_min) or pd.isna(overall_max):
                return train_col.astype('float32'), test_col.astype('float32')
            
            if overall_min >= 0:
                if overall_max < 255:
                    return train_col.astype('uint8'), test_col.astype('uint8')
                elif overall_max < 65535:
                    return train_col.astype('uint16'), test_col.astype('uint16')
                else:
                    return train_col.astype('uint32'), test_col.astype('uint32')
            else:
                if overall_min > -128 and overall_max < 127:
                    return train_col.astype('int8'), test_col.astype('int8')
                elif overall_min > -32768 and overall_max < 32767:
                    return train_col.astype('int16'), test_col.astype('int16')
                else:
                    return train_col.astype('int32'), test_col.astype('int32')
                    
        except Exception:
            return train_col.astype('float32'), test_col.astype('float32')
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 피처 정리"""
        logger.info("기본 피처 정리 시작")
        
        try:
            cols_to_remove = []
            
            batch_size = 30  # 64GB 환경에서 더 큰 배치
            for i in range(0, len(X_train.columns), batch_size):
                batch_cols = X_train.columns[i:i + batch_size]
                
                for col in batch_cols:
                    try:
                        if X_train[col].nunique() <= 1:
                            cols_to_remove.append(col)
                    except Exception:
                        continue
                
                memory_status = self.memory_monitor.get_memory_status()
                if memory_status['should_simplify']:
                    break
            
            if cols_to_remove:
                X_train = X_train.drop(columns=cols_to_remove)
                X_test = X_test.drop(columns=[col for col in cols_to_remove if col in X_test.columns])
                self.removed_columns.extend(cols_to_remove)
                logger.info(f"상수 컬럼 {len(cols_to_remove)}개 제거")
            
        except Exception as e:
            logger.warning(f"기본 피처 정리 실패: {e}")
        
        return X_train, X_test
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """교차 피처 생성"""
        logger.info("교차 피처 생성 시작")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            # 64GB 환경에서 더 많은 교차 피처 생성
            max_interactions = 20 if not memory_status['should_simplify'] else 12
            
            # 중요한 피처들 식별
            important_features = []
            
            # feat_으로 시작하는 피처들 우선
            for prefix in ['feat_e', 'feat_d', 'feat_c', 'feat_b']:
                prefix_features = [col for col in X_train.columns if col.startswith(prefix)][:4]
                important_features.extend(prefix_features)
            
            # 범주형 피처들
            categorical_features = []
            for col in X_train.columns:
                if col not in important_features and X_train[col].nunique() < 100:
                    categorical_features.append(col)
                    if len(categorical_features) >= 4:
                        break
            
            important_features.extend(categorical_features[:4])
            important_features = important_features[:12]
            
            interaction_count = 0
            for i, feat1 in enumerate(important_features):
                if interaction_count >= max_interactions:
                    break
                    
                for j, feat2 in enumerate(important_features[i+1:], i+1):
                    if interaction_count >= max_interactions:
                        break
                    
                    try:
                        interaction_name = f'{feat1}_x_{feat2}'
                        
                        # 곱셈 교차
                        X_train[interaction_name] = (X_train[feat1] * X_train[feat2]).astype('float32')
                        X_test[interaction_name] = (X_test[feat1] * X_test[feat2]).astype('float32')
                        
                        self.interaction_features.append(interaction_name)
                        self.generated_features.append(interaction_name)
                        interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"교차 피처 {feat1} x {feat2} 생성 실패: {e}")
                    
                    if interaction_count % 5 == 0:
                        self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"교차 피처 {len(self.interaction_features)}개 생성")
            
        except Exception as e:
            logger.error(f"교차 피처 생성 실패: {e}")
        
        return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                       y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """타겟 인코딩 피처 생성"""
        logger.info("타겟 인코딩 피처 생성 시작")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            # 64GB 환경에서 더 많은 타겟 인코딩 피처 생성
            max_target_features = 15 if not memory_status['should_simplify'] else 8
            
            # 범주형 피처들 선택
            categorical_candidates = []
            for col in X_train.columns:
                unique_count = X_train[col].nunique()
                if 2 <= unique_count <= 500:
                    categorical_candidates.append(col)
            
            categorical_candidates = categorical_candidates[:max_target_features]
            
            for col in categorical_candidates:
                try:
                    # 베이지안 평활화를 적용한 타겟 인코딩
                    target_mean = y_train.mean()
                    
                    # 그룹별 통계
                    group_stats = pd.DataFrame()
                    group_stats['count'] = X_train[col].value_counts()
                    group_stats['mean'] = X_train.groupby(col)[y_train.name if hasattr(y_train, 'name') else 0].agg(lambda x: y_train.iloc[x.index].mean())
                    
                    # 베이지안 평활화
                    alpha = 100
                    group_stats['target_encoded'] = (group_stats['mean'] * group_stats['count'] + target_mean * alpha) / (group_stats['count'] + alpha)
                    
                    # 매핑 적용
                    target_map = group_stats['target_encoded'].to_dict()
                    
                    feature_name = f'{col}_target_encoded'
                    X_train[feature_name] = X_train[col].map(target_map).fillna(target_mean).astype('float32')
                    X_test[feature_name] = X_test[col].map(target_map).fillna(target_mean).astype('float32')
                    
                    self.target_encoding_features.append(feature_name)
                    self.generated_features.append(feature_name)
                    self.target_encoders[col] = target_map
                    
                except Exception as e:
                    logger.warning(f"타겟 인코딩 {col} 실패: {e}")
                
                if len(self.target_encoding_features) % 5 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"타겟 인코딩 피처 {len(self.target_encoding_features)}개 생성")
            
        except Exception as e:
            logger.error(f"타겟 인코딩 피처 생성 실패: {e}")
        
        return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """시간 기반 피처 생성"""
        logger.info("시간 기반 피처 생성 시작")
        
        try:
            # 기본 시간 인덱스 피처
            X_train['time_index'] = (X_train.index / len(X_train)).astype('float32')
            X_test['time_index'] = (X_test.index / len(X_test)).astype('float32')
            
            # 분위수 기반 시간 구간
            X_train['time_quartile'] = pd.qcut(X_train.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            X_test['time_quartile'] = pd.qcut(X_test.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            
            # 10분위수 기반 시간 구간
            X_train['time_decile'] = pd.qcut(X_train.index, q=10, labels=list(range(10))).astype('int8')
            X_test['time_decile'] = pd.qcut(X_test.index, q=10, labels=list(range(10))).astype('int8')
            
            # 시간대별 상대 위치
            X_train['time_sin'] = np.sin(2 * np.pi * X_train['time_index']).astype('float32')
            X_test['time_sin'] = np.sin(2 * np.pi * X_test['time_index']).astype('float32')
            
            X_train['time_cos'] = np.cos(2 * np.pi * X_train['time_index']).astype('float32')
            X_test['time_cos'] = np.cos(2 * np.pi * X_test['time_index']).astype('float32')
            
            # 64GB 환경에서 추가 시간 피처
            memory_status = self.memory_monitor.get_memory_status()
            if not memory_status['should_simplify']:
                # 시간 구간별 상대적 위치
                X_train['time_relative_pos'] = ((X_train.index % 1000) / 1000).astype('float32')
                X_test['time_relative_pos'] = ((X_test.index % 1000) / 1000).astype('float32')
                
                # 시간 기반 순환 피처
                X_train['time_cycle'] = (X_train.index % 5000 / 5000).astype('float32')
                X_test['time_cycle'] = (X_test.index % 5000 / 5000).astype('float32')
                
                temporal_features = ['time_index', 'time_quartile', 'time_decile', 'time_sin', 'time_cos', 
                                   'time_relative_pos', 'time_cycle']
            else:
                temporal_features = ['time_index', 'time_quartile', 'time_decile', 'time_sin', 'time_cos']
            
            self.temporal_features.extend(temporal_features)
            self.generated_features.extend(temporal_features)
            
            logger.info(f"시간 기반 피처 {len(temporal_features)}개 생성")
            
        except Exception as e:
            logger.error(f"시간 기반 피처 생성 실패: {e}")
        
        return X_train, X_test
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """통계 피처 생성"""
        logger.info("통계 피처 생성 시작")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            
            # 피처 그룹별 통계
            feature_groups = {
                'feat_e': [col for col in X_train.columns if col.startswith('feat_e')][:6],
                'feat_d': [col for col in X_train.columns if col.startswith('feat_d')][:6],
                'feat_c': [col for col in X_train.columns if col.startswith('feat_c')][:6],
                'feat_b': [col for col in X_train.columns if col.startswith('feat_b')][:6]
            }
            
            for group_name, group_cols in feature_groups.items():
                if len(group_cols) >= 2:
                    try:
                        # 그룹 합계
                        sum_feature = f'{group_name}_sum'
                        X_train[sum_feature] = X_train[group_cols].sum(axis=1).astype('float32')
                        X_test[sum_feature] = X_test[group_cols].sum(axis=1).astype('float32')
                        
                        # 그룹 평균
                        mean_feature = f'{group_name}_mean'
                        X_train[mean_feature] = X_train[group_cols].mean(axis=1).astype('float32')
                        X_test[mean_feature] = X_test[group_cols].mean(axis=1).astype('float32')
                        
                        # 64GB 환경에서 추가 통계 피처
                        if not memory_status['should_simplify']:
                            # 그룹 표준편차
                            std_feature = f'{group_name}_std'
                            X_train[std_feature] = X_train[group_cols].std(axis=1).fillna(0).astype('float32')
                            X_test[std_feature] = X_test[group_cols].std(axis=1).fillna(0).astype('float32')
                            
                            # 그룹 최대값
                            max_feature = f'{group_name}_max'
                            X_train[max_feature] = X_train[group_cols].max(axis=1).astype('float32')
                            X_test[max_feature] = X_test[group_cols].max(axis=1).astype('float32')
                            
                            # 그룹 최소값
                            min_feature = f'{group_name}_min'
                            X_train[min_feature] = X_train[group_cols].min(axis=1).astype('float32')
                            X_test[min_feature] = X_test[group_cols].min(axis=1).astype('float32')
                            
                            group_features = [sum_feature, mean_feature, std_feature, max_feature, min_feature]
                        else:
                            group_features = [sum_feature, mean_feature]
                        
                        self.statistical_features.extend(group_features)
                        self.generated_features.extend(group_features)
                        
                    except Exception as e:
                        logger.warning(f"{group_name} 그룹 통계 생성 실패: {e}")
                
                if len(self.statistical_features) % 5 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"통계 피처 {len(self.statistical_features)}개 생성")
            
        except Exception as e:
            logger.error(f"통계 피처 생성 실패: {e}")
        
        return X_train, X_test
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """빈도 기반 피처 생성"""
        logger.info("빈도 기반 피처 생성 시작")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            # 64GB 환경에서 더 많은 빈도 피처 생성
            max_freq_features = 12 if not memory_status['should_simplify'] else 6
            
            # 빈도 인코딩할 피처들 선택
            freq_candidates = []
            for col in X_train.columns:
                unique_count = X_train[col].nunique()
                if 5 <= unique_count <= 1000:
                    freq_candidates.append(col)
            
            freq_candidates = freq_candidates[:max_freq_features]
            
            for col in freq_candidates:
                try:
                    # 빈도 계산
                    value_counts = X_train[col].value_counts()
                    freq_map = value_counts.to_dict()
                    
                    # 빈도 피처
                    freq_feature = f'{col}_freq'
                    X_train[freq_feature] = X_train[col].map(freq_map).fillna(0).astype('int32')
                    X_test[freq_feature] = X_test[col].map(freq_map).fillna(0).astype('int32')
                    
                    # 희귀도 피처
                    rarity_feature = f'{col}_rarity'
                    max_freq = max(freq_map.values())
                    rarity_map = {k: max_freq / v for k, v in freq_map.items()}
                    
                    X_train[rarity_feature] = X_train[col].map(rarity_map).fillna(max_freq).astype('float32')
                    X_test[rarity_feature] = X_test[col].map(rarity_map).fillna(max_freq).astype('float32')
                    
                    freq_features = [freq_feature, rarity_feature]
                    self.frequency_features.extend(freq_features)
                    self.generated_features.extend(freq_features)
                    
                except Exception as e:
                    logger.warning(f"빈도 피처 {col} 생성 실패: {e}")
                
                if len(self.frequency_features) % 6 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            logger.info(f"빈도 기반 피처 {len(self.frequency_features)}개 생성")
            
        except Exception as e:
            logger.error(f"빈도 기반 피처 생성 실패: {e}")
        
        return X_train, X_test
    
    def _encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """범주형 피처 인코딩"""
        current_categorical_cols = []
        for col in X_train.columns:
            try:
                dtype_str = str(X_train[col].dtype)
                if dtype_str in ['object', 'category', 'string']:
                    current_categorical_cols.append(col)
            except Exception:
                continue
        
        if not current_categorical_cols:
            return X_train, X_test
        
        logger.info(f"범주형 피처 인코딩 시작: {len(current_categorical_cols)}개")
        
        try:
            memory_status = self.memory_monitor.get_memory_status()
            # 64GB 환경에서 더 많은 범주형 피처 처리
            max_categorical = 15 if not memory_status['should_simplify'] else 8
            
            for col in current_categorical_cols[:max_categorical]:
                try:
                    train_values = X_train[col].astype(str).fillna('missing')
                    test_values = X_test[col].astype(str).fillna('missing')
                    
                    # 고카디널리티 처리
                    unique_count = len(train_values.unique())
                    max_categories = 50 if not memory_status['should_simplify'] else 30
                    
                    if unique_count > max_categories:
                        top_categories = train_values.value_counts().head(max_categories).index
                        train_values = train_values.where(train_values.isin(top_categories), 'other')
                        test_values = test_values.where(test_values.isin(top_categories), 'other')
                    
                    # Label Encoding
                    try:
                        le = LabelEncoder()
                        le.fit(train_values)
                        
                        train_encoded = le.transform(train_values).astype('int16')
                        
                        test_encoded = []
                        for val in test_values:
                            if val in le.classes_:
                                test_encoded.append(le.transform([val])[0])
                            else:
                                test_encoded.append(-1)
                        
                        X_train[f'{col}_encoded'] = train_encoded
                        X_test[f'{col}_encoded'] = np.array(test_encoded, dtype='int16')
                        
                        self.label_encoders[col] = le
                        self.generated_features.append(f'{col}_encoded')
                        
                    except Exception as e:
                        logger.warning(f"{col} Label Encoding 실패: {e}")
                    
                except Exception as e:
                    logger.warning(f"범주형 피처 {col} 처리 실패: {e}")
                
                if len(self.generated_features) % 5 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            # 원본 범주형 컬럼 제거
            try:
                existing_categorical = [col for col in current_categorical_cols if col in X_train.columns]
                if existing_categorical:
                    X_train = X_train.drop(columns=existing_categorical)
                    X_test = X_test.drop(columns=[col for col in existing_categorical if col in X_test.columns])
                    self.removed_columns.extend(existing_categorical)
            except Exception as e:
                logger.warning(f"범주형 컬럼 제거 실패: {e}")
            
            logger.info(f"범주형 피처 인코딩 완료")
            
        except Exception as e:
            logger.error(f"범주형 피처 인코딩 실패: {e}")
        
        return X_train, X_test
    
    def _create_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """수치형 피처 변환"""
        logger.info("수치형 피처 변환 시작")
        
        try:
            current_numeric_cols = [col for col in X_train.columns 
                                  if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                                           'float16', 'float32', 'float64']]
            
            if not current_numeric_cols:
                return X_train, X_test
            
            memory_status = self.memory_monitor.get_memory_status()
            feature_count = 0
            # 64GB 환경에서 더 많은 수치형 피처 변환
            max_features = 18 if not memory_status['should_simplify'] else 8
            
            for col in current_numeric_cols[:15]:
                try:
                    if feature_count >= max_features:
                        break
                    
                    # 로그 변환
                    train_positive = (X_train[col] > 0) & X_train[col].notna()
                    test_positive = (X_test[col] > 0) & X_test[col].notna()
                    
                    if train_positive.sum() > len(X_train) * 0.7:
                        try:
                            X_train[f'{col}_log'] = np.where(train_positive, np.log1p(X_train[col]), 0).astype('float32')
                            X_test[f'{col}_log'] = np.where(test_positive, np.log1p(X_test[col]), 0).astype('float32')
                            
                            self.generated_features.append(f'{col}_log')
                            feature_count += 1
                        except Exception as e:
                            logger.warning(f"{col} 로그 변환 실패: {e}")
                    
                    # 제곱근 변환
                    if feature_count < max_features:
                        train_non_negative = (X_train[col] >= 0) & X_train[col].notna()
                        test_non_negative = (X_test[col] >= 0) & X_test[col].notna()
                        
                        if train_non_negative.sum() > len(X_train) * 0.8:
                            try:
                                X_train[f'{col}_sqrt'] = np.where(train_non_negative, np.sqrt(X_train[col]), 0).astype('float32')
                                X_test[f'{col}_sqrt'] = np.where(test_non_negative, np.sqrt(X_test[col]), 0).astype('float32')
                                
                                self.generated_features.append(f'{col}_sqrt')
                                feature_count += 1
                            except Exception as e:
                                logger.warning(f"{col} 제곱근 변환 실패: {e}")
                    
                    if feature_count >= max_features:
                        break
                        
                except Exception as e:
                    logger.warning(f"수치형 피처 {col} 처리 실패: {e}")
            
            logger.info(f"수치형 피처 변환 완료: {feature_count}개")
            
        except Exception as e:
            logger.error(f"수치형 피처 변환 실패: {e}")
        
        return X_train, X_test
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """최종 데이터 정리"""
        logger.info("최종 데이터 정리 시작")
        
        try:
            # 공통 컬럼만 유지
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            
            # 컬럼 수 제한 (64GB 환경에서 더 많은 컬럼 허용)
            memory_status = self.memory_monitor.get_memory_status()
            max_cols = 250 if not memory_status['should_simplify'] else 150
            
            if len(common_columns) > max_cols:
                logger.warning(f"컬럼 수가 많아 {max_cols}개로 제한")
                selected_columns = common_columns[:max_cols]
                X_train = X_train[selected_columns]
                X_test = X_test[selected_columns]
            
            # 데이터 타입 강제 통일
            for col in X_train.columns:
                try:
                    train_dtype = X_train[col].dtype
                    test_dtype = X_test[col].dtype
                    
                    if train_dtype == 'object' or test_dtype == 'object':
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                    elif train_dtype != test_dtype or str(train_dtype) not in ['float32', 'int32', 'int16', 'int8', 'uint8', 'uint16']:
                        X_train[col] = X_train[col].astype('float32')
                        X_test[col] = X_test[col].astype('float32')
                        
                except Exception as e:
                    logger.warning(f"컬럼 {col} 타입 정리 실패: {e}")
                    try:
                        X_train[col] = 0.0
                        X_test[col] = 0.0
                    except Exception:
                        pass
            
            # 결측치 및 무한값 처리
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            X_train = X_train.replace([np.inf, -np.inf], [1e6, -1e6])
            X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # 최종 검증
            if list(X_train.columns) != list(X_test.columns):
                logger.warning("컬럼 불일치 감지, 재정렬 수행")
                common_columns = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_columns]
                X_test = X_test[common_columns]
            
            logger.info("최종 데이터 정리 완료")
            
        except Exception as e:
            logger.error(f"최종 데이터 정리 실패: {e}")
        
        return X_train, X_test
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """처리 완료"""
        try:
            processing_time = time.time() - self.processing_stats['start_time']
            memory_usage = self.memory_monitor.get_memory_usage()
            
            self.final_feature_columns = list(X_train.columns)
            
            # 피처 타입별 통계
            self.processing_stats['feature_types_count'] = {
                'interaction': len(self.interaction_features),
                'target_encoding': len(self.target_encoding_features),
                'temporal': len(self.temporal_features),
                'statistical': len(self.statistical_features),
                'frequency': len(self.frequency_features)
            }
            
            self.processing_stats.update({
                'total_features_generated': len(self.generated_features),
                'processing_time': processing_time,
                'memory_usage': memory_usage
            })
            
            logger.info(f"피처 엔지니어링 통계:")
            logger.info(f"  - 처리 시간: {processing_time:.2f}초")
            logger.info(f"  - 생성된 피처: {len(self.generated_features)}개")
            logger.info(f"  - 제거된 피처: {len(self.removed_columns)}개")
            logger.info(f"  - 최종 피처 수: {X_train.shape[1]}개")
            logger.info(f"  - 메모리 사용량: {memory_usage:.2f}GB")
            logger.info(f"  - 교차 피처: {len(self.interaction_features)}개")
            logger.info(f"  - 타겟 인코딩: {len(self.target_encoding_features)}개")
            logger.info(f"  - 시간 피처: {len(self.temporal_features)}개")
            logger.info(f"  - 통계 피처: {len(self.statistical_features)}개")
            logger.info(f"  - 빈도 피처: {len(self.frequency_features)}개")
            
            self.memory_monitor.force_memory_cleanup()
            
        except Exception as e:
            logger.warning(f"처리 완료 실패: {e}")
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 피처만 생성"""
        logger.warning("기본 피처만 생성")
        
        try:
            # 수치형 컬럼만 선택
            numeric_cols = []
            for col in train_df.columns:
                if col != target_col and train_df[col].dtype in ['int8', 'int16', 'int32', 'int64', 
                                                                'uint8', 'uint16', 'uint32', 'uint64',
                                                                'float16', 'float32', 'float64']:
                    numeric_cols.append(col)
                    if len(numeric_cols) >= 100:
                        break
            
            if not numeric_cols:
                numeric_cols = [col for col in train_df.columns if col != target_col][:100]
            
            X_train = train_df[numeric_cols].copy()
            X_test = test_df[numeric_cols].copy() if set(numeric_cols).issubset(test_df.columns) else test_df.iloc[:, :len(numeric_cols)].copy()
            
            # float32로 변환
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                    if col in X_test.columns:
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                except Exception:
                    X_train[col] = 0.0
                    if col in X_test.columns:
                        X_test[col] = 0.0
            
            logger.info(f"기본 피처만 생성 완료: {X_train.shape}")
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"기본 피처 생성 실패: {e}")
            n_train = len(train_df)
            n_test = len(test_df)
            
            X_train = pd.DataFrame({
                'dummy_feature_1': np.random.normal(0, 1, n_train).astype('float32'),
                'dummy_feature_2': np.random.uniform(0, 1, n_train).astype('float32'),
                'dummy_feature_3': np.ones(n_train, dtype='float32')
            })
            
            X_test = pd.DataFrame({
                'dummy_feature_1': np.random.normal(0, 1, n_test).astype('float32'),
                'dummy_feature_2': np.random.uniform(0, 1, n_test).astype('float32'),
                'dummy_feature_3': np.ones(n_test, dtype='float32')
            })
            
            return X_train, X_test
    
    def get_feature_columns_for_inference(self) -> List[str]:
        """추론에 사용할 피처 컬럼 순서 반환"""
        return self.final_feature_columns.copy() if self.final_feature_columns else []
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약 정보"""
        return {
            'total_generated_features': len(self.generated_features),
            'generated_features': self.generated_features,
            'removed_columns': self.removed_columns,
            'final_feature_columns': self.final_feature_columns,
            'original_feature_order': self.original_feature_order,
            'interaction_features': self.interaction_features,
            'target_encoding_features': self.target_encoding_features,
            'temporal_features': self.temporal_features,
            'statistical_features': self.statistical_features,
            'frequency_features': self.frequency_features,
            'processing_stats': self.processing_stats,
            'encoders_count': {
                'target_encoders': len(self.target_encoders),
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            }
        }

FeatureEngineer = CTRFeatureEngineer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 코드
    np.random.seed(42)
    n_samples = 50000
    
    train_data = {
        'clicked': np.random.binomial(1, 0.0191, n_samples),
        'feat_e_1': np.random.normal(0, 100, n_samples),
        'feat_e_2': np.random.normal(50, 25, n_samples),
        'feat_d_1': np.random.poisson(1, n_samples),
        'feat_d_2': np.random.poisson(2, n_samples),
        'feat_c_1': np.random.uniform(0, 10, n_samples),
        'feat_c_2': np.random.uniform(5, 15, n_samples),
        'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'user_id': [f'user_{i % 10000}' for i in range(n_samples)]
    }
    
    test_data = {col: val for col, val in train_data.items() if col != 'clicked'}
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # 피처 엔지니어링 테스트
    try:
        engineer = CTRFeatureEngineer()
        X_train, X_test = engineer.create_all_features(train_df, test_df)
        
        print(f"원본 데이터: {train_df.shape}, {test_df.shape}")
        print(f"처리 후: {X_train.shape}, {X_test.shape}")
        print(f"생성된 피처: {len(engineer.generated_features)}개")
        
        summary = engineer.get_feature_importance_summary()
        print(f"피처 타입별 생성 개수: {summary['processing_stats']['feature_types_count']}")
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")