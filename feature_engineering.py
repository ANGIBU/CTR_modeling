# feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
import gc
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Psutil import 안전 처리
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config import Config

# 안전한 로거 초기화
def get_safe_logger(name: str):
    """안전한 로거 생성"""
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
    """메모리 모니터링 클래스"""
    
    def __init__(self):
        self.monitoring_enabled = PSUTIL_AVAILABLE
        self.lock = threading.Lock()
        self._last_check_time = 0
        self._check_interval = 10.0
    
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
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
        """사용 가능한 메모리 (GB)"""
        if not self.monitoring_enabled:
            return 20.0
        
        try:
            with self.lock:
                return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 20.0
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        try:
            available = self.get_available_memory()
            return available < 5.0
        except Exception:
            return False
    
    def force_memory_cleanup(self):
        """강제 메모리 정리"""
        try:
            gc.collect()
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
            usage = self.get_memory_usage()
            available = self.get_available_memory()
            
            if context:
                logger.info(f"메모리 상태 [{context}]: 사용 {usage:.1f}GB, 가용 {available:.1f}GB")
                
        except Exception as e:
            logger.warning(f"메모리 상태 로깅 실패: {e}")

class CTRFeatureEngineer:
    """CTR 예측에 특화된 피처 엔지니어링 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
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
        
        # 메모리 효율성 설정
        self.memory_efficient_mode = True
        self.freq_encoders = {}
        self.statistical_features = {}
        
        # 성능 통계
        self.processing_stats = {
            'start_time': time.time(),
            'total_features_generated': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0
        }
        
        logger.info("CTR 피처 엔지니어 초기화 완료")
    
    def set_memory_efficient_mode(self, enabled: bool = True):
        """메모리 효율 모드 설정"""
        self.memory_efficient_mode = enabled
        mode_str = "활성화" if enabled else "비활성화"
        logger.info(f"메모리 효율 모드 {mode_str}")
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CTR 특화 피처 엔지니어링 파이프라인"""
        logger.info("=== CTR 피처 엔지니어링 시작 ===")
        
        try:
            # 초기 설정
            self._initialize_processing(train_df, test_df, target_col)
            
            # 1. 기본 데이터 준비
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            # 2. 컬럼 분류
            self._classify_columns_safe(X_train)
            
            # 3. 데이터 타입 통일
            X_train, X_test = self._unify_data_types_safe(X_train, X_test)
            
            # 4. ID 피처 처리 (메모리 효율 모드에서는 간단하게)
            if not self.memory_efficient_mode:
                X_train, X_test = self._process_id_features_safe(X_train, X_test)
            
            # 5. 기본 피처 정리
            X_train, X_test = self._clean_basic_features_safe(X_train, X_test)
            
            # 6. 범주형 피처 인코딩
            X_train, X_test = self._encode_categorical_features_safe(X_train, X_test, y_train)
            
            # 7. 수치형 피처 생성 (메모리 효율 모드에서는 제한적)
            if not self.memory_efficient_mode:
                X_train, X_test = self._create_numeric_features_safe(X_train, X_test)
            
            # 8. CTR 특화 피처 생성
            X_train, X_test = self._create_ctr_features_safe(X_train, X_test, y_train)
            
            # 9. 최종 데이터 정리
            X_train, X_test = self._final_data_cleanup_safe(X_train, X_test)
            
            # 10. 처리 완료
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== CTR 피처 엔지니어링 완료: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"피처 엔지니어링 실패: {e}")
            self.memory_monitor.force_memory_cleanup()
            raise
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """처리 초기화"""
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
            # 타겟 컬럼 확인
            if target_col not in train_df.columns:
                available_targets = [col for col in train_df.columns if 'click' in col.lower()]
                if available_targets:
                    target_col = available_targets[0]
                    logger.warning(f"타겟 컬럼 변경: {target_col}")
                else:
                    raise ValueError(f"타겟 컬럼 '{target_col}'을 찾을 수 없습니다")
            
            # 데이터 분리
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
    
    def _classify_columns_safe(self, df: pd.DataFrame):
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
                        self.id_columns.append(col)
                    elif dtype_str in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                     'float16', 'float32', 'float64']:
                        # 수치형 컬럼 중 고유값 비율로 ID 여부 판단
                        try:
                            unique_ratio = df[col].nunique() / len(df)
                            if unique_ratio > 0.9:
                                self.id_columns.append(col)
                            else:
                                self.numeric_columns.append(col)
                        except Exception:
                            self.numeric_columns.append(col)
                    else:
                        self.categorical_columns.append(col)
                        
                except Exception as e:
                    logger.warning(f"컬럼 {col} 분류 실패: {e}")
                    self.numeric_columns.append(col)
            
            logger.info(f"컬럼 분류 완료 - 수치형: {len(self.numeric_columns)}, "
                       f"범주형: {len(self.categorical_columns)}, ID: {len(self.id_columns)}")
            
        except Exception as e:
            logger.error(f"컬럼 분류 전체 실패: {e}")
            self.numeric_columns = list(df.columns)
            self.categorical_columns = []
            self.id_columns = []
    
    def _unify_data_types_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 타입 통일"""
        logger.info("데이터 타입 통일 시작")
        
        try:
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            processed_count = 0
            
            for col in common_columns:
                try:
                    train_dtype = str(X_train[col].dtype)
                    test_dtype = str(X_test[col].dtype)
                    
                    # 특수 컬럼 처리 (seq 등)
                    if col == 'seq' or 'seq' in str(col).lower():
                        X_train[col] = self._safe_hash_column(X_train[col])
                        X_test[col] = self._safe_hash_column(X_test[col])
                        processed_count += 1
                        continue
                    
                    # 타입 불일치 해결
                    if train_dtype != test_dtype or train_dtype in ['object', 'category']:
                        # 수치형 변환 시도
                        try:
                            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                        except Exception:
                            # 해시 변환
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
                    
                    # 주기적 메모리 정리
                    if processed_count % 50 == 0:
                        self.memory_monitor.force_memory_cleanup()
                        
                except Exception as e:
                    logger.warning(f"컬럼 {col} 타입 통일 실패: {e}")
                    try:
                        X_train[col] = 0.0
                        X_test[col] = 0.0
                    except Exception:
                        pass
            
            logger.info(f"데이터 타입 통일 완료: {processed_count}/{len(common_columns)}개 컬럼")
            
        except Exception as e:
            logger.error(f"데이터 타입 통일 전체 실패: {e}")
        
        return X_train, X_test
    
    def _safe_hash_column(self, series: pd.Series) -> pd.Series:
        """안전한 컬럼 해시 변환"""
        try:
            def safe_hash(x):
                try:
                    if pd.isna(x):
                        return 0
                    str_val = str(x)[:50]
                    return hash(str_val) % 100000
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
    
    def _process_id_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ID 피처 처리"""
        if not self.id_columns:
            return X_train, X_test
        
        logger.info(f"ID 피처 처리 시작: {len(self.id_columns)}개")
        
        try:
            for col in self.id_columns[:5]:  # 최대 5개만 처리
                if col not in X_train.columns:
                    continue
                
                try:
                    # 해시 피처 생성
                    train_hash = self._safe_hash_column(X_train[col])
                    test_hash = self._safe_hash_column(X_test[col])
                    
                    X_train[f'{col}_hash'] = train_hash
                    X_test[f'{col}_hash'] = test_hash
                    self.generated_features.append(f'{col}_hash')
                    
                except Exception as e:
                    logger.warning(f"ID 피처 {col} 처리 실패: {e}")
            
            # 원본 ID 컬럼 제거
            existing_id_cols = [col for col in self.id_columns if col in X_train.columns]
            if existing_id_cols:
                X_train = X_train.drop(columns=existing_id_cols)
                X_test = X_test.drop(columns=existing_id_cols)
                self.removed_columns.extend(existing_id_cols)
            
            logger.info(f"ID 피처 처리 완료: {len([f for f in self.generated_features if 'hash' in f])}개 생성")
            
        except Exception as e:
            logger.error(f"ID 피처 처리 실패: {e}")
        
        return X_train, X_test
    
    def _clean_basic_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 피처 정리"""
        logger.info("기본 피처 정리 시작")
        
        try:
            cols_to_remove = []
            
            for col in X_train.columns:
                try:
                    if X_train[col].nunique() <= 1:
                        cols_to_remove.append(col)
                except Exception:
                    continue
            
            if cols_to_remove:
                X_train = X_train.drop(columns=cols_to_remove)
                X_test = X_test.drop(columns=cols_to_remove)
                self.removed_columns.extend(cols_to_remove)
                logger.info(f"상수 컬럼 {len(cols_to_remove)}개 제거")
            
        except Exception as e:
            logger.warning(f"기본 피처 정리 실패: {e}")
        
        return X_train, X_test
    
    def _encode_categorical_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
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
            for col in current_categorical_cols:
                try:
                    # 문자열 변환
                    train_values = X_train[col].astype(str).fillna('missing')
                    test_values = X_test[col].astype(str).fillna('missing')
                    
                    # 고카디널리티 처리
                    unique_count = len(train_values.unique())
                    max_categories = 20 if self.memory_efficient_mode else 100
                    
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
                    
                    # 빈도 인코딩
                    try:
                        freq_map = train_values.value_counts().to_dict()
                        X_train[f'{col}_freq'] = train_values.map(freq_map).fillna(0).astype('int16')
                        X_test[f'{col}_freq'] = test_values.map(freq_map).fillna(0).astype('int16')
                        
                        self.freq_encoders[col] = freq_map
                        self.generated_features.append(f'{col}_freq')
                        
                    except Exception as e:
                        logger.warning(f"{col} 빈도 인코딩 실패: {e}")
                    
                except Exception as e:
                    logger.warning(f"범주형 피처 {col} 처리 실패: {e}")
                
                # 주기적 메모리 정리
                if len(self.generated_features) % 10 == 0:
                    self.memory_monitor.force_memory_cleanup()
            
            # 원본 범주형 컬럼 제거
            try:
                existing_categorical = [col for col in current_categorical_cols if col in X_train.columns]
                if existing_categorical:
                    X_train = X_train.drop(columns=existing_categorical)
                    X_test = X_test.drop(columns=existing_categorical)
            except Exception as e:
                logger.warning(f"범주형 컬럼 제거 실패: {e}")
            
            logger.info(f"범주형 피처 인코딩 완료: {len([f for f in self.generated_features if 'encoded' in f or 'freq' in f])}개 생성")
            
        except Exception as e:
            logger.error(f"범주형 피처 인코딩 실패: {e}")
        
        return X_train, X_test
    
    def _create_numeric_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """수치형 피처 생성"""
        logger.info("수치형 피처 생성 시작")
        
        try:
            current_numeric_cols = [col for col in X_train.columns 
                                  if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                                           'float16', 'float32', 'float64']]
            
            if not current_numeric_cols:
                return X_train, X_test
            
            feature_count = 0
            max_features = 5 if self.memory_efficient_mode else 15
            
            for col in current_numeric_cols[:10]:
                try:
                    if self.memory_monitor.check_memory_pressure():
                        logger.warning("메모리 압박으로 수치형 피처 생성 중단")
                        break
                    
                    # 로그 변환
                    train_positive = (X_train[col] > 0) & X_train[col].notna()
                    test_positive = (X_test[col] > 0) & X_test[col].notna()
                    
                    if train_positive.sum() > len(X_train) * 0.8:
                        try:
                            X_train[f'{col}_log'] = np.where(train_positive, np.log1p(X_train[col]), 0).astype('float32')
                            X_test[f'{col}_log'] = np.where(test_positive, np.log1p(X_test[col]), 0).astype('float32')
                            
                            self.generated_features.append(f'{col}_log')
                            feature_count += 1
                        except Exception as e:
                            logger.warning(f"{col} 로그 변환 실패: {e}")
                    
                    if feature_count >= max_features:
                        break
                        
                except Exception as e:
                    logger.warning(f"수치형 피처 {col} 처리 실패: {e}")
            
            logger.info(f"수치형 피처 생성 완료: {feature_count}개")
            
        except Exception as e:
            logger.error(f"수치형 피처 생성 실패: {e}")
        
        return X_train, X_test
    
    def _create_ctr_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CTR 특화 피처 생성"""
        logger.info("CTR 특화 피처 생성 시작")
        
        try:
            # 시간적 피처
            X_train['time_index'] = (X_train.index / len(X_train)).astype('float32')
            X_test['time_index'] = (X_test.index / len(X_test)).astype('float32')
            
            X_train['position_quartile'] = pd.qcut(X_train.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            X_test['position_quartile'] = pd.qcut(X_test.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            
            self.generated_features.extend(['time_index', 'position_quartile'])
            
            # 피처 그룹별 통계 (메모리 효율 모드가 아닐 때만)
            if not self.memory_efficient_mode:
                try:
                    self._create_feature_group_statistics(X_train, X_test)
                except Exception as e:
                    logger.warning(f"피처 그룹 통계 생성 실패: {e}")
            
            logger.info("CTR 특화 피처 생성 완료")
            
        except Exception as e:
            logger.error(f"CTR 특화 피처 생성 실패: {e}")
        
        return X_train, X_test
    
    def _create_feature_group_statistics(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """피처 그룹별 통계 생성"""
        feature_groups = {
            'feat_e': [col for col in X_train.columns if col.startswith('feat_e')],
            'feat_d': [col for col in X_train.columns if col.startswith('feat_d')],
            'feat_c': [col for col in X_train.columns if col.startswith('feat_c')]
        }
        
        for group_name, group_cols in feature_groups.items():
            if len(group_cols) > 1:
                try:
                    # 그룹 합계
                    X_train[f'{group_name}_sum'] = X_train[group_cols].sum(axis=1).astype('float32')
                    X_test[f'{group_name}_sum'] = X_test[group_cols].sum(axis=1).astype('float32')
                    
                    # 그룹 평균
                    X_train[f'{group_name}_mean'] = X_train[group_cols].mean(axis=1).astype('float32')
                    X_test[f'{group_name}_mean'] = X_test[group_cols].mean(axis=1).astype('float32')
                    
                    self.generated_features.extend([
                        f'{group_name}_sum', f'{group_name}_mean'
                    ])
                    
                except Exception as e:
                    logger.warning(f"{group_name} 그룹 통계 생성 실패: {e}")
    
    def _final_data_cleanup_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """최종 데이터 정리"""
        logger.info("최종 데이터 정리 시작")
        
        try:
            # 공통 컬럼만 유지
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            
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
            
            self.memory_monitor.force_memory_cleanup()
            
        except Exception as e:
            logger.warning(f"처리 완료 실패: {e}")
    
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
            'id_columns_processed': self.id_columns,
            'memory_efficient_mode': self.memory_efficient_mode,
            'processing_stats': self.processing_stats,
            'encoders_count': {
                'label_encoders': len(self.label_encoders),
                'freq_encoders': len(self.freq_encoders),
                'scalers': len(self.scalers)
            }
        }

# 기존 코드와의 호환성을 위한 별칭
FeatureEngineer = CTRFeatureEngineer

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 10000
    
    train_data = {
        'clicked': np.random.binomial(1, 0.02, n_samples),
        'feat_e_1': np.random.normal(0, 100, n_samples),
        'feat_c_1': np.random.poisson(1, n_samples),
        'feat_b_1': np.random.uniform(0, 10, n_samples),
        'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'user_id': [f'user_{i % 1000}' for i in range(n_samples)]
    }
    
    test_data = {col: val for col, val in train_data.items() if col != 'clicked'}
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # 피처 엔지니어링 테스트
    try:
        engineer = CTRFeatureEngineer()
        engineer.set_memory_efficient_mode(True)
        X_train, X_test = engineer.create_all_features(train_df, test_df)
        
        print(f"원본 데이터: {train_df.shape}, {test_df.shape}")
        print(f"처리 후: {X_train.shape}, {X_test.shape}")
        print(f"생성된 피처: {len(engineer.generated_features)}개")
        
        summary = engineer.get_feature_importance_summary()
        print(f"처리 통계: {summary['processing_stats']}")
        
    except Exception as e:
        logger.error(f"테스트 실행 실패: {e}")