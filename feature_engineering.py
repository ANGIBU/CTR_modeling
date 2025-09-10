# feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class MemoryOptimizedFeatureEngineer:
    """극한 메모리 최적화 피처 엔지니어링 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_encoders = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        self.generated_features = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.removed_columns = []
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """극한 메모리 최적화 피처 엔지니어링 파이프라인"""
        logger.info("극한 메모리 최적화 피처 엔지니어링 시작")
        
        # 학습 데이터에서 타겟 분리
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.copy()
        
        # 초기 메모리 상태
        initial_memory = self.get_memory_usage()
        logger.info(f"초기 메모리 사용량: {initial_memory:.2f} GB")
        
        # 컬럼 타입 사전 분류
        self._classify_columns(X_train)
        
        # 1. ID성 컬럼 강화 제거
        X_train, X_test = self._remove_id_columns_enhanced(X_train, X_test)
        
        # 2. 기본 피처 정리
        X_train, X_test = self._clean_basic_features(X_train, X_test)
        
        # 3. 범주형 피처 인코딩 (극한 최적화)
        X_train, X_test = self._encode_categorical_features_optimized(X_train, X_test, y_train)
        
        # 4. 수치형 피처 생성 (제한적)
        X_train, X_test = self._create_numerical_features_limited(X_train, X_test)
        
        # 5. 메모리 체크 후 조건부 상호작용 피처
        current_memory = self.get_memory_usage()
        logger.info(f"상호작용 피처 생성 전 메모리: {current_memory:.2f} GB")
        
        if current_memory < 45:
            X_train, X_test = self._create_minimal_interaction_features(X_train, X_test)
        else:
            logger.warning("메모리 부족으로 상호작용 피처 생성 건너뛰기")
        
        # 6. 최종 데이터 정리 및 검증
        X_train, X_test = self._final_data_cleanup(X_train, X_test, y_train)
        
        # 최종 상태
        final_memory = self.get_memory_usage()
        logger.info(f"최종 메모리 사용량: {final_memory:.2f} GB")
        logger.info(f"피처 엔지니어링 완료 - 최종 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def _remove_id_columns_enhanced(self, 
                                  X_train: pd.DataFrame, 
                                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """강화된 ID성 컬럼 제거"""
        logger.info("강화된 ID성 컬럼 제거 시작")
        
        # 명시적 ID 컬럼들
        explicit_id_columns = ['ID', 'id', 'seq', 'user_id', 'ad_id', 'session_id', 'request_id']
        
        # 실제 존재하는 명시적 ID 컬럼 찾기
        existing_id_columns = [col for col in explicit_id_columns if col in X_train.columns]
        
        # 패턴 기반 ID 컬럼 찾기
        pattern_id_columns = []
        for col in X_train.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in ['_id', 'id_', 'uuid', 'key', 'index']):
                pattern_id_columns.append(col)
        
        # 유니크 비율 기반 ID 컬럼 탐지 (샘플링)
        sample_size = min(100000, len(X_train))
        if len(X_train) > sample_size:
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            sample_data = X_train.iloc[sample_indices]
        else:
            sample_data = X_train
        
        high_cardinality_columns = []
        for col in X_train.columns:
            try:
                unique_ratio = sample_data[col].nunique() / len(sample_data)
                if unique_ratio > 0.7:  # 70% 이상 유니크
                    high_cardinality_columns.append(col)
                    logger.info(f"{col}: ID성 컬럼 후보 (유니크 비율: {unique_ratio:.3f})")
            except:
                continue
        
        # 모든 ID 컬럼 통합
        all_id_columns = list(set(existing_id_columns + pattern_id_columns + high_cardinality_columns))
        
        if all_id_columns:
            logger.info(f"제거할 ID성 컬럼들: {all_id_columns}")
            
            # 컬럼 제거
            X_train = X_train.drop(columns=all_id_columns, errors='ignore')
            X_test = X_test.drop(columns=all_id_columns, errors='ignore')
            
            # 제거된 컬럼 기록
            self.removed_columns.extend(all_id_columns)
            
            # 분류된 컬럼 목록에서도 제거
            self.numeric_columns = [col for col in self.numeric_columns if col not in all_id_columns]
            self.categorical_columns = [col for col in self.categorical_columns if col not in all_id_columns]
        
        logger.info(f"ID성 컬럼 제거 완료. 남은 컬럼 수: {X_train.shape[1]}")
        return X_train, X_test
    
    def _classify_columns(self, df: pd.DataFrame):
        """컬럼 타입 분류"""
        logger.info("컬럼 타입 수동 분류 시작")
        
        self.numeric_columns = []
        self.categorical_columns = []
        
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            
            if dtype_str in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                           'float16', 'float32', 'float64']:
                self.numeric_columns.append(col)
            elif dtype_str in ['object', 'category', 'string', 'bool']:
                self.categorical_columns.append(col)
            else:
                # 샘플링해서 타입 확인
                sample = df[col].dropna().iloc[:1000] if len(df[col].dropna()) > 0 else df[col].iloc[:1000]
                
                try:
                    pd.to_numeric(sample)
                    self.numeric_columns.append(col)
                except:
                    self.categorical_columns.append(col)
        
        logger.info(f"수치형 컬럼: {len(self.numeric_columns)}개, 범주형 컬럼: {len(self.categorical_columns)}개")
    
    def _clean_basic_features(self, 
                            X_train: pd.DataFrame, 
                            X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 피처 정리"""
        logger.info("기본 피처 정리 시작")
        
        cols_to_remove = []
        
        # 상수 컬럼 제거
        for col in X_train.columns:
            try:
                if X_train[col].nunique() <= 1:
                    cols_to_remove.append(col)
                    logger.info(f"{col}: 상수 컬럼 제거")
            except:
                continue
        
        # 컬럼 제거
        if cols_to_remove:
            X_train = X_train.drop(columns=cols_to_remove)
            X_test = X_test.drop(columns=cols_to_remove)
            
            # 제거된 컬럼 기록
            self.removed_columns.extend(cols_to_remove)
            
            # 분류된 컬럼 목록에서도 제거
            self.numeric_columns = [col for col in self.numeric_columns if col not in cols_to_remove]
            self.categorical_columns = [col for col in self.categorical_columns if col not in cols_to_remove]
        
        # 메모리 정리
        gc.collect()
        
        return X_train, X_test
    
    def _encode_categorical_features_optimized(self, 
                                             X_train: pd.DataFrame, 
                                             X_test: pd.DataFrame, 
                                             y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """극한 최적화 범주형 피처 인코딩"""
        logger.info("극한 최적화 범주형 피처 인코딩 시작")
        
        # 현재 남아있는 범주형 컬럼 재확인
        current_categorical_cols = []
        for col in X_train.columns:
            if str(X_train[col].dtype) in ['object', 'category', 'string']:
                current_categorical_cols.append(col)
        
        # 메모리 모니터링
        for col in current_categorical_cols:
            try:
                memory_before = self.get_memory_usage()
                logger.info(f"{col} 인코딩 시작 (메모리: {memory_before:.2f} GB)")
                
                unique_count = X_train[col].nunique()
                
                # 메모리 부족 방지
                if memory_before > 50:
                    logger.warning(f"{col}: 메모리 부족으로 인코딩 스킵")
                    continue
                
                # 고카디널리티 처리
                if unique_count > 10000:
                    top_categories = X_train[col].value_counts().head(100).index
                    X_train[col] = X_train[col].where(X_train[col].isin(top_categories), 'other')
                    X_test[col] = X_test[col].where(X_test[col].isin(top_categories), 'other')
                    logger.info(f"{col}: 초고카디널리티 - 상위 100개만 유지")
                    
                elif unique_count > 1000:
                    top_categories = X_train[col].value_counts().head(500).index
                    X_train[col] = X_train[col].where(X_train[col].isin(top_categories), 'other')
                    X_test[col] = X_test[col].where(X_test[col].isin(top_categories), 'other')
                    logger.info(f"{col}: 고카디널리티 - 상위 500개만 유지")
                
                # Label Encoding
                try:
                    le = LabelEncoder()
                    
                    # 결측치 처리
                    X_train[col] = X_train[col].fillna('missing')
                    X_test[col] = X_test[col].fillna('missing')
                    
                    # 전체 카테고리 수집
                    all_values = pd.concat([X_train[col], X_test[col]], ignore_index=True).unique()
                    le.fit(all_values.astype(str))
                    
                    X_train[f'{col}_encoded'] = le.transform(X_train[col].astype(str)).astype('int16')
                    X_test[f'{col}_encoded'] = le.transform(X_test[col].astype(str)).astype('int16')
                    
                    self.label_encoders[col] = le
                    self.generated_features.append(f'{col}_encoded')
                    
                except Exception as e:
                    logger.warning(f"{col} Label Encoding 실패: {str(e)}")
                    continue
                
                # 빈도 인코딩
                try:
                    freq_map = X_train[col].value_counts().to_dict()
                    X_train[f'{col}_freq'] = X_train[col].map(freq_map).fillna(0).astype('int16')
                    X_test[f'{col}_freq'] = X_test[col].map(freq_map).fillna(0).astype('int16')
                    self.generated_features.append(f'{col}_freq')
                    
                except Exception as e:
                    logger.warning(f"{col} 빈도 인코딩 실패: {str(e)}")
                
                # 메모리 정리
                try:
                    del freq_map, all_values
                except:
                    pass
                gc.collect()
                
                memory_after = self.get_memory_usage()
                logger.info(f"{col} 인코딩 완료 (메모리: {memory_after:.2f} GB)")
                
            except Exception as e:
                logger.error(f"{col} 인코딩 전체 실패: {str(e)}")
                continue
        
        # 원본 범주형 컬럼 제거
        try:
            existing_categorical = [col for col in current_categorical_cols if col in X_train.columns]
            if existing_categorical:
                X_train = X_train.drop(columns=existing_categorical)
                X_test = X_test.drop(columns=existing_categorical)
                logger.info(f"원본 범주형 컬럼 제거: {existing_categorical}")
        except Exception as e:
            logger.warning(f"범주형 컬럼 제거 실패: {str(e)}")
        
        # 메모리 정리
        gc.collect()
        
        logger.info(f"범주형 피처 인코딩 완료 - 생성된 피처: {len(self.generated_features)}개")
        return X_train, X_test
    
    def _create_numerical_features_limited(self, 
                                         X_train: pd.DataFrame, 
                                         X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """제한적 수치형 피처 생성"""
        logger.info("제한적 수치형 피처 생성 시작")
        
        # 메모리 체크
        current_memory = self.get_memory_usage()
        if current_memory > 40:
            logger.warning("메모리 부족으로 수치형 피처 생성 제한")
            max_features = 5
        else:
            max_features = 15
        
        # 현재 수치형 컬럼 재확인
        current_numeric_cols = []
        for col in X_train.columns:
            if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                    'float16', 'float32', 'float64']:
                current_numeric_cols.append(col)
        
        feature_count = 0
        
        for col in current_numeric_cols[:max_features]:
            try:
                # 메모리 체크
                if self.get_memory_usage() > 45:
                    break
                
                # 로그 변환 (양수 데이터만)
                if (X_train[col] > 0).all():
                    X_train[f'{col}_log'] = np.log1p(X_train[col]).astype('float32')
                    X_test[f'{col}_log'] = np.log1p(X_test[col]).astype('float32')
                    self.generated_features.append(f'{col}_log')
                    feature_count += 1
                
                # 이상치 플래그
                if feature_count < max_features:
                    q1 = X_train[col].quantile(0.25)
                    q3 = X_train[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    X_train[f'{col}_outlier'] = (
                        (X_train[col] < lower_bound) | (X_train[col] > upper_bound)
                    ).astype('int8')
                    
                    X_test[f'{col}_outlier'] = (
                        (X_test[col] < lower_bound) | (X_test[col] > upper_bound)
                    ).astype('int8')
                    
                    self.generated_features.append(f'{col}_outlier')
                    feature_count += 1
                
                # 메모리 정리
                if feature_count % 5 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.warning(f"{col} 수치형 피처 생성 실패: {str(e)}")
                continue
        
        logger.info(f"수치형 피처 생성 완료: {feature_count}개")
        return X_train, X_test
    
    def _create_minimal_interaction_features(self, 
                                           X_train: pd.DataFrame, 
                                           X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """최소한의 상호작용 피처 생성"""
        logger.info("최소한의 상호작용 피처 생성 시작")
        
        # 수치형 컬럼 안전하게 확인
        safe_numeric_cols = []
        for col in X_train.columns:
            if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                    'float16', 'float32', 'float64']:
                safe_numeric_cols.append(col)
        
        # 메모리 기반 제한
        current_memory = self.get_memory_usage()
        if current_memory > 35:
            max_interactions = 10
            max_cols = 5
        elif current_memory > 25:
            max_interactions = 30
            max_cols = 8
        else:
            max_interactions = 50
            max_cols = 10
        
        important_cols = safe_numeric_cols[:max_cols]
        interaction_count = 0
        
        logger.info(f"상호작용 피처 제한: {max_interactions}개, 컬럼: {max_cols}개")
        
        for i, col1 in enumerate(important_cols):
            if interaction_count >= max_interactions:
                break
            
            for j, col2 in enumerate(important_cols[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                try:
                    # 메모리 체크
                    if self.get_memory_usage() > 50:
                        logger.warning("메모리 한계로 상호작용 피처 생성 중단")
                        break
                    
                    # 곱셈 피처만 생성
                    X_train[f'{col1}_x_{col2}'] = (X_train[col1] * X_train[col2]).astype('float32')
                    X_test[f'{col1}_x_{col2}'] = (X_test[col1] * X_test[col2]).astype('float32')
                    
                    self.generated_features.append(f'{col1}_x_{col2}')
                    interaction_count += 1
                    
                    # 메모리 정리
                    if interaction_count % 10 == 0:
                        gc.collect()
                        current_mem = self.get_memory_usage()
                        logger.info(f"상호작용 피처 진행: {interaction_count}/{max_interactions}, 메모리: {current_mem:.2f} GB")
                
                except Exception as e:
                    logger.warning(f"{col1}, {col2} 상호작용 피처 실패: {str(e)}")
                    continue
        
        logger.info(f"상호작용 피처 생성 완료: {interaction_count}개")
        return X_train, X_test
    
    def _final_data_cleanup(self, 
                          X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, 
                          y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """최종 데이터 정리 및 검증"""
        logger.info("최종 데이터 정리 시작")
        
        # 1. 데이터 타입 강제 통일
        problematic_columns = []
        
        for col in X_train.columns:
            try:
                # X_train과 X_test의 데이터 타입 확인
                train_dtype = X_train[col].dtype
                test_dtype = X_test[col].dtype
                
                # object 타입 처리
                if train_dtype == 'object' or test_dtype == 'object':
                    logger.warning(f"{col}: object 타입 발견, 수치형으로 강제 변환")
                    
                    # 수치형 변환 시도
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                    except:
                        # 변환 실패 시 제거 대상으로 추가
                        problematic_columns.append(col)
                        continue
                
                # 카테고리 타입 처리
                elif train_dtype.name == 'category' or test_dtype.name == 'category':
                    logger.warning(f"{col}: category 타입 발견, 수치형으로 변환")
                    try:
                        if train_dtype.name == 'category':
                            X_train[col] = X_train[col].cat.codes.astype('float32')
                        if test_dtype.name == 'category':
                            X_test[col] = X_test[col].cat.codes.astype('float32')
                    except:
                        problematic_columns.append(col)
                        continue
                
                # 수치형 타입 통일 (float32)
                elif train_dtype != test_dtype or train_dtype != 'float32':
                    X_train[col] = X_train[col].astype('float32')
                    X_test[col] = X_test[col].astype('float32')
            
            except Exception as e:
                logger.error(f"{col} 데이터 타입 처리 실패: {str(e)}")
                problematic_columns.append(col)
        
        # 2. 문제가 있는 컬럼 제거
        if problematic_columns:
            logger.warning(f"문제가 있는 컬럼 제거: {problematic_columns}")
            X_train = X_train.drop(columns=problematic_columns, errors='ignore')
            X_test = X_test.drop(columns=problematic_columns, errors='ignore')
            self.removed_columns.extend(problematic_columns)
        
        # 3. 결측치 및 무한값 처리
        # 결측치 처리
        if X_train.isnull().any().any():
            logger.warning("X_train에 결측치 발견, 0으로 대치")
            X_train = X_train.fillna(0)
        
        if X_test.isnull().any().any():
            logger.warning("X_test에 결측치 발견, 0으로 대치")
            X_test = X_test.fillna(0)
        
        # 무한값 처리
        try:
            # X_train 무한값 처리
            inf_mask_train = np.isinf(X_train.values)
            if inf_mask_train.any():
                logger.warning("X_train에 무한값 발견, 클리핑 처리")
                X_train = X_train.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # X_test 무한값 처리
            inf_mask_test = np.isinf(X_test.values)
            if inf_mask_test.any():
                logger.warning("X_test에 무한값 발견, 클리핑 처리")
                X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
                
        except Exception as e:
            logger.warning(f"무한값 처리 중 오류: {str(e)}")
        
        # 4. 최종 검증
        logger.info("최종 데이터 검증 시작")
        
        # 컬럼 일치성 확인
        if list(X_train.columns) != list(X_test.columns):
            logger.error("X_train과 X_test의 컬럼이 일치하지 않습니다!")
            # 공통 컬럼만 유지
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            logger.info(f"공통 컬럼만 유지: {len(common_columns)}개")
        
        # 데이터 타입 확인
        for col in X_train.columns:
            train_dtype = X_train[col].dtype
            test_dtype = X_test[col].dtype
            
            if train_dtype != test_dtype:
                logger.warning(f"{col}: 데이터 타입 불일치 - train: {train_dtype}, test: {test_dtype}")
                # 더 안전한 타입으로 통일
                X_train[col] = X_train[col].astype('float32')
                X_test[col] = X_test[col].astype('float32')
            
            # object 타입이 남아있는지 확인
            if train_dtype == 'object' or test_dtype == 'object':
                logger.error(f"{col}: object 타입이 여전히 남아있음!")
                # 강제 제거
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        
        # 5. 메모리 기반 피처 선택
        current_memory = self.get_memory_usage()
        
        if current_memory > 55 or X_train.shape[1] > 1000:
            logger.info("메모리 절약을 위한 피처 선택 수행")
            
            try:
                # 분산 기반 필터링
                variances = X_train.var()
                high_var_features = variances[variances > 1e-6].index.tolist()
                
                if len(high_var_features) < X_train.shape[1]:
                    X_train = X_train[high_var_features]
                    X_test = X_test[high_var_features]
                    logger.info(f"분산 기반 피처 선택: {len(high_var_features)}개 유지")
                
            except Exception as e:
                logger.warning(f"피처 선택 실패: {str(e)}")
        
        # 최종 데이터 타입 확인
        logger.info(f"X_train 최종 데이터 타입: {X_train.dtypes.value_counts().to_dict()}")
        logger.info(f"X_test 최종 데이터 타입: {X_test.dtypes.value_counts().to_dict()}")
        
        # object 타입이 남아있는지 최종 확인
        train_object_cols = X_train.select_dtypes(include=['object']).columns
        test_object_cols = X_test.select_dtypes(include=['object']).columns
        
        if len(train_object_cols) > 0 or len(test_object_cols) > 0:
            logger.error(f"object 타입 컬럼이 남아있습니다! train: {list(train_object_cols)}, test: {list(test_object_cols)}")
            # 모든 object 컬럼 제거
            all_object_cols = list(set(train_object_cols) | set(test_object_cols))
            X_train = X_train.drop(columns=all_object_cols, errors='ignore')
            X_test = X_test.drop(columns=all_object_cols, errors='ignore')
            logger.info(f"object 컬럼 강제 제거: {all_object_cols}")
        
        # 메모리 정리
        gc.collect()
        
        logger.info("최종 데이터 정리 완료")
        return X_train, X_test
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약 정보"""
        return {
            'total_generated_features': len(self.generated_features),
            'generated_features': self.generated_features,
            'removed_columns': self.removed_columns,
            'encoders_count': {
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            }
        }

# 기존 FeatureEngineer 클래스를 최적화된 버전으로 교체
FeatureEngineer = MemoryOptimizedFeatureEngineer