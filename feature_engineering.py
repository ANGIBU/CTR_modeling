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
        
        # 컬럼 타입 사전 분류 (select_dtypes 사용하지 않음)
        self._classify_columns(X_train)
        
        # 1. 기본 피처 정리
        X_train, X_test = self._clean_basic_features(X_train, X_test)
        
        # 2. 범주형 피처 인코딩 (극한 최적화)
        X_train, X_test = self._encode_categorical_features_optimized(X_train, X_test, y_train)
        
        # 3. 수치형 피처 생성 (제한적)
        X_train, X_test = self._create_numerical_features_limited(X_train, X_test)
        
        # 4. 메모리 체크 후 조건부 상호작용 피처
        current_memory = self.get_memory_usage()
        logger.info(f"상호작용 피처 생성 전 메모리: {current_memory:.2f} GB")
        
        if current_memory < 45:  # 45GB 미만일 때만 상호작용 피처 생성
            X_train, X_test = self._create_minimal_interaction_features(X_train, X_test)
        else:
            logger.warning("메모리 부족으로 상호작용 피처 생성 건너뛰기")
        
        # 5. 최종 최적화
        X_train, X_test = self._final_optimization(X_train, X_test, y_train)
        
        # 최종 상태
        final_memory = self.get_memory_usage()
        logger.info(f"최종 메모리 사용량: {final_memory:.2f} GB")
        logger.info(f"피처 엔지니어링 완료 - 최종 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def _classify_columns(self, df: pd.DataFrame):
        """컬럼 타입 분류 (select_dtypes 대신 수동 분류)"""
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
        """기본 피처 정리 (메모리 효율적)"""
        logger.info("기본 피처 정리 시작")
        
        # 메모리 절약을 위한 청크 처리
        chunk_size = 1000000  # 100만 행씩 처리
        
        # ID성 컬럼 및 상수 컬럼 제거
        cols_to_remove = []
        
        for col in X_train.columns:
            # 유니크 비율 확인 (샘플링)
            if len(X_train) > chunk_size:
                sample_size = min(chunk_size, len(X_train))
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                sample_data = X_train.iloc[sample_indices]
            else:
                sample_data = X_train
            
            unique_ratio = sample_data[col].nunique() / len(sample_data)
            
            # ID성 컬럼 (유니크 비율 70% 이상)
            if unique_ratio > 0.7:
                cols_to_remove.append(col)
                logger.info(f"{col}: ID성 컬럼 제거 (유니크 비율: {unique_ratio:.3f})")
            
            # 상수 컬럼
            elif sample_data[col].nunique() <= 1:
                cols_to_remove.append(col)
                logger.info(f"{col}: 상수 컬럼 제거")
        
        # 컬럼 제거
        if cols_to_remove:
            X_train = X_train.drop(columns=cols_to_remove)
            X_test = X_test.drop(columns=cols_to_remove)
            
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
        
        # 메모리 모니터링
        for col in self.categorical_columns[:]:  # 복사본으로 반복
            try:
                memory_before = self.get_memory_usage()
                logger.info(f"{col} 인코딩 시작 (메모리: {memory_before:.2f} GB)")
                
                unique_count = X_train[col].nunique()
                
                # 메모리 부족 방지를 위한 조건부 처리
                if memory_before > 50:  # 50GB 초과 시 인코딩 스킵
                    logger.warning(f"{col}: 메모리 부족으로 인코딩 스킵")
                    continue
                
                if unique_count > 10000:  # 초고카디널리티
                    logger.info(f"{col}: 초고카디널리티 컬럼 - 상위 100개만 유지")
                    top_categories = X_train[col].value_counts().head(100).index
                    X_train[col] = X_train[col].where(X_train[col].isin(top_categories), 'other')
                    X_test[col] = X_test[col].where(X_test[col].isin(top_categories), 'other')
                
                elif unique_count > 1000:  # 고카디널리티
                    logger.info(f"{col}: 고카디널리티 컬럼 - 상위 500개만 유지")
                    top_categories = X_train[col].value_counts().head(500).index
                    X_train[col] = X_train[col].where(X_train[col].isin(top_categories), 'other')
                    X_test[col] = X_test[col].where(X_test[col].isin(top_categories), 'other')
                
                # Label Encoding만 수행 (메모리 절약)
                le = LabelEncoder()
                
                # 전체 카테고리 수집 (메모리 효율적)
                try:
                    all_values = pd.concat([X_train[col], X_test[col]], ignore_index=True).unique()
                    le.fit(all_values.astype(str))
                    
                    X_train[f'{col}_encoded'] = le.transform(X_train[col].astype(str)).astype('int16')
                    X_test[f'{col}_encoded'] = le.transform(X_test[col].astype(str)).astype('int16')
                    
                    self.label_encoders[col] = le
                    self.generated_features.append(f'{col}_encoded')
                    
                except Exception as e:
                    logger.warning(f"{col} Label Encoding 실패: {str(e)}")
                    continue
                
                # 빈도 인코딩 (메모리 절약 버전)
                try:
                    freq_map = X_train[col].value_counts().to_dict()
                    X_train[f'{col}_freq'] = X_train[col].map(freq_map).fillna(0).astype('int16')
                    X_test[f'{col}_freq'] = X_test[col].map(freq_map).fillna(0).astype('int16')
                    self.generated_features.append(f'{col}_freq')
                    
                except Exception as e:
                    logger.warning(f"{col} 빈도 인코딩 실패: {str(e)}")
                
                # 메모리 정리
                del freq_map, all_values
                gc.collect()
                
                memory_after = self.get_memory_usage()
                logger.info(f"{col} 인코딩 완료 (메모리: {memory_after:.2f} GB)")
                
            except Exception as e:
                logger.error(f"{col} 인코딩 전체 실패: {str(e)}")
                continue
        
        # 원본 범주형 컬럼 제거
        try:
            X_train = X_train.drop(columns=self.categorical_columns)
            X_test = X_test.drop(columns=self.categorical_columns)
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
        if current_memory > 40:  # 40GB 초과 시 수치형 피처 생성 제한
            logger.warning("메모리 부족으로 수치형 피처 생성 제한")
            max_features = 5
        else:
            max_features = 15
        
        # 현재 수치형 컬럼 재확인 (타입 안전)
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
                    
                    # 곱셈 피처만 생성 (나눗셈은 제외)
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
    
    def _final_optimization(self, 
                          X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, 
                          y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """최종 최적화"""
        logger.info("최종 최적화 시작")
        
        # 데이터 타입 최적화
        for col in X_train.columns:
            if X_train[col].dtype == 'int64':
                if X_train[col].min() >= 0:
                    if X_train[col].max() < 65535:
                        X_train[col] = X_train[col].astype('uint16')
                        X_test[col] = X_test[col].astype('uint16')
            elif X_train[col].dtype == 'float64':
                X_train[col] = X_train[col].astype('float32')
                X_test[col] = X_test[col].astype('float32')
        
        # 메모리 기반 피처 선택
        current_memory = self.get_memory_usage()
        
        if current_memory > 55 or X_train.shape[1] > 1000:
            logger.info("메모리 절약을 위한 피처 선택 수행")
            
            # 간단한 피처 선택
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
        
        # 메모리 정리
        gc.collect()
        
        logger.info("최종 최적화 완료")
        return X_train, X_test
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약 정보"""
        return {
            'total_generated_features': len(self.generated_features),
            'generated_features': self.generated_features,
            'encoders_count': {
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            }
        }

# 기존 FeatureEngineer 클래스를 최적화된 버전으로 교체
FeatureEngineer = MemoryOptimizedFeatureEngineer