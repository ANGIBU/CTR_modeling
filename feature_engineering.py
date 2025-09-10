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
    """메모리 최적화 피처 엔지니어링 클래스"""
    
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
        self.final_feature_columns = []  # 최종 피처 컬럼 순서 저장
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """메모리 최적화 피처 엔지니어링 파이프라인"""
        logger.info("메모리 최적화 피처 엔지니어링 시작")
        
        # 학습 데이터에서 타겟 분리
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.copy()
        
        # 초기 메모리 상태
        initial_memory = self.get_memory_usage()
        logger.info(f"초기 메모리 사용량: {initial_memory:.2f} GB")
        
        # 컬럼 타입 사전 분류
        self._classify_columns(X_train)
        
        # 1. ID성 컬럼 제거
        X_train, X_test = self._remove_id_columns_consistent(X_train, X_test)
        
        # 2. 기본 피처 정리
        X_train, X_test = self._clean_basic_features_consistent(X_train, X_test)
        
        # 3. 범주형 피처 인코딩
        X_train, X_test = self._encode_categorical_features_consistent(X_train, X_test, y_train)
        
        # 4. 수치형 피처 생성
        X_train, X_test = self._create_numerical_features_consistent(X_train, X_test)
        
        # 5. 상호작용 피처 생성
        current_memory = self.get_memory_usage()
        logger.info(f"상호작용 피처 생성 전 메모리: {current_memory:.2f} GB")
        
        if current_memory < 35:
            X_train, X_test = self._create_interaction_features_consistent(X_train, X_test)
        else:
            logger.warning("메모리 부족으로 상호작용 피처 생성 건너뛰기")
        
        # 6. 최종 데이터 정리 및 검증
        X_train, X_test = self._final_data_cleanup_consistent(X_train, X_test, y_train)
        
        # 7. 피처 컬럼 순서 일치 보장
        X_train, X_test = self._ensure_feature_consistency(X_train, X_test)
        
        # 최종 상태
        final_memory = self.get_memory_usage()
        logger.info(f"최종 메모리 사용량: {final_memory:.2f} GB")
        logger.info(f"피처 엔지니어링 완료 - 최종 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def _classify_columns(self, df: pd.DataFrame):
        """컬럼 타입 분류"""
        logger.info("컬럼 타입 분류 시작")
        
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
                try:
                    sample = df[col].dropna().iloc[:1000] if len(df[col].dropna()) > 0 else df[col].iloc[:1000]
                    pd.to_numeric(sample)
                    self.numeric_columns.append(col)
                except:
                    self.categorical_columns.append(col)
        
        logger.info(f"수치형 컬럼: {len(self.numeric_columns)}개, 범주형 컬럼: {len(self.categorical_columns)}개")
    
    def _remove_id_columns_consistent(self, 
                                    X_train: pd.DataFrame, 
                                    X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """일관된 ID성 컬럼 제거"""
        logger.info("일관된 ID성 컬럼 제거 시작")
        
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
        
        # 유니크 비율 기반 ID 컬럼 탐지 (학습 데이터만 기준)
        sample_size = min(50000, len(X_train))
        if len(X_train) > sample_size:
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            sample_data = X_train.iloc[sample_indices]
        else:
            sample_data = X_train
        
        high_cardinality_columns = []
        for col in X_train.columns:
            try:
                unique_ratio = sample_data[col].nunique() / len(sample_data)
                if unique_ratio > 0.8:
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
    
    def _clean_basic_features_consistent(self, 
                                       X_train: pd.DataFrame, 
                                       X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """일관된 기본 피처 정리"""
        logger.info("일관된 기본 피처 정리 시작")
        
        cols_to_remove = []
        
        # 상수 컬럼 제거 (학습 데이터 기준)
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
    
    def _encode_categorical_features_consistent(self, 
                                              X_train: pd.DataFrame, 
                                              X_test: pd.DataFrame, 
                                              y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """일관된 범주형 피처 인코딩"""
        logger.info("일관된 범주형 피처 인코딩 시작")
        
        # 현재 남아있는 범주형 컬럼 재확인
        current_categorical_cols = []
        for col in X_train.columns:
            if str(X_train[col].dtype) in ['object', 'category', 'string']:
                current_categorical_cols.append(col)
        
        # 각 범주형 컬럼에 대해 일관된 인코딩
        for col in current_categorical_cols:
            try:
                memory_before = self.get_memory_usage()
                logger.info(f"{col} 인코딩 시작 (메모리: {memory_before:.2f} GB)")
                
                # 메모리 부족 방지
                if memory_before > 35:
                    logger.warning(f"{col}: 메모리 부족으로 인코딩 스킵")
                    continue
                
                # 결측치 처리
                X_train[col] = X_train[col].fillna('missing')
                X_test[col] = X_test[col].fillna('missing')
                
                # 전체 카테고리 수집 (학습+테스트)
                all_values = pd.concat([X_train[col], X_test[col]], ignore_index=True).unique()
                unique_count = len(all_values)
                
                # 고카디널리티 처리
                if unique_count > 10000:
                    top_categories = X_train[col].value_counts().head(100).index
                    X_train[col] = X_train[col].where(X_train[col].isin(top_categories), 'other')
                    X_test[col] = X_test[col].where(X_test[col].isin(top_categories), 'other')
                    logger.info(f"{col}: 초고카디널리티 - 상위 100개만 유지")
                    
                    # 카테고리 재수집
                    all_values = pd.concat([X_train[col], X_test[col]], ignore_index=True).unique()
                    
                elif unique_count > 1000:
                    top_categories = X_train[col].value_counts().head(500).index
                    X_train[col] = X_train[col].where(X_train[col].isin(top_categories), 'other')
                    X_test[col] = X_test[col].where(X_test[col].isin(top_categories), 'other')
                    logger.info(f"{col}: 고카디널리티 - 상위 500개만 유지")
                    
                    # 카테고리 재수집
                    all_values = pd.concat([X_train[col], X_test[col]], ignore_index=True).unique()
                
                # Label Encoding
                try:
                    le = LabelEncoder()
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
    
    def _create_numerical_features_consistent(self, 
                                            X_train: pd.DataFrame, 
                                            X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """일관된 수치형 피처 생성"""
        logger.info("일관된 수치형 피처 생성 시작")
        
        # 메모리 체크
        current_memory = self.get_memory_usage()
        if current_memory > 30:
            logger.warning("메모리 부족으로 수치형 피처 생성 제한")
            max_features = 5
        else:
            max_features = 15
        
        # 현재 수치형 컬럼 재확인 (두 데이터프레임에 공통으로 존재하는 컬럼만)
        current_numeric_cols = []
        for col in X_train.columns:
            if (X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                     'float16', 'float32', 'float64'] and 
                col in X_test.columns):
                current_numeric_cols.append(col)
        
        feature_count = 0
        
        for col in current_numeric_cols[:max_features]:
            try:
                # 메모리 체크
                if self.get_memory_usage() > 35:
                    break
                
                # 로그 변환 (양수 데이터만) - 학습 데이터 기준으로 결정
                if (X_train[col] > 0).all():
                    X_train[f'{col}_log'] = np.log1p(X_train[col]).astype('float32')
                    X_test[f'{col}_log'] = np.log1p(X_test[col]).astype('float32')
                    self.generated_features.append(f'{col}_log')
                    feature_count += 1
                
                # 이상치 플래그 - 학습 데이터 통계로 임계값 설정
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
    
    def _create_interaction_features_consistent(self, 
                                              X_train: pd.DataFrame, 
                                              X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """일관된 상호작용 피처 생성"""
        logger.info("일관된 상호작용 피처 생성 시작")
        
        # 수치형 컬럼 안전하게 확인 (두 데이터프레임에 공통으로 존재)
        safe_numeric_cols = []
        for col in X_train.columns:
            if (X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                     'float16', 'float32', 'float64'] and 
                col in X_test.columns):
                safe_numeric_cols.append(col)
        
        # 메모리 기반 제한
        current_memory = self.get_memory_usage()
        if current_memory > 25:
            max_interactions = 15
            max_cols = 6
        elif current_memory > 20:
            max_interactions = 35
            max_cols = 8
        else:
            max_interactions = 50
            max_cols = 10
        
        # 중요한 피처 그룹별로 선별
        important_groups = {
            'feat_e': [col for col in safe_numeric_cols if col.startswith('feat_e')],
            'feat_d': [col for col in safe_numeric_cols if col.startswith('feat_d')],
            'feat_c': [col for col in safe_numeric_cols if col.startswith('feat_c')],
            'feat_b': [col for col in safe_numeric_cols if col.startswith('feat_b')],
            'history_b': [col for col in safe_numeric_cols if col.startswith('history_b')]
        }
        
        # 각 그룹에서 상위 컬럼만 선택 (학습 데이터 분산 기준)
        important_cols = []
        for group_name, group_cols in important_groups.items():
            if group_cols:
                try:
                    variances = X_train[group_cols].var()
                    top_cols = variances.nlargest(min(2, len(group_cols))).index.tolist()
                    important_cols.extend(top_cols)
                except:
                    important_cols.extend(group_cols[:2])
        
        # 전체에서 상위 컬럼만 사용
        important_cols = important_cols[:max_cols]
        interaction_count = 0
        
        logger.info(f"상호작용 피처 제한: {max_interactions}개, 컬럼: {len(important_cols)}개")
        
        # 1. 그룹 내 상호작용
        for group_name, group_cols in important_groups.items():
            if interaction_count >= max_interactions:
                break
            
            group_important = [col for col in group_cols if col in important_cols]
            if len(group_important) >= 2:
                for i in range(min(2, len(group_important))):
                    for j in range(i+1, min(3, len(group_important))):
                        if interaction_count >= max_interactions:
                            break
                        
                        col1, col2 = group_important[i], group_important[j]
                        try:
                            # 메모리 체크
                            if self.get_memory_usage() > 35:
                                logger.warning("메모리 한계로 상호작용 피처 생성 중단")
                                break
                            
                            # 곱셈 피처
                            X_train[f'{col1}_x_{col2}'] = (X_train[col1] * X_train[col2]).astype('float32')
                            X_test[f'{col1}_x_{col2}'] = (X_test[col1] * X_test[col2]).astype('float32')
                            self.generated_features.append(f'{col1}_x_{col2}')
                            interaction_count += 1
                            
                        except Exception as e:
                            logger.warning(f"{col1}, {col2} 상호작용 피처 실패: {str(e)}")
                            continue
        
        # 2. 그룹 간 상호작용 (중요한 것만)
        if interaction_count < max_interactions and self.get_memory_usage() < 33:
            cross_group_pairs = [
                ('feat_e', 'feat_d'),
                ('feat_c', 'feat_b'), 
                ('feat_e', 'history_b'),
                ('feat_d', 'history_b')
            ]
            
            for group1, group2 in cross_group_pairs:
                if interaction_count >= max_interactions:
                    break
                
                cols1 = [col for col in important_groups.get(group1, []) if col in important_cols]
                cols2 = [col for col in important_groups.get(group2, []) if col in important_cols]
                
                if cols1 and cols2:
                    col1 = cols1[0]
                    col2 = cols2[0]
                    
                    try:
                        X_train[f'{col1}_x_{col2}'] = (X_train[col1] * X_train[col2]).astype('float32')
                        X_test[f'{col1}_x_{col2}'] = (X_test[col1] * X_test[col2]).astype('float32')
                        self.generated_features.append(f'{col1}_x_{col2}')
                        interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"{col1}, {col2} 그룹간 상호작용 피처 실패: {str(e)}")
                        continue
        
        # 3. 비율 피처 (나눗셈)
        if interaction_count < max_interactions and self.get_memory_usage() < 32:
            for i, col1 in enumerate(important_cols[:5]):
                for j, col2 in enumerate(important_cols[i+1:6], i+1):
                    if interaction_count >= max_interactions:
                        break
                    
                    try:
                        # 0으로 나누기 방지
                        ratio_train = X_train[col1] / (X_train[col2] + 1e-8)
                        ratio_test = X_test[col1] / (X_test[col2] + 1e-8)
                        
                        # 무한값 처리
                        ratio_train = np.clip(ratio_train, -1e6, 1e6)
                        ratio_test = np.clip(ratio_test, -1e6, 1e6)
                        
                        X_train[f'{col1}_div_{col2}'] = ratio_train.astype('float32')
                        X_test[f'{col1}_div_{col2}'] = ratio_test.astype('float32')
                        self.generated_features.append(f'{col1}_div_{col2}')
                        interaction_count += 1
                        
                    except Exception as e:
                        logger.warning(f"{col1}, {col2} 비율 피처 실패: {str(e)}")
                        continue
        
        # 메모리 정리
        if interaction_count % 10 == 0:
            gc.collect()
            current_mem = self.get_memory_usage()
            logger.info(f"상호작용 피처 진행: {interaction_count}/{max_interactions}, 메모리: {current_mem:.2f} GB")
        
        logger.info(f"상호작용 피처 생성 완료: {interaction_count}개")
        return X_train, X_test
    
    def _final_data_cleanup_consistent(self, 
                                     X_train: pd.DataFrame, 
                                     X_test: pd.DataFrame, 
                                     y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """일관된 최종 데이터 정리"""
        logger.info("일관된 최종 데이터 정리 시작")
        
        # 1. 공통 컬럼만 유지
        common_columns = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]
        
        logger.info(f"공통 컬럼만 유지: {len(common_columns)}개")
        
        # 2. 데이터 타입 강제 통일
        problematic_columns = []
        
        for col in X_train.columns:
            try:
                # X_train과 X_test의 데이터 타입 확인
                train_dtype = X_train[col].dtype
                test_dtype = X_test[col].dtype
                
                # object 타입 처리
                if train_dtype == 'object' or test_dtype == 'object':
                    logger.warning(f"{col}: object 타입 발견, 수치형으로 변환")
                    
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                    except:
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
        
        # 3. 문제가 있는 컬럼 제거
        if problematic_columns:
            logger.warning(f"문제가 있는 컬럼 제거: {problematic_columns}")
            X_train = X_train.drop(columns=problematic_columns, errors='ignore')
            X_test = X_test.drop(columns=problematic_columns, errors='ignore')
            self.removed_columns.extend(problematic_columns)
        
        # 4. 결측치 처리
        if X_train.isnull().any().any():
            logger.warning("X_train에 결측치 발견, 0으로 대치")
            X_train = X_train.fillna(0)
        
        if X_test.isnull().any().any():
            logger.warning("X_test에 결측치 발견, 0으로 대치")
            X_test = X_test.fillna(0)
        
        # 5. 메모리 안전 무한값 처리
        try:
            for col in X_train.columns:
                try:
                    if np.isinf(X_train[col]).any():
                        logger.warning(f"X_train {col}에 무한값 발견, 클리핑 처리")
                        X_train[col] = X_train[col].replace([np.inf, -np.inf], [1e6, -1e6])
                    
                    if np.isinf(X_test[col]).any():
                        logger.warning(f"X_test {col}에 무한값 발견, 클리핑 처리")
                        X_test[col] = X_test[col].replace([np.inf, -np.inf], [1e6, -1e6])
                        
                except Exception as e:
                    logger.warning(f"{col} 무한값 처리 실패: {str(e)}")
                    continue
                
        except Exception as e:
            logger.warning(f"무한값 처리 중 오류: {str(e)}")
        
        # 6. 최종 검증
        logger.info("최종 데이터 검증 시작")
        
        # 컬럼 일치성 재확인
        if list(X_train.columns) != list(X_test.columns):
            logger.error("X_train과 X_test의 컬럼이 일치하지 않습니다!")
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_columns]
            X_test = X_test[common_columns]
            logger.info(f"공통 컬럼만 재유지: {len(common_columns)}개")
        
        # 데이터 타입 재확인
        for col in X_train.columns:
            train_dtype = X_train[col].dtype
            test_dtype = X_test[col].dtype
            
            if train_dtype != test_dtype:
                logger.warning(f"{col}: 데이터 타입 불일치 - train: {train_dtype}, test: {test_dtype}")
                X_train[col] = X_train[col].astype('float32')
                X_test[col] = X_test[col].astype('float32')
        
        # object 타입이 남아있는지 확인
        train_object_cols = X_train.select_dtypes(include=['object']).columns
        test_object_cols = X_test.select_dtypes(include=['object']).columns
        
        if len(train_object_cols) > 0 or len(test_object_cols) > 0:
            logger.error(f"object 타입 컬럼이 남아있습니다! train: {list(train_object_cols)}, test: {list(test_object_cols)}")
            all_object_cols = list(set(train_object_cols) | set(test_object_cols))
            X_train = X_train.drop(columns=all_object_cols, errors='ignore')
            X_test = X_test.drop(columns=all_object_cols, errors='ignore')
            logger.info(f"object 컬럼 강제 제거: {all_object_cols}")
        
        # 7. 메모리 기반 피처 선택
        current_memory = self.get_memory_usage()
        
        if current_memory > 40 or X_train.shape[1] > 500:
            logger.info("메모리 절약을 위한 피처 선택 수행")
            
            try:
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
        
        # 메모리 정리
        gc.collect()
        
        logger.info("일관된 최종 데이터 정리 완료")
        return X_train, X_test
    
    def _ensure_feature_consistency(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """피처 일관성 보장"""
        logger.info("피처 일관성 최종 보장 시작")
        
        # 현재 컬럼 상태 확인
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        if train_cols != test_cols:
            logger.warning("학습/테스트 데이터 컬럼 불일치 발견")
            logger.warning(f"학습에만 있는 컬럼: {train_cols - test_cols}")
            logger.warning(f"테스트에만 있는 컬럼: {test_cols - train_cols}")
            
            # 공통 컬럼만 유지
            common_cols = list(train_cols & test_cols)
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            logger.info(f"공통 컬럼만 유지: {len(common_cols)}개")
        
        # 컬럼 순서 정렬 (일관성 보장)
        columns_sorted = sorted(X_train.columns)
        X_train = X_train[columns_sorted]
        X_test = X_test[columns_sorted]
        
        # 최종 피처 컬럼 순서 저장
        self.final_feature_columns = list(X_train.columns)
        
        # 최종 검증
        assert X_train.shape[1] == X_test.shape[1], f"피처 수 불일치: train {X_train.shape[1]}, test {X_test.shape[1]}"
        assert list(X_train.columns) == list(X_test.columns), "컬럼 순서 불일치"
        
        logger.info(f"피처 일관성 보장 완료 - 최종 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약 정보"""
        return {
            'total_generated_features': len(self.generated_features),
            'generated_features': self.generated_features,
            'removed_columns': self.removed_columns,
            'final_feature_columns': self.final_feature_columns,
            'encoders_count': {
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            }
        }

# 기존 FeatureEngineer 클래스를 최적화된 버전으로 교체
FeatureEngineer = MemoryOptimizedFeatureEngineer