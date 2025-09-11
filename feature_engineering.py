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
import hashlib
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    """CTR 예측에 특화된 피처 엔지니어링 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
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
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CTR 예측에 특화된 피처 엔지니어링 파이프라인"""
        logger.info("CTR 피처 엔지니어링 시작")
        
        # 학습 데이터에서 타겟 분리
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.copy()
        
        # 초기 메모리 상태
        initial_memory = self.get_memory_usage()
        logger.info(f"초기 메모리 사용량: {initial_memory:.2f} GB")
        
        # 컬럼 타입 분류
        self._classify_columns(X_train)
        
        # 1. ID 피처 처리 (제거하지 않고 해시 인코딩)
        X_train, X_test = self._process_id_features(X_train, X_test)
        
        # 2. 기본 피처 정리
        X_train, X_test = self._clean_basic_features(X_train, X_test)
        
        # 3. 범주형 피처 인코딩 (누수 방지)
        X_train, X_test = self._encode_categorical_features(X_train, X_test, y_train)
        
        # 4. 수치형 피처 생성
        X_train, X_test = self._create_numerical_features(X_train, X_test)
        
        # 5. CTR 특화 피처 생성
        X_train, X_test = self._create_ctr_features(X_train, X_test, y_train)
        
        # 6. 상호작용 피처 생성
        current_memory = self.get_memory_usage()
        if current_memory < 35:
            X_train, X_test = self._create_interaction_features(X_train, X_test)
        else:
            logger.warning("메모리 부족으로 상호작용 피처 생성 건너뛰기")
        
        # 7. 최종 데이터 정리
        X_train, X_test = self._final_data_cleanup(X_train, X_test)
        
        # 8. 피처 컬럼 순서 일치 보장
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
        self.id_columns = []
        
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            col_lower = str(col).lower()
            
            # ID성 컬럼 식별
            if any(pattern in col_lower for pattern in ['id', 'uuid', 'key']):
                self.id_columns.append(col)
            elif dtype_str in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                             'float16', 'float32', 'float64']:
                # 카디널리티 체크로 ID성 여부 재확인
                if df[col].nunique() / len(df) > 0.95:
                    self.id_columns.append(col)
                else:
                    self.numeric_columns.append(col)
            elif dtype_str in ['object', 'category', 'string', 'bool']:
                self.categorical_columns.append(col)
        
        logger.info(f"수치형 컬럼: {len(self.numeric_columns)}개")
        logger.info(f"범주형 컬럼: {len(self.categorical_columns)}개")
        logger.info(f"ID 컬럼: {len(self.id_columns)}개")
    
    def _process_id_features(self, 
                           X_train: pd.DataFrame, 
                           X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ID 피처 처리 (해시 인코딩으로 활용)"""
        logger.info("ID 피처 해시 인코딩 시작")
        
        for col in self.id_columns:
            if col in X_train.columns:
                try:
                    # 해시 인코딩
                    X_train[f'{col}_hash'] = X_train[col].astype(str).apply(
                        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 1000000
                    ).astype('int32')
                    
                    X_test[f'{col}_hash'] = X_test[col].astype(str).apply(
                        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 1000000
                    ).astype('int32')
                    
                    # 카운트 인코딩
                    value_counts = X_train[col].value_counts()
                    X_train[f'{col}_count'] = X_train[col].map(value_counts).fillna(0).astype('int16')
                    X_test[f'{col}_count'] = X_test[col].map(value_counts).fillna(0).astype('int16')
                    
                    self.generated_features.extend([f'{col}_hash', f'{col}_count'])
                    logger.info(f"{col}: 해시 및 카운트 인코딩 완료")
                    
                except Exception as e:
                    logger.warning(f"{col} ID 피처 처리 실패: {str(e)}")
        
        # 원본 ID 컬럼 제거
        existing_id_cols = [col for col in self.id_columns if col in X_train.columns]
        if existing_id_cols:
            X_train = X_train.drop(columns=existing_id_cols)
            X_test = X_test.drop(columns=existing_id_cols)
            self.removed_columns.extend(existing_id_cols)
            logger.info(f"원본 ID 컬럼 제거: {existing_id_cols}")
        
        return X_train, X_test
    
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
            self.removed_columns.extend(cols_to_remove)
        
        # 메모리 정리
        gc.collect()
        
        return X_train, X_test
    
    def _encode_categorical_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """범주형 피처 인코딩 (데이터 누수 방지)"""
        logger.info("범주형 피처 인코딩 시작")
        
        # 현재 남아있는 범주형 컬럼 재확인
        current_categorical_cols = []
        for col in X_train.columns:
            if str(X_train[col].dtype) in ['object', 'category', 'string']:
                current_categorical_cols.append(col)
        
        for col in current_categorical_cols:
            try:
                # 결측치 처리
                X_train[col] = X_train[col].fillna('missing')
                X_test[col] = X_test[col].fillna('missing')
                
                # 고카디널리티 처리
                unique_count = X_train[col].nunique()
                if unique_count > 1000:
                    top_categories = X_train[col].value_counts().head(500).index
                    X_train[col] = X_train[col].where(X_train[col].isin(top_categories), 'other')
                    X_test[col] = X_test[col].where(X_test[col].isin(top_categories), 'other')
                    logger.info(f"{col}: 고카디널리티 - 상위 500개만 유지")
                
                # Label Encoding (학습 데이터만으로 학습)
                try:
                    le = LabelEncoder()
                    le.fit(X_train[col].astype(str))
                    
                    X_train[f'{col}_encoded'] = le.transform(X_train[col].astype(str)).astype('int16')
                    
                    # 테스트 데이터의 미지 카테고리 처리
                    test_encoded = []
                    for val in X_test[col].astype(str):
                        if val in le.classes_:
                            test_encoded.append(le.transform([val])[0])
                        else:
                            test_encoded.append(-1)  # 미지 카테고리
                    
                    X_test[f'{col}_encoded'] = np.array(test_encoded).astype('int16')
                    
                    self.label_encoders[col] = le
                    self.generated_features.append(f'{col}_encoded')
                    
                except Exception as e:
                    logger.warning(f"{col} Label Encoding 실패: {str(e)}")
                    continue
                
                # 빈도 인코딩 (학습 데이터만으로 계산)
                try:
                    freq_map = X_train[col].value_counts().to_dict()
                    X_train[f'{col}_freq'] = X_train[col].map(freq_map).fillna(0).astype('int16')
                    X_test[f'{col}_freq'] = X_test[col].map(freq_map).fillna(0).astype('int16')
                    self.generated_features.append(f'{col}_freq')
                    
                except Exception as e:
                    logger.warning(f"{col} 빈도 인코딩 실패: {str(e)}")
                
                # 타겟 인코딩 (교차검증으로 누수 방지)
                if len(X_train) > 10000:  # 충분한 데이터가 있을 때만
                    try:
                        X_train[f'{col}_target'] = self._safe_target_encoding(
                            X_train[col], y_train, smoothing=self.config.FEATURE_CONFIG['target_encoding_smoothing']
                        ).astype('float32')
                        
                        # 테스트 데이터는 전체 학습 데이터 기반 타겟 인코딩
                        target_map = X_train.groupby(col)[f'{col}_target'].mean().to_dict()
                        global_mean = y_train.mean()
                        X_test[f'{col}_target'] = X_test[col].map(target_map).fillna(global_mean).astype('float32')
                        
                        self.generated_features.append(f'{col}_target')
                        
                    except Exception as e:
                        logger.warning(f"{col} 타겟 인코딩 실패: {str(e)}")
                
                # 메모리 정리
                gc.collect()
                
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
        
        logger.info(f"범주형 피처 인코딩 완료")
        return X_train, X_test
    
    def _safe_target_encoding(self, series: pd.Series, target: pd.Series, smoothing: int = 100) -> pd.Series:
        """안전한 타겟 인코딩 (교차검증 방식)"""
        from sklearn.model_selection import KFold
        
        result = np.zeros(len(series))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        global_mean = target.mean()
        
        for train_idx, val_idx in kf.split(series):
            # 훈련 세트에서 타겟 평균 계산
            train_stats = pd.Series(target.iloc[train_idx], index=series.iloc[train_idx]).groupby(series.iloc[train_idx]).agg(['mean', 'count'])
            
            # 스무딩 적용
            smoothed_means = (train_stats['mean'] * train_stats['count'] + global_mean * smoothing) / (train_stats['count'] + smoothing)
            
            # 검증 세트에 적용
            result[val_idx] = series.iloc[val_idx].map(smoothed_means).fillna(global_mean)
        
        return pd.Series(result, index=series.index)
    
    def _create_numerical_features(self, 
                                 X_train: pd.DataFrame, 
                                 X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """수치형 피처 생성"""
        logger.info("수치형 피처 생성 시작")
        
        # 현재 수치형 컬럼 재확인
        current_numeric_cols = []
        for col in X_train.columns:
            if X_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                                    'float16', 'float32', 'float64']:
                current_numeric_cols.append(col)
        
        feature_count = 0
        max_features = 15
        
        for col in current_numeric_cols[:max_features]:
            try:
                if self.get_memory_usage() > 35:
                    break
                
                # 로그 변환
                if (X_train[col] > 0).all():
                    X_train[f'{col}_log'] = np.log1p(X_train[col]).astype('float32')
                    X_test[f'{col}_log'] = np.log1p(X_test[col]).astype('float32')
                    self.generated_features.append(f'{col}_log')
                    feature_count += 1
                
                # 이상치 플래그
                if feature_count < max_features:
                    q1, q3 = X_train[col].quantile([0.25, 0.75])
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
                
                if feature_count % 5 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.warning(f"{col} 수치형 피처 생성 실패: {str(e)}")
                continue
        
        logger.info(f"수치형 피처 생성 완료: {feature_count}개")
        return X_train, X_test
    
    def _create_ctr_features(self, 
                           X_train: pd.DataFrame, 
                           X_test: pd.DataFrame,
                           y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CTR 예측에 특화된 피처 생성"""
        logger.info("CTR 특화 피처 생성 시작")
        
        try:
            # 시간적 피처 (인덱스 기반)
            X_train['time_index'] = (X_train.index / len(X_train)).astype('float32')
            X_test['time_index'] = (X_test.index / len(X_test)).astype('float32')
            
            # 상대적 위치 피처
            X_train['position_quartile'] = pd.qcut(X_train.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            X_test['position_quartile'] = pd.qcut(X_test.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            
            # 세션 위치 피처 (100개씩 묶어서 세션으로 가정)
            X_train['session_position'] = (X_train.index % 100).astype('int8')
            X_test['session_position'] = (X_test.index % 100).astype('int8')
            
            self.generated_features.extend(['time_index', 'position_quartile', 'session_position'])
            
            # 피처 그룹별 통계 (CTR 예측에 중요한 패턴)
            feature_groups = {
                'feat_e': [col for col in X_train.columns if col.startswith('feat_e')],
                'feat_d': [col for col in X_train.columns if col.startswith('feat_d')],
                'feat_c': [col for col in X_train.columns if col.startswith('feat_c')],
                'feat_b': [col for col in X_train.columns if col.startswith('feat_b')]
            }
            
            for group_name, group_cols in feature_groups.items():
                if group_cols and len(group_cols) > 1:
                    try:
                        # 그룹 합계
                        X_train[f'{group_name}_sum'] = X_train[group_cols].sum(axis=1).astype('float32')
                        X_test[f'{group_name}_sum'] = X_test[group_cols].sum(axis=1).astype('float32')
                        
                        # 그룹 평균
                        X_train[f'{group_name}_mean'] = X_train[group_cols].mean(axis=1).astype('float32')
                        X_test[f'{group_name}_mean'] = X_test[group_cols].mean(axis=1).astype('float32')
                        
                        # 그룹 표준편차
                        X_train[f'{group_name}_std'] = X_train[group_cols].std(axis=1).fillna(0).astype('float32')
                        X_test[f'{group_name}_std'] = X_test[group_cols].std(axis=1).fillna(0).astype('float32')
                        
                        self.generated_features.extend([f'{group_name}_sum', f'{group_name}_mean', f'{group_name}_std'])
                        
                    except Exception as e:
                        logger.warning(f"{group_name} 그룹 통계 생성 실패: {str(e)}")
            
            logger.info("CTR 특화 피처 생성 완료")
            
        except Exception as e:
            logger.error(f"CTR 특화 피처 생성 실패: {str(e)}")
        
        return X_train, X_test
    
    def _create_interaction_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """상호작용 피처 생성"""
        logger.info("상호작용 피처 생성 시작")
        
        # 중요한 피처들 선별
        important_features = []
        
        # 피처 그룹별 대표 피처 선택
        feature_groups = {
            'feat_e': [col for col in X_train.columns if col.startswith('feat_e')],
            'feat_d': [col for col in X_train.columns if col.startswith('feat_d')],
            'feat_c': [col for col in X_train.columns if col.startswith('feat_c')],
            'feat_b': [col for col in X_train.columns if col.startswith('feat_b')]
        }
        
        for group_cols in feature_groups.values():
            if group_cols:
                # 각 그룹에서 분산이 가장 큰 피처 선택
                try:
                    variances = X_train[group_cols].var()
                    top_feature = variances.idxmax()
                    important_features.append(top_feature)
                except:
                    important_features.append(group_cols[0])
        
        # 생성된 피처들 중에서도 선별
        generated_numeric = [col for col in self.generated_features 
                           if col in X_train.columns and X_train[col].dtype in ['float32', 'int16', 'int32']]
        important_features.extend(generated_numeric[:3])
        
        # 중복 제거
        important_features = list(set(important_features))[:8]
        
        interaction_count = 0
        max_interactions = 20
        
        # 곱셈 상호작용
        for i, col1 in enumerate(important_features):
            for j, col2 in enumerate(important_features[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                try:
                    if self.get_memory_usage() > 35:
                        break
                    
                    X_train[f'{col1}_x_{col2}'] = (X_train[col1] * X_train[col2]).astype('float32')
                    X_test[f'{col1}_x_{col2}'] = (X_test[col1] * X_test[col2]).astype('float32')
                    self.generated_features.append(f'{col1}_x_{col2}')
                    interaction_count += 1
                    
                except Exception as e:
                    logger.warning(f"{col1}, {col2} 상호작용 피처 실패: {str(e)}")
                    continue
        
        # 나눗셈 상호작용 (안전한 형태)
        for i, col1 in enumerate(important_features[:4]):
            for j, col2 in enumerate(important_features[i+1:5], i+1):
                if interaction_count >= max_interactions:
                    break
                
                try:
                    ratio_train = X_train[col1] / (np.abs(X_train[col2]) + 1e-8)
                    ratio_test = X_test[col1] / (np.abs(X_test[col2]) + 1e-8)
                    
                    ratio_train = np.clip(ratio_train, -1e6, 1e6)
                    ratio_test = np.clip(ratio_test, -1e6, 1e6)
                    
                    X_train[f'{col1}_div_{col2}'] = ratio_train.astype('float32')
                    X_test[f'{col1}_div_{col2}'] = ratio_test.astype('float32')
                    self.generated_features.append(f'{col1}_div_{col2}')
                    interaction_count += 1
                    
                except Exception as e:
                    logger.warning(f"{col1}, {col2} 비율 피처 실패: {str(e)}")
                    continue
        
        logger.info(f"상호작용 피처 생성 완료: {interaction_count}개")
        return X_train, X_test
    
    def _final_data_cleanup(self, 
                          X_train: pd.DataFrame, 
                          X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """최종 데이터 정리"""
        logger.info("최종 데이터 정리 시작")
        
        # 1. 공통 컬럼만 유지
        common_columns = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]
        
        # 2. 데이터 타입 강제 통일
        for col in X_train.columns:
            try:
                train_dtype = X_train[col].dtype
                test_dtype = X_test[col].dtype
                
                # object 타입 처리
                if train_dtype == 'object' or test_dtype == 'object':
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                
                # 수치형 타입 통일
                elif train_dtype != test_dtype or train_dtype != 'float32':
                    X_train[col] = X_train[col].astype('float32')
                    X_test[col] = X_test[col].astype('float32')
                    
            except Exception as e:
                logger.warning(f"{col} 데이터 타입 처리 실패: {str(e)}")
                continue
        
        # 3. 결측치 및 무한값 처리
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        X_train = X_train.replace([np.inf, -np.inf], [1e6, -1e6])
        X_test = X_test.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 4. 최종 검증
        assert list(X_train.columns) == list(X_test.columns), "컬럼 불일치"
        
        # 5. 메모리 정리
        gc.collect()
        
        logger.info("최종 데이터 정리 완료")
        return X_train, X_test
    
    def _ensure_feature_consistency(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """피처 일관성 보장"""
        logger.info("피처 일관성 보장 시작")
        
        # 컬럼 순서 정렬
        columns_sorted = sorted(X_train.columns)
        X_train = X_train[columns_sorted]
        X_test = X_test[columns_sorted]
        
        # 최종 피처 컬럼 순서 저장
        self.final_feature_columns = list(X_train.columns)
        
        logger.info(f"피처 일관성 보장 완료 - 최종 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약 정보"""
        return {
            'total_generated_features': len(self.generated_features),
            'generated_features': self.generated_features,
            'removed_columns': self.removed_columns,
            'final_feature_columns': self.final_feature_columns,
            'id_columns_processed': self.id_columns,
            'encoders_count': {
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            }
        }

# 기존 클래스명 유지
FeatureEngineer = CTRFeatureEngineer