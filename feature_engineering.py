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
        self.original_feature_order = []
        self.memory_efficient_mode = False
        
    def set_memory_efficient_mode(self, enabled: bool = True):
        """메모리 효율 모드 설정"""
        self.memory_efficient_mode = enabled
        if enabled:
            logger.info("메모리 효율 모드 활성화")
        else:
            logger.info("일반 모드로 설정")
    
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def get_available_memory(self) -> float:
        """사용 가능한 메모리 (GB)"""
        return psutil.virtual_memory().available / (1024**3)
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """메모리 효율적인 피처 엔지니어링 파이프라인"""
        logger.info("CTR 피처 엔지니어링 시작")
        
        # 학습 데이터에서 타겟 분리
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.copy()
        
        # 원본 피처 순서 저장
        self.original_feature_order = sorted(X_train.columns.tolist())
        logger.info(f"원본 피처 순서 저장: {len(self.original_feature_order)}개")
        
        # 초기 메모리 상태
        initial_memory = self.get_memory_usage()
        available_memory = self.get_available_memory()
        logger.info(f"초기 메모리: 사용 {initial_memory:.2f} GB, 사용 가능 {available_memory:.2f} GB")
        
        # 메모리 부족 시 자동으로 효율 모드 활성화
        if available_memory < 10 and not self.memory_efficient_mode:
            self.set_memory_efficient_mode(True)
            logger.warning("메모리 부족으로 자동으로 효율 모드 활성화")
        
        # 컬럼 타입 분류
        self._classify_columns(X_train)
        
        # 1. 기본 데이터 타입 통일
        X_train, X_test = self._unify_data_types(X_train, X_test)
        
        # 2. ID 피처 처리
        X_train, X_test = self._process_id_features(X_train, X_test)
        
        # 3. 기본 피처 정리
        X_train, X_test = self._clean_basic_features(X_train, X_test)
        
        # 4. 범주형 피처 인코딩
        X_train, X_test = self._encode_categorical_features(X_train, X_test, y_train)
        
        # 5. 수치형 피처 생성 (메모리 모드에 따라)
        if self.memory_efficient_mode:
            X_train, X_test = self._create_essential_features_only(X_train, X_test)
        else:
            X_train, X_test = self._create_numerical_features(X_train, X_test)
        
        # 6. CTR 특화 피처 생성
        X_train, X_test = self._create_ctr_features(X_train, X_test, y_train)
        
        # 7. 상호작용 피처 생성 (메모리 여유가 있을 때만)
        available_memory = self.get_available_memory()
        if available_memory > 8 and not self.memory_efficient_mode:
            X_train, X_test = self._create_interaction_features(X_train, X_test)
        else:
            logger.info("메모리 절약을 위해 상호작용 피처 생성 건너뛰기")
        
        # 8. 최종 데이터 정리
        X_train, X_test = self._final_data_cleanup(X_train, X_test)
        
        # 9. 피처 컬럼 순서 일치 보장
        X_train, X_test = self._ensure_consistent_feature_order(X_train, X_test)
        
        # 최종 상태
        final_memory = self.get_memory_usage()
        final_available = self.get_available_memory()
        logger.info(f"최종 메모리: 사용 {final_memory:.2f} GB, 사용 가능 {final_available:.2f} GB")
        logger.info(f"피처 엔지니어링 완료 - 최종 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def _unify_data_types(self, 
                         X_train: pd.DataFrame, 
                         X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 데이터 타입 통일"""
        logger.info("데이터 타입 통일 시작")
        
        try:
            # 공통 컬럼만 처리
            common_columns = list(set(X_train.columns) & set(X_test.columns))
            
            for col in common_columns:
                try:
                    train_dtype = str(X_train[col].dtype)
                    test_dtype = str(X_test[col].dtype)
                    
                    # 타입이 다른 경우 통일
                    if train_dtype != test_dtype:
                        logger.info(f"{col}: 타입 불일치 - train: {train_dtype}, test: {test_dtype}")
                        
                        # category 타입 특별 처리
                        if 'category' in train_dtype or 'category' in test_dtype:
                            # 둘 다 문자열로 변환
                            X_train[col] = X_train[col].astype('str')
                            X_test[col] = X_test[col].astype('str')
                            logger.info(f"{col}: str로 통일")
                        # 수치형으로 변환 시도
                        elif pd.api.types.is_numeric_dtype(X_train[col]) or pd.api.types.is_numeric_dtype(X_test[col]):
                            # 수치형으로 통일
                            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').astype('float32')
                            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').astype('float32')
                            logger.info(f"{col}: float32로 통일")
                        else:
                            # 문자열로 통일
                            X_train[col] = X_train[col].astype('str')
                            X_test[col] = X_test[col].astype('str')
                            logger.info(f"{col}: str로 통일")
                    
                    # 동일한 타입이어도 최적화
                    elif train_dtype == 'object':
                        # object 타입은 문자열로 명시적 변환
                        X_train[col] = X_train[col].astype('str')
                        X_test[col] = X_test[col].astype('str')
                    elif 'category' in train_dtype:
                        # category 타입은 문자열로 변환
                        X_train[col] = X_train[col].astype('str')
                        X_test[col] = X_test[col].astype('str')
                    elif train_dtype in ['int64', 'float64']:
                        # 큰 타입은 작은 타입으로 변환
                        if train_dtype == 'int64':
                            # 범위 확인 후 적절한 타입 선택
                            try:
                                min_val = min(X_train[col].min(), X_test[col].min())
                                max_val = max(X_train[col].max(), X_test[col].max())
                                
                                if pd.isna(min_val) or pd.isna(max_val):
                                    X_train[col] = X_train[col].astype('float32')
                                    X_test[col] = X_test[col].astype('float32')
                                elif min_val >= 0 and max_val < 65535:
                                    X_train[col] = X_train[col].astype('uint16')
                                    X_test[col] = X_test[col].astype('uint16')
                                else:
                                    X_train[col] = X_train[col].astype('int32')
                                    X_test[col] = X_test[col].astype('int32')
                            except:
                                X_train[col] = X_train[col].astype('float32')
                                X_test[col] = X_test[col].astype('float32')
                        else:  # float64
                            X_train[col] = X_train[col].astype('float32')
                            X_test[col] = X_test[col].astype('float32')
                
                except Exception as e:
                    logger.warning(f"{col} 타입 통일 실패: {str(e)}")
                    # 실패 시 기본값으로 설정
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype('float32')
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype('float32')
                    except:
                        X_train[col] = X_train[col].astype('str')
                        X_test[col] = X_test[col].astype('str')
            
            logger.info("데이터 타입 통일 완료")
            
        except Exception as e:
            logger.error(f"데이터 타입 통일 전체 실패: {str(e)}")
        
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
        """ID 피처 처리"""
        logger.info("ID 피처 해시 인코딩 시작")
        
        for col in self.id_columns:
            if col in X_train.columns:
                try:
                    # 안전한 문자열 변환
                    train_values = X_train[col].astype(str).fillna('unknown')
                    test_values = X_test[col].astype(str).fillna('unknown')
                    
                    # 메모리 효율 모드에서는 해시만 생성
                    if self.memory_efficient_mode:
                        X_train[f'{col}_hash'] = train_values.apply(
                            lambda x: hash(str(x)) % 100000
                        ).astype('int32')
                        
                        X_test[f'{col}_hash'] = test_values.apply(
                            lambda x: hash(str(x)) % 100000
                        ).astype('int32')
                        
                        self.generated_features.append(f'{col}_hash')
                        logger.info(f"{col}: 해시 인코딩 완료 (효율 모드)")
                    else:
                        # 일반 모드에서는 해시 + 카운트
                        X_train[f'{col}_hash'] = train_values.apply(
                            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 1000000
                        ).astype('int32')
                        
                        X_test[f'{col}_hash'] = test_values.apply(
                            lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 1000000
                        ).astype('int32')
                        
                        # 카운트 인코딩
                        value_counts = train_values.value_counts()
                        X_train[f'{col}_count'] = train_values.map(value_counts).fillna(0).astype('int16')
                        X_test[f'{col}_count'] = test_values.map(value_counts).fillna(0).astype('int16')
                        
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
        
        # 메모리 정리
        gc.collect()
        
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
            except:
                continue
        
        # 컬럼 제거
        if cols_to_remove:
            X_train = X_train.drop(columns=cols_to_remove)
            X_test = X_test.drop(columns=cols_to_remove)
            self.removed_columns.extend(cols_to_remove)
            logger.info(f"상수 컬럼 {len(cols_to_remove)}개 제거")
        
        # 메모리 정리
        gc.collect()
        
        return X_train, X_test
    
    def _encode_categorical_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """범주형 피처 인코딩"""
        logger.info("범주형 피처 인코딩 시작")
        
        # 현재 남아있는 범주형 컬럼 재확인
        current_categorical_cols = []
        for col in X_train.columns:
            dtype_str = str(X_train[col].dtype)
            if dtype_str in ['object', 'category', 'string'] or X_train[col].dtype.name == 'category':
                current_categorical_cols.append(col)
        
        for col in current_categorical_cols:
            try:
                # 안전한 문자열 변환
                train_values = X_train[col].astype('str').fillna('missing')
                test_values = X_test[col].astype('str').fillna('missing')
                
                # 고카디널리티 처리
                unique_count = len(train_values.unique())
                max_categories = 100 if self.memory_efficient_mode else 500
                
                if unique_count > max_categories:
                    top_categories = train_values.value_counts().head(max_categories).index
                    train_values = train_values.where(train_values.isin(top_categories), 'other')
                    test_values = test_values.where(test_values.isin(top_categories), 'other')
                    logger.info(f"{col}: 고카디널리티 - 상위 {max_categories}개만 유지")
                
                # Label Encoding
                try:
                    le = LabelEncoder()
                    le.fit(train_values)
                    
                    X_train[f'{col}_encoded'] = le.transform(train_values).astype('int16')
                    
                    # 테스트 데이터의 미지 카테고리 처리
                    test_encoded = []
                    for val in test_values:
                        if val in le.classes_:
                            test_encoded.append(le.transform([val])[0])
                        else:
                            test_encoded.append(-1)
                    
                    X_test[f'{col}_encoded'] = np.array(test_encoded).astype('int16')
                    
                    self.label_encoders[col] = le
                    self.generated_features.append(f'{col}_encoded')
                    
                except Exception as e:
                    logger.warning(f"{col} Label Encoding 실패: {str(e)}")
                    continue
                
                # 빈도 인코딩
                try:
                    freq_map = train_values.value_counts().to_dict()
                    X_train[f'{col}_freq'] = train_values.map(freq_map).fillna(0).astype('int16')
                    X_test[f'{col}_freq'] = test_values.map(freq_map).fillna(0).astype('int16')
                    self.generated_features.append(f'{col}_freq')
                    
                except Exception as e:
                    logger.warning(f"{col} 빈도 인코딩 실패: {str(e)}")
                
                # 타겟 인코딩 (메모리 효율 모드에서는 스킵)
                if not self.memory_efficient_mode and len(X_train) > 10000:
                    try:
                        target_encoding = self._safe_target_encoding(
                            train_values, y_train, smoothing=self.config.FEATURE_CONFIG['target_encoding_smoothing']
                        )
                        X_train[f'{col}_target'] = target_encoding.astype('float32')
                        
                        target_map = pd.DataFrame({
                            'category': train_values,
                            'target': target_encoding
                        }).groupby('category')['target'].mean().to_dict()
                        
                        global_mean = y_train.mean()
                        X_test[f'{col}_target'] = test_values.map(target_map).fillna(global_mean).astype('float32')
                        
                        self.generated_features.append(f'{col}_target')
                        
                    except Exception as e:
                        logger.warning(f"{col} 타겟 인코딩 실패: {str(e)}")
                
                # 메모리 정리
                if len(self.generated_features) % 10 == 0:
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
        except Exception as e:
            logger.warning(f"범주형 컬럼 제거 실패: {str(e)}")
        
        # 메모리 정리
        gc.collect()
        
        return X_train, X_test
    
    def _safe_target_encoding(self, series: pd.Series, target: pd.Series, smoothing: int = 100) -> pd.Series:
        """안전한 타겟 인코딩"""
        from sklearn.model_selection import KFold
        
        result = np.zeros(len(series))
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        global_mean = target.mean()
        
        for train_idx, val_idx in kf.split(series):
            # 훈련 세트에서 타겟 평균 계산
            train_stats = pd.DataFrame({
                'category': series.iloc[train_idx],
                'target': target.iloc[train_idx]
            }).groupby('category').agg({'target': ['mean', 'count']})
            
            train_stats.columns = ['mean', 'count']
            
            # 스무딩 적용
            smoothed_means = (train_stats['mean'] * train_stats['count'] + global_mean * smoothing) / (train_stats['count'] + smoothing)
            
            # 검증 세트에 적용
            result[val_idx] = series.iloc[val_idx].map(smoothed_means).fillna(global_mean)
        
        return pd.Series(result, index=series.index)
    
    def _create_essential_features_only(self, 
                                      X_train: pd.DataFrame, 
                                      X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """메모리 효율 모드: 필수 피처만 생성"""
        logger.info("필수 피처만 생성 (메모리 효율 모드)")
        
        # 현재 수치형 컬럼 재확인
        current_numeric_cols = []
        for col in X_train.columns:
            try:
                if pd.api.types.is_numeric_dtype(X_train[col]) and not pd.api.types.is_bool_dtype(X_train[col]):
                    if X_train[col].notna().sum() > len(X_train) * 0.5:
                        current_numeric_cols.append(col)
            except:
                continue
        
        # 중요한 컬럼만 선택 (최대 5개)
        selected_cols = current_numeric_cols[:5]
        
        for col in selected_cols:
            try:
                # 데이터 유효성 검사
                train_valid = X_train[col].notna()
                test_valid = X_test[col].notna()
                
                if train_valid.sum() == 0 or test_valid.sum() == 0:
                    continue
                
                # 로그 변환 (안전한 경우만)
                train_positive = (X_train[col] > 0) & train_valid
                test_positive = (X_test[col] > 0) & test_valid
                
                if train_positive.sum() > len(X_train) * 0.8 and test_positive.sum() > len(X_test) * 0.8:
                    X_train[f'{col}_log'] = np.where(
                        train_positive,
                        np.log1p(X_train[col]),
                        0
                    ).astype('float32')
                    
                    X_test[f'{col}_log'] = np.where(
                        test_positive,
                        np.log1p(X_test[col]),
                        0
                    ).astype('float32')
                    
                    self.generated_features.append(f'{col}_log')
                
            except Exception as e:
                logger.warning(f"{col} 필수 피처 생성 실패: {str(e)}")
                continue
        
        logger.info(f"필수 피처 생성 완료: {len([f for f in self.generated_features if '_log' in f])}개")
        return X_train, X_test
    
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
                available_memory = self.get_available_memory()
                if available_memory < 5:
                    logger.warning("메모리 부족으로 피처 생성 중단")
                    break
                
                # 로그 변환
                if (X_train[col] > 0).all() and (X_test[col] > 0).all():
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
            # 시간적 피처
            X_train['time_index'] = (X_train.index / len(X_train)).astype('float32')
            X_test['time_index'] = (X_test.index / len(X_test)).astype('float32')
            
            # 상대적 위치 피처
            X_train['position_quartile'] = pd.qcut(X_train.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            X_test['position_quartile'] = pd.qcut(X_test.index, q=4, labels=[0, 1, 2, 3]).astype('int8')
            
            self.generated_features.extend(['time_index', 'position_quartile'])
            
            # 메모리 효율 모드가 아닐 때만 추가 피처 생성
            if not self.memory_efficient_mode:
                # 세션 위치 피처
                X_train['session_position'] = (X_train.index % 100).astype('int8')
                X_test['session_position'] = (X_test.index % 100).astype('int8')
                self.generated_features.append('session_position')
                
                # 피처 그룹별 통계
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
                            
                            self.generated_features.extend([f'{group_name}_sum', f'{group_name}_mean'])
                            
                        except Exception as e:
                            logger.warning(f"{group_name} 그룹 통계 생성 실패: {str(e)}")
            
            logger.info("CTR 특화 피처 생성 완료")
            
        except Exception as e:
            logger.error(f"CTR 특화 피처 생성 실패: {str(e)}")
        
        # 메모리 정리
        gc.collect()
        
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
        important_features = list(set(important_features))[:6]
        
        interaction_count = 0
        max_interactions = 10
        
        # 곱셈 상호작용
        for i, col1 in enumerate(important_features):
            for j, col2 in enumerate(important_features[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                try:
                    available_memory = self.get_available_memory()
                    if available_memory < 5:
                        break
                    
                    X_train[f'{col1}_x_{col2}'] = (X_train[col1] * X_train[col2]).astype('float32')
                    X_test[f'{col1}_x_{col2}'] = (X_test[col1] * X_test[col2]).astype('float32')
                    self.generated_features.append(f'{col1}_x_{col2}')
                    interaction_count += 1
                    
                except Exception as e:
                    logger.warning(f"{col1}, {col2} 상호작용 피처 실패: {str(e)}")
                    continue
        
        logger.info(f"상호작용 피처 생성 완료: {interaction_count}개")
        
        # 메모리 정리
        gc.collect()
        
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
    
    def _ensure_consistent_feature_order(self, 
                                        X_train: pd.DataFrame, 
                                        X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """피처 순서 일관성 보장"""
        logger.info("피처 순서 일관성 보장 시작")
        
        # 현재 컬럼들을 원본 순서와 생성된 순서로 분류
        current_columns = list(X_train.columns)
        
        # 원본 피처 중 남아있는 것들
        remaining_original = [col for col in self.original_feature_order if col in current_columns]
        
        # 생성된 피처들
        generated_cols = [col for col in current_columns if col not in self.original_feature_order]
        generated_cols_sorted = sorted(generated_cols)
        
        # 최종 컬럼 순서
        final_order = remaining_original + generated_cols_sorted
        
        # 순서 적용
        X_train = X_train[final_order]
        X_test = X_test[final_order]
        
        # 최종 피처 순서 저장
        self.final_feature_columns = final_order
        
        logger.info(f"피처 순서 일관성 보장 완료 - 원본 {len(remaining_original)}개 + 생성 {len(generated_cols_sorted)}개")
        
        return X_train, X_test
    
    def get_feature_columns_for_inference(self) -> List[str]:
        """추론에 사용할 피처 컬럼 순서 반환"""
        return self.final_feature_columns.copy()
    
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
            'encoders_count': {
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            }
        }

# 기존 클래스명 유지
FeatureEngineer = CTRFeatureEngineer