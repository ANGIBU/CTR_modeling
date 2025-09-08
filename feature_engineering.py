# feature_engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """고급 피처 엔지니어링 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.target_encoders = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        self.generated_features = []
        
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """모든 피처 생성 파이프라인 실행"""
        logger.info("피처 엔지니어링 파이프라인 시작")
        
        # 학습 데이터에서 타겟 분리
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.copy()
        
        # 1. 기본 피처 정리
        X_train, X_test = self._clean_basic_features(X_train, X_test)
        
        # 2. 범주형 피처 인코딩
        X_train, X_test = self._encode_categorical_features(X_train, X_test, y_train)
        
        # 3. 수치형 피처 생성
        X_train, X_test = self._create_numerical_features(X_train, X_test)
        
        # 4. 상호작용 피처 생성
        if self.config.FEATURE_CONFIG['interaction_features']:
            X_train, X_test = self._create_interaction_features(X_train, X_test)
        
        # 5. 집계 피처 생성
        X_train, X_test = self._create_aggregation_features(X_train, X_test, y_train)
        
        # 6. 통계적 피처 생성
        if self.config.FEATURE_CONFIG['statistical_features']:
            X_train, X_test = self._create_statistical_features(X_train, X_test)
        
        # 7. 스케일링
        X_train, X_test = self._scale_features(X_train, X_test)
        
        # 8. 피처 선택
        X_train, X_test = self._select_features(X_train, X_test, y_train)
        
        logger.info(f"피처 엔지니어링 완료 - 최종 피처 수: {X_train.shape[1]}")
        
        return X_train, X_test
    
    def _clean_basic_features(self, 
                            X_train: pd.DataFrame, 
                            X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """기본 피처 정리"""
        logger.info("기본 피처 정리 시작")
        
        # 상수 피처 제거
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            X_train = X_train.drop(columns=constant_features)
            X_test = X_test.drop(columns=constant_features)
            logger.info(f"상수 피처 제거: {len(constant_features)}개")
        
        # 높은 결측치 비율 피처 제거 (80% 이상)
        high_missing_features = []
        for col in X_train.columns:
            missing_ratio = X_train[col].isnull().sum() / len(X_train)
            if missing_ratio > 0.8:
                high_missing_features.append(col)
        
        if high_missing_features:
            X_train = X_train.drop(columns=high_missing_features)
            X_test = X_test.drop(columns=high_missing_features)
            logger.info(f"높은 결측치 피처 제거: {len(high_missing_features)}개")
        
        return X_train, X_test
    
    def _encode_categorical_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """범주형 피처 인코딩"""
        logger.info("범주형 피처 인코딩 시작")
        
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # 빈도 기반 필터링
            freq_threshold = self.config.FEATURE_CONFIG['frequency_threshold']
            value_counts = X_train[col].value_counts()
            rare_values = value_counts[value_counts < freq_threshold].index
            
            # 희귀값을 'rare'로 대체
            X_train[col] = X_train[col].replace(rare_values, 'rare')
            X_test[col] = X_test[col].replace(rare_values, 'rare')
            
            # 테스트 데이터에서만 나타나는 값 처리
            train_values = set(X_train[col].unique())
            X_test[col] = X_test[col].apply(lambda x: x if x in train_values else 'unknown')
            
            # Target Encoding
            if X_train[col].nunique() > 2:  # 이진 변수가 아닌 경우
                te = TargetEncoder(
                    smoothing=self.config.FEATURE_CONFIG['target_encoding_smoothing']
                )
                
                # 학습 데이터로 인코더 훈련
                X_train[f'{col}_target_encoded'] = te.fit_transform(
                    X_train[[col]], y_train
                ).flatten()
                
                # 테스트 데이터 변환
                X_test[f'{col}_target_encoded'] = te.transform(
                    X_test[[col]]
                ).flatten()
                
                self.target_encoders[col] = te
                self.generated_features.append(f'{col}_target_encoded')
            
            # Label Encoding
            le = LabelEncoder()
            X_train[f'{col}_label_encoded'] = le.fit_transform(X_train[col].astype(str))
            
            # 테스트 데이터 변환 (unknown 값 처리)
            test_encoded = []
            for val in X_test[col].astype(str):
                if val in le.classes_:
                    test_encoded.append(le.transform([val])[0])
                else:
                    test_encoded.append(-1)  # unknown 값
            
            X_test[f'{col}_label_encoded'] = test_encoded
            self.label_encoders[col] = le
            self.generated_features.append(f'{col}_label_encoded')
            
            # 빈도 인코딩
            freq_map = X_train[col].value_counts().to_dict()
            X_train[f'{col}_frequency'] = X_train[col].map(freq_map)
            X_test[f'{col}_frequency'] = X_test[col].map(freq_map).fillna(0)
            self.generated_features.append(f'{col}_frequency')
        
        # 원본 범주형 컬럼 제거
        X_train = X_train.drop(columns=categorical_cols)
        X_test = X_test.drop(columns=categorical_cols)
        
        logger.info(f"범주형 피처 인코딩 완료 - 생성된 피처: {len(self.generated_features)}개")
        return X_train, X_test
    
    def _create_numerical_features(self, 
                                 X_train: pd.DataFrame, 
                                 X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """수치형 피처 생성"""
        logger.info("수치형 피처 생성 시작")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 로그 변환 (양수인 경우)
            if (X_train[col] > 0).all():
                X_train[f'{col}_log'] = np.log1p(X_train[col])
                X_test[f'{col}_log'] = np.log1p(X_test[col])
                self.generated_features.append(f'{col}_log')
            
            # 제곱근 변환 (양수인 경우)
            if (X_train[col] >= 0).all():
                X_train[f'{col}_sqrt'] = np.sqrt(X_train[col])
                X_test[f'{col}_sqrt'] = np.sqrt(X_test[col])
                self.generated_features.append(f'{col}_sqrt')
            
            # 이상치 기반 이진 피처
            q1 = X_train[col].quantile(0.25)
            q3 = X_train[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            X_train[f'{col}_is_outlier'] = (
                (X_train[col] < lower_bound) | (X_train[col] > upper_bound)
            ).astype(int)
            
            X_test[f'{col}_is_outlier'] = (
                (X_test[col] < lower_bound) | (X_test[col] > upper_bound)
            ).astype(int)
            
            self.generated_features.append(f'{col}_is_outlier')
            
            # 구간화 피처
            X_train[f'{col}_binned'] = pd.cut(X_train[col], bins=5, labels=False, duplicates='drop')
            X_test[f'{col}_binned'] = pd.cut(X_test[col], bins=5, labels=False, duplicates='drop')
            
            # NaN 값을 0으로 대체 (구간화에서 발생할 수 있음)
            X_train[f'{col}_binned'] = X_train[f'{col}_binned'].fillna(0)
            X_test[f'{col}_binned'] = X_test[f'{col}_binned'].fillna(0)
            
            self.generated_features.append(f'{col}_binned')
        
        logger.info(f"수치형 피처 생성 완료")
        return X_train, X_test
    
    def _create_interaction_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """상호작용 피처 생성"""
        logger.info("상호작용 피처 생성 시작")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # 수치형 변수 간 상호작용 (상위 10개 변수만)
        important_numeric_cols = numeric_cols[:10]
        
        interaction_count = 0
        for i in range(len(important_numeric_cols)):
            for j in range(i + 1, len(important_numeric_cols)):
                col1, col2 = important_numeric_cols[i], important_numeric_cols[j]
                
                # 곱셈 상호작용
                X_train[f'{col1}_mul_{col2}'] = X_train[col1] * X_train[col2]
                X_test[f'{col1}_mul_{col2}'] = X_test[col1] * X_test[col2]
                
                # 나눗셈 상호작용 (0으로 나누기 방지)
                X_train[f'{col1}_div_{col2}'] = X_train[col1] / (X_train[col2] + 1e-8)
                X_test[f'{col1}_div_{col2}'] = X_test[col1] / (X_test[col2] + 1e-8)
                
                self.generated_features.extend([f'{col1}_mul_{col2}', f'{col1}_div_{col2}'])
                interaction_count += 2
                
                # 너무 많은 상호작용 피처 생성 방지
                if interaction_count >= 50:
                    break
            
            if interaction_count >= 50:
                break
        
        logger.info(f"상호작용 피처 생성 완료 - 생성된 피처: {interaction_count}개")
        return X_train, X_test
    
    def _create_aggregation_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, 
                                   y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """집계 피처 생성 (그룹별 통계)"""
        logger.info("집계 피처 생성 시작")
        
        # 범주형으로 인코딩된 피처들을 그룹화 키로 사용
        categorical_encoded_cols = [col for col in X_train.columns 
                                  if 'label_encoded' in col or 'target_encoded' in col]
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in categorical_encoded_cols]
        
        # 상위 5개 범주형 피처만 사용 (계산 비용 고려)
        group_cols = categorical_encoded_cols[:5]
        
        for group_col in group_cols:
            if group_col not in X_train.columns:
                continue
                
            # 그룹별 수치형 피처 집계
            for num_col in numeric_cols[:10]:  # 상위 10개 수치형 피처만
                # 평균
                group_mean = X_train.groupby(group_col)[num_col].mean()
                X_train[f'{group_col}_{num_col}_mean'] = X_train[group_col].map(group_mean)
                X_test[f'{group_col}_{num_col}_mean'] = X_test[group_col].map(group_mean).fillna(group_mean.mean())
                
                # 표준편차
                group_std = X_train.groupby(group_col)[num_col].std()
                X_train[f'{group_col}_{num_col}_std'] = X_train[group_col].map(group_std)
                X_test[f'{group_col}_{num_col}_std'] = X_test[group_col].map(group_std).fillna(group_std.mean())
                
                self.generated_features.extend([
                    f'{group_col}_{num_col}_mean',
                    f'{group_col}_{num_col}_std'
                ])
            
            # 그룹별 타겟 통계 (학습 데이터만)
            group_target_mean = y_train.groupby(X_train[group_col]).mean()
            group_target_count = y_train.groupby(X_train[group_col]).count()
            
            X_train[f'{group_col}_target_mean'] = X_train[group_col].map(group_target_mean)
            X_train[f'{group_col}_target_count'] = X_train[group_col].map(group_target_count)
            
            X_test[f'{group_col}_target_mean'] = X_test[group_col].map(group_target_mean).fillna(group_target_mean.mean())
            X_test[f'{group_col}_target_count'] = X_test[group_col].map(group_target_count).fillna(group_target_count.mean())
            
            self.generated_features.extend([
                f'{group_col}_target_mean',
                f'{group_col}_target_count'
            ])
        
        logger.info(f"집계 피처 생성 완료")
        return X_train, X_test
    
    def _create_statistical_features(self, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """통계적 피처 생성"""
        logger.info("통계적 피처 생성 시작")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # 행별 통계 피처
        X_train['row_sum'] = X_train[numeric_cols].sum(axis=1)
        X_train['row_mean'] = X_train[numeric_cols].mean(axis=1)
        X_train['row_std'] = X_train[numeric_cols].std(axis=1)
        X_train['row_min'] = X_train[numeric_cols].min(axis=1)
        X_train['row_max'] = X_train[numeric_cols].max(axis=1)
        X_train['row_median'] = X_train[numeric_cols].median(axis=1)
        X_train['row_skew'] = X_train[numeric_cols].skew(axis=1)
        X_train['row_kurt'] = X_train[numeric_cols].kurtosis(axis=1)
        
        X_test['row_sum'] = X_test[numeric_cols].sum(axis=1)
        X_test['row_mean'] = X_test[numeric_cols].mean(axis=1)
        X_test['row_std'] = X_test[numeric_cols].std(axis=1)
        X_test['row_min'] = X_test[numeric_cols].min(axis=1)
        X_test['row_max'] = X_test[numeric_cols].max(axis=1)
        X_test['row_median'] = X_test[numeric_cols].median(axis=1)
        X_test['row_skew'] = X_test[numeric_cols].skew(axis=1)
        X_test['row_kurt'] = X_test[numeric_cols].kurtosis(axis=1)
        
        row_features = ['row_sum', 'row_mean', 'row_std', 'row_min', 'row_max', 
                       'row_median', 'row_skew', 'row_kurt']
        self.generated_features.extend(row_features)
        
        # NaN 값 처리
        for feature in row_features:
            X_train[feature] = X_train[feature].fillna(0)
            X_test[feature] = X_test[feature].fillna(0)
        
        logger.info(f"통계적 피처 생성 완료")
        return X_train, X_test
    
    def _scale_features(self, 
                       X_train: pd.DataFrame, 
                       X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """피처 스케일링"""
        logger.info("피처 스케일링 시작")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # 표준화
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        self.scalers['standard'] = scaler
        
        logger.info(f"피처 스케일링 완료")
        return X_train_scaled, X_test_scaled
    
    def _select_features(self, 
                        X_train: pd.DataFrame, 
                        X_test: pd.DataFrame, 
                        y_train: pd.Series,
                        k: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """피처 선택"""
        logger.info("피처 선택 시작")
        
        # 피처 수가 k보다 적으면 전체 사용
        if X_train.shape[1] <= k:
            logger.info(f"피처 수({X_train.shape[1]})가 k({k})보다 적어 전체 피처 사용")
            return X_train, X_test
        
        # 상호정보량 기반 피처 선택
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        X_train_selected = pd.DataFrame(
            selector.fit_transform(X_train, y_train),
            columns=X_train.columns[selector.get_support()],
            index=X_train.index
        )
        
        X_test_selected = pd.DataFrame(
            selector.transform(X_test),
            columns=X_test.columns[selector.get_support()],
            index=X_test.index
        )
        
        logger.info(f"피처 선택 완료 - 선택된 피처: {X_train_selected.shape[1]}개")
        
        return X_train_selected, X_test_selected
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """피처 중요도 요약 정보"""
        return {
            'total_generated_features': len(self.generated_features),
            'generated_features': self.generated_features,
            'encoders_count': {
                'target_encoders': len(self.target_encoders),
                'label_encoders': len(self.label_encoders),
                'scalers': len(self.scalers)
            }
        }