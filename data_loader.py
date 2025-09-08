# data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """데이터 로딩 및 기본 전처리 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.train_data = None
        self.test_data = None
        self.feature_columns = None
        self.target_column = 'clicked'
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """학습 및 테스트 데이터 로딩"""
        try:
            logger.info("데이터 로딩 시작")
            
            # 학습 데이터 로딩
            self.train_data = pd.read_parquet(self.config.TRAIN_PATH)
            logger.info(f"학습 데이터 형태: {self.train_data.shape}")
            
            # 테스트 데이터 로딩
            self.test_data = pd.read_parquet(self.config.TEST_PATH)
            logger.info(f"테스트 데이터 형태: {self.test_data.shape}")
            
            # 피처 컬럼 정의 (타겟 제외)
            self.feature_columns = [col for col in self.train_data.columns 
                                  if col != self.target_column]
            logger.info(f"피처 컬럼 수: {len(self.feature_columns)}")
            
            return self.train_data, self.test_data
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            raise
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리 수행"""
        logger.info("기본 전처리 시작")
        
        # 복사본 생성
        df_processed = df.copy()
        
        # 결측치 확인
        missing_info = df_processed.isnull().sum()
        if missing_info.sum() > 0:
            logger.warning(f"결측치 발견: \n{missing_info[missing_info > 0]}")
            
            # 수치형 변수: 중앙값으로 대치
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
                    logger.info(f"{col}: 결측치를 중앙값 {median_val}로 대치")
            
            # 범주형 변수: 최빈값으로 대치
            categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df_processed[col].isnull().sum() > 0:
                    mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
                    logger.info(f"{col}: 결측치를 최빈값 {mode_val}로 대치")
        
        # 데이터 타입 최적화
        df_processed = self._optimize_dtypes(df_processed)
        
        logger.info("기본 전처리 완료")
        return df_processed
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """메모리 사용량 최적화를 위한 데이터 타입 변환"""
        logger.info("데이터 타입 최적화 시작")
        
        # 정수형 최적화
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # 실수형 최적화
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # 범주형 최적화
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            num_unique = df[col].nunique()
            if num_unique < df.shape[0] * 0.5:  # 유니크 값이 전체의 50% 미만이면 카테고리로 변환
                df[col] = df[col].astype('category')
        
        logger.info("데이터 타입 최적화 완료")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 요약 정보 생성"""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB 단위
        }
        
        # 수치형 변수 통계
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # 범주형 변수 통계
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical_summary'] = {}
            for col in categorical_cols:
                summary['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head().to_dict()
                }
        
        # 타겟 변수 분포 (학습 데이터인 경우)
        if self.target_column in df.columns:
            target_dist = df[self.target_column].value_counts()
            summary['target_distribution'] = {
                'counts': target_dist.to_dict(),
                'proportions': (target_dist / len(df)).to_dict(),
                'ctr': df[self.target_column].mean()
            }
        
        return summary
    
    def create_cross_validation_folds(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series, 
                                    n_splits: int = None) -> StratifiedKFold:
        """계층화된 교차검증 폴드 생성"""
        if n_splits is None:
            n_splits = self.config.N_SPLITS
        
        skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=self.config.RANDOM_STATE
        )
        
        logger.info(f"{n_splits}개 폴드로 교차검증 설정 완료")
        return skf
    
    def train_test_split_data(self, 
                            df: pd.DataFrame, 
                            test_size: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """학습/검증 데이터 분할"""
        if test_size is None:
            test_size = self.config.TEST_SIZE
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"데이터 분할 완료 - 학습: {X_train.shape}, 검증: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    
    def load_submission_template(self) -> pd.DataFrame:
        """제출 템플릿 로딩"""
        try:
            submission = pd.read_csv(self.config.SUBMISSION_PATH)
            logger.info(f"제출 템플릿 로딩 완료: {submission.shape}")
            return submission
        except Exception as e:
            logger.error(f"제출 템플릿 로딩 실패: {str(e)}")
            raise
    
    def save_processed_data(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          suffix: str = "processed"):
        """전처리된 데이터 저장"""
        try:
            train_path = self.config.DATA_DIR / f"train_{suffix}.parquet"
            test_path = self.config.DATA_DIR / f"test_{suffix}.parquet"
            
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path, index=False)
            
            logger.info(f"전처리 데이터 저장 완료: {train_path}, {test_path}")
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {str(e)}")
            raise

class DataValidator:
    """데이터 품질 검증 클래스"""
    
    @staticmethod
    def validate_data_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """학습/테스트 데이터 일관성 검증"""
        validation_results = {}
        
        # 컬럼 일관성 검증
        train_cols = set(train_df.columns) - {'clicked'}
        test_cols = set(test_df.columns)
        
        validation_results['missing_in_test'] = list(train_cols - test_cols)
        validation_results['extra_in_test'] = list(test_cols - train_cols)
        
        # 데이터 타입 일관성 검증
        common_cols = train_cols & test_cols
        dtype_mismatches = []
        
        for col in common_cols:
            if train_df[col].dtype != test_df[col].dtype:
                dtype_mismatches.append({
                    'column': col,
                    'train_dtype': str(train_df[col].dtype),
                    'test_dtype': str(test_df[col].dtype)
                })
        
        validation_results['dtype_mismatches'] = dtype_mismatches
        
        # 범위 일관성 검증 (수치형 변수)
        range_issues = []
        numeric_cols = [col for col in common_cols if train_df[col].dtype in ['int64', 'float64']]
        
        for col in numeric_cols:
            train_min, train_max = train_df[col].min(), train_df[col].max()
            test_min, test_max = test_df[col].min(), test_df[col].max()
            
            if test_min < train_min or test_max > train_max:
                range_issues.append({
                    'column': col,
                    'train_range': [train_min, train_max],
                    'test_range': [test_min, test_max]
                })
        
        validation_results['range_issues'] = range_issues
        
        return validation_results
    
    @staticmethod
    def check_data_leakage(df: pd.DataFrame, target_col: str = 'clicked') -> Dict[str, Any]:
        """데이터 누수 가능성 검증"""
        leakage_results = {}
        
        # 타겟과 완전히 상관관계가 있는 피처 탐지
        feature_cols = [col for col in df.columns if col != target_col]
        high_correlation_features = []
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                corr = abs(df[col].corr(df[target_col]))
                if corr > 0.95:  # 95% 이상 상관관계
                    high_correlation_features.append({
                        'feature': col,
                        'correlation': corr
                    })
        
        leakage_results['high_correlation_features'] = high_correlation_features
        
        return leakage_results