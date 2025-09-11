# data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
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
            
            # CTR 정보 로깅
            actual_ctr = self.train_data[self.target_column].mean()
            logger.info(f"실제 CTR: {actual_ctr:.4f}")
            
            return self.train_data, self.test_data
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            raise
    
    def load_data_chunked(self, 
                         max_train_size: int = 1000000,
                         max_test_size: int = 300000,
                         chunk_size: int = 100000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """메모리 효율적인 샘플링 기반 데이터 로딩"""
        logger.info(f"샘플링 기반 데이터 로딩 시작 - 학습: {max_train_size:,}, 테스트: {max_test_size:,}")
        
        try:
            # 학습 데이터 로딩 및 샘플링
            logger.info("학습 데이터 로딩 시작")
            
            # 전체 학습 데이터 로딩
            logger.info("전체 학습 데이터 로딩 중")
            train_df_full = pd.read_parquet(self.config.TRAIN_PATH)
            logger.info(f"전체 학습 데이터 로딩 완료: {train_df_full.shape}")
            
            # 필요한 크기로 샘플링
            if len(train_df_full) > max_train_size:
                logger.info(f"학습 데이터 샘플링: {len(train_df_full):,} → {max_train_size:,}")
                # 시간적 순서를 고려한 샘플링 (앞에서부터 선택)
                train_df = train_df_full.iloc[:max_train_size].copy()
                del train_df_full
                gc.collect()
            else:
                train_df = train_df_full
            
            # 메모리 최적화
            train_df = self._optimize_chunk_memory(train_df)
            logger.info(f"학습 데이터 처리 완료: {train_df.shape}")
            
            # 테스트 데이터 로딩 및 샘플링
            logger.info("테스트 데이터 로딩 시작")
            
            # 전체 테스트 데이터 로딩
            logger.info("전체 테스트 데이터 로딩 중")
            test_df_full = pd.read_parquet(self.config.TEST_PATH)
            logger.info(f"전체 테스트 데이터 로딩 완료: {test_df_full.shape}")
            
            # 필요한 크기로 샘플링
            if len(test_df_full) > max_test_size:
                logger.info(f"테스트 데이터 샘플링: {len(test_df_full):,} → {max_test_size:,}")
                # 시간적 순서를 고려한 샘플링 (앞에서부터 선택)
                test_df = test_df_full.iloc[:max_test_size].copy()
                del test_df_full
                gc.collect()
            else:
                test_df = test_df_full
            
            # 메모리 최적화
            test_df = self._optimize_chunk_memory(test_df)
            logger.info(f"테스트 데이터 처리 완료: {test_df.shape}")
            
            # 피처 컬럼 정의
            self.feature_columns = [col for col in train_df.columns 
                                  if col != self.target_column]
            
            # CTR 정보
            if self.target_column in train_df.columns:
                actual_ctr = train_df[self.target_column].mean()
                logger.info(f"실제 CTR: {actual_ctr:.4f}")
            
            logger.info("샘플링 기반 데이터 로딩 완료")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"샘플링 데이터 로딩 실패: {str(e)}")
            gc.collect()
            raise
    
    def _optimize_chunk_memory(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """청크별 메모리 최적화"""
        try:
            # 데이터 타입 최적화
            for col in chunk.columns:
                if chunk[col].dtype == 'float64':
                    chunk[col] = pd.to_numeric(chunk[col], downcast='float')
                elif chunk[col].dtype in ['int64', 'int32']:
                    chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
                elif chunk[col].dtype == 'object':
                    # 범주형으로 변환 시도
                    if chunk[col].nunique() / len(chunk) < 0.5:
                        chunk[col] = chunk[col].astype('category')
            
            return chunk
        except Exception as e:
            logger.warning(f"청크 메모리 최적화 실패: {str(e)}")
            return chunk
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리 수행"""
        logger.info("기본 전처리 시작")
        
        # 복사본 생성
        df_processed = df.copy()
        
        # 결측치 확인
        missing_info = df_processed.isnull().sum()
        if missing_info.sum() > 0:
            logger.warning(f"결측치 발견: {missing_info.sum()}개")
            
            # 수치형 변수: 중앙값으로 대치
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_processed[col].isnull().sum() > 0:
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
            
            # 범주형 변수: 최빈값으로 대치
            categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df_processed[col].isnull().sum() > 0:
                    mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
        
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
            if num_unique < df.shape[0] * 0.5:
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
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
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
    
    def create_time_series_folds(self, 
                                X: pd.DataFrame, 
                                y: pd.Series, 
                                n_splits: int = None) -> TimeSeriesSplit:
        """시간적 순서를 고려한 교차검증 폴드 생성"""
        if n_splits is None:
            n_splits = self.config.N_SPLITS
        
        # TimeSeriesSplit 사용으로 시간적 순서 보장
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"시간적 교차검증 {n_splits}개 폴드 설정 완료")
        return tscv
    
    def create_cross_validation_folds(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series, 
                                    n_splits: int = None) -> TimeSeriesSplit:
        """CTR 데이터에 적합한 교차검증 폴드 생성"""
        if n_splits is None:
            n_splits = self.config.N_SPLITS
        
        # CTR 데이터는 시간적 순서가 중요하므로 TimeSeriesSplit 사용
        return self.create_time_series_folds(X, y, n_splits)
    
    def memory_efficient_train_test_split(self,
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """메모리 효율적인 시간적 순서 기반 데이터 분할"""
        logger.info("메모리 효율적인 데이터 분할 시작")
        
        try:
            # 입력 검증
            if len(X_train) != len(y_train):
                raise ValueError(f"X와 y의 길이가 다릅니다. X: {len(X_train)}, y: {len(y_train)}")
            
            if len(X_train) == 0:
                raise ValueError("분할할 데이터가 없습니다.")
            
            # 시간적 순서를 고려한 분할 인덱스 계산
            split_idx = int(len(X_train) * (1 - test_size))
            
            logger.info(f"분할 인덱스: {split_idx}, 전체 크기: {len(X_train)}")
            
            # 분할 수행
            X_train_split = X_train.iloc[:split_idx].copy()
            X_val_split = X_train.iloc[split_idx:].copy()
            y_train_split = y_train.iloc[:split_idx].copy()
            y_val_split = y_train.iloc[split_idx:].copy()
            
            # 메모리 정리
            del X_train, y_train
            gc.collect()
            
            # 결과 검증
            self._validate_split_results(X_train_split, X_val_split, y_train_split, y_val_split)
            
            logger.info(f"메모리 효율적 분할 완료 - 학습: {X_train_split.shape}, 검증: {X_val_split.shape}")
            
            return X_train_split, X_val_split, y_train_split, y_val_split
            
        except Exception as e:
            logger.error(f"메모리 효율적 데이터 분할 실패: {str(e)}")
            gc.collect()
            raise
    
    def train_test_split_data(self, 
                            df: pd.DataFrame, 
                            test_size: float = None,
                            target_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """시간적 순서를 고려한 학습/검증 데이터 분할"""
        logger.info("시간적 순서를 고려한 학습/검증 데이터 분할 시작")
        
        if test_size is None:
            test_size = self.config.TEST_SIZE
        
        if target_col is None:
            target_col = self.target_column
        
        # 타겟 컬럼 존재 확인
        if target_col not in df.columns:
            raise ValueError(f"타겟 컬럼 '{target_col}'이 데이터에 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
        
        # 데이터 전처리 및 검증
        df_clean = self._preprocess_for_split(df, target_col)
        
        # 피처와 타겟 분리
        X, y = self._safe_feature_target_split(df_clean, target_col)
        
        # 시간적 순서를 고려한 분할 수행
        try:
            X_train, X_val, y_train, y_val = self._perform_temporal_split(X, y, test_size)
            
            logger.info(f"데이터 분할 완료 - 학습: {X_train.shape}, 검증: {X_val.shape}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.error(f"데이터 분할 실패: {str(e)}")
            logger.error(f"X 형태: {X.shape if X is not None else 'None'}, y 형태: {y.shape if y is not None else 'None'}")
            if y is not None:
                logger.error(f"y 유니크 값: {y.unique()}")
            raise
    
    def _perform_temporal_split(self, X: pd.DataFrame, y: pd.Series, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """시간적 순서를 고려한 데이터 분할"""
        logger.info("시간적 순서 기반 데이터 분할 수행")
        
        # 입력 데이터 최종 검증
        if X is None or y is None:
            raise ValueError("X 또는 y가 None입니다.")
        
        if len(X) != len(y):
            raise ValueError(f"X와 y의 길이가 다릅니다. X: {len(X)}, y: {len(y)}")
        
        if len(X) == 0:
            raise ValueError("분할할 데이터가 없습니다.")
        
        # 시간적 순서를 고려한 분할 (처음 80%를 학습, 마지막 20%를 검증)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        # 분할 결과 검증
        self._validate_split_results(X_train, X_val, y_train, y_val)
        
        logger.info("시간적 순서 기반 분할 성공")
        return X_train, X_val, y_train, y_val
    
    def _preprocess_for_split(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """분할 전 데이터 전처리"""
        logger.info("분할 전 데이터 전처리 시작")
        
        df_clean = df.copy()
        
        # 1. 컬럼명 정리
        df_clean.columns = [str(col) if col is not None else f'col_{i}' for i, col in enumerate(df_clean.columns)]
        
        # 2. None 값이나 빈 문자열 컬럼명 처리
        new_columns = []
        for i, col in enumerate(df_clean.columns):
            if col is None or str(col).strip() == '' or str(col) == 'None':
                new_columns.append(f'feature_{i}')
            else:
                new_columns.append(str(col))
        df_clean.columns = new_columns
        
        # 3. 중복 컬럼명 처리
        seen_columns = {}
        final_columns = []
        for col in df_clean.columns:
            if col in seen_columns:
                seen_columns[col] += 1
                final_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                final_columns.append(col)
        df_clean.columns = final_columns
        
        # 4. 타겟 컬럼 검증
        if target_col not in df_clean.columns:
            raise ValueError(f"전처리 후 타겟 컬럼 '{target_col}'이 없습니다.")
        
        # 5. 타겟 결측치 처리
        if df_clean[target_col].isnull().any():
            logger.warning("타겟 변수에 결측치 발견, 해당 행 제거")
            df_clean = df_clean.dropna(subset=[target_col])
        
        # 6. 타겟 값 검증
        unique_targets = df_clean[target_col].unique()
        if len(unique_targets) < 2:
            raise ValueError(f"타겟 변수의 유니크 값이 부족합니다: {unique_targets}")
        
        logger.info(f"전처리 후 데이터 형태: {df_clean.shape}")
        return df_clean
    
    def _safe_feature_target_split(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """안전한 피처-타겟 분리"""
        logger.info("피처-타겟 분리 시작")
        
        # 피처 컬럼 동적 계산
        all_columns = list(df.columns)
        feature_columns = [col for col in all_columns if col != target_col]
        
        # 유효한 피처 컬럼만 선택
        valid_feature_columns = []
        for col in feature_columns:
            if col is not None and str(col).strip() != '' and col in df.columns:
                valid_feature_columns.append(col)
        
        if len(valid_feature_columns) == 0:
            raise ValueError("사용 가능한 피처 컬럼이 없습니다.")
        
        logger.info(f"유효한 피처 컬럼 수: {len(valid_feature_columns)}")
        
        # 피처와 타겟 분리
        X = df[valid_feature_columns].copy()
        y = df[target_col].copy()
        
        # 피처 데이터 정리
        X = self._clean_feature_data(X)
        
        # 타겟 데이터 정리
        y = self._clean_target_data(y)
        
        # 길이 확인
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            logger.warning(f"X와 y 길이 불일치, 최소 길이로 맞춤: {min_len}")
        
        return X, y
    
    def _clean_feature_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """피처 데이터 정리"""
        logger.info("피처 데이터 정리 시작")
        
        # object 타입 컬럼 처리
        object_columns = X.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            logger.warning(f"object 타입 컬럼 발견: {object_columns}")
            for col in object_columns:
                try:
                    # 수치형 변환 시도
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    logger.info(f"{col}: 수치형으로 변환")
                except:
                    # 변환 실패 시 제거
                    logger.warning(f"{col}: 변환 실패로 제거")
                    X = X.drop(columns=[col])
        
        # 결측치 처리
        if X.isnull().any().any():
            logger.warning("피처 데이터에 결측치 발견, 0으로 대치")
            X = X.fillna(0)
        
        # 무한값 처리
        if np.isinf(X.values).any():
            logger.warning("피처 데이터에 무한값 발견, 클리핑 처리")
            X = X.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 데이터 타입 통일 (float32)
        for col in X.columns:
            try:
                if X[col].dtype != 'float32':
                    X[col] = X[col].astype('float32')
            except:
                logger.warning(f"{col}: float32 변환 실패, 제거")
                X = X.drop(columns=[col])
        
        return X
    
    def _clean_target_data(self, y: pd.Series) -> pd.Series:
        """타겟 데이터 정리"""
        logger.info("타겟 데이터 정리 시작")
        
        # 결측치 확인
        if y.isnull().any():
            logger.warning("타겟 데이터에 결측치 발견")
            # 결측치가 있는 인덱스 기록
            null_indices = y.isnull()
            if null_indices.sum() > 0:
                logger.warning(f"결측치 개수: {null_indices.sum()}")
        
        # 타겟 값을 정수형으로 변환
        try:
            y = y.astype('int8')
        except:
            logger.warning("타겟을 int8로 변환 실패, int32 사용")
            y = y.astype('int32')
        
        return y
    
    def _validate_split_results(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                               y_train: pd.Series, y_val: pd.Series):
        """분할 결과 검증"""
        logger.info("분할 결과 검증 시작")
        
        # 기본 형태 검증
        assert len(X_train) == len(y_train), f"학습 데이터 길이 불일치: {len(X_train)} != {len(y_train)}"
        assert len(X_val) == len(y_val), f"검증 데이터 길이 불일치: {len(X_val)} != {len(y_val)}"
        assert len(X_train.columns) == len(X_val.columns), f"피처 수 불일치: {len(X_train.columns)} != {len(X_val.columns)}"
        
        # 컬럼 일치성 검증
        if list(X_train.columns) != list(X_val.columns):
            logger.error("학습/검증 데이터의 컬럼이 다릅니다!")
            raise ValueError("학습/검증 데이터의 컬럼 불일치")
        
        # 데이터 타입 검증
        for col in X_train.columns:
            if X_train[col].dtype != X_val[col].dtype:
                logger.warning(f"{col}: 데이터 타입 불일치 - train: {X_train[col].dtype}, val: {X_val[col].dtype}")
        
        # 타겟 분포 확인
        train_dist = y_train.value_counts()
        val_dist = y_val.value_counts()
        
        logger.info(f"학습 데이터 타겟 분포: {train_dist.to_dict()}")
        logger.info(f"검증 데이터 타겟 분포: {val_dist.to_dict()}")
        
        # 최소 요구사항 확인
        if len(X_train) < 10 or len(X_val) < 10:
            logger.warning("분할된 데이터 크기가 매우 작습니다!")
        
        if X_train.shape[1] == 0:
            raise ValueError("피처가 하나도 없습니다!")
        
        logger.info("분할 결과 검증 완료")
    
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
                if corr > 0.95:
                    high_correlation_features.append({
                        'feature': col,
                        'correlation': corr
                    })
        
        leakage_results['high_correlation_features'] = high_correlation_features
        
        return leakage_results