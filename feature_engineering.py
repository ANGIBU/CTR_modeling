# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Union
import logging
import time
import gc
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from config import Config
from data_loader import MemoryMonitor

logger = logging.getLogger(__name__)

class CTRFeatureEngineer:
    """CTR feature engineering class"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        # Processing modes
        self.quick_mode = False
        self.memory_efficient_mode = True
        
        # Feature engineering state variables
        self.target_encoders = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}
        self.generated_features = []
        self.numerical_features = []
        self.categorical_features = []
        self.interaction_features = []
        self.target_encoding_features = []
        self.temporal_features = []
        self.statistical_features = []
        self.frequency_features = []
        self.polynomial_features = []
        self.cross_features = []
        self.binning_features = []
        self.rank_features = []
        self.ratio_features = []
        self.removed_columns = []
        self.original_feature_order = []
        self.final_feature_columns = []
        self.target_column = None
        
        # Processing statistics for monitoring
        self.processing_stats = {
            'start_time': None,
            'processing_time': 0,
            'memory_usage': 0,
            'feature_types_count': {},
            'total_features_generated': 0
        }
    
    def set_memory_efficient_mode(self, enabled: bool):
        """Enable or disable memory efficient mode"""
        self.memory_efficient_mode = enabled
        if enabled:
            logger.info("Memory efficient mode enabled - simplified features only")
        else:
            logger.info("Memory efficient mode disabled - full feature engineering")
    
    def set_quick_mode(self, enabled: bool):
        """Enable or disable quick mode for rapid testing"""
        self.quick_mode = enabled
        if enabled:
            logger.info("Quick mode enabled - basic features only for rapid testing")
        else:
            logger.info("Quick mode disabled - comprehensive feature engineering")
    
    def engineer_features(self, 
                         train_df: pd.DataFrame, 
                         test_df: pd.DataFrame, 
                         target_col: str = 'clicked') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Main feature engineering entry point"""
        try:
            logger.info("=== CTR Feature engineering started ===")
            
            if self.quick_mode:
                return self.create_quick_features(train_df, test_df, target_col)
            else:
                return self.create_all_features(train_df, test_df, target_col)
                
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return None, None
    
    def create_quick_features(self, 
                            train_df: pd.DataFrame, 
                            test_df: pd.DataFrame, 
                            target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Quick feature engineering for rapid testing"""
        logger.info("Creating basic features for quick testing")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            # 1. Basic data preparation
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            # 2. Column classification
            self._classify_columns(X_train)
            
            # 3. Data type unification
            X_train, X_test = self._unify_data_types_safe(X_train, X_test)
            
            # 4. Fill missing values with safe categorical handling
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            # 5. Safe categorical encoding
            X_train, X_test = self._encode_categorical_safe(X_train, X_test)
            
            # 6. Simple numeric normalization
            X_train, X_test = self._normalize_numeric_basic(X_train, X_test)
            
            # 7. Remove any problematic columns
            X_train, X_test = self._clean_final_features(X_train, X_test)
            
            # Store final feature order
            self.final_feature_columns = list(X_train.columns)
            
            logger.info(f"Quick feature engineering completed: {X_train.shape[1]} features")
            logger.info(f"Final features - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Quick feature engineering failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def create_all_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          target_col: str = 'clicked') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete feature engineering pipeline for full dataset processing"""
        logger.info("Creating comprehensive features for full dataset")
        
        try:
            self._initialize_processing(train_df, test_df, target_col)
            
            # 1. Basic data preparation
            X_train, X_test, y_train = self._prepare_basic_data(train_df, test_df, target_col)
            
            # 2. Column classification
            self._classify_columns(X_train)
            
            # 3. Safe data type unification
            X_train, X_test = self._unify_data_types_safe(X_train, X_test)
            
            # 4. Basic feature cleanup
            X_train, X_test = self._clean_basic_features(X_train, X_test)
            
            # Memory check for full processing
            memory_status = self.memory_monitor.get_memory_status()
            if not memory_status['should_cleanup'] and memory_status['available_gb'] > 8:
                logger.info("Full feature engineering enabled - sufficient memory available")
                
                # 5. Target encoding features
                X_train, X_test = self._create_target_encoding_features(X_train, X_test, y_train)
                
                # 6. Interaction feature creation
                X_train, X_test = self._create_interaction_features(X_train, X_test, y_train)
                
                # 7. Statistical features
                X_train, X_test = self._create_statistical_features(X_train, X_test)
                
                # 8. Frequency-based features
                X_train, X_test = self._create_frequency_features(X_train, X_test)
                
                # 9. Cross features
                X_train, X_test = self._create_cross_features(X_train, X_test, y_train)
                
                # 10. Time-based features
                X_train, X_test = self._create_temporal_features(X_train, X_test)
                
                # 11. Binning features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_binning', True):
                    X_train, X_test = self._create_binning_features_fixed(X_train, X_test)
                
                # 12. Rank features
                X_train, X_test = self._create_rank_features(X_train, X_test)
                
                # 13. Ratio features
                X_train, X_test = self._create_ratio_features(X_train, X_test)
                
                # 14. Polynomial features
                if self.config.FEATURE_ENGINEERING_CONFIG.get('enable_polynomial_features', True):
                    X_train, X_test = self._create_polynomial_features(X_train, X_test)
                
            else:
                logger.warning("Simplified feature engineering due to memory constraints")
            
            # 15. Safe categorical feature encoding
            X_train, X_test = self._encode_categorical_features_safe(X_train, X_test, y_train)
            
            # 16. Numeric feature transformation
            X_train, X_test = self._create_numeric_features(X_train, X_test)
            
            # 17. Final data cleanup
            X_train, X_test = self._final_data_cleanup(X_train, X_test)
            
            self._finalize_processing(X_train, X_test)
            
            logger.info(f"=== Feature engineering completed: {X_train.shape} ===")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            self.memory_monitor.force_memory_cleanup()
            
            # Use basic features only on failure
            logger.warning("Using basic features only due to error")
            return self._create_basic_features_only(train_df, test_df, target_col)
    
    def _initialize_processing(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """Initialize feature engineering processing"""
        try:
            self.processing_stats['start_time'] = time.time()
            
            # Target column detection
            self.target_column = self._detect_target_column(train_df, target_col)
            
            self.original_feature_order = sorted([col for col in train_df.columns if col != self.target_column])
            
            mode_info = "QUICK MODE" if self.quick_mode else "FULL MODE"
            logger.info(f"Feature engineering initialization ({mode_info})")
            logger.info(f"Initial data: Training {train_df.shape}, Test {test_df.shape}")
            logger.info(f"Target column: {self.target_column}")
            logger.info(f"Original feature count: {len(self.original_feature_order)}")
            
            self.memory_monitor.log_memory_status("Initialization")
            
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            self.target_column = target_col
    
    def _detect_target_column(self, train_df: pd.DataFrame, provided_target_col: str = None) -> str:
        """Detect CTR target column with validation"""
        try:
            # First, try provided target column
            if provided_target_col and provided_target_col in train_df.columns:
                unique_values = train_df[provided_target_col].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    positive_ratio = train_df[provided_target_col].mean()
                    logger.info(f"Target column confirmed: {provided_target_col} (CTR: {positive_ratio:.4f})")
                    return provided_target_col
            
            # Search for common CTR target column names
            ctr_target_names = ['clicked', 'click', 'is_click', 'target', 'label', 'y']
            
            for col_name in ctr_target_names:
                if col_name in train_df.columns:
                    unique_values = train_df[col_name].dropna().unique()
                    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                        positive_ratio = train_df[col_name].mean()
                        logger.info(f"Target column detected: {col_name} (CTR: {positive_ratio:.4f})")
                        return col_name
            
            # Fallback
            return provided_target_col or 'clicked'
            
        except Exception as e:
            logger.warning(f"Target column detection failed: {e}")
            return provided_target_col or 'clicked'
    
    def _prepare_basic_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepare basic data for feature engineering"""
        try:
            # Extract target
            if target_col in train_df.columns:
                y_train = train_df[target_col].copy()
                X_train = train_df.drop(columns=[target_col]).copy()
            else:
                logger.warning(f"Target column '{target_col}' not found, using zeros")
                y_train = pd.Series([0] * len(train_df))
                X_train = train_df.copy()
            
            X_test = test_df.copy()
            
            # Ensure consistent columns
            common_cols = sorted(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            logger.info(f"Data preparation completed - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def _classify_columns(self, X: pd.DataFrame):
        """Classify columns into numerical and categorical"""
        try:
            self.numerical_features = []
            self.categorical_features = []
            
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    unique_count = X[col].nunique()
                    if unique_count > 20:  # Treat as numerical
                        self.numerical_features.append(col)
                    else:
                        self.categorical_features.append(col)
                else:
                    self.categorical_features.append(col)
            
            logger.info(f"Column classification completed - Numeric: {len(self.numerical_features)}, Categorical: {len(self.categorical_features)}")
            
        except Exception as e:
            logger.warning(f"Column classification failed: {e}")
    
    def _unify_data_types_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe data type unification"""
        try:
            for col in X_train.columns:
                if col in X_test.columns:
                    train_type = X_train[col].dtype
                    test_type = X_test[col].dtype
                    
                    if train_type != test_type:
                        # Convert to string as fallback
                        X_train[col] = X_train[col].astype('object')
                        X_test[col] = X_test[col].astype('object')
            
            return X_train, X_test
            
        except Exception as e:
            logger.warning(f"Data type unification failed: {e}")
            return X_train, X_test
    
    def _safe_fillna(self, X: pd.DataFrame) -> pd.DataFrame:
        """Safe missing value filling"""
        try:
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna('unknown')
            
            return X
            
        except Exception as e:
            logger.warning(f"Safe fillna failed: {e}")
            return X
    
    def _encode_categorical_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical encoding"""
        try:
            for col in self.categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    # Use label encoder
                    le = LabelEncoder()
                    
                    # Combine train and test for consistent encoding
                    combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
                    le.fit(combined)
                    
                    X_train[col] = le.transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))
                    
                    self.label_encoders[col] = le
            
            return X_train, X_test
            
        except Exception as e:
            logger.warning(f"Categorical encoding failed: {e}")
            return X_train, X_test
    
    def _normalize_numeric_basic(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic numeric normalization"""
        try:
            for col in self.numerical_features:
                if col in X_train.columns and col in X_test.columns:
                    scaler = StandardScaler()
                    X_train[col] = scaler.fit_transform(X_train[[col]])
                    X_test[col] = scaler.transform(X_test[[col]])
                    
                    self.scalers[col] = scaler
            
            return X_train, X_test
            
        except Exception as e:
            logger.warning(f"Numeric normalization failed: {e}")
            return X_train, X_test
    
    def _clean_final_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean final features"""
        try:
            # Remove infinite values
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            # Remove NaN values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            return X_train, X_test
            
        except Exception as e:
            logger.warning(f"Final feature cleaning failed: {e}")
            return X_train, X_test
    
    def _create_target_encoding_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create CTR-optimized target encoding features"""
        try:
            logger.info("Creating CTR target encoding features")
            
            categorical_cols = [col for col in self.categorical_features if col in X_train.columns][:8]
            
            for col in categorical_cols:
                try:
                    # Use smoothed target encoding for better CTR prediction
                    encoder = TargetEncoder(
                        categories='auto',
                        target_type='binary',
                        smooth='auto',
                        cv=5,
                        random_state=42
                    )
                    
                    encoded_train = encoder.fit_transform(X_train[[col]], y_train)
                    encoded_test = encoder.transform(X_test[[col]])
                    
                    feature_name = f"{col}_target_enc"
                    X_train[feature_name] = encoded_train.ravel()
                    X_test[feature_name] = encoded_test.ravel()
                    
                    # CTR-specific statistic features
                    value_counts = X_train[col].value_counts()
                    ctr_by_category = X_train.groupby(col).apply(lambda x: y_train.iloc[x.index].mean())
                    
                    # Category frequency feature
                    freq_feature = f"{col}_freq"
                    X_train[freq_feature] = X_train[col].map(value_counts)
                    X_test[freq_feature] = X_test[col].map(value_counts).fillna(0)
                    
                    # Category CTR variance feature
                    ctr_var_feature = f"{col}_ctr_var"
                    ctr_variance = X_train.groupby(col).apply(lambda x: y_train.iloc[x.index].var()).fillna(0)
                    X_train[ctr_var_feature] = X_train[col].map(ctr_variance)
                    X_test[ctr_var_feature] = X_test[col].map(ctr_variance).fillna(0)
                    
                    self.target_encoding_features.extend([feature_name, freq_feature, ctr_var_feature])
                    self.target_encoders[col] = encoder
                    
                except Exception as e:
                    logger.warning(f"Target encoding failed for {col}: {e}")
                    continue
            
            logger.info(f"Target encoding completed: {len(self.target_encoding_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Target encoding features failed: {e}")
            return X_train, X_test
    
    def _create_interaction_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create CTR-optimized interaction features"""
        try:
            logger.info("Creating CTR interaction features")
            
            # Select most important features for interactions
            important_features = self._select_important_features(X_train, y_train, max_features=10)
            
            interaction_count = 0
            max_interactions = 15
            
            for i in range(len(important_features)):
                for j in range(i + 1, len(important_features)):
                    if interaction_count >= max_interactions:
                        break
                    
                    feat1, feat2 = important_features[i], important_features[j]
                    
                    try:
                        # Multiplication interaction
                        mult_feature = f"{feat1}_x_{feat2}"
                        X_train[mult_feature] = X_train[feat1] * X_train[feat2]
                        X_test[mult_feature] = X_test[feat1] * X_test[feat2]
                        
                        # Addition interaction
                        add_feature = f"{feat1}_plus_{feat2}"
                        X_train[add_feature] = X_train[feat1] + X_train[feat2]
                        X_test[add_feature] = X_test[feat1] + X_test[feat2]
                        
                        self.interaction_features.extend([mult_feature, add_feature])
                        interaction_count += 2
                        
                    except Exception as e:
                        logger.warning(f"Interaction feature creation failed for {feat1}x{feat2}: {e}")
                        continue
            
            logger.info(f"Interaction features completed: {len(self.interaction_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Interaction features failed: {e}")
            return X_train, X_test
    
    def _select_important_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 10) -> List[str]:
        """Select most important features for interaction creation"""
        try:
            selector = SelectKBest(score_func=f_classif, k=min(max_features, X.shape[1]))
            selector.fit(X, y)
            
            feature_scores = selector.scores_
            feature_names = X.columns[selector.get_support()]
            
            # Sort by score
            scored_features = list(zip(feature_names, feature_scores))
            scored_features.sort(key=lambda x: x[1], reverse=True)
            
            return [feat[0] for feat in scored_features]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return list(X.columns)[:max_features]
    
    def _create_statistical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create statistical aggregation features"""
        try:
            logger.info("Creating statistical features")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns][:6]
            
            if len(numeric_features) >= 2:
                # Row-wise statistical features
                X_train['row_mean'] = X_train[numeric_features].mean(axis=1)
                X_test['row_mean'] = X_test[numeric_features].mean(axis=1)
                
                X_train['row_std'] = X_train[numeric_features].std(axis=1)
                X_test['row_std'] = X_test[numeric_features].std(axis=1)
                
                X_train['row_max'] = X_train[numeric_features].max(axis=1)
                X_test['row_max'] = X_test[numeric_features].max(axis=1)
                
                X_train['row_min'] = X_train[numeric_features].min(axis=1)
                X_test['row_min'] = X_test[numeric_features].min(axis=1)
                
                self.statistical_features.extend(['row_mean', 'row_std', 'row_max', 'row_min'])
            
            logger.info(f"Statistical features completed: {len(self.statistical_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Statistical features failed: {e}")
            return X_train, X_test
    
    def _create_frequency_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create frequency-based features for CTR prediction"""
        try:
            logger.info("Creating frequency features")
            
            categorical_cols = [col for col in self.categorical_features if col in X_train.columns][:5]
            
            for col in categorical_cols:
                try:
                    # Value frequency
                    freq_map = X_train[col].value_counts()
                    
                    freq_feature = f"{col}_value_freq"
                    X_train[freq_feature] = X_train[col].map(freq_map)
                    X_test[freq_feature] = X_test[col].map(freq_map).fillna(0)
                    
                    # Relative frequency
                    rel_freq_feature = f"{col}_rel_freq"
                    total_count = len(X_train)
                    X_train[rel_freq_feature] = X_train[freq_feature] / total_count
                    X_test[rel_freq_feature] = X_test[freq_feature] / total_count
                    
                    self.frequency_features.extend([freq_feature, rel_freq_feature])
                    
                except Exception as e:
                    logger.warning(f"Frequency feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Frequency features completed: {len(self.frequency_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Frequency features failed: {e}")
            return X_train, X_test
    
    def _create_cross_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create cross features between categorical variables"""
        try:
            logger.info("Creating cross features")
            
            categorical_cols = [col for col in self.categorical_features if col in X_train.columns][:4]
            
            cross_count = 0
            max_crosses = 6
            
            for i in range(len(categorical_cols)):
                for j in range(i + 1, len(categorical_cols)):
                    if cross_count >= max_crosses:
                        break
                    
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    
                    try:
                        cross_feature = f"{col1}_cross_{col2}"
                        
                        # Create cross feature by combining values
                        X_train[cross_feature] = X_train[col1].astype(str) + "_" + X_train[col2].astype(str)
                        X_test[cross_feature] = X_test[col1].astype(str) + "_" + X_test[col2].astype(str)
                        
                        # Encode the cross feature
                        le = LabelEncoder()
                        combined_values = pd.concat([X_train[cross_feature], X_test[cross_feature]])
                        le.fit(combined_values)
                        
                        X_train[cross_feature] = le.transform(X_train[cross_feature])
                        X_test[cross_feature] = le.transform(X_test[cross_feature])
                        
                        self.cross_features.append(cross_feature)
                        self.label_encoders[cross_feature] = le
                        cross_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Cross feature creation failed for {col1}x{col2}: {e}")
                        continue
            
            logger.info(f"Cross features completed: {len(self.cross_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Cross features failed: {e}")
            return X_train, X_test
    
    def _create_temporal_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create time-based features if temporal columns exist"""
        try:
            logger.info("Creating temporal features")
            
            # Look for potential time-related columns
            time_cols = [col for col in X_train.columns if any(time_word in col.lower() for time_word in ['time', 'hour', 'day', 'date', 'timestamp'])]
            
            for col in time_cols[:3]:  # Limit to 3 columns
                try:
                    if X_train[col].dtype in ['int64', 'float64']:
                        # Assume it's a numeric time feature
                        
                        # Hour of day (if values are in 0-23 range)
                        if X_train[col].max() <= 23 and X_train[col].min() >= 0:
                            hour_feature = f"{col}_hour_bin"
                            X_train[hour_feature] = pd.cut(X_train[col], bins=4, labels=['morning', 'afternoon', 'evening', 'night'])
                            X_test[hour_feature] = pd.cut(X_test[col], bins=4, labels=['morning', 'afternoon', 'evening', 'night'])
                            
                            # Encode
                            le = LabelEncoder()
                            combined_hour = pd.concat([X_train[hour_feature].astype(str), X_test[hour_feature].astype(str)])
                            le.fit(combined_hour)
                            
                            X_train[hour_feature] = le.transform(X_train[hour_feature].astype(str))
                            X_test[hour_feature] = le.transform(X_test[hour_feature].astype(str))
                            
                            self.temporal_features.append(hour_feature)
                            self.label_encoders[hour_feature] = le
                    
                except Exception as e:
                    logger.warning(f"Temporal feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Temporal features completed: {len(self.temporal_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Temporal features failed: {e}")
            return X_train, X_test
    
    def _create_binning_features_fixed(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create binning features with proper handling"""
        try:
            logger.info("Creating binning features")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns][:4]
            
            for col in numeric_features:
                try:
                    # Create quantile-based bins
                    _, bins = pd.qcut(X_train[col], q=5, retbins=True, duplicates='drop')
                    
                    bin_feature = f"{col}_binned"
                    X_train[bin_feature] = pd.cut(X_train[col], bins=bins, include_lowest=True, duplicates='drop')
                    X_test[bin_feature] = pd.cut(X_test[col], bins=bins, include_lowest=True, duplicates='drop')
                    
                    # Encode bins
                    le = LabelEncoder()
                    combined_bins = pd.concat([X_train[bin_feature].astype(str), X_test[bin_feature].astype(str)])
                    le.fit(combined_bins)
                    
                    X_train[bin_feature] = le.transform(X_train[bin_feature].astype(str))
                    X_test[bin_feature] = le.transform(X_test[bin_feature].astype(str))
                    
                    self.binning_features.append(bin_feature)
                    self.label_encoders[bin_feature] = le
                    
                except Exception as e:
                    logger.warning(f"Binning feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Binning features completed: {len(self.binning_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Binning features failed: {e}")
            return X_train, X_test
    
    def _create_rank_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create ranking features for CTR prediction"""
        try:
            logger.info("Creating rank features")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns][:5]
            
            for col in numeric_features:
                try:
                    feature_name = f"{col}_rank"
                    
                    # Create rank feature based on training data distribution
                    X_train[feature_name] = X_train[col].rank(pct=True)
                    
                    # For test set, use training set distribution
                    X_test[feature_name] = X_test[col].rank(pct=True)
                    
                    self.rank_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Rank feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Rank features completed: {len(self.rank_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Rank features failed: {e}")
            return X_train, X_test
    
    def _create_ratio_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create ratio features with memory efficiency"""
        try:
            if len(self.numerical_features) < 2:
                return X_train, X_test
            
            logger.info("Creating ratio features")
            
            # Select numerical features for ratio calculation
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:6]
            
            ratio_count = 0
            max_ratios = 8
            
            for i in range(len(numeric_features_selected)):
                for j in range(i + 1, len(numeric_features_selected)):
                    if ratio_count >= max_ratios:
                        break
                    
                    col1, col2 = numeric_features_selected[i], numeric_features_selected[j]
                    
                    try:
                        # Create ratio feature
                        feature_name = f"{col1}_div_{col2}"
                        
                        denominator_train = X_train[col2].replace(0, 0.001)
                        denominator_test = X_test[col2].replace(0, 0.001)
                        
                        X_train[feature_name] = X_train[col1] / denominator_train
                        X_test[feature_name] = X_test[col1] / denominator_test
                        
                        # Handle infinite values
                        X_train[feature_name] = X_train[feature_name].replace([np.inf, -np.inf], 0)
                        X_test[feature_name] = X_test[feature_name].replace([np.inf, -np.inf], 0)
                        
                        self.ratio_features.append(feature_name)
                        ratio_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Ratio feature creation failed for {col1}/{col2}: {e}")
                        continue
            
            logger.info(f"Ratio features completed: {len(self.ratio_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Ratio features failed: {e}")
            return X_train, X_test
    
    def _create_polynomial_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create polynomial features with memory efficiency"""
        try:
            if len(self.numerical_features) == 0:
                return X_train, X_test
            
            logger.info("Creating polynomial features")
            
            # Select top numerical features for polynomial transformation
            numeric_features_selected = [col for col in self.numerical_features if col in X_train.columns][:3]
            
            for col in numeric_features_selected:
                try:
                    # Create squared feature
                    feature_name = f"{col}_squared"
                    X_train[feature_name] = X_train[col] ** 2
                    X_test[feature_name] = X_test[col] ** 2
                    
                    self.polynomial_features.append(feature_name)
                    
                except Exception as e:
                    logger.warning(f"Polynomial feature creation failed for {col}: {e}")
                    continue
            
            logger.info(f"Polynomial features completed: {len(self.polynomial_features)} features created")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Polynomial features failed: {e}")
            return X_train, X_test
    
    def _encode_categorical_features_safe(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safe categorical feature encoding"""
        try:
            # Get all categorical features including cross features
            all_categorical_features = self.categorical_features + self.cross_features
            
            for col in all_categorical_features:
                if col in X_train.columns and col in X_test.columns:
                    try:
                        # Safe label encoding with object conversion
                        train_str = X_train[col].astype('object').fillna('missing')
                        test_str = X_test[col].astype('object').fillna('missing')
                        
                        # Use existing encoder if available
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                        else:
                            le = LabelEncoder()
                            combined = pd.concat([train_str, test_str])
                            le.fit(combined)
                            self.label_encoders[col] = le
                        
                        # Transform safely
                        try:
                            X_train[col] = le.transform(train_str)
                            X_test[col] = le.transform(test_str)
                        except ValueError:
                            # Handle unseen categories
                            X_train[col] = train_str.map(dict(zip(le.classes_, le.transform(le.classes_)))).fillna(0)
                            X_test[col] = test_str.map(dict(zip(le.classes_, le.transform(le.classes_)))).fillna(0)
                        
                    except Exception as e:
                        logger.warning(f"Categorical encoding failed for {col}: {e}")
                        continue
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Categorical feature encoding failed: {e}")
            return X_train, X_test
    
    def _create_numeric_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create numeric feature transformations"""
        try:
            logger.info("Creating numeric transformations")
            
            numeric_features = [col for col in self.numerical_features if col in X_train.columns]
            
            for col in numeric_features:
                if col not in self.scalers:
                    try:
                        scaler = StandardScaler()
                        X_train[col] = scaler.fit_transform(X_train[[col]])
                        X_test[col] = scaler.transform(X_test[[col]])
                        
                        self.scalers[col] = scaler
                        
                    except Exception as e:
                        logger.warning(f"Numeric transformation failed for {col}: {e}")
                        continue
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Numeric feature creation failed: {e}")
            return X_train, X_test
    
    def _final_data_cleanup(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Final data cleanup and validation"""
        try:
            logger.info("Final data cleanup")
            
            # Remove infinite and NaN values
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Ensure consistent columns
            common_cols = sorted(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            # Convert all to numeric where possible
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='ignore')
                    X_test[col] = pd.to_numeric(X_test[col], errors='ignore')
                except Exception:
                    continue
            
            logger.info(f"Final cleanup completed - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Final data cleanup failed: {e}")
            return X_train, X_test
    
    def _finalize_processing(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Finalize feature engineering processing"""
        try:
            self.final_feature_columns = list(X_train.columns)
            
            # Update processing stats
            self.processing_stats['processing_time'] = time.time() - self.processing_stats['start_time']
            self.processing_stats['total_features_generated'] = len(self.final_feature_columns)
            self.processing_stats['feature_types_count'] = {
                'target_encoding': len(self.target_encoding_features),
                'interaction': len(self.interaction_features),
                'statistical': len(self.statistical_features),
                'frequency': len(self.frequency_features),
                'cross': len(self.cross_features),
                'temporal': len(self.temporal_features),
                'binning': len(self.binning_features),
                'rank': len(self.rank_features),
                'ratio': len(self.ratio_features),
                'polynomial': len(self.polynomial_features)
            }
            
            logger.info("Feature engineering processing completed")
            logger.info(f"Processing time: {self.processing_stats['processing_time']:.2f}s")
            logger.info(f"Total features generated: {self.processing_stats['total_features_generated']}")
            logger.info(f"Feature type counts: {self.processing_stats['feature_types_count']}")
            
        except Exception as e:
            logger.warning(f"Processing finalization failed: {e}")
    
    def _clean_basic_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean basic features"""
        try:
            # Remove columns with too many missing values
            missing_threshold = 0.5
            
            for col in list(X_train.columns):
                train_missing = X_train[col].isnull().sum() / len(X_train)
                test_missing = X_test[col].isnull().sum() / len(X_test) if col in X_test.columns else 1.0
                
                if train_missing > missing_threshold or test_missing > missing_threshold:
                    X_train = X_train.drop(columns=[col], errors='ignore')
                    X_test = X_test.drop(columns=[col], errors='ignore')
                    self.removed_columns.append(col)
            
            # Fill remaining missing values
            X_train = self._safe_fillna(X_train)
            X_test = self._safe_fillna(X_test)
            
            return X_train, X_test
            
        except Exception as e:
            logger.warning(f"Basic feature cleaning failed: {e}")
            return X_train, X_test
    
    def _create_minimal_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal features as fallback"""
        try:
            logger.warning("Creating minimal features as fallback")
            
            # Extract target
            if target_col in train_df.columns:
                X_train = train_df.drop(columns=[target_col]).copy()
            else:
                X_train = train_df.copy()
            
            X_test = test_df.copy()
            
            # Keep only common columns
            common_cols = sorted(set(X_train.columns) & set(X_test.columns))
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]
            
            # Simple preprocessing
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Convert to numeric
            for col in X_train.columns:
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                except Exception:
                    # Use label encoding for non-numeric
                    le = LabelEncoder()
                    combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
                    le.fit(combined)
                    X_train[col] = le.transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))
            
            logger.info(f"Minimal features created - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test
            
        except Exception as e:
            logger.error(f"Minimal feature creation failed: {e}")
            raise
    
    def _create_basic_features_only(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create only basic features when full processing fails"""
        try:
            logger.warning("Using basic features only due to processing failure")
            return self.create_quick_features(train_df, test_df, target_col)
            
        except Exception as e:
            logger.error(f"Basic feature creation failed: {e}")
            return self._create_minimal_features(train_df, test_df, target_col)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature engineering information"""
        return {
            'final_feature_count': len(self.final_feature_columns),
            'processing_stats': self.processing_stats,
            'feature_types': {
                'numerical': len(self.numerical_features),
                'categorical': len(self.categorical_features),
                'target_encoding': len(self.target_encoding_features),
                'interaction': len(self.interaction_features),
                'statistical': len(self.statistical_features),
                'frequency': len(self.frequency_features),
                'cross': len(self.cross_features),
                'temporal': len(self.temporal_features),
                'binning': len(self.binning_features),
                'rank': len(self.rank_features),
                'ratio': len(self.ratio_features),
                'polynomial': len(self.polynomial_features)
            },
            'removed_columns': self.removed_columns,
            'quick_mode': self.quick_mode,
            'memory_efficient_mode': self.memory_efficient_mode
        }