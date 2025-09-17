# inference.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pickle
import time
import json
from pathlib import Path
import threading
from collections import OrderedDict
import hashlib
import warnings
import gc
import os
import sys
warnings.filterwarnings('ignore')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class CTRInferenceEngine:
    """CTR 예측 전용 실시간 추론 엔진 - 규칙 준수 독립 실행 + Calibration 지원"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.feature_engineer = None
        self.is_loaded = False
        self.inference_stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_latency_ms': 0.0,
            'ctr_predictions': [],
            'calibration_stats': {
                'calibrated_predictions': 0,
                'raw_predictions': 0,
                'calibration_improvements': []
            }
        }
        self.lock = threading.Lock()
        
        # CTR 기본값 및 피처 설정
        self.ctr_baseline = 0.0201
        self.default_features = self._initialize_default_features()
        self.expected_feature_columns = None
        
        # 카테고리 매핑
        self.category_mappings = {
            'device': {'mobile': 1, 'desktop': 2, 'tablet': 3, 'unknown': 0},
            'page': {'home': 1, 'search': 2, 'product': 3, 'checkout': 4, 'category': 5, 'unknown': 0},
            'time_of_day': {'morning': 1, 'afternoon': 2, 'evening': 3, 'night': 4, 'unknown': 0},
            'day_of_week': {'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 
                          'friday': 5, 'saturday': 6, 'sunday': 7, 'unknown': 0}
        }
        
        # 메모리 모니터링
        self.memory_monitor = MemoryMonitor()
    
    def _initialize_default_features(self) -> Dict[str, float]:
        """기본 피처 값 초기화"""
        return {
            'feat_e_1': 65.0, 'feat_e_3': 5.0, 'feat_e_4': -0.05, 'feat_e_5': -0.02,
            'feat_e_6': -0.04, 'feat_e_7': 21.0, 'feat_e_8': -172.0, 'feat_e_9': -10.0,
            'feat_e_10': -270.0, 'feat_d_1': 0.39, 'feat_d_2': 1.93, 'feat_d_3': 1.76,
            'feat_d_4': 6.0, 'feat_d_5': -0.29, 'feat_d_6': -0.41,
            'feat_c_1': 0.0, 'feat_c_2': 0.0, 'feat_c_3': 0.0, 'feat_c_4': 0.0,
            'feat_c_5': 0.0, 'feat_c_6': 0.0, 'feat_c_7': 0.0, 'feat_c_8': 0.0,
            'feat_b_1': 0.0, 'feat_b_2': 0.0, 'feat_b_3': 0.0, 'feat_b_4': 0.0,
            'feat_b_5': 0.0, 'feat_b_6': 0.0
        }
    
    def load_models(self) -> bool:
        """저장된 모델들 로딩 - Calibration 지원"""
        logger.info(f"모델 로딩 시작: {self.model_dir}")
        
        if not self.model_dir.exists():
            logger.error(f"모델 디렉터리가 존재하지 않습니다: {self.model_dir}")
            return False
        
        try:
            # 고성능 모델 우선 로딩
            model_files = list(self.model_dir.glob("*_high_performance_model.pkl"))
            if not model_files:
                # 기본 모델 로딩
                model_files = list(self.model_dir.glob("*_model.pkl"))
            
            if not model_files:
                logger.error("로딩할 모델 파일이 없습니다")
                return False
            
            # 모델 로딩
            calibrated_count = 0
            for model_file in model_files:
                try:
                    model_name = model_file.stem.replace('_high_performance_model', '').replace('_model', '')
                    
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.models[model_name] = model
                    
                    # 캘리브레이션 상태 확인
                    calibration_status = "No"
                    if hasattr(model, 'is_calibrated') and model.is_calibrated:
                        calibrated_count += 1
                        calibration_method = "Unknown"
                        if hasattr(model, 'calibrator') and model.calibrator:
                            calibration_method = getattr(model.calibrator, 'best_method', 'Unknown')
                        calibration_status = f"Yes ({calibration_method})"
                    
                    logger.info(f"{model_name} 모델 로딩 완료 - Calibration: {calibration_status}")
                    
                    # 첫 모델에서 피처 순서 추출
                    if self.expected_feature_columns is None and hasattr(model, 'feature_names'):
                        self.expected_feature_columns = model.feature_names
                        logger.info(f"피처 순서 설정: {len(self.expected_feature_columns)}개")
                        
                except Exception as e:
                    logger.error(f"{model_name} 모델 로딩 실패: {e}")
                    continue
            
            # 피처 엔지니어 로딩
            feature_engineer_path = self.model_dir / "feature_engineer.pkl"
            if feature_engineer_path.exists():
                try:
                    with open(feature_engineer_path, 'rb') as f:
                        self.feature_engineer = pickle.load(f)
                    logger.info("피처 엔지니어 로딩 완료")
                    
                    if hasattr(self.feature_engineer, 'final_feature_columns'):
                        if self.expected_feature_columns is None:
                            self.expected_feature_columns = self.feature_engineer.final_feature_columns
                            logger.info(f"피처 엔지니어에서 피처 순서 설정: {len(self.expected_feature_columns)}개")
                            
                except Exception as e:
                    logger.warning(f"피처 엔지니어 로딩 실패: {e}")
            
            self.is_loaded = len(self.models) > 0
            logger.info(f"총 {len(self.models)}개 모델 로딩 완료 (캘리브레이션 적용: {calibrated_count}개)")
            
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            return False
    
    def _create_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """입력 데이터로부터 피처 생성"""
        try:
            features = self.default_features.copy()
            
            # 사용자 ID 해시 피처
            if 'user_id' in input_data:
                user_hash = int(hashlib.md5(str(input_data['user_id']).encode()).hexdigest(), 16) % 1000000
                features['user_hash'] = user_hash
                features['user_id_encoded'] = hash(str(input_data['user_id'])) % 10000
            
            # 광고 ID 해시 피처
            if 'ad_id' in input_data:
                ad_hash = int(hashlib.md5(str(input_data['ad_id']).encode()).hexdigest(), 16) % 1000000
                features['ad_hash'] = ad_hash
                features['ad_id_encoded'] = hash(str(input_data['ad_id'])) % 10000
            
            # 컨텍스트 피처 처리
            context = input_data.get('context', {})
            
            # 디바이스 타입
            device = context.get('device', 'unknown')
            features['device_encoded'] = self.category_mappings['device'].get(device.lower(), 0)
            
            # 페이지 타입
            page = context.get('page', 'unknown')
            features['page_encoded'] = self.category_mappings['page'].get(page.lower(), 0)
            
            # 시간대 정보
            hour = context.get('hour', 12)
            features['hour'] = hour
            features['is_morning'] = 1 if 6 <= hour < 12 else 0
            features['is_afternoon'] = 1 if 12 <= hour < 18 else 0
            features['is_evening'] = 1 if 18 <= hour < 22 else 0
            features['is_night'] = 1 if hour >= 22 or hour < 6 else 0
            
            # 요일 정보
            day_of_week = context.get('day_of_week', 'unknown')
            features['day_of_week_encoded'] = self.category_mappings['day_of_week'].get(day_of_week.lower(), 0)
            features['is_weekend'] = 1 if day_of_week.lower() in ['saturday', 'sunday'] else 0
            
            # 사용자-광고 상호작용 피처
            if 'user_id' in input_data and 'ad_id' in input_data:
                interaction_hash = hash(f"{input_data['user_id']}_{input_data['ad_id']}") % 100000
                features['user_ad_interaction'] = interaction_hash
            
            # 세션 정보
            features['session_position'] = context.get('position', 1)
            features['is_first_position'] = 1 if context.get('position', 1) == 1 else 0
            
            # 시간 피처
            current_time = time.time()
            features['time_index'] = (current_time % 86400) / 86400
            
            # 추가 컨텍스트 피처
            for key, value in context.items():
                if key not in ['device', 'page', 'hour', 'day_of_week', 'position']:
                    try:
                        features[f'context_{key}'] = float(value)
                    except:
                        features[f'context_{key}'] = hash(str(value)) % 1000
            
            # DataFrame 생성
            df = pd.DataFrame([features])
            
            # 데이터 타입 통일
            for col in df.columns:
                try:
                    df[col] = df[col].astype('float32')
                except:
                    df[col] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"피처 생성 실패: {e}")
            df = pd.DataFrame([self.default_features])
            return df.astype('float32')
    
    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """모델 학습 시와 동일한 피처 순서 보장"""
        try:
            if self.expected_feature_columns is None:
                return df
            
            # 누락된 피처를 0으로 채움
            for feature in self.expected_feature_columns:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # 학습 시와 동일한 순서로 재정렬
            df = df[self.expected_feature_columns]
            
            return df
            
        except Exception as e:
            logger.error(f"피처 일관성 보장 실패: {e}")
            return df
    
    def predict_single(self, input_data: Dict[str, Any], 
                      model_name: Optional[str] = None, 
                      use_calibration: bool = True) -> Dict[str, Any]:
        """단일 CTR 예측 - Calibration 지원"""
        start_time = time.time()
        
        try:
            with self.lock:
                self.inference_stats['total_requests'] += 1
            
            if not self.is_loaded:
                raise ValueError("모델이 로딩되지 않았습니다")
            
            # 피처 생성
            df = self._create_features(input_data)
            
            # 모델 선택
            if model_name is None:
                model_name = list(self.models.keys())[0]
            
            if model_name not in self.models:
                raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다")
            
            model = self.models[model_name]
            
            # 피처 일관성 보장
            df = self._ensure_feature_consistency(df)
            
            # CTR 예측 수행 (Calibration 고려)
            try:
                # 원본 예측
                if hasattr(model, 'predict_proba_raw'):
                    raw_prediction = model.predict_proba_raw(df)[0]
                else:
                    raw_prediction = model.predict_proba(df)[0]
                
                # 캘리브레이션 적용 여부 결정
                if (use_calibration and 
                    hasattr(model, 'is_calibrated') and 
                    model.is_calibrated and 
                    hasattr(model, 'predict_proba')):
                    # 캘리브레이션된 예측 사용
                    prediction_proba = model.predict_proba(df)[0]
                    calibration_applied = True
                    
                    with self.lock:
                        self.inference_stats['calibration_stats']['calibrated_predictions'] += 1
                        if raw_prediction != prediction_proba:
                            improvement = abs(prediction_proba - self.ctr_baseline) - abs(raw_prediction - self.ctr_baseline)
                            self.inference_stats['calibration_stats']['calibration_improvements'].append(improvement)
                else:
                    # 원본 예측 사용
                    prediction_proba = raw_prediction
                    calibration_applied = False
                    
                    with self.lock:
                        self.inference_stats['calibration_stats']['raw_predictions'] += 1
                
                prediction_proba = max(0.0001, min(0.9999, prediction_proba))
                
            except Exception as e:
                logger.warning(f"모델 예측 실패, 기본값 사용: {e}")
                prediction_proba = self.ctr_baseline
                raw_prediction = self.ctr_baseline
                calibration_applied = False
            
            prediction_binary = int(prediction_proba >= 0.5)
            confidence = abs(prediction_proba - 0.5) * 2
            
            # CTR 카테고리 분류
            if prediction_proba < 0.01:
                ctr_category = 'very_low'
            elif prediction_proba < 0.02:
                ctr_category = 'low'
            elif prediction_proba < 0.05:
                ctr_category = 'medium'
            elif prediction_proba < 0.1:
                ctr_category = 'high'
            else:
                ctr_category = 'very_high'
            
            latency = (time.time() - start_time) * 1000
            
            # 통계 업데이트
            with self.lock:
                self.inference_stats['successful_predictions'] += 1
                self.inference_stats['ctr_predictions'].append(prediction_proba)
                if len(self.inference_stats['ctr_predictions']) > 1000:
                    self.inference_stats['ctr_predictions'] = self.inference_stats['ctr_predictions'][-1000:]
                self._update_average_latency(latency)
            
            result = {
                'ctr_prediction': float(prediction_proba),
                'raw_prediction': float(raw_prediction) if 'raw_prediction' in locals() else float(prediction_proba),
                'predicted_click': prediction_binary,
                'confidence': float(confidence),
                'ctr_category': ctr_category,
                'model_used': model_name,
                'calibration_applied': calibration_applied,
                'latency_ms': latency,
                'recommendation': 'show' if prediction_proba > self.ctr_baseline else 'skip',
                'timestamp': time.time()
            }
            
            # 캘리브레이션 정보 추가
            if calibration_applied and hasattr(model, 'calibrator') and model.calibrator:
                calibration_summary = model.calibrator.get_calibration_summary()
                result['calibration_method'] = calibration_summary.get('best_method', 'unknown')
                if calibration_summary.get('calibration_scores'):
                    result['calibration_quality'] = max(calibration_summary['calibration_scores'].values())
            
            return result
            
        except Exception as e:
            with self.lock:
                self.inference_stats['failed_predictions'] += 1
            
            logger.error(f"CTR 예측 실패: {e}")
            
            return {
                'ctr_prediction': self.ctr_baseline,
                'raw_prediction': self.ctr_baseline,
                'predicted_click': 0,
                'confidence': 0.0,
                'ctr_category': 'low',
                'model_used': model_name or 'unknown',
                'calibration_applied': False,
                'latency_ms': (time.time() - start_time) * 1000,
                'recommendation': 'skip',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def predict_batch(self, input_data: List[Dict[str, Any]], 
                     model_name: Optional[str] = None,
                     use_calibration: bool = True) -> List[Dict[str, Any]]:
        """배치 CTR 예측 - Calibration 지원"""
        batch_size = 10000
        results = []
        
        logger.info(f"배치 CTR 예측 시작: {len(input_data)}개 샘플 (Calibration: {'On' if use_calibration else 'Off'})")
        
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            
            try:
                # 배치 피처 생성
                batch_features = []
                for item in batch:
                    df = self._create_features(item)
                    batch_features.append(df)
                
                # DataFrame 결합
                combined_df = pd.concat(batch_features, ignore_index=True)
                
                # 모델 선택
                if model_name is None:
                    model_name = list(self.models.keys())[0]
                
                model = self.models[model_name]
                
                # 피처 일관성 보장
                combined_df = self._ensure_feature_consistency(combined_df)
                
                # 배치 예측 (Calibration 고려)
                start_time = time.time()
                try:
                    # 원본 예측
                    if hasattr(model, 'predict_proba_raw'):
                        raw_predictions = model.predict_proba_raw(combined_df)
                    else:
                        raw_predictions = model.predict_proba(combined_df)
                    
                    # 캘리브레이션 적용 여부 결정
                    if (use_calibration and 
                        hasattr(model, 'is_calibrated') and 
                        model.is_calibrated and 
                        hasattr(model, 'predict_proba')):
                        # 캘리브레이션된 예측 사용
                        predictions_proba = model.predict_proba(combined_df)
                        calibration_applied = True
                    else:
                        # 원본 예측 사용
                        predictions_proba = raw_predictions
                        calibration_applied = False
                    
                    predictions_proba = np.clip(predictions_proba, 0.0001, 0.9999)
                    raw_predictions = np.clip(raw_predictions, 0.0001, 0.9999)
                    
                except Exception as e:
                    logger.warning(f"배치 모델 예측 실패: {e}")
                    predictions_proba = np.full(len(batch), self.ctr_baseline)
                    raw_predictions = np.full(len(batch), self.ctr_baseline)
                    calibration_applied = False
                
                predictions_binary = (predictions_proba >= 0.5).astype(int)
                latency = (time.time() - start_time) * 1000
                
                # 캘리브레이션 통계 업데이트
                with self.lock:
                    if calibration_applied:
                        self.inference_stats['calibration_stats']['calibrated_predictions'] += len(batch)
                        for raw_pred, cal_pred in zip(raw_predictions, predictions_proba):
                            if raw_pred != cal_pred:
                                improvement = abs(cal_pred - self.ctr_baseline) - abs(raw_pred - self.ctr_baseline)
                                self.inference_stats['calibration_stats']['calibration_improvements'].append(improvement)
                    else:
                        self.inference_stats['calibration_stats']['raw_predictions'] += len(batch)
                
                # 결과 구성
                for j, (item, prob, raw_prob, binary) in enumerate(zip(batch, predictions_proba, raw_predictions, predictions_binary)):
                    confidence = abs(prob - 0.5) * 2
                    
                    if prob < 0.01:
                        ctr_category = 'very_low'
                    elif prob < 0.02:
                        ctr_category = 'low'
                    elif prob < 0.05:
                        ctr_category = 'medium'
                    elif prob < 0.1:
                        ctr_category = 'high'
                    else:
                        ctr_category = 'very_high'
                    
                    result = {
                        'ctr_prediction': float(prob),
                        'raw_prediction': float(raw_prob),
                        'predicted_click': int(binary),
                        'confidence': float(confidence),
                        'ctr_category': ctr_category,
                        'model_used': model_name,
                        'calibration_applied': calibration_applied,
                        'batch_latency_ms': latency,
                        'batch_index': i + j,
                        'recommendation': 'show' if prob > self.ctr_baseline else 'skip',
                        'timestamp': time.time()
                    }
                    
                    # 캘리브레이션 정보 추가
                    if calibration_applied and hasattr(model, 'calibrator') and model.calibrator:
                        calibration_summary = model.calibrator.get_calibration_summary()
                        result['calibration_method'] = calibration_summary.get('best_method', 'unknown')
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} 예측 실패: {e}")
                
                # 오류 시 기본값으로 채움
                for j in range(len(batch)):
                    result = {
                        'ctr_prediction': self.ctr_baseline,
                        'raw_prediction': self.ctr_baseline,
                        'predicted_click': 0,
                        'confidence': 0.0,
                        'ctr_category': 'low',
                        'model_used': model_name or 'unknown',
                        'calibration_applied': False,
                        'batch_latency_ms': 0.0,
                        'batch_index': i + j,
                        'recommendation': 'skip',
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    results.append(result)
            
            # 메모리 정리
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        logger.info(f"배치 CTR 예측 완료: {len(results)}개 결과")
        return results
    
    def predict_test_data(self, test_df: pd.DataFrame, 
                         model_name: Optional[str] = None,
                         use_calibration: bool = True) -> np.ndarray:
        """테스트 데이터 전체 예측 (submission용) - Calibration 지원"""
        logger.info(f"테스트 데이터 예측 시작: {len(test_df):,}행 (Calibration: {'On' if use_calibration else 'Off'})")
        
        if not self.is_loaded:
            raise ValueError("모델이 로딩되지 않았습니다")
        
        # 모델 선택
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다")
        
        model = self.models[model_name]
        
        # 캘리브레이션 상태 확인
        calibration_available = (hasattr(model, 'is_calibrated') and 
                               model.is_calibrated and 
                               use_calibration)
        
        logger.info(f"사용 모델: {model_name}")
        logger.info(f"캘리브레이션 사용: {'Yes' if calibration_available else 'No'}")
        
        # 데이터 전처리
        processed_df = test_df.copy()
        
        # Object 타입 컬럼 제거
        object_columns = processed_df.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            processed_df = processed_df.drop(columns=object_columns)
        
        # 비수치형 컬럼 제거
        non_numeric_columns = []
        for col in processed_df.columns:
            if not np.issubdtype(processed_df[col].dtype, np.number):
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            processed_df = processed_df.drop(columns=non_numeric_columns)
        
        # 결측치 및 무한값 처리
        processed_df = processed_df.fillna(0)
        processed_df = processed_df.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # 데이터 타입 통일
        for col in processed_df.columns:
            if processed_df[col].dtype != 'float32':
                try:
                    processed_df[col] = processed_df[col].astype('float32')
                except:
                    processed_df[col] = 0.0
        
        # 피처 일관성 보장
        processed_df = self._ensure_feature_consistency(processed_df)
        
        logger.info(f"전처리 완료: {processed_df.shape}")
        
        # 메모리 상태 확인
        memory_status = self.memory_monitor.get_memory_status()
        if memory_status['level'] in ['warning', 'critical']:
            batch_size = 25000
        else:
            batch_size = 50000
        
        # 배치 단위 예측
        predictions = []
        raw_predictions = []
        
        for i in range(0, len(processed_df), batch_size):
            end_idx = min(i + batch_size, len(processed_df))
            batch_df = processed_df.iloc[i:end_idx]
            
            try:
                # 원본 예측
                if hasattr(model, 'predict_proba_raw'):
                    batch_raw_pred = model.predict_proba_raw(batch_df)
                else:
                    batch_raw_pred = model.predict_proba(batch_df)
                
                raw_predictions.extend(batch_raw_pred)
                
                # 캘리브레이션 적용 여부 결정
                if calibration_available:
                    batch_pred = model.predict_proba(batch_df)
                else:
                    batch_pred = batch_raw_pred
                
                batch_pred = np.clip(batch_pred, 0.0001, 0.9999)
                predictions.extend(batch_pred)
                
                logger.info(f"배치 {i//batch_size + 1} 완료 ({i:,}~{end_idx:,})")
                
            except Exception as e:
                logger.warning(f"배치 예측 실패: {e}")
                batch_pred = np.full(len(batch_df), self.ctr_baseline)
                batch_raw_pred = np.full(len(batch_df), self.ctr_baseline)
                predictions.extend(batch_pred)
                raw_predictions.extend(batch_raw_pred)
            
            # 주기적 메모리 정리
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        predictions = np.array(predictions)
        raw_predictions = np.array(raw_predictions)
        
        # CTR 보정
        current_ctr = predictions.mean()
        target_ctr = self.ctr_baseline
        
        if abs(current_ctr - target_ctr) > 0.002:
            logger.info(f"CTR 보정: {current_ctr:.4f} → {target_ctr:.4f}")
            correction_factor = target_ctr / current_ctr if current_ctr > 0 else 1.0
            predictions = predictions * correction_factor
            predictions = np.clip(predictions, 0.0001, 0.9999)
        
        # 캘리브레이션 효과 분석
        if calibration_available and len(raw_predictions) == len(predictions):
            try:
                raw_ctr = raw_predictions.mean()
                calibrated_ctr = predictions.mean()
                ctr_improvement = abs(calibrated_ctr - target_ctr) - abs(raw_ctr - target_ctr)
                
                logger.info(f"캘리브레이션 효과 분석:")
                logger.info(f"  - 원본 CTR: {raw_ctr:.4f}")
                logger.info(f"  - 캘리브레이션 CTR: {calibrated_ctr:.4f}")
                logger.info(f"  - 목표 CTR: {target_ctr:.4f}")
                logger.info(f"  - CTR 개선: {ctr_improvement:+.4f}")
                
                if hasattr(model, 'calibrator') and model.calibrator:
                    calibration_summary = model.calibrator.get_calibration_summary()
                    logger.info(f"  - 캘리브레이션 방법: {calibration_summary.get('best_method', 'unknown')}")
                
            except Exception as e:
                logger.warning(f"캘리브레이션 효과 분석 실패: {e}")
        
        logger.info(f"테스트 데이터 예측 완료: {len(predictions):,}개")
        return predictions
    
    def _update_average_latency(self, new_latency: float):
        """평균 지연시간 업데이트"""
        current_count = self.inference_stats['successful_predictions']
        current_avg = self.inference_stats['average_latency_ms']
        
        new_avg = ((current_avg * (current_count - 1)) + new_latency) / current_count
        self.inference_stats['average_latency_ms'] = new_avg
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계 반환 - Calibration 통계 포함"""
        stats = self.inference_stats.copy()
        
        total = stats['total_requests']
        if total > 0:
            stats['success_rate'] = stats['successful_predictions'] / total
            stats['failure_rate'] = stats['failed_predictions'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # CTR 통계
        if stats['ctr_predictions']:
            ctr_preds = np.array(stats['ctr_predictions'])
            stats['ctr_mean'] = float(ctr_preds.mean())
            stats['ctr_std'] = float(ctr_preds.std())
            stats['ctr_min'] = float(ctr_preds.min())
            stats['ctr_max'] = float(ctr_preds.max())
        else:
            stats['ctr_mean'] = 0.0
            stats['ctr_std'] = 0.0
            stats['ctr_min'] = 0.0
            stats['ctr_max'] = 0.0
        
        # 캘리브레이션 통계
        calibration_stats = stats['calibration_stats']
        total_predictions = calibration_stats['calibrated_predictions'] + calibration_stats['raw_predictions']
        
        if total_predictions > 0:
            calibration_stats['calibration_rate'] = calibration_stats['calibrated_predictions'] / total_predictions
        else:
            calibration_stats['calibration_rate'] = 0.0
        
        if calibration_stats['calibration_improvements']:
            improvements = np.array(calibration_stats['calibration_improvements'])
            calibration_stats['avg_improvement'] = float(improvements.mean())
            calibration_stats['improvement_std'] = float(improvements.std())
            calibration_stats['positive_improvements'] = int(np.sum(improvements > 0))
            calibration_stats['total_improvements'] = len(improvements)
        else:
            calibration_stats['avg_improvement'] = 0.0
            calibration_stats['improvement_std'] = 0.0
            calibration_stats['positive_improvements'] = 0
            calibration_stats['total_improvements'] = 0
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 상태 확인 - Calibration 정보 포함"""
        # 캘리브레이션 상태 확인
        calibrated_models = 0
        calibration_methods = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'is_calibrated') and model.is_calibrated:
                calibrated_models += 1
                if hasattr(model, 'calibrator') and model.calibrator:
                    method = getattr(model.calibrator, 'best_method', 'unknown')
                    calibration_methods[name] = method
        
        return {
            'is_loaded': self.is_loaded,
            'models_count': len(self.models),
            'available_models': list(self.models.keys()),
            'calibrated_models_count': calibrated_models,
            'calibration_methods': calibration_methods,
            'calibration_rate': calibrated_models / max(len(self.models), 1),
            'feature_engineer_loaded': self.feature_engineer is not None,
            'system_status': 'healthy' if self.is_loaded else 'not_ready',
            'ctr_baseline': self.ctr_baseline,
            'expected_features_count': len(self.expected_feature_columns) if self.expected_feature_columns else 0,
            'memory_status': self.memory_monitor.get_memory_status()
        }

class MemoryMonitor:
    """간단한 메모리 모니터링 클래스"""
    
    def __init__(self):
        self.monitoring_enabled = PSUTIL_AVAILABLE
    
    def get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태 반환"""
        if not self.monitoring_enabled:
            return {
                'usage_gb': 2.0,
                'available_gb': 40.0,
                'level': 'normal',
                'should_cleanup': False
            }
        
        try:
            process = psutil.Process()
            usage_gb = process.memory_info().rss / (1024**3)
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            if usage_gb > 45 or available_gb < 10:
                level = "critical"
            elif usage_gb > 35 or available_gb < 20:
                level = "warning"
            else:
                level = "normal"
            
            return {
                'usage_gb': usage_gb,
                'available_gb': available_gb,
                'level': level,
                'should_cleanup': level in ['warning', 'critical']
            }
        except:
            return {
                'usage_gb': 2.0,
                'available_gb': 40.0,
                'level': 'normal',
                'should_cleanup': False
            }

class CTRPredictionAPI:
    """CTR 예측 API - 규칙 준수 독립 실행 + Calibration 지원"""
    
    def __init__(self, model_dir: str = "models"):
        self.engine = CTRInferenceEngine(model_dir)
        self.api_stats = {
            'api_calls': 0,
            'api_errors': 0,
            'start_time': time.time(),
            'calibration_usage': {
                'requests_with_calibration': 0,
                'requests_without_calibration': 0
            }
        }
    
    def initialize(self) -> bool:
        """CTR 예측 API 초기화"""
        logger.info("CTR 예측 API 초기화 시작 (Calibration 지원)")
        
        success = self.engine.load_models()
        
        if success:
            # 캘리브레이션 상태 로깅
            health_check = self.engine.health_check()
            logger.info(f"CTR 예측 API 초기화 완료")
            logger.info(f"총 모델 수: {health_check['models_count']}")
            logger.info(f"캘리브레이션 적용 모델: {health_check['calibrated_models_count']}")
            logger.info(f"캘리브레이션 비율: {health_check['calibration_rate']:.2%}")
        else:
            logger.error("CTR 예측 API 초기화 실패")
        
        return success
    
    def predict_ctr(self, user_id: str, ad_id: str, context: Dict[str, Any] = None, 
                   model_name: Optional[str] = None, use_calibration: bool = True) -> Dict[str, Any]:
        """CTR 예측 메인 API - Calibration 지원"""
        self.api_stats['api_calls'] += 1
        
        # 캘리브레이션 사용량 통계
        if use_calibration:
            self.api_stats['calibration_usage']['requests_with_calibration'] += 1
        else:
            self.api_stats['calibration_usage']['requests_without_calibration'] += 1
        
        try:
            input_data = {
                'user_id': user_id,
                'ad_id': ad_id,
                'context': context or {}
            }
            
            result = self.engine.predict_single(input_data, model_name, use_calibration)
            
            api_response = {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': result['ctr_prediction'],
                'raw_prediction': result.get('raw_prediction', result['ctr_prediction']),
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'ctr_category': result['ctr_category'],
                'calibration_applied': result.get('calibration_applied', False),
                'calibration_method': result.get('calibration_method', 'none'),
                'calibration_quality': result.get('calibration_quality', 0.0),
                'expected_revenue': result['ctr_prediction'] * context.get('bid_amount', 1.0) if context else result['ctr_prediction'],
                'model_info': {
                    'model_name': result['model_used'],
                    'version': 'v2.0',
                    'calibration_supported': True
                },
                'performance': {
                    'latency_ms': result['latency_ms'],
                    'timestamp': result['timestamp']
                }
            }
            
            return api_response
            
        except Exception as e:
            self.api_stats['api_errors'] += 1
            logger.error(f"CTR 예측 API 오류: {e}")
            
            return {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': self.engine.ctr_baseline,
                'raw_prediction': self.engine.ctr_baseline,
                'recommendation': 'skip',
                'confidence': 0.0,
                'ctr_category': 'low',
                'calibration_applied': False,
                'calibration_method': 'none',
                'calibration_quality': 0.0,
                'expected_revenue': 0.0,
                'error': str(e),
                'model_info': {
                    'model_name': 'unknown',
                    'version': 'v2.0',
                    'calibration_supported': True
                },
                'performance': {
                    'latency_ms': 0.0,
                    'timestamp': time.time()
                }
            }
    
    def predict_ctr_batch(self, requests: List[Dict[str, Any]], 
                         model_name: Optional[str] = None,
                         use_calibration: bool = True) -> List[Dict[str, Any]]:
        """배치 CTR 예측 - Calibration 지원"""
        logger.info(f"배치 CTR 예측 요청: {len(requests)}개 (Calibration: {'On' if use_calibration else 'Off'})")
        
        input_data = []
        for req in requests:
            input_item = {
                'user_id': req.get('user_id', ''),
                'ad_id': req.get('ad_id', ''),
                'context': req.get('context', {})
            }
            input_data.append(input_item)
        
        # 캘리브레이션 사용량 통계 업데이트
        if use_calibration:
            self.api_stats['calibration_usage']['requests_with_calibration'] += len(requests)
        else:
            self.api_stats['calibration_usage']['requests_without_calibration'] += len(requests)
        
        predictions = self.engine.predict_batch(input_data, model_name, use_calibration)
        
        api_responses = []
        for i, (req, pred) in enumerate(zip(requests, predictions)):
            context = req.get('context', {})
            response = {
                'user_id': req.get('user_id', ''),
                'ad_id': req.get('ad_id', ''),
                'ctr_prediction': pred['ctr_prediction'],
                'raw_prediction': pred.get('raw_prediction', pred['ctr_prediction']),
                'recommendation': pred['recommendation'],
                'confidence': pred['confidence'],
                'ctr_category': pred['ctr_category'],
                'calibration_applied': pred.get('calibration_applied', False),
                'calibration_method': pred.get('calibration_method', 'none'),
                'expected_revenue': pred['ctr_prediction'] * context.get('bid_amount', 1.0),
                'batch_index': i,
                'model_info': {
                    'model_name': pred['model_used'],
                    'version': 'v2.0',
                    'calibration_supported': True
                },
                'performance': {
                    'batch_latency_ms': pred.get('batch_latency_ms', 0.0),
                    'timestamp': pred['timestamp']
                }
            }
            
            if 'error' in pred:
                response['error'] = pred['error']
            
            api_responses.append(response)
        
        return api_responses
    
    def predict_submission(self, test_df: pd.DataFrame, 
                          model_name: Optional[str] = None,
                          use_calibration: bool = True) -> pd.DataFrame:
        """제출용 예측 생성 - Calibration 지원"""
        logger.info(f"제출용 예측 생성 시작 (Calibration: {'On' if use_calibration else 'Off'})")
        
        predictions = self.engine.predict_test_data(test_df, model_name, use_calibration)
        
        submission = pd.DataFrame({
            'id': range(len(predictions)),
            'clicked': predictions
        })
        
        logger.info(f"제출용 예측 생성 완료: {len(submission):,}행")
        
        # 캘리브레이션 효과 요약 (로그)
        if use_calibration:
            avg_prediction = predictions.mean()
            logger.info(f"평균 예측 CTR: {avg_prediction:.4f}")
            logger.info(f"목표 CTR과의 차이: {abs(avg_prediction - self.engine.ctr_baseline):.4f}")
        
        return submission
    
    def get_api_status(self) -> Dict[str, Any]:
        """API 상태 정보 - Calibration 통계 포함"""
        uptime = time.time() - self.api_stats['start_time']
        
        # 캘리브레이션 사용률 계산
        total_calibration_requests = (self.api_stats['calibration_usage']['requests_with_calibration'] + 
                                    self.api_stats['calibration_usage']['requests_without_calibration'])
        
        calibration_usage_rate = 0.0
        if total_calibration_requests > 0:
            calibration_usage_rate = (self.api_stats['calibration_usage']['requests_with_calibration'] / 
                                    total_calibration_requests)
        
        status = {
            'api_uptime_seconds': uptime,
            'total_api_calls': self.api_stats['api_calls'],
            'api_error_count': self.api_stats['api_errors'],
            'api_success_rate': 1.0 - (self.api_stats['api_errors'] / max(1, self.api_stats['api_calls'])),
            'calibration_usage': {
                'total_requests': total_calibration_requests,
                'with_calibration': self.api_stats['calibration_usage']['requests_with_calibration'],
                'without_calibration': self.api_stats['calibration_usage']['requests_without_calibration'],
                'calibration_usage_rate': calibration_usage_rate
            },
            'engine_status': self.engine.health_check(),
            'inference_stats': self.engine.get_inference_stats()
        }
        
        return status

def create_ctr_prediction_service(model_dir: str = "models") -> CTRPredictionAPI:
    """CTR 예측 서비스 생성 - Calibration 지원"""
    logger.info("CTR 예측 서비스 생성 (Calibration 지원)")
    
    api = CTRPredictionAPI(model_dir)
    
    if not api.initialize():
        logger.error("CTR 예측 서비스 초기화 실패")
        return None
    
    logger.info("CTR 예측 서비스 생성 완료")
    return api

if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    service = create_ctr_prediction_service()
    
    if service:
        status = service.get_api_status()
        print("API 상태:", json.dumps(status, indent=2, default=str))
        
        # 단일 예측 테스트 (Calibration 사용)
        test_prediction_calibrated = service.predict_ctr(
            user_id="test_user_123",
            ad_id="test_ad_456",
            context={
                'device': 'mobile',
                'page': 'home',
                'hour': 14,
                'day_of_week': 'tuesday',
                'position': 1,
                'bid_amount': 0.5
            },
            use_calibration=True
        )
        
        print("CTR 예측 결과 (Calibration On):", json.dumps(test_prediction_calibrated, indent=2, default=str))
        
        # 단일 예측 테스트 (Calibration 미사용)
        test_prediction_raw = service.predict_ctr(
            user_id="test_user_123",
            ad_id="test_ad_456",
            context={
                'device': 'mobile',
                'page': 'home',
                'hour': 14,
                'day_of_week': 'tuesday',
                'position': 1,
                'bid_amount': 0.5
            },
            use_calibration=False
        )
        
        print("CTR 예측 결과 (Calibration Off):", json.dumps(test_prediction_raw, indent=2, default=str))
        
        # 캘리브레이션 효과 비교
        calibrated_ctr = test_prediction_calibrated['ctr_prediction']
        raw_ctr = test_prediction_raw['ctr_prediction']
        
        if calibrated_ctr != raw_ctr:
            improvement = abs(calibrated_ctr - 0.0201) - abs(raw_ctr - 0.0201)
            print(f"\n캘리브레이션 효과:")
            print(f"원본 CTR: {raw_ctr:.4f}")
            print(f"캘리브레이션 CTR: {calibrated_ctr:.4f}")
            print(f"목표 대비 개선: {improvement:+.4f}")
        else:
            print(f"\n캘리브레이션이 적용되지 않았거나 효과가 없습니다.")