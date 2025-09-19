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
    """CTR 예측 전용 실시간 추론 엔진 - 최종 완성 버전"""
    
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
            },
            'ensemble_stats': {
                'ensemble_predictions': 0,
                'single_model_predictions': 0,
                'ensemble_improvements': []
            }
        }
        self.lock = threading.Lock()
        
        self.ctr_baseline = 0.0201
        self.default_features = self._initialize_default_features()
        self.expected_feature_columns = None
        
        self.category_mappings = {
            'device': {'mobile': 1, 'desktop': 2, 'tablet': 3, 'unknown': 0},
            'page': {'home': 1, 'search': 2, 'product': 3, 'checkout': 4, 'category': 5, 'unknown': 0},
            'time_of_day': {'morning': 1, 'afternoon': 2, 'evening': 3, 'night': 4, 'unknown': 0},
            'day_of_week': {'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 
                          'friday': 5, 'saturday': 6, 'sunday': 7, 'unknown': 0}
        }
        
        self.memory_monitor = MemoryMonitor()
        self.ensemble_manager = None
        self.postprocessor = CTRPostProcessor()
    
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
        """저장된 모델들 로딩 - 앙상블 지원"""
        logger.info(f"모델 로딩 시작: {self.model_dir}")
        
        if not self.model_dir.exists():
            logger.error(f"모델 디렉터리가 존재하지 않습니다: {self.model_dir}")
            return False
        
        try:
            model_files = list(self.model_dir.glob("*_model.pkl"))
            
            if not model_files:
                logger.error("로딩할 모델 파일이 없습니다")
                return False
            
            calibrated_count = 0
            for model_file in model_files:
                try:
                    model_name = model_file.stem.replace('_model', '')
                    
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.models[model_name] = model
                    
                    calibration_status = "No"
                    if hasattr(model, 'is_calibrated') and model.is_calibrated:
                        calibrated_count += 1
                        calibration_method = "Unknown"
                        if hasattr(model, 'calibrator') and model.calibrator:
                            calibration_method = getattr(model.calibrator, 'best_method', 'Unknown')
                        calibration_status = f"Yes ({calibration_method})"
                    
                    logger.info(f"{model_name} 모델 로딩 완료 - Calibration: {calibration_status}")
                    
                    if self.expected_feature_columns is None and hasattr(model, 'feature_names'):
                        self.expected_feature_columns = model.feature_names
                        logger.info(f"피처 순서 설정: {len(self.expected_feature_columns)}개")
                        
                except Exception as e:
                    logger.error(f"{model_name} 모델 로딩 실패: {e}")
                    continue
            
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
            
            ensemble_manager_path = self.model_dir / "ensemble_manager.pkl"
            if ensemble_manager_path.exists():
                try:
                    with open(ensemble_manager_path, 'rb') as f:
                        self.ensemble_manager = pickle.load(f)
                    logger.info("앙상블 매니저 로딩 완료")
                except Exception as e:
                    logger.warning(f"앙상블 매니저 로딩 실패: {e}")
            
            self.is_loaded = len(self.models) > 0
            logger.info(f"총 {len(self.models)}개 모델 로딩 완료 (캘리브레이션 적용: {calibrated_count}개)")
            logger.info(f"앙상블 매니저: {'로딩됨' if self.ensemble_manager else '없음'}")
            
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            return False
    
    def _create_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """입력 데이터로부터 피처 생성"""
        try:
            features = self.default_features.copy()
            
            if 'user_id' in input_data:
                user_hash = int(hashlib.md5(str(input_data['user_id']).encode()).hexdigest(), 16) % 1000000
                features['user_hash'] = user_hash
                features['user_id_encoded'] = hash(str(input_data['user_id'])) % 10000
            
            if 'ad_id' in input_data:
                ad_hash = int(hashlib.md5(str(input_data['ad_id']).encode()).hexdigest(), 16) % 1000000
                features['ad_hash'] = ad_hash
                features['ad_id_encoded'] = hash(str(input_data['ad_id'])) % 10000
            
            context = input_data.get('context', {})
            
            device = context.get('device', 'unknown')
            features['device_encoded'] = self.category_mappings['device'].get(device.lower(), 0)
            
            page = context.get('page', 'unknown')
            features['page_encoded'] = self.category_mappings['page'].get(page.lower(), 0)
            
            hour = context.get('hour', 12)
            features['hour'] = hour
            features['is_morning'] = 1 if 6 <= hour < 12 else 0
            features['is_afternoon'] = 1 if 12 <= hour < 18 else 0
            features['is_evening'] = 1 if 18 <= hour < 22 else 0
            features['is_night'] = 1 if hour >= 22 or hour < 6 else 0
            
            day_of_week = context.get('day_of_week', 'unknown')
            features['day_of_week_encoded'] = self.category_mappings['day_of_week'].get(day_of_week.lower(), 0)
            features['is_weekend'] = 1 if day_of_week.lower() in ['saturday', 'sunday'] else 0
            
            if 'user_id' in input_data and 'ad_id' in input_data:
                interaction_hash = hash(f"{input_data['user_id']}_{input_data['ad_id']}") % 100000
                features['user_ad_interaction'] = interaction_hash
            
            features['session_position'] = context.get('position', 1)
            features['is_first_position'] = 1 if context.get('position', 1) == 1 else 0
            
            current_time = time.time()
            features['time_index'] = (current_time % 86400) / 86400
            
            for key, value in context.items():
                if key not in ['device', 'page', 'hour', 'day_of_week', 'position']:
                    try:
                        features[f'context_{key}'] = float(value)
                    except:
                        features[f'context_{key}'] = hash(str(value)) % 1000
            
            df = pd.DataFrame([features])
            
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
            
            for feature in self.expected_feature_columns:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            df = df[self.expected_feature_columns]
            
            return df
            
        except Exception as e:
            logger.error(f"피처 일관성 보장 실패: {e}")
            return df
    
    def predict_single(self, input_data: Dict[str, Any], 
                      model_name: Optional[str] = None, 
                      use_calibration: bool = True,
                      use_ensemble: bool = True,
                      use_postprocessing: bool = True) -> Dict[str, Any]:
        """단일 CTR 예측 - 최종 완성 버전"""
        start_time = time.time()
        
        try:
            with self.lock:
                self.inference_stats['total_requests'] += 1
            
            if not self.is_loaded:
                raise ValueError("모델이 로딩되지 않았습니다")
            
            df = self._create_features(input_data)
            df = self._ensure_feature_consistency(df)
            
            prediction_method = "single_model"
            ensemble_used = False
            calibration_applied = False
            postprocessing_applied = False
            
            try:
                if use_ensemble and self.ensemble_manager is not None:
                    try:
                        prediction_proba = self.ensemble_manager.predict_with_best_ensemble(df)[0]
                        prediction_method = "ensemble"
                        ensemble_used = True
                        
                        with self.lock:
                            self.inference_stats['ensemble_stats']['ensemble_predictions'] += 1
                        
                        logger.debug("앙상블 예측 사용")
                    except Exception as e:
                        logger.warning(f"앙상블 예측 실패, 단일 모델 사용: {e}")
                        use_ensemble = False
                
                if not use_ensemble:
                    if model_name is None:
                        model_name = list(self.models.keys())[0]
                    
                    if model_name not in self.models:
                        raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다")
                    
                    model = self.models[model_name]
                    
                    if hasattr(model, 'predict_proba_raw'):
                        raw_prediction = model.predict_proba_raw(df)[0]
                    else:
                        raw_prediction = model.predict_proba(df)[0]
                    
                    if (use_calibration and 
                        hasattr(model, 'is_calibrated') and 
                        model.is_calibrated and 
                        hasattr(model, 'predict_proba')):
                        prediction_proba = model.predict_proba(df)[0]
                        calibration_applied = True
                        
                        with self.lock:
                            self.inference_stats['calibration_stats']['calibrated_predictions'] += 1
                            if raw_prediction != prediction_proba:
                                improvement = abs(prediction_proba - self.ctr_baseline) - abs(raw_prediction - self.ctr_baseline)
                                self.inference_stats['calibration_stats']['calibration_improvements'].append(improvement)
                    else:
                        prediction_proba = raw_prediction
                        
                        with self.lock:
                            self.inference_stats['calibration_stats']['raw_predictions'] += 1
                    
                    with self.lock:
                        self.inference_stats['ensemble_stats']['single_model_predictions'] += 1
                
                if use_postprocessing:
                    original_prediction = prediction_proba
                    prediction_proba = self.postprocessor.apply_postprocessing(
                        np.array([prediction_proba]), 
                        target_ctr=self.ctr_baseline
                    )[0]
                    
                    if abs(prediction_proba - original_prediction) > 1e-6:
                        postprocessing_applied = True
                
                prediction_proba = max(0.0001, min(0.9999, prediction_proba))
                
            except Exception as e:
                logger.warning(f"모델 예측 실패, 기본값 사용: {e}")
                prediction_proba = self.ctr_baseline
                raw_prediction = self.ctr_baseline
                calibration_applied = False
                ensemble_used = False
                postprocessing_applied = False
            
            prediction_binary = int(prediction_proba >= 0.5)
            confidence = abs(prediction_proba - 0.5) * 2
            
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
                'model_used': prediction_method,
                'ensemble_used': ensemble_used,
                'calibration_applied': calibration_applied,
                'postprocessing_applied': postprocessing_applied,
                'latency_ms': latency,
                'recommendation': 'show' if prediction_proba > self.ctr_baseline else 'skip',
                'timestamp': time.time()
            }
            
            if calibration_applied and hasattr(self.models.get(model_name or list(self.models.keys())[0]), 'calibrator'):
                model_obj = self.models.get(model_name or list(self.models.keys())[0])
                if model_obj and model_obj.calibrator:
                    calibration_summary = model_obj.calibrator.get_calibration_summary()
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
                'ensemble_used': False,
                'calibration_applied': False,
                'postprocessing_applied': False,
                'latency_ms': (time.time() - start_time) * 1000,
                'recommendation': 'skip',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def predict_batch(self, input_data: List[Dict[str, Any]], 
                     model_name: Optional[str] = None,
                     use_calibration: bool = True,
                     use_ensemble: bool = True,
                     use_postprocessing: bool = True) -> List[Dict[str, Any]]:
        """배치 CTR 예측 - 최종 완성 버전"""
        batch_size = 15000
        results = []
        
        logger.info(f"배치 CTR 예측 시작: {len(input_data)}개 샘플")
        logger.info(f"설정 - Calibration: {'On' if use_calibration else 'Off'}, "
                   f"Ensemble: {'On' if use_ensemble else 'Off'}, "
                   f"Postprocessing: {'On' if use_postprocessing else 'Off'}")
        
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            
            try:
                batch_features = []
                for item in batch:
                    df = self._create_features(item)
                    batch_features.append(df)
                
                combined_df = pd.concat(batch_features, ignore_index=True)
                combined_df = self._ensure_feature_consistency(combined_df)
                
                start_time = time.time()
                ensemble_used = False
                calibration_applied = False
                postprocessing_applied = False
                
                try:
                    if use_ensemble and self.ensemble_manager is not None:
                        try:
                            predictions_proba = self.ensemble_manager.predict_with_best_ensemble(combined_df)
                            prediction_method = "ensemble"
                            ensemble_used = True
                            
                            with self.lock:
                                self.inference_stats['ensemble_stats']['ensemble_predictions'] += len(batch)
                        except Exception as e:
                            logger.warning(f"배치 앙상블 예측 실패: {e}")
                            use_ensemble = False
                    
                    if not use_ensemble:
                        if model_name is None:
                            model_name = list(self.models.keys())[0]
                        
                        model = self.models[model_name]
                        prediction_method = f"single_{model_name}"
                        
                        if hasattr(model, 'predict_proba_raw'):
                            raw_predictions = model.predict_proba_raw(combined_df)
                        else:
                            raw_predictions = model.predict_proba(combined_df)
                        
                        if (use_calibration and 
                            hasattr(model, 'is_calibrated') and 
                            model.is_calibrated and 
                            hasattr(model, 'predict_proba')):
                            predictions_proba = model.predict_proba(combined_df)
                            calibration_applied = True
                        else:
                            predictions_proba = raw_predictions
                        
                        with self.lock:
                            self.inference_stats['ensemble_stats']['single_model_predictions'] += len(batch)
                    
                    if use_postprocessing:
                        original_predictions = predictions_proba.copy()
                        predictions_proba = self.postprocessor.apply_postprocessing(
                            predictions_proba, 
                            target_ctr=self.ctr_baseline
                        )
                        
                        if not np.allclose(predictions_proba, original_predictions, atol=1e-6):
                            postprocessing_applied = True
                    
                    predictions_proba = np.clip(predictions_proba, 0.0001, 0.9999)
                    
                    if 'raw_predictions' not in locals():
                        raw_predictions = predictions_proba
                    
                    raw_predictions = np.clip(raw_predictions, 0.0001, 0.9999)
                    
                except Exception as e:
                    logger.warning(f"배치 모델 예측 실패: {e}")
                    predictions_proba = np.full(len(batch), self.ctr_baseline)
                    raw_predictions = np.full(len(batch), self.ctr_baseline)
                    prediction_method = "default"
                    calibration_applied = False
                    ensemble_used = False
                    postprocessing_applied = False
                
                predictions_binary = (predictions_proba >= 0.5).astype(int)
                latency = (time.time() - start_time) * 1000
                
                with self.lock:
                    if calibration_applied:
                        self.inference_stats['calibration_stats']['calibrated_predictions'] += len(batch)
                        for raw_pred, cal_pred in zip(raw_predictions, predictions_proba):
                            if raw_pred != cal_pred:
                                improvement = abs(cal_pred - self.ctr_baseline) - abs(raw_pred - self.ctr_baseline)
                                self.inference_stats['calibration_stats']['calibration_improvements'].append(improvement)
                    else:
                        self.inference_stats['calibration_stats']['raw_predictions'] += len(batch)
                
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
                        'model_used': prediction_method,
                        'ensemble_used': ensemble_used,
                        'calibration_applied': calibration_applied,
                        'postprocessing_applied': postprocessing_applied,
                        'batch_latency_ms': latency,
                        'batch_index': i + j,
                        'recommendation': 'show' if prob > self.ctr_baseline else 'skip',
                        'timestamp': time.time()
                    }
                    
                    if calibration_applied and model_name and model_name in self.models:
                        model_obj = self.models[model_name]
                        if hasattr(model_obj, 'calibrator') and model_obj.calibrator:
                            calibration_summary = model_obj.calibrator.get_calibration_summary()
                            result['calibration_method'] = calibration_summary.get('best_method', 'unknown')
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} 예측 실패: {e}")
                
                for j in range(len(batch)):
                    result = {
                        'ctr_prediction': self.ctr_baseline,
                        'raw_prediction': self.ctr_baseline,
                        'predicted_click': 0,
                        'confidence': 0.0,
                        'ctr_category': 'low',
                        'model_used': 'error',
                        'ensemble_used': False,
                        'calibration_applied': False,
                        'postprocessing_applied': False,
                        'batch_latency_ms': 0.0,
                        'batch_index': i + j,
                        'recommendation': 'skip',
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    results.append(result)
            
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        logger.info(f"배치 CTR 예측 완료: {len(results)}개 결과")
        return results
    
    def predict_test_data(self, test_df: pd.DataFrame, 
                         model_name: Optional[str] = None,
                         use_calibration: bool = True,
                         use_ensemble: bool = True,
                         use_postprocessing: bool = True) -> np.ndarray:
        """테스트 데이터 전체 예측 (submission용) - 최종 완성 버전"""
        logger.info(f"테스트 데이터 예측 시작: {len(test_df):,}행")
        logger.info(f"설정 - Calibration: {'On' if use_calibration else 'Off'}, "
                   f"Ensemble: {'On' if use_ensemble else 'Off'}, "
                   f"Postprocessing: {'On' if use_postprocessing else 'Off'}")
        
        if not self.is_loaded:
            raise ValueError("모델이 로딩되지 않았습니다")
        
        ensemble_available = use_ensemble and self.ensemble_manager is not None
        
        if model_name is None and not ensemble_available:
            model_name = list(self.models.keys())[0]
        
        if not ensemble_available and model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다")
        
        logger.info(f"사용 방법: {'Ensemble' if ensemble_available else f'Single Model ({model_name})'}")
        
        processed_df = test_df.copy()
        
        object_columns = processed_df.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            processed_df = processed_df.drop(columns=object_columns)
        
        non_numeric_columns = []
        for col in processed_df.columns:
            if not np.issubdtype(processed_df[col].dtype, np.number):
                non_numeric_columns.append(col)
        
        if non_numeric_columns:
            processed_df = processed_df.drop(columns=non_numeric_columns)
        
        processed_df = processed_df.fillna(0)
        processed_df = processed_df.replace([np.inf, -np.inf], [1e6, -1e6])
        
        for col in processed_df.columns:
            if processed_df[col].dtype != 'float32':
                try:
                    processed_df[col] = processed_df[col].astype('float32')
                except:
                    processed_df[col] = 0.0
        
        processed_df = self._ensure_feature_consistency(processed_df)
        
        logger.info(f"전처리 완료: {processed_df.shape}")
        
        memory_status = self.memory_monitor.get_memory_status()
        if memory_status['level'] in ['warning', 'critical']:
            batch_size = 30000
        else:
            batch_size = 80000
        
        predictions = []
        raw_predictions = []
        
        for i in range(0, len(processed_df), batch_size):
            end_idx = min(i + batch_size, len(processed_df))
            batch_df = processed_df.iloc[i:end_idx]
            
            try:
                ensemble_used = False
                calibration_applied = False
                
                if ensemble_available:
                    try:
                        batch_pred = self.ensemble_manager.predict_with_best_ensemble(batch_df)
                        batch_raw_pred = batch_pred.copy()
                        ensemble_used = True
                        logger.debug(f"배치 {i//batch_size + 1}: 앙상블 예측 사용")
                    except Exception as e:
                        logger.warning(f"배치 앙상블 예측 실패: {e}")
                        ensemble_available = False
                
                if not ensemble_available:
                    model = self.models[model_name]
                    
                    if hasattr(model, 'predict_proba_raw'):
                        batch_raw_pred = model.predict_proba_raw(batch_df)
                    else:
                        batch_raw_pred = model.predict_proba(batch_df)
                    
                    if (use_calibration and 
                        hasattr(model, 'is_calibrated') and 
                        model.is_calibrated):
                        batch_pred = model.predict_proba(batch_df)
                        calibration_applied = True
                    else:
                        batch_pred = batch_raw_pred
                
                if use_postprocessing:
                    original_batch_pred = batch_pred.copy()
                    batch_pred = self.postprocessor.apply_postprocessing(
                        batch_pred,
                        target_ctr=self.ctr_baseline
                    )
                    
                    if not np.allclose(batch_pred, original_batch_pred, atol=1e-6):
                        logger.debug(f"배치 {i//batch_size + 1}: 후처리 적용됨")
                
                batch_pred = np.clip(batch_pred, 0.0001, 0.9999)
                batch_raw_pred = np.clip(batch_raw_pred, 0.0001, 0.9999)
                
                predictions.extend(batch_pred)
                raw_predictions.extend(batch_raw_pred)
                
                logger.info(f"배치 {i//batch_size + 1} 완료 ({i:,}~{end_idx:,}) - "
                          f"Ensemble: {'Yes' if ensemble_used else 'No'}, "
                          f"Calibration: {'Yes' if calibration_applied else 'No'}")
                
            except Exception as e:
                logger.warning(f"배치 예측 실패: {e}")
                batch_pred = np.full(len(batch_df), self.ctr_baseline)
                batch_raw_pred = np.full(len(batch_df), self.ctr_baseline)
                predictions.extend(batch_pred)
                raw_predictions.extend(batch_raw_pred)
            
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        predictions = np.array(predictions)
        raw_predictions = np.array(raw_predictions)
        
        current_ctr = predictions.mean()
        target_ctr = self.ctr_baseline
        
        if abs(current_ctr - target_ctr) > 0.002:
            logger.info(f"CTR 보정: {current_ctr:.4f} → {target_ctr:.4f}")
            correction_factor = target_ctr / current_ctr if current_ctr > 0 else 1.0
            predictions = predictions * correction_factor
            predictions = np.clip(predictions, 0.0001, 0.9999)
        
        if ensemble_available and len(raw_predictions) == len(predictions):
            try:
                raw_ctr = raw_predictions.mean()
                final_ctr = predictions.mean()
                
                logger.info(f"최종 예측 분석:")
                logger.info(f"  - 원본 CTR: {raw_ctr:.4f}")
                logger.info(f"  - 최종 CTR: {final_ctr:.4f}")
                logger.info(f"  - 목표 CTR: {target_ctr:.4f}")
                logger.info(f"  - 사용 방법: {'Ensemble' if ensemble_used else 'Single Model'}")
                
                if use_calibration and calibration_applied:
                    logger.info(f"  - 캘리브레이션: 적용됨")
                    
                if use_postprocessing:
                    logger.info(f"  - 후처리: 적용됨")
                
            except Exception as e:
                logger.warning(f"예측 분석 실패: {e}")
        
        logger.info(f"테스트 데이터 예측 완료: {len(predictions):,}개")
        return predictions
    
    def _update_average_latency(self, new_latency: float):
        """평균 지연시간 업데이트"""
        current_count = self.inference_stats['successful_predictions']
        current_avg = self.inference_stats['average_latency_ms']
        
        new_avg = ((current_avg * (current_count - 1)) + new_latency) / current_count
        self.inference_stats['average_latency_ms'] = new_avg
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계 반환 - 최종 완성 버전"""
        stats = self.inference_stats.copy()
        
        total = stats['total_requests']
        if total > 0:
            stats['success_rate'] = stats['successful_predictions'] / total
            stats['failure_rate'] = stats['failed_predictions'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
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
        
        ensemble_stats = stats['ensemble_stats']
        total_ensemble_predictions = ensemble_stats['ensemble_predictions'] + ensemble_stats['single_model_predictions']
        
        if total_ensemble_predictions > 0:
            ensemble_stats['ensemble_usage_rate'] = ensemble_stats['ensemble_predictions'] / total_ensemble_predictions
        else:
            ensemble_stats['ensemble_usage_rate'] = 0.0
        
        if ensemble_stats['ensemble_improvements']:
            ensemble_improvements = np.array(ensemble_stats['ensemble_improvements'])
            ensemble_stats['avg_ensemble_improvement'] = float(ensemble_improvements.mean())
            ensemble_stats['positive_ensemble_improvements'] = int(np.sum(ensemble_improvements > 0))
        else:
            ensemble_stats['avg_ensemble_improvement'] = 0.0
            ensemble_stats['positive_ensemble_improvements'] = 0
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 상태 확인 - 최종 완성 버전"""
        calibrated_models = 0
        calibration_methods = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'is_calibrated') and model.is_calibrated:
                calibrated_models += 1
                if hasattr(model, 'calibrator') and model.calibrator:
                    method = getattr(model.calibrator, 'best_method', 'unknown')
                    calibration_methods[name] = method
        
        ensemble_status = {
            'available': self.ensemble_manager is not None,
            'type': 'unknown',
            'calibrated': False
        }
        
        if self.ensemble_manager is not None:
            if hasattr(self.ensemble_manager, 'best_ensemble') and self.ensemble_manager.best_ensemble:
                ensemble_status['type'] = self.ensemble_manager.best_ensemble.name
                ensemble_status['calibrated'] = getattr(self.ensemble_manager.best_ensemble, 'is_calibrated', False)
        
        return {
            'is_loaded': self.is_loaded,
            'models_count': len(self.models),
            'available_models': list(self.models.keys()),
            'calibrated_models_count': calibrated_models,
            'calibration_methods': calibration_methods,
            'calibration_rate': calibrated_models / max(len(self.models), 1),
            'ensemble_status': ensemble_status,
            'feature_engineer_loaded': self.feature_engineer is not None,
            'postprocessor_available': self.postprocessor is not None,
            'system_status': 'healthy' if self.is_loaded else 'not_ready',
            'ctr_baseline': self.ctr_baseline,
            'expected_features_count': len(self.expected_feature_columns) if self.expected_feature_columns else 0,
            'memory_status': self.memory_monitor.get_memory_status()
        }

class CTRPostProcessor:
    """CTR 예측 후처리 클래스"""
    
    def __init__(self):
        self.outlier_threshold_low = 0.0001
        self.outlier_threshold_high = 0.9999
        self.diversity_threshold = 1000
        
    def apply_postprocessing(self, predictions: np.ndarray, 
                           target_ctr: float = 0.0201,
                           apply_outlier_clipping: bool = True,
                           apply_diversity_enhancement: bool = True,
                           apply_ctr_alignment: bool = True) -> np.ndarray:
        """전체 후처리 적용"""
        
        processed = predictions.copy()
        
        if apply_outlier_clipping:
            processed = self._clip_outliers(processed)
        
        if apply_diversity_enhancement:
            processed = self._enhance_diversity(processed)
        
        if apply_ctr_alignment:
            processed = self._align_ctr(processed, target_ctr)
        
        processed = np.clip(processed, self.outlier_threshold_low, self.outlier_threshold_high)
        
        return processed
    
    def _clip_outliers(self, predictions: np.ndarray) -> np.ndarray:
        """이상값 제거"""
        try:
            q1 = np.percentile(predictions, 1)
            q99 = np.percentile(predictions, 99)
            
            outlier_low = max(q1 * 0.1, self.outlier_threshold_low)
            outlier_high = min(q99 * 1.1, self.outlier_threshold_high)
            
            clipped = np.clip(predictions, outlier_low, outlier_high)
            
            return clipped
            
        except Exception as e:
            logger.warning(f"이상값 제거 실패: {e}")
            return predictions
    
    def _enhance_diversity(self, predictions: np.ndarray) -> np.ndarray:
        """예측 다양성 향상"""
        try:
            unique_count = len(np.unique(predictions))
            
            if unique_count < self.diversity_threshold:
                noise_scale = max(predictions.std() * 0.008, 1e-6)
                noise = np.random.normal(0, noise_scale, len(predictions))
                
                enhanced = predictions + noise
                enhanced = np.clip(enhanced, self.outlier_threshold_low, self.outlier_threshold_high)
                
                return enhanced
            
            return predictions
            
        except Exception as e:
            logger.warning(f"다양성 향상 실패: {e}")
            return predictions
    
    def _align_ctr(self, predictions: np.ndarray, target_ctr: float) -> np.ndarray:
        """CTR 정렬"""
        try:
            current_ctr = predictions.mean()
            
            if abs(current_ctr - target_ctr) > 0.001:
                if current_ctr > 0:
                    correction_factor = target_ctr / current_ctr
                    aligned = predictions * correction_factor
                    
                    aligned = np.clip(aligned, self.outlier_threshold_low, self.outlier_threshold_high)
                    
                    return aligned
            
            return predictions
            
        except Exception as e:
            logger.warning(f"CTR 정렬 실패: {e}")
            return predictions

class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
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
    """CTR 예측 API - 최종 완성 버전"""
    
    def __init__(self, model_dir: str = "models"):
        self.engine = CTRInferenceEngine(model_dir)
        self.api_stats = {
            'api_calls': 0,
            'api_errors': 0,
            'start_time': time.time(),
            'feature_usage': {
                'calibration_requests': 0,
                'ensemble_requests': 0,
                'postprocessing_requests': 0,
                'full_pipeline_requests': 0
            }
        }
    
    def initialize(self) -> bool:
        """CTR 예측 API 초기화"""
        logger.info("CTR 예측 API 초기화 시작 (최종 완성 버전)")
        
        success = self.engine.load_models()
        
        if success:
            health_check = self.engine.health_check()
            logger.info(f"CTR 예측 API 초기화 완료")
            logger.info(f"총 모델 수: {health_check['models_count']}")
            logger.info(f"캘리브레이션 적용 모델: {health_check['calibrated_models_count']}")
            logger.info(f"캘리브레이션 비율: {health_check['calibration_rate']:.2%}")
            logger.info(f"앙상블 사용 가능: {health_check['ensemble_status']['available']}")
            logger.info(f"후처리 사용 가능: {health_check['postprocessor_available']}")
        else:
            logger.error("CTR 예측 API 초기화 실패")
        
        return success
    
    def predict_ctr(self, user_id: str, ad_id: str, context: Dict[str, Any] = None, 
                   model_name: Optional[str] = None, 
                   use_calibration: bool = True,
                   use_ensemble: bool = True,
                   use_postprocessing: bool = True) -> Dict[str, Any]:
        """CTR 예측 메인 API - 최종 완성 버전"""
        self.api_stats['api_calls'] += 1
        
        if use_calibration and use_ensemble and use_postprocessing:
            self.api_stats['feature_usage']['full_pipeline_requests'] += 1
        else:
            if use_calibration:
                self.api_stats['feature_usage']['calibration_requests'] += 1
            if use_ensemble:
                self.api_stats['feature_usage']['ensemble_requests'] += 1
            if use_postprocessing:
                self.api_stats['feature_usage']['postprocessing_requests'] += 1
        
        try:
            input_data = {
                'user_id': user_id,
                'ad_id': ad_id,
                'context': context or {}
            }
            
            result = self.engine.predict_single(
                input_data, 
                model_name, 
                use_calibration,
                use_ensemble,
                use_postprocessing
            )
            
            api_response = {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': result['ctr_prediction'],
                'raw_prediction': result.get('raw_prediction', result['ctr_prediction']),
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'ctr_category': result['ctr_category'],
                'processing_pipeline': {
                    'ensemble_used': result.get('ensemble_used', False),
                    'calibration_applied': result.get('calibration_applied', False),
                    'postprocessing_applied': result.get('postprocessing_applied', False),
                    'calibration_method': result.get('calibration_method', 'none'),
                    'calibration_quality': result.get('calibration_quality', 0.0)
                },
                'expected_revenue': result['ctr_prediction'] * context.get('bid_amount', 1.0) if context else result['ctr_prediction'],
                'model_info': {
                    'model_name': result['model_used'],
                    'version': 'v3.0_final',
                    'features_supported': {
                        'calibration': True,
                        'ensemble': True,
                        'postprocessing': True
                    }
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
                'processing_pipeline': {
                    'ensemble_used': False,
                    'calibration_applied': False,
                    'postprocessing_applied': False,
                    'calibration_method': 'none',
                    'calibration_quality': 0.0
                },
                'expected_revenue': 0.0,
                'error': str(e),
                'model_info': {
                    'model_name': 'unknown',
                    'version': 'v3.0_final',
                    'features_supported': {
                        'calibration': True,
                        'ensemble': True,
                        'postprocessing': True
                    }
                },
                'performance': {
                    'latency_ms': 0.0,
                    'timestamp': time.time()
                }
            }
    
    def predict_submission(self, test_df: pd.DataFrame, 
                          model_name: Optional[str] = None,
                          use_calibration: bool = True,
                          use_ensemble: bool = True,
                          use_postprocessing: bool = True) -> pd.DataFrame:
        """제출용 예측 생성 - 최종 완성 버전"""
        logger.info(f"제출용 예측 생성 시작 (최종 완성 버전)")
        logger.info(f"파이프라인 설정 - Calibration: {'On' if use_calibration else 'Off'}, "
                   f"Ensemble: {'On' if use_ensemble else 'Off'}, "
                   f"Postprocessing: {'On' if use_postprocessing else 'Off'}")
        
        predictions = self.engine.predict_test_data(
            test_df, 
            model_name, 
            use_calibration,
            use_ensemble,
            use_postprocessing
        )
        
        submission = pd.DataFrame({
            'id': range(len(predictions)),
            'clicked': predictions
        })
        
        logger.info(f"제출용 예측 생성 완료: {len(submission):,}행")
        
        avg_prediction = predictions.mean()
        logger.info(f"평균 예측 CTR: {avg_prediction:.4f}")
        logger.info(f"목표 CTR과의 차이: {abs(avg_prediction - self.engine.ctr_baseline):.4f}")
        logger.info(f"예측 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
        
        return submission
    
    def get_api_status(self) -> Dict[str, Any]:
        """API 상태 정보 - 최종 완성 버전"""
        uptime = time.time() - self.api_stats['start_time']
        
        total_feature_requests = sum(self.api_stats['feature_usage'].values())
        feature_usage_rates = {}
        
        if total_feature_requests > 0:
            for feature, count in self.api_stats['feature_usage'].items():
                feature_usage_rates[feature] = count / total_feature_requests
        
        status = {
            'api_uptime_seconds': uptime,
            'total_api_calls': self.api_stats['api_calls'],
            'api_error_count': self.api_stats['api_errors'],
            'api_success_rate': 1.0 - (self.api_stats['api_errors'] / max(1, self.api_stats['api_calls'])),
            'feature_usage_stats': {
                'total_feature_requests': total_feature_requests,
                'usage_counts': self.api_stats['feature_usage'].copy(),
                'usage_rates': feature_usage_rates,
                'full_pipeline_rate': feature_usage_rates.get('full_pipeline_requests', 0.0)
            },
            'engine_status': self.engine.health_check(),
            'inference_stats': self.engine.get_inference_stats()
        }
        
        return status

def create_ctr_prediction_service(model_dir: str = "models") -> CTRPredictionAPI:
    """CTR 예측 서비스 생성 - 최종 완성 버전"""
    logger.info("CTR 예측 서비스 생성 (최종 완성 버전)")
    
    api = CTRPredictionAPI(model_dir)
    
    if not api.initialize():
        logger.error("CTR 예측 서비스 초기화 실패")
        return None
    
    logger.info("CTR 예측 서비스 생성 완료")
    return api

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    service = create_ctr_prediction_service()
    
    if service:
        status = service.get_api_status()
        print("API 상태:", json.dumps(status, indent=2, default=str))
        
        test_prediction = service.predict_ctr(
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
            use_calibration=True,
            use_ensemble=True,
            use_postprocessing=True
        )
        
        print("CTR 예측 결과 (완전 파이프라인):", json.dumps(test_prediction, indent=2, default=str))
        
        print(f"\n최종 완성 시스템 상태:")
        print(f"앙상블 사용: {test_prediction['processing_pipeline']['ensemble_used']}")
        print(f"캘리브레이션 적용: {test_prediction['processing_pipeline']['calibration_applied']}")
        print(f"후처리 적용: {test_prediction['processing_pipeline']['postprocessing_applied']}")
        print(f"예측 CTR: {test_prediction['ctr_prediction']:.4f}")
        print(f"신뢰도: {test_prediction['confidence']:.3f}")
        print(f"응답 시간: {test_prediction['performance']['latency_ms']:.2f}ms")