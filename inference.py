# inference.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
import pickle
import time
import json
from pathlib import Path
import threading
from collections import OrderedDict
import hashlib
import warnings
warnings.filterwarnings('ignore')

from config import Config
from models import BaseModel
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class ModelCache:
    """모델 캐싱 시스템"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: str, value: Any):
        """캐시에 값 저장"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self):
        """캐시 초기화"""
        with self.lock:
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

class CTRFeatureCache:
    """CTR 예측용 피처 캐싱 시스템"""
    
    def __init__(self, max_size: int = 10000):
        self.user_features = ModelCache(max_size)
        self.ad_features = ModelCache(max_size)
        self.context_features = ModelCache(max_size)
        self.interaction_features = ModelCache(max_size)
    
    def get_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """사용자 피처 조회"""
        return self.user_features.get(user_id)
    
    def cache_user_features(self, user_id: str, features: Dict[str, Any]):
        """사용자 피처 캐싱"""
        self.user_features.put(user_id, features)
    
    def get_ad_features(self, ad_id: str) -> Optional[Dict[str, Any]]:
        """광고 피처 조회"""
        return self.ad_features.get(ad_id)
    
    def cache_ad_features(self, ad_id: str, features: Dict[str, Any]):
        """광고 피처 캐싱"""
        self.ad_features.put(ad_id, features)
    
    def get_interaction_features(self, user_id: str, ad_id: str) -> Optional[Dict[str, Any]]:
        """사용자-광고 상호작용 피처 조회"""
        interaction_key = f"{user_id}_{ad_id}"
        return self.interaction_features.get(interaction_key)
    
    def cache_interaction_features(self, user_id: str, ad_id: str, features: Dict[str, Any]):
        """사용자-광고 상호작용 피처 캐싱"""
        interaction_key = f"{user_id}_{ad_id}"
        self.interaction_features.put(interaction_key, features)
    
    def clear_all(self):
        """모든 캐시 초기화"""
        self.user_features.clear()
        self.ad_features.clear()
        self.context_features.clear()
        self.interaction_features.clear()

class CTRInferenceEngine:
    """CTR 예측 특화 실시간 추론 엔진"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.models = {}
        self.feature_engineer = None
        self.feature_cache = CTRFeatureCache(
            max_size=config.INFERENCE_CONFIG['cache_size']
        )
        self.is_loaded = False
        self.inference_stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_latency': 0.0,
            'cache_hit_rate': 0.0,
            'ctr_predictions': []
        }
        self.lock = threading.Lock()
        
        # CTR 예측용 기본 피처 설정
        self.default_numeric_features = self._get_default_features()
        
        # 학습된 모델의 피처 순서 저장
        self.expected_feature_columns = None
        
        # CTR 예측용 카테고리 매핑
        self.category_mappings = {
            'gender': {'male': 1, 'female': 2, 'unknown': 0},
            'age_group': {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6, 'unknown': 0},
            'device': {'mobile': 1, 'desktop': 2, 'tablet': 3, 'unknown': 0},
            'page': {'home': 1, 'search': 2, 'product': 3, 'checkout': 4, 'category': 5, 'unknown': 0},
            'time_of_day': {'morning': 1, 'afternoon': 2, 'evening': 3, 'night': 4, 'unknown': 0},
            'day_of_week': {'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 
                          'friday': 5, 'saturday': 6, 'sunday': 7, 'unknown': 0}
        }
        
        # CTR 도메인 지식 기반 기본값
        self.ctr_baseline = 0.0191  # 실제 관찰된 CTR
    
    def _get_default_features(self) -> Dict[str, float]:
        """CTR 예측용 기본 피처 값"""
        return {
            'feat_e_1': 65.0,
            'feat_e_3': 5.0,
            'feat_e_4': -0.05,
            'feat_e_5': -0.02,
            'feat_e_6': -0.04,
            'feat_e_7': 21.0,
            'feat_e_8': -172.0,
            'feat_e_9': -10.0,
            'feat_e_10': -270.0,
            'feat_d_1': 0.39,
            'feat_d_2': 1.93,
            'feat_d_3': 1.76,
            'feat_d_4': 6.0,
            'feat_d_5': -0.29,
            'feat_d_6': -0.41,
            'feat_c_1': 0.0,
            'feat_c_2': 0.0,
            'feat_c_3': 0.0,
            'feat_c_4': 0.0,
            'feat_c_5': 0.0,
            'feat_c_6': 0.0,
            'feat_c_7': 0.0,
            'feat_c_8': 0.0,
            'feat_b_1': 0.0,
            'feat_b_2': 0.0,
            'feat_b_3': 0.0,
            'feat_b_4': 0.0,
            'feat_b_5': 0.0,
            'feat_b_6': 0.0
        }
    
    def load_models(self, model_dir: Path = None) -> bool:
        """저장된 모델들 로딩"""
        if model_dir is None:
            model_dir = self.config.MODEL_DIR
        
        model_dir = Path(model_dir)
        logger.info(f"모델 로딩 시작: {model_dir}")
        
        try:
            # 모델 파일 찾기
            model_files = list(model_dir.glob("*_model.pkl"))
            
            if not model_files:
                logger.error("로딩할 모델 파일이 없습니다.")
                return False
            
            # 모델 로딩
            for model_file in model_files:
                model_name = model_file.stem.replace('_model', '')
                
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    self.models[model_name] = model
                    logger.info(f"{model_name} 모델 로딩 완료")
                    
                    # 첫 번째 모델에서 피처 순서 추출
                    if self.expected_feature_columns is None and hasattr(model, 'feature_names'):
                        self.expected_feature_columns = model.feature_names
                        logger.info(f"모델 피처 순서 설정: {len(self.expected_feature_columns)}개")
                    
                except Exception as e:
                    logger.error(f"{model_name} 모델 로딩 실패: {str(e)}")
                    continue
            
            # 피처 엔지니어 로딩
            feature_engineer_path = model_dir / "feature_engineer.pkl"
            if feature_engineer_path.exists():
                try:
                    with open(feature_engineer_path, 'rb') as f:
                        self.feature_engineer = pickle.load(f)
                    logger.info("피처 엔지니어 로딩 완료")
                    
                    # 피처 엔지니어에서 피처 순서 정보 추출
                    if hasattr(self.feature_engineer, 'final_feature_columns'):
                        if self.expected_feature_columns is None:
                            self.expected_feature_columns = self.feature_engineer.final_feature_columns
                            logger.info(f"피처 엔지니어에서 피처 순서 설정: {len(self.expected_feature_columns)}개")
                    
                except Exception as e:
                    logger.warning(f"피처 엔지니어 로딩 실패: {str(e)}")
                    self.feature_engineer = FeatureEngineer(self.config)
            else:
                logger.warning("피처 엔지니어 파일이 없습니다. 새로 생성합니다.")
                self.feature_engineer = FeatureEngineer(self.config)
            
            self.is_loaded = len(self.models) > 0
            logger.info(f"총 {len(self.models)}개 모델 로딩 완료")
            
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            return False
    
    def _create_ctr_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """CTR 예측용 피처 생성"""
        try:
            # 기본 피처로 초기화
            features = self.default_numeric_features.copy()
            
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
            
            # 세션 내 위치 (가상)
            features['session_position'] = context.get('position', 1)
            features['is_first_position'] = 1 if context.get('position', 1) == 1 else 0
            
            # 시간적 피처 (현재 시점 기준)
            current_time = time.time()
            features['time_index'] = (current_time % 86400) / 86400  # 하루 내 시간 비율
            
            # 추가 컨텍스트 피처들
            for key, value in context.items():
                if key not in ['device', 'page', 'hour', 'day_of_week', 'position']:
                    try:
                        # 수치형 변환 시도
                        features[f'context_{key}'] = float(value)
                    except:
                        # 문자열은 해시로 변환
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
            logger.error(f"CTR 피처 생성 실패: {str(e)}")
            # 기본 피처만 반환
            df = pd.DataFrame([self.default_numeric_features])
            return df.astype('float32')
    
    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """모델 학습 시와 동일한 피처 순서 보장"""
        try:
            if self.expected_feature_columns is None:
                logger.warning("예상 피처 순서 정보가 없습니다")
                return df
            
            # 누락된 피처를 0으로 채움
            for feature in self.expected_feature_columns:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # 학습 시와 동일한 순서로 재정렬
            df = df[self.expected_feature_columns]
            
            logger.debug(f"피처 일관성 보장 완료: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"피처 일관성 보장 실패: {str(e)}")
            return df
    
    def predict_single(self, 
                      input_data: Dict[str, Any],
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """단일 CTR 예측"""
        
        start_time = time.time()
        
        try:
            with self.lock:
                self.inference_stats['total_requests'] += 1
            
            # 입력 검증
            if not self.is_loaded:
                raise ValueError("모델이 로딩되지 않았습니다.")
            
            # CTR 특화 피처 생성
            df = self._create_ctr_features(input_data)
            
            # 모델 선택
            if model_name is None:
                # 기본적으로 첫 번째 모델 사용
                model_name = list(self.models.keys())[0]
            
            if model_name not in self.models:
                raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다.")
            
            model = self.models[model_name]
            
            # 피처 일관성 보장 (학습된 모델 기준)
            df = self._ensure_feature_consistency(df)
            
            # CTR 예측 수행
            try:
                prediction_proba = model.predict_proba(df)[0]
                
                # CTR 값 검증 및 보정
                prediction_proba = max(0.0001, min(0.9999, prediction_proba))
                
            except Exception as e:
                logger.warning(f"모델 예측 실패, 기본값 사용: {str(e)}")
                prediction_proba = self.ctr_baseline
            
            prediction_binary = int(prediction_proba >= 0.5)
            
            # CTR 예측 신뢰도 계산
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
            
            # 응답 시간 계산
            latency = (time.time() - start_time) * 1000  # 밀리초
            
            # 통계 업데이트
            with self.lock:
                self.inference_stats['successful_predictions'] += 1
                self.inference_stats['ctr_predictions'].append(prediction_proba)
                # 최근 1000개만 유지
                if len(self.inference_stats['ctr_predictions']) > 1000:
                    self.inference_stats['ctr_predictions'] = self.inference_stats['ctr_predictions'][-1000:]
                self._update_average_latency(latency)
            
            result = {
                'ctr_prediction': float(prediction_proba),
                'predicted_click': prediction_binary,
                'confidence': float(confidence),
                'ctr_category': ctr_category,
                'model_used': model_name,
                'latency_ms': latency,
                'recommendation': 'show' if prediction_proba > self.ctr_baseline else 'skip',
                'timestamp': time.time()
            }
            
            # 로그 (낮은 빈도로)
            if self.inference_stats['total_requests'] % 1000 == 0:
                logger.info(f"CTR 예측 완료 (누적: {self.inference_stats['total_requests']}회)")
            
            return result
            
        except Exception as e:
            with self.lock:
                self.inference_stats['failed_predictions'] += 1
            
            logger.error(f"CTR 예측 실패: {str(e)}")
            
            return {
                'ctr_prediction': self.ctr_baseline,
                'predicted_click': 0,
                'confidence': 0.0,
                'ctr_category': 'low',
                'model_used': model_name or 'unknown',
                'latency_ms': (time.time() - start_time) * 1000,
                'recommendation': 'skip',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def predict_batch(self, 
                     input_data: List[Dict[str, Any]],
                     model_name: Optional[str] = None,
                     batch_size: int = None) -> List[Dict[str, Any]]:
        """배치 CTR 예측"""
        
        if batch_size is None:
            batch_size = self.config.INFERENCE_CONFIG['batch_size']
        
        logger.info(f"배치 CTR 예측 시작: {len(input_data)}개 샘플")
        
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            
            try:
                # 배치 피처 생성
                batch_features = []
                for item in batch:
                    df = self._create_ctr_features(item)
                    batch_features.append(df)
                
                # DataFrame 결합
                combined_df = pd.concat(batch_features, ignore_index=True)
                
                # 모델 선택
                if model_name is None:
                    model_name = list(self.models.keys())[0]
                
                model = self.models[model_name]
                
                # 피처 일관성 보장
                combined_df = self._ensure_feature_consistency(combined_df)
                
                # 배치 예측
                start_time = time.time()
                try:
                    predictions_proba = model.predict_proba(combined_df)
                    # CTR 값 검증 및 보정
                    predictions_proba = np.clip(predictions_proba, 0.0001, 0.9999)
                except Exception as e:
                    logger.warning(f"배치 모델 예측 실패, 기본값 사용: {str(e)}")
                    predictions_proba = np.full(len(batch), self.ctr_baseline)
                
                predictions_binary = (predictions_proba >= 0.5).astype(int)
                latency = (time.time() - start_time) * 1000
                
                # 결과 구성
                for j, (item, prob, binary) in enumerate(zip(batch, predictions_proba, predictions_binary)):
                    confidence = abs(prob - 0.5) * 2
                    
                    # CTR 카테고리 분류
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
                        'predicted_click': int(binary),
                        'confidence': float(confidence),
                        'ctr_category': ctr_category,
                        'model_used': model_name,
                        'batch_latency_ms': latency,
                        'batch_index': i + j,
                        'recommendation': 'show' if prob > self.ctr_baseline else 'skip',
                        'timestamp': time.time()
                    }
                    results.append(result)
                
            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} CTR 예측 실패: {str(e)}")
                
                # 오류 발생 시 기본값으로 채움
                for j in range(len(batch)):
                    result = {
                        'ctr_prediction': self.ctr_baseline,
                        'predicted_click': 0,
                        'confidence': 0.0,
                        'ctr_category': 'low',
                        'model_used': model_name or 'unknown',
                        'batch_latency_ms': 0.0,
                        'batch_index': i + j,
                        'recommendation': 'skip',
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    results.append(result)
        
        logger.info(f"배치 CTR 예측 완료: {len(results)}개 결과")
        return results
    
    def _update_average_latency(self, new_latency: float):
        """평균 지연시간 업데이트"""
        current_count = self.inference_stats['successful_predictions']
        current_avg = self.inference_stats['average_latency']
        
        # 이동 평균 계산
        new_avg = ((current_avg * (current_count - 1)) + new_latency) / current_count
        self.inference_stats['average_latency'] = new_avg
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """CTR 예측 통계 반환"""
        stats = self.inference_stats.copy()
        
        # 성공률 계산
        total = stats['total_requests']
        if total > 0:
            stats['success_rate'] = stats['successful_predictions'] / total
            stats['failure_rate'] = stats['failed_predictions'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # CTR 예측 통계
        if stats['ctr_predictions']:
            ctr_preds = np.array(stats['ctr_predictions'])
            stats['ctr_mean'] = float(ctr_preds.mean())
            stats['ctr_std'] = float(ctr_preds.std())
            stats['ctr_min'] = float(ctr_preds.min())
            stats['ctr_max'] = float(ctr_preds.max())
            stats['ctr_percentiles'] = {
                'p50': float(np.percentile(ctr_preds, 50)),
                'p90': float(np.percentile(ctr_preds, 90)),
                'p95': float(np.percentile(ctr_preds, 95)),
                'p99': float(np.percentile(ctr_preds, 99))
            }
        else:
            stats['ctr_mean'] = 0.0
            stats['ctr_std'] = 0.0
            stats['ctr_min'] = 0.0
            stats['ctr_max'] = 0.0
            stats['ctr_percentiles'] = {}
        
        # 캐시 통계 추가
        cache_stats = self.feature_cache.user_features.get_stats()
        stats['cache_hit_rate'] = cache_stats['hit_rate']
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            'is_loaded': self.is_loaded,
            'models_count': len(self.models),
            'available_models': list(self.models.keys()),
            'feature_engineer_loaded': self.feature_engineer is not None,
            'cache_size': len(self.feature_cache.user_features.cache),
            'system_status': 'healthy' if self.is_loaded else 'not_ready',
            'ctr_baseline': self.ctr_baseline,
            'expected_features_count': len(self.expected_feature_columns) if self.expected_feature_columns else 0
        }

class CTRPredictionAPI:
    """CTR 예측 API 래퍼"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.engine = CTRInferenceEngine(config)
        self.api_stats = {
            'api_calls': 0,
            'api_errors': 0,
            'start_time': time.time()
        }
    
    def initialize(self, model_dir: Path = None) -> bool:
        """CTR 예측 API 초기화"""
        logger.info("CTR 예측 API 초기화 시작")
        
        success = self.engine.load_models(model_dir)
        
        if success:
            logger.info("CTR 예측 API 초기화 완료")
        else:
            logger.error("CTR 예측 API 초기화 실패")
        
        return success
    
    def predict_ctr(self, 
                   user_id: str,
                   ad_id: str,
                   context: Dict[str, Any] = None,
                   model_name: Optional[str] = None) -> Dict[str, Any]:
        """CTR 예측 메인 API"""
        
        self.api_stats['api_calls'] += 1
        
        try:
            # 입력 데이터 구성
            input_data = {
                'user_id': user_id,
                'ad_id': ad_id,
                'context': context or {}
            }
            
            # CTR 예측 수행
            result = self.engine.predict_single(input_data, model_name)
            
            # API 응답 형태로 변환
            api_response = {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': result['ctr_prediction'],
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'ctr_category': result['ctr_category'],
                'expected_revenue': result['ctr_prediction'] * context.get('bid_amount', 1.0) if context else result['ctr_prediction'],
                'model_info': {
                    'model_name': result['model_used'],
                    'version': self.config.INFERENCE_CONFIG['model_version']
                },
                'performance': {
                    'latency_ms': result['latency_ms'],
                    'timestamp': result['timestamp']
                }
            }
            
            return api_response
            
        except Exception as e:
            self.api_stats['api_errors'] += 1
            logger.error(f"CTR 예측 API 오류: {str(e)}")
            
            return {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': self.engine.ctr_baseline,
                'recommendation': 'skip',
                'confidence': 0.0,
                'ctr_category': 'low',
                'expected_revenue': 0.0,
                'error': str(e),
                'model_info': {
                    'model_name': 'unknown',
                    'version': self.config.INFERENCE_CONFIG['model_version']
                },
                'performance': {
                    'latency_ms': 0.0,
                    'timestamp': time.time()
                }
            }
    
    def predict_ctr_batch(self, 
                         requests: List[Dict[str, Any]],
                         model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """배치 CTR 예측"""
        
        logger.info(f"배치 CTR 예측 요청: {len(requests)}개")
        
        # 입력 데이터 변환
        input_data = []
        for req in requests:
            input_item = {
                'user_id': req.get('user_id', ''),
                'ad_id': req.get('ad_id', ''),
                'context': req.get('context', {})
            }
            input_data.append(input_item)
        
        # 배치 예측 수행
        predictions = self.engine.predict_batch(input_data, model_name)
        
        # API 응답 형태로 변환
        api_responses = []
        for i, (req, pred) in enumerate(zip(requests, predictions)):
            context = req.get('context', {})
            response = {
                'user_id': req.get('user_id', ''),
                'ad_id': req.get('ad_id', ''),
                'ctr_prediction': pred['ctr_prediction'],
                'recommendation': pred['recommendation'],
                'confidence': pred['confidence'],
                'ctr_category': pred['ctr_category'],
                'expected_revenue': pred['ctr_prediction'] * context.get('bid_amount', 1.0),
                'batch_index': i,
                'model_info': {
                    'model_name': pred['model_used'],
                    'version': self.config.INFERENCE_CONFIG['model_version']
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
    
    def get_api_status(self) -> Dict[str, Any]:
        """API 상태 정보"""
        uptime = time.time() - self.api_stats['start_time']
        
        status = {
            'api_uptime_seconds': uptime,
            'total_api_calls': self.api_stats['api_calls'],
            'api_error_count': self.api_stats['api_errors'],
            'api_success_rate': 1.0 - (self.api_stats['api_errors'] / max(1, self.api_stats['api_calls'])),
            'engine_status': self.engine.health_check(),
            'inference_stats': self.engine.get_inference_stats()
        }
        
        return status

def create_ctr_prediction_service(config: Config = Config) -> CTRPredictionAPI:
    """CTR 예측 서비스 생성 팩토리 함수"""
    
    logger.info("CTR 예측 서비스 생성")
    
    # API 인스턴스 생성
    api = CTRPredictionAPI(config)
    
    # 초기화
    if not api.initialize():
        logger.error("CTR 예측 서비스 초기화 실패")
        return None
    
    logger.info("CTR 예측 서비스 생성 완료")
    return api

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 서비스 생성
    service = create_ctr_prediction_service()
    
    if service:
        # 상태 확인
        status = service.get_api_status()
        print("API 상태:", json.dumps(status, indent=2))
        
        # 단일 CTR 예측 테스트
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
            }
        )
        
        print("CTR 예측 결과:", json.dumps(test_prediction, indent=2))