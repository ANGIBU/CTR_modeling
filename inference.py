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
                # LRU 업데이트
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
                    # 가장 오래된 항목 제거
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

class FeatureCache:
    """피처 캐싱 시스템"""
    
    def __init__(self, max_size: int = 10000):
        self.user_features = ModelCache(max_size)
        self.ad_features = ModelCache(max_size)
        self.context_features = ModelCache(max_size)
    
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
    
    def clear_all(self):
        """모든 캐시 초기화"""
        self.user_features.clear()
        self.ad_features.clear()
        self.context_features.clear()

class RealTimeInferenceEngine:
    """실시간 추론 엔진"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.models = {}
        self.feature_engineer = None
        self.feature_cache = FeatureCache(
            max_size=config.INFERENCE_CONFIG['cache_size']
        )
        self.is_loaded = False
        self.inference_stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_latency': 0.0,
            'cache_hit_rate': 0.0
        }
        self.lock = threading.Lock()
        
        # 추론용 기본 피처 설정
        self.default_numeric_features = [
            'feat_e_1', 'feat_e_3', 'feat_e_4', 'feat_e_5', 'feat_e_6', 'feat_e_7',
            'feat_e_8', 'feat_e_9', 'feat_e_10', 'feat_d_1', 'feat_d_2', 'feat_d_3',
            'feat_d_4', 'feat_d_5', 'feat_d_6', 'feat_c_1', 'feat_c_2', 'feat_c_3',
            'feat_c_4', 'feat_c_5', 'feat_c_6', 'feat_c_7', 'feat_c_8',
            'feat_b_1', 'feat_b_2', 'feat_b_3', 'feat_b_4', 'feat_b_5', 'feat_b_6'
        ]
        
        # 추론 시 카테고리 매핑
        self.category_mappings = {
            'gender': {'male': 1, 'female': 2, 'unknown': 0},
            'age_group': {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6, 'unknown': 0},
            'device': {'mobile': 1, 'desktop': 2, 'tablet': 3, 'unknown': 0},
            'page': {'home': 1, 'search': 2, 'product': 3, 'checkout': 4, 'unknown': 0}
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
                
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                self.models[model_name] = model
                logger.info(f"{model_name} 모델 로딩 완료")
            
            # 피처 엔지니어 로딩
            feature_engineer_path = model_dir / "feature_engineer.pkl"
            if feature_engineer_path.exists():
                with open(feature_engineer_path, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
                logger.info("피처 엔지니어 로딩 완료")
            else:
                logger.warning("피처 엔지니어 파일이 없습니다. 새로 생성합니다.")
                self.feature_engineer = FeatureEngineer(self.config)
            
            self.is_loaded = True
            logger.info(f"총 {len(self.models)}개 모델 로딩 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            return False
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """입력 데이터 전처리 (수치형으로 변환)"""
        try:
            # DataFrame으로 변환
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = input_data.copy()
            
            # 범주형 데이터를 수치형으로 변환
            for col in df.columns:
                if df[col].dtype == 'object':
                    if col in self.category_mappings:
                        # 사전 정의된 매핑 사용
                        df[col] = df[col].map(self.category_mappings[col]).fillna(0)
                    else:
                        # 문자열을 해시값으로 변환
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if df[col].isnull().any():
                                # 수치 변환 실패 시 해시 사용
                                df[col] = df[col].astype(str).apply(lambda x: hash(str(x)) % 1000000)
                        except:
                            df[col] = 0
                else:
                    df[col] = df[col].fillna(0)
            
            # 기본 피처들 추가 (없는 경우)
            for feature in self.default_numeric_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # 데이터 타입을 float32로 통일
            for col in df.columns:
                try:
                    df[col] = df[col].astype('float32')
                except:
                    df[col] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"입력 데이터 전처리 실패: {str(e)}")
            raise
    
    def predict_single(self, 
                      input_data: Dict[str, Any],
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """단일 샘플 예측"""
        
        start_time = time.time()
        
        try:
            with self.lock:
                self.inference_stats['total_requests'] += 1
            
            # 입력 검증
            if not self.is_loaded:
                raise ValueError("모델이 로딩되지 않았습니다.")
            
            # 전처리
            df = self.preprocess_input(input_data)
            
            # 모델 선택
            if model_name is None:
                # 기본적으로 첫 번째 모델 사용
                model_name = list(self.models.keys())[0]
            
            if model_name not in self.models:
                raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다.")
            
            model = self.models[model_name]
            
            # 피처 수가 부족한 경우 기본값으로 채움
            required_features = getattr(model, 'n_features_in_', None)
            if required_features and df.shape[1] < required_features:
                for i in range(df.shape[1], required_features):
                    df[f'dummy_feature_{i}'] = 0.0
            
            # 예측 수행
            prediction_proba = model.predict_proba(df)[0]
            prediction_binary = int(prediction_proba >= 0.5)
            
            # 응답 시간 계산
            latency = (time.time() - start_time) * 1000  # 밀리초
            
            # 통계 업데이트
            with self.lock:
                self.inference_stats['successful_predictions'] += 1
                self._update_average_latency(latency)
            
            result = {
                'click_probability': float(prediction_proba),
                'predicted_click': prediction_binary,
                'model_used': model_name,
                'latency_ms': latency,
                'timestamp': time.time()
            }
            
            # 로그 (낮은 빈도로)
            if self.inference_stats['total_requests'] % 1000 == 0:
                logger.info(f"예측 완료 (누적: {self.inference_stats['total_requests']}회)")
            
            return result
            
        except Exception as e:
            with self.lock:
                self.inference_stats['failed_predictions'] += 1
            
            logger.error(f"예측 실패: {str(e)}")
            
            return {
                'click_probability': 0.0,
                'predicted_click': 0,
                'model_used': model_name or 'unknown',
                'latency_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def predict_batch(self, 
                     input_data: List[Dict[str, Any]],
                     model_name: Optional[str] = None,
                     batch_size: int = None) -> List[Dict[str, Any]]:
        """배치 예측"""
        
        if batch_size is None:
            batch_size = self.config.INFERENCE_CONFIG['batch_size']
        
        logger.info(f"배치 예측 시작: {len(input_data)}개 샘플")
        
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            
            try:
                # DataFrame으로 변환
                df = self.preprocess_input(batch)
                
                # 모델 선택
                if model_name is None:
                    model_name = list(self.models.keys())[0]
                
                model = self.models[model_name]
                
                # 배치 예측
                start_time = time.time()
                predictions_proba = model.predict_proba(df)
                predictions_binary = (predictions_proba >= 0.5).astype(int)
                latency = (time.time() - start_time) * 1000
                
                # 결과 구성
                for j, (prob, binary) in enumerate(zip(predictions_proba, predictions_binary)):
                    result = {
                        'click_probability': float(prob),
                        'predicted_click': int(binary),
                        'model_used': model_name,
                        'batch_latency_ms': latency,
                        'batch_index': i + j,
                        'timestamp': time.time()
                    }
                    results.append(result)
                
            except Exception as e:
                logger.error(f"배치 {i//batch_size + 1} 예측 실패: {str(e)}")
                
                # 오류 발생 시 기본값으로 채움
                for j in range(len(batch)):
                    result = {
                        'click_probability': 0.0,
                        'predicted_click': 0,
                        'model_used': model_name or 'unknown',
                        'batch_latency_ms': 0.0,
                        'batch_index': i + j,
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    results.append(result)
        
        logger.info(f"배치 예측 완료: {len(results)}개 결과")
        return results
    
    def _update_average_latency(self, new_latency: float):
        """평균 지연시간 업데이트"""
        current_count = self.inference_stats['successful_predictions']
        current_avg = self.inference_stats['average_latency']
        
        # 이동 평균 계산
        new_avg = ((current_avg * (current_count - 1)) + new_latency) / current_count
        self.inference_stats['average_latency'] = new_avg
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계 반환"""
        stats = self.inference_stats.copy()
        
        # 성공률 계산
        total = stats['total_requests']
        if total > 0:
            stats['success_rate'] = stats['successful_predictions'] / total
            stats['failure_rate'] = stats['failed_predictions'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
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
            'system_status': 'healthy' if self.is_loaded else 'not_ready'
        }

class CTRPredictionAPI:
    """CTR 예측 API 래퍼"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.engine = RealTimeInferenceEngine(config)
        self.api_stats = {
            'api_calls': 0,
            'api_errors': 0,
            'start_time': time.time()
        }
    
    def initialize(self, model_dir: Path = None) -> bool:
        """API 초기화"""
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
            # 입력 데이터 구성 (모든 값을 수치형으로 처리)
            input_data = {}
            
            # 사용자 ID를 해시값으로 변환
            try:
                input_data['user_hash'] = hash(str(user_id)) % 1000000
            except:
                input_data['user_hash'] = 0
            
            # 광고 ID를 해시값으로 변환
            try:
                input_data['ad_hash'] = hash(str(ad_id)) % 1000000
            except:
                input_data['ad_hash'] = 0
            
            # 컨텍스트 추가 (수치형으로 변환)
            if context:
                for key, value in context.items():
                    try:
                        # 숫자로 변환 시도
                        input_data[key] = float(value)
                    except:
                        # 변환 실패 시 해시값 사용
                        input_data[key] = hash(str(value)) % 1000
            
            # 기본 피처들 추가
            default_features = {
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
                'feat_d_6': -0.41
            }
            
            for key, value in default_features.items():
                if key not in input_data:
                    input_data[key] = value
            
            # 예측 수행
            result = self.engine.predict_single(input_data, model_name)
            
            # API 응답 형태로 변환
            api_response = {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': result['click_probability'],
                'recommendation': 'show' if result['predicted_click'] else 'skip',
                'confidence': abs(result['click_probability'] - 0.5) * 2,  # 0~1 사이 신뢰도
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
                'ctr_prediction': 0.0,
                'recommendation': 'skip',
                'confidence': 0.0,
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
            input_item = {}
            
            # 기본 정보
            try:
                input_item['user_hash'] = hash(str(req.get('user_id', ''))) % 1000000
                input_item['ad_hash'] = hash(str(req.get('ad_id', ''))) % 1000000
            except:
                input_item['user_hash'] = 0
                input_item['ad_hash'] = 0
            
            # 추가 컨텍스트
            context = req.get('context', {})
            for key, value in context.items():
                try:
                    input_item[key] = float(value)
                except:
                    input_item[key] = hash(str(value)) % 1000
            
            input_data.append(input_item)
        
        # 배치 예측 수행
        predictions = self.engine.predict_batch(input_data, model_name)
        
        # API 응답 형태로 변환
        api_responses = []
        for i, (req, pred) in enumerate(zip(requests, predictions)):
            response = {
                'user_id': req.get('user_id', ''),
                'ad_id': req.get('ad_id', ''),
                'ctr_prediction': pred['click_probability'],
                'recommendation': 'show' if pred['predicted_click'] else 'skip',
                'confidence': abs(pred['click_probability'] - 0.5) * 2,
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

def create_prediction_service(config: Config = Config) -> CTRPredictionAPI:
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
    service = create_prediction_service()
    
    if service:
        # 상태 확인
        status = service.get_api_status()
        print("API 상태:", json.dumps(status, indent=2))
        
        # 단일 예측 테스트
        test_prediction = service.predict_ctr(
            user_id="test_user_123",
            ad_id="test_ad_456",
            context={'device': 'mobile', 'page': 'home'}
        )
        
        print("예측 결과:", json.dumps(test_prediction, indent=2))