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
    """CTR prediction dedicated real-time inference engine"""
    
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
        """Initialize default feature values"""
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
        """Load saved models"""
        logger.info(f"Model loading started: {self.model_dir}")
        
        if not self.model_dir.exists():
            logger.error(f"Model directory does not exist: {self.model_dir}")
            return False
        
        try:
            model_files = list(self.model_dir.glob("*_model.pkl"))
            
            if not model_files:
                logger.error("No model files to load")
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
                    
                    logger.info(f"{model_name} model loading completed - Calibration: {calibration_status}")
                    
                    if self.expected_feature_columns is None and hasattr(model, 'feature_names'):
                        self.expected_feature_columns = model.feature_names
                        logger.info(f"Feature order set: {len(self.expected_feature_columns)} features")
                        
                except Exception as e:
                    logger.error(f"{model_name} model loading failed: {e}")
                    continue
            
            feature_engineer_path = self.model_dir / "feature_engineer.pkl"
            if feature_engineer_path.exists():
                try:
                    with open(feature_engineer_path, 'rb') as f:
                        self.feature_engineer = pickle.load(f)
                    logger.info("Feature engineer loading completed")
                    
                    if hasattr(self.feature_engineer, 'final_feature_columns'):
                        if self.expected_feature_columns is None:
                            self.expected_feature_columns = self.feature_engineer.final_feature_columns
                            logger.info(f"Feature order set from feature engineer: {len(self.expected_feature_columns)} features")
                            
                except Exception as e:
                    logger.warning(f"Feature engineer loading failed: {e}")
            
            ensemble_manager_path = self.model_dir / "ensemble_manager.pkl"
            if ensemble_manager_path.exists():
                try:
                    with open(ensemble_manager_path, 'rb') as f:
                        self.ensemble_manager = pickle.load(f)
                    logger.info("Ensemble manager loading completed")
                except Exception as e:
                    logger.warning(f"Ensemble manager loading failed: {e}")
            
            self.is_loaded = len(self.models) > 0
            logger.info(f"Total {len(self.models)} models loaded (calibration applied: {calibrated_count})")
            logger.info(f"Ensemble manager: {'Loaded' if self.ensemble_manager else 'None'}")
            
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def _create_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Create features from input data"""
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
            logger.error(f"Feature creation failed: {e}")
            df = pd.DataFrame([self.default_features])
            return df.astype('float32')
    
    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure same feature order as model training"""
        try:
            if self.expected_feature_columns is None:
                return df
            
            for feature in self.expected_feature_columns:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            df = df[self.expected_feature_columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Feature consistency check failed: {e}")
            return df
    
    def predict_single(self, input_data: Dict[str, Any], 
                      model_name: Optional[str] = None, 
                      use_calibration: bool = False,
                      use_ensemble: bool = True,
                      use_postprocessing: bool = False) -> Dict[str, Any]:
        """Single CTR prediction"""
        start_time = time.time()
        
        try:
            with self.lock:
                self.inference_stats['total_requests'] += 1
            
            if not self.is_loaded:
                raise ValueError("Models not loaded")
            
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
                        
                        logger.debug("Using ensemble prediction")
                    except Exception as e:
                        logger.warning(f"Ensemble prediction failed, using single model: {e}")
                        use_ensemble = False
                
                if not use_ensemble:
                    if model_name is None:
                        model_name = list(self.models.keys())[0]
                    
                    if model_name not in self.models:
                        raise ValueError(f"Model '{model_name}' not found")
                    
                    model = self.models[model_name]
                    
                    if hasattr(model, 'predict_proba_raw'):
                        raw_prediction = model.predict_proba_raw(df)[0]
                    else:
                        raw_prediction = model.predict_proba(df)[0]
                    
                    if use_calibration and hasattr(model, 'is_calibrated') and model.is_calibrated:
                        prediction_proba = model.predict_proba(df)[0]
                        calibration_applied = True
                        
                        with self.lock:
                            self.inference_stats['calibration_stats']['calibrated_predictions'] += 1
                    else:
                        prediction_proba = raw_prediction
                        
                        with self.lock:
                            self.inference_stats['calibration_stats']['raw_predictions'] += 1
                    
                    with self.lock:
                        self.inference_stats['ensemble_stats']['single_model_predictions'] += 1
                
                prediction_proba = max(0.0001, min(0.9999, prediction_proba))
                
            except Exception as e:
                logger.warning(f"Model prediction failed, using default value: {e}")
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
            
            return result
            
        except Exception as e:
            with self.lock:
                self.inference_stats['failed_predictions'] += 1
            
            logger.error(f"CTR prediction failed: {e}")
            
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
                     use_calibration: bool = False,
                     use_ensemble: bool = True,
                     use_postprocessing: bool = False) -> List[Dict[str, Any]]:
        """Batch CTR prediction"""
        batch_size = 15000
        results = []
        
        logger.info(f"Batch CTR prediction started: {len(input_data)} samples")
        
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
                
                try:
                    if use_ensemble and self.ensemble_manager is not None:
                        try:
                            predictions_proba = self.ensemble_manager.predict_with_best_ensemble(combined_df)
                            prediction_method = "ensemble"
                            ensemble_used = True
                            
                            with self.lock:
                                self.inference_stats['ensemble_stats']['ensemble_predictions'] += len(batch)
                        except Exception as e:
                            logger.warning(f"Batch ensemble prediction failed: {e}")
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
                        
                        if use_calibration and hasattr(model, 'is_calibrated') and model.is_calibrated:
                            predictions_proba = model.predict_proba(combined_df)
                            calibration_applied = True
                        else:
                            predictions_proba = raw_predictions
                        
                        with self.lock:
                            self.inference_stats['ensemble_stats']['single_model_predictions'] += len(batch)
                    
                    predictions_proba = np.clip(predictions_proba, 0.0001, 0.9999)
                    
                    if 'raw_predictions' not in locals():
                        raw_predictions = predictions_proba
                    
                    raw_predictions = np.clip(raw_predictions, 0.0001, 0.9999)
                    
                except Exception as e:
                    logger.warning(f"Batch model prediction failed: {e}")
                    predictions_proba = np.full(len(batch), self.ctr_baseline)
                    raw_predictions = np.full(len(batch), self.ctr_baseline)
                    prediction_method = "default"
                    calibration_applied = False
                    ensemble_used = False
                
                predictions_binary = (predictions_proba >= 0.5).astype(int)
                latency = (time.time() - start_time) * 1000
                
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
                        'postprocessing_applied': False,
                        'batch_latency_ms': latency,
                        'batch_index': i + j,
                        'recommendation': 'show' if prob > self.ctr_baseline else 'skip',
                        'timestamp': time.time()
                    }
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} prediction failed: {e}")
                
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
        
        logger.info(f"Batch CTR prediction completed: {len(results)} results")
        return results
    
    def predict_test_data(self, test_df: pd.DataFrame, 
                         model_name: Optional[str] = None,
                         use_calibration: bool = False,
                         use_ensemble: bool = True,
                         use_postprocessing: bool = False) -> np.ndarray:
        """Full test data prediction (for submission)"""
        logger.info(f"Test data prediction started: {len(test_df):,} rows")
        
        if not self.is_loaded:
            raise ValueError("Models not loaded")
        
        ensemble_available = use_ensemble and self.ensemble_manager is not None
        
        if model_name is None and not ensemble_available:
            model_name = list(self.models.keys())[0]
        
        if not ensemble_available and model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        logger.info(f"Method used: {'Ensemble' if ensemble_available else f'Single Model ({model_name})'}")
        
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
        
        logger.info(f"Preprocessing completed: {processed_df.shape}")
        
        memory_status = self.memory_monitor.get_memory_status()
        if memory_status['level'] in ['warning', 'critical']:
            batch_size = 30000
        else:
            batch_size = 80000
        
        predictions = []
        
        for i in range(0, len(processed_df), batch_size):
            end_idx = min(i + batch_size, len(processed_df))
            batch_df = processed_df.iloc[i:end_idx]
            
            try:
                ensemble_used = False
                
                if ensemble_available:
                    try:
                        batch_pred = self.ensemble_manager.predict_with_best_ensemble(batch_df)
                        ensemble_used = True
                        logger.debug(f"Batch {i//batch_size + 1}: Using ensemble prediction")
                    except Exception as e:
                        logger.warning(f"Batch ensemble prediction failed: {e}")
                        ensemble_available = False
                
                if not ensemble_available:
                    model = self.models[model_name]
                    
                    if hasattr(model, 'predict_proba_raw'):
                        batch_pred = model.predict_proba_raw(batch_df)
                    else:
                        batch_pred = model.predict_proba(batch_df)
                    
                    if use_calibration and hasattr(model, 'is_calibrated') and model.is_calibrated:
                        batch_pred = model.predict_proba(batch_df)
                
                batch_pred = np.clip(batch_pred, 0.0001, 0.9999)
                
                predictions.extend(batch_pred)
                
                logger.info(f"Batch {i//batch_size + 1} completed ({i:,}~{end_idx:,})")
                
            except Exception as e:
                logger.warning(f"Batch prediction failed: {e}")
                batch_pred = np.full(len(batch_df), self.ctr_baseline)
                predictions.extend(batch_pred)
            
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        predictions = np.array(predictions)
        
        logger.info(f"Test data prediction completed: {len(predictions):,}")
        return predictions
    
    def _update_average_latency(self, new_latency: float):
        """Update average latency"""
        current_count = self.inference_stats['successful_predictions']
        current_avg = self.inference_stats['average_latency_ms']
        
        new_avg = ((current_avg * (current_count - 1)) + new_latency) / current_count
        self.inference_stats['average_latency_ms'] = new_avg
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Return inference statistics"""
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
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """System status check"""
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
    """CTR prediction post-processing class - simplified"""
    
    def __init__(self):
        self.outlier_threshold_low = 0.0001
        self.outlier_threshold_high = 0.9999
        
    def apply_postprocessing(self, predictions: np.ndarray, 
                           target_ctr: float = 0.0191,
                           apply_outlier_clipping: bool = True) -> np.ndarray:
        """Apply simplified post-processing"""
        
        processed = predictions.copy()
        
        if apply_outlier_clipping:
            processed = np.clip(processed, self.outlier_threshold_low, self.outlier_threshold_high)
        
        return processed

class MemoryMonitor:
    """Memory monitoring class"""
    
    def __init__(self):
        self.monitoring_enabled = PSUTIL_AVAILABLE
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Return memory status"""
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
    """CTR prediction API"""
    
    def __init__(self, model_dir: str = "models"):
        self.engine = CTRInferenceEngine(model_dir)
        self.api_stats = {
            'api_calls': 0,
            'api_errors': 0,
            'start_time': time.time()
        }
    
    def initialize(self) -> bool:
        """Initialize CTR prediction API"""
        logger.info("CTR prediction API initialization started")
        
        success = self.engine.load_models()
        
        if success:
            health_check = self.engine.health_check()
            logger.info(f"CTR prediction API initialization completed")
            logger.info(f"Total models: {health_check['models_count']}")
            logger.info(f"Ensemble available: {health_check['ensemble_status']['available']}")
        else:
            logger.error("CTR prediction API initialization failed")
        
        return success
    
    def predict_ctr(self, user_id: str, ad_id: str, context: Dict[str, Any] = None, 
                   model_name: Optional[str] = None, 
                   use_calibration: bool = False,
                   use_ensemble: bool = True) -> Dict[str, Any]:
        """Main CTR prediction API"""
        self.api_stats['api_calls'] += 1
        
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
                False
            )
            
            api_response = {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': result['ctr_prediction'],
                'raw_prediction': result.get('raw_prediction', result['ctr_prediction']),
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'ctr_category': result['ctr_category'],
                'expected_revenue': result['ctr_prediction'] * context.get('bid_amount', 1.0) if context else result['ctr_prediction'],
                'model_info': {
                    'model_name': result['model_used'],
                    'version': 'v3.1'
                },
                'performance': {
                    'latency_ms': result['latency_ms'],
                    'timestamp': result['timestamp']
                }
            }
            
            return api_response
            
        except Exception as e:
            self.api_stats['api_errors'] += 1
            logger.error(f"CTR prediction API error: {e}")
            
            return {
                'user_id': user_id,
                'ad_id': ad_id,
                'ctr_prediction': self.engine.ctr_baseline,
                'raw_prediction': self.engine.ctr_baseline,
                'recommendation': 'skip',
                'confidence': 0.0,
                'ctr_category': 'low',
                'expected_revenue': 0.0,
                'error': str(e),
                'model_info': {
                    'model_name': 'unknown',
                    'version': 'v3.1'
                },
                'performance': {
                    'latency_ms': 0.0,
                    'timestamp': time.time()
                }
            }
    
    def predict_submission(self, test_df: pd.DataFrame, 
                          model_name: Optional[str] = None,
                          use_calibration: bool = False,
                          use_ensemble: bool = True) -> pd.DataFrame:
        """Generate submission predictions"""
        logger.info(f"Submission prediction generation started")
        
        predictions = self.engine.predict_test_data(
            test_df, 
            model_name, 
            use_calibration,
            use_ensemble,
            False
        )
        
        try:
            sample_submission = pd.read_csv('data/sample_submission.csv')
            if len(sample_submission) != len(predictions):
                logger.warning(f"Sample submission length ({len(sample_submission)}) != predictions length ({len(predictions)})")
                submission = pd.DataFrame({
                    'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                    'clicked': predictions
                })
            else:
                submission = pd.DataFrame({
                    'ID': sample_submission['ID'].values,
                    'clicked': predictions
                })
        except Exception as e:
            logger.warning(f"Could not load sample submission: {e}")
            submission = pd.DataFrame({
                'ID': [f"TEST_{i:07d}" for i in range(len(predictions))],
                'clicked': predictions
            })
        
        logger.info(f"Submission prediction generation completed: {len(submission):,} rows")
        
        avg_prediction = predictions.mean()
        logger.info(f"Average predicted CTR: {avg_prediction:.4f}")
        logger.info(f"Prediction range: {predictions.min():.4f} ~ {predictions.max():.4f}")
        
        return submission
    
    def get_api_status(self) -> Dict[str, Any]:
        """API status information"""
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

def create_ctr_prediction_service(model_dir: str = "models") -> CTRPredictionAPI:
    """Create CTR prediction service"""
    logger.info("CTR prediction service creation")
    
    api = CTRPredictionAPI(model_dir)
    
    if not api.initialize():
        logger.error("CTR prediction service initialization failed")
        return None
    
    logger.info("CTR prediction service creation completed")
    return api

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    service = create_ctr_prediction_service()
    
    if service:
        status = service.get_api_status()
        print("API status:", json.dumps(status, indent=2, default=str))
        
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
            use_calibration=False,
            use_ensemble=True
        )
        
        print("CTR prediction result:", json.dumps(test_prediction, indent=2, default=str))