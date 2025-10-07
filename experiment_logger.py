# experiment_logger.py

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """Experiment parameter and result logging system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.experiment_log_path = self.log_dir / "experiment_log.txt"
        self.experiment_counter_path = self.log_dir / "experiment_counter.json"
        
        self.current_experiment_number = self._load_experiment_counter()
        
        logger.info(f"Experiment logger initialized: {self.experiment_log_path}")
    
    def _load_experiment_counter(self) -> int:
        """Load experiment counter"""
        try:
            if self.experiment_counter_path.exists():
                with open(self.experiment_counter_path, 'r') as f:
                    data = json.load(f)
                    return data.get('counter', 0)
            return 0
        except Exception as e:
            logger.warning(f"Failed to load experiment counter: {e}")
            return 0
    
    def _save_experiment_counter(self):
        """Save experiment counter"""
        try:
            with open(self.experiment_counter_path, 'w') as f:
                json.dump({'counter': self.current_experiment_number}, f)
        except Exception as e:
            logger.warning(f"Failed to save experiment counter: {e}")
    
    def log_experiment(self, 
                      config: Any,
                      results: Optional[Dict[str, Any]] = None,
                      model_name: str = "Unknown",
                      notes: str = ""):
        """Log experiment parameters and results"""
        try:
            self.current_experiment_number += 1
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = []
            log_entry.append(f"\n{'='*70}")
            log_entry.append(f"[{timestamp}] Experiment #{self.current_experiment_number}")
            log_entry.append(f"{'='*70}")
            
            log_entry.append(f"\nModel: {model_name}")
            
            if notes:
                log_entry.append(f"Notes: {notes}")
            
            log_entry.append("\n--- Key Parameters ---")
            params = self._extract_key_parameters(config, model_name)
            for param_name, param_value in params.items():
                log_entry.append(f"  {param_name}: {param_value}")
            
            if results:
                log_entry.append("\n--- Results ---")
                
                if 'model_performances' in results and results['model_performances']:
                    for model, perf in results['model_performances'].items():
                        log_entry.append(f"\n  [{model}]")
                        log_entry.append(f"    AUC: {perf.get('auc', 0.0):.4f}")
                        log_entry.append(f"    AP: {perf.get('ap', 0.0):.4f}")
                        log_entry.append(f"    Actual CTR: {perf.get('actual_ctr', 0.0):.6f}")
                        log_entry.append(f"    Predicted CTR: {perf.get('predicted_ctr', 0.0):.6f}")
                        log_entry.append(f"    CTR Bias: {perf.get('ctr_bias', 0.0):.6f}")
                
                if 'prediction_stats' in results:
                    stats = results['prediction_stats']
                    log_entry.append("\n  [Prediction Stats]")
                    log_entry.append(f"    Mean: {stats.get('mean', 0.0):.6f}")
                    log_entry.append(f"    Std: {stats.get('std', 0.0):.6f}")
                    log_entry.append(f"    Min: {stats.get('min', 0.0):.6f}")
                    log_entry.append(f"    Max: {stats.get('max', 0.0):.6f}")
                
                if 'execution_time' in results:
                    log_entry.append(f"\n  Execution Time: {results['execution_time']:.1f}s")
                
                if 'successful_models' in results:
                    log_entry.append(f"  Successful Models: {results['successful_models']}")
                
                if 'ensemble_used' in results:
                    log_entry.append(f"  Ensemble Used: {'Yes' if results['ensemble_used'] else 'No'}")
                
                if 'gpu_used' in results:
                    log_entry.append(f"  GPU Used: {'Yes' if results['gpu_used'] else 'No'}")
            
            log_entry.append(f"\n{'='*70}\n")
            
            with open(self.experiment_log_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_entry))
            
            self._save_experiment_counter()
            
            logger.info(f"Experiment #{self.current_experiment_number} logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log experiment: {e}")
    
    def _extract_key_parameters(self, config: Any, model_name: str) -> Dict[str, Any]:
        """Extract key parameters from config"""
        params = {}
        
        try:
            if model_name.lower() in ['xgboost', 'xgboost_gpu']:
                xgb_config = config.MODEL_TRAINING_CONFIG.get('xgboost_gpu', {})
                params['tree_method'] = xgb_config.get('tree_method', 'hist')
                params['max_depth'] = xgb_config.get('max_depth', 6)
                params['learning_rate'] = xgb_config.get('learning_rate', 0.05)
                params['subsample'] = xgb_config.get('subsample', 0.9)
                params['colsample_bytree'] = xgb_config.get('colsample_bytree', 0.8)
                params['scale_pos_weight'] = xgb_config.get('scale_pos_weight', 1.0)
                params['min_child_weight'] = xgb_config.get('min_child_weight', 10)
                params['gamma'] = xgb_config.get('gamma', 0.1)
                params['reg_alpha'] = xgb_config.get('reg_alpha', 0.05)
                params['reg_lambda'] = xgb_config.get('reg_lambda', 1.5)
                params['max_bin'] = xgb_config.get('max_bin', 256)
            
            elif model_name.lower() == 'logistic':
                log_config = config.MODEL_TRAINING_CONFIG.get('logistic', {})
                params['C'] = log_config.get('C', 0.5)
                params['penalty'] = log_config.get('penalty', 'l2')
                params['solver'] = log_config.get('solver', 'saga')
                params['max_iter'] = log_config.get('max_iter', 100)
            
            params['cv_folds'] = config.CV_FOLDS
            params['use_cross_validation'] = config.USE_CROSS_VALIDATION
            params['calibration_method'] = config.CALIBRATION_METHOD
            params['calibration_mandatory'] = config.CALIBRATION_MANDATORY
            
            params['target_combined_score'] = config.TARGET_COMBINED_SCORE
            params['target_ctr'] = config.TARGET_CTR
            
        except Exception as e:
            logger.warning(f"Failed to extract parameters: {e}")
        
        return params
    
    def add_note_to_last_experiment(self, note: str):
        """Add note to last experiment"""
        try:
            with open(self.experiment_log_path, 'a', encoding='utf-8') as f:
                f.write(f"Additional Note: {note}\n")
            
            logger.info("Note added to last experiment")
            
        except Exception as e:
            logger.error(f"Failed to add note: {e}")
    
    def get_experiment_summary(self, last_n: int = 5) -> str:
        """Get summary of last N experiments"""
        try:
            if not self.experiment_log_path.exists():
                return "No experiments logged yet"
            
            with open(self.experiment_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            experiments = content.split('='*70)
            recent_experiments = experiments[-last_n*2:] if len(experiments) > last_n*2 else experiments
            
            return '='*70 + ''.join(recent_experiments)
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return f"Error: {e}"
    
    def clear_log(self):
        """Clear experiment log"""
        try:
            if self.experiment_log_path.exists():
                self.experiment_log_path.unlink()
            
            self.current_experiment_number = 0
            self._save_experiment_counter()
            
            logger.info("Experiment log cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear log: {e}")

def create_experiment_logger(log_dir: str = "logs") -> ExperimentLogger:
    """Create experiment logger instance"""
    return ExperimentLogger(log_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    exp_logger = ExperimentLogger()
    
    class MockConfig:
        MODEL_TRAINING_CONFIG = {
            'xgboost_gpu': {
                'tree_method': 'gpu_hist',
                'max_depth': 6,
                'learning_rate': 0.05,
                'scale_pos_weight': 51.43
            }
        }
        CV_FOLDS = 5
        USE_CROSS_VALIDATION = True
        CALIBRATION_METHOD = 'isotonic'
        CALIBRATION_MANDATORY = True
        TARGET_COMBINED_SCORE = 0.35
        TARGET_CTR = 0.0191
    
    test_results = {
        'model_performances': {
            'xgboost_gpu': {
                'auc': 0.7856,
                'ap': 0.0812,
                'actual_ctr': 0.0191,
                'predicted_ctr': 0.0188,
                'ctr_bias': -0.0003
            }
        },
        'execution_time': 125.3,
        'successful_models': 1,
        'ensemble_used': False,
        'gpu_used': True
    }
    
    exp_logger.log_experiment(
        config=MockConfig(),
        results=test_results,
        model_name="xgboost_gpu",
        notes="Testing parameter adjustment"
    )
    
    print("Test experiment logged")
    print("\nLast 5 experiments:")
    print(exp_logger.get_experiment_summary(5))