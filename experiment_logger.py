# experiment_logger.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from config import Config

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """Experiment parameters and results logging system"""
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Config.EXPERIMENTS_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_count = self._get_experiment_count()
        
    def _get_experiment_count(self) -> int:
        """Get current experiment count from log file"""
        if not self.log_path.exists():
            return 0
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                count = content.count('Experiment #')
                return count
        except Exception as e:
            logger.warning(f"Failed to read experiment count: {e}")
            return 0
    
    def log_experiment(self, 
                      model_name: str,
                      parameters: Dict[str, Any],
                      results: Dict[str, Any],
                      execution_info: Dict[str, Any],
                      notes: str = "Mode: Full") -> None:
        """Log experiment with parameters and results"""
        
        self.experiment_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = self._format_log_entry(
            experiment_num=self.experiment_count,
            timestamp=timestamp,
            model_name=model_name,
            parameters=parameters,
            results=results,
            execution_info=execution_info,
            notes=notes
        )
        
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            logger.info(f"Experiment #{self.experiment_count} logged successfully")
        except Exception as e:
            logger.error(f"Failed to log experiment: {e}")
    
    def _format_log_entry(self,
                         experiment_num: int,
                         timestamp: str,
                         model_name: str,
                         parameters: Dict[str, Any],
                         results: Dict[str, Any],
                         execution_info: Dict[str, Any],
                         notes: str) -> str:
        """Format log entry with detailed information"""
        
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append(f"[{timestamp}] Experiment #{experiment_num}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Model: {model_name}")
        lines.append(f"Notes: {notes}")
        lines.append("")
        
        lines.append("--- Key Parameters ---")
        for key, value in sorted(parameters.items()):
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        lines.append("--- Results ---")
        lines.append("")
        
        if 'model_performances' in results and results['model_performances']:
            for model_name, perf in results['model_performances'].items():
                lines.append(f"  [{model_name}]")
                if 'auc' in perf:
                    lines.append(f"    AUC: {perf['auc']:.4f}")
                if 'ap' in perf:
                    lines.append(f"    AP: {perf['ap']:.4f}")
                if 'actual_ctr' in perf:
                    lines.append(f"    Actual CTR: {perf['actual_ctr']:.6f}")
                if 'predicted_ctr' in perf:
                    lines.append(f"    Predicted CTR: {perf['predicted_ctr']:.6f}")
                if 'ctr_bias' in perf:
                    lines.append(f"    CTR Bias: {perf['ctr_bias']:.6f}")
                lines.append("")
        
        if 'prediction_stats' in results:
            pred_stats = results['prediction_stats']
            lines.append(f"  [Prediction Stats]")
            if 'mean' in pred_stats:
                lines.append(f"    Mean: {pred_stats['mean']:.6f}")
            if 'std' in pred_stats:
                lines.append(f"    Std: {pred_stats['std']:.6f}")
            if 'min' in pred_stats:
                lines.append(f"    Min: {pred_stats['min']:.6f}")
            if 'max' in pred_stats:
                lines.append(f"    Max: {pred_stats['max']:.6f}")
            lines.append("")
        
        if execution_info:
            if 'execution_time' in execution_info:
                lines.append(f"  Execution Time: {execution_info['execution_time']:.1f}s")
            if 'successful_models' in execution_info:
                lines.append(f"  Successful Models: {execution_info['successful_models']}")
            if 'ensemble_used' in execution_info:
                lines.append(f"  Ensemble Used: {'Yes' if execution_info['ensemble_used'] else 'No'}")
            if 'gpu_used' in execution_info:
                lines.append(f"  GPU Used: {'Yes' if execution_info['gpu_used'] else 'No'}")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("")
        
        return "\n".join(lines)
    
    def get_best_experiment(self) -> Optional[Dict[str, Any]]:
        """Get best experiment from log file"""
        if not self.log_path.exists():
            return None
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            experiments = content.split("=" * 70)
            best_score = 0.0
            best_experiment = None
            
            for exp in experiments:
                if 'AUC:' in exp:
                    try:
                        auc_line = [line for line in exp.split('\n') if 'AUC:' in line][0]
                        auc_score = float(auc_line.split('AUC:')[1].strip())
                        if auc_score > best_score:
                            best_score = auc_score
                            best_experiment = exp
                    except:
                        continue
            
            return {'score': best_score, 'details': best_experiment} if best_experiment else None
            
        except Exception as e:
            logger.error(f"Failed to get best experiment: {e}")
            return None
    
    def get_recent_experiments(self, count: int = 5) -> List[str]:
        """Get recent experiments from log file"""
        if not self.log_path.exists():
            return []
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            experiments = [exp for exp in content.split("=" * 70) if 'Experiment #' in exp]
            return experiments[-count:] if experiments else []
            
        except Exception as e:
            logger.error(f"Failed to get recent experiments: {e}")
            return []
    
    def clear_log(self) -> bool:
        """Clear experiment log file"""
        try:
            if self.log_path.exists():
                self.log_path.unlink()
            self.experiment_count = 0
            logger.info("Experiment log cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear experiment log: {e}")
            return False
    
    def export_to_json(self, output_path: Optional[Path] = None) -> bool:
        """Export experiments to JSON format"""
        if not self.log_path.exists():
            logger.warning("No experiment log to export")
            return False
        
        output_path = output_path or self.log_path.parent / "experiments.json"
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            experiments = []
            for exp in content.split("=" * 70):
                if 'Experiment #' in exp:
                    exp_dict = self._parse_experiment_text(exp)
                    if exp_dict:
                        experiments.append(exp_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(experiments, f, indent=2)
            
            logger.info(f"Experiments exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export experiments: {e}")
            return False
    
    def _parse_experiment_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse experiment text to dictionary"""
        try:
            lines = text.strip().split('\n')
            exp_dict = {}
            
            for line in lines:
                if 'Experiment #' in line:
                    exp_dict['experiment_num'] = int(line.split('#')[1].strip(']'))
                    exp_dict['timestamp'] = line.split('[')[1].split(']')[0]
                elif 'Model:' in line:
                    exp_dict['model'] = line.split(':')[1].strip()
                elif 'AUC:' in line:
                    exp_dict['auc'] = float(line.split(':')[1].strip())
                elif 'AP:' in line:
                    exp_dict['ap'] = float(line.split(':')[1].strip())
                elif 'Execution Time:' in line:
                    exp_dict['execution_time'] = float(line.split(':')[1].strip().replace('s', ''))
            
            return exp_dict if exp_dict else None
            
        except Exception as e:
            logger.warning(f"Failed to parse experiment: {e}")
            return None

def log_training_experiment(model_name: str,
                           config: Config,
                           results: Dict[str, Any],
                           execution_time: float,
                           quick_mode: bool = False) -> None:
    """Helper function to log training experiment"""
    
    experiment_logger = ExperimentLogger()
    
    parameters = {
        'cv_folds': config.CV_FOLDS,
        'use_cross_validation': True,
        'calibration_method': config.CALIBRATION_CONFIG.get('methods', ['isotonic'])[0],
        'calibration_mandatory': config.CALIBRATION_CONFIG.get('enabled', False),
        'target_combined_score': config.TARGET_COMBINED_SCORE,
        'target_ctr': config.TARGET_CTR
    }
    
    if model_name in config.MODEL_TRAINING_CONFIG:
        model_params = config.MODEL_TRAINING_CONFIG[model_name]
        parameters.update(model_params)
    
    execution_info = {
        'execution_time': execution_time,
        'successful_models': results.get('successful_models', 0),
        'ensemble_used': results.get('ensemble_used', False),
        'gpu_used': config.GPU_AVAILABLE
    }
    
    notes = "Mode: Quick (50 samples)" if quick_mode else "Mode: Full"
    
    experiment_logger.log_experiment(
        model_name=model_name,
        parameters=parameters,
        results=results,
        execution_info=execution_info,
        notes=notes
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Experiment Logger Test ===")
    
    test_logger = ExperimentLogger()
    
    test_params = {
        'C': 0.5,
        'penalty': 'l2',
        'solver': 'saga',
        'max_iter': 100,
        'cv_folds': 5
    }
    
    test_results = {
        'model_performances': {
            'logistic': {
                'auc': 0.5358,
                'ap': 0.0217,
                'actual_ctr': 0.019075,
                'predicted_ctr': 0.019075,
                'ctr_bias': 0.000000
            },
            'xgboost_gpu': {
                'auc': 0.7393,
                'ap': 0.0787,
                'actual_ctr': 0.019075,
                'predicted_ctr': 0.019075,
                'ctr_bias': 0.000000
            }
        },
        'prediction_stats': {
            'mean': 0.241058,
            'std': 0.074164,
            'min': 0.162903,
            'max': 0.905850
        }
    }
    
    test_execution = {
        'execution_time': 331.5,
        'successful_models': 2,
        'ensemble_used': True,
        'gpu_used': True
    }
    
    test_logger.log_experiment(
        model_name='logistic',
        parameters=test_params,
        results=test_results,
        execution_info=test_execution,
        notes='Mode: Full'
    )
    
    print("Test log entry created successfully")
    print(f"Log file: {test_logger.log_path}")
    
    best = test_logger.get_best_experiment()
    if best:
        print(f"\nBest experiment score: {best['score']:.4f}")