# evaluation.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.metrics import (
    average_precision_score, log_loss, roc_auc_score, 
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

logger = logging.getLogger(__name__)

class CTRMetrics:
    """CTR 예측 전용 평가 지표 클래스"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        self.ap_weight = config.EVALUATION_CONFIG['ap_weight']
        self.wll_weight = config.EVALUATION_CONFIG['wll_weight']
        self.class_weight_ratio = config.EVALUATION_CONFIG['class_weight_ratio']
    
    def average_precision(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Average Precision (AP) 계산"""
        try:
            ap_score = average_precision_score(y_true, y_pred_proba)
            return ap_score
        except Exception as e:
            logger.error(f"AP 계산 오류: {str(e)}")
            return 0.0
    
    def weighted_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Weighted Log Loss (WLL) 계산 - 클래스 균형화"""
        try:
            # 클래스 가중치 계산
            pos_weight = self.class_weight_ratio
            neg_weight = 1 - self.class_weight_ratio
            
            # 각 샘플별 가중치 적용
            sample_weights = np.where(y_true == 1, pos_weight, neg_weight)
            
            # 로그 손실 계산 (클립핑으로 수치 안정성 확보)
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            
            # 가중 로그 손실 직접 계산
            log_loss_values = -(y_true * np.log(y_pred_proba_clipped) + 
                              (1 - y_true) * np.log(1 - y_pred_proba_clipped))
            
            weighted_log_loss = np.average(log_loss_values, weights=sample_weights)
            
            return weighted_log_loss
            
        except Exception as e:
            logger.error(f"WLL 계산 오류: {str(e)}")
            return float('inf')
    
    def combined_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Combined Score = 0.5 * AP + 0.5 * (1/(1+WLL))"""
        try:
            ap_score = self.average_precision(y_true, y_pred_proba)
            wll_score = self.weighted_log_loss(y_true, y_pred_proba)
            
            # WLL을 0-1 스케일로 변환 (1/(1+WLL))
            wll_normalized = 1 / (1 + wll_score)
            
            # 최종 점수 계산
            combined = self.ap_weight * ap_score + self.wll_weight * wll_normalized
            
            return combined
            
        except Exception as e:
            logger.error(f"Combined Score 계산 오류: {str(e)}")
            return 0.0
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray, 
                               y_pred_proba: np.ndarray,
                               threshold: float = 0.5) -> Dict[str, float]:
        """종합적인 평가 지표 계산"""
        
        # 이진 예측
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {}
        
        try:
            # 대회 평가 지표
            metrics['ap'] = self.average_precision(y_true, y_pred_proba)
            metrics['wll'] = self.weighted_log_loss(y_true, y_pred_proba)
            metrics['combined_score'] = self.combined_score(y_true, y_pred_proba)
            
            # 기본 분류 지표
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            # 혼동 행렬 기반 지표
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # F1 Score
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0.0
            
            # CTR 관련 지표
            metrics['ctr_actual'] = y_true.mean()
            metrics['ctr_predicted'] = y_pred_proba.mean()
            metrics['ctr_bias'] = metrics['ctr_predicted'] - metrics['ctr_actual']
            
            # 분포 관련 지표
            metrics['prediction_std'] = y_pred_proba.std()
            metrics['prediction_entropy'] = self._calculate_entropy(y_pred_proba)
            
        except Exception as e:
            logger.error(f"종합 평가 계산 오류: {str(e)}")
        
        return metrics
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """예측 확률의 엔트로피 계산"""
        try:
            # 확률 클립핑
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            
            # 엔트로피 계산
            entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            
            return entropy
        except:
            return 0.0
    
    def calculate_pr_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Precision-Recall AUC 계산"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = np.trapz(precision, recall)
            
            return pr_auc, precision, recall
        except Exception as e:
            logger.error(f"PR-AUC 계산 오류: {str(e)}")
            return 0.0, np.array([]), np.array([])
    
    def find_optimal_threshold(self, 
                             y_true: np.ndarray, 
                             y_pred_proba: np.ndarray,
                             metric: str = 'f1') -> Tuple[float, float]:
        """최적 임계값 찾기"""
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            elif metric == 'accuracy':
                score = (y_true == y_pred).mean()
            
            elif metric == 'combined':
                score = self.combined_score(y_true, y_pred_proba)
            
            else:
                continue
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score

class ModelComparator:
    """다중 모델 비교 클래스"""
    
    def __init__(self):
        self.metrics_calculator = CTRMetrics()
        self.comparison_results = {}
    
    def compare_models(self, 
                      models_predictions: Dict[str, np.ndarray],
                      y_true: np.ndarray) -> pd.DataFrame:
        """여러 모델의 성능 비교"""
        
        results = []
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                metrics = self.metrics_calculator.comprehensive_evaluation(y_true, y_pred_proba)
                metrics['model_name'] = model_name
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"{model_name} 모델 평가 실패: {str(e)}")
        
        # DataFrame으로 변환
        comparison_df = pd.DataFrame(results)
        
        if not comparison_df.empty:
            # 모델명을 인덱스로 설정
            comparison_df.set_index('model_name', inplace=True)
            
            # 주요 지표로 정렬
            comparison_df.sort_values('combined_score', ascending=False, inplace=True)
        
        self.comparison_results = comparison_df
        
        return comparison_df
    
    def rank_models(self, 
                   ranking_metric: str = 'combined_score') -> pd.DataFrame:
        """모델 순위 매기기"""
        
        if self.comparison_results.empty:
            logger.warning("비교 결과가 없습니다.")
            return pd.DataFrame()
        
        ranking_df = self.comparison_results.copy()
        ranking_df['rank'] = ranking_df[ranking_metric].rank(ascending=False)
        ranking_df.sort_values('rank', inplace=True)
        
        return ranking_df[['rank', ranking_metric, 'ap', 'wll', 'auc', 'f1']]
    
    def get_best_model(self, metric: str = 'combined_score') -> Tuple[str, float]:
        """최고 성능 모델 반환"""
        
        if self.comparison_results.empty:
            return None, 0.0
        
        best_idx = self.comparison_results[metric].idxmax()
        best_score = self.comparison_results.loc[best_idx, metric]
        
        return best_idx, best_score

class EvaluationVisualizer:
    """평가 결과 시각화 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_roc_curves(self, 
                       models_predictions: Dict[str, np.ndarray],
                       y_true: np.ndarray,
                       save_path: Optional[str] = None):
        """ROC 곡선 시각화"""
        
        plt.figure(figsize=self.figsize)
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC: {auc_score:.3f})')
                
            except Exception as e:
                logger.error(f"{model_name} ROC 곡선 그리기 실패: {str(e)}")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self,
                                    models_predictions: Dict[str, np.ndarray],
                                    y_true: np.ndarray,
                                    save_path: Optional[str] = None):
        """Precision-Recall 곡선 시각화"""
        
        plt.figure(figsize=self.figsize)
        
        metrics_calc = CTRMetrics()
        
        for model_name, y_pred_proba in models_predictions.items():
            try:
                pr_auc, precision, recall = metrics_calc.calculate_pr_auc(y_true, y_pred_proba)
                
                plt.plot(recall, precision, label=f'{model_name} (PR-AUC: {pr_auc:.3f})')
                
            except Exception as e:
                logger.error(f"{model_name} PR 곡선 그리기 실패: {str(e)}")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self,
                            comparison_df: pd.DataFrame,
                            metrics: List[str] = None,
                            save_path: Optional[str] = None):
        """모델 성능 비교 차트"""
        
        if metrics is None:
            metrics = ['combined_score', 'ap', 'auc', 'f1']
        
        # 사용 가능한 지표만 필터링
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            logger.warning("시각화할 지표가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            if i < len(axes):
                ax = axes[i]
                
                comparison_df[metric].plot(kind='bar', ax=ax)
                ax.set_title(f'{metric.upper()} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        # 빈 서브플롯 숨기기
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_distribution(self,
                                   y_true: np.ndarray,
                                   y_pred_proba: np.ndarray,
                                   model_name: str = "Model",
                                   save_path: Optional[str] = None):
        """예측 확률 분포 시각화"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 클래스별 예측 확률 분포
        axes[0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Not Clicked', density=True)
        axes[0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Clicked', density=True)
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{model_name} - Prediction Distribution by Class')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 전체 예측 확률 분포
        axes[1].hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(y_pred_proba.mean(), color='red', linestyle='--', label=f'Mean: {y_pred_proba.mean():.3f}')
        axes[1].axvline(y_true.mean(), color='green', linestyle='--', label=f'Actual CTR: {y_true.mean():.3f}')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Overall Prediction Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class EvaluationReporter:
    """평가 보고서 생성 클래스"""
    
    def __init__(self):
        self.metrics_calculator = CTRMetrics()
        self.visualizer = EvaluationVisualizer()
    
    def generate_comprehensive_report(self,
                                    models_predictions: Dict[str, np.ndarray],
                                    y_true: np.ndarray,
                                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """종합 평가 보고서 생성"""
        
        logger.info("종합 평가 보고서 생성 시작")
        
        # 모델 비교
        comparator = ModelComparator()
        comparison_df = comparator.compare_models(models_predictions, y_true)
        
        # 최고 성능 모델
        best_model, best_score = comparator.get_best_model()
        
        # 보고서 데이터 구성
        report = {
            'summary': {
                'total_models': len(models_predictions),
                'best_model': best_model,
                'best_score': best_score,
                'data_size': len(y_true),
                'actual_ctr': y_true.mean()
            },
            'detailed_comparison': comparison_df.to_dict(),
            'model_rankings': comparator.rank_models().to_dict()
        }
        
        # 시각화 생성 (출력 디렉터리가 지정된 경우)
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # ROC 곡선
            self.visualizer.plot_roc_curves(
                models_predictions, y_true,
                save_path=f"{output_dir}/roc_curves.png"
            )
            
            # PR 곡선
            self.visualizer.plot_precision_recall_curves(
                models_predictions, y_true,
                save_path=f"{output_dir}/pr_curves.png"
            )
            
            # 모델 비교
            self.visualizer.plot_model_comparison(
                comparison_df,
                save_path=f"{output_dir}/model_comparison.png"
            )
            
            # 최고 모델의 예측 분포
            if best_model and best_model in models_predictions:
                self.visualizer.plot_prediction_distribution(
                    y_true, models_predictions[best_model], best_model,
                    save_path=f"{output_dir}/prediction_distribution.png"
                )
        
        logger.info("종합 평가 보고서 생성 완료")
        
        return report