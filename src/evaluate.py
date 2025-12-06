"""
Evaluation Module: Compare baseline and transformer models
Generate comprehensive evaluation metrics and visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class ModelEvaluator:
    """
    Evaluate and compare different models
    """
    
    def __init__(self):
        self.label_names = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
    def compare_models(self, baseline_metrics: dict, transformer_metrics: dict):
        """
        Compare baseline and transformer models
        
        Args:
            baseline_metrics: Metrics from baseline model
            transformer_metrics: Metrics from transformer model
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Create comparison DataFrame
        metrics_to_compare = ['accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall']
        
        comparison_data = {
            'Metric': [],
            'Baseline': [],
            'Transformer': [],
            'Improvement': []
        }
        
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics.get(metric, 0)
            transformer_val = transformer_metrics.get(metric, 0)
            improvement = ((transformer_val - baseline_val) / baseline_val) * 100
            
            comparison_data['Metric'].append(metric.replace('_', ' ').title())
            comparison_data['Baseline'].append(f"{baseline_val:.4f}")
            comparison_data['Transformer'].append(f"{transformer_val:.4f}")
            comparison_data['Improvement'].append(f"{improvement:+.2f}%")
        
        df = pd.DataFrame(comparison_data)
        print("\n", df.to_string(index=False))
        
        return df
    
    def plot_model_comparison(self, baseline_metrics: dict, transformer_metrics: dict,
                             save_path: str = 'figures/model_comparison.png'):
        """Plot side-by-side comparison of models"""
        metrics_to_plot = ['accuracy', 'f1_macro', 'precision', 'recall']
        
        baseline_vals = [baseline_metrics[m] for m in metrics_to_plot]
        transformer_vals = [transformer_metrics[m] for m in metrics_to_plot]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (TF-IDF + LogReg)',
                      color='skyblue', edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, transformer_vals, width, label='Transformer (DistilBERT)',
                      color='lightgreen', edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison plot to {save_path}")
        
        plt.show()
        
        return save_path
    
    def plot_per_class_f1(self, y_true, y_pred_baseline, y_pred_transformer,
                          save_path: str = 'figures/per_class_f1.png'):
        """Plot F1-scores per class for both models"""
        from sklearn.metrics import f1_score
        
        classes = list(self.label_names.values())
        
        # Calculate per-class F1 scores
        baseline_f1 = f1_score(y_true, y_pred_baseline, average=None)
        transformer_f1 = f1_score(y_true, y_pred_transformer, average=None)
        
        x = np.arange(len(classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline',
                      color='coral', edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, transformer_f1, width, label='Transformer',
                      color='mediumseagreen', edgecolor='black', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved per-class F1 plot to {save_path}")
        
        plt.show()
        
        return save_path
    
    def plot_confusion_matrices_comparison(self, cm_baseline, cm_transformer,
                                          save_path: str = 'figures/confusion_matrices_comparison.png'):
        """Plot confusion matrices side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Baseline confusion matrix
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names.values(),
                   yticklabels=self.label_names.values(),
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Baseline Model\n(TF-IDF + Logistic Regression)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)
        
        # Transformer confusion matrix
        sns.heatmap(cm_transformer, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.label_names.values(),
                   yticklabels=self.label_names.values(),
                   ax=axes[1], cbar_kws={'label': 'Count'})
        axes[1].set_title('Transformer Model\n(DistilBERT)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)
        
        plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrices comparison to {save_path}")
        
        plt.show()
        
        return save_path
    
    def analyze_misclassifications(self, texts, y_true, y_pred, n_samples: int = 5):
        """Analyze and display misclassified samples"""
        print("\n" + "="*70)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*70)
        
        # Find misclassified indices
        misclassified_idx = np.where(np.array(y_true) != np.array(y_pred))[0]
        
        print(f"\nTotal misclassifications: {len(misclassified_idx)}")
        print(f"Misclassification rate: {len(misclassified_idx)/len(y_true)*100:.2f}%")
        
        # Sample some misclassifications
        sample_idx = np.random.choice(misclassified_idx, min(n_samples, len(misclassified_idx)), replace=False)
        
        print(f"\nSample misclassifications:")
        print("-" * 70)
        
        for i, idx in enumerate(sample_idx, 1):
            true_label = self.label_names[y_true[idx]]
            pred_label = self.label_names[y_pred[idx]]
            text = texts[idx][:200] + "..."
            
            print(f"\nExample {i}:")
            print(f"Text: {text}")
            print(f"True Label: {true_label}")
            print(f"Predicted Label: {pred_label}")
            print("-" * 70)
    
    def generate_full_report(self, baseline_metrics, transformer_metrics,
                            cm_baseline, cm_transformer,
                            y_true, y_pred_baseline, y_pred_transformer):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*70)
        
        # Model comparison
        comparison_df = self.compare_models(baseline_metrics, transformer_metrics)
        
        # Plot comparisons
        self.plot_model_comparison(baseline_metrics, transformer_metrics)
        self.plot_per_class_f1(y_true, y_pred_baseline, y_pred_transformer)
        self.plot_confusion_matrices_comparison(cm_baseline, cm_transformer)
        
        print("\n" + "="*70)
        print("EVALUATION REPORT COMPLETE")
        print("="*70)
        print("\n✓ All visualizations saved to 'figures/' directory")
        print("✓ Check MLflow UI for detailed experiment tracking")


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*70)
    
    evaluator = ModelEvaluator()
    
    # Example metrics (replace with actual metrics from MLflow)
    baseline_metrics = {
        'accuracy': 0.8750,
        'f1_macro': 0.8745,
        'f1_micro': 0.8750,
        'precision': 0.8751,
        'recall': 0.8745
    }
    
    transformer_metrics = {
        'accuracy': 0.9380,
        'f1_macro': 0.9379,
        'f1_micro': 0.9380,
        'precision': 0.9381,
        'recall': 0.9379
    }
    
    # Generate comparison report
    evaluator.compare_models(baseline_metrics, transformer_metrics)
    evaluator.plot_model_comparison(baseline_metrics, transformer_metrics)
    
    print("\n✓ To run full evaluation, train both models first:")
    print("  1. python src/train_baseline.py")
    print("  2. python src/train_transformer.py")
    print("  3. Then use MLflow to compare results")


if __name__ == "__main__":
    main()
