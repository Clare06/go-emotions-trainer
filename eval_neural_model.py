import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import os
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SarcasmEvaluationMetrics:
    """🎯 Comprehensive Evaluation Metrics for Sarcasm Detection"""

    def __init__(self, model, scaler, feature_cols, device):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device
        self.class_names = ['Non-Sarcastic', 'Sarcastic']

    def evaluate_comprehensive(self, test_loader, save_dir="evaluation_results"):
        """🔍 Complete evaluation with all metrics and visualizations"""

        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{save_dir}_{timestamp}"
        os.makedirs(save_path, exist_ok=True)

        print(f"🎯 Starting comprehensive evaluation...")
        print(f"📁 Results will be saved to: {save_path}")

        # Get predictions and probabilities
        results = self._get_predictions(test_loader)

        # Calculate all metrics
        metrics = self._calculate_metrics(results)

        # Generate all visualizations
        self._create_visualizations(results, metrics, save_path)

        # Save detailed results
        self._save_results(results, metrics, save_path)

        return metrics, save_path

    def _get_predictions(self, test_loader):
        """Get model predictions and probabilities"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs)
        }

    def _calculate_metrics(self, results):
        """Calculate comprehensive metrics"""
        y_true = results['labels']
        y_pred = results['predictions']
        y_probs = results['probabilities']

        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_binary': f1_score(y_true, y_pred, average='binary'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),

            # Per-class metrics
            'precision_per_class': precision_score(y_true, y_pred, average=None),
            'recall_per_class': recall_score(y_true, y_pred, average=None),
            'f1_per_class': f1_score(y_true, y_pred, average=None),

            # ROC and PR metrics
            'roc_auc': auc(*roc_curve(y_true, y_probs[:, 1])[:2]),
            'avg_precision': average_precision_score(y_true, y_probs[:, 1]),

            # Confusion matrix
            'confusion_matrix': confusion_matrix(y_true, y_pred),

            # Classification report
            'classification_report': classification_report(y_true, y_pred,
                                                           target_names=self.class_names,
                                                           output_dict=True)
        }

        return metrics

    def _create_visualizations(self, results, metrics, save_path):
        """Create all visualization plots"""

        # 1. Confusion Matrix
        self._plot_confusion_matrix(metrics['confusion_matrix'], save_path)

        # 2. ROC Curve
        self._plot_roc_curve(results, save_path)

        # 3. Precision-Recall Curve
        self._plot_precision_recall_curve(results, save_path)

        # 4. Class Distribution
        self._plot_class_distribution(results, save_path)

        # 5. Probability Distribution
        self._plot_probability_distribution(results, save_path)

        # 6. Metrics Summary
        self._plot_metrics_summary(metrics, save_path)

        # 7. Per-Class Performance
        self._plot_per_class_performance(metrics, save_path)

        # 8. Error Analysis
        self._plot_error_analysis(results, save_path)

    def _plot_confusion_matrix(self, cm, save_path):
        """Plot confusion matrix with percentages"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=self.class_names, yticklabels=self.class_names)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)

        # Percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                    xticklabels=self.class_names, yticklabels=self.class_names)
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_roc_curve(self, results, save_path):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'][:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3,
                 label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve',
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_path}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_precision_recall_curve(self, results, save_path):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(results['labels'],
                                                      results['probabilities'][:, 1])
        avg_precision = average_precision_score(results['labels'],
                                                results['probabilities'][:, 1])

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=3,
                 label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_path}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_class_distribution(self, results, save_path):
        """Plot class distribution in predictions vs true labels"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # True labels distribution
        true_counts = np.bincount(results['labels'])
        ax1.bar(self.class_names, true_counts, color=['skyblue', 'lightcoral'])
        ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12)
        for i, v in enumerate(true_counts):
            ax1.text(i, v + 1, str(v), ha='center', fontweight='bold')

        # Predicted labels distribution
        pred_counts = np.bincount(results['predictions'])
        ax2.bar(self.class_names, pred_counts, color=['lightgreen', 'orange'])
        ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        for i, v in enumerate(pred_counts):
            ax2.text(i, v + 1, str(v), ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{save_path}/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_probability_distribution(self, results, save_path):
        """Plot probability distributions for each class"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Sarcastic probability for true sarcastic samples
        sarc_true_probs = results['probabilities'][results['labels'] == 1, 1]
        axes[0, 0].hist(sarc_true_probs, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_title('Sarcastic Probability\n(True Sarcastic Samples)', fontweight='bold')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(sarc_true_probs), color='darkred', linestyle='--',
                           label=f'Mean: {np.mean(sarc_true_probs):.3f}')
        axes[0, 0].legend()

        # Sarcastic probability for true non-sarcastic samples
        sarc_false_probs = results['probabilities'][results['labels'] == 0, 1]
        axes[0, 1].hist(sarc_false_probs, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('Sarcastic Probability\n(True Non-Sarcastic Samples)', fontweight='bold')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(sarc_false_probs), color='darkblue', linestyle='--',
                           label=f'Mean: {np.mean(sarc_false_probs):.3f}')
        axes[0, 1].legend()

        # Combined distribution
        axes[1, 0].hist(sarc_true_probs, bins=15, alpha=0.6, color='red',
                        label='True Sarcastic', edgecolor='black')
        axes[1, 0].hist(sarc_false_probs, bins=15, alpha=0.6, color='blue',
                        label='True Non-Sarcastic', edgecolor='black')
        axes[1, 0].set_title('Sarcastic Probability Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # Confidence distribution
        confidences = np.max(results['probabilities'], axis=1)
        axes[1, 1].hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(confidences), color='darkgreen', linestyle='--',
                           label=f'Mean: {np.mean(confidences):.3f}')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{save_path}/probability_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_metrics_summary(self, metrics, save_path):
        """Plot comprehensive metrics summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Overall metrics bar chart
        overall_metrics = {
            'Accuracy': metrics['accuracy'],
            'F1-Macro': metrics['f1_macro'],
            'F1-Weighted': metrics['f1_weighted'],
            'Precision': metrics['precision_macro'],
            'Recall': metrics['recall_macro'],
            'ROC-AUC': metrics['roc_auc'],
            'Avg Precision': metrics['avg_precision']
        }

        bars = ax1.bar(overall_metrics.keys(), overall_metrics.values(),
                       color=['skyblue', 'lightgreen', 'orange', 'lightcoral',
                              'gold', 'plum', 'lightgray'])
        ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1)
        for bar, value in zip(bars, overall_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Per-class F1 scores
        ax2.bar(self.class_names, metrics['f1_per_class'],
                color=['lightblue', 'lightcoral'])
        ax2.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_ylim(0, 1)
        for i, v in enumerate(metrics['f1_per_class']):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

        # Per-class Precision
        ax3.bar(self.class_names, metrics['precision_per_class'],
                color=['lightgreen', 'orange'])
        ax3.set_title('Precision by Class', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Precision', fontsize=12)
        ax3.set_ylim(0, 1)
        for i, v in enumerate(metrics['precision_per_class']):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

        # Per-class Recall
        ax4.bar(self.class_names, metrics['recall_per_class'],
                color=['gold', 'plum'])
        ax4.set_title('Recall by Class', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Recall', fontsize=12)
        ax4.set_ylim(0, 1)
        for i, v in enumerate(metrics['recall_per_class']):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{save_path}/metrics_summary.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_per_class_performance(self, metrics, save_path):
        """Plot detailed per-class performance comparison"""
        classes = self.class_names
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 8))

        bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')

        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{save_path}/per_class_performance.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_error_analysis(self, results, save_path):
        """Plot error analysis"""
        # Get misclassified samples
        misclassified = results['predictions'] != results['labels']

        # False positives and false negatives
        fp_mask = (results['labels'] == 0) & (results['predictions'] == 1)
        fn_mask = (results['labels'] == 1) & (results['predictions'] == 0)

        fp_count = np.sum(fp_mask)
        fn_count = np.sum(fn_mask)
        tp_count = np.sum((results['labels'] == 1) & (results['predictions'] == 1))
        tn_count = np.sum((results['labels'] == 0) & (results['predictions'] == 0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Error type distribution
        error_types = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
        error_counts = [tp_count, tn_count, fp_count, fn_count]
        colors = ['lightgreen', 'lightblue', 'orange', 'lightcoral']

        bars = ax1.bar(error_types, error_counts, color=colors)
        ax1.set_title('Prediction Type Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12)
        for bar, count in zip(bars, error_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(count), ha='center', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Confidence of misclassified samples
        if np.sum(misclassified) > 0:
            misclass_confidences = np.max(results['probabilities'][misclassified], axis=1)
            correct_confidences = np.max(results['probabilities'][~misclassified], axis=1)

            ax2.hist(correct_confidences, bins=15, alpha=0.6, label='Correct',
                     color='green', edgecolor='black')
            ax2.hist(misclass_confidences, bins=15, alpha=0.6, label='Misclassified',
                     color='red', edgecolor='black')
            ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No Misclassifications!', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=16, fontweight='bold')
            ax2.set_title('Perfect Classification!', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{save_path}/error_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    def _save_results(self, results, metrics, save_path):
        """Save detailed results to files"""

        # Save metrics to JSON
        import json

        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            elif isinstance(value, np.integer):
                metrics_json[key] = int(value)
            elif isinstance(value, np.floating):
                metrics_json[key] = float(value)
            elif key == 'classification_report':
                metrics_json[key] = value  # Already a dict
            elif key == 'confusion_matrix':
                metrics_json[key] = value.tolist()
            else:
                metrics_json[key] = value

        with open(f"{save_path}/metrics.json", 'w') as f:
            json.dump(metrics_json, f, indent=2)

        # Save detailed results to CSV
        results_df = pd.DataFrame({
            'true_label': results['labels'],
            'predicted_label': results['predictions'],
            'probability_non_sarcastic': results['probabilities'][:, 0],
            'probability_sarcastic': results['probabilities'][:, 1],
            'confidence': np.max(results['probabilities'], axis=1),
            'correct': results['labels'] == results['predictions']
        })
        results_df.to_csv(f"{save_path}/detailed_results.csv", index=False)

        # Save summary report
        with open(f"{save_path}/evaluation_summary.txt", 'w') as f:
            f.write("🎯 SARCASM DETECTION MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write("📊 OVERALL PERFORMANCE:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"Average Precision: {metrics['avg_precision']:.4f}\n\n")

            f.write("📋 PER-CLASS PERFORMANCE:\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n")

            f.write(f"\n🎯 CONFUSION MATRIX:\n")
            f.write(f"{metrics['confusion_matrix']}\n")

            f.write(f"\n📈 DETAILED CLASSIFICATION REPORT:\n")
            f.write(classification_report(results['labels'], results['predictions'],
                                          target_names=self.class_names))

        print(f"📁 All results saved to: {save_path}")
        print("📊 Files generated:")
        print(f"  - confusion_matrix.png")
        print(f"  - roc_curve.png")
        print(f"  - precision_recall_curve.png")
        print(f"  - class_distribution.png")
        print(f"  - probability_distributions.png")
        print(f"  - metrics_summary.png")
        print(f"  - per_class_performance.png")
        print(f"  - error_analysis.png")
        print(f"  - metrics.json")
        print(f"  - detailed_results.csv")
        print(f"  - evaluation_summary.txt")


def load_model_and_evaluate(model_path, scaler_path, test_data_path):
    """🔍 Load trained model and perform comprehensive evaluation"""

    # Load model and scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    feature_cols = checkpoint['feature_cols']

    # Recreate model architecture
    from sarcasm_detector_trainer import OptimizedSarcasmDetector
    model = OptimizedSarcasmDetector(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Prepare test data
    exclude_cols = ['sarcasm_label', 'original_text', 'text_preview', 'text_index', 'dominant_emotion']
    X_test = test_df[feature_cols].values
    y_test = test_df['sarcasm_label'].values

    # Handle NaN values and scale
    X_test = np.nan_to_num(X_test, nan=0.0)
    X_test_scaled = scaler.transform(X_test)

    # Create test loader
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled),
                                 torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create evaluator and run evaluation
    evaluator = SarcasmEvaluationMetrics(model, scaler, feature_cols, device)
    metrics, save_path = evaluator.evaluate_comprehensive(test_loader)

    return metrics, save_path


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "sarcasm_training_results_20250725_035320/sarcasm_detector.pth"
    SCALER_PATH = "sarcasm_training_results_20250725_035320/scaler.pkl"
    TEST_DATA_PATH = "cleaned_sarc/test.csv"

    print("🚀 Starting comprehensive model evaluation...")
    metrics, results_path = load_model_and_evaluate(MODEL_PATH, SCALER_PATH, TEST_DATA_PATH)

    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {results_path}")
    print(f"🎯 Model Accuracy: {metrics['accuracy']:.4f}")
    print(f"📊 F1-Score: {metrics['f1_macro']:.4f}")