# effective_eval_with_optimal_thresholds.py
import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    hamming_loss, jaccard_score, classification_report
)
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Emotion labels (27 emotions, no neutral)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'
]

# Your optimal thresholds
OPTIMAL_THRESHOLDS = {
    "admiration": 0.35,
    "amusement": 0.35,
    "anger": 0.25,
    "annoyance": 0.15,
    "approval": 0.30,
    "caring": 0.25,
    "confusion": 0.2,
    "curiosity": 0.30,
    "desire": 0.2,
    "disappointment": 0.25,
    "disapproval": 0.2,
    "disgust": 0.25,
    "embarrassment": 0.2,
    "excitement": 0.30,
    "fear": 0.4,
    "gratitude": 0.45,
    "grief": 0.15,
    "joy": 0.25,
    "love": 0.35,
    "nervousness": 0.25,
    "optimism": 0.4,
    "pride": 0.15,
    "realization": 0.15,
    "relief": 0.55,
    "remorse": 0.35,
    "sadness": 0.25,
    "surprise": 0.25
}


class EmotionDataset(Dataset):
    """Dataset class for emotion classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def load_data():
    """Load and prepare the test dataset"""
    print("Loading test dataset...")

    try:
        test_df = pd.read_csv("dataset/goemotions_test.csv")
        print(f"Successfully loaded test dataset with {len(test_df)} samples")
        texts = test_df["text"].tolist()
        labels = test_df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values
        return texts, labels

    except FileNotFoundError:
        print("Could not find 'dataset/goemotions_test.csv'. Trying alternative files...")
        alternative_files = [
            "dataset/test_dataset.csv",
            "dataset/test.csv",
            "dataset/goemotions_test_data.csv"
        ]

        for file_path in alternative_files:
            try:
                test_df = pd.read_csv(file_path)
                print(f"Successfully loaded test dataset from {file_path} with {len(test_df)} samples")
                texts = test_df["text"].tolist()
                labels = test_df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values
                return texts, labels
            except FileNotFoundError:
                continue

        raise FileNotFoundError("Could not find test dataset files")


def load_model():
    """Load the trained model and tokenizer"""
    MODEL_PATH = os.getenv("MODEL_PATH", "saved_model_xlm-roberta-base")
    MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")

    print(f"Loading model from: {MODEL_PATH}")

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    return model, tokenizer, MODEL_NAME


def run_evaluation(model, tokenizer, test_texts, test_labels):
    """Run model evaluation on test set"""
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_probs = []
    all_labels = []

    print("Running evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if (i + 1) % 100 == 0:
                print(f"Processing batch {i + 1}/{len(test_loader)}")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs, all_labels


def apply_optimal_thresholds(probabilities):
    """Apply optimal thresholds to get predictions"""
    predictions = np.zeros_like(probabilities, dtype=int)

    for i, emotion in enumerate(emotion_labels):
        threshold = OPTIMAL_THRESHOLDS[emotion]
        predictions[:, i] = (probabilities[:, i] > threshold).astype(int)

    return predictions


def compute_metrics(true_labels, predictions, method_name):
    """Compute comprehensive evaluation metrics"""
    metrics = {
        f'{method_name}_f1_micro': f1_score(true_labels, predictions, average='micro'),
        f'{method_name}_f1_macro': f1_score(true_labels, predictions, average='macro'),
        f'{method_name}_f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        f'{method_name}_precision_micro': precision_score(true_labels, predictions, average='micro', zero_division=0),
        f'{method_name}_precision_macro': precision_score(true_labels, predictions, average='macro', zero_division=0),
        f'{method_name}_recall_micro': recall_score(true_labels, predictions, average='micro', zero_division=0),
        f'{method_name}_recall_macro': recall_score(true_labels, predictions, average='macro', zero_division=0),
        f'{method_name}_hamming_loss': hamming_loss(true_labels, predictions),
        f'{method_name}_jaccard_micro': jaccard_score(true_labels, predictions, average='micro'),
        f'{method_name}_jaccard_macro': jaccard_score(true_labels, predictions, average='macro'),
        f'{method_name}_subset_accuracy': accuracy_score(true_labels, predictions),
    }
    return metrics


def generate_classification_report(true_labels, predictions, method_name):
    """Generate detailed classification report"""
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=emotion_labels,
        output_dict=True,
        zero_division=0
    )
    class_report_df = pd.DataFrame(class_report).transpose()
    return class_report_df


def compare_methods(default_metrics, optimal_metrics, default_report, optimal_report):
    """Compare default vs optimal threshold performance"""

    print("\n" + "=" * 80)
    print("🎯 PERFORMANCE COMPARISON: DEFAULT (0.5) vs OPTIMAL THRESHOLDS")
    print("=" * 80)

    # Overall metrics comparison
    comparison_data = []

    metric_pairs = [
        ('f1_micro', 'F1-Micro'),
        ('f1_macro', 'F1-Macro'),
        ('f1_weighted', 'F1-Weighted'),
        ('precision_macro', 'Precision-Macro'),
        ('recall_macro', 'Recall-Macro'),
        ('subset_accuracy', 'Subset Accuracy'),
        ('hamming_loss', 'Hamming Loss')
    ]

    print(f"\n📊 OVERALL METRICS COMPARISON:")
    print(f"{'Metric':<20} {'Default':<10} {'Optimal':<10} {'Improvement':<12} {'% Change':<10}")
    print("-" * 70)

    for metric_key, metric_name in metric_pairs:
        default_val = default_metrics[f'default_{metric_key}']
        optimal_val = optimal_metrics[f'optimal_{metric_key}']

        if metric_key == 'hamming_loss':
            # Lower is better for hamming loss
            improvement = default_val - optimal_val
            pct_change = (improvement / default_val) * 100 if default_val > 0 else 0
            improvement_str = f"{improvement:+.4f}"
        else:
            # Higher is better for other metrics
            improvement = optimal_val - default_val
            pct_change = (improvement / default_val) * 100 if default_val > 0 else 0
            improvement_str = f"{improvement:+.4f}"

        print(f"{metric_name:<20} {default_val:<10.4f} {optimal_val:<10.4f} {improvement_str:<12} {pct_change:+.2f}%")

        comparison_data.append({
            'metric': metric_name,
            'default': default_val,
            'optimal': optimal_val,
            'improvement': improvement,
            'pct_change': pct_change
        })

    # Per-emotion F1 score comparison
    print(f"\n📈 PER-EMOTION F1-SCORE IMPROVEMENTS:")
    print(f"{'Emotion':<15} {'Default':<8} {'Optimal':<8} {'Improvement':<12} {'% Change':<10}")
    print("-" * 60)

    emotion_improvements = []
    for emotion in emotion_labels:
        default_f1 = default_report.loc[emotion, 'f1-score']
        optimal_f1 = optimal_report.loc[emotion, 'f1-score']
        improvement = optimal_f1 - default_f1
        pct_change = (improvement / default_f1) * 100 if default_f1 > 0 else 0

        print(f"{emotion:<15} {default_f1:<8.3f} {optimal_f1:<8.3f} {improvement:+.3f}        {pct_change:+.1f}%")

        emotion_improvements.append({
            'emotion': emotion,
            'default_f1': default_f1,
            'optimal_f1': optimal_f1,
            'improvement': improvement,
            'pct_change': pct_change,
            'threshold': OPTIMAL_THRESHOLDS[emotion]
        })

    # Summary statistics
    emotion_improvements_df = pd.DataFrame(emotion_improvements)

    print(f"\n📋 SUMMARY STATISTICS:")
    print(
        f"Emotions with improved F1-score: {len(emotion_improvements_df[emotion_improvements_df['improvement'] > 0])}/27")
    print(
        f"Emotions with >5% improvement: {len(emotion_improvements_df[emotion_improvements_df['pct_change'] > 5])}/27")
    print(
        f"Emotions with >10% improvement: {len(emotion_improvements_df[emotion_improvements_df['pct_change'] > 10])}/27")
    print(f"Average F1-score improvement: {emotion_improvements_df['improvement'].mean():.4f}")
    print(
        f"Maximum F1-score improvement: {emotion_improvements_df['improvement'].max():.4f} ({emotion_improvements_df.loc[emotion_improvements_df['improvement'].idxmax(), 'emotion']})")
    print(f"Average percentage improvement: {emotion_improvements_df['pct_change'].mean():.2f}%")

    return pd.DataFrame(comparison_data), emotion_improvements_df


def create_comparison_visualizations(comparison_df, emotion_improvements_df, output_dir):
    """Create comprehensive comparison visualizations"""

    print("📊 Creating comparison visualizations...")

    # 1. Overall metrics comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Bar chart of overall metrics
    metrics_to_plot = comparison_df[comparison_df['metric'] != 'Hamming Loss']  # Exclude hamming loss
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, metrics_to_plot['default'], width, label='Default (0.5)', color='lightcoral',
                    alpha=0.8)
    bars2 = ax1.bar(x + width / 2, metrics_to_plot['optimal'], width, label='Optimal Thresholds', color='lightgreen',
                    alpha=0.8)

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance: Default vs Optimal Thresholds', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_plot['metric'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Percentage improvements
    ax2.bar(metrics_to_plot['metric'], metrics_to_plot['pct_change'], color='orange', alpha=0.8)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Percentage Improvement (%)')
    ax2.set_title('Percentage Improvement with Optimal Thresholds', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(metrics_to_plot['pct_change']):
        ax2.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Top 15 emotion improvements
    top_15_emotions = emotion_improvements_df.nlargest(15, 'improvement')
    ax3.barh(range(len(top_15_emotions)), top_15_emotions['improvement'], color='lightblue', alpha=0.8)
    ax3.set_yticks(range(len(top_15_emotions)))
    ax3.set_yticklabels(top_15_emotions['emotion'])
    ax3.set_xlabel('F1-Score Improvement')
    ax3.set_title('Top 15 Emotions: F1-Score Improvements', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Threshold distribution
    ax4.hist(list(OPTIMAL_THRESHOLDS.values()), bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Default Threshold')
    ax4.axvline(x=np.mean(list(OPTIMAL_THRESHOLDS.values())), color='green', linestyle='--', linewidth=2,
                label=f'Mean Optimal: {np.mean(list(OPTIMAL_THRESHOLDS.values())):.3f}')
    ax4.set_xlabel('Threshold Value')
    ax4.set_ylabel('Number of Emotions')
    ax4.set_title('Distribution of Optimal Thresholds', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Detailed F1-score comparison heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Default F1-scores heatmap
    default_f1_grid = emotion_improvements_df['default_f1'].values.reshape(3, 9)
    emotion_grid = np.array(emotion_labels).reshape(3, 9)

    im1 = ax1.imshow(default_f1_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_title('F1-Scores with Default Thresholds (0.5)', fontsize=14, fontweight='bold')

    for i in range(3):
        for j in range(9):
            ax1.text(j, i, f'{emotion_grid[i, j]}\n{default_f1_grid[i, j]:.3f}',
                     ha="center", va="center", fontsize=9, fontweight='bold')

    ax1.set_xticks([])
    ax1.set_yticks([])

    # Optimal F1-scores heatmap
    optimal_f1_grid = emotion_improvements_df['optimal_f1'].values.reshape(3, 9)

    im2 = ax2.imshow(optimal_f1_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('F1-Scores with Optimal Thresholds', fontsize=14, fontweight='bold')

    for i in range(3):
        for j in range(9):
            threshold = OPTIMAL_THRESHOLDS[emotion_grid[i, j]]
            ax2.text(j, i, f'{emotion_grid[i, j]}\n{optimal_f1_grid[i, j]:.3f}\n(t={threshold:.2f})',
                     ha="center", va="center", fontsize=8, fontweight='bold')

    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='F1-Score')
    plt.colorbar(im2, ax=ax2, label='F1-Score')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_comparison_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Improvement scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(emotion_improvements_df['default_f1'], emotion_improvements_df['optimal_f1'],
                          c=emotion_improvements_df['pct_change'], cmap='RdYlGn',
                          s=100, alpha=0.7, edgecolors='black')

    # Add diagonal line (no improvement line)
    min_val = min(emotion_improvements_df['default_f1'].min(), emotion_improvements_df['optimal_f1'].min())
    max_val = max(emotion_improvements_df['default_f1'].max(), emotion_improvements_df['optimal_f1'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='No Improvement Line')

    plt.xlabel('Default F1-Score (0.5 threshold)')
    plt.ylabel('Optimal F1-Score')
    plt.title('F1-Score Improvement: Default vs Optimal Thresholds', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Percentage Improvement (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate extreme points
    max_improvement_idx = emotion_improvements_df['improvement'].idxmax()
    max_improvement_emotion = emotion_improvements_df.loc[max_improvement_idx]
    plt.annotate(f"{max_improvement_emotion['emotion']}\n+{max_improvement_emotion['improvement']:.3f}",
                 xy=(max_improvement_emotion['default_f1'], max_improvement_emotion['optimal_f1']),
                 xytext=(10, 10), textcoords='offset points', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ All comparison visualizations saved to {output_dir}")


def save_comprehensive_results(default_metrics, optimal_metrics, comparison_df, emotion_improvements_df, output_dir,
                               model_name):
    """Save comprehensive comparison results"""

    # Save detailed comparison report
    with open(f"{output_dir}/threshold_optimization_report.txt", "w", encoding="utf-8") as f:
        f.write("🎯 THRESHOLD OPTIMIZATION EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("📊 EXECUTIVE SUMMARY:\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Emotions: {len(emotion_labels)}\n")
        f.write(f"Default Threshold: 0.5 (uniform)\n")
        f.write(
            f"Optimal Thresholds: Emotion-specific (range: {min(OPTIMAL_THRESHOLDS.values()):.2f} - {max(OPTIMAL_THRESHOLDS.values()):.2f})\n\n")

        f.write("🎯 KEY IMPROVEMENTS:\n")
        f1_macro_improvement = optimal_metrics['optimal_f1_macro'] - default_metrics['default_f1_macro']
        f1_micro_improvement = optimal_metrics['optimal_f1_micro'] - default_metrics['default_f1_micro']

        f.write(
            f"- F1-Macro: {default_metrics['default_f1_macro']:.4f} → {optimal_metrics['optimal_f1_macro']:.4f} (+{f1_macro_improvement:.4f})\n")
        f.write(
            f"- F1-Micro: {default_metrics['default_f1_micro']:.4f} → {optimal_metrics['optimal_f1_micro']:.4f} (+{f1_micro_improvement:.4f})\n")
        f.write(
            f"- Subset Accuracy: {default_metrics['default_subset_accuracy']:.4f} → {optimal_metrics['optimal_subset_accuracy']:.4f}\n")
        f.write(
            f"- Hamming Loss: {default_metrics['default_hamming_loss']:.4f} → {optimal_metrics['optimal_hamming_loss']:.4f}\n\n")

        f.write("📈 EMOTION-LEVEL ANALYSIS:\n")
        emotions_improved = len(emotion_improvements_df[emotion_improvements_df['improvement'] > 0])
        f.write(f"- Emotions with improved F1-score: {emotions_improved}/27 ({emotions_improved / 27 * 100:.1f}%)\n")
        f.write(f"- Average F1-score improvement: {emotion_improvements_df['improvement'].mean():.4f}\n")
        f.write(
            f"- Maximum improvement: {emotion_improvements_df['improvement'].max():.4f} ({emotion_improvements_df.loc[emotion_improvements_df['improvement'].idxmax(), 'emotion']})\n")
        f.write(
            f"- Emotions with >10% improvement: {len(emotion_improvements_df[emotion_improvements_df['pct_change'] > 10])}/27\n\n")

        f.write("🎛️ THRESHOLD ANALYSIS:\n")
        f.write(f"- Average optimal threshold: {np.mean(list(OPTIMAL_THRESHOLDS.values())):.3f}\n")
        f.write(f"- Threshold range: {min(OPTIMAL_THRESHOLDS.values()):.2f} - {max(OPTIMAL_THRESHOLDS.values()):.2f}\n")
        f.write(f"- Emotions with threshold < 0.3: {len([t for t in OPTIMAL_THRESHOLDS.values() if t < 0.3])}/27\n")
        f.write(f"- Emotions with threshold > 0.4: {len([t for t in OPTIMAL_THRESHOLDS.values() if t > 0.4])}/27\n\n")

        f.write("🏆 TOP 10 MOST IMPROVED EMOTIONS:\n")
        top_10 = emotion_improvements_df.nlargest(10, 'improvement')
        for _, row in top_10.iterrows():
            f.write(
                f"- {row['emotion']}: {row['default_f1']:.3f} → {row['optimal_f1']:.3f} (+{row['improvement']:.3f}, {row['pct_change']:+.1f}%)\n")

    # Save detailed data
    comparison_df.to_csv(f"{output_dir}/metrics_comparison.csv", index=False)
    emotion_improvements_df.to_csv(f"{output_dir}/emotion_improvements.csv", index=False)

    # Save optimal thresholds
    with open(f"{output_dir}/optimal_thresholds_used.json", 'w') as f:
        json.dump(OPTIMAL_THRESHOLDS, f, indent=2)

    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"🎯 F1-Macro Improvement: {f1_macro_improvement:+.4f}")
    print(f"🎯 F1-Micro Improvement: {f1_micro_improvement:+.4f}")
    print(f"🎯 Emotions Improved: {emotions_improved}/27")


def main():
    """Main evaluation pipeline with optimal thresholds"""
    try:
        print("🚀 EFFECTIVE EVALUATION WITH OPTIMAL THRESHOLDS")
        print("=" * 60)

        # Load test data
        test_texts, test_labels = load_data()
        print(f"Loaded {len(test_texts)} samples from test dataset")

        # Load model
        model, tokenizer, model_name = load_model()

        # Run evaluation to get probabilities
        all_probs, true_labels = run_evaluation(model, tokenizer, test_texts, test_labels)

        # Generate predictions with both methods
        print("\n🔄 Generating predictions with different thresholds...")

        # Default threshold (0.5)
        default_predictions = (all_probs > 0.5).astype(int)

        # Optimal thresholds
        optimal_predictions = apply_optimal_thresholds(all_probs)

        # Compute metrics for both methods
        print("📊 Computing metrics...")
        default_metrics = compute_metrics(true_labels, default_predictions, 'default')
        optimal_metrics = compute_metrics(true_labels, optimal_predictions, 'optimal')

        # Generate classification reports
        default_report = generate_classification_report(true_labels, default_predictions, 'default')
        optimal_report = generate_classification_report(true_labels, optimal_predictions, 'optimal')

        # Compare methods
        comparison_df, emotion_improvements_df = compare_methods(
            default_metrics, optimal_metrics, default_report, optimal_report
        )

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/optimal_threshold_eval_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Create visualizations
        create_comparison_visualizations(comparison_df, emotion_improvements_df, output_dir)

        # Save comprehensive results
        save_comprehensive_results(default_metrics, optimal_metrics, comparison_df,
                                   emotion_improvements_df, output_dir, model_name)

    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()