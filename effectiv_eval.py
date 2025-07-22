# clean_evaluate_goemotions.py
import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
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

# Load environment variables
load_dotenv()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


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
        # Load the pre-split test dataset
        test_df = pd.read_csv("dataset/goemotions_test.csv")
        print(f"Successfully loaded test dataset with {len(test_df)} samples")

        # Extract texts and labels
        texts = test_df["text"].tolist()
        labels = test_df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values

        return texts, labels

    except FileNotFoundError:
        print("Could not find 'dataset/goemotions_test.csv'. Trying alternative file names...")

        # Try alternative file names
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

        # If no test file found, raise error with helpful message
        raise FileNotFoundError(
            "Could not find test dataset. Please ensure one of the following files exists:\n"
            "- dataset/goemotions_test.csv\n"
            "- dataset/test_dataset.csv\n"
            "- dataset/test.csv\n"
            "- dataset/goemotions_test_data.csv"
        )


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
    # Create dataset and dataloader
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Batch prediction
    all_probs = []
    all_labels = []

    print("Running evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if (i + 1) % 10 == 0:
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


def compute_metrics(true_labels, predictions):
    """Compute comprehensive evaluation metrics"""
    metrics = {
        'f1_micro': f1_score(true_labels, predictions, average='micro'),
        'f1_macro': f1_score(true_labels, predictions, average='macro'),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        'precision_micro': precision_score(true_labels, predictions, average='micro', zero_division=0),
        'precision_macro': precision_score(true_labels, predictions, average='macro', zero_division=0),
        'recall_micro': recall_score(true_labels, predictions, average='micro', zero_division=0),
        'recall_macro': recall_score(true_labels, predictions, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(true_labels, predictions),
        'jaccard_micro': jaccard_score(true_labels, predictions, average='micro'),
        'jaccard_macro': jaccard_score(true_labels, predictions, average='macro'),
        'subset_accuracy': accuracy_score(true_labels, predictions),
    }
    return metrics


def generate_classification_report(true_labels, predictions):
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


def create_visualizations(class_report_df, output_dir):
    """Create comprehensive visualizations"""

    # 1. Class distribution plot
    plt.figure(figsize=(15, 8))
    support = class_report_df.loc[emotion_labels, "support"].astype(int)
    sns.barplot(x=emotion_labels, y=support.values)
    plt.title("Class Distribution in Test Set", fontsize=16)
    plt.xticks(rotation=90)
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance metrics per emotion
    metrics_df = class_report_df.loc[emotion_labels, ["precision", "recall", "f1-score"]]
    metrics_df = metrics_df.reset_index().melt(id_vars="index", var_name="metric")

    plt.figure(figsize=(18, 10))
    sns.barplot(data=metrics_df, x="index", y="value", hue="metric")
    plt.title("Performance Metrics per Emotion", fontsize=16)
    plt.xticks(rotation=90)
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_emotion_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. F1-Score heatmap
    f1_scores = class_report_df.loc[emotion_labels, "f1-score"].values.reshape(4, 7)  # 4x7 grid

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        f1_scores,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        xticklabels=emotion_labels[:7],
        yticklabels=[emotion_labels[i:i + 7] for i in range(0, len(emotion_labels), 7)],
        cbar_kws={'label': 'F1-Score'}
    )
    plt.title("F1-Score Heatmap by Emotion")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def run_threshold_analysis(all_probs, true_labels, output_dir):
    """Run threshold analysis for each emotion"""
    thresholds = np.linspace(0.1, 0.9, 9)
    best_thresholds = []

    print("Running threshold analysis...")
    for emotion in emotion_labels:
        idx = emotion_labels.index(emotion)
        f1_scores = []

        for thresh in thresholds:
            preds_thresh = (all_probs[:, idx] > thresh).astype(int)
            f1 = f1_score(true_labels[:, idx], preds_thresh, zero_division=0)
            f1_scores.append(f1)

        best_idx = int(np.argmax(f1_scores))
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        best_thresholds.append({
            "emotion": emotion,
            "best_threshold": best_thresh,
            "best_f1": best_f1,
            "default_f1": f1_scores[4]  # 0.5 threshold
        })

        # Save plot for each emotion
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores, marker='o', color='teal', linewidth=2)
        plt.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_thresh:.1f}')
        plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Default: 0.5')
        plt.title(f"F1-Score vs Threshold for '{emotion}'")
        plt.xlabel("Threshold")
        plt.ylabel("F1-Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/threshold_analysis_{emotion}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Save threshold analysis results
    threshold_df = pd.DataFrame(best_thresholds)
    threshold_df['improvement'] = (
            (threshold_df['best_f1'] - threshold_df['default_f1']) / threshold_df['default_f1'] * 100).round(2)
    threshold_df.to_csv(f"{output_dir}/best_thresholds.csv", index=False)

    return threshold_df


def save_results(metrics, class_report_df, threshold_df, output_dir, model_name):
    """Save all evaluation results"""

    # Save comprehensive metrics
    with open(f"{output_dir}/evaluation_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"📊 Evaluation Results for {model_name}\n")
        f.write("=" * 60 + "\n\n")

        f.write("🎯 OVERALL METRICS:\n")
        for k, v in metrics.items():
            clean_k = k.replace('_', ' ').title()
            f.write(f"- {clean_k}: {v:.4f}\n")

        f.write(f"\n📈 PER-CLASS SUMMARY:\n")
        avg_f1 = class_report_df.loc[emotion_labels, "f1-score"].mean()
        best_emotion = class_report_df.loc[emotion_labels, "f1-score"].idxmax()
        worst_emotion = class_report_df.loc[emotion_labels, "f1-score"].idxmin()

        f.write(f"- Average F1-Score: {avg_f1:.4f}\n")
        f.write(f"- Best performing emotion: {best_emotion} ({class_report_df.loc[best_emotion, 'f1-score']:.4f})\n")
        f.write(f"- Worst performing emotion: {worst_emotion} ({class_report_df.loc[worst_emotion, 'f1-score']:.4f})\n")

        f.write(f"\n🎛️ THRESHOLD OPTIMIZATION:\n")
        f.write(
            f"- Emotions that benefit from threshold tuning: {len(threshold_df[threshold_df['improvement'] > 1])}\n")
        f.write(f"- Maximum improvement possible: {threshold_df['improvement'].max():.2f}%\n")
        f.write(f"- Average improvement with optimal thresholds: {threshold_df['improvement'].mean():.2f}%\n")

    # Save classification report
    class_report_df.to_csv(f"{output_dir}/classification_report.csv")

    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"🎯 Overall F1-Micro: {metrics['f1_micro']:.4f}")
    print(f"🎯 Overall F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"🎯 Subset Accuracy: {metrics['subset_accuracy']:.4f}")


def main():
    """Main evaluation pipeline"""
    try:
        # Load test data (already split)
        test_texts, test_labels = load_data()
        print(f"Loaded {len(test_texts)} samples from test dataset")

        # Load model
        model, tokenizer, model_name = load_model()

        # Run evaluation
        all_probs, true_labels = run_evaluation(model, tokenizer, test_texts, test_labels)

        # Generate predictions with default threshold
        predictions = (all_probs > 0.5).astype(int)

        # Compute metrics
        metrics = compute_metrics(true_labels, predictions)

        # Generate classification report
        class_report_df = generate_classification_report(true_labels, predictions)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{model_name}_eval_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Generate visualizations
        create_visualizations(class_report_df, output_dir)

        # Run threshold analysis
        threshold_df = run_threshold_analysis(all_probs, true_labels, output_dir)

        # Save all results
        save_results(metrics, class_report_df, threshold_df, output_dir, model_name)

    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()