# enhanced_evaluate_goemotions_metrics.py
import pandas as pd
import torch
import numpy as np
import re
import os
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, jaccard_score, \
    classification_report
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class NegationAwareEmotionPredictor:
    def __init__(self, model, tokenizer, emotion_labels, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.emotion_labels = emotion_labels

        # Define negation patterns
        self.negation_patterns = [
            r'\bnot\s+', r'\bno\s+', r'\bnever\s+', r'\bnobody\s+',
            r'\bnothing\s+', r'\bnowhere\s+', r'\bneither\s+', r'\bnor\s+',
            r"n't\s+", r'\bwithout\s+', r'\bhardly\s+', r'\bbarely\s+', r'\bscarcely\s+'
        ]

        # Emotional opposites mapping
        self.emotion_opposites = {
            'sadness': ['joy', 'optimism', 'relief'],
            'anger': ['relief', 'approval', 'caring'],
            'fear': ['relief', 'optimism', 'approval'],
            'disgust': ['admiration', 'approval'],
            'disappointment': ['relief', 'optimism', 'approval'],
            'disapproval': ['approval', 'admiration'],
            'annoyance': ['relief', 'approval', 'caring'],
            'embarrassment': ['pride', 'relief'],
            'grief': ['relief', 'joy', 'optimism'],
            'nervousness': ['relief', 'optimism'],
            'remorse': ['relief', 'approval']
        }

    def detect_negation_context(self, text):
        """Detect if text contains negation and what emotions might be negated"""
        text_lower = text.lower()
        negations_found = []

        for pattern in self.negation_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start_pos = match.end()
                remaining_text = text_lower[start_pos:start_pos + 50]
                words_after = remaining_text.split()[:3]

                negations_found.append({
                    'negation_word': match.group().strip(),
                    'position': match.start(),
                    'context_after': words_after
                })

        return negations_found

    def apply_negation_adjustment(self, text, probs):
        """Apply negation-aware probability adjustments"""
        negations = self.detect_negation_context(text)
        if not negations:
            return probs, False

        adjusted_probs = probs.copy()
        text_lower = text.lower()

        for negation in negations:
            context_words = ' '.join(negation['context_after']).lower()

            for i, emotion in enumerate(self.emotion_labels):
                emotion_found = False

                # Direct emotion word match
                if emotion in context_words:
                    emotion_found = True
                # Partial word matches for emotion stems
                elif any(emotion.startswith(word.rstrip('.,!?')) for word in context_words.split() if len(word) > 3):
                    emotion_found = True

                # Special cases for common emotion words
                emotion_variants = {
                    'sadness': ['sad', 'unhappy', 'down'],
                    'anger': ['angry', 'mad', 'furious'],
                    'excitement': ['excited', 'thrilled'],
                    'fear': ['scared', 'afraid', 'frightened'],
                    'joy': ['happy', 'joyful', 'glad'],
                    'disappointment': ['disappointed', 'letdown']
                }

                if emotion in emotion_variants:
                    for variant in emotion_variants[emotion]:
                        if variant in context_words:
                            emotion_found = True
                            break

                if emotion_found:
                    # Reduce this emotion's probability
                    reduction_factor = 0.15
                    adjusted_probs[i] *= reduction_factor

                    # Boost opposite emotions if they exist
                    if emotion in self.emotion_opposites:
                        for opposite in self.emotion_opposites[emotion]:
                            if opposite in self.emotion_labels:
                                opp_idx = self.emotion_labels.index(opposite)
                                boost_factor = 2.0
                                adjusted_probs[opp_idx] = min(1.0, adjusted_probs[opp_idx] * boost_factor)

        return adjusted_probs, True


# Enhanced Dataset class
class EnhancedEmotionDataset(Dataset):
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
            'text': text  # Include original text for negation analysis
        }


def evaluate_with_negation_handling(model, tokenizer, test_loader, emotion_labels, device, negation_predictor):
    """Enhanced evaluation with negation handling"""
    model.eval()

    all_probs_original = []
    all_probs_negation = []
    all_labels = []
    negation_stats = {'total_samples': 0, 'negation_detected': 0, 'adjustments_made': 0}

    print("Running enhanced evaluation with negation handling...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"Processing batch {i + 1}/{len(test_loader)}")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text']

            # Get original model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs_original = torch.sigmoid(logits).cpu().numpy()

            # Apply negation handling
            probs_negation = []
            for j, (prob_row, text) in enumerate(zip(probs_original, texts)):
                negation_stats['total_samples'] += 1

                adjusted_probs, had_negation = negation_predictor.apply_negation_adjustment(text, prob_row)
                probs_negation.append(adjusted_probs)

                if had_negation:
                    negation_stats['negation_detected'] += 1
                    if not np.allclose(prob_row, adjusted_probs, atol=1e-6):
                        negation_stats['adjustments_made'] += 1

            probs_negation = np.array(probs_negation)

            all_probs_original.append(probs_original)
            all_probs_negation.append(probs_negation)
            all_labels.append(labels.cpu().numpy())

    # Concatenate all results
    all_probs_original = np.concatenate(all_probs_original, axis=0)
    all_probs_negation = np.concatenate(all_probs_negation, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs_original, all_probs_negation, all_labels, negation_stats


def compute_metrics(y_true, y_pred, prefix=""):
    """Compute evaluation metrics"""
    metrics = {
        f'{prefix}f1_micro': f1_score(y_true, y_pred, average='micro'),
        f'{prefix}f1_macro': f1_score(y_true, y_pred, average='macro'),
        f'{prefix}precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        f'{prefix}recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        f'{prefix}hamming_loss': hamming_loss(y_true, y_pred),
        f'{prefix}jaccard_micro': jaccard_score(y_true, y_pred, average='micro'),
        f'{prefix}subset_accuracy': accuracy_score(y_true, y_pred),
    }
    return metrics


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your already split test dataset
    print("Loading test dataset...")
    # Replace with your actual test dataset path
    test_df = pd.read_csv("dataset/goemotions_test.csv")  # Update this path to your test file

    # Emotion labels
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise'
    ]

    test_texts = test_df["text"].tolist()
    test_labels = test_df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values

    print(f"Test samples: {len(test_texts)}")

    # Load model and tokenizer
    MODEL_PATH = os.getenv("MODEL_PATH", "saved_model_xlm-roberta-base")
    MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")

    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)

    # Create negation predictor
    negation_predictor = NegationAwareEmotionPredictor(model, tokenizer, emotion_labels, device)

    # Create test dataset and dataloader
    test_dataset = EnhancedEmotionDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Run evaluation
    probs_original, probs_negation, true_labels, negation_stats = evaluate_with_negation_handling(
        model, tokenizer, test_loader, emotion_labels, device, negation_predictor
    )

    # Generate predictions with threshold 0.5
    preds_original = (probs_original > 0.5).astype(int)
    preds_negation = (probs_negation > 0.5).astype(int)

    # Compute metrics for both approaches
    metrics_original = compute_metrics(true_labels, preds_original, "original_")
    metrics_negation = compute_metrics(true_labels, preds_negation, "negation_")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{MODEL_NAME}_negation_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save comprehensive metrics
    all_metrics = {**metrics_original, **metrics_negation}

    with open(f"{output_dir}/evaluation_metrics.txt", "w", encoding="utf-8") as f:
        f.write("📊 Enhanced Evaluation Metrics with Negation Handling\n")
        f.write("=" * 60 + "\n\n")

        f.write("🔍 Negation Detection Statistics:\n")
        f.write(f"- Total samples: {negation_stats['total_samples']}\n")
        f.write(
            f"- Negation detected: {negation_stats['negation_detected']} ({negation_stats['negation_detected'] / negation_stats['total_samples'] * 100:.1f}%)\n")
        f.write(
            f"- Adjustments made: {negation_stats['adjustments_made']} ({negation_stats['adjustments_made'] / negation_stats['total_samples'] * 100:.1f}%)\n\n")

        f.write("📈 ORIGINAL MODEL METRICS:\n")
        for k, v in metrics_original.items():
            clean_k = k.replace('original_', '').replace('_', ' ').title()
            f.write(f"- {clean_k}: {v:.4f}\n")

        f.write(f"\n📈 NEGATION-AWARE METRICS:\n")
        for k, v in metrics_negation.items():
            clean_k = k.replace('negation_', '').replace('_', ' ').title()
            f.write(f"- {clean_k}: {v:.4f}\n")

        f.write(f"\n📊 IMPROVEMENT ANALYSIS:\n")
        for metric in ['f1_micro', 'f1_macro', 'precision_micro', 'recall_micro', 'subset_accuracy']:
            orig_val = metrics_original[f'original_{metric}']
            neg_val = metrics_negation[f'negation_{metric}']
            improvement = ((neg_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
            f.write(f"- {metric.replace('_', ' ').title()}: {improvement:+.2f}%\n")

    # Generate classification reports
    class_report_original = classification_report(
        true_labels, preds_original, target_names=emotion_labels, output_dict=True, zero_division=0
    )
    class_report_negation = classification_report(
        true_labels, preds_negation, target_names=emotion_labels, output_dict=True, zero_division=0
    )

    pd.DataFrame(class_report_original).transpose().to_csv(f"{output_dir}/classification_report_original.csv")
    pd.DataFrame(class_report_negation).transpose().to_csv(f"{output_dir}/classification_report_negation.csv")

    # Generate visualizations
    generate_enhanced_visualizations(
        true_labels, preds_original, preds_negation,
        emotion_labels, output_dir, class_report_original, class_report_negation
    )

    # Threshold analysis for negation-aware predictions
    print("Running threshold analysis...")
    run_threshold_analysis(probs_negation, true_labels, emotion_labels, output_dir)

    print(f"\n✅ Enhanced evaluation completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"🔍 Negation detected in {negation_stats['negation_detected']}/{negation_stats['total_samples']} samples")
    print(f"⚙️  Adjustments made to {negation_stats['adjustments_made']} samples")


def generate_enhanced_visualizations(true_labels, preds_original, preds_negation, emotion_labels, output_dir,
                                     class_report_orig, class_report_neg):
    """Generate enhanced visualizations comparing both approaches"""

    # 1. Performance comparison plot
    metrics_to_compare = ['precision', 'recall', 'f1-score']
    comparison_data = []

    for emotion in emotion_labels:
        if emotion in class_report_orig and emotion in class_report_neg:
            for metric in metrics_to_compare:
                comparison_data.append({
                    'emotion': emotion,
                    'metric': metric,
                    'original': class_report_orig[emotion][metric],
                    'negation_aware': class_report_neg[emotion][metric]
                })

    comparison_df = pd.DataFrame(comparison_data)

    plt.figure(figsize=(20, 12))
    for i, metric in enumerate(metrics_to_compare):
        plt.subplot(3, 1, i + 1)
        metric_data = comparison_df[comparison_df['metric'] == metric]

        x = np.arange(len(emotion_labels))
        width = 0.35

        plt.bar(x - width / 2, metric_data['original'], width, label='Original', alpha=0.7)
        plt.bar(x + width / 2, metric_data['negation_aware'], width, label='Negation-Aware', alpha=0.7)

        plt.xlabel('Emotions')
        plt.ylabel(metric.title())
        plt.title(f'{metric.title()} Comparison: Original vs Negation-Aware')
        plt.xticks(x, emotion_labels, rotation=90)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Improvement heatmap
    improvement_matrix = np.zeros((len(emotion_labels), len(metrics_to_compare)))

    for i, emotion in enumerate(emotion_labels):
        for j, metric in enumerate(metrics_to_compare):
            if emotion in class_report_orig and emotion in class_report_neg:
                orig_val = class_report_orig[emotion][metric]
                neg_val = class_report_neg[emotion][metric]
                improvement = ((neg_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
                improvement_matrix[i, j] = improvement

    plt.figure(figsize=(8, 15))
    sns.heatmap(
        improvement_matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        xticklabels=[m.title() for m in metrics_to_compare],
        yticklabels=emotion_labels,
        cbar_kws={'label': 'Improvement (%)'}
    )
    plt.title('Performance Improvement with Negation Handling (%)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def run_threshold_analysis(probs, true_labels, emotion_labels, output_dir):
    """Run threshold analysis for negation-aware predictions"""
    thresholds = np.linspace(0.1, 0.9, 9)
    best_thresholds = []

    for emotion in emotion_labels:
        idx = emotion_labels.index(emotion)
        f1_scores = []

        for thresh in thresholds:
            preds_thresh = (probs[:, idx] > thresh).astype(int)
            f1 = f1_score(true_labels[:, idx], preds_thresh, zero_division=0)
            f1_scores.append(f1)

        best_idx = int(np.argmax(f1_scores))
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        best_thresholds.append({
            "emotion": emotion,
            "best_threshold": best_thresh,
            "best_f1": best_f1
        })

        # Save plot for each emotion
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores, marker='o', color='teal')
        plt.title(f"F1-Score vs Threshold for '{emotion}' (Negation-Aware)")
        plt.xlabel("Threshold")
        plt.ylabel("F1-Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/threshold_analysis_{emotion}.png")
        plt.close()

    # Save CSV summary
    pd.DataFrame(best_thresholds).to_csv(f"{output_dir}/best_thresholds_negation_aware.csv", index=False)


if __name__ == "__main__":
    main()