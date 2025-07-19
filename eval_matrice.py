# evaluate_goemotions_metrics.py
import pandas as pd
import torch
import numpy as np
# from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, jaccard_score, \
    classification_report
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load all datasets
df1 = pd.read_csv("dataset/goemotions_1.csv")
df2 = pd.read_csv("dataset/goemotions_2.csv")
df3 = pd.read_csv("dataset/goemotions_3.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)

# Emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

texts = df["text"].tolist()
labels = df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values

# Split dataset
_, val_texts, _, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
# MODEL_NAME=os.getenv("MODEL_NAME")
# # Load model
# MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_PATH = "saved_model_xlm-roberta-base"
MODEL_NAME = "xlm-roberta-base"
# Before saving plots
os.makedirs(f"results/{MODEL_NAME}", exist_ok=True)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)

model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


# Dataset class
class EmotionDataset(Dataset):
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


# Create DataLoader
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # Adjust batch size as needed

# Batch prediction
all_probs = []
all_labels = []

print("Running evaluation...")
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        print(f"Processing batch {i + 1}/{len(val_loader)}")
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
preds = (all_probs > 0.5).astype(int)
true_labels = all_labels

# Compute metrics with zero_division handling
metrics = {
    'f1_micro': f1_score(true_labels, preds, average='micro'),
    'f1_macro': f1_score(true_labels, preds, average='macro'),
    'precision_micro': precision_score(true_labels, preds, average='micro', zero_division=0),
    'recall_micro': recall_score(true_labels, preds, average='micro', zero_division=0),
    'hamming_loss': hamming_loss(true_labels, preds),
    'jaccard_micro': jaccard_score(true_labels, preds, average='micro'),
    'subset_accuracy': accuracy_score(true_labels, preds),
}

# Classification report with zero_division
class_report = classification_report(
    true_labels,
    preds,
    target_names=emotion_labels,
    output_dict=True,
    zero_division=0
)
class_report_df = pd.DataFrame(class_report).transpose()

# Create output directory
os.makedirs("results", exist_ok=True)

# Save metrics with UTF-8 encoding
with open("results/"+MODEL_NAME+"/evaluation_metrics.txt", "w", encoding="utf-8") as f:
    f.write("📊 Evaluation Metrics on Validation Set:\n")
    for k, v in metrics.items():
        f.write(f"- {k}: {v:.4f}\n")

# Save classification report
class_report_df.to_csv("results/"+MODEL_NAME+"/classification_report.csv")

# ... [after saving metrics] ...

# 1. Plot emotion distribution
plt.figure(figsize=(15, 8))
support = class_report_df.loc[emotion_labels, "support"].astype(int)
sns.barplot(x=emotion_labels, y=support.values)
plt.title("Class Distribution in Validation Set")
plt.xticks(rotation=90)
plt.ylabel("Number of Samples")
plt.savefig("results/"+MODEL_NAME+"/class_distribution.png", bbox_inches='tight')
plt.close()

# 2. Plot performance metrics per emotion
metrics_df = class_report_df.loc[emotion_labels, ["precision", "recall", "f1-score"]]
metrics_df = metrics_df.reset_index().melt(id_vars="index", var_name="metric")

plt.figure(figsize=(18, 10))
sns.barplot(data=metrics_df, x="index", y="value", hue="metric")
plt.title("Performance Metrics per Emotion")
plt.xticks(rotation=90)
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.savefig("results/"+MODEL_NAME+"/per_emotion_metrics.png", bbox_inches='tight')
plt.close()

# 3. Plot confusion matrix for top emotions (simplified)
top_emotions = ["neutral", "admiration", "gratitude", "anger", "joy", "love"]
top_indices = [emotion_labels.index(e) for e in top_emotions]

# Create simplified confusion matrix
conf_matrix = np.zeros((len(top_emotions), len(top_emotions)))
for i, true_idx in enumerate(top_indices):
    for j, pred_idx in enumerate(top_indices):
        conf_matrix[i, j] = np.logical_and(
            true_labels[:, true_idx],
            preds[:, pred_idx]
        ).sum()

plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt=".0f",
    cmap="Blues",
    xticklabels=top_emotions,
    yticklabels=top_emotions
)
plt.title("Confusion Matrix (Top Emotions)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("results/"+MODEL_NAME+"/confusion_matrix_top.png", bbox_inches='tight')
plt.close()

# 4. Plot threshold analysis for a key emotion
emotion = "grief"
idx = emotion_labels.index(emotion)

thresholds = np.linspace(0.1, 0.9, 9)
f1_scores = []
for thresh in thresholds:
    preds_thresh = (all_probs[:, idx] > thresh).astype(int)
    f1 = f1_score(true_labels[:, idx], preds_thresh, zero_division=0)
    f1_scores.append(f1)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, marker='o')
plt.title(f"F1-Score vs Threshold for {emotion}")
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.grid(True)
plt.savefig(f"results/"+MODEL_NAME+"/threshold_analysis_{emotion}.png", bbox_inches='tight')
plt.close()

print("\n📈 Generated visualizations:")
print(f"- Class distribution: results/"+MODEL_NAME+"/class_distribution.png")
print(f"- Per-emotion metrics: results/"+MODEL_NAME+"/per_emotion_metrics.png")
print(f"- Confusion matrix (top emotions): results/"+MODEL_NAME+"/confusion_matrix_top.png")
print(f"- Threshold analysis: results/"+MODEL_NAME+"/threshold_analysis_{emotion}.png")