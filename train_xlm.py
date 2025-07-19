# train_goemotions_roberta.py

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, jaccard_score

# Load datasets
df1 = pd.read_csv("dataset/goemotions_1.csv")
df2 = pd.read_csv("dataset/goemotions_2.csv")
df3 = pd.read_csv("dataset/goemotions_3.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

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
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenizer & Encoding
MODEL_NAME = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = GoEmotionsDataset(train_enc, train_labels)
val_dataset = GoEmotionsDataset(val_enc, val_labels)

# Model definition
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(emotion_labels),
    problem_type="multi_label_classification"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Metrics
def compute_metrics(p):
    logits, labels = p.predictions, p.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    return {
        'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'precision_micro': precision_score(labels, preds, average='micro', zero_division=0),
        'recall_micro': recall_score(labels, preds, average='micro', zero_division=0),
        'hamming_loss': hamming_loss(labels, preds),
        'jaccard_micro': jaccard_score(labels, preds, average='micro', zero_division=0),
        'subset_accuracy': accuracy_score(labels, preds)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_xlm_roberta",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    # save_steps = 1000,  # Or 500
    eval_strategy = "epoch",
    # eval_steps = 1000,  # Match s
    logging_dir="./logs_xlm",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    greater_is_better=True,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=2,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()

# Save model
model_path = f"saved_model_{MODEL_NAME.replace('/', '_')}"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Prediction function
def predict_emotion(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    top_indices = probs.argsort()[-3:][::-1]
    print("\nInput:", text)
    print("Top predicted emotions:")
    for idx in top_indices:
        print(f"- {emotion_labels[idx]} ({probs[idx]*100:.1f}%)")

# Sample test
predict_emotion("I feel so disappointed but also kind of relieved.")
