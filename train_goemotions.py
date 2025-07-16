# train_goemotions.py

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load all 3 datasets
df1 = pd.read_csv("dataset/goemotions_1.csv")
df2 = pd.read_csv("dataset/goemotions_2.csv")
df3 = pd.read_csv("dataset/goemotions_3.csv")

# Combine
df = pd.concat([df1, df2, df3], ignore_index=True)

# Get emotion columns (all except 'text')
# Define emotion labels manually (GoEmotions original 28)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Now extract text + clean emotion columns
texts = df["text"].tolist()
labels = df[emotion_labels].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values


# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Dataset class
class GoEmotionsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = GoEmotionsDataset(train_enc, train_labels)
val_dataset = GoEmotionsDataset(val_enc, val_labels)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(emotion_labels),
    problem_type="multi_label_classification"
)

# Move model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save final model
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

# Prediction function
def predict_emotion(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to GPU
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # bring back to CPU for numpy

    top_indices = probs.argsort()[-3:][::-1]
    print("\nInput:", text)
    print("Top predicted emotions:")
    for idx in top_indices:
        print(f"- {emotion_labels[idx]} ({probs[idx]*100:.1f}%)")

# Try a sample
predict_emotion("I feel so disappointed but also kind of relieved.")

