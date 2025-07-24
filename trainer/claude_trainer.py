# emotion_trainer.py
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os

# Configuration
MODEL_NAME = "xlm-roberta-base"
BATCH_SIZE = 8
MAX_LENGTH = 128
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Generate timestamp for unique model naming
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_OUTPUT_DIR = f"emotion_model_{TIMESTAMP}"
RESULTS_DIR = f"./results_{TIMESTAMP}"
LOGS_DIR = f"./logs_{TIMESTAMP}"

print(f"Training session: {TIMESTAMP}")
print(f"Model will be saved to: {MODEL_OUTPUT_DIR}")

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Metric Calculation
def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    preds = (probs > threshold).int()
    
    # Calculate metrics
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': accuracy
    }

# Compute Metrics Function
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return multi_label_metrics(preds, p.label_ids)

# Load datasets
train_df = pd.read_csv('../dataset/goemotions_train.csv')
val_df = pd.read_csv('../dataset/goemotions_val.csv')

# Initialize tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

# Create datasets
train_dataset = EmotionDataset(
    train_df['text'].tolist(),
    train_df[EMOTIONS].values,
    tokenizer,
    MAX_LENGTH
)

val_dataset = EmotionDataset(
    val_df['text'].tolist(),
    val_df[EMOTIONS].values,
    tokenizer,
    MAX_LENGTH
)

# Load model
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTIONS),
    problem_type="multi_label_classification"
)

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Training arguments optimized for RTX 4050 6GB with timestamped directories
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=LOGS_DIR,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    fp16=True,
    gradient_accumulation_steps=4,
    report_to='none'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Start training
print("Starting training...")
print(f"Training outputs will be saved to: {RESULTS_DIR}")
print(f"Logs will be saved to: {LOGS_DIR}")

trainer.train()

# Save final model with timestamp
print(f"Saving model to: {MODEL_OUTPUT_DIR}")
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

# Save training summary
summary_file = f"{MODEL_OUTPUT_DIR}/training_summary.txt"
with open(summary_file, 'w') as f:
    f.write(f"Training Summary\n")
    f.write(f"================\n")
    f.write(f"Timestamp: {TIMESTAMP}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Max Length: {MAX_LENGTH}\n")
    f.write(f"Number of Emotions: {len(EMOTIONS)}\n")
    f.write(f"Training Data: {len(train_dataset)} samples\n")
    f.write(f"Validation Data: {len(val_dataset)} samples\n")
    f.write(f"Results Directory: {RESULTS_DIR}\n")
    f.write(f"Logs Directory: {LOGS_DIR}\n")

print(f"Training complete. Model saved to: {MODEL_OUTPUT_DIR}")
print(f"Training summary saved to: {summary_file}")
print(f"Previous models are preserved and not overwritten.")