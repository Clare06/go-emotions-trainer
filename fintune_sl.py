# srilanka_emotion_finetuner.py
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
import warnings
from nlpaug.augmenter.word import SynonymAug, ContextualWordEmbsAug
warnings.filterwarnings('ignore')

# Configuration
PRETRAINED_MODEL_PATH = "saved_model_xlm_roberta_negation_head_only_20250722_151517"  # Replace with your trained model path
BATCH_SIZE = 16  # Smaller for limited GPU memory
MAX_LENGTH = 128
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'  # No neutral as requested
]

# Optimal thresholds from your previous training
OPTIMAL_THRESHOLDS = {
    'admiration': 0.4, 'amusement': 0.4, 'anger': 0.3, 'annoyance': 0.1,
    'approval': 0.3, 'caring': 0.3, 'confusion': 0.2, 'curiosity': 0.3,
    'desire': 0.2, 'disappointment': 0.2, 'disapproval': 0.2, 'disgust': 0.3,
    'embarrassment': 0.2, 'excitement': 0.3, 'fear': 0.4, 'gratitude': 0.5,
    'grief': 0.1, 'joy': 0.3, 'love': 0.4, 'nervousness': 0.4,
    'optimism': 0.4, 'pride': 0.2, 'realization': 0.1, 'relief': 0.4,
    'remorse': 0.4, 'sadness': 0.3, 'surprise': 0.3
}

# Generate timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_OUTPUT_DIR = f"srilanka_emotion_model_{TIMESTAMP}"
RESULTS_DIR = f"./srilanka_results_{TIMESTAMP}"
LOGS_DIR = f"./srilanka_logs_{TIMESTAMP}"

print(f"Sri Lanka Fine-tuning Session: {TIMESTAMP}")
print(f"Loading base model from: {PRETRAINED_MODEL_PATH}")
print(f"Fine-tuned model will be saved to: {MODEL_OUTPUT_DIR}")


class SriLankaEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Clean and preprocess text
        text = text.strip()

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


def multi_label_metrics(predictions, labels, thresholds=None):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    if thresholds is None:
        # Use default threshold
        preds = (probs > 0.5).int()
    else:
        # Use optimal thresholds
        preds = torch.zeros_like(probs)
        for i, emotion in enumerate(EMOTIONS):
            if emotion in thresholds:
                preds[:, i] = (probs[:, i] > thresholds[emotion]).int()
            else:
                preds[:, i] = (probs[:, i] > 0.5).int()

    # Calculate metrics
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    # Calculate per-emotion F1 scores
    f1_per_emotion = f1_score(labels, preds, average=None, zero_division=0)

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_emotion': f1_per_emotion.tolist()
    }


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    metrics = multi_label_metrics(preds, p.label_ids, OPTIMAL_THRESHOLDS)
    return {
        'f1_micro': metrics['f1_micro'],
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted']
    }


# Load Sri Lankan dataset
print("Loading Sri Lankan emotion dataset...")
srilanka_df = pd.read_csv('dataset/cleaned_emotion_data_ef.csv')

# Ensure we have the required emotion columns (excluding neutral)
available_emotions = [col for col in EMOTIONS if col in srilanka_df.columns]
print(f"Available emotions in dataset: {len(available_emotions)}")
print(f"Missing emotions: {set(EMOTIONS) - set(available_emotions)}")
syn_aug = SynonymAug(aug_src='wordnet')
# Contextual word replacement
contextual_aug = ContextualWordEmbsAug(model_path='xlm-roberta-base', action="substitute")
# Split into train/validation (80/20 split for small dataset)
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    srilanka_df['text'].tolist(),
    srilanka_df[available_emotions].values,
    test_size=0.2,
    random_state=42,
    stratify=None  # Can't stratify with multi-label
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Load pre-trained model and tokenizer
print("Loading pre-trained emotion model...")
tokenizer = XLMRobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)

# First, load the model with original configuration (including neutral)
original_model = XLMRobertaForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_PATH,
    problem_type="multi_label_classification"
)

# Create new model with correct number of labels (27 without neutral)
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",  # Load base model
    num_labels=len(available_emotions),
    problem_type="multi_label_classification"
)

# Transfer weights from original model (excluding neutral-related weights)
print("Transferring weights from pre-trained model (excluding neutral)...")

# Copy all roberta layers
model.roberta.load_state_dict(original_model.roberta.state_dict())

# Copy classifier weights but exclude neutral (assuming it's the last emotion)
original_classifier_weight = original_model.classifier.out_proj.weight.data
original_classifier_bias = original_model.classifier.out_proj.bias.data

# Map emotions to indices (excluding neutral)
emotion_mapping = []
original_emotions_with_neutral = EMOTIONS + ['neutral']  # Your original training had neutral
for emotion in available_emotions:
    if emotion in original_emotions_with_neutral:
        emotion_mapping.append(original_emotions_with_neutral.index(emotion))

print(f"Emotion mapping: {list(zip(available_emotions, emotion_mapping))}")

# Copy weights for available emotions
model.classifier.out_proj.weight.data = original_classifier_weight[emotion_mapping, :]
model.classifier.out_proj.bias.data = original_classifier_bias[emotion_mapping]

# Copy dense layer if it exists
if hasattr(original_model.classifier, 'dense') and hasattr(model.classifier, 'dense'):
    model.classifier.dense.load_state_dict(original_model.classifier.dense.state_dict())

print("Weight transfer complete!")
del original_model  # Free memory

# Freeze early layers (0-8) to preserve general linguistic knowledge
print("Freezing early transformer layers (0-8)...")
for name, param in model.named_parameters():
    if 'roberta.encoder.layer.' in name:
        layer_num = int(name.split('roberta.encoder.layer.')[1].split('.')[0])
        if layer_num <= 8:  # Freeze layers 0-8
            param.requires_grad = False
            print(f"Frozen: {name}")

# Keep layers 9-11 and classifier trainable
print("Keeping layers 9-11 and classifier trainable...")
trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"Trainable: {name}")

print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# Create datasets
train_dataset = SriLankaEmotionDataset(
    train_texts, train_labels, tokenizer, MAX_LENGTH
)
val_dataset = SriLankaEmotionDataset(
    val_texts, val_labels, tokenizer, MAX_LENGTH
)

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Training arguments optimized for small dataset and RTX 4050
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=10,  # More epochs for small dataset
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=2e-6,  # Very low learning rate to prevent catastrophic forgetting
    warmup_steps=50,  # Reduced warmup for small dataset
    weight_decay=0.01,
    logging_dir=LOGS_DIR,
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    fp16=True,
    gradient_accumulation_steps=8,  # Increase for effective larger batch size
    gradient_checkpointing=True,  # Save memory
    dataloader_pin_memory=False,  # Save GPU memory
    remove_unused_columns=True,
    report_to='none',
    seed=42
)


# Custom Trainer for better handling
# Custom Trainer for better handling
class SriLankaEmotionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Fixed compute_loss method that handles the num_items_in_batch parameter
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Use BCEWithLogitsLoss for multi-label classification
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Initialize trainer
trainer = SriLankaEmotionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Start fine-tuning
print("Starting Sri Lanka context fine-tuning...")
print("=" * 60)
print(f"Dataset size: {len(srilanka_df)} samples")
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Emotions: {available_emotions}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {BATCH_SIZE * training_args.gradient_accumulation_steps}")
print("=" * 60)

# Train
trainer.train()

# Evaluate with optimal thresholds
print("\nEvaluating with optimal thresholds...")
predictions = trainer.predict(val_dataset)
final_metrics = multi_label_metrics(
    predictions.predictions,
    predictions.label_ids,
    OPTIMAL_THRESHOLDS
)

print(f"Final F1 Micro: {final_metrics['f1_micro']:.4f}")
print(f"Final F1 Macro: {final_metrics['f1_macro']:.4f}")
print(f"Final F1 Weighted: {final_metrics['f1_weighted']:.4f}")

# Per-emotion performance
print("\nPer-emotion F1 scores:")
for i, emotion in enumerate(available_emotions):
    if i < len(final_metrics['f1_per_emotion']):
        print(f"{emotion}: {final_metrics['f1_per_emotion'][i]:.4f}")

# Save model
print(f"\nSaving fine-tuned model to: {MODEL_OUTPUT_DIR}")
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

# Save configuration
config_file = f"{MODEL_OUTPUT_DIR}/finetuning_config.txt"
with open(config_file, 'w') as f:
    f.write(f"Sri Lanka Fine-tuning Configuration\n")
    f.write(f"=====================================\n")
    f.write(f"Timestamp: {TIMESTAMP}\n")
    f.write(f"Base Model: {PRETRAINED_MODEL_PATH}\n")
    f.write(f"Dataset Size: {len(srilanka_df)} samples\n")
    f.write(f"Training Samples: {len(train_texts)}\n")
    f.write(f"Validation Samples: {len(val_texts)}\n")
    f.write(f"Frozen Layers: 0-8\n")
    f.write(f"Trainable Layers: 9-11 + classifier\n")
    f.write(f"Learning Rate: {training_args.learning_rate}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Gradient Accumulation: {training_args.gradient_accumulation_steps}\n")
    f.write(f"Epochs: {training_args.num_train_epochs}\n")
    f.write(f"Available Emotions: {available_emotions}\n")
    f.write(f"\nFinal Metrics:\n")
    f.write(f"F1 Micro: {final_metrics['f1_micro']:.4f}\n")
    f.write(f"F1 Macro: {final_metrics['f1_macro']:.4f}\n")
    f.write(f"F1 Weighted: {final_metrics['f1_weighted']:.4f}\n")

    f.write(f"\nOptimal Thresholds Used:\n")
    for emotion, threshold in OPTIMAL_THRESHOLDS.items():
        if emotion in available_emotions:
            f.write(f"{emotion}: {threshold}\n")

print(f"Fine-tuning complete!")
print(f"Model saved to: {MODEL_OUTPUT_DIR}")
print(f"Configuration saved to: {config_file}")

# Test with sample texts
# Test with sample texts
print("\nTesting with sample texts...")
sample_texts = [
    "මට ඔබගේ සහාය ගැන ස්තූතියි",  # Sinhala: Thank you for your help
    "இது மிகவும் அருமையான செய்தி",  # Tamil: This is very good news
    "This is absolutely amazing work machan!",  # Sri Lankan English
    "අපි කණගාටුවෙන් ඉන්නවා මේ ගැන"  # Sinhala: We are sad about this
]

# Ensure model is in evaluation mode and get device
model.eval()
device = next(model.parameters()).device
print(f"Model device: {device}")

with torch.no_grad():
    for text in sample_texts:
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze()

        print(f"\nText: {text}")
        predicted_emotions = []
        for i, emotion in enumerate(available_emotions):
            prob = probs[i].item()
            threshold = OPTIMAL_THRESHOLDS.get(emotion, 0.5)
            if prob > threshold:
                predicted_emotions.append(f"{emotion}({prob:.3f})")

        print(f"Predicted: {', '.join(predicted_emotions) if predicted_emotions else 'No emotions above threshold'}")