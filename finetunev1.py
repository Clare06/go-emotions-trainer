# Enhanced fine-tuning with layer protection for small datasets
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import os

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
class NegationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def freeze_lower_layers(model, layers_to_freeze=6):
    """
    Freeze the first N layers to protect learned representations
    XLM-RoBERTa-base has 12 layers (0-11)
    """
    print(f"🔒 Freezing first {layers_to_freeze} layers...")

    # Freeze embeddings
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    # Freeze specified encoder layers
    for layer_idx in range(layers_to_freeze):
        for param in model.roberta.encoder.layer[layer_idx].parameters():
            param.requires_grad = False

    # Keep classification head and upper layers trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")
    return model


def gradual_unfreezing_fine_tune(original_model_path, negation_csv_path):
    """
    Fine-tune with gradual unfreezing strategy for small datasets
    """

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    print("🔧 Loading negation training data...")
    df = pd.read_csv(negation_csv_path)

    # Prepare data
    texts = df["text"].tolist()
    labels = df[emotion_labels].values.astype(float)

    print(f"📊 Loaded {len(texts)} negation examples")

    # Load model and tokenizer
    print("🤖 Loading your trained model...")
    model = XLMRobertaForSequenceClassification.from_pretrained(original_model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(original_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize data
    print("🔤 Tokenizing negation examples...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    negation_dataset = NegationDataset(encodings, labels)

    # Strategy 1: Freeze lower layers (most conservative)
    model_frozen = freeze_lower_layers(model, layers_to_freeze=8)  # Freeze first 8 layers

    # Very conservative training args for small dataset
    training_args_conservative = TrainingArguments(
        output_dir="./results_xlm_roberta_negation_conservative",
        num_train_epochs=2,  # Very few epochs
        per_device_train_batch_size=4,  # Small batch size
        per_device_eval_batch_size=4,
        save_strategy="epoch",
        logging_dir="./logs_xlm_negation_conservative",
        logging_steps=5,
        learning_rate=5e-6,  # Very low learning rate
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        warmup_steps=5,
        dataloader_drop_last=False,
        gradient_accumulation_steps=4,  # Simulate larger batch
    )

    def compute_metrics(p):
        logits, labels = p.predictions, p.label_ids
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)

        return {
            'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'accuracy': accuracy_score(labels, preds)
        }

    # Phase 1: Train with frozen lower layers
    print("🚀 Phase 1: Training with frozen lower layers...")
    trainer = Trainer(
        model=model_frozen,
        args=training_args_conservative,
        train_dataset=negation_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save conservative model
    conservative_path = f"saved_model_xlm_roberta_negation_conservative_{TIMESTAMP}"
    model.save_pretrained(conservative_path)
    tokenizer.save_pretrained(conservative_path)

    print(f"✅ Conservative fine-tuning complete!")
    print(f"📁 Model saved to: {conservative_path}")

    return conservative_path


def adapter_based_fine_tune(original_model_path, negation_csv_path):
    """
    Alternative: Use adapter-like approach by only training classification head
    """

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    print("🎯 Adapter-style fine-tuning (classification head only)...")

    # Load data
    df = pd.read_csv(negation_csv_path)
    texts = df["text"].tolist()
    labels = df[emotion_labels].values.astype(float)

    # Load model
    model = XLMRobertaForSequenceClassification.from_pretrained(original_model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(original_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze ENTIRE roberta backbone - only train classifier
    print("🔒 Freezing entire RoBERTa backbone...")
    for param in model.roberta.parameters():
        param.requires_grad = False

    # Keep only classifier trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")

    # Tokenize
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    negation_dataset = NegationDataset(encodings, labels)

    # Training args for head-only training
    training_args_head_only = TrainingArguments(
        output_dir=f"./results_xlm_roberta_negation_head_only_{TIMESTAMP}",
        num_train_epochs=5,  # Can train longer since only head
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="epoch",
        logging_dir="./logs_xlm_negation_head_only",
        logging_steps=5,
        learning_rate=1e-4,  # Higher LR for classification head
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        warmup_steps=10,
        dataloader_drop_last=False,
    )

    def compute_metrics(p):
        logits, labels = p.predictions, p.label_ids
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)
        return {
            'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args_head_only,
        train_dataset=negation_dataset,
        compute_metrics=compute_metrics
    )

    print("🚀 Training classification head only...")
    trainer.train()

    # Save head-only model
    head_only_path = f"saved_model_xlm_roberta_negation_head_only_{TIMESTAMP}"
    model.save_pretrained(head_only_path)
    tokenizer.save_pretrained(head_only_path)

    print(f"✅ Head-only fine-tuning complete!")
    print(f"📁 Model saved to: {head_only_path}")

    return head_only_path


def data_augmentation_for_negation(original_csv_path):
    """
    Augment the small negation dataset to make it larger and more robust
    """

    print("🔄 Augmenting negation dataset...")

    # Load original negation data
    df = pd.read_csv(original_csv_path)

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    augmented_examples = []

    # Add paraphrase variations
    negation_patterns = [
        "I am not {emotion}",
        "I'm not {emotion}",
        "I don't feel {emotion}",
        "I never feel {emotion}",
        "This doesn't make me {emotion}",
        "I'm no longer {emotion}",
        "I refuse to be {emotion}",
        "I won't be {emotion}",
        "I can't be {emotion}",
        "I'm definitely not {emotion}",
        "I absolutely don't feel {emotion}",
        "Not feeling {emotion} at all",
        "Zero {emotion} here",
        "No {emotion} from me"
    ]

    emotion_adjectives = {
        'sadness': ['sad', 'unhappy', 'down', 'blue', 'depressed'],
        'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed'],
        'fear': ['afraid', 'scared', 'terrified', 'frightened', 'worried'],
        'joy': ['happy', 'joyful', 'cheerful', 'glad', 'delighted'],
        'excitement': ['excited', 'thrilled', 'enthusiastic', 'pumped'],
        'disappointment': ['disappointed', 'let down', 'discouraged'],
        'disgust': ['disgusted', 'repulsed', 'revolted', 'sickened'],
        'nervousness': ['nervous', 'anxious', 'worried', 'tense']
    }

    # Generate augmented examples
    for emotion, adjectives in emotion_adjectives.items():
        for adj in adjectives:
            for pattern in negation_patterns[:8]:  # Use first 8 patterns
                text = pattern.format(emotion=adj)

                # Create label vector
                row = {"text": text}
                for label in emotion_labels:
                    row[label] = 0

                # Set appropriate labels for negation
                row['neutral'] = 1
                if emotion in ['sadness', 'anger', 'fear', 'disappointment', 'disgust', 'nervousness']:
                    row['relief'] = 1
                elif emotion in ['joy', 'excitement']:
                    row['disappointment'] = 0.5  # Mild disappointment

                augmented_examples.append(row)

    # Create augmented dataframe
    augmented_df = pd.DataFrame(augmented_examples)

    # Combine with original
    combined_df = pd.concat([df, augmented_df], ignore_index=True)

    # Save augmented dataset
    augmented_path = "negation_training_data_augmented.csv"
    combined_df.to_csv(augmented_path, index=False)

    print(f"✅ Dataset augmented from {len(df)} to {len(combined_df)} examples")
    print(f"📁 Saved to: {augmented_path}")

    return augmented_path, combined_df


# Main execution with protection strategies
if __name__ == "__main__":

    ORIGINAL_MODEL_PATH = "emotion_model_20250722_050505"
    NEGATION_CSV_PATH = "negation_training_data.csv"

    if not os.path.exists(ORIGINAL_MODEL_PATH):
        print(f"❌ Original model not found at: {ORIGINAL_MODEL_PATH}")
        exit()

    print("🎯 Choose fine-tuning strategy:")
    print("1. Conservative (freeze lower layers)")
    print("2. Head-only (freeze entire backbone)")
    print("3. Augment data first, then conservative")

    strategy = input("Enter choice (1/2/3): ").strip()

    if strategy == "1":
        print("\n🔒 Using conservative layer-freezing approach...")
        model_path = gradual_unfreezing_fine_tune(ORIGINAL_MODEL_PATH, NEGATION_CSV_PATH)

    elif strategy == "2":
        print("\n🎯 Using head-only training approach...")
        model_path = adapter_based_fine_tune(ORIGINAL_MODEL_PATH, NEGATION_CSV_PATH)

    elif strategy == "3":
        print("\n🔄 First augmenting dataset...")
        augmented_path, _ = data_augmentation_for_negation(NEGATION_CSV_PATH)
        print("\n🔒 Now using conservative approach with augmented data...")
        model_path = gradual_unfreezing_fine_tune(ORIGINAL_MODEL_PATH, augmented_path)

    else:
        print("❌ Invalid choice")
        exit()

    print(f"\n✅ Fine-tuning complete! Model saved to: {model_path}")
    print("\n💡 Recommendations for small datasets:")
    print("- Use validation set to monitor for overfitting")
    print("- Test on original GoEmotions test set to check for catastrophic forgetting")
    print("- Consider ensemble methods or regularization")
    print("- Monitor both negation performance AND original emotion detection")