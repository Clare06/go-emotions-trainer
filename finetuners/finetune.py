# Generate negation training dataset and fine-tune model

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, jaccard_score
import os


# Step 1: Generate Negation Training Data
def generate_negation_dataset():
    """Generate comprehensive negation training examples"""

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    # Negation examples with proper labels
    negation_examples = [
        # Sadness negations → should be neutral/relief, NOT sadness
        {"text": "I am not sad", "sadness": 0, "neutral": 1, "relief": 1},
        {"text": "I'm not feeling sad anymore", "sadness": 0, "relief": 1, "neutral": 1},
        {"text": "I don't feel sad about this", "sadness": 0, "neutral": 1},
        {"text": "This doesn't make me sad", "sadness": 0, "neutral": 1},
        {"text": "I'm not unhappy", "sadness": 0, "neutral": 1, "relief": 1},
        {"text": "I never felt sad", "sadness": 0, "neutral": 1},
        {"text": "Without sadness, I moved on", "sadness": 0, "relief": 1, "neutral": 1},
        {"text": "I'm no longer sad", "sadness": 0, "relief": 1, "neutral": 1},

        # Anger negations → should be neutral/relief, NOT anger
        {"text": "I am not angry", "anger": 0, "neutral": 1, "relief": 1},
        {"text": "I'm not mad at you", "anger": 0, "neutral": 1, "caring": 1},
        {"text": "This doesn't make me angry", "anger": 0, "neutral": 1},
        {"text": "I never get angry about this", "anger": 0, "neutral": 1},
        {"text": "I'm not furious", "anger": 0, "neutral": 1, "relief": 1},
        {"text": "Without anger, I approached the problem", "anger": 0, "neutral": 1, "approval": 1},
        {"text": "I don't feel angry anymore", "anger": 0, "relief": 1, "neutral": 1},

        # Fear negations → should be neutral/relief, NOT fear
        {"text": "I am not afraid", "fear": 0, "neutral": 1, "relief": 1},
        {"text": "I'm not scared", "fear": 0, "neutral": 1, "relief": 1},
        {"text": "This doesn't frighten me", "fear": 0, "neutral": 1, "approval": 1},
        {"text": "I never felt scared", "fear": 0, "neutral": 1},
        {"text": "Without fear, I took the leap", "fear": 0, "relief": 1, "optimism": 1},
        {"text": "I'm no longer terrified", "fear": 0, "relief": 1, "neutral": 1},

        # Excitement negations → should be neutral/disappointment, NOT excitement
        {"text": "I am not excited", "excitement": 0, "neutral": 1, "disappointment": 1},
        {"text": "I'm not thrilled about this", "excitement": 0, "disappointment": 1, "neutral": 1},
        {"text": "This doesn't excite me", "excitement": 0, "neutral": 1},
        {"text": "I never felt excited", "excitement": 0, "neutral": 1},
        {"text": "I'm not enthusiastic", "excitement": 0, "neutral": 1, "disappointment": 1},

        # Joy/Happiness negations → should be neutral/sadness, NOT joy
        {"text": "I am not happy", "joy": 0, "neutral": 1, "sadness": 1},
        {"text": "This doesn't make me happy", "joy": 0, "neutral": 1, "disappointment": 1},
        {"text": "I'm not joyful", "joy": 0, "neutral": 1, "sadness": 1},
        {"text": "I never felt happy about it", "joy": 0, "neutral": 1, "disappointment": 1},
        {"text": "Without joy, I continued", "joy": 0, "neutral": 1, "sadness": 1},

        # Disappointment negations
        {"text": "I am not disappointed", "disappointment": 0, "neutral": 1, "relief": 1},
        {"text": "This doesn't disappoint me", "disappointment": 0, "neutral": 1, "approval": 1},
        {"text": "I'm not let down", "disappointment": 0, "neutral": 1, "relief": 1},
        {"text": "I never felt disappointed", "disappointment": 0, "neutral": 1, "approval": 1},

        # Disgust negations
        {"text": "I am not disgusted", "disgust": 0, "neutral": 1, "relief": 1},
        {"text": "This doesn't disgust me", "disgust": 0, "neutral": 1, "approval": 1},
        {"text": "I'm not repulsed", "disgust": 0, "neutral": 1},

        # Nervousness negations
        {"text": "I am not nervous", "nervousness": 0, "neutral": 1, "relief": 1},
        {"text": "I'm not anxious", "nervousness": 0, "neutral": 1, "relief": 1},
        {"text": "This doesn't make me nervous", "nervousness": 0, "neutral": 1},

        # Approval negations
        {"text": "I do not approve", "approval": 0, "disapproval": 1, "neutral": 1},
        {"text": "I'm not impressed", "approval": 0, "disappointment": 1, "neutral": 1},
        {"text": "This doesn't get my approval", "approval": 0, "disapproval": 1},

        # Multiple negations in complex sentences
        {"text": "I'm not sad, just not excited either", "sadness": 0, "excitement": 0, "neutral": 1},
        {"text": "I don't feel angry or disappointed", "anger": 0, "disappointment": 0, "neutral": 1, "relief": 1},
        {"text": "I'm neither happy nor sad about this", "joy": 0, "sadness": 0, "neutral": 1},
        {"text": "I'm not scared but I'm not confident", "fear": 0, "neutral": 1, "nervousness": 1},

        # Positive examples (control group - no negation)
        {"text": "I am sad", "sadness": 1, "neutral": 0},
        {"text": "I feel angry", "anger": 1, "neutral": 0},
        {"text": "I am excited", "excitement": 1, "neutral": 0},
        {"text": "I feel happy", "joy": 1, "neutral": 0},
        {"text": "I am disappointed", "disappointment": 1, "neutral": 0},
        {"text": "I feel scared", "fear": 1, "neutral": 0},
        {"text": "I am disgusted", "disgust": 1, "neutral": 0},
        {"text": "I feel nervous", "nervousness": 1, "neutral": 0},

        # Contrasting pairs
        {"text": "I was sad but I'm not sad anymore", "sadness": 0, "relief": 1, "neutral": 1},
        {"text": "I used to be angry but not now", "anger": 0, "relief": 1, "neutral": 1},
        {"text": "I'm relieved I'm not scared", "fear": 0, "relief": 1, "neutral": 1},
    ]

    # Convert to DataFrame format
    rows = []
    for example in negation_examples:
        row = {"text": example["text"]}
        # Initialize all emotions to 0
        for emotion in emotion_labels:
            row[emotion] = 0
        # Set specified emotions to their values
        for emotion, value in example.items():
            if emotion != "text" and emotion in emotion_labels:
                row[emotion] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# Step 2: Save the dataset
def save_negation_dataset():
    """Generate and save the negation dataset"""
    df = generate_negation_dataset()
    df.to_csv("negation_training_data.csv", index=False)
    print(f"✅ Generated {len(df)} negation training examples")
    print(f"📁 Saved to: negation_training_data.csv")
    print("\n📊 Sample rows:")
    print(df.head(3).to_string())
    return df


# Step 3: Fine-tuning code
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


def fine_tune_with_negation_data(original_model_path, negation_csv_path):
    """Fine-tune existing model with negation data"""

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

    # Load pre-trained model and tokenizer
    print("🤖 Loading your trained model...")
    model = XLMRobertaForSequenceClassification.from_pretrained(original_model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(original_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize negation data
    print("🔤 Tokenizing negation examples...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    # Create dataset
    negation_dataset = NegationDataset(encodings, labels)

    # Fine-tuning arguments (lighter training since we're fine-tuning)
    training_args = TrainingArguments(
        output_dir="./results_xlm_roberta_negation_finetuned",
        num_train_epochs=3,  # Fewer epochs for fine-tuning
        per_device_train_batch_size=16,  # Can use larger batch for small dataset
        per_device_eval_batch_size=16,
        save_strategy="epoch",
        logging_dir="./logs_xlm_negation",
        logging_steps=5,
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        warmup_steps=10,  # Short warmup
        dataloader_drop_last=False,  # Important for small dataset
    )

    # Metrics function
    def compute_metrics(p):
        logits, labels = p.predictions, p.label_ids
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)

        return {
            'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'accuracy': accuracy_score(labels, preds)
        }

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=negation_dataset,
        compute_metrics=compute_metrics
    )

    print("🚀 Starting negation fine-tuning...")
    trainer.train()

    # Save the fine-tuned model
    final_model_path = "saved_model_xlm_roberta_negation_aware"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"✅ Fine-tuning complete!")
    print(f"📁 Model saved to: {final_model_path}")

    return final_model_path


def test_negation_improvements(model_path):
    """Test the fine-tuned model on negation examples"""

    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    # Load fine-tuned model
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict_emotion(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        return probs

    # Test cases
    test_cases = [
        "I am not sad",
        "I'm not angry",
        "I am sad",  # Control
        "I feel angry",  # Control
        "This doesn't make me excited",
        "I'm not disappointed",
    ]

    print("\n🧪 Testing fine-tuned model on negation cases:")
    print("=" * 60)

    for text in test_cases:
        probs = predict_emotion(text)
        top_indices = probs.argsort()[-3:][::-1]

        print(f"\nInput: '{text}'")
        print("Top 3 predictions:")
        for idx in top_indices:
            print(f"  {emotion_labels[idx]}: {probs[idx] * 100:.1f}%")


# Main execution
if __name__ == "__main__":
    print("🎯 Generating Negation Training Dataset...")
    negation_df = save_negation_dataset()

    # print("\n" + "=" * 60)
    # print("🚀 Starting Fine-tuning Process...")
    #
    # # Update this path to your trained model
    # ORIGINAL_MODEL_PATH = "saved_model_xlm-roberta-base"  # Your existing model
    # NEGATION_CSV_PATH = "negation_training_data.csv"
    #
    # if os.path.exists(ORIGINAL_MODEL_PATH):
    #     fine_tuned_path = fine_tune_with_negation_data(ORIGINAL_MODEL_PATH, NEGATION_CSV_PATH)
    #
    #     print("\n" + "=" * 60)
    #     print("🧪 Testing the fine-tuned model...")
    #     test_negation_improvements(fine_tuned_path)
    # else:
    #     print(f"❌ Original model not found at: {ORIGINAL_MODEL_PATH}")
    #     print("Please update ORIGINAL_MODEL_PATH to point to your trained model directory")
    #

# Additional helper: Combine with original dataset
def combine_with_original_dataset(original_csv_paths, negation_csv_path, output_path):
    """Combine negation data with original GoEmotions data for full retraining"""

    print("📊 Combining datasets...")

    # Load original data
    dfs = []
    for path in original_csv_paths:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if dfs:
        original_df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(original_df)} original examples")
    else:
        print("❌ No original dataset found")
        return

    # Load negation data
    negation_df = pd.read_csv(negation_csv_path)
    print(f"✅ Loaded {len(negation_df)} negation examples")

    # Combine
    combined_df = pd.concat([original_df, negation_df], ignore_index=True)
    combined_df.to_csv(output_path, index=False)

    print(f"📁 Combined dataset saved to: {output_path}")
    print(f"📊 Total examples: {len(combined_df)}")
    print(f"   - Original: {len(original_df)}")
    print(f"   - Negation: {len(negation_df)}")

    return combined_df