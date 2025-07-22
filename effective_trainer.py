import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GoEmotions label mapping
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


def load_and_preprocess_data():
    """Load and preprocess the GoEmotions dataset."""
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions")

    # Convert multi-label to single-label by taking the first label
    def convert_labels(examples):
        # Take the first label for each example (convert from multi-label to single-label)
        examples['labels'] = [labels[0] if labels else 27 for labels in examples['labels']]  # 27 is neutral
        return examples

    dataset = dataset.map(convert_labels, batched=True)
    return dataset


def tokenize_function(examples, tokenizer):
    """Tokenize the input text."""
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # Ensure labels are included in the tokenized output
    tokenized["labels"] = examples["labels"]
    return tokenized


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1
    }


def main():
    """Main training and evaluation function."""
    print("Starting Go Emotions Trainer...")

    # Model configuration
    model_name = "bert-base-uncased"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS)
    )

    # Load dataset
    dataset = load_and_preprocess_data()

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    # Remove unnecessary columns
    tokenized_dataset = tokenized_dataset.remove_columns(['text', 'id'])

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # Reduced batch size for stability
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_pin_memory=False,  # Disable pin_memory for MPS
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test Results: {test_results}")

    # Save the model
    trainer.save_model("./go-emotions-model")
    tokenizer.save_pretrained("./go-emotions-model")

    print("Training completed! Model saved to ./go-emotions-model")


def predict_emotion(text, model_path="./go-emotions-model"):
    """Predict emotion for a given text."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()

        predicted_emotion = EMOTION_LABELS[predicted_class_id]
        return predicted_emotion, confidence
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None, None


def interactive_test():
    """Interactive mode for testing user inputs."""
    print("\n=== Go Emotions Interactive Tester ===")
    print("Enter text to predict emotions (type 'quit' to exit)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nEnter text: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter some text.")
                continue

            emotion, confidence = predict_emotion(user_input)

            if emotion and confidence:
                print(f"Predicted Emotion: {emotion}")
                print(f"Confidence: {confidence:.3f}")
            else:
                print("Could not predict emotion. Please check if the model is properly saved.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "predict":
            if len(sys.argv) > 2:
                text = " ".join(sys.argv[2:])
                emotion, confidence = predict_emotion(text)
                if emotion and confidence:
                    print(f"Text: {text}")
                    print(f"Predicted Emotion: {emotion} (confidence: {confidence:.3f})")
                else:
                    print("Could not predict emotion. Please check if the model exists.")
            else:
                print("Usage: python main.py predict 'your text here'")
        elif sys.argv[1] == "test":
            interactive_test()
        elif sys.argv[1] == "train":
            main()
        else:
            print("Usage:")
            print("  python main.py train          - Train the model")
            print("  python main.py test           - Interactive testing mode")
            print("  python main.py predict 'text' - Predict single text")
    else:
        # Default to interactive testing if model exists, otherwise train
        if os.path.exists("./go-emotions-model"):
            print("Model found! Starting interactive test mode...")
            interactive_test()
        else:
            print("No model found. Starting training...")
            main()