# test_goemotions.py

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Emotion labels (same order used in training)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Load model and tokenizer from saved folder
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")

# Move to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Top 3 predicted emotions
    top_indices = probs.argsort()[-3:][::-1]
    print(f"\nInput: {text}")
    print("Top predicted emotions:")
    for idx in top_indices:
        print(f"- {emotion_labels[idx]} ({probs[idx]*100:.1f}%)")

# ✨ Try testing here
if __name__ == "__main__":
    predict_emotion("I'm not sure if I'm happy or just confused.")
    predict_emotion("This makes me so proud and joyful!")
    predict_emotion("I don't care anymore. I'm done.")
