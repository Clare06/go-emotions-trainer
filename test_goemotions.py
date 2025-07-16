import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize

# Download tokenizer model (if not already)
# nltk.download("punkt")

# Emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Load model + tokenizer
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 🔹 Utility: Split into clean sentences
def split_into_sentences(text):
    sentences = sent_tokenize(text.strip())
    return [s.strip() for s in sentences if s.strip()]

# 🔹 Predict for a single sentence
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    return probs

# 🔹 Predict for full paragraph (auto split, print debug)
def predict_paragraph_emotions(paragraph):
    sentences = split_into_sentences(paragraph)
    all_probs = []

    print(f"\n📘 Total Sentences Detected: {len(sentences)}")

    for i, sent in enumerate(sentences):
        probs = predict_emotion(sent)
        all_probs.append(probs)

        # Show top emotions for each sentence
        top_indices = probs.argsort()[-3:][::-1]
        print(f"\n🔹 Sentence {i+1}: {sent}")
        for idx in top_indices:
            print(f"   - {emotion_labels[idx]} ({probs[idx]*100:.1f}%)")

    # Aggregate all
    avg_probs = np.mean(all_probs, axis=0)
    # Sort all emotions by descending probability
    sorted_indices = np.argsort(avg_probs)[::-1]

    print("\n🔻 Final Aggregated Emotion Prediction (All Sentences):")
    for idx in sorted_indices:
        if avg_probs[idx] > 0.01:  # show only >1% scores
            print(f"   - {emotion_labels[idx]} ({avg_probs[idx] * 100:.1f}%)")


# 🧪 Example input: long review (just copy-paste in one line!)
if __name__ == "__main__":
    review = (
        "I’m very disappointed with the Moratuwa Pizza Hut outlet. Most of the time, the pizzas barely have any cheese, which completely ruins the taste. The quality of the food is consistently poor, and it's definitely not what you'd expect from a brand like Pizza Hut. Honestly, this is the worst Pizza Hut outlet I’ve experienced. Really hope the management looks into this seriously and makes improvements."
    )

    predict_paragraph_emotions(review)
