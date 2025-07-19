from dotenv import load_dotenv
load_dotenv()
import torch
import os
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from deep_translator import GoogleTranslator

import nltk
from nltk.tokenize import sent_tokenize

# Download tokenizer model (if not already)
# nltk.download("punkt")
# 1. Get model path from environment variable
MODEL_PATH = os.getenv("MODEL_PATH")  # Default if not set
print(f"🔧 Using model: {MODEL_PATH}")

# Emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Load model + tokenizer
# model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
# tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def translate_tamil_to_english(text):
    try:
        translated = GoogleTranslator(source='ta', target='en').translate(text)
        return translated
    except Exception as e:
        print("⚠️ Translation failed:", e)
        return text  # fallback to original

# 🔹 Utility: Split into clean sentences
def split_into_sentences(text, max_token_length=512):
    sentences = sent_tokenize(text.strip())
    result = []
    current_chunk = ""

    for sentence in sentences:
        # Check if adding this sentence would exceed the token limit
        if len(current_chunk) + len(sentence) + 1 <= max_token_length:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            # Save the current chunk and start a new one
            if current_chunk:
                result.append(current_chunk.strip())
            current_chunk = sentence

    # Append any remaining chunk
    if current_chunk:
        result.append(current_chunk.strip())

    return result

# 🔹 Predict for a single sentence
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    return probs

# 🔹 Predict for full paragraph (auto split, print debug)
def predict_paragraph_emotions(paragraph):
    sentences = split_into_sentences(paragraph)
    all_probs = []
    token_counts = []
    max_token_length = 512

    print(f"\n📘 Total Sentences Detected: {len(sentences)}")

    # Visualize sentence impact weights
    total_tokens = sum(token_counts)
    print("\n📊 Sentence Impact Weights:")
    for i, count in enumerate(token_counts):
        weight_percent = count / total_tokens * 100
        print(f"  Sentence {i + 1}: {weight_percent:.1f}% of total tokens")

    for i, sent in enumerate(sentences):
        tokens = tokenizer.tokenize(sent)
        token_count = min(len(tokens), max_token_length)
        token_counts.append(token_count)

        probs = predict_emotion(sent)
        all_probs.append(probs)

        top_indices = probs.argsort()[-3:][::-1]
        print(f"\n🔹 Sentence {i + 1} ({token_count} tokens): {sent}")
        for idx in top_indices:
            print(f"   - {emotion_labels[idx]} ({probs[idx] * 100:.1f}%)")

    # Token-weighted aggregation
    total_tokens = sum(token_counts)
    weighted_probs = np.zeros(len(emotion_labels))

    for i in range(len(all_probs)):
        weighted_probs += all_probs[i] * token_counts[i]

    avg_probs = weighted_probs / total_tokens

    print("\n🔻 Token-Weighted Emotion Prediction:")
    sorted_indices = np.argsort(avg_probs)[::-1]
    for idx in sorted_indices:
        if avg_probs[idx] > 0.01:
            print(f"   - {emotion_labels[idx]} ({avg_probs[idx] * 100:.1f}%)")

    # Hybrid weighting: Optional but recommended
    hybrid_probs = np.zeros(len(emotion_labels))
    total_weight = 0

    for i in range(len(all_probs)):
        intensity_factor = 1 + np.max(all_probs[i])
        weight = token_counts[i] * intensity_factor
        hybrid_probs += all_probs[i] * weight
        total_weight += weight

    hybrid_probs /= total_weight

    print("\n🔻 Hybrid Weighted Prediction (Tokens + Intensity):")
    sorted_hybrid = np.argsort(hybrid_probs)[::-1]
    for idx in sorted_hybrid:
        if hybrid_probs[idx] > 0.01:
            print(f"   - {emotion_labels[idx]} ({hybrid_probs[idx] * 100:.1f}%)")




# 🧪 Example input: long review (just copy-paste in one line!)
# if __name__ == "__main__":
#     review = (
#       "I’m very disappointed with the Moratuwa Pizza Hut outlet. Most of the time, the pizzas barely have any cheese, which completely ruins the taste. The quality of the food is consistently poor, and it's definitely not what you'd expect from a brand like Pizza Hut. Honestly, this is the worst Pizza Hut outlet I’ve experienced. Really hope the management looks into this seriously and makes improvements."
#     )
#
#     predict_paragraph_emotions(review)

if __name__ == "__main__":
    review = "I’m very disappointed with the Moratuwa Pizza Hut outlet. Most of the time, the pizzas barely have any cheese, which completely ruins the taste. The quality of the food is consistently poor, and it's definitely not what you'd expect from a brand like Pizza Hut. Honestly, this is the worst Pizza Hut outlet I’ve experienced. Really hope the management looks into this seriously and makes improvements.I’m very disappointed with the Moratuwa Pizza Hut outlet. Most of the time, the pizzas barely have any cheese, which completely ruins the taste. The quality of the food is consistently poor, and it's definitely not what you'd expect from a brand like Pizza Hut. Honestly, this is the worst Pizza Hut outlet I’ve experienced. Really hope the management looks into this seriously and makes improvements."

    # Translate Tamil → English
    # translated_review = translate_tamil_to_english(review)
    # print(f"\n🌐 Translated Review:\n{translated_review}\n")

    # Now predict using the translated English
    # predict_paragraph_emotions(translated_review)
    # paragraph = "I felt energized within five minutes, but it lasted for about 45 minutes. I paid $3.99 for this drink. I could have just drunk a cup of coffee and saved my money."
    # predict_paragraph_emotions(paragraph)
    predict_paragraph_emotions(review)
    # print(predict_emotion(review))