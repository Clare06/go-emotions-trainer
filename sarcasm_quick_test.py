# sarcasm_quick_test.py
import pandas as pd
import numpy as np
from predict_emo import get_emotion_prediction  # Your existing emotion predictor
from sarcasm_detector import get_sarcasm_detector, predict_sarcasm
import warnings

warnings.filterwarnings('ignore')


class SarcasmQuickTester:
    def __init__(self):
        print("🚀 Loading models...")

        # Load sarcasm model
        print("🎭 Loading sarcasm model...")
        self.sarcasm_model, self.scaler, self.feature_cols, self.device = get_sarcasm_detector()

        # Emotion labels (same order as in training)
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        print("✅ Models loaded successfully!")

    def extract_features_from_text(self, text, emotion_predictions):
        """Extract features exactly like in training"""

        # Initialize features dictionary
        features = {}

        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?') + 1

        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['capitalization_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        # Emotion features from your emotion model
        for i, emotion in enumerate(self.emotion_labels):
            if i < len(emotion_predictions):
                features[f'emotion_{emotion}'] = emotion_predictions[i]
            else:
                features[f'emotion_{emotion}'] = 0.0

        # Advanced sarcasm-specific features
        positive_emotions = ['joy', 'love', 'excitement', 'gratitude', 'admiration', 'optimism', 'amusement', 'pride',
                             'caring', 'approval']
        negative_emotions = ['anger', 'disgust', 'fear', 'grief', 'sadness', 'disappointment', 'disapproval',
                             'annoyance', 'remorse']

        pos_sum = sum(
            emotion_predictions[self.emotion_labels.index(e)] for e in positive_emotions if e in self.emotion_labels)
        neg_sum = sum(
            emotion_predictions[self.emotion_labels.index(e)] for e in negative_emotions if e in self.emotion_labels)

        features['positive_sum'] = pos_sum
        features['negative_sum'] = neg_sum
        features['pos_neg_ratio'] = pos_sum / (neg_sum + 1e-8)
        features['emotional_polarity'] = pos_sum - neg_sum

        # Statistical features
        features['max_emotion_prob'] = np.max(emotion_predictions)
        features['min_emotion_prob'] = np.min(emotion_predictions)
        features['mean_emotion_prob'] = np.mean(emotion_predictions)
        features['emotion_variance'] = np.var(emotion_predictions)
        features['emotion_std'] = np.std(emotion_predictions)

        # Incongruency features (key for sarcasm)
        joy_idx = self.emotion_labels.index('joy') if 'joy' in self.emotion_labels else 0
        anger_idx = self.emotion_labels.index('anger') if 'anger' in self.emotion_labels else 0
        features['joy_anger_contrast'] = abs(emotion_predictions[joy_idx] - emotion_predictions[anger_idx])

        # Convert to array in the same order as training features
        feature_array = np.array([features.get(col, 0.0) for col in self.feature_cols])

        return feature_array

    def test_text(self, text):
        """Test a single text for sarcasm"""
        print(f"\n🔍 Testing: '{text}'")

        # Step 1: Get emotion predictions
        print("📊 Getting emotion predictions...")
        emotion_predictions = get_emotion_prediction(text)

        # Step 2: Extract features
        print("🔧 Extracting features...")
        features = self.extract_features_from_text(text, emotion_predictions)

        # Step 3: Predict sarcasm
        print("🎭 Predicting sarcasm...")
        prediction, probabilities = predict_sarcasm(features, self.sarcasm_model, self.scaler, self.device)

        # Step 4: Display results
        print("\n" + "=" * 60)
        print("📋 RESULTS:")
        print(f"Text: {text}")
        print(f"🎭 Sarcasm Prediction: {'SARCASTIC' if prediction == 1 else 'NOT SARCASTIC'}")
        print(f"📊 Confidence: {probabilities[prediction]:.4f}")
        print(f"🎯 Probabilities:")
        print(f"   Non-Sarcastic: {probabilities[0]:.4f}")
        print(f"   Sarcastic: {probabilities[1]:.4f}")

        # Top emotions
        top_emotions_idx = np.argsort(emotion_predictions)[-3:][::-1]
        print(f"🎨 Top 3 Emotions:")
        for idx in top_emotions_idx:
            if idx < len(self.emotion_labels):
                emotion_name = self.emotion_labels[idx]
                emotion_score = emotion_predictions[idx]
                print(f"   {emotion_name}: {emotion_score:.4f}")

        print("=" * 60)

        return prediction, probabilities


def main():
    """Interactive testing"""
    try:
        tester = SarcasmQuickTester()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure your model files exist at the specified paths!")
        return

    print("\n🎯 SARCASM DETECTION QUICK TESTER")
    print("Type 'quit' to exit\n")

    # Test samples
    test_samples = [
        "I am going to hostel after a long holiday at home with my parents, sad",
        "Oh great, another meeting!",
        "I love working overtime on weekends",
        "This is the best day ever",
        "Yeah, because that's exactly what I wanted to hear",
        "I'm really happy about this situation"
    ]

    print("🧪 Testing sample texts:")
    for sample in test_samples:
        try:
            tester.test_text(sample)
            input("Press Enter to continue...")
        except Exception as e:
            print(f"❌ Error testing '{sample}': {e}")
            continue

    # Interactive testing
    while True:
        user_input = input("\nEnter text to test (or 'quit' to exit): ").strip()

        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break

        if user_input:
            try:
                tester.test_text(user_input)
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("Please enter some text to test!")


if __name__ == "__main__":
    main()