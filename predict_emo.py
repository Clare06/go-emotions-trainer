import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels with their optimal thresholds from your analysis
OPTIMAL_THRESHOLDS = {
    'admiration': 0.4,
    'amusement': 0.4,
    'anger': 0.3,
    'annoyance': 0.1,
    'approval': 0.3,
    'caring': 0.3,
    'confusion': 0.2,
    'curiosity': 0.3,
    'desire': 0.2,
    'disappointment': 0.2,
    'disapproval': 0.2,
    'disgust': 0.3,
    'embarrassment': 0.2,
    'excitement': 0.3,
    'fear': 0.4,
    'gratitude': 0.5,
    'grief': 0.1,
    'joy': 0.3,
    'love': 0.4,
    'nervousness': 0.4,
    'optimism': 0.4,
    'pride': 0.2,
    'realization': 0.1,
    'relief': 0.4,
    'remorse': 0.4,
    'sadness': 0.3,
    'surprise': 0.3
}

emotion_labels = list(OPTIMAL_THRESHOLDS.keys())


class OptimizedEmotionPredictor:
    """Enhanced emotion prediction with optimized thresholds and paragraph processing"""

    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("MODEL_PATH", "saved_model_xlm-roberta-base")
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from: {self.model_path}")

        try:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_path)
            self.model = XLMRobertaForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def translate_tamil_to_english(self, text):
        """Translate Tamil text to English"""
        try:
            translated = GoogleTranslator(source='ta', target='en').translate(text)
            return translated
        except Exception as e:
            print("⚠️ Translation failed:", e)
            return text  # fallback to original

    def split_into_sentences(self, text, max_token_length=512):
        """Split text into sentences while respecting token limits"""
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

    def predict_sentence_emotion(self, text):
        """Predict emotions for a single sentence"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return np.zeros(len(emotion_labels))

        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]

            return probabilities

        except Exception as e:
            print(f"Error in sentence prediction: {str(e)}")
            return np.zeros(len(emotion_labels))

    def predict_single_text(self, text, return_probabilities=False, min_confidence=0.0):
        """
        Predict emotions for a single text using optimized thresholds

        Args:
            text (str): Input text to analyze
            return_probabilities (bool): Whether to return probability scores
            min_confidence (float): Minimum confidence threshold for predictions

        Returns:
            dict: Prediction results with emotions and optionally probabilities
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {"error": "Invalid input text"}

        probabilities = self.predict_sentence_emotion(text)

        # Apply optimized thresholds
        predicted_emotions = []
        emotion_scores = {}

        for i, emotion in enumerate(emotion_labels):
            prob = probabilities[i]
            threshold = OPTIMAL_THRESHOLDS[emotion]

            emotion_scores[emotion] = {
                'probability': float(prob),
                'threshold': threshold,
                'predicted': bool(prob > threshold and prob >= min_confidence)
            }

            if prob > threshold and prob >= min_confidence:
                predicted_emotions.append(emotion)

        # Prepare results
        result = {
            'text': text,
            'predicted_emotions': predicted_emotions,
            'emotion_count': len(predicted_emotions)
        }

        if return_probabilities:
            result['detailed_scores'] = emotion_scores
            result['top_5_emotions'] = self._get_top_emotions(emotion_scores, top_k=5)

        return result

    def predict_paragraph_emotions(self, paragraph, verbose=True, translation=False):
        """
        Enhanced paragraph emotion prediction with sentence-level analysis

        Args:
            paragraph (str): Input paragraph text
            verbose (bool): Whether to print detailed analysis
            translation (bool): Whether to attempt translation from Tamil

        Returns:
            dict: Comprehensive paragraph emotion analysis
        """
        if translation:
            paragraph = self.translate_tamil_to_english(paragraph)
            if verbose:
                print(f"\n🌐 Translated Text: {paragraph}\n")

        sentences = self.split_into_sentences(paragraph)
        all_probs = []
        token_counts = []
        max_token_length = 512

        if verbose:
            print(f"\n📘 Total Sentences Detected: {len(sentences)}")

        for i, sent in enumerate(sentences):
            tokens = self.tokenizer.tokenize(sent)
            token_count = min(len(tokens), max_token_length)
            token_counts.append(token_count)

            probs = self.predict_sentence_emotion(sent)
            all_probs.append(probs)

            if verbose:
                top_indices = probs.argsort()[-3:][::-1]
                print(f"\n🔹 Sentence {i + 1} ({token_count} tokens): {sent}")
                for idx in top_indices:
                    print(f"   - {emotion_labels[idx]} ({probs[idx] * 100:.1f}%)")

        # Token-weighted average
        total_tokens = sum(token_counts)
        weighted_probs = np.zeros(len(emotion_labels))

        for i in range(len(all_probs)):
            weighted_probs += all_probs[i] * token_counts[i]

        avg_probs = weighted_probs / total_tokens

        if verbose:
            print("\n🔻 Token-Weighted Emotion Prediction:")
            sorted_indices = np.argsort(avg_probs)[::-1]
            for idx in sorted_indices:
                if avg_probs[idx] > 0.01:
                    print(f"   - {emotion_labels[idx]} ({avg_probs[idx] * 100:.1f}%)")

        # Hybrid: tokens + intensity
        hybrid_probs = np.zeros(len(emotion_labels))
        total_weight = 0

        for i in range(len(all_probs)):
            intensity_factor = 1 + np.max(all_probs[i])
            weight = token_counts[i] * intensity_factor
            hybrid_probs += all_probs[i] * weight
            total_weight += weight

        hybrid_probs /= total_weight

        if verbose:
            print("\n🔻 Hybrid Weighted Prediction (Tokens + Intensity):")
            sorted_hybrid = np.argsort(hybrid_probs)[::-1]
            for idx in sorted_hybrid:
                if hybrid_probs[idx] > 0.01:
                    print(f"   - {emotion_labels[idx]} ({hybrid_probs[idx] * 100:.1f}%)")

        # Apply optimized thresholds
        filtered_emotions = {}
        predicted_emotions = []

        if verbose:
            print("\n✅ Final Filtered Emotions (Above Custom Thresholds):")

        for idx in np.argsort(hybrid_probs)[::-1]:
            emotion = emotion_labels[idx]
            score = hybrid_probs[idx]
            threshold = OPTIMAL_THRESHOLDS.get(emotion, 0.5)

            if score >= threshold:
                if verbose:
                    print(f"   ✔ {emotion} ({score * 100:.1f}%) — above threshold {threshold:.2f}")
                filtered_emotions[emotion] = round(score, 4)
                predicted_emotions.append(emotion)

        # Calculate sentiment analysis
        sentiment_analysis = self._calculate_sentiment_analysis(filtered_emotions, hybrid_probs)

        return {
            'text': paragraph,
            'sentences': sentences,
            'sentence_count': len(sentences),
            'total_tokens': total_tokens,
            'predicted_emotions': predicted_emotions,
            'emotion_count': len(predicted_emotions),
            'filtered_emotions': filtered_emotions,
            'all_probabilities': {emotion_labels[i]: round(hybrid_probs[i], 4) for i in range(len(emotion_labels))},
            'sentiment_analysis': sentiment_analysis,
            'processing_method': 'paragraph_weighted'
        }

    def predict_batch(self, texts, return_probabilities=False, min_confidence=0.0, use_paragraph_mode=False):
        """
        Predict emotions for multiple texts

        Args:
            texts (list): List of input texts
            return_probabilities (bool): Whether to return probability scores
            min_confidence (float): Minimum confidence threshold
            use_paragraph_mode (bool): Whether to use paragraph processing for longer texts

        Returns:
            list: List of prediction results
        """
        results = []

        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                print(f"Processing text {i + 1}/{len(texts)}")

            # Decide whether to use paragraph mode based on text length
            if use_paragraph_mode and len(text.split()) > 20:
                result = self.predict_paragraph_emotions(text, verbose=False)
            else:
                result = self.predict_single_text(text, return_probabilities, min_confidence)

            results.append(result)

        return results

    def _get_top_emotions(self, emotion_scores, top_k=5):
        """Get top K emotions by probability"""
        sorted_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )

        return [
            {
                'emotion': emotion,
                'probability': scores['probability'],
                'threshold': scores['threshold'],
                'predicted': scores['predicted']
            }
            for emotion, scores in sorted_emotions[:top_k]
        ]

    def _calculate_sentiment_analysis(self, filtered_emotions, all_probabilities):
        """Calculate comprehensive sentiment analysis"""
        positive_emotions = ['admiration', 'amusement', 'approval', 'caring', 'excitement',
                             'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief']
        negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
                             'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness']

        positive_score = sum(all_probabilities[i] for i, emotion in enumerate(emotion_labels)
                             if emotion in positive_emotions and emotion in filtered_emotions)
        negative_score = sum(all_probabilities[i] for i, emotion in enumerate(emotion_labels)
                             if emotion in negative_emotions and emotion in filtered_emotions)

        # Determine overall sentiment
        if positive_score > negative_score:
            sentiment = "Positive"
        elif negative_score > positive_score:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Emotion intensity
        max_prob = max(all_probabilities) if len(all_probabilities) > 0 else 0
        if max_prob > 0.8:
            intensity = "Very High"
        elif max_prob > 0.6:
            intensity = "High"
        elif max_prob > 0.4:
            intensity = "Medium"
        else:
            intensity = "Low"

        return {
            'overall_sentiment': sentiment,
            'positive_score': round(positive_score, 3),
            'negative_score': round(negative_score, 3),
            'emotion_intensity': intensity,
            'max_probability': round(max_prob, 3)
        }

    def analyze_text_detailed(self, text, use_paragraph_mode=None):
        """
        Comprehensive analysis of a text with detailed insights
        Auto-detects whether to use paragraph mode based on text length

        Args:
            text (str): Input text to analyze
            use_paragraph_mode (bool): Force paragraph mode (None for auto-detect)

        Returns:
            dict: Detailed analysis results
        """
        # Auto-detect paragraph mode based on sentence count or word count
        if use_paragraph_mode is None:
            word_count = len(text.split())
            sentence_count = len(sent_tokenize(text))
            use_paragraph_mode = word_count > 20 or sentence_count > 2

        if use_paragraph_mode:
            return self.predict_paragraph_emotions(text, verbose=True)
        else:
            result = self.predict_single_text(text, return_probabilities=True)

            if 'error' in result:
                return result

            # Add sentiment analysis for single text
            detailed_scores = result['detailed_scores']
            all_probs = np.array([scores['probability'] for scores in detailed_scores.values()])
            filtered_emotions = {emotion: scores['probability']
                                 for emotion, scores in detailed_scores.items()
                                 if scores['predicted']}

            sentiment_analysis = self._calculate_sentiment_analysis(filtered_emotions, all_probs)
            result['sentiment_analysis'] = sentiment_analysis

            # Categorize emotions by confidence
            high_confidence = []  # > 0.7
            medium_confidence = []  # 0.4 - 0.7
            low_confidence = []  # < 0.4

            for emotion, scores in detailed_scores.items():
                prob = scores['probability']

                if prob > 0.7:
                    high_confidence.append((emotion, prob))
                elif prob > 0.4:
                    medium_confidence.append((emotion, prob))
                else:
                    low_confidence.append((emotion, prob))

            result['confidence_breakdown'] = {
                'high_confidence': [(e, round(p, 3)) for e, p in high_confidence],
                'medium_confidence': [(e, round(p, 3)) for e, p in medium_confidence],
                'low_confidence': [(e, round(p, 3)) for e, p in low_confidence]
            }

            return result


def compare_thresholds(text, predictor):
    """
    Compare predictions using default (0.5) vs optimized thresholds

    Args:
        text (str): Input text to analyze
        predictor (OptimizedEmotionPredictor): Predictor instance

    Returns:
        dict: Comparison results
    """
    # Get probabilities
    probabilities = predictor.predict_sentence_emotion(text)

    # Default threshold predictions
    default_predictions = []
    optimized_predictions = []

    comparison = {}

    for i, emotion in enumerate(emotion_labels):
        prob = probabilities[i]
        default_threshold = 0.5
        optimal_threshold = OPTIMAL_THRESHOLDS[emotion]

        default_pred = prob > default_threshold
        optimal_pred = prob > optimal_threshold

        if default_pred:
            default_predictions.append(emotion)
        if optimal_pred:
            optimized_predictions.append(emotion)

        comparison[emotion] = {
            'probability': round(float(prob), 3),
            'default_threshold': default_threshold,
            'optimal_threshold': optimal_threshold,
            'default_prediction': default_pred,
            'optimal_prediction': optimal_pred,
            'threshold_difference': optimal_threshold - default_threshold
        }

    return {
        'text': text,
        'default_predictions': default_predictions,
        'optimized_predictions': optimized_predictions,
        'default_count': len(default_predictions),
        'optimized_count': len(optimized_predictions),
        'detailed_comparison': comparison
    }


# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = OptimizedEmotionPredictor()

    # Test texts - mix of short and long
    test_texts = [
        "இது மிகவும் அருமையான செய்தி",
        "ethana maru thanak..",
        "මම සතුටින් ඉන්නවා",
        "EHA GEDARA BALLATA MEHA GEDARATA ENNA NA When the port city was being built SIGIRI GALA KADALA KALU GAL GANNA WEI In my POV we cannot expect any development from such narrow minded peeps",
        "mama ada rata kawa",
        "අම්මේ, අද රට කවා ",
        "நான் இன்று ரொட்டி சாப்பிட்டேன்",
        "epa kanna one"
    ]

    print("=== ENHANCED EMOTION PREDICTION DEMO ===\n")

    for i, text in enumerate(test_texts):
        print(f"\n{'=' * 60}")
        print(f"TEST {i + 1}: Processing text...")
        print(f"Length: {len(text.split())} words")
        print(f"Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"{'=' * 60}")

        # Detailed analysis (auto-detects paragraph vs single text)
        detailed = predictor.analyze_text_detailed(text)

        if 'error' in detailed:
            print(f"❌ Error: {detailed['error']}")
            continue

        print(f"\n📊 RESULTS:")
        print(f"Processing method: {detailed.get('processing_method', 'single_text')}")
        print(f"Predicted emotions: {detailed['predicted_emotions']}")
        print(f"Emotion count: {detailed['emotion_count']}")
        print(f"Sentiment: {detailed['sentiment_analysis']['overall_sentiment']}")
        print(f"Intensity: {detailed['sentiment_analysis']['emotion_intensity']}")

        if 'top_5_emotions' in detailed:
            # Single text mode
            top_emotions_str = [f"{e['emotion']}: {e['probability']:.3f}"
                                for e in detailed['top_5_emotions'][:3]]
            print(f"Top 3 emotions: {top_emotions_str}")

            # Compare with default thresholds
            comparison = compare_thresholds(text, predictor)
            print(
                f"Default vs Optimized: {len(comparison['default_predictions'])} → {len(comparison['optimized_predictions'])} emotions")
        else:
            # Paragraph mode
            print(f"Sentences processed: {detailed['sentence_count']}")
            print(f"Total tokens: {detailed['total_tokens']}")
            if detailed['filtered_emotions']:
                top_3 = list(detailed['filtered_emotions'].items())[:3]
                print(f"Top 3 emotions: {[f'{e}: {s:.3f}' for e, s in top_3]}")

        print("-" * 60)