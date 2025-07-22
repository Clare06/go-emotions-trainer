# Enhanced Emotion Prediction with Negation Handling
import os

from dotenv import load_dotenv
load_dotenv()
# Enhanced Emotion Prediction with Negation Handling

import torch
import numpy as np
import re
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification


class NegationAwareEmotionPredictor:
    def __init__(self, model_path, emotion_labels):
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.emotion_labels = emotion_labels

        # Define negation patterns and opposing emotions
        self.negation_patterns = [
            r'\bnot\s+',
            r'\bno\s+',
            r'\bnever\s+',
            r'\bnobody\s+',
            r'\bnothing\s+',
            r'\bnowhere\s+',
            r'\bneither\s+',
            r'\bnor\s+',
            r"n't\s+",
            r'\bwithout\s+',
            r'\bhardly\s+',
            r'\bbarely\s+',
            r'\bscarcely\s+',
        ]

        # Emotional opposites mapping
        self.emotion_opposites = {
            'sadness': ['joy', 'happiness', 'optimism', 'relief'],
            'anger': ['calm', 'relief', 'approval', 'caring'],
            'fear': ['relief', 'optimism', 'approval'],
            'disgust': ['admiration', 'approval'],
            'disappointment': ['relief', 'optimism', 'approval'],
            'disapproval': ['approval', 'admiration'],
            'annoyance': ['relief', 'approval', 'caring'],
            'embarrassment': ['pride', 'relief'],
            'grief': ['relief', 'joy', 'optimism'],
            'nervousness': ['relief', 'optimism'],
            'remorse': ['relief', 'approval']
        }

    def detect_negation_context(self, text):
        """Detect if text contains negation and what emotions might be negated"""
        text_lower = text.lower()
        negations_found = []

        for pattern in self.negation_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Get words after negation (within 3 words)
                start_pos = match.end()
                remaining_text = text_lower[start_pos:start_pos + 50]  # Look ahead 50 chars
                words_after = remaining_text.split()[:3]  # Next 3 words

                negations_found.append({
                    'negation_word': match.group().strip(),
                    'position': match.start(),
                    'context_after': words_after
                })

        return negations_found

    def preprocess_for_negation(self, text):
        """Enhanced preprocessing that marks negated contexts"""
        # Method 1: Add special tokens around negated phrases
        negation_processed = text

        patterns_with_replacement = [
            (r'\bnot\s+(\w+)', r'NOT_\1'),
            (r'\bno\s+(\w+)', r'NO_\1'),
            (r'\bnever\s+(\w+)', r'NEVER_\1'),
            (r"(\w+)n't\s+(\w+)", r'\1_NOT_\2'),
            (r'\bwithout\s+(\w+)', r'WITHOUT_\1'),
        ]

        for pattern, replacement in patterns_with_replacement:
            negation_processed = re.sub(pattern, replacement, negation_processed, flags=re.IGNORECASE)

        return negation_processed

    def predict_with_negation_handling(self, text, use_preprocessing=True, apply_negation_logic=True):
        """Main prediction function with negation awareness"""

        # Step 1: Detect negation context
        negations = self.detect_negation_context(text)
        has_negation = len(negations) > 0

        # Step 2: Get predictions from multiple approaches
        results = {}

        # Original prediction
        original_probs = self._get_raw_prediction(text)
        results['original'] = original_probs

        if use_preprocessing and has_negation:
            # Preprocessed prediction
            processed_text = self.preprocess_for_negation(text)
            processed_probs = self._get_raw_prediction(processed_text)
            results['preprocessed'] = processed_probs

        # Step 3: Apply negation logic if detected
        final_probs = original_probs.copy()

        if apply_negation_logic and has_negation:
            final_probs = self._apply_negation_adjustment(text, original_probs, negations)
            results['negation_adjusted'] = final_probs

        return {
            'probabilities': final_probs,
            'has_negation': has_negation,
            'negation_contexts': negations,
            'debug_results': results
        }

    def _get_raw_prediction(self, text):
        """Get raw model predictions"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits)[0].cpu().numpy()
        return probs

    def _apply_negation_adjustment(self, text, probs, negations):
        """Apply negation-aware probability adjustments"""
        adjusted_probs = probs.copy()
        text_lower = text.lower()

        print(f"DEBUG: Original text: '{text}'")
        print(f"DEBUG: Negations found: {negations}")

        # For each detected negation, reduce related emotion probabilities
        for negation in negations:
            context_words = ' '.join(negation['context_after']).lower()
            print(f"DEBUG: Looking at context after '{negation['negation_word']}': '{context_words}'")

            # Check which emotions might be negated based on context
            for i, emotion in enumerate(self.emotion_labels):
                emotion_found = False

                # Direct emotion word match
                if emotion in context_words:
                    emotion_found = True
                    print(f"DEBUG: Found exact emotion '{emotion}' in context")

                # Partial word matches for emotion stems
                elif any(emotion.startswith(word.rstrip('.,!?')) for word in context_words.split() if len(word) > 3):
                    emotion_found = True
                    print(f"DEBUG: Found emotion stem '{emotion}' in context")

                # Special cases for common emotion words
                emotion_variants = {
                    'sadness': ['sad', 'unhappy', 'down'],
                    'anger': ['angry', 'mad', 'furious'],
                    'excitement': ['excited', 'thrilled'],
                    'fear': ['scared', 'afraid', 'frightened'],
                    'joy': ['happy', 'joyful', 'glad'],
                    'disappointment': ['disappointed', 'letdown']
                }

                if emotion in emotion_variants:
                    for variant in emotion_variants[emotion]:
                        if variant in context_words:
                            emotion_found = True
                            print(f"DEBUG: Found emotion variant '{variant}' for '{emotion}'")
                            break

                if emotion_found:
                    original_prob = adjusted_probs[i]
                    # Significantly reduce this emotion's probability
                    reduction_factor = 0.15  # Reduce to 15% of original
                    adjusted_probs[i] *= reduction_factor
                    print(f"DEBUG: Reduced '{emotion}' from {original_prob:.3f} to {adjusted_probs[i]:.3f}")

                    # Boost opposite emotions if they exist
                    if emotion in self.emotion_opposites:
                        for opposite in self.emotion_opposites[emotion]:
                            if opposite in self.emotion_labels:
                                opp_idx = self.emotion_labels.index(opposite)
                                original_opp = adjusted_probs[opp_idx]
                                # Boost opposite emotion
                                boost_factor = 2.0
                                adjusted_probs[opp_idx] = min(1.0, adjusted_probs[opp_idx] * boost_factor)
                                print(
                                    f"DEBUG: Boosted opposite '{opposite}' from {original_opp:.3f} to {adjusted_probs[opp_idx]:.3f}")

        return adjusted_probs

    def predict_and_explain(self, text, top_k=5):
        """Predict emotions with detailed explanation"""
        result = self.predict_with_negation_handling(text)

        probs = result['probabilities']
        top_indices = probs.argsort()[-top_k:][::-1]

        print(f"Input: '{text}'")
        print(f"Negation detected: {'Yes' if result['has_negation'] else 'No'}")

        if result['has_negation']:
            print("Negation contexts found:")
            for neg in result['negation_contexts']:
                print(f"  - '{neg['negation_word']}' followed by: {neg['context_after']}")

        print(f"\nTop {top_k} predicted emotions:")
        for idx in top_indices:
            print(f"  {self.emotion_labels[idx]}: {probs[idx] * 100:.1f}%")

        # Show comparison if negation was handled
        if 'negation_adjusted' in result['debug_results']:
            print("\n--- Comparison ---")
            original_top = result['debug_results']['original'].argsort()[-3:][::-1]
            adjusted_top = result['debug_results']['negation_adjusted'].argsort()[-3:][::-1]

            print("Original top 3:")
            for idx in original_top:
                print(f"  {self.emotion_labels[idx]}: {result['debug_results']['original'][idx] * 100:.1f}%")

            print("After negation handling:")
            for idx in adjusted_top:
                print(f"  {self.emotion_labels[idx]}: {result['debug_results']['negation_adjusted'][idx] * 100:.1f}%")

        return result


# Usage example
if __name__ == "__main__":
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    MODEL_PATH = os.getenv("MODEL_PATH")  # Default if not set
    print(f"🔧 Using model: {MODEL_PATH}")
    predictor = NegationAwareEmotionPredictor(MODEL_PATH, emotion_labels)

    # Test cases
    test_sentences = [
        "I never feel cheerful",
        "I am not well",
        "This doesn't make me happy",
        "I never felt disappointed",
        "I am sad",  # Control case without negation
        "Without fear, I approached the situation",
        "I'm not excited about this",
        "I'm not sad",
        "I'm not anger",
    ]

    for sentence in test_sentences:
        print("=" * 60)
        predictor.predict_and_explain(sentence)
        print()


# Alternative lightweight solution for your existing code
def quick_negation_fix(text, probs, emotion_labels):
    """Quick fix to add to your existing prediction function"""

    # Simple negation detection
    negation_words = ['not', 'no', 'never', "n't", 'without', 'hardly', 'barely']
    text_lower = text.lower()

    has_negation = any(neg in text_lower for neg in negation_words)

    if has_negation:
        adjusted_probs = probs.copy()

        # Define emotions that are commonly negated
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'disappointment',
                             'disapproval', 'annoyance', 'embarrassment', 'grief', 'nervousness']

        # If negation is detected, reduce negative emotion probabilities
        for i, emotion in enumerate(emotion_labels):
            if emotion in negative_emotions:
                # Check if emotion word appears near negation
                for neg_word in negation_words:
                    if f"{neg_word} {emotion}" in text_lower or f"{neg_word}ly {emotion}" in text_lower:
                        adjusted_probs[i] *= 0.2  # Reduce significantly
                        break
                    elif neg_word in text_lower and emotion in text_lower:
                        adjusted_probs[i] *= 0.5  # Moderate reduction

        # Boost neutral and positive emotions slightly
        neutral_idx = emotion_labels.index('neutral') if 'neutral' in emotion_labels else None
        if neutral_idx is not None:
            adjusted_probs[neutral_idx] *= 1.3

        return adjusted_probs

    return probs