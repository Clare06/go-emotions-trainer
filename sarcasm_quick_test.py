# sarcasm_quick_test_with_kg.py
import pandas as pd
import numpy as np
from predict_emo import get_emotion_prediction  # Your existing emotion predictor
from sarcasm_detector import get_sarcasm_detector, predict_sarcasm
import warnings
import networkx as nx
from nltk import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

warnings.filterwarnings('ignore')


class EnhancedSarcasmTester:
    def __init__(self):
        print("🚀 Loading models...")

        # Load sarcasm model
        print("🎭 Loading sarcasm model...")
        self.sarcasm_model, self.scaler, self.feature_cols, self.device = get_sarcasm_detector()

        # Emotion labels (same order as in training, but without neutral for KG)
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise'
        ]

        # For features extraction (includes neutral)
        self.feature_emotion_labels = self.emotion_labels + ['neutral']

        # Opposites for negation cross-boost
        self.OPPOSITES = {
            'admiration': ['disapproval', 'disgust', 'annoyance'],
            'amusement': ['annoyance', 'disappointment', 'sadness'],
            'anger': ['relief', 'approval', 'caring'],
            'annoyance': ['approval', 'relief', 'joy'],
            'approval': ['disapproval', 'anger', 'disappointment'],
            'caring': ['disapproval', 'disgust', 'anger'],
            'confusion': ['realization', 'approval', 'optimism'],
            'curiosity': ['disappointment', 'annoyance', 'disapproval'],
            'desire': ['disgust', 'disappointment', 'remorse'],
            'disappointment': ['joy', 'excitement', 'relief'],
            'disapproval': ['approval', 'admiration', 'gratitude'],
            'disgust': ['admiration', 'love', 'approval'],
            'embarrassment': ['pride', 'optimism', 'relief'],
            'excitement': ['disappointment', 'sadness', 'annoyance'],
            'fear': ['relief', 'optimism', 'pride'],
            'gratitude': ['disapproval', 'anger', 'remorse'],
            'grief': ['joy', 'relief', 'optimism'],
            'joy': ['sadness', 'disappointment', 'grief'],
            'love': ['disgust', 'disapproval', 'anger'],
            'nervousness': ['relief', 'optimism', 'pride'],
            'optimism': ['disappointment', 'fear', 'sadness'],
            'pride': ['embarrassment', 'remorse', 'disapproval'],
            'realization': ['confusion', 'disappointment', 'surprise'],
            'relief': ['fear', 'nervousness', 'disappointment'],
            'remorse': ['pride', 'approval', 'relief'],
            'sadness': ['joy', 'excitement', 'relief'],
            'surprise': ['realization', 'annoyance', 'disappointment']
        }

        print("✅ Models loaded successfully!")

    def build_emotion_kg(self, include_negation=True):
        """Build emotion knowledge graph with triggers and optional negation"""
        G = nx.Graph()

        # Add emotion nodes
        for emotion in self.emotion_labels:
            G.add_node(emotion)

        # Emotion triggers (from your code)
        triggers = {
            'admiration': ['stunning view', 'beautiful architecture', 'amazing design', 'impressive facilities',
                           'gorgeous sunset', 'breathtaking scenery', 'elegant decor', 'wonderful craftsmanship',
                           'incredible location', 'sundara'],
            'amusement': ['funny staff', 'hilarious experience', 'entertaining show', 'playful atmosphere', 'joke',
                          'amusing incident', 'lighthearted vibe', 'comical error', 'witty service', 'enjoyable games',
                          'laughable mistake', 'cheerful crowd'],
            'anger': ['rude staff', 'overpriced', 'terrible service', 'frustrating wait', 'annoying noise',
                      'fight with manager', 'infuriating delay', 'outrageous bill', 'aggravating crowd',
                      'irritating insects', 'furious about cleanliness', 'enraging scam'],
            'annoyance': ['minor issue', 'slight delay', 'bothersome smell', 'irritating music', 'pesky mosquitoes',
                          'annoying crowd', 'frustrating parking', 'mild discomfort', 'bothersome noise',
                          'irksome wait',
                          'petty complaint', 'nagging problem'],
            'approval': ['great value', 'recommend highly', 'worth visiting', 'excellent choice', 'approve of service',
                         'good decision', 'positive experience', 'thumbs up', 'well done', 'satisfied customer',
                         'endorse this place', 'favorable review'],
            'caring': ['helpful staff', 'thoughtful service', 'caring host', 'attentive care', 'warm welcome',
                       'supportive environment', 'kind gesture', 'empathetic response', 'nurturing atmosphere',
                       'considerate amenities', 'gentle handling', 'protective measures'],
            'confusion': ['confusing layout', 'unclear directions', 'mixed signals', 'baffling menu', 'puzzling rules',
                          'disorienting paths', 'bewildering experience', 'uncertain about quality',
                          'muddled instructions',
                          'perplexing pricing', 'lost in crowd'],
            'curiosity': ['intriguing history', 'mysterious ruins', 'curious artifacts', 'exploring hidden spots',
                          'wondering about', 'fascinating facts', 'inquisitive tour', 'eager to discover',
                          'piqued interest', 'questioning origins', 'alluring mystery'],
            'desire': ['craving food', 'want to return', 'longing for relaxation', 'eager to stay', 'desire luxury',
                       'yearning for adventure', 'wishing for more', 'tempting menu', 'hankering for view',
                       'coveting experience', 'aspiring visit'],
            'disappointment': ['below expectations', 'let down', 'disappointing food', 'failed promise',
                               'regret visiting',
                               'underwhelming view', 'dashed hopes', 'mediocre service', 'unfulfilled hype',
                               'sad letdown',
                               'frustrated outcome', 'disheartening stay'],
            'disapproval': ['poor quality', 'not recommended', 'disapprove of hygiene', 'bad choice',
                            'unacceptable behavior', 'frown upon', 'negative review', 'criticize management',
                            'object to noise', 'condemn facilities', 'reject this place', 'dislike strongly'],
            'disgust': ['dirty room', 'filthy bathroom', 'disgusting smell', 'revolting food', 'nasty insects',
                        'gross hygiene', 'repulsive odor', 'sickening sight', 'appalling cleanliness',
                        'vile conditions',
                        'nauseating experience', 'kadu'],
            'embarrassment': ['awkward situation', 'embarrassing mistake', 'humiliating service', 'shameful experience',
                              'cringeworthy moment', 'red-faced error', 'mortifying incident', 'uncomfortable vibe',
                              'disgraceful handling', 'belittling staff', 'shaming review'],
            'excitement': ['thrilling adventure', 'exciting activities', 'buzzing atmosphere', 'electrifying event',
                           'pumped up', 'adrenaline rush', 'vibrant energy', 'exhilarating view', 'heart-pounding fun',
                           'eager anticipation', 'lively crowd', 'dynamic place'],
            'fear': ['scary area', 'unsafe at night', 'frightening crowd', 'alarming noise', 'terrifying experience',
                     'nerve-wracking path', 'dreadful security', 'intimidating surroundings', 'fearful of theft',
                     'spooky ambiance', 'anxious about safety'],
            'gratitude': ['thankful for service', 'appreciative host', 'grateful experience', 'thanks to staff',
                          'obliged for help', 'indebted to', 'appreciate kindness', 'thankful view', 'gracious welcome',
                          'blessed stay', 'heartfelt thanks', 'nandri'],
            'grief': ['tragic loss', 'mournful memory', 'heartbreaking event', 'sorrowful place', 'grieving over',
                      'painful reminder', 'devastating news', 'lamenting failure', 'woeful experience',
                      'bereaved feeling',
                      'mourn loss', 'deep sorrow'],
            'joy': ['happy stay', 'joyful experience', 'delightful food', 'cheerful atmosphere', 'blissful relaxation',
                    'ecstatic view', 'gleeful moments', 'merry crowd', 'uplifting vibe', 'fun celebration',
                    'radiant happiness', 'santhosam'],
            'love': ['adore this place', 'love the view', 'cherish memories', 'passionate about', 'fond of service',
                     'heartwarming', 'beloved spot', 'affectionate welcome', 'endearing charm', 'romantic setting',
                     'treasure experience', 'priyam'],
            'nervousness': ['anxious wait', 'nervous about safety', 'tense atmosphere', 'apprehensive crowd',
                            'worried service', 'uneasy feeling', 'jittery experience', 'fidgety moments',
                            'restless night',
                            'edgy vibe', 'nervous anticipation'],
            'optimism': ['hopeful return', 'positive outlook', 'optimistic about', 'bright future visit',
                         'encouraging signs', 'upbeat review', 'promising place', 'confident recommendation',
                         'hopeful improvement', 'cheerful prospects', 'asai'],
            'pride': ['proud achievement', 'pride in heritage', 'boastful review', 'honored to visit', 'self-satisfied',
                      'dignified place', 'prestigious location', 'arrogant charm', 'vainglorious staff',
                      'noble feeling',
                      'elevated status'],
            'realization': ['sudden insight', 'eye-opening experience', 'dawning awareness', 'realized truth',
                            'aha moment',
                            'epiphany about quality', 'uncovered fact', 'revelation in review', 'discovered hidden gem',
                            'understood issue', 'clarifying visit'],
            'relief': ['relieved after', 'sigh of relief', 'eased tension', 'comforting end', 'stress-free stay',
                       'calming atmosphere', 'soothing experience', 'unburdened feeling', 'relaxed finally',
                       'alleviated worry', 'peaceful resolution'],
            'remorse': ['regret choosing', 'sorry for visit', 'remorseful review', 'guilty pleasure', 'apologetic tone',
                        'rueful experience', 'penitent feeling', 'contrite about', 'ashamed of choice',
                        'repentant stay',
                        'sorrowful regret'],
            'sadness': ['sad experience', 'depressing place', 'heartbreaking view', 'melancholy atmosphere',
                        'downcast mood', 'gloomy stay', 'tearful memory', 'mournful night', 'despondent review',
                        'woeful disappointment', 'dukka'],
            'surprise': ['unexpected delight', 'shocking discovery', 'surprising quality', 'astonishing view',
                         'jaw-dropping moment', 'unforeseen issue', 'startling event', 'amazing twist',
                         'pleasant shock',
                         'unanticipated fun', 'bewildering surprise']
        }

        # Add trigger words as nodes and connect to emotions
        for emotion, words in triggers.items():
            for word in words:
                G.add_node(word)
                G.add_edge(word, emotion, weight=0.1)

        # Add negation words if requested
        if include_negation:
            negations = ['not', 'no', 'never', 'isnt', 'arent', 'wasnt', 'wont', 'cant', 'dont', 'didnt']
            for neg in negations:
                G.add_node(neg)
                for emotion in self.emotion_labels:
                    G.add_edge(neg, emotion, weight=-0.1)

        return G

    def apply_kg_boost_to_emotions(self, emotion_probs, text, use_negation=True):
        """Apply KG boost to emotion probabilities with negation handling"""
        kg = self.build_emotion_kg(include_negation=use_negation)
        refined_probs = np.copy(emotion_probs)

        # Tokenize text
        tokens = word_tokenize(text.lower())
        adjustment = np.zeros(len(self.emotion_labels))

        print(f"🔍 Analyzing tokens: {tokens}")

        for j, token in enumerate(tokens):
            if token in kg:
                print(f"   📍 Found trigger word: '{token}'")

                for neighbor in kg.neighbors(token):
                    if neighbor in self.emotion_labels:
                        idx = self.emotion_labels.index(neighbor)
                        weight = kg[token][neighbor]['weight']

                        # Check for negation (previous token)
                        if use_negation and j > 0 and tokens[j - 1] in ['not', 'no', 'never', 'isnt', 'arent', 'wasnt',
                                                                        'wont', 'cant', 'dont', 'didnt']:
                            print(f"   🚫 NEGATION detected for '{neighbor}': flipping sentiment")

                            # Apply negative boost to the original emotion
                            adjustment[idx] += weight * -1

                            # Boost opposite emotions
                            for opp in self.OPPOSITES.get(neighbor, []):
                                if opp in self.emotion_labels:
                                    opp_idx = self.emotion_labels.index(opp)
                                    adjustment[opp_idx] += 0.05
                                    print(f"   ⬆️ Boosting opposite emotion '{opp}' by 0.05")
                        else:
                            # Normal positive boost
                            adjustment[idx] += weight
                            print(f"   ⬆️ Boosting '{neighbor}' by {weight}")

        # Apply adjustments
        refined_probs += adjustment
        refined_probs = np.clip(refined_probs, 0, 1)

        # Show significant changes
        changes = adjustment[np.abs(adjustment) > 0.01]
        if len(changes) > 0:
            print(f"🎯 Significant KG adjustments applied: {len(changes)} emotions modified")

        return refined_probs

    def get_enhanced_emotion_predictions(self, text, mode='kg_with_neg'):
        """Get emotion predictions with KG boost options"""
        print(f"📊 Getting emotion predictions (mode: {mode})...")

        # Get base emotion predictions
        base_emotions = get_emotion_prediction(text)

        if mode == 'raw':
            return base_emotions
        elif mode == 'kg_no_neg':
            return self.apply_kg_boost_to_emotions(base_emotions, text, use_negation=False)
        elif mode == 'kg_with_neg':
            return self.apply_kg_boost_to_emotions(base_emotions, text, use_negation=True)
        else:
            return base_emotions

    def extract_features_from_text(self, text, emotion_predictions):
        """Extract features exactly like in training (includes neutral)"""

        # Pad emotion predictions to include neutral (set to 0)
        full_emotion_predictions = np.append(emotion_predictions, 0.0)  # Add neutral = 0

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

        # Emotion features (including neutral)
        for i, emotion in enumerate(self.feature_emotion_labels):
            if i < len(full_emotion_predictions):
                features[f'emotion_{emotion}'] = full_emotion_predictions[i]
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

        return feature_array, features

    def test_text_comprehensive(self, text, modes=['raw', 'kg_no_neg', 'kg_with_neg']):
        """Test text with different KG boost modes"""
        print(f"\n🔍 COMPREHENSIVE TESTING: '{text}'")
        print("=" * 80)

        results = {}

        for mode in modes:
            print(f"\n🎯 MODE: {mode.upper()}")
            print("-" * 40)

            # Get emotion predictions with current mode
            emotion_predictions = self.get_enhanced_emotion_predictions(text, mode)

            # Extract features
            features, feature_dict = self.extract_features_from_text(text, emotion_predictions)

            # Predict sarcasm
            prediction, probabilities = predict_sarcasm(features, self.sarcasm_model, self.scaler, self.device)

            # Store results
            results[mode] = {
                'prediction': prediction,
                'probabilities': probabilities,
                'emotion_predictions': emotion_predictions,
                'features': feature_dict
            }

            # Display results
            print(f"🎭 Sarcasm: {'SARCASTIC' if prediction == 1 else 'NOT SARCASTIC'}")
            print(f"📊 Confidence: {probabilities[prediction]:.4f}")
            print(f"🎯 Probabilities: Non-Sarcastic: {probabilities[0]:.4f}, Sarcastic: {probabilities[1]:.4f}")

            # Top emotions
            top_emotions_idx = np.argsort(emotion_predictions)[-3:][::-1]
            print(f"🎨 Top 3 Emotions:")
            for idx in top_emotions_idx:
                if idx < len(self.emotion_labels):
                    emotion_name = self.emotion_labels[idx]
                    emotion_score = emotion_predictions[idx]
                    print(f"   {emotion_name}: {emotion_score:.4f}")

            # Key sarcasm indicators
            print(f"🔥 Key Features:")
            print(f"   Positive Sum: {feature_dict['positive_sum']:.3f}")
            print(f"   Negative Sum: {feature_dict['negative_sum']:.3f}")
            print(f"   Emotional Polarity: {feature_dict['emotional_polarity']:.3f}")
            print(f"   Joy-Anger Contrast: {feature_dict['joy_anger_contrast']:.3f}")

        # Compare results across modes
        print(f"\n📊 COMPARISON ACROSS MODES:")
        print("-" * 40)
        for mode in modes:
            pred = results[mode]['prediction']
            conf = results[mode]['probabilities'][pred]
            print(f"{mode:12} | {'SARCASTIC' if pred == 1 else 'NOT SARCASTIC':13} | Confidence: {conf:.4f}")

        return results

    def test_text(self, text, mode='kg_with_neg'):
        """Test a single text for sarcasm with specified mode"""
        print(f"\n🔍 Testing: '{text}' (Mode: {mode})")

        # Get enhanced emotion predictions
        emotion_predictions = self.get_enhanced_emotion_predictions(text, mode)

        # Extract features
        features, feature_dict = self.extract_features_from_text(text, emotion_predictions)

        # Predict sarcasm
        prediction, probabilities = predict_sarcasm(features, self.sarcasm_model, self.scaler, self.device)

        # Display results
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

        # Key sarcasm features
        print(f"🔥 Sarcasm Indicators:")
        print(f"   Emotional Polarity: {feature_dict['emotional_polarity']:.3f}")
        print(f"   Joy-Anger Contrast: {feature_dict['joy_anger_contrast']:.3f}")
        print(f"   Pos/Neg Ratio: {feature_dict['pos_neg_ratio']:.3f}")

        print("=" * 60)

        return prediction, probabilities


def main():
    """Interactive testing with KG boost options"""
    try:
        tester = EnhancedSarcasmTester()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure your model files exist at the specified paths!")
        return

    print("\n🎯 ENHANCED SARCASM DETECTION WITH KG BOOST")
    print("Available modes: raw, kg_no_neg, kg_with_neg")
    print("Type 'quit' to exit\n")

    # Test samples
    test_samples = [
        "I am going to hostel after a long holiday at home with my parents, sad",
        "Oh great, another meeting!",
        "I love working overtime on weekends",
        "This is not a great day",
        "Yeah, because that's exactly what I wanted to hear",
        "I'm not really happy about this situation"
    ]

    print("🧪 Testing sample texts with comprehensive analysis:")
    for sample in test_samples:
        try:
            tester.test_text_comprehensive(sample)
            input("\nPress Enter to continue...")
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
            # Ask for mode
            mode_input = input(
                "Choose mode (raw/kg_no_neg/kg_with_neg) or press Enter for comprehensive: ").strip().lower()

            try:
                if mode_input in ['raw', 'kg_no_neg', 'kg_with_neg']:
                    tester.test_text(user_input, mode_input)
                else:
                    tester.test_text_comprehensive(user_input)
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("Please enter some text to test!")


if __name__ == "__main__":
    main()