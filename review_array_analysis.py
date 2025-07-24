import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
from scipy import stats
from collections import Counter
import re
import warnings

from predict_emo import OptimizedEmotionPredictor

warnings.filterwarnings('ignore')


class AdvancedReviewAggregator(OptimizedEmotionPredictor):
    """
    Advanced Review Sentiment Aggregator with Statistical Weighting

    Key Features:
    - Token-based quality weighting (not word count)
    - Emotion intensity scoring within positive/negative categories
    - Improved outlier detection focusing on quality/confidence
    - Returns normalized sentiment score (-1 to +1)
    """

    def __init__(self, model_path=None):
        super().__init__(model_path)

        # Define emotion intensity weights based on psychological research
        self.EMOTION_WEIGHTS = {
            # Positive emotions (weighted by intensity/impact)
            'joy': 1.0,  # Pure happiness - highest positive weight
            'love': 0.95,  # Deep affection - very strong
            'excitement': 0.90,  # High energy positive
            'gratitude': 0.85,  # Appreciation - strong positive
            'admiration': 0.80,  # Respect/awe - strong positive
            'optimism': 0.75,  # Hope for future - moderate-strong
            'amusement': 0.70,  # Light happiness - moderate
            'pride': 0.65,  # Self-satisfaction - moderate
            'caring': 0.60,  # Concern for others - moderate
            'approval': 0.55,  # Agreement/acceptance - mild-moderate
            'relief': 0.50,  # Stress removal - mild positive
            'desire': 0.45,  # Wanting something - mild positive

            # Negative emotions (weighted by severity/impact)
            'anger': -1.0,  # Rage - highest negative weight
            'disgust': -0.95,  # Revulsion - very strong negative
            'fear': -0.90,  # Terror/anxiety - very strong negative
            'grief': -0.85,  # Deep sadness - strong negative
            'sadness': -0.80,  # General sadness - strong negative
            'disappointment': -0.75,  # Unmet expectations - moderate-strong
            'disapproval': -0.70,  # Rejection - moderate negative
            'annoyance': -0.65,  # Irritation - moderate negative
            'embarrassment': -0.60,  # Shame - moderate negative
            'nervousness': -0.55,  # Anxiety - mild-moderate negative
            'remorse': -0.50,  # Regret - mild negative

            # Ambiguous emotions (context-dependent, lower weights)
            'surprise': 0.0,  # Can be positive or negative
            'confusion': -0.15,  # Slight negative (unclear communication)
            'curiosity': 0.10,  # Slight positive (engagement)
            'realization': 0.05,  # Neutral to slight positive (understanding)
        }

        # Review quality factors
        self.MIN_MEANINGFUL_LENGTH = 5  # tokens
        self.PARAGRAPH_THRESHOLD = 20  # tokens for paragraph weight
        self.MAX_REVIEW_WEIGHT = 5.0  # Cap for very long reviews
        self.MIN_REVIEW_WEIGHT = 0.1  # Floor for very short reviews

    def calculate_review_quality_weight_improved(self, review_text, emotion_count, processing_method):
        """
        Improved quality weighting using actual tokenizer and removing unnecessary penalties

        Args:
            review_text (str): Original review text
            emotion_count (int): Number of emotions detected
            processing_method (str): Processing method used

        Returns:
            dict: Quality weight information
        """
        # Get actual meaningful tokens (what the model actually processes)
        tokens = self.tokenizer.tokenize(review_text)
        meaningful_tokens = [t for t in tokens if
                             not t.startswith('##') and t not in ['[CLS]', '[SEP]', '<pad>', '<s>', '</s>']]
        token_count = len(meaningful_tokens)

        # Sentence structure analysis
        sentences = sent_tokenize(review_text)
        sentence_count = len(sentences)

        # Base weight using meaningful tokens (not raw word count)
        if token_count <= 2:
            quality_weight = 0.15
            quality_reason = "Very short review (1-2 tokens)"
        elif token_count <= 5:
            quality_weight = 0.4
            quality_reason = "Short review (3-5 tokens)"
        elif token_count <= 10:
            quality_weight = 0.7
            quality_reason = "Brief review (6-10 tokens)"
        elif token_count <= self.PARAGRAPH_THRESHOLD:
            quality_weight = 1.0
            quality_reason = "Standard review"
        else:
            # Logarithmic scaling for longer reviews
            quality_weight = min(1.5 + np.log10(token_count / self.PARAGRAPH_THRESHOLD), self.MAX_REVIEW_WEIGHT)
            quality_reason = f"Detailed review ({token_count} tokens)"

        bonuses = []
        penalties = []

        # Emotion diversity bonus (indicates thoughtful review)
        if emotion_count >= 3:
            emotion_bonus = min(0.3, emotion_count * 0.08)
            quality_weight += emotion_bonus
            bonuses.append(f"Emotion diversity (+{emotion_bonus:.2f})")

        # Multi-sentence structure bonus
        if sentence_count >= 2:
            structure_bonus = min(0.25, sentence_count * 0.08)
            quality_weight += structure_bonus
            bonuses.append(f"Multi-sentence (+{structure_bonus:.2f})")

        # Processing method bonus (paragraph analysis is more thorough)
        if processing_method == 'paragraph_weighted':
            paragraph_bonus = 0.4
            quality_weight += paragraph_bonus
            bonuses.append(f"Paragraph analysis (+{paragraph_bonus:.2f})")

        # Only keep extreme repetition penalty (like "good good good good good")
        words = review_text.lower().split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # More than 70% repetitive
                repetition_penalty = -0.4
                quality_weight += repetition_penalty
                penalties.append(f"Extreme repetition (-{abs(repetition_penalty):.2f})")

        # Bounds
        quality_weight = max(self.MIN_REVIEW_WEIGHT, min(quality_weight, self.MAX_REVIEW_WEIGHT))

        return {
            'weight': round(quality_weight, 3),
            'base_reason': quality_reason,
            'bonuses': bonuses,
            'penalties': penalties,
            'token_count': token_count,
            'meaningful_tokens': meaningful_tokens[:10],  # Show first 10 for debugging
            'sentence_count': sentence_count
        }

    def calculate_weighted_sentiment_score(self, emotions_dict, emotion_confidence):
        """
        Calculate weighted sentiment score using emotion intensities

        Args:
            emotions_dict (dict): Detected emotions with probabilities
            emotion_confidence (dict): Confidence scores for emotions

        Returns:
            dict: Sentiment analysis with weighted scores
        """
        total_weighted_score = 0.0
        total_confidence = 0.0
        emotion_breakdown = {}

        # Calculate weighted sentiment for each detected emotion
        for emotion, probability in emotions_dict.items():
            if emotion in self.EMOTION_WEIGHTS:
                emotion_weight = self.EMOTION_WEIGHTS[emotion]
                confidence = emotion_confidence.get(emotion, probability)

                # Weighted contribution = emotion_intensity × probability × confidence
                contribution = emotion_weight * probability * confidence
                total_weighted_score += contribution
                total_confidence += confidence

                emotion_breakdown[emotion] = {
                    'probability': round(probability, 4),
                    'intensity_weight': emotion_weight,
                    'confidence': round(confidence, 4),
                    'contribution': round(contribution, 4)
                }

        # Normalize by total confidence to get final sentiment score
        if total_confidence > 0:
            normalized_sentiment = total_weighted_score / total_confidence
        else:
            normalized_sentiment = 0.0

        # Ensure within [-1, 1] range
        normalized_sentiment = max(-1.0, min(1.0, normalized_sentiment))

        # Calculate sentiment statistics
        positive_emotions = [e for e, w in self.EMOTION_WEIGHTS.items() if w > 0]
        negative_emotions = [e for e, w in self.EMOTION_WEIGHTS.items() if w < 0]
        ambiguous_emotions = [e for e, w in self.EMOTION_WEIGHTS.items() if w == 0]

        positive_count = sum(1 for e in emotions_dict.keys() if e in positive_emotions)
        negative_count = sum(1 for e in emotions_dict.keys() if e in negative_emotions)
        ambiguous_count = sum(1 for e in emotions_dict.keys() if e in ambiguous_emotions)

        return {
            'weighted_sentiment_score': round(normalized_sentiment, 4),
            'emotion_breakdown': emotion_breakdown,
            'sentiment_stats': {
                'positive_emotions': positive_count,
                'negative_emotions': negative_count,
                'ambiguous_emotions': ambiguous_count,
                'total_confidence': round(total_confidence, 4),
                'raw_weighted_score': round(total_weighted_score, 4)
            }
        }

    def improved_outlier_detection(self, reviews_data, method='confidence_based'):
        """
        Better outlier detection focusing on review quality, not just sentiment extremes

        Args:
            reviews_data (list): List of individual review results
            method (str): Detection method ('confidence_based' or 'quality_based')

        Returns:
            dict: Outlier analysis results
        """
        if len(reviews_data) < 4:
            return {
                'outlier_indices': [],
                'method': 'insufficient_data',
                'total_reviews': len(reviews_data),
                'outliers_detected': 0
            }

        outlier_indices = []

        if method == 'confidence_based':
            # Focus on reviews with suspiciously low model confidence
            confidences = []
            for i, review_data in enumerate(reviews_data):
                # Calculate average confidence across predicted emotions
                emotions = review_data.get('sentiment_analysis', {}).get('emotion_breakdown', {})
                if emotions:
                    avg_confidence = np.mean([data['confidence'] for data in emotions.values()])
                    confidences.append((i, avg_confidence))
                else:
                    confidences.append((i, 0.0))

            # Mark reviews with very low confidence as potential outliers
            confidence_scores = [conf for _, conf in confidences]
            if len(confidence_scores) > 1:
                threshold = np.percentile(confidence_scores, 10)  # Bottom 10%
                outlier_indices = [i for i, conf in confidences if conf < threshold and conf < 0.3]

        elif method == 'quality_based':
            # Focus on reviews with very low quality weights
            quality_weights = [review.get('quality_weight_info', {}).get('weight', 1.0)
                               for review in reviews_data]
            threshold = np.percentile(quality_weights, 5)  # Bottom 5%
            outlier_indices = [i for i, weight in enumerate(quality_weights)
                               if weight < threshold and weight < 0.2]

        return {
            'outlier_indices': outlier_indices,
            'method': method,
            'total_reviews': len(reviews_data),
            'outliers_detected': len(outlier_indices),
            'outlier_percentage': round(len(outlier_indices) / len(reviews_data) * 100, 2) if reviews_data else 0
        }

    def detect_outlier_reviews(self, review_scores, method='iqr', threshold=1.5):
        """
        Detect outlier reviews using statistical methods (fallback method)

        Args:
            review_scores (list): List of sentiment scores
            method (str): 'iqr' or 'zscore'
            threshold (float): Threshold for outlier detection

        Returns:
            dict: Outlier analysis results
        """
        scores_array = np.array(review_scores)

        if len(scores_array) < 4:
            return {
                'outlier_indices': [],
                'method': 'insufficient_data',
                'total_reviews': len(scores_array),
                'outliers_detected': 0
            }

        outlier_indices = []

        if method == 'iqr':
            Q1 = np.percentile(scores_array, 25)
            Q3 = np.percentile(scores_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_indices = [i for i, score in enumerate(scores_array)
                               if score < lower_bound or score > upper_bound]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(scores_array))
            outlier_indices = [i for i, z in enumerate(z_scores) if z > threshold]

        return {
            'outlier_indices': outlier_indices,
            'method': method,
            'total_reviews': len(scores_array),
            'outliers_detected': len(outlier_indices),
            'outlier_percentage': round(len(outlier_indices) / len(scores_array) * 100, 2)
        }

    def aggregate_reviews_sentiment(self, reviews,
                                    return_detailed=True,
                                    outlier_detection=True,
                                    min_confidence=0.0,
                                    translation=False):
        """
        🚀 MAIN AGGREGATION FUNCTION 🚀

        Analyzes multiple reviews and returns aggregated sentiment score (-1 to +1)

        Args:
            reviews (list): List of review texts
            return_detailed (bool): Return detailed analysis for each review
            outlier_detection (bool): Apply outlier detection and handling
            min_confidence (float): Minimum confidence threshold for emotions
            translation (bool): Attempt translation from other languages

        Returns:
            dict: Comprehensive aggregated sentiment analysis
        """

        if not reviews or len(reviews) == 0:
            return {'error': 'No reviews provided'}

        print(f"🔍 Processing {len(reviews)} reviews...")

        individual_results = []
        all_sentiment_scores = []
        all_quality_weights = []
        all_emotions_aggregated = Counter()
        processing_stats = {'paragraph_mode': 0, 'single_text_mode': 0, 'errors': 0}

        # Step 1: Process each review individually
        for i, review in enumerate(reviews):
            print(f"   📝 Processing review {i + 1}/{len(reviews)}")

            try:
                # Analyze single review with detailed insights
                result = self.analyze_text_detailed(review, use_paragraph_mode=None)

                if 'error' in result:
                    processing_stats['errors'] += 1
                    continue

                # Track processing method
                if result.get('processing_method') == 'paragraph_weighted':
                    processing_stats['paragraph_mode'] += 1
                else:
                    processing_stats['single_text_mode'] += 1

                # Get emotions (handle both single text and paragraph results)
                if 'filtered_emotions' in result:
                    # Paragraph mode
                    emotions = result['filtered_emotions']
                    emotion_confidence = {e: prob for e, prob in emotions.items()}
                else:
                    # Single text mode - extract from detailed_scores
                    emotions = {e: scores['probability']
                                for e, scores in result.get('detailed_scores', {}).items()
                                if scores.get('predicted', False)}
                    emotion_confidence = {e: scores['probability']
                                          for e, scores in result.get('detailed_scores', {}).items()}

                # Calculate quality weight for this review - FIXED METHOD NAME AND PARAMETERS
                quality_info = self.calculate_review_quality_weight_improved(
                    review,
                    result['emotion_count'],
                    result.get('processing_method', 'single_text')
                )

                # Calculate weighted sentiment score
                sentiment_analysis = self.calculate_weighted_sentiment_score(emotions, emotion_confidence)

                # Store individual result
                individual_result = {
                    'review_index': i,
                    'review_text': review,
                    'review_preview': review[:100] + '...' if len(review) > 100 else review,
                    'processing_method': result.get('processing_method', 'single_text'),
                    'emotions_detected': result['predicted_emotions'],
                    'emotion_count': result['emotion_count'],
                    'quality_weight_info': quality_info,
                    'sentiment_analysis': sentiment_analysis,
                    'original_sentiment': result.get('sentiment_analysis', {})
                }

                individual_results.append(individual_result)
                all_sentiment_scores.append(sentiment_analysis['weighted_sentiment_score'])
                all_quality_weights.append(quality_info['weight'])

                # Aggregate emotions across all reviews
                for emotion in result['predicted_emotions']:
                    all_emotions_aggregated[emotion] += quality_info['weight']

            except Exception as e:
                print(f"❌ Error processing review {i + 1}: {str(e)}")
                processing_stats['errors'] += 1
                continue

        if len(individual_results) == 0:
            return {'error': 'No reviews could be processed successfully'}

        print(f"✅ Successfully processed {len(individual_results)} reviews")

        # Step 2: Outlier Detection and Handling
        outlier_info = {}
        final_sentiment_scores = all_sentiment_scores.copy()
        final_quality_weights = all_quality_weights.copy()

        if outlier_detection and len(all_sentiment_scores) >= 4:
            # Use improved outlier detection method
            outlier_info = self.improved_outlier_detection(individual_results, method='confidence_based')

            if outlier_info['outliers_detected'] > 0:
                print(f"🚨 Detected {outlier_info['outliers_detected']} outlier reviews")

                # Reduce weight of outlier reviews (don't completely remove them)
                for outlier_idx in outlier_info['outlier_indices']:
                    if outlier_idx < len(final_quality_weights):
                        original_weight = final_quality_weights[outlier_idx]
                        final_quality_weights[outlier_idx] *= 0.3  # Reduce to 30% of original weight
                        individual_results[outlier_idx]['outlier_status'] = {
                            'is_outlier': True,
                            'original_weight': original_weight,
                            'reduced_weight': final_quality_weights[outlier_idx],
                            'outlier_method': outlier_info['method']
                        }

        # Step 3: Calculate Final Aggregated Sentiment Score
        if len(final_sentiment_scores) > 0:
            # Weighted average sentiment score
            weighted_sum = sum(score * weight for score, weight in zip(final_sentiment_scores, final_quality_weights))
            total_weight = sum(final_quality_weights)

            final_aggregated_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
            final_aggregated_sentiment = max(-1.0, min(1.0, final_aggregated_sentiment))  # Ensure [-1, 1]

        else:
            final_aggregated_sentiment = 0.0

        # Step 4: Calculate Confidence Metrics
        score_std = np.std(all_sentiment_scores) if len(all_sentiment_scores) > 1 else 0.0
        confidence_score = max(0.0, 1.0 - (score_std * 2))  # Higher std = lower confidence

        # Step 5: Get Top 6 Emotions (weighted by quality)
        top_6_emotions = [emotion for emotion, count in all_emotions_aggregated.most_common(6)]

        # Step 6: Sentiment Category Classification
        if final_aggregated_sentiment > 0.1:
            sentiment_category = 'Positive'
        elif final_aggregated_sentiment < -0.1:
            sentiment_category = 'Negative'
        else:
            sentiment_category = 'Neutral'

        # Intensity classification
        abs_sentiment = abs(final_aggregated_sentiment)
        if abs_sentiment > 0.7:
            intensity = 'Very Strong'
        elif abs_sentiment > 0.5:
            intensity = 'Strong'
        elif abs_sentiment > 0.3:
            intensity = 'Moderate'
        else:
            intensity = 'Mild'

        # Build final result
        aggregated_result = {
            # 🎯 MAIN OUTPUT
            'final_sentiment_score': round(final_aggregated_sentiment, 4),  # [-1, 1]
            'sentiment_category': sentiment_category,
            'intensity': intensity,
            'confidence_score': round(confidence_score, 4),

            # 📊 TOP INSIGHTS
            'top_6_emotions': top_6_emotions,
            'total_reviews_processed': len(individual_results),
            'emotion_distribution': dict(all_emotions_aggregated.most_common(10)),

            # 📈 STATISTICS
            'sentiment_statistics': {
                'mean_sentiment': round(np.mean(all_sentiment_scores), 4),
                'median_sentiment': round(np.median(all_sentiment_scores), 4),
                'std_sentiment': round(score_std, 4),
                'min_sentiment': round(np.min(all_sentiment_scores), 4),
                'max_sentiment': round(np.max(all_sentiment_scores), 4),
                'quality_weight_stats': {
                    'mean_weight': round(np.mean(all_quality_weights), 4),
                    'total_weight': round(sum(all_quality_weights), 4)
                }
            },

            # 🔍 PROCESSING INFO
            'processing_stats': processing_stats,
            'outlier_analysis': outlier_info,

            # 📝 DETAILED RESULTS (if requested)
            'individual_reviews': individual_results if return_detailed else f"{len(individual_results)} reviews processed"
        }

        print(f"🎉 Final Aggregated Sentiment: {final_aggregated_sentiment:.4f} ({sentiment_category} - {intensity})")
        print(f"🏆 Top 6 Emotions: {', '.join(top_6_emotions)}")

        return aggregated_result


# Example usage function
def analyze_array_reviews(reviews_list):
    """
    Convenience function to analyze restaurant/place reviews

    Args:
        reviews_list (list): List of review strings

    Returns:
        dict: Sentiment analysis results with score from -1 to +1
    """
    # Initialize the aggregator
    aggregator = AdvancedReviewAggregator()

    # Analyze reviews
    results = aggregator.aggregate_reviews_sentiment(
        reviews=reviews_list,
        return_detailed=True,  # Set to False for API efficiency
        outlier_detection=True,
        min_confidence=0.0
    )

    return results


# 🧪 TESTING SECTION
if __name__ == "__main__":
    # Test with sample reviews of varying quality
    test_reviews = [
        "2024 December Very old place we got their best room called sweet room But Ac machine was not adjustable Half of the place is surrounded by swamp land so bad odour and mosquitoes Can't stay outside Not maintained when reserving room they said there is a restaurant but No restaurant They were not willing to provide snacks or soft drink even we request provided dinner also in bad quality Even it is beach front the road leading to the beach is crowded with fishermen and drunkards Feel not safe",
        "Great!",  # One word - should get low weight
        "Amazing food and service, really enjoyed our time here. The staff was friendly and the ambiance was perfect for a romantic dinner.",
        "ok",  # One word - very low weight
        "Terrible experience. Food was cold, service was slow, and the place was dirty. Would never recommend this to anyone. The manager was rude and didn't care about our complaints. Worst restaurant ever!",
        "Good good good good good",  # Repetitive - should get penalized
        "ABSOLUTELY AMAZING BEST PLACE EVER!!!",  # No caps penalty now
        "The restaurant has excellent food quality, great ambiance, and outstanding service. We ordered the seafood platter and it was fresh and delicious. The staff was attentive and made sure we had everything we needed. Highly recommend for special occasions."
    ]

    print("🚀 TESTING ADVANCED REVIEW AGGREGATOR")
    print("=" * 60)

    results = analyze_array_reviews(test_reviews)
    print(results['final_sentiment_score'])
    if 'error' not in results:
        print(f"\n🎯 FINAL RESULT:")
        print(
            f"   Sentiment Score: {results['final_sentiment_score']} ({results['sentiment_category']} - {results['intensity']})")
        print(f"   Top 6 Emotions: {', '.join(results['top_6_emotions'])}")
        print(f"   Confidence: {results['confidence_score']:.3f}")
        print(f"   Reviews Processed: {results['total_reviews_processed']}")

        if results.get('outlier_analysis', {}).get('outliers_detected', 0) > 0:
            print(f"   Outliers Detected: {results['outlier_analysis']['outliers_detected']}")
    else:
        print(f"❌ Error: {results['error']}")