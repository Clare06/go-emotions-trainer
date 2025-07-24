#!/usr/bin/env python3
"""
Quick Sarcasm Detection Test Script
Usage: python quick_sarcasm_test.py
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from sarcasm_detector import get_sarcasm_detector


def quick_test():
    """Quick interactive test"""
    print("🎭 Quick Sarcasm Detection Test")
    print("=" * 40)

    # Load detector
    try:
        detector = get_sarcasm_detector()
        print("✅ Detector loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return

    # Test samples
    test_samples = [
        "Oh great, another meeting!",
        "Thank you for your help!",
        "Perfect, just what I needed today...",
        "I love this new feature!",
        "Sure, that makes perfect sense",
        "The weather is nice today"
    ]

    print("🧪 Testing sample texts:")
    print("-" * 40)

    for i, text in enumerate(test_samples, 1):
        print(f"\n📝 Test {i}: '{text}'")
        try:
            result = detector.predict_sarcasm(text)
            print(f"🎯 Prediction: {result['prediction']}")
            print(f"🔐 Confidence: {result['confidence']:.3f}")
            print(f"📊 Sarcastic Prob: {result['probabilities']['sarcastic']:.3f}")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Interactive testing
    print(f"\n{'=' * 40}")
    print("🎮 Interactive Testing (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            text = input("\n💬 Enter text to test: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not text:
                print("⚠️ Please enter some text")
                continue

            result = detector.predict_sarcasm(text)

            print(f"\n🎯 Result:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Probabilities:")
            print(f"    - Non-Sarcastic: {result['probabilities']['non_sarcastic']:.3f}")
            print(f"    - Sarcastic: {result['probabilities']['sarcastic']:.3f}")

            if 'key_features' in result:
                print(f"  Key Features:")
                for feat, val in result['key_features'].items():
                    print(f"    - {feat}: {val:.4f}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    quick_test()