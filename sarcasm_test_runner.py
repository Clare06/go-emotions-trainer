import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
from threading import Thread
import time
import traceback

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from sarcasm_detector import get_sarcasm_detector, ThreadSafeSarcasmDetector
except ImportError as e:
    print(f"❌ Error importing sarcasm detector: {e}")
    print("Make sure sarcasm_detector.py is in the same directory")
    sys.exit(1)


class SarcasmTestRunner:
    """🎯 Comprehensive Sarcasm Detection Test Runner"""

    def __init__(self, model_path=None, scaler_path=None):
        self.model_path = model_path or "models/sarcasm_detector.pth"
        self.scaler_path = scaler_path or "models/scaler.pkl"

        # Test results storage
        self.test_results = []
        self.performance_metrics = {}

        print("🚀 Initializing Sarcasm Test Runner...")

        # Load detector
        try:
            if model_path and scaler_path:
                self.detector = ThreadSafeSarcasmDetector(model_path, scaler_path)
            else:
                self.detector = get_sarcasm_detector()
            print("✅ Sarcasm detector loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading detector: {e}")
            self.detector = None

    def run_single_test(self, text, expected_label=None, test_name=""):
        """Run a single sarcasm detection test"""
        if not self.detector:
            return {"error": "Detector not loaded"}

        start_time = time.time()

        try:
            result = self.detector.predict_sarcasm(text)
            processing_time = time.time() - start_time

            test_result = {
                'test_name': test_name,
                'text': text,
                'predicted_label': result.get('prediction', 'Unknown'),
                'confidence': result.get('confidence', 0.0),
                'probabilities': result.get('probabilities', {}),
                'key_features': result.get('key_features', {}),
                'processing_time': processing_time,
                'expected_label': expected_label,
                'timestamp': datetime.now().isoformat()
            }

            # Add accuracy if expected label provided
            if expected_label:
                predicted_binary = 1 if result.get('prediction') == 'Sarcastic' else 0
                expected_binary = 1 if expected_label == 'Sarcastic' else 0
                test_result['correct'] = predicted_binary == expected_binary
                test_result['accuracy'] = 1.0 if test_result['correct'] else 0.0

            self.test_results.append(test_result)
            return test_result

        except Exception as e:
            error_result = {
                'test_name': test_name,
                'text': text,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            self.test_results.append(error_result)
            return error_result

    def run_batch_test(self, test_cases, show_progress=True):
        """Run multiple test cases"""
        print(f"\n🧪 Running batch test with {len(test_cases)} cases...")

        batch_results = []
        total_processing_time = 0

        for i, test_case in enumerate(test_cases):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(test_cases)} tests completed")

            if isinstance(test_case, dict):
                text = test_case.get('text', '')
                expected = test_case.get('expected', None)
                name = test_case.get('name', f'Test_{i + 1}')
            else:
                text = test_case
                expected = None
                name = f'Test_{i + 1}'

            result = self.run_single_test(text, expected, name)
            batch_results.append(result)

            if 'processing_time' in result:
                total_processing_time += result['processing_time']

        # Calculate batch statistics
        valid_results = [r for r in batch_results if 'error' not in r]
        error_count = len(batch_results) - len(valid_results)

        batch_stats = {
            'total_tests': len(test_cases),
            'successful_tests': len(valid_results),
            'failed_tests': error_count,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(test_cases) if test_cases else 0,
            'success_rate': len(valid_results) / len(test_cases) if test_cases else 0
        }

        # Calculate accuracy if expected labels provided
        accuracy_tests = [r for r in valid_results if 'correct' in r]
        if accuracy_tests:
            correct_predictions = sum(1 for r in accuracy_tests if r['correct'])
            batch_stats['accuracy'] = correct_predictions / len(accuracy_tests)
            batch_stats['accuracy_test_count'] = len(accuracy_tests)

        print(f"\n📊 Batch Test Results:")
        print(f"  ✅ Successful tests: {batch_stats['successful_tests']}")
        print(f"  ❌ Failed tests: {batch_stats['failed_tests']}")
        print(f"  ⏱️  Average processing time: {batch_stats['average_processing_time']:.3f}s")
        print(f"  📈 Success rate: {batch_stats['success_rate']:.1%}")

        if 'accuracy' in batch_stats:
            print(f"  🎯 Accuracy: {batch_stats['accuracy']:.1%} ({batch_stats['accuracy_test_count']} tests)")

        self.performance_metrics.update(batch_stats)
        return batch_results, batch_stats

    def run_performance_test(self, test_text="This is a test message", iterations=100):
        """Run performance benchmarking"""
        print(f"\n⚡ Running performance test ({iterations} iterations)...")

        processing_times = []
        successful_tests = 0

        start_time = time.time()

        for i in range(iterations):
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i + 1}/{iterations}")

            iteration_start = time.time()
            result = self.detector.predict_sarcasm(f"{test_text} - iteration {i}")
            iteration_time = time.time() - iteration_start

            processing_times.append(iteration_time)

            if 'error' not in result:
                successful_tests += 1

        total_time = time.time() - start_time

        perf_stats = {
            'total_iterations': iterations,
            'successful_iterations': successful_tests,
            'total_time': total_time,
            'average_time': np.mean(processing_times),
            'min_time': np.min(processing_times),
            'max_time': np.max(processing_times),
            'std_time': np.std(processing_times),
            'throughput': iterations / total_time,  # tests per second
            'success_rate': successful_tests / iterations
        }

        print(f"\n⚡ Performance Test Results:")
        print(f"  🎯 Throughput: {perf_stats['throughput']:.1f} tests/second")
        print(f"  ⏱️  Average time: {perf_stats['average_time']:.3f}s")
        print(f"  📊 Min/Max time: {perf_stats['min_time']:.3f}s / {perf_stats['max_time']:.3f}s")
        print(f"  📈 Success rate: {perf_stats['success_rate']:.1%}")

        self.performance_metrics.update(perf_stats)
        return perf_stats

    def run_thread_safety_test(self, num_threads=5, iterations_per_thread=20):
        """Test thread safety with concurrent requests"""
        print(f"\n🧵 Running thread safety test ({num_threads} threads, {iterations_per_thread} iterations each)...")

        results_dict = {}
        threads = []

        def thread_worker(thread_id):
            thread_results = []
            for i in range(iterations_per_thread):
                test_text = f"Thread {thread_id} test {i}: This is a sample sarcastic comment!"
                result = self.detector.predict_sarcasm(test_text)
                thread_results.append({
                    'thread_id': thread_id,
                    'iteration': i,
                    'text': test_text,
                    'result': result,
                    'timestamp': time.time()
                })
            results_dict[thread_id] = thread_results

        # Start all threads
        start_time = time.time()
        for thread_id in range(num_threads):
            thread = Thread(target=thread_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Analyze results
        all_results = []
        for thread_results in results_dict.values():
            all_results.extend(thread_results)

        successful_results = [r for r in all_results if 'error' not in r['result']]

        thread_safety_stats = {
            'num_threads': num_threads,
            'iterations_per_thread': iterations_per_thread,
            'total_tests': len(all_results),
            'successful_tests': len(successful_results),
            'total_time': total_time,
            'concurrent_throughput': len(all_results) / total_time,
            'success_rate': len(successful_results) / len(all_results) if all_results else 0
        }

        print(f"\n🧵 Thread Safety Test Results:")
        print(f"  🎯 Concurrent throughput: {thread_safety_stats['concurrent_throughput']:.1f} tests/second")
        print(f"  ✅ Successful tests: {thread_safety_stats['successful_tests']}/{thread_safety_stats['total_tests']}")
        print(f"  📈 Success rate: {thread_safety_stats['success_rate']:.1%}")
        print(f"  ⏱️  Total time: {thread_safety_stats['total_time']:.2f}s")

        self.performance_metrics.update(thread_safety_stats)
        return thread_safety_stats, results_dict

    def run_comprehensive_test_suite(self):
        """Run all test suites"""
        print("🚀 Starting Comprehensive Sarcasm Detection Test Suite")
        print("=" * 60)

        # Test cases with expected labels
        test_cases = [
            {"text": "Oh great, another meeting!", "expected": "Sarcastic", "name": "Classic_Sarcasm_1"},
            {"text": "Perfect, just what I needed today...", "expected": "Sarcastic", "name": "Classic_Sarcasm_2"},
            {"text": "Thank you for your help!", "expected": "Non-Sarcastic", "name": "Genuine_Thanks"},
            {"text": "I love this new feature!", "expected": "Non-Sarcastic", "name": "Genuine_Praise"},
            {"text": "Wow, this is absolutely brilliant!", "expected": "Sarcastic", "name": "Potential_Sarcasm"},
            {"text": "Sure, that makes perfect sense", "expected": "Sarcastic", "name": "Subtle_Sarcasm"},
            {"text": "The weather is nice today", "expected": "Non-Sarcastic", "name": "Neutral_Statement"},
            {"text": "I'm so excited to work overtime again", "expected": "Sarcastic", "name": "Work_Sarcasm"},
            {"text": "This movie was really entertaining", "expected": "Non-Sarcastic", "name": "Movie_Review"},
            {"text": "Oh wonderful, more homework!", "expected": "Sarcastic", "name": "Student_Sarcasm"},
        ]

        # 1. Batch Test
        batch_results, batch_stats = self.run_batch_test(test_cases)

        # 2. Performance Test
        perf_stats = self.run_performance_test()

        # 3. Thread Safety Test
        thread_stats, thread_results = self.run_thread_safety_test()

        # 4. Edge Cases Test
        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "a",  # Single character
            "This is a very long sentence that contains multiple clauses and should test the model's ability to handle longer inputs with various emotional indicators and potential sarcastic undertones that might be present in extended text.",
            # Long text
            "🙄 Sure, that's helpful",  # With emoji
            "CAPS LOCK SARCASM IS THE BEST!!!",  # All caps
            "...",  # Punctuation only
        ]

        print(f"\n🔍 Testing edge cases...")
        edge_results = []
        for i, text in enumerate(edge_cases):
            result = self.run_single_test(text, test_name=f"Edge_Case_{i + 1}")
            edge_results.append(result)

        # Generate comprehensive report
        self.generate_test_report()

        return {
            'batch_results': batch_results,
            'batch_stats': batch_stats,
            'performance_stats': perf_stats,
            'thread_safety_stats': thread_stats,
            'edge_case_results': edge_results,
            'all_test_results': self.test_results
        }

    def generate_test_report(self, save_to_file=True):
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            'test_summary': {
                'timestamp': timestamp,
                'total_tests': len(self.test_results),
                'performance_metrics': self.performance_metrics,
                'model_info': {
                    'model_path': self.model_path,
                    'scaler_path': self.scaler_path,
                    'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                }
            },
            'detailed_results': self.test_results
        }

        if save_to_file:
            # Create results directory
            results_dir = f"sarcasm_test_results_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)

            # Save JSON report
            with open(f"{results_dir}/test_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Save CSV with test results
            if self.test_results:
                df_results = pd.DataFrame(self.test_results)
                df_results.to_csv(f"{results_dir}/test_results.csv", index=False, encoding='utf-8')

            # Save summary text report
            with open(f"{results_dir}/summary.txt", 'w', encoding='utf-8') as f:
                f.write("🎯 SARCASM DETECTION TEST SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Test Date: {timestamp}\n")
                f.write(f"Total Tests: {len(self.test_results)}\n\n")

                f.write("📊 PERFORMANCE METRICS:\n")
                for key, value in self.performance_metrics.items():
                    f.write(f"  {key}: {value}\n")

                f.write("\n🔍 INDIVIDUAL TEST RESULTS:\n")
                for i, result in enumerate(self.test_results[:10]):  # First 10 tests
                    f.write(f"\nTest {i + 1}: {result.get('test_name', 'Unknown')}\n")
                    f.write(f"  Text: {result.get('text', 'N/A')[:100]}...\n")
                    f.write(f"  Prediction: {result.get('predicted_label', 'N/A')}\n")
                    f.write(f"  Confidence: {result.get('confidence', 0):.3f}\n")
                    if 'correct' in result:
                        f.write(f"  Correct: {result['correct']}\n")

            print(f"\n📁 Test report saved to: {results_dir}")
            return results_dir

        return report

    def display_sample_predictions(self, num_samples=5):
        """Display sample predictions with detailed analysis"""
        print(f"\n🔍 Sample Predictions (Last {num_samples} tests):")
        print("=" * 80)

        recent_results = self.test_results[-num_samples:] if self.test_results else []

        for i, result in enumerate(recent_results, 1):
            print(f"\n📝 Sample {i}:")
            print(f"  Text: '{result.get('text', 'N/A')}'")
            print(f"  🎯 Prediction: {result.get('predicted_label', 'Unknown')}")
            print(f"  🔐 Confidence: {result.get('confidence', 0):.3f}")

            if 'probabilities' in result:
                probs = result['probabilities']
                print(f"  📊 Probabilities:")
                print(f"    - Non-Sarcastic: {probs.get('non_sarcastic', 0):.3f}")
                print(f"    - Sarcastic: {probs.get('sarcastic', 0):.3f}")

            if 'key_features' in result and result['key_features']:
                print(f"  🔑 Key Features:")
                for feat, val in result['key_features'].items():
                    print(f"    - {feat}: {val:.4f}")

            if 'expected_label' in result:
                print(f"  ✅ Expected: {result['expected_label']}")
                print(f"  🎯 Correct: {result.get('correct', 'Unknown')}")

            print(f"  ⏱️  Processing Time: {result.get('processing_time', 0):.3f}s")
            print("-" * 80)


def main():
    """Main test runner function"""
    print("🎯 Sarcasm Detection Test Runner")
    print("=" * 50)

    # Initialize test runner with your model paths
    MODEL_PATH = "models/sarcasm_detector.pth"  # Update path as needed
    SCALER_PATH = "models/scaler.pkl"  # Update path as needed

    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script or ensure model files are in the correct location")
        return

    if not os.path.exists(SCALER_PATH):
        print(f"❌ Scaler file not found: {SCALER_PATH}")
        print("Please update SCALER_PATH in the script or ensure model files are in the correct location")
        return

    # Initialize test runner
    try:
        test_runner = SarcasmTestRunner(MODEL_PATH, SCALER_PATH)
    except Exception as e:
        print(f"❌ Failed to initialize test runner: {e}")
        traceback.print_exc()
        return

    # Run comprehensive test suite
    try:
        results = test_runner.run_comprehensive_test_suite()

        # Display sample predictions
        test_runner.display_sample_predictions(5)

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()