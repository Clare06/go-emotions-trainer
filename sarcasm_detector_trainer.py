import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 🎯 RTX 4050 OPTIMIZATION SETTINGS
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Device setup with RTX 4050 optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # RTX 4050 memory optimization
    torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of 6GB
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
else:
    print("⚠️ CUDA not available, using CPU")


class SarcasmFeatureDataset(Dataset):
    """Custom dataset for sarcasm features"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class OptimizedSarcasmDetector(nn.Module):
    """🔥 RTX 4050 Optimized Neural Network for Sarcasm Detection"""

    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
        super(OptimizedSarcasmDetector, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Batch normalization for stability
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 2))  # Binary classification

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


def load_and_prepare_data(data_folder):
    """Load and prepare the split dataset"""
    print("📊 Loading dataset...")

    train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_folder, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))

    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    # Define feature columns (exclude metadata)
    exclude_cols = ['sarcasm_label', 'original_text', 'text_preview', 'text_index', 'dominant_emotion']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    print(f"📈 Total features: {len(feature_cols)}")

    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['sarcasm_label'].values

    X_valid = valid_df[feature_cols].values
    y_valid = valid_df['sarcasm_label'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['sarcasm_label'].values

    # Handle any NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_valid = np.nan_to_num(X_valid, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale features for better performance
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train), (X_valid_scaled, y_valid), (X_test_scaled, y_test), scaler, feature_cols


def train_model(model, train_loader, valid_loader, num_epochs=20, learning_rate=0.001):
    """🚀 RTX 4050 Optimized Training Loop"""

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': [],
        'train_f1': [], 'valid_f1': []
    }

    best_valid_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"🏋️ Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())

            # Clear cache periodically for RTX 4050
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_preds, valid_labels = [], []

        with torch.no_grad():
            for batch_features, batch_labels in valid_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                valid_loss += loss.item()
                valid_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                valid_labels.extend(batch_labels.cpu().numpy())

        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds)
        valid_acc = accuracy_score(valid_labels, valid_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')

        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['valid_loss'].append(valid_loss / len(valid_loader))
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['train_f1'].append(train_f1)
        history['valid_f1'].append(valid_f1)

        # Learning rate scheduling
        scheduler.step(valid_loss / len(valid_loader))

        # Early stopping and best model saving
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Valid Loss: {valid_loss / len(valid_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Valid Acc: {valid_acc:.4f} | "
              f"Valid F1: {valid_f1:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping
        # if patience_counter >= 20:
        #     print(f"🛑 Early stopping at epoch {epoch + 1}")
        #     break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Best model loaded (F1: {best_valid_f1:.4f})")

    return model, history


def evaluate_model(model, test_loader, class_names=['Non-Sarcastic', 'Sarcastic']):
    """🎯 Comprehensive Model Evaluation"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("🧪 Evaluating on test set...")

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"\n📊 TEST SET RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\n📋 Classification Report:\n{report}")

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_training_history(history, save_path):
    """📊 Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['valid_loss'], label='Valid Loss', color='red')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='blue')
    axes[0, 1].plot(history['valid_acc'], label='Valid Accuracy', color='red')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train F1', color='blue')
    axes[1, 0].plot(history['valid_f1'], label='Valid F1', color='red')
    axes[1, 0].set_title('Training & Validation F1-Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Confusion Matrix placeholder
    axes[1, 1].text(0.5, 0.5, 'Confusion Matrix\n(See separate plot)',
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(labels, predictions, save_path):
    """📊 Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


class SarcasmPredictor:
    """🎯 Easy-to-use sarcasm prediction interface"""

    def __init__(self, model, scaler, feature_cols, device):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device

        # You'll need to import your emotion predictor
        from predict_emo import OptimizedEmotionPredictor
        self.emotion_predictor = OptimizedEmotionPredictor()

        # Sarcasm feature extraction logic
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
        ]

        self.positive_emotions = ['joy', 'love', 'excitement', 'gratitude', 'admiration',
                                  'optimism', 'amusement', 'pride', 'caring', 'approval']
        self.negative_emotions = ['anger', 'disgust', 'fear', 'grief', 'sadness',
                                  'disappointment', 'disapproval', 'annoyance', 'remorse']
        self.ambiguous_emotions = ['surprise', 'confusion', 'curiosity', 'realization']

    def extract_features_from_text(self, text):
        """Extract features from raw text (same as SarcasmFeatureExtractor)"""
        emotion_probs = self.emotion_predictor.predict_sentence_emotion_thread_safe(text)

        features = {}

        # Raw emotion probabilities (no duplicates)
        for i, label in enumerate(self.emotion_labels):
            features[f'{label}_prob'] = float(emotion_probs[i])

        # Statistical features
        features['max_prob'] = float(np.max(emotion_probs))
        features['min_prob'] = float(np.min(emotion_probs))
        features['mean_prob'] = float(np.mean(emotion_probs))
        features['variance'] = float(np.var(emotion_probs))
        features['std_dev'] = float(np.std(emotion_probs))
        features['dominant_emotion_idx'] = int(np.argmax(emotion_probs))

        # Sarcasm-specific features
        pos_sum = sum(emotion_probs[i] for i, label in enumerate(self.emotion_labels)
                      if label in self.positive_emotions)
        neg_sum = sum(emotion_probs[i] for i, label in enumerate(self.emotion_labels)
                      if label in self.negative_emotions)
        amb_sum = sum(emotion_probs[i] for i, label in enumerate(self.emotion_labels)
                      if label in self.ambiguous_emotions)

        features['positive_sum'] = float(pos_sum)
        features['negative_sum'] = float(neg_sum)
        features['ambiguous_sum'] = float(amb_sum)
        features['pos_neg_ratio'] = float(pos_sum / (neg_sum + 1e-8))
        features['emotional_polarity'] = float(pos_sum - neg_sum)

        # Advanced sarcasm detection features (key indicators!)
        joy_prob = emotion_probs[self.emotion_labels.index('joy')]
        anger_prob = emotion_probs[self.emotion_labels.index('anger')]
        love_prob = emotion_probs[self.emotion_labels.index('love')]
        disgust_prob = emotion_probs[self.emotion_labels.index('disgust')]

        features['incongruency_score'] = float(abs(joy_prob - anger_prob) + abs(love_prob - disgust_prob))
        features['emotional_flatness'] = float(1.0 - np.std(emotion_probs))

        sorted_probs = np.sort(emotion_probs)[::-1]
        features['peak_sharpness'] = float(sorted_probs[0] / (sorted_probs[1] + 1e-8))

        # More advanced features...
        approval_prob = emotion_probs[self.emotion_labels.index('approval')]
        disapproval_prob = emotion_probs[self.emotion_labels.index('disapproval')]
        excitement_prob = emotion_probs[self.emotion_labels.index('excitement')]
        disappointment_prob = emotion_probs[self.emotion_labels.index('disappointment')]
        gratitude_prob = emotion_probs[self.emotion_labels.index('gratitude')]
        annoyance_prob = emotion_probs[self.emotion_labels.index('annoyance')]
        surprise_prob = emotion_probs[self.emotion_labels.index('surprise')]
        amusement_prob = emotion_probs[self.emotion_labels.index('amusement')]

        features['approval_disapproval_gap'] = float(abs(approval_prob - disapproval_prob))
        features['excitement_disappointment_gap'] = float(abs(excitement_prob - disappointment_prob))
        features['gratitude_annoyance_gap'] = float(abs(gratitude_prob - annoyance_prob))

        # Complexity measures
        features['active_emotion_count'] = int(sum(1 for p in emotion_probs if p > 0.15))
        features['emotion_gini_coefficient'] = self.calculate_gini(emotion_probs)
        features['emotion_concentration'] = float(np.sum(emotion_probs ** 2))
        features['emotion_diversity_index'] = float(1.0 - np.sum(emotion_probs ** 2))

        # Threshold features
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            features[f'count_above_{threshold}'] = int(np.sum(emotion_probs > threshold))

        # Entropy
        non_zero_probs = emotion_probs[emotion_probs > 1e-8]
        if len(non_zero_probs) > 0:
            features['entropy'] = float(-np.sum(non_zero_probs * np.log(non_zero_probs)))
        else:
            features['entropy'] = 0.0

        # Advanced sarcasm patterns
        features['emotional_contradiction'] = float(min(pos_sum, neg_sum) * 2)
        features['surprise_amplified_contrast'] = float(surprise_prob * features['incongruency_score'])
        features['irony_indicator'] = float(approval_prob * disappointment_prob * 4)
        features['classic_sarcasm_pattern'] = float(amusement_prob * anger_prob * 3)
        features['polite_sarcasm_pattern'] = float(approval_prob * (neg_sum / (pos_sum + 1e-8)))
        features['bitter_joy_pattern'] = float(
            joy_prob * (emotion_probs[self.emotion_labels.index('sadness')] + anger_prob))

        return features

    def calculate_gini(self, probabilities):
        """Calculate Gini coefficient"""
        sorted_probs = np.sort(probabilities)
        n = len(sorted_probs)
        if n == 0 or np.sum(sorted_probs) == 0:
            return 0.0
        cumsum = np.cumsum(sorted_probs)
        gini = (n + 1 - 2 * np.sum(cumsum)) / (n * np.sum(sorted_probs))
        return float(max(0.0, min(1.0, gini)))

    def predict(self, text):
        """Predict sarcasm from raw text"""
        # Extract features
        features_dict = self.extract_features_from_text(text)

        # Convert to feature vector in correct order
        feature_vector = []
        for col in self.feature_cols:
            feature_vector.append(features_dict.get(col, 0.0))

        # Scale and predict
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Model prediction
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(feature_vector_scaled).to(self.device)
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])

        return {
            'text': text,
            'prediction': 'Sarcastic' if pred == 1 else 'Non-Sarcastic',
            'confidence': float(probs[pred]),
            'probabilities': {
                'non_sarcastic': float(probs[0]),
                'sarcastic': float(probs[1])
            },
            'key_features': {
                'incongruency_score': features_dict.get('incongruency_score', 0),
                'emotion_gini_coefficient': features_dict.get('emotion_gini_coefficient', 0),
                'emotional_contradiction': features_dict.get('emotional_contradiction', 0),
                'classic_sarcasm_pattern': features_dict.get('classic_sarcasm_pattern', 0)
            }
        }


def main():
    """🎯 Main training pipeline"""
    print("🚀 Starting RTX 4050 Optimized Sarcasm Detection Training")
    print(f"Device: {device}")

    # Configuration
    DATA_FOLDER = "cleaned_sarc"
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 25

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"sarcasm_training_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), scaler, feature_cols = load_and_prepare_data(DATA_FOLDER)

    print(f"📊 Feature count: {X_train.shape[1]}")
    print(f"📊 Class distribution - Train: {np.bincount(y_train)}")

    # Create datasets and loaders
    train_dataset = SarcasmFeatureDataset(X_train, y_train)
    valid_dataset = SarcasmFeatureDataset(X_valid, y_valid)
    test_dataset = SarcasmFeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    # Initialize model
    model = OptimizedSarcasmDetector(
        input_size=X_train.shape[1],
        hidden_sizes=[512, 256, 128],  # RTX 4050 optimized
        dropout_rate=0.4
    ).to(device)

    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("🏋️ Starting training...")
    trained_model, history = train_model(
        model, train_loader, valid_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )

    # Evaluate model
    print("🧪 Evaluating model...")
    test_results = evaluate_model(trained_model, test_loader)

    # Save model and scaler
    model_save_path = os.path.join(results_dir, "sarcasm_detector.pth")
    scaler_save_path = os.path.join(results_dir, "scaler.pkl")

    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'feature_cols': feature_cols,
        'model_config': {
            'input_size': X_train.shape[1],
            'hidden_sizes': [512, 256, 128],
            'dropout_rate': 0.4
        }
    }, model_save_path)

    import pickle
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"💾 Model saved to: {model_save_path}")
    print(f"💾 Scaler saved to: {scaler_save_path}")

    # Generate plots
    plot_training_history(history, os.path.join(results_dir, "training_history.png"))
    plot_confusion_matrix(test_results['labels'], test_results['predictions'],
                          os.path.join(results_dir, "confusion_matrix.png"))

    # Test the predictor
    print("\n🧪 Testing SarcasmPredictor...")
    predictor = SarcasmPredictor(trained_model, scaler, feature_cols, device)

    test_texts = [
        "Oh great, another meeting!",  # Sarcastic
        "I love this new feature!",  # Non-sarcastic
        "Perfect, just what I needed today...",  # Sarcastic
        "Thank you for your help!"  # Non-sarcastic
    ]

    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n📝 Text: '{text}'")
        print(f"🎯 Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
        print(f"🔍 Key features: {result['key_features']}")

    print(f"\n✅ Training completed! Results saved in: {results_dir}")

    return trained_model, scaler, feature_cols, test_results


if __name__ == "__main__":
    model, scaler, feature_cols, results = main()