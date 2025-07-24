import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")


class SarcasmFeatureDataset(Dataset):
    """Custom dataset for sarcasm features"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ImbalanceAwareSarcasmDetector(nn.Module):
    """🔥 Class Imbalance Aware Neural Network"""

    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.5):
        super(ImbalanceAwareSarcasmDetector, self).__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 2))

        self.network = nn.Sequential(*layers)

        # Focal loss parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


class FocalLoss(nn.Module):
    """🎯 Focal Loss for handling class imbalance"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_balanced_sampler(labels):
    """Create weighted sampler to balance classes"""
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def load_and_prepare_data(data_folder):
    """Load and prepare the split dataset with class analysis"""
    print("📊 Loading dataset...")

    train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_folder, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))

    # Analyze class distribution
    print("\n📈 CLASS DISTRIBUTION ANALYSIS:")
    for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        counts = df['sarcasm_label'].value_counts().sort_index()
        total = len(df)
        print(f"{name}: Non-sarcastic: {counts[0]} ({counts[0] / total:.1%}), "
              f"Sarcastic: {counts[1]} ({counts[1] / total:.1%})")

    # Define feature columns
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

    # Handle NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_valid = np.nan_to_num(X_valid, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale features
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train), (X_valid_scaled, y_valid), (X_test_scaled, y_test), scaler, feature_cols


def train_model_with_imbalance_handling(model, train_loader, valid_loader, num_epochs=30, learning_rate=0.001,
                                        class_weights=None):
    """🚀 Enhanced training with class imbalance handling"""

    # Multiple loss functions to try
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"📊 Using weighted CrossEntropyLoss with weights: {class_weights}")
    else:
        criterion = FocalLoss(alpha=1, gamma=2)
        print("📊 Using Focal Loss for imbalance handling")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'valid_loss': [], 'valid_acc': [], 'valid_f1': []
    }

    best_valid_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    patience_limit = 8  # Increased patience

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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())

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

        # Calculate metrics with focus on sarcasm detection
        train_acc = accuracy_score(train_labels, train_preds)
        valid_acc = accuracy_score(valid_labels, valid_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')

        # Class-specific metrics for monitoring
        valid_f1_sarcastic = f1_score(valid_labels, valid_preds, labels=[1], average='macro')

        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['valid_loss'].append(valid_loss / len(valid_loader))
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['train_f1'].append(train_f1)
        history['valid_f1'].append(valid_f1)

        scheduler.step(valid_loss / len(valid_loader))

        # Early stopping based on F1 score
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start

        # Enhanced logging
        print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Valid Loss: {valid_loss / len(valid_loader):.4f} | "
              f"Valid F1: {valid_f1:.4f} | "
              f"Sarcasm F1: {valid_f1_sarcastic:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Best model loaded (F1: {best_valid_f1:.4f})")

    return model, history


def evaluate_model_detailed(model, test_loader, class_names=['Non-Sarcastic', 'Sarcastic']):
    """🎯 Detailed evaluation with class-specific metrics"""
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

    # Comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Class-specific F1 scores
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    print(f"\n📊 DETAILED TEST RESULTS:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"Non-Sarcastic F1: {f1_per_class[0]:.4f}")
    print(f"Sarcastic F1: {f1_per_class[1]:.4f}")

    # Detailed classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\n📋 Classification Report:\n{report}")

    # Confusion matrix analysis
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n🔍 Confusion Matrix Analysis:")
    print(f"True Negatives (correct non-sarcastic): {cm[0, 0]}")
    print(f"False Positives (incorrectly flagged as sarcastic): {cm[0, 1]}")
    print(f"False Negatives (missed sarcastic): {cm[1, 0]}")
    print(f"True Positives (correct sarcastic): {cm[1, 1]}")

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }


def main():
    """🎯 Main training pipeline with imbalance handling"""
    print("🚀 Starting IMBALANCE-AWARE Sarcasm Detection Training")
    print(f"Device: {device}")

    # Configuration - Adjusted for imbalanced data
    DATA_FOLDER = "sarcasm_emotion_features_split"
    BATCH_SIZE = 32  # Smaller batch size for better gradient updates
    LEARNING_RATE = 0.0005  # Lower learning rate for stability
    NUM_EPOCHS = 40  # More epochs for imbalanced learning

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"sarcasm_imbalance_fixed_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), scaler, feature_cols = load_and_prepare_data(DATA_FOLDER)

    # Calculate class weights for imbalance handling
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"\n⚖️ Computed class weights: {class_weights}")
    print(f"Non-sarcastic weight: {class_weights[0]:.3f}")
    print(f"Sarcastic weight: {class_weights[1]:.3f}")

    # Create datasets
    train_dataset = SarcasmFeatureDataset(X_train, y_train)
    valid_dataset = SarcasmFeatureDataset(X_valid, y_valid)
    test_dataset = SarcasmFeatureDataset(X_test, y_test)

    # Create balanced sampler for training
    train_sampler = create_balanced_sampler(y_train)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    # Initialize model with higher dropout for imbalanced data
    model = ImbalanceAwareSarcasmDetector(
        input_size=X_train.shape[1],
        hidden_sizes=[256, 128, 64],  # Slightly smaller to prevent overfitting
        dropout_rate=0.6  # Higher dropout
    ).to(device)

    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model with imbalance handling
    print("🏋️ Starting imbalance-aware training...")
    trained_model, history = train_model_with_imbalance_handling(
        model, train_loader, valid_loader,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        class_weights=class_weights
    )

    # Detailed evaluation
    print("🧪 Running detailed evaluation...")
    test_results = evaluate_model_detailed(trained_model, test_loader)

    # Save everything
    model_save_path = os.path.join(results_dir, "imbalance_aware_sarcasm_detector.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'feature_cols': feature_cols,
        'class_weights': class_weights,
        'model_config': {
            'input_size': X_train.shape[1],
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.6
        }
    }, model_save_path)

    import pickle
    with open(os.path.join(results_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"💾 Model saved to: {model_save_path}")
    print(f"\n✅ Training completed! Results saved in: {results_dir}")

    # Check if sarcasm detection improved
    sarcasm_f1 = test_results['f1_per_class'][1]
    if sarcasm_f1 > 0.3:
        print(f"🎉 SUCCESS! Sarcasm F1-Score: {sarcasm_f1:.4f} - Model can detect sarcasm!")
    else:
        print(f"⚠️ Sarcasm F1-Score: {sarcasm_f1:.4f} - Still needs improvement")

    return trained_model, scaler, feature_cols, test_results


if __name__ == "__main__":
    model, scaler, feature_cols, results = main()