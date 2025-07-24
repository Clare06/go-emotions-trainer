# sarcasm_detector.py
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def get_sarcasm_detector(model_path="sarcasm_training_results_20250725_035320/sarcasm_detector.pth",
                         scaler_path="sarcasm_training_results_20250725_035320/scaler.pkl"):
    """Load and return the trained sarcasm detection model and scaler"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    feature_cols = checkpoint['feature_cols']

    # Recreate model architecture
    from sarcasm_detector_trainer import OptimizedSarcasmDetector
    model = OptimizedSarcasmDetector(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler, feature_cols, device


def predict_sarcasm(features, model, scaler, device):
    """Predict sarcasm from features"""

    # Handle NaN values and scale
    features = np.nan_to_num(features, nan=0.0)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)

    return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]