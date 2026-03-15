"""
Model training and evaluation utilities.
"""

from typing import Any, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class BaseModel:
    """Base class for ML models."""
    
    def __init__(self):
        self.model = None
        self.history = {}
    
    def train(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
        """
        raise NotImplementedError("Subclasses must implement train() method")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
