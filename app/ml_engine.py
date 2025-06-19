"""
Machine Learning engine for financial predictions.
Implements Strategy pattern for different ML algorithms.
"""
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from loguru import logger
from config import get_settings, MODELS_DIR

settings = get_settings()


@dataclass(frozen=True)
class ModelMetrics:
    """Immutable model performance metrics."""
    mae: float
    r2_score: float
    training_samples: int
    feature_count: int
    trained_at: datetime


class MLStrategy(ABC):
    """Abstract base class for ML strategies."""
    
    @abstractmethod
    async def train(self, data: pd.DataFrame) -> ModelMetrics:
        """Train the model with given data."""
        pass
    
    @abstractmethod
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: Path) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: Path) -> None:
        """Load model from disk."""
        pass


class AnomalyDetectionStrategy(MLStrategy):
    """Anomaly detection using Isolation Forest."""
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=settings.anomaly_contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        features = data.copy()
        
        # Price-based features
        features['price_change'] = features['close_price'].pct_change()
        features['price_volatility'] = features['close_price'].rolling(5).std()
        features['volume_change'] = features['volume'].pct_change()
        
        # Technical indicators
        features['sma_5'] = features['close_price'].rolling(5).mean()
        features['sma_20'] = features['close_price'].rolling(20).mean()
        features['price_to_sma'] = features['close_price'] / features['sma_20']
        
        # Remove non-numeric and missing values
        numeric_features = features.select_dtypes(include=[np.number]).dropna()
        self.feature_columns = numeric_features.columns.tolist()
        
        return numeric_features
    
    async def train(self, data: pd.DataFrame) -> ModelMetrics:
        """Train anomaly detection model."""
        features = self._prepare_features(data)
        
        if len(features) < 10:
            raise ValueError("Insufficient data for training")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(scaled_features)
        
        # Calculate metrics
        anomaly_scores = self.model.decision_function(scaled_features)
        predictions = self.model.predict(scaled_features)
        
        return ModelMetrics(
            mae=np.mean(np.abs(anomaly_scores)),
            r2_score=len(predictions[predictions == -1]) / len(predictions),
            training_samples=len(features),
            feature_count=len(self.feature_columns),
            trained_at=datetime.now()
        )
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the data."""
        features = self._prepare_features(data)
        
        if not self.feature_columns:
            raise ValueError("Model not trained")
        
        # Ensure same features as training
        features = features[self.feature_columns]
        scaled_features = self.scaler.transform(features)
        
        # Generate predictions
        anomaly_scores = self.model.decision_function(scaled_features)
        predictions = self.model.predict(scaled_features)
        
        anomalies = []
        for idx, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
            if pred == -1:  # Anomaly detected
                anomalies.append({
                    "index": idx,
                    "date": data.index[idx] if hasattr(data.index, 'date') else idx,
                    "anomaly_score": float(score),
                    "severity": "high" if score < -0.5 else "medium"
                })
        
        return {
            "anomaly_score": float(np.mean(np.abs(anomaly_scores))),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(predictions),
            "detected_anomalies": anomalies
        }
    
    def save_model(self, filepath: Path) -> None:
        """Save model and scaler."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: Path) -> None:
        """Load model and scaler."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_columns = model_data["feature_columns"]


class VolatilityPredictionStrategy(MLStrategy):
    """Volatility prediction using Random Forest."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for volatility prediction."""
        features = data.copy()
        
        # Calculate volatility target (rolling standard deviation)
        volatility_window = settings.volatility_window
        target = features['close_price'].rolling(volatility_window).std()
        
        # Price features
        features['returns'] = features['close_price'].pct_change()
        features['log_returns'] = np.log(features['close_price'] / features['close_price'].shift(1))
        features['price_range'] = (features['high_price'] - features['low_price']) / features['close_price']
        
        # Lagged volatility features
        for lag in [1, 5, 10]:
            features[f'volatility_lag_{lag}'] = target.shift(lag)
        
        # Volume features
        features['volume_sma'] = features['volume'].rolling(10).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']
        
        # Select numeric features and remove NaN
        numeric_features = features.select_dtypes(include=[np.number])
        valid_data = pd.concat([numeric_features, target], axis=1).dropna()
        
        if len(valid_data) == 0:
            raise ValueError("No valid data after feature preparation")
        
        feature_data = valid_data.iloc[:, :-1]  # All columns except target
        target_data = valid_data.iloc[:, -1]    # Last column (target)
        
        self.feature_columns = feature_data.columns.tolist()
        return feature_data, target_data
    
    async def train(self, data: pd.DataFrame) -> ModelMetrics:
        """Train volatility prediction model."""
        features, target = self._prepare_features(data)
        
        if len(features) < 50:
            raise ValueError("Insufficient data for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return ModelMetrics(
            mae=mae,
            r2_score=r2,
            training_samples=len(X_train),
            feature_count=len(self.feature_columns),
            trained_at=datetime.now()
        )
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Predict volatility."""
        features, _ = self._prepare_features(data)
        
        if not self.feature_columns:
            raise ValueError("Model not trained")
        
        # Use latest data for prediction
        latest_features = features[self.feature_columns].tail(1)
        scaled_features = self.scaler.transform(latest_features)
        
        # Generate prediction
        volatility_pred = self.model.predict(scaled_features)[0]
        
        # Calculate confidence interval (simple approach)
        feature_importance = self.model.feature_importances_
        confidence = float(np.mean(feature_importance))
        
        return {
            "predicted_volatility": float(volatility_pred),
            "confidence_score": confidence,
            "volatility_percentage": float(volatility_pred * 100),
            "confidence_interval": {
                "lower": float(volatility_pred * 0.8),
                "upper": float(volatility_pred * 1.2)
            }
        }
    
    def save_model(self, filepath: Path) -> None:
        """Save model and scaler."""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: Path) -> None:
        """Load model and scaler."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_columns = model_data["feature_columns"]


class MLEngine:
    """Main ML engine implementing Strategy pattern."""
    
    def __init__(self):
        self.strategies = {
            "anomaly": AnomalyDetectionStrategy(),
            "volatility": VolatilityPredictionStrategy()
        }
        self.model_metrics = {}
    
    async def train_model(self, model_type: str, data: pd.DataFrame) -> ModelMetrics:
        """Train specific model type."""
        if model_type not in self.strategies:
            raise ValueError(f"Unknown model type: {model_type}")
        
        strategy = self.strategies[model_type]
        metrics = await strategy.train(data)
        
        # Save model
        model_path = MODELS_DIR / f"{model_type}_model.pkl"
        strategy.save_model(model_path)
        
        # Store metrics
        self.model_metrics[model_type] = metrics
        logger.info(f"Trained {model_type} model: MAE={metrics.mae:.4f}, R2={metrics.r2_score:.4f}")
        
        return metrics
    
    async def predict(self, model_type: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions using specified model."""
        if model_type not in self.strategies:
            raise ValueError(f"Unknown model type: {model_type}")
        
        strategy = self.strategies[model_type]
        
        # Load model if not trained
        model_path = MODELS_DIR / f"{model_type}_model.pkl"
        if model_path.exists() and not hasattr(strategy.model, 'estimators_'):
            strategy.load_model(model_path)
        
        return await strategy.predict(data)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {}
        for model_type in self.strategies:
            model_path = MODELS_DIR / f"{model_type}_model.pkl"
            metrics = self.model_metrics.get(model_type)
            
            status[model_type] = {
                "model_exists": model_path.exists(),
                "last_trained": metrics.trained_at.isoformat() if metrics else None,
                "performance": {
                    "mae": metrics.mae if metrics else None,
                    "r2_score": metrics.r2_score if metrics else None,
                    "training_samples": metrics.training_samples if metrics else None
                } if metrics else None
            }
        
        return status
