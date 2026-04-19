"""
Tests for model training module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor
from src.models.model_training import ModelTrainer


class TestModelTrainer:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        })
        
        y = pd.Series(
            100000 + 50000 * X["feature1"] + 
            20000 * X["feature2"] + np.random.randn(n_samples) * 10000
        )
        
        return X, y
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer(model_type="xgboost")
        assert trainer.model_type == "xgboost"
        
        trainer = ModelTrainer(model_type="random_forest")
        assert trainer.model_type == "random_forest"
        
        with pytest.raises(ValueError):
            ModelTrainer(model_type="invalid")
    
    def test_train_model(self, sample_data):
        """Test model training."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="xgboost")
        
        model = trainer.train(X, y)
        assert model is not None
        assert trainer.model is not None
        assert "train_metrics" in trainer.training_history
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="xgboost")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["r2"] > 0  # Should be better than mean prediction
    
    def test_predict(self, sample_data):
        """Test predictions."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="xgboost")
        trainer.train(X, y)
        
        predictions = trainer.predict(X)
        assert len(predictions) == len(X)
        assert predictions.dtype in [np.float32, np.float64]
    
    def test_feature_importance(self, sample_data):
        """Test feature importance calculation."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="xgboost")
        trainer.train(X, y)
        
        assert trainer.feature_importance is not None
        assert len(trainer.feature_importance) == X.shape[1]
        assert "importance_pct" in trainer.feature_importance.columns