"""
Tests for prediction module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from src.data.preprocessing import DataPreprocessor
from src.models.model_training import ModelTrainer
from src.prediction.prediction import RentalPricePredictor


class TestRentalPricePredictor:
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor for testing."""
        np.random.seed(42)
        n_samples = 200
        
        # Create synthetic data
        df = pd.DataFrame({
            "price": 200000 + np.random.randn(n_samples) * 50000,
            "lotSize": np.random.uniform(0.1, 2.0, n_samples),
            "age": np.random.randint(0, 50, n_samples),
            "landValue": np.random.uniform(20000, 100000, n_samples),
            "livingArea": np.random.uniform(800, 3000, n_samples),
            "pctCollege": np.random.randint(20, 80, n_samples),
            "bedrooms": np.random.randint(2, 6, n_samples),
            "fireplaces": np.random.randint(0, 3, n_samples),
            "bathrooms": np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], n_samples),
            "rooms": np.random.randint(4, 12, n_samples),
            "heating": np.random.choice(["hot air", "hot water/steam", "electric"], n_samples),
            "fuel": np.random.choice(["gas", "oil", "electric"], n_samples),
            "sewer": np.random.choice(["septic", "public/commercial"], n_samples),
            "waterfront": np.random.choice(["Yes", "No"], n_samples, p=[0.1, 0.9]),
            "newConstruction": np.random.choice(["Yes", "No"], n_samples, p=[0.05, 0.95]),
            "centralAir": np.random.choice(["Yes", "No"], n_samples, p=[0.4, 0.6])
        })
        
        # Train preprocessor and model
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        X = preprocessor.prepare_features(df_clean, fit=True)
        y = df_clean["price"]
        
        trainer = ModelTrainer(model_type="xgboost")
        trainer.train(X, y)
        
        # Save to temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            preprocessor_path = Path(tmpdir) / "preprocessor.pkl"
            
            trainer.save_model(model_path)
            preprocessor.save_preprocessor(preprocessor_path)
            
            predictor = RentalPricePredictor(model_path, preprocessor_path)
            
            return predictor, df
    
    def test_initialization(self, trained_predictor):
        """Test predictor initialization."""
        predictor, _ = trained_predictor
        assert predictor.is_fitted
        assert predictor.model_trainer.model is not None
        assert predictor.preprocessor.preprocessing_pipeline is not None
    
    def test_predict_single(self, trained_predictor):
        """Test single prediction."""
        predictor, df = trained_predictor
        
        features = {
            "lotSize": 0.5,
            "age": 15,
            "landValue": 50000,
            "livingArea": 1500,
            "pctCollege": 50,
            "bedrooms": 3,
            "fireplaces": 1,
            "bathrooms": 2.0,
            "rooms": 6,
            "heating": "hot air",
            "fuel": "gas",
            "sewer": "septic",
            "waterfront": "No",
            "newConstruction": "No",
            "centralAir": "No"
        }
        
        result = predictor.predict_single(features)
        
        assert "predicted_price" in result
        assert "confidence_interval" in result
        assert result["predicted_price"] > 0
    
    def test_predict_batch(self, trained_predictor):
        """Test batch prediction."""
        predictor, df = trained_predictor
        
        # Use first 5 rows for prediction
        test_df = df.drop("price", axis=1).head(5)
        
        results = predictor.predict_batch(test_df)
        
        assert len(results) == 5
        assert "predicted_price" in results.columns
        assert results["predicted_price"].notna().all()
    
    def test_analyze_scenario(self, trained_predictor):
        """Test scenario analysis."""
        predictor, _ = trained_predictor
        
        base_features = {
            "lotSize": 0.5, "age": 15, "landValue": 50000,
            "livingArea": 1500, "pctCollege": 50,
            "bedrooms": 3, "bathrooms": 2.0, "rooms": 6,
            "fireplaces": 1, "heating": "hot air", "fuel": "gas",
            "sewer": "septic", "waterfront": "No",
            "newConstruction": "No", "centralAir": "No"
        }
        
        scenarios = [
            {
                "name": "With AC",
                "description": "Add central air",
                "modifications": {"centralAir": "Yes"}
            }
        ]
        
        results = predictor.analyze_scenario(base_features, scenarios)
        
        assert len(results) == 2  # Base + 1 scenario
        assert "predicted_price" in results.columns
        assert "price_change" in results.columns
    
    def test_get_feature_importance(self, trained_predictor):
        """Test feature importance retrieval."""
        predictor, _ = trained_predictor
        
        importance_df = predictor.get_feature_importance()
        
        assert importance_df is not None
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
    
    def test_get_model_metrics(self, trained_predictor):
        """Test model metrics retrieval."""
        predictor, _ = trained_predictor
        
        metrics = predictor.get_model_metrics()
        
        assert "r2" in metrics or len(metrics) > 0