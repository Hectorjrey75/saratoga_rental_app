
## 16. tests/test_preprocessing.py

```python
"""
Tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            "price": [200000, 300000, 250000, 350000, 280000],
            "lotSize": [0.5, 0.8, 0.3, 1.0, 0.6],
            "age": [15, 8, 25, 3, 12],
            "landValue": [50000, 75000, 40000, 90000, 60000],
            "livingArea": [1500, 2200, 1200, 2800, 1800],
            "pctCollege": [50, 65, 40, 70, 55],
            "bedrooms": [3, 4, 2, 5, 3],
            "fireplaces": [1, 2, 0, 2, 1],
            "bathrooms": [2.0, 2.5, 1.5, 3.0, 2.0],
            "rooms": [6, 8, 5, 10, 7],
            "heating": ["hot air", "hot air", "electric", "hot water", "hot air"],
            "fuel": ["gas", "gas", "electric", "oil", "gas"],
            "sewer": ["septic", "public", "septic", "public", "septic"],
            "waterfront": ["No", "No", "Yes", "No", "No"],
            "newConstruction": ["No", "Yes", "No", "Yes", "No"],
            "centralAir": ["No", "Yes", "No", "Yes", "No"]
        })
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(scaler_type="standard")
        assert preprocessor.scaler_type == "standard"
        
        preprocessor = DataPreprocessor(scaler_type="minmax")
        assert preprocessor.scaler_type == "minmax"
        
        with pytest.raises(ValueError):
            DataPreprocessor(scaler_type="invalid")
    
    def test_clean_data(self, sample_data):
        """Test data cleaning."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_data)
        
        assert df_clean.shape[0] == sample_data.shape[0]
        assert "price_per_sqft" in df_clean.columns
        assert "age_category" in df_clean.columns
    
    def test_prepare_features_fit(self, sample_data):
        """Test feature preparation with fitting."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_data)
        X_processed = preprocessor.prepare_features(df_clean, fit=True)
        
        assert X_processed.shape[0] == sample_data.shape[0]
        assert preprocessor.preprocessing_pipeline is not None
        assert len(preprocessor.feature_names) > 0
    
    def test_prepare_features_transform(self, sample_data):
        """Test feature preparation without fitting."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_data)
        
        # Fit first
        preprocessor.prepare_features(df_clean, fit=True)
        
        # Then transform
        X_processed = preprocessor.prepare_features(df_clean, fit=False)
        assert X_processed.shape[0] == sample_data.shape[0]
    
    def test_split_data(self, sample_data):
        """Test data splitting."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_data)
        X = preprocessor.prepare_features(df_clean, fit=True)
        y = df_clean["price"]
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            X, y, test_size=0.2, val_size=0.2
        )
        
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)