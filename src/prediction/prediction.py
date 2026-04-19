"""
Prediction module for rental price forecasting.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from src.config.settings import config
from src.data.preprocessing import DataPreprocessor
from src.models.model_training import ModelTrainer
from src.utils.helpers import format_currency, calculate_metrics


class RentalPricePredictor:
    """Main prediction interface for rental prices."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to fitted preprocessor
        """
        self.model_trainer = ModelTrainer()
        self.preprocessor = DataPreprocessor()
        self.is_fitted = False
        
        if model_path and model_path.exists():
            try:
                self.model_trainer.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        
        if preprocessor_path and preprocessor_path.exists():
            try:
                self.preprocessor = DataPreprocessor.load_preprocessor(preprocessor_path)
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
            except Exception as e:
                logger.error(f"Failed to load preprocessor: {e}")
                raise
        
        # Check if both model and preprocessor are loaded
        if self.model_trainer.model is not None and self.preprocessor.preprocessing_pipeline is not None:
            self.is_fitted = True
            logger.info("Predictor fully initialized and ready")
        else:
            logger.warning("Predictor not fully initialized. Model or preprocessor missing.")
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to match training data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features added
        """
        df_feat = df.copy()
        
        # Room density (rooms per living area)
        if "rooms" in df_feat.columns and "livingArea" in df_feat.columns:
            df_feat["room_density"] = df_feat["rooms"] / df_feat["livingArea"].replace(0, np.nan)
            # Replace infinite values
            df_feat["room_density"] = df_feat["room_density"].replace([np.inf, -np.inf], 0)
        
        # Lot size per bedroom
        if "lotSize" in df_feat.columns and "bedrooms" in df_feat.columns:
            df_feat["lot_per_bedroom"] = df_feat["lotSize"] / df_feat["bedrooms"].replace(0, np.nan)
            # Replace infinite values
            df_feat["lot_per_bedroom"] = df_feat["lot_per_bedroom"].replace([np.inf, -np.inf], 0)
        
        # Age category
        if "age" in df_feat.columns:
            try:
                df_feat["age_category"] = pd.cut(
                    df_feat["age"],
                    bins=[-1, 5, 15, 30, 50, 100, 1000],
                    labels=["New", "Recent", "Moderate", "Old", "Very Old", "Historic"]
                )
            except Exception as e:
                logger.debug(f"Could not create age_category: {e}")
                df_feat["age_category"] = "Moderate"
        
        # Total bathrooms
        if "bathrooms" in df_feat.columns:
            df_feat["total_baths"] = df_feat["bathrooms"]
        
        # Living area category
        if "livingArea" in df_feat.columns:
            try:
                df_feat["living_area_category"] = pd.cut(
                    df_feat["livingArea"],
                    bins=[-1, 1000, 1500, 2000, 2500, 100000],
                    labels=["Small", "Medium-Small", "Medium", "Medium-Large", "Large"]
                )
            except Exception as e:
                logger.debug(f"Could not create living_area_category: {e}")
                df_feat["living_area_category"] = "Medium"
        
        return df_feat
    
    def predict_single(
        self,
        features: Dict[str, Union[float, int, str]]
    ) -> Dict[str, Union[float, Dict]]:
        """
        Predict rental price for a single property.
        
        Args:
            features: Dictionary with property features
            
        Returns:
            Dictionary with prediction and metadata
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Load or train a model first.")
        
        logger.info("Making single prediction")
        logger.debug(f"Input features: {features}")
        
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([features])
            
            # Add derived features
            input_df = self._add_derived_features(input_df)
            
            # Preprocess features
            X_processed = self.preprocessor.prepare_features(input_df, fit=False)
            
            # Make prediction
            prediction = float(self.model_trainer.predict(X_processed)[0])
            
            # Calculate prediction interval
            if "std_residual" in self.model_trainer.metrics:
                std_residual = self.model_trainer.metrics["std_residual"]
                lower_bound = prediction - 1.96 * std_residual
                upper_bound = prediction + 1.96 * std_residual
            elif "rmse" in self.model_trainer.metrics:
                # Use RMSE as approximation
                rmse = self.model_trainer.metrics["rmse"]
                lower_bound = prediction - 1.96 * rmse
                upper_bound = prediction + 1.96 * rmse
            else:
                # Default to 15% interval
                lower_bound = prediction * 0.85
                upper_bound = prediction * 1.15
            
            # Ensure lower bound is not negative
            lower_bound = max(0, lower_bound)
            
            result = {
                "predicted_price": prediction,
                "predicted_price_formatted": format_currency(prediction),
                "confidence_interval": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                    "lower_formatted": format_currency(lower_bound),
                    "upper_formatted": format_currency(upper_bound)
                },
                "input_features": features,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Prediction successful: {result['predicted_price_formatted']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Predict rental prices for multiple properties.
        
        Args:
            data: DataFrame or list of dictionaries with property features
            return_confidence: Whether to return confidence intervals
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Load or train a model first.")
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            input_df = data.copy()
        
        logger.info(f"Making batch prediction for {len(input_df)} properties")
        
        try:
            # Store original data for results
            original_df = input_df.copy()
            
            # Add derived features
            input_df = self._add_derived_features(input_df)
            
            # Preprocess features
            X_processed = self.preprocessor.prepare_features(input_df, fit=False)
            
            # Make predictions
            predictions = self.model_trainer.predict(X_processed)
            
            # Create results DataFrame
            results = original_df.copy()
            results["predicted_price"] = predictions
            
            if return_confidence:
                if "std_residual" in self.model_trainer.metrics:
                    std_residual = self.model_trainer.metrics["std_residual"]
                    results["ci_lower"] = np.maximum(0, predictions - 1.96 * std_residual)
                    results["ci_upper"] = predictions + 1.96 * std_residual
                elif "rmse" in self.model_trainer.metrics:
                    rmse = self.model_trainer.metrics["rmse"]
                    results["ci_lower"] = np.maximum(0, predictions - 1.96 * rmse)
                    results["ci_upper"] = predictions + 1.96 * rmse
            
            logger.info(f"Batch prediction completed. Mean price: {format_currency(predictions.mean())}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def analyze_scenario(
        self,
        base_features: Dict[str, Union[float, int, str]],
        scenarios: List[Dict[str, Union[float, int, str, Dict]]]
    ) -> pd.DataFrame:
        """
        Analyze different scenarios by modifying base features.
        
        Args:
            base_features: Base property features
            scenarios: List of scenario configurations. Each scenario should have:
                - name: str
                - description: str
                - modifications: dict of feature changes
            
        Returns:
            DataFrame with scenario analysis
        """
        logger.info(f"Analyzing {len(scenarios)} scenarios")
        
        results = []
        
        # Add base scenario
        try:
            base_prediction = self.predict_single(base_features)
            results.append({
                "scenario": "Base",
                "description": "Configuración actual de la propiedad",
                "predicted_price": base_prediction["predicted_price"],
                "price_change": 0.0,
                "price_change_pct": 0.0
            })
            logger.debug(f"Base scenario predicted: {base_prediction['predicted_price_formatted']}")
        except Exception as e:
            logger.error(f"Error predicting base scenario: {e}")
            raise
        
        # Analyze each scenario
        for i, scenario in enumerate(scenarios):
            try:
                # Get scenario configuration
                scenario_name = scenario.get("name", f"Escenario {i+1}")
                scenario_desc = scenario.get("description", "")
                modifications = scenario.get("modifications", {})
                
                # Merge base features with scenario modifications
                scenario_features = base_features.copy()
                scenario_features.update(modifications)
                
                # Make prediction
                prediction = self.predict_single(scenario_features)
                
                # Calculate changes
                base_price = base_prediction["predicted_price"]
                new_price = prediction["predicted_price"]
                price_change = new_price - base_price
                price_change_pct = (price_change / base_price * 100) if base_price > 0 else 0
                
                results.append({
                    "scenario": scenario_name,
                    "description": scenario_desc,
                    "predicted_price": new_price,
                    "price_change": price_change,
                    "price_change_pct": price_change_pct
                })
                
                logger.debug(f"Scenario '{scenario_name}' predicted: {prediction['predicted_price_formatted']} (change: {price_change_pct:+.1f}%)")
                
            except Exception as e:
                logger.error(f"Error analyzing scenario '{scenario.get('name', f'Escenario {i+1}')}': {e}")
                # Add error entry
                results.append({
                    "scenario": scenario.get("name", f"Escenario {i+1}"),
                    "description": scenario.get("description", ""),
                    "predicted_price": np.nan,
                    "price_change": np.nan,
                    "price_change_pct": np.nan,
                    "error": str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Scenario analysis completed. {len(results_df)} scenarios analyzed.")
        
        return results_df
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the model."""
        if self.model_trainer.feature_importance is not None:
            return self.model_trainer.feature_importance.copy()
        return None
    
    def get_model_metrics(self) -> Dict:
        """Get model performance metrics."""
        return self.model_trainer.metrics.copy()
    
    def recursive_forecast(
        self,
        base_features: Dict[str, Union[float, int, str]],
        periods: int = 12,
        growth_rate: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate recursive multi-step forecasts.
        
        Args:
            base_features: Base property features
            periods: Number of periods to forecast
            growth_rate: Expected annual growth rate (e.g., 0.02 for 2%)
            
        Returns:
            DataFrame with recursive forecasts
        """
        logger.info(f"Generating recursive forecast for {periods} periods")
        
        try:
            # Monthly growth rate
            monthly_rate = (1 + growth_rate) ** (1/12) - 1
            
            # Get base prediction
            base_prediction = self.predict_single(base_features)
            current_price = base_prediction["predicted_price"]
            
            # Get confidence interval width
            ci_lower_base = base_prediction["confidence_interval"]["lower"]
            ci_upper_base = base_prediction["confidence_interval"]["upper"]
            base_width = (ci_upper_base - ci_lower_base) / 2
            
            forecasts = []
            
            for period in range(periods + 1):
                if period == 0:
                    price = current_price
                    ci_lower = ci_lower_base
                    ci_upper = ci_upper_base
                else:
                    # Apply growth
                    price = current_price * (1 + monthly_rate) ** period
                    
                    # Expand confidence interval over time (uncertainty increases with sqrt(time))
                    uncertainty_factor = np.sqrt(period + 1)
                    ci_lower = max(0, price - base_width * uncertainty_factor)
                    ci_upper = price + base_width * uncertainty_factor
                
                forecasts.append({
                    "period": period,
                    "forecasted_price": price,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper
                })
            
            forecasts_df = pd.DataFrame(forecasts)
            
            final_price = forecasts_df.iloc[-1]['forecasted_price']
            logger.info(f"Recursive forecast completed. Final price: {format_currency(final_price)}")
            
            return forecasts_df
            
        except Exception as e:
            logger.error(f"Recursive forecast failed: {e}")
            raise
    
    def sensitivity_analysis(
        self,
        base_features: Dict[str, Union[float, int, str]],
        feature: str,
        values: List[Union[float, int, str]]
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a single feature.
        
        Args:
            base_features: Base property features
            feature: Feature to vary
            values: List of values to test
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        logger.info(f"Performing sensitivity analysis on '{feature}' with {len(values)} values")
        
        results = []
        
        for value in values:
            try:
                test_features = base_features.copy()
                test_features[feature] = value
                
                prediction = self.predict_single(test_features)
                
                results.append({
                    feature: value,
                    "predicted_price": prediction["predicted_price"],
                    "ci_lower": prediction["confidence_interval"]["lower"],
                    "ci_upper": prediction["confidence_interval"]["upper"]
                })
                
            except Exception as e:
                logger.error(f"Error analyzing value '{value}' for feature '{feature}': {e}")
        
        if not results:
            raise ValueError(f"No valid results for sensitivity analysis on '{feature}'")
        
        results_df = pd.DataFrame(results)
        
        # Calculate marginal effects for numeric features
        numeric_features = self.preprocessor.numeric_features if hasattr(self.preprocessor, 'numeric_features') else []
        if feature in numeric_features and len(results_df) > 1:
            # Sort by feature value to calculate proper differences
            results_df = results_df.sort_values(feature)
            results_df["marginal_effect"] = results_df["predicted_price"].diff()
            # Calculate per-unit change
            feature_diff = results_df[feature].diff()
            results_df["marginal_effect_per_unit"] = results_df["marginal_effect"] / feature_diff.replace(0, np.nan)
        
        logger.info(f"Sensitivity analysis completed. {len(results_df)} results.")
        
        return results_df
    
    def compare_properties(
        self,
        properties: List[Dict[str, Union[float, int, str]]],
        names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple properties.
        
        Args:
            properties: List of property feature dictionaries
            names: Optional list of property names
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(properties)} properties")
        
        if names is None:
            names = [f"Propiedad {i+1}" for i in range(len(properties))]
        
        results = []
        
        for i, (prop_features, name) in enumerate(zip(properties, names)):
            try:
                prediction = self.predict_single(prop_features)
                
                result = {
                    "name": name,
                    "predicted_price": prediction["predicted_price"],
                    "predicted_price_formatted": prediction["predicted_price_formatted"],
                    "livingArea": prop_features.get("livingArea", np.nan),
                    "bedrooms": prop_features.get("bedrooms", np.nan),
                    "bathrooms": prop_features.get("bathrooms", np.nan),
                    "age": prop_features.get("age", np.nan),
                    "waterfront": prop_features.get("waterfront", "No"),
                    "centralAir": prop_features.get("centralAir", "No")
                }
                
                # Calculate price per sqft
                if "livingArea" in prop_features and prop_features["livingArea"] > 0:
                    result["price_per_sqft"] = prediction["predicted_price"] / prop_features["livingArea"]
                else:
                    result["price_per_sqft"] = np.nan
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting property '{name}': {e}")
                results.append({
                    "name": name,
                    "predicted_price": np.nan,
                    "predicted_price_formatted": "Error",
                    "error": str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        # Calculate rankings
        if "predicted_price" in results_df.columns:
            results_df["rank"] = results_df["predicted_price"].rank(ascending=False)
        
        logger.info(f"Property comparison completed")
        
        return results_df
    
    def get_price_breakdown(
        self,
        features: Dict[str, Union[float, int, str]]
    ) -> Dict[str, float]:
        """
        Get a breakdown of how each feature contributes to the price.
        
        Args:
            features: Property features
            
        Returns:
            Dictionary with feature contributions
        """
        logger.info("Calculating price breakdown")
        
        # Get base prediction
        base_prediction = self.predict_single(features)
        base_price = base_prediction["predicted_price"]
        
        # Get feature importance
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            logger.warning("No feature importance available")
            return {"base_price": base_price}
        
        breakdown = {"base_price": base_price}
        
        # Calculate approximate contribution of top features
        top_features = importance_df.head(10)
        total_importance = top_features["importance"].sum()
        
        for _, row in top_features.iterrows():
            feature_name = row["feature"]
            importance_pct = row["importance"] / total_importance
            contribution = base_price * importance_pct
            breakdown[f"contribution_{feature_name}"] = contribution
        
        logger.info("Price breakdown calculated")
        
        return breakdown
    
    def validate_input_features(
        self,
        features: Dict[str, Union[float, int, str]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that input features contain all required fields.
        
        Args:
            features: Input feature dictionary
            
        Returns:
            Tuple of (is_valid, list of missing or invalid fields)
        """
        required_fields = [
            "lotSize", "age", "landValue", "livingArea", "pctCollege",
            "bedrooms", "fireplaces", "bathrooms", "rooms",
            "heating", "fuel", "sewer", "waterfront", "newConstruction", "centralAir"
        ]
        
        issues = []
        
        # Check missing fields
        for field in required_fields:
            if field not in features:
                issues.append(f"Missing field: {field}")
        
        # Validate numeric fields
        numeric_fields = ["lotSize", "age", "landValue", "livingArea", "pctCollege", 
                         "bedrooms", "fireplaces", "bathrooms", "rooms"]
        
        for field in numeric_fields:
            if field in features:
                try:
                    value = float(features[field])
                    if value < 0:
                        issues.append(f"Negative value for {field}: {value}")
                except (ValueError, TypeError):
                    issues.append(f"Invalid numeric value for {field}: {features[field]}")
        
        # Validate categorical fields
        categorical_fields = {
            "heating": ["hot air", "hot water/steam", "electric"],
            "fuel": ["gas", "oil", "electric"],
            "sewer": ["septic", "public/commercial", "none"],
            "waterfront": ["Yes", "No"],
            "newConstruction": ["Yes", "No"],
            "centralAir": ["Yes", "No"]
        }
        
        for field, valid_values in categorical_fields.items():
            if field in features:
                if features[field] not in valid_values:
                    issues.append(f"Invalid value for {field}: '{features[field]}'. Valid: {valid_values}")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Input validation failed: {issues}")
        
        return is_valid, issues


# Convenience functions for quick predictions
def quick_predict(features: Dict) -> Dict:
    """
    Quick prediction using default model paths.
    
    Args:
        features: Property features dictionary
        
    Returns:
        Prediction result dictionary
    """
    from src.config.settings import MODELS_DIR
    
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    
    predictor = RentalPricePredictor(model_path, preprocessor_path)
    return predictor.predict_single(features)


def quick_batch_predict(data: pd.DataFrame) -> pd.DataFrame:
    """
    Quick batch prediction using default model paths.
    
    Args:
        data: DataFrame with property features
        
    Returns:
        DataFrame with predictions
    """
    from src.config.settings import MODELS_DIR
    
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    
    predictor = RentalPricePredictor(model_path, preprocessor_path)
    return predictor.predict_batch(data)