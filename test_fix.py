# test_fix.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import MODELS_DIR
from src.prediction.prediction import RentalPricePredictor

# Cargar predictor
predictor = RentalPricePredictor(
    model_path=MODELS_DIR / "best_model.pkl",
    preprocessor_path=MODELS_DIR / "preprocessor.pkl"
)

# Probar predicción
features = {
    "lotSize": 0.5, "age": 15, "landValue": 50000,
    "livingArea": 1500, "pctCollege": 50,
    "bedrooms": 3, "fireplaces": 1, "bathrooms": 2.0, "rooms": 6,
    "heating": "hot air", "fuel": "gas", "sewer": "septic",
    "waterfront": "No", "newConstruction": "No", "centralAir": "No"
}

result = predictor.predict_single(features)
print(f"✅ Predicción exitosa: {result['predicted_price_formatted']}")

# Probar análisis de escenarios
scenarios = [{
    "name": "Con AC",
    "description": "Añadir aire acondicionado",
    "modifications": {"centralAir": "Yes"}
}]

results = predictor.analyze_scenario(features, scenarios)
print(f"✅ Análisis de escenarios exitoso: {len(results)} escenarios")
print(results)