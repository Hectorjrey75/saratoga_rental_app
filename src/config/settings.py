"""
Configuration settings for the rental price prediction system.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths - CORREGIDO
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directorio creado/verificado: {directory}")


@dataclass
class DataConfig:
    """Data configuration."""
    target_column: str = "price"
    numeric_features: list = field(default_factory=lambda: [
        "lotSize", "age", "landValue", "livingArea", "pctCollege",
        "bedrooms", "fireplaces", "bathrooms", "rooms"
    ])
    categorical_features: list = field(default_factory=lambda: [
        "heating", "fuel", "sewer", "waterfront", "newConstruction", "centralAir"
    ])
    test_size: float = 0.2
    random_state: int = 42
    validation_size: float = 0.2


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "xgboost"
    cv_folds: int = 5
    scoring: str = "neg_mean_squared_error"
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "xgboost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    })


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    rotation: str = "10 MB"
    retention: str = "10 days"
    log_file: Path = LOGS_DIR / "app.log"


@dataclass
class AppConfig:
    """Application configuration."""
    title: str = "🏠 Sistema Profesional de Predicción de Precios de Alquiler"
    description: str = "Predicción de precios de viviendas usando Machine Learning"
    version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8501))


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    app: AppConfig = field(default_factory=AppConfig)
    
    # Rutas como atributos de clase
    PROJECT_ROOT : Path = PROJECT_ROOT
    DATA_DIR : Path = DATA_DIR
    RAW_DATA_DIR : Path = RAW_DATA_DIR
    PROCESSED_DATA_DIR : Path = PROCESSED_DATA_DIR
    MODELS_DIR : Path = MODELS_DIR
    LOGS_DIR : Path = LOGS_DIR



# Global configuration instance
config = Config()