"""
Utility helper functions.
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger


def save_pickle(obj: Any, path: Path) -> None:
    """Save object as pickle file."""
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save pickle: {e}")
        raise


def load_pickle(path: Path) -> Any:
    """Load object from pickle file."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load pickle: {e}")
        raise


def save_json(data: Dict, path: Path) -> None:
    """Save dictionary as JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise


def load_json(path: Path) -> Dict:
    """Load dictionary from JSON file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"JSON loaded from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        raise


def generate_run_id() -> str:
    """Generate unique run identifier."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error
    )
    
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100
    }
    
    return metrics


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def create_feature_importance_df(
    feature_names: List[str],
    importance_values: np.ndarray
) -> pd.DataFrame:
    """Create feature importance DataFrame."""
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    })
    df = df.sort_values("importance", ascending=False)
    df["cumulative_importance"] = df["importance"].cumsum()
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    return df