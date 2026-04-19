"""
Model training module for rental price prediction.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import json
from loguru import logger

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    KFold
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Importar con manejo de errores para entornos sin estas librerías
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost no está instalado. Usa 'pip install xgboost'")

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM no está instalado. Usa 'pip install lightgbm'")

from src.config.settings import config
from src.utils.helpers import save_pickle, load_pickle, save_json, calculate_metrics


class ModelTrainer:
    """Model training and evaluation pipeline."""
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.training_history = {}
        self.metrics = {}
        
        logger.info(f"ModelTrainer initialized with {model_type}")
    
    def _get_model(self, params: Optional[Dict] = None) -> Any:
        """Get model instance with optional parameters."""
        
        # Filtrar parámetros según el tipo de modelo
        if params is None:
            params = {}
        else:
            params = params.copy()
        
        models = {}
        
        # XGBoost
        if XGB_AVAILABLE:
            # Parámetros válidos para XGBoost
            xgb_params = {
                k: v for k, v in params.items() 
                if k in ['n_estimators', 'max_depth', 'learning_rate', 
                        'subsample', 'colsample_bytree', 'colsample_bylevel',
                        'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']
            }
            models["xgboost"] = XGBRegressor(
                random_state=config.data.random_state,
                n_jobs=-1,
                **xgb_params
            )
        
        # LightGBM
        if LGBM_AVAILABLE:
            # Parámetros válidos para LightGBM
            lgbm_params = {
                k: v for k, v in params.items() 
                if k in ['n_estimators', 'max_depth', 'learning_rate',
                        'num_leaves', 'subsample', 'colsample_bytree',
                        'min_child_samples', 'reg_alpha', 'reg_lambda']
            }
            models["lightgbm"] = LGBMRegressor(
                random_state=config.data.random_state,
                n_jobs=-1,
                verbose=-1,
                **lgbm_params
            )
        
        # Random Forest - solo acepta ciertos parámetros
        rf_params = {
            k: v for k, v in params.items() 
            if k in ['n_estimators', 'max_depth', 'min_samples_split',
                    'min_samples_leaf', 'max_features', 'bootstrap']
        }
        models["random_forest"] = RandomForestRegressor(
            random_state=config.data.random_state,
            n_jobs=-1,
            **rf_params
        )
        
        # Gradient Boosting
        gb_params = {
            k: v for k, v in params.items() 
            if k in ['n_estimators', 'max_depth', 'learning_rate',
                    'subsample', 'min_samples_split', 'min_samples_leaf']
        }
        models["gradient_boosting"] = GradientBoostingRegressor(
            random_state=config.data.random_state,
            **gb_params
        )
        
        # Modelos lineales
        linear_params = {
            k: v for k, v in params.items() 
            if k in ['alpha', 'l1_ratio', 'fit_intercept']
        }
        models["ridge"] = Ridge(random_state=config.data.random_state, **linear_params)
        models["lasso"] = Lasso(random_state=config.data.random_state, **linear_params)
        models["elasticnet"] = ElasticNet(random_state=config.data.random_state, **linear_params)
        
        if self.model_type not in models:
            available_models = list(models.keys())
            raise ValueError(f"Unknown model type: {self.model_type}. Available: {available_models}")
        
        return models[self.model_type]
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> Any:
        """
        Train the model.
        """
        logger.info(f"Training {self.model_type} model")
        
        # Initialize model with filtered parameters
        self.model = self._get_model(params)
        
        # Train model
        try:
            if X_val is not None and y_val is not None and self.model_type in ["xgboost", "lightgbm"]:
                # Use validation set for early stopping if supported
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        self.training_history["train_metrics"] = calculate_metrics(y_train, train_pred)
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            self.training_history["val_metrics"] = calculate_metrics(y_val, val_pred)
        
        # Store feature importance
        self._calculate_feature_importance(X_train.columns.tolist())
        
        logger.info(f"Model trained. Train R²: {self.training_history['train_metrics']['r2']:.4f}")
        
        return self.model
    
    def _calculate_feature_importance(self, feature_names: List[str]) -> None:
        """Calculate and store feature importance."""
        try:
            if hasattr(self.model, "feature_importances_"):
                importance_values = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importance_values = np.abs(self.model.coef_)
                if len(importance_values.shape) > 1:
                    importance_values = importance_values.mean(axis=0)
            else:
                logger.warning("Model does not provide feature importance")
                return
            
            self.feature_importance = pd.DataFrame({
                "feature": feature_names[:len(importance_values)],
                "importance": importance_values
            }).sort_values("importance", ascending=False)
            
            self.feature_importance["importance_pct"] = (
                self.feature_importance["importance"] / self.feature_importance["importance"].sum() * 100
            )
            self.feature_importance["cumulative_pct"] = self.feature_importance["importance_pct"].cumsum()
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
    
    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv_folds: int = 5,
        scoring: str = "neg_mean_squared_error",
        search_type: str = "random"
    ) -> Dict:
        """
        Perform hyperparameter tuning.
        """
        logger.info(f"Starting hyperparameter tuning with {search_type} search")
        
        # Get parameter grid específico para el modelo
        if param_grid is None:
            param_grid = config.model.hyperparameters.get(self.model_type, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid for {self.model_type}. Skipping tuning.")
            return {}
        
        # Filtrar parámetros según el modelo
        filtered_param_grid = self._filter_param_grid(param_grid)
        
        if not filtered_param_grid:
            logger.warning(f"No valid parameters for {self.model_type}. Skipping tuning.")
            return {}
        
        # Initialize base model
        base_model = self._get_model()
        
        # Setup cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=config.data.random_state)
        
        # Setup search
        if search_type == "grid":
            search = GridSearchCV(
                base_model,
                filtered_param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
        else:
            # Calcular n_iter basado en combinaciones posibles
            n_combinations = np.prod([len(v) for v in filtered_param_grid.values()])
            n_iter = min(20, n_combinations)
            
            search = RandomizedSearchCV(
                base_model,
                filtered_param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0,
                random_state=config.data.random_state
            )
        
        # Fit search
        search.fit(X_train, y_train)
        
        # Store results
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        
        # Log results
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {-search.best_score_:.4f}")
        
        return self.best_params
    
    def _filter_param_grid(self, param_grid: Dict) -> Dict:
        """Filtrar parámetros válidos para el modelo actual."""
        
        # Parámetros válidos por tipo de modelo
        valid_params = {
            "xgboost": ['n_estimators', 'max_depth', 'learning_rate', 
                       'subsample', 'colsample_bytree', 'min_child_weight', 'gamma'],
            "lightgbm": ['n_estimators', 'max_depth', 'learning_rate',
                        'num_leaves', 'subsample', 'colsample_bytree'],
            "random_forest": ['n_estimators', 'max_depth', 'min_samples_split',
                             'min_samples_leaf', 'max_features'],
            "gradient_boosting": ['n_estimators', 'max_depth', 'learning_rate',
                                 'subsample', 'min_samples_split', 'min_samples_leaf'],
            "ridge": ['alpha', 'fit_intercept'],
            "lasso": ['alpha', 'fit_intercept'],
            "elasticnet": ['alpha', 'l1_ratio', 'fit_intercept']
        }
        
        allowed = valid_params.get(self.model_type, [])
        
        filtered = {}
        for param, values in param_grid.items():
            if param in allowed:
                filtered[param] = values
            else:
                logger.debug(f"Ignoring parameter '{param}' for model '{self.model_type}'")
        
        return filtered
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model on test data")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = calculate_metrics(y_test, y_pred)
        
        # Additional metrics
        residuals = y_test - y_pred
        self.metrics.update({
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "max_error": np.max(np.abs(residuals)),
            "median_absolute_error": np.median(np.abs(residuals))
        })
        
        # Log results
        logger.info(f"Test Metrics:")
        logger.info(f"  R²: {self.metrics['r2']:.4f}")
        logger.info(f"  RMSE: {self.metrics['rmse']:,.2f}")
        logger.info(f"  MAE: {self.metrics['mae']:,.2f}")
        logger.info(f"  MAPE: {self.metrics['mape']:.2f}%")
        
        return self.metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, List[float]]:
        """Perform cross-validation on the model."""
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Initialize model with best params if available
        model = self._get_model(self.best_params)
        
        # Perform CV
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=config.data.random_state)
        
        try:
            scores = {
                "r2": cross_val_score(model, X, y, cv=cv, scoring="r2"),
                "neg_mse": cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"),
                "neg_mae": cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
            }
            
            # Convert negative scores to positive
            cv_stats = {
                "r2": {
                    "mean": np.mean(scores["r2"]),
                    "std": np.std(scores["r2"]),
                    "values": scores["r2"].tolist()
                },
                "rmse": {
                    "mean": np.sqrt(-np.mean(scores["neg_mse"])),
                    "std": np.std(np.sqrt(-scores["neg_mse"])),
                    "values": np.sqrt(-scores["neg_mse"]).tolist()
                },
                "mae": {
                    "mean": -np.mean(scores["neg_mae"]),
                    "std": np.std(-scores["neg_mae"]),
                    "values": (-scores["neg_mae"]).tolist()
                }
            }
            
            self.training_history["cv_scores"] = cv_stats
            
            logger.info(f"CV R²: {cv_stats['r2']['mean']:.4f} (+/- {cv_stats['r2']['std']:.4f})")
            logger.info(f"CV RMSE: {cv_stats['rmse']['mean']:,.2f} (+/- {cv_stats['rmse']['std']:,.2f})")
            
            return cv_stats
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def save_model(self, path: Path) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_pickle(self.model, path)
        
        # Save model metadata
        metadata = {
            "model_type": self.model_type,
            "best_params": self.best_params,
            "metrics": self.metrics,
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.feature_importance is not None:
            metadata["feature_importance"] = self.feature_importance.to_dict()
        
        metadata_path = path.with_suffix(".json")
        save_json(metadata, metadata_path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """Load a trained model from disk."""
        self.model = load_pickle(path)
        
        # Try to load metadata
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            self.model_type = metadata.get("model_type", self.model_type)
            self.best_params = metadata.get("best_params")
            self.metrics = metadata.get("metrics", {})
            
            if metadata.get("feature_importance"):
                self.feature_importance = pd.DataFrame(metadata["feature_importance"])
            
            self.training_history = metadata.get("training_history", {})
        
        logger.info(f"Model loaded from {path}")
    
    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models_to_compare: List[str] = None
    ) -> pd.DataFrame:
        """Compare multiple models."""
        if models_to_compare is None:
            models_to_compare = ["random_forest", "gradient_boosting", "ridge"]
            if XGB_AVAILABLE:
                models_to_compare.insert(0, "xgboost")
            if LGBM_AVAILABLE:
                models_to_compare.insert(1, "lightgbm")
        
        logger.info(f"Comparing models: {models_to_compare}")
        
        results = []
        
        for model_name in models_to_compare:
            logger.info(f"Training {model_name}...")
            
            try:
                trainer = ModelTrainer(model_name)
                trainer.train(X_train, y_train)
                metrics = trainer.evaluate(X_test, y_test)
                
                results.append({
                    "model": model_name,
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "mape": metrics["mape"]
                })
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results.append({
                    "model": model_name,
                    "r2": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan
                })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values("r2", ascending=False)
        
        logger.info("Model comparison completed")
        return comparison_df