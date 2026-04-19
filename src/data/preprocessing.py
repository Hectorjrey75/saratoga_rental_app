"""
Data preprocessing module for rental price prediction.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from loguru import logger

from src.config.settings import config
from src.utils.helpers import save_pickle, load_pickle, save_json


class DataPreprocessor:
    """Data preprocessing pipeline for rental price prediction."""
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.preprocessing_pipeline: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.numeric_features = config.data.numeric_features.copy()
        self.categorical_features = config.data.categorical_features.copy()
        self.target_column = config.data.target_column
        
        # Initialize scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        logger.info(f"DataPreprocessor initialized with {scaler_type} scaler")
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def explore_data(self, df: pd.DataFrame) -> Dict:
        """
        Perform exploratory data analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with EDA statistics
        """
        logger.info("Performing exploratory data analysis")
        
        eda_stats = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "numeric_stats": df.describe().to_dict(),
            "categorical_stats": {}
        }
        
        # Categorical statistics
        for col in self.categorical_features:
            if col in df.columns:
                eda_stats["categorical_stats"][col] = {
                    "unique_values": df[col].nunique(),
                    "value_counts": df[col].value_counts().to_dict()
                }
        
        # Check for target column
        if self.target_column in df.columns:
            eda_stats["target_stats"] = {
                "mean": df[self.target_column].mean(),
                "median": df[self.target_column].median(),
                "std": df[self.target_column].std(),
                "min": df[self.target_column].min(),
                "max": df[self.target_column].max(),
                "skewness": df[self.target_column].skew(),
                "kurtosis": df[self.target_column].kurtosis()
            }
        
        logger.info("EDA completed successfully")
        return eda_stats
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data")
        df_clean = df.copy()
        
        # Handle missing values for numeric features
        for col in self.numeric_features:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # Handle missing values for categorical features
        for col in self.categorical_features:
            if col in df_clean.columns:
                mode_vals = df_clean[col].mode()
                mode_val = mode_vals[0] if not mode_vals.empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(mode_val)
        
        # Handle outliers using IQR method for numeric features
        for col in self.numeric_features:
            if col in df_clean.columns and col != self.target_column:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.warning(f"Found {outlier_count} outliers in {col}")
                    # Cap outliers instead of removing
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        # Feature engineering
        df_clean = self._create_features(df_clean)
        
        logger.info(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        df_feat = df.copy()
        
        # Price per square foot of living area (solo si price existe - para entrenamiento)
        if "livingArea" in df_feat.columns and self.target_column in df_feat.columns:
            df_feat["price_per_sqft"] = df_feat[self.target_column] / df_feat["livingArea"].replace(0, np.nan)
        
        # Room density (rooms per living area)
        if "rooms" in df_feat.columns and "livingArea" in df_feat.columns:
            df_feat["room_density"] = df_feat["rooms"] / df_feat["livingArea"].replace(0, np.nan)
            # Añadir a numeric_features si no existe
            if "room_density" not in self.numeric_features:
                self.numeric_features.append("room_density")
        
        # Lot size per bedroom
        if "lotSize" in df_feat.columns and "bedrooms" in df_feat.columns:
            df_feat["lot_per_bedroom"] = df_feat["lotSize"] / df_feat["bedrooms"].replace(0, np.nan)
            # Añadir a numeric_features si no existe
            if "lot_per_bedroom" not in self.numeric_features:
                self.numeric_features.append("lot_per_bedroom")
        
        # Age category
        if "age" in df_feat.columns:
            try:
                df_feat["age_category"] = pd.cut(
                    df_feat["age"],
                    bins=[-1, 5, 15, 30, 50, 100, 1000],
                    labels=["New", "Recent", "Moderate", "Old", "Very Old", "Historic"]
                )
                if "age_category" not in self.categorical_features:
                    self.categorical_features.append("age_category")
            except Exception as e:
                logger.warning(f"Could not create age_category: {e}")
        
        # Total bathrooms (igual que bathrooms pero para claridad)
        if "bathrooms" in df_feat.columns:
            df_feat["total_baths"] = df_feat["bathrooms"]
            if "total_baths" not in self.numeric_features:
                self.numeric_features.append("total_baths")
        
        # Living area category
        if "livingArea" in df_feat.columns:
            try:
                df_feat["living_area_category"] = pd.cut(
                    df_feat["livingArea"],
                    bins=[-1, 1000, 1500, 2000, 2500, 100000],
                    labels=["Small", "Medium-Small", "Medium", "Medium-Large", "Large"]
                )
                if "living_area_category" not in self.categorical_features:
                    self.categorical_features.append("living_area_category")
            except Exception as e:
                logger.warning(f"Could not create living_area_category: {e}")
        
        # Price category (solo si price existe)
        if self.target_column in df_feat.columns:
            try:
                df_feat["price_category"] = pd.qcut(
                    df_feat[self.target_column],
                    q=4,
                    labels=["Budget", "Standard", "Premium", "Luxury"]
                )
            except Exception as e:
                logger.warning(f"Could not create price_category: {e}")
        
        logger.info(f"Created new features. Total numeric: {len(self.numeric_features)}, categorical: {len(self.categorical_features)}")
        
        return df_feat
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for prediction."""
        df_feat = df.copy()
        
        # Room density
        if "rooms" in df_feat.columns and "livingArea" in df_feat.columns:
            df_feat["room_density"] = df_feat["rooms"] / df_feat["livingArea"].replace(0, np.nan)
        
        # Lot per bedroom
        if "lotSize" in df_feat.columns and "bedrooms" in df_feat.columns:
            df_feat["lot_per_bedroom"] = df_feat["lotSize"] / df_feat["bedrooms"].replace(0, np.nan)
        
        # Age category
        if "age" in df_feat.columns:
            try:
                df_feat["age_category"] = pd.cut(
                    df_feat["age"],
                    bins=[-1, 5, 15, 30, 50, 100, 1000],
                    labels=["New", "Recent", "Moderate", "Old", "Very Old", "Historic"]
                )
            except Exception:
                pass
        
        # Total baths
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
            except Exception:
                pass
        
        return df_feat
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for model training or prediction.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the preprocessor (True for training, False for prediction)
            
        Returns:
            Processed feature DataFrame
        """
        logger.info(f"Preparing features (fit={fit})")
        
        # Asegurar que las características existen
        df = df.copy()
        
        # Para predicción, crear las mismas características que en entrenamiento
        if not fit:
            df = self._add_derived_features(df)
        
        # Update feature lists based on available columns
        available_numeric = []
        for col in self.numeric_features:
            if col in df.columns:
                available_numeric.append(col)
            else:
                logger.debug(f"Numeric feature '{col}' not found in data")
        
        available_categorical = []
        for col in self.categorical_features:
            if col in df.columns:
                available_categorical.append(col)
            else:
                logger.debug(f"Categorical feature '{col}' not found in data")
        
        logger.info(f"Available numeric features: {len(available_numeric)}")
        logger.info(f"Available categorical features: {len(available_categorical)}")
        
        if fit:
            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", self.scaler)
            ])
            
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None))
            ])
            
            transformers = []
            if available_numeric:
                transformers.append(("num", numeric_transformer, available_numeric))
            if available_categorical:
                transformers.append(("cat", categorical_transformer, available_categorical))
            
            if not transformers:
                raise ValueError("No features available for preprocessing")
            
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=transformers,
                remainder="drop"
            )
            
            # Fit and transform
            X_processed = self.preprocessing_pipeline.fit_transform(df)
            
            # Get feature names
            self.feature_names = self._get_feature_names()
            logger.info(f"Fitted preprocessor. Output shape: {X_processed.shape}")
            
        else:
            if self.preprocessing_pipeline is None:
                raise ValueError("Preprocessor not fitted. Call prepare_features with fit=True first.")
            
            # Verificar que todas las columnas esperadas estén presentes
            expected_columns = []
            for name, transformer, columns in self.preprocessing_pipeline.transformers_:
                expected_columns.extend(columns)
            
            # Añadir columnas faltantes con valores por defecto
            for col in expected_columns:
                if col not in df.columns:
                    logger.warning(f"Missing column '{col}', adding with default value")
                    if col in self.numeric_features:
                        df[col] = 0.0
                    else:
                        df[col] = "Unknown"
            
            # Transform only
            try:
                X_processed = self.preprocessing_pipeline.transform(df)
            except Exception as e:
                logger.error(f"Error during transform: {e}")
                logger.info(f"Expected columns: {expected_columns}")
                logger.info(f"Available columns: {df.columns.tolist()}")
                raise
        
        # Convert to DataFrame with feature names
        if self.feature_names:
            # Asegurar que el número de columnas coincida
            if X_processed.shape[1] == len(self.feature_names):
                X_processed_df = pd.DataFrame(
                    X_processed,
                    columns=self.feature_names,
                    index=df.index
                )
            else:
                logger.warning(f"Feature name mismatch: {X_processed.shape[1]} vs {len(self.feature_names)}")
                # Generar nombres genéricos
                generic_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                X_processed_df = pd.DataFrame(
                    X_processed,
                    columns=generic_names,
                    index=df.index
                )
        else:
            generic_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
            X_processed_df = pd.DataFrame(
                X_processed,
                columns=generic_names,
                index=df.index
            )
        
        return X_processed_df
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names from fitted preprocessor."""
        if self.preprocessing_pipeline is None:
            return []
        
        feature_names = []
        
        for name, transformer, columns in self.preprocessing_pipeline.transformers_:
            if name == "num":
                feature_names.extend(columns)
            elif name == "cat":
                # Obtener nombres de características one-hot encoded
                onehot_step = transformer.named_steps.get("onehot", None)
                if onehot_step is not None:
                    try:
                        # Para versiones más recientes de scikit-learn
                        if hasattr(onehot_step, "get_feature_names_out"):
                            cat_features = onehot_step.get_feature_names_out(columns)
                            feature_names.extend(cat_features.tolist())
                        else:
                            # Fallback para versiones anteriores
                            for i, col in enumerate(columns):
                                if hasattr(onehot_step, "categories_"):
                                    categories = onehot_step.categories_[i]
                                    for cat in categories:
                                        feature_names.append(f"{col}_{cat}")
                                else:
                                    feature_names.append(f"{col}_encoded")
                    except Exception as e:
                        logger.warning(f"Could not get categorical feature names: {e}")
                        for col in columns:
                            feature_names.append(f"{col}_encoded")
        
        return feature_names
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set
            val_size: Proportion of validation set
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        test_size = test_size or config.data.test_size
        val_size = val_size or config.data.validation_size
        
        logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=config.data.random_state
        )
        
        # Second split: train vs val
        if len(X_temp) > 1:
            val_size_adjusted = val_size / (1 - test_size) if (1 - test_size) > 0 else 0.2
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=config.data.random_state
            )
        else:
            # Si hay muy pocos datos, usar los mismos para todo
            X_train = X_val = X_temp
            y_train = y_val = y_temp
            logger.warning("Not enough data for proper split. Using same data for train and val.")
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, path: Path) -> None:
        """Save the fitted preprocessor to disk."""
        # Guardar también las listas de características
        state = {
            "preprocessor": self,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "feature_names": self.feature_names,
            "scaler_type": self.scaler_type
        }
        save_pickle(state, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load_preprocessor(cls, path: Path) -> "DataPreprocessor":
        """Load a fitted preprocessor from disk."""
        state = load_pickle(path)
        
        if isinstance(state, dict):
            preprocessor = state["preprocessor"]
            preprocessor.numeric_features = state.get("numeric_features", preprocessor.numeric_features)
            preprocessor.categorical_features = state.get("categorical_features", preprocessor.categorical_features)
            preprocessor.feature_names = state.get("feature_names", preprocessor.feature_names)
        else:
            # Para compatibilidad con versiones anteriores
            preprocessor = state
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        output_dir: Path
    ) -> None:
        """Save processed data to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(output_dir / "X_train.csv", index=False)
        X_val.to_csv(output_dir / "X_val.csv", index=False)
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_val.to_csv(output_dir / "y_val.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)
        
        # Guardar también los nombres de las características
        feature_names_df = pd.DataFrame({"feature_names": self.feature_names})
        feature_names_df.to_csv(output_dir / "feature_names.csv", index=False)
        
        logger.info(f"Processed data saved to {output_dir}")
    
    def get_feature_info(self) -> Dict:
        """Get information about features used in preprocessing."""
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "feature_names": self.feature_names,
            "total_features": len(self.feature_names)
        }